//! Headless A/B timing of [`PhysicsWorld::step`] against [`PhysicsWorld::step_collisions_last`].
//!
//! # Purpose
//!
//! `step_collisions_last` runs the collision detection at the end of the step, so the contacts
//! match the poses read between steps. This benchmark measures what that order costs or saves
//! compared with `step`, headless.
//!
//! Scenes:
//! - The ten 2D stress-test scenes, copied from `examples2d/stress_tests/` without the viewer
//!   code: Balls, Boxes, Capsules, Convex polygons, Heightfield, Pyramid, Verticals stacks, and
//!   the three "(Stress test) joint" scenes. Nothing changes between steps, so both orders run
//!   the same stages.
//! - "Spawner": a box pile hit by a projectile inserted every N steps, with old projectiles
//!   removed. A collider insertion makes a collisions-last step detect collisions before its
//!   solve as well (a catch-up), which the stress scenes never trigger.
//!
//! # Method
//!
//! Each scene is built twice: world A steps with `step`, world B with `step_collisions_last`
//! (initialized once with `initialize_collisions_last_with_events`). Both run the untimed
//! warm-up steps, then the timed iterations. The order alternates: even iterations step A then
//! B, odd iterations B then A, so neither order systematically runs second on a warmer cache.
//! Scene callbacks (the spawner's insertions and removals) run before a step, at the same step
//! indices in both worlds, and are not timed.
//!
//! Each step call is timed with `std::time::Instant`. The pipeline's `counters.step_time` is not
//! used: its timers only measure when rapier is built with the `profiler` feature, and that
//! feature adds a timer call around every internal stage of both orders.
//!
//! The summary reports, per scene, the mean, standard deviation, median, quartiles and IQR of
//! both orders, the percent difference of the means, and Welch's t-test (unequal variances) with
//! its two-tailed p-value. After the run, every body pose in both worlds must be finite, and no
//! body or collider may have been quarantined.
//!
//! # Running
//!
//! Always run it in release mode:
//!
//! ```text
//! cargo test -p rapier2d --release --test collisions_last_timing -- --ignored --nocapture
//! ```
//!
//! With `enhanced-determinism` and `serde-serialize`:
//!
//! ```text
//! cargo test -p rapier2d --release --features enhanced-determinism,serde-serialize \
//!     --test collisions_last_timing -- --ignored --nocapture
//! ```
//!
//! Without `--ignored`, only the fast unit tests of the statistics functions run.
//!
//! # Knobs (environment variables, read at run time)
//!
//! - `COLLISIONS_LAST_TIMING_ITERS`: timed iterations per scene (default 500, minimum 2).
//! - `COLLISIONS_LAST_TIMING_WARMUP`: untimed warm-up steps per scene (default 20).
//! - `COLLISIONS_LAST_TIMING_SCENES`: comma-separated filter. A scene runs if its name contains
//!   one of the entries, ignoring case; for example `joint,spawner`. Default: all scenes.
//! - `COLLISIONS_LAST_TIMING_SPAWN_EVERY`: the Spawner scene inserts a projectile every this many
//!   steps (default 10, minimum 1).
//!
//! # Output
//!
//! The summary is printed to stdout and written to
//! `$CARGO_TARGET_DIR/collisions_last_timing_2d_summary.txt`, or to
//! `<workspace>/target/collisions_last_timing_2d_summary.txt` when `CARGO_TARGET_DIR` is unset.
//!
//! # Warning
//!
//! Run it on an idle machine. Other builds, test runs or heavy programs running at the same time
//! skew the numbers. The alternation cancels slow drifts that affect both orders equally, but not
//! the extra noise.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rapier2d::prelude::*;

const SUMMARY_FILE_NAME: &str = "collisions_last_timing_2d_summary.txt";

struct Knobs {
    iters: usize,
    warmup: usize,
    /// Lowercased filter entries; `None` selects every scene.
    scenes: Option<Vec<String>>,
    spawn_every: usize,
}

impl Knobs {
    fn from_env() -> Self {
        let scenes = std::env::var("COLLISIONS_LAST_TIMING_SCENES")
            .ok()
            .map(|filter| {
                filter
                    .split(',')
                    .map(|entry| entry.trim().to_lowercase())
                    .filter(|entry| !entry.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|entries| !entries.is_empty());
        Self {
            iters: env_usize("COLLISIONS_LAST_TIMING_ITERS", 500, 2),
            warmup: env_usize("COLLISIONS_LAST_TIMING_WARMUP", 20, 0),
            scenes,
            spawn_every: env_usize("COLLISIONS_LAST_TIMING_SPAWN_EVERY", 10, 1),
        }
    }

    fn selects(&self, name: &str) -> bool {
        let name = name.to_lowercase();
        self.scenes
            .as_ref()
            .is_none_or(|entries| entries.iter().any(|entry| name.contains(entry.as_str())))
    }
}

/// Reads a non-negative integer knob. Unset or empty gives `default`; anything unparsable or
/// below `min` panics, so a typo never silently runs the default.
fn env_usize(name: &str, default: usize, min: usize) -> usize {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    if raw.trim().is_empty() {
        return default;
    }
    match raw.trim().parse::<usize>() {
        Ok(value) if value >= min => value,
        _ => panic!("{name}={raw:?}: expected an integer >= {min}"),
    }
}

/// A benchmark scene. Built once per stepping order, so each world gets its own hooks and callback
/// state.
struct Scene {
    world: PhysicsWorld,
    hooks: Box<dyn PhysicsHooks>,
    before_step: Option<BeforeStep>,
    /// Appended to the scene name in the summary.
    detail: Option<String>,
}

/// Called with the step index (warm-up steps included) before each step, outside the timed
/// region.
type BeforeStep = Box<dyn FnMut(usize, &mut PhysicsWorld)>;

impl Scene {
    fn plain(world: PhysicsWorld) -> Self {
        Self {
            world,
            hooks: Box::new(()),
            before_step: None,
            detail: None,
        }
    }
}

type SceneBuilder = fn(&Knobs) -> Scene;

/// The stress scenes first, under their testbed names, then the Spawner scene.
const SCENES: &[(&str, SceneBuilder)] = &[
    ("Balls", balls),
    ("Boxes", boxes),
    ("Capsules", capsules),
    ("Convex polygons", convex_polygons),
    ("Heightfield", heightfield),
    ("Pyramid", pyramid),
    ("Verticals stacks", vertical_stacks),
    ("(Stress test) joint ball", joint_ball),
    ("(Stress test) joint fixed", joint_fixed),
    ("(Stress test) joint prismatic", joint_prismatic),
    ("Spawner", spawner),
];

/// Copy of `examples2d/stress_tests/balls2.rs`.
fn balls(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let num = 50;
    let rad = 1.0;

    let shiftx = rad * 2.5;
    let shifty = rad * 2.0;
    let centerx = shiftx * (num as f32) / 2.0;
    let centery = shifty / 2.0;

    for i in 0..num {
        for j in 0usize..num * 5 {
            let x = i as f32 * shiftx - centerx;
            let y = j as f32 * shifty + centery;

            let status = if j == 0 {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };

            let rigid_body = RigidBodyBuilder::new(status).translation(Vec2::new(x, y));
            let collider = ColliderBuilder::ball(rad);
            let _ = world.insert(rigid_body, collider);
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/boxes2.rs`.
fn boxes(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let ground_size = 25.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::cuboid(ground_size, 1.2);
    let _ = world.insert(rigid_body, collider);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(ground_size, ground_size * 2.0));
    let collider = ColliderBuilder::cuboid(ground_size * 2.0, 1.2);
    let _ = world.insert(rigid_body, collider);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(-ground_size, ground_size * 2.0));
    let collider = ColliderBuilder::cuboid(ground_size * 2.0, 1.2);
    let _ = world.insert(rigid_body, collider);

    let num = 26;
    let rad = 0.5;

    let shift = rad * 2.0;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;

    for i in 0..num {
        for j in 0usize..num * 5 {
            let x = i as f32 * shift - centerx;
            let y = j as f32 * shift + centery + 2.0;

            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
            let collider = ColliderBuilder::cuboid(rad, rad);
            let _ = world.insert(rigid_body, collider);
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/capsules2.rs`.
fn capsules(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let ground_size = 25.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::cuboid(ground_size, 1.2);
    let _ = world.insert(rigid_body, collider);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(ground_size, ground_size * 4.0));
    let collider = ColliderBuilder::cuboid(ground_size * 4.0, 1.2);
    let _ = world.insert(rigid_body, collider);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(-ground_size, ground_size * 4.0));
    let collider = ColliderBuilder::cuboid(ground_size * 4.0, 1.2);
    let _ = world.insert(rigid_body, collider);

    let num = 26;
    let numy = num * 5;
    let rad = 0.5;

    let shift = rad * 2.0;
    let shifty = rad * 5.0;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;

    for i in 0..num {
        for j in 0usize..numy {
            let x = i as f32 * shift - centerx;
            let y = j as f32 * shifty + centery + 3.0;

            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
            let collider = ColliderBuilder::capsule_y(rad * 1.5, rad);
            let _ = world.insert(rigid_body, collider);
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/convex_polygons2.rs`, except for the random generator. The
/// example uses `rand::rngs::StdRng`, which is not a dependency of `rapier2d`, so the points come
/// from the `oorandom` dev-dependency (same seed, same sampling order, different values). The
/// scene structure is unchanged: 5 random convex hulls of 10 points, reused over 26×130 bodies.
fn convex_polygons(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let ground_size = 30.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::cuboid(ground_size, 1.2);
    let _ = world.insert(rigid_body, collider);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(ground_size, ground_size * 2.0));
    let collider = ColliderBuilder::cuboid(ground_size * 2.0, 1.2);
    let _ = world.insert(rigid_body, collider);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(-ground_size, ground_size * 2.0));
    let collider = ColliderBuilder::cuboid(ground_size * 2.0, 1.2);
    let _ = world.insert(rigid_body, collider);

    let num = 26;
    let scale = 2.0;
    let border_rad = 0.0;

    let shift = border_rad * 2.0 + scale;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;

    let mut rng = oorandom::Rand32::new(0);

    let poly_shapes: Vec<_> = (0..5)
        .map(|_| {
            let mut points = Vec::new();
            for _ in 0..10 {
                let pt = Vec2::new(rng.rand_float(), rng.rand_float());
                points.push(pt * scale);
            }
            SharedShape::convex_hull(&points).unwrap()
        })
        .collect();

    for i in 0..num {
        for j in 0usize..num * 5 {
            let x = i as f32 * shift - centerx;
            let y = j as f32 * shift * 2.0 + centery + 2.0;

            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
            let collider = ColliderBuilder::new(poly_shapes[i % 5].clone());
            let _ = world.insert(rigid_body, collider);
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/heightfield2.rs`.
fn heightfield(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let ground_size = Vec2::new(50.0, 1.0);
    let nsubdivs = 2000;

    let heights = (0..nsubdivs + 1)
        .map(|i| {
            if i == 0 || i == nsubdivs {
                80.0
            } else {
                (i as f32 * ground_size.x / (nsubdivs as f32)).cos() * 2.0
            }
        })
        .collect();

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::heightfield(heights, ground_size);
    let _ = world.insert(rigid_body, collider);

    let num = 26;
    let rad = 0.5;

    let shift = rad * 2.0;
    let centerx = shift * (num / 2) as f32;
    let centery = shift / 2.0;

    for i in 0..num {
        for j in 0usize..num * 5 {
            let x = i as f32 * shift - centerx;
            let y = j as f32 * shift + centery + 3.0;

            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));

            if j % 2 == 0 {
                let collider = ColliderBuilder::cuboid(rad, rad);
                let _ = world.insert(rigid_body, collider);
            } else {
                let collider = ColliderBuilder::ball(rad);
                let _ = world.insert(rigid_body, collider);
            }
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/pyramid2.rs`.
fn pyramid(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let ground_size = 100.0;
    let ground_thickness = 1.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::cuboid(ground_size, ground_thickness);
    let _ = world.insert(rigid_body, collider);

    let num = 100;
    let rad = 0.5;

    let shift = rad * 2.0;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0 + ground_thickness + rad * 1.5;

    for i in 0usize..num {
        for j in i..num {
            let fj = j as f32;
            let fi = i as f32;
            let x = (fi * shift / 2.0) + (fj - fi) * shift - centerx;
            let y = fi * shift + centery;

            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
            let collider = ColliderBuilder::cuboid(rad, rad);
            let _ = world.insert(rigid_body, collider);
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/vertical_stacks2.rs`.
fn vertical_stacks(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let num = 80;
    let rad = 0.5;

    let ground_size = num as f32 * rad * 10.0;
    let ground_thickness = 1.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::cuboid(ground_size, ground_thickness);
    let _ = world.insert(rigid_body, collider);

    let shiftx_centerx = [
        (rad * 2.0, -(num as f32) * rad * 2.0 * 1.5),
        (rad * 2.0 + rad, num as f32 * rad * 2.0 * 1.5),
    ];

    for (shiftx, centerx) in shiftx_centerx {
        let shifty = rad * 2.0;
        let centery = shifty / 2.0 + ground_thickness;

        for i in 0..num {
            for j in 0usize..1 + i * 2 {
                let fj = j as f32;
                let fi = i as f32;
                let x = (fj - fi) * shiftx + centerx;
                let y = (num as f32 - fi - 1.0) * shifty + centery;

                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
                let collider = ColliderBuilder::cuboid(rad, rad);
                let _ = world.insert(rigid_body, collider);
            }
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/joint_ball2.rs`.
fn joint_ball(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let rad = 0.4;
    let numi = 100; // Num vertical nodes.
    let numk = 100; // Num horizontal nodes.
    let shift = 1.0;

    let mut body_handles = Vec::new();

    for k in 0..numk {
        for i in 0..numi {
            let fk = k as f32;
            let fi = i as f32;

            let status = if k >= numk / 2 - 3 && k <= numk / 2 + 3 && i == 0 {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };

            let rigid_body =
                RigidBodyBuilder::new(status).translation(Vec2::new(fk * shift, -fi * shift));
            let collider = ColliderBuilder::ball(rad);
            let (child_handle, _) = world.insert(rigid_body, collider);

            // Vertical joint.
            if i > 0 {
                let parent_handle = *body_handles.last().unwrap();
                let joint = RevoluteJointBuilder::new().local_anchor2(Vec2::new(0.0, shift));
                world.insert_impulse_joint(parent_handle, child_handle, joint);
            }

            // Horizontal joint.
            if k > 0 {
                let parent_index = body_handles.len() - numi;
                let parent_handle = body_handles[parent_index];
                let joint = RevoluteJointBuilder::new().local_anchor2(Vec2::new(-shift, 0.0));
                world.insert_impulse_joint(parent_handle, child_handle, joint);
            }

            body_handles.push(child_handle);
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/joint_fixed2.rs`.
fn joint_fixed(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let rad = 0.4;
    let num = 30; // Num vertical nodes.
    let shift = 1.0;

    let mut body_handles = Vec::new();

    for xx in 0..4 {
        let x = xx as f32 * shift * (num as f32 + 2.0);

        for yy in 0..4 {
            let y = yy as f32 * shift * (num as f32 + 4.0);

            for k in 0..num {
                for i in 0..num {
                    let fk = k as f32;
                    let fi = i as f32;

                    let status = if k == 0 {
                        RigidBodyType::Fixed
                    } else {
                        RigidBodyType::Dynamic
                    };

                    let rigid_body = RigidBodyBuilder::new(status)
                        .translation(Vec2::new(x + fk * shift, y - fi * shift));
                    let collider = ColliderBuilder::ball(rad);
                    let (child_handle, _) = world.insert(rigid_body, collider);

                    // Vertical joint.
                    if i > 0 {
                        let parent_handle = *body_handles.last().unwrap();
                        let joint =
                            FixedJointBuilder::new().local_frame2(Pose2::translation(0.0, shift));
                        world.insert_impulse_joint(parent_handle, child_handle, joint);
                    }

                    // Horizontal joint.
                    if k > 0 {
                        let parent_index = body_handles.len() - num;
                        let parent_handle = body_handles[parent_index];
                        let joint =
                            FixedJointBuilder::new().local_frame2(Pose2::translation(-shift, 0.0));
                        world.insert_impulse_joint(parent_handle, child_handle, joint);
                    }

                    body_handles.push(child_handle);
                }
            }
        }
    }

    Scene::plain(world)
}

/// Copy of `examples2d/stress_tests/joint_prismatic2.rs`.
fn joint_prismatic(_: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    let rad = 0.4;
    let num = 10;
    let shift = 1.0;

    for l in 0..25 {
        let y = l as f32 * shift * (num as f32 + 2.0) * 2.0;

        for j in 0..50 {
            let x = j as f32 * shift * 4.0;

            let ground = RigidBodyBuilder::fixed().translation(Vec2::new(x, y));
            let collider = ColliderBuilder::cuboid(rad, rad);
            let (mut curr_parent, _) = world.insert(ground, collider);

            for i in 0..num {
                let y = y - (i + 1) as f32 * shift;
                let density = 1.0;
                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
                let collider = ColliderBuilder::cuboid(rad, rad).density(density);
                let (curr_child, _) = world.insert(rigid_body, collider);

                let axis = if i % 2 == 0 {
                    Vec2::new(1.0, 1.0).normalize()
                } else {
                    Vec2::new(-1.0, 1.0).normalize()
                };

                let prism = PrismaticJointBuilder::new(axis)
                    .local_anchor2(Vec2::new(0.0, shift))
                    .limits([-1.5, 1.5]);
                world.insert_impulse_joint(curr_parent, curr_child, prism);

                curr_parent = curr_child;
            }
        }
    }

    Scene::plain(world)
}

/// Steps a projectile lives before it is removed.
const PROJECTILE_LIFETIME_STEPS: usize = 180;

/// A 16×16 box pile on the ground, awake, in the projectiles' path, and a 16×8 box pile that
/// starts asleep behind a divider wall. From step 0, a ball is fired at the awake pile every
/// `spawn_every` steps and removed `PROJECTILE_LIFETIME_STEPS` steps later.
fn spawner(knobs: &Knobs) -> Scene {
    let mut world = PhysicsWorld::new();

    // Ground (top at y = 1), outer walls and the divider.
    let _ = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(60.0, 1.0),
    );
    for (x, half_height) in [(-40.0, 20.0), (40.0, 20.0), (12.0, 10.0)] {
        let _ = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(x, 1.0 + half_height)),
            ColliderBuilder::cuboid(0.5, half_height),
        );
    }

    for (x0, columns, rows, sleeping) in [(-8.0, 16, 16, false), (14.0, 16, 8, true)] {
        for i in 0..columns {
            for j in 0..rows {
                let center = Vector::new(x0 + 0.5 + i as Real, 1.5 + j as Real);
                let _ = world.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(center)
                        .sleeping(sleeping),
                    ColliderBuilder::cuboid(0.5, 0.5),
                );
            }
        }
    }

    let spawn_every = knobs.spawn_every;
    let mut live: VecDeque<(usize, RigidBodyHandle)> = VecDeque::new();
    let before_step = move |step: usize, world: &mut PhysicsWorld| {
        while let Some(&(born, handle)) = live.front() {
            if step < born + PROJECTILE_LIFETIME_STEPS {
                break;
            }
            let _ = world.remove_body(handle);
            let _ = live.pop_front();
        }
        if step % spawn_every == 0 {
            // Sweep the launch height over the pile, one tile higher per shot.
            let shot = step / spawn_every;
            let height = 2.0 + (shot % 12) as Real;
            let (handle, _) = world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(-30.0, height))
                    .linvel(Vector::new(25.0, 0.0)),
                ColliderBuilder::ball(0.3).density(10.0),
            );
            live.push_back((step, handle));
        }
    };

    Scene {
        world,
        hooks: Box::new(()),
        before_step: Some(Box::new(before_step)),
        detail: Some(format!(
            "projectile collider inserted every {spawn_every} steps, removed after {PROJECTILE_LIFETIME_STEPS}"
        )),
    }
}

#[derive(Clone, Copy)]
enum Order {
    Step,
    CollisionsLast,
}

/// One stepping order of one scene, and what it recorded.
struct Arm {
    scene: Scene,
    order: Order,
    samples_ms: Vec<f64>,
    /// Bodies and colliders quarantined over the whole run.
    quarantined: usize,
}

impl Arm {
    fn new(build: SceneBuilder, knobs: &Knobs, order: Order) -> Self {
        let mut scene = build(knobs);
        // Only measures with the `profiler` feature; the samples come from `Instant` (see the
        // module docs).
        scene.world.physics_pipeline.counters.enable();
        if let Order::CollisionsLast = order {
            scene
                .world
                .initialize_collisions_last_with_events(&*scene.hooks, &());
        }
        Self {
            scene,
            order,
            samples_ms: Vec::with_capacity(knobs.iters),
            quarantined: 0,
        }
    }

    /// Runs step `step` and records its duration when `timed`.
    fn step(&mut self, step: usize, timed: bool) {
        let scene = &mut self.scene;
        if let Some(before_step) = scene.before_step.as_mut() {
            before_step(step, &mut scene.world);
        }

        let hooks = &*scene.hooks;
        let start = Instant::now();
        match self.order {
            Order::Step => scene.world.step_with_events(hooks, &()),
            Order::CollisionsLast => scene.world.step_collisions_last_with_events(hooks, &()),
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        if timed {
            self.samples_ms.push(elapsed_ms);
        }
        let quarantine = scene.world.quarantine();
        self.quarantined += quarantine.bodies().len() + quarantine.colliders().len();
    }

    fn assert_sane(&self, scene_name: &str) {
        let label = match self.order {
            Order::Step => "step()",
            Order::CollisionsLast => "step_collisions_last()",
        };
        for (handle, body) in self.scene.world.bodies.iter() {
            let translation = body.translation();
            assert!(
                translation.is_finite() && body.rotation().angle().is_finite(),
                "{scene_name}, {label}: body {handle:?} has a non-finite pose ({translation:?})"
            );
        }
        assert_eq!(
            self.quarantined, 0,
            "{scene_name}, {label}: bodies or colliders were quarantined (non-finite state)"
        );
    }
}

#[test]
#[ignore = "timing-based perf check; run manually in release"]
fn step_vs_step_collisions_last() {
    let knobs = Knobs::from_env();
    let selected: Vec<_> = SCENES
        .iter()
        .filter(|(name, _)| knobs.selects(name))
        .collect();
    assert!(
        !selected.is_empty(),
        "COLLISIONS_LAST_TIMING_SCENES={:?} matches no scene. Scenes: {:?}",
        std::env::var("COLLISIONS_LAST_TIMING_SCENES").unwrap_or_default(),
        SCENES.iter().map(|(name, _)| *name).collect::<Vec<_>>()
    );

    let mut lines = summary_header(&knobs);

    for &&(name, build) in &selected {
        println!("Running {name}...");
        let mut a = Arm::new(build, &knobs, Order::Step);
        let mut b = Arm::new(build, &knobs, Order::CollisionsLast);
        let title = match &a.scene.detail {
            Some(detail) => format!("{name} ({detail})"),
            None => name.to_string(),
        };

        for step in 0..knobs.warmup + knobs.iters {
            let timed = step >= knobs.warmup;
            let iteration = if timed { step - knobs.warmup } else { step };
            if iteration % 2 == 0 {
                a.step(step, timed);
                b.step(step, timed);
            } else {
                b.step(step, timed);
                a.step(step, timed);
            }
        }

        a.assert_sane(name);
        b.assert_sane(name);
        lines.extend(scene_summary(&title, &a.samples_ms, &b.samples_ms));
    }

    let text = lines.join("\n") + "\n";
    println!("\n{text}");
    let path = summary_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    }
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    println!("Summary written to {}", path.display());
}

fn summary_header(knobs: &Knobs) -> Vec<String> {
    macro_rules! enabled_features {
        ($($feature:tt),* $(,)?) => {{
            let mut enabled: Vec<&str> = Vec::new();
            $(if cfg!(feature = $feature) { enabled.push($feature); })*
            enabled
        }};
    }
    let features = enabled_features![
        "dim2",
        "f32",
        "std",
        "alloc",
        "block-solver",
        "parallel",
        "simd8",
        "serde-serialize",
        "enhanced-determinism",
        "debug-render",
        "profiler",
        "bytemuck",
        "unsync-callbacks",
        "solver-bounds-checks",
        "dev-remove-slow-accessors",
        "debug-disable-legitimate-fe-exceptions",
    ];

    let mut lines = vec![
        "Benchmark Summary: rapier step() vs step_collisions_last()".to_string(),
        format!(
            "Iterations per benchmark: {} timed, after {} untimed warm-up steps",
            knobs.iters, knobs.warmup
        ),
        "Method: two identical worlds per scene; even iterations step the step() world first, \
         odd iterations the step_collisions_last() world first; each step call timed with \
         std::time::Instant"
            .to_string(),
        format!("rapier2d features: {}", features.join(", ")),
        format!("Git revision: {}", git_revision()),
    ];
    if cfg!(debug_assertions) {
        lines.push(
            "WARNING: debug build; the timings are not representative (use --release)".into(),
        );
    }
    lines.push(String::new());
    lines.push(String::new());
    lines
}

/// The per-scene block.
fn scene_summary(title: &str, step: &[f64], collisions_last: &[f64]) -> Vec<String> {
    let (r_mean, r_std, r_med, r_q1, r_q3) = compute_stats(step);
    let (c_mean, c_std, c_med, c_q1, c_q3) = compute_stats(collisions_last);
    let (t_stat, df, p_value) = welch_t_test(step, collisions_last, r_mean, c_mean, r_std, c_std);
    let pct_diff = if r_mean.abs() > 1e-12 {
        (c_mean - r_mean) / r_mean * 100.0
    } else {
        0.0
    };
    let sig = if p_value < 0.01 {
        "YES (p < 0.01)"
    } else if p_value < 0.05 {
        "YES (p < 0.05)"
    } else {
        "NO (p >= 0.05)"
    };

    vec![
        format!("=== {title} ==="),
        format!("  N = {}", step.len()),
        String::new(),
        format!(
            "  {:30} {:>12} {:>12}",
            "", "step()", "step_collisions_last()"
        ),
        format!("  {:30} {:>12.4} {:>12.4}", "Mean (ms):", r_mean, c_mean),
        format!("  {:30} {:>12.4} {:>12.4}", "Std Dev (ms):", r_std, c_std),
        format!("  {:30} {:>12.4} {:>12.4}", "Median (ms):", r_med, c_med),
        format!("  {:30} {:>12.4} {:>12.4}", "Q1 (ms):", r_q1, c_q1),
        format!("  {:30} {:>12.4} {:>12.4}", "Q3 (ms):", r_q3, c_q3),
        format!(
            "  {:30} {:>12.4} {:>12.4}",
            "IQR (ms):",
            r_q3 - r_q1,
            c_q3 - c_q1
        ),
        String::new(),
        format!("  Difference: {pct_diff:+.2}% (collisions_last vs step)"),
        format!("  Welch's t = {t_stat:.4}, df = {df:.1}, p = {p_value:.6}"),
        format!("  Statistically significant: {sig}"),
        String::new(),
    ]
}

fn summary_path() -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    target.join(SUMMARY_FILE_NAME)
}

/// Short commit hash of the checkout, marked when tracked files have uncommitted changes.
fn git_revision() -> String {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    match git(&["rev-parse", "--short=10", "HEAD"]) {
        Some(revision) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|status| !status.is_empty());
            if dirty {
                format!("{revision} (with uncommitted changes)")
            } else {
                revision
            }
        }
        None => "unavailable".to_string(),
    }
}

/// Returns (mean, std_dev, median, q1, q3) for the given data.
fn compute_stats(data: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = percentile_sorted(&sorted, 50.0);
    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);

    (mean, std_dev, median, q1, q3)
}

/// Linear interpolation percentile on already-sorted data.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = lo + 1;
    let frac = rank - lo as f64;
    if hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Welch's t-test for two independent samples with unequal variances.
/// Returns (t_statistic, degrees_of_freedom, p_value).
fn welch_t_test(
    a: &[f64],
    b: &[f64],
    mean_a: f64,
    mean_b: f64,
    std_a: f64,
    std_b: f64,
) -> (f64, f64, f64) {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let var_a = std_a * std_a;
    let var_b = std_b * std_b;
    let se = (var_a / n_a + var_b / n_b).sqrt();

    if se < 1e-15 {
        return (0.0, n_a + n_b - 2.0, 1.0);
    }

    let t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom.
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = num / denom;

    // Two-tailed p-value via regularized incomplete beta function.
    let p = two_tailed_t_p_value(t.abs(), df);
    (t, df, p)
}

/// Two-tailed p-value for Student's t distribution:
/// p = I_{df/(df+t^2)}(df/2, 1/2), where I is the regularized incomplete beta.
fn two_tailed_t_p_value(t_abs: f64, df: f64) -> f64 {
    let x = df / (df + t_abs * t_abs);
    regularized_incomplete_beta(x, df / 2.0, 0.5)
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction (Lentz).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation if needed for better convergence.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();
    let prefix = ln_prefix.exp();

    // Lentz's continued fraction.
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0_f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;

        // Even step: d_{2m}
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + num_even / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        f *= c * d;

        // Odd step: d_{2m+1}
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + num_odd / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    prefix * f
}

/// ln(Beta(a,b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation for ln(Gamma(x)).
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + i as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

fn assert_close(got: f64, expected: f64, epsilon: f64, what: &str) {
    assert!(
        (got - expected).abs() <= epsilon,
        "{what}: got {got}, expected {expected} ± {epsilon}"
    );
}

#[test]
fn stats_two_tailed_t_p_values_match_known_values() {
    // Two-tailed 5% critical values of Student's t (tables give t to 3 decimals).
    assert_close(two_tailed_t_p_value(2.228, 10.0), 0.05, 5e-4, "df = 10");
    assert_close(two_tailed_t_p_value(2.042, 30.0), 0.05, 5e-4, "df = 30");
    // Closed forms: df = 1 is the Cauchy distribution, P(|T| > 1) = 1/2; df = 2 gives
    // P(|T| > t) = 1 - t / sqrt(2 + t^2).
    assert_close(two_tailed_t_p_value(1.0, 1.0), 0.5, 1e-8, "df = 1");
    assert_close(
        two_tailed_t_p_value(2.0, 2.0),
        1.0 - 2.0 / 6.0_f64.sqrt(),
        1e-8,
        "df = 2",
    );
    assert_close(two_tailed_t_p_value(0.0, 10.0), 1.0, 1e-12, "t = 0");
    // Large df, as with many iterations: approaches the normal distribution's 1.959964.
    assert_close(
        two_tailed_t_p_value(1.959964, 1.0e5),
        0.05,
        1e-4,
        "df = 1e5",
    );
}

#[test]
fn stats_regularized_incomplete_beta_symmetry_and_closed_forms() {
    // I_x(a, b) = 1 - I_{1-x}(b, a), on both sides of the continued fraction's branch point.
    for (x, a, b) in [
        (0.1, 2.0, 3.0),
        (0.3, 0.5, 5.0),
        (0.7, 15.0, 0.5),
        (0.5, 250.0, 0.5),
        (0.99, 3.5, 1.25),
    ] {
        let lhs = regularized_incomplete_beta(x, a, b);
        let rhs = 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
        assert_close(lhs, rhs, 1e-10, &format!("symmetry at ({x}, {a}, {b})"));
    }

    // The symmetry above holds by construction away from the branch point, so also check the
    // continued fraction against closed forms on both branches.
    for x in [0.05, 0.3, 0.5, 0.8, 0.97] {
        assert_close(
            regularized_incomplete_beta(x, 3.0, 1.0),
            x * x * x,
            1e-9,
            &format!("I_{x}(3, 1) = x^3"),
        );
        assert_close(
            regularized_incomplete_beta(x, 1.0, 2.5),
            1.0 - (1.0 - x).powf(2.5),
            1e-9,
            &format!("I_{x}(1, 2.5) = 1 - (1 - x)^2.5"),
        );
        assert_close(
            regularized_incomplete_beta(x, 2.0, 2.0),
            3.0 * x * x - 2.0 * x * x * x,
            1e-9,
            &format!("I_{x}(2, 2) = 3x^2 - 2x^3"),
        );
    }
}

#[test]
fn stats_compute_stats_hand_computed_sample() {
    // Sorted: 2 4 4 4 5 5 7 9. Mean 5; squared deviations sum to 32, so the sample variance
    // is 32 / 7. Percentile ranks (n - 1 = 7): median 3.5 -> 4.5, Q1 1.75 -> 4, Q3 5.25 -> 5.5.
    let (mean, std_dev, median, q1, q3) = compute_stats(&[5.0, 2.0, 9.0, 4.0, 7.0, 4.0, 5.0, 4.0]);
    assert_close(mean, 5.0, 1e-12, "mean");
    assert_close(std_dev, (32.0_f64 / 7.0).sqrt(), 1e-12, "std dev");
    assert_close(median, 4.5, 1e-12, "median");
    assert_close(q1, 4.0, 1e-12, "Q1");
    assert_close(q3, 5.5, 1e-12, "Q3");
}

#[test]
fn stats_welch_t_test_hand_computed_samples() {
    // a: mean 3, variance 2.5; b: mean 6, variance 10. se^2 = 2.5/5 + 10/5 = 2.5, so
    // t = -3 / sqrt(2.5), and df = 2.5^2 / (0.5^2 / 4 + 2^2 / 4) = 6.25 / 1.0625.
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 4.0, 6.0, 8.0, 10.0];
    let (mean_a, std_a, ..) = compute_stats(&a);
    let (mean_b, std_b, ..) = compute_stats(&b);
    let (t, df, p) = welch_t_test(&a, &b, mean_a, mean_b, std_a, std_b);
    assert_close(t, -3.0 / 2.5_f64.sqrt(), 1e-12, "t");
    assert_close(df, 6.25 / 1.0625, 1e-12, "df");
    assert!((0.05..0.2).contains(&p), "p = {p}");

    // Identical constant samples: no difference at all.
    let c = [1.0; 4];
    assert_eq!(welch_t_test(&c, &c, 1.0, 1.0, 0.0, 0.0), (0.0, 6.0, 1.0));
}
