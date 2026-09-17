//! Headless timing of [`PhysicsWorld::step`] on the ten 2D stress scenes, for A/B comparisons
//! between two builds of rapier (a branch against the commit it is based on). It uses only the
//! public API of `step`, so the same file runs unchanged on both sides.
//! Each timed step is appended as one CSV row to `STEP_TIMING_OUT`; a driver script
//! interleaves runs of the two binaries and compares the samples.
//!
//! ```text
//! STEP_TIMING_OUT=/tmp/ab.csv STEP_TIMING_LABEL=branch \
//!     cargo test -p rapier2d --release --test step_timing -- --ignored --nocapture
//! ```
//!
//! Knobs (environment variables): `STEP_TIMING_ITERS` (timed steps per scene, default
//! 500), `STEP_TIMING_WARMUP` (untimed steps, default 20), `STEP_TIMING_SCENES`
//! (comma-separated, case-insensitive substring filter), `STEP_TIMING_OUT` (CSV path,
//! appended to; unset: no file), `STEP_TIMING_LABEL` (the `build` column, default
//! `unlabeled`). The CSV columns are `build,scene,step,ms`.
//!
//! Run it on an idle machine, in release mode.

use std::io::Write;
use std::time::Instant;

use rapier2d::prelude::*;

struct Knobs {
    iters: usize,
    warmup: usize,
    scenes: Option<Vec<String>>,
    out: Option<String>,
    label: String,
}

impl Knobs {
    fn from_env() -> Self {
        let scenes = std::env::var("STEP_TIMING_SCENES")
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
            iters: env_usize("STEP_TIMING_ITERS", 500, 2),
            warmup: env_usize("STEP_TIMING_WARMUP", 20, 0),
            scenes,
            out: std::env::var("STEP_TIMING_OUT")
                .ok()
                .filter(|path| !path.trim().is_empty()),
            label: std::env::var("STEP_TIMING_LABEL")
                .ok()
                .filter(|label| !label.trim().is_empty())
                .unwrap_or_else(|| "unlabeled".to_string()),
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

type SceneBuilder = fn() -> PhysicsWorld;

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
];

/// Copy of `examples2d/stress_tests/balls2.rs`.
fn balls() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/boxes2.rs`.
fn boxes() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/capsules2.rs`.
fn capsules() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/convex_polygons2.rs`, except for the random generator. The
/// example uses `rand::rngs::StdRng`, which is not a dependency of `rapier2d`, so the points come
/// from the `oorandom` dev-dependency (same seed, same sampling order, different values). The
/// scene structure is unchanged: 5 random convex hulls of 10 points, reused over 26×130 bodies.
fn convex_polygons() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/heightfield2.rs`.
fn heightfield() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/pyramid2.rs`.
fn pyramid() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/vertical_stacks2.rs`.
fn vertical_stacks() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/joint_ball2.rs`.
fn joint_ball() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/joint_fixed2.rs`.
fn joint_fixed() -> PhysicsWorld {
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

    world
}

/// Copy of `examples2d/stress_tests/joint_prismatic2.rs`.
fn joint_prismatic() -> PhysicsWorld {
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

    world
}

#[test]
#[ignore = "timing-based perf check; run manually in release"]
fn step_timing() {
    let knobs = Knobs::from_env();
    let selected: Vec<_> = SCENES
        .iter()
        .filter(|(name, _)| knobs.selects(name))
        .collect();
    assert!(!selected.is_empty(), "STEP_TIMING_SCENES matches no scene");
    let mut csv = knobs.out.as_ref().map(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("cannot open {path}: {e}"))
    });

    println!("build: {}", knobs.label);
    for &&(name, build) in &selected {
        let mut world = build();
        let mut samples = Vec::with_capacity(knobs.iters);
        for step in 0..knobs.warmup + knobs.iters {
            let start = Instant::now();
            world.step();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if step >= knobs.warmup {
                samples.push(ms);
            }
        }
        for (handle, body) in world.bodies.iter() {
            assert!(
                body.translation().is_finite() && body.rotation().angle().is_finite(),
                "{name}: body {handle:?} has a non-finite pose"
            );
        }
        assert!(
            world.quarantine().is_empty(),
            "{name}: non-finite state was quarantined"
        );

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        println!(
            "{name:32} N = {:4}  mean {mean:9.4} ms  median {median:9.4} ms",
            samples.len()
        );
        if let Some(csv) = csv.as_mut() {
            for (step, ms) in samples.iter().enumerate() {
                writeln!(csv, "{},{},{},{}", knobs.label, name, step, ms)
                    .expect("cannot write the CSV");
            }
        }
    }
}
