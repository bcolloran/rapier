//! Headless timing of [`PhysicsWorld::step`] on the eleven 3D stress scenes, for A/B comparisons
//! between two builds of rapier (a branch against the commit it is based on). It uses only the
//! public API of `step`, so the same file runs unchanged on both sides.
//! Each timed step is appended as one CSV row to `STEP_TIMING_OUT`; a driver script
//! interleaves runs of the two binaries and compares the samples.
//!
//! ```text
//! STEP_TIMING_OUT=/tmp/ab.csv STEP_TIMING_LABEL=branch \
//!     cargo test -p rapier3d --release --test step_timing -- --ignored --nocapture
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

use rapier3d::na::ComplexField;
use rapier3d::prelude::*;

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
    ("Convex polyhedron", convex_polyhedron),
    ("Heightfield", heightfield),
    ("Stacks", stacks),
    ("Pyramid", pyramid),
    ("Trimesh", trimesh),
    ("ImpulseJoint ball", joint_ball),
    ("ImpulseJoint fixed", joint_fixed),
    ("ImpulseJoint prismatic", joint_prismatic),
];

/// Copy of `examples3d/stress_tests/balls3.rs`.
fn balls() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let num = 20;
    let rad = 1.0;

    let shift = rad * 2.0 + 1.0;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;
    let centerz = shift * (num as f32) / 2.0;

    for i in 0..num {
        for j in 0usize..num {
            for k in 0..num {
                let x = i as f32 * shift - centerx;
                let y = j as f32 * shift + centery;
                let z = k as f32 * shift - centerz;

                let status = if j == 0 {
                    RigidBodyType::Fixed
                } else {
                    RigidBodyType::Dynamic
                };
                let density = 0.477;

                let rigid_body = RigidBodyBuilder::new(status).translation(Vec3::new(x, y, z));
                let collider = ColliderBuilder::ball(rad).density(density);
                let _ = world.insert(rigid_body, collider);
            }
        }
    }

    world
}

/// Copy of `examples3d/stress_tests/boxes3.rs`.
fn boxes() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let ground_size = 200.1;
    let ground_height = 0.1;

    let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height, 0.0));
    let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
    let _ = world.insert(rigid_body, collider);

    let num = 10;
    let rad = 1.0;

    let shift = rad * 2.0;
    let centerx = shift * (num / 2) as f32;
    let centery = shift / 2.0;
    let centerz = shift * (num / 2) as f32;

    let mut offset = -(num as f32) * (rad * 2.0) * 0.5;

    for j in 0usize..num {
        for i in 0..num {
            for k in 0usize..num {
                let x = i as f32 * shift - centerx + offset;
                let y = j as f32 * shift + centery;
                let z = k as f32 * shift - centerz + offset;

                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
                let collider = ColliderBuilder::cuboid(rad, rad, rad);
                let _ = world.insert(rigid_body, collider);
            }
        }

        offset -= 0.05 * rad * (num as f32 - 1.0);
    }

    world
}

/// Copy of `examples3d/stress_tests/capsules3.rs`.
fn capsules() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let ground_size = 200.1;
    let ground_height = 0.1;

    let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height, 0.0));
    let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
    let _ = world.insert(rigid_body, collider);

    let num = 8;
    let rad = 1.0;

    let shift = rad * 2.0 + rad;
    let shifty = rad * 4.0;
    let centerx = shift * (num / 2) as f32;
    let centery = shift / 2.0;
    let centerz = shift * (num / 2) as f32;

    let mut offset = -(num as f32) * (rad * 2.0 + rad) * 0.5;

    for j in 0usize..47 {
        for i in 0..num {
            for k in 0usize..num {
                let x = i as f32 * shift - centerx + offset;
                let y = j as f32 * shifty + centery + 3.0;
                let z = k as f32 * shift - centerz + offset;

                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
                let collider = ColliderBuilder::capsule_y(rad, rad);
                let _ = world.insert(rigid_body, collider);
            }
        }

        offset -= 0.05 * rad * (num as f32 - 1.0);
    }

    world
}

/// Copy of `examples3d/stress_tests/convex_polyhedron3.rs`, except for the random generator. The
/// example uses `rand::rngs::StdRng`, which is not a dependency of `rapier3d`, so the points come
/// from the `oorandom` dev-dependency (same seed, same sampling order, different values). The
/// scene structure is unchanged: 5 random round convex hulls of 10 points, reused over 47×8×8
/// bodies.
fn convex_polyhedron() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let ground_size = 200.1;
    let ground_height = 0.1;

    let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height, 0.0));
    let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
    let _ = world.insert(rigid_body, collider);

    let num = 8;
    let scale = 2.0;
    let rad = 1.0;
    let border_rad = 0.1;

    let shift = border_rad * 2.0 + scale;
    let centerx = shift * (num / 2) as f32;
    let centery = shift / 2.0;
    let centerz = shift * (num / 2) as f32;

    let mut offset = -(num as f32) * shift * 0.5;

    let mut rng = oorandom::Rand32::new(0);

    let poly_shape: Vec<_> = (0..5)
        .map(|_| {
            let mut points = Vec::new();
            for _ in 0..10 {
                let pt = Vec3::new(rng.rand_float(), rng.rand_float(), rng.rand_float());
                points.push(pt * scale);
            }
            SharedShape::round_convex_hull(&points, border_rad).unwrap()
        })
        .collect();

    for j in 0usize..47 {
        for i in 0..num {
            for k in 0usize..num {
                let x = i as f32 * shift - centerx + offset;
                let y = j as f32 * shift + centery + 3.0;
                let z = k as f32 * shift - centerz + offset;

                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
                let collider = ColliderBuilder::new(poly_shape[(i + k) % 5].clone());
                let _ = world.insert(rigid_body, collider);
            }
        }

        offset -= 0.05 * rad * (num as f32 - 1.0);
    }

    world
}

/// The heightfield of `examples3d/stress_tests/heightfield3.rs` and `trimesh3.rs`.
fn stress_heights(ground_size: Vec3, nsubdivs: usize) -> Array2<f32> {
    Array2::from_fn(nsubdivs + 1, nsubdivs + 1, |i, j| {
        if i == 0 || i == nsubdivs || j == 0 || j == nsubdivs {
            10.0
        } else {
            let x = i as f32 * ground_size.x / (nsubdivs as f32);
            let z = j as f32 * ground_size.z / (nsubdivs as f32);

            // NOTE: make sure we use the sin/cos from simba to ensure
            // cross-platform determinism of the example when the
            // enhanced_determinism feature is enabled.
            <f32 as ComplexField>::sin(x) + <f32 as ComplexField>::cos(z)
        }
    })
}

/// The 47×8×8 pile of alternating cuboid and ball layers of `examples3d/stress_tests/heightfield3.rs`
/// and `trimesh3.rs`.
fn stress_pile(world: &mut PhysicsWorld) {
    let num = 8;
    let rad = 1.0;

    let shift = rad * 2.0 + rad;
    let centerx = shift * (num / 2) as f32;
    let centery = shift / 2.0;
    let centerz = shift * (num / 2) as f32;

    for j in 0usize..47 {
        for i in 0..num {
            for k in 0usize..num {
                let x = i as f32 * shift - centerx;
                let y = j as f32 * shift + centery + 3.0;
                let z = k as f32 * shift - centerz;

                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));

                if j % 2 == 0 {
                    let collider = ColliderBuilder::cuboid(rad, rad, rad);
                    let _ = world.insert(rigid_body, collider);
                } else {
                    let collider = ColliderBuilder::ball(rad);
                    let _ = world.insert(rigid_body, collider);
                }
            }
        }
    }
}

/// Copy of `examples3d/stress_tests/heightfield3.rs`.
fn heightfield() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let ground_size = Vec3::new(200.0, 1.0, 200.0);
    let heights = stress_heights(ground_size, 20);

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::heightfield(heights, ground_size);
    let _ = world.insert(rigid_body, collider);

    stress_pile(&mut world);

    world
}

/// Copy of `examples3d/stress_tests/trimesh3.rs`: the same pile on the heightfield converted to
/// a triangle mesh.
fn trimesh() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let ground_size = Vec3::new(200.0, 1.0, 200.0);
    let heightfield = HeightField::new(stress_heights(ground_size, 20), ground_size);
    let (vertices, indices) = heightfield.to_trimesh();

    let rigid_body = RigidBodyBuilder::fixed();
    let collider = ColliderBuilder::trimesh(vertices, indices).unwrap();
    let _ = world.insert(rigid_body, collider);

    stress_pile(&mut world);

    world
}

/// Copy of `examples3d/stress_tests/pyramid3.rs`: 50 layers of shrunken 1.95-cubes on a 2.25
/// pitch with a 1.0 brick offset, about 43k bodies.
fn pyramid() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81, 0.0);

    let _ = world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -1.0, 0.0)),
        ColliderBuilder::cuboid(100.0, 1.0, 100.0),
    );

    let pyramid_height = 50i32;
    let box_size = 2.0;
    let box_separation = 0.5;
    let half_box_size = 0.5 * box_size;
    let h = half_box_size - 0.025;

    for i in 0..pyramid_height {
        let brick = if i & 1 != 0 { half_box_size } else { 0.0 };
        let y = 1.0 + (box_size + box_separation) * i as f32;

        for j in i / 2..pyramid_height - (i + 1) / 2 {
            for k in i / 2..pyramid_height - (i + 1) / 2 {
                let x = -(pyramid_height as f32) + (box_size + 0.25) * j as f32 + brick;
                let z = -(pyramid_height as f32) + (box_size + 0.25) * k as f32 + brick;

                let _ = world.insert(
                    RigidBodyBuilder::dynamic().translation(Vector::new(x, y, z)),
                    ColliderBuilder::cuboid(h, h, h).density(1000.0),
                );
            }
        }
    }

    world
}

/// Helper of `examples3d/stress_tests/stacks3.rs`.
fn create_tower_circle(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    offset: Vec3,
    stack_height: usize,
    nsubdivs: usize,
    half_extents: Vec3,
) {
    let ang_step = std::f32::consts::PI * 2.0 / nsubdivs as f32;
    let radius = 1.3 * nsubdivs as f32 * half_extents.x / std::f32::consts::PI;

    let shift = half_extents * 2.0;
    for i in 0usize..stack_height {
        for j in 0..nsubdivs {
            let fj = j as f32;
            let fi = i as f32;
            let y = fi * shift.y;
            let pos = Pose3::new(offset, Vec3::Y * (fi / 2.0 + fj) * ang_step)
                .prepend_translation(Vec3::new(0.0, y, radius));

            let rigid_body = RigidBodyBuilder::dynamic().pose(pos);
            let handle = bodies.insert(rigid_body);
            let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z);
            colliders.insert_with_parent(collider, handle, bodies);
        }
    }
}

/// Helper of `examples3d/stress_tests/stacks3.rs`.
fn create_wall(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    offset: Vec3,
    stack_height: usize,
    half_extents: Vec3,
) {
    let shift = half_extents * 2.0;
    for i in 0usize..stack_height {
        for j in i..stack_height {
            let fj = j as f32;
            let fi = i as f32;
            let x = offset.x;
            let y = fi * shift.y + offset.y;
            let z = (fi * shift.z / 2.0) + (fj - fi) * shift.z + offset.z
                - stack_height as f32 * half_extents.z;

            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
            let handle = bodies.insert(rigid_body);
            let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z);
            colliders.insert_with_parent(collider, handle, bodies);
        }
    }
}

/// Helper of `examples3d/stress_tests/stacks3.rs`.
fn create_pyramid(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    offset: Vec3,
    stack_height: usize,
    half_extents: Vec3,
) {
    let shift = half_extents * 2.0;

    for i in 0usize..stack_height {
        for j in i..stack_height {
            for k in i..stack_height {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                let x = (fi * shift.x / 2.0) + (fk - fi) * shift.x + offset.x
                    - stack_height as f32 * half_extents.x;
                let y = fi * shift.y + offset.y;
                let z = (fi * shift.z / 2.0) + (fj - fi) * shift.z + offset.z
                    - stack_height as f32 * half_extents.z;

                let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
                let handle = bodies.insert(rigid_body);
                let collider =
                    ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z);
                colliders.insert_with_parent(collider, handle, bodies);
            }
        }
    }
}

/// Copy of `examples3d/stress_tests/stacks3.rs`.
fn stacks() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let ground_size = 200.0;
    let ground_height = 0.1;

    let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height, 0.0));
    let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
    let _ = world.insert(rigid_body, collider);

    let cube_size = 1.0;
    let hext = Vec3::splat(cube_size);
    let bottomy = cube_size * 50.0;
    for x in [-110.0, -80.0, -50.0, -20.0] {
        create_pyramid(
            &mut world.bodies,
            &mut world.colliders,
            Vec3::new(x, bottomy, 0.0),
            12,
            hext,
        );
    }
    for x in [-2.0, 4.0, 10.0] {
        create_wall(
            &mut world.bodies,
            &mut world.colliders,
            Vec3::new(x, bottomy, 0.0),
            12,
            hext,
        );
    }
    create_tower_circle(
        &mut world.bodies,
        &mut world.colliders,
        Vec3::new(25.0, bottomy, 0.0),
        8,
        24,
        hext,
    );

    world
}

/// Copy of `examples3d/stress_tests/joint_ball3.rs`.
fn joint_ball() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let rad = 0.4;
    let num = 100;
    let shift = 1.0;

    let mut body_handles = Vec::new();

    for k in 0..num {
        for i in 0..num {
            let fk = k as f32;
            let fi = i as f32;

            let status = if i == 0 && (k % 4 == 0 || k == num - 1) {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };

            let rigid_body =
                RigidBodyBuilder::new(status).translation(Vec3::new(fk * shift, 0.0, fi * shift));
            let collider = ColliderBuilder::ball(rad);
            let (child_handle, _) = world.insert(rigid_body, collider);

            // Vertical joint.
            if i > 0 {
                let parent_handle = *body_handles.last().unwrap();
                let joint = SphericalJointBuilder::new().local_anchor2(Vec3::new(0.0, 0.0, -shift));
                world.insert_impulse_joint(parent_handle, child_handle, joint);
            }

            // Horizontal joint.
            if k > 0 {
                let parent_index = body_handles.len() - num;
                let parent_handle = body_handles[parent_index];
                let joint = SphericalJointBuilder::new().local_anchor2(Vec3::new(-shift, 0.0, 0.0));
                world.insert_impulse_joint(parent_handle, child_handle, joint);
            }

            body_handles.push(child_handle);
        }
    }

    world
}

/// Copy of `examples3d/stress_tests/joint_fixed3.rs`.
fn joint_fixed() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let rad = 0.4;
    let num = 5;
    let shift = 1.0;

    let mut body_handles = Vec::new();

    for m in 0..10 {
        let z = m as f32 * shift * (num as f32 + 2.0);

        for l in 0..10 {
            let y = l as f32 * shift * 3.0;

            for j in 0..5 {
                let x = j as f32 * shift * (num as f32) * 2.0;

                for k in 0..num {
                    for i in 0..num {
                        let fk = k as f32;
                        let fi = i as f32;

                        // NOTE: the num - 2 test is to avoid two consecutive
                        // fixed bodies. Because physx will crash if we add
                        // a joint between these.

                        let status = if i == 0 && (k % 4 == 0 && k != num - 2 || k == num - 1) {
                            RigidBodyType::Fixed
                        } else {
                            RigidBodyType::Dynamic
                        };

                        let rigid_body = RigidBodyBuilder::new(status).translation(Vec3::new(
                            x + fk * shift,
                            y,
                            z + fi * shift,
                        ));
                        let collider = ColliderBuilder::ball(rad);
                        let (child_handle, _) = world.insert(rigid_body, collider);

                        // Vertical joint.
                        if i > 0 {
                            let parent_handle = *body_handles.last().unwrap();
                            let joint =
                                FixedJointBuilder::new().local_anchor2(Vec3::new(0.0, 0.0, -shift));
                            world.insert_impulse_joint(parent_handle, child_handle, joint);
                        }

                        // Horizontal joint.
                        if k > 0 {
                            let parent_index = body_handles.len() - num;
                            let parent_handle = body_handles[parent_index];
                            let joint =
                                FixedJointBuilder::new().local_anchor2(Vec3::new(-shift, 0.0, 0.0));
                            world.insert_impulse_joint(parent_handle, child_handle, joint);
                        }

                        body_handles.push(child_handle);
                    }
                }
            }
        }
    }

    world
}

/// Copy of `examples3d/stress_tests/joint_prismatic3.rs`.
fn joint_prismatic() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();

    let rad = 0.4;
    let num = 5;
    let shift = 1.0;

    for m in 0..8 {
        let z = m as f32 * shift * (num as f32 + 2.0);

        for l in 0..8 {
            let y = l as f32 * shift * (num as f32) * 2.0;

            for j in 0..50 {
                let x = j as f32 * shift * 4.0;

                let ground = RigidBodyBuilder::fixed().translation(Vec3::new(x, y, z));
                let collider = ColliderBuilder::cuboid(rad, rad, rad);
                let (mut curr_parent, _) = world.insert(ground, collider);

                for i in 0..num {
                    let z = z + (i + 1) as f32 * shift;
                    let density = 1.0;
                    let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
                    let collider = ColliderBuilder::cuboid(rad, rad, rad).density(density);
                    let (curr_child, _) = world.insert(rigid_body, collider);

                    let axis = if i % 2 == 0 {
                        Vec3::new(1.0, 1.0, 0.0).normalize()
                    } else {
                        Vec3::new(-1.0, 1.0, 0.0).normalize()
                    };

                    let prism = PrismaticJointBuilder::new(axis)
                        .local_anchor2(Vec3::new(0.0, 0.0, -shift))
                        .limits([-2.0, 0.0]);
                    world.insert_impulse_joint(curr_parent, curr_child, prism);

                    curr_parent = curr_child;
                }
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
                body.translation().is_finite() && body.rotation().is_finite(),
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
