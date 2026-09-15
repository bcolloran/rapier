//! Contact adhesion (3D): clustered multi-manifold pairs, the area-based tangential extent and
//! the 3D lever arms.
//!
//! In 3D a box against a triangle mesh produces one manifold per triangle; contact clustering
//! merges them into cluster manifolds, and those clusters are what the solver (and therefore
//! adhesion) sees. The requested adhesion must be applied once per cluster, and the
//! composition-invariant kinds (`adhesion_pressure`, `adhesion_budget`) must give a tiled mesh
//! ceiling the same hold/break threshold as one monolithic cuboid.
//!
//! Every scenario runs in both stepping modes (see [`Mode`]): stock `step` and
//! `step_collisions_last`.

use std::sync::Mutex;

use rapier3d::prelude::*;

const G: Real = 9.81;

/// How a test advances its world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `PhysicsWorld::step_with_events`.
    Stock,
    /// `PhysicsWorld::step_collisions_last_with_events`, initialized before the first step.
    CollisionsLast,
}

/// Generates one `#[test]` per stepping mode for `fn $name(mode: Mode)`, in a module named after
/// the function, so a failure reads `$name::stock` or `$name::collisions_last`.
macro_rules! both_modes {
    ($name:ident) => {
        mod $name {
            #[test]
            fn stock() {
                super::$name(super::Mode::Stock);
            }

            #[test]
            fn collisions_last() {
                super::$name(super::Mode::CollisionsLast);
            }
        }
    };
}

/// Advances `world` by one step in `mode`. Every test steps through here.
fn step(mode: Mode, world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks) {
    match mode {
        Mode::Stock => world.step_with_events(hooks, &()),
        Mode::CollisionsLast => {
            if !world.physics_pipeline.collisions_last_initialized() {
                world.initialize_collisions_last_with_events(hooks, &());
            }
            world.step_collisions_last_with_events(hooks, &());
        }
    }
}

fn run(mode: Mode, world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks, steps: usize) {
    for _ in 0..steps {
        step(mode, world, hooks);
    }
}

fn new_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -G, 0.0);
    world
}

#[derive(Clone, Copy, Debug)]
enum Request {
    Force(Real),
    Pressure(Real),
    Budget(Real),
}

/// Requests `request` on every manifold involving `collider`.
struct AdhesionHook {
    collider: ColliderHandle,
    request: Request,
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        if context.collider1 != self.collider && context.collider2 != self.collider {
            return;
        }
        match self.request {
            Request::Force(f) => *context.adhesion_force = f,
            Request::Pressure(p) => *context.adhesion_pressure = p,
            Request::Budget(total) => {
                *context.adhesion_budget = Some(AdhesionBudget {
                    owner: self.collider,
                    channel: 0,
                    total,
                })
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ceiling {
    /// One thick cuboid whose bottom face is the plane y = 0.
    Cuboid,
    /// A flat 4x4-quad triangle mesh in the plane y = 0, facing down: a box under it touches
    /// several triangles, i.e. several manifolds of one contact pair.
    Trimesh,
}

/// A hook-enabled fixed ceiling whose underside is the plane y = 0, and a 1x1x1 dynamic box
/// (mass 1) whose top face touches it, with a constant downward `load` on the box.
/// Returns `(ceiling collider, box body, box collider)`.
fn ceiling_and_hanging_box(
    world: &mut PhysicsWorld,
    ceiling: Ceiling,
    load: Real,
) -> (ColliderHandle, RigidBodyHandle, ColliderHandle) {
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let builder = match ceiling {
        Ceiling::Cuboid => {
            ColliderBuilder::cuboid(5.0, 0.5, 5.0).translation(Vector::new(0.0, 0.5, 0.0))
        }
        Ceiling::Trimesh => {
            const N: u32 = 4;
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            for i in 0..=N {
                for j in 0..=N {
                    vertices.push(Vector::new(
                        i as Real - N as Real / 2.0,
                        0.0,
                        j as Real - N as Real / 2.0,
                    ));
                }
            }
            for i in 0..N {
                for j in 0..N {
                    let a = i * (N + 1) + j;
                    let b = a + 1;
                    let c = a + (N + 1);
                    let d = c + 1;
                    // Wound so the face normals point down (-y), toward the box.
                    indices.push([a, c, b]);
                    indices.push([b, c, d]);
                }
            }
            ColliderBuilder::trimesh_with_flags(vertices, indices, TriMeshFlags::FIX_INTERNAL_EDGES)
                .unwrap()
        }
    };
    let ceiling_co = world.insert_collider(
        builder.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );

    let (box_body, box_co) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, -0.5, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    world.bodies[box_body].add_force(Vector::new(0.0, -load, 0.0), true);
    (ceiling_co, box_body, box_co)
}

/// Where the box ends up after `steps`, plus what the contact pair looked like after the first
/// step: `(final y, plain manifold count, cluster count, solver-manifold extents)`.
fn hang(
    mode: Mode,
    ceiling: Ceiling,
    request: Request,
    load: Real,
    steps: usize,
) -> (Real, usize, usize, Vec<Real>) {
    let mut world = new_world();
    let (ceiling_co, box_body, box_co) = ceiling_and_hanging_box(&mut world, ceiling, load);
    let hook = AdhesionHook {
        collider: ceiling_co,
        request,
    };
    step(mode, &mut world, &hook);
    let pair = world
        .narrow_phase
        .contact_pair(ceiling_co, box_co)
        .expect("ceiling/box contact pair");
    let manifolds = pair.manifolds.len();
    let clusters = pair.solver_clusters.len();
    let extents = pair
        .solver_manifolds()
        .iter()
        .filter(|m| m.data.num_active_contacts() > 0)
        .map(|m| m.data.tangential_extent())
        .collect();
    run(mode, &mut world, &hook, steps - 1);
    (
        world.bodies[box_body].translation().y,
        manifolds,
        clusters,
        extents,
    )
}

fn clustered_mesh_ceiling_matches_monolithic_hold_and_break_threshold(mode: Mode) {
    // The box weighs w ~= 9.81 N. A request worth w + 3 N over the 1 m² face holds a 1 N load
    // (w + 1 < w + 3) and breaks under a 5 N load (w + 5 > w + 3), whether the ceiling is one
    // cuboid or a tiled mesh clustered from many per-triangle manifolds.
    let w = G;
    for request in [Request::Pressure(w + 3.0), Request::Budget(w + 3.0)] {
        for ceiling in [Ceiling::Cuboid, Ceiling::Trimesh] {
            let (y, manifolds, clusters, extents) = hang(mode, ceiling, request, 1.0, 240);
            if ceiling == Ceiling::Trimesh {
                assert!(
                    manifolds > 1 && clusters == 1,
                    "the mesh ceiling should be one cluster over several manifolds \
                     ({manifolds} manifolds, {clusters} clusters)"
                );
            }
            assert!(
                extents.len() == 1 && (extents[0] - 1.0).abs() < 0.05,
                "{ceiling:?}: one solver manifold spanning the 1 m² face expected, got extents {extents:?}"
            );
            assert!(
                y > -0.6,
                "{ceiling:?} {request:?}: box should hold a 1 N load (y = {y})"
            );

            let (y, ..) = hang(mode, ceiling, request, 5.0, 240);
            assert!(
                y < -1.0,
                "{ceiling:?} {request:?}: box should break free under a 5 N load (y = {y})"
            );
        }
    }
}
both_modes!(clustered_mesh_ceiling_matches_monolithic_hold_and_break_threshold);

fn adhesion_force_on_a_clustered_pair_is_applied_once(mode: Mode) {
    // `adhesion_force` is per solver manifold. On the clustered mesh ceiling the solver sees one
    // cluster, so w + 3 N is applied once and breaks under a 5 N load, exactly like on the
    // cuboid. Applying it through the per-triangle manifolds too would multiply the pull and
    // keep the box hanging.
    let w = G;
    for ceiling in [Ceiling::Cuboid, Ceiling::Trimesh] {
        let (y, ..) = hang(mode, ceiling, Request::Force(w + 3.0), 1.0, 240);
        assert!(
            y > -0.6,
            "{ceiling:?}: box should hold a 1 N load (y = {y})"
        );
        let (y, manifolds, clusters, _) = hang(mode, ceiling, Request::Force(w + 3.0), 5.0, 240);
        if ceiling == Ceiling::Trimesh {
            assert!(manifolds > 1 && clusters == 1);
        }
        assert!(
            y < -1.0,
            "{ceiling:?}: a w + 3 N force must not hold a 5 N load (y = {y})"
        );
    }
}
both_modes!(adhesion_force_on_a_clustered_pair_is_applied_once);

fn face_contact_extent_is_about_one_square_meter(mode: Mode) {
    // A 1x1x1 box resting on a floor: its four face corners span 1 m², both inside the hook and
    // in the value cached on the manifold afterwards.
    struct Recorder {
        collider: ColliderHandle,
        seen: Mutex<Vec<Real>>,
    }
    impl PhysicsHooks for Recorder {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            if context.collider1 == self.collider || context.collider2 == self.collider {
                self.seen.lock().unwrap().push(context.tangential_extent());
            }
        }
    }

    let mut world = new_world();
    let floor_body = world.insert_body(RigidBodyBuilder::fixed());
    let floor = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5, 5.0).translation(Vector::new(0.0, -0.5, 0.0)),
        Some(floor_body),
    );
    let (_, box_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.5, 0.0)),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5).active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
    );
    let hook = Recorder {
        collider: box_co,
        seen: Mutex::new(Vec::new()),
    };
    run(mode, &mut world, &hook, 30);

    let seen = hook.seen.lock().unwrap().clone();
    let last = *seen.last().expect("the hook never ran");
    assert!(
        (last - 1.0).abs() < 1.0e-3,
        "extent of a 1x1 face contact should be ~1 m², got {last}"
    );
    let pair = world.narrow_phase.contact_pair(floor, box_co).unwrap();
    let manifold = &pair.solver_manifolds()[0];
    assert_eq!(manifold.data.num_active_contacts(), 4);
    assert_eq!(manifold.data.tangential_extent(), last);
}
both_modes!(face_contact_extent_is_about_one_square_meter);

fn adhesion_applies_no_net_torque_off_center(mode: Mode) {
    // The 3D counterpart of the 2D `adhesion_applies_no_net_torque`, off center along both
    // horizontal axes. Adhesion is an internal action-reaction pair, so it must add zero net
    // torque to the system it acts within; that holds only if each side's torque uses its own
    // lever arm to the contact. Rig: a 10 x 1 x 4 m plank hung from a joint 3 m above its center
    // that lets it tilt about both horizontal axes (a stable pendulum about each) but not turn
    // about the vertical one, which nothing restores, with a capsule standing on each of
    // two diagonally opposite spots, (-4, +1) and (+4, -1) in (x, z), so their weights balance
    // about both axes. Only the first capsule adheres (20 N, ~2x its weight). A zero or wrong
    // lever arm on either body leaves an unbalanced torque of up to 20 N * 4 m about z and
    // 20 N * 1 m about x, which would settle the plank at a visible tilt; the 0.002 rad
    // threshold is the 2D test's. Measured in both modes: 1.3e-6 rad without adhesion, 7.9e-6 rad
    // with it. The rotation about the vertical axis is locked because this load pattern turns a
    // plank that is free to yaw slowly (~8e-4 rad/s) whether the pull comes from adhesion or from
    // an equivalent pair of user forces, with or without contact recycling: that is the rig, not
    // the adhesion.
    let settle = |adhesion: Option<Real>| -> Real {
        let mut world = new_world();
        let pivot =
            world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 3.0, 0.0)));
        let plank_body = world.insert_body(
            RigidBodyBuilder::dynamic()
                .angular_damping(2.0) // settle oscillations so the final angle is read at rest
                .can_sleep(false),
        );
        world.insert_collider(
            ColliderBuilder::cuboid(5.0, 0.5, 2.0)
                .mass(1.0)
                .friction(2.0),
            Some(plank_body),
        );
        // A ball joint that also locks the rotation about the vertical axis, which gravity
        // doesn't restore (see the comment above).
        world.insert_impulse_joint(
            pivot,
            plank_body,
            GenericJointBuilder::new(JointAxesMask::LIN_AXES | JointAxesMask::ANG_Y)
                .local_anchor2(Vector::new(0.0, 3.0, 0.0)),
        );

        let mut adhesive_capsule = None;
        for side in [-1.0 as Real, 1.0] {
            let cap_body = world.insert_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(side * 4.0, 1.5, -side))
                    .lock_rotations()
                    .can_sleep(false),
            );
            let adheres = side < 0.0 && adhesion.is_some();
            let mut capsule = ColliderBuilder::capsule_y(0.5, 0.5).mass(1.0).friction(2.0);
            if adheres {
                capsule = capsule.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
            }
            let capsule = world.insert_collider(capsule, Some(cap_body));
            if adheres {
                adhesive_capsule = Some(capsule);
            }
        }

        match (adhesion, adhesive_capsule) {
            (Some(force), Some(capsule)) => {
                let hook = AdhesionHook {
                    collider: capsule,
                    request: Request::Force(force),
                };
                run(mode, &mut world, &hook, 600);
            }
            _ => run(mode, &mut world, &(), 600),
        }
        // The rotation angle, precise for small angles (unlike an `acos` of the quaternion).
        world.bodies[plank_body]
            .rotation()
            .to_scaled_axis()
            .length()
    };

    let level = settle(None);
    let adhering = settle(Some(20.0));
    assert!(level < 0.002, "control plank settled tilted ({level} rad)");
    assert!(
        adhering < 0.002,
        "adhesion applied a steady net torque: plank settled at {adhering} rad"
    );
}
both_modes!(adhesion_applies_no_net_torque_off_center);
