//! Adhesion contract exercised in both dimensions and precisions.

#[cfg(feature = "serde-serialize")]
use crate::alloc_prelude::*;
use crate::prelude::*;
use approx::assert_abs_diff_eq;

fn vector(x: Real, y: Real) -> Vector {
    #[cfg(feature = "dim2")]
    {
        Vector::new(x, y)
    }
    #[cfg(feature = "dim3")]
    {
        Vector::new(x, y, 0.0)
    }
}

fn cuboid(x: Real, y: Real) -> ColliderBuilder {
    #[cfg(feature = "dim2")]
    {
        ColliderBuilder::cuboid(x, y)
    }
    #[cfg(feature = "dim3")]
    {
        ColliderBuilder::cuboid(x, y, 0.5)
    }
}

struct Hook<F>(F);

impl<F: Fn(&mut ContactModificationContext) + Send + Sync> PhysicsHooks for Hook<F> {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        (self.0)(context);
    }
}

fn hanging_box(tiles: bool, origin: Vector) -> (PhysicsWorld, RigidBodyHandle, ColliderHandle) {
    let mut world = PhysicsWorld::new();
    let ceiling = world.insert_body(RigidBodyBuilder::fixed().translation(origin));
    for x in if tiles { &[-0.5, 0.5][..] } else { &[0.0][..] } {
        world.insert_collider(
            cuboid(if tiles { 0.5 } else { 2.0 }, 0.5)
                .translation(vector(*x, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling),
        );
    }
    let (body, collider) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(origin + vector(0.0, -0.5))
            .can_sleep(false),
        cuboid(0.5, 0.5).mass(1.0),
    );
    (world, body, collider)
}

fn step(world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks, count: usize) {
    for _ in 0..count {
        world.step_with_events(hooks, &());
        assert!(world.quarantine().is_empty());
    }
}

#[test]
fn force_pressure_and_budget_add_together() {
    for enabled in [
        [true, false, false],
        [false, true, false],
        [false, false, true],
        [true; 3],
    ] {
        let (mut world, body, owner) = hanging_box(false, Vector::ZERO);
        let hook = Hook(|context: &mut ContactModificationContext| {
            let rigid = context.rigid_mut().unwrap();
            *rigid.adhesion_force = if enabled[0] { 4.0 } else { 0.0 };
            *rigid.adhesion_pressure = if enabled[1] { 4.0 } else { 0.0 };
            *rigid.adhesion_budget = enabled[2].then_some(AdhesionBudget {
                owner,
                channel: 0,
                total: 4.0,
            });
        });
        step(&mut world, &hook, 120);
        let y = world.bodies[body].translation().y;
        if enabled == [true; 3] {
            assert!(
                y > -0.6,
                "the combined 12 N request must hold the unit mass: {y}"
            );
        } else {
            assert!(
                y < -1.0,
                "a single 4 N request must not hold the unit mass: {y}"
            );
        }
    }
}

#[test]
fn budget_uses_maximum_request_and_separates_owners() {
    for (separate_owners, load, holds) in
        [(false, 1.0, true), (false, 5.0, false), (true, 5.0, true)]
    {
        let (mut world, body, owner) = hanging_box(true, Vector::ZERO);
        world.bodies[body].add_force(vector(0.0, -load), true);
        let hook = Hook(|context: &mut ContactModificationContext| {
            let surface = if context.collider1 == owner {
                context.collider2
            } else {
                context.collider1
            };
            let total = if world_surface_is_left(context, surface) {
                8.0
            } else {
                12.0
            };
            *context.rigid_mut().unwrap().adhesion_budget = Some(AdhesionBudget {
                owner: if separate_owners { surface } else { owner },
                channel: 0,
                total,
            });
        });
        step(&mut world, &hook, 120);
        let y = world.bodies[body].translation().y;
        assert!(
            if holds { y > -0.6 } else { y < -1.0 },
            "owners={separate_owners}, load={load}, y={y}"
        );
    }
}

fn world_surface_is_left(context: &ContactModificationContext, surface: ColliderHandle) -> bool {
    context.colliders[surface].translation().x < 0.0
}

#[test]
fn pressure_holds_a_translated_patch_and_respects_subdivision() {
    for tiles in [false, true] {
        for origin in [Vector::ZERO, Vector::splat(10000.0)] {
            let (mut world, body, _) = hanging_box(tiles, origin);
            let hook = Hook(|context: &mut ContactModificationContext| {
                let rigid = context.rigid_mut().unwrap();
                let extent = rigid.tangential_extent();
                assert!(extent > 0.0 && extent <= 1.01);
                *rigid.adhesion_pressure = 20.0;
            });
            step(&mut world, &hook, 120);
            assert_abs_diff_eq!(
                world.bodies[body].translation().y - origin.y,
                -0.5,
                epsilon = 0.05
            );
            for pair in world.narrow_phase.contact_pairs() {
                for manifold in pair.solver_manifolds() {
                    assert!(manifold.data.tangential_extent(&world.bodies) > 0.0);
                }
            }
        }
    }
}

#[test]
fn hook_withdrawal_resets_every_request_kind() {
    for kind in 0..3 {
        let (mut world, body, owner) = hanging_box(false, Vector::ZERO);
        let hook = Hook(|context: &mut ContactModificationContext| {
            let rigid = context.rigid_mut().unwrap();
            match kind {
                0 => *rigid.adhesion_force = 30.0,
                1 => *rigid.adhesion_pressure = 30.0,
                _ => {
                    *rigid.adhesion_budget = Some(AdhesionBudget {
                        owner,
                        channel: 0,
                        total: 30.0,
                    })
                }
            }
        });
        step(&mut world, &hook, 60);
        assert!(world.bodies[body].translation().y > -0.6);
        // Keep the flag but stop making requests: each callback starts from zero/None.
        step(&mut world, &(), 120);
        assert!(world.bodies[body].translation().y < -1.0);
    }
}

#[test]
fn negative_budget_is_inert() {
    let run = |total| {
        let (mut world, body, owner) = hanging_box(false, Vector::ZERO);
        let hook = Hook(|context: &mut ContactModificationContext| {
            *context.rigid_mut().unwrap().adhesion_budget = Some(AdhesionBudget {
                owner,
                channel: 0,
                total,
            });
        });
        step(&mut world, &hook, 60);
        world.bodies[body].translation().y
    };
    assert_abs_diff_eq!(run(-30.0), run(0.0), epsilon = 1.0e-5);
}

#[test]
fn point_contact_receives_budget_but_no_pressure() {
    for budgeted in [false, true] {
        let (mut world, body, owner) = hanging_box(false, Vector::ZERO);
        world.colliders[owner].set_shape(SharedShape::ball(0.5));
        let hook = Hook(|context: &mut ContactModificationContext| {
            let rigid = context.rigid_mut().unwrap();
            assert_eq!(rigid.tangential_extent(), 0.0);
            if budgeted {
                *rigid.adhesion_budget = Some(AdhesionBudget {
                    owner,
                    channel: 0,
                    total: 30.0,
                });
            } else {
                *rigid.adhesion_pressure = 1000.0;
            }
        });
        step(&mut world, &hook, 120);
        let y = world.bodies[body].translation().y;
        assert!(
            if budgeted { y > -0.6 } else { y < -1.0 },
            "budgeted={budgeted}, y={y}"
        );
    }
}

#[cfg(feature = "serde-serialize")]
#[test]
fn adhesion_snapshot_roundtrip_preserves_requests_and_motion() {
    let (mut world, body, owner) = hanging_box(true, Vector::ZERO);
    let hook = Hook(|context: &mut ContactModificationContext| {
        let rigid = context.rigid_mut().unwrap();
        *rigid.adhesion_force = 2.0;
        *rigid.adhesion_pressure = 3.0;
        *rigid.adhesion_budget = Some(AdhesionBudget {
            owner,
            channel: 7,
            total: 20.0,
        });
    });
    step(&mut world, &hook, 30);
    let bytes = bincode::serialize(&world).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
    let requests: Vec<_> = restored
        .narrow_phase
        .contact_pairs()
        .flat_map(|pair| pair.solver_manifolds())
        .collect();
    assert_eq!(requests.len(), 2);
    for manifold in requests {
        assert_eq!(manifold.data.adhesion_force, 2.0);
        assert_eq!(manifold.data.adhesion_pressure, 3.0);
        assert_eq!(
            manifold.data.adhesion_budget,
            Some(AdhesionBudget {
                owner,
                channel: 7,
                total: 20.0
            })
        );
    }
    step(&mut world, &hook, 60);
    step(&mut restored, &hook, 60);
    assert!(
        (world.bodies[body].translation() - restored.bodies[body].translation()).length() < 1.0e-5
    );
}
