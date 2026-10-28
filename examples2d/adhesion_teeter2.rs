use kiss3d::color::Color;
use rapier_testbed2d::TestbedViewer;
use rapier2d::prelude::*;

/// Teeter-totter torque-balance test: does adhesion perturb an otherwise balanced pivot?
///
/// Each cell is a 10 m × 1 m plank (mass 1 kg) pinned to the world through a revolute joint at
/// its center of mass. Two rotation-locked capsules (2 m tall, 1 m wide, mass 1 kg each) stand on
/// it, one meter in from each end (contact points at ±4 m along the plank). The **left** capsule
/// requests adhesion; the **right** one never does. Friction is high (μ = 2) so nothing slips —
/// slippage is deliberately out of scope.
///
/// Adhesion is an *internal* action-reaction pair: the engine pulls the capsule toward the plank
/// and the plank toward the capsule with equal and opposite forces at the same contact points, so
/// it must contribute **zero net force and zero net torque** to the plank+capsule system. The test
/// criterion is therefore: **within each row, all four columns must move identically**. If an
/// adhesion column diverges from the no-adhesion column, adhesion is leaking momentum.
///
/// - **Columns (x):** no adhesion; `adhesion_force`; `adhesion_pressure`; `adhesion_budget`
///   (left capsule tinted gray / red / blue / green respectively). Note the capsules touch the
///   plank with the *curved bottom cap* — a single-point contact — so the pressure column applies
///   exactly zero force (pressure × zero extent) and must trivially match the no-adhesion column,
///   while force and budget both deliver the full 20 N at the point.
/// - **Rows (y):** initial plank tilt — 0°, 15°, 30°, 45°.
///
/// ⚠ **Read this before judging "it moved!":** only the **flat row is an equilibrium** — and even
/// it is an *unstable* one. A tilted plank with equal masses standing *on top* is NOT balanced,
/// even in perfect physics: the contact points sit half a plank-thickness above the pivot line,
/// so tilting shifts both of them horizontally toward the low side, producing a genuine net
/// torque `m·g·sin θ` (≈ 2.5 N·m at 15°) that tips the plank *further* — the classic instability
/// of carrying mass above the pivot. Measured: the 15° row gains ~0.28 rad within 3 s, in **all
/// four columns alike** (that equality is the pass criterion); once a row tips past the friction
/// slip angle (atan 2 ≈ 63°) the capsules slide off and the cell goes chaotic. The flat row holds
/// for a while, but being an unstable equilibrium it amplifies femto-scale solver asymmetries
/// exponentially (time constant ~2.2 s) and every flat cell visibly tips within ~20–30 s — the
/// adhesion cells a little sooner than the bare one (their settle transient is a slightly larger
/// seed), which is amplified noise, not a steady torque. The decisive check lives in the test
/// suite (`adhesion_applies_no_net_torque`): the same rig hung from a pivot *above* the mass line
/// — a stable pendulum that cannot amplify seeds — settles dead level (≤ 9 µrad) with adhesion
/// active. If your game shows a teeter drifting, compare against a no-adhesion twin before
/// blaming adhesion: pivot-below-mass drift is real physics.
#[derive(Clone, Copy, PartialEq)]
enum AdhesionMode {
    None,
    Force,
    Pressure,
    Budget,
}

struct AdheringCapsule {
    handle: ColliderHandle,
    mode: AdhesionMode,
}

struct TeeterHook {
    capsules: Vec<AdheringCapsule>,
}

impl PhysicsHooks for TeeterHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let (collider1, collider2) = (context.collider1, context.collider2);
        let Some(rigid) = context.rigid_mut() else {
            return;
        };

        for params in &self.capsules {
            if collider1 == params.handle || collider2 == params.handle {
                match params.mode {
                    AdhesionMode::None => {}
                    AdhesionMode::Force => *rigid.adhesion_force = ADHESION,
                    AdhesionMode::Pressure => {
                        // The bottom cap is a point contact: extent = 0, so this applies nothing.
                        // Kept as a column to make that property visible.
                        *rigid.adhesion_pressure = ADHESION / CAP_WIDTH;
                    }
                    AdhesionMode::Budget => {
                        *rigid.adhesion_budget = Some(AdhesionBudget {
                            owner: params.handle,
                            channel: 0,
                            total: ADHESION,
                        });
                    }
                }
                return;
            }
        }
    }
}

const PLANK_HALF_LEN: Real = 5.0; // 10 m long
const PLANK_HALF_THICK: Real = 0.5; // 1 m thick
const PLANK_MASS: Real = 1.0;
const CAP_RADIUS: Real = 0.5; // 1 m wide
const CAP_HALF_HEIGHT: Real = 0.5; // segment 1 m + two 0.5 caps ⇒ 2 m tall
const CAP_MASS: Real = 1.0;
const CAP_STANCE: Real = 4.0; // contact point: 1 m in from each 5 m half-end
const ADHESION: Real = 20.0; // ~2× a capsule's weight
const CAP_WIDTH: Real = 2.0 * CAP_RADIUS;
const FRICTION: Real = 2.0; // high: even the adhesion-free capsule must never slip (μ > tan 45°)
const GRAVITY: Real = 9.81;

/// One cell: pinned plank at `angle_deg` + two standing capsules; the left one adheres via `mode`.
fn add_cell(
    world: &mut PhysicsWorld,
    colors: &mut Vec<(ColliderHandle, Color)>,
    center: Vector,
    angle_deg: Real,
    mode: AdhesionMode,
    hook_capsules: &mut Vec<AdheringCapsule>,
) {
    let bodies = &mut world.bodies;
    let colliders = &mut world.colliders;
    let impulse_joints = &mut world.impulse_joints;
    let theta = angle_deg.to_radians();
    let along = Vector::new(theta.cos(), theta.sin()); // plank long axis, +x end up for θ > 0
    let normal = Vector::new(-theta.sin(), theta.cos()); // out of the plank's top face

    // The plank, pinned to a fixed body through a revolute joint at its center of mass.
    let pivot = bodies.insert(RigidBodyBuilder::fixed().translation(center));
    let plank_body = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(center)
            .rotation(theta)
            .can_sleep(false), // keep slow drifts observable
    );
    let plank = colliders.insert_with_parent(
        ColliderBuilder::cuboid(PLANK_HALF_LEN, PLANK_HALF_THICK)
            .mass(PLANK_MASS)
            .friction(FRICTION),
        plank_body,
        bodies,
    );
    colors.push((plank, Color::new(0.6, 0.6, 0.6, 1.0)));
    impulse_joints.insert(pivot, plank_body, RevoluteJointBuilder::new(), true);

    // Capsules: vertical (rotation locked), bottom cap exactly tangent to the plank's top face at
    // ±CAP_STANCE along the plank. Tangency: cap center = surface point + normal · CAP_RADIUS;
    // body center is half the capsule height above the bottom cap center, vertically.
    for (side, adheres) in [(-1.0, true), (1.0, false)] {
        let surface_point = center + along * (side * CAP_STANCE) + normal * PLANK_HALF_THICK;
        let bottom_cap_center = surface_point + normal * CAP_RADIUS;
        let body_center = bottom_cap_center + Vector::new(0.0, CAP_HALF_HEIGHT);

        let cap_body = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(body_center)
                .lock_rotations()
                .can_sleep(false),
        );
        let mut capsule = ColliderBuilder::capsule_y(CAP_HALF_HEIGHT, CAP_RADIUS)
            .mass(CAP_MASS)
            .friction(FRICTION);
        let adhesive = adheres && mode != AdhesionMode::None;
        if adhesive {
            capsule = capsule.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
        }
        let capsule = colliders.insert_with_parent(capsule, cap_body, bodies);

        let color = if !adheres {
            Color::new(0.75, 0.75, 0.75, 1.0) // right capsule: always inert, light gray
        } else {
            match mode {
                AdhesionMode::None => Color::new(0.55, 0.55, 0.55, 1.0),
                AdhesionMode::Force => Color::new(1.0, 0.3, 0.15, 1.0),
                AdhesionMode::Pressure => Color::new(0.25, 0.55, 1.0, 1.0),
                AdhesionMode::Budget => Color::new(0.25, 0.9, 0.4, 1.0),
            }
        };
        colors.push((capsule, color));

        if adhesive {
            hook_capsules.push(AdheringCapsule {
                handle: capsule,
                mode,
            });
        }
    }
}

fn build_world() -> (
    PhysicsWorld,
    impl PhysicsHooks,
    Vec<(ColliderHandle, Color)>,
) {
    let mut colors = Vec::new();
    let mut world = PhysicsWorld::new();
    let mut hook_capsules = Vec::new();

    let modes = [
        AdhesionMode::None,
        AdhesionMode::Force,
        AdhesionMode::Pressure,
        AdhesionMode::Budget,
    ];
    let angles = [0.0, 15.0, 30.0, 45.0]; // bottom row → top row

    const COL_DX: Real = 14.0;
    const ROW_DY: Real = 16.0;
    let x_center = (modes.len() as Real - 1.0) / 2.0 * COL_DX;
    let y_center = (angles.len() as Real - 1.0) / 2.0 * ROW_DY;

    for (ci, &mode) in modes.iter().enumerate() {
        for (ri, &angle) in angles.iter().enumerate() {
            let x = ci as Real * COL_DX - x_center;
            let y = ri as Real * ROW_DY - y_center;
            add_cell(
                &mut world,
                &mut colors,
                Vector::new(x, y),
                angle,
                mode,
                &mut hook_capsules,
            );
        }
    }

    let physics_hooks = TeeterHook {
        capsules: hook_capsules,
    };

    world.gravity = Vector::new(0.0, -GRAVITY);
    (world, physics_hooks, colors)
}

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    let (mut world, physics_hooks, colors) = build_world();
    viewer.set_world(&mut world);
    for (collider, color) in colors {
        viewer.set_initial_collider_color(collider, color);
    }
    viewer.look_at(Vec2::new(0.0, 0.0), 9.0);
    while viewer.render_frame(&mut world).await {
        if viewer.simulating() {
            world.step_with_events(&physics_hooks, &());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn adhesion_scenario_runs_without_invalid_state() {
        let (mut world, hooks, _) = super::build_world();
        for _ in 0..720 {
            world.step_with_events(&hooks, &());
            assert!(
                world.bodies.iter().all(|(_, body)| {
                    body.translation().is_finite() && body.linvel().is_finite()
                })
            );
            assert!(world.physics_pipeline.quarantine().is_empty());
        }
    }
}
