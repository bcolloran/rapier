use kiss3d::color::Color;
use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Friction-solver / multi-collider equivalence test (prerequisite for trusting adhesion across
/// multiple contacts).
///
/// A capsule lies lengthwise on a straight slope and slides down under gravity. Each slope is 10
/// units long but is built two different ways that *should* behave identically:
/// - a **single** rectangle collider, or
/// - **10 unit squares** laid end-to-end (so a sliding body crosses 9 seams and is usually in
///   contact with 2–3 square colliders at once — the tricky boundary condition).
///
/// Three dimensions are varied at once:
/// - **Rows (y): adhesion force**, 0 at the bottom, increasing upward.
/// - **Outer column groups: slope angle** — 20°, then 40°, then 60° from horizontal.
/// - **Inner paired columns: surface composition** — single rectangle (left of each pair) vs. the
///   10-square strip (right of each pair), sitting side by side so they are easy to compare.
///
/// The capsule (cap radius 0.5, total length 3, so a 2-unit flat side) starts at the high end with
/// its flat side resting on the top 2 m of the slope.
///
/// What to look for:
/// - **Bottom row (zero adhesion)** is the friction-solver baseline: within every pair the single
///   rectangle and the 10-square strip must slide *identically* (same speed, same final position).
///   At 20° the capsule stays put (tan 20° < μ = 0.5); at 40° and 60° it slides. If the two members
///   of a pair diverge here, the friction solver is mishandling multi-collider contact — that is the
///   thing this row exists to rule out.
/// - **Adhesion rows** then probe adhesion across multiple colliders. Adhesion is requested *per
///   contact manifold*, and the segmented strip presents a separate manifold per square, so a capsule
///   spanning N squares receives roughly N× the total inward pull of the single rectangle. Expect the
///   square-strip capsule to cling harder / at a lower adhesion than its rectangle twin — e.g. around
///   40° in a mid row the rectangle still creeps while the strip has already locked. That asymmetry
///   is the multi-collider adhesion behaviour to be aware of.
struct AdhesionSlideHook {
    // (capsule collider, adhesion force) — only capsules that actually request adhesion are listed.
    capsules: Vec<(ColliderHandle, Real)>,
}

impl PhysicsHooks for AdhesionSlideHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for &(capsule, adhesion) in &self.capsules {
            if context.collider1 == capsule || context.collider2 == capsule {
                // Applied to every manifold this capsule is part of — including each square it
                // straddles, which is exactly the multi-collider case under test.
                *context.adhesion_force = adhesion;
                return;
            }
        }
    }
}

const SURFACE_LEN: Real = 10.0;
const SURFACE_HALF_THICK: Real = 0.5; // single-rectangle half-thickness; matches the 1×1 squares
const SQUARE_HALF: Real = 0.5; // 1×1 squares
const CAP_RADIUS: Real = 0.5;
const CAP_HALF_HEIGHT: Real = 1.0; // segment length 2 ⇒ total capsule length 3, flat side 2 units
const FRICTION: Real = 0.5;
const GRAVITY: Real = 9.81;

/// One grid cell: a fixed slope (single rectangle or 10 squares) plus a capsule resting flat on its
/// top 2 m, plus a catch floor below. Registers the capsule for adhesion if `adhesion > 0`.
#[allow(clippy::too_many_arguments)]
fn add_cell(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    testbed: &mut Testbed,
    center: Vector,
    angle_deg: Real,
    squares: bool,
    adhesion: Real,
    color: Color,
    hook_capsules: &mut Vec<(ColliderHandle, Real)>,
) {
    let theta = angle_deg.to_radians();
    let up_slope = Vector::new(theta.cos(), theta.sin()); // long axis, pointing to the high end
    let top_normal = Vector::new(-theta.sin(), theta.cos()); // out of the slope's top face

    // All static geometry for this cell shares one fixed body.
    let statics = bodies.insert(RigidBodyBuilder::fixed());

    if squares {
        // 10 squares centred at s = -4.5, -3.5, … 4.5 along the slope — abutting, tops co-planar.
        for k in 0..10 {
            let s = -SURFACE_LEN / 2.0 + SQUARE_HALF + k as Real;
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(SQUARE_HALF, SQUARE_HALF)
                    .translation(center + up_slope * s)
                    .rotation(theta)
                    .friction(FRICTION),
                statics,
                bodies,
            );
        }
    } else {
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(SURFACE_LEN / 2.0, SURFACE_HALF_THICK)
                .translation(center)
                .rotation(theta)
                .friction(FRICTION),
            statics,
            bodies,
        );
    }

    // Catch floor so a slid-off capsule doesn't wander into other cells.
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(7.0, 0.3)
            .translation(center + Vector::new(0.0, -6.5))
            .friction(FRICTION),
        statics,
        bodies,
    );

    // Capsule: flat side on the top 2 m. Its centre sits at s = SURFACE_LEN/2 − CAP_HALF_HEIGHT = 4
    // along the slope (so the 2-unit flat side covers s ∈ [3, 5]), lifted off the top face by the
    // cap radius. Rotation is left free so any spurious seam torque would show up as wobble.
    let cap_center = center
        + up_slope * (SURFACE_LEN / 2.0 - CAP_HALF_HEIGHT)
        + top_normal * (SURFACE_HALF_THICK + CAP_RADIUS);
    let cap_body = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(cap_center)
            .rotation(theta),
    );
    let mut capsule = ColliderBuilder::capsule_x(CAP_HALF_HEIGHT, CAP_RADIUS).friction(FRICTION);
    if adhesion > 0.0 {
        // Only adhering capsules enable the hook, so the zero-adhesion baseline row is pure default.
        capsule = capsule.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
    }
    let capsule = colliders.insert_with_parent(capsule, cap_body, bodies);
    testbed.set_initial_collider_color(capsule, color);
    if adhesion > 0.0 {
        hook_capsules.push((capsule, adhesion));
    }
}

pub fn init_world(testbed: &mut Testbed) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let mut hook_capsules = Vec::new();

    let angles = [20.0, 40.0, 60.0];
    let adhesions = [0.0, 12.0, 25.0, 45.0]; // bottom row → top row
    let max_adhesion = 45.0;

    const PAIR_DX: Real = 13.0; // rectangle ↔ square within an angle group
    const GROUP_DX: Real = 28.0; // angle group ↔ angle group
    const ROW_DY: Real = 15.0;

    let x_center = (2.0 * GROUP_DX + PAIR_DX) / 2.0;
    let y_center = (adhesions.len() as Real - 1.0) / 2.0 * ROW_DY;

    for (gi, &angle) in angles.iter().enumerate() {
        for (ci, &squares) in [false, true].iter().enumerate() {
            let x = gi as Real * GROUP_DX + ci as Real * PAIR_DX - x_center;
            for (ri, &adhesion) in adhesions.iter().enumerate() {
                let y = ri as Real * ROW_DY - y_center;
                // Capsule colour: red (no adhesion) → green (strong), by row.
                let frac = adhesion / max_adhesion;
                let color = Color::new(1.0 - frac, frac, 0.2, 1.0);
                add_cell(
                    &mut bodies,
                    &mut colliders,
                    testbed,
                    Vector::new(x, y),
                    angle,
                    squares,
                    adhesion,
                    color,
                    &mut hook_capsules,
                );
            }
        }
    }

    let physics_hooks = AdhesionSlideHook {
        capsules: hook_capsules,
    };

    testbed.set_world_with_params(
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        Vector::new(0.0, -GRAVITY),
        physics_hooks,
    );
    testbed.look_at(Vec2::new(0.0, 0.0), 8.0);
}
