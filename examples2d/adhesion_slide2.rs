use kiss3d::color::Color;
use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Friction-solver / multi-collider equivalence test (prerequisite for trusting adhesion across
/// multiple contacts).
///
/// A capsule lies lengthwise on a straight slope and slides down under gravity. Each slope is 10
/// units long but is built two different ways that *should* behave identically:
/// - a **single** rectangle collider, or
/// - **10 unit squares** laid end-to-end, striped light/dark gray so the seams are visible (a
///   sliding body crosses 9 seams and is usually in contact with 2–3 square colliders at once —
///   the tricky boundary condition).
///
/// Four dimensions are varied at once:
/// - **Row blocks (y): adhesion force**, 0 at the bottom, increasing upward.
/// - **Outer column groups (x): slope angle** — 20°, then 40°, then 60° from horizontal.
/// - **Inner 2×2 per block/group intersection:**
///   - columns: single rectangle (left) vs. the 10-square strip (right);
///   - rows: friction from the **collider builder** (lower, the usual path) vs. friction forced by
///     the **contact-modification hook** (upper, blue-tinted capsule). The hook capsules are built
///     with friction 0 and the hook overwrites every solver contact's `friction` to the same
///     combined value the builder path produces — if Rapier used anything but the per-contact
///     value, or contact modification interfered with the friction solver's bookkeeping, the two
///     sub-rows would diverge.
///
/// The capsule (cap radius 0.5, total length 3, so a 2-unit flat side) starts at the high end with
/// its flat side resting on the top 2 m of the slope.
///
/// What to look for:
/// - **Bottom block (zero adhesion)** is the friction-solver baseline: all four members of each
///   2×2 must slide *identically* (same speed, same final position). At 20° the capsule stays put
///   (tan 20° < μ = 0.5); at 40° and 60° it slides. Friction cannot "double count" across the
///   squares because the solver bounds each contact's friction by μ × that contact's *solved
///   normal impulse*, and the normal impulses always partition the capsule's weight no matter how
///   many colliders carry it. Hook-set friction feeds the exact same per-contact value, so it
///   inherits the same invariance.
/// - **Adhesion blocks** then probe adhesion across multiple colliders, using the
///   composition-invariant `adhesion_pressure` request (force per meter of contact length). The
///   old per-manifold `adhesion_force` would give the segmented strip ~3× the pull (one manifold
///   per straddled square) — the strip would lock while its rectangle twin still creeps. With
///   pressure, the squares' contact spans partition the capsule's 2 m patch, so both compositions
///   receive the same total pull: every 2×2 must stay fully symmetric, matching the zero-adhesion
///   baseline's symmetry. Any rect-vs-strip divergence in an adhesion block is a regression.
struct CapsuleParams {
    handle: ColliderHandle,
    /// Total adhesion (in force units) over the capsule's 2-unit flat side; the hook requests it
    /// as a pressure of `adhesion / 2` per meter, which is composition-invariant.
    adhesion: Real,
    /// If set, the capsule collider was built frictionless and the hook re-applies FRICTION on
    /// every solver contact — testing that hook-set friction behaves like builder-set friction.
    hook_friction: bool,
}

struct AdhesionSlideHook {
    capsules: Vec<CapsuleParams>,
}

impl PhysicsHooks for AdhesionSlideHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for params in &self.capsules {
            if context.collider1 == params.handle || context.collider2 == params.handle {
                if params.adhesion > 0.0 {
                    // Requested as a *pressure* (force per meter of contact length): each straddled
                    // square's manifold contributes proportionally to its share of the capsule's
                    // 2 m flat side, so the strip totals the same pull as the single rectangle.
                    *context.adhesion_pressure = params.adhesion / CAP_FLAT_SIDE;
                }
                if params.hook_friction {
                    // `SolverContact::friction` is the already-combined coefficient for this
                    // contact point; write the same value the builder path yields (Average of
                    // 0.5 and 0.5).
                    for contact in context.solver_contacts.iter_mut() {
                        contact.friction = FRICTION;
                    }
                }
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
const CAP_FLAT_SIDE: Real = 2.0 * CAP_HALF_HEIGHT; // patch length the adhesion pressure acts on
const FRICTION: Real = 0.5;
const GRAVITY: Real = 9.81;

/// One grid cell: a fixed slope (single rectangle or 10 striped squares) plus a capsule resting
/// flat on its top 2 m, plus a catch floor below. Registers the capsule with the hook if it needs
/// adhesion or hook-driven friction.
#[allow(clippy::too_many_arguments)]
fn add_cell(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    testbed: &mut Testbed,
    center: Vector,
    angle_deg: Real,
    squares: bool,
    adhesion: Real,
    hook_friction: bool,
    color: Color,
    hook_capsules: &mut Vec<CapsuleParams>,
) {
    let theta = angle_deg.to_radians();
    let up_slope = Vector::new(theta.cos(), theta.sin()); // long axis, pointing to the high end
    let top_normal = Vector::new(-theta.sin(), theta.cos()); // out of the slope's top face

    // All static geometry for this cell shares one fixed body.
    let statics = bodies.insert(RigidBodyBuilder::fixed());

    if squares {
        // 10 squares centred at s = -4.5, -3.5, … 4.5 along the slope — abutting, tops co-planar.
        // Alternate light/dark gray so the individual colliders (and their seams) stay visible.
        for k in 0..10 {
            let s = -SURFACE_LEN / 2.0 + SQUARE_HALF + k as Real;
            let square = colliders.insert_with_parent(
                ColliderBuilder::cuboid(SQUARE_HALF, SQUARE_HALF)
                    .translation(center + up_slope * s)
                    .rotation(theta)
                    .friction(FRICTION),
                statics,
                bodies,
            );
            let shade = if k % 2 == 0 { 0.7 } else { 0.42 };
            testbed.set_initial_collider_color(square, Color::new(shade, shade, shade, 1.0));
        }
    } else {
        let rect = colliders.insert_with_parent(
            ColliderBuilder::cuboid(SURFACE_LEN / 2.0, SURFACE_HALF_THICK)
                .translation(center)
                .rotation(theta)
                .friction(FRICTION),
            statics,
            bodies,
        );
        // Uniform mid-gray so the pair reads as the same material as its striped twin.
        testbed.set_initial_collider_color(rect, Color::new(0.56, 0.56, 0.56, 1.0));
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
    // Hook-friction capsules are built frictionless: everything they get, the hook must provide.
    let builder_friction = if hook_friction { 0.0 } else { FRICTION };
    let mut capsule =
        ColliderBuilder::capsule_x(CAP_HALF_HEIGHT, CAP_RADIUS).friction(builder_friction);
    let needs_hook = adhesion > 0.0 || hook_friction;
    if needs_hook {
        capsule = capsule.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
    }
    let capsule = colliders.insert_with_parent(capsule, cap_body, bodies);
    testbed.set_initial_collider_color(capsule, color);
    if needs_hook {
        hook_capsules.push(CapsuleParams {
            handle: capsule,
            adhesion,
            hook_friction,
        });
    }
}

pub fn init_world(testbed: &mut Testbed) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let mut hook_capsules = Vec::new();

    let angles = [20.0, 40.0, 60.0];
    let adhesions = [0.0, 12.0, 25.0, 45.0]; // bottom block → top block
    let max_adhesion = 45.0;

    const PAIR_DX: Real = 13.0; // rectangle ↔ square strip within an angle group
    const GROUP_DX: Real = 34.0; // angle group ↔ angle group
    const SUB_DY: Real = 15.0; // builder-friction ↔ hook-friction within an adhesion block
    const BLOCK_DY: Real = 34.0; // adhesion block ↔ adhesion block

    let x_center = (2.0 * GROUP_DX + PAIR_DX) / 2.0;
    let y_center = ((adhesions.len() as Real - 1.0) * BLOCK_DY + SUB_DY) / 2.0;

    for (gi, &angle) in angles.iter().enumerate() {
        for (ci, &squares) in [false, true].iter().enumerate() {
            let x = gi as Real * GROUP_DX + ci as Real * PAIR_DX - x_center;
            for (ri, &adhesion) in adhesions.iter().enumerate() {
                // Lower sub-row: friction from the collider builder (the baseline path).
                // Upper sub-row: friction forced per-contact by the hook (blue-tinted capsule).
                for (fi, &hook_friction) in [false, true].iter().enumerate() {
                    let y = ri as Real * BLOCK_DY + fi as Real * SUB_DY - y_center;
                    // Capsule colour: red (no adhesion) → green (strong) by block; the blue channel
                    // marks the hook-friction sub-row.
                    let frac = adhesion / max_adhesion;
                    let blue = if hook_friction { 0.9 } else { 0.15 };
                    let color = Color::new(1.0 - frac, frac, blue, 1.0);
                    add_cell(
                        &mut bodies,
                        &mut colliders,
                        testbed,
                        Vector::new(x, y),
                        angle,
                        squares,
                        adhesion,
                        hook_friction,
                        color,
                        &mut hook_capsules,
                    );
                }
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
    testbed.look_at(Vec2::new(0.0, 0.0), 5.5);
}
