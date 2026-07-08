use kiss3d::color::Color;
use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Side-by-side comparison of the three adhesion request flavors on surfaces that *should* behave
/// identically.
///
/// A capsule lies lengthwise on a straight slope and slides down under gravity. Each slope is 10
/// units long but is built two different ways:
/// - a **single** rectangle collider, or
/// - **10 unit squares** laid end-to-end, striped light/dark gray so the seams are visible (a
///   sliding body crosses 9 seams and is usually in contact with 2–3 square colliders at once —
///   the tricky boundary condition).
///
/// Four dimensions are varied at once:
/// - **Row blocks (y): adhesion magnitude** (total newtons over the capsule's 2 m flat side),
///   0 at the bottom, increasing upward.
/// - **Outer column groups (x): slope angle** — 20°, then 40°, then 60° from horizontal.
/// - **Inner columns per group:** single rectangle (left) vs. the 10-square strip (right).
/// - **Sub-rows within a block: how the adhesion is requested** —
///   - bottom: `adhesion_force` (absolute force *per manifold*),
///   - middle: `adhesion_pressure` (force per meter of contact length; blue-tinted capsule),
///   - top: `adhesion_budget` (fixed total shared by all of the capsule's manifolds; bright blue).
///
/// What to look for:
/// - **Bottom block (zero adhesion)** is the friction baseline: all six members of each group must
///   slide *identically*. At 20° the capsule stays put (tan 20° < μ = 0.5); at 40° and 60° it
///   slides. Friction cannot double count across the squares because the solver bounds each
///   contact's friction by μ × that contact's *solved normal impulse*, and normal impulses always
///   partition the capsule's weight no matter how many colliders carry it.
/// - **`adhesion_force` sub-rows show the double-counting bug**: the request is per manifold, and
///   the strip presents one manifold per straddled square (~3 at once), so the strip capsule gets
///   ~3× the pull of its rectangle twin — expect it to cling harder / lock while the rectangle
///   still creeps (clearest around 40° at 12 N).
/// - **`adhesion_pressure` sub-rows are composition-invariant**: the squares' contact spans
///   partition the capsule's 2 m patch, so both compositions receive the same total pull — the
///   rect/strip pair must behave identically, matching the baseline's symmetry.
/// - **`adhesion_budget` sub-rows are also invariant** (the pool total is shared by however many
///   manifolds enroll), and additionally would stay invariant under *overlapping* colliders and
///   keep working on point contacts — cases where pressure falls short.
#[derive(Clone, Copy, PartialEq)]
enum AdhesionMode {
    /// `adhesion_force`: absolute per-manifold force — multiplies with the manifold count.
    Force,
    /// `adhesion_pressure`: force per meter of contact — composition-invariant.
    Pressure,
    /// `adhesion_budget`: fixed total shared by the capsule's manifolds — composition- and
    /// overlap-invariant.
    Budget,
}

struct CapsuleParams {
    handle: ColliderHandle,
    /// Total adhesion (in force units) over the capsule's 2-unit flat side.
    adhesion: Real,
    mode: AdhesionMode,
}

struct AdhesionSlideHook {
    capsules: Vec<CapsuleParams>,
}

impl PhysicsHooks for AdhesionSlideHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for params in &self.capsules {
            if context.collider1 == params.handle || context.collider2 == params.handle {
                match params.mode {
                    AdhesionMode::Force => {
                        // Per manifold — each straddled square adds a full copy (the bug).
                        *context.adhesion_force = params.adhesion;
                    }
                    AdhesionMode::Pressure => {
                        // Per meter of contact — the straddled squares' spans partition the 2 m
                        // flat side, so the strip totals the same pull as the rectangle.
                        *context.adhesion_pressure = params.adhesion / CAP_FLAT_SIDE;
                    }
                    AdhesionMode::Budget => {
                        // Fixed total shared by every manifold of this capsule's pool, however
                        // many colliders implement the contact.
                        *context.adhesion_budget = Some(AdhesionBudget {
                            owner: params.handle,
                            channel: 0,
                            total: params.adhesion,
                        });
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
/// flat on its top 2 m, plus a catch floor below. Registers the capsule with the hook if it
/// requests adhesion.
#[allow(clippy::too_many_arguments)]
fn add_cell(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    testbed: &mut Testbed,
    center: Vector,
    angle_deg: Real,
    squares: bool,
    adhesion: Real,
    mode: AdhesionMode,
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
    let mut capsule = ColliderBuilder::capsule_x(CAP_HALF_HEIGHT, CAP_RADIUS).friction(FRICTION);
    if adhesion > 0.0 {
        // Only adhering capsules enable the hook, so the zero-adhesion baseline block is pure
        // default behavior.
        capsule = capsule.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
    }
    let capsule = colliders.insert_with_parent(capsule, cap_body, bodies);
    testbed.set_initial_collider_color(capsule, color);
    if adhesion > 0.0 {
        hook_capsules.push(CapsuleParams {
            handle: capsule,
            adhesion,
            mode,
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
    // Bottom → top within a block. In the zero-adhesion block the mode is irrelevant (three
    // identical baselines); everywhere else the force sub-row is the one that misbehaves on the
    // strip.
    let modes = [
        AdhesionMode::Force,
        AdhesionMode::Pressure,
        AdhesionMode::Budget,
    ];

    const PAIR_DX: Real = 13.0; // rectangle ↔ square strip within an angle group
    const GROUP_DX: Real = 34.0; // angle group ↔ angle group
    const SUB_DY: Real = 15.0; // force ↔ pressure ↔ budget within an adhesion block
    const BLOCK_DY: Real = 49.0; // adhesion block ↔ adhesion block (3 sub-rows + gap)

    let x_center = (2.0 * GROUP_DX + PAIR_DX) / 2.0;
    let y_center = ((adhesions.len() as Real - 1.0) * BLOCK_DY
        + (modes.len() as Real - 1.0) * SUB_DY)
        / 2.0;

    for (gi, &angle) in angles.iter().enumerate() {
        for (ci, &squares) in [false, true].iter().enumerate() {
            let x = gi as Real * GROUP_DX + ci as Real * PAIR_DX - x_center;
            for (ri, &adhesion) in adhesions.iter().enumerate() {
                for (mi, &mode) in modes.iter().enumerate() {
                    let y = ri as Real * BLOCK_DY + mi as Real * SUB_DY - y_center;
                    // Capsule colour: red (no adhesion) → green (strong) by block; the blue
                    // channel marks the request flavor (force 0.1, pressure 0.55, budget 1.0).
                    let frac = adhesion / max_adhesion;
                    let blue = match mode {
                        AdhesionMode::Force => 0.1,
                        AdhesionMode::Pressure => 0.55,
                        AdhesionMode::Budget => 1.0,
                    };
                    let color = Color::new(1.0 - frac, frac, blue, 1.0);
                    add_cell(
                        &mut bodies,
                        &mut colliders,
                        testbed,
                        Vector::new(x, y),
                        angle,
                        squares,
                        adhesion,
                        mode,
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
    testbed.look_at(Vec2::new(0.0, 0.0), 3.6);
}
