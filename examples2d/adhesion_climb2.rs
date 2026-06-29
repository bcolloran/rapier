use kiss3d::color::Color;
use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Tangent velocity **and** adhesion, interacting on the *same* contact.
///
/// Each cell is a rotation-locked capsule standing on a big static disk, its lower cap touching the
/// disk at 22.5° above the +X axis. The capsule's contact with the disk is driven like a conveyor
/// belt (`tangent_velocity`) *up-slope* towards the top of the disk, and is simultaneously glued to
/// the disk with an attractive `adhesion_force` — both set on the same manifold from the same hook.
/// The **adhesion** sticks only while the capsule's **lower** cap is touching the disk — i.e. while
/// the contact normal from the capsule towards the disk points downward (negative Y) — modelling a
/// foot that grips only on its sole. The conveyor **belt** is applied at *any* contact angle.
///
/// Why the two settings interact: the conveyor can only haul the capsule up-slope through *friction*,
/// whose limit is `μ · N`, and the only thing pinning the capsule to the slope is the adhesion. With
/// no adhesion the belt's friction budget can't overcome gravity's down-slope pull, so the capsule
/// just slips off; with enough adhesion the very same belt drives it cleanly over the top. So the
/// adhesion is what makes the tangent-velocity drive *work* at all.
///
/// Grid (4×4):
/// - Columns (x) vary the adhesion force, linearly from 0 (col 1) up to the analytic value that lets
///   the capsule *barely* crest the top at the slowest belt speed (col 4). The hardest point of the
///   climb is the at-rest start (22.5°): if the grip can't start the climb there it never does, so
///   the threshold is sharp — cols 1–3 slip straight back, only col 4 climbs.
/// - Rows (y) vary the belt speed so that, with *perfect* grip, the capsule would reach the top in a
///   geometrically-interpolated time from 10 s (row 1, slowest) to 1 s (row 4, fastest). The start is
///   the bottleneck and the capsule is barely moving there regardless of belt speed, so col 4
///   adhesion carries *every* row over the top — the rows differ in how fast they climb, not whether
///   they make it. (col 4 is tuned to the slowest row, so it lags its ideal 10 s and crests in ~11 s.)
///
/// Net picture: the whole col-4 column climbs (faster down the rows); the three weaker columns slip
/// off at the start — adhesion gates the climb, belt speed sets its pace.
struct AdhesionClimbHook {
    cells: Vec<Cell>,
}

struct Cell {
    disk: ColliderHandle,
    capsule: ColliderHandle,
    /// Up-slope belt speed in world units per second.
    belt_speed: Real,
    /// Attractive force holding the capsule onto the disk.
    adhesion: Real,
}

impl PhysicsHooks for AdhesionClimbHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for cell in &self.cells {
            // Match this manifold to a (disk, capsule) cell, in either collider ordering.
            let capsule_is_collider1 = if context.collider1 == cell.capsule
                && context.collider2 == cell.disk
            {
                true
            } else if context.collider1 == cell.disk && context.collider2 == cell.capsule {
                false
            } else {
                continue;
            };

            // `context.normal` points from collider1's exterior towards collider2. Build the disk's
            // outward normal (pointing from the disk surface towards the capsule).
            let disk_outward = if capsule_is_collider1 {
                -*context.normal // normal points capsule -> disk, flip it
            } else {
                *context.normal // normal already points disk -> capsule
            };

            // Adhesion sticks only on the capsule's LOWER cap: the normal *from the capsule towards
            // the disk* must point downward (negative Y). That normal is the opposite of the disk's
            // outward normal. (Models a foot that only grips on its sole.)
            let capsule_to_disk = -disk_outward;
            if capsule_to_disk.y < 0.0 {
                *context.adhesion_force = cell.adhesion;
            }

            // The conveyor drive, by contrast, is applied at ANY contact angle. Up-slope tangent =
            // the outward normal rotated +90° (CCW), pointing towards the top of the disk (increasing
            // polar angle).
            let up_slope = Vector::new(-disk_outward.y, disk_outward.x);

            // `tangent_velocity` is the target *relative* surface velocity `v(collider2) −
            // v(collider1)` at the contact. The disk is static, so to drive the capsule to
            // `belt_speed · up_slope` we use +T when the capsule is collider2 and −T when collider1.
            let sign = if capsule_is_collider1 { -1.0 } else { 1.0 };
            let belt = up_slope * (cell.belt_speed * sign);
            for contact in context.solver_contacts.iter_mut() {
                contact.tangent_velocity = belt;
            }
            return;
        }
    }
}

const DISK_RADIUS: Real = 5.0;
const CAP_RADIUS: Real = 0.5;
const CAP_HALF_HEIGHT: Real = 1.0; // total capsule height = 2·1 + 2·0.5 = 3
const FRICTION: Real = 0.5; // both colliders; default Average combine ⇒ μ = 0.5
const GRAVITY: Real = 9.81;
const START_ANGLE_DEG: Real = 22.5;

/// Builds one cell: a fixed disk, a rotation-locked capsule touching it at `START_ANGLE_DEG`, and a
/// catch floor below so a capsule that slips back stays in its own cell.
fn add_cell(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    testbed: &mut Testbed,
    center: Vector,
    belt_speed: Real,
    adhesion: Real,
    capsule_color: Color,
) -> Cell {
    // Fixed disk with contact modification enabled.
    let disk_body = bodies.insert(RigidBodyBuilder::fixed());
    let disk = colliders.insert_with_parent(
        ColliderBuilder::ball(DISK_RADIUS)
            .translation(center)
            .friction(FRICTION)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        disk_body,
        bodies,
    );

    // Capsule: lower cap exactly tangent to the disk at the start angle. The cap circle (radius
    // CAP_RADIUS) is tangent to the disk when its center sits at distance DISK_RADIUS + CAP_RADIUS
    // from the disk center, along the contact direction. The capsule body center is one half-height
    // above its lower cap center (the capsule stays vertical: rotation is locked).
    let theta = START_ANGLE_DEG.to_radians();
    let contact_dir = Vector::new(theta.cos(), theta.sin());
    let lower_cap_center = center + contact_dir * (DISK_RADIUS + CAP_RADIUS);
    let capsule_center = lower_cap_center + Vector::new(0.0, CAP_HALF_HEIGHT);

    let capsule_body = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(capsule_center)
            .lock_rotations()
            .can_sleep(false), // keep the belt able to start a stationary capsule moving
    );
    let capsule = colliders.insert_with_parent(
        ColliderBuilder::capsule_y(CAP_HALF_HEIGHT, CAP_RADIUS).friction(FRICTION),
        capsule_body,
        bodies,
    );
    testbed.set_initial_collider_color(capsule, capsule_color);

    // Catch floor below the disk so a slipped capsule stays put in its cell.
    let floor_body = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(DISK_RADIUS + 1.0, 0.3)
            .translation(center + Vector::new(0.0, -(DISK_RADIUS + 3.5))),
        floor_body,
        bodies,
    );

    Cell {
        disk,
        capsule,
        belt_speed,
        adhesion,
    }
}

pub fn init_world(testbed: &mut Testbed) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let mut cells = Vec::new();

    let theta = START_ANGLE_DEG.to_radians();

    // Capsule mass (density 1): rectangle (2r × 2·half) + the two caps forming a disk (π r²).
    let capsule_area =
        (2.0 * CAP_RADIUS) * (2.0 * CAP_HALF_HEIGHT) + std::f32::consts::PI * CAP_RADIUS * CAP_RADIUS;
    let weight = capsule_area * GRAVITY;

    // Critical adhesion (col 4): the climb is hardest at the start angle, where quasi-statically the
    // up-slope friction budget μ·(A + weight·sinθ) must overcome the down-slope gravity weight·cosθ.
    // Solving for A and adding a small margin so it *just* crests at the slowest belt speed.
    let critical_adhesion = weight * (theta.cos() / FRICTION - theta.sin());
    let col4_adhesion = critical_adhesion * 1.08;

    // The capsule center travels on a circle of radius DISK_RADIUS + CAP_RADIUS. Arc length from the
    // start angle to the top of the disk; belt speed = arc / time so that, with perfect grip, the
    // capsule reaches the top in `time` seconds.
    let arc_length =
        (DISK_RADIUS + CAP_RADIUS) * (std::f32::consts::FRAC_PI_2 - theta);

    const COLS: usize = 4;
    const ROWS: usize = 4;
    const COL_SPACING: Real = 13.0;
    const ROW_SPACING: Real = 19.0;
    const TIME_SLOW: Real = 10.0; // row 1
    const TIME_FAST: Real = 1.0; // row 4

    for col in 0..COLS {
        // Adhesion: linear 0 → col4_adhesion across the columns.
        let col_frac = col as Real / (COLS as Real - 1.0);
        let adhesion = col4_adhesion * col_frac;
        // Capsule colour goes red (weak) → green (strong) so the adhesion gradient is visible.
        let color = Color::new(1.0 - col_frac, col_frac, 0.2, 1.0);

        for row in 0..ROWS {
            // Time-to-top: geometric interpolation TIME_SLOW → TIME_FAST down the rows.
            let row_frac = row as Real / (ROWS as Real - 1.0);
            let time = TIME_SLOW * (TIME_FAST / TIME_SLOW).powf(row_frac);
            let belt_speed = arc_length / time;

            let x = (col as Real - (COLS as Real - 1.0) / 2.0) * COL_SPACING;
            let y = ((ROWS as Real - 1.0) / 2.0 - row as Real) * ROW_SPACING;

            let cell = add_cell(
                &mut bodies,
                &mut colliders,
                testbed,
                Vector::new(x, y),
                belt_speed,
                adhesion,
                color,
            );
            cells.push(cell);
        }
    }

    let physics_hooks = AdhesionClimbHook { cells };

    testbed.set_world_with_params(
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        Vector::new(0.0, -GRAVITY),
        physics_hooks,
    );
    testbed.look_at(Vec2::new(0.0, 0.0), 9.0);
}
