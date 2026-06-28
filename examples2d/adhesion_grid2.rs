use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Looks up a per-wall adhesion force, so every wall in the grid can use a different strength.
struct AdhesionGridHook {
    walls: Vec<(ColliderHandle, Real)>,
}

impl PhysicsHooks for AdhesionGridHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for &(wall, force) in &self.walls {
            if context.collider1 == wall || context.collider2 == wall {
                *context.adhesion_force = force;
                return;
            }
        }
    }
}

const WALL_HALF_THICKNESS: Real = 0.15;
const WALL_HALF_LENGTH: Real = 1.3;
const BOX_HALF: Real = 0.4;
const FRICTION: Real = 0.5;

/// Adds one grid cell: a fixed wall tilted by `angle` (radians) with adhesion enabled, and a box
/// adhered to the *left face, near the top* of that wall.
fn add_cell(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    walls: &mut Vec<(ColliderHandle, Real)>,
    center: Vector,
    angle: Real,
    adhesion: Real,
) {
    let wall_body = bodies.insert(RigidBodyBuilder::fixed());
    let wall = colliders.insert_with_parent(
        ColliderBuilder::cuboid(WALL_HALF_THICKNESS, WALL_HALF_LENGTH)
            .translation(center)
            .rotation(angle)
            .friction(FRICTION)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        wall_body,
        bodies,
    );
    walls.push((wall, adhesion));

    // The wall's local -X (left face normal) and local +Y (along its length), both rotated.
    let left_normal = Vector::new(-angle.cos(), -angle.sin());
    let up = Vector::new(-angle.sin(), angle.cos());
    // Box on the left face, near the top, just touching (a hair of overlap => an active contact).
    let box_center = center
        + up * (WALL_HALF_LENGTH * 0.55)
        + left_normal * (WALL_HALF_THICKNESS + BOX_HALF - 0.01);
    let box_body = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(box_center)
            .rotation(angle),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(BOX_HALF, BOX_HALF).friction(FRICTION),
        box_body,
        bodies,
    );
}

pub fn init_world(testbed: &mut Testbed) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let mut walls = Vec::new();

    let deg = |d: Real| d * std::f32::consts::PI / 180.0;

    // Columns vary the wall tilt across the x-axis. Positive angle leans the wall's top to the LEFT
    // (CCW), negative to the right: 45° left, 10° left, vertical, 10° right, 45° right.
    let column_x = [-10.0, -5.0, 0.0, 5.0, 10.0];
    let column_angle = [deg(45.0), deg(10.0), deg(0.0), deg(-10.0), deg(-45.0)];

    // Rows vary the adhesion strength down the y-axis. The top row is strong enough that nothing
    // moves (stuck fast, no sliding even with this friction); the bottom row has zero adhesion so
    // its boxes fall / slide away immediately. The middle rows are a hand-tuned gradient which,
    // together with the per-column tilt, produces a range of sticking and sliding behaviours.
    let row_y = [9.0, 4.5, 0.0, -4.5, -9.0];
    let row_adhesion = [150.0, 30.0, 11.0, 6.0, 0.0];

    for (&x, &angle) in column_x.iter().zip(column_angle.iter()) {
        for (&y, &adhesion) in row_y.iter().zip(row_adhesion.iter()) {
            add_cell(
                &mut bodies,
                &mut colliders,
                &mut walls,
                Vector::new(x, y),
                angle,
                adhesion,
            );
        }
    }

    // A floor to catch boxes that fall off the weak / zero-adhesion rows.
    let floor_body = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(40.0, 0.5).translation(Vector::new(0.0, -12.0)),
        floor_body,
        &mut bodies,
    );

    let physics_hooks = AdhesionGridHook { walls };

    testbed.set_world_with_params(
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        Vector::new(0.0, -9.81),
        physics_hooks,
    );
    testbed.look_at(Vec2::new(0.0, 0.0), 14.0);
}
