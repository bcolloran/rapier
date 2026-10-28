use kiss3d::color::Color;
use rapier_testbed2d::TestbedViewer;
use rapier2d::prelude::*;

/// Per-wall adhesion selected by which face the contact is on: the left face uses `left`, the right
/// face uses `right`, and the wall's top/bottom get none.
struct AdhesionGridHook {
    // (wall collider, wall angle, left-face adhesion, right-face adhesion)
    walls: Vec<(ColliderHandle, Real, Real, Real)>,
}

impl PhysicsHooks for AdhesionGridHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let (collider1, collider2) = (context.collider1, context.collider2);
        let Some(rigid) = context.rigid_mut() else {
            return;
        };

        for &(wall, angle, left, right) in &self.walls {
            // The contact normal pointing *out of the wall* (the manifold normal points out of
            // collider1, so flip it when the wall is collider2).
            let outward = if collider1 == wall {
                *rigid.normal
            } else if collider2 == wall {
                -*rigid.normal
            } else {
                continue;
            };

            // Decompose that normal in the (rotated) wall's local frame.
            let right_axis = Vector::new(angle.cos(), angle.sin());
            let up_axis = Vector::new(-angle.sin(), angle.cos());
            let along_face = outward.dot(right_axis); // + = right face, - = left face
            let along_length = outward.dot(up_axis); // dominant = top/bottom face

            // Only adhere on the side faces; leave top/bottom contacts at zero adhesion.
            if along_face.abs() > along_length.abs() {
                *rigid.adhesion_force = if along_face > 0.0 { right } else { left };
            }
            return;
        }
    }
}

const WALL_HALF_THICKNESS: Real = 0.15;
const WALL_HALF_LENGTH: Real = 1.3;
const BOX_HALF: Real = 0.4;
const FRICTION: Real = 0.5;

/// Adds a dynamic box clinging to one face of a wall. `side` is +1 for the right face, -1 for the
/// left face.
fn add_side_box(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    wall_center: Vector,
    angle: Real,
    side: Real,
) {
    let face_normal = Vector::new(side * angle.cos(), side * angle.sin());
    let up = Vector::new(-angle.sin(), angle.cos());
    let box_center = wall_center
        + up * (WALL_HALF_LENGTH * 0.55)
        + face_normal * (WALL_HALF_THICKNESS + BOX_HALF - 0.01);
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

/// Adds one grid cell: a fixed wall tilted by `angle` with a box adhered to *each* side. The left
/// face uses `left_adhesion`, the right face uses twice that.
fn add_cell(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    walls: &mut Vec<(ColliderHandle, Real, Real, Real)>,
    center: Vector,
    angle: Real,
    left_adhesion: Real,
) {
    let right_adhesion = 2.0 * left_adhesion;

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
    walls.push((wall, angle, left_adhesion, right_adhesion));

    add_side_box(bodies, colliders, center, angle, -1.0); // left face
    add_side_box(bodies, colliders, center, angle, 1.0); // right face
}

fn build_world() -> (
    PhysicsWorld,
    impl PhysicsHooks,
    Vec<(ColliderHandle, Color)>,
) {
    let colors = Vec::new();
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

    // Rows vary the LEFT-face adhesion down the y-axis (the right face always gets twice as much,
    // so the right box clings harder than the left). Top row holds; bottom row (zero) lets the left
    // box go immediately; the middle is a hand-tuned gradient.
    let row_y = [9.0, 4.5, 0.0, -4.5, -9.0];
    let row_adhesion = [26.0, 13.0, 11.0, 6.0, 0.0];

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

    let mut world = PhysicsWorld::new();
    world.bodies = bodies;
    world.colliders = colliders;
    world.impulse_joints = impulse_joints;
    world.multibody_joints = multibody_joints;
    world.gravity = Vector::new(0.0, -9.81);
    (world, physics_hooks, colors)
}

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    let (mut world, physics_hooks, colors) = build_world();
    viewer.set_world(&mut world);
    for (collider, color) in colors {
        viewer.set_initial_collider_color(collider, color);
    }
    viewer.look_at(Vec2::new(0.0, 0.0), 20.0);
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
