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
        for &(wall, angle, left, right) in &self.walls {
            // The contact normal pointing *out of the wall* (the manifold normal points out of
            // collider1, so flip it when the wall is collider2).
            let outward = if context.collider1 == wall {
                *context.normal
            } else if context.collider2 == wall {
                -*context.normal
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
                *context.adhesion_force = if along_face > 0.0 { right } else { left };
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
fn add_side_box(world: &mut PhysicsWorld, wall_center: Vector, angle: Real, side: Real) {
    let face_normal = Vector::new(side * angle.cos(), side * angle.sin());
    let up = Vector::new(-angle.sin(), angle.cos());
    let box_center = wall_center
        + up * (WALL_HALF_LENGTH * 0.55)
        + face_normal * (WALL_HALF_THICKNESS + BOX_HALF - 0.01);
    let box_body = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(box_center)
            .rotation(angle),
    );
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(BOX_HALF, BOX_HALF).friction(FRICTION),
        box_body,
        &mut world.bodies,
    );
}

/// Adds one grid cell: a fixed wall tilted by `angle` with a box adhered to *each* side. The left
/// face uses `left_adhesion`, the right face uses twice that.
fn add_cell(
    world: &mut PhysicsWorld,
    walls: &mut Vec<(ColliderHandle, Real, Real, Real)>,
    center: Vector,
    angle: Real,
    left_adhesion: Real,
) {
    let right_adhesion = 2.0 * left_adhesion;

    let wall_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let wall = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(WALL_HALF_THICKNESS, WALL_HALF_LENGTH)
            .translation(center)
            .rotation(angle)
            .friction(FRICTION)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        wall_body,
        &mut world.bodies,
    );
    walls.push((wall, angle, left_adhesion, right_adhesion));

    add_side_box(world, center, angle, -1.0); // left face
    add_side_box(world, center, angle, 1.0); // right face
}

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    // Step with `step_collisions_last` (the game's stepping mode) instead of `step`. Changing it
    // restarts the example, so both modes start from the same scene.
    let collisions_last = viewer
        .example_settings_mut()
        .get_or_set_bool("Collisions last", true);

    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81);
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
            add_cell(&mut world, &mut walls, Vector::new(x, y), angle, adhesion);
        }
    }

    // A floor to catch boxes that fall off the weak / zero-adhesion rows.
    let floor_body = world.bodies.insert(RigidBodyBuilder::fixed());
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(40.0, 0.5).translation(Vector::new(0.0, -12.0)),
        floor_body,
        &mut world.bodies,
    );

    let physics_hooks = AdhesionGridHook { walls };

    viewer.set_world(&mut world);
    viewer.look_at(Vec2::new(0.0, 0.0), 20.0);

    // After `set_world`, which replaces the broad-phase.
    if collisions_last {
        world.initialize_collisions_last_with_events(&physics_hooks, &());
    }

    while viewer.render_frame(&mut world).await {
        if viewer.simulating() {
            if collisions_last {
                world.step_collisions_last_with_events(&physics_hooks, &());
            } else {
                world.step_with_events(&physics_hooks, &());
            }
        }
    }
    Ok(())
}
