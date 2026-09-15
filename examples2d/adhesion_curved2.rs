use kiss3d::color::Color;
use rapier_testbed2d::TestbedViewer;
use rapier2d::prelude::*;

/// Per-object adhesion: each dynamic object (ball or capsule) gets its own adhesion force, looked
/// up by collider when it touches the central disk (adhesion works on round shapes too).
struct AdhesionCurvedHook {
    objects: Vec<(ColliderHandle, Real)>,
}

impl PhysicsHooks for AdhesionCurvedHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for &(object, force) in &self.objects {
            if context.collider1 == object || context.collider2 == object {
                *context.adhesion_force = force;
                return;
            }
        }
    }
}

const FRICTION: Real = 0.5;
const DISK_RADIUS: Real = 3.0;
const STRONG_ADHESION: Real = 40.0;

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81);
    let mut objects = Vec::new();

    let disk_center = Vector::new(0.0, 4.0);

    // The fixed central disk (a curved surface), with contact modification enabled.
    let disk_body = world.bodies.insert(RigidBodyBuilder::fixed());
    world.colliders.insert_with_parent(
        ColliderBuilder::ball(DISK_RADIUS)
            .translation(disk_center)
            .friction(FRICTION)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        disk_body,
        &mut world.bodies,
    );

    // A ring of dynamic objects placed all the way around the disk, alternating ball / capsule.
    // Objects with strong adhesion cling everywhere (even hanging underneath); objects with zero
    // adhesion fall off wherever gravity isn't holding them onto the disk (sides and underside).
    let count = 12;
    for i in 0..count {
        let phi = i as Real / count as Real * std::f32::consts::TAU;
        let dir = Vector::new(phi.cos(), phi.sin());

        // Strong for two out of every four, so both shapes appear in clinging and falling groups.
        let adhesion = if i % 4 < 2 { STRONG_ADHESION } else { 0.0 };

        if i % 2 == 0 {
            // Ball touching the disk surface.
            let r = 0.35;
            let pos = disk_center + dir * (DISK_RADIUS + r - 0.02);
            let body = world
                .bodies
                .insert(RigidBodyBuilder::dynamic().translation(pos));
            let collider = world.colliders.insert_with_parent(
                ColliderBuilder::ball(r).friction(FRICTION),
                body,
                &mut world.bodies,
            );
            objects.push((collider, adhesion));
        } else {
            // Capsule lying tangent to the disk (long axis = local +Y, rotated to the tangent).
            let half_height = 0.4;
            let r = 0.22;
            let pos = disk_center + dir * (DISK_RADIUS + r - 0.02);
            let body = world
                .bodies
                .insert(RigidBodyBuilder::dynamic().translation(pos).rotation(phi));
            let collider = world.colliders.insert_with_parent(
                ColliderBuilder::capsule_y(half_height, r).friction(FRICTION),
                body,
                &mut world.bodies,
            );
            // Tint the capsules so they stand out from the balls in the visualization.
            viewer.set_initial_collider_color(collider, Color::new(1.0, 0.6, 0.1, 1.0));
            objects.push((collider, adhesion));
        }
    }

    // A floor to catch the objects that fall.
    let floor_body = world.bodies.insert(RigidBodyBuilder::fixed());
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(20.0, 0.5).translation(Vector::new(0.0, -3.0)),
        floor_body,
        &mut world.bodies,
    );

    let physics_hooks = AdhesionCurvedHook { objects };

    viewer.set_world(&mut world);
    viewer.look_at(disk_center, 30.0);

    while viewer.render_frame(&mut world).await {
        if viewer.simulating() {
            // Detects collisions last when Settings > Advanced > "Collisions last" is checked
            // (`set_world` applies it to `world.collisions_last`).
            world.step_with_events(&physics_hooks, &());
        }
    }
    Ok(())
}
