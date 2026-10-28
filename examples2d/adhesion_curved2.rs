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
        let (collider1, collider2) = (context.collider1, context.collider2);
        let Some(rigid) = context.rigid_mut() else {
            return;
        };

        for &(object, force) in &self.objects {
            if collider1 == object || collider2 == object {
                *rigid.adhesion_force = force;
                return;
            }
        }
    }
}

const FRICTION: Real = 0.5;
const DISK_RADIUS: Real = 3.0;
const STRONG_ADHESION: Real = 40.0;

fn build_world() -> (
    PhysicsWorld,
    impl PhysicsHooks,
    Vec<(ColliderHandle, Color)>,
) {
    let mut colors = Vec::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let mut objects = Vec::new();

    let disk_center = Vector::new(0.0, 4.0);

    // The fixed central disk (a curved surface), with contact modification enabled.
    let disk_body = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(
        ColliderBuilder::ball(DISK_RADIUS)
            .translation(disk_center)
            .friction(FRICTION)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        disk_body,
        &mut bodies,
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
            let body = bodies.insert(RigidBodyBuilder::dynamic().translation(pos));
            let collider = colliders.insert_with_parent(
                ColliderBuilder::ball(r).friction(FRICTION),
                body,
                &mut bodies,
            );
            objects.push((collider, adhesion));
        } else {
            // Capsule lying tangent to the disk (long axis = local +Y, rotated to the tangent).
            let half_height = 0.4;
            let r = 0.22;
            let pos = disk_center + dir * (DISK_RADIUS + r - 0.02);
            let body = bodies.insert(RigidBodyBuilder::dynamic().translation(pos).rotation(phi));
            let collider = colliders.insert_with_parent(
                ColliderBuilder::capsule_y(half_height, r).friction(FRICTION),
                body,
                &mut bodies,
            );
            // Tint the capsules so they stand out from the balls in the visualization.
            colors.push((collider, Color::new(1.0, 0.6, 0.1, 1.0)));
            objects.push((collider, adhesion));
        }
    }

    // A floor to catch the objects that fall.
    let floor_body = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(20.0, 0.5).translation(Vector::new(0.0, -3.0)),
        floor_body,
        &mut bodies,
    );

    let physics_hooks = AdhesionCurvedHook { objects };

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
    viewer.look_at(Vector::new(0.0, 4.0), 30.0);
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
