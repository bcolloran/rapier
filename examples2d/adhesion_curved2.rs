use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Per-object adhesion: each dynamic object (ball or capsule) is registered with its own adhesion
/// force, looked up by collider when it touches the central disk. This proves adhesion is decided
/// per contact manifold even for round shapes with no flat faces (the manifold still has a contact
/// point and normal).
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

pub fn init_world(testbed: &mut Testbed) {
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

        // Strong for two out of every four, so both shapes appear in the clinging and falling groups.
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
            // Capsule lying tangent to the disk (long axis along the local +Y, rotated to the tangent).
            let half_height = 0.4;
            let r = 0.22;
            let pos = disk_center + dir * (DISK_RADIUS + r - 0.02);
            let body = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(pos)
                    .rotation(phi),
            );
            let collider = colliders.insert_with_parent(
                ColliderBuilder::capsule_y(half_height, r).friction(FRICTION),
                body,
                &mut bodies,
            );
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

    testbed.set_world_with_params(
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        Vector::new(0.0, -9.81),
        physics_hooks,
    );
    testbed.look_at(disk_center, 10.0);
}
