use rapier_testbed2d::Testbed;
use rapier2d::prelude::*;

/// Sets a fixed adhesion force on every contact manifold that involves one of the registered
/// "sticky" surface colliders.
struct AdhesionHook {
    sticky: Vec<ColliderHandle>,
    force: Real,
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        if self.sticky.contains(&context.collider1) || self.sticky.contains(&context.collider2) {
            *context.adhesion_force = self.force;
        }
    }
}

const ADHESION_FORCE: Real = 60.0;

/// Adds a fixed slab tilted by `angle` (radians) with adhesion enabled, plus a 1x1 dynamic box
/// clinging to its outward face. The surface collider is registered as "sticky".
fn add_sticky_surface(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    sticky: &mut Vec<ColliderHandle>,
    center: Vector,
    half_len: Real,
    angle: Real,
    friction: Real,
) {
    // The fixed surface (a thin slab), rotated, with contact modification enabled.
    let surface_body = bodies.insert(RigidBodyBuilder::fixed());
    let surface = colliders.insert_with_parent(
        ColliderBuilder::cuboid(half_len, 0.25)
            .translation(center)
            .rotation(angle)
            .friction(friction)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        surface_body,
        bodies,
    );
    sticky.push(surface);

    // The slab's outward face normal (its local +Y rotated by `angle`); the box clings on this
    // side.
    let face_normal = Vector::new(-angle.sin(), angle.cos());
    // Place the box just touching that face (a hair of overlap guarantees an active contact).
    let box_center = center + face_normal * (0.25 + 0.5 - 0.01);
    let box_body = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(box_center)
            .rotation(angle),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5).friction(friction),
        box_body,
        bodies,
    );
}

pub fn init_world(testbed: &mut Testbed) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let mut sticky = Vec::new();

    let deg = |d: Real| d * std::f32::consts::PI / 180.0;

    // A row of sticky surfaces from vertical (90°) through a full overhang / ceiling (180°), all
    // with high friction: each box clings to the (increasingly overhanging) face and is held in
    // place — hanging on even when the surface is past vertical.
    add_sticky_surface(&mut bodies, &mut colliders, &mut sticky, Vector::new(-12.0, 6.0), 4.0, deg(90.0), 1.0);
    add_sticky_surface(&mut bodies, &mut colliders, &mut sticky, Vector::new(-6.0, 6.0), 4.0, deg(120.0), 1.0);
    add_sticky_surface(&mut bodies, &mut colliders, &mut sticky, Vector::new(0.0, 6.0), 4.0, deg(150.0), 1.0);
    add_sticky_surface(&mut bodies, &mut colliders, &mut sticky, Vector::new(6.0, 6.0), 4.0, deg(180.0), 1.0);

    // A beyond-vertical overhang with LOW friction: the box stays attached (held normal-wise by
    // adhesion) but slides down along the surface under gravity.
    add_sticky_surface(&mut bodies, &mut colliders, &mut sticky, Vector::new(15.0, 9.0), 7.0, deg(135.0), 0.03);

    // A floor to catch anything that slides off the low-friction overhang.
    let floor_body = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(35.0, 0.5).translation(Vector::new(0.0, -6.0)),
        floor_body,
        &mut bodies,
    );

    let physics_hooks = AdhesionHook {
        sticky,
        force: ADHESION_FORCE,
    };

    testbed.set_world_with_params(
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        Vector::new(0.0, -9.81),
        physics_hooks,
    );
    testbed.look_at(Vec2::new(2.0, 2.0), 18.0);
}
