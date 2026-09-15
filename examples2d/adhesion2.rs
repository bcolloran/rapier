use rapier_testbed2d::TestbedViewer;
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
    world: &mut PhysicsWorld,
    sticky: &mut Vec<ColliderHandle>,
    center: Vector,
    half_len: Real,
    angle: Real,
    friction: Real,
) {
    // The fixed surface (a thin slab), rotated, with contact modification enabled.
    let surface_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let surface = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(half_len, 0.25)
            .translation(center)
            .rotation(angle)
            .friction(friction)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        surface_body,
        &mut world.bodies,
    );
    sticky.push(surface);

    // The slab's outward face normal (its local +Y rotated by `angle`); the box clings on this
    // side.
    let face_normal = Vector::new(-angle.sin(), angle.cos());
    // Place the box just touching that face (a hair of overlap guarantees an active contact).
    let box_center = center + face_normal * (0.25 + 0.5 - 0.01);
    let box_body = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(box_center)
            .rotation(angle),
    );
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5).friction(friction),
        box_body,
        &mut world.bodies,
    );
}

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81);
    let mut sticky = Vec::new();

    let deg = |d: Real| d * std::f32::consts::PI / 180.0;

    // A row of sticky surfaces from vertical (90°) through a full overhang / ceiling (180°), all
    // with high friction: each box clings to the (increasingly overhanging) face and is held in
    // place — hanging on even when the surface is past vertical.
    add_sticky_surface(
        &mut world,
        &mut sticky,
        Vector::new(-12.0, 6.0),
        4.0,
        deg(90.0),
        1.0,
    );
    add_sticky_surface(
        &mut world,
        &mut sticky,
        Vector::new(-6.0, 6.0),
        4.0,
        deg(120.0),
        1.0,
    );
    add_sticky_surface(
        &mut world,
        &mut sticky,
        Vector::new(0.0, 6.0),
        4.0,
        deg(150.0),
        1.0,
    );
    add_sticky_surface(
        &mut world,
        &mut sticky,
        Vector::new(6.0, 6.0),
        4.0,
        deg(180.0),
        1.0,
    );

    // A beyond-vertical overhang with LOW friction: the box stays attached (held normal-wise by
    // adhesion) but slides down along the surface under gravity.
    add_sticky_surface(
        &mut world,
        &mut sticky,
        Vector::new(15.0, 9.0),
        7.0,
        deg(135.0),
        0.03,
    );

    // A floor to catch anything that slides off the low-friction overhang.
    let floor_body = world.bodies.insert(RigidBodyBuilder::fixed());
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(35.0, 0.5).translation(Vector::new(0.0, -6.0)),
        floor_body,
        &mut world.bodies,
    );

    let physics_hooks = AdhesionHook {
        sticky,
        force: ADHESION_FORCE,
    };

    viewer.set_world(&mut world);
    viewer.look_at(Vec2::new(2.0, 2.0), 18.0);

    while viewer.render_frame(&mut world).await {
        if viewer.simulating() {
            // Detects collisions last when Settings > Advanced > "Collisions last" is checked
            // (`set_world` applies it to `world.collisions_last`).
            world.step_with_events(&physics_hooks, &());
        }
    }
    Ok(())
}
