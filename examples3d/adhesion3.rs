use rapier_testbed3d::TestbedViewer;
use rapier3d::prelude::*;

/// The pull each sticky surface receives, in newtons. A box of 1 m on each side weighs about
/// 9.81 N, thus this holds it and about 20 N more.
const ADHESION: Real = 30.0;

/// Which kind of request the hook makes; the setting of the testbed selects it.
#[derive(Clone, Copy)]
enum Kind {
    Force,
    Pressure,
    Budget,
}

/// Requests an adhesion on every contact manifold of a sticky collider.
struct AdhesionHook {
    sticky: Vec<ColliderHandle>,
    kind: Kind,
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let Some(sticky) = self
            .sticky
            .iter()
            .find(|handle| **handle == context.collider1 || **handle == context.collider2)
        else {
            return;
        };
        match self.kind {
            Kind::Force => *context.adhesion_force = ADHESION,
            // The face of each box is 1 m², thus this is the same pull as `Kind::Force` over one
            // collider.
            Kind::Pressure => *context.adhesion_pressure = ADHESION,
            Kind::Budget => {
                *context.adhesion_budget = Some(AdhesionBudget {
                    owner: *sticky,
                    channel: 0,
                    total: ADHESION,
                })
            }
        }
    }
}

/// Adds a fixed slab tilted by `angle` about the z axis, with a box of 1 m on each side that
/// clings to its outward face. The box is the sticky collider.
fn add_sticky_slab(
    world: &mut PhysicsWorld,
    sticky: &mut Vec<ColliderHandle>,
    center: Vector,
    angle: Real,
    friction: Real,
) {
    let slab_body = world.insert_body(RigidBodyBuilder::fixed());
    world.insert_collider(
        ColliderBuilder::cuboid(3.0, 0.25, 3.0)
            .translation(center)
            .rotation(Vector::new(0.0, 0.0, angle))
            .friction(friction),
        Some(slab_body),
    );

    // The outward face of the slab is its local +y turned by `angle`. The box sits against it,
    // with a little overlap so that the contact is active at once.
    let face_normal = Vector::new(-angle.sin(), angle.cos(), 0.0);
    let (_, collider) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(center + face_normal * (0.25 + 0.5 - 0.01))
            .rotation(Vector::new(0.0, 0.0, angle)),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5)
            .friction(friction)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
    );
    sticky.push(collider);
}

/// Adds a ceiling made of `tiles` by `tiles` colliders, with a box of 1 m on each side that hangs
/// under its middle and receives the downward load `load`. The box is the sticky collider.
fn add_tiled_ceiling(
    world: &mut PhysicsWorld,
    sticky: &mut Vec<ColliderHandle>,
    center: Vector,
    tiles: usize,
    load: Real,
) {
    let half_width = 3.0 / tiles as Real;
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    for i in 0..tiles {
        for j in 0..tiles {
            // The tiles are centered on the odd multiples of their half-width, thus a seam falls
            // under the middle of the box whenever there is more than one of them.
            let offset = |k: usize| half_width * (2 * k + 1) as Real - 3.0;
            world.insert_collider(
                ColliderBuilder::cuboid(half_width, 0.25, half_width)
                    .translation(center + Vector::new(offset(i), 0.25, offset(j))),
                Some(ceiling_body),
            );
        }
    }

    let (body, collider) = world.insert(
        RigidBodyBuilder::dynamic().translation(center - Vector::new(0.0, 0.5, 0.0)),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5).active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
    );
    world.bodies[body].add_force(Vector::new(0.0, -load, 0.0), true);
    sticky.push(collider);
}

pub async fn run(viewer: &mut TestbedViewer) -> anyhow::Result<()> {
    /*
     * The kind of adhesion the hook requests.
     */
    let kind = match viewer.example_settings_mut().get_or_set_string(
        "Adhesion",
        0,
        vec![
            "Force".to_string(),
            "Pressure".to_string(),
            "Budget".to_string(),
        ],
    ) {
        1 => Kind::Pressure,
        2 => Kind::Budget,
        _ => Kind::Force,
    };

    /*
     * World
     */
    let mut world = PhysicsWorld::new();
    let mut sticky = Vec::new();

    /*
     * A row of sticky slabs, from vertical to a full overhang. Each box holds on, even where
     * the surface points downward. The last slab has a low friction: its box stays against the
     * surface, but it slides down along it.
     */
    let angles: [Real; 4] = [90.0, 120.0, 150.0, 180.0];
    for (k, angle) in angles.into_iter().enumerate() {
        let center = Vector::new(k as Real * 7.0 - 18.0, 6.0, 0.0);
        add_sticky_slab(&mut world, &mut sticky, center, angle.to_radians(), 1.0);
    }
    let overhang: Real = 135.0;
    add_sticky_slab(
        &mut world,
        &mut sticky,
        Vector::new(14.0, 8.0, 0.0),
        overhang.to_radians(),
        0.03,
    );

    /*
     * A row of ceilings of the same size, made of more and more tiles. Each holds a box that
     * carries a load of 25 N, which one collider worth of pull cannot hold.
     *
     * A `Force` is requested once for each manifold: the box under one collider falls, but the
     * boxes under the ceilings of several tiles receive several times the pull and hang. A
     * `Pressure` and a `Budget` give every box the same pull, thus the whole row falls. Change
     * the setting to see the row become uniform.
     */
    for (k, tiles) in [1, 2, 3, 4].into_iter().enumerate() {
        let center = Vector::new(k as Real * 8.0 - 12.0, -3.0, 0.0);
        add_tiled_ceiling(&mut world, &mut sticky, center, tiles, 25.0);
    }

    /*
     * A floor to catch what falls.
     */
    let floor_body = world.insert_body(RigidBodyBuilder::fixed());
    world.insert_collider(
        ColliderBuilder::cuboid(35.0, 0.5, 10.0).translation(Vector::new(0.0, -12.0, 0.0)),
        Some(floor_body),
    );

    let physics_hooks = AdhesionHook { sticky, kind };

    /*
     * Set up the testbed.
     */
    viewer.set_world(&mut world);
    viewer.look_at(Vec3::new(0.0, 0.0, 45.0), Vec3::ZERO);

    while viewer.render_frame(&mut world).await {
        if viewer.simulating() {
            world.step_with_events(&physics_hooks, &());
        }
    }
    Ok(())
}
