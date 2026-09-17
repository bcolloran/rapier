//! Contact adhesion: the attractive pull that a contact-modification hook requests through
//! `ContactModificationContext::adhesion_force`, `adhesion_pressure` and `adhesion_budget`.
//!
//! The engine applies the pull as an ordinary external force on both bodies just before the
//! contact solver runs, thus the push-only contacts supply the reaction that holds the bodies
//! together, the threshold at which they separate, and the friction. Most scenes use a box of
//! 1 m on each side (mass 1, weight `G`): the adhesion holds while it is larger than the weight
//! plus the load, and the reaction to the pull gives friction up to `mu` times the adhesion.
//!
//! The scenes are built in the xy plane and are 1 m deep in 3D, thus the same numbers apply in
//! both dimensions: a box of 1 m on each side has a mass of 1 in 2D and in 3D, and the extent of
//! its face is 1 (one meter of length in 2D, one square meter of area in 3D).

use crate::alloc_prelude::*;
use crate::prelude::*;

const G: Real = 9.81;

/// The half-depth of every shape in 3D. It makes each scene 1 m deep.
#[cfg(feature = "dim3")]
const HALF_DEPTH: Real = 0.5;

fn new_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = xy(0.0, -G);
    world
}

fn run(world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks, steps: usize) {
    for _ in 0..steps {
        world.step_with_events(hooks, &());
    }
}

fn assert_close(got: Real, expected: Real, epsilon: Real, what: &str) {
    assert!(
        (got - expected).abs() <= epsilon,
        "{what}: got {got}, expected {expected} +/- {epsilon}"
    );
}

/// A point of the xy plane the scenes are built in.
#[cfg(feature = "dim2")]
fn xy(x: Real, y: Real) -> Vector {
    Vector::new(x, y)
}

/// A point of the xy plane the scenes are built in.
#[cfg(feature = "dim3")]
fn xy(x: Real, y: Real) -> Vector {
    Vector::new(x, y, 0.0)
}

/// A box with the given half-extents in the xy plane.
#[cfg(feature = "dim2")]
fn cuboid(hx: Real, hy: Real) -> ColliderBuilder {
    ColliderBuilder::cuboid(hx, hy)
}

/// A box with the given half-extents in the xy plane.
#[cfg(feature = "dim3")]
fn cuboid(hx: Real, hy: Real) -> ColliderBuilder {
    ColliderBuilder::cuboid(hx, hy, HALF_DEPTH)
}

/// A rotation of `angle` in the xy plane, as the builders take it.
#[cfg(feature = "dim2")]
fn spin(angle: Real) -> Real {
    angle
}

/// A rotation of `angle` in the xy plane, as the builders take it.
#[cfg(feature = "dim3")]
fn spin(angle: Real) -> Vector {
    Vector::new(0.0, 0.0, angle)
}

/// The rotation of `body` in the xy plane.
#[cfg(feature = "dim2")]
fn tilt(body: &RigidBody) -> Real {
    body.rotation().angle()
}

/// The rotation of `body` in the xy plane.
#[cfg(feature = "dim3")]
fn tilt(body: &RigidBody) -> Real {
    body.rotation().to_scaled_axis().z
}

/// A hinge about the axis perpendicular to the xy plane.
#[cfg(feature = "dim2")]
fn hinge() -> RevoluteJointBuilder {
    RevoluteJointBuilder::new()
}

/// A hinge about the axis perpendicular to the xy plane.
#[cfg(feature = "dim3")]
fn hinge() -> RevoluteJointBuilder {
    RevoluteJointBuilder::new(Vector::Z)
}

/// The bit pattern of a number, so that two runs can be compared exactly.
fn bits_of(x: &Real) -> u64 {
    u64::from(x.to_bits())
}

/// The bit patterns of the pose and the velocity of every body, to compare two runs exactly.
fn body_bits(world: &PhysicsWorld) -> Vec<u64> {
    let mut handles: Vec<_> = world.bodies.iter().map(|(handle, _)| handle).collect();
    handles.sort_by_key(|handle| handle.into_raw_parts());
    let mut bits = Vec::new();
    for handle in handles {
        let body = &world.bodies[handle];
        bits.extend(body.translation().to_array().iter().map(bits_of));
        bits.extend(body.linvel().to_array().iter().map(bits_of));
        bits.push(bits_of(&tilt(body)));
    }
    bits
}

/*
 * Hooks.
 */

/// What a hook requests on a manifold.
#[derive(Clone, Copy, Debug)]
enum Request {
    Force(Real),
    Pressure(Real),
    Budget(Real),
}

impl Request {
    fn apply(self, context: &mut ContactModificationContext, owner: ColliderHandle, channel: u32) {
        match self {
            Request::Force(force) => *context.adhesion_force = force,
            Request::Pressure(pressure) => *context.adhesion_pressure = pressure,
            Request::Budget(total) => {
                *context.adhesion_budget = Some(AdhesionBudget {
                    owner,
                    channel,
                    total,
                })
            }
        }
    }
}

/// Requests `request` on every manifold that involves `collider`.
struct AdhesionHook {
    collider: ColliderHandle,
    request: Request,
}

impl AdhesionHook {
    fn new(collider: ColliderHandle, request: Request) -> Self {
        Self { collider, request }
    }
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        if context.collider1 == self.collider || context.collider2 == self.collider {
            self.request.apply(context, self.collider, 0);
        }
    }
}

/// Requests a different `request` for each of several colliders, for scenes that mix the kinds
/// of adhesion.
struct MixedHook {
    requests: Vec<(ColliderHandle, Request)>,
}

impl PhysicsHooks for MixedHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for (collider, request) in &self.requests {
            if context.collider1 == *collider || context.collider2 == *collider {
                request.apply(context, *collider, 0);
            }
        }
    }
}

/*
 * Scenes.
 */

/// A fixed surface whose underside is the plane y = 0, with contact modification enabled, and a
/// dynamic box of 1 m on each side that hangs under it at `x`. Returns the collider of the
/// ceiling and the body of the box.
fn ceiling_and_hanging_box_at(
    world: &mut PhysicsWorld,
    x: Real,
) -> (ColliderHandle, RigidBodyHandle) {
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let ceiling = world.insert_collider(
        cuboid(5.0, 0.5)
            .translation(xy(x, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );

    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(xy(x, -0.5)),
        cuboid(0.5, 0.5),
    );

    (ceiling, box_body)
}

fn ceiling_and_hanging_box(world: &mut PhysicsWorld) -> (ColliderHandle, RigidBodyHandle) {
    ceiling_and_hanging_box_at(world, 0.0)
}

/// A fixed vertical wall whose left face is the plane x = 0, with contact modification enabled,
/// and a dynamic box of 1 m on each side that touches it from the left. Both have the friction
/// `mu`. Returns the collider of the wall and the body of the box.
fn wall_and_box(world: &mut PhysicsWorld, mu: Real) -> (ColliderHandle, RigidBodyHandle) {
    let wall_body = world.insert_body(RigidBodyBuilder::fixed());
    let wall = world.insert_collider(
        cuboid(0.5, 10.0)
            .translation(xy(0.5, 0.0))
            .friction(mu)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(wall_body),
    );

    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(xy(-0.5, 0.0)),
        cuboid(0.5, 0.5).friction(mu),
    );

    (wall, box_body)
}

/*
 * `adhesion_force`: an absolute force for each manifold.
 */

#[test]
fn adhesion_holds_box_below_ceiling() {
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    // The adhesion is much larger than the weight of the box (about 9.81 N): it must hang.
    let hook = AdhesionHook::new(ceiling, Request::Force(30.0));
    run(&mut world, &hook, 300);

    let body = &world.bodies[box_body];
    let y = body.translation().y;
    assert!(
        y > -0.6,
        "the box fell to y = {y} although the pull is large"
    );
    // It must also be at rest: the pull must not add energy over many steps.
    assert_close(body.linvel().y, 0.0, 0.1, "vertical speed of the box");
}

#[test]
fn weak_adhesion_lets_box_fall() {
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    // The adhesion is much smaller than the weight of the box: it cannot hold it.
    let hook = AdhesionHook::new(ceiling, Request::Force(3.0));
    run(&mut world, &hook, 120);

    let y = world.bodies[box_body].translation().y;
    assert!(y < -1.0, "the box should have fallen but is at y = {y}");
}

#[test]
fn adhesion_break_threshold() {
    // The box holds while the pull is larger than its weight plus the load...
    let hang = |load: Real| -> Real {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let weight = world.bodies[box_body].mass() * G;
        world.bodies[box_body].add_force(xy(0.0, -load), true);
        let hook = AdhesionHook::new(ceiling, Request::Force(weight + 3.0));
        run(&mut world, &hook, 300);
        world.bodies[box_body].translation().y
    };

    let y = hang(1.0);
    assert!(y > -0.6, "the box should hold a load of 1 N (y = {y})");
    // ...and separates as soon as the load passes the threshold.
    let y = hang(5.0);
    assert!(
        y < -1.0,
        "the box should break free of a 5 N load (y = {y})"
    );
}

#[test]
fn adhesion_friction_holds_box_on_vertical_wall() {
    // The friction the pull makes available is mu * 20 = 20 N, more than the weight of the box:
    // it must not slide.
    let mut world = new_world();
    let (wall, box_body) = wall_and_box(&mut world, 1.0);
    let y0 = world.bodies[box_body].translation().y;

    let hook = AdhesionHook::new(wall, Request::Force(20.0));
    run(&mut world, &hook, 300);

    let body = &world.bodies[box_body];
    assert_close(body.translation().x, -0.5, 0.05, "x of the box on the wall");
    assert_close(body.translation().y, y0, 0.1, "y of the box on the wall");
}

#[test]
fn adhesion_low_friction_slides_but_stays_attached() {
    // The friction the pull makes available is mu * 20 = 2 N, less than the weight of the box:
    // it slides down, but it stays against the wall.
    let mut world = new_world();
    let (wall, box_body) = wall_and_box(&mut world, 0.1);
    let y0 = world.bodies[box_body].translation().y;

    let hook = AdhesionHook::new(wall, Request::Force(20.0));
    // Few enough steps to keep the box inside the vertical extent of the wall.
    run(&mut world, &hook, 60);

    let body = &world.bodies[box_body];
    let y = body.translation().y;
    assert!(y < y0 - 0.5, "the box should have slid down (y = {y})");
    assert_close(body.translation().x, -0.5, 0.1, "x of the sliding box");
}

#[test]
fn zero_adhesion_is_inert() {
    // A hook that requests no adhesion must leave an ordinary contact: the box rests on the
    // ground and is not pulled into it.
    let mut world = new_world();
    let ground_body = world.insert_body(RigidBodyBuilder::fixed());
    let ground = world.insert_collider(
        cuboid(5.0, 0.5)
            .translation(xy(0.0, -0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ground_body),
    );
    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(xy(0.0, 0.5)),
        cuboid(0.5, 0.5),
    );

    let hook = AdhesionHook::new(ground, Request::Force(0.0));
    run(&mut world, &hook, 300);

    let body = &world.bodies[box_body];
    assert_close(body.translation().y, 0.5, 0.05, "y of the resting box");
    assert_close(body.linvel().y, 0.0, 0.05, "vertical speed of the box");
}

#[test]
fn adhesion_holds_box_on_a_surface_past_the_vertical() {
    // A slab tilted by 135 degrees overhangs: its sticky face points partly downward and gravity
    // peels the box away from it. A large pull and a large friction must still hold the box in
    // place.
    let angle = Real::to_radians(135.0);
    let mut world = new_world();

    let center = xy(0.0, 5.0);
    let slab_body = world.insert_body(RigidBodyBuilder::fixed());
    let slab = world.insert_collider(
        cuboid(4.0, 0.25)
            .translation(center)
            .rotation(spin(angle))
            .friction(1.0)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(slab_body),
    );

    // The box clings to the outward face of the slab, which is its local +y turned by `angle`.
    let face_normal = xy(-angle.sin(), angle.cos());
    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(center + face_normal * (0.25 + 0.5 - 0.01))
            .rotation(spin(angle)),
        cuboid(0.5, 0.5).friction(1.0),
    );

    let start = world.bodies[box_body].translation();
    let hook = AdhesionHook::new(slab, Request::Force(60.0));
    run(&mut world, &hook, 300);

    let end = world.bodies[box_body].translation();
    assert_close(end.x, start.x, 0.15, "x of the box on the overhang");
    assert_close(end.y, start.y, 0.15, "y of the box on the overhang");
}

#[test]
fn adhesion_applies_no_net_torque() {
    // Adhesion acts on the two bodies of a manifold in opposite directions, thus it must add no
    // torque to the system it acts inside. The scene is a plank of 10 m hung from a hinge 3 m
    // above its center, with a box standing 4 m out on each side, and only the left box adheres
    // to it. A steady net torque T would settle the plank at a permanent angle of about
    // T / (M g d); the threshold of 0.002 rad sees about 0.1 N m.
    let settle = |adhesion: Option<Real>| -> Real {
        let mut world = new_world();
        let pivot = world.insert_body(RigidBodyBuilder::fixed().translation(xy(0.0, 3.0)));
        let plank_body = world.insert_body(
            RigidBodyBuilder::dynamic()
                // Damped, so that the angle can be read once the plank is at rest.
                .angular_damping(2.0)
                .can_sleep(false),
        );
        world.insert_collider(cuboid(5.0, 0.5).mass(1.0).friction(2.0), Some(plank_body));
        world.insert_impulse_joint(pivot, plank_body, hinge().local_anchor2(xy(0.0, 3.0)));

        let mut adhesive = None;
        let sides: [Real; 2] = [-1.0, 1.0];
        for side in sides {
            let body = world.insert_body(
                RigidBodyBuilder::dynamic()
                    .translation(xy(side * 4.0, 1.0))
                    .lock_rotations()
                    .can_sleep(false),
            );
            let adheres = side < 0.0 && adhesion.is_some();
            let mut builder = cuboid(0.5, 0.5).mass(1.0).friction(2.0);
            if adheres {
                builder = builder.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
            }
            let collider = world.insert_collider(builder, Some(body));
            if adheres {
                adhesive = Some(collider);
            }
        }

        match (adhesion, adhesive) {
            (Some(force), Some(collider)) => {
                let hook = AdhesionHook::new(collider, Request::Force(force));
                run(&mut world, &hook, 600);
            }
            _ => run(&mut world, &(), 600),
        }
        tilt(&world.bodies[plank_body])
    };

    let level = settle(None);
    let adhering = settle(Some(20.0));
    assert!(
        level.abs() < 0.002,
        "the plank of the reference run settled tilted by {level} rad"
    );
    assert!(
        adhering.abs() < 0.002,
        "adhesion applied a steady torque: the plank settled at {adhering} rad"
    );
}

#[test]
fn adhesion_does_not_drag_a_dominant_body() {
    // The solver holds a body of a higher dominance group in place for its contacts with bodies
    // of a lower group: it receives no contact reaction. Pulling it would therefore drag it
    // through the contact, thus only the body of the lower group is pulled. Without gravity, the
    // dominant slab must not move at all, while the adhesion still holds the small box against a
    // load of 20 N that pulls it away.
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let (slab_body, slab) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(xy(0.0, 0.5))
            .dominance_group(10)
            .can_sleep(false),
        cuboid(2.0, 0.5).active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
    );
    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(xy(0.0, -0.5))
            .can_sleep(false),
        cuboid(0.5, 0.5),
    );
    world.bodies[box_body].add_force(xy(0.0, -20.0), true);

    let hook = AdhesionHook::new(slab, Request::Force(50.0));
    run(&mut world, &hook, 180);

    let slab_body = &world.bodies[slab_body];
    assert_eq!(
        slab_body.translation(),
        xy(0.0, 0.5),
        "adhesion dragged the dominant slab"
    );
    assert_eq!(slab_body.linvel(), Vector::ZERO);

    let y = world.bodies[box_body].translation().y;
    assert_close(y, -0.5, 0.05, "y of the box held against a load of 20 N");
}

#[test]
fn adhesion_holds_a_multibody_link() {
    // The solver reads the manifolds of a multibody through a list of its own, and not through
    // the color buckets that hold the contacts between two plain bodies. The adhesion must reach
    // that list too. The link can only slide up and down, thus it hangs or it falls.
    let hangs = |force: Real| -> bool {
        let mut world = new_world();
        Ceiling::One.build(&mut world);
        // The base has no collider. It sits where the link starts, because inserting the joint
        // places the link from the base and the rest position of the joint.
        let base = world.insert_body(RigidBodyBuilder::fixed().translation(xy(0.0, -0.5)));
        let (link_body, link) = world.insert(
            RigidBodyBuilder::dynamic().translation(xy(0.0, -0.5)),
            cuboid(0.5, 0.5).active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        );
        world.insert_multibody_joint(base, link_body, PrismaticJointBuilder::new(Vector::Y));

        let hook = AdhesionHook::new(link, Request::Force(force));
        run(&mut world, &hook, 200);
        world.bodies[link_body].translation().y > -0.6
    };

    assert!(!hangs(0.0), "the link should fall with no adhesion");
    assert!(hangs(30.0), "the adhesion should hold the link");
}

#[test]
fn non_positive_and_nan_requests_do_not_change_the_simulation() {
    // A request that is zero, negative or not a number must write no force at all, thus the
    // simulation must match a hook that requests nothing, bit for bit.
    struct Inert;
    impl PhysicsHooks for Inert {
        fn modify_solver_contacts(&self, _: &mut ContactModificationContext) {}
    }
    struct NonPositive {
        values: [Real; 3],
    }
    impl PhysicsHooks for NonPositive {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            let [force, pressure, total] = self.values;
            *context.adhesion_force = force;
            *context.adhesion_pressure = pressure;
            *context.adhesion_budget = Some(AdhesionBudget {
                owner: context.collider1,
                channel: 7,
                total,
            });
        }
    }

    let simulate = |hooks: &dyn PhysicsHooks| -> Vec<u64> {
        let mut world = new_world();
        ceiling_and_box(&mut world, Ceiling::OverlappingTiles);
        ceiling_and_hanging_box_at(&mut world, 30.0);
        run(&mut world, hooks, 90);
        body_bits(&world)
    };

    let reference = simulate(&Inert);
    for values in [
        [0.0, 0.0, 0.0],
        [-30.0, -30.0, -30.0],
        [Real::NAN, Real::NAN, Real::NAN],
        [Real::NEG_INFINITY, Real::NEG_INFINITY, Real::NEG_INFINITY],
    ] {
        assert!(
            simulate(&NonPositive { values }) == reference,
            "the request {values:?} changed the simulation"
        );
    }
}

/*
 * The lifetime of a request: it is made again at each update of a pair, and it stops as soon as
 * the hook stops.
 */

#[test]
fn adhesion_stops_when_the_hook_flag_is_removed() {
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    let hook = AdhesionHook::new(ceiling, Request::Force(30.0));
    run(&mut world, &hook, 120);
    let y = world.bodies[box_body].translation().y;
    assert!(y > -0.6, "the box should hang while the hook runs");

    // Withdraw contact modification. The manifold is still alive, but it must not keep the
    // adhesion that was requested last. (The box is woken up: a sleeping pair is frozen as a
    // whole, which is not the case under test.)
    world.colliders[ceiling].set_active_hooks(ActiveHooks::empty());
    world.bodies[box_body].wake_up(true);
    run(&mut world, &hook, 120);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y < -1.0,
        "the box should fall once the flag is gone (y = {y})"
    );
}

#[test]
fn adhesion_stops_when_the_hook_flag_is_removed_from_a_resting_pair() {
    // The contact recycling path. `Collider::set_active_hooks` does not mark the collider as
    // changed, and a pair at rest almost does not move, thus the pair would qualify for
    // recycling as soon as the flag is gone, and would keep its adhesion. The narrow phase must
    // refuse to recycle a pair that stored a request, so that the next update clears it. The
    // scene is a box on a hooked floor, held down by 30 N of adhesion against a net upward load
    // of 5 N: it must lift off once the flag is removed.
    let mut world = new_world();
    assert!(world.integration_parameters.contact_recycling);
    let floor_body = world.insert_body(RigidBodyBuilder::fixed());
    let floor = world.insert_collider(
        cuboid(5.0, 0.5)
            .translation(xy(0.0, -0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(floor_body),
    );
    let (box_body, box_collider) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(xy(0.0, 0.5))
            .can_sleep(false),
        cuboid(0.5, 0.5),
    );
    let weight = world.bodies[box_body].mass() * G;
    world.bodies[box_body].add_force(xy(0.0, weight + 5.0), true);

    let hook = AdhesionHook::new(floor, Request::Force(30.0));
    run(&mut world, &hook, 120);
    let y = world.bodies[box_body].translation().y;
    assert_close(y, 0.5, 0.05, "y of the box held on the floor");

    world.colliders[floor].set_active_hooks(ActiveHooks::empty());
    run(&mut world, &hook, 1);
    let pair = world
        .narrow_phase
        .contact_pair(floor, box_collider)
        .unwrap();
    assert!(
        pair.manifolds
            .iter()
            .all(|manifold| manifold.data.adhesion_force == 0.0),
        "the first update after the removal of the flag must clear the request"
    );

    run(&mut world, &hook, 60);
    let y = world.bodies[box_body].translation().y;
    assert!(y > 1.0, "the box should lift off (y = {y})");
}

#[test]
fn adhesion_survives_sleep_and_wake_by_impact() {
    // A box held under a ceiling falls asleep. The narrow phase then skips its pair and the hook
    // does not run, thus the manifold must keep the adhesion requested before the sleep, and a
    // ball thrown into the box from below must not detach it.
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
    let box_collider = world.bodies[box_body].colliders()[0];
    let hook = AdhesionHook::new(ceiling, Request::Force(30.0));

    let mut slept = false;
    for _ in 0..600 {
        run(&mut world, &hook, 1);
        if world.bodies[box_body].is_sleeping() {
            slept = true;
            break;
        }
    }
    assert!(slept, "the box that adheres never fell asleep");
    run(&mut world, &hook, 30);
    let y_asleep = world.bodies[box_body].translation().y;

    let pair = world
        .narrow_phase
        .contact_pair(ceiling, box_collider)
        .expect("the ceiling and the box must have a contact pair");
    assert!(
        pair.manifolds
            .iter()
            .any(|manifold| manifold.data.adhesion_force == 30.0),
        "a sleeping pair must keep the request made last"
    );

    // Throw a ball up into the box.
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(xy(0.1, -3.0))
            .linvel(xy(0.0, 12.0)),
        ColliderBuilder::ball(0.25),
    );
    let mut woke = false;
    for _ in 0..60 {
        run(&mut world, &hook, 1);
        woke |= !world.bodies[box_body].is_sleeping();
        let y = world.bodies[box_body].translation().y;
        assert!(
            y > y_asleep - 0.05,
            "the box detached when it woke (y = {y})"
        );
    }
    assert!(woke, "the ball never woke the box");

    run(&mut world, &hook, 240);
    let y = world.bodies[box_body].translation().y;
    assert_close(y, y_asleep, 0.05, "y of the box after it woke");
}

#[test]
fn contact_recycling_does_not_change_adhesion_results() {
    // A pair with hooks is never recycled, and every pair of this scene involves a hooked
    // collider, thus the result must be the same with recycling on and off, bit for bit.
    let simulate = |recycling: bool| -> Vec<u64> {
        let mut world = new_world();
        world.integration_parameters.contact_recycling = recycling;
        let (tiled_box, _) = ceiling_and_box(&mut world, Ceiling::OverlappingTiles);
        let (ceiling, _) = ceiling_and_hanging_box_at(&mut world, 30.0);
        let wall_body = world.insert_body(RigidBodyBuilder::fixed());
        let wall = world.insert_collider(
            cuboid(0.5, 10.0)
                .translation(xy(-29.5, 0.0))
                .friction(0.1)
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(wall_body),
        );
        world.insert(
            RigidBodyBuilder::dynamic().translation(xy(-30.5, 0.0)),
            cuboid(0.5, 0.5).friction(0.1),
        );
        let hook = MixedHook {
            requests: vec![
                (tiled_box, Request::Budget(12.0)),
                (ceiling, Request::Force(30.0)),
                (wall, Request::Pressure(20.0)),
            ],
        };
        run(&mut world, &hook, 240);
        body_bits(&world)
    };

    assert!(
        simulate(true) == simulate(false),
        "the result differs with contact recycling on and off"
    );
}

/*
 * `adhesion_pressure` and `adhesion_budget`: the pull must not depend on the number of colliders
 * that make a surface. Each test measures the largest load the box holds, which is the pull it
 * receives minus its weight.
 */

/// How the ceiling of the composition tests is made of colliders. Every form covers at least the
/// plane y = 0 from x = -3 to x = 3.
#[derive(Clone, Copy, Debug)]
enum Ceiling {
    /// One collider.
    One,
    /// Tiles of the given half-width, side by side, with a seam at x = 0. The face of the box
    /// straddles that seam, thus its pair with the ceiling has several manifolds.
    Tiles(Real),
    /// Tiles of half-width 0.5, plus a second layer shifted by half a tile: almost every point of
    /// the surface belongs to two colliders. The patches that overlap are measured two times,
    /// which only a budget can handle.
    OverlappingTiles,
}

impl Ceiling {
    /// Adds the colliders of this ceiling to `world`, under one fixed body.
    fn build(self, world: &mut PhysicsWorld) {
        let body = world.insert_body(RigidBodyBuilder::fixed());
        let mut tile = |half_width: Real, center: Real| {
            world.insert_collider(
                cuboid(half_width, 0.5).translation(xy(center, 0.5)),
                Some(body),
            );
        };
        // Tiles centered on the odd multiples of their half-width put their seams on the even
        // ones, thus a seam always falls at x = 0, under the middle of the box.
        let mut tiles = |half_width: Real, shift: Real| {
            for k in 0..(3.0 / (2.0 * half_width)).ceil() as usize {
                let center = half_width * (2 * k + 1) as Real;
                tile(half_width, center + shift);
                tile(half_width, -center + shift);
            }
        };
        match self {
            Ceiling::One => tile(3.0, 0.0),
            Ceiling::Tiles(half_width) => tiles(half_width, 0.0),
            Ceiling::OverlappingTiles => {
                tiles(0.5, 0.0);
                tiles(0.5, 0.5);
            }
        }
    }
}

/// A ceiling of the given form, and a dynamic box of 1 m on each side that hangs under it, with
/// contact modification enabled on the box. Returns the collider of the box and its body.
fn ceiling_and_box(
    world: &mut PhysicsWorld,
    ceiling: Ceiling,
) -> (ColliderHandle, RigidBodyHandle) {
    ceiling.build(world);
    let (box_body, box_collider) = world.insert(
        RigidBodyBuilder::dynamic().translation(xy(0.0, -0.5)),
        cuboid(0.5, 0.5).active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
    );
    (box_collider, box_body)
}

/// The largest downward load, in newtons, that `request` holds a box of 1 m on each side against,
/// under a ceiling of the given form. The hook is on the box, thus every manifold of the ceiling
/// carries the request.
fn max_held_load(ceiling: Ceiling, request: Request) -> Real {
    let holds = |load: Real| -> bool {
        let mut world = new_world();
        let (box_collider, box_body) = ceiling_and_box(&mut world, ceiling);
        world.bodies[box_body].add_force(xy(0.0, -load), true);
        let hook = AdhesionHook::new(box_collider, request);
        run(&mut world, &hook, 150);
        world.bodies[box_body].translation().y > -0.6
    };

    let (mut held, mut lost) = (0.0, 100.0);
    assert!(holds(held), "{ceiling:?} {request:?} holds no load at all");
    assert!(
        !holds(lost),
        "{ceiling:?} {request:?} holds a load of {lost} N"
    );
    // Twelve halvings of the bracket give a resolution of about 0.02 N.
    for _ in 0..12 {
        let load = 0.5 * (held + lost);
        if holds(load) {
            held = load;
        } else {
            lost = load;
        }
    }
    0.5 * (held + lost)
}

#[test]
fn pressure_is_composition_invariant() {
    // The face of the box has an extent of 1, thus a pressure of 30 is a pull of 30 N and holds
    // a load of 30 N less the weight of the box. It must hold the same load however many tiles
    // make the ceiling, because the patches of tiles that are side by side divide the face of
    // the box between them.
    let request = Request::Pressure(30.0);
    let one = max_held_load(Ceiling::One, request);
    assert_close(one, 30.0 - G, 0.5, "load held under one collider");
    for half_width in [1.0, 0.5, 0.2] {
        let tiled = max_held_load(Ceiling::Tiles(half_width), request);
        assert_close(tiled, one, 0.5, "load held under tiles");
    }
}

#[test]
fn adhesion_force_is_requested_once_for_each_manifold() {
    // The contrast that gives `adhesion_pressure` and `adhesion_budget` their reason to exist:
    // the same request made on each manifold of a ceiling of two tiles pulls two times.
    let request = Request::Force(30.0);
    let one = max_held_load(Ceiling::One, request);
    let tiles = max_held_load(Ceiling::Tiles(1.0), request);
    assert_close(tiles, 2.0 * 30.0 - G, 1.0, "load held under two tiles");
    assert!(tiles > one + 20.0, "the two tiles should pull much more");
}

#[test]
fn budget_is_composition_and_overlap_invariant() {
    // A budget holds the same load under one collider, under tiles, and under tiles that overlap.
    let request = Request::Budget(30.0);
    let one = max_held_load(Ceiling::One, request);
    assert_close(one, 30.0 - G, 0.5, "load held under one collider");
    for ceiling in [
        Ceiling::Tiles(1.0),
        Ceiling::Tiles(0.5),
        Ceiling::Tiles(0.2),
        Ceiling::OverlappingTiles,
    ] {
        let tiled = max_held_load(ceiling, request);
        assert_close(
            tiled,
            one,
            0.5,
            "load held under a ceiling of several colliders",
        );
    }
}

#[test]
fn pressure_and_budget_match_the_equivalent_force_on_one_collider() {
    // On a surface that makes one manifold the three kinds of request describe the same physics,
    // because the face of the box has an extent of 1.
    let force = max_held_load(Ceiling::One, Request::Force(30.0));
    let pressure = max_held_load(Ceiling::One, Request::Pressure(30.0));
    let budget = max_held_load(Ceiling::One, Request::Budget(30.0));
    assert_close(pressure, force, 1.0e-3, "load held under a pressure");
    assert_close(budget, force, 1.0e-3, "load held under a budget");
}

#[test]
fn the_kinds_of_request_add_up() {
    // A hook can make more than one kind of request on a manifold. The pulls add together.
    struct AllThree(ColliderHandle);
    impl PhysicsHooks for AllThree {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            if context.collider1 == self.0 || context.collider2 == self.0 {
                *context.adhesion_force = 10.0;
                *context.adhesion_pressure = 10.0;
                *context.adhesion_budget = Some(AdhesionBudget {
                    owner: self.0,
                    channel: 0,
                    total: 10.0,
                });
            }
        }
    }

    let mut world = new_world();
    let (box_collider, box_body) = ceiling_and_box(&mut world, Ceiling::One);
    // A pull of 30 N holds this load; each kind of request alone, worth 10 N, does not.
    world.bodies[box_body].add_force(xy(0.0, -(20.0 - G)), true);
    run(&mut world, &AllThree(box_collider), 200);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y > -0.6,
        "the three requests should add up to 30 N (y = {y})"
    );
}

#[test]
fn pressure_holds_box_below_ceiling() {
    // The top face of the box has an extent of 1, thus the pressure is also the force.
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    let hook = AdhesionHook::new(ceiling, Request::Pressure(30.0));
    run(&mut world, &hook, 300);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y > -0.6,
        "the box fell to y = {y} although the pull is large"
    );
}

#[test]
fn pressure_scales_with_the_size_of_the_patch() {
    // The patch of a box 2 m wide is two times the patch of a box 1 m wide, and so is its
    // weight. The same pressure must therefore hold both, and a pressure below the weight for
    // each unit of patch must hold neither.
    let hangs = |pressure: Real, half_width: Real| -> bool {
        let mut world = new_world();
        let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
        let ceiling = world.insert_collider(
            cuboid(5.0, 0.5)
                .translation(xy(0.0, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling_body),
        );
        let (box_body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(xy(0.0, -0.5)),
            cuboid(half_width, 0.5),
        );

        let hook = AdhesionHook::new(ceiling, Request::Pressure(pressure));
        run(&mut world, &hook, 200);
        world.bodies[box_body].translation().y > -0.6
    };

    // A box of 1 unit of patch weighs G, and a box of 2 units of patch weighs 2 G.
    assert!(
        hangs(2.0 * G, 0.5),
        "two G for each unit holds the small box"
    );
    assert!(hangs(2.0 * G, 1.0), "and it holds the wide box as well");
    assert!(!hangs(0.5 * G, 0.5), "half a G for each unit is too weak");
    assert!(
        !hangs(0.5 * G, 1.0),
        "and it is too weak for the wide box too"
    );
}

#[test]
fn pressure_on_a_point_contact_is_inert() {
    // A ball that touches the ceiling makes a manifold of one point: it has no extent, thus even
    // a very large pressure gives no force and the ball falls as it does with no adhesion.
    let fall = |pressure: Real| -> Real {
        let mut world = new_world();
        let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
        let ceiling = world.insert_collider(
            cuboid(5.0, 0.5)
                .translation(xy(0.0, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling_body),
        );
        let (ball_body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(xy(0.0, -0.5)),
            ColliderBuilder::ball(0.5),
        );
        let hook = AdhesionHook::new(ceiling, Request::Pressure(pressure));
        run(&mut world, &hook, 120);
        world.bodies[ball_body].translation().y
    };

    let free = fall(0.0);
    assert!(free < -1.0, "the ball should fall with no adhesion");
    assert_close(fall(1000.0), free, 1.0e-3, "y of the ball under a pressure");
}

#[test]
fn negative_pressure_is_ignored() {
    // A negative request only pulls less, never pushes: the box falls exactly as it does with no
    // adhesion at all.
    let fall = |pressure: Real| -> Real {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let y0 = world.bodies[box_body].translation().y;
        let hook = AdhesionHook::new(ceiling, Request::Pressure(pressure));
        run(&mut world, &hook, 60);
        y0 - world.bodies[box_body].translation().y
    };

    let zero = fall(0.0);
    assert!(zero > 0.5, "the box should fall with no adhesion");
    assert_close(fall(-30.0), zero, 1.0e-3, "fall under a negative pressure");
}

#[test]
fn negative_adhesion_is_ignored() {
    // Like a negative pressure: the box falls exactly as it does with no adhesion at all.
    let fall = |force: Real| -> Real {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let y0 = world.bodies[box_body].translation().y;
        let hook = AdhesionHook::new(ceiling, Request::Force(force));
        run(&mut world, &hook, 60);
        y0 - world.bodies[box_body].translation().y
    };

    let zero = fall(0.0);
    assert!(zero > 0.5, "the box should fall with no adhesion");
    assert_close(fall(-30.0), zero, 1.0e-3, "fall under a negative force");
}

/*
 * `adhesion_budget`: the channels of a pool, and the share a point contact receives.
 */

/// A ceiling made of two tiles, and a box under it whose top face straddles the seam, with a
/// downward `load` on the box. Returns the collider of the box, its body and its weight.
fn seam_ceiling_and_box(
    world: &mut PhysicsWorld,
    load: Real,
) -> (ColliderHandle, RigidBodyHandle, Real) {
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let sides: [Real; 2] = [-1.0, 1.0];
    for side in sides {
        world.insert_collider(
            cuboid(1.0, 0.5)
                .translation(xy(side, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling_body),
        );
    }
    let (box_body, box_collider) = world.insert(
        RigidBodyBuilder::dynamic().translation(xy(0.0, -0.5)),
        cuboid(0.5, 0.5),
    );
    let weight = world.bodies[box_body].mass() * G;
    world.bodies[box_body].add_force(xy(0.0, -load), true);
    (box_collider, box_body, weight)
}

#[test]
fn budget_channels_are_independent_pools() {
    // A box hangs under a ceiling of two tiles, with a load of 5 N and a budget worth its weight
    // plus 3 N. The two manifolds in one pool share that total, which the load exceeds, and the
    // box falls. In two channels they make two pools, which together hold it. This is what a
    // body in a corner needs: it spends the budget of its feet and the budget of its flank at
    // the same time.
    struct PerTile {
        collider: ColliderHandle,
        total: Real,
        one_channel: bool,
    }
    impl PhysicsHooks for PerTile {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            let tile = if context.collider1 == self.collider {
                context.collider2
            } else if context.collider2 == self.collider {
                context.collider1
            } else {
                return;
            };
            let (index, _) = tile.into_raw_parts();
            let channel = if self.one_channel { 0 } else { index };
            Request::Budget(self.total).apply(context, self.collider, channel);
        }
    }

    let hang = |one_channel: bool| -> Real {
        let mut world = new_world();
        let (box_collider, box_body, weight) = seam_ceiling_and_box(&mut world, 5.0);
        let hook = PerTile {
            collider: box_collider,
            total: weight + 3.0,
            one_channel,
        };
        run(&mut world, &hook, 200);
        world.bodies[box_body].translation().y
    };

    let y = hang(true);
    assert!(
        y < -1.0,
        "one shared pool should not hold the load (y = {y})"
    );
    let y = hang(false);
    assert!(y > -0.6, "two channels should hold the load (y = {y})");
}

#[test]
fn budget_gives_a_point_contact_the_full_total() {
    // A ball under the ceiling makes a manifold of one point. A pressure is inert there, but a
    // pool that has this manifold only gives it the full total, thus the ball hangs.
    let mut world = new_world();
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    world.insert_collider(
        cuboid(5.0, 0.5)
            .translation(xy(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );
    let (ball_body, ball) = world.insert(
        RigidBodyBuilder::dynamic().translation(xy(0.0, -0.5)),
        ColliderBuilder::ball(0.5),
    );

    // The ball weighs about 7.7 N in 2D and 5.1 N in 3D.
    let hook = AdhesionHook::new(ball, Request::Budget(30.0));
    run(&mut world, &hook, 300);

    let y = world.bodies[ball_body].translation().y;
    assert!(
        y > -0.6,
        "the ball fell to y = {y} although it has a budget"
    );
}

/*
 * 3D only: a pair whose manifolds are clustered, and a patch that is a line.
 */

/// A fixed ceiling whose underside is the plane y = 0, made of a triangle mesh of 4 by 4 quads,
/// with contact modification enabled, and a box of 1 m on each side that hangs under it with the
/// downward load `load`. A box under a mesh touches several triangles, thus the pair has several
/// manifolds, which contact clustering merges into one cluster.
#[cfg(feature = "dim3")]
fn mesh_ceiling_and_hanging_box(
    world: &mut PhysicsWorld,
    load: Real,
) -> (ColliderHandle, RigidBodyHandle, ColliderHandle) {
    const N: u32 = 4;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=N {
        for j in 0..=N {
            vertices.push(Vector::new(
                i as Real - N as Real / 2.0,
                0.0,
                j as Real - N as Real / 2.0,
            ));
        }
    }
    for i in 0..N {
        for j in 0..N {
            let a = i * (N + 1) + j;
            let (b, c) = (a + 1, a + N + 1);
            // Wound so that the normals of the faces point down, toward the box.
            indices.push([a, c, b]);
            indices.push([b, c, c + 1]);
        }
    }

    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let ceiling = world.insert_collider(
        ColliderBuilder::trimesh_with_flags(vertices, indices, TriMeshFlags::FIX_INTERNAL_EDGES)
            .unwrap()
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );
    let (box_body, box_collider) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(xy(0.0, -0.5))
            .can_sleep(false),
        cuboid(0.5, 0.5),
    );
    world.bodies[box_body].add_force(xy(0.0, -load), true);
    (ceiling, box_body, box_collider)
}

#[cfg(feature = "dim3")]
#[test]
fn a_clustered_pair_holds_like_one_collider() {
    // The box weighs about 9.81 N. A request worth its weight plus 3 N over its face of 1 m2
    // holds a load of 1 N and breaks under a load of 5 N, whether the ceiling is one cuboid or a
    // mesh whose manifolds are clustered. The adhesion must therefore be applied once for each
    // cluster, and not once for each manifold inside it.
    let hang = |mesh: bool, request: Request, load: Real| -> Real {
        let mut world = new_world();
        let (ceiling, box_body, box_collider) = if mesh {
            mesh_ceiling_and_hanging_box(&mut world, load)
        } else {
            let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
            let ceiling = world.insert_collider(
                cuboid(5.0, 0.5)
                    .translation(xy(0.0, 0.5))
                    .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
                Some(ceiling_body),
            );
            let (box_body, box_collider) = world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(xy(0.0, -0.5))
                    .can_sleep(false),
                cuboid(0.5, 0.5),
            );
            world.bodies[box_body].add_force(xy(0.0, -load), true);
            (ceiling, box_body, box_collider)
        };

        let hook = AdhesionHook::new(ceiling, request);
        run(&mut world, &hook, 1);
        if mesh {
            let pair = world
                .narrow_phase
                .contact_pair(ceiling, box_collider)
                .expect("the ceiling and the box must have a contact pair");
            assert!(
                pair.manifolds.len() > 1 && pair.solver_clusters.len() == 1,
                "the mesh ceiling should give one cluster over several manifolds \
                 ({} manifolds, {} clusters)",
                pair.manifolds.len(),
                pair.solver_clusters.len()
            );
        }
        run(&mut world, &hook, 239);
        world.bodies[box_body].translation().y
    };

    for request in [Request::Pressure(G + 3.0), Request::Budget(G + 3.0)] {
        for mesh in [false, true] {
            let y = hang(mesh, request, 1.0);
            assert!(
                y > -0.6,
                "mesh {mesh}, {request:?}: should hold 1 N (y = {y})"
            );
            let y = hang(mesh, request, 5.0);
            assert!(y < -1.0, "mesh {mesh}, {request:?}: should break (y = {y})");
        }
    }
}

#[cfg(feature = "dim3")]
#[test]
fn pressure_on_a_line_contact_is_inert() {
    // A capsule that lies against the ceiling touches it along a line, which has no area. A
    // pressure therefore gives no force, but a budget still holds the capsule.
    let hangs = |request: Request| -> bool {
        let mut world = new_world();
        let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
        let ceiling = world.insert_collider(
            ColliderBuilder::cuboid(5.0, 0.5, 5.0)
                .translation(xy(0.0, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling_body),
        );
        let (capsule_body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(xy(0.0, -0.25)),
            ColliderBuilder::capsule_x(1.0, 0.25),
        );
        let hook = AdhesionHook::new(ceiling, request);
        run(&mut world, &hook, 120);
        world.bodies[capsule_body].translation().y > -0.5
    };

    assert!(!hangs(Request::Pressure(1000.0)), "a line has no area");
    assert!(hangs(Request::Budget(30.0)), "a budget should hold it");
}
