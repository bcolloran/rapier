//! First-order tracking on shapes with corners: a box on a ramp and a tumbling box, where the first-order update drops the second-order part of the body's rotation that body-fixed tracking keeps.

use rapier2d::prelude::*;
use test_case::test_case;

const HALF_EXTENT: Real = 0.25;

/// A world with a fixed floor tilted by `ramp_angle` (radians, rising to the right) and one box of [`HALF_EXTENT`] tracked as `tracking`, at `position` turned by `angle`, moving at `linvel` and `angvel`.
fn box_world(
    ramp_angle: Real,
    tracking: SubstepContactTracking,
    position: Vector,
    angle: Real,
    linvel: Vector,
    angvel: Real,
) -> (PhysicsWorld, RigidBodyHandle) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81);
    let normal = Vector::new(-ramp_angle.sin(), ramp_angle.cos());
    world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(20.0, 1.0)
            .translation(-normal)
            .rotation(ramp_angle)
            .friction(1.0),
    );
    let (body, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(position)
            .rotation(angle)
            .linvel(linvel)
            .angvel(angvel)
            .can_sleep(false),
        ColliderBuilder::cuboid(HALF_EXTENT, HALF_EXTENT)
            .friction(1.0)
            .substep_contact_tracking(tracking)
            .expect("any shape takes first-order tracking"),
    );
    (world, body)
}

/// The deepest a box corner sits below a floor through the origin with normal `normal`, m; positive when it overlaps.
fn depth(world: &PhysicsWorld, body: RigidBodyHandle, normal: Vector) -> Real {
    let pose = world.bodies[body].position();
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .iter()
        .map(|&(x, y)| {
            -pose
                .transform_point(Vector::new(x, y) * HALF_EXTENT)
                .dot(normal)
        })
        .fold(Real::NEG_INFINITY, Real::max)
}

/// The box's pose after `steps` steps, and the deepest overlap it reached.
fn run(
    world: &mut PhysicsWorld,
    body: RigidBodyHandle,
    normal: Vector,
    steps: usize,
) -> (Pose, Real) {
    let mut deepest: Real = 0.0;
    for _ in 0..steps {
        world.step();
        deepest = deepest.max(depth(world, body, normal));
    }
    (*world.bodies[body].position(), deepest)
}

/// A box resting on a ramp it grips does not turn, so first-order tracking reads the same separations as body-fixed and the box stays where body-fixed leaves it.
#[test_case(0.0 ; "flat floor")]
#[test_case(20.0_f32.to_radians() ; "20 degree ramp")]
fn a_box_resting_on_a_ramp_tracks_the_same_to_first_order(ramp_angle: Real) {
    let normal = Vector::new(-ramp_angle.sin(), ramp_angle.cos());
    let start = normal * HALF_EXTENT;
    let path = |tracking| {
        let (mut world, body) =
            box_world(ramp_angle, tracking, start, ramp_angle, Vector::ZERO, 0.0);
        run(&mut world, body, normal, 120)
    };
    let (body_fixed, _) = path(SubstepContactTracking::BodyFixed);
    let (first_order, deepest) = path(SubstepContactTracking::FirstOrder);
    let moved = (first_order.translation - body_fixed.translation).length();
    assert!(moved < 1.0e-4, "first order ends {moved} m from body-fixed");
    assert!(deepest < 5.0e-3, "the box sinks {deepest} m into the ramp");
}

/// A box thrown spinning onto the floor tumbles over its corners. First order drops the second-order part of each substep's rotation, the part that lifts or lowers a corner as the box turns, and the box still settles on a face, no deeper than body-fixed lets it.
#[test_case(5.0 ; "5 rad/s")]
#[test_case(20.0 ; "20 rad/s")]
fn a_tumbling_box_settles_on_the_floor_to_first_order(angvel: Real) {
    let start = Vector::new(0.0, 0.6);
    let linvel = Vector::new(2.0, -1.0);
    let settle = |tracking| {
        let (mut world, body) = box_world(0.0, tracking, start, 0.3, linvel, angvel);
        let (pose, deepest) = run(&mut world, body, Vector::Y, 300);
        let rb = &world.bodies[body];
        (pose, deepest, rb.linvel().length() + rb.angvel().abs())
    };
    let (_, body_fixed_deepest, _) = settle(SubstepContactTracking::BodyFixed);
    let (pose, deepest, speed) = settle(SubstepContactTracking::FirstOrder);
    eprintln!(
        "tumbling box at {angvel} rad/s: deepest overlap first order {deepest} m, body-fixed {body_fixed_deepest} m"
    );
    let resting_height = pose.translation.y;
    assert!(speed < 1.0e-2, "the box still moves at {speed}");
    assert!(
        (resting_height - HALF_EXTENT).abs() < 5.0e-3,
        "the box rests with its centre {resting_height} m up, not on a face"
    );
    assert!(
        deepest < body_fixed_deepest + 5.0e-3,
        "first order sinks {deepest} m, body-fixed {body_fixed_deepest} m"
    );
}
