//! Substep contact tracking: a collider can choose how the solver moves its side of a contact through the substeps. With `CurvatureCenter` (balls) or `FirstOrder` (any shape), a ball's spin never changes how its contacts push. Every ball here is frictionless, so its spin and its translation can only interact through the separation.
#![cfg(feature = "substep-contact-tracking")]

mod boxes;
mod identity;
mod world;

use rapier2d::prelude::*;
use test_case::test_case;
use world::{BallBody, RADIUS, Surface, World};

/// Fast enough to turn a body-fixed anchor 0.59 rad per substep, as the game's fastest wheel.
const FAST_SPIN: Real = 142.0;

const CENTER: SubstepContactTracking = SubstepContactTracking::CurvatureCenter;
const FIRST_ORDER: SubstepContactTracking = SubstepContactTracking::FirstOrder;
const BODY_FIXED: SubstepContactTracking = SubstepContactTracking::BodyFixed;

/// The ball's centre over `steps` steps, starting 1 mm into `surface`'s first plane, at rest.
fn resting_trajectory(
    body: BallBody,
    surface: Surface,
    tracking: SubstepContactTracking,
    spin: Real,
    steps: usize,
) -> Vec<Vector> {
    let (point, normal) = surface.planes()[0];
    let center = point + normal * (RADIUS - 0.001);
    let center = match surface {
        Surface::VGroove => Vector::new(0.0, (RADIUS - 0.001) * core::f32::consts::SQRT_2),
        _ => center,
    };
    World::new(body, surface, tracking, center, Vector::ZERO, spin).trajectory(steps)
}

fn max_distance(a: &[Vector], b: &[Vector]) -> Real {
    a.iter()
        .zip(b)
        .map(|(p, q)| (*p - *q).length())
        .fold(0.0, Real::max)
}

/// With either tracking the spin is invisible to the contact: the spinning ball follows the non-spinning one's path on every surface, including two manifolds at once in the groove.
#[test_case(BallBody::Rigid, Surface::Flat, CENTER ; "centre, rigid ball on a flat floor")]
#[test_case(BallBody::Rigid, Surface::Slope, CENTER ; "centre, rigid ball sliding down a slope")]
#[test_case(BallBody::Rigid, Surface::VGroove, CENTER ; "centre, rigid ball in a V groove")]
#[test_case(BallBody::MultibodyWheel, Surface::Flat, CENTER ; "centre, multibody wheel on a flat floor")]
#[test_case(BallBody::MultibodyWheel, Surface::Slope, CENTER ; "centre, multibody wheel sliding down a slope")]
#[test_case(BallBody::MultibodyWheel, Surface::VGroove, CENTER ; "centre, multibody wheel in a V groove")]
#[test_case(BallBody::Rigid, Surface::Flat, FIRST_ORDER ; "first order, rigid ball on a flat floor")]
#[test_case(BallBody::Rigid, Surface::Slope, FIRST_ORDER ; "first order, rigid ball sliding down a slope")]
#[test_case(BallBody::Rigid, Surface::VGroove, FIRST_ORDER ; "first order, rigid ball in a V groove")]
#[test_case(BallBody::MultibodyWheel, Surface::Flat, FIRST_ORDER ; "first order, multibody wheel on a flat floor")]
#[test_case(BallBody::MultibodyWheel, Surface::Slope, FIRST_ORDER ; "first order, multibody wheel sliding down a slope")]
#[test_case(BallBody::MultibodyWheel, Surface::VGroove, FIRST_ORDER ; "first order, multibody wheel in a V groove")]
fn tracking_makes_the_spin_invisible_to_the_contact(
    body: BallBody,
    surface: Surface,
    tracking: SubstepContactTracking,
) {
    let still = resting_trajectory(body, surface, tracking, 0.0, 120);
    let spinning = resting_trajectory(body, surface, tracking, FAST_SPIN, 120);
    let deviation = max_distance(&still, &spinning);
    assert!(
        deviation < 1.0e-4,
        "the spinning ball strays {deviation} m from the still one"
    );
}

/// The artifact the tracking removes: with body-fixed anchors the same spin moves the ball off the still ball's path by millimetres. Without this the test above could pass with the option doing nothing.
#[test_case(BallBody::Rigid ; "rigid ball")]
#[test_case(BallBody::MultibodyWheel ; "multibody wheel")]
fn body_fixed_anchors_let_the_spin_move_the_ball(body: BallBody) {
    let still = resting_trajectory(body, Surface::Flat, BODY_FIXED, 0.0, 120);
    let spinning = resting_trajectory(body, Surface::Flat, BODY_FIXED, FAST_SPIN, 120);
    let deviation = max_distance(&still, &spinning);
    assert!(
        deviation > 1.0e-3,
        "the spinning ball strays only {deviation} m"
    );
}

/// The gap left after one step by a ball falling at 3 m/s from 1 cm above the floor, with no gravity: fast enough to close the gap within the step, so the speculative contact stops it where the solver's separation says the floor is.
fn falling_gap(body: BallBody, tracking: SubstepContactTracking, spin: Real) -> Real {
    let center = Vector::new(0.0, RADIUS + 0.01);
    let velocity = Vector::new(0.0, -3.0);
    let mut world = World::new(body, Surface::Flat, tracking, center, velocity, spin);
    world.gravity = Vector::ZERO;
    world.step();
    world.gap(Surface::Flat)
}

/// The solver's separation in every substep is the true gap: a spinning ball falling onto the floor stops where the still ball stops, on the surface.
#[test_case(BallBody::Rigid, CENTER ; "centre, rigid ball")]
#[test_case(BallBody::MultibodyWheel, CENTER ; "centre, multibody wheel")]
#[test_case(BallBody::Rigid, FIRST_ORDER ; "first order, rigid ball")]
#[test_case(BallBody::MultibodyWheel, FIRST_ORDER ; "first order, multibody wheel")]
fn a_falling_tracked_ball_stops_where_a_still_one_does(
    body: BallBody,
    tracking: SubstepContactTracking,
) {
    let still = falling_gap(body, tracking, 0.0);
    let spinning = falling_gap(body, tracking, 100.0);
    assert!(
        still.abs() < 1.0e-3,
        "the still ball ends {still} m from the floor"
    );
    assert!(
        (spinning - still).abs() < 1.0e-5,
        "the spinning ball ends {spinning} m from the floor, the still one {still} m"
    );
}

/// A ball off its body's centre of mass, resting on the floor while its body spins clockwise: the spin swings the ball's centre down into the floor within the step. Both trackings follow the centre's swing (the curvature centre exactly, first order to first order), so the floor pushes in every substep and the ball ends out of the floor; body-fixed rides the contact round the rim, reads the ball as leaving, and lets it sink. The two tracked results are printed; at this spin they agree.
#[test]
fn an_off_centre_spinning_ball_stays_out_of_the_floor_when_tracked() {
    // Clockwise, so the ball's centre swings down into the floor it starts on.
    let spin = -40.0;
    let gap = |tracking| {
        let center = Vector::new(0.0, RADIUS);
        let mut world = World::new(
            BallBody::OffCentre,
            Surface::Flat,
            tracking,
            center,
            Vector::ZERO,
            spin,
        );
        world.gravity = Vector::ZERO;
        world.step();
        world.gap(Surface::Flat)
    };
    let (center, first_order, body_fixed) = (gap(CENTER), gap(FIRST_ORDER), gap(BODY_FIXED));
    eprintln!(
        "off-centre ball spinning at {spin} rad/s ends: curvature centre {center} m, first order {first_order} m, body-fixed {body_fixed} m from the floor"
    );
    assert!(
        center > -1.0e-3,
        "the curvature-centre ball ends {center} m from the floor"
    );
    assert!(
        first_order > -1.0e-3,
        "the first-order ball ends {first_order} m from the floor"
    );
    assert!(
        body_fixed < -1.0e-3,
        "the body-fixed ball ends only {body_fixed} m from the floor"
    );
}

/// With body-fixed anchors the spin lifts the anchor, the solver sees a larger gap in the later substeps, and the spinning ball ends deeper in the floor than the still one.
#[test_case(BallBody::Rigid ; "rigid ball")]
#[test_case(BallBody::MultibodyWheel ; "multibody wheel")]
fn a_falling_body_fixed_ball_overshoots_into_the_floor(body: BallBody) {
    let still = falling_gap(body, BODY_FIXED, 0.0);
    let spinning = falling_gap(body, BODY_FIXED, 100.0);
    assert!(
        spinning < still - 1.0e-3,
        "the spinning ball ends {spinning} m from the floor, the still one {still} m"
    );
}

/// Two runs of one tracked scene agree bit for bit.
#[test_case(CENTER ; "curvature centre")]
#[test_case(FIRST_ORDER ; "first order")]
fn tracked_runs_repeat_exactly(tracking: SubstepContactTracking) {
    let run = || {
        resting_trajectory(
            BallBody::MultibodyWheel,
            Surface::VGroove,
            tracking,
            FAST_SPIN,
            60,
        )
    };
    let bits = |path: Vec<Vector>| -> Vec<[u32; 2]> {
        path.iter()
            .map(|p| [p.x.to_bits(), p.y.to_bits()])
            .collect()
    };
    assert_eq!(bits(run()), bits(run()));
}

#[test_case(ColliderBuilder::cuboid(0.3, 0.3) ; "cuboid")]
#[test_case(ColliderBuilder::capsule_y(0.3, 0.3) ; "capsule")]
#[test_case(ColliderBuilder::compound(vec![(Pose::IDENTITY, SharedShape::ball(0.3))]) ; "compound of one ball")]
fn the_curvature_centre_is_refused_on_other_shapes(builder: ColliderBuilder) {
    let shape = builder.shape.shape_type();
    assert_eq!(
        builder.clone().substep_contact_tracking(CENTER).err(),
        Some(CurvatureCenterNeedsBall { shape })
    );
    let mut collider = builder.build();
    assert!(collider.set_substep_contact_tracking(CENTER).is_err());
    assert_eq!(collider.substep_contact_tracking(), BODY_FIXED);
    assert!(collider.set_substep_contact_tracking(FIRST_ORDER).is_ok());
    assert_eq!(collider.substep_contact_tracking(), FIRST_ORDER);
}

/// The tracking shares the collider's hook bits without showing in them, and survives a change of hooks.
#[test_case(CENTER ; "curvature centre")]
#[test_case(FIRST_ORDER ; "first order")]
fn the_tracking_is_independent_of_the_hooks(tracking: SubstepContactTracking) {
    let mut collider = ColliderBuilder::ball(0.3)
        .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS)
        .substep_contact_tracking(tracking)
        .unwrap()
        .build();
    assert_eq!(collider.active_hooks(), ActiveHooks::MODIFY_SOLVER_CONTACTS);
    collider.set_active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS);
    assert_eq!(collider.substep_contact_tracking(), tracking);
    assert_eq!(collider.active_hooks(), ActiveHooks::FILTER_CONTACT_PAIRS);
    collider.set_substep_contact_tracking(BODY_FIXED).unwrap();
    assert_eq!(collider.active_hooks(), ActiveHooks::FILTER_CONTACT_PAIRS);
}
