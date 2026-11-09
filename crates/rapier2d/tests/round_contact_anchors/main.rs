//! Round contact anchors: a ball whose collider uses `ContactAnchor::Round` has its solver separation measured from a point that keeps its detected offset from the ball's centre, so the ball's spin never changes how its contacts push. Every ball here is frictionless, so its spin and its translation can only interact through that separation.

mod world;

use rapier2d::prelude::*;
use test_case::test_case;
use world::{BallBody, RADIUS, Surface, World};

/// Fast enough to turn the anchor 0.59 rad per substep, as the game's fastest wheel.
const FAST_SPIN: Real = 142.0;

/// The ball's centre over `steps` steps, starting 1 mm into `surface`'s first plane, at rest.
fn resting_trajectory(
    body: BallBody,
    surface: Surface,
    anchor: ContactAnchor,
    spin: Real,
    steps: usize,
) -> Vec<Vector> {
    let (point, normal) = surface.planes()[0];
    let center = point + normal * (RADIUS - 0.001);
    let center = match surface {
        Surface::VGroove => Vector::new(0.0, (RADIUS - 0.001) * core::f32::consts::SQRT_2),
        _ => center,
    };
    World::new(body, surface, anchor, center, Vector::ZERO, spin).trajectory(steps)
}

fn max_distance(a: &[Vector], b: &[Vector]) -> Real {
    a.iter()
        .zip(b)
        .map(|(p, q)| (*p - *q).length())
        .fold(0.0, Real::max)
}

/// With round anchors the spin is invisible to the contact: the spinning ball follows the non-spinning one's path on every surface, including two manifolds at once in the groove.
#[test_case(BallBody::Rigid, Surface::Flat ; "rigid ball on a flat floor")]
#[test_case(BallBody::Rigid, Surface::Slope ; "rigid ball sliding down a slope")]
#[test_case(BallBody::Rigid, Surface::VGroove ; "rigid ball in a V groove")]
#[test_case(BallBody::MultibodyWheel, Surface::Flat ; "multibody wheel on a flat floor")]
#[test_case(BallBody::MultibodyWheel, Surface::Slope ; "multibody wheel sliding down a slope")]
#[test_case(BallBody::MultibodyWheel, Surface::VGroove ; "multibody wheel in a V groove")]
fn round_anchors_make_the_spin_invisible_to_the_contact(body: BallBody, surface: Surface) {
    let still = resting_trajectory(body, surface, ContactAnchor::Round, 0.0, 120);
    let spinning = resting_trajectory(body, surface, ContactAnchor::Round, FAST_SPIN, 120);
    let deviation = max_distance(&still, &spinning);
    assert!(
        deviation < 1.0e-4,
        "the spinning ball strays {deviation} m from the still one"
    );
}

/// The artifact the round anchor removes: with body-fixed anchors the same spin moves the ball off the still ball's path by millimetres. Without this the test above could pass with the option doing nothing.
#[test_case(BallBody::Rigid ; "rigid ball")]
#[test_case(BallBody::MultibodyWheel ; "multibody wheel")]
fn body_fixed_anchors_let_the_spin_move_the_ball(body: BallBody) {
    let anchor = ContactAnchor::BodyFixed;
    let still = resting_trajectory(body, Surface::Flat, anchor, 0.0, 120);
    let spinning = resting_trajectory(body, Surface::Flat, anchor, FAST_SPIN, 120);
    let deviation = max_distance(&still, &spinning);
    assert!(
        deviation > 1.0e-3,
        "the spinning ball strays only {deviation} m"
    );
}

/// The gap left after one step by a ball falling at 3 m/s from 1 cm above the floor, with no gravity: fast enough to close the gap within the step, so the speculative contact stops it where the solver's separation says the floor is.
fn falling_gap(body: BallBody, anchor: ContactAnchor, spin: Real) -> Real {
    let center = Vector::new(0.0, RADIUS + 0.01);
    let velocity = Vector::new(0.0, -3.0);
    let mut world = World::new(body, Surface::Flat, anchor, center, velocity, spin);
    world.gravity = Vector::ZERO;
    world.step();
    world.gap(Surface::Flat)
}

/// The solver's separation in every substep is the true gap: a spinning ball falling onto the floor stops where the still ball stops, on the surface.
#[test_case(BallBody::Rigid ; "rigid ball")]
#[test_case(BallBody::MultibodyWheel ; "multibody wheel")]
fn a_falling_round_anchor_ball_stops_where_a_still_one_does(body: BallBody) {
    let still = falling_gap(body, ContactAnchor::Round, 0.0);
    let spinning = falling_gap(body, ContactAnchor::Round, 100.0);
    assert!(
        still.abs() < 1.0e-3,
        "the still ball ends {still} m from the floor"
    );
    assert!(
        (spinning - still).abs() < 1.0e-5,
        "the spinning ball ends {spinning} m from the floor, the still one {still} m"
    );
}

/// With body-fixed anchors the spin lifts the anchor, the solver sees a larger gap in the later substeps, and the spinning ball ends deeper in the floor than the still one.
#[test_case(BallBody::Rigid ; "rigid ball")]
#[test_case(BallBody::MultibodyWheel ; "multibody wheel")]
fn a_falling_body_fixed_ball_overshoots_into_the_floor(body: BallBody) {
    let still = falling_gap(body, ContactAnchor::BodyFixed, 0.0);
    let spinning = falling_gap(body, ContactAnchor::BodyFixed, 100.0);
    assert!(
        spinning < still - 1.0e-3,
        "the spinning ball ends {spinning} m from the floor, the still one {still} m"
    );
}

/// Two runs of one round-anchor scene agree bit for bit.
#[test]
fn round_anchor_runs_repeat_exactly() {
    let run = || {
        resting_trajectory(
            BallBody::MultibodyWheel,
            Surface::VGroove,
            ContactAnchor::Round,
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
fn round_anchors_are_refused_on_other_shapes(builder: ColliderBuilder) {
    let shape = builder.shape.shape_type();
    assert_eq!(
        builder.clone().contact_anchor(ContactAnchor::Round).err(),
        Some(RoundAnchorNeedsBall { shape })
    );
    let mut collider = builder.build();
    assert!(collider.set_contact_anchor(ContactAnchor::Round).is_err());
    assert_eq!(collider.contact_anchor(), ContactAnchor::BodyFixed);
}

#[test]
fn a_new_shape_that_is_not_a_ball_drops_the_round_anchor() {
    let mut collider = ColliderBuilder::ball(0.3)
        .contact_anchor(ContactAnchor::Round)
        .unwrap()
        .build();
    collider.set_shape(SharedShape::ball(0.5));
    assert_eq!(collider.contact_anchor(), ContactAnchor::Round);
    collider.set_shape(SharedShape::cuboid(0.3, 0.3));
    assert_eq!(collider.contact_anchor(), ContactAnchor::BodyFixed);
}
