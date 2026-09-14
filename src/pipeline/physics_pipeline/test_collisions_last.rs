//! Tests of the collisions-last stepping mode: [`PhysicsPipeline::step_collisions_last`] and
//! [`PhysicsPipeline::initialize_collisions_last`].
//!
//! The `step_collisions_last_*` and `manual_init_*` tests are ported from the fork's 0.32
//! implementation; the others cover what changed with the 0.35 pipeline (early narrow-phase
//! user changes, catch-up detection, CCD substeps, the deferred BVH optimization, snapshots).

use crate::alloc_prelude::*;
use crate::math::{Real, Rotation, Vector};
use crate::prelude::*;

#[cfg(feature = "dim2")]
fn v(x: Real, y: Real) -> Vector {
    Vector::new(x, y)
}

#[cfg(feature = "dim3")]
fn v(x: Real, y: Real) -> Vector {
    Vector::new(x, y, 0.0)
}

/// A box `2 hx` wide and `2 hy` tall (and `2 hx` deep in 3D).
#[cfg(feature = "dim2")]
fn cuboid(hx: Real, hy: Real) -> ColliderBuilder {
    ColliderBuilder::cuboid(hx, hy)
}

#[cfg(feature = "dim3")]
fn cuboid(hx: Real, hy: Real) -> ColliderBuilder {
    ColliderBuilder::cuboid(hx, hy, hx)
}

#[cfg(feature = "dim2")]
fn revolute() -> RevoluteJointBuilder {
    RevoluteJointBuilder::new()
}

#[cfg(feature = "dim3")]
fn revolute() -> RevoluteJointBuilder {
    RevoluteJointBuilder::new(Vector::Z)
}

#[cfg(feature = "dim2")]
fn rotation(angle: Real) -> Rotation {
    Rotation::from_angle(angle)
}

#[cfg(feature = "dim3")]
fn rotation(angle: Real) -> Rotation {
    Rotation::from_axis_angle(Vector::Z, angle)
}

/// A ball of radius 0.5 dropped from `y = 5` onto a fixed floor whose top is at `y = 0.1`.
/// Returns the world, the ball, the ball's collider and the floor's collider.
fn ball_on_floor(
    restitution: Real,
) -> (
    PhysicsWorld,
    RigidBodyHandle,
    ColliderHandle,
    ColliderHandle,
) {
    let mut world = PhysicsWorld::new();
    let (_, floor_co) = world.insert(RigidBodyBuilder::fixed(), cuboid(10.0, 0.1));
    let (ball, ball_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 5.0)),
        ColliderBuilder::ball(0.5).restitution(restitution),
    );
    (world, ball, ball_co, floor_co)
}

fn assert_body_finite(world: &PhysicsWorld, handle: RigidBodyHandle) {
    let rb = &world.bodies[handle];
    assert!(
        rb.position().is_finite() && rb.vels().is_finite(),
        "body {handle:?} has non-finite state"
    );
}

fn assert_all_bodies_finite(world: &PhysicsWorld) {
    for (handle, _) in world.rigid_bodies() {
        assert_body_finite(world, handle);
    }
}

/// Every contact and intersection pair only references live colliders.
fn assert_no_stale_pairs(world: &PhysicsWorld) {
    for pair in world.contact_pairs() {
        assert!(
            world.colliders.contains(pair.collider1) && world.colliders.contains(pair.collider2),
            "contact pair references a removed collider"
        );
    }
    for (h1, h2, _) in world.narrow_phase.intersection_pairs() {
        assert!(
            world.colliders.contains(h1) && world.colliders.contains(h2),
            "intersection pair references a removed collider"
        );
    }
}

/// The touching contact pairs and intersecting sensor pairs, as sorted collider index pairs.
fn pair_state(world: &PhysicsWorld) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let key = |a: ColliderHandle, b: ColliderHandle| {
        let (a, b) = (a.into_raw_parts().0, b.into_raw_parts().0);
        (a.min(b), a.max(b))
    };
    let mut touching: Vec<_> = world
        .contact_pairs()
        .filter(|pair| pair.has_any_active_contact())
        .map(|pair| key(pair.collider1, pair.collider2))
        .collect();
    touching.sort_unstable();
    let mut intersecting: Vec<_> = world
        .narrow_phase
        .intersection_pairs()
        .filter(|(_, _, intersecting)| *intersecting)
        .map(|(h1, h2, _)| key(h1, h2))
        .collect();
    intersecting.sort_unstable();
    (touching, intersecting)
}

// =========================================================================
// Ported from 0.32: automatic initialization on the first step.
// =========================================================================

/// Same as [`step_collisions_last_basic`], but through the raw pipeline API.
#[allow(clippy::too_many_arguments)]
fn run_step_collisions_last(
    pipeline: &mut PhysicsPipeline,
    gravity: Vector,
    integration_parameters: &IntegrationParameters,
    islands: &mut IslandManager,
    broad_phase: &mut BroadPhaseBvh,
    narrow_phase: &mut NarrowPhase,
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    impulse_joints: &mut ImpulseJointSet,
    multibody_joints: &mut MultibodyJointSet,
    ccd: &mut CCDSolver,
    n: usize,
) {
    for _ in 0..n {
        pipeline.step_collisions_last(
            gravity,
            integration_parameters,
            islands,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            ccd,
            &(),
            &(),
        );
    }
}

#[test]
fn step_collisions_last_basic() {
    let mut pipeline = PhysicsPipeline::new();
    let gravity = Vector::Y * -9.81;
    let integration_parameters = IntegrationParameters::default();
    let mut broad_phase = BroadPhaseBvh::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut ccd = CCDSolver::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut islands = IslandManager::new();

    // A dynamic ball above a fixed floor.
    let floor = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(cuboid(10.0, 0.1), floor, &mut bodies);
    let ball = bodies.insert(RigidBodyBuilder::dynamic().translation(v(0.0, 5.0)));
    colliders.insert_with_parent(
        ColliderBuilder::ball(0.5).restitution(0.7),
        ball,
        &mut bodies,
    );

    assert!(!pipeline.collisions_last_initialized());
    run_step_collisions_last(
        &mut pipeline,
        gravity,
        &integration_parameters,
        &mut islands,
        &mut broad_phase,
        &mut narrow_phase,
        &mut bodies,
        &mut colliders,
        &mut impulse_joints,
        &mut multibody_joints,
        &mut ccd,
        100,
    );
    assert!(pipeline.collisions_last_initialized());

    // The ball has fallen, and not through the floor.
    let y = bodies[ball].translation().y;
    assert!(y < 5.0, "the ball should have fallen from y=5, but y={y}");
    assert!(
        y > -1.0,
        "the ball should not fall through the floor, but y={y}"
    );
}

/// Steps a resting-ball scene with stock `step` and with collisions-last (optionally
/// initialized by hand) and checks that the ball settles at the same height.
fn check_matches_step(manual_init: bool) {
    let (mut stock, stock_ball, _, _) = ball_on_floor(0.7);
    let (mut last, last_ball, _, _) = ball_on_floor(0.7);
    if manual_init {
        last.initialize_collisions_last_with_events(&(), &());
    }

    // Long enough for the bouncing to die out and the ball to come to rest.
    for _ in 0..600 {
        stock.step();
        last.step_collisions_last();
    }

    let stock_y = stock.bodies[stock_ball].translation().y;
    let last_y = last.bodies[last_ball].translation().y;
    // Floor top (0.1) + ball radius (0.5).
    let rest_y = 0.6;
    assert!(
        (stock_y - rest_y).abs() < 0.01,
        "stock step did not settle on the floor: y={stock_y}"
    );
    assert!(
        (stock_y - last_y).abs() < 1.0e-3,
        "step() and step_collisions_last() settled at different heights: \
         step={stock_y}, step_collisions_last={last_y}"
    );
}

#[test]
fn step_collisions_last_matches_step() {
    check_matches_step(false);
}

/// Adds a body and a joint between collisions-last steps. In 0.32 this panicked with an
/// out-of-bounds index in `select_active_interactions`, because the new body was not
/// processed by the user-changes stage before the solve.
fn check_joint_added_between_steps(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    let fixed_body = world.insert_body(RigidBodyBuilder::fixed());
    world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 5.0)),
        ColliderBuilder::ball(0.5),
    );
    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
    }

    for _ in 0..10 {
        world.step_collisions_last();
    }

    let (new_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(2.0, 5.0)),
        ColliderBuilder::ball(0.5),
    );
    world.insert_impulse_joint(fixed_body, new_body, revolute());

    for _ in 0..10 {
        world.step_collisions_last();
    }

    assert_all_bodies_finite(&world);
}

#[test]
fn step_collisions_last_joint_added_between_steps() {
    check_joint_added_between_steps(false);
}

/// Removes a body resting on the floor between collisions-last steps. In 0.32 this panicked
/// ("No element at index") because the island update walked the removed body's stale contact
/// pairs before the end-of-step detection removed them.
fn check_body_removal_between_steps(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    world.insert(RigidBodyBuilder::fixed(), cuboid(10.0, 0.1));
    let (ball1, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 2.0)),
        ColliderBuilder::ball(0.5),
    );
    let (ball2, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(1.5, 2.0)),
        ColliderBuilder::ball(0.5),
    );
    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
    }

    // The balls land on the floor.
    for _ in 0..60 {
        world.step_collisions_last();
    }

    world.remove_body(ball1);
    for _ in 0..10 {
        world.step_collisions_last();
        assert_no_stale_pairs(&world);
    }

    assert_body_finite(&world, ball2);
}

#[test]
fn step_collisions_last_body_removal_between_steps() {
    check_body_removal_between_steps(false);
}

/// Two overlapping balls of radius 10 on a fixed and a kinematic body.
fn check_kinematic_and_fixed_contact_crash(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.insert(RigidBodyBuilder::fixed(), ColliderBuilder::ball(10.0));
    world.insert(
        RigidBodyBuilder::kinematic_position_based(),
        ColliderBuilder::ball(10.0),
    );
    if manual_init {
        // Kinematic-fixed contacts are not active by default (`ActiveCollisionTypes`): only
        // check that the initialization runs.
        world.initialize_collisions_last_with_events(&(), &());
    }

    world.step_collisions_last();
}

#[test]
fn step_collisions_last_kinematic_and_fixed_contact_crash() {
    check_kinematic_and_fixed_contact_crash(false);
}

/// Inserts two dynamic bodies, a kinematic one and a fixed one, then removes them all before
/// the first step (includes the 0.32 regression where deleting a kinematic body crashed).
fn check_rigid_body_removal_before_step(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let handles = [
        world.insert_body(RigidBodyBuilder::dynamic()),
        world.insert_body(RigidBodyBuilder::dynamic()),
        world.insert_body(RigidBodyBuilder::kinematic_position_based()),
        world.insert_body(RigidBodyBuilder::fixed()),
    ];
    for handle in handles {
        world.remove_body(handle);
    }

    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
        assert_eq!(
            world.contact_pairs().count(),
            0,
            "no contact pairs expected after removing all bodies"
        );
        // A second initialization does nothing.
        world.initialize_collisions_last_with_events(&(), &());
    }

    world.step_collisions_last();
}

#[test]
fn step_collisions_last_rigid_body_removal_before_step() {
    check_rigid_body_removal_before_step(false);
}

/// Removes a collider, then its body, before the first step.
fn check_collider_removal_before_step(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    let (body, collider) = world.insert(RigidBodyBuilder::dynamic(), ColliderBuilder::ball(1.0));
    world.remove_collider(collider);
    world.remove_body(body);

    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
        assert_eq!(
            world.contact_pairs().count(),
            0,
            "no contact pairs expected after removing all colliders"
        );
    }

    for _ in 0..10 {
        world.step_collisions_last();
    }
}

#[test]
fn step_collisions_last_collider_removal_before_step() {
    check_collider_removal_before_step(false);
}

/// Switches a kinematic body with mass to dynamic between steps: the next step must move it
/// with gravity and keep it awake.
fn check_rigid_body_type_changed_dynamic_is_in_active_set(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    let handle =
        world.insert_body(RigidBodyBuilder::kinematic_position_based().additional_mass(1.0));
    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
        // No colliders, so no contacts at t=0.
        assert_eq!(world.contact_pairs().count(), 0);
    }

    world.step_collisions_last();
    world.bodies[handle].set_body_type(RigidBodyType::Dynamic, true);
    world.step_collisions_last();

    let body = &world.bodies[handle];
    assert!(
        body.translation().y < 0.0,
        "gravity should apply after switching to dynamic"
    );
    assert!(!body.is_sleeping());
}

#[test]
fn step_collisions_last_rigid_body_type_changed_dynamic_is_in_active_set() {
    check_rigid_body_type_changed_dynamic_is_in_active_set(false);
}

/// A revolute joint between a fixed and a dynamic body, stepped with `dt = 0`.
fn check_joint_step_delta_time_0(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 0.0;
    let fixed = world.insert_body(RigidBodyBuilder::fixed().additional_mass(1.0));
    let dynamic = world.insert_body(RigidBodyBuilder::dynamic().additional_mass(1.0));
    world.insert_impulse_joint(
        fixed,
        dynamic,
        revolute()
            .local_anchor1(v(0.0, 1.0))
            .local_anchor2(v(0.0, -3.0)),
    );
    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
    }

    world.step_collisions_last();

    assert_body_finite(&world, dynamic);
}

#[test]
fn step_collisions_last_joint_step_delta_time_0() {
    check_joint_step_delta_time_0(false);
}

/// Teleports and disables a body between steps, then teleports and re-enables it (the 0.32
/// `test_multi_sap_disable_body` scenario, made dimension-generic).
fn check_multi_sap_disable_body(manual_init: bool) {
    let mut world = PhysicsWorld::new();
    let ground_co = world.insert_collider(cuboid(100.0, 0.1), None);
    let (ball, ball_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 10.0)),
        ColliderBuilder::ball(0.5).restitution(0.7),
    );

    if manual_init {
        world.initialize_collisions_last_with_events(&(), &());
        // At t=0 the ball is at y=10 and the ground at y=0: no contact.
        let contact = world.contact_pair(ball_co, ground_co);
        assert!(
            contact.is_none_or(|pair| !pair.has_any_active_contact()),
            "the ball at y=10 should not touch the ground at t=0"
        );
    }

    world.step_collisions_last();

    let body = &mut world.bodies[ball];
    body.set_translation(v(1.0, 1.0), true);
    body.set_rotation(rotation(1.0), true);
    body.set_enabled(false);
    world.step_collisions_last();

    let body = &mut world.bodies[ball];
    body.set_translation(v(0.0, 0.0), true);
    body.set_rotation(rotation(0.0), true);
    body.set_enabled(true);
    world.step_collisions_last();

    assert_body_finite(&world, ball);
}

#[test]
fn step_collisions_last_multi_sap_disable_body() {
    check_multi_sap_disable_body(false);
}

// =========================================================================
// Ported from 0.32: manual `initialize_collisions_last` before the first step.
// =========================================================================

#[test]
fn manual_init_basic() {
    let mut pipeline = PhysicsPipeline::new();
    let gravity = Vector::Y * -9.81;
    let integration_parameters = IntegrationParameters::default();
    let mut broad_phase = BroadPhaseBvh::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut ccd = CCDSolver::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut islands = IslandManager::new();

    let floor = bodies.insert(RigidBodyBuilder::fixed());
    let floor_co = colliders.insert_with_parent(cuboid(10.0, 0.1), floor, &mut bodies);
    let ball = bodies.insert(RigidBodyBuilder::dynamic().translation(v(0.0, 5.0)));
    let ball_co = colliders.insert_with_parent(
        ColliderBuilder::ball(0.5).restitution(0.7),
        ball,
        &mut bodies,
    );

    // Populates the narrow-phase with the t=0 collision data.
    pipeline.initialize_collisions_last(
        &integration_parameters,
        &mut islands,
        &mut broad_phase,
        &mut narrow_phase,
        &mut bodies,
        &mut colliders,
        &mut impulse_joints,
        &mut multibody_joints,
        &mut ccd,
        &(),
        &(),
    );
    assert!(pipeline.collisions_last_initialized());

    // At t=0 the ball is at y=5 and the floor at y=0: no contact.
    let contact = narrow_phase.contact_pair(ball_co, floor_co);
    assert!(
        contact.is_none_or(|pair| !pair.has_any_active_contact()),
        "the ball at y=5 should not touch the floor at t=0"
    );
    // The initialization didn't move anything.
    assert_eq!(bodies[ball].translation().y, 5.0);

    run_step_collisions_last(
        &mut pipeline,
        gravity,
        &integration_parameters,
        &mut islands,
        &mut broad_phase,
        &mut narrow_phase,
        &mut bodies,
        &mut colliders,
        &mut impulse_joints,
        &mut multibody_joints,
        &mut ccd,
        100,
    );

    let y = bodies[ball].translation().y;
    assert!(y < 5.0, "the ball should have fallen, but y={y}");
    assert!(
        y > -1.0,
        "the ball should not fall through the floor, but y={y}"
    );
}

#[test]
fn manual_init_kinematic_and_fixed_contact_crash() {
    check_kinematic_and_fixed_contact_crash(true);
}

#[test]
fn manual_init_rigid_body_removal_before_step() {
    check_rigid_body_removal_before_step(true);
}

#[test]
fn manual_init_collider_removal_before_step() {
    check_collider_removal_before_step(true);
}

#[test]
fn manual_init_rigid_body_type_changed_dynamic_is_in_active_set() {
    check_rigid_body_type_changed_dynamic_is_in_active_set(true);
}

#[test]
fn manual_init_joint_step_delta_time_0() {
    check_joint_step_delta_time_0(true);
}

#[test]
fn manual_init_matches_step() {
    check_matches_step(true);
}

#[test]
fn manual_init_joint_added_between_steps() {
    check_joint_added_between_steps(true);
}

#[test]
fn manual_init_body_removal_between_steps() {
    check_body_removal_between_steps(true);
}

#[test]
fn manual_init_multi_sap_disable_body() {
    check_multi_sap_disable_body(true);
}

#[test]
fn manual_init_is_idempotent() {
    let mut world = PhysicsWorld::new();
    // Two overlapping balls at the origin.
    world.insert(RigidBodyBuilder::fixed(), ColliderBuilder::ball(1.0));
    world.insert(RigidBodyBuilder::dynamic(), ColliderBuilder::ball(1.0));

    world.initialize_collisions_last_with_events(&(), &());
    let contact_count = world.contact_pairs().count();
    assert!(contact_count > 0, "expected contacts after the first init");
    assert!(world.physics_pipeline.collisions_last_initialized());

    // The second initialization does nothing.
    world.initialize_collisions_last_with_events(&(), &());
    assert_eq!(
        world.contact_pairs().count(),
        contact_count,
        "the second init should not change the contact count"
    );

    for _ in 0..10 {
        world.step_collisions_last();
    }
    assert_all_bodies_finite(&world);
}

// =========================================================================
// 0.35: user changes, CCD substeps, deferred BVH optimization, snapshots.
// =========================================================================

/// Re-parents a collider, changes a body type and toggles a sensor between steps. These
/// changes recolor and relink pairs in the narrow-phase and invalidate stored contacts, so
/// they run through the catch-up detection; the debug validators of the solver graph and the
/// persistent islands check the result each step. The same edits applied to a stock-stepped
/// world must lead to the same pairs once things settle.
#[test]
fn step_collisions_last_reparent_body_type_and_sensor_toggle() {
    struct Scene {
        world: PhysicsWorld,
        box_a: RigidBodyHandle,
        box_a_co: ColliderHandle,
        box_b: RigidBodyHandle,
        ball: RigidBodyHandle,
        extra: ColliderHandle,
    }

    let scene = || {
        let mut world = PhysicsWorld::new();
        world.insert(RigidBodyBuilder::fixed(), cuboid(10.0, 0.1));
        let (box_a, box_a_co) = world.insert(
            RigidBodyBuilder::dynamic().translation(v(-2.0, 0.6)),
            cuboid(0.5, 0.5),
        );
        let (box_b, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(v(2.0, 0.6)),
            cuboid(0.5, 0.5),
        );
        let (ball, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(v(0.0, 0.6)),
            ColliderBuilder::ball(0.5),
        );
        // Sits on top of box A (same parent: no contact), and moves to box B.
        let extra = world.insert_collider(cuboid(0.25, 0.25).translation(v(0.0, 0.8)), Some(box_a));
        Scene {
            world,
            box_a,
            box_a_co,
            box_b,
            ball,
            extra,
        }
    };

    let edit = |scene: &mut Scene, step: usize| {
        let world = &mut scene.world;
        match step {
            20 => world
                .colliders
                .set_parent(scene.extra, Some(scene.box_b), &mut world.bodies),
            30 => {
                world.bodies[scene.ball].set_body_type(RigidBodyType::KinematicPositionBased, true)
            }
            40 => world.bodies[scene.ball].set_body_type(RigidBodyType::Dynamic, true),
            50 => world.colliders[scene.box_a_co].set_sensor(true),
            52 => world.colliders[scene.box_a_co].set_sensor(false),
            _ => {}
        }
    };

    let mut stock = scene();
    let mut last = scene();
    for step in 0..120 {
        edit(&mut stock, step);
        stock.world.step();
        edit(&mut last, step);
        last.world.step_collisions_last();
        assert_all_bodies_finite(&last.world);
        assert_no_stale_pairs(&last.world);

        if step == 21 {
            // The re-parented collider moved with its new body.
            assert_eq!(last.world.colliders[last.extra].parent(), Some(last.box_b));
        }
        if step == 51 {
            let (_, intersecting) = pair_state(&last.world);
            assert!(
                !intersecting.is_empty(),
                "the box turned into a sensor should intersect the floor"
            );
        }
    }

    assert_eq!(
        pair_state(&last.world),
        pair_state(&stock.world),
        "collisions-last and stock stepping disagree on the settled pairs"
    );
    let _ = last.box_a;
}

/// `max_ccd_substeps = 4`, a fast CCD body, and colliders removed between steps. The removals
/// skip the catch-up detection, so the intermediate CCD-substep detection is the first to
/// apply them to the broad-phase, and the end-of-step detection must not apply them again.
#[test]
fn step_collisions_last_ccd_substeps_with_removal() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.max_ccd_substeps = 4;

    // The wall's face is at x = 4.9.
    world.insert_collider(cuboid(0.1, 5.0).translation(v(5.0, 0.0)), None);
    // A fixed collider and a dynamic body near the bullet's path, removed mid-flight.
    let obstacle = world.insert_collider(cuboid(0.5, 0.5).translation(v(0.0, 3.0)), None);
    let (target, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(2.0, 1.5)),
        cuboid(0.3, 0.3),
    );
    // 200 m/s: about 3.3 m per step, so it reaches the wall during the third step.
    let (bullet, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(v(-5.0, 0.0))
            .linvel(v(200.0, 0.0))
            .ccd_enabled(true),
        ColliderBuilder::ball(0.2),
    );

    world.step_collisions_last();
    world.step_collisions_last();

    world.remove_collider(obstacle);
    world.remove_body(target);

    world.step_collisions_last();
    assert!(
        world.physics_pipeline.counters.ccd.num_substeps > 1,
        "the bullet's impact should split the step into CCD substeps"
    );

    for _ in 0..10 {
        let x = world.bodies[bullet].translation().x;
        assert!(x <= 4.9, "the bullet tunneled through the wall: x={x}");
        assert_all_bodies_finite(&world);
        assert_no_stale_pairs(&world);
        assert_eq!(
            world.broad_phase.tree.leaf_count() as usize,
            world.colliders.len(),
            "the broad-phase should hold exactly one leaf per live collider"
        );
        world.step_collisions_last();
    }
}

/// A ray cast between collisions-last steps hits the fixed ground. The end-of-step detection
/// can hand the BVH to a deferred optimization; if the step returned without joining it, the
/// broad-phase tree would be empty for queries and snapshots between steps.
#[test]
fn step_collisions_last_ray_cast_between_steps_hits_static() {
    let mut world = PhysicsWorld::new();
    let (_, ground) = world.insert(
        RigidBodyBuilder::fixed().translation(v(0.0, -0.5)),
        cuboid(100.0, 0.5),
    );
    // Enough moving leaves to put the broad-phase in its bulk regime, where the tree
    // optimization is deferred.
    for i in 0..40 {
        world.insert(
            RigidBodyBuilder::dynamic()
                .translation(v(i as Real * 1.1 - 22.0, 2.0 + (i % 5) as Real)),
            ColliderBuilder::ball(0.5),
        );
    }

    let ray = Ray::new(v(60.0, 10.0), -Vector::Y);
    for _ in 0..60 {
        world.step_collisions_last();
        assert_eq!(
            world.broad_phase.tree.leaf_count() as usize,
            world.colliders.len(),
            "the broad-phase tree is missing leaves between steps"
        );
        let hit = world.cast_ray(&ray, Real::MAX, true, QueryFilter::default());
        assert_eq!(
            hit.map(|(handle, _)| handle),
            Some(ground),
            "the ray should hit the fixed ground between steps"
        );
    }
}

/// Collision events in emission order, as `(started, lower collider index, higher collider
/// index)`.
#[derive(Default)]
struct CollisionLog(std::sync::Mutex<Vec<(bool, u32, u32)>>);

impl CollisionLog {
    /// The events recorded since the last call.
    fn take(&self) -> Vec<(bool, u32, u32)> {
        core::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl EventHandler for CollisionLog {
    fn handle_collision_event(
        &self,
        _: &RigidBodySet,
        _: &ColliderSet,
        event: CollisionEvent,
        _: Option<&ContactPair>,
    ) {
        let (started, h1, h2) = match event {
            CollisionEvent::Started(h1, h2, _) => (true, h1, h2),
            CollisionEvent::Stopped(h1, h2, _) => (false, h1, h2),
        };
        let (a, b) = (h1.into_raw_parts().0, h2.into_raw_parts().0);
        self.0.lock().unwrap().push((started, a.min(b), a.max(b)));
    }

    fn handle_contact_force_event(
        &self,
        _: Real,
        _: &RigidBodySet,
        _: &ColliderSet,
        _: &ContactPair,
        _: Real,
    ) {
    }
}

/// Probe for the narrow-phase's pending solver-graph maintenance. A collisions-last step ends
/// with a detection, so the pairs it changed (here: a contact that just started) are only
/// reconciled into the solver contact graph by the next solve. A catch-up detection in the next
/// step replaces that pending list. Twin worlds, one of them with an unrelated change that
/// triggers the catch-up right after the contact started, must keep the ball and the collision
/// events bit-identical. In debug builds the solver-graph validator also checks the graph.
#[test]
fn step_collisions_last_catch_up_after_contact_start() {
    let make = || {
        let (mut world, ball, ball_co, floor_co) = ball_on_floor(0.0);
        world.colliders[ball_co].set_active_events(ActiveEvents::COLLISION_EVENTS);
        let far = world.insert_collider(cuboid(0.5, 0.5).translation(v(50.0, 50.0)), None);
        (world, ball, ball_co, floor_co, far)
    };
    let (mut edited, ball, ball_co, floor_co, far) = make();
    let (mut twin, ..) = make();
    let (edited_log, twin_log) = (CollisionLog::default(), CollisionLog::default());

    let touching = |world: &PhysicsWorld| {
        world
            .contact_pair(ball_co, floor_co)
            .is_some_and(|pair| pair.has_any_active_contact())
    };

    let mut edited_at = None;
    for step in 0..120 {
        let was_touching = touching(&edited);
        edited.step_collisions_last_with_events(&(), &edited_log);
        twin.step_collisions_last_with_events(&(), &twin_log);
        if edited_at.is_none() && !was_touching && touching(&edited) {
            // Sensor toggle: outside the deferrable changes, so the next step catches up.
            edited.colliders[far].set_sensor(true);
            edited_at = Some(step);
        }

        let (a, b) = (&edited.bodies[ball], &twin.bodies[ball]);
        assert_eq!(
            (a.translation(), a.linvel()),
            (b.translation(), b.linvel()),
            "the unrelated change altered the ball's motion at step {step} \
             (change made after step {edited_at:?})"
        );
        assert_eq!(
            edited_log.take(),
            twin_log.take(),
            "the unrelated change altered the collision events at step {step}"
        );
    }
    assert!(edited_at.is_some(), "the ball never touched the floor");
}

/// Counts `modify_solver_contacts` calls.
#[derive(Default)]
struct HookCalls(core::sync::atomic::AtomicUsize);

impl HookCalls {
    fn take(&self) -> usize {
        self.0.swap(0, core::sync::atomic::Ordering::Relaxed)
    }
}

impl PhysicsHooks for HookCalls {
    fn modify_solver_contacts(&self, _: &mut ContactModificationContext) {
        self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Inserting a collider triggers the catch-up detection, so collision detection runs twice on
/// that step (the contact hook runs twice), but the collision events are not duplicated: the
/// per-pair event sequence matches stock stepping.
#[test]
fn step_collisions_last_catch_up_does_not_duplicate_collision_events() {
    const INSERT_STEP: usize = 10;
    let events = ActiveEvents::COLLISION_EVENTS;

    let scene = || {
        let mut world = PhysicsWorld::new();
        world.insert(
            RigidBodyBuilder::fixed(),
            cuboid(10.0, 0.1).active_events(events),
        );
        // Resting on the floor (top at y = 1.1).
        world.insert(
            RigidBodyBuilder::dynamic()
                .translation(v(0.0, 0.6))
                .can_sleep(false),
            cuboid(0.5, 0.5).active_events(events),
        );
        world
    };
    // Slightly overlapping the box, so their contact starts during the insertion step.
    let insert = |world: &mut PhysicsWorld| {
        world.insert(
            RigidBodyBuilder::dynamic().translation(v(0.0, 1.38)),
            ColliderBuilder::ball(0.3)
                .active_events(events)
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        )
    };

    let (mut stock, mut last) = (scene(), scene());
    let (stock_log, last_log) = (CollisionLog::default(), CollisionLog::default());
    let (stock_hooks, last_hooks) = (HookCalls::default(), HookCalls::default());
    let (mut stock_events, mut last_events) = (Vec::new(), Vec::new());
    let mut ball_co = None;

    for step in 0..60 {
        if step == INSERT_STEP {
            insert(&mut stock);
            ball_co = Some(insert(&mut last).1.into_raw_parts().0);
        }
        stock.step_with_events(&stock_hooks, &stock_log);
        last.step_collisions_last_with_events(&last_hooks, &last_log);

        let step_events = last_log.take();
        let (stock_calls, last_calls) = (stock_hooks.take(), last_hooks.take());
        if step == INSERT_STEP {
            let ball_co = ball_co.unwrap();
            let ball_starts = step_events
                .iter()
                .filter(|(started, a, b)| *started && (*a == ball_co || *b == ball_co))
                .count();
            assert_eq!(
                ball_starts, 1,
                "the inserted ball's contact should start exactly once: {step_events:?}"
            );
            assert!(
                last_calls > 0 && last_calls == 2 * stock_calls,
                "the insertion step should run the contact hook for both detections: \
                 collisions-last {last_calls}, stock {stock_calls}"
            );
        }
        stock_events.extend(stock_log.take());
        last_events.extend(step_events);
    }

    // Per pair, the event sequence alternates started/stopped, and matches stock stepping.
    let per_pair = |events: &[(bool, u32, u32)]| {
        let mut pairs: Vec<((u32, u32), Vec<bool>)> = Vec::new();
        for (started, a, b) in events {
            match pairs.iter_mut().find(|(pair, _)| *pair == (*a, *b)) {
                Some((_, sequence)) => sequence.push(*started),
                None => pairs.push(((*a, *b), alloc::vec![*started])),
            }
        }
        pairs.sort_unstable();
        pairs
    };
    let last_pairs = per_pair(&last_events);
    for (pair, sequence) in &last_pairs {
        for (i, started) in sequence.iter().enumerate() {
            assert_eq!(
                *started,
                i % 2 == 0,
                "pair {pair:?} has a duplicated collision event: {sequence:?}"
            );
        }
    }
    assert_eq!(
        last_pairs.len(),
        2,
        "expected box-floor and ball-box events"
    );
    assert_eq!(last_pairs, per_pair(&stock_events));
}

/// Snapshot probe for the narrow-phase's pending solver-graph maintenance (see
/// [`step_collisions_last_catch_up_after_contact_start`]). The pending list is not serialized,
/// so the snapshot is taken right after a collisions-last step in which a contact started
/// during the end-of-step detection. Restoring into a fresh world and pipeline with
/// `set_collisions_last_initialized(true)` and stepping on must continue exactly as not saving
/// would have: the snapshots must stay byte-identical. Balls keep spawning (catch-up steps) and
/// getting removed on both sides of the snapshot. In debug builds the solver-graph validator
/// also checks both runs.
#[cfg(feature = "serde-serialize")]
fn check_snapshot_roundtrip_after_contact_start(min_steps: usize) {
    let mut world = PhysicsWorld::new();
    world.insert(
        RigidBodyBuilder::fixed().translation(v(0.0, -0.5)),
        cuboid(20.0, 0.5),
    );
    for i in 0..6 {
        for j in 0..3 {
            world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(v(i as Real * 1.05 - 3.0, j as Real * 1.05 + 0.55)),
                cuboid(0.5, 0.5),
            );
        }
    }

    let is_edit_step = |step: usize| step % 7 == 0 || step % 11 == 5;
    let edit = |world: &mut PhysicsWorld, step: usize, spawned: &mut Vec<RigidBodyHandle>| {
        if step % 7 == 0 {
            let x = (step % 5) as Real * 3.0 - 6.0;
            let (rb, _) = world.insert(
                RigidBodyBuilder::dynamic().translation(v(x, 4.5)),
                ColliderBuilder::ball(0.4),
            );
            spawned.push(rb);
        }
        if step % 11 == 5 && spawned.len() > 2 {
            let rb = spawned.remove(0);
            world.remove_body(rb);
        }
    };

    // Step until a contact starts during the end-of-step detection of a step without edits
    // (so no catch-up detection ran before the solve).
    let mut spawned = Vec::new();
    let mut step = 0;
    loop {
        let touching_before = pair_state(&world).0;
        edit(&mut world, step, &mut spawned);
        world.step_collisions_last();
        let started = pair_state(&world)
            .0
            .iter()
            .any(|pair| !touching_before.contains(pair));
        step += 1;
        if step >= min_steps && !is_edit_step(step - 1) && started {
            break;
        }
        assert!(
            step < min_steps + 300,
            "no contact started after step {min_steps}"
        );
    }

    let snapshot = bincode::serialize(&world).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&snapshot).unwrap();
    restored
        .physics_pipeline
        .set_collisions_last_initialized(true);
    let mut spawned_restored = spawned.clone();

    for i in 0..60 {
        edit(&mut world, step + i, &mut spawned);
        world.step_collisions_last();
        edit(&mut restored, step + i, &mut spawned_restored);
        restored.step_collisions_last();
        assert!(
            bincode::serialize(&world).unwrap() == bincode::serialize(&restored).unwrap(),
            "the restored world diverged {} step(s) after the restore (saved after step {step})",
            i + 1
        );
    }
}

/// The minimal case of [`check_snapshot_roundtrip_after_contact_start`]: a single ball, saved
/// right after the step in which it starts touching the floor. With nothing else waking or
/// sleeping, the next solve maintains the solver graph incrementally, so it relies entirely on
/// the maintenance that was pending when the snapshot was taken.
#[cfg(feature = "serde-serialize")]
fn check_snapshot_roundtrip_single_ball() {
    let (mut world, _, ball_co, floor_co) = ball_on_floor(0.0);
    let touching = |world: &PhysicsWorld| {
        world
            .contact_pair(ball_co, floor_co)
            .is_some_and(|pair| pair.has_any_active_contact())
    };

    let mut steps = 0;
    while !touching(&world) {
        world.step_collisions_last();
        steps += 1;
        assert!(steps < 200, "the ball never touched the floor");
    }

    let snapshot = bincode::serialize(&world).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&snapshot).unwrap();
    restored
        .physics_pipeline
        .set_collisions_last_initialized(true);

    for i in 0..60 {
        world.step_collisions_last();
        restored.step_collisions_last();
        assert!(
            bincode::serialize(&world).unwrap() == bincode::serialize(&restored).unwrap(),
            "the restored single-ball world diverged {} step(s) after the restore (saved \
             after step {steps})",
            i + 1
        );
    }
}

#[cfg(feature = "serde-serialize")]
#[test]
fn step_collisions_last_snapshot_roundtrip() {
    check_snapshot_roundtrip_single_ball();
    check_snapshot_roundtrip_after_contact_start(20);
    check_snapshot_roundtrip_after_contact_start(45);
    check_snapshot_roundtrip_after_contact_start(80);
}

/// A stack of `count` boxes resting on a fixed floor whose top is at `y = 0.1`, bottom box first.
fn box_stack(count: usize) -> (PhysicsWorld, Vec<RigidBodyHandle>) {
    let mut world = PhysicsWorld::new();
    world.insert(RigidBodyBuilder::fixed(), cuboid(10.0, 0.1));
    let boxes = (0..count)
        .map(|i| {
            world
                .insert(
                    RigidBodyBuilder::dynamic().translation(v(0.0, 0.6 + i as Real)),
                    cuboid(0.5, 0.5),
                )
                .0
        })
        .collect();
    (world, boxes)
}

/// Steps `world` (with `step`, or with `step_collisions_last`) until all `bodies` sleep.
fn step_until_asleep(world: &mut PhysicsWorld, collisions_last: bool, bodies: &[RigidBodyHandle]) {
    for _ in 0..3000 {
        if collisions_last {
            world.step_collisions_last();
        } else {
            world.step();
        }
        if bodies
            .iter()
            .all(|handle| world.bodies[*handle].is_sleeping())
        {
            return;
        }
    }
    panic!("the bodies never fell asleep");
}

/// Wakes the top box of a sleeping stack between steps, in twin scenes stepped with stock `step`
/// and with collisions-last. Falling asleep count-cleared the solver hints of the stack's pairs:
/// stock stepping repairs them in the detection that precedes the solve, but collisions-last
/// solves first. Unrepaired, the woken stack would be solved for one step without its resting
/// contacts and sink under gravity, so no box may sink more than with stock stepping.
fn check_wake_after_sleep(count: usize) {
    let (mut stock, stock_boxes) = box_stack(count);
    let (mut last, last_boxes) = box_stack(count);
    step_until_asleep(&mut stock, false, &stock_boxes);
    step_until_asleep(&mut last, true, &last_boxes);

    let heights = |world: &PhysicsWorld, boxes: &[RigidBodyHandle]| -> Vec<Real> {
        boxes
            .iter()
            .map(|handle| world.bodies[*handle].translation().y)
            .collect()
    };
    let stock_rest = heights(&stock, &stock_boxes);
    let last_rest = heights(&last, &last_boxes);

    let (stock_top, last_top) = (*stock_boxes.last().unwrap(), *last_boxes.last().unwrap());
    stock.wake_up(stock_top, true);
    last.wake_up(last_top, true);
    assert!(last_boxes.iter().all(|h| !last.bodies[*h].is_sleeping()));

    for step in 0..10 {
        stock.step();
        last.step_collisions_last();
        let stock_now = heights(&stock, &stock_boxes);
        let last_now = heights(&last, &last_boxes);
        for i in 0..count {
            let stock_sink = stock_rest[i] - stock_now[i];
            let last_sink = last_rest[i] - last_now[i];
            assert!(
                last_sink <= stock_sink + 1.0e-4,
                "box {i} of {count} sank {last_sink} m with collisions-last but {stock_sink} m \
                 with stock stepping, {} step(s) after the wake-up",
                step + 1
            );
        }
    }
}

#[test]
fn step_collisions_last_wake_after_sleep() {
    check_wake_after_sleep(3);
    check_wake_after_sleep(1);
}
