//! Tests of the collisions-last stepping order: [`PhysicsPipeline::step_collisions_last`] and
//! [`PhysicsPipeline::initialize_collisions_last`].
//!
//! The regression scenes of `test.rs`, stepped in that order; the changes made between steps
//! (their narrow-phase half applied before the solve); CCD substeps; the deferred BVH
//! optimization; snapshots; the bit-identity of both orders when nothing changes between steps;
//! and the contact data matching the poses read between steps.

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

// The regression scenes of `test.rs`, stepped with the collision detection last.

/// A ball dropped onto a floor, through the raw pipeline API. The initialization populates the
/// narrow-phase with the collision data at `t = 0` without moving anything; the ball then falls
/// and lands on the floor.
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

    let floor = bodies.insert(RigidBodyBuilder::fixed());
    let floor_co = colliders.insert_with_parent(cuboid(10.0, 0.1), floor, &mut bodies);
    let ball = bodies.insert(RigidBodyBuilder::dynamic().translation(v(0.0, 5.0)));
    let ball_co = colliders.insert_with_parent(
        ColliderBuilder::ball(0.5).restitution(0.7),
        ball,
        &mut bodies,
    );

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

    // At t = 0 the ball is at y = 5 and the floor at y = 0: no contact, and nothing moved.
    let contact = narrow_phase.contact_pair(ball_co, floor_co);
    assert!(
        contact.is_none_or(|pair| !pair.has_any_active_contact()),
        "the ball at y = 5 should not touch the floor at t = 0"
    );
    assert_eq!(bodies[ball].translation().y, 5.0);

    for _ in 0..100 {
        pipeline.step_collisions_last(
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
            &(),
            &(),
        );
    }

    let y = bodies[ball].translation().y;
    assert!(y < 5.0, "the ball should have fallen, but y = {y}");
    assert!(
        y > -1.0,
        "the ball should not fall through the floor, but y = {y}"
    );
}

/// A bouncing ball settles at the same height with `step` and with `step_collisions_last`.
#[test]
fn step_collisions_last_matches_step() {
    let (mut standard, standard_ball, _, _) = ball_on_floor(0.7);
    let (mut last, last_ball, _, _) = ball_on_floor(0.7);
    last.initialize_collisions_last();

    // Long enough for the bouncing to die out and the ball to come to rest.
    for _ in 0..600 {
        standard.step();
        last.step_collisions_last();
    }

    let standard_y = standard.bodies[standard_ball].translation().y;
    let last_y = last.bodies[last_ball].translation().y;
    // Floor top (0.1) + ball radius (0.5).
    let rest_y = 0.6;
    assert!(
        (standard_y - rest_y).abs() < 0.01,
        "`step` did not settle the ball on the floor: y = {standard_y}"
    );
    assert!(
        (standard_y - last_y).abs() < 1.0e-3,
        "`step` and `step_collisions_last` settled at different heights: {standard_y} and {last_y}"
    );
}

/// Adds a body and a joint between steps. The user-changes stage must process the new body
/// before the solve, which indexes bodies by their slot in the active set, and the joint must act
/// from the step of its insertion on: the new body, which the joint swings 5.4 m, must follow the
/// path it follows with `step`.
#[test]
fn step_collisions_last_joint_added_between_steps() {
    let new_body_path = |collisions_last: bool| {
        let mut world = PhysicsWorld::new();
        let fixed_body = world.insert_body(RigidBodyBuilder::fixed());
        world.insert(
            RigidBodyBuilder::dynamic().translation(v(0.0, 5.0)),
            ColliderBuilder::ball(0.5),
        );
        let step = |world: &mut PhysicsWorld| {
            if collisions_last {
                world.step_collisions_last();
            } else {
                world.step();
            }
        };
        if collisions_last {
            world.initialize_collisions_last();
        }

        for _ in 0..10 {
            step(&mut world);
        }

        let (new_body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(v(2.0, 5.0)),
            ColliderBuilder::ball(0.5),
        );
        world.insert_impulse_joint(fixed_body, new_body, revolute());

        let mut path = Vec::new();
        for _ in 0..10 {
            step(&mut world);
            path.push(world.bodies[new_body].translation());
        }

        assert_all_bodies_finite(&world);
        path
    };

    let last = new_body_path(true);
    let standard = new_body_path(false);
    for (i, (last, standard)) in last.iter().zip(&standard).enumerate() {
        assert!(
            (*last - *standard).length() <= 1.0e-4,
            "the new body moved differently from `step` {} step(s) after the joint insertion: \
             collisions-last {last:?}, `step` {standard:?}",
            i + 1
        );
    }
}

/// Removes a body resting on the floor between steps. Its contact pairs must leave the
/// narrow-phase before the island update walks the pairs of the remaining bodies.
#[test]
fn step_collisions_last_body_removal_between_steps() {
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
    world.initialize_collisions_last();

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

/// Two overlapping balls of radius 10 on a fixed and a kinematic body: their pair is not active
/// by default (`ActiveCollisionTypes`).
#[test]
fn step_collisions_last_kinematic_and_fixed_contact_crash() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.insert(RigidBodyBuilder::fixed(), ColliderBuilder::ball(10.0));
    world.insert(
        RigidBodyBuilder::kinematic_position_based(),
        ColliderBuilder::ball(10.0),
    );
    world.initialize_collisions_last();
    world.step_collisions_last();
}

/// Inserts two dynamic bodies, a kinematic one and a fixed one, then removes them all before the
/// initialization.
#[test]
fn step_collisions_last_rigid_body_removal_before_step() {
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

    world.initialize_collisions_last();
    assert_eq!(
        world.contact_pairs().count(),
        0,
        "no contact pairs expected after removing all bodies"
    );
    world.step_collisions_last();
}

/// Removes a collider, then its body, before the initialization.
#[test]
fn step_collisions_last_collider_removal_before_step() {
    let mut world = PhysicsWorld::new();
    let (body, collider) = world.insert(RigidBodyBuilder::dynamic(), ColliderBuilder::ball(1.0));
    world.remove_collider(collider);
    world.remove_body(body);

    world.initialize_collisions_last();
    assert_eq!(
        world.contact_pairs().count(),
        0,
        "no contact pairs expected after removing all colliders"
    );
    for _ in 0..10 {
        world.step_collisions_last();
    }
}

/// Switches a kinematic body with mass to dynamic between steps: the next step must move it
/// with gravity and keep it awake.
#[test]
fn step_collisions_last_rigid_body_type_changed_dynamic_is_in_active_set() {
    let mut world = PhysicsWorld::new();
    let handle =
        world.insert_body(RigidBodyBuilder::kinematic_position_based().additional_mass(1.0));
    world.initialize_collisions_last();

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

/// A revolute joint between a fixed and a dynamic body, stepped with `dt = 0`.
#[test]
fn step_collisions_last_joint_step_delta_time_0() {
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
    world.initialize_collisions_last();

    world.step_collisions_last();

    assert_body_finite(&world, dynamic);
}

/// Teleports and disables a body between steps, then teleports and re-enables it.
#[test]
fn step_collisions_last_multi_sap_disable_body() {
    let mut world = PhysicsWorld::new();
    let ground_co = world.insert_collider(cuboid(100.0, 0.1), None);
    let (ball, ball_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 10.0)),
        ColliderBuilder::ball(0.5).restitution(0.7),
    );

    world.initialize_collisions_last();
    // At t = 0 the ball is at y = 10 and the ground at y = 0: no contact.
    let contact = world.contact_pair(ball_co, ground_co);
    assert!(
        contact.is_none_or(|pair| !pair.has_any_active_contact()),
        "the ball at y = 10 should not touch the ground at t = 0"
    );

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

// Changes made between steps, CCD substeps, the deferred BVH optimization, snapshots.

/// `max_ccd_substeps = 4`, a fast CCD body, and colliders removed between steps. The pairs of
/// the removed colliders leave the narrow-phase before the solve; their broad-phase leaves only
/// leave at the end-of-step detection, so the CCD sweeps of that step see them, and must ignore
/// them.
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
    world.initialize_collisions_last();

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

/// `max_ccd_substeps = 4`, a fast CCD bullet, and two colliders in its path removed between steps,
/// each right before the bullet sweeps through the space it occupied: a fixed collider, then a
/// dynamic body. The broad-phase still holds the removed collider's leaf during that step's CCD
/// sweeps (it drops it in the end-of-step detection). The bullet must not collide with those
/// ghosts: step for step, it must move as in the same scene without the two colliders, with as
/// many CCD substeps.
#[test]
fn step_collisions_last_ccd_ignores_colliders_removed_between_steps() {
    let simulate = |with_ghosts: bool| {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        world.integration_parameters.max_ccd_substeps = 4;

        // The wall's face is at x = 4.9.
        world.insert_collider(cuboid(0.1, 5.0).translation(v(5.0, 0.0)), None);
        // Straddling the bullet's path at x = 0 and x = 2.5.
        let ghosts = with_ghosts.then(|| {
            let fixed = world.insert_collider(cuboid(0.5, 0.5), None);
            let (body, _) = world.insert(
                RigidBodyBuilder::dynamic().translation(v(2.5, 0.0)),
                cuboid(0.3, 0.3),
            );
            (fixed, body)
        });
        // 200 m/s: about 3.3 m per step from x = -5, so it crosses x = 0 during step 1 and
        // x = 2.5 during step 2, where it reaches the wall.
        let (bullet, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(v(-5.0, 0.0))
                .linvel(v(200.0, 0.0))
                .ccd_enabled(true),
            ColliderBuilder::ball(0.2),
        );

        world.initialize_collisions_last();

        let mut trace = Vec::new();
        for step in 0..6 {
            if let Some((fixed, body)) = ghosts {
                match step {
                    1 => {
                        world.remove_collider(fixed);
                    }
                    2 => {
                        world.remove_body(body);
                    }
                    _ => {}
                }
                if step == 1 || step == 2 {
                    assert_eq!(
                        world.broad_phase.tree.leaf_count() as usize,
                        world.colliders.len() + 1,
                        "the removed collider's leaf should stay in the broad-phase until the next \
                         detection"
                    );
                }
            }
            world.step_collisions_last();
            assert_all_bodies_finite(&world);
            assert_no_stale_pairs(&world);
            let rb = &world.bodies[bullet];
            trace.push((
                rb.translation(),
                rb.linvel(),
                world.physics_pipeline.counters.ccd.num_substeps,
            ));
        }
        trace
    };

    let control = simulate(false);
    let ghosts = simulate(true);
    assert!(
        control[1].0.x > 0.7 && control[2].2 > 1,
        "the bullet should pass x = 0.7 during step 1, and hit the wall in CCD substeps during step \
         2: {control:?}"
    );
    for (step, (ghost, control)) in ghosts.iter().zip(&control).enumerate() {
        assert!(
            ghost.2 == control.2
                && (ghost.0 - control.0).length() <= 1.0e-5
                && (ghost.1 - control.1).length() <= 1.0e-5,
            "step {step}: the bullet collided with a removed collider (position, velocity, CCD \
             substeps with the removed colliders: {ghost:?}; without them: {control:?})"
        );
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

    world.initialize_collisions_last();

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

/// Twin worlds, one of them with an unrelated change (a far collider turned into a sensor) made
/// right after a contact started in an end-of-step detection, must keep the ball and the
/// collision events bit-identical. In debug builds the solver-graph validator also checks the
/// graph.
#[test]
fn step_collisions_last_unrelated_change_after_contact_start() {
    let make = || {
        let (mut world, ball, ball_co, floor_co) = ball_on_floor(0.0);
        world.colliders[ball_co].set_active_events(ActiveEvents::COLLISION_EVENTS);
        let far = world.insert_collider(cuboid(0.5, 0.5).translation(v(50.0, 50.0)), None);
        (world, ball, ball_co, floor_co, far)
    };
    let (mut edited, ball, ball_co, floor_co, far) = make();
    let (mut twin, ..) = make();
    let (edited_log, twin_log) = (CollisionLog::default(), CollisionLog::default());
    edited.initialize_collisions_last_with_events(&(), &edited_log);
    twin.initialize_collisions_last_with_events(&(), &twin_log);

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

/// Snapshot probe for the narrow-phase's solver-graph maintenance. Its list of pairs to
/// reconcile is not serialized, so the snapshot is taken right after a collisions-last step in
/// which a contact started during the end-of-step detection. Restoring into a fresh world and
/// pipeline and stepping on must continue exactly as not saving would have: the snapshots must
/// stay byte-identical. Balls keep spawning and
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

    world.initialize_collisions_last();

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

    // Step until a contact starts during the end-of-step detection of a step without edits.
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
    world.initialize_collisions_last();
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

/// Collision and contact-force events in emission order, as `(kind, collider1 index, collider2
/// index, force magnitude bits)`, with kind 0 = started, 1 = stopped, 2 = contact force (the
/// magnitude is 0 for collision events).
#[derive(Default)]
struct EventOrderLog(std::sync::Mutex<Vec<(u8, u32, u32, u64)>>);

impl EventOrderLog {
    /// The events recorded since the last call.
    fn take(&self) -> Vec<(u8, u32, u32, u64)> {
        core::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl EventHandler for EventOrderLog {
    fn handle_collision_event(
        &self,
        _: &RigidBodySet,
        _: &ColliderSet,
        event: CollisionEvent,
        _: Option<&ContactPair>,
    ) {
        let (kind, h1, h2) = match event {
            CollisionEvent::Started(h1, h2, _) => (0, h1, h2),
            CollisionEvent::Stopped(h1, h2, _) => (1, h1, h2),
        };
        self.0
            .lock()
            .unwrap()
            .push((kind, h1.into_raw_parts().0, h2.into_raw_parts().0, 0));
    }

    fn handle_contact_force_event(
        &self,
        _: Real,
        _: &RigidBodySet,
        _: &ColliderSet,
        pair: &ContactPair,
        total_force_magnitude: Real,
    ) {
        self.0.lock().unwrap().push((
            2,
            pair.collider1.into_raw_parts().0,
            pair.collider2.into_raw_parts().0,
            (total_force_magnitude as f64).to_bits(),
        ));
    }
}

/// A contact-modification hook that writes the friction of every manifold it sees and counts
/// its calls.
#[cfg(feature = "serde-serialize")]
#[derive(Default)]
struct FrictionHook(core::sync::atomic::AtomicUsize);

#[cfg(feature = "serde-serialize")]
impl FrictionHook {
    fn take(&self) -> usize {
        self.0.swap(0, core::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(feature = "serde-serialize")]
impl PhysicsHooks for FrictionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        *context.friction = 0.3;
    }
}

/// Steps `world` and `restored` (a bincode snapshot of it) side by side for `steps` steps, with
/// `step` or with `step_collisions_last`, each with its own [`FrictionHook`] and
/// [`EventOrderLog`]. After every step, the two worlds must have emitted the same events in the
/// same order, called the hook as often, and serialize to the same bytes. Returns the number of
/// contact-force events emitted, so callers can check their scene exercised them.
#[cfg(feature = "serde-serialize")]
fn assert_restored_twin_matches(
    world: &mut PhysicsWorld,
    restored: &mut PhysicsWorld,
    collisions_last: bool,
    steps: usize,
    what: &str,
) -> usize {
    let (log, restored_log) = (EventOrderLog::default(), EventOrderLog::default());
    let (hooks, restored_hooks) = (FrictionHook::default(), FrictionHook::default());
    let mut force_events = 0;
    for i in 0..steps {
        if collisions_last {
            world.step_collisions_last_with_events(&hooks, &log);
            restored.step_collisions_last_with_events(&restored_hooks, &restored_log);
        } else {
            world.step_with_events(&hooks, &log);
            restored.step_with_events(&restored_hooks, &restored_log);
        }
        let (events, restored_events) = (log.take(), restored_log.take());
        force_events += events.iter().filter(|event| event.0 == 2).count();
        assert_eq!(
            events,
            restored_events,
            "{what}: the uninterrupted world (left) and the restored world (right) emitted \
             different events {} step(s) after the restore",
            i + 1
        );
        assert_eq!(
            hooks.take(),
            restored_hooks.take(),
            "{what}: the contact hook ran a different number of times {} step(s) after the restore",
            i + 1
        );
        assert!(
            bincode::serialize(world).unwrap() == bincode::serialize(restored).unwrap(),
            "{what}: the restored world's bytes diverged {} step(s) after the restore",
            i + 1
        );
    }
    force_events
}

/// Saves a world right after both contacts of a box start in the same detection, then enables
/// contact-force events on the box in the uninterrupted and the restored world, and steps both
/// with `step_collisions_last` (or with `step`, the control).
///
/// The box falls flat onto the seam of two abutting fixed tiles (one of them hooked), so both
/// pairs change solver-graph membership in the same detection. In collisions-last mode that is
/// the end-of-step detection, whose maintenance must not leave the narrow-phase's list of
/// changed pairs filled: the list is not serialized, so only the uninterrupted world would still
/// hold it at the next solve. Enabling force events flags both pairs for force-event
/// reconciliation, which walks that list (ascending edge ids) before the flagged pairs (graph
/// adjacency order): with a stale list the two worlds would order their force-event pairs
/// differently, so their bytes and the order of their force events would differ. Standard
/// stepping's detection always replaces the list before its solve.
#[cfg(feature = "serde-serialize")]
fn check_restore_keeps_force_event_order(collisions_last: bool) {
    let what = if collisions_last {
        "collisions-last seam landing"
    } else {
        "standard seam landing (control)"
    };
    let mut world = PhysicsWorld::new();
    let tile1 = world.insert_collider(cuboid(1.0, 0.1).translation(v(-1.0, 0.0)), None);
    let tile2 = world.insert_collider(
        cuboid(1.0, 0.1)
            .translation(v(1.0, 0.0))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        None,
    );
    let (_, box_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 1.0)),
        cuboid(0.5, 0.5).active_events(ActiveEvents::COLLISION_EVENTS),
    );
    let touching = |world: &PhysicsWorld, tile: ColliderHandle| {
        world
            .contact_pair(tile, box_co)
            .is_some_and(|pair| pair.has_any_active_contact())
    };

    let (hooks, log) = (FrictionHook::default(), EventOrderLog::default());
    if collisions_last {
        world.initialize_collisions_last_with_events(&hooks, &log);
    }
    let mut steps = 0;
    loop {
        if collisions_last {
            world.step_collisions_last_with_events(&hooks, &log);
        } else {
            world.step_with_events(&hooks, &log);
        }
        steps += 1;
        let (on1, on2) = (touching(&world, tile1), touching(&world, tile2));
        if on1 || on2 {
            assert!(
                on1 && on2,
                "{what}: the box's two contacts did not start in the same step"
            );
            break;
        }
        assert!(steps < 500, "{what}: the box never landed");
    }

    let snapshot = bincode::serialize(&world).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&snapshot).unwrap();

    // Changes only the event flags: the collider is marked modified, with no other change.
    let events = ActiveEvents::COLLISION_EVENTS | ActiveEvents::CONTACT_FORCE_EVENTS;
    world.colliders[box_co].set_active_events(events);
    restored.colliders[box_co].set_active_events(events);

    let force_events = assert_restored_twin_matches(
        &mut world,
        &mut restored,
        collisions_last,
        30,
        &alloc::format!("{what} (saved after step {steps})"),
    );
    assert!(
        force_events > 0,
        "{what}: the scene emitted no contact-force events"
    );
}

#[cfg(feature = "serde-serialize")]
#[test]
fn step_collisions_last_snapshot_restore_keeps_force_event_order() {
    check_restore_keeps_force_event_order(true);
    check_restore_keeps_force_event_order(false);
}

/// `initialize_collisions_last` on a world first stepped with `step`, whose solver contact graph
/// is therefore already valid. The initialization ends with a detection, which here starts a
/// contact, so it leaves a pending solver-graph maintenance behind unless it ends the way a
/// collisions-last step does. A snapshot taken right after the initialization must restore into a
/// world that steps on bit-identically, with the same events.
#[cfg(feature = "serde-serialize")]
#[test]
fn initialize_collisions_last_after_stock_steps_snapshot_roundtrip() {
    let events = ActiveEvents::COLLISION_EVENTS | ActiveEvents::CONTACT_FORCE_EVENTS;
    let make = || {
        let (mut world, _, ball_co, floor_co) = ball_on_floor(0.0);
        world.colliders[ball_co].set_active_events(events);
        world.colliders[floor_co].set_active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
        (world, ball_co, floor_co)
    };
    let touching = |world: &PhysicsWorld, ball_co: ColliderHandle, floor_co: ColliderHandle| {
        world
            .contact_pair(ball_co, floor_co)
            .is_some_and(|pair| pair.has_any_active_contact())
    };
    let hooks = FrictionHook::default();

    // The standard step whose detection (at the start of the step) starts the contact.
    let (mut probe, ball_co, floor_co) = make();
    let mut contact_step = 0;
    while !touching(&probe, ball_co, floor_co) {
        probe.step_with_events(&hooks, &());
        contact_step += 1;
        assert!(contact_step < 200, "the ball never touched the floor");
    }

    // One standard step fewer: the detection of the initialization starts the contact instead.
    let (mut world, ..) = make();
    for _ in 1..contact_step {
        world.step_with_events(&hooks, &());
    }
    assert!(!touching(&world, ball_co, floor_co));
    world.initialize_collisions_last_with_events(&hooks, &());
    assert!(
        touching(&world, ball_co, floor_co),
        "the initialization's detection should start the contact"
    );

    let snapshot = bincode::serialize(&world).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&snapshot).unwrap();

    let force_events = assert_restored_twin_matches(
        &mut world,
        &mut restored,
        true,
        60,
        &alloc::format!("initialization after {} standard steps", contact_step - 1),
    );
    assert!(
        force_events > 0,
        "the scene emitted no contact-force events"
    );
}

// With nothing changed between steps, both orders are the same computation observed at a
// different point: the bodies move bit-identically.

/// Steps twin worlds built by `scene`, one with `step` and one with `step_collisions_last`, and
/// asserts after every step that every body has bit-identical pose and velocities, that the
/// collision events standard stepping emits during step `k` were emitted by collisions-last one
/// detection earlier (by its initialization for `k = 0`), and that the contact-force events of
/// step `k` are the same. `observe` sees the standard world after every step. Returns the number
/// of events compared.
fn assert_twin_bit_identical(
    name: &str,
    scene: impl Fn() -> PhysicsWorld,
    steps: usize,
    mut observe: impl FnMut(usize, &PhysicsWorld),
) -> usize {
    let mut standard = scene();
    let mut last = scene();
    let (standard_log, last_log) = (EventOrderLog::default(), EventOrderLog::default());
    last.initialize_collisions_last_with_events(&(), &last_log);

    let split = |events: Vec<(u8, u32, u32, u64)>| {
        let (mut collision, mut force): (Vec<_>, Vec<_>) =
            events.into_iter().partition(|event| event.0 < 2);
        collision.sort_unstable();
        force.sort_unstable();
        (collision, force)
    };
    let (mut last_collision, _) = split(last_log.take());
    let mut compared = 0;

    for step in 0..steps {
        standard.step_with_events(&(), &standard_log);
        last.step_collisions_last_with_events(&(), &last_log);

        let (standard_collision, standard_force) = split(standard_log.take());
        let (next_last_collision, last_force) = split(last_log.take());
        assert_eq!(
            last_collision, standard_collision,
            "{name}: the collision events standard stepping emitted during step {step} (right) \
             differ from those collisions-last emitted one detection earlier (left)"
        );
        assert_eq!(
            last_force, standard_force,
            "{name}: the contact-force events of step {step} differ between collisions-last \
             (left) and standard stepping (right)"
        );
        compared += standard_collision.len() + standard_force.len();
        last_collision = next_last_collision;

        for (handle, rb) in standard.rigid_bodies() {
            let lb = &last.bodies[handle];
            assert!(
                rb.position().is_finite() && rb.vels().is_finite(),
                "{name}: body {handle:?} went non-finite at step {step}"
            );
            assert!(
                rb.translation() == lb.translation()
                    && rb.rotation() == lb.rotation()
                    && rb.linvel() == lb.linvel()
                    && rb.angvel() == lb.angvel(),
                "{name}: body {handle:?} differs between the two orders at step {step}: standard \
                 {:?} {:?} {:?} {:?}, collisions-last {:?} {:?} {:?} {:?}",
                rb.translation(),
                rb.rotation(),
                rb.linvel(),
                rb.angvel(),
                lb.translation(),
                lb.rotation(),
                lb.linvel(),
                lb.angvel(),
            );
        }
        observe(step, &standard);
    }
    compared
}

/// Boxes settling on a floor (they fall asleep), a revolute chain, a sensor, a CCD ball, and a
/// ball dropped from high enough to land on the pile after it fell asleep: a contact-driven
/// wake-up with nothing changed between steps.
fn pile_scene() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    let events = ActiveEvents::COLLISION_EVENTS | ActiveEvents::CONTACT_FORCE_EVENTS;
    world.insert(
        RigidBodyBuilder::fixed().translation(v(0.0, -0.5)),
        cuboid(20.0, 0.5),
    );
    for i in 0..8 {
        for j in 0..3 {
            let jitter = (i as Real * 0.013 + j as Real * 0.017) % 0.05;
            world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(v(i as Real * 1.05 - 4.0 + jitter, j as Real * 1.05 + 0.55)),
                cuboid(0.5, 0.5)
                    .active_events(events)
                    .contact_force_event_threshold(0.0),
            );
        }
    }
    let mut prev = world.insert_body(RigidBodyBuilder::fixed().translation(v(0.0, 7.0)));
    for i in 0..4 {
        let rb = world.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(v(0.6 * (i + 1) as Real, 7.0))
                .can_sleep(false),
        );
        world.insert_collider(ColliderBuilder::ball(0.25), Some(rb));
        world.insert_impulse_joint(
            prev,
            rb,
            revolute()
                .local_anchor1(v(0.3, 0.0))
                .local_anchor2(v(-0.3, 0.0)),
        );
        prev = rb;
    }
    world.insert_collider(
        cuboid(3.0, 0.5)
            .translation(v(0.0, 4.0))
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS),
        None,
    );
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(v(-8.0, 2.0))
            .linvel(v(90.0, -20.0))
            .ccd_enabled(true)
            .can_sleep(false),
        ColliderBuilder::ball(0.15),
    );
    // Lands on the pile after about 3 s, once it fell asleep.
    world.insert(
        RigidBodyBuilder::dynamic().translation(v(0.0, 45.0)),
        ColliderBuilder::ball(0.4).active_events(events),
    );
    world
}

#[test]
fn both_orders_move_bodies_identically_pile() {
    let (mut slept, mut woke) = (false, false);
    let compared = assert_twin_bit_identical("pile", pile_scene, 400, |_, world| {
        let sleeping = world
            .rigid_bodies()
            .filter(|(_, rb)| rb.is_sleeping())
            .count();
        if sleeping > 0 {
            slept = true;
        } else if slept {
            woke = true;
        }
    });
    assert!(compared > 0, "the pile emitted no events");
    assert!(
        slept && woke,
        "the pile should fall asleep and be woken by the falling ball (slept: {slept}, woke: \
         {woke})"
    );
}

/// Many bouncing balls in a box in zero gravity: chaotic, so any difference between the two
/// orders would grow quickly. With nothing changed between steps there must be none.
fn balls_scene() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let wall = |world: &mut PhysicsWorld, pos: Vector, hx: Real, hy: Real| {
        world.insert_collider(
            cuboid(hx, hy)
                .translation(pos)
                .restitution(1.0)
                .friction(0.0),
            None,
        );
    };
    wall(&mut world, v(0.0, -10.5), 10.5, 0.5);
    wall(&mut world, v(0.0, 10.5), 10.5, 0.5);
    wall(&mut world, v(-10.5, 0.0), 0.5, 10.0);
    wall(&mut world, v(10.5, 0.0), 0.5, 10.0);
    for i in 0..12 {
        for j in 0..10 {
            let pos = v(i as Real * 1.6 - 8.8, j as Real * 1.8 - 8.1);
            let vel = v(
                ((i * 7 + j * 3) % 11) as Real - 5.0,
                ((i * 5 + j * 11) % 13) as Real - 6.0,
            ) * 1.5;
            world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(pos)
                    .linvel(vel)
                    .can_sleep(false),
                ColliderBuilder::ball(0.4)
                    .restitution(1.0)
                    .friction(0.0)
                    .active_events(ActiveEvents::COLLISION_EVENTS),
            );
        }
    }
    world
}

#[test]
fn both_orders_move_bodies_identically_chaotic_balls() {
    let compared = assert_twin_bit_identical("balls", balls_scene, 600, |_, _| {});
    assert!(
        compared > 100,
        "the balls emitted too few events: {compared}"
    );
}

// The narrow-phase describes the current poses between steps.

/// After every collisions-last step, the deepest contact of a ball bouncing on the floor is as
/// deep as the ball's current position says. With standard stepping the same contact describes the
/// poses the step started from, so while the ball moves it is off by about the distance it
/// travelled during the step. Contact recycling is off: a recycled pair keeps its contact data
/// until the pair moves by more than the recycle distance, in both orders.
#[test]
fn step_collisions_last_contacts_match_current_poses() {
    let probe = |collisions_last: bool| -> (Real, Real, usize) {
        let (mut world, ball, ball_co, floor_co) = ball_on_floor(0.3);
        world.integration_parameters.contact_recycling = false;
        if collisions_last {
            world.initialize_collisions_last();
        }
        let (mut max_err, mut max_moving_err, mut samples) = (0.0 as Real, 0.0 as Real, 0);
        for _ in 0..240 {
            if collisions_last {
                world.step_collisions_last();
            } else {
                world.step();
            }
            let Some(pair) = world.contact_pair(ball_co, floor_co) else {
                continue;
            };
            let Some((_, contact)) = pair.find_deepest_contact() else {
                continue;
            };
            // Floor top at 0.1, ball radius 0.5.
            let analytic = world.bodies[ball].translation().y - 0.6;
            let err = (contact.dist - analytic).abs();
            max_err = max_err.max(err);
            if world.bodies[ball].linvel().y.abs() > 0.5 {
                max_moving_err = max_moving_err.max(err);
            }
            samples += 1;
        }
        (max_err, max_moving_err, samples)
    };

    let (last_err, _, samples) = probe(true);
    assert!(samples > 10, "the ball's contact was never observed");
    assert!(
        last_err < 1.0e-4,
        "with collisions-last the contact distance is off by {last_err} m from the current poses"
    );
    let (_, standard_moving_err, _) = probe(false);
    assert!(
        standard_moving_err > 1.0e-3,
        "with standard stepping the contact distance should lag the moving ball by about one step \
         of motion, but the largest error was {standard_moving_err} m"
    );
}
