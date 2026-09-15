//! Contact adhesion together with `step_collisions_last` (2D).
//!
//! bc's game requests adhesion (plus a drive `tangent_velocity` and a per-manifold friction) in
//! `modify_solver_contacts`, steps with `step_collisions_last`, and restores bincode snapshots
//! for rollback netplay, comparing checksums across clients. These tests check that combination:
//! restores stay byte-identical, a woken adhered body stays attached, a removed hook flag stops
//! the pull, and a catch-up detection doesn't apply the adhesion twice.

use std::sync::atomic::{AtomicUsize, Ordering};

use rapier2d::prelude::*;

const G: Real = 9.81;

/// How a test advances its world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `PhysicsWorld::step_with_events`.
    Stock,
    /// `PhysicsWorld::step_collisions_last_with_events`, initialized before the first step.
    CollisionsLast,
}

/// Advances `world` by one step in `mode`.
fn step(mode: Mode, world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks) {
    match mode {
        Mode::Stock => world.step_with_events(hooks, &()),
        Mode::CollisionsLast => {
            if !world.physics_pipeline.collisions_last_initialized() {
                world.initialize_collisions_last_with_events(hooks, &());
            }
            world.step_collisions_last_with_events(hooks, &());
        }
    }
}

fn run(mode: Mode, world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks, steps: usize) {
    for _ in 0..steps {
        step(mode, world, hooks);
    }
}

fn new_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -G);
    world
}

fn assert_close(got: Real, expected: Real, epsilon: Real, what: &str) {
    assert!(
        (got - expected).abs() <= epsilon,
        "{what}: got {got}, expected {expected} ± {epsilon}"
    );
}

/// Bit patterns of the pose and velocity of `handles`, in order, for bit-identity comparisons.
fn bits_of(world: &PhysicsWorld, handles: &[RigidBodyHandle]) -> Vec<u32> {
    let mut bits = Vec::new();
    for h in handles {
        let rb = &world.bodies[*h];
        let t = rb.translation();
        let v = rb.linvel();
        for x in [t.x, t.y, rb.rotation().angle(), v.x, v.y, rb.angvel()] {
            bits.push(x.to_bits());
        }
    }
    bits
}

/// Requests a fixed adhesion force on every manifold involving one of its colliders, and counts
/// its calls.
struct AdhesionHook {
    requests: Vec<(ColliderHandle, Real)>,
    calls: AtomicUsize,
}

impl AdhesionHook {
    fn new(requests: Vec<(ColliderHandle, Real)>) -> Self {
        Self {
            requests,
            calls: AtomicUsize::new(0),
        }
    }

    /// The number of calls since the last call to this method.
    fn take_calls(&self) -> usize {
        self.calls.swap(0, Ordering::Relaxed)
    }
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        for (collider, force) in &self.requests {
            if context.collider1 == *collider || context.collider2 == *collider {
                *context.adhesion_force = *force;
            }
        }
    }
}

/// A fixed ceiling whose bottom face is at y = 0 (contact modification enabled) and a 1x1 dynamic
/// box (mass 1) hanging under it. Returns `(ceiling collider, box body)`.
fn ceiling_and_hanging_box(
    world: &mut PhysicsWorld,
    can_sleep: bool,
) -> (ColliderHandle, RigidBodyHandle) {
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let ceiling = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );
    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, -0.5))
            .can_sleep(can_sleep),
        ColliderBuilder::cuboid(0.5, 0.5),
    );
    (ceiling, box_body)
}

/*
 * Snapshot restores with both features.
 */

#[cfg(feature = "serde-serialize")]
mod snapshot {
    use super::*;
    use std::sync::Mutex;

    /// Drives capsules the way bc's game drives its characters. Every manifold involving a capsule
    /// gets a per-manifold friction, a conveyor `tangent_velocity` along the contact tangent, and
    /// an adhesion request: an `adhesion_force` for the even-indexed capsules, an
    /// `adhesion_budget` for the odd-indexed ones. Counts its calls.
    pub(super) struct GameHook {
        pub(super) capsules: Vec<ColliderHandle>,
        pub(super) calls: AtomicUsize,
    }

    impl GameHook {
        pub(super) fn new(capsules: Vec<ColliderHandle>) -> Self {
            Self {
                capsules,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl PhysicsHooks for GameHook {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let find = |handle| self.capsules.iter().position(|c| *c == handle);
            // `tangent_velocity` is the target velocity of collider2 relative to collider1.
            let (index, capsule, sign) = match (find(context.collider1), find(context.collider2)) {
                (_, Some(i)) => (i, context.collider2, 1.0),
                (Some(i), None) => (i, context.collider1, -1.0),
                (None, None) => return,
            };
            *context.friction = 0.9;
            let n = *context.normal;
            let tangent = Vector::new(-n.y, n.x);
            for contact in context.solver_contacts.iter_mut() {
                contact.tangent_velocity = tangent * (sign * 1.2);
            }
            if index % 2 == 0 {
                *context.adhesion_force = 18.0;
            } else {
                *context.adhesion_budget = Some(AdhesionBudget {
                    owner: capsule,
                    channel: 0,
                    total: 18.0,
                });
            }
        }
    }

    /// Collision and contact-force events, as `(kind, collider1 index, collider2 index, force
    /// magnitude bits)` with kind 0 = started, 1 = stopped, 2 = contact force.
    #[derive(Default)]
    pub(super) struct EventLog(Mutex<Vec<(u8, u32, u32, u32)>>);

    impl EventLog {
        /// The events recorded since the last call, sorted (the `parallel` feature may report the
        /// events of one step in any order).
        pub(super) fn take_sorted(&self) -> Vec<(u8, u32, u32, u32)> {
            let mut events = core::mem::take(&mut *self.0.lock().unwrap());
            events.sort_unstable();
            events
        }
    }

    impl EventHandler for EventLog {
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
                total_force_magnitude.to_bits(),
            ));
        }
    }

    /// A game-shaped scene: capsules walking on a floor, on a slope of abutting tiles and against
    /// a wall, next to plain boxes. Returns the world and the capsule colliders and bodies.
    pub(super) fn game_scene() -> (PhysicsWorld, Vec<ColliderHandle>, Vec<RigidBodyHandle>) {
        let mut world = new_world();
        let events = ActiveEvents::COLLISION_EVENTS | ActiveEvents::CONTACT_FORCE_EVENTS;

        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5)),
            ColliderBuilder::cuboid(30.0, 0.5),
        );
        // A 35° slope of 8 abutting 1 m tiles, rising to the right from (2, 0).
        let theta = (35.0 as Real).to_radians();
        let up_slope = Vector::new(theta.cos(), theta.sin());
        let top_normal = Vector::new(-theta.sin(), theta.cos());
        let statics = world.insert_body(RigidBodyBuilder::fixed());
        let origin = Vector::new(2.0, 0.0);
        for k in 0..8 {
            world.insert_collider(
                ColliderBuilder::cuboid(0.5, 0.25)
                    .translation(origin + up_slope * (0.5 + k as Real) - top_normal * 0.25)
                    .rotation(theta)
                    .friction(0.6),
                Some(statics),
            );
        }
        // A wall on the left.
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(-10.0, 4.0)),
            ColliderBuilder::cuboid(0.5, 4.5),
        );

        let mut capsules = Vec::new();
        let mut capsule_bodies = Vec::new();
        let mut add_capsule = |world: &mut PhysicsWorld, pos: Vector, rot: Real| {
            let (body, co) = world.insert(
                RigidBodyBuilder::dynamic().translation(pos).rotation(rot),
                ColliderBuilder::capsule_y(0.3, 0.25)
                    .friction(0.8)
                    .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS)
                    .active_events(events)
                    .contact_force_event_threshold(5.0),
            );
            capsules.push(co);
            capsule_bodies.push(body);
        };
        // Two on the slope, one against the wall, three on the floor.
        add_capsule(
            &mut world,
            origin + up_slope * 2.5 + top_normal * 0.6,
            theta,
        );
        add_capsule(
            &mut world,
            origin + up_slope * 5.5 + top_normal * 0.6,
            theta,
        );
        add_capsule(&mut world, Vector::new(-9.25, 0.6), 0.0);
        for i in 0..3 {
            add_capsule(&mut world, Vector::new(-6.0 + 2.5 * i as Real, 0.6), 0.0);
        }

        for i in 0..4 {
            world.insert(
                RigidBodyBuilder::dynamic().translation(Vector::new(10.0 + 1.5 * i as Real, 0.5)),
                ColliderBuilder::cuboid(0.4, 0.4)
                    .active_events(events)
                    .contact_force_event_threshold(0.0),
            );
        }

        (world, capsules, capsule_bodies)
    }

    /// A world restored from a snapshot of the uninterrupted world, stepped in lockstep with it.
    pub(super) struct Twin {
        pub(super) world: PhysicsWorld,
        pub(super) hook: GameHook,
        pub(super) log: EventLog,
        pub(super) saved_after: usize,
        pub(super) flag_set: bool,
        /// Steps after the restore at which the bytes and the hook calls first differed from the
        /// uninterrupted world.
        pub(super) first_byte_divergence: Option<usize>,
        pub(super) first_call_divergence: Option<usize>,
    }
}

/// Saves the game-shaped scene after steps 5, 30, 90 and 200, and restores each snapshot twice
/// into a fresh world with a fresh pipeline: once with `set_collisions_last_initialized(true)`,
/// once without. Each restored world steps in lockstep with the uninterrupted one for 60 steps,
/// with the same between-step edits: a capsule jump (a deferred change) at step 40, a ball dropped
/// in at step 150 (an insertion, so collisions-last catches up) and a box removal at step 220.
///
/// With the flag, every step must emit the same (sorted) events, call the hook as often and
/// serialize to the same bytes as the uninterrupted world: 240 compared steps, 3,920,609 bytes in
/// all when this test was written. Without the flag, the first step after the restore runs the
/// initial detection again, so the hook runs more often on that step and the restored world no
/// longer matches the uninterrupted one; that is why the flag exists. When this test was written,
/// all four unflagged restores differed in bytes, events and hook calls from the first step on.
#[cfg(feature = "serde-serialize")]
#[test]
fn collisions_last_snapshot_restore_with_adhesion_and_drive_is_byte_identical() {
    use snapshot::*;

    const SAVE_POINTS: [usize; 4] = [5, 30, 90, 200];
    const HORIZON: usize = 60;
    const JUMP_STEP: usize = 40;
    const BALL_STEP: usize = 150;
    const REMOVE_STEP: usize = 220;

    let (mut world, capsules, capsule_bodies) = game_scene();
    let hook = GameHook::new(capsules.clone());
    let log = EventLog::default();
    world.initialize_collisions_last_with_events(&hook, &log);
    log.take_sorted();
    let plain_box = world
        .bodies
        .iter()
        .map(|(h, _)| h)
        .filter(|h| !capsule_bodies.contains(h) && world.bodies[*h].is_dynamic())
        .max_by_key(|h| h.into_raw_parts())
        .unwrap();

    // Returns the inserted ball's handles, so the twins can check they allocate the same ones.
    let edit = |world: &mut PhysicsWorld, step: usize| {
        match step {
            JUMP_STEP => world.bodies[capsule_bodies[3]].set_linvel(Vector::new(1.0, 4.0), true),
            BALL_STEP => {
                return Some(world.insert(
                    RigidBodyBuilder::dynamic().translation(Vector::new(-4.0, 3.0)),
                    ColliderBuilder::ball(0.3),
                ));
            }
            REMOVE_STEP => {
                world.remove_body(plain_box);
            }
            _ => {}
        }
        None
    };

    let mut twins: Vec<Twin> = Vec::new();
    let (mut compared_steps, mut compared_bytes) = (0, 0);
    let (mut hook_calls, mut force_events) = (0, 0);
    for step in 0..SAVE_POINTS[3] + HORIZON {
        if SAVE_POINTS.contains(&step) {
            let snapshot = bincode::serialize(&world).unwrap();
            for flag_set in [true, false] {
                let mut restored: PhysicsWorld = bincode::deserialize(&snapshot).unwrap();
                assert!(
                    !restored.physics_pipeline.collisions_last_initialized(),
                    "the pipeline is not part of the snapshot"
                );
                if flag_set {
                    restored
                        .physics_pipeline
                        .set_collisions_last_initialized(true);
                }
                twins.push(Twin {
                    world: restored,
                    hook: GameHook::new(capsules.clone()),
                    log: EventLog::default(),
                    saved_after: step,
                    flag_set,
                    first_byte_divergence: None,
                    first_call_divergence: None,
                });
            }
        }

        let inserted = edit(&mut world, step);
        world.step_collisions_last_with_events(&hook, &log);
        let events = log.take_sorted();
        let calls = hook.calls.swap(0, Ordering::Relaxed);
        hook_calls += calls;
        force_events += events.iter().filter(|event| event.0 == 2).count();
        let bytes = bincode::serialize(&world).unwrap();

        for twin in twins
            .iter_mut()
            .filter(|twin| step < twin.saved_after + HORIZON)
        {
            let after = step + 1 - twin.saved_after;
            let twin_inserted = edit(&mut twin.world, step);
            assert_eq!(twin_inserted, inserted, "the twin allocated other handles");
            twin.world
                .step_collisions_last_with_events(&twin.hook, &twin.log);
            let twin_events = twin.log.take_sorted();
            let twin_calls = twin.hook.calls.swap(0, Ordering::Relaxed);
            let twin_bytes = bincode::serialize(&twin.world).unwrap();

            if twin.flag_set {
                let saved_after = twin.saved_after;
                assert_eq!(
                    twin_events, events,
                    "restored after step {saved_after}: the events differ {after} step(s) after \
                     the restore"
                );
                assert_eq!(
                    twin_calls, calls,
                    "restored after step {saved_after}: the hook ran a different number of times \
                     {after} step(s) after the restore"
                );
                assert!(
                    twin_bytes == bytes,
                    "restored after step {saved_after}: the bytes differ {after} step(s) after the \
                     restore"
                );
                compared_steps += 1;
                compared_bytes += bytes.len();
            } else {
                if twin.first_byte_divergence.is_none() && twin_bytes != bytes {
                    twin.first_byte_divergence = Some(after);
                }
                if twin.first_call_divergence.is_none() && twin_calls != calls {
                    twin.first_call_divergence = Some(after);
                }
            }
        }
    }

    assert_eq!(compared_steps, SAVE_POINTS.len() * HORIZON);
    assert!(
        compared_bytes > 0 && hook_calls > 0 && force_events > 0,
        "the scene should run the hook and emit contact-force events (hook calls {hook_calls}, \
         force events {force_events})"
    );
    for twin in twins.iter().filter(|twin| !twin.flag_set) {
        let saved_after = twin.saved_after;
        assert_eq!(
            twin.first_call_divergence,
            Some(1),
            "restored after step {saved_after} without the flag: the first step should run the \
             initial detection again, so call the hook more often"
        );
        assert!(
            twin.first_byte_divergence.is_some(),
            "restored after step {saved_after} without the flag: running the initial detection \
             again no longer changes the simulation, so the flag may no longer be needed"
        );
    }
}

/*
 * A woken adhered box.
 */

/// Hangs a box under an adhesive ceiling until it has slept for 30 steps, then throws a ball
/// (without gravity) at its side, and returns the largest distance between the box's height and
/// its sleeping height over the next 60 steps.
fn drift_after_side_hit(mode: Mode) -> Real {
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world, true);
    let hook = AdhesionHook::new(vec![(ceiling, 30.0)]);

    let mut slept = false;
    for _ in 0..600 {
        step(mode, &mut world, &hook);
        if world.bodies[box_body].is_sleeping() {
            slept = true;
            break;
        }
    }
    assert!(slept, "the adhered box never fell asleep ({mode:?})");
    run(mode, &mut world, &hook, 30);
    assert!(world.bodies[box_body].is_sleeping());
    let y_asleep = world.bodies[box_body].translation().y;

    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-3.0, -0.5))
            .linvel(Vector::new(8.0, 0.0))
            .gravity_scale(0.0),
        ColliderBuilder::ball(0.25),
    );
    let mut woke = false;
    let mut max_drift: Real = 0.0;
    for _ in 0..60 {
        step(mode, &mut world, &hook);
        woke |= !world.bodies[box_body].is_sleeping();
        max_drift = max_drift.max((world.bodies[box_body].translation().y - y_asleep).abs());
    }
    assert!(woke, "the thrown ball never woke the box ({mode:?})");
    max_drift
}

/// A box held under a ceiling by adhesion falls asleep, then a ball thrown at its side wakes it.
/// The ball's contact starts during a detection, which wakes the box while its ceiling pair is
/// still count-cleared from sleep. Stock `step` solves that step without the pair (upstream
/// behavior), so neither the ceiling nor the adhesion holds the box for one step and it drops.
/// Collisions-last puts the pair back into the solve first (`requalify_woken_pair_hints`), so the
/// box must stay where it slept.
///
/// Measured in every debug build of the quick gates (default, enhanced-determinism + serde,
/// parallel + serde): stock stepping 1.66 mm (1.6625e-3 m), collisions-last 3.0e-8 m.
#[test]
fn collisions_last_woken_adhered_box_stays_attached() {
    let stock = drift_after_side_hit(Mode::Stock);
    let collisions_last = drift_after_side_hit(Mode::CollisionsLast);
    assert!(
        collisions_last < 1.0e-4,
        "collisions-last: the woken box moved {collisions_last} m from where it slept (stock \
         stepping: {stock} m)"
    );
    assert!(
        collisions_last <= stock,
        "collisions-last moved the woken box further than stock stepping ({collisions_last} m, \
         stock {stock} m)"
    );
}

/*
 * A hook flag removed from an awake resting pair.
 */

/// A box resting on a hooked floor, never sleeping, with a net upward load of 5 N held down by
/// 30 N of adhesion. After 120 steps the floor loses its hook flag, optionally together with the
/// insertion of a far collider, which makes collisions-last catch up. Returns the box's vertical
/// velocity after each of the next 3 steps, and whether the first of them cleared the stored
/// request.
fn velocities_after_hook_flag_removal(mode: Mode, catch_up: bool) -> ([Real; 3], bool) {
    let mut world = new_world();
    assert!(world.integration_parameters.contact_recycling);
    let floor_body = world.insert_body(RigidBodyBuilder::fixed());
    let floor = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, -0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(floor_body),
    );
    let (box_body, box_co) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.5))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.5, 0.5),
    );
    let w = world.bodies[box_body].mass() * G;
    world.bodies[box_body].add_force(Vector::new(0.0, w + 5.0), true);
    let hook = AdhesionHook::new(vec![(floor, 30.0)]);
    run(mode, &mut world, &hook, 120);
    assert_close(
        world.bodies[box_body].translation().y,
        0.5,
        0.05,
        "adhered box resting on the floor",
    );

    world.colliders[floor].set_active_hooks(ActiveHooks::empty());
    if catch_up {
        world.insert_collider(
            ColliderBuilder::cuboid(0.5, 0.5).translation(Vector::new(100.0, 100.0)),
            None,
        );
    }
    let mut velocities = [0.0; 3];
    let mut cleared = false;
    for (i, velocity) in velocities.iter_mut().enumerate() {
        step(mode, &mut world, &hook);
        *velocity = world.bodies[box_body].linvel().y;
        if i == 0 {
            let pair = world.narrow_phase.contact_pair(floor, box_co).unwrap();
            cleared = pair.manifolds.iter().all(|m| m.data.adhesion_force == 0.0);
        }
    }
    (velocities, cleared)
}

/// The awake case of hook-flag removal (the sleeping case is
/// `collisions_last_wake_applies_no_adhesion_after_hook_flag_removal` in `adhesion.rs`): a resting
/// pair loses the flag without being woken. Without the pull, the box accelerates up at 5 m/s².
///
/// - Stock `step` updates the pair before the solve: no pull from the first step on.
/// - Collisions-last without other changes solves that step with the contacts of the previous
///   detection, flags included (documented as one step late), so that one solve still pulls. Its
///   end-of-step detection must then fully update the pair instead of recycling it, which clears
///   the request (the recycle poisoning in the narrow phase), so the pull stops within one step.
/// - Collisions-last with a catch-up detection on the same step updates the pair before the
///   solve, like stock stepping.
///
/// Measured: stock vy = [0.0833, 0.1667, 0.2500] m/s; collisions-last [7.5e-9, 0.0833, 0.1667];
/// collisions-last with the catch-up identical to stock.
#[test]
fn collisions_last_pull_stops_when_hook_flag_is_removed_from_an_awake_resting_pair() {
    let dt = new_world().integration_parameters.dt;
    let (stock, stock_cleared) = velocities_after_hook_flag_removal(Mode::Stock, false);
    let (deferred, deferred_cleared) =
        velocities_after_hook_flag_removal(Mode::CollisionsLast, false);
    let (caught_up, caught_up_cleared) =
        velocities_after_hook_flag_removal(Mode::CollisionsLast, true);

    assert!(
        stock_cleared && deferred_cleared && caught_up_cleared,
        "the first step after the flag removal must clear the stored request (stock \
         {stock_cleared}, collisions-last {deferred_cleared}, with catch-up {caught_up_cleared})"
    );
    assert_close(
        stock[0],
        5.0 * dt,
        1.0e-4,
        "stock: vy one step after the removal",
    );
    assert!(
        deferred[0].abs() < 1.0e-4,
        "collisions-last: the solve of the removal step should still pull (vy = {})",
        deferred[0]
    );
    for i in 0..2 {
        assert_close(
            deferred[i + 1],
            stock[i],
            1.0e-5,
            &format!(
                "collisions-last: vy {} step(s) after the removal, against stock one step earlier",
                i + 2
            ),
        );
    }
    for i in 0..3 {
        assert_close(
            caught_up[i],
            stock[i],
            1.0e-5,
            &format!(
                "collisions-last with a catch-up: vy {} step(s) after the removal, against stock",
                i + 1
            ),
        );
    }
}

/*
 * A catch-up detection with adhesion.
 */

/// Inserting a collider makes collisions-last catch up: the step runs its collision detection (and
/// the hook) twice, before and after the solve. The adhesion requested by both runs must still be
/// applied once, by the one solve. Twin worlds, one of them with a far collider inserted while a
/// box hangs under an adhesive ceiling and another box slides down an adhesive low-friction wall,
/// must keep both boxes bit-identical. The slider shows a double pull directly: its friction is
/// `mu * adhesion`, so its vertical velocity changes by `(mu * F - g) dt` per step with the pull
/// applied once (-0.1302 m/s) and `(2 mu * F - g) dt` with it applied twice (-0.0968 m/s).
#[test]
fn collisions_last_catch_up_applies_adhesion_once() {
    const INSERT_STEP: usize = 40;
    const MU: Real = 0.1;
    const WALL_ADHESION: Real = 20.0;
    let make = || {
        let mut world = new_world();
        let (ceiling, hanging) = ceiling_and_hanging_box(&mut world, false);
        let wall_body = world.insert_body(RigidBodyBuilder::fixed());
        let wall = world.insert_collider(
            ColliderBuilder::cuboid(0.5, 20.0)
                .translation(Vector::new(20.5, 0.0))
                .friction(MU)
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(wall_body),
        );
        let (slider, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(19.5, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5).friction(MU),
        );
        let hook = AdhesionHook::new(vec![(ceiling, 30.0), (wall, WALL_ADHESION)]);
        (world, hook, hanging, slider)
    };

    let (mut edited, edited_hook, hanging, slider) = make();
    let (mut twin, twin_hook, ..) = make();
    let dt = edited.integration_parameters.dt;
    for step_index in 0..INSERT_STEP + 30 {
        let slider_vy = edited.bodies[slider].linvel().y;
        if step_index == INSERT_STEP {
            edited.insert_collider(
                ColliderBuilder::cuboid(0.5, 0.5).translation(Vector::new(100.0, 100.0)),
                None,
            );
        }
        step(Mode::CollisionsLast, &mut edited, &edited_hook);
        step(Mode::CollisionsLast, &mut twin, &twin_hook);

        let (edited_calls, twin_calls) = (edited_hook.take_calls(), twin_hook.take_calls());
        if step_index == INSERT_STEP {
            assert!(
                twin_calls > 0 && edited_calls == 2 * twin_calls,
                "the insertion step should run the hook for both detections (edited world \
                 {edited_calls} calls, twin {twin_calls})"
            );
            let slider_dvy = edited.bodies[slider].linvel().y - slider_vy;
            assert_close(
                slider_dvy,
                (MU * WALL_ADHESION - G) * dt,
                1.0e-4,
                "slider vertical velocity change on the catch-up step (a pull applied once)",
            );
        }
        assert!(
            bits_of(&edited, &[hanging, slider]) == bits_of(&twin, &[hanging, slider]),
            "the catch-up detection changed the adhered boxes at step {step_index} (collider \
             inserted before step {INSERT_STEP})"
        );
    }
}
