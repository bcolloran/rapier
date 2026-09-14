//! Fork-owned determinism goldens for bc's rapier fork.
//!
//! Upstream's `snapshot_portability` goldens hash the whole serialized world, so they must be
//! re-minted whenever the fork adds serialized state (for example the adhesion fields on
//! `ContactManifoldData`). These goldens hash only *simulation results*: every serialized
//! container except the narrow-phase, plus the sorted per-step event stream. They pin that the
//! fork's additions leave stock `PhysicsPipeline::step` bit-identical.
//!
//! Never re-mint these to make a fork change pass. If one moves, the fork changed the behavior
//! of stock `step`, which is a bug in the fork change.
//!
//! Run it the same way as the upstream goldens, natively and under wasm32:
//!
//! ```text
//! cargo test -p rapier2d --release --features enhanced-determinism,serde-serialize \
//!     --test fork_goldens
//! ```
#![cfg(all(feature = "serde-serialize", feature = "enhanced-determinism"))]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rapier2d::prelude::*;

/// FNV-1a digest of the scene state after its steps, per scene.
const MIXED_GOLDEN: u64 = 0xa9f4_44db_810d_922d;
const HOOKED_GOLDEN: u64 = 0x81bf_32df_72af_2aa3;

/// FNV-1a digests of the same scenes stepped with collisions-last
/// (`initialize_collisions_last_with_events`, then `step_collisions_last_with_events`), minted
/// when that mode was ported onto rapier 0.35.3. The digest also covers the initialization's
/// events. Do not re-mint them to make an unrelated change pass.
const MIXED_COLLISIONS_LAST_GOLDEN: u64 = 0x4775_892b_f8fb_3ba5;
const HOOKED_COLLISIONS_LAST_GOLDEN: u64 = 0x7959_d7bb_5c1a_8e78;

const MIXED_STEPS: usize = 60;
const HOOKED_STEPS: usize = 240;

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// Records collision and contact-force events. Events may arrive in any order within a step
/// (the `parallel` feature dispatches narrow-phase work to workers), so each step's events
/// are sorted before they are digested.
#[derive(Default)]
struct EventLog {
    events: Mutex<Vec<(u8, u32, u32, u64, u64)>>,
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
        self.events.lock().unwrap().push((
            kind,
            h1.into_raw_parts().0,
            h2.into_raw_parts().0,
            0,
            0,
        ));
    }

    fn handle_contact_force_event(
        &self,
        dt: Real,
        _: &RigidBodySet,
        _: &ColliderSet,
        contact_pair: &ContactPair,
        total_force_magnitude: Real,
    ) {
        let event = ContactForceEvent::from_contact_pair(dt, contact_pair, total_force_magnitude);
        let force = event.total_force;
        self.events.lock().unwrap().push((
            2 + event.started as u8,
            event.collider1.into_raw_parts().0,
            event.collider2.into_raw_parts().0,
            total_force_magnitude.to_bits() as u64,
            ((force.x.to_bits() as u64) << 32) | force.y.to_bits() as u64,
        ));
    }
}

impl EventLog {
    /// Digests this step's events and tallies them by kind: collision started, collision
    /// stopped, contact force (continuing), contact force (`started`).
    fn digest_step(&self, fnv: &mut Fnv, counts: &mut [usize; 4]) {
        let mut events = core::mem::take(&mut *self.events.lock().unwrap());
        events.sort_unstable();
        fnv.u64(events.len() as u64);
        for (kind, h1, h2, a, b) in events {
            counts[kind as usize] += 1;
            fnv.u64(kind as u64);
            fnv.u64(((h1 as u64) << 32) | h2 as u64);
            fnv.u64(a);
            fnv.u64(b);
        }
    }
}

/// Digest of every serialized container except the narrow-phase (which carries fork-added
/// per-manifold fields); the bodies alone would already reflect any contact-solve change.
fn digest_world(world: &PhysicsWorld, fnv: &mut Fnv) {
    fnv.bytes(&bincode::serialize(&world.bodies).unwrap());
    fnv.bytes(&bincode::serialize(&world.colliders).unwrap());
    fnv.bytes(&bincode::serialize(&world.islands).unwrap());
    fnv.bytes(&bincode::serialize(&world.broad_phase).unwrap());
    fnv.bytes(&bincode::serialize(&world.impulse_joints).unwrap());
    fnv.bytes(&bincode::serialize(&world.multibody_joints).unwrap());
}

/// The same mixed scene as `snapshot_portability.rs`: box contacts (some settling into sleep),
/// an impulse-joint chain, a multibody articulation, a sensor and a CCD body.
fn mixed_scene() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81);
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5)),
        ColliderBuilder::cuboid(20.0, 0.5),
    );

    for i in 0..10 {
        for j in 0..4 {
            let jitter = (i as Real * 0.013 + j as Real * 0.017) % 0.05;
            world.insert(
                RigidBodyBuilder::dynamic().translation(Vector::new(
                    i as Real * 1.05 - 5.0 + jitter,
                    j as Real * 1.05 + 0.55,
                )),
                ColliderBuilder::cuboid(0.5, 0.5).active_events(ActiveEvents::COLLISION_EVENTS),
            );
        }
    }

    let mut prev = world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 7.0)));
    for i in 0..4 {
        let rb = world.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.6 * (i + 1) as Real, 7.0))
                .can_sleep(false),
        );
        world.insert_collider(ColliderBuilder::ball(0.25), Some(rb));
        world.insert_impulse_joint(
            prev,
            rb,
            RevoluteJointBuilder::new()
                .local_anchor1(Vector::X * 0.3)
                .local_anchor2(Vector::X * -0.3),
        );
        prev = rb;
    }

    let size = 0.4;
    let mut last = world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(6.0, 4.0)));
    for i in 0..6 {
        let rb = world.insert_body(RigidBodyBuilder::dynamic().can_sleep(false));
        world.insert_collider(
            ColliderBuilder::cuboid(size / 8.0, size / 2.0).density(1.0),
            Some(rb),
        );
        let joint = RevoluteJointBuilder::new()
            .local_anchor1(Vector::new(0.0, size / 2.0 * (i != 0) as usize as Real))
            .local_anchor2(Vector::new(0.0, -size / 2.0))
            .build()
            .data;
        world.insert_multibody_joint(last, rb, joint);
        last = rb;
    }

    world.insert_collider(
        ColliderBuilder::cuboid(3.0, 0.5)
            .translation(Vector::new(0.0, 4.0))
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS),
        None,
    );
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-8.0, 2.0))
            .linvel(Vector::new(90.0, -20.0))
            .ccd_enabled(true)
            .can_sleep(false),
        ColliderBuilder::ball(0.15),
    );

    world
}

/// Mirrors how the game drives its player through the contact-modification hook: the hook
/// writes a conveyor-belt `tangent_velocity` and an effective per-manifold friction.
struct DriveHook {
    capsules: Vec<ColliderHandle>,
    calls: AtomicUsize,
}

impl PhysicsHooks for DriveHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let sign = if self.capsules.contains(&context.collider2) {
            1.0
        } else {
            -1.0
        };
        *context.friction = 1.2;
        for contact in context.solver_contacts.iter_mut() {
            contact.tangent_velocity.x = sign * 1.5;
        }
    }
}

/// Capsules on a floor, a slope and against a wall, driven by `DriveHook`, next to plain boxes;
/// all report collision and contact-force events.
fn hooked_scene() -> (PhysicsWorld, DriveHook) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -9.81);
    let events = ActiveEvents::COLLISION_EVENTS | ActiveEvents::CONTACT_FORCE_EVENTS;

    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5)),
        ColliderBuilder::cuboid(30.0, 0.5),
    );
    world.insert(
        RigidBodyBuilder::fixed()
            .translation(Vector::new(8.0, 2.0))
            .rotation(0.6),
        ColliderBuilder::cuboid(6.0, 0.25),
    );
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(-12.0, 4.0)),
        ColliderBuilder::cuboid(0.5, 4.5),
    );

    let mut capsules = Vec::new();
    for i in 0..6 {
        let (_, handle) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(-10.0 + 3.5 * i as Real, 1.5 + 0.5 * i as Real)),
            ColliderBuilder::capsule_y(0.3, 0.25)
                .friction(0.8)
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS)
                .active_events(events)
                .contact_force_event_threshold(5.0),
        );
        capsules.push(handle);
    }

    for i in 0..4 {
        world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(-4.0 + 2.0 * i as Real, 3.0)),
            ColliderBuilder::cuboid(0.4, 0.4)
                .active_events(events)
                .contact_force_event_threshold(0.0),
        );
    }

    let hooks = DriveHook {
        capsules,
        calls: AtomicUsize::new(0),
    };
    (world, hooks)
}

fn check(name: &str, got: u64, golden: u64) {
    assert_eq!(
        got, golden,
        "\n{name}: simulation results differ from the fork golden.\n  golden: {golden:#018x}\n  \
         got:    {got:#018x}\nA fork change altered stock `step` behavior. Do not re-mint this \
         golden to make the change pass.\n",
    );
}

// Each test first asserts that its scene exercised what the golden is meant to guard, so a
// scene that silently stops producing contacts cannot keep a golden green.

#[test]
fn mixed_scene_matches_fork_golden() {
    let mut world = mixed_scene();
    let log = EventLog::default();
    let mut fnv = Fnv::new();
    let mut counts = [0; 4];
    for _ in 0..MIXED_STEPS {
        world.step_with_events(&(), &log);
        log.digest_step(&mut fnv, &mut counts);
    }
    digest_world(&world, &mut fnv);
    assert!(
        counts[0] > 0,
        "mixed_scene produced no collision events: {counts:?}"
    );
    check("mixed_scene", fnv.0, MIXED_GOLDEN);
}

#[test]
fn hooked_scene_matches_fork_golden() {
    let (mut world, hooks) = hooked_scene();
    let log = EventLog::default();
    let mut fnv = Fnv::new();
    let mut counts = [0; 4];
    for _ in 0..HOOKED_STEPS {
        world.step_with_events(&hooks, &log);
        log.digest_step(&mut fnv, &mut counts);
    }
    digest_world(&world, &mut fnv);
    let calls = hooks.calls.load(Ordering::Relaxed);
    assert!(calls > 0, "hooked_scene never ran its contact hook");
    assert!(
        counts[0] > 0 && counts[2] > 0 && counts[3] > 0,
        "hooked_scene is missing collision or contact-force events: {counts:?}"
    );
    check("hooked_scene", fnv.0, HOOKED_GOLDEN);
}

fn check_collisions_last(name: &str, got: u64, golden: u64) {
    assert_eq!(
        got, golden,
        "\n{name}: collisions-last simulation results differ from the fork golden.\n  \
         golden: {golden:#018x}\n  got:    {got:#018x}\nA change altered \
         `step_collisions_last` behavior. Do not re-mint this golden to make an unrelated \
         change pass.\n",
    );
}

#[test]
fn mixed_scene_collisions_last_matches_fork_golden() {
    let mut world = mixed_scene();
    let log = EventLog::default();
    let mut fnv = Fnv::new();
    let mut counts = [0; 4];
    world.initialize_collisions_last_with_events(&(), &log);
    log.digest_step(&mut fnv, &mut counts);
    for _ in 0..MIXED_STEPS {
        world.step_collisions_last_with_events(&(), &log);
        log.digest_step(&mut fnv, &mut counts);
    }
    digest_world(&world, &mut fnv);
    assert!(
        counts[0] > 0,
        "mixed_scene (collisions-last) produced no collision events: {counts:?}"
    );
    check_collisions_last(
        "mixed_scene_collisions_last",
        fnv.0,
        MIXED_COLLISIONS_LAST_GOLDEN,
    );
}

#[test]
fn hooked_scene_collisions_last_matches_fork_golden() {
    let (mut world, hooks) = hooked_scene();
    let log = EventLog::default();
    let mut fnv = Fnv::new();
    let mut counts = [0; 4];
    world.initialize_collisions_last_with_events(&hooks, &log);
    log.digest_step(&mut fnv, &mut counts);
    for _ in 0..HOOKED_STEPS {
        world.step_collisions_last_with_events(&hooks, &log);
        log.digest_step(&mut fnv, &mut counts);
    }
    digest_world(&world, &mut fnv);
    let calls = hooks.calls.load(Ordering::Relaxed);
    assert!(
        calls > 0,
        "hooked_scene (collisions-last) never ran its contact hook"
    );
    assert!(
        counts[0] > 0 && counts[2] > 0 && counts[3] > 0,
        "hooked_scene (collisions-last) is missing collision or contact-force events: {counts:?}"
    );
    check_collisions_last(
        "hooked_scene_collisions_last",
        fnv.0,
        HOOKED_COLLISIONS_LAST_GOLDEN,
    );
}
