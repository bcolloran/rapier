//! Determinism goldens for the two stepping orders on the same scenes: `step`, whose results
//! the collisions-last code must leave untouched, and `step_collisions_last`.
//!
//! `snapshot_portability` hashes the whole serialized world, so its golden moves whenever the
//! stored layout changes. These goldens hash only simulation results: every serialized
//! container except the narrow-phase (the bodies alone reflect any change to the contact
//! solve), plus the sorted per-step event stream. Re-mint a golden only for a change meant to
//! alter the results of that stepping order, and say why in the commit.
//!
//! Run it natively and under wasm32, like `snapshot_portability`:
//!
//! ```text
//! cargo test -p rapier3d --release --features enhanced-determinism,serde-serialize \
//!     --test collisions_last_determinism
//! ```
#![cfg(all(feature = "serde-serialize", feature = "enhanced-determinism"))]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rapier3d::prelude::*;

/// FNV-1a digest of the scene state after its steps and of its events, per scene and stepping
/// order. The collisions-last digests also cover the events of the initialization.
const MIXED_STEP_GOLDEN: u64 = 0x1022_733d_1b9c_3ba2;
const HOOKED_STEP_GOLDEN: u64 = 0x903c_4ded_77b1_53a0;
const MIXED_COLLISIONS_LAST_GOLDEN: u64 = 0x5b18_a861_e3dd_c865;
const HOOKED_COLLISIONS_LAST_GOLDEN: u64 = 0x1f02_924e_0548_d78d;

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
    events: Mutex<Vec<(u8, u32, u32, u32, [u32; 3])>>,
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
            [0; 3],
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
            total_force_magnitude.to_bits(),
            [force.x.to_bits(), force.y.to_bits(), force.z.to_bits()],
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
        for (kind, h1, h2, magnitude, force) in events {
            counts[kind as usize] += 1;
            fnv.u64(kind as u64);
            fnv.u64(((h1 as u64) << 32) | h2 as u64);
            fnv.u64(magnitude as u64);
            for component in force {
                fnv.u64(component as u64);
            }
        }
    }
}

/// Digest of every serialized container except the narrow-phase: the bodies alone already
/// reflect any change to the contact solve, and the stored contacts' layout may change without
/// the results changing.
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
    world.gravity = Vector::new(0.0, -9.81, 0.0);
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(20.0, 0.5, 20.0),
    );

    for i in 0..6 {
        for j in 0..3 {
            for k in 0..6 {
                let jitter = (i as Real * 0.013 + k as Real * 0.017) % 0.05;
                world.insert(
                    RigidBodyBuilder::dynamic().translation(Vector::new(
                        i as Real * 1.05 - 3.0 + jitter,
                        j as Real * 1.05 + 0.55,
                        k as Real * 1.05 - 3.0 - jitter,
                    )),
                    ColliderBuilder::cuboid(0.5, 0.5, 0.5)
                        .active_events(ActiveEvents::COLLISION_EVENTS),
                );
            }
        }
    }

    let mut prev =
        world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 7.0, 0.0)));
    for i in 0..4 {
        let rb = world.insert_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.6 * (i + 1) as Real, 7.0, 0.0))
                .can_sleep(false),
        );
        world.insert_collider(ColliderBuilder::ball(0.25), Some(rb));
        world.insert_impulse_joint(
            prev,
            rb,
            SphericalJointBuilder::new()
                .local_anchor1(Vector::X * 0.3)
                .local_anchor2(Vector::X * -0.3),
        );
        prev = rb;
    }

    let size = 0.4;
    let mut last =
        world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(6.0, 4.0, 0.0)));
    for i in 0..6 {
        let rb = world.insert_body(RigidBodyBuilder::dynamic().can_sleep(false));
        world.insert_collider(
            ColliderBuilder::cuboid(size / 8.0, size / 2.0, size / 8.0).density(1.0),
            Some(rb),
        );
        let joint = SphericalJointBuilder::new()
            .local_anchor1(Vector::new(
                0.0,
                size / 2.0 * (i != 0) as usize as Real,
                0.0,
            ))
            .local_anchor2(Vector::new(0.0, -size / 2.0, 0.0))
            .build()
            .data;
        world.insert_multibody_joint(last, rb, joint);
        last = rb;
    }

    world.insert_collider(
        ColliderBuilder::cuboid(3.0, 0.5, 3.0)
            .translation(Vector::new(0.0, 4.0, 0.0))
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS),
        None,
    );
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-8.0, 2.0, 0.5))
            .linvel(Vector::new(90.0, -20.0, 1.0))
            .ccd_enabled(true)
            .can_sleep(false),
        ColliderBuilder::ball(0.15),
    );

    world
}

/// A contact-modification hook that writes a conveyor-belt `tangent_velocity` and an effective
/// per-manifold friction, and counts its calls.
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
    world.gravity = Vector::new(0.0, -9.81, 0.0);
    let events = ActiveEvents::COLLISION_EVENTS | ActiveEvents::CONTACT_FORCE_EVENTS;

    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(30.0, 0.5, 30.0),
    );
    world.insert(
        RigidBodyBuilder::fixed()
            .translation(Vector::new(8.0, 2.0, 0.0))
            .rotation(Vector::Z * 0.6),
        ColliderBuilder::cuboid(6.0, 0.25, 6.0),
    );
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(-12.0, 4.0, 0.0)),
        ColliderBuilder::cuboid(0.5, 4.5, 6.0),
    );

    let mut capsules = Vec::new();
    for i in 0..6 {
        let (_, handle) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(
                -10.0 + 3.5 * i as Real,
                1.5 + 0.5 * i as Real,
                (i as Real - 2.5) * 0.4,
            )),
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
            RigidBodyBuilder::dynamic().translation(Vector::new(-4.0 + 2.0 * i as Real, 3.0, 0.2)),
            ColliderBuilder::cuboid(0.4, 0.4, 0.4)
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

fn assert_golden(name: &str, got: u64, golden: u64) {
    assert_eq!(
        got, golden,
        "\n{name}: simulation results differ from the golden.\n  golden: {golden:#018x}\n  \
         got:    {got:#018x}\nRe-mint the golden only for a change meant to alter these results, \
         and say why in the commit.\n",
    );
}

// Each test first asserts that its scene exercised what the golden is meant to guard, so a
// scene that silently stops producing contacts cannot keep a golden green.

#[test]
fn mixed_scene_step_golden() {
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
    assert_golden("mixed_scene, step", fnv.0, MIXED_STEP_GOLDEN);
}

#[test]
fn hooked_scene_step_golden() {
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
    assert_golden("hooked_scene, step", fnv.0, HOOKED_STEP_GOLDEN);
}

#[test]
fn mixed_scene_collisions_last_golden() {
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
    assert_golden(
        "mixed_scene, step_collisions_last",
        fnv.0,
        MIXED_COLLISIONS_LAST_GOLDEN,
    );
}

#[test]
fn hooked_scene_collisions_last_golden() {
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
    assert_golden(
        "hooked_scene, step_collisions_last",
        fnv.0,
        HOOKED_COLLISIONS_LAST_GOLDEN,
    );
}
