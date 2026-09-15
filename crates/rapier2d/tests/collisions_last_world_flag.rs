//! `PhysicsWorld::collisions_last` selects the stepping order of `PhysicsWorld::step`, which is
//! how the testbed's Settings > Advanced > "Collisions last" checkbox applies to every example.
//! A stock step clears the collisions-last initialization, so switching back to collisions-last
//! mid-run runs its initial collision detection again at the current poses.

use rapier2d::prelude::*;

/// Boxes dropped in stacks onto a floor. The bottom boxes hit the floor around step 19.
fn scene() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(20.0, 0.5),
    );
    for i in 0..6 {
        for j in 0..3 {
            world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(i as Real * 1.2 - 3.0, 1.5 + j as Real * 1.05)),
                ColliderBuilder::cuboid(0.5, 0.5),
            );
        }
    }
    world
}

/// Every body's pose and velocity, as raw bits.
fn state_bits(world: &PhysicsWorld) -> Vec<u32> {
    let mut bits = Vec::new();
    for (_, rb) in world.rigid_bodies() {
        let (translation, linvel) = (rb.translation(), rb.linvel());
        bits.extend([
            translation.x.to_bits(),
            translation.y.to_bits(),
            rb.rotation().angle().to_bits(),
            linvel.x.to_bits(),
            linvel.y.to_bits(),
            rb.angvel().to_bits(),
        ]);
    }
    bits
}

/// The number of touching contact pairs: what a caller reads from the narrow-phase between steps.
fn touching_pairs(world: &PhysicsWorld) -> usize {
    world
        .contact_pairs()
        .filter(|pair| pair.has_any_active_contact())
        .count()
}

#[test]
fn collisions_last_flag_makes_step_detect_collisions_last() {
    let mut flagged = scene();
    flagged.collisions_last = true;
    let mut explicit = scene();
    let mut stock = scene();

    let mut flagged_contacts = Vec::new();
    let mut stock_contacts = Vec::new();
    for step in 0..120 {
        flagged.step();
        explicit.step_collisions_last();
        stock.step();
        assert_eq!(
            state_bits(&flagged),
            state_bits(&explicit),
            "step {step}: `step` with `collisions_last` set moved the bodies differently from \
             `step_collisions_last`"
        );
        assert_eq!(
            touching_pairs(&flagged),
            touching_pairs(&explicit),
            "step {step}: `step` with `collisions_last` set reports different contacts from \
             `step_collisions_last`"
        );
        flagged_contacts.push(touching_pairs(&flagged));
        stock_contacts.push(touching_pairs(&stock));
    }

    // With no changes between steps, both stepping orders solve with contacts detected at the same
    // poses (stock at the start of step k + 1, collisions-last at the end of step k), so the bodies
    // move identically. What differs is what the narrow-phase reports between steps:
    // collisions-last reports the contacts of the new poses, which stock only detects at the start
    // of its next step.
    assert_eq!(state_bits(&flagged), state_bits(&stock));
    assert_ne!(
        flagged_contacts, stock_contacts,
        "the scene never changed its contacts, so it can't tell the stepping orders apart"
    );
    assert_eq!(
        flagged_contacts[..119],
        stock_contacts[1..],
        "collisions-last should report each step's contacts one step before stock does"
    );
}

#[test]
fn stock_step_clears_collisions_last_initialization() {
    let mut world = scene();
    world.step_collisions_last();
    assert!(world.physics_pipeline.collisions_last_initialized());
    world.step();
    assert!(!world.physics_pipeline.collisions_last_initialized());
}

#[test]
fn switching_the_flag_mid_run_reinitializes_collisions_last() {
    let mut flipped = scene();
    let mut reference = scene();

    // Collisions-last, then stock up to the moment the boxes hit the floor, in both worlds.
    flipped.collisions_last = true;
    for _ in 0..12 {
        flipped.step();
        reference.step_collisions_last();
    }
    flipped.collisions_last = false;
    for _ in 0..8 {
        flipped.step();
        reference.step();
    }

    // Back to collisions-last during the impacts. The reference re-runs the initial collision
    // detection explicitly; the flipped world must do the same on its own instead of solving with
    // the contacts of the last stock step's start-of-step poses.
    reference
        .physics_pipeline
        .set_collisions_last_initialized(false);
    reference.initialize_collisions_last_with_events(&(), &());
    flipped.collisions_last = true;
    for step in 0..40 {
        flipped.step();
        reference.step_collisions_last();
        assert_eq!(
            state_bits(&flipped),
            state_bits(&reference),
            "step {step} after switching back to collisions-last"
        );
    }
}
