//! With the feature on and no collider tracked, everything is as without the feature: the snapshot-portability scene, built here under the feature, still meets the golden size and digest committed for the build without it, and a collider whose tracking went back to body-fixed serializes like one never tracked.

#[path = "../snapshot_portability.rs"]
mod snapshot_portability;

#[cfg(feature = "serde-serialize")]
use rapier2d::prelude::*;
#[cfg(feature = "serde-serialize")]
use test_case::test_case;

/// A collider set to `tracking` and back to body-fixed is the collider it was, byte for byte.
#[cfg(feature = "serde-serialize")]
#[test_case(SubstepContactTracking::CurvatureCenter ; "curvature centre")]
#[test_case(SubstepContactTracking::FirstOrder ; "first order")]
fn a_collider_back_at_body_fixed_serializes_as_one_never_tracked(tracking: SubstepContactTracking) {
    let never = ColliderBuilder::ball(0.3)
        .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS)
        .build();
    let mut back = never.clone();
    back.set_substep_contact_tracking(tracking).unwrap();
    back.set_substep_contact_tracking(SubstepContactTracking::BodyFixed)
        .unwrap();
    assert_eq!(
        bincode::serialize(&back).unwrap(),
        bincode::serialize(&never).unwrap()
    );
}

/// A tracked collider adds no bytes either: its tracking is two bits of the hook flags it already stores.
#[cfg(feature = "serde-serialize")]
#[test_case(SubstepContactTracking::CurvatureCenter ; "curvature centre")]
#[test_case(SubstepContactTracking::FirstOrder ; "first order")]
fn a_tracked_collider_is_the_size_of_an_untracked_one(tracking: SubstepContactTracking) {
    let untracked = ColliderBuilder::ball(0.3).build();
    let tracked = ColliderBuilder::ball(0.3)
        .substep_contact_tracking(tracking)
        .unwrap()
        .build();
    assert_eq!(
        bincode::serialize(&tracked).unwrap().len(),
        bincode::serialize(&untracked).unwrap().len()
    );
}
