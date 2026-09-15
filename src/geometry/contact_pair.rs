use super::CollisionEvent;
use crate::alloc_prelude::*;
use crate::dynamics::{RigidBodyHandle, RigidBodySet};
use crate::geometry::{ColliderHandle, ColliderSet, Contact, ContactManifold};
use crate::math::{Pose, Real, TangentImpulse, Vector};
use crate::pipeline::EventHandler;
use crate::prelude::CollisionEventFlags;
use crate::utils::ScalarType;
use crate::utils::SolverBlock;
use parry::math::{SIMD_WIDTH, SimdReal};
use parry::query::ContactManifoldsWorkspace;
// Only `relative_pose_drift`’s 2D branch needs the no-std float methods.
#[cfg(all(not(feature = "std"), feature = "dim2"))]
use simba::scalar::ComplexField as _;

bitflags::bitflags! {
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    /// Flags affecting the behavior of the constraints solver for a given contact manifold.
    pub struct SolverFlags: u32 {
        /// The constraint solver will take this contact manifold into
        /// account for force computation.
        const COMPUTE_IMPULSES = 0b001;
    }
}

impl Default for SolverFlags {
    fn default() -> Self {
        SolverFlags::COMPUTE_IMPULSES
    }
}

bitflags::bitflags! {
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    #[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
    /// Event bookkeeping bits of a contact pair.
    ///
    /// Serialized as a single byte with the same values as the `start_event_emitted`
    /// bool it replaces, so snapshots keep their exact byte layout.
    pub(crate) struct PairEventStatus: u8 {
        /// A `CollisionEvent::Started` was emitted for this pair.
        const START_EVENT_EMITTED = 0b01;
        /// The pair's total contact force exceeded its force-event threshold at the
        /// previous step. [`ContactForceEvent::started`] is derived from it, and it
        /// resets when the force drops back below the threshold or the pair stops
        /// touching.
        const INITIAL_FORCE_THRESHOLD_EVENT_EMITTED = 0b10;
    }
}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
/// A single contact between two collider.
pub struct ContactData {
    /// The impulse, along the contact normal, applied by this contact to the first collider's rigid-body.
    ///
    /// The impulse applied to the second collider's rigid-body is given by `-impulse`.
    pub impulse: Real,
    /// The friction impulse along the vector orthonormal to the contact normal, applied to the first
    /// collider's rigid-body.
    pub tangent_impulse: TangentImpulse<Real>,
    /// The impulse retained for warmstarting the next simulation step.
    pub warmstart_impulse: Real,
    /// The friction impulse retained for warmstarting the next simulation step.
    pub warmstart_tangent_impulse: TangentImpulse<Real>,
    /// The twist impulse retained for warmstarting the next simulation step.
    #[cfg(feature = "dim3")]
    pub warmstart_twist_impulse: Real,
    /// The friction warm-start impulse as a **world-space** vector — the canonical
    /// value 3D friction warm-starts from,
    /// projected onto the constraint's current tangent basis at constraint generation.
    /// Warm-starting from the raw [`Self::warmstart_tangent_impulse`] components would
    /// silently rotate the friction force whenever the basis changes, kicking resting stacks.
    #[cfg(feature = "dim3")]
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub warmstart_tangent_world: Vector,
    /// The solver's lever arm for the first body: contact point relative to the body's CoM,
    /// in **world space**, frozen at the pair's last full narrow-phase update (anchor
    /// freezing) and used verbatim while recycled. Load-bearing for tall-stack stability:
    /// re-linearizing the arms every step under heavy warm-started impulses is a state-
    /// proportional energy pump (lean mode). Separations still track the bodies' rigid motion.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub solver_dp1: Vector,
    /// The solver's lever arm for the second body (see [`Self::solver_dp1`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub solver_dp2: Vector,
}

impl Default for ContactData {
    fn default() -> Self {
        Self {
            impulse: 0.0,
            tangent_impulse: na::zero(),
            warmstart_impulse: 0.0,
            warmstart_tangent_impulse: na::zero(),
            #[cfg(feature = "dim3")]
            warmstart_twist_impulse: 0.0,
            #[cfg(feature = "dim3")]
            warmstart_tangent_world: Vector::ZERO,
            solver_dp1: Vector::ZERO,
            solver_dp2: Vector::ZERO,
        }
    }
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
/// The description of all the contacts between a pair of colliders.
pub struct IntersectionPair {
    /// Are the colliders intersecting?
    pub intersecting: bool,
    /// Was a `CollisionEvent::Started` emitted for this collider?
    pub(crate) start_event_emitted: bool,
}

impl IntersectionPair {
    pub(crate) fn new() -> Self {
        Self {
            intersecting: false,
            start_event_emitted: false,
        }
    }

    pub(crate) fn emit_start_event(
        &mut self,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        collider1: ColliderHandle,
        collider2: ColliderHandle,
        events: &dyn EventHandler,
    ) {
        self.start_event_emitted = true;
        events.handle_collision_event(
            bodies,
            colliders,
            CollisionEvent::Started(collider1, collider2, CollisionEventFlags::SENSOR),
            None,
        );
    }

    pub(crate) fn emit_stop_event(
        &mut self,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        collider1: ColliderHandle,
        collider2: ColliderHandle,
        events: &dyn EventHandler,
    ) {
        self.start_event_emitted = false;
        events.handle_collision_event(
            bodies,
            colliders,
            CollisionEvent::Stopped(collider1, collider2, CollisionEventFlags::SENSOR),
            None,
        );
    }
}

/// Sentinel color for pairs currently holding no solver graph color.
pub(crate) const SOLVER_COLOR_UNCOLORED: u8 = u8::MAX;
/// Color assigned when the parallel color space is exhausted (or for extra manifolds
/// of multi-manifold pairs); such constraints are solved sequentially.
pub(crate) const SOLVER_COLOR_OVERFLOW: u8 = 128;
/// Number of low colors dynamic-vs-dynamic contacts may use; `..128` is reserved for
/// dynamic-vs-fixed so those always iterate last, giving
/// fixed geometry the final say each sweep and reducing push-through of piled bodies.
pub(crate) const SOLVER_DYNAMIC_COLOR_COUNT: u32 = 120;

#[cfg(feature = "serde-serialize")]
fn default_solver_color() -> u8 {
    SOLVER_COLOR_UNCOLORED
}
#[cfg(feature = "serde-serialize")]
fn default_solver_color_bodies() -> [u32; 2] {
    [u32::MAX; 2]
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone)]
/// All contact information between two colliding colliders.
///
/// When two colliders are touching, a ContactPair stores all the contact points, normals,
/// and forces between them. You can access this through the narrow phase or in event handlers.
///
/// ## Contact manifolds
///
/// The contacts are organized into "manifolds" - groups of contact points that share similar
/// properties (like being on the same face). Most collider pairs have 1 manifold, but complex
/// shapes may have multiple.
///
/// ## Use cases
///
/// - Reading contact normals for custom physics
/// - Checking penetration depth
/// - Analyzing impact forces
/// - Implementing custom contact responses
///
/// # Example
/// ```
/// # use rapier3d::prelude::*;
/// # use rapier3d::geometry::ContactPair;
/// # let contact_pair = ContactPair::default();
/// if let Some((manifold, contact)) = contact_pair.find_deepest_contact() {
///     println!("Deepest penetration: {}", -contact.dist);
///     println!("Contact normal: {:?}", manifold.data.normal);
/// }
/// ```
pub struct ContactPair {
    /// The first collider involved in the contact pair.
    pub collider1: ColliderHandle,
    /// The second collider involved in the contact pair.
    pub collider2: ColliderHandle,
    /// The set of contact manifolds between the two colliders.
    ///
    /// All contact manifold contain themselves contact points between the colliders.
    /// Note that contact points in the contact manifold do not take into account the
    /// [`Collider::contact_skin`] which only affects the constraint solver and the
    /// [`SolverContact`].
    ///
    /// [`Collider::contact_skin`]: crate::geometry::Collider::contact_skin
    pub manifolds: Vec<ContactManifold>,
    /// Cluster manifolds handed to the constraint solver instead of `manifolds` when
    /// contact clustering applies (see [`IntegrationParameters::contact_clustering`]);
    /// empty otherwise. They merge the points of manifolds sharing (nearly) the same
    /// contact normal and hold the contact impulses actually applied by the solver.
    ///
    /// [`IntegrationParameters::contact_clustering`]: crate::dynamics::IntegrationParameters::contact_clustering
    pub solver_clusters: Vec<ContactManifold>,
    /// The clusters solved at the previous step, kept as the warm-start source (and
    /// reused as scratch buffers) when rebuilding `solver_clusters` each frame.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) solver_clusters_prev: Vec<ContactManifold>,
    /// The persistent solver graph color of this pair: same-color active pairs never share
    /// a rigid-body, so one color solves concurrently. Maintained incrementally on contact
    /// start/stop; `SOLVER_COLOR_UNCOLORED` inactive, `SOLVER_COLOR_OVERFLOW` no free color.
    #[cfg_attr(feature = "serde-serialize", serde(default = "default_solver_color"))]
    pub(crate) solver_color: u8,
    /// The body mask slots on which this pair's color bit is set (u32::MAX = none).
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_solver_color_bodies")
    )]
    pub(crate) solver_color_bodies: [u32; 2],
    /// Event bookkeeping: `CollisionEvent::Started` emission and force-event
    /// threshold status.
    pub(crate) event_status: PairEventStatus,
    pub(crate) workspace: Option<ContactManifoldsWorkspace>,
    /// State cached at the last full narrow-phase update, allowing the update to be
    /// skipped ("recycled") while the colliders' relative pose stays within
    /// `IntegrationParameters::contact_recycling`'s drift threshold.
    ///
    /// Part of the snapshot: a restored pair must resume recycling from the same
    /// reference pose, or its first update recomputes manifolds (and re-derives the
    /// world-frozen solver anchors) where the uninterrupted run would have recycled.
    pub(crate) recycle_state: Option<ContactRecycleState>,
}

/// The relative configuration of a contact pair at its last full narrow-phase
/// update, used by contact recycling to bound how much the pair moved since.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub(crate) struct ContactRecycleState {
    /// Pose of the second collider relative to the first at the last full update.
    pub pos12: Pose,
    /// World rotation of the first collider at the last full update. The frozen world-space anchors
    /// ([`ContactData::solver_dp1`]) mean recycling must also bound each body's *absolute* rotation:
    /// rotating rigidly together keeps the relative pose but invalidates world-frozen arms (bound: `cos Δθ > 0.98`, ~11.5°).
    pub rot1: crate::math::Rotation,
    /// World rotation of the second collider at the last full update.
    pub rot2: crate::math::Rotation,
    /// Conservative bound on the distance of any point of either shape from its
    /// collider origin, used to convert a relative rotation into a point-drift bound.
    pub max_extent: Real,
    /// The maximum relative-pose drift below which this pair can be recycled,
    /// precomputed at the last full update (it depends on whether the pair had
    /// active contacts, which recycling doesn't change). Negative when the pair must not be
    /// recycled from this state: its full update stored an adhesion request, which only a full
    /// update can clear once the hook flag is gone.
    pub max_drift: Real,
}

/// `cos Δθ` between two world rotations (in 3D, computed from the quaternion dot
/// `cos(Δθ/2)` as `2·dot² − 1`), for the per-body rotation bound of contact
/// recycling.
#[inline]
pub(crate) fn relative_rot_cos(base: &crate::math::Rotation, cur: &crate::math::Rotation) -> Real {
    #[cfg(feature = "dim2")]
    {
        base.dot(*cur)
    }
    #[cfg(feature = "dim3")]
    {
        let c = base.dot(*cur);
        2.0 * c * c - 1.0
    }
}

/// Straight-line bound on how far any point within `max_extent` of the origin moved
/// between poses `base` and `cur`: translation delta + rotation *chord* `2·max_extent·sin(Δθ/2)`.
/// Tighter than the arc length `max_extent·Δθ`, and no `atan2`/`acos`.
#[inline]
pub(crate) fn relative_pose_drift(base: &Pose, cur: &Pose, max_extent: Real) -> Real {
    let trans = (cur.translation - base.translation).length();
    let delta_rot = cur.rotation * base.rotation.inverse();
    #[cfg(feature = "dim2")]
    let rot_chord = {
        // To avoid explicit trigonometric functions, use the identity:
        // `sin(Δθ/2) = |sin Δθ| / sqrt(2(1 + cos Δθ))`
        let (sin, cos) = (delta_rot.sin(), delta_rot.cos());
        let denom = 2.0 * (1.0 + cos);
        let half_sin = if denom > 1.0e-6 {
            sin.abs() / denom.sqrt()
        } else {
            1.0
        };

        2.0 * half_sin * max_extent
    };
    #[cfg(feature = "dim3")]
    let rot_chord = {
        // A unit quaternion's vector part is already `sin(Δθ/2)` about its axis.
        2.0 * Vector::new(delta_rot.x, delta_rot.y, delta_rot.z).length() * max_extent
    };
    trans + rot_chord
}

impl Default for ContactPair {
    fn default() -> Self {
        Self::new(ColliderHandle::invalid(), ColliderHandle::invalid())
    }
}

impl ContactPair {
    pub(crate) fn new(collider1: ColliderHandle, collider2: ColliderHandle) -> Self {
        Self {
            collider1,
            collider2,
            manifolds: Vec::new(),
            solver_clusters: Vec::new(),
            solver_clusters_prev: Vec::new(),
            solver_color: SOLVER_COLOR_UNCOLORED,
            solver_color_bodies: [u32::MAX; 2],
            event_status: PairEventStatus::empty(),
            workspace: None,
            recycle_state: None,
        }
    }

    /// Resets a retired pair to the exact state [`Self::new`] would produce,
    /// keeping the (outer) buffer capacities so pooled reuse skips their
    /// reallocation on pair-churn-heavy scenes.
    pub(crate) fn reset_for_reuse(&mut self, collider1: ColliderHandle, collider2: ColliderHandle) {
        self.collider1 = collider1;
        self.collider2 = collider2;
        self.manifolds.clear();
        self.solver_clusters.clear();
        self.solver_clusters_prev.clear();
        self.solver_color = SOLVER_COLOR_UNCOLORED;
        self.solver_color_bodies = [u32::MAX; 2];
        self.event_status = PairEventStatus::empty();
        self.workspace = None;
        self.recycle_state = None;
    }

    /// The manifolds actually seen by the constraint solver: the contact clusters if
    /// clustering applied to this pair, the plain manifolds otherwise.
    pub fn solver_manifolds(&self) -> &[ContactManifold] {
        if self.solver_clusters.is_empty() {
            &self.manifolds
        } else {
            &self.solver_clusters
        }
    }

    /// Mutable twin of [`Self::solver_manifolds`]: the manifolds the constraint
    /// solver actually sees (the solver clusters if any, else the plain manifolds).
    #[cfg_attr(feature = "parallel", allow(dead_code))] // Single-threaded solver path.
    pub(crate) fn solver_manifolds_mut(&mut self) -> &mut [ContactManifold] {
        if self.solver_clusters.is_empty() {
            &mut self.manifolds
        } else {
            &mut self.solver_clusters
        }
    }

    /// Is there any active contact in this contact pair?
    pub fn has_any_active_contact(&self) -> bool {
        self.solver_manifolds()
            .iter()
            .any(|m| !m.data.solver_contacts.is_empty())
    }

    /// Clears all the contacts of this contact pair.
    pub fn clear(&mut self) {
        self.manifolds.clear();
        self.solver_clusters.clear();
        self.solver_clusters_prev.clear();
        self.workspace = None;
        self.recycle_state = None;
    }

    // NOTE: while recycled, a pair's world-space solver data (normal, frozen lever arms — see
    // `ContactData::solver_dp1`) keeps its last-full-update values (anchor freezing): the solver
    // rebuilds world points/separations from body-local anchors + current poses, so no per-step refresh; user data stays stale within the recycle drift bound.

    /// The total impulse (force × time) applied by all contacts.
    ///
    /// This is the accumulated force that pushed the colliders apart.
    /// Useful for determining impact strength.
    pub fn total_impulse(&self) -> Vector {
        self.solver_manifolds()
            .iter()
            .map(|m| m.total_impulse() * m.data.normal)
            .sum()
    }

    /// The total magnitude of all contact impulses (sum of lengths, not length of sum).
    ///
    /// This is what's compared against `contact_force_event_threshold`.
    pub fn total_impulse_magnitude(&self) -> Real {
        self.solver_manifolds()
            .iter()
            .fold(0.0, |a, m| a + m.total_impulse())
    }

    /// Finds the strongest contact impulse and its direction.
    ///
    /// Returns `(magnitude, normal_direction)` of the strongest individual contact.
    pub fn max_impulse(&self) -> (Real, Vector) {
        let mut result = (0.0, Vector::ZERO);

        for m in self.solver_manifolds() {
            let impulse = m.total_impulse();

            if impulse > result.0 {
                result = (impulse, m.data.normal);
            }
        }

        result
    }

    /// Finds the contact point with the deepest penetration.
    ///
    /// When objects overlap, this returns the contact point that's penetrating the most.
    /// Useful for:
    /// - Finding the "worst" overlap
    /// - Determining primary contact direction
    /// - Custom penetration resolution
    ///
    /// Returns both the contact point and its parent manifold.
    ///
    /// # Example
    /// ```
    /// # use rapier3d::prelude::*;
    /// # use rapier3d::geometry::ContactPair;
    /// # let pair = ContactPair::default();
    /// if let Some((manifold, contact)) = pair.find_deepest_contact() {
    ///     let penetration_depth = -contact.dist;  // Negative dist = penetration
    ///     println!("Deepest penetration: {} units", penetration_depth);
    /// }
    /// ```
    #[profiling::function]
    pub fn find_deepest_contact(&self) -> Option<(&ContactManifold, &Contact)> {
        let mut deepest = None;

        for m2 in &self.manifolds {
            let deepest_candidate = m2.find_deepest_contact();

            deepest = match (deepest, deepest_candidate) {
                (_, None) => deepest,
                (None, Some(c2)) => Some((m2, c2)),
                (Some((m1, c1)), Some(c2)) => {
                    if c1.dist <= c2.dist {
                        Some((m1, c1))
                    } else {
                        Some((m2, c2))
                    }
                }
            }
        }

        deepest
    }

    pub(crate) fn emit_start_event(
        &mut self,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        events: &dyn EventHandler,
    ) {
        self.event_status
            .insert(PairEventStatus::START_EVENT_EMITTED);

        events.handle_collision_event(
            bodies,
            colliders,
            CollisionEvent::Started(self.collider1, self.collider2, CollisionEventFlags::empty()),
            Some(self),
        );
    }

    pub(crate) fn emit_stop_event(
        &mut self,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        events: &dyn EventHandler,
    ) {
        // Not touching anymore: the force-event threshold status resets with it.
        self.event_status = PairEventStatus::empty();

        events.handle_collision_event(
            bodies,
            colliders,
            CollisionEvent::Stopped(self.collider1, self.collider2, CollisionEventFlags::empty()),
            Some(self),
        );
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
/// A contact manifold between two colliders.
///
/// A contact manifold describes a set of contacts between two colliders. All the contact
/// part of the same contact manifold share the same contact normal and contact kinematics.
pub struct ContactManifoldData {
    // The following are set by the narrow-phase.
    /// The first rigid-body involved in this contact manifold.
    pub rigid_body1: Option<RigidBodyHandle>,
    /// The second rigid-body involved in this contact manifold.
    pub rigid_body2: Option<RigidBodyHandle>,
    // We put the following fields here to avoids reading the colliders inside of the
    // contact preparation method.
    /// Flags used to control some aspects of the constraints solver for this contact manifold.
    pub solver_flags: SolverFlags,
    /// The solver graph color of this manifold (copied from its contact pair during
    /// constraint selection; extra manifolds of a same pair are sent to the overflow
    /// color since they share their bodies).
    #[cfg_attr(feature = "serde-serialize", serde(default = "default_solver_color"))]
    pub(crate) solver_color: u8,
    /// The solver-body index (`active_set_id`) of each rigid-body, or `u32::MAX` for a
    /// world-attached side (fixed, sleeping, no body). Stamped by constraint selection so
    /// the assembly never re-reads the rigid-body set.
    pub(crate) solver_body_ids: [u32; 2],
    /// This manifold's persistent position (bucket + index) in the narrow-phase's
    /// `SolverContactGraph`, maintained incrementally so the solver reads a ready color-grouped
    /// contact list without re-selecting/re-sorting. `GraphPos::NONE` when not solver-active.
    #[cfg_attr(feature = "parallel", allow(dead_code))] // Single-threaded solver path.
    pub(crate) graph_pos: crate::dynamics::solver::solver_contact_graph::GraphPos,
    /// The world-space contact normal shared by all the contact in this contact manifold.
    // NOTE: read the comment of `solver_contacts` regarding serialization. It applies
    // to this field as well.
    pub normal: Vector,
    /// The contacts that will be seen by the constraints solver for computing forces.
    // NOTE: unfortunately, we can't ignore this field when serialize
    // the contact manifold data. The reason is that the solver contacts
    // won't be updated for sleeping bodies. So it means that for one
    // frame, we won't have any solver contacts when waking up an island
    // after a deserialization. Not only does this break post-snapshot
    // determinism, but it will also skip constraint resolution for these
    // contacts during one frame.
    //
    // An alternative would be to skip the serialization of `solver_contacts` and
    // find a way to recompute them right after the deserialization process completes.
    // However, this would be an expensive operation. And doing this efficiently as part
    // of the narrow-phase update or the contact manifold collect will likely lead to tricky
    // bugs too.
    //
    // So right now it is best to just serialize this field and keep it that way until it
    // is proven to be actually problematic in real applications (in terms of snapshot size for example).
    pub solver_contacts: SolverContacts,
    /// The relative dominance of the bodies involved in this contact manifold.
    pub relative_dominance: i16,
    /// A user-defined piece of data.
    pub user_data: u32,
    /// The effective friction coefficient of this manifold's contacts (combined from
    /// both colliders' materials; identical for every contact of the manifold).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub friction: Real,
    /// The effective restitution coefficient of this manifold's contacts.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub restitution: Real,
    /// Attractive adhesion force pulling the two bodies together (e.g. glue or suction).
    ///
    /// Applied along [`Self::normal`] (spread over [`Self::solver_contacts`]) as an external
    /// force on both bodies before the contact solver runs, so the push-only contacts provide
    /// the reaction (and hence friction). `0.0` (default) is an ordinary contact; non-positive
    /// and non-finite values are ignored (a non-finite value adds nothing, and doesn't cancel the
    /// manifold's other adhesion terms). Written by the narrow-phase each time
    /// [`PhysicsHooks::modify_solver_contacts`] runs on this manifold (reset to `0.0` when the
    /// pair has no [`ActiveHooks::MODIFY_SOLVER_CONTACTS`]).
    ///
    /// This is an absolute force *per manifold*: a surface split into N colliders produces ~N
    /// manifolds and therefore ~N× the total pull. For adhesion that behaves identically on a
    /// monolithic surface and on the same surface decomposed into many small colliders, use
    /// [`Self::adhesion_pressure`] or [`Self::adhesion_budget`] instead.
    ///
    /// For a 3D pair whose manifolds are clustered for the solver
    /// ([`ContactPair::solver_clusters`] is not empty), the hook runs on the clusters, so the
    /// adhesion request ([`Self::adhesion_force`], [`Self::adhesion_pressure`],
    /// [`Self::adhesion_budget`]) and [`Self::tangential_extent`] live on the clusters (see
    /// [`ContactPair::solver_manifolds`]); the pair's plain [`ContactPair::manifolds`] show `0.0`
    /// and `None`.
    ///
    /// The adhesion is added to the bodies' effective force for the solve, and that force is still
    /// in place when the next step starts, so the next step's CCD activation check (which predicts
    /// each body's motion from its velocity and forces) sees it.
    ///
    /// Serialized: a restored snapshot resumes with the adhesion requested before it was taken.
    ///
    /// [`PhysicsHooks::modify_solver_contacts`]: crate::pipeline::PhysicsHooks::modify_solver_contacts
    /// [`ActiveHooks::MODIFY_SOLVER_CONTACTS`]: crate::pipeline::ActiveHooks::MODIFY_SOLVER_CONTACTS
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub adhesion_force: Real,
    /// Attractive adhesion *pressure* pulling the two bodies together, expressed per unit of
    /// contact patch: force per meter of contact length in 2D, force per square meter of contact
    /// area in 3D.
    ///
    /// The resulting force for this manifold is `adhesion_pressure * tangential_extent()`, applied
    /// exactly like [`Self::adhesion_force`] (the two add up). Because the contact spans of
    /// abutting colliders partition the body's total contact patch, this formulation is
    /// composition-invariant: splitting one big collider into many small ones leaves the total
    /// adhesion force (and its torque) unchanged. The flip side is that point contacts (2D) and
    /// point/line contacts (3D) have zero extent and receive no pressure adhesion — use
    /// [`Self::adhesion_force`] or [`Self::adhesion_budget`] for those. Overlapping colliders
    /// double-count the overlapped span; [`Self::adhesion_budget`] does not.
    ///
    /// `0.0` (default) is an ordinary contact; non-positive and non-finite values are ignored, and
    /// so is a non-finite product with the extent (an infinite pressure on a point contact).
    /// Written like [`Self::adhesion_force`].
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub adhesion_pressure: Real,
    /// Membership of this manifold in a budgeted adhesion pool (see [`AdhesionBudget`]).
    ///
    /// `None` (default) is an ordinary contact. Written like [`Self::adhesion_force`].
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub adhesion_budget: Option<AdhesionBudget>,
    /// The tangential extent of [`Self::solver_contacts`] measured right after the
    /// contact-modification hook last ran on this manifold (see [`Self::tangential_extent`]);
    /// `0.0` for manifolds of pairs without `MODIFY_SOLVER_CONTACTS`.
    ///
    /// Cached because the solver contacts are converted to body-local anchors right after the
    /// hook, after which the world-space patch can no longer be measured cheaply.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) adhesion_extent: Real,
}

/// A *budgeted* adhesion request: a fixed total adhesion force shared by every manifold enrolled
/// in the same pool during the same timestep.
///
/// This is the right model for "this region of my body has stickiness `total`": however many
/// manifolds happen to implement the region's contact this step — one big collider, many abutting
/// tiles, *overlapping* colliders, a seam-straddling pair of point contacts — the force applied
/// across all of them always sums to exactly `total`. Neither
/// [`ContactManifoldData::adhesion_force`] (which multiplies with the manifold count) nor
/// [`ContactManifoldData::adhesion_pressure`] (which double-counts overlapping coverage and
/// vanishes on point contacts) has that property.
///
/// Pools are keyed by `(owner, channel)`:
/// - `owner` is an opaque identity, never dereferenced by the engine — typically the collider the
///   budget belongs to (e.g. a player capsule). Distinct bodies must use distinct owners or their
///   budgets merge.
/// - `channel` distinguishes independent regions of the same owner (e.g. `0` = feet, `1` = flank),
///   so a body in a corner can spend its feet budget *and* its flank budget simultaneously.
///
/// Within a pool, the effective total is the **maximum** of the enrolled manifolds' `total`
/// requests (max, not sum, so duplicated or overlapping colliders can never inflate it), and it is
/// distributed over the enrolled manifolds proportionally to their tangential extent (point
/// contacts get a small floor weight, so a lone point contact still receives the full total, while
/// a degenerate sliver alongside real area contacts receives almost nothing).
///
/// Only manifolds actually seen by the constraint solver this step (at least one solver contact,
/// at least one awake dynamic body) take part in a pool.
///
/// A member's share is computed from the whole pool before the engine checks which of the member's
/// two bodies it can pull: only awake dynamic bodies are pulled, and never a side the solver
/// treats as world-attached through dominance. A member whose pullable side is skipped (kinematic,
/// fixed, sleeping, or dominance-world-attached) still takes its share of the pool's weight, and
/// that share pulls nothing on that side. So the shares of a pool add up to its total only when
/// every member's pullable side is actually pulled.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct AdhesionBudget {
    /// Opaque pool identity; typically the collider this budget belongs to.
    pub owner: ColliderHandle,
    /// Distinguishes independent budgets of the same owner (e.g. feet vs. flank).
    pub channel: u32,
    /// Total adhesion force shared by the pool this step. Non-positive and non-finite values are
    /// ignored: such a request joins no pool.
    pub total: Real,
}

/// A single solver contact.
pub type SolverContact = SolverContactGeneric<Real, 1>;

/// The container of a manifold's solver contacts. In 2D a manifold has at most 2 active
/// contacts, so they are stored inline: the solver's contact gathers read one contiguous
/// manifold instead of chasing a heap allocation per manifold (a dependent cache miss
/// on every SIMD lane of every constraint, every step).
#[cfg(feature = "dim2")]
pub type SolverContacts = arrayvec::ArrayVec<SolverContact, 2>;
/// The container of a manifold's solver contacts. In 3D, composite-shape manifolds can
/// exceed the solver's per-constraint point cap, so they stay heap-allocated.
#[cfg(feature = "dim3")]
pub type SolverContacts = Vec<SolverContact>;
/// A group of `SIMD_WIDTH` solver contacts stored in SoA fashion for SIMD optimizations.
pub type SimdSolverContact = SolverContactGeneric<SimdReal, SIMD_WIDTH>;

/// A contact seen by the constraints solver for computing forces.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "N: serde::Serialize, N::Vector: serde::Serialize, [ContactId; LANES]: serde::Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "N: serde::Deserialize<'de>, N::Vector: serde::Deserialize<'de>, [ContactId; LANES]: serde::Deserialize<'de>"
    ))
)]
#[repr(C)]
#[repr(align(16))]
pub struct SolverContactGeneric<N: ScalarType, const LANES: usize> {
    // IMPORTANT: don't change the fields unless `SimdSolverContactRepr` is also changed.
    // TOTAL: 8/8 lanes in 2D (two 16B SIMD rows), 11/12 in 3D. Friction/restitution live
    // on `ContactManifoldData`, is-new in bit 31 of `contact_id`, warm-starts on the manifold points.
    /// The contact point on the first body's surface (contact skin baked in), in that
    /// body's CoM-centered local frame so it rides rigidly with the body — what lets
    /// contact recycling skip the per-frame world refresh. World-space instead for a side
    /// without a solver body (none, or world-attached by dominance — fixed bodies included).
    /// Inside [`PhysicsHooks::modify_solver_contacts`] this always holds the fresh
    /// **world-space** point (hooks run before localization).
    ///
    /// [`PhysicsHooks::modify_solver_contacts`]: crate::pipeline::PhysicsHooks::modify_solver_contacts
    pub anchor1: N::Vector, // 2/3
    /// The contact point on the second body's surface, expressed like
    /// [`Self::anchor1`] (world-space when the second side is world-attached, i.e.
    /// `relative_dominance < 0`, or inside the contact-modification hook).
    pub anchor2: N::Vector, // 2/3
    /// Distance between the contact points along the normal at the last full contact
    /// update (negative = penetration), minus the contact skins. Writable from
    /// [`PhysicsHooks::modify_solver_contacts`] (the delta is baked into the anchors after
    /// the hook); afterwards the solver re-derives the live separation and never reads this.
    ///
    /// [`PhysicsHooks::modify_solver_contacts`]: crate::pipeline::PhysicsHooks::modify_solver_contacts
    pub dist: N, // 1/1
    /// The desired tangent relative velocity at the contact point.
    ///
    /// This is set to zero by default. Set to a non-zero value to
    /// simulate, e.g., conveyor belts.
    pub tangent_velocity: N::Vector, // 2/3
    /// The index of the manifold contact used to generate this solver contact, in the
    /// low 31 bits; bit 31 ([`NEW_CONTACT_BIT`]) is set if this contact did not exist
    /// during the last *full* contact update (recycled steps leave it untouched; the
    /// solver derives contact newness from the warm-start state instead).
    pub contact_id: [ContactId; LANES], // 1/1
    #[cfg(feature = "dim3")]
    pub(crate) padding: [N; 1],
}

/// The storage type of [`SolverContactGeneric::contact_id`]: one `Real`-sized slot
/// per lane, so that a lane of the AoSoA struct keeps the same layout as a scalar
/// contact. At `f32` a slot is exactly the `u32` id; at `f64` the high 32 bits are
/// unused padding.
#[cfg(feature = "f32")]
pub type ContactId = u32;
/// See [`ContactId`].
#[cfg(feature = "f64")]
pub type ContactId = u64;

/// Bit set in [`SolverContactGeneric::contact_id`] when the contact did not exist
/// during the previous timestep.
pub const NEW_CONTACT_BIT: ContactId = 1 << 31;

// One scalar `SolverContact` reinterpreted as fixed 128-bit blocks for the
// AoS↔SoA gather. The blocks are always 4-wide (`SolverBlock`), independent of
// `SIMD_WIDTH`, so this holds at both 4 and 8 lanes.
#[repr(C)]
#[repr(align(16))]
pub struct SimdSolverContactRepr {
    data0: SolverBlock,
    data1: SolverBlock,
    #[cfg(feature = "dim3")]
    data2: SolverBlock,
}

// NOTE: if these assertion fail with a weird "0 - 1 would overflow" error, it means the equality doesn’t hold.
static_assertions::const_assert_eq!(
    align_of::<SimdSolverContactRepr>(),
    align_of::<SolverContact>()
);
static_assertions::assert_eq_size!(SimdSolverContactRepr, SolverContact);
// The SoA gather result is at least as aligned as the AoS lane array (equal at 4
// lanes; at 8 lanes `SimdReal` is 32-byte-aligned while the scalar array is 16).
static_assertions::const_assert_eq!(
    align_of::<SimdSolverContact>() % align_of::<[SolverContact; SIMD_WIDTH]>(),
    0
);
static_assertions::assert_eq_size!(SimdSolverContact, [SolverContact; SIMD_WIDTH]);

impl SimdSolverContact {
    /// Gathers one solver contact per lane, at a per-lane index (the lanes of a
    /// constraint chunk may have different active-contact counts, so callers
    /// clamp each lane's index to its own count).
    ///
    /// # Safety
    ///
    /// Every `ks[k]` must be a valid index into `contacts[k]` — the gather reads each
    /// lane's slice unchecked.
    pub unsafe fn gather_unchecked(
        contacts: &[&[SolverContact]; SIMD_WIDTH],
        ks: [usize; SIMD_WIDTH],
    ) -> Self {
        // TODO PERF: double-check that the compiler is using simd loads and
        //       isn’t generating useless copies.

        let data_repr: &[&[SimdSolverContactRepr]; SIMD_WIDTH] =
            unsafe { core::mem::transmute(contacts) };
        use crate::utils::transpose_wide;

        // One 128-bit block per lane, gathered at each lane's own `ks` index.
        let aos0: [_; SIMD_WIDTH] =
            core::array::from_fn(|k| unsafe { data_repr[k].get_unchecked(ks[k]).data0.0 });
        let aos1: [_; SIMD_WIDTH] =
            core::array::from_fn(|k| unsafe { data_repr[k].get_unchecked(ks[k]).data1.0 });
        let soa0 = transpose_wide(aos0);
        let soa1 = transpose_wide(aos1);

        #[cfg(feature = "dim2")]
        unsafe {
            core::mem::transmute::<[[SimdReal; 4]; 2], SimdSolverContact>([soa0, soa1])
        }

        #[cfg(feature = "dim3")]
        {
            let aos2: [_; SIMD_WIDTH] =
                core::array::from_fn(|k| unsafe { data_repr[k].get_unchecked(ks[k]).data2.0 });
            let soa2 = transpose_wide(aos2);

            unsafe {
                core::mem::transmute::<[[SimdReal; 4]; 3], SimdSolverContact>([soa0, soa1, soa2])
            }
        }
    }
}

impl<N: ScalarType, const LANES: usize> SolverContactGeneric<N, LANES> {
    /// The manifold contact indices, with the is-new bit masked off.
    ///
    /// These indices are only valid within the timestep that produced this solver
    /// contact: manifold points may be reordered or replaced by the next narrow-phase
    /// update.
    #[inline]
    pub fn contact_indices(&self) -> [ContactId; LANES] {
        self.contact_id.map(|id| id & !NEW_CONTACT_BIT)
    }
}

/// Should a contact be treated as bouncy? (SIMD lanes; `1.0` = bouncy.) Restitution is
/// per-manifold ([`ContactManifoldData::restitution`]); `is_new` is decoded from bit 31
/// ([`NEW_CONTACT_BIT`]) of [`SolverContactGeneric::contact_id`].
pub fn is_bouncy_simd(restitution: SimdReal, is_new: SimdReal) -> SimdReal {
    use na::{SimdPartialOrd, SimdValue};

    let one = SimdReal::splat(1.0);
    let zero = SimdReal::splat(0.0);

    // Treat new collisions as bouncing at first, unless we have zero restitution.
    let if_new = one.select(restitution.simd_gt(zero), zero);

    // If the contact is still here one step later, it is now a resting contact.
    // The exception is very high restitutions, which can never rest
    let if_not_new = one.select(restitution.simd_ge(one), zero);

    if_new.select(is_new.simd_ne(zero), if_not_new)
}

/// Scalar variant of [`is_bouncy_simd`].
pub fn is_bouncy(restitution: Real, is_new: bool) -> Real {
    if is_new {
        (restitution > 0.0) as u32 as Real
    } else {
        (restitution >= 1.0) as u32 as Real
    }
}

impl Default for ContactManifoldData {
    fn default() -> Self {
        Self::new(None, None, SolverFlags::empty())
    }
}

impl ContactManifoldData {
    pub(crate) fn new(
        rigid_body1: Option<RigidBodyHandle>,
        rigid_body2: Option<RigidBodyHandle>,
        solver_flags: SolverFlags,
    ) -> ContactManifoldData {
        Self {
            rigid_body1,
            rigid_body2,
            solver_flags,
            solver_color: SOLVER_COLOR_UNCOLORED,
            solver_body_ids: [u32::MAX; 2],
            graph_pos: crate::dynamics::solver::solver_contact_graph::GraphPos::NONE,
            normal: Vector::ZERO,
            solver_contacts: SolverContacts::new(),
            relative_dominance: 0,
            user_data: 0,
            friction: 0.0,
            restitution: 0.0,
            adhesion_force: 0.0,
            adhesion_pressure: 0.0,
            adhesion_budget: None,
            adhesion_extent: 0.0,
        }
    }

    /// Resolves the world-space contact points (one per body surface) of one solver
    /// contact: body-local anchors ([`SolverContactGeneric::anchor1`]) are resolved through
    /// the bodies' current poses (a world-attached side's anchor already is a world point).
    /// The points differ by roughly the separation along the normal; their midpoint is the
    /// effective solver contact point.
    pub fn solver_contact_world_points(
        &self,
        contact: &SolverContact,
        bodies: &crate::dynamics::RigidBodySet,
    ) -> (Vector, Vector) {
        let resolve =
            |anchor: Vector, handle: Option<RigidBodyHandle>, world_attached: bool| match handle
                .filter(|_| !world_attached)
                .and_then(|h| bodies.get(h))
            {
                Some(rb) => rb.pos.position * (rb.mprops.local_mprops.local_com + anchor),
                None => anchor,
            };
        (
            resolve(
                contact.anchor1,
                self.rigid_body1,
                self.relative_dominance > 0,
            ),
            resolve(
                contact.anchor2,
                self.rigid_body2,
                self.relative_dominance < 0,
            ),
        )
    }

    /// Number of actives contacts, i.e., contacts that will be seen by
    /// the constraints solver.
    #[inline]
    pub fn num_active_contacts(&self) -> usize {
        self.solver_contacts.len()
    }

    /// The tangential extent of this manifold's contact patch: the spread of its solver contact
    /// points perpendicular to [`Self::normal`] (a length in 2D, an area in 3D).
    ///
    /// This is what [`Self::adhesion_pressure`] gets multiplied by, and the weight of this
    /// manifold in its [`Self::adhesion_budget`] pool. Manifolds with fewer than 2 (2D) / 3 (3D)
    /// solver contacts — point and line contacts — have zero extent.
    ///
    /// This is the value cached when the contact-modification hook last ran on this manifold
    /// (measured right after the hook returned, so contacts it added, removed or moved count),
    /// not a fresh measurement: it is `0.0` for manifolds of pairs without
    /// [`ActiveHooks::MODIFY_SOLVER_CONTACTS`], and keeps its value while the pair is not
    /// updated (e.g. both bodies asleep).
    ///
    /// For a 3D pair clustered for the solver, the value lives on the clusters (see
    /// [`ContactPair::solver_manifolds`]); the pair's plain manifolds return `0.0`.
    ///
    /// [`ActiveHooks::MODIFY_SOLVER_CONTACTS`]: crate::pipeline::ActiveHooks::MODIFY_SOLVER_CONTACTS
    #[inline]
    pub fn tangential_extent(&self) -> Real {
        self.adhesion_extent
    }
}

/// The world-space effective point of a solver contact (the midpoint of its two anchors).
///
/// Only meaningful while the anchors still hold world-space points, i.e. inside the
/// contact-modification hook and before the narrow-phase localizes them.
#[inline]
fn solver_contact_world_midpoint(contact: &SolverContact) -> Vector {
    (contact.anchor1 + contact.anchor2) * 0.5
}

/// Tangential extent (2D: length) of a set of solver contacts perpendicular to `normal`: the
/// spread of their projections on the contact tangent. Fewer than 2 contacts give `0.0`.
///
/// Reads world-space anchors, so it is only valid before the narrow-phase localizes them (inside
/// the contact-modification hook, or right after it). Assumes `normal` is unit-length.
#[cfg(feature = "dim2")]
pub(crate) fn solver_contacts_tangential_extent(
    normal: Vector,
    contacts: &[SolverContact],
) -> Real {
    use crate::utils::OrthonormalBasis;

    if contacts.len() < 2 {
        return 0.0;
    }

    let tangent = normal.orthonormal_vector();
    let mut min_s = tangent.dot(solver_contact_world_midpoint(&contacts[0]));
    let mut max_s = min_s;
    for contact in &contacts[1..] {
        let s = tangent.dot(solver_contact_world_midpoint(contact));
        min_s = min_s.min(s);
        max_s = max_s.max(s);
    }
    max_s - min_s
}

/// Tangential extent (3D: area) of a set of solver contacts perpendicular to `normal`.
///
/// This is the area of the polygon formed by the contact points projected on the plane orthogonal
/// to `normal` (points ordered by angle about their centroid), which is exact for contact points
/// in convex position — the case produced by the narrow phase. Fewer than 3 contacts give `0.0`.
///
/// Reads world-space anchors, so it is only valid before the narrow-phase localizes them (inside
/// the contact-modification hook, or right after it). Assumes `normal` is unit-length.
///
/// Allocation-free for up to 4 contacts (every manifold the narrow-phase reduces); larger
/// hook-edited sets fall back to a heap buffer. The ordering is an insertion sort over angles
/// computed once each with the crate's `atan2` (libm under `enhanced-determinism`), so the
/// result is deterministic.
#[cfg(feature = "dim3")]
pub(crate) fn solver_contacts_tangential_extent(
    normal: Vector,
    contacts: &[SolverContact],
) -> Real {
    use crate::utils::OrthonormalBasis;

    let n = contacts.len();
    if n < 3 {
        return 0.0;
    }

    let [b1, b2] = normal.orthonormal_basis();
    let project = |c: &SolverContact| {
        let p = solver_contact_world_midpoint(c);
        [b1.dot(p), b2.dot(p), 0.0]
    };

    let mut stack: [[Real; 3]; 4] = [[0.0; 3]; 4];
    let mut heap: Vec<[Real; 3]>;
    let pts: &mut [[Real; 3]] = if n <= stack.len() {
        for (dst, c) in stack.iter_mut().zip(contacts) {
            *dst = project(c);
        }
        &mut stack[..n]
    } else {
        heap = contacts.iter().map(project).collect();
        &mut heap
    };

    let inv_n = 1.0 / n as Real;
    let (mut cx, mut cy): (Real, Real) = (0.0, 0.0);
    for p in pts.iter() {
        cx += p[0];
        cy += p[1];
    }
    cx *= inv_n;
    cy *= inv_n;

    // Angle about the centroid, stored in the third slot and computed once per point. Called
    // through `RealField` (not the inherent `f32::atan2`) so `enhanced-determinism` routes it
    // through libm.
    for p in pts.iter_mut() {
        p[2] = <Real as simba::scalar::RealField>::atan2(p[1] - cy, p[0] - cx);
    }

    // Stable insertion sort by angle (small `n`, no allocation, NaN angles stay in place).
    for i in 1..n {
        let key = pts[i];
        let mut j = i;
        while j > 0 && pts[j - 1][2] > key[2] {
            pts[j] = pts[j - 1];
            j -= 1;
        }
        pts[j] = key;
    }

    let mut twice_area = 0.0;
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        twice_area += p[0] * q[1] - q[0] * p[1];
    }
    let area = twice_area * 0.5;
    if area < 0.0 { -area } else { area }
}

/// Additional methods for the contact manifold.
pub trait ContactManifoldExt {
    /// Computes the sum of all the impulses applied by contacts from this contact manifold.
    fn total_impulse(&self) -> Real;
}

impl ContactManifoldExt for ContactManifold {
    fn total_impulse(&self) -> Real {
        self.points.iter().map(|pt| pt.data.impulse).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::relative_pose_drift;
    use crate::math::{Pose, Real, Vector};

    /// A pose compared against *itself* must report no drift beyond rounding noise (the
    /// `cur ∘ base⁻¹` product is not exactly the identity), whatever the orientation and
    /// however far the shape reaches: it must never eat into the sleep gate's allowance.
    #[test]
    fn same_pose_never_drifts() {
        // The sleep gate's per-step allowance at the default threshold and 60 Hz.
        let allowance = 0.05 * (1.0 / 60.0) * 2.0;
        let max_extent = 4.0;

        let mut worst: Real = 0.0;
        let mut n = 0;
        for i in 0..40 {
            for j in 0..10 {
                let angle = i as Real * 0.157;
                #[cfg(feature = "dim2")]
                let pose = Pose::new(Vector::new(j as Real * 3.7, 1.0), angle);
                #[cfg(feature = "dim3")]
                let pose = Pose::new(
                    Vector::new(j as Real * 3.7, 1.0, -2.0),
                    Vector::new(0.3, -0.7, 0.15).normalize() * angle,
                );
                worst = worst.max(relative_pose_drift(&pose, &pose, max_extent));
                n += 1;
            }
        }
        let rounding_noise = 8.0 * Real::EPSILON * max_extent;
        assert!(
            worst <= rounding_noise,
            "an unmoved pose drifted by up to {worst} over {n} orientations \
             (rounding noise is {rounding_noise}, the sleep gate allows {allowance} per step)"
        );
    }
}
