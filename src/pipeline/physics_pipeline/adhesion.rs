//! Contact adhesion: the attractive force that a contact-modification hook requests through
//! [`ContactModificationContext::adhesion_force`](crate::pipeline::ContactModificationContext::adhesion_force)
//! and its two composition-invariant forms.

use crate::alloc_prelude::*;

use crate::dynamics::RigidBodySet;
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::SolverContactGraph;
use crate::geometry::{ColliderHandle, ContactManifold, NEW_CONTACT_BIT, SolverContact};
use crate::math::{Real, Vector};
use crate::utils::CrossProduct;

/// The smallest weight a manifold can have in an adhesion pool. It keeps point contacts, whose
/// contact patch has no extent, in their pool: a pool that contains one point contact only
/// receives the full total, but a degenerate manifold next to real patches receives almost
/// nothing.
const MIN_POOL_WEIGHT: Real = 1.0e-3;

/// An adhesion pool: the manifolds enrolled with one `(owner, channel)` key share `total`,
/// divided between them in proportion to their weights.
struct Pool {
    key: (ColliderHandle, u32),
    total: Real,
    weight_sum: Real,
}

/// Applies the adhesion that the contact-modification hook requested, as an external force on
/// both bodies of each manifold.
///
/// It must run after the effective force of every body is computed and before the contact solver
/// reads it. The contacts stay push-only, thus they supply the reaction that holds the bodies
/// together, the threshold at which they separate, and the friction.
///
/// The manifolds are visited in the order of the solver contact graph, which is the same in a
/// serial and in a parallel build. The first pass builds the adhesion pools. The narrow phase
/// says whether any hook made a request at all, thus a simulation that uses no adhesion never
/// reaches this function.
pub(super) fn apply_contact_adhesion(
    graph: &SolverContactGraph,
    store: &ManifoldStore,
    bodies: &mut RigidBodySet,
) {
    let manifolds = || {
        graph
            .buckets()
            .flat_map(|(_, refs)| refs.iter())
            .chain(graph.generic())
            .filter(|r| !r.is_padding())
            .map(|r| store.get(*r))
            .filter(|manifold| !manifold.data.solver_contacts.is_empty())
    };

    // Empty until a manifold enrolls in a pool, thus it allocates nothing in the common case.
    let mut pools: Vec<Pool> = Vec::new();
    let mut any_request = false;

    for manifold in manifolds() {
        let data = &manifold.data;
        any_request |= data.adhesion_force > 0.0 || data.adhesion_pressure > 0.0;
        if let Some(budget) = data.adhesion_budget.filter(|budget| budget.total > 0.0) {
            any_request = true;
            let key = (budget.owner, budget.channel);
            let weight = tangential_extent(manifold).max(MIN_POOL_WEIGHT);
            match pools.iter_mut().find(|pool| pool.key == key) {
                // The total of a pool is the largest of the requests, not their sum, so that
                // colliders that overlap cannot increase it.
                Some(pool) => {
                    pool.total = pool.total.max(budget.total);
                    pool.weight_sum += weight;
                }
                None => pools.push(Pool {
                    key,
                    total: budget.total,
                    weight_sum: weight,
                }),
            }
        }
    }

    if !any_request {
        return;
    }

    for manifold in manifolds() {
        let data = &manifold.data;
        let mut force = data.adhesion_force.max(0.0);
        let budget = data.adhesion_budget.filter(|budget| budget.total > 0.0);
        if data.adhesion_pressure > 0.0 || budget.is_some() {
            let extent = tangential_extent(manifold);
            if data.adhesion_pressure > 0.0 {
                force += data.adhesion_pressure * extent;
            }
            if let Some(budget) = budget {
                let key = (budget.owner, budget.channel);
                if let Some(pool) = pools.iter().find(|pool| pool.key == key) {
                    // The shares of the manifolds of a pool add up to its total.
                    force += pool.total / pool.weight_sum * extent.max(MIN_POOL_WEIGHT);
                }
            }
        }
        if force.is_nan() || force <= 0.0 {
            continue;
        }

        // The normal points from the first collider toward the second one, thus a force along
        // it on the first body, and against it on the second one, pulls them together.
        let per_contact = data.normal * (force / data.solver_contacts.len() as Real);
        // A body that the solver treats as world-attached because of its dominance group
        // receives no contact reaction, thus pulling it would drag its partner through the
        // contact.
        for (handle, world_attached, first) in [
            (data.rigid_body1, data.relative_dominance > 0, true),
            (data.rigid_body2, data.relative_dominance < 0, false),
        ] {
            if world_attached {
                continue;
            }
            let Some(rb) = handle.and_then(|handle| bodies.get_mut_internal(handle)) else {
                continue;
            };
            if !rb.is_dynamic() || rb.is_sleeping() {
                continue;
            }
            let force = if first { per_contact } else { -per_contact };
            for contact in &data.solver_contacts {
                rb.forces.force += force;
                rb.forces.torque += solver_arm(manifold, contact, first).gcross(force);
            }
        }
    }
}

/// The lever arm the solver uses for `contact` on the first body of `manifold` (when `first`) or
/// on its second body: the contact point relative to the center of mass, in world space, frozen
/// at the pair's last full update (see [`ContactData::solver_dp1`](crate::geometry::ContactData::solver_dp1)).
fn solver_arm(manifold: &ContactManifold, contact: &SolverContact, first: bool) -> Vector {
    let data = &manifold.points[(contact.contact_id[0] & !NEW_CONTACT_BIT) as usize].data;
    if first {
        data.solver_dp1
    } else {
        data.solver_dp2
    }
}

/// The extent of the contact patch of `manifold`, measured perpendicular to its normal: the
/// spread of its solver contact points in 2D.
///
/// A manifold with less than 2 solver contacts is a point contact. Its extent is `0.0`.
#[cfg(feature = "dim2")]
fn tangential_extent(manifold: &ContactManifold) -> Real {
    use crate::utils::OrthonormalBasis;

    let contacts = &manifold.data.solver_contacts;
    if contacts.len() < 2 {
        return 0.0;
    }

    let tangent = manifold.data.normal.orthonormal_vector();
    let mut min = tangent.dot(solver_arm(manifold, &contacts[0], true));
    let mut max = min;
    for contact in &contacts[1..] {
        let extent = tangent.dot(solver_arm(manifold, contact, true));
        min = min.min(extent);
        max = max.max(extent);
    }
    max - min
}

/// The extent of the contact patch of `manifold`, measured perpendicular to its normal: the area
/// of the polygon that its solver contact points make in 3D.
///
/// The points are projected on the plane perpendicular to the normal and put in the order of
/// their angle about their centroid. This is the exact area for points in convex position, which
/// is what the narrow phase produces. A manifold with less than 3 solver contacts is a point or
/// a line contact. Its extent is `0.0`.
#[cfg(feature = "dim3")]
fn tangential_extent(manifold: &ContactManifold) -> Real {
    use crate::math::MAX_MANIFOLD_POINTS;
    use crate::utils::OrthonormalBasis;

    let contacts = &manifold.data.solver_contacts;
    let len = contacts.len();
    if len < 3 {
        return 0.0;
    }

    let [b1, b2] = manifold.data.normal.orthonormal_basis();
    // The third value of each point is its angle about the centroid, filled in below.
    let project = |contact: &SolverContact| {
        let point = solver_arm(manifold, contact, true);
        [b1.dot(point), b2.dot(point), 0.0]
    };
    // The narrow phase never makes more than `MAX_MANIFOLD_POINTS` solver contacts. Only a hook
    // that adds contacts of its own needs the heap.
    let mut inline = [[0.0; 3]; MAX_MANIFOLD_POINTS];
    let mut heap;
    let points: &mut [[Real; 3]] = if len <= MAX_MANIFOLD_POINTS {
        for (slot, contact) in inline.iter_mut().zip(contacts) {
            *slot = project(contact);
        }
        &mut inline[..len]
    } else {
        heap = contacts.iter().map(project).collect::<Vec<_>>();
        &mut heap
    };

    let inv_len = 1.0 / len as Real;
    let center = [
        points.iter().map(|p| p[0]).sum::<Real>() * inv_len,
        points.iter().map(|p| p[1]).sum::<Real>() * inv_len,
    ];
    for point in points.iter_mut() {
        // Through `RealField` and not the inherent `atan2`, so that `enhanced-determinism`
        // routes it to `libm`.
        point[2] =
            <Real as simba::scalar::RealField>::atan2(point[1] - center[1], point[0] - center[0]);
    }

    // Insertion sort by angle: the point count is small and this needs no allocation.
    for i in 1..len {
        let key = points[i];
        let mut j = i;
        while j > 0 && points[j - 1][2] > key[2] {
            points[j] = points[j - 1];
            j -= 1;
        }
        points[j] = key;
    }

    let mut twice_area = 0.0;
    for (i, p) in points.iter().enumerate() {
        let q = points[(i + 1) % len];
        twice_area += p[0] * q[1] - q[0] * p[1];
    }
    (twice_area * 0.5).abs()
}
