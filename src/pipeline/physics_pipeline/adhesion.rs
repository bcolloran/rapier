//! Contact adhesion forces accumulated after external forces and before constraint solving.

use crate::alloc_prelude::*;
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::SolverContactGraph;
use crate::dynamics::{RigidBodyHandle, RigidBodySet};
use crate::geometry::contact_patch::contact_patch;
use crate::geometry::{AdhesionBudget, ColliderHandle};
use crate::math::{Real, Vector};
use crate::utils::CrossProduct;
use parry::utils::hashmap::HashMap;

/// Gives a lone point contact the full budget while keeping its weight small beside a face.
const MIN_BUDGET_WEIGHT: Real = 1.0e-3;

#[derive(Default)]
struct BudgetPool {
    total: Real,
    weight: Real,
}

struct Request {
    bodies: [Option<RigidBodyHandle>; 2],
    normal: Vector,
    force: Real,
    point: Vector,
    pressure_force: Real,
    pressure_point: Vector,
    budget: Option<AdhesionBudget>,
    weight: Real,
}

pub(super) fn apply_adhesion(
    bodies: &mut RigidBodySet,
    graph: &SolverContactGraph,
    manifolds: &ManifoldStore<'_>,
) {
    let mut requests = Vec::new();
    let mut pools: HashMap<(ColliderHandle, u32), BudgetPool> = HashMap::default();

    // The graph contains only solver-active manifolds. Include articulated-body contacts
    // from the generic bucket as well as the ordinary rigid-body color buckets.
    for contact_ref in graph
        .buckets()
        .flat_map(|(_, bucket)| bucket)
        .chain(graph.generic())
    {
        let data = &manifolds.get(*contact_ref).data;
        let force = data.adhesion_force.max(0.0);
        let pressure = data.adhesion_pressure.max(0.0);
        let budget = data.adhesion_budget.filter(|budget| budget.total > 0.0);
        if data.solver_contacts.is_empty() || (force == 0.0 && pressure == 0.0 && budget.is_none())
        {
            continue;
        }

        let points = || {
            data.solver_contacts.iter().map(|contact| {
                let (p1, p2) = data.solver_contact_world_points(contact, bodies);
                (p1 + p2) * 0.5
            })
        };
        let point = points().sum::<Vector>() / data.solver_contacts.len() as Real;
        // Measure once per participating manifold, shared by pressure and budget weighting.
        let patch =
            (pressure > 0.0 || budget.is_some()).then(|| contact_patch(data.normal, points()));
        let extent = patch.as_ref().map_or(0.0, |patch| patch.extent);
        let weight = extent.max(MIN_BUDGET_WEIGHT);
        if let Some(budget) = budget {
            let pool = pools.entry((budget.owner, budget.channel)).or_default();
            pool.total = pool.total.max(budget.total);
            pool.weight += weight;
        }
        requests.push(Request {
            bodies: [data.rigid_body1, data.rigid_body2],
            normal: data.normal,
            force,
            point,
            pressure_force: pressure * extent,
            pressure_point: patch.map_or(point, |patch| patch.centroid),
            budget,
            weight,
        });
    }

    // Iterate requests in graph order; hash-map iteration never influences force accumulation.
    for request in requests {
        let mut force = request.force;
        if let Some(budget) = request.budget {
            let pool = &pools[&(budget.owner, budget.channel)];
            force += pool.total * (request.weight / pool.weight);
        }
        if force == 0.0 && request.pressure_force == 0.0 {
            continue;
        }
        for (handle, sign) in [(request.bodies[0], 1.0), (request.bodies[1], -1.0)] {
            let Some(body) = handle.and_then(|handle| bodies.get_mut_internal(handle)) else {
                continue;
            };
            if !body.is_dynamic() {
                continue;
            }
            // Each action/reaction pair acts at the same world point, conserving its wrench.
            // Uniform pressure acts at the area centroid, preserving torque under subdivision.
            for (magnitude, point) in [
                (force, request.point),
                (request.pressure_force, request.pressure_point),
            ] {
                if magnitude == 0.0 {
                    continue;
                }
                let applied = request.normal * (magnitude * sign);
                body.forces.force += applied;
                body.forces.torque += (point - body.mprops.world_com).gcross(applied);
            }
        }
    }
}
