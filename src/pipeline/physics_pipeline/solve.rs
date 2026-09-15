//! Per-substep pipeline stages: broad/narrow-phase collision detection and the
//! island build + staged velocity constraint solve.

use crate::alloc_prelude::*;

use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::SolverContactGraph;
use crate::dynamics::{
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, RigidBodySet,
};
use crate::geometry::{BroadPhaseBvh, ColliderHandle, ColliderSet, NEW_CONTACT_BIT, NarrowPhase};
use crate::math::{Real, Vector};
use crate::pipeline::{EventHandler, PhysicsHooks};
use crate::utils::CrossProduct;

use super::PhysicsPipeline;

/// What one parallel chunk of the body-update pass reduces to: whether any of its bodies
/// asked for extra solver iterations, its best island-split bid (score, island id), and its
/// sleep observations.
#[cfg(feature = "parallel")]
type BodyUpdateChunkResult = (bool, Option<(Real, u32)>, Vec<(u32, bool)>);

/// The narrow-phase's per-body solver-color-mask slice, type-erased (the pointer is held as a
/// `usize`) so it stays holdable across the exclusive narrow-phase borrow of the solver scope.
#[derive(Copy, Clone)]
struct ErasedColorMasks {
    ptr: usize,
    len: usize,
}

impl ErasedColorMasks {
    fn erase(masks: &[u128]) -> Self {
        Self {
            ptr: masks.as_ptr() as usize,
            len: masks.len(),
        }
    }

    /// # Safety
    /// The erased slice must still be live, and unmutated since [`Self::erase`].
    unsafe fn as_slice<'a>(self) -> &'a [u128] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const u128, self.len) }
    }
}

impl PhysicsPipeline {
    pub(super) fn detect_collisions(
        &mut self,
        integration_parameters: &IntegrationParameters,
        islands: &mut IslandManager,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        impulse_joints: &ImpulseJointSet,
        multibody_joints: &MultibodyJointSet,
        modified_colliders: &[ColliderHandle],
        removed_colliders: &[ColliderHandle],
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
        handle_user_changes: bool,
    ) {
        self.counters.stages.collision_detection_time.resume();
        self.counters.cd.broad_phase_time.resume();

        // A tree-optimization pass from a previous call may still be pending.
        self.join_deferred_bvh_optimize(broad_phase);

        // Update broad-phase.
        self.broad_phase_events.clear();
        broad_phase.update(
            integration_parameters,
            colliders,
            bodies,
            modified_colliders,
            removed_colliders,
            &mut self.broad_phase_events,
        );

        // Run the update's deferred (quality-only) tree optimization on another thread
        // while the narrow phase and solver don't need the tree;
        // joined by `join_deferred_bvh_optimize` before next use.
        if let Some(task) = broad_phase.take_deferred_optimize() {
            // Deferring is unconditional — the tree this step's pair traversal already
            // walked must stay un-optimized until the join, in every build — but the
            // *execution* needs a spare worker. `step` itself runs inside the pool (see
            // `PhysicsPipeline::step`), so on a single-worker pool a detached task would
            // queue behind the `recv` waiting for it: deadlock. Hand those to the join
            // point instead, which runs them inline.
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 {
                let (tx, rx) = std::sync::mpsc::channel();
                let mut task = task;
                rayon::spawn(move || {
                    task.run();
                    let _ = tx.send(task);
                });
                *self.deferred_bvh.get_mut().unwrap() = Some(rx);
            } else {
                self.deferred_bvh_inline = Some(task);
            }

            #[cfg(not(feature = "parallel"))]
            {
                self.deferred_bvh_inline = Some(task);
            }
        }

        self.counters.cd.broad_phase_time.pause();
        self.counters.cd.narrow_phase_time.resume();

        // Update narrow-phase.
        if handle_user_changes {
            narrow_phase.handle_user_changes(
                Some(islands),
                modified_colliders,
                removed_colliders,
                colliders,
                bodies,
                events,
            );
        }
        narrow_phase.register_pairs(
            Some(islands),
            colliders,
            bodies,
            &self.broad_phase_events,
            events,
        );
        narrow_phase.compute_contacts(
            integration_parameters.prediction_distance(),
            integration_parameters.dt,
            integration_parameters.contact_clustering,
            if integration_parameters.contact_recycling {
                integration_parameters.contact_recycle_distance()
            } else {
                0.0
            },
            islands,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            modified_colliders,
            hooks,
            events,
        );
        narrow_phase.compute_intersections(
            islands,
            bodies,
            colliders,
            modified_colliders,
            hooks,
            events,
        );

        self.counters.cd.narrow_phase_time.pause();
        self.counters.stages.collision_detection_time.pause();
    }

    pub(super) fn build_islands_and_solve_velocity_constraints(
        &mut self,
        gravity: Vector,
        integration_parameters: &IntegrationParameters,
        islands: &mut IslandManager,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        impulse_joints: &mut ImpulseJointSet,
        multibody_joints: &mut MultibodyJointSet,
        events: &dyn EventHandler,
    ) {
        // Persistent islands, two tiers: a bounded local dual search settles each removal (proves
        // connectivity — common case, island never marked dirty — or peels the detached side at its cost);
        // the rest falls to the deferred union-find split (one island/step), run before the fused traversal so split bids see the post-split state.
        self.counters.stages.island_construction_time.resume();
        islands.persistent.resolve_removals(
            bodies,
            colliders,
            narrow_phase,
            impulse_joints,
            multibody_joints,
            integration_parameters.length_unit,
        );
        islands.persistent.run_pending_split(bodies);
        self.counters.stages.island_construction_time.pause();

        // Single fused traversal of the active bodies: sleep-energy/candidacy update (must
        // run after the narrow-phase wake-ups, before the sleep traversals below) +
        // effective external forces. Pass cost is dominated by body cache lines, not math.
        self.counters.stages.update_time.resume();
        // OR-reduction over the active bodies: does any awake body request
        // extra substeps? Gates the substep-group partition below.
        let mut any_extra_iterations = false;
        // Persistent-island split-candidate bid: the sleepiest body whose island has
        // pending removals nominates it for next step's (single) split.
        // `(sleepiness, island id)`, ties toward the larger id.
        let mut split_bid: Option<(Real, u32)> = None;
        // Deterministic bid reduction: max score wins; ties break toward the larger
        // island id.
        fn better_bid(best: &mut Option<(Real, u32)>, score: Real, island_id: u32) {
            match *best {
                Some((s, id)) if score < s || (score == s && island_id <= id) => {}
                _ => *best = Some((score, island_id)),
            }
        }
        // Sleep observation for the whole-island decision, run-length
        // compressed: consecutive bodies of the same island fold into one
        // `(island id, all eligible so far)` entry.
        let observe = |rb: &crate::dynamics::RigidBody, out: &mut Vec<(u32, bool)>| {
            let island_id = rb.ids.island_id;
            if island_id == crate::dynamics::INVALID_ISLAND {
                return;
            }
            let eligible = rb.activation.is_eligible_for_sleep();
            match out.last_mut() {
                Some((last_id, last_eligible)) if *last_id == island_id => {
                    *last_eligible &= eligible;
                }
                _ => out.push((island_id, eligible)),
            }
        };
        let bid = |rb: &crate::dynamics::RigidBody,
                   persistent: &crate::dynamics::PersistentIslands,
                   best: &mut Option<(Real, u32)>| {
            if rb.activation.is_eligible_for_sleep() {
                let island_id = rb.ids.island_id;
                if island_id != crate::dynamics::INVALID_ISLAND
                    && persistent.split_allowed(island_id)
                {
                    let score = rb.activation.time_since_can_sleep;
                    better_bid(best, score, island_id);
                }
            }
        };
        self.sleep_observations.clear();
        #[cfg(not(feature = "parallel"))]
        {
            let dt = integration_parameters.dt;
            let length_unit = integration_parameters.length_unit;
            let observations = &mut self.sleep_observations;
            for handle in islands.active_bodies() {
                let rb = bodies.index_mut_internal(handle);
                IslandManager::update_body_energy(rb, dt, length_unit);
                let effective_mass = rb.mprops.effective_mass();
                rb.forces
                    .compute_effective_force_and_torque(gravity, effective_mass);
                any_extra_iterations |= rb.additional_solver_iterations() > 0;
                bid(rb, &islands.persistent, &mut split_bid);
                observe(rb, observations);
            }
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let dt = integration_parameters.dt;
            let length_unit = integration_parameters.length_unit;
            self.active_body_handles.clear();
            self.active_body_handles.extend(islands.active_bodies());
            let bodies_ptr = core::sync::atomic::AtomicPtr::new(bodies as *mut RigidBodySet);
            let persistent = &islands.persistent;
            let chunk_results: Vec<BodyUpdateChunkResult> = self
                .active_body_handles
                .par_chunks(256)
                .map(|chunk| {
                    // SAFETY: every body handle is distinct, so the mutated bodies are disjoint.
                    let bodies =
                        unsafe { &mut *bodies_ptr.load(core::sync::atomic::Ordering::Relaxed) };
                    let mut any_extra = false;
                    let mut chunk_bid = None;
                    let mut observations = Vec::new();
                    for handle in chunk {
                        let rb = bodies.index_mut_internal(*handle);
                        IslandManager::update_body_energy(rb, dt, length_unit);
                        let effective_mass = rb.mprops.effective_mass();
                        rb.forces
                            .compute_effective_force_and_torque(gravity, effective_mass);
                        any_extra |= rb.additional_solver_iterations() > 0;
                        bid(rb, persistent, &mut chunk_bid);
                        observe(rb, &mut observations);
                    }
                    (any_extra, chunk_bid, observations)
                })
                .collect();
            // Chunks are collected in order, so the reduction stays deterministic.
            for (any_extra, chunk_bid, observations) in &chunk_results {
                any_extra_iterations |= any_extra;
                if let Some((score, island_id)) = *chunk_bid {
                    better_bid(&mut split_bid, score, island_id);
                }
                self.sleep_observations.extend_from_slice(observations);
            }
        }
        // Promote the winning bid to next step's pending split.
        if let Some((_, island_id)) = split_bid {
            islands.persistent.schedule_split(island_id);
        }
        self.counters.stages.update_time.pause();

        self.counters.stages.island_construction_time.resume();
        // NOTE: islands update must be done after the narrow-phase.
        islands.update_islands(
            bodies,
            colliders,
            narrow_phase,
            impulse_joints,
            multibody_joints,
            &self.sleep_observations,
        );

        // Substep solve-groups: partition the awake body list by effective
        // `additional_solver_iterations`. Must run after `update_islands` (last mutator of the
        // awake list) and before the maintenance below (consumes the body order). No elevated body => one branch.
        islands.update_substep_groups(
            any_extra_iterations,
            bodies,
            narrow_phase,
            impulse_joints,
            multibody_joints,
        );

        self.counters.stages.island_construction_time.pause();

        self.counters
            .stages
            .island_constraints_collection_time
            .resume();
        // Per-body contact-color masks for coloring joints in the contacts' color space, captured
        // type-erased before the narrow-phase is mutably borrowed for the solver scope.
        // SAFETY (used below): nothing mutates the narrow-phase while the solver runs.
        let contact_color_masks = ErasedColorMasks::erase(narrow_phase.body_solver_color_masks());

        // Raw parts of the solver-facing manifold store, captured before the shared graph borrow
        // below. SAFETY (used below): the contact graph is not mutated while the solvers run.
        let manifold_store_parts = narrow_phase.manifold_store_parts();
        // Incrementally reconcile the persistent per-color solver contact graph with this
        // step's changed contacts. The assemblies consume the buckets directly — nothing
        // is collected, selected or sorted per step.
        narrow_phase.maintain_solver_contact_graph(islands, bodies, colliders, multibody_joints);
        if !self.joint_selection_primed {
            impulse_joints.invalidate_selection_memo();
            self.joint_selection_primed = true;
        }
        impulse_joints.select_active_interactions(
            islands,
            bodies,
            &mut self.joint_constraint_indices,
        );
        self.counters
            .stages
            .island_constraints_collection_time
            .pause();

        // NOTE: world-space mass-properties are NOT recomputed before the solver: they were
        // refreshed by `advance_to_final_positions`, the user-changes handler, or multibody forward
        // kinematics; effective forces by the fused traversal above.
        self.counters.stages.solver_time.resume();

        // Manifold store: raw ContactRef resolution for constraint generation and impulse
        // writeback. SAFETY: parts captured above; the contact graph is not mutated for
        // the rest of the step (solver scope).
        let manifold_store = unsafe {
            crate::dynamics::solver::manifold_store::ManifoldStore::from_parts(manifold_store_parts)
        };

        // Solve the single awake island. The staged solver is the only solver: a parallel
        // build fans the colored constraints across `num_threads` workers; otherwise
        // `num_threads` = 1 and it runs inline, skipping all cross-worker coordination.
        if let Some(island_id) = islands.awake_island {
            #[cfg(feature = "parallel")]
            let num_threads = rayon::current_num_threads();
            #[cfg(not(feature = "parallel"))]
            let num_threads = 1;

            // Contact adhesion requested by the hooks: added to the effective forces the fused
            // traversal assigned above, right before the solver reads them.
            apply_contact_adhesion(
                &mut self.adhesion_pools,
                narrow_phase.solver_graph(),
                &manifold_store,
                bodies,
            );

            let joint_assembly_epoch = impulse_joints.assembly_epoch;
            self.staged_solver.init_and_solve(
                num_threads,
                island_id,
                &mut self.counters,
                integration_parameters,
                islands,
                bodies,
                narrow_phase.solver_graph(),
                &manifold_store,
                impulse_joints.joints_mut(),
                &self.joint_constraint_indices,
                joint_assembly_epoch,
                multibody_joints,
                unsafe { contact_color_masks.as_slice() },
            );
        }

        // Generate contact force events if needed, and update each pair's
        // above-threshold status (the source of `ContactForceEvent::started`).
        narrow_phase.emit_contact_force_events(
            integration_parameters.dt,
            bodies,
            colliders,
            events,
        );

        self.counters.stages.solver_time.pause();
    }
}

/// Weight floor for adhesion budget distribution: keeps point contacts (zero tangential extent)
/// in their pool — a lone point contact receives the full budget, while a degenerate sliver
/// manifold alongside real area contacts receives almost nothing.
const ADHESION_BUDGET_MIN_WEIGHT: Real = 1.0e-3;

/// Scratch adhesion budget pools: `((owner, channel), (pool total, sum of member weights))`.
pub(super) type AdhesionPools = Vec<((ColliderHandle, u32), (Real, Real))>;

/// Whether an adhesion value counts: positive and finite. A non-finite request adds nothing.
#[inline]
fn adhesion_term_applies(value: Real) -> bool {
    value > 0.0 && value.is_finite()
}

/// Applies the adhesion requested through `PhysicsHooks::modify_solver_contacts` (stored on each
/// manifold by the narrow phase) as external forces on the bodies, right before the contact
/// solver reads them ("Design A": the unchanged push-only contacts then produce the holding
/// reaction, the break threshold and friction).
///
/// Must run after the fused traversal assigned the effective forces, so nothing accumulates.
/// A body woken by `update_islands`' first-step bootstrap after a legacy-snapshot restore missed
/// that traversal; its effective force is stale this step, adhesion included.
///
/// Visits exactly the solver-active manifolds — the solver contact graph's color buckets in
/// ascending color order, then the generic (multibody) list, resolved through the manifold store
/// (so 3D clusters, not their plain manifolds) — in an order that is identical in serial and
/// parallel builds and survives snapshots. Serial; the first pass builds the budget pools and
/// returns early when nothing requests adhesion.
///
/// For each manifold with `k` solver contacts, the magnitude `F` is the sum of those of the terms
/// `force`, `pressure · extent` and `pool_total · weight / pool_weight_sum` that are positive and
/// finite. A non-finite term adds nothing and doesn't discard the other terms of its manifold, and a
/// budget whose total is not positive and finite joins no pool. A manifold whose `F` is not positive
/// and finite (nothing requested, or an overflowing sum) writes nothing. Otherwise `F`
/// is split into `k` equal forces `f = normal · F / k`, one per solver contact: `+f` on the
/// first body and `-f` on the second (the normal points from collider1 toward collider2, so
/// this pulls them together), each with the torque of the solver's own frozen lever arm
/// (`ContactData::solver_dp1`/`solver_dp2`).
///
/// Only awake dynamic bodies are pulled, and never a side the solver treats as world-attached
/// through dominance (`relative_dominance > 0` for the first body, `< 0` for the second, the
/// convention of the narrow-phase localization and the constraint builders): that side gets no
/// contact reaction, so pulling it would drag the dominant body through the contact.
fn apply_contact_adhesion(
    pools: &mut AdhesionPools,
    graph: &SolverContactGraph,
    store: &ManifoldStore,
    bodies: &mut RigidBodySet,
) {
    let refs = || {
        graph
            .buckets()
            .flat_map(|(_, refs)| refs.iter())
            .chain(graph.generic().iter())
            .filter(|r| !r.is_padding())
    };

    pools.clear();
    let mut any_request = false;
    for r in refs() {
        let data = &store.get(*r).data;
        if data.solver_contacts.is_empty() {
            continue;
        }
        any_request |= data.adhesion_force > 0.0 || data.adhesion_pressure > 0.0;
        if let Some(budget) = &data.adhesion_budget {
            if adhesion_term_applies(budget.total) {
                any_request = true;
                let key = (budget.owner, budget.channel);
                let weight = data.adhesion_extent.max(ADHESION_BUDGET_MIN_WEIGHT);
                match pools.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, (total, weight_sum))) => {
                        *total = total.max(budget.total);
                        *weight_sum += weight;
                    }
                    None => pools.push((key, (budget.total, weight))),
                }
            }
        }
    }
    if !any_request {
        return;
    }

    for r in refs() {
        let manifold = store.get(*r);
        let data = &manifold.data;
        let num_contacts = data.solver_contacts.len();
        if num_contacts == 0 {
            continue;
        }

        // Each term counts on its own, only if it is positive and finite, so a non-finite request
        // (an infinite pressure on a zero extent is NaN) can't discard a valid term.
        let mut magnitude: Real = 0.0;
        if adhesion_term_applies(data.adhesion_force) {
            magnitude += data.adhesion_force;
        }
        let pressure_force = data.adhesion_pressure * data.adhesion_extent;
        if adhesion_term_applies(pressure_force) {
            magnitude += pressure_force;
        }
        if let Some(budget) = &data.adhesion_budget {
            if adhesion_term_applies(budget.total) {
                let key = (budget.owner, budget.channel);
                if let Some((_, (total, weight_sum))) = pools.iter().find(|(k, _)| *k == key) {
                    let weight = data.adhesion_extent.max(ADHESION_BUDGET_MIN_WEIGHT);
                    // This manifold's share; the shares of a pool sum to its total.
                    let share = (*total / *weight_sum) * weight;
                    if adhesion_term_applies(share) {
                        magnitude += share;
                    }
                }
            }
        }
        // Nothing requested (or an overflowing sum): write nothing, so stock results stay
        // bit-identical.
        if !adhesion_term_applies(magnitude) {
            continue;
        }

        let per_contact = data.normal * (magnitude / num_contacts as Real);
        let rel_dom = data.relative_dominance;
        for (handle, world_attached, first) in [
            (data.rigid_body1, rel_dom > 0, true),
            (data.rigid_body2, rel_dom < 0, false),
        ] {
            if world_attached {
                continue;
            }
            let Some(rb) = handle.and_then(|h| bodies.get_mut_internal(h)) else {
                continue;
            };
            if !rb.is_dynamic() || rb.is_sleeping() {
                continue;
            }
            let force = if first { per_contact } else { -per_contact };
            for contact in &data.solver_contacts {
                let cid = (contact.contact_id[0] & !NEW_CONTACT_BIT) as usize;
                let point = &manifold.points[cid].data;
                let arm = if first {
                    point.solver_dp1
                } else {
                    point.solver_dp2
                };
                rb.forces.force += force;
                rb.forces.torque += arm.gcross(force);
            }
        }
    }
}
