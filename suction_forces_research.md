# Suction-cup contact forces in Rapier — research findings

> **Purpose.** Assess whether attractive ("suction cup") forces can be applied between two bodies at their contact points, driven from inside a `modify_solver_contacts` physics hook, integrated in the *same* step, reusing data already present in the contact-modification context.
>
> **Scope.** 2D (`dim2`). Most of the analysis applies to 3D too, but code references and design decisions are framed for 2D.
>
> **Status.** Research + design exploration for review. Nothing here is implemented. All file/line links resolve from the repository root.

---

## 0. TL;DR

- **Doing this through the stock contact-modification API alone is not possible.** Three independent blockers (all confirmed in code): the normal contact impulse is hard-clamped to be non-negative (push-only); the friction cone is defined as `friction × normal_impulse`, so a pulling normal force would break it; and the hook is handed an *immutable* `&RigidBodySet`, so it cannot apply impulses.
- **It is achievable with a contained engine change.** The decisive fact is *timing*: the hook runs during the narrow phase, and external forces are summed into each body in a window *after* the narrow phase but *before* the solver. So the engine can convert hook-provided suction into body forces in that window and have it integrated the same step.
- **Two designs are viable.** 
**Design A** (recommended): the engine applies *your* force at the contact points via the ordinary external-force path; the still-push-only contact reacts, yielding holding behaviour, a natural break threshold, and friction — with zero changes to the solver core.
**Design B**: change the contact solver's normal clamp from `≥ 0` to `≥ −A` so the contact itself can pull up to a limit `A`; more physically faithful (no preload) but it edits the delicate clamp, friction cone, and the 2-contact block solver.
- **Do not add fields to `SolverContact`.** It is SIMD-packed with compile-time size/alignment assertions and hand-written transpose code. New suction data belongs on [`ContactManifoldData`](src/geometry/contact_pair.rs#L312-L348) instead.

---

## 1. How Rapier represents contacts

This section answers a question raised during review: *if you only have "a manifold", where is the world point at which an impulse would be applied? There is no apply-impulse-across-manifold function.* That instinct is correct, and here is the precise data model.

### 1.1 Manifolds, points, and solver contacts

- A **`ContactManifold`** describes contact between **one pair of collider features**. It is the parry type `ContactManifold<ContactManifoldData, ContactData>` (re-exported in [`src/geometry/mod.rs`](src/geometry/mod.rs)). A pair of simple convex shapes typically produces **one** manifold; compound/heightfield/trimesh shapes produce **several** (one per colliding sub-piece).
- The manifold carries a single shared **world-space normal** and a list of **contact points**. The solver's per-point view lives in [`ContactManifoldData::solver_contacts`](src/geometry/contact_pair.rs#L343).
- A **`SolverContact`** is the solver's per-point record, and it **already carries its own world-space point**. See [`SolverContactGeneric`](src/geometry/contact_pair.rs#L372-L405):

```rust
pub struct SolverContactGeneric<N: ScalarType, const LANES: usize> {
    pub point: N::Vector,            // world-space contact point (midpoint of the two surface points)
    pub dist: N,                     // signed gap along the normal (negative = penetration)
    pub friction: N,                 // per-point
    pub restitution: N,              // per-point
    pub tangent_velocity: N::Vector, // per-point (conveyor belts etc.)
    pub warmstart_impulse: N,
    pub warmstart_tangent_impulse: TangentImpulse<N>,
    pub warmstart_twist_impulse: N,
    pub is_new: N,
    pub contact_id: [u32; LANES],
    // (dim3 only) padding
}
```

The world point is computed in the narrow phase as the midpoint of the two surface points — [`narrow_phase.rs:1052`](src/geometry/narrow_phase.rs#L1052), where each `SolverContact` is built ([`narrow_phase.rs:1054-1072`](src/geometry/narrow_phase.rs#L1054-L1072)).

### 1.2 What this means for "where is the world point?"

There is indeed **no** "manifold world point", and Rapier has no apply-impulse-across-manifold operation. Any force/impulse is applied at a **specific contact point's world position** — and those positions already exist as `solver_contacts[i].point`. Therefore:

- A **per-point** suction model applies a force at `solver_contacts[i].point` for each point `i`. Every application has a concrete world point. This is consistent with how `friction` is already specified per point.
- A **per-manifold** suction model is a convenience: you provide **one** number, and the engine *still* applies it at real world points by distributing it across the manifold's active contact points (e.g. evenly, at each `solver_contacts[i].point`). This is consistent with how the single `normal` is shared per manifold.
  - Distributing the pull across the patch (rather than lumping it at a centroid) also produces a correct **peeling-resisting torque** — a cup resists being levered off at an edge.

So "per-manifold" never means "applied at no point"; it means "one knob, applied at the same per-point world positions under the hood." (Granularity trade-offs in §8.)

---

## 2. The contact-modification API

### 2.1 What the hook receives

[`ContactModificationContext`](src/pipeline/physics_hooks.rs#L23-L46):

```rust
pub struct ContactModificationContext<'a> {
    pub bodies: &'a RigidBodySet,        // IMMUTABLE
    pub colliders: &'a ColliderSet,      // IMMUTABLE
    pub collider1: ColliderHandle,
    pub collider2: ColliderHandle,
    pub rigid_body1: Option<RigidBodyHandle>,
    pub rigid_body2: Option<RigidBodyHandle>,
    pub manifold: &'a ContactManifold,            // read-only
    pub solver_contacts: &'a mut Vec<SolverContact>, // mutable
    pub normal: &'a mut Vector,                    // mutable (per-manifold)
    pub user_data: &'a mut u32,                    // mutable, persists across steps
}
```

The hook is [`PhysicsHooks::modify_solver_contacts`](src/pipeline/physics_hooks.rs#L232-L257). It is only called when a collider in the pair sets [`ActiveHooks::MODIFY_SOLVER_CONTACTS`](src/pipeline/physics_hooks.rs#L147).

**Key limitation:** `bodies` and `colliders` are shared/immutable references. The hook can read body state but **cannot apply impulses or forces** to bodies.

### 2.2 What the hook can change today

Per-point (on each `SolverContact`): `friction`, `restitution`, `tangent_velocity`, `dist`, the warmstart impulses, and the contact set itself (remove/clear/add). Per-manifold: the shared `normal`, and a persistent `user_data: u32`.

Existing precedents for these modifications:
- Conveyor belts via `tangent_velocity`, and one-way platforms via contact removal — [`ContactModificationContext::update_as_oneway_platform`](src/pipeline/physics_hooks.rs#L48-L113), used by [`examples2d/one_way_platforms2.rs`](examples2d/one_way_platforms2.rs).

### 2.3 Where the hook runs (call site)

In the narrow phase, immediately after the solver contacts are generated and before the solver runs — [`narrow_phase.rs:1076-1101`](src/geometry/narrow_phase.rs#L1076-L1101). The context's modifiable copies are written back into `manifold.data` right after the call:

```rust
hooks.modify_solver_contacts(&mut context);
manifold.data.solver_contacts = modifiable_solver_contacts;
manifold.data.normal = modifiable_normal;
manifold.data.user_data = modifiable_user_data;
```

This write-back is exactly where a new suction field would be carried in/out.

---

## 3. The constraint solver internals (the crux)

### 3.1 The normal impulse is hard-clamped to ≥ 0 (push-only)

The per-contact normal solve, in [`ContactConstraintNormalPart::solve`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L262-L287):

```rust
let dvel = /* relative normal velocity */ + self.rhs;
let new_impulse = cfm_factor * (self.impulse - self.r * dvel).simd_max(N::zero());  // <-- clamp >= 0
```

[`contact_constraint_element.rs:278`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L278). The `simd_max(0)` forbids a negative (pulling) accumulated impulse. This is the fundamental unilateral-contact enforcement.

The same non-negativity is baked into the **2-contact block solver** (used when a manifold has two coupled contacts), via the case-selection in [`solve_mlcp_two_constraints`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L289-L319) / [`solve_pair`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L321-L366), e.g. [`contact_constraint_element.rs:308-313`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L308-L313):

```rust
let keep0 = new_impulse0.x.simd_ge(N::zero()) & new_impulse0.y.simd_ge(N::zero());
// ...
```

The generic/multibody path clamps the same way (`.max(0.0)`) in `generic_contact_constraint_element.rs::generic_solve`.

**Consequence:** nothing writable on a `SolverContact` can produce a negative normal impulse. Setting `dist` large/positive merely makes the contact a dormant speculative contact; it never pulls.

### 3.2 Friction is tied to the (non-negative) normal impulse

This confirms the reviewer's specific concern. The friction (tangent) impulse is clamped to a cone whose radius is `friction × normal_impulse`. In 2D, [`ContactConstraintTangentPart::solve`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L131-L159):

```rust
let new_impulse = (self.impulse[0] - self.r[0] * dvel).simd_clamp(-limit, limit);
```

[`contact_constraint_element.rs:150`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L150). The `limit` passed in is `friction × normal_part.impulse` (computed in the constraint solve loop). If `normal_impulse` were negative, `limit` would be negative and `simd_clamp(-limit, limit)` would be inverted/ill-defined. So **any scheme that makes the normal impulse negative must also fix the friction limit.**

### 3.3 The right-hand side: restitution, Baumgarte bias, speculative distance

The normal-part target (`rhs`) is assembled from:
- a restitution term, only when the contact is "bouncy" ([`SolverContact::is_bouncy`](src/geometry/contact_pair.rs#L572-L585)),
- a speculative/separation term `dist.max(0) × inv_dt`, and
- a Baumgarte stabilisation bias clamped to `[-max_corrective_velocity, 0]` (it only ever *removes* penetration, never creates separation).

This assembly lives in the contact-constraint builder/update (`generic_contact_constraint.rs`, normal-part block) and the SIMD variant (`contact_with_coulomb_friction.rs`). The clamp of the bias to `≤ 0` is another reason the existing machinery cannot be coerced into pulling by tweaking `dist`.

### 3.4 Warmstarting tolerates signed impulses

Warmstart simply re-applies the stored `impulse` to the solver velocities — [`ContactConstraintNormalPart::warmstart`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L246-L260). It is sign-agnostic, so a negative impulse (Design B) would warmstart correctly without changes.

### 3.5 `SolverContact` is SIMD-packed — do not add fields to it

`SolverContact` is packed into a fixed number of SIMD lanes with hand-written transpose code and compile-time assertions — [`contact_pair.rs:407-429`](src/geometry/contact_pair.rs#L407-L429), and the warning comment at [`contact_pair.rs:373`](src/geometry/contact_pair.rs#L373) (*"don't change the fields unless `SimdSolverContactRepr` is also changed"*). Adding a field there is expensive and error-prone.

**Conclusion:** any new suction data should live on the plain, non-SIMD [`ContactManifoldData`](src/geometry/contact_pair.rs#L312-L348) (which already holds `solver_contacts`, `normal`, `user_data`) and be surfaced through the hook context — not on `SolverContact`.

---

## 4. Step ordering and the in-step opportunity

The reason a same-step implementation is possible at all is the ordering inside [`PhysicsPipeline::step`](src/pipeline/physics_pipeline.rs):

1. **Narrow phase** builds solver contacts and calls the hook — [`narrow_phase.rs:1076-1101`](src/geometry/narrow_phase.rs#L1076-L1101).
2. **External forces are summed into each body once per step** — the loop at [`physics_pipeline.rs:247-258`](src/pipeline/physics_pipeline.rs#L247-L258) calls [`compute_effective_force_and_torque`](src/dynamics/rigid_body_components.rs#L976-L979), which sets `forces.force = user_force + gravity·mass·scale` and `forces.torque = user_torque`.
3. **The solver runs.** It seeds per-substep velocity increments from `forces.force` / `forces.torque` — [`velocity_solver.rs:118-120`](src/dynamics/solver/velocity_solver.rs#L118-L120) — then iterates the contact/joint constraints.

So there is a window **between step 2 and step 3** where, if the engine adds a force contribution derived from the hook's suction value, it will be integrated **this step**, and the contact constraints (which run afterward) react to it normally. This is the foundation of Design A.

---

## 5. External force / impulse application — what is immediate vs deferred

Relevant to both designs and to the "one-step delay" the reviewer wants to avoid.

- **Accumulated, deferred (per-step):** [`add_force`](src/dynamics/rigid_body.rs#L1186-L1194) and [`add_force_at_point`](src/dynamics/rigid_body.rs#L1242-L1251) add into `forces.user_force` / `forces.user_torque`. These are **persistent** — they are *not* auto-cleared each step; they remain until the user calls [`reset_forces`](src/dynamics/rigid_body.rs#L1126) / [`reset_torques`](src/dynamics/rigid_body.rs#L1140).
- **Immediate:** [`apply_impulse`](src/dynamics/rigid_body.rs#L1280-L1288) and [`apply_impulse_at_point`](src/dynamics/rigid_body.rs#L1341-L1345) modify `vels` directly. These need `&mut RigidBody`, which the hook does **not** have.
- **The per-step effective fields** `forces.force` / `forces.torque` are overwritten every step by [`compute_effective_force_and_torque`](src/dynamics/rigid_body_components.rs#L976-L979). Writing suction into *these* (after step 2 above, before the solver) is inherently transient and avoids interfering with the user's persistent `user_force` bookkeeping. See [`RigidBodyForces`](src/dynamics/rigid_body_components.rs#L916-L931).

This is why a naive "do it yourself" approach has a one-step delay: the hook can only read bodies, so you would record contact data during step N and `apply_impulse_at_point` at the start of step N+1.

---

## 6. Existing bidirectional constraints (for contrast)

Rapier *does* have constraints that pull, but they are joints, not contacts:
- [`SpringJoint`](src/dynamics/joint/spring_joint.rs) is a force-based motor pulling two bodies toward a rest length.
- Joint **motors** use symmetric impulse bounds `[-max_impulse, max_impulse]` (see `joint_constraint_builder.rs`), i.e. they are bidirectional — unlike the contact normal's `[0, ∞)`. The stiffness/damping → `erp`/`cfm` mapping is in [`MotorModel::combine_coefficients`](src/dynamics/joint/motor_model.rs).

These confirm the engine *can* represent attraction, but only outside the contact solver. A `SpringJoint` is a poor fit for suction-at-a-contact: its anchors are fixed local points, it does not track the moving contact point, and it would require creating/destroying joints as contacts come and go. There is **no** public API for a user-defined custom constraint inside the solver.

---

## 7. The two designs

### 7.1 Design A — engine applies your force at the contact points, in-step (recommended)

**Idea.** Don't try to make the *contact constraint* pull. Instead, the engine applies a normal force **you specify** as an ordinary external force (same machinery as gravity), at the contact point(s), in the window from §4. The contact constraint stays push-only and untouched.

**Mechanism (one timestep):**
1. Narrow phase builds `solver_contacts` (each with a world `point`) and calls your hook; you set a suction value on the manifold/context (granularity in §8).
2. Right after the effective-force loop ([`physics_pipeline.rs:247-258`](src/pipeline/physics_pipeline.rs#L247-L258)), a new engine pass iterates active manifolds with suction set and, for each contact point `i`, adds at `solver_contacts[i].point`:
   - to body1: `+f_i · normal`
   - to body2: `−f_i · normal` (equal and opposite),
   plus the matching torque `(point − com) × force`. It also wakes the bodies. **Crucially, it writes the per-step effective `forces.force` / `forces.torque`, not the persistent `user_force`** (see §5), so there is no cross-step accumulation.
3. The solver integrates these forces this step; the push-only contact reacts to the inward pull.

**Why it produces suction-cup behaviour for free:**
- *Holding + break threshold.* The inward pull would drive the surfaces together; the push-only contact pushes back to prevent penetration, so its normal impulse rises to ≈ the pull. The net internal force between the surfaces equals the suction. If an external load tries to separate them with *less* than the suction, the contact stays loaded and they stay stuck. If the load *exceeds* the suction, the net becomes separating, the contact impulse drops to zero, and they detach — a natural break threshold at your force value.
- *Friction emerges automatically.* The friction limit is `friction × normal_impulse`, and `normal_impulse` now includes the reaction to the suction, so a strongly-sucked cup is correspondingly hard to slide. **The friction cone is never touched** (this sidesteps blocker §3.2 entirely).
- *Same-step, no user bookkeeping.* Meets both primary goals.

**Characteristics / caveats:**
- It is a **constant-pull** model: the cup always pulls with force `F`, so there is an internal **preload** (the surfaces are actively squeezed together). Physically that is what a sealed suction cup does; numerically it is stable because the contact balances it exactly. But it does exert force even when nothing is trying to separate the bodies (a real cup at rest is neutral).
- It acts **only where there is an active solver contact** — when bodies touch or are within the speculative-contact prediction distance. It can pull an object the last little bit onto a surface, but it is **not** action-at-a-distance attraction across an arbitrary gap.

**Code touch-points (A), all additive:**
- Add suction storage to [`ContactManifoldData`](src/geometry/contact_pair.rs#L312-L348).
- Surface it mutably on [`ContactModificationContext`](src/pipeline/physics_hooks.rs#L23-L46) and carry it through the write-back at [`narrow_phase.rs:1076-1101`](src/geometry/narrow_phase.rs#L1076-L1101).
- Add the per-manifold force pass in [`PhysicsPipeline::step`](src/pipeline/physics_pipeline.rs#L247-L258), just after the effective-force loop (mind the `parallel` feature variant later in that file).
- **No** changes to any solver-internals, clamp, friction, or SIMD code.

### 7.2 Design B — adhesion lower-bound inside the contact solver (alternative)

**Idea.** Give the normal contact constraint a negative lower bound: clamp the accumulated normal impulse to `[−A, +∞)` instead of `[0, +∞)`, where `A` is a per-contact maximum adhesion impulse set in the hook. The contact can then pull (negative impulse) up to `A`.

**Behaviour, and how it differs from A.** This is a **reactive** holding force, not a constant pull. At rest (no relative normal velocity, nothing separating), the solved impulse is ≈ 0 → **no preload, no net force**. It develops a pulling impulse only when something tries to separate the surfaces, capped at `A`; exceed `A` and they detach. This is arguably the more physically faithful "max holding force" / glue model.

**What must change (B), all in delicate code:**
- **The clamp**, in every normal solve path used in 2D: `(...).simd_max(N::zero())` → `(...).simd_max(-A)` at [`contact_constraint_element.rs:278`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L278). Warmstart already tolerates signed impulses (§3.4).
- **The friction cone** must never see a negative limit: use `friction × normal_impulse.max(0)` (or a notion of total clamp force) at [`contact_constraint_element.rs:150`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L150). A modelling decision: should a purely-pulling cup still resist sliding? `normal_impulse.max(0)` gives no friction during pure adhesion; using the holding-force magnitude would.
- **The block solver.** [`solve_mlcp_two_constraints`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L289-L319) / [`solve_pair`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L321-L366) hard-code the `≥ 0` MLCP case analysis and cannot represent a `−A` lower bound without a rewrite. Simplest path: **disable the block solver for manifolds with adhesion** and fall back to the scalar per-contact `solve` (slightly worse convergence on 2-point manifolds, but correct).
- **Carry the adhesion value** from the hook to the constraint builder, again via [`ContactManifoldData`](src/geometry/contact_pair.rs#L312-L348) (not `SolverContact`, per §3.5), read where normal parts are built (`generic_contact_constraint.rs` and the SIMD `contact_with_coulomb_friction.rs`).

---

## 8. Granularity: per-point vs per-manifold

Recall (§1.2): the force is **always** applied at per-point world positions. The only question is how you *specify* it.

- **Per-manifold (one knob).** A single `Real` on [`ContactManifoldData`](src/geometry/contact_pair.rs#L312-L348), exposed as `context.suction`. The engine splits it across the active contact points (also yielding the peeling-resisting torque). Simplest storage/API; mirrors how `normal` is per-manifold; good for a uniform seal.
- **Per-point (max fidelity).** A parallel `Vec<Real>` (or `[Real; MAX_MANIFOLD_POINTS]`) on `ContactManifoldData`, exposed as `context.suction` indexed like `solver_contacts`. Lets suction vary across the patch (e.g. model edge peeling). More API surface; mirrors how `friction` is per-point. It cannot live on `SolverContact` itself (§3.5), so it is a sibling array indexed in lockstep.

For Design B, the adhesion limit is conceptually per-point (it caps each contact's pull), so per-point storage fits most naturally; per-manifold (split) is also possible.

---

## 9. Side-by-side comparison

| | **A — external force** | **B — adhesion lower-bound** |
|---|---|---|
| You control | a force `F` (constant pull) | a max holding impulse `A` (reactive cap) |
| At rest, nothing pulling | net pull `F` (preload) | ~0 (no preload) |
| Resists separation up to | `F` | `A` |
| Friction while stuck | yes, scales with `F` (automatic) | needs an explicit modelling choice |
| Touches clamp / friction cone / block solver | no | yes (all three) |
| Touches SIMD-packed `SolverContact` code | no | no (value rides on `ContactManifoldData`) |
| Same-step integration | yes | yes |
| Risk / blast radius | low (reuses external-force path) | higher (edits solver core) |
| Action at a distance | only within speculative range | only within speculative range |

**Current recommendation.** Design A. It matches the "apply equal/opposite forces at the contact point" mental model, is low-risk, and already delivers same-step suction plus friction. Design B is worth it only if a no-preload, velocity-stabilising "max holding force" is specifically required. A reasonable path is to prototype A first and revisit B only if the constant-pull preload feels wrong.

---

## 10. Open decisions
- **Model:** A (constant pull) vs B (adhesion cap) — §7, §9.
- **Granularity:** per-manifold vs per-point — §8.
- (Confirmed) **Scope:** 2D only.

---

## 11. Verification plan (once a design is chosen)
- A new 2D example modelled on [`examples2d/one_way_platforms2.rs`](examples2d/one_way_platforms2.rs): a `PhysicsHooks` that sets suction on a chosen collider pair; confirm bodies stick, and that pulling harder than the suction releases them.
- Integration test: two stacked 2D boxes with suction; assert the upper box resists a separating impulse below the threshold and detaches above it; assert lateral friction holds while stuck.
- `cargo build` / `cargo test` for the `dim2` feature set.

---

## Appendix — key code references

| Topic | Location |
|---|---|
| Contact-modification context | [`physics_hooks.rs:23-46`](src/pipeline/physics_hooks.rs#L23-L46) |
| Hook trait method | [`physics_hooks.rs:232-257`](src/pipeline/physics_hooks.rs#L232-L257) |
| `MODIFY_SOLVER_CONTACTS` flag | [`physics_hooks.rs:147`](src/pipeline/physics_hooks.rs#L147) |
| One-way-platform helper (precedent) | [`physics_hooks.rs:48-113`](src/pipeline/physics_hooks.rs#L48-L113) |
| Solver-contact construction + world point | [`narrow_phase.rs:1054-1072`](src/geometry/narrow_phase.rs#L1054-L1072) |
| Hook call site + write-back | [`narrow_phase.rs:1076-1101`](src/geometry/narrow_phase.rs#L1076-L1101) |
| `SolverContact` definition | [`contact_pair.rs:372-405`](src/geometry/contact_pair.rs#L372-L405) |
| `SolverContact` SIMD packing asserts | [`contact_pair.rs:407-429`](src/geometry/contact_pair.rs#L407-L429) |
| `ContactManifoldData` | [`contact_pair.rs:312-348`](src/geometry/contact_pair.rs#L312-L348) |
| `is_bouncy` | [`contact_pair.rs:572-585`](src/geometry/contact_pair.rs#L572-L585) |
| Normal impulse clamp (≥ 0) | [`contact_constraint_element.rs:278`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L278) |
| Friction cone clamp (2D) | [`contact_constraint_element.rs:150`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L150) |
| Block solver (2 coupled contacts) | [`contact_constraint_element.rs:289-366`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L289-L366) |
| Normal-part warmstart (signed-safe) | [`contact_constraint_element.rs:246-260`](src/dynamics/solver/contact_constraint/contact_constraint_element.rs#L246-L260) |
| Effective force/torque per step | [`rigid_body_components.rs:976-979`](src/dynamics/rigid_body_components.rs#L976-L979) |
| `RigidBodyForces` fields | [`rigid_body_components.rs:916-931`](src/dynamics/rigid_body_components.rs#L916-L931) |
| `add_force_at_point` (persistent) | [`rigid_body.rs:1242-1251`](src/dynamics/rigid_body.rs#L1242-L1251) |
| `apply_impulse_at_point` (immediate) | [`rigid_body.rs:1341-1345`](src/dynamics/rigid_body.rs#L1341-L1345) |
| `reset_forces` | [`rigid_body.rs:1126`](src/dynamics/rigid_body.rs#L1126) |
| Effective-force summation loop | [`physics_pipeline.rs:247-258`](src/pipeline/physics_pipeline.rs#L247-L258) |
| Force → per-substep velocity increment | [`velocity_solver.rs:118-120`](src/dynamics/solver/velocity_solver.rs#L118-L120) |
| Spring joint / motors (bidirectional, for contrast) | [`spring_joint.rs`](src/dynamics/joint/spring_joint.rs), [`motor_model.rs`](src/dynamics/joint/motor_model.rs) |
