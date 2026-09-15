# Fork changes — bcolloran/rapier `my-changes-0.35.3`

This fork of [dimforge/rapier](https://github.com/dimforge/rapier) carries two features that bc's
game (PVE, godot_rapier) depends on. This file records what the fork changes, why, where the code
lives, what guards it, and how to carry it to the next upstream release.

- **Base:** upstream `master` at `d90dd956` (v0.35.3 plus three post-release fixes).
- **Branch naming:** `my-changes-<upstream version>`. godot_rapier's
  `scripts/update_rapier.sh --branch <name>` switches the game to a branch.
- **Consumer build:** `rapier2d`, f32, features `serde-serialize` and `enhanced-determinism`, no
  `simd8` and no `parallel`. The game runs deterministic rollback netplay and hashes serialized
  snapshots across clients.

---

## 1. `step_collisions_last`

**What.** `PhysicsPipeline::step_collisions_last` is a drop-in replacement for `step`, with the
same parameters. It integrates first and runs collision detection **last**. When it returns, the
narrow phase (contacts, intersections, collision events) matches the new body poses.

Companion API:
- `initialize_collisions_last(…, ccd_solver, hooks, events)` runs the first detection, so contacts
  exist at t = 0. It is idempotent, and the first `step_collisions_last` calls it automatically.
- `collisions_last_initialized()` and `set_collisions_last_initialized(bool)`. When a fresh
  pipeline steps a restored snapshot, set the flag instead of re-running init. Re-running init
  changes broad-phase and narrow-phase state and breaks cross-client checksums; a test asserts the
  divergence.
- `PhysicsWorld::{step_collisions_last, step_collisions_last_with_events,
  initialize_collisions_last_with_events}`.

**Why.** The game reads contacts between ticks, for movement and jump logic. With stock `step`,
those contacts describe the poses at the start of the tick, so they are one tick stale.

**Where.**
- `src/pipeline/physics_pipeline/substep.rs`: a private `StepMode { Standard, CollisionsLastInit,
  CollisionsLast }` parameter on `step_inner`.
  - One code path serves every mode, so upstream fixes to `step_inner` reach all of them.
  - In `Standard` mode every fork branch is a no-op.
- `src/pipeline/physics_pipeline/mod.rs`: the public methods (with the same thread-pool install as
  `step`), the initialized flag, and `finish_collisions_last_detection`.
- `src/geometry/narrow_phase/solver_graph.rs`: `requalify_woken_pair_hints` and
  `clear_solver_graph_dirty`.
- `src/pipeline/physics_world.rs`: the world wrappers.

**How the collisions-last mode differs from stock:**
1. Narrow-phase user-change handling runs **before** the solve: removals, recoloring, island
   relinks and wake-ups.
2. **Catch-up.** In some cases a full `detect_collisions` also runs before the solve. It runs when
   a collider is inserted, enabled or disabled; when it changes shape, collision groups, sensor
   status or parent; when its parent body changes type or dominance; or when an impulse or
   multibody joint is inserted or removed.
   - Why: contact anchors are frozen in body frames, so they would be wrong for one solve.
   - Effect: on those frames the **hooks run twice**. Collision events are not duplicated, and
     adhesion is applied once (tested).
3. The final `detect_collisions` runs after the substep loop, with the whole step's `dt` and
   `handle_user_changes = false`.
4. `finish_collisions_last_detection` ends every collisions-last detection, including init's:
   - It runs `maintain_solver_contact_graph` and clears the non-serialized `solver_graph_dirty`
     list. Without that clear, a restored world ordered its contact-force-event pairs differently
     from the uninterrupted one, and their bincode bytes differed.
   - It **joins** the deferred BVH optimization. Without the join, queries and snapshots between
     steps would see an empty broad-phase tree.
5. The wake gap is fixed: `requalify_woken_pair_hints` runs before the solve. It repairs the solver
   hints that a sleeping body's pairs had count-cleared. Without it, a woken stack sank 1.7 mm on
   its first solve. With it, a woken adhered box drifts 3e-8 m, where stock drops 1.66 mm.

**Accepted trade-off.** Some changes reach the contacts one step late: they get fresh contacts only
from the end-of-step detection. The rustdoc of `step_collisions_last` lists every case:
- teleports, including CCD tunneling through a wall teleported that step;
- removals;
- friction and restitution;
- active hooks, events and collision types;
- center-of-mass changes;
- edits to an existing joint.

**Guarded by** `src/pipeline/physics_pipeline/test_collisions_last.rs`, which runs in all four
crates:
- 21 ports of the 0.32 tests;
- catch-up (re-parent, body type and sensor, with events and positions compared against stock), CCD
  substeps with removals (a bullet sweeping stale BVH leaves), ray casts between steps, and the wake
  gap;
- snapshot restores (bytes, events and hook calls), including after init on a valid graph;
- joint insertion and removal, which matches stock within 1e-5.

The collisions-last variants in `crates/rapier2d/tests/fork_goldens.rs` guard it too.

## 2. Contact adhesion

**What.** Inside `PhysicsHooks::modify_solver_contacts`, a hook can request a pull between the two
bodies of a contact manifold. There are three kinds of request. They add together, and zero or
`None` does nothing:

| Request | Unit | Behavior |
|---|---|---|
| `*context.adhesion_force` | N per manifold | Simple, but counts once per manifold: a body on N abutting colliders gets ~N× the pull. |
| `*context.adhesion_pressure` | N/m (2D), N/m² (3D) | Multiplied by the manifold's tangential extent: 2D is the contact-point spread ⊥ normal, 3D the projected polygon area, and 0 below 2 (2D) or 3 (3D) points. Independent of how a surface is split into colliders. |
| `*context.adhesion_budget = Some(AdhesionBudget { owner, channel, total })` | N total | Manifolds in one step with the same `(owner, channel)` share a pool. Pool total = **max** of the requests. Share ∝ `max(extent, 1e-3)`. Independent of splitting and overlap; a point contact gets the full total; channels are independent. |

- **Ignored values:** non-positive and non-finite ones. A non-finite term doesn't cancel the
  manifold's other terms, and a budget with a non-finite total joins no pool.
- **Extent:** `ContactModificationContext::tangential_extent()` computes it inside the hook.
  `ContactManifoldData::tangential_extent()` returns the value cached when the hook last ran.
- **3D clustered pairs:** the request and the extent live on the solver clusters
  (`ContactPair::solver_manifolds`).

**Why.**
- The player moves only through contact modification: a `tangent_velocity` drive, an effective μ,
  and adhesion.
- Plain Coulomb friction grips too weakly on slopes of about 30° and steeper, and wall-slide needs a
  tunable terminal velocity.
- A stock hook cannot pull: the normal impulse is ≥ 0, friction is μ × the normal impulse, and hooks
  get only `&RigidBodySet`.
- godot_rapier ADR-0003: contact effects go through Rapier's contact solver, never as body forces
  that fight it.
- Details are in `suction_forces_research.md`. Design A, a pre-solve external force, was chosen.
  Design B, a negative normal-impulse bound, was rejected.

**Where.**
- **Data:** `src/geometry/contact_pair.rs`.
  - `ContactManifoldData` gets `adhesion_force`, `adhesion_pressure`, `adhesion_budget` and the
    private `adhesion_extent`, all **serialized**. The collisions-last solve and woken pairs read
    the persisted values, so the values must survive a snapshot restore.
  - The same file defines `AdhesionBudget` and the extent helper.
- **Hook plumbing:** `src/geometry/narrow_phase/pair_update.rs::process_pair`.
  - Locals are passed to the hook and written back.
  - The extent is cached before the anchors are localized.
  - Every field resets when `MODIFY_SOLVER_CONTACTS` is off, and on plain manifolds when 3D
    clustering applies.
  - A full update that stores a request poisons the pair's recycle state, so a removed hook flag
    can't leave a stale pull on a recycled pair.
- **Application:** `src/pipeline/physics_pipeline/solve.rs::apply_contact_adhesion`. It is called
  from `build_islands_and_solve_velocity_constraints` immediately before
  `staged_solver.init_and_solve`.
  - It iterates the solver contact graph: the exact solver-active manifolds, in deterministic order.
  - For each solver contact it adds `±normal·F/k` to `rb.forces.force` and the matching torque,
    using the solver's own lever arms (`ContactData::solver_dp1/dp2`).
  - Only awake dynamic bodies are pulled, and never a side the solver treats as world-attached
    through dominance.
  - It returns early when no manifold requests adhesion. A manifold whose terms are all zero,
    negative or non-finite writes nothing, so stock results stay bit-identical.
  - In collisions-last mode, a pair requalified on wake has its adhesion zeroed when neither
    collider still has the hook flag.

**Costs and side effects:**
- **Stock `step` sweep.** Every substep reads each solver-active manifold once before the early
  return, even in worlds without hooks. See Open decisions.
- **CCD sees adhesion.** `rb.forces.force` still holds the adhesion force after the solve, and the
  next step's CCD activation check reads it before the fused traversal reassigns forces. This is
  deterministic.
- **Budget shares.** A member whose pullable side is skipped (kinematic, fixed, sleeping, or
  dominance-world-attached) still takes its share of the pool's weight. That share pulls nothing on
  that side.
- **Hook-flag removal delay.** In collisions-last mode, removing the hook flag from an awake,
  resting pair takes effect one step late unless the same step triggers a catch-up. This is the
  documented delay for any hook output.

**Deferred on purpose.** In-constraint adhesion: a normal-impulse lower bound plus a shifted friction
limit. Bound sums still scale with the manifold count, and the 2×2 block solver hard-codes λ ≥ 0.

**Known facts for hook authors:**
- `tangent_velocity` is `v(collider2) − v(collider1)`.
- Friction set in a hook applies per manifold (`*context.friction`).
- Very large adhesion values (e.g. 1e5 N on a body of a few kg) are numerically explosive. Test with
  moderate multiples of the analytic threshold.

**Guarded by:**
- `crates/rapier2d/tests/adhesion.rs`: every test runs in both step modes (59 cases).
  - 20 ports of the 0.32 tests;
  - sleep and wake, with a two-sided tolerance per mode;
  - recycling, with hook-less resting pairs that really recycle;
  - dominance, both orderings;
  - bit-identity of zero, negative and non-finite requests, and the cached extent;
  - hook-flag removal (hanging, resting, woken);
  - non-finite terms.
- `crates/rapier3d/tests/adhesion3.rs`, both modes: clustered pairs applied once, 3D extent, and
  off-center lever arms (no net torque).
- `crates/rapier2d/tests/adhesion_collisions_last.rs`:
  - snapshot restore with adhesion and the tangent drive, byte-identical over 240 compared steps
    and 3.9 MB, plus the divergence without the flag;
  - a woken adhered box that stays attached;
  - hook-flag removal on an awake pair;
  - adhesion applied once on a catch-up frame.

## 3. Tooling

- **Demos:** `examples2d/adhesion{,_grid,_curved,_climb,_slide,_teeter}2.rs`, in the "Adhesion"
  group of `all_examples2`. Each has a "Collisions last" setting, on by default. Run them with
  `cargo run --release -p rapier-examples-2d --bin all_examples2`.
- **Benchmark:** `crates/rapier2d/tests/collisions_last_bench.rs`, an ignored A/B timing test with
  Welch's t-test.
  - Scenes: the ten June 2026 stress scenes, "Adhesion tiles" (game-shaped), and "Spawner" (a
    collider inserted every N steps, to measure catch-up cost).
  - Steps are timed with `std::time::Instant`, because rapier's counters only measure with the
    `profiler` feature.
  - Run it with `cargo test -p rapier2d --release --test collisions_last_bench -- --ignored --nocapture`.
  - Knobs: `COLLISIONS_LAST_BENCH_ITERS`, `_WARMUP`, `_SCENES`, `_SPAWN_EVERY`.
- **Verification tooling used for the port**, kept outside the repo in
  `/data/code_projects/rust/rapier-port/`:
  - `port-gates.sh`: mirrors CI's jobs, keeps going after failures, and logs each gate.
  - `bin/cargo`: a shim that makes a shared build directory safe across worktrees.

## 4. Goldens

- **`crates/rapier2d/tests/fork_goldens.rs`** is fork-owned.
  - It hashes simulation results only: every serialized container except the narrow phase, plus
    the sorted per-step event stream.
  - It has three scenes, each in stock and collisions-last modes:
    - mixed (contacts, sleep, joints, multibody, sensor, CCD);
    - hooked (a `tangent_velocity` drive and friction);
    - adhesion (tiles and slope with force, pressure and budget, plus a sleeping box that wakes).
  - **Never re-mint it to make a fork change pass.** If it moves, the change altered behavior. The
    only re-mint so far was the collisions-last hooked scene, when the wake-gap fix intentionally
    changed collisions-last results.
  - It is verified identical in debug, release, `parallel` and wasm32.
- **Upstream goldens re-minted by this fork.** The adhesion fields add exactly 13 bytes per
  serialized manifold: 4 + 4 + a 1-byte `None` tag + 4. Nothing else changed.

  | Golden | Manifolds after 60 steps | Before | After |
  |---|---|---|---|
  | `crates/rapier2d/tests/snapshot_portability.rs` | 133 (+1,729 B) | `(88_572, 0x9e84_4370_55e1_8c88)` | `(90_301, 0x0b49_066a_911b_8d9e)` |
  | `crates/rapier3d/tests/snapshot_portability.rs` | 889, 0 clusters (+11,557 B) | `(481_608, 0xedfe_b12c_c030_e444)` | `(493_165, 0xc77a_b42d_83c2_86fa)` |
  | `crates/rapier3d/tests/parallel_path_parity.rs` | n/a (hash only) | `0x7c57_2d38_0b87_2d4c` | `0xe1a7_71ca_4441_1890` |

## 5. Removed or replaced since `my-changes-0.32.0`

- `ContactForceEvent::first_tick` is dropped. Use upstream `ContactForceEvent::started` instead: it
  is true on the step a pair's total force first exceeds its threshold, and resets when the force
  drops back below it or the pair separates.
- The testbed capsule-coloring fix is upstream (`set_color_recursive`).
- The per-example toggle and `collisions_last_bench.rs` replace the testbed's
  `Harness::use_step_collisions_last` flag and the `app.rs` benchmark.
- `StepMode` replaces the public stage helpers `user_changes_stage_part_1/2`, `substeps_stage` and
  `update_mass_properties_of_moved_bodies_stage`.

## 6. Behavior that changed vs `my-changes-0.32.0`

1. In collisions-last mode, collider user changes are handled before the solve. Catch-up frames
   (see §1) detect twice, so hooks run twice on them.
2. In collisions-last mode, new colliders get contacts before that step's solve, and a woken island
   solves with its resting contacts. On 0.32 both were one step late.
3. `initialize_collisions_last` takes `ccd_solver`, and there is a new initialized-flag setter.
4. `ContactManifoldData::tangential_extent()` returns the value cached when the hook ran.
5. Adhesion is not applied to a dominance-world-attached side. 0.32 applied it to any dynamic body.
6. Infinite adhesion requests now add nothing. 0.32 applied an infinite force.
7. `first_tick` is removed in favor of `started`.
8. The snapshot byte format changed, so old bincode snapshots do not load.

## 6b. Known limitations (found during the port, not fixed)

- **Shared with stock `step` (upstream behavior):**
  - With contact recycling on (the default), a resting pair keeps its contacts after a
    contacts-disabled joint is inserted, until its next full update. That is about 2.8e-2 m/s of
    jolt in 3D, in both modes.
  - A center-of-mass change (`LOCAL_MASS_PROPERTIES`) is in the recycle-safe set, so recycled pairs
    keep stale CoM-relative anchors. Found by reading the code; not measured.
  - Joint island events (`island_events`, `island_chain_events`) are not serialized. A snapshot
    taken between a joint removal and the next step loses the island unlink, and in collisions-last
    mode the catch-up trigger too. This is harmless when inputs are re-applied after a restore, as
    rollback does.
- **Collisions-last only:** editing an existing joint, such as its `contacts_enabled` flag, triggers
  no catch-up, so the edit reaches the contacts one step late.
- **Cost:** `requalify_woken_pair_hints` walks every edge of every awake body each step. Catch-up
  frames (for example any frame that spawns a collider) run collision detection twice. The
  benchmark's "Spawner" scene measures this.

## 6c. Open decisions

1. **Stock `step` adhesion sweep.** Worlds without hooks pay one read per solver-active manifold
   per substep. Removing that cost needs a sticky, serialized "adhesion requested" flag, because
   sleeping and requalified pairs keep old requests without the hook running again. Keep the sweep
   (simple, measured by the benchmark), or add the flag?
2. **Verification tooling.** `port-gates.sh` and the cargo shim live outside the repo. Commit them
   under `scripts/fork/`, so the next upgrade starts with them, or keep them outside?

## 7. Carrying the fork to the next upstream release

1. Read this file.
2. Make a worktree from the new upstream tip. Add `fork_goldens.rs` first, and run the full gates
   as a baseline.
3. Port in order: goldens → collisions-last → adhesion → demos → benchmark. Collisions-last and
   adhesion touch different files, so they can go in parallel.
4. Diff these upstream hot spots first:
   - `step_inner` and `detect_collisions`;
   - the `process_pair` hook site and anchor localization;
   - the force traversal and solver entry in `build_islands_and_solve_velocity_constraints`;
   - `SolverContact` and `ContactManifoldData`;
   - the non-serialized narrow-phase state (`solver_graph_dirty`) and broad-phase deferred
     optimization;
   - the testbed example API.
5. Keep `fork_goldens` unchanged through every step. Re-mint upstream serialized-world goldens only
   for fork-added encoding, and prove it with the byte-count check.
6. Never let two worktrees share a cargo build directory without a guard: cargo silently reuses the
   other checkout's crates.
7. After the fork is green, migrate godot_rapier with `update_rapier.sh`, then run its testbed,
   surrogate corpus and 4-client E2E checksum gates.
