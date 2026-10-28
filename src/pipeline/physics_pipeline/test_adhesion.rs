//! Force, pressure, and shared-budget adhesion regression scenarios.

use crate::prelude::*;
use approx::assert_abs_diff_eq;

const G: Real = 9.81;

/// Applies a fixed adhesion force to every contact manifold that involves `collider`.
struct AdhesionHook {
    collider: ColliderHandle,
    force: Real,
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let (collider1, collider2) = (context.collider1, context.collider2);
        let Some(rigid) = context.rigid_mut() else {
            return;
        };

        if collider1 == self.collider || collider2 == self.collider {
            *rigid.adhesion_force = self.force;
        }
    }
}

/// Minimal world wrapper so each test can `step()` without repeating the long argument list.
struct World {
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    soft_bodies: SoftBodySet,
    ccd: CCDSolver,
    params: IntegrationParameters,
    gravity: Vector,
}

impl World {
    fn new(gravity: Vector) -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            ccd: CCDSolver::new(),
            params: IntegrationParameters::default(),
            gravity,
        }
    }

    fn step(&mut self, hooks: &dyn PhysicsHooks, n: usize) {
        for _ in 0..n {
            self.pipeline.step(
                self.gravity,
                &self.params,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                &mut self.ccd,
                hooks,
                &(),
            );
        }
    }
}

/// Fixed horizontal surface (thick cuboid) whose *bottom* face is at y = 0 with contact
/// modification enabled, plus a 1x1 dynamic box whose *top* face starts touching it from below
/// (i.e. hanging under the ceiling). Returns `(ceiling_collider, box_body)`.
fn ceiling_and_hanging_box(world: &mut World) -> (ColliderHandle, RigidBodyHandle) {
    let ceiling_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let ceiling = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        ceiling_body,
        &mut world.bodies,
    );

    let box_body = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)));
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5),
        box_body,
        &mut world.bodies,
    );

    (ceiling, box_body)
}

/// Fixed *vertical* wall whose left face is at x = 0 (contact modification enabled), plus a 1x1
/// dynamic box with friction `mu` whose right face starts touching that wall from the left.
fn wall_and_box(world: &mut World, mu: Real) -> (ColliderHandle, RigidBodyHandle) {
    let wall_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let wall = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 10.0)
            .translation(Vector::new(0.5, 0.0))
            .friction(mu)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        wall_body,
        &mut world.bodies,
    );

    let box_body = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(-0.5, 0.0)));
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5).friction(mu),
        box_body,
        &mut world.bodies,
    );

    (wall, box_body)
}

#[test]
fn adhesion_holds_box_below_ceiling() {
    let mut world = World::new(Vector::new(0.0, -G));
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    // Adhesion comfortably exceeds the box weight (m*g ~= 9.81): it should hang, not fall.
    let hook = AdhesionHook {
        collider: ceiling,
        force: 30.0,
    };
    world.step(&hook, 300);

    let box_rb = &world.bodies[box_body];
    let y = box_rb.translation().y;
    let vy = box_rb.linvel().y;

    // Still hanging just under the ceiling (started at y = -0.5), not fallen.
    assert!(y > -0.6, "box fell to y = {y} despite strong adhesion");
    // Stability: at rest and finite, i.e. no spurious energy gain over many steps.
    assert!(box_rb.translation().x.is_finite() && vy.is_finite());
    assert_abs_diff_eq!(vy, 0.0, epsilon = 0.1);
}

#[test]
fn weak_adhesion_lets_box_fall() {
    let mut world = World::new(Vector::new(0.0, -G));
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    // Adhesion far below the box weight: it cannot hold, so the box falls away.
    let hook = AdhesionHook {
        collider: ceiling,
        force: 3.0,
    };
    world.step(&hook, 120);

    let y = world.bodies[box_body].translation().y;
    assert!(y < -1.0, "box should have fallen but is at y = {y}");
}

#[test]
fn adhesion_break_threshold() {
    // Holds when adhesion exceeds weight + opposing load...
    {
        let mut world = World::new(Vector::new(0.0, -G));
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let w = world.bodies[box_body].mass() * G;
        // adhesion (w + 3) > weight + load (w + 1)  => holds.
        world.bodies[box_body].add_force(Vector::new(0.0, -1.0), true);
        let hook = AdhesionHook {
            collider: ceiling,
            force: w + 3.0,
        };
        world.step(&hook, 300);
        let y = world.bodies[box_body].translation().y;
        assert!(
            y > -0.6,
            "box should hold under sub-threshold load (y = {y})"
        );
    }
    // ...and detaches once the opposing load pushes past the adhesion.
    {
        let mut world = World::new(Vector::new(0.0, -G));
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let w = world.bodies[box_body].mass() * G;
        // adhesion (w + 3) < weight + load (w + 5)  => breaks free.
        world.bodies[box_body].add_force(Vector::new(0.0, -5.0), true);
        let hook = AdhesionHook {
            collider: ceiling,
            force: w + 3.0,
        };
        world.step(&hook, 200);
        let y = world.bodies[box_body].translation().y;
        assert!(
            y < -1.0,
            "box should break free under over-threshold load (y = {y})"
        );
    }
}

#[test]
fn adhesion_friction_holds_box_on_vertical_wall() {
    // High friction: mu*adhesion = 1.0 * 20 = 20 > m*g  => the box does not slide.
    let mut world = World::new(Vector::new(0.0, -G));
    let (wall, box_body) = wall_and_box(&mut world, 1.0);
    let y0 = world.bodies[box_body].translation().y;

    let hook = AdhesionHook {
        collider: wall,
        force: 20.0,
    };
    world.step(&hook, 300);

    let box_rb = &world.bodies[box_body];
    // Stuck to the wall (x unchanged) and not sliding down (y unchanged).
    assert_abs_diff_eq!(box_rb.translation().x, -0.5, epsilon = 0.05);
    assert_abs_diff_eq!(box_rb.translation().y, y0, epsilon = 0.1);
}

#[test]
fn adhesion_low_friction_slides_but_stays_attached() {
    // Low friction: mu*adhesion = 0.1 * 20 = 2.0 < m*g  => the box slides down the wall...
    let mut world = World::new(Vector::new(0.0, -G));
    let (wall, box_body) = wall_and_box(&mut world, 0.1);
    let y0 = world.bodies[box_body].translation().y;

    let hook = AdhesionHook {
        collider: wall,
        force: 20.0,
    };
    // Few enough steps that the box stays within the (tall) wall's vertical extent.
    world.step(&hook, 60);

    let box_rb = &world.bodies[box_body];
    // ...yet remains attached to the wall (x essentially unchanged).
    assert!(
        box_rb.translation().y < y0 - 0.5,
        "box should have slid down (y = {})",
        box_rb.translation().y
    );
    assert_abs_diff_eq!(box_rb.translation().x, -0.5, epsilon = 0.1);
}

#[test]
fn zero_adhesion_is_inert() {
    // A box resting on the ground with a hook that requests *zero* adhesion must behave exactly
    // like an ordinary contact: it rests on top and is not pulled into the ground.
    let mut world = World::new(Vector::new(0.0, -G));
    let ground_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let ground = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, -0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        ground_body,
        &mut world.bodies,
    );
    let box_body = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.5)));
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5),
        box_body,
        &mut world.bodies,
    );

    let hook = AdhesionHook {
        collider: ground,
        force: 0.0,
    };
    world.step(&hook, 300);

    let box_rb = &world.bodies[box_body];
    // Rests on top of the ground (around y = 0.5), at rest.
    assert_abs_diff_eq!(box_rb.translation().y, 0.5, epsilon = 0.05);
    assert_abs_diff_eq!(box_rb.linvel().y, 0.0, epsilon = 0.05);
}

#[test]
fn adhesion_holds_box_on_beyond_vertical_overhang() {
    // A slab tilted past vertical (135°) overhangs, so its sticky face points partly downward
    // and gravity peels the box away. With strong adhesion and high friction it must still
    // cling on, barely moving — the headline "hang on past vertical" behaviour.
    let angle = Real::to_radians(135.0);
    let mut world = World::new(Vector::new(0.0, -G));

    let surf_center = Vector::new(0.0, 5.0);
    let surf_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let surf = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(4.0, 0.25)
            .translation(surf_center)
            .rotation(angle)
            .friction(1.0)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        surf_body,
        &mut world.bodies,
    );

    // Box clinging to the slab's outward face (its local +Y rotated by `angle`).
    let face_normal = Vector::new(-angle.sin(), angle.cos());
    let box_center = surf_center + face_normal * (0.25 + 0.5 - 0.01);
    let box_body = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(box_center)
            .rotation(angle),
    );
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5).friction(1.0),
        box_body,
        &mut world.bodies,
    );

    let p0 = world.bodies[box_body].translation();
    let hook = AdhesionHook {
        collider: surf,
        force: 60.0,
    };
    world.step(&hook, 300);

    let p = world.bodies[box_body].translation();
    assert_abs_diff_eq!(p.x, p0.x, epsilon = 0.15);
    assert_abs_diff_eq!(p.y, p0.y, epsilon = 0.15);
}

#[test]
fn negative_adhesion_is_ignored() {
    // A negative request must be clamped to zero (adhesion only pulls, never pushes), so the
    // box falls *exactly* as it would with zero adhesion — not pushed away from the ceiling.
    let drop = |force: Real| -> Real {
        let mut world = World::new(Vector::new(0.0, -G));
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let hook = AdhesionHook {
            collider: ceiling,
            force,
        };
        let y0 = world.bodies[box_body].translation().y;
        world.step(&hook, 60);
        y0 - world.bodies[box_body].translation().y // distance fallen
    };

    let zero = drop(0.0);
    let negative = drop(-30.0);
    assert!(zero > 0.5, "box should fall under gravity with no adhesion");
    // Negative adhesion behaves identically to zero (it is not a push-apart force).
    assert_abs_diff_eq!(negative, zero, epsilon = 1.0e-3);
}

/*
 * Adhesion pressure (`ModifiableManifold::adhesion_pressure`): intensive adhesion,
 * force per unit of contact extent. The headline property is composition invariance — the
 * same behaviour whether a surface is one big collider or many small abutting ones.
 */

/// Applies a fixed adhesion pressure to every contact manifold that involves `collider`.
struct PressureHook {
    collider: ColliderHandle,
    pressure: Real,
}

impl PhysicsHooks for PressureHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let (collider1, collider2) = (context.collider1, context.collider2);
        let Some(rigid) = context.rigid_mut() else {
            return;
        };

        if collider1 == self.collider || collider2 == self.collider {
            *rigid.adhesion_pressure = self.pressure;
        }
    }
}

/// How the 10-unit test slope is decomposed into colliders.
#[derive(Clone, Copy, PartialEq)]
enum Slope {
    /// One 10-unit rectangle.
    Rect,
    /// 10 abutting unit squares (a body straddles 2-3 of them).
    Tiles,
    /// 10 abutting unit squares PLUS 9 more shifted by half a tile, co-planar tops: almost
    /// the whole surface is covered by TWO overlapping colliders.
    OverlappingTiles,
}

/// A capsule (2-unit flat side) lying on the top 2 m of a 10-unit slope at 40°, decomposed
/// per `slope`, all surface colliders under one fixed body. Returns the capsule collider
/// (hook-enabled) and body.
fn slope_and_capsule(world: &mut World, slope: Slope) -> (ColliderHandle, RigidBodyHandle) {
    let theta = Real::to_radians(40.0);
    let up_slope = Vector::new(theta.cos(), theta.sin());
    let top_normal = Vector::new(-theta.sin(), theta.cos());

    let statics = world.bodies.insert(RigidBodyBuilder::fixed());
    if slope == Slope::Rect {
        world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(5.0, 0.5)
                .rotation(theta)
                .friction(0.5),
            statics,
            &mut world.bodies,
        );
    } else {
        for k in 0..10 {
            let s = -4.5 + k as Real;
            world.colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.5, 0.5)
                    .translation(up_slope * s)
                    .rotation(theta)
                    .friction(0.5),
                statics,
                &mut world.bodies,
            );
        }
        if slope == Slope::OverlappingTiles {
            // Second, half-tile-shifted layer of tiles with the SAME top surface: doubled
            // coverage without changing the geometry a sliding body sees.
            for k in 0..9 {
                let s = -4.0 + k as Real;
                world.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(0.5, 0.5)
                        .translation(up_slope * s)
                        .rotation(theta)
                        .friction(0.5),
                    statics,
                    &mut world.bodies,
                );
            }
        }
    }

    // Flat side covers the top 2 m of the slope (s in [3, 5]).
    let cap_center = up_slope * 4.0 + top_normal * 1.0;
    let cap_body = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(cap_center)
            .rotation(theta),
    );
    let capsule = world.colliders.insert_with_parent(
        ColliderBuilder::capsule_x(1.0, 0.5)
            .friction(0.5)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        cap_body,
        &mut world.bodies,
    );
    (capsule, cap_body)
}

#[test]
fn pressure_adhesion_is_composition_invariant_on_slope() {
    // Per-manifold `adhesion_force` on a segmented slope multiplies with the manifold count
    // (the capsule straddles ~3 squares), so at 12 N the rectangle capsule creeps while the
    // square-strip capsule locks solid. `adhesion_pressure` must not: with P = 12 N / 2 m,
    // both compositions get 12 N total and must creep the same distance.
    let pressure = 6.0; // N per meter over the 2-unit flat side => 12 N total
    let run = |slope: Slope| -> Real {
        let mut world = World::new(Vector::new(0.0, -G));
        let (capsule, cap_body) = slope_and_capsule(&mut world, slope);
        let hook = PressureHook {
            collider: capsule,
            pressure,
        };
        let start = world.bodies[cap_body].translation();
        world.step(&hook, 180);
        let end = world.bodies[cap_body].translation();
        let theta = Real::to_radians(40.0);
        -(end - start).dot(Vector::new(theta.cos(), theta.sin()))
    };

    let d_rect = run(Slope::Rect);
    let d_strip = run(Slope::Tiles);

    // Both creep (neither locks — a lock here is the force-mode double-counting bug)...
    assert!(
        d_rect > 0.5,
        "rectangle capsule should creep (slid {d_rect})"
    );
    assert!(d_strip > 0.5, "strip capsule should creep (slid {d_strip})");
    // ...and they creep the same distance.
    assert_abs_diff_eq!(d_rect, d_strip, epsilon = 0.15);
}

#[test]
fn pressure_matches_equivalent_force_on_monolithic_surface() {
    // On a single-manifold surface the two flavors describe the same physics:
    // P * extent == F. The capsule's flat side is 2 units, so P = F / 2.
    let force = 12.0;
    let run = |use_pressure: bool| -> Real {
        let mut world = World::new(Vector::new(0.0, -G));
        let (capsule, cap_body) = slope_and_capsule(&mut world, Slope::Rect);
        let start = world.bodies[cap_body].translation();
        if use_pressure {
            let hook = PressureHook {
                collider: capsule,
                pressure: force / 2.0,
            };
            world.step(&hook, 180);
        } else {
            let hook = AdhesionHook {
                collider: capsule,
                force,
            };
            world.step(&hook, 180);
        }
        let end = world.bodies[cap_body].translation();
        let theta = Real::to_radians(40.0);
        -(end - start).dot(Vector::new(theta.cos(), theta.sin()))
    };

    let d_pressure = run(true);
    let d_force = run(false);
    assert_abs_diff_eq!(d_pressure, d_force, epsilon = 0.3);
}

#[test]
fn pressure_holds_box_below_ceiling() {
    // The hanging box's top face is 1 unit wide (extent 1), so pressure == resulting force.
    let mut world = World::new(Vector::new(0.0, -G));
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    let hook = PressureHook {
        collider: ceiling,
        pressure: 30.0, // x extent 1 => 30 N, comfortably above the ~9.81 N weight
    };
    world.step(&hook, 300);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y > -0.6,
        "box fell to y = {y} despite strong adhesion pressure"
    );
}

#[test]
fn pressure_on_point_contact_is_inert() {
    // A ball touching the ceiling makes a single-point manifold: zero tangential extent, so
    // even an enormous pressure produces zero force and the ball falls freely.
    let mut world = World::new(Vector::new(0.0, -G));
    let ceiling_body = world.bodies.insert(RigidBodyBuilder::fixed());
    let ceiling = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        ceiling_body,
        &mut world.bodies,
    );
    let ball_body = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)));
    world
        .colliders
        .insert_with_parent(ColliderBuilder::ball(0.5), ball_body, &mut world.bodies);

    let hook = PressureHook {
        collider: ceiling,
        pressure: 1000.0,
    };
    world.step(&hook, 120);

    let y = world.bodies[ball_body].translation().y;
    assert!(y < -1.0, "point-contact ball should fall (y = {y})");
}

#[test]
fn adhesion_stops_when_hook_flag_is_removed() {
    // Regression test: `manifold.data.adhesion_force`/`adhesion_pressure` used to be written
    // only inside the MODIFY_SOLVER_CONTACTS branch of the narrow phase, so removing the hook
    // flag from a collider mid-contact left the last requested value on the live manifold as a
    // permanent phantom force. The box must fall as soon as the flag is gone.
    let mut world = World::new(Vector::new(0.0, -G));
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    let hook = AdhesionHook {
        collider: ceiling,
        force: 30.0,
    };
    world.step(&hook, 120);
    let y_held = world.bodies[box_body].translation().y;
    assert!(y_held > -0.6, "box should hang while the hook is active");

    // Withdraw contact modification entirely; the still-alive manifold must not keep the
    // last-requested adhesion. (Wake the box: a sleeping pair is frozen wholesale, which
    // isn't the scenario under test.)
    world.colliders[ceiling].set_active_hooks(ActiveHooks::empty());
    world.bodies[box_body].wake_up(true);
    world.step(&hook, 120);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y < -1.0,
        "box should fall once the hook flag is removed, but is at y = {y}"
    );
}

#[test]
fn negative_pressure_is_ignored() {
    // Like negative force: clamped to zero, not a push-apart force.
    let drop = |pressure: Real| -> Real {
        let mut world = World::new(Vector::new(0.0, -G));
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let hook = PressureHook {
            collider: ceiling,
            pressure,
        };
        let y0 = world.bodies[box_body].translation().y;
        world.step(&hook, 60);
        y0 - world.bodies[box_body].translation().y
    };

    let zero = drop(0.0);
    let negative = drop(-30.0);
    assert!(zero > 0.5, "box should fall under gravity with no adhesion");
    assert_abs_diff_eq!(negative, zero, epsilon = 1.0e-3);
}

/*
 * Budgeted adhesion (`ModifiableManifold::adhesion_budget`): a fixed total shared by
 * all manifolds of an (owner, channel) pool in the same step. Headline properties: the total
 * is invariant under surface decomposition AND collider overlap, and a lone point contact
 * still receives the full total.
 */

/// Enrolls every manifold involving `collider` in one budget pool with the given total.
struct BudgetHook {
    collider: ColliderHandle,
    total: Real,
}

impl PhysicsHooks for BudgetHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let (collider1, collider2) = (context.collider1, context.collider2);
        let Some(rigid) = context.rigid_mut() else {
            return;
        };

        if collider1 == self.collider || collider2 == self.collider {
            *rigid.adhesion_budget = Some(AdhesionBudget {
                owner: self.collider,
                channel: 0,
                total: self.total,
            });
        }
    }
}

/// Down-slope distance slid in 3 s by a budgeted capsule on the given slope decomposition.
fn budget_slide(slope: Slope, total: Real) -> Real {
    let mut world = World::new(Vector::new(0.0, -G));
    let (capsule, cap_body) = slope_and_capsule(&mut world, slope);
    let hook = BudgetHook {
        collider: capsule,
        total,
    };
    let start = world.bodies[cap_body].translation();
    world.step(&hook, 180);
    let end = world.bodies[cap_body].translation();
    let theta = Real::to_radians(40.0);
    -(end - start).dot(Vector::new(theta.cos(), theta.sin()))
}

#[test]
fn budget_is_composition_and_overlap_invariant_on_slope() {
    // The same 12 N budget must produce the same creep whether the slope is one rectangle,
    // 10 abutting tiles, or 19 OVERLAPPING tiles (double coverage) — the overlap case is the
    // one `adhesion_pressure` cannot handle (overlapping spans double the measured extent).
    let d_rect = budget_slide(Slope::Rect, 12.0);
    let d_tiles = budget_slide(Slope::Tiles, 12.0);
    let d_overlap = budget_slide(Slope::OverlappingTiles, 12.0);

    assert!(
        d_rect > 0.5,
        "rectangle capsule should creep (slid {d_rect})"
    );
    assert!(d_tiles > 0.5, "tiled capsule should creep (slid {d_tiles})");
    assert!(
        d_overlap > 0.5,
        "overlapping-tiles capsule should creep (slid {d_overlap})"
    );
    assert_abs_diff_eq!(d_rect, d_tiles, epsilon = 0.15);
    assert_abs_diff_eq!(d_rect, d_overlap, epsilon = 0.15);
}

#[test]
fn budget_matches_equivalent_force_on_monolithic_surface() {
    // A pool with a single enrolled manifold is exactly `adhesion_force`.
    let d_budget = budget_slide(Slope::Rect, 12.0);
    let mut world = World::new(Vector::new(0.0, -G));
    let (capsule, cap_body) = slope_and_capsule(&mut world, Slope::Rect);
    let hook = AdhesionHook {
        collider: capsule,
        force: 12.0,
    };
    let start = world.bodies[cap_body].translation();
    world.step(&hook, 180);
    let end = world.bodies[cap_body].translation();
    let theta = Real::to_radians(40.0);
    let d_force = -(end - start).dot(Vector::new(theta.cos(), theta.sin()));

    assert_abs_diff_eq!(d_budget, d_force, epsilon = 1.0e-3);
}

#[test]
fn budget_total_is_capped_across_seam_straddling_manifolds() {
    // A box hangs under a ceiling made of TWO tiles, its top face straddling the seam: two
    // manifolds share the pool. The hold threshold must be the pool total — not 2x it, which
    // is what per-manifold `adhesion_force` gives.
    let run = |budgeted: bool, load: Real| -> Real {
        let mut world = World::new(Vector::new(0.0, -G));
        let ceiling_body = world.bodies.insert(RigidBodyBuilder::fixed());
        for k in [-1.0, 1.0] {
            world.colliders.insert_with_parent(
                ColliderBuilder::cuboid(1.0, 0.5)
                    .translation(Vector::new(k, 0.5))
                    .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
                ceiling_body,
                &mut world.bodies,
            );
        }
        let box_body = world
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)));
        let box_co = world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.5, 0.5),
            box_body,
            &mut world.bodies,
        );
        let w = world.bodies[box_body].mass() * G;
        world.bodies[box_body].add_force(Vector::new(0.0, -load), true);

        let total = w + 3.0;
        if budgeted {
            let hook = BudgetHook {
                collider: box_co,
                total,
            };
            world.step(&hook, 200);
        } else {
            let hook = AdhesionHook {
                collider: box_co,
                force: total,
            };
            world.step(&hook, 200);
        }
        world.bodies[box_body].translation().y
    };

    // Budget w + 3 vs load 5: total pull (w + 3) < weight + load (w + 5) => must break free.
    let y = run(true, 5.0);
    assert!(y < -1.0, "budgeted box should break free (y = {y})");
    // ...but still holds a sub-threshold load.
    let y = run(true, 1.0);
    assert!(y > -0.6, "budgeted box should hold a 1 N load (y = {y})");
    // Contrast: per-manifold force double-counts across the two manifolds (2w + 6 total pull)
    // and wrongly survives the over-threshold load. This pins the difference.
    let y = run(false, 5.0);
    assert!(
        y > -0.6,
        "force-mode box should (incorrectly) keep hanging via 2x pull (y = {y})"
    );
}

#[test]
fn budget_channels_are_independent_pools() {
    // Same rig as the seam-straddling test, but the hook enrolls each ceiling tile's manifold
    // in a DIFFERENT channel: two pools of (w + 3) each => 2w + 6 total pull, which holds the
    // w + 5 load that a single shared pool (previous test) breaks under. This is the corner
    // semantics: a body spends its feet budget AND its flank budget simultaneously.
    struct TwoChannelHook {
        collider: ColliderHandle,
        total: Real,
    }
    impl PhysicsHooks for TwoChannelHook {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            let (collider1, collider2) = (context.collider1, context.collider2);
            let Some(rigid) = context.rigid_mut() else {
                return;
            };

            let other = if collider1 == self.collider {
                collider2
            } else if collider2 == self.collider {
                collider1
            } else {
                return;
            };
            // Channel keyed by which tile this manifold touches.
            let (index, _) = other.0.into_raw_parts();
            *rigid.adhesion_budget = Some(AdhesionBudget {
                owner: self.collider,
                channel: index,
                total: self.total,
            });
        }
    }

    let mut world = World::new(Vector::new(0.0, -G));
    let ceiling_body = world.bodies.insert(RigidBodyBuilder::fixed());
    for k in [-1.0, 1.0] {
        world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(1.0, 0.5)
                .translation(Vector::new(k, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            ceiling_body,
            &mut world.bodies,
        );
    }
    let box_body = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)));
    let box_co = world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5),
        box_body,
        &mut world.bodies,
    );
    let w = world.bodies[box_body].mass() * G;
    world.bodies[box_body].add_force(Vector::new(0.0, -5.0), true);

    let hook = TwoChannelHook {
        collider: box_co,
        total: w + 3.0,
    };
    world.step(&hook, 200);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y > -0.6,
        "two independent channels should hold the w + 5 load together (y = {y})"
    );
}

#[test]
fn adhesion_applies_no_net_torque() {
    // Wrench-neutrality: adhesion is an internal action-reaction pair, so it must contribute
    // zero net torque to the system it acts within. Rig: a 10 m plank hung from a pivot 3 m
    // above its center (a STABLE pendulum — unlike a center-pivot teeter, it cannot amplify
    // numerical seeds), with a capsule standing 4 m out on EACH side and only the LEFT one
    // adhering (20 N, ~2x its weight). Any steady net torque T from the adhesion would settle
    // the plank at a permanent tilt ~T / (M g d); the 0.002 rad threshold detects ~0.1 N*m.
    let run = |adhesion: Option<Real>| -> Real {
        let mut world = World::new(Vector::new(0.0, -G));
        let pivot = world
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 3.0)));
        let plank_body = world.bodies.insert(
            RigidBodyBuilder::dynamic()
                .angular_damping(2.0) // settle oscillations so the final angle is read at rest
                .can_sleep(false),
        );
        world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(5.0, 0.5).mass(1.0).friction(2.0),
            plank_body,
            &mut world.bodies,
        );
        world.impulse_joints.insert(
            pivot,
            plank_body,
            RevoluteJointBuilder::new().local_anchor2(Vector::new(0.0, 3.0)),
            true,
        );

        let mut adhesive_capsule = None;
        for side in [-1.0, 1.0] {
            let cap_body = world.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(side * 4.0, 1.5))
                    .lock_rotations()
                    .can_sleep(false),
            );
            let adheres = side < 0.0 && adhesion.is_some();
            let mut capsule = ColliderBuilder::capsule_y(0.5, 0.5).mass(1.0).friction(2.0);
            if adheres {
                capsule = capsule.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
            }
            let capsule = world
                .colliders
                .insert_with_parent(capsule, cap_body, &mut world.bodies);
            if adheres {
                adhesive_capsule = Some(capsule);
            }
        }

        match (adhesion, adhesive_capsule) {
            (Some(force), Some(capsule)) => {
                let hook = AdhesionHook {
                    collider: capsule,
                    force,
                };
                world.step(&hook, 600);
            }
            _ => world.step(&(), 600),
        }
        world.bodies[plank_body].rotation().angle()
    };

    let level = run(None);
    let adhering = run(Some(20.0));
    assert!(
        level.abs() < 0.002,
        "control plank settled tilted ({level} rad)"
    );
    assert!(
        adhering.abs() < 0.002,
        "adhesion applied a steady net torque: plank settled at {adhering} rad"
    );
}

#[test]
fn budget_point_contact_receives_full_total() {
    // A ball under the ceiling is a single-point manifold: pressure adhesion is inert there,
    // but a budget pool with one member hands it the full total, so the ball hangs.
    let mut world = World::new(Vector::new(0.0, -G));
    let ceiling_body = world.bodies.insert(RigidBodyBuilder::fixed());
    world.colliders.insert_with_parent(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        ceiling_body,
        &mut world.bodies,
    );
    let ball_body = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)));
    let ball_co = world.colliders.insert_with_parent(
        ColliderBuilder::ball(0.5),
        ball_body,
        &mut world.bodies,
    );

    let hook = BudgetHook {
        collider: ball_co,
        total: 30.0, // ball weighs ~7.7 N
    };
    world.step(&hook, 300);

    let y = world.bodies[ball_body].translation().y;
    assert!(y > -0.6, "ball fell to y = {y} despite a 30 N budget");
}
