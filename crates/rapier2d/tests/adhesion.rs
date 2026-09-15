//! Contact adhesion (2D): the attractive pull a contact-modification hook requests through
//! `ContactModificationContext::adhesion_force`, `adhesion_pressure` and `adhesion_budget`.
//!
//! The engine applies the requested pull as an ordinary external force on both bodies right before
//! the contact solver runs, so the unchanged push-only contacts provide the holding reaction, a
//! natural break threshold and friction. Scenarios mostly use a 1x1 box (mass 1.0, weight `G`):
//! adhesion holds while `adhesion >= weight + load`, and the contact reaction to the pull gives
//! friction up to `mu * adhesion`.
//!
//! Every scenario runs in both stepping modes (see [`Mode`]): stock `step` and
//! `step_collisions_last`, which bc's game uses together with adhesion.

use std::sync::Mutex;

use rapier2d::prelude::*;

const G: Real = 9.81;

/// How a test advances its world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `PhysicsWorld::step_with_events`.
    Stock,
    /// `PhysicsWorld::step_collisions_last_with_events`, initialized before the first step.
    CollisionsLast,
}

/// Generates one `#[test]` per stepping mode for `fn $name(mode: Mode)`, in a module named after
/// the function, so a failure reads `$name::stock` or `$name::collisions_last`.
macro_rules! both_modes {
    ($name:ident) => {
        mod $name {
            #[test]
            fn stock() {
                super::$name(super::Mode::Stock);
            }

            #[test]
            fn collisions_last() {
                super::$name(super::Mode::CollisionsLast);
            }
        }
    };
}

/// Advances `world` by one step in `mode`. Every test steps through here.
fn step(mode: Mode, world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks) {
    match mode {
        Mode::Stock => world.step_with_events(hooks, &()),
        Mode::CollisionsLast => {
            if !world.physics_pipeline.collisions_last_initialized() {
                world.initialize_collisions_last_with_events(hooks, &());
            }
            world.step_collisions_last_with_events(hooks, &());
        }
    }
}

fn run(mode: Mode, world: &mut PhysicsWorld, hooks: &dyn PhysicsHooks, steps: usize) {
    for _ in 0..steps {
        step(mode, world, hooks);
    }
}

fn new_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -G);
    world
}

fn assert_close(got: Real, expected: Real, epsilon: Real, what: &str) {
    assert!(
        (got - expected).abs() <= epsilon,
        "{what}: got {got}, expected {expected} ± {epsilon}"
    );
}

/// Bit patterns of the pose and velocity of `handles`, in order, for bit-identity comparisons.
fn bits_of(world: &PhysicsWorld, handles: &[RigidBodyHandle]) -> Vec<u32> {
    let mut bits = Vec::new();
    for h in handles {
        let rb = &world.bodies[*h];
        let t = rb.translation();
        let v = rb.linvel();
        for x in [t.x, t.y, rb.rotation().angle(), v.x, v.y, rb.angvel()] {
            bits.push(x.to_bits());
        }
    }
    bits
}

/// Bit patterns of every body's pose and velocity, for bit-identity comparisons.
fn body_bits(world: &PhysicsWorld) -> Vec<u32> {
    let mut handles: Vec<_> = world.bodies.iter().map(|(h, _)| h).collect();
    handles.sort_by_key(|h| h.into_raw_parts());
    bits_of(world, &handles)
}

/// Applies a fixed adhesion force to every contact manifold that involves `collider`.
struct AdhesionHook {
    collider: ColliderHandle,
    force: Real,
}

impl PhysicsHooks for AdhesionHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        if context.collider1 == self.collider || context.collider2 == self.collider {
            *context.adhesion_force = self.force;
        }
    }
}

/// Fixed horizontal surface (thick cuboid) whose *bottom* face is at y = 0 with contact
/// modification enabled, plus a 1x1 dynamic box whose *top* face starts touching it from below
/// (i.e. hanging under the ceiling), both centered on `x`. Returns `(ceiling_collider, box_body)`.
fn ceiling_and_hanging_box_at(
    world: &mut PhysicsWorld,
    x: Real,
) -> (ColliderHandle, RigidBodyHandle) {
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let ceiling = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(x, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );

    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(x, -0.5)),
        ColliderBuilder::cuboid(0.5, 0.5),
    );

    (ceiling, box_body)
}

fn ceiling_and_hanging_box(world: &mut PhysicsWorld) -> (ColliderHandle, RigidBodyHandle) {
    ceiling_and_hanging_box_at(world, 0.0)
}

/// Fixed *vertical* wall whose left face is at x = 0 (contact modification enabled), plus a 1x1
/// dynamic box with friction `mu` whose right face starts touching that wall from the left.
fn wall_and_box(world: &mut PhysicsWorld, mu: Real) -> (ColliderHandle, RigidBodyHandle) {
    let wall_body = world.insert_body(RigidBodyBuilder::fixed());
    let wall = world.insert_collider(
        ColliderBuilder::cuboid(0.5, 10.0)
            .translation(Vector::new(0.5, 0.0))
            .friction(mu)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(wall_body),
    );

    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(-0.5, 0.0)),
        ColliderBuilder::cuboid(0.5, 0.5).friction(mu),
    );

    (wall, box_body)
}

fn adhesion_holds_box_below_ceiling(mode: Mode) {
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    // Adhesion comfortably exceeds the box weight (m*g ~= 9.81): it should hang, not fall.
    let hook = AdhesionHook {
        collider: ceiling,
        force: 30.0,
    };
    run(mode, &mut world, &hook, 300);

    let box_rb = &world.bodies[box_body];
    let y = box_rb.translation().y;
    let vy = box_rb.linvel().y;

    // Still hanging just under the ceiling (started at y = -0.5), not fallen.
    assert!(y > -0.6, "box fell to y = {y} despite strong adhesion");
    // Stability: at rest and finite, i.e. no spurious energy gain over many steps.
    assert!(box_rb.translation().x.is_finite() && vy.is_finite());
    assert_close(vy, 0.0, 0.1, "hanging box vertical velocity");
}
both_modes!(adhesion_holds_box_below_ceiling);

fn weak_adhesion_lets_box_fall(mode: Mode) {
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    // Adhesion far below the box weight: it cannot hold, so the box falls away.
    let hook = AdhesionHook {
        collider: ceiling,
        force: 3.0,
    };
    run(mode, &mut world, &hook, 120);

    let y = world.bodies[box_body].translation().y;
    assert!(y < -1.0, "box should have fallen but is at y = {y}");
}
both_modes!(weak_adhesion_lets_box_fall);

fn adhesion_break_threshold(mode: Mode) {
    // Holds when adhesion exceeds weight + opposing load...
    {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let w = world.bodies[box_body].mass() * G;
        // adhesion (w + 3) > weight + load (w + 1)  => holds.
        world.bodies[box_body].add_force(Vector::new(0.0, -1.0), true);
        let hook = AdhesionHook {
            collider: ceiling,
            force: w + 3.0,
        };
        run(mode, &mut world, &hook, 300);
        let y = world.bodies[box_body].translation().y;
        assert!(
            y > -0.6,
            "box should hold under sub-threshold load (y = {y})"
        );
    }
    // ...and detaches once the opposing load pushes past the adhesion.
    {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let w = world.bodies[box_body].mass() * G;
        // adhesion (w + 3) < weight + load (w + 5)  => breaks free.
        world.bodies[box_body].add_force(Vector::new(0.0, -5.0), true);
        let hook = AdhesionHook {
            collider: ceiling,
            force: w + 3.0,
        };
        run(mode, &mut world, &hook, 200);
        let y = world.bodies[box_body].translation().y;
        assert!(
            y < -1.0,
            "box should break free under over-threshold load (y = {y})"
        );
    }
}
both_modes!(adhesion_break_threshold);

fn adhesion_friction_holds_box_on_vertical_wall(mode: Mode) {
    // High friction: mu*adhesion = 1.0 * 20 = 20 > m*g  => the box does not slide.
    let mut world = new_world();
    let (wall, box_body) = wall_and_box(&mut world, 1.0);
    let y0 = world.bodies[box_body].translation().y;

    let hook = AdhesionHook {
        collider: wall,
        force: 20.0,
    };
    run(mode, &mut world, &hook, 300);

    let box_rb = &world.bodies[box_body];
    // Stuck to the wall (x unchanged) and not sliding down (y unchanged).
    assert_close(box_rb.translation().x, -0.5, 0.05, "box x on the wall");
    assert_close(box_rb.translation().y, y0, 0.1, "box y on the wall");
}
both_modes!(adhesion_friction_holds_box_on_vertical_wall);

fn adhesion_low_friction_slides_but_stays_attached(mode: Mode) {
    // Low friction: mu*adhesion = 0.1 * 20 = 2.0 < m*g  => the box slides down the wall...
    let mut world = new_world();
    let (wall, box_body) = wall_and_box(&mut world, 0.1);
    let y0 = world.bodies[box_body].translation().y;

    let hook = AdhesionHook {
        collider: wall,
        force: 20.0,
    };
    // Few enough steps that the box stays within the (tall) wall's vertical extent.
    run(mode, &mut world, &hook, 60);

    let box_rb = &world.bodies[box_body];
    // ...yet remains attached to the wall (x essentially unchanged).
    assert!(
        box_rb.translation().y < y0 - 0.5,
        "box should have slid down (y = {})",
        box_rb.translation().y
    );
    assert_close(box_rb.translation().x, -0.5, 0.1, "sliding box x");
}
both_modes!(adhesion_low_friction_slides_but_stays_attached);

fn zero_adhesion_is_inert(mode: Mode) {
    // A box resting on the ground with a hook that requests *zero* adhesion must behave exactly
    // like an ordinary contact: it rests on top and is not pulled into the ground.
    let mut world = new_world();
    let ground_body = world.insert_body(RigidBodyBuilder::fixed());
    let ground = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, -0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ground_body),
    );
    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 0.5)),
        ColliderBuilder::cuboid(0.5, 0.5),
    );

    let hook = AdhesionHook {
        collider: ground,
        force: 0.0,
    };
    run(mode, &mut world, &hook, 300);

    let box_rb = &world.bodies[box_body];
    // Rests on top of the ground (around y = 0.5), at rest.
    assert_close(box_rb.translation().y, 0.5, 0.05, "resting box y");
    assert_close(
        box_rb.linvel().y,
        0.0,
        0.05,
        "resting box vertical velocity",
    );
}
both_modes!(zero_adhesion_is_inert);

fn adhesion_holds_box_on_beyond_vertical_overhang(mode: Mode) {
    // A slab tilted past vertical (135°) overhangs, so its sticky face points partly downward
    // and gravity peels the box away. With strong adhesion and high friction it must still
    // cling on, barely moving — the headline "hang on past vertical" behaviour.
    let angle = (135.0 as Real).to_radians();
    let mut world = new_world();

    let surf_center = Vector::new(0.0, 5.0);
    let surf_body = world.insert_body(RigidBodyBuilder::fixed());
    let surf = world.insert_collider(
        ColliderBuilder::cuboid(4.0, 0.25)
            .translation(surf_center)
            .rotation(angle)
            .friction(1.0)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(surf_body),
    );

    // Box clinging to the slab's outward face (its local +Y rotated by `angle`).
    let face_normal = Vector::new(-angle.sin(), angle.cos());
    let box_center = surf_center + face_normal * (0.25 + 0.5 - 0.01);
    let (box_body, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(box_center)
            .rotation(angle),
        ColliderBuilder::cuboid(0.5, 0.5).friction(1.0),
    );

    let p0 = world.bodies[box_body].translation();
    let hook = AdhesionHook {
        collider: surf,
        force: 60.0,
    };
    run(mode, &mut world, &hook, 300);

    let p = world.bodies[box_body].translation();
    assert_close(p.x, p0.x, 0.15, "overhang box x");
    assert_close(p.y, p0.y, 0.15, "overhang box y");
}
both_modes!(adhesion_holds_box_on_beyond_vertical_overhang);

fn negative_adhesion_is_ignored(mode: Mode) {
    // A negative request must be clamped to zero (adhesion only pulls, never pushes), so the
    // box falls *exactly* as it would with zero adhesion — not pushed away from the ceiling.
    let drop = |force: Real| -> Real {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let hook = AdhesionHook {
            collider: ceiling,
            force,
        };
        let y0 = world.bodies[box_body].translation().y;
        run(mode, &mut world, &hook, 60);
        y0 - world.bodies[box_body].translation().y // distance fallen
    };

    let zero = drop(0.0);
    let negative = drop(-30.0);
    assert!(zero > 0.5, "box should fall under gravity with no adhesion");
    // Negative adhesion behaves identically to zero (it is not a push-apart force).
    assert_close(
        negative,
        zero,
        1.0e-3,
        "fall distance with negative adhesion",
    );
}
both_modes!(negative_adhesion_is_ignored);

/*
 * Adhesion pressure (`ContactModificationContext::adhesion_pressure`): intensive adhesion,
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
        if context.collider1 == self.collider || context.collider2 == self.collider {
            *context.adhesion_pressure = self.pressure;
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
fn slope_and_capsule(world: &mut PhysicsWorld, slope: Slope) -> (ColliderHandle, RigidBodyHandle) {
    let theta = (40.0 as Real).to_radians();
    let up_slope = Vector::new(theta.cos(), theta.sin());
    let top_normal = Vector::new(-theta.sin(), theta.cos());

    let statics = world.insert_body(RigidBodyBuilder::fixed());
    if slope == Slope::Rect {
        world.insert_collider(
            ColliderBuilder::cuboid(5.0, 0.5)
                .rotation(theta)
                .friction(0.5),
            Some(statics),
        );
    } else {
        for k in 0..10 {
            let s = -4.5 + k as Real;
            world.insert_collider(
                ColliderBuilder::cuboid(0.5, 0.5)
                    .translation(up_slope * s)
                    .rotation(theta)
                    .friction(0.5),
                Some(statics),
            );
        }
        if slope == Slope::OverlappingTiles {
            // Second, half-tile-shifted layer of tiles with the SAME top surface: doubled
            // coverage without changing the geometry a sliding body sees.
            for k in 0..9 {
                let s = -4.0 + k as Real;
                world.insert_collider(
                    ColliderBuilder::cuboid(0.5, 0.5)
                        .translation(up_slope * s)
                        .rotation(theta)
                        .friction(0.5),
                    Some(statics),
                );
            }
        }
    }

    // Flat side covers the top 2 m of the slope (s in [3, 5]).
    let cap_center = up_slope * 4.0 + top_normal * 1.0;
    let (cap_body, capsule) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(cap_center)
            .rotation(theta),
        ColliderBuilder::capsule_x(1.0, 0.5)
            .friction(0.5)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
    );
    (capsule, cap_body)
}

/// Down-slope distance travelled by `body` since `start`.
fn slid(world: &PhysicsWorld, body: RigidBodyHandle, start: Vector) -> Real {
    let theta = (40.0 as Real).to_radians();
    -(world.bodies[body].translation() - start).dot(Vector::new(theta.cos(), theta.sin()))
}

fn pressure_adhesion_is_composition_invariant_on_slope(mode: Mode) {
    // Per-manifold `adhesion_force` on a segmented slope multiplies with the manifold count
    // (the capsule straddles ~3 squares), so at 12 N the rectangle capsule creeps while the
    // square-strip capsule locks solid. `adhesion_pressure` must not: with P = 12 N / 2 m,
    // both compositions get 12 N total and must creep the same distance.
    let pressure = 6.0; // N per meter over the 2-unit flat side => 12 N total
    let slide = |slope: Slope| -> Real {
        let mut world = new_world();
        let (capsule, cap_body) = slope_and_capsule(&mut world, slope);
        let hook = PressureHook {
            collider: capsule,
            pressure,
        };
        let start = world.bodies[cap_body].translation();
        run(mode, &mut world, &hook, 180);
        slid(&world, cap_body, start)
    };

    let d_rect = slide(Slope::Rect);
    let d_strip = slide(Slope::Tiles);

    // Both creep (neither locks — a lock here is the force-mode double-counting bug)...
    assert!(
        d_rect > 0.5,
        "rectangle capsule should creep (slid {d_rect})"
    );
    assert!(d_strip > 0.5, "strip capsule should creep (slid {d_strip})");
    // ...and they creep the same distance.
    assert_close(d_rect, d_strip, 0.15, "rectangle vs strip creep");
}
both_modes!(pressure_adhesion_is_composition_invariant_on_slope);

fn pressure_matches_equivalent_force_on_monolithic_surface(mode: Mode) {
    // On a single-manifold surface the two flavors describe the same physics:
    // P * extent == F. The capsule's flat side is 2 units, so P = F / 2.
    let force = 12.0;
    let slide = |use_pressure: bool| -> Real {
        let mut world = new_world();
        let (capsule, cap_body) = slope_and_capsule(&mut world, Slope::Rect);
        let start = world.bodies[cap_body].translation();
        if use_pressure {
            let hook = PressureHook {
                collider: capsule,
                pressure: force / 2.0,
            };
            run(mode, &mut world, &hook, 180);
        } else {
            let hook = AdhesionHook {
                collider: capsule,
                force,
            };
            run(mode, &mut world, &hook, 180);
        }
        slid(&world, cap_body, start)
    };

    let d_pressure = slide(true);
    let d_force = slide(false);
    assert_close(d_pressure, d_force, 0.3, "pressure vs force creep");
}
both_modes!(pressure_matches_equivalent_force_on_monolithic_surface);

fn pressure_holds_box_below_ceiling(mode: Mode) {
    // The hanging box's top face is 1 unit wide (extent 1), so pressure == resulting force.
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    let hook = PressureHook {
        collider: ceiling,
        pressure: 30.0, // x extent 1 => 30 N, comfortably above the ~9.81 N weight
    };
    run(mode, &mut world, &hook, 300);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y > -0.6,
        "box fell to y = {y} despite strong adhesion pressure"
    );
}
both_modes!(pressure_holds_box_below_ceiling);

fn pressure_on_point_contact_is_inert(mode: Mode) {
    // A ball touching the ceiling makes a single-point manifold: zero tangential extent, so
    // even an enormous pressure produces zero force and the ball falls freely.
    let mut world = new_world();
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    let ceiling = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );
    let (ball_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)),
        ColliderBuilder::ball(0.5),
    );

    let hook = PressureHook {
        collider: ceiling,
        pressure: 1000.0,
    };
    run(mode, &mut world, &hook, 120);

    let y = world.bodies[ball_body].translation().y;
    assert!(y < -1.0, "point-contact ball should fall (y = {y})");
}
both_modes!(pressure_on_point_contact_is_inert);

fn adhesion_stops_when_hook_flag_is_removed(mode: Mode) {
    // Regression test: the adhesion requests used to be written only inside the
    // MODIFY_SOLVER_CONTACTS branch of the narrow phase, so removing the hook flag from a
    // collider mid-contact left the last requested value on the live manifold as a permanent
    // phantom force. The box must fall as soon as the flag is gone.
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);

    let hook = AdhesionHook {
        collider: ceiling,
        force: 30.0,
    };
    run(mode, &mut world, &hook, 120);
    let y_held = world.bodies[box_body].translation().y;
    assert!(y_held > -0.6, "box should hang while the hook is active");

    // Withdraw contact modification entirely; the still-alive manifold must not keep the
    // last-requested adhesion. (Wake the box: a sleeping pair is frozen wholesale, which
    // isn't the scenario under test.)
    world.colliders[ceiling].set_active_hooks(ActiveHooks::empty());
    world.bodies[box_body].wake_up(true);
    run(mode, &mut world, &hook, 120);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y < -1.0,
        "box should fall once the hook flag is removed, but is at y = {y}"
    );
}
both_modes!(adhesion_stops_when_hook_flag_is_removed);

fn adhesion_stops_when_hook_flag_is_removed_from_a_resting_pair(mode: Mode) {
    // The contact-recycling path. `Collider::set_active_hooks` doesn't mark the collider as
    // changed, and a resting pair barely moves, so once the flag is gone the pair qualifies for
    // recycling, which would keep the stale adhesion on its manifolds. The narrow phase must
    // instead refuse to recycle from an update that stored an adhesion request, so the next
    // update clears it. Rig: a box on a hooked floor, held down by 30 N of adhesion against a
    // net upward load of 5 N; it must lift off once the flag is removed.
    let mut world = new_world();
    assert!(world.integration_parameters.contact_recycling);
    let floor_body = world.insert_body(RigidBodyBuilder::fixed());
    let floor = world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, -0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(floor_body),
    );
    let (box_body, box_co) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.5))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.5, 0.5),
    );
    let w = world.bodies[box_body].mass() * G;
    world.bodies[box_body].add_force(Vector::new(0.0, w + 5.0), true);

    let hook = AdhesionHook {
        collider: floor,
        force: 30.0,
    };
    run(mode, &mut world, &hook, 120);
    let y_held = world.bodies[box_body].translation().y;
    assert_close(y_held, 0.5, 0.05, "adhered box resting on the floor");

    world.colliders[floor].set_active_hooks(ActiveHooks::empty());
    step(mode, &mut world, &hook);
    let pair = world.narrow_phase.contact_pair(floor, box_co).unwrap();
    assert!(
        pair.manifolds.iter().all(|m| m.data.adhesion_force == 0.0),
        "the first update after the flag removal must clear the stored adhesion"
    );

    run(mode, &mut world, &hook, 60);
    let y = world.bodies[box_body].translation().y;
    assert!(
        y > 1.0,
        "box should lift off once the hook flag is removed, but is at y = {y}"
    );
}
both_modes!(adhesion_stops_when_hook_flag_is_removed_from_a_resting_pair);

fn negative_pressure_is_ignored(mode: Mode) {
    // Like negative force: clamped to zero, not a push-apart force.
    let drop = |pressure: Real| -> Real {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let hook = PressureHook {
            collider: ceiling,
            pressure,
        };
        let y0 = world.bodies[box_body].translation().y;
        run(mode, &mut world, &hook, 60);
        y0 - world.bodies[box_body].translation().y
    };

    let zero = drop(0.0);
    let negative = drop(-30.0);
    assert!(zero > 0.5, "box should fall under gravity with no adhesion");
    assert_close(
        negative,
        zero,
        1.0e-3,
        "fall distance with negative pressure",
    );
}
both_modes!(negative_pressure_is_ignored);

/*
 * Budgeted adhesion (`ContactModificationContext::adhesion_budget`): a fixed total shared by
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
        if context.collider1 == self.collider || context.collider2 == self.collider {
            *context.adhesion_budget = Some(AdhesionBudget {
                owner: self.collider,
                channel: 0,
                total: self.total,
            });
        }
    }
}

/// Down-slope distance slid in 3 s by a budgeted capsule on the given slope decomposition.
fn budget_slide(mode: Mode, slope: Slope, total: Real) -> Real {
    let mut world = new_world();
    let (capsule, cap_body) = slope_and_capsule(&mut world, slope);
    let hook = BudgetHook {
        collider: capsule,
        total,
    };
    let start = world.bodies[cap_body].translation();
    run(mode, &mut world, &hook, 180);
    slid(&world, cap_body, start)
}

fn budget_is_composition_and_overlap_invariant_on_slope(mode: Mode) {
    // The same 12 N budget must produce the same creep whether the slope is one rectangle,
    // 10 abutting tiles, or 19 OVERLAPPING tiles (double coverage) — the overlap case is the
    // one `adhesion_pressure` cannot handle (overlapping spans double the measured extent).
    let d_rect = budget_slide(mode, Slope::Rect, 12.0);
    let d_tiles = budget_slide(mode, Slope::Tiles, 12.0);
    let d_overlap = budget_slide(mode, Slope::OverlappingTiles, 12.0);

    assert!(
        d_rect > 0.5,
        "rectangle capsule should creep (slid {d_rect})"
    );
    assert!(d_tiles > 0.5, "tiled capsule should creep (slid {d_tiles})");
    assert!(
        d_overlap > 0.5,
        "overlapping-tiles capsule should creep (slid {d_overlap})"
    );
    assert_close(d_rect, d_tiles, 0.15, "rectangle vs tiles creep");
    assert_close(
        d_rect,
        d_overlap,
        0.15,
        "rectangle vs overlapping tiles creep",
    );
}
both_modes!(budget_is_composition_and_overlap_invariant_on_slope);

fn budget_matches_equivalent_force_on_monolithic_surface(mode: Mode) {
    // A pool with a single enrolled manifold is exactly `adhesion_force`.
    let d_budget = budget_slide(mode, Slope::Rect, 12.0);
    let mut world = new_world();
    let (capsule, cap_body) = slope_and_capsule(&mut world, Slope::Rect);
    let hook = AdhesionHook {
        collider: capsule,
        force: 12.0,
    };
    let start = world.bodies[cap_body].translation();
    run(mode, &mut world, &hook, 180);
    let d_force = slid(&world, cap_body, start);

    assert_close(d_budget, d_force, 1.0e-3, "budget vs force creep");
}
both_modes!(budget_matches_equivalent_force_on_monolithic_surface);

/// A box hanging under a ceiling made of TWO tiles, its top face straddling the seam, with a
/// downward `load` on the box. Returns `(box collider, box body, box weight)`.
fn seam_ceiling_and_box(
    world: &mut PhysicsWorld,
    load: Real,
) -> (ColliderHandle, RigidBodyHandle, Real) {
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    for k in [-1.0, 1.0] {
        world.insert_collider(
            ColliderBuilder::cuboid(1.0, 0.5)
                .translation(Vector::new(k, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling_body),
        );
    }
    let (box_body, box_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)),
        ColliderBuilder::cuboid(0.5, 0.5),
    );
    let w = world.bodies[box_body].mass() * G;
    world.bodies[box_body].add_force(Vector::new(0.0, -load), true);
    (box_co, box_body, w)
}

fn budget_total_is_capped_across_seam_straddling_manifolds(mode: Mode) {
    // Two manifolds share the pool. The hold threshold must be the pool total — not 2x it,
    // which is what per-manifold `adhesion_force` gives.
    let hang = |budgeted: bool, load: Real| -> Real {
        let mut world = new_world();
        let (box_co, box_body, w) = seam_ceiling_and_box(&mut world, load);
        let total = w + 3.0;
        if budgeted {
            let hook = BudgetHook {
                collider: box_co,
                total,
            };
            run(mode, &mut world, &hook, 200);
        } else {
            let hook = AdhesionHook {
                collider: box_co,
                force: total,
            };
            run(mode, &mut world, &hook, 200);
        }
        world.bodies[box_body].translation().y
    };

    // Budget w + 3 vs load 5: total pull (w + 3) < weight + load (w + 5) => must break free.
    let y = hang(true, 5.0);
    assert!(y < -1.0, "budgeted box should break free (y = {y})");
    // ...but still holds a sub-threshold load.
    let y = hang(true, 1.0);
    assert!(y > -0.6, "budgeted box should hold a 1 N load (y = {y})");
    // Contrast: per-manifold force double-counts across the two manifolds (2w + 6 total pull)
    // and wrongly survives the over-threshold load. This pins the difference.
    let y = hang(false, 5.0);
    assert!(
        y > -0.6,
        "force-mode box should (incorrectly) keep hanging via 2x pull (y = {y})"
    );
}
both_modes!(budget_total_is_capped_across_seam_straddling_manifolds);

fn budget_channels_are_independent_pools(mode: Mode) {
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
            let other = if context.collider1 == self.collider {
                context.collider2
            } else if context.collider2 == self.collider {
                context.collider1
            } else {
                return;
            };
            // Channel keyed by which tile this manifold touches.
            let (index, _) = other.into_raw_parts();
            *context.adhesion_budget = Some(AdhesionBudget {
                owner: self.collider,
                channel: index,
                total: self.total,
            });
        }
    }

    let mut world = new_world();
    let (box_co, box_body, w) = seam_ceiling_and_box(&mut world, 5.0);
    let hook = TwoChannelHook {
        collider: box_co,
        total: w + 3.0,
    };
    run(mode, &mut world, &hook, 200);

    let y = world.bodies[box_body].translation().y;
    assert!(
        y > -0.6,
        "two independent channels should hold the w + 5 load together (y = {y})"
    );
}
both_modes!(budget_channels_are_independent_pools);

fn adhesion_applies_no_net_torque(mode: Mode) {
    // Wrench-neutrality: adhesion is an internal action-reaction pair, so it must contribute
    // zero net torque to the system it acts within. Rig: a 10 m plank hung from a pivot 3 m
    // above its center (a STABLE pendulum — unlike a center-pivot teeter, it cannot amplify
    // numerical seeds), with a capsule standing 4 m out on EACH side and only the LEFT one
    // adhering (20 N, ~2x its weight). Any steady net torque T from the adhesion would settle
    // the plank at a permanent tilt ~T / (M g d); the 0.002 rad threshold detects ~0.1 N*m.
    let settle = |adhesion: Option<Real>| -> Real {
        let mut world = new_world();
        let pivot = world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 3.0)));
        let plank_body = world.insert_body(
            RigidBodyBuilder::dynamic()
                .angular_damping(2.0) // settle oscillations so the final angle is read at rest
                .can_sleep(false),
        );
        world.insert_collider(
            ColliderBuilder::cuboid(5.0, 0.5).mass(1.0).friction(2.0),
            Some(plank_body),
        );
        world.insert_impulse_joint(
            pivot,
            plank_body,
            RevoluteJointBuilder::new().local_anchor2(Vector::new(0.0, 3.0)),
        );

        let mut adhesive_capsule = None;
        for side in [-1.0 as Real, 1.0] {
            let cap_body = world.insert_body(
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
            let capsule = world.insert_collider(capsule, Some(cap_body));
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
                run(mode, &mut world, &hook, 600);
            }
            _ => run(mode, &mut world, &(), 600),
        }
        world.bodies[plank_body].rotation().angle()
    };

    let level = settle(None);
    let adhering = settle(Some(20.0));
    assert!(
        level.abs() < 0.002,
        "control plank settled tilted ({level} rad)"
    );
    assert!(
        adhering.abs() < 0.002,
        "adhesion applied a steady net torque: plank settled at {adhering} rad"
    );
}
both_modes!(adhesion_applies_no_net_torque);

fn budget_point_contact_receives_full_total(mode: Mode) {
    // A ball under the ceiling is a single-point manifold: pressure adhesion is inert there,
    // but a budget pool with one member hands it the full total, so the ball hangs.
    let mut world = new_world();
    let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
    world.insert_collider(
        ColliderBuilder::cuboid(5.0, 0.5)
            .translation(Vector::new(0.0, 0.5))
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        Some(ceiling_body),
    );
    let (ball_body, ball_co) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)),
        ColliderBuilder::ball(0.5),
    );

    let hook = BudgetHook {
        collider: ball_co,
        total: 30.0, // ball weighs ~7.7 N
    };
    run(mode, &mut world, &hook, 300);

    let y = world.bodies[ball_body].translation().y;
    assert!(y > -0.6, "ball fell to y = {y} despite a 30 N budget");
}
both_modes!(budget_point_contact_receives_full_total);

/*
 * Behavior added on top of the 0.32 suite: sleep/wake, contact recycling, dominance, inert
 * requests and the cached extent.
 */

fn adhesion_survives_sleep_and_wake_by_impact(mode: Mode) {
    // A box held under a ceiling falls asleep. Its pair is then skipped by the narrow phase
    // (the hook doesn't run), so the manifold must keep the adhesion requested before sleep,
    // and a thrown ball waking the box from below must not detach it.
    //
    // Measured in every debug build of the quick gates (default, enhanced-determinism + serde,
    // parallel + serde), as the largest |y - y_asleep| over the 60 steps after the throw: stock
    // stepping 28.45 mm, collisions-last 0.30 mm; neither ever goes below its sleeping height.
    // The ball's contact starts during a detection, which wakes the box while its ceiling pair is
    // still count-cleared from sleep: stock stepping solves that step without the ceiling (upstream
    // behavior), so the ball knocks the box up into it, while collisions-last puts the pair back
    // into the solve first (`requalify_woken_pair_hints`). 240 steps later both are back within
    // 1e-6 m (stock 6.9e-7 m, collisions-last 3.6e-7 m).
    let (wake_tolerance, final_tolerance): (Real, Real) = match mode {
        Mode::Stock => (0.03, 1.0e-5),
        Mode::CollisionsLast => (5.0e-4, 1.0e-5),
    };
    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
    let box_co = world.bodies[box_body].colliders()[0];
    let hook = AdhesionHook {
        collider: ceiling,
        force: 30.0,
    };

    let mut slept = false;
    for _ in 0..600 {
        step(mode, &mut world, &hook);
        if world.bodies[box_body].is_sleeping() {
            slept = true;
            break;
        }
    }
    assert!(slept, "the adhered box never fell asleep");
    run(mode, &mut world, &hook, 30);
    assert!(world.bodies[box_body].is_sleeping());
    let y_asleep = world.bodies[box_body].translation().y;

    let pair = world
        .narrow_phase
        .contact_pair(ceiling, box_co)
        .expect("ceiling/box contact pair");
    assert!(
        pair.manifolds.iter().any(|m| m.data.adhesion_force == 30.0),
        "a sleeping pair must keep its last adhesion request"
    );

    // Throw a ball up into the box.
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.1, -3.0))
            .linvel(Vector::new(0.0, 12.0)),
        ColliderBuilder::ball(0.25),
    );
    let mut woke = false;
    for _ in 0..60 {
        step(mode, &mut world, &hook);
        woke |= !world.bodies[box_body].is_sleeping();
        let y = world.bodies[box_body].translation().y;
        assert!(
            (y - y_asleep).abs() < wake_tolerance,
            "box moved {} m from where it slept on wake (y = {y}, tolerance {wake_tolerance} m)",
            y - y_asleep
        );
    }
    assert!(woke, "the thrown ball never woke the box");

    run(mode, &mut world, &hook, 240);
    let y = world.bodies[box_body].translation().y;
    assert!(
        (y - y_asleep).abs() < final_tolerance,
        "box should still hang where it slept (y = {y}, was {y_asleep}, tolerance \
         {final_tolerance} m)"
    );
}
both_modes!(adhesion_survives_sleep_and_wake_by_impact);

/// One hook for scenes mixing adhesion kinds: `(collider, request)` pairs, matched against
/// either side of each manifold.
#[derive(Clone, Copy)]
enum Request {
    Force(Real),
    Pressure(Real),
    Budget(Real),
}

struct MixedHook {
    requests: Vec<(ColliderHandle, Request)>,
}

impl PhysicsHooks for MixedHook {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        for (collider, request) in &self.requests {
            if context.collider1 != *collider && context.collider2 != *collider {
                continue;
            }
            match *request {
                Request::Force(f) => *context.adhesion_force = f,
                Request::Pressure(p) => *context.adhesion_pressure = p,
                Request::Budget(total) => {
                    *context.adhesion_budget = Some(AdhesionBudget {
                        owner: *collider,
                        channel: 0,
                        total,
                    })
                }
            }
        }
    }
}

fn contact_recycling_does_not_change_adhesion_results(mode: Mode) {
    // Pairs with hooks are never recycled, so turning contact recycling on must leave the adhered
    // bodies bit-identical, even while hook-less resting pairs in the same world do get recycled.
    // Those plain pairs are what makes this test able to fail: they put the recycling path to
    // work next to the adhesion (their own bodies' results differ with recycling on and off,
    // which the test checks), on bodies that don't touch the adhered ones.
    let simulate = |recycling: bool| -> (Vec<u32>, Vec<u32>) {
        let mut world = new_world();
        world.integration_parameters.contact_recycling = recycling;
        let (capsule, cap_body) = slope_and_capsule(&mut world, Slope::OverlappingTiles);
        let (ceiling, box_body) = ceiling_and_hanging_box_at(&mut world, 30.0);
        let wall_body = world.insert_body(RigidBodyBuilder::fixed());
        let wall = world.insert_collider(
            ColliderBuilder::cuboid(0.5, 10.0)
                .translation(Vector::new(-29.5, 0.0))
                .friction(0.1)
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(wall_body),
        );
        let (wall_box, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(-30.5, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5).friction(0.1),
        );

        // Hook-less resting pairs: a two-high row of boxes on a plain floor, away from the
        // adhesive rigs. They never sleep, so they keep going through the contact update.
        let floor_body = world.insert_body(RigidBodyBuilder::fixed());
        world.insert_collider(
            ColliderBuilder::cuboid(5.0, 0.5).translation(Vector::new(60.0, -0.5)),
            Some(floor_body),
        );
        let mut plain = Vec::new();
        for i in 0..3 {
            for j in 0..2 {
                let (body, _) = world.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(Vector::new(58.0 + 2.0 * i as Real, 0.5 + j as Real))
                        .can_sleep(false),
                    ColliderBuilder::cuboid(0.5, 0.5),
                );
                plain.push(body);
            }
        }

        let hook = MixedHook {
            requests: vec![
                (capsule, Request::Budget(12.0)),
                (ceiling, Request::Force(30.0)),
                (wall, Request::Pressure(20.0)),
            ],
        };
        run(mode, &mut world, &hook, 240);
        (
            bits_of(&world, &[cap_body, box_body, wall_box]),
            bits_of(&world, &plain),
        )
    };

    let (adhered_on, plain_on) = simulate(true);
    let (adhered_off, plain_off) = simulate(false);
    assert!(
        plain_on != plain_off,
        "the plain resting pairs gave the same results with recycling on and off: the scene no \
         longer exercises contact recycling"
    );
    assert!(
        adhered_on == adhered_off,
        "adhesion results differ with contact recycling on and off"
    );
}
both_modes!(contact_recycling_does_not_change_adhesion_results);

/// A body of a higher dominance group is world-attached for its contacts with lower-group
/// bodies: the solver gives it no contact reaction. Pulling it would therefore drag it through
/// the contact, so adhesion only pulls the lower-group side. Zero gravity: the dominant slab must
/// not move at all, while the adhesion still holds the small box against a 20 N load pulling it
/// away. `slab_first` inserts the slab before the box; the pair's relative dominance (collider1's
/// group minus collider2's) is returned, so callers can check which side was world-attached.
fn check_dominant_slab_is_not_dragged(mode: Mode, slab_first: bool) -> i16 {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let insert_slab = |world: &mut PhysicsWorld| {
        world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.5))
                .dominance_group(10)
                .can_sleep(false),
            ColliderBuilder::cuboid(2.0, 0.5).active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
        )
    };
    let insert_box = |world: &mut PhysicsWorld| {
        world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, -0.5))
                .can_sleep(false),
            ColliderBuilder::cuboid(0.5, 0.5),
        )
    };
    let ((slab_body, slab_co), (box_body, box_co)) = if slab_first {
        let slab = insert_slab(&mut world);
        (slab, insert_box(&mut world))
    } else {
        let the_box = insert_box(&mut world);
        (insert_slab(&mut world), the_box)
    };
    world.bodies[box_body].add_force(Vector::new(0.0, -20.0), true);

    let hook = AdhesionHook {
        collider: slab_co,
        force: 50.0,
    };
    run(mode, &mut world, &hook, 180);

    let slab = &world.bodies[slab_body];
    assert_eq!(
        slab.translation(),
        Vector::new(0.0, 0.5),
        "the dominant slab was dragged by adhesion (slab_first = {slab_first})"
    );
    assert_eq!(slab.linvel(), Vector::ZERO);
    assert_eq!(slab.angvel(), 0.0);

    let y = world.bodies[box_body].translation().y;
    assert_close(y, -0.5, 0.05, "adhered box against a 20 N load");

    let pair = world
        .narrow_phase
        .contact_pair(slab_co, box_co)
        .expect("slab/box contact pair");
    pair.manifolds[0].data.relative_dominance
}

fn adhesion_does_not_drag_a_dominant_body(mode: Mode) {
    let relative_dominance = check_dominant_slab_is_not_dragged(mode, true);
    assert!(
        relative_dominance > 0,
        "the dominant slab should be the pair's first side (relative dominance \
         {relative_dominance})"
    );
}
both_modes!(adhesion_does_not_drag_a_dominant_body);

fn adhesion_does_not_drag_a_dominant_second_body(mode: Mode) {
    // The mirror of `adhesion_does_not_drag_a_dominant_body`: inserted after the box, the
    // dominant slab is the pair's second side, which exercises the other skip branch.
    let relative_dominance = check_dominant_slab_is_not_dragged(mode, false);
    assert!(
        relative_dominance < 0,
        "the dominant slab should be the pair's second side (relative dominance \
         {relative_dominance})"
    );
}
both_modes!(adhesion_does_not_drag_a_dominant_second_body);

fn non_positive_and_nan_requests_are_bit_identical_to_no_request(mode: Mode) {
    // Zero, negative and NaN requests of every kind produce no force write at all, so the
    // simulation matches a hook that requests nothing, bit for bit.
    struct Inert;
    impl PhysicsHooks for Inert {
        fn modify_solver_contacts(&self, _: &mut ContactModificationContext) {}
    }
    struct NonPositive {
        values: [Real; 3],
    }
    impl PhysicsHooks for NonPositive {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            let [force, pressure, total] = self.values;
            *context.adhesion_force = force;
            *context.adhesion_pressure = pressure;
            *context.adhesion_budget = Some(AdhesionBudget {
                owner: context.collider1,
                channel: 7,
                total,
            });
        }
    }

    let simulate = |hooks: &dyn PhysicsHooks| -> Vec<u32> {
        let mut world = new_world();
        ceiling_and_hanging_box_at(&mut world, 0.0);
        slope_and_capsule(&mut world, Slope::Tiles);
        run(mode, &mut world, hooks, 90);
        body_bits(&world)
    };

    let reference = simulate(&Inert);
    for values in [
        [0.0, 0.0, 0.0],
        [-30.0, -30.0, -30.0],
        [Real::NAN, Real::NAN, Real::NAN],
        [Real::NEG_INFINITY, Real::NEG_INFINITY, Real::NEG_INFINITY],
    ] {
        assert!(
            simulate(&NonPositive { values }) == reference,
            "request {values:?} changed the simulation"
        );
    }
}
both_modes!(non_positive_and_nan_requests_are_bit_identical_to_no_request);

fn tangential_extent_is_cached_at_hook_time(mode: Mode) {
    // `ContactManifoldData::tangential_extent` is the value measured right after the hook ran
    // (the same number `ContactModificationContext::tangential_extent` gives inside the hook),
    // and 0 for a pair without the hook flag.
    struct Recorder {
        collider: ColliderHandle,
        seen: Mutex<Vec<Real>>,
    }
    impl PhysicsHooks for Recorder {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            if context.collider1 == self.collider || context.collider2 == self.collider {
                self.seen.lock().unwrap().push(context.tangential_extent());
            }
        }
    }

    let mut world = new_world();
    let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
    let box_co = world.bodies[box_body].colliders()[0];
    // A second box resting on a plain (hook-less) floor far away.
    let floor_body = world.insert_body(RigidBodyBuilder::fixed());
    let floor = world.insert_collider(
        ColliderBuilder::cuboid(2.0, 0.5).translation(Vector::new(40.0, -0.5)),
        Some(floor_body),
    );
    let (rest_body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(40.0, 0.5)),
        ColliderBuilder::cuboid(0.5, 0.5),
    );
    let rest_co = world.bodies[rest_body].colliders()[0];

    let hook = Recorder {
        collider: ceiling,
        seen: Mutex::new(Vec::new()),
    };
    step(mode, &mut world, &hook);
    step(mode, &mut world, &hook);

    let last_seen = *hook
        .seen
        .lock()
        .unwrap()
        .last()
        .expect("the hook never ran");
    assert_close(last_seen, 1.0, 1.0e-3, "extent of a 1 m box face (2D)");
    let hooked = world.narrow_phase.contact_pair(ceiling, box_co).unwrap();
    assert_eq!(hooked.manifolds[0].data.tangential_extent(), last_seen);

    let plain = world.narrow_phase.contact_pair(floor, rest_co).unwrap();
    assert!(plain.has_any_active_contact());
    assert_eq!(plain.manifolds[0].data.tangential_extent(), 0.0);
}
both_modes!(tangential_extent_is_cached_at_hook_time);

#[test]
fn collisions_last_wake_applies_no_adhesion_after_hook_flag_removal() {
    // A box held under a ceiling by adhesion falls asleep. The ceiling then loses its hook flag,
    // which is not a collider change, so the sleeping pair isn't updated and keeps its request.
    // On the step the box is woken, stock `step` fully updates the pair before its solve (a pair
    // whose last update stored adhesion can't be recycled), which drops the request: the box
    // starts to fall. `step_collisions_last` solves before its detection and puts the woken pair
    // back into the solver without updating it, so the stale request must not hold the box there.
    let wake_step_velocity = |mode: Mode| -> Real {
        let mut world = new_world();
        let (ceiling, box_body) = ceiling_and_hanging_box(&mut world);
        let hook = AdhesionHook {
            collider: ceiling,
            force: 30.0,
        };

        let mut slept = false;
        for _ in 0..600 {
            step(mode, &mut world, &hook);
            if world.bodies[box_body].is_sleeping() {
                slept = true;
                break;
            }
        }
        assert!(slept, "the adhered box never fell asleep ({mode:?})");

        world.colliders[ceiling].set_active_hooks(ActiveHooks::empty());
        step(mode, &mut world, &hook);
        assert!(
            world.bodies[box_body].is_sleeping(),
            "removing the hook flag woke the box ({mode:?})"
        );

        world.bodies[box_body].wake_up(true);
        step(mode, &mut world, &hook);
        world.bodies[box_body].linvel().y
    };

    let dt = new_world().integration_parameters.dt;
    let stock = wake_step_velocity(Mode::Stock);
    let collisions_last = wake_step_velocity(Mode::CollisionsLast);
    assert!(
        stock < -0.5 * G * dt,
        "stock stepping: the box should start falling on the wake step (vy = {stock})"
    );
    assert_close(
        collisions_last,
        stock,
        1.0e-3,
        "collisions-last vertical velocity on the wake step, against stock stepping",
    );
}

fn non_finite_adhesion_term_does_not_drop_the_finite_terms(mode: Mode) {
    // A ball under a hooked ceiling touches it at a single point, so its manifold's tangential
    // extent is 0 and an infinite pressure gives an infinite * 0 = NaN pressure force. That term
    // must add nothing while the finite force requested on the same manifold still applies: the
    // run must be bit-identical to requesting the force alone.
    struct ForceAndPressure {
        collider: ColliderHandle,
        pressure: Option<Real>,
    }
    impl PhysicsHooks for ForceAndPressure {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            if context.collider1 == self.collider || context.collider2 == self.collider {
                *context.adhesion_force = 30.0;
                if let Some(pressure) = self.pressure {
                    *context.adhesion_pressure = pressure;
                }
            }
        }
    }

    let simulate = |pressure: Option<Real>| -> (Vec<u32>, Real) {
        let mut world = new_world();
        let ceiling_body = world.insert_body(RigidBodyBuilder::fixed());
        let ceiling = world.insert_collider(
            ColliderBuilder::cuboid(5.0, 0.5)
                .translation(Vector::new(0.0, 0.5))
                .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS),
            Some(ceiling_body),
        );
        let (ball_body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -0.5)),
            ColliderBuilder::ball(0.5),
        );
        let hook = ForceAndPressure {
            collider: ceiling,
            pressure,
        };
        run(mode, &mut world, &hook, 120);
        (body_bits(&world), world.bodies[ball_body].translation().y)
    };

    let (force_only, y) = simulate(None);
    assert!(
        y > -0.6,
        "the 30 N force alone should hold the ball (y = {y})"
    );
    for pressure in [Real::INFINITY, Real::NAN, Real::NEG_INFINITY] {
        assert!(
            simulate(Some(pressure)).0 == force_only,
            "a pressure of {pressure} on a point contact changed the result of the finite force"
        );
    }
}
both_modes!(non_finite_adhesion_term_does_not_drop_the_finite_terms);

fn positive_infinite_requests_are_bit_identical_to_no_request(mode: Mode) {
    // A non-finite request adds nothing, +inf included: each kind alone, and all three together,
    // match a hook that requests nothing, bit for bit.
    struct Inert;
    impl PhysicsHooks for Inert {
        fn modify_solver_contacts(&self, _: &mut ContactModificationContext) {}
    }
    struct Infinite {
        force: bool,
        pressure: bool,
        budget: bool,
    }
    impl PhysicsHooks for Infinite {
        fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
            if self.force {
                *context.adhesion_force = Real::INFINITY;
            }
            if self.pressure {
                *context.adhesion_pressure = Real::INFINITY;
            }
            if self.budget {
                *context.adhesion_budget = Some(AdhesionBudget {
                    owner: context.collider1,
                    channel: 7,
                    total: Real::INFINITY,
                });
            }
        }
    }

    let simulate = |hooks: &dyn PhysicsHooks| -> Vec<u32> {
        let mut world = new_world();
        ceiling_and_hanging_box_at(&mut world, 0.0);
        slope_and_capsule(&mut world, Slope::Tiles);
        run(mode, &mut world, hooks, 90);
        body_bits(&world)
    };

    let reference = simulate(&Inert);
    for (force, pressure, budget) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (true, true, true),
    ] {
        let hook = Infinite {
            force,
            pressure,
            budget,
        };
        assert!(
            simulate(&hook) == reference,
            "an infinite request (force {force}, pressure {pressure}, budget {budget}) changed \
             the simulation"
        );
    }
}
both_modes!(positive_infinite_requests_are_bit_identical_to_no_request);
