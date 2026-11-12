//! A frictionless ball, a rigid body or a multibody wheel, spinning over fixed surfaces.

use rapier2d::prelude::*;

pub const RADIUS: Real = 0.3;

/// How far an [`BallBody::OffCentre`] ball sits from its body's origin, m.
pub const OFF_CENTRE: Real = 0.2;

/// How the ball is simulated: its contacts take the rigid-body or the multibody (generic) constraint path.
#[derive(Copy, Clone, Debug)]
pub enum BallBody {
    Rigid,
    /// A wheel on a revolute joint under a torso that translates freely and cannot rotate, as the game's wheeled player.
    MultibodyWheel,
    /// A rigid body whose ball is [`OFF_CENTRE`] to the right of the body's origin, with an extra point mass at the origin, so the centre of mass lies between them. Its `spin` turns it about the ball's centre: the centre of mass moves round it, the ball spins in place.
    OffCentre,
}

/// The fixed surfaces under the ball. Each is a set of half-planes `(point, outward normal)`, one manifold each.
#[derive(Copy, Clone, Debug)]
pub enum Surface {
    Flat,
    /// A plane rising 30° to the right.
    Slope,
    /// Two planes at ±45° meeting under the ball.
    VGroove,
}

impl Surface {
    pub fn planes(self) -> Vec<(Vector, Vector)> {
        let tilted = |angle: Real| Vector::new(-angle.sin(), angle.cos());
        match self {
            Surface::Flat => vec![(Vector::ZERO, Vector::Y)],
            Surface::Slope => vec![(Vector::ZERO, tilted(30.0_f32.to_radians()))],
            Surface::VGroove => vec![
                (Vector::ZERO, tilted(45.0_f32.to_radians())),
                (Vector::ZERO, tilted(-45.0_f32.to_radians())),
            ],
        }
    }
}

pub struct World {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    soft_bodies: SoftBodySet,
    pipeline: PhysicsPipeline,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    islands: IslandManager,
    ccd: CCDSolver,
    params: IntegrationParameters,
    pub gravity: Vector,
    pub ball: RigidBodyHandle,
    wheel_joint: Option<MultibodyJointHandle>,
    /// The ball's centre in its body's frame.
    offset: Vector,
}

impl World {
    /// A ball of [`RADIUS`] centred at `center`, moving at `velocity` and spinning at `spin`, over `surface`.
    pub fn new(
        body: BallBody,
        surface: Surface,
        tracking: SubstepContactTracking,
        center: Vector,
        velocity: Vector,
        spin: Real,
    ) -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let ground = bodies.insert(RigidBodyBuilder::fixed());
        for (point, normal) in surface.planes() {
            let angle = normal.y.atan2(normal.x) - core::f32::consts::FRAC_PI_2;
            let half_thickness = 1.0;
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(20.0, half_thickness)
                    .translation(point - normal * half_thickness)
                    .rotation(angle)
                    .friction(0.0),
                ground,
                &mut bodies,
            );
        }
        let offset = match body {
            BallBody::OffCentre => Vector::X * OFF_CENTRE,
            _ => Vector::ZERO,
        };
        let mut ball_body = RigidBodyBuilder::dynamic()
            .translation(center - offset)
            .linvel(velocity)
            .angvel(spin)
            .can_sleep(false);
        if let BallBody::OffCentre = body {
            ball_body =
                ball_body.additional_mass_properties(MassProperties::new(Vector::ZERO, 200.0, 1.0));
        }
        let ball = bodies.insert(ball_body);
        let ball_collider = ColliderBuilder::ball(RADIUS)
            .translation(offset)
            .friction(0.0)
            .density(1000.0)
            .substep_contact_tracking(tracking)
            .expect("a ball takes any substep contact tracking");
        colliders.insert_with_parent(ball_collider, ball, &mut bodies);
        if let BallBody::OffCentre = body {
            // The centre of mass circles the ball's centre: its velocity is the spin's at its offset from that centre, plus the ball's own.
            let rb = &mut bodies[ball];
            let from_center = rb.center_of_mass() - center;
            rb.set_linvel(
                velocity + Vector::new(-from_center.y, from_center.x) * spin,
                true,
            );
        }

        let mut multibody_joints = MultibodyJointSet::new();
        let wheel_joint = match body {
            BallBody::Rigid | BallBody::OffCentre => None,
            BallBody::MultibodyWheel => {
                let root = bodies.insert(RigidBodyBuilder::fixed().translation(center));
                let torso = bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(center)
                        .additional_mass(20.0)
                        .can_sleep(false),
                );
                multibody_joints.insert(
                    root,
                    torso,
                    GenericJointBuilder::new(JointAxesMask::ANG_X).build(),
                    true,
                );
                multibody_joints.insert(torso, ball, RevoluteJointBuilder::new().build(), true)
            }
        };
        let mut world = Self {
            bodies,
            colliders,
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints,
            soft_bodies: SoftBodySet::new(),
            pipeline: PhysicsPipeline::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            islands: IslandManager::new(),
            ccd: CCDSolver::new(),
            params: IntegrationParameters::default(),
            gravity: Vector::new(0.0, -9.81),
            ball,
            wheel_joint,
            offset,
        };
        world.set_multibody_velocity(velocity, spin);
        world
    }

    /// Writes the multibody wheel's generalized velocities: the torso's translation and the wheel's spin. No-op for a rigid ball, which took them at creation.
    fn set_multibody_velocity(&mut self, velocity: Vector, spin: Real) {
        let Some(wheel_joint) = self.wheel_joint else {
            return;
        };
        let (multibody, _) = self.multibody_joints.get_mut(wheel_joint).unwrap();
        // Settles the generalized layout: the root is fixed, so the torso's two translations come first.
        multibody.forward_kinematics(&self.bodies, true);
        multibody.update_rigid_bodies(&mut self.bodies, true);
        let (multibody, wheel_link) = self.multibody_joints.get_mut(wheel_joint).unwrap();
        assert_eq!(
            multibody.ndofs(),
            3,
            "the torso's x and y, the wheel's spin"
        );
        multibody.damping_mut().fill(0.0);
        let spin_dof = multibody.link(wheel_link).unwrap().assembly_id();
        let mut v = multibody.generalized_velocity_mut();
        v[0] = velocity.x;
        v[1] = velocity.y;
        v[spin_dof] = spin;
    }

    pub fn step(&mut self) {
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
            &(),
            &(),
        );
    }

    pub fn center(&self) -> Vector {
        self.bodies[self.ball]
            .position()
            .transform_point(self.offset)
    }

    /// The ball's true gap to the nearest plane of `surface`, negative when it overlaps.
    pub fn gap(&self, surface: Surface) -> Real {
        let c = self.center();
        surface
            .planes()
            .into_iter()
            .map(|(point, normal)| (c - point).dot(normal) - RADIUS)
            .fold(Real::INFINITY, Real::min)
    }

    /// The ball's centre after each of `steps` steps.
    pub fn trajectory(&mut self, steps: usize) -> Vec<Vector> {
        (0..steps)
            .map(|_| {
                self.step();
                self.center()
            })
            .collect()
    }
}
