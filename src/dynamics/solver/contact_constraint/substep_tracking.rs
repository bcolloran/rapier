//! The per-substep contact point of a side with [`SubstepContactTracking`](crate::geometry::SubstepContactTracking) other than body-fixed.
//!
//! A body-fixed side re-measures its separation from `pose * local_p`, a point painted on the body. A tracked side measures it from `pose * local + arm + θ × arm`, where `pose` is the side's frame at the substep (the frame the body-fixed anchor rides in) and `θ` the first-order rotation of that frame since the constraint was built:
//!
//! - **Curvature centre:** `local` is the ball's centre in the side's frame and `arm` the world offset from the centre to the contact at the build, with `θ = 0`. The centre's offset from the frame turns with the body; the arm does not.
//! - **First order:** `local` is the frame's origin and `arm` the frozen lever arm from it to the contact, with `θ` the rotation since the build, sine-scaled. The separation then moves by `n·Δx + (arm × n)·θ`.
//!
//! At the build both give the body-fixed point, so the base separation is unchanged. Only the separation uses them: the lever arms, the normal and friction keep the body-fixed point.

use crate::geometry::{ContactManifold, SolverContact, SolverFlags};
use crate::math::{MAX_MANIFOLD_POINTS, Real, Vector};
#[cfg(feature = "dim3")]
use crate::utils::CrossProduct;
use crate::utils::{RotationOps, ScalarType};

/// How one side of a manifold moves its contact point through the substeps.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SideTracking {
    BodyFixed,
    CurvatureCenter,
    FirstOrder,
}

impl SideTracking {
    /// Side `side`'s tracking in a manifold's solver flags.
    pub fn of(flags: SolverFlags, side: usize) -> Self {
        let (center, first_order) = if side == 0 {
            (SolverFlags::CURVATURE_CENTER1, SolverFlags::FIRST_ORDER1)
        } else {
            (SolverFlags::CURVATURE_CENTER2, SolverFlags::FIRST_ORDER2)
        };
        if flags.contains(center) {
            Self::CurvatureCenter
        } else if flags.contains(first_order) {
            Self::FirstOrder
        } else {
            Self::BodyFixed
        }
    }
}

/// Whether any side of a manifold with these flags is tracked.
pub(crate) fn any_tracked(flags: SolverFlags) -> bool {
    flags.intersects(
        SolverFlags::CURVATURE_CENTER1
            | SolverFlags::CURVATURE_CENTER2
            | SolverFlags::FIRST_ORDER1
            | SolverFlags::FIRST_ORDER2,
    )
}

/// One contact point's tracked anchors, one entry per side: the side's point at a substep is `pose * local + arm + θ × arm`. A body-fixed side stores its body-fixed anchor and a zero arm.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct TrackedPoint<N: ScalarType> {
    pub local: [N::Vector; 2],
    pub arm: [N::Vector; 2],
}

/// The world centre of a curvature-centre side's ball at the build, for one solver contact.
///
/// The narrow phase stores the ball side's contact point on the ball's surface along the normal (Parry's ball manifolds put it at `local_n * radius` from the centre), and side 2's anchor at its raw surface point. So side 2's ball centre is `p2 + n·r2`, and side 1's is `p2 − n·(d + r1)`, with `p2` side 2's anchor in world space, `n` the manifold normal from side 1 to side 2, `d` the detected distance and `r` a ball's radius, the length of its contact point.
pub(crate) fn curvature_center(
    manifold: &ContactManifold,
    contact: &SolverContact,
    side: usize,
    p2: Vector,
) -> Vector {
    let cid = (contact.contact_id[0] & !crate::geometry::NEW_CONTACT_BIT) as usize;
    let point = &manifold.points[cid];
    let normal = manifold.data.normal;
    if side == 0 {
        p2 - normal * (point.dist + point.local_p1.length())
    } else {
        p2 + normal * point.local_p2.length()
    }
}

/// The first-order rotation term `θ × arm` for a frame that turned by `delta` since the build: `θ` is the rotation's axis scaled by the sine of its angle, read off the rotation without trigonometry.
#[cfg(feature = "dim2")]
pub(crate) fn first_order_rotation<N: ScalarType>(delta: N::Rotation, arm: N::Vector) -> N::Vector {
    delta.imag().gcross(arm)
}

/// The first-order rotation term `θ × arm` for a frame that turned by `delta` since the build: `θ = 2·w·v` for the unit quaternion `(w, v)`, its axis scaled by the sine of its angle.
#[cfg(feature = "dim3")]
pub(crate) fn first_order_rotation<N: ScalarType>(delta: N::Rotation, arm: N::Vector) -> N::Vector {
    (delta.imag() * (delta.real() * N::splat(2.0))).gcross(arm)
}

/// The tracked anchors of one point of a scalar (generic-path) constraint. `p2` is side 2's solver anchor in world space; `frames[i]` maps side `i`'s frame to world at the build, `body_fixed[i]` is the side's body-fixed anchor in that frame, and `points[i]` the world point side `i`'s lever arm reaches.
pub(crate) fn tracked_point(
    kinds: [SideTracking; 2],
    manifold: &ContactManifold,
    contact: &SolverContact,
    p2: Vector,
    frames: [crate::math::Pose; 2],
    body_fixed: [Vector; 2],
    points: [Vector; 2],
) -> TrackedPoint<Real> {
    let side = |i: usize| match kinds[i] {
        SideTracking::BodyFixed => (body_fixed[i], Vector::ZERO),
        SideTracking::CurvatureCenter => {
            let center = curvature_center(manifold, contact, i, p2);
            (
                frames[i].inverse_transform_point(center),
                points[i] - center,
            )
        }
        SideTracking::FirstOrder => (Vector::ZERO, points[i] - frames[i].translation),
    };
    let (local1, arm1) = side(0);
    let (local2, arm2) = side(1);
    TrackedPoint {
        local: [local1, local2],
        arm: [arm1, arm2],
    }
}

/// The tracking state of a scalar (generic-path) constraint with a tracked side: each side's kind and build-time frame rotation, and each point's tracked anchors.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ScalarTracking {
    pub kinds: [SideTracking; 2],
    pub rotations0: [crate::math::Rotation; 2],
    pub points: [TrackedPoint<Real>; MAX_MANIFOLD_POINTS],
}

impl ScalarTracking {
    /// Side `i`'s point `k` at a substep where its frame is `pose`.
    #[inline]
    pub fn point(&self, k: usize, i: usize, pose: &crate::math::Pose) -> Vector {
        let tracked = &self.points[k];
        let p = *pose * tracked.local[i] + tracked.arm[i];
        if self.kinds[i] == SideTracking::FirstOrder {
            let delta = pose.rotation * self.rotations0[i].inverse();
            p + first_order_rotation::<Real>(delta, tracked.arm[i])
        } else {
            p
        }
    }
}
