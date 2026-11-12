//! The SIMD contact builders' substep tracking: [`super::substep_tracking`] over the lanes of a constraint chunk, each lane with its own manifold's sides.

use super::substep_tracking::{SideTracking, TrackedPoint, any_tracked, first_order_rotation};
use crate::dynamics::solver::solver_body::{SolverPose, SolverTransform};
use crate::geometry::{ContactManifold, NEW_CONTACT_BIT};
use crate::math::{MAX_MANIFOLD_POINTS, Real, SIMD_WIDTH, SimdReal};
use crate::utils::{DotProduct, RotationOps, ScalarType};
use num::Zero;
use simba::simd::{SimdPartialOrd, SimdValue};

type SimdVector = <SimdReal as ScalarType>::Vector;
type SimdRotation = <SimdReal as ScalarType>::Rotation;

/// The tracking state of one SIMD constraint. `active` is written by every build; the rest is read only when it is true.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SimdTracking {
    /// Whether any lane has a tracked side.
    pub active: bool,
    /// 1 on the lanes with a tracked side, 0 elsewhere.
    lanes: SimdReal,
    /// Per side, 1 on the lanes that track it to first order.
    first_order: [SimdReal; 2],
    /// Per side, whether any lane tracks it to first order.
    any_first_order: [bool; 2],
    /// Per side, the frame rotation at the build, inverted.
    rotations0_inv: [SimdRotation; 2],
    points: [TrackedPoint<SimdReal>; MAX_MANIFOLD_POINTS],
}

/// Per lane, 1 where side `side` of the lane's manifold tracks as `kind`.
fn side_mask(
    manifolds: &[&ContactManifold; SIMD_WIDTH],
    side: usize,
    kind: SideTracking,
) -> SimdReal {
    SimdReal::from(array![
        |ii| (SideTracking::of(manifolds[ii].data.solver_flags, side) == kind) as u32 as Real
    ])
}

impl SimdTracking {
    /// Starts a build over `manifolds`, whose side frames are `poses` at the build.
    pub fn begin(
        &mut self,
        manifolds: &[&ContactManifold; SIMD_WIDTH],
        poses: [&SolverPose<SimdReal>; 2],
    ) {
        self.active = manifolds.iter().any(|m| any_tracked(m.data.solver_flags));
        if !self.active {
            return;
        }
        let one = SimdReal::splat(1.0);
        let mut tracked = SimdReal::zero();
        for side in 0..2 {
            let center = side_mask(manifolds, side, SideTracking::CurvatureCenter);
            let first_order = side_mask(manifolds, side, SideTracking::FirstOrder);
            tracked += center + first_order;
            self.first_order[side] = first_order;
            self.any_first_order[side] = manifolds
                .iter()
                .any(|m| SideTracking::of(m.data.solver_flags, side) == SideTracking::FirstOrder);
            self.rotations0_inv[side] = poses[side].rotation.inverse();
        }
        self.lanes = tracked.simd_min(one);
    }

    /// Records point `k`'s tracked anchors. `ks` is each lane's solver contact index, `p2` side 2's anchor in world space, `points` the world points both sides' lever arms reach, and `body_fixed` each side's body-fixed anchor in its frame.
    #[allow(clippy::too_many_arguments)]
    pub fn set_point(
        &mut self,
        k: usize,
        manifolds: &[&ContactManifold; SIMD_WIDTH],
        ks: [usize; SIMD_WIDTH],
        poses: [&SolverPose<SimdReal>; 2],
        p2: SimdVector,
        points: [SimdVector; 2],
        body_fixed: [SimdVector; 2],
    ) {
        let one = SimdReal::splat(1.0);
        let point = |ii: usize| {
            let contact = &manifolds[ii].data.solver_contacts[ks[ii]];
            &manifolds[ii].points[(contact.contact_id[0] & !NEW_CONTACT_BIT) as usize]
        };
        let normal = SimdVector::from(gather![|ii| manifolds[ii].data.normal.into()]);
        let dist = SimdReal::from(gather![|ii| point(ii).dist]);
        let radius1 = SimdReal::from(gather![|ii| point(ii).local_p1.length()]);
        let radius2 = SimdReal::from(gather![|ii| point(ii).local_p2.length()]);
        // The ball centres, as `super::substep_tracking::curvature_center` derives them.
        let centers = [p2 - normal * (dist + radius1), p2 + normal * radius2];
        let mut tracked = TrackedPoint::<SimdReal>::default();
        for side in 0..2 {
            let center_mask = side_mask(manifolds, side, SideTracking::CurvatureCenter);
            let first_order = self.first_order[side];
            let body_fixed_mask = one - center_mask - first_order;
            let center_local = poses[side].inverse_transform_point(centers[side]);
            tracked.local[side] = body_fixed[side] * body_fixed_mask + center_local * center_mask;
            tracked.arm[side] = (points[side] - centers[side]) * center_mask
                + (points[side] - poses[side].translation) * first_order;
        }
        self.points[k] = tracked;
    }

    /// Side `side`'s point `k` at a substep where its frame is `pose`.
    fn point(&self, k: usize, side: usize, pose: &SolverTransform<SimdReal>) -> SimdVector {
        let tracked = &self.points[k];
        let p = pose.transform_point(tracked.local[side]) + tracked.arm[side];
        if self.any_first_order[side] {
            let delta = pose.rotation * self.rotations0_inv[side];
            p + first_order_rotation::<SimdReal>(delta, tracked.arm[side]) * self.first_order[side]
        } else {
            p
        }
    }

    /// Point `k`'s separation: `dist`, the body-fixed one, on lanes without a tracked side, and the tracked one elsewhere. `shift` is the first side's tangent-velocity drift and `info_dist` the rebased base separation.
    pub fn separation(
        &self,
        k: usize,
        poses: [&SolverTransform<SimdReal>; 2],
        dir1: SimdVector,
        info_dist: SimdReal,
        shift: SimdVector,
        dist: SimdReal,
    ) -> SimdReal {
        let p1 = self.point(k, 0, poses[0]) + shift;
        let p2 = self.point(k, 1, poses[1]);
        let tracked = info_dist + (p1 - p2).gdot(dir1);
        tracked.select(self.lanes.simd_gt(SimdReal::zero()), dist)
    }
}
