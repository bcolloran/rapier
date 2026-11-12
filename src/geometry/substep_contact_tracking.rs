//! How the solver moves one collider's side of a contact through the substeps of a step.
//!
//! Collision detection runs once per step. The solver then re-measures each contact's separation at every substep from a point it moves along with the bodies since detection. Rapier moves that point rigidly with the body, so a fast-spinning ball's point rides round its rim and reads as lifting off the surface it rests on. [`SubstepContactTracking`] lets a collider choose another rule for its side. Only the separation changes: the lever arms, the normal and friction keep the point fixed on the body.
//!
//! A collider's choice is two bits of its [`ActiveHooks`], outside the public hook flags, so a collider that keeps the default stores nothing more than before. A manifold's choice is the matching bits of its [`SolverFlags`], set by the narrow phase at every full contact update.

use crate::geometry::{Collider, ColliderBuilder, SolverFlags};
use crate::pipeline::ActiveHooks;
use core::fmt;
use parry::shape::ShapeType;

/// How the solver moves a collider's side of its contacts between detection and each substep's separation.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum SubstepContactTracking {
    /// The point is fixed on the body: it translates and rotates with it. Exact for a flat face or a corner, and Rapier's behaviour.
    #[default]
    BodyFixed,
    /// The point keeps its detected world-space offset from the ball's centre, and the centre moves with the body. The centre's offset from the body's frame rotates with the body, so a ball off its body's centre of mass is tracked too; the offset from the centre to the contact does not rotate, which is where a ball's contact stays however it spins. Balls only.
    CurvatureCenter,
    /// The point moves to first order in the body's motion since detection: its translation, plus the rotation applied to the frozen lever arm as `θ × r`, with `θ` the rotation's sine-scaled axis. The separation then changes by `n·Δx + (r × n)·θ`, the update of PhysX's TGS and Bepu's substepper. A ball or a shape rolling about its own axis has `r × n = 0`, so its spin does not move the separation. Any shape; it drops the second-order term of a tumbling body's rotation.
    FirstOrder,
}

/// [`SubstepContactTracking::CurvatureCenter`] was asked of a collider whose shape is not a ball.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CurvatureCenterNeedsBall {
    /// The shape the collider has.
    pub shape: ShapeType,
}

impl fmt::Display for CurvatureCenterNeedsBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "curvature-centre contact tracking needs a ball collider, and this one is a {:?}: the solver finds the centre from the ball's contact point and normal alone, which only a ball's contact determines",
            self.shape
        )
    }
}

impl core::error::Error for CurvatureCenterNeedsBall {}

/// The [`ActiveHooks`] bit of [`SubstepContactTracking::CurvatureCenter`]: outside every public hook flag.
const CURVATURE_CENTER_BIT: ActiveHooks = ActiveHooks::from_bits_retain(1 << 30);
/// The [`ActiveHooks`] bit of [`SubstepContactTracking::FirstOrder`].
const FIRST_ORDER_BIT: ActiveHooks = ActiveHooks::from_bits_retain(1 << 31);
/// Both tracking bits.
pub(crate) const TRACKING_BITS: ActiveHooks =
    ActiveHooks::from_bits_retain(CURVATURE_CENTER_BIT.bits() | FIRST_ORDER_BIT.bits());

impl SubstepContactTracking {
    /// `Ok(self)` if a collider of shape `shape` may use this tracking.
    pub fn check_shape(self, shape: ShapeType) -> Result<Self, CurvatureCenterNeedsBall> {
        match self {
            Self::CurvatureCenter if shape != ShapeType::Ball => {
                Err(CurvatureCenterNeedsBall { shape })
            }
            _ => Ok(self),
        }
    }

    /// The tracking a collider's hook bits hold.
    pub(crate) fn from_hook_bits(hooks: ActiveHooks) -> Self {
        if hooks.contains(CURVATURE_CENTER_BIT) {
            Self::CurvatureCenter
        } else if hooks.contains(FIRST_ORDER_BIT) {
            Self::FirstOrder
        } else {
            Self::BodyFixed
        }
    }

    /// `hooks` with its tracking bits set to this tracking.
    pub(crate) fn into_hook_bits(self, hooks: ActiveHooks) -> ActiveHooks {
        let hooks = hooks - TRACKING_BITS;
        match self {
            Self::BodyFixed => hooks,
            Self::CurvatureCenter => hooks | CURVATURE_CENTER_BIT,
            Self::FirstOrder => hooks | FIRST_ORDER_BIT,
        }
    }

    /// The manifold flags of this tracking on side `side` (0 or 1) of a pair.
    fn solver_flags(self, side: usize) -> SolverFlags {
        match (self, side) {
            (Self::BodyFixed, _) => SolverFlags::empty(),
            (Self::CurvatureCenter, 0) => SolverFlags::CURVATURE_CENTER1,
            (Self::CurvatureCenter, _) => SolverFlags::CURVATURE_CENTER2,
            (Self::FirstOrder, 0) => SolverFlags::FIRST_ORDER1,
            (Self::FirstOrder, _) => SolverFlags::FIRST_ORDER2,
        }
    }
}

/// The tracking `co` contacts with: its own, except that a collider no longer a ball contacts as [`SubstepContactTracking::BodyFixed`] where it asked for the curvature centre.
pub(crate) fn effective_tracking(co: &Collider) -> SubstepContactTracking {
    let tracking = SubstepContactTracking::from_hook_bits(co.flags.active_hooks);
    tracking
        .check_shape(co.shape.shape_type())
        .unwrap_or(SubstepContactTracking::BodyFixed)
}

/// The manifold flags for a pair of `co1` and `co2`; empty when both track body-fixed.
pub(crate) fn pair_tracking_flags(co1: &Collider, co2: &Collider) -> SolverFlags {
    effective_tracking(co1).solver_flags(0) | effective_tracking(co2).solver_flags(1)
}

impl Collider {
    /// How the solver moves this collider's side of its contacts through the substeps; see [`SubstepContactTracking`].
    ///
    /// A collider asked for [`SubstepContactTracking::CurvatureCenter`] whose shape is no longer a ball still returns it, and contacts as [`SubstepContactTracking::BodyFixed`].
    pub fn substep_contact_tracking(&self) -> SubstepContactTracking {
        SubstepContactTracking::from_hook_bits(self.flags.active_hooks)
    }

    /// Sets how the solver moves this collider's side of its contacts through the substeps.
    ///
    /// Refuses [`SubstepContactTracking::CurvatureCenter`] unless this collider is a ball, and leaves the tracking unchanged then. Takes effect at the pair's next full contact update; a pair with a tracked side is never recycled, so that is the next step.
    pub fn set_substep_contact_tracking(
        &mut self,
        tracking: SubstepContactTracking,
    ) -> Result<(), CurvatureCenterNeedsBall> {
        let tracking = tracking.check_shape(self.shape.shape_type())?;
        self.flags.active_hooks = tracking.into_hook_bits(self.flags.active_hooks);
        Ok(())
    }
}

impl ColliderBuilder {
    /// Sets how the solver moves the collider's side of its contacts through the substeps; see [`SubstepContactTracking`].
    ///
    /// Refuses [`SubstepContactTracking::CurvatureCenter`] unless the builder's shape is a ball.
    pub fn substep_contact_tracking(
        mut self,
        tracking: SubstepContactTracking,
    ) -> Result<Self, CurvatureCenterNeedsBall> {
        self.substep_contact_tracking = tracking.check_shape(self.shape.shape_type())?;
        Ok(self)
    }
}
