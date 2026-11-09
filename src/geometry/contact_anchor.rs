//! How the solver tracks a contact's separation on one collider's side between collision detection and the end of the step.

use crate::geometry::Collider;
use crate::math::Vector;
use core::fmt;
use parry::shape::ShapeType;

/// How the solver follows a collider's side of its contacts through the substeps of one step.
///
/// Collision detection runs once per step. The solver then re-measures each contact's separation at every substep, from a point it moves along with the bodies since detection. This enum picks how that point moves on this collider's side.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ContactAnchor {
    /// The point is fixed on the body: it translates and rotates with it. Right for a flat face resting on another surface, and Rapier's behaviour for every shape.
    #[default]
    BodyFixed,
    /// The point keeps its offset from the ball's centre, as it was at detection, and translates with the centre without rotating. This is where a ball's contact really is as it spins: under the centre, whatever its rotation. A body-fixed point on a spinning ball instead rides round the rim and reads as lifting off the surface. Only the separation uses this point; friction's slip still tracks the point fixed on the body, which a rolling ball leaves at rest. Allowed on balls only.
    Round,
}

/// [`ContactAnchor::Round`] was asked for on a collider whose shape is not a ball.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RoundAnchorNeedsBall {
    /// The shape the collider has.
    pub shape: ShapeType,
}

impl fmt::Display for RoundAnchorNeedsBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "round contact anchors need a ball collider, and this one is a {:?}: only on a ball is the contact always under the centre whatever the rotation",
            self.shape
        )
    }
}

impl core::error::Error for RoundAnchorNeedsBall {}

impl ContactAnchor {
    /// `Ok(self)` if a collider of shape `shape` may use this anchor.
    pub fn check_shape(self, shape: ShapeType) -> Result<Self, RoundAnchorNeedsBall> {
        match self {
            ContactAnchor::Round if shape != ShapeType::Ball => Err(RoundAnchorNeedsBall { shape }),
            _ => Ok(self),
        }
    }
}

/// The world-space centre of `co`'s ball, if `co` uses [`ContactAnchor::Round`] and is still a ball.
///
/// A ball's centre is its collider's origin, so this is the collider's world translation.
pub(crate) fn round_anchor_center(co: &Collider) -> Option<Vector> {
    (co.contact_anchor() == ContactAnchor::Round && co.shape().shape_type() == ShapeType::Ball)
        .then_some(co.pos.translation)
}
