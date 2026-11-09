//! The per-substep separation of contacts with a [`ContactAnchor::Round`](crate::geometry::ContactAnchor::Round) side.
//!
//! A body-fixed side re-measures its separation from `pose * local_p`, a point painted on the body. A round side measures it from `pose * center + arm`: the ball's centre in the side's solver frame, plus the contact's world offset from that centre at constraint generation, which does not rotate. Friction's slip keeps the body-fixed point, the material point a rolling ball leaves at rest.

use crate::utils::ScalarType;

/// One contact point's anchors for the round separation, one entry per side.
///
/// A round side stores its ball's centre in its solver frame and the world offset from it; a body-fixed side stores its body-fixed anchor and a zero offset, so `pose * local + arm` is that side's anchor either way.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct RoundAnchorPoint<N: ScalarType> {
    pub local: [N::Vector; 2],
    pub arm: [N::Vector; 2],
}
