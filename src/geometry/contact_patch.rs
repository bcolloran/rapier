//! Geometry of the convex patch carrying an adhesion pressure.

use crate::math::{Real, Vector};
use crate::utils::OrthonormalBasis;

pub(crate) struct ContactPatch {
    pub extent: Real,
    pub centroid: Vector,
}

#[cfg(feature = "dim2")]
pub(crate) fn contact_patch(
    normal: Vector,
    points: impl IntoIterator<Item = Vector>,
) -> ContactPatch {
    let mut points = points.into_iter();
    let origin = points.next().unwrap_or(Vector::ZERO);
    let tangent = normal.orthonormal_vector();
    let mut min: Real = 0.0;
    let mut max: Real = 0.0;
    for point in points {
        let coordinate = tangent.dot(point - origin);
        min = min.min(coordinate);
        max = max.max(coordinate);
    }
    ContactPatch {
        extent: max - min,
        centroid: origin + tangent * ((min + max) * 0.5),
    }
}

#[cfg(feature = "dim3")]
pub(crate) fn contact_patch(
    normal: Vector,
    points: impl IntoIterator<Item = Vector>,
) -> ContactPatch {
    use crate::alloc_prelude::*;
    use crate::math::Vector2;

    let mut points = points.into_iter();
    let origin = points.next().unwrap_or(Vector::ZERO);
    let [b1, b2] = normal.orthonormal_basis();
    // Subtract before projecting: area must not lose precision as the world translates.
    let mut projected: Vec<_> = core::iter::once(Vector2::ZERO)
        .chain(points.map(|point| {
            let relative = point - origin;
            Vector2::new(b1.dot(relative), b2.dot(relative))
        }))
        .collect();
    if projected.len() < 3 || projected.iter().all(|point| *point == Vector2::ZERO) {
        return ContactPatch {
            extent: 0.0,
            centroid: origin,
        };
    }

    // Monotone-chain hull: hooks can reorder or duplicate points, or leave interior points.
    projected.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    projected.dedup();
    fn push_vertex(hull: &mut Vec<Vector2>, point: Vector2, start: usize) {
        while hull.len() >= start + 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let ab = b - a;
            let ap = point - a;
            if ab.x * ap.y - ab.y * ap.x > 0.0 {
                break;
            }
            hull.pop();
        }
        hull.push(point);
    }
    let mut hull = Vec::with_capacity(projected.len() * 2);
    for &point in &projected {
        push_vertex(&mut hull, point, 0);
    }
    let upper_start = hull.len() - 1;
    for &point in projected.iter().rev().skip(1) {
        push_vertex(&mut hull, point, upper_start);
    }
    hull.pop();
    let mut twice_area: Real = 0.0;
    let mut first_moment = Vector2::ZERO;
    for i in 0..hull.len() {
        let p = hull[i];
        let q = hull[(i + 1) % hull.len()];
        let cross = p.x * q.y - q.x * p.y;
        twice_area += cross;
        first_moment += (p + q) * cross;
    }
    if twice_area == 0.0 {
        return ContactPatch {
            extent: 0.0,
            centroid: origin,
        };
    }
    let center = first_moment / (3.0 * twice_area);
    ContactPatch {
        extent: twice_area.abs() * 0.5,
        centroid: origin + b1 * center.x + b2 * center.y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[cfg(feature = "dim2")]
    #[test]
    fn translated_contact_span_and_midpoint() {
        let patch = contact_patch(
            Vector::Y,
            [Vector::new(10000.0, 10000.0), Vector::new(10002.0, 10000.0)],
        );
        assert_abs_diff_eq!(patch.extent, 2.0);
        assert_abs_diff_eq!(patch.centroid.x, 10001.0);
    }

    #[cfg(feature = "dim3")]
    #[test]
    fn translated_contact_area_and_centroid() {
        for shift in [Vector::ZERO, Vector::splat(10000.0)] {
            let points = [Vector::ZERO, Vector::X, Vector::X + Vector::Z, Vector::Z];
            let patch = contact_patch(Vector::Y, points.map(|point| point + shift));
            assert_abs_diff_eq!(patch.extent, 1.0, epsilon = 1.0e-5);
            assert_abs_diff_eq!(patch.centroid.x - shift.x, 0.5, epsilon = 1.0e-5);
            assert_abs_diff_eq!(patch.centroid.z - shift.z, 0.5, epsilon = 1.0e-5);
        }
    }

    #[cfg(feature = "dim3")]
    #[test]
    fn pressure_resultant_is_invariant_under_patch_subdivision() {
        let points = [
            Vector::ZERO,
            Vector::new(4.0, 0.0, 0.0),
            Vector::new(1.0, 0.0, 2.0),
            Vector::new(0.0, 0.0, 2.0),
        ];
        let whole = contact_patch(Vector::Y, points);
        let left = contact_patch(Vector::Y, [points[0], points[1], points[2]]);
        let right = contact_patch(Vector::Y, [points[0], points[2], points[3]]);
        assert_abs_diff_eq!(whole.extent, 5.0, epsilon = 1.0e-5);
        assert_abs_diff_eq!(whole.extent, left.extent + right.extent, epsilon = 1.0e-5);
        let moment = whole.centroid * whole.extent;
        let subdivided_moment = left.centroid * left.extent + right.centroid * right.extent;
        assert!((moment - subdivided_moment).length() < 1.0e-5);
        assert_abs_diff_eq!(moment.x, 7.0, epsilon = 1.0e-5);
        assert_abs_diff_eq!(moment.z, 4.0, epsilon = 1.0e-5);
    }

    #[cfg(feature = "dim3")]
    #[test]
    fn duplicate_interior_and_collinear_contacts_do_not_inflate_area() {
        let square = [
            Vector::ZERO,
            Vector::X,
            Vector::X + Vector::Z,
            Vector::Z,
            Vector::ZERO,
            (Vector::X + Vector::Z) * 0.5,
        ];
        assert_abs_diff_eq!(
            contact_patch(Vector::Y, square).extent,
            1.0,
            epsilon = 1.0e-5
        );
        assert_eq!(contact_patch(Vector::Y, [Vector::ZERO; 4]).extent, 0.0);
        assert_eq!(
            contact_patch(Vector::Y, [Vector::ZERO, Vector::X, Vector::X * 2.0]).extent,
            0.0
        );
    }
}
