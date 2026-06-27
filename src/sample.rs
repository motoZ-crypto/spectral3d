//! Deterministic, order-independent surface sampling.
//!
//! Each triangle is subdivided into k² congruent sub-triangles (k chosen
//! from its area share) and sampled at sub-triangle centroids with weight
//! area/k². The rule is purely local to each triangle, so it commutes
//! exactly with rigid motion and is unaffected by face/vertex ordering.

use crate::mesh::{cross, sub, Mesh};
use alloc::vec::Vec;
use libm::{ceil, sqrt};

pub struct Samples {
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub total_area: f64,
}

/// Sample the surface at roughly `target` points, returning each point with its
/// integration weight plus the total area (see [`Samples`]). `target` only sets
/// the subdivision density, so the actual count lands near it, not exactly on
/// it. A zero-area mesh yields no samples. The sampling rule is in the module
/// header.
pub fn sample_surface(mesh: &Mesh, target: usize) -> Samples {
    let mut areas = Vec::with_capacity(mesh.faces.len());
    let mut total = 0.0;
    for &f in &mesh.faces {
        let (a, b, c) = mesh.tri(f);
        let n = cross(sub(b, a), sub(c, a));
        let area = 0.5 * sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        areas.push(area);
        total += area;
    }
    let mut points = Vec::new();
    let mut weights = Vec::new();
    if total <= 0.0 {
        return Samples {
            points,
            weights,
            total_area: 0.0,
        };
    }
    let per = total / target.max(1) as f64;
    for (t, &f) in mesh.faces.iter().enumerate() {
        let area = areas[t];
        if area <= 0.0 {
            continue;
        }
        // k = sub-triangles per side, ~proportional to the face's area share.
        // Capped at 64 (<= 4096 samples on one face) so a single outsized face
        // cannot blow up the sample count. That under-samples a face which
        // alone dominates the surface, but the result is deterministic, so it
        // shifts no identity and is absorbed by the quant buckets. The floor of
        // 1 is definitional: a face needs at least one sample.
        let k = (ceil(sqrt(area / per)) as usize).clamp(1, 64);
        let (a, b, c) = mesh.tri(f);
        let w = area / (k * k) as f64;
        let kf = (3 * k) as f64;
        let bary = |u: f64, v: f64| -> [f64; 3] {
            [
                a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
                a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
                a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]),
            ]
        };
        for i in 0..k {
            for j in 0..(k - i) {
                // upward sub-triangle centroid
                points.push(bary((3 * i + 1) as f64 / kf, (3 * j + 1) as f64 / kf));
                weights.push(w);
                // downward sub-triangle centroid
                if i + j < k - 1 {
                    points.push(bary((3 * i + 2) as f64 / kf, (3 * j + 2) as f64 / kf));
                    weights.push(w);
                }
            }
        }
    }
    Samples {
        points,
        weights,
        total_area: total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_area() {
        let m = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            faces: vec![[0, 1, 2]],
        };
        let s = sample_surface(&m, 100);
        let w: f64 = s.weights.iter().sum();
        assert!((w - 2.0).abs() < 1e-12, "sum {w}");
        assert!((s.total_area - 2.0).abs() < 1e-12);
        // centroid of samples == triangle centroid (centroid rule is exact for affine)
        let mut c = [0.0; 3];
        for (p, w) in s.points.iter().zip(&s.weights) {
            for k in 0..3 {
                c[k] += p[k] * w;
            }
        }
        for ck in c.iter_mut() {
            *ck /= 2.0;
        }
        assert!((c[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((c[1] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn subdivision_count_is_k_squared() {
        let m = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
        };
        // target 9 cells on a single triangle → k = 3 → 9 samples
        let s = sample_surface(&m, 9);
        assert_eq!(s.points.len(), 9);
    }
}
