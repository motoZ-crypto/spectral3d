//! Rotation-invariant feature vector.
//!
//! 23 dims, all Lipschitz functionals of the surface:
//! - f0,f1: covariance eigenvalue ratios λ2/λ1, λ3/λ1
//! - f2:    mean radius r̄ (compactness; volume already normalized to 1)
//! - f3..8: area-weighted spherical-harmonic power, l=1..6
//! - f9..14: radius-weighted spherical-harmonic power, l=1..6
//! - f15..22: radial histogram, 8 bins, triangular (linear-interp) kernel
//!
//! Surface area used to be f3. It tangled shape with mesh density: on a dense
//! mesh, scan noise crinkles the surface and inflates the area past a bucket,
//! flipping the identity, while adding little the eigen-ratios and SH power
//! don't already carry. Dropping it is what lets dense real scans re-verify.
//!
//! Per-l SH power Σ_m |a_lm|² is invariant under rotation of the sample
//! point set itself (not merely in the integral limit), so rigid motion
//! changes these features only at float-roundoff level.

use crate::sample::Samples;
use alloc::vec;
use libm::{sqrt, floor};

pub const N_FEATURES: usize = 23;
pub const HIST_BINS: usize = 8;
const HIST_STEP: f64 = 0.25; // bin centers at 0.125 + j·0.25 on t = r/r̄ ∈ [0,2]

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "lam21", "lam31", "rbar",
    "p1", "p2", "p3", "p4", "p5", "p6",
    "q1", "q2", "q3", "q4", "q5", "q6",
    "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7",
];

/// Shape-gate bounds on `lam31 = λ3/λ1`, the smallest-to-largest covariance
/// eigenvalue ratio (always `lam31 ≤ lam21 ≤ 1`, so this one ratio gates both
/// failure modes):
/// - **below [`LAM31_MIN`]** → near-degenerate, the vertex spread collapsed
///   onto a plane with no real third axis to measure against. Reject.
/// - **above [`LAM31_MAX`]** → near-regular (sphere, cube). Too isotropic to
///   anchor a stable identity, and regular shapes are out of scope by design.
///   Reject.
///
/// The admitted window is [0.005, 0.95]: real irregular solids land inside,
/// down to genuinely thin ones like shields and plates, while only the
/// degenerate tail (lam31 → 0) and the near-regular tail (sphere, cube ≈ 1.0)
/// fall outside.
///
/// The floor used to sit at 0.15, on the theory that near-flat shapes have
/// eigen-ratios that swing under scan noise and flip the identity. Measurement
/// killed that theory: lam31 barely moves under noise even on the flattest real
/// scans. The actual noise fragility lived in the surface-area dimension, since
/// dropped from the feature vector. With it gone, thin solids re-verify fine, so
/// the floor came down to 0.005, where it only turns away a vertex cloud with no
/// third axis at all.
///
/// These are deliberate thresholds. They sit at the registration gate, outside
/// feature computation, so the "no hard thresholds inside the feature math"
/// contract still holds. Why neither end is simply dropped:
/// - **Lower bound (near-degenerate).** A near-planar cloud has no stable third
///   axis, so its eigen-ratios are noise. A cheap up-front refusal beats a
///   confusing later "sometimes matches". Recall and legibility, not safety.
/// - **Upper bound (near-regular).** Not droppable. It encodes a policy
///   (a plain sphere or cube must not anchor an asset) and fences off a
///   collision-dense region where two distinct near-isotropic shapes can share
///   an id. Both are false-accepts that verification cannot catch, so dropping
///   this end would silently admit forgery-prone or out-of-scope registrations.
pub const LAM31_MIN: f64 = 0.005;
pub const LAM31_MAX: f64 = 0.95;

/// Registration shape gate: reject weak or regular shapes. Returns a rejection
/// reason if the shape cannot carry a stable identity, else `None`.
///
/// Evaluated **at registration only**. Verification deliberately does not gate.
/// It re-derives whatever hash a fresh scan yields and lets hash equality
/// decide, so a weak re-scan just fails to match, with no separate error path.
pub fn weak_shape(f: &[f64; N_FEATURES]) -> Option<&'static str> {
    let lam31 = f[1];
    if lam31 < LAM31_MIN {
        Some("nearly a flat sheet with no real thickness, too degenerate for a reliable fingerprint. Use a fuller, more solid model.")
    } else if lam31 > LAM31_MAX {
        Some("too round and plain, like a ball or box, for a unique fingerprint. Pick a model with a more distinctive shape.")
    } else {
        None
    }
}

/// Assemble the 23-D feature vector from the covariance eigenvalues and a set
/// of surface samples. The eigenvalues give the shape ratios, the samples give
/// the radius, area, spherical-harmonic, and histogram terms. See the module
/// header for what each dimension holds.
pub fn features(eigvals: [f64; 3], s: &Samples) -> [f64; N_FEATURES] {
    let n = s.points.len();
    let mut w_total = 0.0;
    let mut r = vec![0.0; n];
    for (ri, (p, w)) in r.iter_mut().zip(s.points.iter().zip(s.weights.iter())) {
        *ri = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        w_total += *w;
    }
    let mut rbar = 0.0;
    for (ri, w) in r.iter().zip(s.weights.iter()) {
        rbar += *w * *ri;
    }
    rbar /= w_total;

    // radial histogram with triangular kernel (no hard bin edges)
    let mut hist = [0.0; HIST_BINS];
    for (ri, w) in r.iter().zip(s.weights.iter()) {
        let t = (*ri / rbar).clamp(0.0, 2.0);
        let x = t / HIST_STEP - 0.5;
        let j0 = floor(x);
        let frac = x - j0;
        let j0i = j0 as i64;
        let lo = j0i.clamp(0, HIST_BINS as i64 - 1) as usize;
        let hi = (j0i + 1).clamp(0, HIST_BINS as i64 - 1) as usize;
        hist[lo] += *w * (1.0 - frac);
        hist[hi] += *w * frac;
    }
    for h in hist.iter_mut() {
        *h /= w_total;
    }

    // spherical-harmonic moments: a (area-weighted), b (radius-weighted).
    // A sample at the centroid (r ~ 0) carries no direction, so it drops out
    // of the angular sums. w_sh counts only the weight that does carry a
    // direction, so the power below divides by the same mass it was built from.
    // Using w_total would dilute it by any centroid-grazing samples.
    let mut a = [[0.0f64; 13]; 7];
    let mut bq = [[0.0f64; 13]; 7];
    let mut w_sh = 0.0;
    for (ri, (p, w)) in r.iter().zip(s.points.iter().zip(s.weights.iter())) {
        if *ri < 1e-12 {
            continue;
        }
        w_sh += *w;
        let u = [p[0] / *ri, p[1] / *ri, p[2] / *ri];
        let y = real_sh6(u);
        let rr = *ri / rbar;
        let mut idx = 0;
        for l in 0..=6usize {
            for m in 0..(2 * l + 1) {
                a[l][m] += *w * y[idx];
                bq[l][m] += *w * rr * y[idx];
                idx += 1;
            }
        }
    }
    let mut p = [0.0; 6];
    let mut q = [0.0; 6];
    // w_sh == 0 only if every sample sat at the centroid (no surface), which a
    // valid mesh never produces. Leave the power at zero in that case.
    if w_sh > 0.0 {
        for l in 1..=6usize {
            let mut sa = 0.0;
            let mut sb = 0.0;
            for m in 0..(2 * l + 1) {
                sa += a[l][m] * a[l][m];
                sb += bq[l][m] * bq[l][m];
            }
            p[l - 1] = sqrt(sa) / w_sh;
            q[l - 1] = sqrt(sb) / w_sh;
        }
    }

    let mut f = [0.0; N_FEATURES];
    f[0] = eigvals[1] / eigvals[0];
    f[1] = eigvals[2] / eigvals[0];
    f[2] = rbar;
    f[3..9].copy_from_slice(&p);
    f[9..15].copy_from_slice(&q);
    f[15..23].copy_from_slice(&hist[..8]);
    f
}

/// All real orthonormal spherical harmonics up to l=6 at unit vector u,
/// flattened as [l=0 | l=1 m=-1,0,1 | l=2 m=-2..2 | ...], 49 values.
///
/// Orders 0..4 are the classic explicit Cartesian forms. Orders 5,6 are built
/// from the azimuthal polynomials A_m = Re[(x+iy)^m], B_m = Im[(x+iy)^m] and
/// the Legendre derivatives D_l^m(z) = dᵐ P_l/dzᵐ, scaled by the real-SH norm
/// K_l^m = √((2l+1)/(4π) · (l−m)!/(l+m)!). Every term is sqrt + polynomial, no
/// transcendentals, so the identity stays bit-reproducible across IEEE targets.
pub fn real_sh6(u: [f64; 3]) -> [f64; 49] {
    let (x, y, z) = (u[0], u[1], u[2]);
    let pi = core::f64::consts::PI;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let mut o = [0.0; 49];
    o[0] = 0.5 * sqrt(1.0 / pi);

    // l = 1
    let c1 = sqrt(3.0 / (4.0 * pi));
    o[1] = c1 * y;
    o[2] = c1 * z;
    o[3] = c1 * x;

    // l = 2
    let c2a = 0.5 * sqrt(15.0 / pi);
    o[4] = c2a * x * y;
    o[5] = c2a * y * z;
    o[6] = 0.25 * sqrt(5.0 / pi) * (3.0 * z2 - 1.0);
    o[7] = c2a * x * z;
    o[8] = 0.25 * sqrt(15.0 / pi) * (x2 - y2);

    // l = 3
    o[ 9] = 0.25 * sqrt( 35.0 / (2.0 * pi)) * y * (3.0 * x2 - y2);
    o[10] = 0.5  * sqrt(105.0 /        pi ) * x * y * z;
    o[11] = 0.25 * sqrt( 21.0 / (2.0 * pi)) * y * (5.0 * z2 - 1.0);
    o[12] = 0.25 * sqrt(  7.0 /        pi ) * z * (5.0 * z2 - 3.0);
    o[13] = 0.25 * sqrt( 21.0 / (2.0 * pi)) * x * (5.0 * z2 - 1.0);
    o[14] = 0.25 * sqrt(105.0 /        pi ) * z * (x2 - y2);
    o[15] = 0.25 * sqrt( 35.0 / (2.0 * pi)) * x * (x2 - 3.0 * y2);

    // l = 4
    o[16] =  0.75        * sqrt(35.0 /        pi ) * x * y * (x2 - y2);
    o[17] =  0.75        * sqrt(35.0 / (2.0 * pi)) * y * z * (3.0 * x2 - y2);
    o[18] =  0.75        * sqrt( 5.0 /        pi ) * x * y * (7.0 * z2 - 1.0);
    o[19] =  0.75        * sqrt( 5.0 / (2.0 * pi)) * y * z * (7.0 * z2 - 3.0);
    o[20] = (3.0 / 16.0) * sqrt( 1.0 /        pi ) * (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
    o[21] =  0.75        * sqrt( 5.0 / (2.0 * pi)) * x * z * (7.0 * z2 - 3.0);
    o[22] = (3.0 /  8.0) * sqrt( 5.0 /        pi ) * (x2 - y2) * (7.0 * z2 - 1.0);
    o[23] =  0.75        * sqrt(35.0 / (2.0 * pi)) * x * z * (x2 - 3.0 * y2);
    o[24] = (3.0 / 16.0) * sqrt(35.0 /        pi ) * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2);

    // l = 5, 6: azimuthal Cartesian polynomials A_m = Re[(x+iy)^m] (= am[m])
    // and B_m = Im[(x+iy)^m] (= bm[m]) via the complex-power recurrence.
    let mut am = [0.0f64; 7];
    let mut bm = [0.0f64; 7];
    am[1] = x;
    bm[1] = y;
    for m in 2..=6 {
        am[m] = x * am[m - 1] - y * bm[m - 1];
        bm[m] = x * bm[m - 1] + y * am[m - 1];
    }
    let s2 = core::f64::consts::SQRT_2;

    // l = 5: K_5^m = √(11/(4π) · (5−m)!/(5+m)!), Legendre derivatives D_5^m(z).
    let k5 = |ratio: f64| sqrt(11.0 / (4.0 * pi) * ratio);
    let d5_0 = (63.0 * z2 * z2 * z - 70.0 * z2 * z + 15.0 * z) / 8.0;
    let d5_1 = (315.0 * z2 * z2 - 210.0 * z2 + 15.0) / 8.0;
    let d5_2 = (1260.0 * z2 * z - 420.0 * z) / 8.0;
    let d5_3 = (3780.0 * z2 - 420.0) / 8.0;
    let d5_4 = (7560.0 * z) / 8.0;
    let d5_5 = 945.0;
    o[30] = k5(1.0) * d5_0;
    let n51 = s2 * k5(1.0 / 30.0);
    o[31] = n51 * am[1] * d5_1;
    o[29] = n51 * bm[1] * d5_1;
    let n52 = s2 * k5(1.0 / 840.0);
    o[32] = n52 * am[2] * d5_2;
    o[28] = n52 * bm[2] * d5_2;
    let n53 = s2 * k5(1.0 / 20160.0);
    o[33] = n53 * am[3] * d5_3;
    o[27] = n53 * bm[3] * d5_3;
    let n54 = s2 * k5(1.0 / 362880.0);
    o[34] = n54 * am[4] * d5_4;
    o[26] = n54 * bm[4] * d5_4;
    let n55 = s2 * k5(1.0 / 3628800.0);
    o[35] = n55 * am[5] * d5_5;
    o[25] = n55 * bm[5] * d5_5;

    // l = 6: K_6^m = √(13/(4π) · (6−m)!/(6+m)!), Legendre derivatives D_6^m(z).
    let k6 = |ratio: f64| sqrt(13.0 / (4.0 * pi) * ratio);
    let d6_0 = ( 231.0 * z2 * z2 * z2 - 315.0 * z2 * z2 + 105.0 * z2 - 5.0) / 16.0;
    let d6_1 = (1386.0 * z2 * z2 * z - 1260.0 * z2 * z  + 210.0 * z) / 16.0;
    let d6_2 = (6930.0 * z2 * z2 - 3780.0 * z2 + 210.0) / 16.0;
    let d6_3 = (27720.0 * z2 * z - 7560.0 * z) / 16.0;
    let d6_4 = (83160.0 * z2 - 7560.0) / 16.0;
    let d6_5 = 10395.0 * z;
    let d6_6 = 10395.0;
    o[42] = k6(1.0) * d6_0;
    let n61 = s2 * k6(1.0 / 42.0);
    o[43] = n61 * am[1] * d6_1;
    o[41] = n61 * bm[1] * d6_1;
    let n62 = s2 * k6(1.0 / 1680.0);
    o[44] = n62 * am[2] * d6_2;
    o[40] = n62 * bm[2] * d6_2;
    let n63 = s2 * k6(1.0 / 60480.0);
    o[45] = n63 * am[3] * d6_3;
    o[39] = n63 * bm[3] * d6_3;
    let n64 = s2 * k6(1.0 / 1814400.0);
    o[46] = n64 * am[4] * d6_4;
    o[38] = n64 * bm[4] * d6_4;
    let n65 = s2 * k6(1.0 / 39916800.0);
    o[47] = n65 * am[5] * d6_5;
    o[37] = n65 * bm[5] * d6_5;
    let n66 = s2 * k6(1.0 / 479001600.0);
    o[48] = n66 * am[6] * d6_6;
    o[36] = n66 * bm[6] * d6_6;
    
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::normalize;
    use crate::sample::sample_surface;
    use crate::testutil::{bumpy, reorder, rotate, scale, translate, uv_sphere};

    fn pipeline(mesh: crate::mesh::Mesh) -> [f64; N_FEATURES] {
        let n = normalize(mesh).unwrap();
        let s = sample_surface(&n.mesh, 4096);
        features(n.eigvals, &s)
    }

    /// Unsöld's theorem: Σ_m Y_lm(u)² = (2l+1)/(4π) for any u.
    /// Catches any wrong normalization constant.
    #[test]
    fn unsold_identity() {
        let pi = core::f64::consts::PI;
        let dirs: [[f64; 3]; 5] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-2.0, 1.0, 5.0],
        ];
        for d in dirs {
            let n = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            let d = [d[0] / n, d[1] / n, d[2] / n];
            let y = real_sh6(d);
            let mut idx = 0;
            for l in 0..=6usize {
                let mut s = 0.0;
                for _ in 0..(2 * l + 1) {
                    s += y[idx] * y[idx];
                    idx += 1;
                }
                let expect = (2 * l + 1) as f64 / (4.0 * pi);
                assert!(
                    (s - expect).abs() < 1e-9,
                    "l={l} dir={d:?}: {s} vs {expect}"
                );
            }
        }
    }

    #[test]
    fn sphere_features_match_analytic() {
        let f = pipeline(uv_sphere(48, 96, &|_, _| 1.0));
        // isotropic up to UV-discretization error: a rings=48 polyhedron is
        // not a perfect sphere, its eigenvalue ratios deviate at ~1e-3
        assert!((f[0] - 1.0).abs() < 2e-3, "lam21 {}", f[0]);
        assert!((f[1] - 1.0).abs() < 2e-3, "lam31 {}", f[1]);
        // unit-volume sphere radius = (3/4π)^(1/3) ≈ 0.62035
        assert!((f[2] - 0.62035).abs() < 0.01, "rbar {}", f[2]);
        // symmetric: SH power near zero for l ≥ 1
        for l in 0..6 {
            assert!(f[3 + l] < 0.02, "p{} = {}", l + 1, f[3 + l]);
            assert!(f[9 + l] < 0.02, "q{} = {}", l + 1, f[9 + l]);
        }
        // radial mass concentrated around t = 1 (bins 3 and 4)
        assert!(f[15 + 3] + f[15 + 4] > 0.95, "hist {:?}", &f[15..23]);
    }

    /// The design's core claims, verified at unit level: rigid motion,
    /// uniform scale and index reordering shift every feature by no more
    /// than float roundoff.
    #[test]
    fn invariance_rigid_scale_reorder() {
        let base = bumpy(48, 96);
        let f0 = pipeline(base.clone());

        let rotated = translate(
            &rotate(&base, [0.267261, 0.534522, 0.801784], 1.234),
            [0.3, -0.2, 0.5],
        );
        let f1 = pipeline(rotated);

        let scaled = scale(&base, 7.3);
        let f2 = pipeline(scaled);

        let reordered = reorder(&base, 13, 7);
        let f3 = pipeline(reordered);

        for d in 0..N_FEATURES {
            assert!(
                (f1[d] - f0[d]).abs() < 1e-9,
                "rigid {} d{}: {} vs {}",
                FEATURE_NAMES[d],
                d,
                f1[d],
                f0[d]
            );
            assert!(
                (f2[d] - f0[d]).abs() < 1e-9,
                "scale {} d{}: {} vs {}",
                FEATURE_NAMES[d],
                d,
                f2[d],
                f0[d]
            );
            assert!(
                (f3[d] - f0[d]).abs() < 1e-9,
                "reorder {} d{}: {} vs {}",
                FEATURE_NAMES[d],
                d,
                f3[d],
                f0[d]
            );
        }
    }

    /// Stretch must move features (discrimination direction).
    #[test]
    fn stretch_changes_eig_ratio() {
        let base = uv_sphere(48, 96, &|_, _| 1.0);
        let mut stretched = base.clone();
        for p in &mut stretched.vertices {
            p[2] *= 1.2;
        }
        let f0 = pipeline(base);
        let f1 = pipeline(stretched);
        assert!(
            (f1[1] - f0[1]).abs() > 0.1,
            "lam31 must move: {} vs {}",
            f1[1],
            f0[1]
        );
    }
}
