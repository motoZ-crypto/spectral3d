//! spectral3d — spectral shape identity.
//!
//! Design contract:
//! - **No reference frame is ever chosen.** Features are integrals over the
//!   solid or surface: volume, centroid, covariance *eigenvalues* (never
//!   eigenvectors), radial distribution, spherical-harmonic power spectra.
//!   All are analytically invariant under rotation and translation, and the
//!   sampling rule commutes with rigid motion, so rigid moves and vertex
//!   reorderings shift features only at float-roundoff level (~1e-12).
//! - **No hard thresholds inside feature computation.** Every feature is a
//!   Lipschitz functional of the surface (smooth kernels, no nearest-
//!   neighbour chains, no DFS). The lone discretization is the final
//!   quantization step. Quantizing a continuous feature means two
//!   measurements of one object can land on opposite sides of a bucket edge,
//!   so a code-offset secure sketch removes that fragility: registration
//!   publishes per-dimension offsets that re-centre the reading, and a fresh
//!   scan within ±½ bucket then recovers one deterministic identity hash that
//!   verification reproduces and compares for equality.

#![no_std]

extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

use alloc::{format, string::String};

pub mod features;
pub mod mesh;
pub mod quant;
pub mod sample;
#[cfg(test)]
pub(crate) mod testutil;

pub use features::{features, weak_shape, LAM31_MAX, LAM31_MIN, N_FEATURES};
pub use mesh::{normalize, Mesh, MeshError, Normalized};
pub use quant::{Helper, QuantParams, PROTOCOL, QUANT_STEP, SCALE_MAX, SCALE_MIN};
pub use sample::{sample_surface, Samples};

/// End-to-end parameters: sampling density + quantization knobs.
#[derive(Debug, Clone)]
pub struct SpectralParams {
    /// Target number of surface sample points.
    pub target_samples: usize,
    pub quant: QuantParams,
}

impl Default for SpectralParams {
    fn default() -> Self {
        Self {
            target_samples: 4096,
            quant: QuantParams::default(),
        }
    }
}

/// Scan an object, a triangle mesh (vertex and face arrays) → 23-D
/// rotation/translation/scale-invariant spectral features (raw, unquantized).
///
/// This is the spectral analysis at the heart of the library. The exact
/// computation [`register`] and [`verify`] run, minus the quantization that
/// turns features into a noise-tolerant identity. Surface integrals only
/// (covariance eigenvalue ratios, spherical-harmonic power, radial histogram),
/// all analytically invariant under rigid motion and uniform scale.
///
/// Call this directly when you want the features but not the identity sketch,
/// e.g. a downstream protocol that supplies its own (finer) discretization
/// instead of the noise-robust bucketing of [`register`]. `target_samples` sets
/// the surface sampling density (the registration default is 4096). `scan` does
/// no quantization, so the `QuantParams` scale guard does not apply here.
pub fn scan(mesh: Mesh, target_samples: usize) -> Result<[f64; N_FEATURES], MeshError> {
    let n = normalize(mesh)?;
    let s = sample_surface(&n.mesh, target_samples);
    Ok(features(n.eigvals, &s))
}

/// The identity-path feature pipeline: validate the quant scale, then [`scan`].
/// The scale guard lives here, not in `scan`, because it only matters once the
/// features get quantized (which `scan` does not do). One guard covers both
/// register and verify, since both funnel through here.
fn pipeline(mesh: Mesh, params: &SpectralParams) -> Result<[f64; N_FEATURES], MeshError> {
    // `scale` is a public, unchecked knob. A non-finite, zero, negative, or
    // wildly large value sails straight through the bucket division into a
    // meaningless hash. At worst every bucket collapses to one constant and
    // every object verifies as every other (a universal false-accept). Reject
    // those up front. The [SCALE_MIN, SCALE_MAX] window is a config tripwire,
    // not a tuning limit (see the consts).
    let s = params.quant.scale;
    if !(quant::SCALE_MIN..=quant::SCALE_MAX).contains(&s) {
        return Err(MeshError::InvalidParam(format!(
            "quant scale {s} outside the sane window [{}, {}]: a config tripwire \
             against collapsing all identities, not a tuning limit",
            quant::SCALE_MIN, quant::SCALE_MAX
        )));
    }
    scan(mesh, params.target_samples)
}

/// Register an object: a triangle mesh → (identity HASH ID, public helper data).
///
/// Store both. The helper data is what verifies later scans. It leaks only the
/// within-bucket phase (scan noise), never the identity. This is the
/// registration half of the fuzzy extractor.
///
/// Registration applies the shape gate ([`weak_shape`]): a geometrically valid
/// but ill-conditioned (near-flat) or near-regular (sphere, cube) shape is
/// refused with [`MeshError::WeakShape`], since it cannot anchor an identity
/// that survives scan noise. Verification stays ungated by design.
pub fn register(mesh: Mesh, params: &SpectralParams) -> Result<(String, Helper), MeshError> {
    let f = pipeline(mesh, params)?;
    if let Some(why) = features::weak_shape(&f) {
        return Err(MeshError::WeakShape(why.into()));
    }
    Ok(quant::sketch(&f, &params.quant))
}

/// Verify a fresh scan against a registered identity's helper data, returning
/// the HASH ID to compare with the stored one. They are equal iff the scanned
/// object is the same up to pose/scale/noise within the sketch's ±½-bucket
/// recovery radius.
pub fn verify(mesh: Mesh, helper: &Helper, params: &SpectralParams) -> Result<String, MeshError> {
    let f = pipeline(mesh, params)?;
    Ok(quant::recover(&f, helper, &params.quant))
}

#[cfg(test)]
mod e2e {
    use super::*;
    use alloc::vec::Vec;
    use crate::mesh::Mesh;
    use crate::testutil::{
        bumpy, bumpy_b, radial_noise, reorder, rms_radius, rotate, scale, superellipsoid,
        tamper_bump, translate, Rng,
    };

    /// Register a mesh → (identity hash, public helper data).
    fn registered(mesh: &Mesh, p: &SpectralParams) -> (String, Helper) {
        register(mesh.clone(), p).unwrap()
    }

    /// Does `mesh` verify as the registered identity, using its helper data?
    fn keeps(mesh: &Mesh, reg: &(String, Helper), p: &SpectralParams) -> bool {
        verify(mesh.clone(), &reg.1, p).unwrap() == reg.0
    }

    #[test]
    fn rigid_scale_reorder_keep_registered_identity() {
        let base = bumpy(48, 96);
        let p = SpectralParams::default();
        let reg = registered(&base, &p);

        let moved = translate(&rotate(&base, [1.0, -2.0, 0.5], 2.345), [4.0, 1.0, -3.0]);
        let scaled = scale(&base, 0.013);
        let reordered = reorder(&base, 101, 37);

        for (name, m) in [("rigid", moved), ("scale", scaled), ("reorder", reordered)] {
            assert!(keeps(&m, &reg, &p), "{name}: registered hash lost");
        }
    }

    /// A pose change must not break the identity, across many random poses,
    /// not just one.
    #[test]
    fn random_rigid_poses_keep_registered_identity() {
        let base = bumpy(48, 96);
        let p = SpectralParams::default();
        let reg = registered(&base, &p);
        let mut rng = Rng::new(0x5EED_0001);
        for s in 0..8 {
            let axis = rng.unit_vec();
            let angle = rng.range(0.0, core::f64::consts::TAU);
            let t = [
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
            ];
            let m = translate(&rotate(&base, axis, angle), t);
            assert!(keeps(&m, &reg, &p), "pose {s}: registered hash lost");
        }
    }

    /// A different scanner must not break the identity: the same smooth
    /// surface retessellated at several resolutions (the remesh op).
    #[test]
    fn remesh_keeps_registered_identity() {
        let p = SpectralParams::default();
        let reg = registered(&bumpy(48, 96), &p);
        for (rings, segs) in [(32, 64), (40, 80), (64, 128), (96, 192)] {
            assert!(
                keeps(&bumpy(rings, segs), &reg, &p),
                "remesh {rings}x{segs}: registered hash lost"
            );
        }
    }

    /// Scan noise must not break the identity: 0.5% radial noise stays well
    /// inside the recovery radius.
    #[test]
    fn noise_keeps_registered_identity() {
        let base = bumpy(48, 96);
        let rms = rms_radius(&base);
        let p = SpectralParams::default();
        let reg = registered(&base, &p);
        for s in 0..4u64 {
            let m = radial_noise(&base, 0.005 * rms, &mut Rng::new(0xA015E + s));
            assert!(keeps(&m, &reg, &p), "noise 0.5% seed {s}: registered hash lost");
        }
    }

    /// At 1% noise the identity degrades gracefully. The bumpy shape is
    /// well-conditioned, so a clear majority of seeds must still hold.
    #[test]
    fn noise_1pct_degrades_gracefully() {
        let base = bumpy(48, 96);
        let rms = rms_radius(&base);
        let p = SpectralParams::default();
        let reg = registered(&base, &p);
        let kept = (0..8u64)
            .filter(|&s| {
                let m = radial_noise(&base, 0.01 * rms, &mut Rng::new(0xB0153 + s));
                keeps(&m, &reg, &p)
            })
            .count();
        assert!(kept >= 6, "noise 1%: only {kept}/8 seeds kept identity");
    }

    /// Slight tampering must not break the identity, so a tweaked copy cannot
    /// register as a fresh asset. Here the probe is a 1% local bump.
    #[test]
    fn tamper_1pct_keeps_registered_identity() {
        let base = bumpy(48, 96);
        let rms = rms_radius(&base);
        let p = SpectralParams::default();
        let reg = registered(&base, &p);
        let mut rng = Rng::new(0x007A_3BEE);
        for s in 0..4 {
            let m = tamper_bump(&base, 0.01 * rms, 0.3, rng.unit_vec());
            assert!(keeps(&m, &reg, &p), "tamper 1% probe {s}: registered hash lost");
        }
    }

    /// Discrimination: a 5% z-stretch is already a different object, so neither
    /// 1.05 nor 1.2 may recover the base identity.
    #[test]
    fn stretch_changes_identity() {
        let base = bumpy(48, 96);
        let p = SpectralParams::default();
        let (ida, ha) = register(base.clone(), &p).unwrap();
        for k in [1.05f64, 1.2] {
            let mut stretched = base.clone();
            for v in &mut stretched.vertices {
                v[2] *= k;
            }
            // even with the base object's own helper data, a 5% stretch must
            // not recover the base identity
            let got = verify(stretched, &ha, &p).unwrap();
            assert_ne!(got, ida, "stretch {k}: still matches base identity");
        }
    }

    /// One object, one hash: distinct shapes must get distinct identities.
    /// Only well-conditioned shapes appear here. Regular ones (cube, sphere)
    /// live in [`weak_shapes_are_rejected`], since the shape gate refuses them.
    #[test]
    fn distinct_shapes_share_no_hashes() {
        let p = SpectralParams::default();
        let shapes: Vec<(&str, Mesh)> = vec![
            ("bumpy", bumpy(48, 96)),                            // lam31 ≈ 0.81
            ("rounded_box", superellipsoid(48, 96, 1.0, 0.8, 0.6, 8.0)), // ≈ 0.36
            ("ellipsoid", superellipsoid(48, 96, 1.0, 0.85, 0.7, 2.0)),  // ≈ 0.49
        ];
        let ids: Vec<(&str, String)> = shapes
            .iter()
            .map(|(n, m)| (*n, register(m.clone(), &p).unwrap().0))
            .collect();
        for i in 0..ids.len() {
            for j in i + 1..ids.len() {
                assert_ne!(
                    ids[i].1, ids[j].1,
                    "{} and {} share an identity hash",
                    ids[i].0, ids[j].0
                );
            }
        }
    }

    /// Shape gate: reject weak or regular shapes. Near-regular ones (sphere,
    /// cube, lam31 → 1) and near-flat ones (oblate disc, lam31 → 0) must be
    /// refused at registration with [`MeshError::WeakShape`], so they can never
    /// bind to an asset.
    #[test]
    fn weak_shapes_are_rejected() {
        let p = SpectralParams::default();
        let weak: Vec<(&str, Mesh)> = vec![
            ("sphere", superellipsoid(48, 96, 1.0, 1.0, 1.0, 2.0)), // lam31 ≈ 1.00
            ("cube", superellipsoid(48, 96, 1.0, 1.0, 1.0, 8.0)),   // lam31 ≈ 1.00
            ("near_sphere", bumpy_b(48, 96)), // lam31 ≈ 0.953, a hair over the 0.95 ceiling
            ("flat_disc", superellipsoid(48, 96, 1.0, 1.0, 0.03, 2.0)), // lam31 ≈ 0.001, below the 0.005 floor
        ];
        for (name, m) in weak {
            match register(m, &p) {
                Err(MeshError::WeakShape(_)) => {}
                other => panic!("{name}: expected WeakShape rejection, got {other:?}"),
            }
        }
    }

    /// The reason the upper ceiling exists, proven on its own with no gate in
    /// the loop. The near-regular corner is collision-dense. Two visibly
    /// different solids, a soft (p=6) and a harder (p=8) rounded cube, quantize
    /// to one identity hash. register() never appears here. This pins the raw
    /// feature and quantization behavior, so it holds however the gate is
    /// wired, or even with the gate gone.
    #[test]
    fn near_regular_corner_collides_without_a_gate() {
        let p = SpectralParams::default();
        // Around the gate: pipeline yields the raw features, sketch hashes
        // them. That is exactly register() minus the WeakShape refusal.
        let probe = |m: &Mesh| {
            let f = pipeline(m.clone(), &p).unwrap();
            (f[1], quant::sketch(&f, &p.quant).0)
        };
        let a = superellipsoid(48, 96, 1.0, 1.0, 1.0, 6.0);
        let b = superellipsoid(48, 96, 1.0, 1.0, 1.0, 8.0);
        let (la, ha) = probe(&a);
        let (lb, hb) = probe(&b);
        assert!(
            la > LAM31_MAX && lb > LAM31_MAX,
            "fixtures must sit in the near-regular corner the ceiling guards (lam31 {la:.4}, {lb:.4})"
        );
        assert_ne!(a.vertices, b.vertices, "fixtures must be genuinely different solids");
        assert_eq!(
            ha, hb,
            "two distinct near-regular solids collapse to one identity, the false-accept the ceiling fences off"
        );
    }

    /// The gate must admit genuine, well-conditioned shapes.
    #[test]
    fn well_conditioned_shapes_are_accepted() {
        let p = SpectralParams::default();
        let ok: Vec<(&str, Mesh)> = vec![
            ("bumpy", bumpy(48, 96)),
            ("rounded_box", superellipsoid(48, 96, 1.0, 0.8, 0.6, 8.0)),
            ("ellipsoid", superellipsoid(48, 96, 1.0, 0.85, 0.7, 2.0)),
        ];
        for (name, m) in ok {
            assert!(register(m, &p).is_ok(), "{name}: should pass the gate");
        }
    }

    /// Determinism canary: the registered hash of a fixed mesh under the
    /// fixed protocol. A failure means one of three things: an unintended
    /// pipeline change (fix it), an intended one (bump PROTOCOL and
    /// re-bake), or a `libm`-crate version bump that shifted a
    /// transcendental. The feature math routes cbrt/acos/cos through the
    /// `libm` crate, so identity is bit-reproducible on every IEEE target
    /// once that crate version is pinned.
    #[test]
    fn golden_registration_hash() {
        let p = SpectralParams::default();
        let (reg, _) = registered(&bumpy(48, 96), &p);
        assert_eq!(
            reg,
            "8e28a19cad7cc17852b354732d17e0cac7fb593aca93bb26ea875c8a2011dc43"
        );
    }

    #[test]
    fn degenerate_mesh_is_rejected() {
        // flat (zero-volume) triangle pair
        let flat = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2], [0, 2, 1]],
        };
        let p = SpectralParams::default();
        assert!(register(flat, &p).is_err());
    }

    /// A bad quant scale must be refused on both the register and verify paths,
    /// rather than dividing by it and hashing the garbage.
    #[test]
    fn invalid_quant_scale_is_rejected() {
        let mesh = bumpy(48, 96);
        let helper = Helper {
            offsets: [0.0; N_FEATURES],
        };
        // includes the collapse extremes. 1e-300 saturates the bucket cast and
        // 1e300 rounds every bucket to 0, both making every object share one id.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300, 1e300] {
            let p = SpectralParams {
                target_samples: 4096,
                quant: QuantParams { scale: bad },
            };
            assert!(
                matches!(register(mesh.clone(), &p), Err(MeshError::InvalidParam(_))),
                "scale {bad} should be rejected at register",
            );
            assert!(
                matches!(verify(mesh.clone(), &helper, &p), Err(MeshError::InvalidParam(_))),
                "scale {bad} should be rejected at verify",
            );
        }
    }

    /// `scan` is exactly the registration pipeline minus the identity sketch:
    /// quantizing a scan with the same params reproduces register()'s hash,
    /// for a shape the gate admits. Pins the contract that the public `scan`
    /// and the `register` path share one spectral analysis.
    #[test]
    fn scan_then_sketch_matches_register() {
        let p = SpectralParams::default();
        let mesh = bumpy(48, 96);
        let f = scan(mesh.clone(), p.target_samples).unwrap();
        assert_eq!(
            quant::sketch(&f, &p.quant).0,
            register(mesh, &p).unwrap().0,
            "scan + sketch must reproduce register's identity"
        );
    }
}
