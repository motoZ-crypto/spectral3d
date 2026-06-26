//! Per-dimension quantization with a code-offset secure sketch, then a plain
//! cryptographic hash (SHA-256). Together they make a noise-tolerant identity.
//!
//! The pipeline's only discretization lives here. Each feature dimension is
//! divided by a protocol-fixed bucket width and rounded to an integer bucket.
//! The concatenated bucket vector is the object identity.
//!
//! **Secure sketch (noise tolerance).** A bare quantizer is fragile at bucket
//! boundaries: two scans of one object straddling an edge round to different
//! buckets. So registration publishes per-dimension *helper data*, the
//! sub-bucket offset that re-centres the reading in its bucket. Applying that
//! same offset at verification guarantees any later reading within ±½ bucket of
//! the registered one recovers the exact same bucket, i.e. one deterministic
//! identity hash rather than a candidate list. The offset reveals only the
//! within-bucket phase (the scan noise), not the bucket index (the identity).
//! That is the secure-sketch leakage guarantee.
//!
//! **On privacy: bounded leakage, NOT zero-knowledge.** The quantized bucket
//! vector is a low-entropy "password". Across real objects only ~6 to 15 bits
//! of it actually vary (tolerance to pose and noise is bought by discarding
//! detail), and the bucket widths are public protocol constants. A published
//! HASH ID is therefore *not* zero-knowledge: an attacker enumerates the
//! reachable bucket vectors, hashes each, and recovers the coarse shape
//! statistics behind any ID.
//!
//! A slow or memory-hard hash does **not** fix that. The ID must be
//! deterministic, secret-free, and used for equality comparison, so the salt
//! can only be a constant or a function of the identity itself. The attacker
//! then precomputes ONE universal {bucket vector → hash} table and reverses
//! every ID by O(1) lookup, amortizing the per-guess cost away after a single
//! table build (~tens of seconds at this entropy). Memory-hardening buys
//! something only once the upstream entropy is high enough (~2^30+) that
//! building the table is itself a barrier, which spectral3d alone is far from.
//! So the ID is a plain SHA-256, and the ceiling is stated honestly: privacy
//! is "bounded leakage, brute-forceable", set by the feature entropy upstream,
//! and only raising that entropy (or an architectural measure) can move it.

use sha2::{Digest, Sha256};

use crate::features::N_FEATURES;

/// Protocol identifier, part of every hash preimage, so any parameter or
/// pipeline revision lands in a disjoint hash universe. Bump it whenever the
/// feature math or quantization changes, then re-bake the golden hash.
pub const PROTOCOL: &str = "spectral3d-v1";

/// Per-dimension bucket widths, in feature units, indexed like
/// [`crate::features::FEATURE_NAMES`].
///
/// Sized by how each dimension responds to noise versus signal. The SH orders
/// absorb surface crinkle, so they get coarse buckets and 1% surface noise stays
/// well under one bucket. The discrimination load rides on the eigenvalue ratios,
/// whose buckets stay fine (a 5% z-stretch already moves lam21/lam31 by ~1.9
/// buckets). Tolerance and discrimination thus land on *different* dimensions
/// instead of fighting over one knob.
pub const QUANT_STEP: [f64; N_FEATURES] = [
    0.03, // lam21: 1% noise shifts it ~0.1 bucket, a 5% stretch >=1.7 buckets,
    //       well past the ½-bucket recovery radius, so a 5% stretch reads as a
    //       different identity
    0.03, // lam31 (same as lam21)
    0.08, // rbar  (1% noise shifts it ~0.14 unit)
    // SH power: every order gets a coarse 0.10 bucket. Surface noise lands mostly
    // in the even orders, but on dense meshes the odd orders drift too, and the
    // old fine 0.025 there flipped the identity on dense real scans. Widening
    // them cost no discrimination on the test corpus (zero new collisions).
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10, // p1..p6 (area-weighted)
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10, // q1..q6 (radius-weighted)
    0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, // h0..h7 (1% noise <=0.16)
];

/// Sanity window for [`QuantParams::scale`]: a config tripwire, not a
/// tuning threshold. The endpoints sit a millionfold either side of the
/// default 1.0, far past any useful tuning (below `SCALE_MIN` buckets are too
/// fine to tolerate any noise, above `SCALE_MAX` they are too coarse to
/// discriminate). Their only job is to reject a misconfigured scale that would
/// collapse every bucket to one constant and make every object verify as every
/// other (a universal false-accept). This is parameter validation, not a
/// feature-computation threshold, so it leaves the Lipschitz / no-hard-threshold
/// design of the features untouched. It trips only on the arithmetically
/// degenerate extremes. A merely mistuned scale (say 100x) still erodes
/// discrimination quietly, so a "distinct objects get distinct ids" check stays
/// the deployer's job, not something a parameter guard can stand in for.
pub const SCALE_MIN: f64 = 1e-6;
pub const SCALE_MAX: f64 = 1e6;

#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Global multiplier on every bucket width (tuning knob). A larger
    /// scale widens buckets: more noise tolerance, coarser discrimination.
    pub scale: f64,
}

impl Default for QuantParams {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

/// Public helper data for the code-offset secure sketch: the per-dimension
/// sub-bucket offset, in (-0.5, 0.5] bucket units. Published alongside the
/// registered identity and required to verify a later scan. It leaks the
/// within-bucket phase (scan noise) but not the bucket index (identity).
#[derive(Debug, Clone, PartialEq)]
pub struct Helper {
    pub offsets: [f64; N_FEATURES],
}

/// Registration: feature vector → (identity HASH ID, public helper data).
///
/// Each dimension's bucket is the nearest integer of `f / step`. The offset
/// records how far the reading sat from that bucket's centre, so [`recover`]
/// can re-centre a fresh reading onto the same bucket.
pub fn sketch(f: &[f64; N_FEATURES], p: &QuantParams) -> (String, Helper) {
    let mut b = [0i64; N_FEATURES];
    let mut offsets = [0.0f64; N_FEATURES];
    for i in 0..N_FEATURES {
        let x = f[i] / (QUANT_STEP[i] * p.scale);
        let c = x.round();
        b[i] = c as i64;
        offsets[i] = x - c;
    }
    (hash_buckets(PROTOCOL, &b), Helper { offsets })
}

/// Verification: feature vector + registered helper → HASH ID.
///
/// Re-centres each dimension by the registered offset before rounding, so the
/// result equals the registered ID iff every dimension drifted strictly less
/// than ½ bucket from registration. One deterministic identity, no candidate
/// enumeration.
pub fn recover(f: &[f64; N_FEATURES], h: &Helper, p: &QuantParams) -> String {
    let mut b = [0i64; N_FEATURES];
    for i in 0..N_FEATURES {
        let x = f[i] / (QUANT_STEP[i] * p.scale);
        b[i] = (x - h.offsets[i]).round() as i64;
    }
    hash_buckets(PROTOCOL, &b)
}

fn hash_buckets(protocol: &str, b: &[i64; N_FEATURES]) -> String {
    // Canonical preimage: protocol || 0x00 || buckets (little-endian).
    // Position-sensitive and protocol-prefixed, so distinct protocols and
    // distinct bucket orderings land in disjoint hash universes.
    let mut h = Sha256::new();
    h.update(protocol.as_bytes());
    h.update([0u8]);
    for &x in b {
        h.update(x.to_le_bytes());
    }
    let out = h.finalize();

    let mut s = String::with_capacity(out.len() * 2);
    for byte in out {
        use core::fmt::Write;
        let _ = write!(s, "{byte:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features_at(mult: f64) -> [f64; N_FEATURES] {
        let mut f = [0.0; N_FEATURES];
        for i in 0..N_FEATURES {
            f[i] = QUANT_STEP[i] * mult;
        }
        f
    }

    #[test]
    fn sketch_is_deterministic() {
        let p = QuantParams::default();
        let f = features_at(3.0);
        assert_eq!(sketch(&f, &p), sketch(&f, &p));
        assert_eq!(sketch(&f, &p).0.len(), 64); // hex of 32 bytes
    }

    #[test]
    fn offsets_are_sub_bucket() {
        let p = QuantParams::default();
        let mut f = features_at(3.0);
        f[2] += 0.3 * QUANT_STEP[2];
        f[5] -= 0.4 * QUANT_STEP[5];
        let (_, h) = sketch(&f, &p);
        for o in h.offsets {
            assert!(o.abs() <= 0.5 + 1e-9, "offset {o} is not sub-bucket");
        }
    }

    /// Secure-sketch contract: a fresh reading within ±½ bucket of registration
    /// recovers the exact registered ID, even when the registration reading sat
    /// right on a bucket edge, with no candidate enumeration. The helper
    /// re-centres it.
    #[test]
    fn straddle_within_half_bucket_recovers_id() {
        let p = QuantParams::default();
        let mut fa = features_at(3.0);
        fa[2] = QUANT_STEP[2] * 4.5; // registration sits exactly on an edge
        let (id, h) = sketch(&fa, &p);
        // every dimension drifts 0.45 bucket on the re-scan
        let mut fb = fa;
        for i in 0..N_FEATURES {
            fb[i] += 0.45 * QUANT_STEP[i];
        }
        assert_eq!(recover(&fb, &h, &p), id, "within ½ bucket must recover id");
    }

    #[test]
    fn drift_past_half_bucket_changes_id() {
        let p = QuantParams::default();
        let fa = features_at(3.0);
        let (id, h) = sketch(&fa, &p);
        let mut fb = fa;
        fb[0] += 0.6 * QUANT_STEP[0]; // one dim past the recovery radius
        assert_ne!(recover(&fb, &h, &p), id);
    }

    #[test]
    fn far_features_differ() {
        let p = QuantParams::default();
        let fa = features_at(3.0);
        let (id, h) = sketch(&fa, &p);
        let fb = features_at(6.0); // three buckets away on every dim
        assert_ne!(recover(&fb, &h, &p), id);
    }

    /// Domain separation: any protocol/parameter revision must yield a
    /// disjoint hash universe (the property the PROTOCOL preimage prefix
    /// exists for).
    #[test]
    fn protocol_string_separates_hash_universes() {
        let mut b = [0i64; N_FEATURES];
        for (i, x) in b.iter_mut().enumerate() {
            *x = i as i64 - 3; // includes negatives
        }
        assert_ne!(
            hash_buckets(PROTOCOL, &b),
            hash_buckets("spectral3d-v0", &b)
        );
        // and the encoding is position-sensitive, not just value-sensitive
        let mut swapped = b;
        swapped.swap(0, 1);
        assert_ne!(hash_buckets(PROTOCOL, &b), hash_buckets(PROTOCOL, &swapped));
    }
}
