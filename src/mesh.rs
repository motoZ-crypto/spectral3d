//! Triangle-mesh loading and exact volume integrals.
//!
//! Everything is computed by summing signed tetrahedra (divergence theorem).
//! The sums are order-independent up to floating-point roundoff, and exactly
//! equivariant under rigid motion and scaling. No reference frame, axis
//! ordering, or start vertex is ever chosen.

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeshError {
    Parse(String),
    /// Geometry too degenerate to measure: a zero or near-zero volume (flat
    /// or coplanar mesh), or a covariance with a non-finite or non-positive
    /// eigenvalue. Raised by [`normalize`] before feature extraction.
    Degenerate(String),
    /// A geometrically valid shape that still can't anchor a stable identity.
    /// Two cases: ill-conditioned (near-flat, eigen-ratios blow up under noise),
    /// or too regular (near sphere or cube, nothing to anchor on). Raised by
    /// the registration shape gate. See [`crate::features::weak_shape`].
    WeakShape(String),
    /// The face set isn't a closed, consistently oriented surface: an open
    /// shell, a hole, a non-manifold edge, or mixed winding. The divergence-
    /// theorem integrals only mean anything on a watertight oriented manifold.
    /// A single missing face already makes the volume swing with the coordinate
    /// origin, so this is rejected up front instead of silently hashing a
    /// pose-dependent identity. Raised by [`normalize`].
    NotClosed(String),
    /// The shells don't all wind the same way, or one shell closes around a
    /// vanishing volume. [`Mesh::check_closed`] confirms each connected shell is
    /// individually closed and consistently wound, but it can't tell how the
    /// shells sit relative to each other. Each is independently orientable, so
    /// one can end up flipped inside-out against the rest. A flipped shell
    /// subtracts its own mass from the signed-volume integrals and hands the
    /// same physical object a winding-dependent identity. Multiple shells are
    /// fine (a car body plus four wheels is still one object). They just have to
    /// agree on a sign. Raised by [`normalize`].
    InconsistentShells(String),
    /// A pipeline parameter is out of range, currently a non-finite or
    /// non-positive quantization `scale`, which would feed garbage through the
    /// bucket division. Raised by [`crate::register`] and [`crate::verify`]
    /// before any hashing.
    InvalidParam(String),
}

impl core::fmt::Display for MeshError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MeshError::Parse             (m) => write!(f, "obj parse error: {m}"),
            MeshError::Degenerate        (m) => write!(f, "degenerate mesh: {m}"),
            MeshError::WeakShape         (m) => write!(f, "weak shape (rejected at registration): {m}"),
            MeshError::NotClosed         (m) => write!(f, "mesh is not a closed oriented manifold: {m}"),
            MeshError::InconsistentShells(m) => write!(f, "mesh shells aren't consistently oriented: {m}"),
            MeshError::InvalidParam      (m) => write!(f, "invalid parameter: {m}"),
        }
    }
}

impl std::error::Error for MeshError {}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Mesh {
    /// Minimal OBJ reader: `v x y z` and `f i j k ...` (polygons fan-
    /// triangulated from the first listed vertex, `i/t/n` and negative
    /// indices supported). Everything else is ignored.
    ///
    /// The fan apex is the face's first vertex. A planar polygon gives the
    /// same integrals however it is written, but a *non-planar* n-gon (n >= 4)
    /// is genuinely ambiguous: a cyclic rewrite of its vertex list fans into
    /// different triangles and a slightly different solid. Such a face has no
    /// well-defined surface to begin with, so feed triangles or planar
    /// polygons when a stable identity matters.
    pub fn parse_obj(bytes: &[u8]) -> Result<Mesh, MeshError> {
        let text = String::from_utf8_lossy(bytes);
        let mut vertices: Vec<[f64; 3]> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            match it.next() {
                Some("v") => {
                    let mut p = [0f64; 3];
                    for c in p.iter_mut() {
                        let tok = it.next().ok_or_else(|| {
                            MeshError::Parse(format!("line {}: vertex needs 3 coords", lineno + 1))
                        })?;
                        *c = tok.parse::<f64>().map_err(|_| {
                            MeshError::Parse(format!("line {}: bad float '{tok}'", lineno + 1))
                        })?;
                    }
                    vertices.push(p);
                }
                Some("f") => {
                    let mut idx: Vec<u32> = Vec::new();
                    for tok in it {
                        let first = tok.split('/').next().unwrap_or("");
                        let i = first.parse::<i64>().map_err(|_| {
                            MeshError::Parse(format!("line {}: bad index '{tok}'", lineno + 1))
                        })?;
                        let resolved = if i > 0 {
                            i - 1
                        } else if i < 0 {
                            vertices.len() as i64 + i
                        } else {
                            return Err(MeshError::Parse(format!(
                                "line {}: zero index",
                                lineno + 1
                            )));
                        };
                        if resolved < 0 || resolved >= vertices.len() as i64 {
                            return Err(MeshError::Parse(format!(
                                "line {}: index {} out of range",
                                lineno + 1,
                                i
                            )));
                        }
                        idx.push(resolved as u32);
                    }
                    if idx.len() < 3 {
                        return Err(MeshError::Parse(format!(
                            "line {}: face needs >= 3 vertices",
                            lineno + 1
                        )));
                    }
                    for k in 1..idx.len() - 1 {
                        faces.push([idx[0], idx[k], idx[k + 1]]);
                    }
                }
                _ => {}
            }
        }
        if vertices.is_empty() || faces.is_empty() {
            return Err(MeshError::Parse("no geometry".into()));
        }
        Ok(Mesh { vertices, faces })
    }

    pub(crate) fn tri(&self, f: [u32; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
        (
            self.vertices[f[0] as usize],
            self.vertices[f[1] as usize],
            self.vertices[f[2] as usize],
        )
    }

    /// Arithmetic mean of the vertices: a translation-equivariant,
    /// order-independent local reference point.
    ///
    /// The integrals below sum tetrahedra against this point instead of the
    /// world origin. Mathematically identical for a closed mesh, but it keeps
    /// the sums honest under large coordinates. Against the origin, a 1e9 offset
    /// alone turns a unit box's volume into ~1e9 of pure cancellation noise.
    /// Folding the mean out first builds every tetrahedron from object-sized
    /// numbers, so nothing large is ever subtracted.
    fn vertex_mean(&self) -> [f64; 3] {
        let mut o = [0.0; 3];
        for v in &self.vertices {
            for k in 0..3 {
                o[k] += v[k];
            }
        }
        let n = self.vertices.len().max(1) as f64;
        for ok in o.iter_mut() {
            *ok /= n;
        }
        o
    }

    /// Signed volume and volume centroid (tetrahedra against the vertex mean,
    /// then the centroid is shifted back to world coordinates).
    pub fn volume_centroid(&self) -> (f64, [f64; 3]) {
        let o = self.vertex_mean();
        let mut vol = 0.0;
        let mut c = [0.0; 3];
        for &f in &self.faces {
            let (a, b, cc) = self.tri(f);
            let (a, b, cc) = (sub(a, o), sub(b, o), sub(cc, o));
            let v = dot(a, cross(b, cc)) / 6.0;
            vol += v;
            for k in 0..3 {
                c[k] += v * (a[k] + b[k] + cc[k]) / 4.0;
            }
        }
        if vol.abs() > 1e-300 {
            for ck in c.iter_mut() {
                *ck /= vol;
            }
            // Centroid came out relative to the reference point, so add it back.
            for k in 0..3 {
                c[k] += o[k];
            }
        }
        (vol, c)
    }

    /// Signed second moment ∫ x xᵀ dV about the origin.
    ///
    /// For a tetrahedron (0, a, b, c) with signed volume v:
    /// ∫ x xᵀ dV = v/20 · (aaᵀ + bbᵀ + ccᵀ + ssᵀ), s = a+b+c.
    ///
    /// Integrated about the vertex mean and carried back to the origin by the
    /// parallel-axis identity, for the same large-coordinate reason as
    /// [`Mesh::volume_centroid`]. In the pipeline the mesh is already centred
    /// when this runs (mean ≈ 0), so the shift is a no-op there. It only earns
    /// its keep on a direct call against an off-origin mesh.
    pub fn second_moment(&self) -> [[f64; 3]; 3] {
        let o = self.vertex_mean();
        let mut mo = [[0.0; 3]; 3]; // ∫ (x-o)(x-o)ᵀ dV
        let mut first = [0.0; 3]; // ∫ (x-o) dV
        let mut vol = 0.0;
        for &f in &self.faces {
            let (a, b, c) = self.tri(f);
            let (a, b, c) = (sub(a, o), sub(b, o), sub(c, o));
            let v = dot(a, cross(b, c)) / 6.0;
            vol += v;
            let s = [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
            for k in 0..3 {
                first[k] += v * s[k] / 4.0;
            }
            for i in 0..3 {
                for j in 0..3 {
                    mo[i][j] += v / 20.0 * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j] + s[i] * s[j]);
                }
            }
        }
        // Parallel axis: M₀ = M_o + first·oᵀ + o·firstᵀ + V·o·oᵀ.
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = mo[i][j] + first[i] * o[j] + o[i] * first[j] + vol * o[i] * o[j];
            }
        }
        m
    }

    /// Verify the faces form a closed, consistently-oriented triangle mesh:
    /// every undirected edge shared by exactly two triangles of opposite
    /// winding. This is the precondition the divergence-theorem integrals
    /// assume. Without it the volume and moments are neither physically
    /// meaningful nor translation-invariant. Disjoint closed shells each pass
    /// here, since their edges still pair up internally. Pinning their relative
    /// winding is [`Mesh::check_consistent_shells`]'s job, run right after in
    /// [`normalize`].
    pub fn check_closed(&self) -> Result<(), MeshError> {
        use std::collections::HashMap;
        // Per undirected edge: how many half-edges touch it, and their net
        // winding (opposite directions cancel to zero).
        let mut edges: HashMap<(u32, u32), (u32, i32)> = HashMap::new();
        for &[i, j, k] in &self.faces {
            for (a, b) in [(i, j), (j, k), (k, i)] {
                if a == b {
                    return Err(MeshError::NotClosed(format!("degenerate edge ({a}, {a})")));
                }
                let (key, dir) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                let e = edges.entry(key).or_insert((0, 0));
                e.0 += 1;
                e.1 += dir;
            }
        }
        // `edges` is a HashMap, so when several edges are malformed, which one
        // gets reported first varies between runs. That only colours the
        // diagnostic text of an already-rejected mesh. The accept/reject
        // verdict, and every valid mesh's identity, come from ordered Vec
        // traversals and stay deterministic.
        for (&(a, b), &(count, winding)) in &edges {
            if count != 2 {
                return Err(MeshError::NotClosed(format!(
                    "edge ({a}, {b}) touches {count} face(s), expected 2 (open or non-manifold)"
                )));
            }
            if winding != 0 {
                return Err(MeshError::NotClosed(format!(
                    "edge ({a}, {b}) has inconsistent winding"
                )));
            }
        }
        Ok(())
    }

    /// Verify the faces bound a consistent solid: one or more closed shells that
    /// all wind the same way. [`Mesh::check_closed`] has already passed, so every
    /// edge has exactly two faces and each connected component is its own closed,
    /// consistently-wound shell. What it can't see is how the shells sit relative
    /// to each other. Each is independently orientable, so one can be wound
    /// inside-out against the rest, and a flipped shell would subtract its mass
    /// from the signed-volume integrals (see [`MeshError::InconsistentShells`]).
    ///
    /// Rather than demand a single component, which would turn away a car body
    /// with four separate wheels and most other multi-part assets, this groups
    /// faces into shells with union-find, measures each shell's signed volume,
    /// and requires every shell to share one sign. A shell wrapped around a
    /// vanishing volume has no trustworthy sign, so it's rejected outright
    /// instead of seeding a non-deterministic accept/reject across scans.
    ///
    /// Each shell is closed, so its signed volume doesn't depend on the
    /// reference point. The shared vertex mean only keeps the tetrahedra
    /// object-sized, same reason as [`Mesh::volume_centroid`].
    pub fn check_consistent_shells(&self) -> Result<(), MeshError> {
        use std::collections::HashMap;
        let n = self.faces.len();
        // Union-find over faces. Two faces merge when they share an edge.
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            x
        }
        let mut seen: HashMap<(u32, u32), usize> = HashMap::new();
        for (fi, &[i, j, k]) in self.faces.iter().enumerate() {
            for (a, b) in [(i, j), (j, k), (k, i)] {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&fj) = seen.get(&key) {
                    let (ra, rb) = (find(&mut parent, fi), find(&mut parent, fj));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                } else {
                    seen.insert(key, fi);
                }
            }
        }

        // Signed volume per shell, tetrahedra against the shared vertex mean.
        let o = self.vertex_mean();
        let mut vol: HashMap<usize, f64> = HashMap::new();
        for fi in 0..n {
            let r = find(&mut parent, fi);
            let (a, b, c) = self.tri(self.faces[fi]);
            let (a, b, c) = (sub(a, o), sub(b, o), sub(c, o));
            *vol.entry(r).or_insert(0.0) += dot(a, cross(b, c)) / 6.0;
        }

        // The largest shell fixes the reference sign and the degeneracy floor.
        // A shell far smaller than that carries no reliable sign, so it counts
        // as a degenerate zero-volume shell, not a real part.
        let mut reference = 0.0_f64;
        let mut vmax = 0.0_f64;
        for &v in vol.values() {
            if v.abs() > vmax {
                vmax = v.abs();
                reference = v;
            }
        }
        if vmax <= 1e-300 {
            // Every shell vanishes. Leave the empty-volume verdict to `normalize`.
            return Ok(());
        }
        let floor = vmax * 1e-12;
        let want_positive = reference > 0.0;
        for &v in vol.values() {
            if v.abs() < floor {
                return Err(MeshError::InconsistentShells(format!(
                    "a shell encloses ~0 volume ({v}), orientation undefined"
                )));
            }
            if (v > 0.0) != want_positive {
                return Err(MeshError::InconsistentShells(format!(
                    "shells disagree on winding (a shell has signed volume {v}), the integrals would cancel"
                )));
            }
        }
        Ok(())
    }
}

/// Mesh translated to its volume centroid and scaled to unit volume,
/// plus the sorted (descending) eigenvalues of the solid's covariance.
///
/// Only eigen*values* are kept. Eigenvectors, and with them every PCA
/// sign and ordering ambiguity, are never used.
pub struct Normalized {
    pub mesh: Mesh,
    pub eigvals: [f64; 3],
}

/// Validate a mesh and bring it into canonical form: confirm it is a closed
/// solid (one or more consistently-wound shells), recentre it on its volume
/// centroid, and rescale to unit volume. Returns the canonical mesh and the
/// covariance eigenvalues (see [`Normalized`]).
///
/// Errors with [`MeshError::NotClosed`] or [`MeshError::InconsistentShells`]
/// for a bad surface, or [`MeshError::Degenerate`] for a vanishing volume or a
/// covariance that is not finite and positive.
pub fn normalize(mut mesh: Mesh) -> Result<Normalized, MeshError> {
    mesh.check_closed()?;
    mesh.check_consistent_shells()?;
    let (vol, c) = mesh.volume_centroid();
    if !vol.is_finite() || vol.abs() < 1e-12 {
        return Err(MeshError::Degenerate(format!("volume too small: {vol}")));
    }
    for p in &mut mesh.vertices {
        for k in 0..3 {
            p[k] -= c[k];
        }
    }
    let s = libm::cbrt(vol.abs()).recip();
    for p in &mut mesh.vertices {
        for pk in p.iter_mut() {
            *pk *= s;
        }
    }
    // Covariance about the (now-origin) centroid. Sign cancels in M/V.
    let (vol2, _) = mesh.volume_centroid();
    let m = mesh.second_moment();
    let mut cov = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] = m[i][j] / vol2;
        }
    }
    let ev = eigvals_sym3(cov);
    if ev.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return Err(MeshError::Degenerate("bad covariance".into()));
    }
    Ok(Normalized { mesh, eigvals: ev })
}

/// Eigenvalues of a symmetric 3×3 matrix, descending. Closed-form
/// trigonometric solution, deterministic, no iteration.
///
/// The trig branch already emits e1≥e2≥e3 by construction (cos is monotone
/// over the three offset angles), but near a degenerate covariance two roots
/// collide and floating-point roundoff can violate the order by an epsilon.
/// That is enough to swap which ratio lands in lam21 vs lam31 and flip the
/// identity hash between two scans of the same object. A final descending sort
/// makes "descending" a literal guarantee, at zero cost for well-conditioned
/// inputs (already ordered → no-op). It does **not** rescue the genuinely
/// ill-conditioned tail (a near-flat shape whose λ3/λ1 swings wildly under
/// noise even when correctly ordered). That is the job of the registration
/// shape gate ([`crate::features::weak_shape`]).
pub fn eigvals_sym3(a: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    if p2 <= 1e-300 {
        return [q, q, q];
    }
    let p = (p2 / 6.0).sqrt();
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = (a[i][j] - if i == j { q } else { 0.0 }) / p;
        }
    }
    let detb = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (detb / 2.0).clamp(-1.0, 1.0);
    let phi = libm::acos(r) / 3.0;
    let e1 = q + 2.0 * p * libm::cos(phi);
    let e3 = q + 2.0 * p * libm::cos(phi + 2.0 * core::f64::consts::PI / 3.0);
    let e2 = 3.0 * q - e1 - e3;
    // Defensive descending sort: closes any roundoff-induced order violation
    // near degeneracy so lam21/lam31 never swap between scans (see doc above).
    let mut ev = [e1, e2, e3];
    // Total comparator so this stays panic-free on non-finite input (the public
    // API contract): any NaN/inf sorts as equal rather than blowing up. Callers
    // already screen the result for finiteness (see `normalize`).
    ev.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    ev
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box [0,a]×[0,b]×[0,c], outward orientation.
    pub(crate) fn box_mesh(a: f64, b: f64, c: f64) -> Mesh {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [a, b, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, c],
            [a, 0.0, c],
            [a, b, c],
            [0.0, b, c],
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2], // bottom
            [4, 5, 6],
            [4, 6, 7], // top
            [0, 1, 5],
            [0, 5, 4], // front
            [2, 3, 7],
            [2, 7, 6], // back
            [0, 4, 7],
            [0, 7, 3], // left
            [1, 2, 6],
            [1, 6, 5], // right
        ];
        Mesh { vertices, faces }
    }

    const CUBE_OBJ: &str = "\
# unit cube, mixed face formats
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1//1 3//1 2//1
f 1 4 3
f 5/1/1 6/1/1 7/1/1
f 5 7 8
f -8 -7 -3
f -8 -3 -4
f 3//2 4//2 8//2
f 3 8 7
f 1//3 5//3 8//3
f 1 8 4
f 2//4 3//4 7//4
f 2 7 6
";

    #[test]
    fn parse_obj_mixed_formats() {
        let m = Mesh::parse_obj(CUBE_OBJ.as_bytes()).unwrap();
        assert_eq!(m.vertices.len(), 8);
        assert_eq!(m.faces.len(), 12);
        let (v, c) = m.volume_centroid();
        assert!((v - 1.0).abs() < 1e-12, "volume {v}");
        for k in 0..3 {
            assert!((c[k] - 0.5).abs() < 1e-12, "centroid {c:?}");
        }
    }

    #[test]
    fn parse_obj_quads_fan() {
        let quad_cube = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 3 4 8 7
f 1 5 8 4
f 2 3 7 6
";
        let m = Mesh::parse_obj(quad_cube.as_bytes()).unwrap();
        assert_eq!(m.faces.len(), 12);
        let (v, _) = m.volume_centroid();
        assert!((v - 1.0).abs() < 1e-12, "volume {v}");
    }

    #[test]
    fn box_volume_and_centroid() {
        let m = box_mesh(1.0, 2.0, 3.0);
        let (v, c) = m.volume_centroid();
        assert!((v - 6.0).abs() < 1e-10, "volume {v}");
        assert!((c[0] - 0.5).abs() < 1e-12);
        assert!((c[1] - 1.0).abs() < 1e-12);
        assert!((c[2] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn box_inertia_ratios() {
        // Covariance of a solid box = diag(a²,b²,c²)/12 → ratios 4/9, 1/9.
        let n = normalize(box_mesh(1.0, 2.0, 3.0)).unwrap();
        let [e1, e2, e3] = n.eigvals;
        assert!((e2 / e1 - 4.0 / 9.0).abs() < 1e-9, "{:?}", n.eigvals);
        assert!((e3 / e1 - 1.0 / 9.0).abs() < 1e-9, "{:?}", n.eigvals);
        // unit volume after normalization
        let (v, _) = n.mesh.volume_centroid();
        assert!((v.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cube_is_isotropic() {
        let n = normalize(box_mesh(2.0, 2.0, 2.0)).unwrap();
        let [e1, e2, e3] = n.eigvals;
        assert!((e2 / e1 - 1.0).abs() < 1e-9);
        assert!((e3 / e1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn inverted_orientation_same_result() {
        let m = box_mesh(1.0, 2.0, 3.0);
        let flipped = Mesh {
            vertices: m.vertices.clone(),
            faces: m.faces.iter().map(|f| [f[0], f[2], f[1]]).collect(),
        };
        let (v, _) = flipped.volume_centroid();
        assert!((v + 6.0).abs() < 1e-10, "inverted volume {v}");
        let a = normalize(m).unwrap();
        let b = normalize(flipped).unwrap();
        for k in 0..3 {
            assert!((a.eigvals[k] - b.eigvals[k]).abs() < 1e-12);
        }
    }

    /// Large coordinates must not move the result. Summed against the world
    /// origin, a 1e9 offset would bury this box's volume of 6 under ~2.17e9 of
    /// cancellation noise and hand the same object a per-origin identity. The
    /// vertex-mean reference point keeps both the volume and the eigenvalues put.
    #[test]
    fn large_translation_keeps_volume_and_eigvals() {
        let base = box_mesh(1.0, 2.0, 3.0);
        let mut shifted = base.clone();
        for p in &mut shifted.vertices {
            for pk in p.iter_mut() {
                *pk += 1e9;
            }
        }
        let (v, _) = shifted.volume_centroid();
        assert!((v - 6.0).abs() < 1e-4, "shifted volume {v}");
        let a = normalize(base).unwrap();
        let b = normalize(shifted).unwrap();
        for k in 0..3 {
            assert!(
                (a.eigvals[k] - b.eigvals[k]).abs() < 1e-6,
                "eigvals drift under 1e9 shift: {:?} vs {:?}",
                a.eigvals,
                b.eigvals
            );
        }
    }

    #[test]
    fn closed_box_passes_check() {
        assert!(box_mesh(1.0, 2.0, 3.0).check_closed().is_ok());
    }

    /// A dropped face leaves an open shell with non-zero signed volume. A bare
    /// `volume < eps` check would wave it through, but the closure check rejects it.
    #[test]
    fn open_mesh_is_rejected() {
        let mut m = box_mesh(1.0, 2.0, 3.0);
        m.faces.remove(0);
        assert!(matches!(normalize(m), Err(MeshError::NotClosed(_))));
    }

    /// One flipped triangle: edges still pair up (count 2) but no longer oppose,
    /// so the winding check fires.
    #[test]
    fn inconsistent_winding_is_rejected() {
        let mut m = box_mesh(1.0, 2.0, 3.0);
        let f = m.faces[0];
        m.faces[0] = [f[0], f[2], f[1]];
        assert!(matches!(normalize(m), Err(MeshError::NotClosed(_))));
    }

    /// Two disjoint boxes wound the same way are a legitimate multi-part object,
    /// a car body and its wheels being the canonical case. Each shell is closed
    /// and they agree on a sign, so `normalize` lets them through.
    #[test]
    fn consistent_multipart_solid_is_accepted() {
        let one = box_mesh(1.0, 2.0, 3.0);
        let off = one.vertices.len() as u32;
        let mut vertices = one.vertices.clone();
        let mut faces = one.faces.clone();
        for v in &one.vertices {
            vertices.push([v[0] + 10.0, v[1], v[2]]); // a clear copy alongside
        }
        for f in &one.faces {
            faces.push([f[0] + off, f[1] + off, f[2] + off]);
        }
        let two = Mesh { vertices, faces };
        assert!(two.check_closed().is_ok(), "both shells are individually closed");
        assert!(two.check_consistent_shells().is_ok(), "same winding, must pass");
        assert!(normalize(two).is_ok());
    }

    /// The same two boxes, but the second shell is wound inside-out. Its signed
    /// volume flips sign, the pair no longer agrees, and the integrals would
    /// cancel one shell against the other. This is the case the shell check
    /// exists for.
    #[test]
    fn flipped_shell_is_rejected() {
        let one = box_mesh(1.0, 2.0, 3.0);
        let off = one.vertices.len() as u32;
        let mut vertices = one.vertices.clone();
        let mut faces = one.faces.clone();
        for v in &one.vertices {
            vertices.push([v[0] + 10.0, v[1], v[2]]);
        }
        for f in &one.faces {
            faces.push([f[0] + off, f[2] + off, f[1] + off]); // reversed winding
        }
        let two = Mesh { vertices, faces };
        assert!(two.check_closed().is_ok(), "a uniformly flipped shell is still closed");
        assert!(matches!(normalize(two), Err(MeshError::InconsistentShells(_))));
    }

    /// A shell wrapped around zero volume (here a doubled triangle) has no
    /// trustworthy orientation, so it can't seed a sign. It is rejected rather
    /// than left to flip the verdict between scans.
    #[test]
    fn degenerate_shell_is_rejected() {
        let mut m = box_mesh(1.0, 2.0, 3.0);
        let base = m.vertices.len() as u32;
        m.vertices.push([100.0, 0.0, 0.0]);
        m.vertices.push([101.0, 0.0, 0.0]);
        m.vertices.push([100.0, 1.0, 0.0]);
        // Two coincident triangles of opposite winding: closed, but zero volume.
        m.faces.push([base, base + 1, base + 2]);
        m.faces.push([base, base + 2, base + 1]);
        assert!(m.check_closed().is_ok(), "doubled triangle pairs its edges");
        assert!(matches!(
            normalize(m),
            Err(MeshError::InconsistentShells(_))
        ));
    }

    #[test]
    fn eigvals_known_matrix() {
        // diag(3,1,2) plus symmetry check
        let e = eigvals_sym3([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
        assert!((e[0] - 3.0).abs() < 1e-12);
        assert!((e[1] - 2.0).abs() < 1e-12);
        assert!((e[2] - 1.0).abs() < 1e-12);
    }

    /// Order guarantee holds even for near-degenerate and perturbed matrices,
    /// where the closed form could otherwise emit an epsilon-out-of-order root.
    #[test]
    fn eigvals_are_descending() {
        let cases: [[[f64; 3]; 3]; 4] = [
            // near triple-degenerate (near-sphere covariance) with tiny skew
            [[1.0, 1e-9, 0.0], [1e-9, 1.0, 1e-9], [0.0, 1e-9, 1.0 + 1e-10]],
            // near double-degenerate top pair (oblate / disc-like)
            [[2.0, 1e-7, 0.0], [1e-7, 2.0, 0.0], [0.0, 0.0, 0.2]],
            // near double-degenerate bottom pair (prolate / rod-like)
            [[5.0, 0.0, 0.0], [0.0, 0.5, 1e-7], [0.0, 1e-7, 0.5]],
            // generic skew
            [[3.0, 0.4, 0.1], [0.4, 1.2, 0.2], [0.1, 0.2, 0.7]],
        ];
        for a in cases {
            let e = eigvals_sym3(a);
            assert!(e[0] >= e[1] && e[1] >= e[2], "not descending: {e:?}");
        }
    }
}
