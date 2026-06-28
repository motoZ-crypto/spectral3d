//! Shared test helpers: deterministic shape generators and mesh transforms.
#![allow(dead_code)]

use crate::mesh::Mesh;
use alloc::vec::Vec;
use core::f64::consts::PI;

/// UV-sphere with radial function r(θ, φ). Closed manifold with two poles.
pub fn uv_sphere(rings: usize, segs: usize, radial: &dyn Fn(f64, f64) -> f64) -> Mesh {
    let mut vertices = Vec::new();
    vertices.push([0.0, 0.0, radial(0.0, 0.0)]); // north pole
    for i in 1..rings {
        let theta = PI * i as f64 / rings as f64;
        for j in 0..segs {
            let phi = 2.0 * PI * j as f64 / segs as f64;
            let r = radial(theta, phi);
            vertices.push([
                r * theta.sin() * phi.cos(),
                r * theta.sin() * phi.sin(),
                r * theta.cos(),
            ]);
        }
    }
    vertices.push([0.0, 0.0, -radial(PI, 0.0)]); // south pole
    let south = (vertices.len() - 1) as u32;

    let ring = |i: usize, j: usize| -> u32 { (1 + (i - 1) * segs + (j % segs)) as u32 };
    let mut faces = Vec::new();
    for j in 0..segs {
        faces.push([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..rings - 1 {
        for j in 0..segs {
            let a = ring(i, j);
            let b = ring(i + 1, j);
            let c = ring(i + 1, j + 1);
            let d = ring(i, j + 1);
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
    for j in 0..segs {
        faces.push([south, ring(rings - 1, j + 1), ring(rings - 1, j)]);
    }
    Mesh { vertices, faces }
}

/// Deterministic asymmetric bumpy shape (smooth, star-shaped).
pub fn bumpy(rings: usize, segs: usize) -> Mesh {
    uv_sphere(rings, segs, &|theta: f64, phi: f64| {
        1.0 + 0.15 * (3.0 * theta).sin() * (2.0 * phi).cos()
            + 0.08 * (5.0 * phi).cos() * theta.sin() * theta.sin()
            + 0.05 * (2.0 * theta).cos() * phi.sin()
    })
}

/// A second, genuinely different bumpy shape (for collision tests).
pub fn bumpy_b(rings: usize, segs: usize) -> Mesh {
    uv_sphere(rings, segs, &|theta: f64, phi: f64| {
        1.0 + 0.12 * (2.0 * theta).sin() * (3.0 * phi).cos()
            + 0.07 * (4.0 * phi).sin() * theta.sin()
            + 0.06 * (4.0 * theta).cos() * phi.cos()
    })
}

/// Superellipsoid: p=2 ellipsoid, p=8 rounded box. Star-shaped, so the
/// same surface regenerates exactly at any (rings, segs) resolution.
pub fn superellipsoid(rings: usize, segs: usize, a: f64, b: f64, c: f64, p: f64) -> Mesh {
    uv_sphere(rings, segs, &move |theta: f64, phi: f64| {
        let d = [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ];
        let q = (d[0] / a).abs().powf(p) + (d[1] / b).abs().powf(p) + (d[2] / c).abs().powf(p);
        q.powf(-1.0 / p)
    })
}

/// Rodrigues rotation about a (normalized internally) axis.
pub fn rotate(mesh: &Mesh, axis: [f64; 3], angle: f64) -> Mesh {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = angle.sin_cos();
    let mut out = mesh.clone();
    for p in &mut out.vertices {
        let kxp = [
            k[1] * p[2] - k[2] * p[1],
            k[2] * p[0] - k[0] * p[2],
            k[0] * p[1] - k[1] * p[0],
        ];
        let kdp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
        for d in 0..3 {
            p[d] = p[d] * c + kxp[d] * s + k[d] * kdp * (1.0 - c);
        }
    }
    out
}

pub fn translate(mesh: &Mesh, t: [f64; 3]) -> Mesh {
    let mut out = mesh.clone();
    for p in &mut out.vertices {
        for d in 0..3 {
            p[d] += t[d];
        }
    }
    out
}

pub fn scale(mesh: &Mesh, s: f64) -> Mesh {
    let mut out = mesh.clone();
    for p in &mut out.vertices {
        for pd in p.iter_mut() {
            *pd *= s;
        }
    }
    out
}

/// Same geometry, different indexing: cyclically shift vertex ids by
/// `vshift` (remapping faces) and rotate the face list by `fshift`.
pub fn reorder(mesh: &Mesh, vshift: usize, fshift: usize) -> Mesh {
    let nv = mesh.vertices.len();
    let mut vertices = vec![[0.0; 3]; nv];
    for (old, &v) in mesh.vertices.iter().enumerate() {
        vertices[(old + vshift) % nv] = v;
    }
    let mut faces: Vec<[u32; 3]> = mesh
        .faces
        .iter()
        .map(|f| {
            [
                ((f[0] as usize + vshift) % nv) as u32,
                ((f[1] as usize + vshift) % nv) as u32,
                ((f[2] as usize + vshift) % nv) as u32,
            ]
        })
        .collect();
    let nf = faces.len().max(1);
    faces.rotate_left(fshift % nf);
    Mesh { vertices, faces }
}

/// Deterministic RNG (SplitMix64). Same stream on every platform, so
/// perturbation seeds reproduce anywhere.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = (1.0 - self.next_f64()).max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    pub fn unit_vec(&mut self) -> [f64; 3] {
        loop {
            let v = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if n2 > 1e-6 && n2 <= 1.0 {
                let n = n2.sqrt();
                return [v[0] / n, v[1] / n, v[2] / n];
            }
        }
    }
}

/// RMS distance of vertices from the vertex centroid. A scale proxy for
/// expressing noise and tamper magnitudes as a fraction of object size.
pub fn rms_radius(mesh: &Mesh) -> f64 {
    let n = mesh.vertices.len().max(1) as f64;
    let mut c = [0.0; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            c[k] += v[k];
        }
    }
    for ck in c.iter_mut() {
        *ck /= n;
    }
    let sum: f64 = mesh
        .vertices
        .iter()
        .map(|v| {
            (v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2)
        })
        .sum();
    (sum / n).sqrt()
}

fn radial_dir(v: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let d = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len > 1e-12 {
        [d[0] / len, d[1] / len, d[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn vertex_centroid(mesh: &Mesh) -> [f64; 3] {
    let n = mesh.vertices.len().max(1) as f64;
    let mut c = [0.0; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            c[k] += v[k];
        }
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Scan-noise model: independent radial displacement per vertex,
/// v += dir(v) * N(0, sigma_abs).
pub fn radial_noise(mesh: &Mesh, sigma_abs: f64, rng: &mut Rng) -> Mesh {
    let c = vertex_centroid(mesh);
    let mut out = mesh.clone();
    for v in &mut out.vertices {
        let d = radial_dir(*v, c);
        let off = sigma_abs * rng.normal();
        for k in 0..3 {
            v[k] += d[k] * off;
        }
    }
    out
}

/// Forgery probe: local gaussian bump of amplitude `amp_abs` and angular
/// width `sigma_ang` centered on direction `dir`.
pub fn tamper_bump(mesh: &Mesh, amp_abs: f64, sigma_ang: f64, dir: [f64; 3]) -> Mesh {
    let c = vertex_centroid(mesh);
    let mut out = mesh.clone();
    for v in &mut out.vertices {
        let d = radial_dir(*v, c);
        let cosang = (d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2]).clamp(-1.0, 1.0);
        let ang = cosang.acos();
        let off = amp_abs * (-0.5 * (ang / sigma_ang).powi(2)).exp();
        for k in 0..3 {
            v[k] += d[k] * off;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_is_closed_and_outward() {
        let m = uv_sphere(32, 64, &|_, _| 1.0);
        let (v, c) = m.volume_centroid();
        assert!(
            (v - 4.0 * PI / 3.0).abs() / (4.0 * PI / 3.0) < 0.01,
            "volume {v}"
        );
        for k in 0..3 {
            assert!(c[k].abs() < 1e-9, "centroid {c:?}");
        }
    }
}
