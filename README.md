# Spectral3d

A spectral hashing method for 3D objects that generates stable shape fingerprints, allowing offline verification without needing to check against a central database.

## The problem

A file hash can tell you whether a file has changed, but it cannot tell you whether two different scans represent the same physical object.

The same mug scanned at different times may produce different meshes due to changes in pose, triangulation, vertex ordering, scale, or scan noise. A practical shape fingerprint must remain stable under these variations.

## What it does

`spectral3d` captures the overall shape of a mesh using global geometric statistics independent of mesh triangulation and spatial orientation.

Register an object once. You get back its fingerprint and a bit of public helper data. Check a later scan against that helper and it recovers the same fingerprint for the same object, a different one otherwise.

## Results

| Scenario                              | Same object recognized |
| ------------------------------------- | ---------------------: |
| Different pose, mesh layout, or scale |               **100%** |
| Typical scan noise                    |            **92–100%** |
| Deliberate shape deformation          |                 **0%** |
| Unrelated objects falsely matched     |                 **0%** |

The method remains stable under normal scanning variations and rejects meaningful geometric changes and unrelated objects.

## Why it matters

- Binds identity to the physical object rather than a specific scan file.
- Enables local verification without relying on a central server.
- Can be combined with weight, material, or other physical signals.

## Gates

`spectral3d` refuses any input it can't turn into a stable fingerprint. The gates, what each turns away, and why:

| Gate                 | Rejects                                                                               | Why                                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Malformed faces      | A face index pointing past the vertex array, or a face that repeats a vertex          | The vertex and face arrays must name real triangles before any integral can index a vertex.                                                 |
| Open or non-manifold | Holes, open edges, edges shared by three or more faces, mismatched winding            | The volume integrals only hold on a watertight, oriented surface. One missing face lets the volume drift with the origin.                   |
| Inconsistent shells  | Multi-part meshes whose shells wind opposite ways, or a shell around near-zero volume | A flipped shell subtracts its mass from the integrals and hands the same object a different fingerprint.                                    |
| Empty volume         | Zero or near-zero enclosed volume                                                     | A flat or hollow mesh has no size to normalize against.                                                                                     |
| Flat spread          | A vertex spread collapsed onto a plane or line (a degenerate covariance)              | No stable axes to measure the features against.                                                                                             |
| Weak shape           | Near-regular solids like spheres and boxes, and the fully degenerate near-flat case   | Near-regular ones crowd one corner where different solids collapse to a single fingerprint. A vertex cloud flattened onto a plane has no third axis to measure against. |
| Bad scale parameter  | A quantization scale outside its sane window                                          | An extreme scale folds every shape into one fingerprint, matching everything.                                                               |

## Limits

- The fingerprint is one-way but low-entropy. A determined attacker can brute-force the coarse shape behind it, so treat it as an identity tag, not a secret or an auth token.
