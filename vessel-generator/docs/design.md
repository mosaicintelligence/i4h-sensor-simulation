# vessel-generator design notes

Internal design reference for implementers. For user-facing documentation
start with [quickstart.md](quickstart.md).

## Module map

```
vesselgen/
  config.py        dataclasses; GenerationConfig.sample() draws a VesselConfig
  centerline.py    parametric centerline + local frame (tangent / normal / binormal)
  cross_section.py Fourier-perturbed lumen contours
  wall.py          per-angle wall thickness + LayeredWallConfig builder
  sweep.py         centerline + cross-section field → closed trimesh
  inclusions.py    in-wall lesion meshes
  guidewire.py     tungsten guidewire cylinder
  bifurcation.py   side-branch attachment via boolean union (per-layer for trilaminar)
  vessel.py        top-level Vessel: branch tree, surfaces, sampling API
  sampling.py      pose sampling + per-frame ground truth
  io.py            on-disk format: OBJ + vessel.json + branches/
  labels.py        segmentation label IDs (rendering layer)
  sim_randomization.py  per-frame simulator diversity (rendering layer)
  visualize.py     QC plots
  library.py       batch generation
  tools/           CLI entrypoints
```

## Coordinate convention

- Vessel axis along **Y**, cross-sections in the **xz** plane, units mm.
- Matches the simulator's `phantom_maker.generate_cylinder_thick_mesh`
  convention so OBJ files load directly.

## Outward normals in memory, inward normals on disk

trimesh and manifold3d assume **outward** face normals on closed solids.
The simulator expects **inward** normals on phantom surfaces.

Resolution: in-memory meshes use outward normals. `io.save_vessel` flips
winding at OBJ export. `io.load_vessel` flips back on import.

## Lumen and wall parameterisation

Both lumen radius and wall thickness use:

```
f(theta, s) = mean(s) * (1 + sum_k a_k(s) * cos(k * theta + phi_k(s)))
```

with bounded perturbation amplitudes and slowly drifting phases along
arclength.

- **Lumen**: k = 2..6, default `max_perturbation_frac = 0.18`.
- **Wall**: k = 1..3, default `max_perturbation_frac = 0.6`.

## Bifurcations (side branches)

A side-branch is constructed by:

1. Recessing the daughter origin into the parent (capped at 0.85 × parent
   local radius) so the daughter wall crosses the parent on one side only.
2. Sweeping the daughter into closed meshes.
3. Boolean-unioning parent and daughter (per-layer for trilaminar walls).

If the union fails, `BifurcationError` is raised; batch code retries with
a new seed.

### Y-junction (not implemented)

Full Y-junctions were prototyped and removed. Side branches are the only
bifurcation flavour in the current codebase. See [limitations](#limitations).

## Current capabilities

The generator currently supports:

- Trilaminar intima/media/adventitia walls (`LayeredWallConfig`)
- In-wall lesions (calcified, lipid, fibrous, thrombus)
- Tungsten guidewire with lateral offset
- Side-branch bifurcations (including trilaminar parents)
- Wall-contact pose sampling
- Frame-level geometric ground truth (contours, distances, CSA, branch visibility)

## Pose sampling

Three usage modes:

1. **Random:** `Vessel.sample_pose(rng)` — branch weighted by arclength,
   uniform arclength, rejection-sampled 2D position, small probe tilt.
2. **Constrained:** `Vessel.sample_pose_in_branch(name, rng, arclength_mm=...)`.
3. **Explicit:** `Vessel.pose_at(position, probe_axis_world=...)`.

Returns `PoseSample` with fields that plug into `rs.IVUSProbe`.

## Ground truth extraction

`Vessel.ground_truth_at(pose)` slices lumen and outer meshes with the
imaging plane, projects to 2D polygons, and computes per-angle wall
distances via ray–segment intersection.

## Limitations

| Limitation | Notes |
|------------|-------|
| Straight centerlines only | Curvature parameters reserved in `CenterlineConfig` |
| Y-junctions | Not implemented; side branches only |
| Lesions + bifurcation | Mutually exclusive (interface smoothing unsafe post-union) |
| Multiple bifurcations | Schema supports a list; default sampler caps at one |
| `wall_contact_probability` on `GenerationConfig` | Not auto-forwarded by `Vessel.sample_pose()`; pass explicitly |

## Dependencies

- `numpy`, `scipy`, `trimesh`, `manifold3d`, `networkx`
- `rtree` (optional; accelerates `contains` checks)
- `matplotlib` (QC previews only)
- `shapely` reserved for future polygon operations

Rendering layer additionally requires raysim and
`instrument-calibration/p035_visions/`.

## Tests

```bash
pytest vessel-generator/tests/
```

Coverage includes centerline frames, cross-section positivity, sweep
watertightness, OBJ normal orientation, side-branch builds, pose
sampling, ground-truth extraction, layered walls, lesions, guidewire,
and simulator material name consistency.
