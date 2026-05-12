# vessel-generator design notes

This document captures the design choices behind the `vesselgen` package
and what is intentionally **not** in v1.

## Module map

```
vesselgen/
  config.py        dataclasses for every parameter; GenerationConfig.sample() draws a VesselConfig
  centerline.py    parametric centerline + local frame (tangent / normal / binormal)
  cross_section.py Fourier-perturbed lumen contours that drift smoothly along arclength
  wall.py          per-angle wall thickness with low-mode perturbations
  sweep.py         centerline + cross-section field -> closed (capped) trimesh
  bifurcation.py   side-branch and Y-junction attachment via boolean union
  vessel.py        top-level Vessel: branch tree, unified meshes, sampling API
  sampling.py      pose sampling + per-frame ground truth (lumen contour, distances, CSA, ...)
  io.py            on-disk format: lumen.obj, outer.obj, vessel.json, branches/branch_NN.json
  visualize.py     QC plots
  library.py       batch generation
  tools/           CLI entrypoints
```

## Coordinate convention

- Vessel axis along **Y**, cross-sections in the **xz** plane, units mm.
- Matches the simulator's `phantom_maker.generate_cylinder_thick_mesh`
  convention so the OBJ files load directly with the existing IVUS
  example.

## Outward normals in memory, inward normals on disk

trimesh and the manifold3d boolean engine assume **outward** face normals
on closed solids. The simulator (per `phantom_maker.py`) expects
**inward** normals on its phantom surfaces so rays cast from inside the
lumen hit the front face.

Resolution: in-memory meshes use outward normals. `io.save_vessel` flips
the winding at OBJ export time. `io.load_vessel` flips back so a loaded
`Vessel` has the same in-memory convention as a freshly built one.

## Lumen and wall parameterisation

Both the lumen radius and the wall thickness are written as

```
f(theta, s) = mean(s) * (1 + sum_k a_k(s) * cos(k * theta + phi_k(s)))
```

with two crucial properties:

1. The amplitudes `a_k` are scaled so that the *total* perturbation is
   bounded by `max_perturbation_frac`. This guarantees the contour is
   never degenerate regardless of mode superposition.
2. The phases `phi_k` drift slowly with arclength so adjacent
   cross-sections look similar. Without phase drift, the lumen would
   appear to spin frame-to-frame.

Mode choices:

- **Lumen**: k = 2..6 (k=2 is ovality, k=3 is trefoil, k=4 looks
  quasi-square). Default `max_perturbation_frac = 0.18`.
- **Wall**: k = 1..3 only. k=1 produces a clearly thicker side ("eccentric
  intima"), k=2 produces oval thickening. Default
  `max_perturbation_frac = 0.6` — large enough to make the wall asymmetry
  obvious in renderings.

## Bifurcations (side branches only in v1)

A side-branch is constructed by:

1. Building a daughter centerline whose origin is *recessed* into the
   parent centerline by a small distance ``r_recess`` along the
   daughter's outgoing direction. ``r_recess`` is computed as
   ``recess_frac_of_parent_radius * parent_local_mean_radius`` and
   **clamped** to never exceed ``0.85 * parent_local_mean_radius``. The
   clamp is the critical fix for the "X-on-a-stick" failure mode: with a
   recess larger than the parent radius, the daughter origin lands on
   the *opposite* side of the parent and the daughter mesh crosses the
   parent twice, producing two ostia (one on each side). The clamp
   guarantees the daughter cap stays inside the parent, so the daughter
   wall crosses the parent wall on exactly one side and the second
   ostium cannot form.
2. Sweeping the daughter into closed lumen and outer trimeshes.
3. Calling `trimesh.boolean.union([parent, daughter], engine="manifold")`
   on the lumen meshes and again on the outer meshes.

If the boolean union fails (rare; happens at degenerate emergence
angles), `BifurcationError` is raised so the batch generator can
re-sample.

### Y-junction (deferred)

A full Y-junction (parent splits into two daughters at a single station)
was prototyped and removed from v1. The straightforward implementation
-- truncate the parent at the junction, attach two daughters at the
distal end, and boolean-union all three -- produces visibly wrong
ostium geometry: the two daughter caps overlap each other inside the
parent stub before the union is computed, the resulting union surface
has a triangular fin or a self-intersection at the apex of the split,
and the catheter sees a non-physical inner ridge. A correct
implementation needs at least one of:

- a custom blended apex surface joining the three branches with a
  tangent-continuous saddle, instead of relying on a single boolean
  union of three primitives;
- a smarter daughter construction that trims the daughter caps against
  each other before unioning;
- or a downstream manifold-cleanup pass (e.g. quadric remeshing of the
  ostium triangle) to remove the self-intersections the boolean leaves
  behind.

None of these are needed for the porcine-lab gate (the model needs to
recognise side branches; full Y-junctions are not part of the v1
diversity envelope) so the feature is removed from the codebase. The
``SideBranchConfig`` schema and the boolean-union plumbing are kept and
will accept the additional logic when Y-junctions are revisited.

## Pose sampling

The training pipeline asks for **frame-level snapshots**, not pullbacks.
The sampling API supports three usage modes:

### Arbitrary random sampling (the typical training-loop call)

`Vessel.sample_pose(rng)` does:

1. **Pick a branch** weighted by arclength (so longer branches get more
   samples).
2. **Pick an arclength** uniformly along the branch.
3. **Pick a 2D position inside the lumen contour** at that arclength,
   rejection-sampled. The interpolated lumen contour is reused for the
   in-polygon test; an `edge_margin_mm` parameter enforces a minimum
   wall clearance, or 0 to allow probe-against-wall poses for training
   the contact-signal head.
4. **Pick a small probe-axis tilt** uniformly in
   `[0, max_tilt_deg]`, with random azimuth around the local tangent.
5. **Build the rotation matrix** that maps the probe-local frame
   (where the probe long axis is +Y at zero rotation, matching the
   simulator's `IVUSProbe`) to world coordinates, then convert to Euler
   XYZ degrees.

Every call returns a different pose; the simulator can render an
unbounded number of distinct frames per vessel.

### Constrained sampling

`Vessel.sample_pose_in_branch(branch_name, rng, arclength_mm=None)`
restricts the sample to one named branch. If `arclength_mm` is
provided, the pose is also pinned to that station; otherwise arclength
is sampled uniformly along the branch as in the arbitrary case. Useful
for stratified training (e.g. "give me 50 frames per branch per 5 mm
station" for diversity coverage).

### Explicit pose construction

`Vessel.pose_at(position, probe_axis_world=None,
require_inside_lumen=True)` builds a `PoseSample` at a user-specified 3D
position with no random sampling. The probe long axis defaults to the
local tangent of the nearest branch's centerline. Useful for replay,
hand-crafted regression cases, and for any caller that already knows
exactly where the probe should be.

`Vessel.contains_point(position)` is a quick true/false check against
the lumen mesh.

### Returned object

The returned `PoseSample` has `position`, `rotation_euler_deg`,
`rotation_matrix`, `probe_axis_world`, `branch_id`, `branch_name`,
`arclength_mm`, `centerline_offset_mm`, and `tilt_deg`. The first two
fields plug straight into `rs.IVUSProbe(rs.Pose(position=..., rotation=...))`.

## Ground truth extraction

`Vessel.ground_truth_at(pose)` slices the lumen and outer meshes with
the imaging plane (perpendicular to `pose.probe_axis_world`, through
`pose.position`) using `trimesh.section`. The resulting Path3D is
projected onto the probe-local x and z basis vectors to give 2D polygons.

For each angular bin `theta`, a ray from the probe origin at angle
`theta` is intersected with every polygon segment using a closed-form
2x2 line-segment / ray solve. The shortest positive intersection is the
distance-to-wall at that angle. NaN is returned for rays that miss
within the imaging window (e.g. rays into the open ostium of a
bifurcation when the probe is in the parent).

The ground-truth object also exposes the polygons themselves (so the DL
team can rasterise their own segmentation labels), per-angle wall
thickness, CSA, equivalent lumen diameter, and the list of visible
branch IDs.

## What's deliberately not in v1

- **Curved 3D centerlines.** The `CenterlineConfig` accepts curvature
  parameters but they are reserved; `Centerline` always builds a
  straight line. The local-frame API (`tangent(s)`, `frame(s)`,
  `project()`) is already polymorphic, so a curved centerline can drop
  in without changing any caller.
- **Y-junctions.** Removed from v1; see "Y-junction (deferred)" above
  for what was tried and what would need to be added.
- **Multiple bifurcations per vessel.** The schema supports a list of
  side branches, but the default `GenerationConfig` caps at one for
  reliability. Increasing the cap requires the boolean engine to be
  resilient to repeated unions.
- **Pathology features.** No plaque, calcium, stent, dissection, or
  thrombus modelling. The simulator's material set already supports
  these as separate meshes if needed; a v2 extension can place plaque
  inclusions inside the lumen mesh.
- **Tissue-layer modelling.** Two surfaces only (lumen + adventitia
  outer), mapped to two simulator materials (`vessel_wall` and
  `extravascular`). Intima / media / adventitia separation is a v2
  concern.

## Dependencies

- `numpy`, `scipy` for arithmetic.
- `trimesh` + `manifold3d` for meshes and boolean operations.
- `networkx` for trimesh's section-to-path assembly.
- `rtree` for trimesh's `contains` ray-cast acceleration (the sampler
  uses it as a sanity check; if `rtree` is unavailable, the code still
  runs but `Vessel.lumen_mesh.contains()` will not).
- `shapely` reserved for v2 polygon operations.
- `matplotlib` for QC previews only; not required at import time of the
  core API.

## Smoke tests

`tests/test_geometry.py` covers:

- Centerline frame is orthonormal.
- Cross-section radii are positive everywhere and drift smoothly along
  arclength.
- Single-branch sweep produces a watertight outward-normal mesh.
- Saved OBJ is watertight with **inward** normals.
- Vessels build with no bifurcation, with a side branch, and with a
  Y-junction.
- `sample_pose` returns positions actually inside the lumen.
- Ground-truth extraction returns positive distances and a finite CSA.
- Bifurcated vessels report multiple visible branch IDs at appropriate
  poses.

Run with `pytest vessel-generator/tests/`.
