# vessel-generator

Procedural geometry for vessel segments used as **training data** for the
IVUS deep-learning navigation model. This module is intentionally separate
from the simulator (`i4h-sensor-simulation/`) and from the instrument
calibration code (`instrument-calibration/`). Its only job is to emit
realistic, varied **vessel volumes** with **per-vessel ground truth**, in
a form the simulator can consume directly.

## What it produces

For each generated vessel, the module writes a folder:

```
out/<vessel_id>/
  lumen.obj          # closed, inward-facing lumen surface (simulator material: vessel_wall)
  outer.obj          # closed, inward-facing outer surface (simulator material: extravascular)
  preview.png        # 3D preview + cross-section gallery (QC)
  vessel.json        # branch topology, generation parameters, seed, axis convention
  branches/
    branch_<id>.json # per-branch centerline, local frames, cross-section parameters
```

Conventions:
- **Vessel axis along Y**, cross-sections in the **xz** plane (matches
  `i4h-sensor-simulation/ultrasound-raytracing/utils/phantom_maker.py`).
- **Units: mm** everywhere.
- **Inward-facing normals** on both surfaces, so rays from a probe inside
  the lumen hit the front face. Each vessel surface is a **closed
  watertight mesh**; the simulator can read the inner surface directly with
  the `vessel_wall` material and the outer surface with the `extravascular`
  material.

## Why volumes, not pullback paths

The DL training pipeline samples **frame-level snapshots**: it picks a
random point inside the lumen and a small probe-axis tilt, then renders one
IVUS frame. There is no pullback. This module therefore exposes the
sampling API needed for that workflow.

### Arbitrary random sampling (the typical training-loop call)

```python
from vesselgen import Vessel
import numpy as np

v = Vessel.load("out/vessel_0001")
rng = np.random.default_rng(0)

for _ in range(100):
    pose = v.sample_pose(rng, max_tilt_deg=15.0, edge_margin_mm=0.2)
    gt = v.ground_truth_at(pose)
    # pose.position, pose.rotation_euler_deg feed straight into rs.IVUSProbe
    # gt.distance_to_lumen_wall_mm, gt.lumen_contour_polygons,
    # gt.branch_ids_visible, gt.lumen_csa_mm2, ...
```

Every call to `sample_pose` returns a different pose with no
preselection: branch is sampled weighted by arclength, then arclength
uniformly along that branch, then a 2D in-lumen position rejection-
sampled (with optional `edge_margin_mm` wall clearance), then a small
probe-axis tilt. **The simulator can render an unbounded number of
distinct frames per vessel.**

### When you want a specific pose

```python
# Pose pinned to a specific branch, optionally a specific arclength station
pose = v.sample_pose_in_branch("daughter_a", rng, arclength_mm=5.0)

# Pose at an explicitly chosen 3D point (probe axis defaults to local
# vessel tangent; pass probe_axis_world to override)
pose = v.pose_at(position=np.array([0.0, 2.5, 0.5]))

# Quick check whether an arbitrary 3D point is inside the lumen
v.contains_point(np.array([0.0, 0.0, 0.0]))
```

### Bifurcation snapshots are automatic

When the imaging plane (perpendicular to the probe long axis) crosses an
ostium, `gt.lumen_contour_polygons` returns multiple polygons and
`gt.branch_ids_visible` lists every branch the plane intersects — that is
how the model learns to recognise bifurcations.

## What's modelled

- **Non-circular lumen contours.** Smooth Fourier-perturbed shapes (modes
  k = 2..6) that drift along arclength so adjacent cross-sections look
  similar but never identical.
- **Variable wall thickness.** Per-angle thickness modulation (k = 1..3)
  so the wall is thick on one side and thin on another, with the pattern
  varying along the length.
- **Diameter taper.** Mean lumen radius varies smoothly along arclength.
- **Side-branch ostia.** A daughter vessel buds off the parent at an
  angle and continues away. The parent passes through unchanged. The
  daughter origin is recessed *just inside* the parent (capped at 0.85
  of the parent's local radius) so the daughter wall crosses the parent
  wall on exactly one side, never both. The daughter is stitched into
  the parent with a **trimesh + manifold3d boolean union** so the lumen
  is a single watertight surface with a clean ostium.

  Side-branch is the only bifurcation flavour modelled in v1. Full
  Y-junctions (parent splits into two daughters) were prototyped and
  pulled because the ostium geometry needs more work; see
  ``docs/design.md`` for what was tried.

## Install

From repo root:

```bash
python -m pip install -e vessel-generator
```

## CLIs

```bash
# One vessel, parameters from a YAML preset (see configs/)
vesselgen-vessel --preset peripheral_straight --out out/vessel_0001

# A batch of N vessels with sampled parameters
vesselgen-dataset --preset peripheral_mixed --n 64 --out out/dataset

# Sample N (pose, ground truth) tuples from one vessel
vesselgen-frames --vessel out/vessel_0001 --n 200 --out out/vessel_0001/frames
```

## Layout

```
vessel-generator/
  vesselgen/            # importable package
    config.py           # generation parameter dataclasses
    centerline.py       # parametric centerlines + Frenet/Bishop frames
    cross_section.py    # Fourier-perturbed lumen contours
    wall.py             # per-angle wall thickness model
    sweep.py            # cross-section -> closed trimesh
    bifurcation.py      # daughter branch attachment via boolean union
    vessel.py           # Vessel class (branches + unified mesh + sampling)
    sampling.py         # pose sampling + ground-truth extraction
    io.py               # mesh + manifest IO
    visualize.py        # 3D + 2D QC plots
    library.py          # batch generation
    tools/              # CLIs
  tests/                # smoke tests
  examples/             # standalone scripts
  docs/                 # design notes
```
