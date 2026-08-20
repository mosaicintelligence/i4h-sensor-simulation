# Configuration

This page documents the main knobs for anatomy realism and batch diversity.
For CLI flags see [CLI reference](cli-reference.md).

## Anatomical scale

Batch defaults in `GenerationConfig` target **large peripheral** vessels
(PV .035 femoral/iliac scale), not coronary:

| Draw | Lumen diameter | Wall thickness | Segment length |
|------|----------------|----------------|----------------|
| Typical (72%) | 8–13 mm | 0.65–1.2 mm | 45–75 mm |
| Aortic-scale (18%) | 16–23 mm | 1.0–1.5 mm | 45–75 mm |
| Adjacent parallel vessels (10%) | 3.6–7.0 mm | 0.3–0.55 mm | 45–75 mm |
| Side branch | 55–80% of parent radius | ~85% of parent wall | ≥ parent length |

The scale buckets are mutually exclusive and drawn from a single partition;
the adjacent-vessels share is carved out of the typical share (aortic stays
at 18%). For typical/aortic draws, inner wall radii start around **4 mm**
so anatomy sits outside the catheter ring-down disc (~2–3.6 mm). The
adjacent-vessels bucket uses a dedicated primary size range so the
parent–neighbor boundary can sit in the ring-down zone, where the
catheter artifact obscures it on B-mode (a merge frame).

Calibration scenarios that motivated these scales:
`instrument-calibration/p035_visions/vessel_evaluation.py` scenarios 06
(femoral) and 07 (aorta).

## Wall layering

| `--layers` | Geometry | IVUS appearance |
|------------|----------|-----------------|
| `1` | Single slab (`lumen.obj` + `outer.obj`) | One bright wall echo |
| `2` | Media + adventitia interfaces | Partial layering |
| `3` | Intima + media + adventitia | Bright–dark–bright signature |

**Defaults:**

- `vesselgen-vessel`: `--layers 1` (explicit opt-in for trilaminar)
- `vesselgen-dataset`: `layered_wall_probability = 0.7`
- `render_paired_dataset.py`: trilaminar pinned for all vessels

## Side branches

Controlled by `--side-branch` (single vessel) or
`side_branch_probability` (batch, default 0.45).

Side branches attach via trimesh boolean union. At most one side branch per
vessel in the default sampler (`max_side_branches = 1`).

Trilaminar walls **can** coexist with side branches (per-layer boolean
union). The paired renderer supports both in the same vessel.

**`side_branch_probability` is a conditional rate, not a whole-dataset
rate.** Side branches and adjacent vessels are mutually exclusive
topologies, so the probability is applied **only to vessels that were not
drawn as the adjacent-vessels type**. Read it as *"45% of my
non-adjacent-vessel-type vessels get a side branch"*, not *"45% of the whole
dataset"*. The whole-dataset side-branch fraction is

```
side_branch_probability * (1 - adjacent_vessel_probability)
# defaults: 0.45 * (1 - 0.10) = 0.405  → ~41% of all vessels
```

## Adjacent parallel vessels

Two parallel, non-touching neighbor vessels (the artery-next-to-vein case).
The catheter stays in the **parent** lumen; neighbors are visible because
the IVUS beam reaches them laterally. When ring-down obscures the wall
between parent and neighbor, segmentation models tend to merge the two
structures on B-mode — this scenario exists to teach that appearance. A
sampled pose is a **merge frame** when the parent–neighbor boundary falls
inside the catheter ring-down zone (about 2.8 mm), so the septum is
obscured and the vessels read as one. Unlike side branches, neighbors do
**not** fuse into the parent wall — they are separate tubes offset laterally
in the cross-section plane. At build time their per-layer meshes are
concatenated into the parent's surface meshes, so they **share the parent's
materials and segmentation labels** (they do not get their own label IDs).

Controlled by `--adjacent-vessel` (single vessel, plus `--n-adjacent`,
`--adjacent-gap-mm`, `--adjacent-radius-frac`, `--adjacent-azimuth-deg`) or
`adjacent_vessel_probability` (batch).

| Knob | Default | Notes |
|------|---------|-------|
| `adjacent_vessel_probability` | 0.10 | Own scale bucket carved from the typical share; whole-dataset rate of adjacent-vessels draws (1–2 parallel neighbors, probe in parent). Mutually exclusive with side branches. |
| `adjacent_parent_radius_mm_range` | (1.8, 3.5) | Lumen radius (mm) of the adjacent-case primary. Sized so the parent–neighbor boundary can fall in ring-down with pose bias; wider than the tightest ring-down-only range to retain primary-lumen visibility in centered poses. |
| `adjacent_parent_wall_thickness_mm_range` | (0.3, 0.55) | Wall thickness of the adjacent-case primary; kept thin so the septum stays near ring-down scale. |
| `adjacent_vessel_count_range` | (1, 2) | Number of neighbors |
| `adjacent_vessel_radius_frac_range` | (0.7, 1.1) | Neighbor lumen radius as a fraction of the primary radius (70–110%). Does not change the neighbor's near-wall distance, only how much FOV it fills. |
| `adjacent_vessel_gap_mm_range` | (0.02, 0.15) | Target edge-to-edge slack added on top of the measured outer radii. Near zero (strictly positive so shells stay disjoint) so the vessel–vessel boundary sits in the ring-down zone, where the ring-down artifact obscures it. |

Neighbor wall thickness is **not** copied verbatim from the primary: each neighbor inherits the parent's layered structure but the total thickness is rescaled to ``max(adjacent_parent_wall_thickness_mm_range[0], parent_wall × radius_frac)``.

**Tight, non-overlapping placement:** each neighbor's center-to-center
offset is `measured(parent) + measured(neighbor) + gap`, where
`measured_outer_radius_mm` reconstructs the deterministic lumen + wall
fields the mesh builder produces for that branch's seed and returns the
exact maximum outer radius over every station and angle. Because it is the
true maximum (not a loose bound that sums independent worst-case lumen and
wall perturbations, which peak at different angles), the neighbor sits as
close as the geometry allows while the realised edge-to-edge clearance
equals `gap` exactly. For two neighbors the second is packed right next to
the first at the minimum separation that still clears it (`delta_min`, law
of cosines), so a pair forms a tight cluster on one side of the parent
rather than spreading around it. Neighbor walls are separated by at least
`gap` at every station and angle; the mesh shells are genuinely disjoint
(validated in `tests/test_adjacent_vessels.py`).

**Mutual exclusivity:** `adjacent_vessels` and `side_branches` cannot be set
on the same `VesselConfig` (raises `ValueError`), and the batch sampler
never draws both. `adjacent_vessel_probability + aortic_scale_probability`
must be `<= 1`.

**On-disk meshes (for rendering):** neighbor meshes are concatenated into the
merged top-level surfaces for the geometry stack, and *also* written per-object
under `objects/` with a manifest `objects` section (`surfaces_are_merged: true`)
so raysim can render each vessel as a distinct nested object. See
[simulator-integration.md](simulator-integration.md#adjacent-vessels-merged-vs-per-object-meshes).

**Neighbor-directed pose bias:** `adjacent_vessel_pose_bias_prob` (default
0.8) controls the fraction of poses drawn eccentric toward a neighbor (when
several neighbors are present, one is chosen at random for that pose), which
raises the rate of merge frames. From a centered probe the vessel–vessel
boundary sits at `parent_outer_radius + gap`, always outside the about
2.8 mm ring-down zone; an eccentric probe on the neighbor side pulls the
boundary into ring-down so it is obscured on the B-mode. Passed to
`sampling.sample_pose(neighbor_bias_prob=...)`; a no-op for vessels with
no neighbors. Kept below 1.0 so a minority of frames still show a
resolved boundary. With the default radius range and thin wall, the
merge-frame rate tracks the bias probability closely (about 80% at 0.8
bias).

## Lesions

In-wall lesion kinds: `hard` (calcified), `soft_lipid`, `fibrous`,
`thrombus`. Lesions cluster on a per-vessel diseased sector for eccentric
plaque.

**Constraint:** lesions are skipped when a vessel has side branches (the
interface-smoothing step is not yet safe after bifurcation boolean union).

| Knob | Default | Notes |
|------|---------|-------|
| `diseased_vessel_probability` | 0.35 | Fraction of eligible vessels |
| `lesions_per_vessel_range` | (1, 3) | Count when diseased |
| `lesion_arc_extent_deg_range` | (60, 150) | Circumferential extent |
| `lesion_axial_extent_mm_range` | (6, 16) | Along-vessel extent |

Single-vessel CLI: `--n-lesions N --lesion-kinds hard,soft_lipid` (requires
no `--side-branch`).

## Guidewire

Optional tungsten cylinder along the parent centerline with lateral offset.

| Knob | Default |
|------|---------|
| `guidewire_probability` | 0.70 (batch, when no bifurcation) |
| `guidewire_diameter_choices_mm` | 0.36, 0.46, 0.89 (0.014", 0.018", 0.035") |
| `guidewire_lateral_offset_frac_range` | (0.4, 0.85) of lumen radius |

Single-vessel CLI: `--guidewire --guidewire-diameter-mm 0.36`.

## Pose sampling

| Knob | Default | Purpose |
|------|---------|---------|
| `wall_contact_probability` | 0.12 | Probe-against-wall frames |
| `max_tilt_deg` | 15.0 | Probe-axis tilt |
| `edge_margin_mm` | 0.15–0.2 | Minimum wall clearance |

Wall-contact poses produce the bright contact rim and opposite-side shadow
artefacts seen in real IVUS.

## GenerationConfig (batch)

The full dataclass lives in `vesselgen/config.py`. Key fields:

```python
from vesselgen.config import GenerationConfig

cfg = GenerationConfig(
    side_branch_probability=0.45,       # conditional on non-adjacent type
    adjacent_vessel_probability=0.10,   # adjacent parallel-vessels bucket
    adjacent_vessel_gap_mm_range=(0.02, 0.15),
    layered_wall_probability=0.7,
    n_layers_weights=(0.15, 0.15, 0.70),  # 1-, 2-, 3-layer draw
    diseased_vessel_probability=0.35,
    guidewire_probability=0.70,
    wall_contact_probability=0.12,
)
```

Pass a custom config to `vesselgen.library.generate_dataset()` or adjust
the paired renderer's internal config in
`render_paired_dataset.py::generate_paired_dataset()`.

## Programmatic single vessel

```python
from vesselgen.config import VesselConfig, BranchConfig, LayeredWallConfig
from vesselgen.vessel import Vessel
from vesselgen.io import save_vessel

cfg = VesselConfig(
    parent=BranchConfig(...),
    seed=42,
    name="my_vessel",
)
vessel = Vessel.from_config(cfg)
save_vessel(vessel, "out/my_vessel")
```

See `vesselgen/tools/generate_vessel.py` for a complete flag-to-config
mapping.
