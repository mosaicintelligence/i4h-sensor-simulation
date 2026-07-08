# Configuration

This page documents the main knobs for anatomy realism and batch diversity.
For CLI flags see [CLI reference](cli-reference.md).

## Anatomical scale

Batch defaults in `GenerationConfig` target **large peripheral** vessels
(PV .035 femoral/iliac scale), not coronary:

| Draw | Lumen diameter | Wall thickness | Segment length |
|------|----------------|----------------|----------------|
| Typical (62%) | 8–13 mm | 0.65–1.2 mm | 45–75 mm |
| Aortic-scale (18%) | 16–23 mm | 1.0–1.5 mm | 45–75 mm |
| Large beyond-FOV (10%) | 24–32 mm | 1.0–1.5 mm | 45–75 mm |
| Small-vessel (10%) | 3.6–7.0 mm | 0.5–0.9 mm | 45–75 mm |
| Side branch | 55–80% of parent radius | ~85% of parent wall | ≥ parent length |

Typical inner wall radii start around **4 mm**, while the small-vessel draw
explicitly introduces cases with wall echoes inside or near the catheter
ring-down disc (~2–3.6 mm).

The **large beyond-FOV** draw scales the lumen up (radius 12–16 mm) so the
wall runs past the imaging field of view (`t_far_mm` = 17.5 / 20 / 30 mm) on
some angular sectors for typical, naturally off-centre poses. Those A-lines
have no wall echo and their `distance_to_lumen_wall_mm` /
`distance_to_outer_wall_mm` are `NaN`. Because triggering depends on the FOV,
these vessels reliably show open sectors at the 17.5/20 mm FOVs and mostly
stay in view at 30 mm. The scale probabilities (`aortic_scale_probability`,
`large_vessel_beyond_fov_probability`) should sum to ≤ 1.0; the remainder is
the typical draw.

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
    side_branch_probability=0.45,
    layered_wall_probability=0.7,
    n_layers_weights=(0.15, 0.15, 0.70),  # 1-, 2-, 3-layer draw
    calcification_probability=0.35,
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
