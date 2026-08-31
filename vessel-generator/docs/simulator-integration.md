# Simulator integration

How vessel-generator meshes connect to the IVUS raytracing simulator and
calibrated probe parameters.

## Calibrated operating point

The canonical simulator config for realistic peripheral IVUS is:

```
instrument-calibration/p035_visions/volcano_s5i.yaml
```

This file calibrates the **Visions PV .035 (10 MHz) catheter** on a
**Volcano s5/s5i console**. The paired-dataset renderer loads it via:

```python
from tier1_evaluation import load_calibrated_config

cfg = load_calibrated_config(VISIONS_DIR / "volcano_s5i.yaml")
```

For calibration methodology and parameter provenance, see
[instrument-calibration/p035_visions/README.md](../../instrument-calibration/p035_visions/README.md).

**Naming note:** this repo uses the `p035_visions` folder. References to
`p034` in older notes refer to the same instrument family; always use
`p035_visions/volcano_s5i.yaml` as the canonical path.

## World model

The paired renderer builds a raysim `World` from the vessel manifest:

1. Load every mesh listed in `vessel.json` with its declared material.
2. Set world background to `lumen` (blood pool inside and beyond the wall).
3. Nest materials in surface order: intima → media → adventitia.

This replaces the older two-mesh model (`lumen.obj` + `outer.obj` with
`extravascular` beyond the outer surface). The adventitia back boundary
is acoustically invisible — rays pass through it into the blood pool.

## Adjacent vessels: merged vs. per-object meshes

Adjacent parallel neighbors are built as separate, non-touching tubes and then
**concatenated** into the parent's per-layer surface meshes. That merged mesh is
what the geometry stack (ground-truth slicing, the segmentation rasterizer,
pose previews) consumes: a single watertight mesh per shell keeps
`trimesh.contains` and contour extraction correct with no special-casing.

That representation does **not** render neighbors correctly, because raysim's
material model assumes:

1. **One `Mesh` = one nested closed shell.** A single `Mesh` that fuses two
   disjoint shells (parent lumen + neighbor lumen) breaks the per-mesh
   crossing/containment bookkeeping in the region between them.
2. **Inside→out traversal from the probe.** The material chain is defined as
   "material entered when a ray crosses this surface *outward*." That holds for
   the parent (the probe sits inside it), but a ray reaches a neighbor from the
   **outside**, so the neighbor must be its own object for the transition to
   resolve.

So when a vessel has neighbors, `save_vessel()` **also** writes each vessel as
its own nested object and records them in the manifest. This duplicates the
geometry on disk on purpose — the merged mesh buys the geometry layer, the
per-object meshes buy the renderer, and a consistency test asserts the merged
surfaces are exactly the concatenation of the per-object meshes so they cannot
drift.

```jsonc
"surfaces_are_merged": true,      // top-level `surfaces` fuse all objects
"objects": [
  {
    "name": "parent", "role": "parent",
    "probe_inside": true,          // rays start inside -> traverse inside->out
    "interior_material": "lumen",  // blood pool inside the innermost shell
    "surrounding_material": "adventitia", // material beyond the outermost shell
    "centerline": { "length_mm": …, "origin": […], "direction": […], … },
    "surfaces": [ {"name": "lumen", "material": "intima", "obj": "objects/parent/lumen.obj"}, … ]
  },
  {
    "name": "adjacent_0", "role": "adjacent",
    "probe_inside": false,         // rays enter from outside -> outside->in
    "interior_material": "lumen",
    "surrounding_material": "adventitia",
    "azimuth_deg": …, "center_offset_mm": …, "center_xy_mm": […],
    "mean_radius_mm": …, "outer_radius_bound_mm": …,
    "centerline": { …resolved (laterally offset) origin… },
    "surfaces": [ …objects/neighbor_00/… ]   // parent's chain + a closing `outer` shell
  }
]
```

A neighbor emits **one more shell than the parent**: the adventitia back
boundary (`outer.obj`), carrying a duplicate of the outermost material. The
parent drops that shell because rays exit it into adventitia and stay there
until the FOV, which is already the truth. A ray passes *clean through* a
neighbor, though, and raysim tracks only a single "material outside" slot
(`Payload.outter_material_id`), so with no final surface to cross the ray would
keep the neighbor's innermost wall material all the way out — a wedge of
phantom wall behind every neighbor. The closing shell has the same material on
both sides, so it is acoustically invisible (R = 0); it exists purely to
restore the material state machine.

The top-level `surfaces` list and its OBJ files are unchanged, so the current
renderer keeps working (it renders the parent correctly and ignores `objects`).
A future rendering PR should load the `objects` decomposition when
`surfaces_are_merged` is true: one raysim `Mesh` per surface per object, using
`probe_inside` to know the traversal direction and `interior_material` /
`surrounding_material` for the blood pool and the tissue beyond each object.

**Open decision for the rendering PR:** the material of the *inter-vessel gap*.
Geometry currently lets rays sit in `adventitia` past the outermost surface, so
`surrounding_material` reports `adventitia` (what is modeled today). A renderer
may prefer `peri_adventitia` there; the manifest records current intent rather
than pre-deciding.

## Per-frame randomization

`vesselgen/sim_randomization.py` samples appearance diversity around the
calibrated operating point:

| Tier | Knobs | Fixed |
|------|-------|-------|
| Tier 1 | Gain, tissue acoustics, artifact reduction on/off | PSF geometry, log compression, image grid |
| Tier 2 | TGC deep scale, ring-down amplitude, scattering resolution, FOV | PSF geometry, display LUT chain |

Randomization is applied per frame in `render_paired_dataset.py` via
`SimRandomizationConfig` and `apply_frame_sim_params()`.

## Acoustic boundary offset

Geometric GT contours represent the true lumen/wall boundary. IVUS echoes
arrive slightly offset due to pulse width and beam width. The renderer
applies `acoustic_boundary_offset_mm()` from
`instrument-calibration/p035_visions/vessel_evaluation.py` when
rasterizing segmentation masks.

## Prerequisites

1. Built raysim CUDA extension — see
   [simulator quick start](../../i4h-sensor-simulation/ultrasound-raytracing/docs/quick_start.md)
2. `conda activate ultrasound` (recommended)
3. Run from monorepo root so `instrument-calibration/p035_visions/` resolves

## CT-pullback bridge

`simulated_pullback_from_CT/src/sim_runner.py` accepts
`vessel_manifest_path=...` and loads manifest meshes using the same
material-chain convention. Useful for combining CT-derived pullbacks with
procedurally generated side branches.

## Troubleshooting

| Issue | Check |
|-------|-------|
| Flat wall echo | Vessel uses single-slab wall; regenerate with `--layers 3` |
| Missing intima label in seg | Ensure trilaminar manifest (post June 2026 world model) |
| Import errors for `tier1_evaluation` | Run from repo root; `p035_visions/` must be on path |
| GPU OOM at scale | Use `--skip-overlay`; reduce `--frames-per-vessel` |

