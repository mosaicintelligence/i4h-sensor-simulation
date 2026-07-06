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

