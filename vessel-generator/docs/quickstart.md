# Quick start

Get from a fresh clone to a realistic IVUS frame in three steps. This guide
covers two workflows:

1. **Single vessel** — generate one anatomy folder for inspection or custom
   integration.
2. **Paired dataset** — generate many vessels and render B-mode + segmentation
   frames for ML training.

For architecture details, see [Pipeline overview](pipeline-overview.md).

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Used by `vesselgen` and the simulator bindings |
| `pip install -e vessel-generator` | Run from the **monorepo root** |
| Built raysim CUDA extension | Required only for IVUS rendering (path 2) |
| `conda activate ultrasound` | Recommended env for raysim (see [simulator quick start](../../i4h-sensor-simulation/ultrasound-raytracing/docs/quick_start.md)) |
| NVIDIA GPU | Required for IVUS rendering |

Geometry-only steps (path 1) do not need a GPU or raysim.

## Calibrated probe parameters

Realistic IVUS rendering uses the **PV .035 / Volcano s5i** calibration
shipped in this repo:

```
instrument-calibration/p035_visions/volcano_s5i.yaml
```

The paired-dataset renderer loads this file via
`load_calibrated_config()` and applies small per-frame randomization around
that operating point (see [Simulator integration](simulator-integration.md)).

**Note on naming:** this repo uses the `p035_visions` calibration folder
(PV .035 catheter). Older references to `p034` point to the same instrument
family; treat `p035_visions/volcano_s5i.yaml` as canonical.

## Path 1 — Single vessel (geometry only)

Generate one peripheral-scale vessel with a trilaminar wall (the minimum
needed for the bright–dark–bright IVUS wall signature):

```bash
# From monorepo root
pip install -e vessel-generator

vesselgen-vessel \
  --out vessel-generator/out/my_vessel \
  --seed 1 \
  --layers 3 \
  --guidewire \
  --n-lesions 1 \
  --lesion-kinds hard
```

Inspect the output:

```
vessel-generator/out/my_vessel/
  lumen.obj
  surfaces/interface_01.obj
  surfaces/interface_02.obj
  lesions/lesion_00_hard.obj      # if --n-lesions > 0
  guidewire.obj                   # if --guidewire
  preview.png                     # QC cross-section gallery
  vessel.json                     # manifest for the simulator
  branches/branch_00.json
```

Optional: sample geometric ground-truth poses (JSON only, no B-mode):

```bash
vesselgen-frames \
  --vessel vessel-generator/out/my_vessel \
  --out vessel-generator/out/my_vessel/frames \
  --n 10 \
  --write-previews
```

See [CLI reference](cli-reference.md) for all `vesselgen-vessel` flags.

## Path 2 — Paired IVUS dataset (geometry + rendering)

This is the end-to-end training-data path. One command generates varied
vessels, samples probe poses, renders B-mode images with raysim, and writes
acoustic-boundary segmentation masks.

```bash
# From monorepo root, with raysim built and conda env active
conda activate ultrasound
pip install -e vessel-generator

python vessel-generator/examples/render_paired_dataset.py \
  --out vessel-generator/out/paired_smoke \
  --n 4 \
  --frames-per-vessel 1 \
  --seed 42
```

Open the first frame:

```
vessel-generator/out/paired_smoke/frames/frame_00000/
  image.png           # polar B-mode display
  segmentation.png    # per-pixel label mask
  overlay.png         # GT contours on B-mode (skip with --skip-overlay)
  image.npy
  segmentation.npy
  metadata.json
```

Scale up when the smoke test looks right:

```bash
python vessel-generator/examples/render_paired_dataset.py \
  --out vessel-generator/out/paired_dataset_100 \
  --n 100 \
  --frames-per-vessel 12 \
  --skip-overlay \
  --resume          # continue an interrupted run
```

### Geometry-only batch (no rendering)

If you only need vessel folders (no IVUS frames), use the dataset CLI:

```bash
vesselgen-dataset \
  --n 64 \
  --out vessel-generator/out/dataset \
  --base-seed 0 \
  --side-branch-probability 0.45
```

Feed the resulting folders into your own simulator driver, or switch to
path 2 above for the built-in paired renderer.

## Choosing a path

| Goal | Command |
|------|---------|
| Inspect one anatomy, tune parameters | `vesselgen-vessel` |
| Batch vessel folders for a custom pipeline | `vesselgen-dataset` |
| Sample pose + geometric GT JSON from a saved vessel | `vesselgen-frames` |
| Full paired B-mode + segmentation dataset | `render_paired_dataset.py` |

## Realism defaults

- **`vesselgen-vessel` defaults to `--layers 1`** (single-slab wall). Pass
  `--layers 3` for trilaminar anatomy.
- **`vesselgen-dataset`** draws layered walls ~70% of the time via
  `GenerationConfig.layered_wall_probability`.
- **`render_paired_dataset.py`** pins trilaminar walls for all vessels
  (see [Configuration](configuration.md)).

Lesions and guidewire are mutually exclusive with side branches in the
single-vessel CLI (`--n-lesions` requires no `--side-branch`). The batch
sampler applies the same constraint automatically.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ImportError: raysim` | CUDA extension not built | Follow [simulator quick start](../../i4h-sensor-simulation/ultrasound-raytracing/docs/quick_start.md) |
| `BifurcationError` during batch gen | Rare boolean-union failure at ostium | Re-run with a different seed; batch code retries automatically |
| Flat wall echo (no bright–dark–bright) | Single-slab wall (`--layers 1`) | Use `--layers 3` or the paired renderer |
| `--preset` not found | Removed from docs; not implemented | Use explicit CLI flags (see [CLI reference](cli-reference.md)) |

## Next steps

- [Configuration](configuration.md) — anatomy and batch sampling knobs
- [On-disk format](on-disk-format.md) — `vessel.json` manifest schema
- [Segmentation labels](segmentation-labels.md) — label IDs and legacy collapse
- [Simulator integration](simulator-integration.md) — calibration, materials, randomization
