# CLI reference

All commands assume you have run `pip install -e vessel-generator` from the
monorepo root.

## `vesselgen-vessel` — one vessel

Generate a single vessel folder with explicit parameters.

```bash
vesselgen-vessel --out OUT_DIR [options]
```

### Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | *(required)* | Output directory |
| `--seed` | `0` | Reproducibility seed |
| `--layers` | `1` | Wall layers: `1` single-slab, `2` media+adventitia, `3` trilaminar |
| `--radius-mm` | `5.0` | Mean lumen radius (~10 mm diameter) |
| `--wall-mm` | `0.85` | Total wall thickness |
| `--length-mm` | `55.0` | Parent segment length |
| `--side-branch` | off | Attach one side-branch bifurcation |
| `--guidewire` | off | Add tungsten guidewire |
| `--n-lesions` | `0` | In-wall lesions (requires no side branch) |
| `--lesion-kinds` | `hard` | Comma-separated: `hard`, `soft_lipid`, `fibrous`, `thrombus` |
| `--no-preview` | off | Skip `preview.png` |

### Examples

Straight trilaminar vessel with guidewire:

```bash
vesselgen-vessel --out out/vessel_straight --seed 1 --layers 3 --guidewire
```

Side-branch bifurcation (single-slab wall on branches):

```bash
vesselgen-vessel --out out/vessel_side --seed 7 --layers 3 --side-branch
```

## `vesselgen-dataset` — batch geometry

Generate N vessel folders with parameters drawn from `GenerationConfig`.

```bash
vesselgen-dataset --out OUT_DIR --n COUNT [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | *(required)* | Output directory |
| `--n` | *(required)* | Number of vessels |
| `--base-seed` | `0` | Seed for the batch |
| `--name-prefix` | `vessel` | Folder prefix (`vessel_0000`, …) |
| `--side-branch-probability` | `0.45` | Fraction of vessels with a side branch |
| `--no-previews` | off | Skip per-vessel `preview.png` |

## `vesselgen-frames` — pose + geometric GT

Sample probe poses and geometric ground truth from a saved vessel. Writes
JSON records only — **no B-mode images**.

```bash
vesselgen-frames --vessel VESSEL_DIR --out OUT_DIR [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vessel` | *(required)* | Path to a saved vessel folder |
| `--out` | *(required)* | Output directory for JSON frames |
| `--n` | `200` | Number of pose samples |
| `--max-tilt-deg` | `15.0` | Maximum probe-axis tilt |
| `--edge-margin-mm` | `0.2` | Minimum clearance from lumen wall |
| `--max-distance-mm` | `30.0` | Imaging FOV radius; A-lines beyond it are `NaN`. Set to `17.5` to reproduce large-vessel open sectors |
| `--write-previews` | off | Write per-frame contour PNGs (draws the FOV circle) |

## `render_paired_dataset.py` — paired IVUS dataset

End-to-end geometry + raysim rendering. Requires built raysim and GPU.

```bash
python vessel-generator/examples/render_paired_dataset.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `vessel-generator/out/paired_dataset_100` | Dataset root |
| `--n` | `100` | Total paired frames across all vessels |
| `--seed` | `42` | Base seed |
| `--frames-per-vessel` | `12` | Frames per vessel before moving on |
| `--max-tilt-deg` | `15.0` | Pose tilt limit |
| `--edge-margin-mm` | `0.15` | Wall clearance for pose sampling |
| `--require-side-branch` | off | Only emit bifurcation vessels |
| `--skip-overlay` | off | Skip overlay PNG (faster at scale) |
| `--resume` | off | Continue an interrupted run |
| `--regenerate-segmentations-only` | off | Recompute masks from saved B-mode |

See [Quick start](quickstart.md) for a minimal smoke-test command.
