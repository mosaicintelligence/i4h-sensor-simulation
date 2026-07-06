# vessel-generator

Procedural vessel geometry for **IVUS training data**. Generates realistic,
varied vessel volumes with per-frame ground truth, then renders paired
B-mode + segmentation frames through the calibrated raysim simulator.

## Quick start

**First realistic IVUS frame:** [docs/quickstart.md](docs/quickstart.md)

```bash
# From monorepo root — geometry only (no GPU)
pip install -e vessel-generator
vesselgen-vessel --out vessel-generator/out/smoke --seed 1 --layers 3 --guidewire

# End-to-end paired IVUS dataset (requires built raysim + GPU)
conda activate ultrasound
python vessel-generator/examples/render_paired_dataset.py \
  --out vessel-generator/out/paired_smoke --n 4 --frames-per-vessel 1
```

## Two workflows

| Workflow | Command | Output |
|----------|---------|--------|
| **Single vessel** | `vesselgen-vessel` | One anatomy folder (OBJ + manifest) |
| **Vessel batch** | `vesselgen-dataset` | N anatomy folders |
| **Pose + GT sampling** | `vesselgen-frames` | JSON pose/GT records (no B-mode) |
| **Paired IVUS dataset** | `render_paired_dataset.py` | B-mode + segmentation frames |

See [Pipeline overview](docs/pipeline-overview.md) for how these connect.

## What's modelled

- Non-circular lumen contours with smooth Fourier perturbation
- Variable, eccentric wall thickness
- Trilaminar intima/media/adventitia (bright–dark–bright IVUS wall)
- In-wall lesions (calcified, lipid, fibrous, thrombus)
- Optional tungsten guidewire with acoustic shadow
- Side-branch bifurcations via boolean union
- Frame-level pose sampling with geometric ground truth

Anatomical scale targets **PV .035 peripheral** vessels (8–13 mm lumen
diameter). Calibration reference:
[instrument-calibration/p035_visions/volcano_s5i.yaml](../instrument-calibration/p035_visions/volcano_s5i.yaml).

## Install

```bash
pip install -e vessel-generator
pytest vessel-generator/tests/
```

IVUS rendering additionally requires the built raysim CUDA extension — see
[simulator quick start](../i4h-sensor-simulation/ultrasound-raytracing/docs/quick_start.md).

## Documentation

| Doc | Contents |
|-----|----------|
| [Quick start](docs/quickstart.md) | First IVUS frame in minutes |
| [CLI reference](docs/cli-reference.md) | All command-line flags |
| [Pipeline overview](docs/pipeline-overview.md) | Geometry → rendering flow |
| [Configuration](docs/configuration.md) | Anatomy and batch sampling knobs |
| [On-disk format](docs/on-disk-format.md) | `vessel.json` manifest schema |
| [Segmentation labels](docs/segmentation-labels.md) | Label IDs and legacy collapse |
| [Simulator integration](docs/simulator-integration.md) | Calibration, materials, randomization |
| [Design notes](docs/design.md) | Coordinate conventions, algorithms, limitations |

## Layout

```
vessel-generator/
  vesselgen/              # importable package
    config.py             # generation parameter dataclasses
    vessel.py             # Vessel class (build, sample, GT)
    io.py                 # mesh + manifest IO
    library.py            # batch generation
    labels.py             # segmentation label IDs
    sim_randomization.py  # per-frame simulator diversity
    tools/                # CLIs (vesselgen-vessel, -dataset, -frames)
  examples/
    render_paired_dataset.py   # end-to-end IVUS dataset builder
  docs/                   # user documentation
  tests/
```

Output directories under `out/` are regenerable and not part of the source
tree. Generate your own with the commands above.

## Sampling API (Python)

```python
from vesselgen import Vessel
import numpy as np

v = Vessel.load("out/vessel_0001")
rng = np.random.default_rng(0)

pose = v.sample_pose(rng, max_tilt_deg=15.0, edge_margin_mm=0.2)
gt = v.ground_truth_at(pose)
# pose.position, pose.rotation_euler_deg → rs.IVUSProbe
# gt.lumen_contour_polygons, gt.distance_to_lumen_wall_mm, ...
```

The simulator can render an unbounded number of distinct frames per vessel
by calling `sample_pose` repeatedly — there is no pullback path.
