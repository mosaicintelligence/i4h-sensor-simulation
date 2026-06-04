# Quick Start

Get your first ultrasound simulation rendered in **under five minutes** on a machine that already has the simulator installed.

> 🚧 **Don't have the simulator built yet?** Start with the [VM setup guide](vm_setup.md) (fresh cloud GPU) or the [bare-metal install in the README](../README.md#installation).
>
> 📖 **Want a deeper walk-through?** The [tutorial](../../docs/ultrasound_simulator_tutorial.md) builds up examples step by step. The [technical overview](../../docs/ultrasound_simulator_technical_guide.md) explains the physics and pipeline.

---

## 1. Verify the install

From `i4h-sensor-simulation/ultrasound-raytracing/`:

```bash
conda activate ultrasound          # or your environment of choice
python -c "import raysim; from raysim.cuda import RaytracingUltrasoundSimulator; print('raysim OK')"
```

If you see `raysim OK`, the extension is built against the active Python and OptiX is reachable.

---

## 2. Run a built-in IVUS example

```bash
python examples/ivus_example.py --single --output-dir /tmp/ivus_quickstart
```

This builds a thick-walled vessel phantom, places an IVUS probe at the lumen center, runs one frame, and writes:

- `/tmp/ivus_quickstart/ivus_frame.png` — unwrapped view (x = angle, y = depth)
- `/tmp/ivus_quickstart/ivus_frame_polar.png` — polar view (clinical IVUS layout)

The frame renders in well under a second on an L4-class GPU.

---

## 3. Build your own scene from scratch

The snippet below is the minimum-viable IVUS scene. Save it as `quickstart_ivus.py` and run with `python quickstart_ivus.py`.

```python
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import raysim.cuda as rs

materials = rs.Materials()

world = rs.World("water")
wall_mat = materials.get_index("vessel_wall")
mesh_dir = os.path.join(os.path.dirname(rs.__file__), "..", "mesh")
world.add(rs.Mesh(os.path.join(mesh_dir, "Cylinder_inner.obj"), wall_mat))
world.add(rs.Mesh(os.path.join(mesh_dir, "Cylinder_outer.obj"),
                  materials.get_index("extravascular")))

probe = rs.IVUSProbe(
    rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.]),
    num_angular_rays=256,
    frequency=40.0,
    f_num=1.0,
)

sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.buffer_size = 4096
sim_params.t_far = 10.0
sim_params.b_mode_size = (512, 512)

simulator = rs.RaytracingUltrasoundSimulator(world, materials)
b_mode = simulator.simulate(probe, sim_params)

plt.imshow(b_mode, cmap="gray", vmin=-60, vmax=0)
plt.xlabel("Angle bin"); plt.ylabel("Depth bin")
plt.savefig("ivus_quickstart.png", bbox_inches="tight", dpi=120)
print("Saved ivus_quickstart.png")
```

**What this gives you:** an unwrapped IVUS frame with a thick-walled vessel, calibrated materials, and the default IVUS PSF/TGC.

---

## 4. Use the calibrated PV .035 / Volcano s5i configuration

For a frame that matches the calibrated PV .035 (10 MHz peripheral IVUS) on a Volcano s5i, point the simulator at the bench-derived YAML:

```python
from raysim.config import IvusSimConfig

cfg = IvusSimConfig.from_yaml(
    "../../instrument-calibration/p035_visions/volcano_s5i.yaml"
)
sim_params = cfg.to_sim_params()
probe = cfg.build_probe(rs.Pose(position=[0., 0., 0.], rotation=[0., 0., 0.]))
materials = cfg.materials()
```

Everything downstream is identical. The YAML carries all calibrated knobs (gain, ring-down template, TGC schedule, noise σ, etc.) and is updated by the calibration scripts in `instrument-calibration/p035_visions/`.

---

## 5. Where to go next

- **Try other probe types and phantoms** — `examples/sphere_sweep.py`, `examples/liver_sweep.py`, `examples/cystic_resolution_phantom_evaluation.py`, `examples/wire_phantom_evaluation.py`.
- **Optimize for a new probe** — follow the calibration pipeline in [`instrument-calibration/p035_visions/README.md`](../../../instrument-calibration/p035_visions/README.md).
- **Evaluate fidelity** — run `tier1_evaluation.py` to regenerate the [Tier 1 report](../../../instrument-calibration/p035_visions/tier1_results/tier1_results.md).
- **Render clinical scenes** — `vessel_evaluation.py` exercises seven vessel scenarios and produces the [vessel evaluation report](../../../instrument-calibration/p035_visions/vessel_evaluation_output/VESSEL_EVALUATION_REPORT.md).
