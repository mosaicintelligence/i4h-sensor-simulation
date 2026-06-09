# Raytracing Ultrasound Simulator

A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX raytracing with Python bindings. Supports curvilinear, linear-array, phased-array, and IVUS probes; ships with a calibrated PV .035 / Volcano s5i configuration for IVUS.

## Documentation map

| If you want to… | Read |
|---|---|
| Get a first frame rendered in 5 minutes | [Quick Start](docs/quick_start.md) |
| Provision a fresh cloud GPU and install everything from scratch | [VM Setup (GCP)](docs/vm_setup.md) |
| Walk through 7+ progressively richer examples | [Tutorial](../docs/ultrasound_simulator_tutorial.md) |
| Understand the physics and pipeline | [Technical Guide](../docs/ultrasound_simulator_technical_guide.md) |
| Read the IVUS-specific implementation deep dive | [IVUS Implementation Writeup](docs/ivus_implementation_writeup.md) |
| Run the simulator via Docker (no local CUDA build) | [Docker Build & Python Interface](docs/docker_build.md) |
| Recalibrate the simulator for a new probe | [Calibration Pipeline](../../instrument-calibration/p035_visions/README.md) |

## Requirements

- NVIDIA GPU with CUDA support (RTX/L4/A10 class or better)
- NVIDIA Driver **555+** (required for OptiX 8.1)
- [CUDA 12.6+](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#)
- [CMake 3.24+](https://cmake.org/)
- Python 3.10
- [NVIDIA OptiX SDK 8.1](https://developer.nvidia.com/designworks/optix/downloads/legacy) (headers fetched automatically; runtime ships with the driver)

## Installation

### Option A: Provision a cloud GPU from scratch
Follow the step-by-step guide in [`docs/vm_setup.md`](docs/vm_setup.md). It covers GCP VM creation, NVIDIA driver, CUDA toolkit, conda, and the OptiX `LD_LIBRARY_PATH` workaround.

### Option B: Bare-metal install on an existing machine

1. **Clone and enter the simulator directory:**

   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
   cd i4h-sensor-simulation/ultrasound-raytracing
   ```

2. **Make sure CUDA is on your path:**

   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **Install the Python package (builds the C++/CUDA extension automatically):**

   ```bash
   # Conda
   conda create -n ultrasound python=3.10 libstdcxx-ng -c conda-forge -y
   conda activate ultrasound
   pip install -e .[all]

   # — or — uv
   uv sync && source .venv/bin/activate
   uv pip install -e .[all]
   ```

   > **Important:** the same Python interpreter you run `pip install -e` with is the one you must use to run examples. Editable rebuilds remove leftover `raysim/ray_sim_python*.so` from another interpreter before linking.

4. **Download the abdominal mesh assets (only needed for abdominal examples):**

   ```bash
   pip install git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git
   i4h-asset-retrieve --download-dir assets --sub-path Props/ABDPhantom/Organs --version 0.2.0
   ln -s assets/8c0bf782eab2f44f1cc82da60eb10f6be8f941406d291b7fbfbdb53c05b3d149/Props/ABDPhantom/Organs mesh
   ```

   IVUS examples use vessel meshes that already live in `mesh/` (`Cylinder.obj`, `Cylinder_inner.obj`, `Cylinder_outer.obj`). Regenerate them with `python utils/phantom_maker.py cylinder --output mesh [--cylinder-thick]`.

## Run an example

After installation:

```bash
# IVUS single frame (uses bundled cylinder mesh)
python examples/ivus_example.py --single --output-dir /tmp/ivus_demo

# Curvilinear sphere sweep
python examples/sphere_sweep.py

# Web viewer (open http://localhost:8000)
python examples/server.py
```

For a C++ example build:

```bash
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -B build-release
cmake --build build-release -j $(nproc)
./build-release/examples/cpp/ray_sim_example
```

## Troubleshooting

- **`OPTIX_ERROR_LIBRARY_NOT_FOUND`** — the loader cannot find `libnvoptix.so.1`. Run `find /usr -name "libnvoptix*"` and prepend the containing directory to `LD_LIBRARY_PATH`. On many "GPU server" GCE images the OptiX runtime is missing entirely; see [`docs/vm_setup.md`](docs/vm_setup.md) for the desktop-driver path that includes it.
- **`ImportError` / undefined Python symbols when importing `raysim`** — usually a stale extension built for a different Python. Re-run `pip install -e .[all]` from this directory with the interpreter you run examples with; the build step clears old `ray_sim_python*.so` files before linking.
- **CMake "generator does not match" error** — delete `build-release/` (or whichever build dir CMake is complaining about) and reinstall.
- **CMake builds for the wrong GPU architecture** — set `export CUDA_VISIBLE_DEVICES=<gpu_idx>` before `pip install` so `CMAKE_CUDA_ARCHITECTURES=native` resolves to a single compute capability. Multi-GPU systems with mixed SMs are the usual culprit.

## Benchmark

To reproduce the published throughput numbers, run `python examples/benchmark.py`. Reference numbers on an RTX 6000 Ada / Ryzen Threadripper PRO 7975WX:

```
Total frames: 200
Average frame time: 0.0073 s
Average FPS: 136.28
Date: 2025-03-16
```
