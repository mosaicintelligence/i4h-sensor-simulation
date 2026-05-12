# Raytracing Ultrasound Simulator

A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX raytracing with Python bindings.

## Features

- GPU acceleration with CUDA and NVIDIA OptiX
- Python interface for ease of use
- Real-time simulation capabilities
- Support for curvilinear, linear, and phased array ultrasound probe simulation

## Benchmark Results
To reproduce these results, run `python examples/benchmark.py`.
```
Benchmark Results:
        Total frames: 200
        Average frame time: 0.0073 seconds
        Average FPS: 136.28
        Minimum FPS: 59.66
        Maximum FPS: 249.62
        Date: 2025-03-16 07:38:46

        System Information:
        GPU: NVIDIA RTX 6000 Ada Generation (48.0 GB, Driver: 565.57.01)
        CPU: AMD Ryzen Threadripper PRO 7975WX 32-Cores (64 cores)

```
## Requirements

- [CUDA 12.6+](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#)
- [NVIDIA Driver 555+](https://www.nvidia.com/en-us/drivers/)
- [CMake 3.24+](https://cmake.org/)
- [NVIDIA OptiX SDK 8.1](https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Docker Installation

Instructions to build and run the examples in a docker environment can be found in the [`docs/docker_build`](docs/docker_build.md).

## Bare-Metal Installation

1. Clone this repository and open the **ultrasound-raytracing** package directory (the folder that contains this `README.md` and `pyproject.toml`):

   ```bash
   git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
   cd i4h-sensor-simulation/ultrasound-raytracing
   ```

   Some repository layouts nest that folder once more (e.g. `i4h-sensor-simulation/i4h-sensor-simulation/ultrasound-raytracing`); use the path where `pyproject.toml` is present.

2. The required OptiX header files are fetched automatically by CMake from the
   open-source [optix-dev](https://github.com/NVIDIA/optix-dev) repository during the
   configure step.  All you need is a recent NVIDIA driver (OptiX is part of the
   driver) and CUDA ≥ 12.6 which are already listed in the requirements section.

3. Install Python dependencies and create virtual environment:

   > Note: The `pip install -e .[all]` command will automatically run the CMake build process. This requires:
   > - CUDA toolkit with `nvcc` in your `$PATH`
   > - CMake 3.24.0 or higher (install with `pip install cmake==3.24.0` if needed)
   > - CUDA paths properly set:
   >   ```bash
   >   export PATH=/usr/local/cuda/bin:$PATH
   >   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   >   ```

   **Option A: Using uv**
   ```bash
   uv sync && source .venv/bin/activate
   uv pip install -e .[all]
   ```

   **Option B: Using conda**
   ```bash
   # Create environment and install dependencies
   conda create -n ultrasound python=3.10 libstdcxx-ng -c conda-forge -y
   conda activate ultrasound
   pip install -e .[all]
   ```

   After installation, use the **same** `python` you used for `pip install -e` when running examples. Rebuilding with `pip install -e .` removes any leftover `raysim/ray_sim_python*.so` from another interpreter before linking, so editable installs stay aligned with your Python version.

3b. **IVUS example meshes (included):** `examples/ivus_example.py` expects Wavefront meshes under `mesh/`. This repository ships `mesh/Cylinder.obj` plus `mesh/Cylinder_inner.obj` / `mesh/Cylinder_outer.obj` for the thick-wall option. To regenerate them (same defaults as the bundled files):

   ```bash
   python utils/phantom_maker.py cylinder --output mesh
   python utils/phantom_maker.py cylinder --output mesh --cylinder-thick
   ```

4. Download mesh data:
   ```bash
   cd ultrasound-raytracing  # ensure you're in the correct directory
   pip install git+https://github.com/isaac-for-healthcare/i4h-asset-catalog.git
   # Download mesh data (~527MB)
   i4h-asset-retrieve --download-dir assets --sub-path Props/ABDPhantom/Organs --version 0.2.0
   # Note: The hash below is specific to version 0.2.0
   ln -s assets/8c0bf782eab2f44f1cc82da60eb10f6be8f941406d291b7fbfbdb53c05b3d149/Props/ABDPhantom/Organs mesh
   ```

   > **Troubleshooting Build Issues:**
   >
   > - In the [CMake setup file](./cmake/SetupCUDA.cmake), the default value for `CMAKE_CUDA_ARCHITECTURES` is set to `native`. This setting **may cause compilation failures** on systems with multiple NVIDIA GPUs that have different compute capabilities.
   >
   > - If you experience this issue, try specifying the GPU you want to use by setting the environment variable `export CUDA_VISIBLE_DEVICES=<selected device number>` before running `pip install`.
   >
   > - **CMake generator mismatch** (`Does not match the generator used previously`): remove the build directory and reinstall, e.g. `rm -rf build-release` then `pip install -e .` again (use the same generator / toolchain as your first successful configure, or delete `build-release` whenever you switch tools).
   >
   > - **`ImportError` / undefined Python symbols when importing `raysim`:** Usually a stale extension built for a different Python. Run `pip install -e .` again from this directory using the interpreter you run examples with; the build step clears old `ray_sim_python*.so` files before linking.

5. Run examples

   **Using uv**
   ```bash
   # Basic example
   uv run examples/sphere_sweep.py

   # Web interface (open http://localhost:8000 afterward)
   uv run examples/server.py
   ```

   **Using conda**
   ```bash
   # Using the system's libstdc++ with LD_PRELOAD if your conda environment's version is too old
   python examples/sphere_sweep.py

   # Web interface
   python examples/server.py
   ```

   **C++ example**

   To build and run C++ examples, you can manually run CMake with `BUILD_EXAMPLES=ON`:
   ```bash
   cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -B build-release
   cmake --build build-release -j $(nproc)
   ./build-release/examples/cpp/ray_sim_example
   ```


## Start Simulating

For a comprehensive guide on using the simulator, understanding its features, and exploring advanced topics, please refer to our documentation:

- **[Getting Started Guide](../docs/ultrasound_simulator_getting_started.md)**: A step-by-step tutorial for beginners.
- **[Technical Guide](../docs/ultrasound_simulator_technical_guide.md)**: An in-depth look at the physics and implementation details.
