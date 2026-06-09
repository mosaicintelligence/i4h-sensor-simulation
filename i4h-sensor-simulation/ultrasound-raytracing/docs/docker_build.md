# Running the simulator via Docker

The raysim repository ships **two** Dockerfiles for two different workflows:

| File | Purpose |
|------|---------|
| `.devcontainer/Dockerfile` | Interactive dev container (VS Code Remote-Containers / `docker run -it`). Source lives on the host and is mounted in; you build inside. |
| `docker/Dockerfile` | Production image. Builds the C++/CUDA simulator and installs `raysim` so `docker run raysim:latest --scene ...` just renders. |

Most users want the production image. The rest of this document covers it.

---

## Prerequisites

- NVIDIA GPU with driver **555+** (OptiX 8.1 requires it)
- Docker Engine ≥ 20.10
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so `docker run --gpus all` works:

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

  Verify with:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
  ```

---

## 1. Build the image

From `ultrasound-raytracing/`:

```bash
docker build -f docker/Dockerfile -t raysim:latest .
```

The default build targets CUDA archs `70;80;86;89;90` (V100 → H100) so the
image is portable across the common data-center GPUs. To slim things down,
override the arch list:

```bash
docker build -f docker/Dockerfile -t raysim:latest \
  --build-arg CUDA_ARCH="89" .          # only sm_89 (RTX 6000 Ada / L40)
```

The build pulls only the simulator source, mesh assets, and Python package
into the image (see `.dockerignore`). The first build takes ~8–15 minutes;
subsequent builds are cached.

---

## 2. Smoke-test the image

The image's default entrypoint is the scene runner:

```bash
docker run --rm --gpus all raysim:latest --help
```

To render a single IVUS frame from the bundled cylinder mesh, create a
minimal scene file in a host directory:

```bash
mkdir -p ivus_demo
cat > ivus_demo/scene.json <<'JSON'
{
  "world": {
    "background_material": "lumen",
    "objects": [
      {"type": "mesh", "path": "Cylinder_inner.obj", "material": "vessel_wall"},
      {"type": "mesh", "path": "Cylinder_outer.obj", "material": "extravascular"}
    ]
  },
  "mesh_search_paths": ["/opt/raysim/mesh"],
  "probe": {
    "type": "ivus",
    "frequency_mhz": 40.0,
    "num_scanlines": 256,
    "pose": {"position_mm": [0,0,0], "rotation_rad": [0,0,0]}
  },
  "sim_params": {"t_far_mm": 10.0, "buffer_size": 4096, "b_mode_size": [512, 512]}
}
JSON

docker run --rm --gpus all \
  -v "$PWD/ivus_demo":/work \
  raysim:latest --scene /work/scene.json --output /work/b_mode.npy
```

The container writes `ivus_demo/b_mode.npy` (and a `b_mode.npy.meta.json` with
the simulator's display extents) on the host.

---

## 3. Python interface (recommended)

`raysim.docker` is a host-side wrapper that lets you build scenes in Python
— exactly like you do with the native simulator — and dispatch the actual
render to the container. The host process never imports the CUDA/OptiX
extension, so you can use this from any machine with Docker and the NVIDIA
runtime.

```python
from raysim.docker import (
    DockerSimulator, MaterialOverride, MeshObject, Pose,
    ProbeSpec, Scene, SimParamsSpec, ensure_image,
)

# (Optional) build the image on first use.
ensure_image("raysim:latest")

# Describe the world.
scene = Scene(background_material="lumen",
              mesh_search_paths=["/work/mesh"])
scene.add(MeshObject(path="Cylinder_inner.obj", material="vessel_wall"))
scene.add(MeshObject(path="Cylinder_outer.obj", material="extravascular"))
scene.override_material(MaterialOverride(name="lumen", mu0=0.1))

# Configure the probe & simulator.
probe = ProbeSpec.ivus(pose=Pose(position_mm=(0, 0, 0)),
                       frequency_mhz=40.0, num_scanlines=256)
sim_params = SimParamsSpec(t_far_mm=10.0, b_mode_size=(512, 512))

# Render. The mesh dir is mounted into the container so meshes resolve.
sim = DockerSimulator(
    image="raysim:latest",
    mounts={"./mesh": "/work/mesh"},
)
b_mode = sim.simulate(scene, probe, sim_params,
                      output_dir="./docker_out", cleanup=False)
print(b_mode.shape)   # e.g. (512, 512)
```

For pullback / sweeps, pass an iterable of `Pose` objects via `frames=` and
the returned array has shape `(num_frames, h, w)`.

Two ready-to-run example scripts ship with the repo:

| Script | What it shows |
|---|---|
| [`examples/docker_quickstart.py`](../examples/docker_quickstart.py) | Minimal ~50-line "hello world" — single IVUS frame from the meshes bundled into the image. No host mounts required. |
| [`examples/docker_ivus_example.py`](../examples/docker_ivus_example.py) | Single frame **and** an N-frame pullback, optional material overrides, host-mounted mesh dir. Matches what `examples/ivus_example.py` does natively. |

Run them with:

```bash
# 1. Quickstart (uses meshes bundled in the image)
python examples/docker_quickstart.py --output-dir /tmp/docker_quickstart

# 2. IVUS example with pullback (mounts ./mesh from the repo)
python examples/docker_ivus_example.py --output-dir /tmp/docker_ivus_demo --pullback

# Add --use-sudo (or set RAYSIM_DOCKER_USE_SUDO=1) if your user is not in
# the `docker` group.
```

Each writes:
- `b_mode.npy` (or `b_mode_stack.npy` for pullback) — float32 array of shape `(H, W)` or `(num_frames, H, W)`.
- `b_mode.npy.meta.json` — display-extent metadata (`min_x`, `max_x`, `min_z`, `max_z`).
- `b_mode.png` — matplotlib rendering for quick inspection.

### Common kwargs

| `DockerSimulator(...)` arg | Meaning |
|---|---|
| `image` | Image tag to invoke (default `raysim:latest`). |
| `mounts` | `{host_path: container_path}` for mesh data, calibration YAMLs, etc. The scene's working dir is always mounted as `/work`. |
| `gpus` | Forwarded to `docker run --gpus`; default `"all"`. Set to `None` to skip the flag (e.g. for CPU smoke tests). |
| `use_sudo` | Prepend `sudo` to docker calls (use when the invoking user isn't in the `docker` group). |
| `run_as_host_user` | When True (default), pass `--user $(id -u):$(id -g)` so outputs are owned by the caller. Set to False if the image needs root. |
| `extra_run_args` | Extra `docker run` flags (e.g. `["--shm-size=2g"]`). |
| `env` | Environment variables to forward into the container. |

### Starting from a calibrated YAML

Both the CLI and the Python API can start from the calibrated PV .035 /
Volcano s5i YAML (`instrument-calibration/p035_visions/volcano_s5i.yaml`) and
override individual fields. Mount the calibration directory and reference it
through `SimParamsSpec.config_yaml`:

```python
sim = DockerSimulator(
    image="raysim:latest",
    mounts={"./instrument-calibration": "/work/instrument-calibration"},
)
sim_params = SimParamsSpec(
    config_yaml="/work/instrument-calibration/p035_visions/volcano_s5i.yaml",
    t_far_mm=8.0,        # override one field
)
```

The runner inside the container loads the YAML via `raysim.config.IvusSimConfig`
and applies your overrides on top.

---

## 4. Dev container workflow

For interactive development (`docker exec`-ing in, rebuilding the C++ tree
on every change, running examples ad-hoc), use the dev container:

```bash
cd ultrasound-raytracing
docker build \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -f .devcontainer/Dockerfile \
  -t raysim-dev .

docker run -it --rm --gpus all --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$(pwd)":/raysim \
  --workdir /raysim \
  raysim-dev
```

Inside the container, build and run as on bare metal:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[all]
python examples/ivus_example.py --single --output-dir /tmp/ivus_demo
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `failed to discover GPU vendor from CDI: no known GPU vendor found` | NVIDIA Container Toolkit not installed or daemon not restarted. | Install the toolkit (see Prerequisites) and `sudo systemctl restart docker`. |
| `permission denied while trying to connect to the docker API` | User not in `docker` group. | `sudo usermod -aG docker $USER && newgrp docker`, or pass `use_sudo=True` to `DockerSimulator`. |
| `OPTIX_ERROR_LIBRARY_NOT_FOUND` inside the container | The host driver doesn't ship `libnvoptix.so.1`. | Update the NVIDIA driver to 555+ (OptiX 8.1 ships with it) or follow `docs/vm_setup.md`. |
| Mesh files not found | The host path isn't mounted, or the path in `scene.json` isn't relative to a `mesh_search_paths` entry. | Add the host dir to `DockerSimulator(mounts=...)` and put the matching container path in `scene.mesh_search_paths`. |
| Build is slow on first run | CUDA arch list is wide (`70;80;86;89;90`). | Build with `--build-arg CUDA_ARCH=89` (or your target SM). |
