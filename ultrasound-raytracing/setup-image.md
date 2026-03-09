# Cloud Environment Setup (Google Cloud)

This guide walks through creating a new Google Cloud VM and installing the raytracing ultrasound simulator so you can run `python examples/sphere_sweep.py`. It uses **Option A**: a base Ubuntu image with the **desktop** NVIDIA driver from Ubuntu’s repository, which is more likely to include the OptiX runtime (`libnvoptix.so.1`) than the “server” driver used on many GPU DL images.

## Prerequisites

- A Google Cloud project with billing enabled
- `gcloud` CLI installed and configured (`gcloud auth login`, `gcloud config set project YOUR_PROJECT_ID`)
- Sufficient quota for the GPU type in your chosen region

## 1. Create the VM

Create an instance with a GPU, Ubuntu 22.04, and a 100 GB boot disk. Adjust `INSTANCE_NAME`, `ZONE`, and `ACCELERATOR` if needed (e.g. `nvidia-tesla-t4` for T4).

```bash
INSTANCE_NAME=raysim-vm
ZONE=us-central1-a
# g2-standard-8 uses NVIDIA L4; use nvidia-tesla-t4 for T4, etc.
ACCELERATOR=type=nvidia-l4,count=1

gcloud compute instances create "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --accelerator="$ACCELERATOR"
```

**Driver and OptiX 8.1:** Do **not** add `--metadata=install-nvidia-driver=True`. Google's automatic driver install typically provides a minimal/server stack that does **not** include the OptiX runtime (`libnvoptix.so.1`), so you would still get `OPTIX_ERROR_LIBRARY_NOT_FOUND`. OptiX 8.1 requires driver **R555 or newer**. This guide uses the **manual** desktop driver install in section 3 (Option 3b) so the OptiX runtime is present.

Wait for the instance to be running, then SSH in:

```bash
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE"
```

## 2. System updates and kernel headers

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) build-essential
```

## 3. NVIDIA driver (desktop variant for OptiX)

The **desktop** driver package from Ubuntu is more likely to include the OptiX runtime than the server driver. Install it (and reboot) unless you used automatic driver install and already have a working driver.
Install the desktop driver (R555+ required for OptiX 8.1) and reboot:

```bash
sudo apt-get install -y nvidia-driver-570
sudo reboot
```

After reboot, SSH in again and confirm:

```bash
nvidia-smi
find /usr -name "libnvoptix*" 2>/dev/null
```

If `find` shows a path (e.g. `/usr/lib/x86_64-linux-gnu/libnvoptix.so.1`), the OptiX runtime is present.

## 4. CUDA toolkit

The project expects CUDA 12.6+ and `nvcc` in your `PATH`. Install the CUDA toolkit (e.g. 12.6) from NVIDIA’s repo or use the version that matches your driver.

**Using NVIDIA repository (Ubuntu 22.04):**

```bash
# Install repo package (adjust URL for your CUDA version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Match your driver: nvidia-smi shows CUDA 12.8 → use 12-8; 12-6 or 12-7 also work
sudo apt-get install -y cuda-toolkit-12-8
```

Add CUDA to your path for the current session and for future logins:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

## 5. Miniconda (recommended)

```bash
mkdir -p ~/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 -f
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 6. Clone the repo and create the conda environment

```bash
cd ~
git clone https://github.com/isaac-for-healthcare/i4h-sensor-simulation.git
cd i4h-sensor-simulation/ultrasound-raytracing
```

Create the environment and install build dependencies, then the project:

```bash
conda create -n ultrasound python=3.10 -y
conda activate ultrasound

pip install "packaging>=23.2" "scikit-build-core>=0.10" pybind11 "cmake>=3.24"
pip install -e .[all] --no-build-isolation
```

If you see `BackendUnavailable: Cannot import 'scikit_build_core.build'`, run the two `pip install` lines again in the same order.

## 7. OptiX library path (for conda)

So the loader can find the OptiX runtime, set `LD_LIBRARY_PATH` before running Python:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

To make this permanent for the `ultrasound` env, add that line to `~/.bashrc` or run it each time after `conda activate ultrasound`.

## 8. Run the example

From the project root:

```bash
cd ~/i4h-sensor-simulation/ultrasound-raytracing
conda activate ultrasound
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
python examples/sphere_sweep.py
```

If you still see `OPTIX_ERROR_LIBRARY_NOT_FOUND`, run `find /usr -name "libnvoptix*"` and add the directory that contains `libnvoptix.so.1` to `LD_LIBRARY_PATH`. If no path is found, the driver stack on this image does not include the OptiX runtime; see the main [README](../README.md) section “OptiX runtime: OPTIX_ERROR_LIBRARY_NOT_FOUND” for other options (e.g. different image or full NVIDIA driver install).

## Summary checklist

| Step | Action |
|------|--------|
| 1 | Create GCP VM with GPU, Ubuntu 22.04, 100 GB disk |
| 2 | `sudo apt install linux-headers-$(uname -r) build-essential` |
| 3 | Install NVIDIA driver: `sudo apt install nvidia-driver-570` (manual, for OptiX 8.1), reboot; confirm `nvidia-smi` and `find /usr -name "libnvoptix*"` |
| 4 | Install CUDA toolkit 12.x, add to `PATH` and `LD_LIBRARY_PATH` |
| 5 | `sudo apt install python3.10 python3.10-dev` |
| 6 | Install Miniconda, `conda init`, `source ~/.bashrc` |
| 7 | Clone repo, `conda create -n ultrasound python=3.10`, activate, `pip install` build deps then `pip install -e .[all] --no-build-isolation` |
| 8 | `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}` (and optionally add to `~/.bashrc`) |
| 9 | `python examples/sphere_sweep.py` |

## References

- Main [README](../README.md) – requirements, bare-metal install, troubleshooting
- [OptiX runtime: OPTIX_ERROR_LIBRARY_NOT_FOUND](../README.md#optix-runtime-optix_error_library_not_found-7804) – if the library is still not found
- [Docker build](docker_build.md) – alternative using Docker

