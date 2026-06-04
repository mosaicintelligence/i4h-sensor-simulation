# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Production Dockerfile for the ultrasound-raytracing simulator.
#
# Builds the C++/CUDA simulator and installs the `raysim` Python package so the
# image is fully self-contained. The default ENTRYPOINT is the scene runner
# (`raysim-run-scene`); pass `--scene path.json --output out.npy` to render.
#
# Build (from the ultrasound-raytracing/ directory):
#   docker build -f docker/Dockerfile -t raysim:latest .
#
# Run (requires NVIDIA Container Toolkit / CDI):
#   docker run --rm --gpus all \
#     -v "$PWD":/work -w /work \
#     raysim:latest --scene scene.json --output out.npy
#
# Multi-arch CUDA build: defaults to 70;80;86;89;90 (V100 → H100) so the image
# works on the common data-center GPUs without needing a GPU at build time.
# Override with `--build-arg CUDA_ARCH=...` to slim down or target a specific
# device.

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=off \
    PYTHONDONTWRITEBYTECODE=1

ARG CUDA_ARCH="70;80;86;89;90"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        ninja-build \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /opt/raysim

# Copy only the files required to build the simulator. We deliberately exclude
# build artefacts, large bench data, and the calibration/evaluation tree so
# the image stays focused on running the simulator.
COPY CMakeLists.txt pyproject.toml README.md ./
COPY cmake/        ./cmake/
COPY csrc/         ./csrc/
COPY include/      ./include/
COPY raysim/       ./raysim/

# Create a venv inside the image so the user's host Python version is irrelevant.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel setuptools

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# Build/install the simulator. Pin CUDA archs for portability and disable C++
# examples (the Python entrypoint is what users invoke).
RUN CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
    pip install --no-cache-dir ".[all]"

# Bundle the scene runner and bundled meshes so users can render IVUS without
# mounting any host data.
COPY docker/scene_runner.py /opt/raysim/scene_runner.py
COPY mesh/ /opt/raysim/mesh/

# ----------------------------------------------------------------------------
# Runtime stage. Slim image with only what's needed to run the simulator.
# ----------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgomp1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/raysim /opt/raysim

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    RAYSIM_BUNDLED_MESHES=/opt/raysim/mesh \
    RAYSIM_INSTALL_DIR=/opt/raysim

WORKDIR /work

# Default entrypoint: scene runner. Override with `--entrypoint python` (etc.)
# for ad-hoc inspection. `--help` prints the CLI usage.
ENTRYPOINT ["python", "/opt/raysim/scene_runner.py"]
CMD ["--help"]
