# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Minimal "hello world" for running raysim inside Docker.

This script never imports the native ``raysim.cuda`` extension; the actual
simulation runs inside the ``raysim:latest`` container. The host process
only needs:

    pip install numpy matplotlib   # raysim.docker has no other deps

Prereqs:
    1. Build the image once: ``docker build -f docker/Dockerfile -t raysim:latest .``
       (or call ``ensure_image()`` from this script with ``--build``).
    2. The NVIDIA Container Toolkit must be installed so ``--gpus all`` works.

Run::

    python examples/docker_quickstart.py                              # default
    python examples/docker_quickstart.py --build                      # build image first
    python examples/docker_quickstart.py --use-sudo                   # if not in docker group
    python examples/docker_quickstart.py --output-dir /tmp/quickstart
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make raysim.docker importable when running this file directly out of a source
# checkout (e.g. before `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from raysim.docker import (  # noqa: E402
    DockerSimulator,
    MeshObject,
    Pose,
    ProbeSpec,
    Scene,
    SimParamsSpec,
    SphereObject,  # noqa: F401 — referenced in the in-code comment about sphere-only scenes
    ensure_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default="raysim:latest",
                        help="Docker image tag to invoke (default: raysim:latest).")
    parser.add_argument("--output-dir", type=Path, default=Path("./docker_quickstart_out"),
                        help="Where to write the rendered B-mode .npy / .png.")
    parser.add_argument("--build", action="store_true",
                        help="Build the image first if it is not present.")
    parser.add_argument("--use-sudo", action="store_true",
                        default=bool(os.environ.get("RAYSIM_DOCKER_USE_SUDO")),
                        help="Prepend `sudo` to all docker commands.")
    args = parser.parse_args()

    if args.build:
        ensure_image(
            args.image,
            dockerfile_dir=Path(__file__).resolve().parents[1],
            dockerfile="docker/Dockerfile",
            use_sudo=args.use_sudo,
        )

    # ----------------------------------------------------------------
    # 1. Describe the world. The container already ships with the bundled
    #    IVUS meshes at /opt/raysim/mesh, so we can reference them without
    #    mounting anything from the host.
    #
    # NOTE: OptiX requires every object in a World to be the same primitive
    # type (either all meshes or all spheres), so don't mix MeshObject and
    # SphereObject in one Scene. Comment out the meshes and add a few
    # SphereObject(...) entries instead to render a sphere-only phantom.
    # ----------------------------------------------------------------
    scene = Scene(
        background_material="lumen",
        mesh_search_paths=["/opt/raysim/mesh"],
    )
    scene.add(MeshObject(path="Cylinder_inner.obj", material="vessel_wall"))
    scene.add(MeshObject(path="Cylinder_outer.obj", material="extravascular"))

    # ----------------------------------------------------------------
    # 2. Configure the probe and simulator the same way you would when
    #    running natively.
    # ----------------------------------------------------------------
    probe = ProbeSpec.ivus(
        pose=Pose(position_mm=(0.0, 0.0, 0.0)),
        frequency_mhz=40.0,
        num_scanlines=256,
    )
    sim_params = SimParamsSpec(t_far_mm=10.0, b_mode_size=(512, 512))

    # ----------------------------------------------------------------
    # 3. Dispatch to the container. No --mounts needed here because the
    #    meshes are bundled into the image.
    # ----------------------------------------------------------------
    sim = DockerSimulator(image=args.image, use_sudo=args.use_sudo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b_mode = sim.simulate(
        scene, probe, sim_params,
        output_dir=args.output_dir, output_name="b_mode.npy",
        cleanup=False,
    )

    # ----------------------------------------------------------------
    # 4. Render a PNG so you can eyeball the result.
    # ----------------------------------------------------------------
    norm = np.clip((b_mode - (-60.0)) / (0.0 - (-60.0)), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="gray", aspect="auto")
    ax.set_title(f"IVUS B-mode via Docker — {b_mode.shape[0]}×{b_mode.shape[1]} (intensity dB-normalized)")
    ax.set_xlabel("Angle bin")
    ax.set_ylabel("Depth sample")
    png_path = args.output_dir / "b_mode.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)

    print(f"\nRendered shape: {b_mode.shape}  dtype: {b_mode.dtype}")
    print(f"Outputs:")
    for p in sorted(args.output_dir.iterdir()):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
