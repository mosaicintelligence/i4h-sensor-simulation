# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""IVUS example dispatched to the raysim Docker image.

Mirrors ``examples/ivus_example.py``, but the host process never imports the
native ``raysim.cuda`` extension. Scenes are described in pure Python via
``raysim.docker`` and rendered inside the ``raysim:latest`` container.

Prerequisites:
    1. Build the image once:
         docker build -f docker/Dockerfile -t raysim:latest .
       (or call ``raysim.docker.ensure_image('raysim:latest')`` from Python).
    2. Install the NVIDIA Container Toolkit so ``docker run --gpus all`` works.
    3. Optionally set ``RAYSIM_DOCKER_USE_SUDO=1`` if your user is not in the
       ``docker`` group.

Run:
    python examples/docker_ivus_example.py --output-dir /tmp/docker_ivus_demo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the host-side helpers importable when running this file directly out
# of a source checkout (e.g. before `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402  (matplotlib import after sys.path tweak)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from raysim.docker import (  # noqa: E402
    DockerSimulator,
    MaterialOverride,
    MeshObject,
    Pose,
    ProbeSpec,
    Scene,
    SimParamsSpec,
    ensure_image,
)


REPO_RAYTRACING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = REPO_RAYTRACING_DIR / "mesh"

MIN_VAL = -60.0
MAX_VAL = 0.0


def build_ivus_scene(mesh_dir: Path) -> Scene:
    """A vessel-like phantom: lumen background + thick cylinder wall.

    Uses the bundled ``mesh/Cylinder_inner.obj`` + ``mesh/Cylinder_outer.obj``
    meshes — when the inner surface is ``vessel_wall`` and the outer is
    ``extravascular`` a refracted ray inside the wall sees a wall->water
    transition and records the back-wall echo.
    """
    scene = Scene(background_material="lumen", mesh_search_paths=[str(mesh_dir)])
    scene.add(MeshObject(path="Cylinder_inner.obj", material="vessel_wall"))
    scene.add(MeshObject(path="Cylinder_outer.obj", material="extravascular"))
    return scene


def build_ivus_probe(position_mm: tuple[float, float, float]) -> ProbeSpec:
    return ProbeSpec.ivus(
        pose=Pose(position_mm=position_mm),
        frequency_mhz=40.0,
        num_scanlines=256,
        pulse_duration_cycles=2.0,
        f_num=1.0,
        speed_of_sound_mm_per_us=1.54,
        element_radius_mm=0.6,
        focal_length_mm=4.0,
        elevational_height_mm=0.0,
        num_elevational_samples=1,
    )


def build_sim_params() -> SimParamsSpec:
    return SimParamsSpec(
        t_far_mm=10.0,
        buffer_size=4096,
        b_mode_size=(512, 512),
        conv_psf=True,
    )


def save_unwrapped_frame(b_mode_image: np.ndarray, path: Path, title: str) -> None:
    """Match the on-host example's matplotlib layout for parity."""
    normalized = np.clip((b_mode_image - MIN_VAL) / (MAX_VAL - MIN_VAL), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(normalized, cmap="gray", aspect="auto")
    ax.set_xlabel("Angle (bin)")
    ax.set_ylabel("Depth (sample)")
    ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax, label="Intensity (normalized)")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render an IVUS frame via the raysim Docker image."
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("RAYSIM_DOCKER_IMAGE", "raysim:latest"),
        help="Container image to invoke (default: raysim:latest).",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="Host directory containing the cylinder meshes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docker_ivus_output"),
        help="Where to write the rendered .npy / .png.",
    )
    parser.add_argument(
        "--pullback",
        action="store_true",
        help="Render a 5-frame pullback in addition to the single frame.",
    )
    parser.add_argument(
        "--build-image",
        action="store_true",
        help="Build the image if it isn't already present locally.",
    )
    parser.add_argument(
        "--use-sudo",
        action="store_true",
        default=bool(os.environ.get("RAYSIM_DOCKER_USE_SUDO")),
        help="Prepend `sudo` to all docker invocations.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_image:
        ensure_image(
            args.image,
            dockerfile_dir=REPO_RAYTRACING_DIR,
            dockerfile="docker/Dockerfile",
            use_sudo=args.use_sudo,
        )

    sim = DockerSimulator(
        image=args.image,
        mounts={str(args.mesh_dir.resolve()): "/work/mesh"},
        use_sudo=args.use_sudo,
    )

    scene = build_ivus_scene(Path("/work/mesh"))
    probe = build_ivus_probe(position_mm=(0.0, 0.0, 0.0))
    sim_params = build_sim_params()

    # Optional: tweak the lumen scattering coefficient to demonstrate the
    # material override pathway. Comment out to use the built-in defaults.
    scene.override_material(
        MaterialOverride(name="lumen", mu0=0.1)
    )

    print(f"[docker_ivus] Rendering single frame via {args.image}...")
    image = sim.simulate(
        scene,
        probe,
        sim_params,
        output_dir=args.output_dir / "single",
        output_name="b_mode.npy",
        cleanup=False,
    )
    print(f"[docker_ivus] Got image of shape {image.shape}, dtype {image.dtype}.")
    save_unwrapped_frame(
        image,
        args.output_dir / "single" / "b_mode.png",
        title="IVUS frame via Docker (single)",
    )

    if args.pullback:
        z_positions = np.linspace(-1.5, 1.5, 5)
        frames = [Pose(position_mm=(0.0, 0.0, float(z))) for z in z_positions]
        print(f"[docker_ivus] Rendering pullback ({len(frames)} frames)...")
        stack = sim.simulate(
            scene,
            probe,
            sim_params,
            frames=frames,
            output_dir=args.output_dir / "pullback",
            output_name="b_mode_stack.npy",
            cleanup=False,
        )
        print(f"[docker_ivus] Got pullback stack of shape {stack.shape}.")
        for i, (img, z) in enumerate(zip(stack, z_positions)):
            save_unwrapped_frame(
                img,
                args.output_dir / "pullback" / f"frame_{i:03d}.png",
                title=f"IVUS pullback z={z:.2f} mm (via Docker)",
            )

    print(f"[docker_ivus] Done. Outputs in {args.output_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
