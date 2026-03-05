# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wire phantom evaluation for IVUS simulation.

Creates a wire phantom (point targets on rings at known radii), visualizes the phantom
geometry, runs IVUS simulation, and saves example images in both linear (unwrapped)
and polar format for performance evaluation.

Wire phantom design (aligned with phasedArray_psf / rotatingSingleElm style):
- One wire per ring in a spiral (5 wires at 1–5 mm radius, each at a different angle)
- IVUS imaging plane (x-z); wires are point-like spheres (bone material) for visible echoes

Usage:
    python examples/wire_phantom_evaluation.py [--output-dir DIR]

Output:
    - Phantom visualization: wire_phantom_geometry.png
    - Simulated B-mode: wire_phantom_unwrapped.png, wire_phantom_polar.png
"""

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs

from ivus_example import (
    get_sim_params,
    save_polar_frame,
    save_unwrapped_frame,
)

OUTPUT_DIR_DEFAULT = "wire_phantom_output"

# Default wire phantom parameters (mm)
# IVUS rays lie in the x-z plane (y=0); wires must be in x-z to be visible.
# One wire per ring, arranged in a spiral (different angle per ring).
N_RINGS = 5
R_MIN_MM = 1.0
R_MAX_MM = 5.0
WIRE_RADIUS_MM = 0.12  # Small spheres to approximate point targets


def make_wire_phantom_positions(
    n_rings=N_RINGS,
    r_min_mm=R_MIN_MM,
    r_max_mm=R_MAX_MM,
):
    """
    Generate wire phantom: one wire per ring in a spiral.

    IVUS rays are in the x-z plane. Each ring gets a single wire at radius r and
    an angle that steps around so the wires form a spiral (not stacked at one angle).
    All wires are at y=0 (imaging plane center). Returns (N, 3) array of (x, y, z) in mm.
    """
    r_range = np.linspace(r_min_mm, r_max_mm, n_rings)
    positions = []
    for ind, r_val in enumerate(r_range):
        # Spiral: one angle per ring, evenly spaced so wires are not on top of each other
        theta = ind * 2 * np.pi / n_rings
        x = r_val * np.sin(theta)
        z = r_val * np.cos(theta)
        y = 0.0  # center of slice
        positions.append([x, y, z])
    return np.array(positions, dtype=np.float32)


def build_wire_phantom_world(materials, wire_positions_mm, wire_radius_mm=WIRE_RADIUS_MM, background="lumen"):
    """
    Build a World containing only the wire phantom (no vessel wall).

    - Background: lumen (blood) or water; use background="water" for stronger contrast.
    - Wires: small spheres with a high-impedance material so the reflection coefficient
      is large enough to be visible. We use "bone" (Z ≈ 7.8 MRayl) so the echo is visible
      after log compression.
    """
    world = rs.World(background)
    wire_material = materials.get_index("bone")

    for pos in wire_positions_mm:
        x, y, z = pos[0], pos[1], pos[2]
        sphere = rs.Sphere(
            np.array([x, y, z], dtype=np.float32),
            wire_radius_mm,
            wire_material,
        )
        world.add(sphere)

    return world


def visualize_phantom(wire_positions_mm, output_path):
    """
    Plot phantom geometry: IVUS imaging plane (x-z) and axial vs radial (y vs r).
    """
    x = wire_positions_mm[:, 0]
    y = wire_positions_mm[:, 1]
    z = wire_positions_mm[:, 2]
    r = np.sqrt(x**2 + z**2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: imaging plane (x-z) where IVUS rays lie – cross-section at probe
    ax = axes[0]
    ax.scatter(x, z, s=20, c="k", alpha=0.9, zorder=2)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Wire phantom (IVUS imaging plane x-z)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    for r_ref in np.unique(np.round(r, 2)):
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r_ref * np.sin(th), r_ref * np.cos(th), "r--", alpha=0.4, lw=0.8)

    # Right: axial (y) vs radial depth (r)
    ax = axes[1]
    ax.scatter(y, r, s=20, c="k", alpha=0.9, zorder=2)
    ax.set_xlabel("y (mm) – axial / slice")
    ax.set_ylabel("r (mm) – radial depth")
    ax.set_title("Wire phantom (axial vs radial)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Wire phantom geometry (validation)")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Phantom visualization saved to {output_path}")


def make_probe(position_mm=(0.0, 0.0, 0.0), rotation_rad=None):
    """IVUS probe at the given position (same defaults as ivus_example)."""
    if rotation_rad is None:
        rotation_rad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return rs.IVUSProbe(
        rs.Pose(position=np.array(position_mm, dtype=np.float32), rotation=rotation_rad),
        num_angular_rays=256,
        frequency=40.0,
        elevational_height=0.0,
        num_el_samples=1,
        f_num=1.0,
        speed_of_sound=1.54,
        pulse_duration=2.0,
    )


def run_wire_phantom_simulation(output_dir, save_phantom_viz=True, background="lumen"):
    """Build wire phantom, simulate one IVUS frame, save unwrapped and polar images."""
    # 1. Wire positions (rings in x-z plane so IVUS rays hit them)
    wire_positions = make_wire_phantom_positions()
    print(f"Wire phantom: {wire_positions.shape[0]} wires (1 per ring, spiral), "
          f"r = {R_MIN_MM}–{R_MAX_MM} mm (IVUS x-z plane)")

    # 2. World and materials
    materials = rs.Materials()
    world = build_wire_phantom_world(
        materials, wire_positions, wire_radius_mm=WIRE_RADIUS_MM, background=background
    )
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    sim_params = get_sim_params()

    # 3. Single frame at origin
    probe = make_probe((0.0, 0.0, 0.0))
    b_mode = simulator.simulate(probe, sim_params)

    os.makedirs(output_dir, exist_ok=True)

    # 4. Phantom visualization (optional)
    if save_phantom_viz:
        vis_path = os.path.join(output_dir, "wire_phantom_geometry.png")
        visualize_phantom(wire_positions, vis_path)

    # 5. Unwrapped and polar images
    path_unwrapped = os.path.join(output_dir, "wire_phantom_unwrapped.png")
    path_polar = os.path.join(output_dir, "wire_phantom_polar.png")
    save_unwrapped_frame(
        b_mode,
        simulator,
        path_unwrapped,
        title="IVUS wire phantom (unwrapped)",
    )
    save_polar_frame(
        b_mode,
        simulator,
        path_polar,
        title="IVUS wire phantom (polar)",
    )
    print(f"Unwrapped image: {path_unwrapped}")
    print(f"Polar image:     {path_polar}")
    return path_unwrapped, path_polar


def main():
    parser = argparse.ArgumentParser(
        description="Run IVUS simulation with a wire phantom and save evaluation images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory (default: {OUTPUT_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip saving phantom geometry visualization.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="lumen",
        choices=("lumen", "water"),
        help="Background material (default: lumen). Use water for stronger contrast.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    run_wire_phantom_simulation(
        output_dir, save_phantom_viz=not args.no_viz, background=args.background
    )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
