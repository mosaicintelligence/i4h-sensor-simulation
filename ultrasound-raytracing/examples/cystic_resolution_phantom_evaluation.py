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
Cystic-resolution phantom evaluation for IVUS simulation.

Creates a cystic-resolution phantom (tissue-mimicking background with anechoic
fluid-filled cysts at known positions/sizes), visualizes the phantom geometry,
runs IVUS simulation, and saves images in both linear (unwrapped) and polar format.

Phantom design:
- Background: vessel_wall (tissue-mimicking, scattering) so uncysted regions show speckle.
- Cysts: spheres filled with blood (fluid, low backscatter) in the IVUS imaging plane (x-z).
- Cysts at several radii and sizes to assess cyst detection and resolution.

Usage:
    python examples/cystic_resolution_phantom_evaluation.py [--output-dir DIR]

Output:
    - Phantom visualization: cystic_phantom_geometry.png
    - Simulated B-mode: cystic_phantom_unwrapped.png, cystic_phantom_polar.png
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

OUTPUT_DIR_DEFAULT = "cystic_phantom_output"

# Cystic phantom: (radius_mm, angle_deg, cyst_radius_mm) in IVUS x-z plane (y=0).
# Angle in degrees; 0 = +z, 90 = +x (matches typical polar convention).
# One entry per cyst: (radial position, angular position, cyst size).
CYST_SPECS = [
    (2.0, 0.0, 0.5),    # 2 mm depth, 0.5 mm radius cyst (small)
    (3.0, 72.0, 0.8),   # 3 mm, 0.8 mm cyst
    (4.0, 144.0, 1.0),  # 4 mm, 1.0 mm cyst
    (5.0, 216.0, 1.2),  # 5 mm, 1.2 mm cyst
    (3.5, 288.0, 0.6),  # 3.5 mm, 0.6 mm cyst (small)
]


def cyst_specs_to_positions(cyst_specs):
    """
    Convert (r_mm, angle_deg, cyst_radius_mm) to (x, y, z, cyst_radius_mm) in mm.
    IVUS imaging plane is x-z; angle 0 at +z, increasing toward +x (theta = 0 -> z, 90 -> x).
    So x = r * sin(angle), z = r * cos(angle), y = 0.
    """
    positions = []
    for r_mm, angle_deg, c_r in cyst_specs:
        th_rad = np.radians(angle_deg)
        x = r_mm * np.sin(th_rad)
        z = r_mm * np.cos(th_rad)
        y = 0.0
        positions.append((x, y, z, c_r))
    return positions


def build_cystic_phantom_world(materials, cyst_positions, background="vessel_wall"):
    """
    Build a World for the cystic-resolution phantom.

    - Background: vessel_wall (tissue-mimicking) so rays that miss geometry see tissue.
    - Cysts: spheres with blood (fluid) so they appear anechoic (dark) relative to tissue.
    """
    world = rs.World(background)
    cyst_material = materials.get_index("water")

    for (x, y, z, c_r) in cyst_positions:
        sphere = rs.Sphere(
            np.array([x, y, z], dtype=np.float32),
            float(c_r),
            cyst_material,
        )
        world.add(sphere)

    return world


def visualize_phantom(cyst_positions, output_path):
    """
    Plot phantom geometry: IVUS imaging plane (x-z) with cyst circles,
    and axial vs radial (y vs r) with cyst radii indicated.
    """
    # cyst_positions: list of (x, y, z, cyst_radius_mm)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: imaging plane (x-z) with cyst centers and circles
    ax = axes[0]
    for (x, y, z, c_r) in cyst_positions:
        ax.scatter(x, z, s=30, c="blue", alpha=0.9, zorder=2)
        circle = plt.Circle((x, z), c_r, fill=False, color="blue", linestyle="-", linewidth=1.5)
        ax.add_patch(circle)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Cystic-resolution phantom (IVUS imaging plane x-z)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    r_vals = [np.sqrt(p[0]**2 + p[2]**2) for p in cyst_positions]
    for r_ref in np.unique(np.round(r_vals, 2)):
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r_ref * np.sin(th), r_ref * np.cos(th), "gray", alpha=0.4, lw=0.8, linestyle="--")

    # Right: axial (y) vs radial depth (r), with cyst extent in r
    ax = axes[1]
    for (x, y, z, c_r) in cyst_positions:
        r = np.sqrt(x**2 + z**2)
        ax.scatter(y, r, s=30, c="blue", alpha=0.9, zorder=2)
        ax.errorbar(y, r, yerr=[[c_r], [c_r]], fmt="none", color="blue", capsize=2)
    ax.set_xlabel("y (mm) – axial / slice")
    ax.set_ylabel("r (mm) – radial depth")
    ax.set_title("Cystic phantom (axial vs radial)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Cystic-resolution phantom geometry")
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


def run_cystic_phantom_simulation(output_dir, save_phantom_viz=True, background="vessel_wall"):
    """Build cystic-resolution phantom, simulate one IVUS frame, save geometry, unwrapped and polar."""
    cyst_positions = cyst_specs_to_positions(CYST_SPECS)
    print(f"Cystic-resolution phantom: {len(cyst_positions)} cysts (tissue background, fluid cysts)")

    materials = rs.Materials()
    world = build_cystic_phantom_world(materials, cyst_positions, background=background)
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    sim_params = get_sim_params()

    probe = make_probe((0.0, 0.0, 0.0))
    b_mode = simulator.simulate(probe, sim_params)

    os.makedirs(output_dir, exist_ok=True)

    if save_phantom_viz:
        vis_path = os.path.join(output_dir, "cystic_phantom_geometry.png")
        visualize_phantom(cyst_positions, vis_path)

    path_unwrapped = os.path.join(output_dir, "cystic_phantom_unwrapped.png")
    path_polar = os.path.join(output_dir, "cystic_phantom_polar.png")
    save_unwrapped_frame(
        b_mode,
        simulator,
        path_unwrapped,
        title="IVUS cystic-resolution phantom (unwrapped)",
    )
    save_polar_frame(
        b_mode,
        simulator,
        path_polar,
        title="IVUS cystic-resolution phantom (polar)",
    )
    print(f"Unwrapped image: {path_unwrapped}")
    print(f"Polar image:     {path_polar}")
    return path_unwrapped, path_polar


def main():
    parser = argparse.ArgumentParser(
        description="Run IVUS simulation with a cystic-resolution phantom and save evaluation images.",
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
        default="vessel_wall",
        choices=("vessel_wall", "liver", "extravascular"),
        help="Background (tissue) material (default: vessel_wall).",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    run_cystic_phantom_simulation(
        output_dir, save_phantom_viz=not args.no_viz, background=args.background
    )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
