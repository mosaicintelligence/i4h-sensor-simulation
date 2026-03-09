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
IVUS (Intravascular Ultrasound) simulation example.

Demonstrates scene setup (vessel-like phantom), single-frame imaging, and pullback
workflow. The image is displayed in unwrapped format: horizontal = angle (0–360°),
vertical = depth (mm).

Physical scaling (IVUS vs abdominal):
- All distances are in mm. Use t_far and scene geometry in the 1–10 mm range.
- Frequency: 20–40 MHz typical for IVUS (higher than abdominal). Set on IVUSProbe.
- Scattering speckle scale is set automatically for IVUS (finer resolution).
- Attenuation uses the probe frequency, so IVUS has stronger depth-dependent loss.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use non-interactive matplotlib backend to avoid Qt/XCB issues
# isort: off
import matplotlib

matplotlib.use("Agg")
# isort: on
import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs
from tqdm import tqdm

# Default output directory and display range
OUTPUT_DIR = "ivus_example_output"
MIN_VAL = -60.0
MAX_VAL = 0.0


def _mesh_path(name):
    """Absolute path to a mesh file under the project mesh/ directory (works from any cwd)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mesh", name)
    )


def build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=True):
    """
    Build a vessel phantom: lumen (water background), vessel wall, and optionally
    interior reflectors. Probe is placed at the vessel center; imaging plane is xz.

    If use_cylinder is True (default), the wall is a cylinder mesh (mesh/Cylinder.obj),
    or with use_thick_cylinder=True both Cylinder_inner.obj and Cylinder_outer.obj.
    No interior spheres are added when using mesh (OptiX requires one geometry type).
    If use_cylinder is False, the wall is a ring of spheres and two interior spheres.
    Generate cylinder meshes with default 129 segments so vertices do not align with 256 rays.
    """
    world = rs.World("water")
    wall_material = materials.get_index("vessel_wall")
    water_material = materials.get_index("vessel_wall")

    if use_cylinder:
        if use_thick_cylinder:
            # Thick wall: inner surface (liver) and outer surface (water so tracer sees liver->water)
            inner_path = _mesh_path("Cylinder_inner.obj")
            outer_path = _mesh_path("Cylinder_outer.obj")
            if not os.path.isfile(inner_path):
                raise FileNotFoundError(
                    f"Cylinder_inner not found: {inner_path}\n"
                    "Generate with: python utils/phantom_maker.py cylinder --output mesh --cylinder-thick"
                )
            if not os.path.isfile(outer_path):
                raise FileNotFoundError(
                    f"Cylinder_outer not found: {outer_path}\n"
                    "Generate with: python utils/phantom_maker.py cylinder --output mesh --cylinder-thick"
                )
            world.add(rs.Mesh(inner_path, wall_material))
            # Outer mesh uses water so refracted ray in wall sees liver->water and records the echo
            world.add(rs.Mesh(outer_path, water_material))
        else:
            # Single-surface cylinder (inward normals so rays from center hit front face)
            cylinder_path = _mesh_path("Cylinder.obj")
            if not os.path.isfile(cylinder_path):
                raise FileNotFoundError(
                    f"Cylinder mesh not found: {cylinder_path}\n"
                    "Generate it with: python utils/phantom_maker.py cylinder --output mesh"
                )
            world.add(rs.Mesh(cylinder_path, wall_material))
    else:
        # Fallback: ring of spheres approximating the wall
        num_wall = 36
        wall_radius_mm = 4.0
        wall_sphere_radius = 0.35
        for i in range(num_wall):
            theta = 2 * np.pi * i / num_wall
            x = wall_radius_mm * np.cos(theta)
            z = wall_radius_mm * np.sin(theta)
            world.add(
                rs.Sphere(
                    np.array([x, 0.0, z], dtype=np.float32),
                    wall_sphere_radius,
                    wall_material,
                )
            )
        # Interior reflectors (only when using spheres; cannot mix mesh + spheres in one GAS)
        world.add(
            rs.Sphere(np.array([1.5, 0.0, 2.0], dtype=np.float32), 0.4, wall_material)
        )
        world.add(
            rs.Sphere(np.array([-2.0, 0.0, 3.0], dtype=np.float32), 0.35, wall_material)
        )

    return world


def make_probe(position_mm, rotation_rad=None):
    """IVUS probe at the given position; rotation defaults to identity."""
    if rotation_rad is None:
        rotation_rad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return rs.IVUSProbe(
        rs.Pose(position=position_mm, rotation=rotation_rad),
        num_angular_rays=256,
        frequency=40.0,
        elevational_height=0.0,
        num_el_samples=1,
        f_num=1.0,
        speed_of_sound=1.54,
        pulse_duration=2.0,
    )


def get_sim_params():
    """Simulation parameters for IVUS (t_far, buffer_size, b_mode_size)."""
    p = rs.SimParams()
    p.conv_psf = True
    p.buffer_size = 4096
    p.t_far = 10.0  # mm
    p.b_mode_size = (512, 512)
    p.enable_cuda_timing = True
    return p


def save_unwrapped_frame(b_mode_image, simulator, path, title="IVUS cross-section (unwrapped)"):
    """Normalize, plot with angle/depth axes, and save to path."""
    min_x = simulator.get_min_x()
    max_x = simulator.get_max_x()
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()
    normalized = np.clip((b_mode_image - MIN_VAL) / (MAX_VAL - MIN_VAL), 0, 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        normalized,
        cmap="gray",
        extent=[min_x, max_x, max_z, min_z],
        aspect="auto",
    )
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax, label="Intensity (normalized)")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def save_polar_frame(b_mode_image, simulator, path, title="IVUS cross-section (polar)"):
    """
    Plot and save the same frame in polar format (standard ultrasound view):
    center = probe, radius = depth, angle = azimuth. 0° at top (12 o'clock).
    """
    min_x = simulator.get_min_x()
    max_x = simulator.get_max_x()
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()
    n_depth, n_angle = b_mode_image.shape
    normalized = np.clip((b_mode_image - MIN_VAL) / (MAX_VAL - MIN_VAL), 0, 1)

    # Edges for pcolormesh: angle in radians; 0° at top via set_theta_zero_location("N")
    theta_deg = np.linspace(min_x, max_x, n_angle + 1)
    theta_rad = np.radians(theta_deg)
    r_edges = np.linspace(min_z, max_z, n_depth + 1)
    Theta, R = np.meshgrid(theta_rad, r_edges)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.pcolormesh(Theta, R, normalized, cmap="gray", shading="flat")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(min_z, max_z)
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, label="Intensity (normalized)", shrink=0.7)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def run_single_frame(output_dir, use_cylinder=True, use_thick_cylinder=True):
    """Run one IVUS frame at the vessel center (origin)."""
    materials = rs.Materials()
    world = build_vessel_world(
        materials, use_cylinder=use_cylinder, use_thick_cylinder=use_thick_cylinder
    )
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    sim_params = get_sim_params()

    position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    probe = make_probe(position)
    b_mode_image = simulator.simulate(probe, sim_params)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ivus_frame.png")
    path_polar = os.path.join(output_dir, "ivus_frame_polar.png")
    save_unwrapped_frame(b_mode_image, simulator, path)
    save_polar_frame(b_mode_image, simulator, path_polar)
    print(f"Single frame saved to {path}")
    return path


def run_pullback(
    output_dir, n_frames=10, z_start=-1.5, z_end=1.5, use_cylinder=True, use_thick_cylinder=True
):
    """
    Run an IVUS pullback along the vessel: move the probe along z and
    simulate one frame at each position. Saves frame_000.png, frame_001.png, ...
    """
    materials = rs.Materials()
    world = build_vessel_world(
        materials, use_cylinder=use_cylinder, use_thick_cylinder=use_thick_cylinder
    )
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    sim_params = get_sim_params()

    z_positions = np.linspace(z_start, z_end, n_frames)
    os.makedirs(output_dir, exist_ok=True)

    for i, z in tqdm(enumerate(z_positions), total=n_frames, desc="Pullback"):
        position = np.array([0.0, 0.0, z], dtype=np.float32)
        probe = make_probe(position)
        b_mode_image = simulator.simulate(probe, sim_params)
        path = os.path.join(output_dir, f"frame_{i:03d}.png")
        save_unwrapped_frame(
            b_mode_image,
            simulator,
            path,
            title=f"IVUS pullback z = {z:.2f} mm",
        )

    print(f"Pullback: {n_frames} frames saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="IVUS example: vessel phantom, single frame, and/or pullback."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        default=True,
        help="Run single frame at vessel center (default: True).",
    )
    parser.add_argument(
        "--no-single",
        action="store_false",
        dest="single",
        help="Disable single frame.",
    )
    parser.add_argument(
        "--pullback",
        action="store_true",
        help="Run pullback along z and save multiple frames.",
    )
    parser.add_argument(
        "--pullback-frames",
        type=int,
        default=10,
        metavar="N",
        help="Number of pullback frames (default: 10).",
    )
    parser.add_argument(
        "--pullback-z",
        type=float,
        nargs=2,
        default=[-1.5, 1.5],
        metavar=("Z_START", "Z_END"),
        help="Pullback z range in mm (default: -1.5 1.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sphere-wall",
        action="store_true",
        help="Use a ring of spheres for the vessel wall instead of the cylinder mesh.",
    )
    parser.add_argument(
        "--thick-cylinder",
        action="store_true",
        help="Use thick-walled cylinder (Cylinder_inner.obj + Cylinder_outer.obj).",
    )
    args = parser.parse_args()

    use_cylinder = not args.sphere_wall
    use_thick = getattr(args, "thick_cylinder", False) and use_cylinder

    if args.single:
        run_single_frame(
            args.output_dir,
            use_cylinder=use_cylinder,
            use_thick_cylinder=use_thick,
        )
    if args.pullback:
        run_pullback(
            args.output_dir,
            n_frames=args.pullback_frames,
            z_start=args.pullback_z[0],
            z_end=args.pullback_z[1],
            use_cylinder=use_cylinder,
            use_thick_cylinder=use_thick,
        )
    if not args.single and not args.pullback:
        print("Run with --single and/or --pullback. Doing single frame.")
        run_single_frame(
            args.output_dir,
            use_cylinder=use_cylinder,
            use_thick_cylinder=use_thick,
        )


if __name__ == "__main__":
    main()
