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
Example: Wire, cystic, and vessel phantoms with simulated B-mode in linear and polar coordinates.

Each scenario produces a figure with:
  1. Phantom layout (object positions in the imaging plane)
  2. Simulated ultrasound image in linear (Cartesian) coordinates
  3. Same image in polar (angle vs range) coordinates

Run from the repository root (build the Python extension first if needed):
  python examples/phantom_examples.py

Outputs are written to phantom_examples_output/.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
import raysim.cuda as rs


def _ensure_ivus_cylinder_meshes():
    """Generate IVUS-oriented cylinder OBJs if not present (axis = x, z_center = 10 mm)."""
    inner = os.path.join(_REPO_ROOT, "mesh", "Cylinder_inner_ivus.obj")
    outer = os.path.join(_REPO_ROOT, "mesh", "Cylinder_outer_ivus.obj")
    if os.path.isfile(inner) and os.path.isfile(outer):
        return
    try:
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(_REPO_ROOT, "utils", "cylinder_obj_to_ivus.py")],
            cwd=_REPO_ROOT,
            check=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Vessel phantom needs mesh/Cylinder_*_ivus.obj. Run: python utils/cylinder_obj_to_ivus.py"
        ) from e


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------

def cartesian_to_angle_depth(im_cart, min_x, max_x, depth_min, depth_max, n_angle=360, n_depth=None):
    """
    Resample Cartesian B-mode (x, depth) into (angle, depth) linear format.
    Returns linear_img with shape (n_depth, n_angle): rows = depth (0 at top), cols = angle (0–360°).
    """
    h, w = im_cart.shape
    if n_depth is None:
        n_depth = h
    angles_rad = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    depths = np.linspace(depth_min, depth_max, n_depth)
    Angle, Depth = np.meshgrid(angles_rad, depths)
    # (angle, depth) -> (x, z): x = depth*sin(angle), z = depth*cos(angle)
    X = Depth * np.sin(Angle)
    Z = Depth * np.cos(Angle)
    # Map to Cartesian image indices: row = depth (0 at top), col = x
    col = (X - min_x) / (max_x - min_x) * (w - 1) if (max_x - min_x) != 0 else np.full_like(X, (w - 1) / 2)
    row = (Z - depth_min) / (depth_max - depth_min) * (h - 1) if (depth_max - depth_min) != 0 else np.zeros_like(Z)
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)
    linear = map_coordinates(
        im_cart, [row.ravel(), col.ravel()], order=1, mode="constant", cval=0.0
    ).reshape(Depth.shape)
    return linear, depths


def angle_depth_to_circular_polar(linear_img, depth_max, size=400):
    """
    Render (depth, angle) linear image as a circular polar display.
    Center = probe (depth 0), radius = depth (mm). Angle goes around (0° at top, clockwise).
    linear_img shape: (n_depth, n_angle) with rows=depth, cols=angle. Returns (size, size) in mm extent.
    """
    n_depth, n_angle = linear_img.shape
    cx = cy = size // 2
    radius_px = float(cx - 1)
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    dx_mm = (jj - cx) / radius_px * depth_max
    dy_mm = (cy - ii) / radius_px * depth_max
    r_mm = np.sqrt(dx_mm**2 + dy_mm**2)
    angle_rad = np.arctan2(dx_mm, dy_mm)
    angle_rad = np.where(angle_rad < 0, angle_rad + 2 * np.pi, angle_rad)
    depth_idx = (r_mm / depth_max) * (n_depth - 1)
    angle_idx = (angle_rad / (2 * np.pi)) * (n_angle - 1)
    depth_idx = np.clip(depth_idx, 0, n_depth - 1)
    angle_idx = np.clip(angle_idx, 0, n_angle - 1)
    out = map_coordinates(
        linear_img, [depth_idx.ravel(), angle_idx.ravel()], order=1, mode="constant", cval=0.0
    ).reshape((size, size))
    out[r_mm > depth_max] = 0.0
    return out


def plot_phantom_layout(ax, objects_xy, title, xlim, zlim, reference_rings=None, axis_labels=None):
    """Draw phantom object centers in the imaging plane. objects_xy: list of (1st, 2nd) or (1st, 2nd, label).
    For IVUS the imaging plane is y-z, so pass (y, z) and axis_labels=('y (mm)', 'z (mm)').
    """
    if axis_labels is None:
        axis_labels = ("x (mm)", "z (mm)")
    for item in objects_xy:
        if len(item) == 3:
            a1, a2, label = item
            ax.scatter([a1], [a2], s=80, zorder=2, label=label)
        else:
            a1, a2 = item
            ax.scatter([a1], [a2], s=80, zorder=2)
    if reference_rings is not None:
        for r_ref in reference_rings:
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(r_ref * np.sin(th), r_ref * np.cos(th), "r--", alpha=0.4, lw=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect("equal")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any(len(o) == 3 for o in objects_xy):
        ax.legend(loc="upper right", fontsize=8)


def run_and_plot_scenario(
    name,
    world,
    objects_for_plot,
    materials,
    probe,
    sim_params,
    output_dir,
    layout_xlim=None,
    layout_zlim=None,
    reference_rings=None,
):
    """Run simulation and create one figure: phantom layout, linear B-mode, polar B-mode.

    simulate() returns Cartesian B-mode; we resample to (angle, depth) for linear and polar.
    """
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    b_mode = simulator.simulate(probe, sim_params)
    min_x = simulator.get_min_x()
    max_x = simulator.get_max_x()
    min_z = simulator.get_min_z()
    max_z = simulator.get_max_z()

    min_val, max_val = -60.0, 0.0
    normalized = np.clip((b_mode - min_val) / (max_val - min_val), 0, 1)
    depth_min = 0.0
    depth_max = max_z
    if min_z < 0 and max_z > 0:
        depth_max = max_z
    linear_img, _ = cartesian_to_angle_depth(
        normalized, min_x, max_x, depth_min, depth_max,
        n_angle=360, n_depth=normalized.shape[0]
    )

    # Build circular polar: center = probe, radius = depth, angle around
    polar_circular = angle_depth_to_circular_polar(linear_img, depth_max, size=400)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # Phantom layout (use phantom-specific limits when provided)
    plot_xlim = (layout_xlim if layout_xlim is not None else (min_x, max_x))
    plot_zlim = (layout_zlim if layout_zlim is not None else (min_z, max_z))
    # IVUS imaging plane is y-z (catheter axis = x); layout shows (y, z)
    plot_phantom_layout(
        axes[0],
        objects_for_plot,
        f"{name}\nPhantom layout",
        plot_xlim,
        plot_zlim,
        reference_rings=reference_rings,
        axis_labels=("y (mm)", "z (mm)"),
    )
    # Linear: x = angle (deg), y = depth (mm), grayscale = intensity (depth 0 at top)
    axes[1].imshow(
        linear_img,
        cmap="gray",
        extent=[0, 360, depth_max, 0],
        aspect="auto",
    )
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Depth (mm)")
    axes[1].set_title(f"{name}\nB-mode (linear)")
    # Polar: circular, angle around, distance from center = depth (mm)
    axes[2].imshow(
        polar_circular,
        cmap="gray",
        extent=[-depth_max, depth_max, -depth_max, depth_max],
        aspect="equal",
    )
    axes[2].set_xlabel("x (mm)")
    axes[2].set_ylabel("z (mm)")
    axes[2].set_title(f"{name}\nB-mode (polar)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_phantom.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Phantom definitions (world + list of (y,z) or (y,z,label) for layout = IVUS imaging plane)
# -----------------------------------------------------------------------------

# Wire phantom: one wire per ring in a spiral (IVUS imaging plane = y-z, catheter axis = x)
N_RINGS = 5
R_MIN_MM = 1.0
R_MAX_MM = 5.0
WIRE_RADIUS_MM = 0.12  # Small spheres to approximate point targets


def make_wire_phantom_positions(n_rings=N_RINGS, r_min_mm=R_MIN_MM, r_max_mm=R_MAX_MM):
    """
    One wire per ring in a spiral. IVUS imaging plane is y-z (rays in y-z; catheter axis = x).
    Wires at x=0 so they lie in the imaging plane. Returns (N, 3) array of (x, y, z) in mm.
    """
    r_range = np.linspace(r_min_mm, r_max_mm, n_rings)
    positions = []
    for ind, r_val in enumerate(r_range):
        theta = ind * 2 * np.pi / n_rings
        x = 0.0  # in imaging plane (y-z)
        y = r_val * np.sin(theta)
        z = r_val * np.cos(theta)
        positions.append([x, y, z])
    return np.array(positions, dtype=np.float32)


def make_wire_phantom(materials):
    """
    Wire phantom: one wire per ring at radii 1–5 mm in a spiral.
    Wires in the IVUS imaging plane (y-z) for correct radial visibility.
    """
    world = rs.World("water")
    wire_positions = make_wire_phantom_positions()
    bone_id = materials.get_index("bone")
    for pos in wire_positions:
        sphere = rs.Sphere(pos, WIRE_RADIUS_MM, bone_id)
        world.add(sphere)
    # Radial distance in imaging plane (y-z)
    r_vals = np.sqrt(wire_positions[:, 1] ** 2 + wire_positions[:, 2] ** 2)
    # Layout plot: imaging plane (y, z)
    objects_for_plot = [
        (wire_positions[i, 1], wire_positions[i, 2], f"r={r_vals[i]:.1f} mm")
        for i in range(len(wire_positions))
    ]
    layout_xlim = (-7.0, 7.0)
    layout_zlim = (-7.0, 7.0)
    reference_rings = np.unique(np.round(r_vals, 2))
    return world, objects_for_plot, layout_xlim, layout_zlim, reference_rings


def make_cystic_phantom(materials):
    """Anechoic cyst (water sphere) in a scattering background (liver). At 12 mm for IVUS depth."""
    world = rs.World("liver")
    water_id = materials.get_index("water")
    cyst_z = 12.0  # mm (IVUS-appropriate depth)
    cyst = rs.Sphere(np.array([0.0, 0.0, cyst_z], dtype=np.float32), 4.0, water_id)
    world.add(cyst)
    objects_for_plot = [(0, cyst_z, "cyst")]
    layout_xlim = (-10.0, 10.0)
    layout_zlim = (5.0, 20.0)
    return world, objects_for_plot, layout_xlim, layout_zlim, None


def make_vessel_phantom(materials):
    """
    Vessel phantom using cylinder OBJ meshes (axis along x, center at z=10 mm).
    Outer cylinder = vessel wall (liver), inner = inner wall; lumen = blood (background).
    Meshes are rotated/placed by utils/cylinder_obj_to_ivus.py -> Cylinder_*_ivus.obj.
    """
    _ensure_ivus_cylinder_meshes()
    mesh_dir = os.path.join(_REPO_ROOT, "mesh")
    world = rs.World("blood")  # lumen and background = blood
    liver_id = materials.get_index("liver")
    blood_id = materials.get_index("blood")
    # Both cylinder surfaces are vessel wall (tissue); interior r < 3.5 mm is lumen (water)
    outer_path = os.path.abspath(os.path.join(mesh_dir, "Cylinder_outer_ivus.obj"))
    inner_path = os.path.abspath(os.path.join(mesh_dir, "Cylinder_inner_ivus.obj"))
    outer_mesh = rs.Mesh(outer_path, blood_id)
    inner_mesh = rs.Mesh(inner_path, liver_id)
    world.add(outer_mesh)
    world.add(inner_mesh)
    vessel_z = 10.0  # mm (matches Z_CENTER in cylinder_obj_to_ivus.py)
    objects_for_plot = [(0, vessel_z, "vessel")]
    layout_xlim = (-8.0, 8.0)
    layout_zlim = (4.0, 16.0)
    return world, objects_for_plot, layout_xlim, layout_zlim, None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    output_dir = "phantom_examples_output"
    os.makedirs(output_dir, exist_ok=True)

    materials = rs.Materials()
    # Vessel phantom: cylinder cross-section in y-z is centered at (y=0, z=10 mm).
    # IVUS catheter must be at that center so 360° rays hit inner/outer walls uniformly.
    pose = rs.Pose(
        position=np.array([0.0, 0.0, 10.0], dtype=np.float32),  # center of vessel (matches Z_CENTER)
        rotation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    # IVUS probe: full 360° radial sweep (point source at catheter center)
    probe = rs.IVUSProbe(
        pose,
        num_elements_x=361,   # 1° angular step over 360°
        frequency=40.0,       # MHz (typical IVUS)
        elevational_height=0.0,  # single slice at vessel center
        num_el_samples=1,
    )
    sim_params = rs.SimParams()
    sim_params.conv_psf = True
    sim_params.buffer_size = 4096  # Must match Hilbert row length (compile-time)
    # IVUS-appropriate depth: typical 5–15 mm; use 20 mm so wire (1–5 mm), cyst, vessel all in range
    sim_params.t_far = 20.0
    sim_params.write_debug_images = True
    sim_params.b_mode_size = (600, 600)
    sim_params.use_scattering = True

    scenarios = [
        # ("Wire phantom", make_wire_phantom(materials)),
        # ("Cystic phantom", make_cystic_phantom(materials)),
        ("Vessel phantom", make_vessel_phantom(materials)),
    ]
    for name, phantom_data in scenarios:
        world, objects_for_plot, layout_xlim, layout_zlim, reference_rings = phantom_data
        run_and_plot_scenario(
            name,
            world,
            objects_for_plot,
            materials,
            probe,
            sim_params,
            output_dir,
            layout_xlim=layout_xlim,
            layout_zlim=layout_zlim,
            reference_rings=reference_rings,
        )


if __name__ == "__main__":
    main()
