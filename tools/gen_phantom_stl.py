# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generate STL files for IVUS calibration phantom hardware.

Produces:
  hardware/wire_spiral_disc.stl   — wire-stringing disc (print 2 copies)
  hardware/wire_spiral_assembly.stl — visual reference (both discs + standoffs)
  hardware/phantom_mold_uniform.stl — open-top cylindrical mold w/ catheter channel
  hardware/phantom_mold_cyst.stl — same mold with 4 cyst-inclusion rod holders
  hardware/phantom_step_mold.stl — concentric ring mold for stepped-α phantom

All dimensions are in millimetres. Print orientation:
  * Discs: flat on bed (5 mm tall), 0.2 mm layer height, ≥ 30% infill, 0.4 mm nozzle.
  * Molds: flat on bed, 0.2 mm layer height, vase-mode walls if available, otherwise
    ≥ 3 perimeters and 20% infill.

Run from the repo root:
    PYTHONPATH=/tmp/calpkgs python3 tools/gen_phantom_stl.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely.geometry as sg
import trimesh
from trimesh.creation import extrude_polygon

trimesh.tol.facet_threshold = 0.01


def _union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.boolean.union(parts, engine="manifold")


def _diff(a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.boolean.difference([a, b], engine="manifold")

OUT = Path(__file__).resolve().parents[1] / "hardware"
OUT.mkdir(parents=True, exist_ok=True)

# Wire-spiral fixture parameters (must match the protocol)
DISC_OD = 50.0          # mm
DISC_THICK = 5.0        # mm
DISC_CENTER_HOLE = 2.5  # mm — catheter clearance hole (catheter OD ≈ 1.17 mm); tight fit aids centering
WIRE_HOLE_DIA = 0.5     # mm — for 25 µm tungsten wire (oversize for easy threading)
N_WIRES = 12
R_MIN = 1.75            # mm — innermost wire radius
DR = 0.18               # mm — radial increment per wire
DTHETA_DEG = 30.0       # azimuthal increment per wire (uniform 30° → 12 wires in one turn)
THETA_OFFSET_DEG = 0.0  # azimuth of wire #0
STANDOFF_HOLE_DIA = 3.2 # mm — clearance for M3 threaded rod / standoff
STANDOFF_BCD = 40.0     # mm — bolt circle diameter for the 3 standoffs (radius 20 mm)
STANDOFF_ANGLES_DEG = (60.0, 180.0, 300.0)
STANDOFF_LEN = 50.0     # mm — standoff height (= disc spacing)
STANDOFF_OD = 5.0       # mm — for the visual assembly only (commercial M3 standoff hex flat-to-flat ≈ 5 mm)


def _circle(cx: float, cy: float, r: float, n: int = 64) -> sg.Polygon:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return sg.Polygon(np.c_[cx + r * np.cos(th), cy + r * np.sin(th)])


def _wire_positions() -> list[tuple[float, float, float]]:
    """Return list of (x, y, theta_deg) for the 12 wire holes."""
    positions = []
    for n in range(N_WIRES):
        r = R_MIN + n * DR
        th_deg = THETA_OFFSET_DEG + n * DTHETA_DEG
        th = np.deg2rad(th_deg)
        positions.append((r * np.cos(th), r * np.sin(th), th_deg))
    return positions


def build_disc_polygon() -> sg.Polygon:
    outer = _circle(0, 0, DISC_OD / 2, n=128)
    holes = [_circle(0, 0, DISC_CENTER_HOLE / 2, n=64).exterior.coords[:]]
    for x, y, _ in _wire_positions():
        holes.append(_circle(x, y, WIRE_HOLE_DIA / 2, n=24).exterior.coords[:])
    for ang_deg in STANDOFF_ANGLES_DEG:
        ang = np.deg2rad(ang_deg)
        sx = STANDOFF_BCD / 2 * np.cos(ang)
        sy = STANDOFF_BCD / 2 * np.sin(ang)
        holes.append(_circle(sx, sy, STANDOFF_HOLE_DIA / 2, n=32).exterior.coords[:])
    return sg.Polygon(outer.exterior.coords[:], holes)


def build_disc_mesh() -> trimesh.Trimesh:
    body = _vert_cyl(DISC_OD / 2, DISC_THICK, sections=128)
    holes: list[trimesh.Trimesh] = []
    holes.append(_vert_cyl(DISC_CENTER_HOLE / 2, DISC_THICK + 1.0, z0=-0.5,
                           sections=64))
    for x, y, _ in _wire_positions():
        holes.append(_vert_cyl(WIRE_HOLE_DIA / 2, DISC_THICK + 1.0, z0=-0.5,
                               cx=x, cy=y, sections=24))
    for ang_deg in STANDOFF_ANGLES_DEG:
        ang = np.deg2rad(ang_deg)
        sx = STANDOFF_BCD / 2 * np.cos(ang)
        sy = STANDOFF_BCD / 2 * np.sin(ang)
        holes.append(_vert_cyl(STANDOFF_HOLE_DIA / 2, DISC_THICK + 1.0,
                               z0=-0.5, cx=sx, cy=sy, sections=32))
    body = _diff(body, _union(holes))
    body.metadata["name"] = "wire_spiral_disc"
    return body


def build_assembly_mesh() -> trimesh.Trimesh:
    """Visual reference: two discs + 3 standoffs (not for printing as one piece)."""
    parts: list[trimesh.Trimesh] = []
    bottom = build_disc_mesh()
    parts.append(bottom)
    top = build_disc_mesh()
    top.apply_translation([0, 0, DISC_THICK + STANDOFF_LEN])
    parts.append(top)
    for ang_deg in STANDOFF_ANGLES_DEG:
        ang = np.deg2rad(ang_deg)
        sx = STANDOFF_BCD / 2 * np.cos(ang)
        sy = STANDOFF_BCD / 2 * np.sin(ang)
        rod = trimesh.creation.cylinder(radius=STANDOFF_OD / 2,
                                        height=STANDOFF_LEN + 2 * DISC_THICK,
                                        sections=24)
        rod.apply_translation([sx, sy, (STANDOFF_LEN + 2 * DISC_THICK) / 2])
        parts.append(rod)
    cath_len = STANDOFF_LEN + 2 * DISC_THICK + 30
    cath = trimesh.creation.cylinder(radius=0.6, height=cath_len, sections=24)
    cath.apply_translation([0, 0, cath_len / 2 - 15])
    parts.append(cath)
    band = trimesh.creation.cylinder(radius=0.7, height=2.0, sections=24)
    band.apply_translation([0, 0, DISC_THICK + STANDOFF_LEN / 2])
    parts.append(band)
    for x, y, _ in _wire_positions():
        wire_h = STANDOFF_LEN + 2 * DISC_THICK
        wire = trimesh.creation.cylinder(radius=0.06, height=wire_h, sections=8)
        wire.apply_translation([x, y, wire_h / 2])
        parts.append(wire)
    return trimesh.util.concatenate(parts)


# Phantom molds (E4 / E5)
MOLD_OD = 60.0          # mm — outer diameter of the mold
MOLD_ID = 50.0          # mm — inner diameter (gives a 5 mm wall)
MOLD_HEIGHT = 60.0      # mm — gives 50 mm of phantom + 10 mm headspace
MOLD_FLOOR = 3.0        # mm — closed bottom thickness
CATH_CHANNEL_DIA = 2.0  # mm — channel formed by a removable PTFE rod ≤ 1.5 mm; 2 mm gives clearance
CYST_ROD_DIA = 4.0      # mm — diameter of cylindrical anechoic inclusions
CYST_RADII_MM = (10.0, 14.0, 14.0, 18.0)  # radial position of cyst rods
CYST_ANGLES_DEG = (0.0, 90.0, 270.0, 180.0)


def _vert_cyl(radius: float, height: float, z0: float = 0.0,
              cx: float = 0.0, cy: float = 0.0,
              sections: int = 64) -> trimesh.Trimesh:
    """Vertical cylinder spanning [z0, z0+height] centered at (cx, cy)."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height,
                                    sections=sections)
    cyl.apply_translation([cx, cy, z0 + height / 2])
    return cyl


def build_uniform_mold_mesh(with_cysts: bool = False) -> trimesh.Trimesh:
    """Open-top cylindrical mold for casting tissue-mimicking material.

    Design rationale: a simple cup with a closed floor and a small through-hole
    in the floor for the catheter PTFE rod. The rod is clamped externally
    above and below the mold so it stays centered while the gel sets.

    Cyst variant adds blind dimples in the floor so PTFE rods (4 mm OD) can be
    pressed in to stand vertically; they form the anechoic cylinders.
    """
    outer = _vert_cyl(MOLD_OD / 2, MOLD_HEIGHT)
    cup = _vert_cyl(MOLD_ID / 2, MOLD_HEIGHT - MOLD_FLOOR + 0.5,
                    z0=MOLD_FLOOR)
    cath_bore = _vert_cyl(CATH_CHANNEL_DIA / 2, MOLD_HEIGHT + 1.0, z0=-0.5)
    body = _diff(outer, _union([cup, cath_bore]))
    if with_cysts:
        dimple_h = MOLD_FLOOR - 1.0
        for r, ang_deg in zip(CYST_RADII_MM, CYST_ANGLES_DEG):
            ang = np.deg2rad(ang_deg)
            sx = r * np.cos(ang)
            sy = r * np.sin(ang)
            dimple = _vert_cyl(CYST_ROD_DIA / 2 + 0.1, dimple_h,
                               z0=1.0, cx=sx, cy=sy, sections=32)
            body = _diff(body, dimple)
    body.metadata["name"] = "phantom_mold"
    return body


# Step-attenuation phantom mold (E7 fallback)
STEP_WALL_RADII = (8.0, 12.0, 16.0, 20.0, 24.0)  # mm — radii of integral divider walls
STEP_WALL_THICK = 0.6  # mm — thin wall (single-perimeter print)


def build_step_mold_mesh() -> trimesh.Trimesh:
    """Multi-chamber concentric mold for a stepped-α phantom.

    The mold is a single solid cylinder with 6 annular chambers cut into
    its top, separated by 5 thin (0.6 mm) integral walls at radii
    8, 12, 16, 20, 24 mm. Each chamber is poured with a different graphite
    concentration to make a stepped-attenuation / stepped-scatter target.
    The central catheter bore runs through the floor.
    """
    body = _vert_cyl(MOLD_OD / 2, MOLD_HEIGHT)
    chamber_h = MOLD_HEIGHT - MOLD_FLOOR + 0.5
    half_w = STEP_WALL_THICK / 2
    bounds = [(0.0, STEP_WALL_RADII[0] - half_w)]
    for i in range(len(STEP_WALL_RADII) - 1):
        bounds.append((STEP_WALL_RADII[i] + half_w,
                       STEP_WALL_RADII[i + 1] - half_w))
    bounds.append((STEP_WALL_RADII[-1] + half_w, MOLD_ID / 2))
    cavities: list[trimesh.Trimesh] = []
    for r_in, r_out in bounds:
        outer = _vert_cyl(r_out, chamber_h, z0=MOLD_FLOOR, sections=64)
        if r_in <= 1e-6:
            cavities.append(outer)
        else:
            inner = _vert_cyl(r_in, chamber_h + 1.0, z0=MOLD_FLOOR - 0.5,
                              sections=64)
            cavities.append(_diff(outer, inner))
    cath_bore = _vert_cyl(CATH_CHANNEL_DIA / 2, MOLD_HEIGHT + 1.0,
                          z0=-0.5)
    cavities.append(cath_bore)
    body = _diff(body, _union(cavities))
    body.metadata["name"] = "phantom_step_mold"
    return body


def _finalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge close vertices and process the mesh in-place to maximize watertightness."""
    mesh.merge_vertices(merge_tex=False, merge_norm=False)
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def main() -> None:
    disc = _finalize(build_disc_mesh())
    disc_path = OUT / "wire_spiral_disc.stl"
    disc.export(disc_path)
    print(f"  wrote {disc_path}  ({disc.vertices.shape[0]} verts, "
          f"{disc.faces.shape[0]} faces)")

    asm = build_assembly_mesh()
    asm_path = OUT / "wire_spiral_assembly.stl"
    asm.export(asm_path)
    print(f"  wrote {asm_path}  (visual reference; do not print as one piece)")

    mold = _finalize(build_uniform_mold_mesh(with_cysts=False))
    mold_path = OUT / "phantom_mold_uniform.stl"
    mold.export(mold_path)
    print(f"  wrote {mold_path}")

    cyst = _finalize(build_uniform_mold_mesh(with_cysts=True))
    cyst_path = OUT / "phantom_mold_cyst.stl"
    cyst.export(cyst_path)
    print(f"  wrote {cyst_path}")

    step = _finalize(build_step_mold_mesh())
    step_path = OUT / "phantom_step_mold.stl"
    step.export(step_path)
    print(f"  wrote {step_path}")

    print("\nWire-spiral fixture geometry:")
    print(f"  Disc OD: {DISC_OD} mm, thickness: {DISC_THICK} mm")
    print(f"  Center hole: ⌀{DISC_CENTER_HOLE} mm "
          f"(catheter clearance for ≤ 1.5 mm OD probe)")
    print(f"  Wire holes: ⌀{WIRE_HOLE_DIA} mm × {N_WIRES} on Archimedean spiral")
    print(f"    r_n = {R_MIN} + n·{DR} mm  →  r ∈ [{R_MIN:.2f}, "
          f"{R_MIN + (N_WIRES - 1) * DR:.2f}] mm")
    print(f"    θ_n = {THETA_OFFSET_DEG} + n·{DTHETA_DEG}°  →  one full turn")
    print(f"  Standoff holes: ⌀{STANDOFF_HOLE_DIA} mm × 3 on "
          f"⌀{STANDOFF_BCD} mm BCD at 60°/180°/300°")
    print(f"  Required: 2× printed disc + 3× M3 × {STANDOFF_LEN} mm threaded "
          f"standoffs (or M3 rod + 6 nuts)")
    print()
    print("Wire (r, θ) layout:")
    for n, (x, y, th) in enumerate(_wire_positions()):
        r = np.hypot(x, y)
        print(f"  wire #{n+1:2d}:  r = {r:.2f} mm,  θ = {th:.0f}°")


if __name__ == "__main__":
    main()
