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
  hardware/wire_spiral_disc.stl              — wire-stringing disc (print 2 copies)
  hardware/wire_spiral_standoff.stl          — printable shouldered standoff (print 3 copies)
  hardware/wire_spiral_assembly.stl          — visual reference (both discs + standoffs)
  hardware/phantom_mold_uniform.stl          — open-top cylindrical mold w/ catheter channel
  hardware/phantom_mold_cyst.stl             — canonical cyst mold (4 mm cyst dimples at {10,14,14,18} mm)
  hardware/phantom_mold_cyst_lid.stl         — registration lid for the canonical 4 mm cyst mold
  hardware/phantom_mold_cyst_interim.stl     — interim cyst mold (8 mm cyst dimples at {12,16,16,20} mm)
  hardware/phantom_mold_cyst_lid_interim.stl — registration lid for the interim 8 mm cyst mold
  hardware/phantom_step_mold.stl             — concentric ring mold for stepped-α phantom

Wire-spiral fixture is sized for the **Visions PV .035 (10 MHz, ⌀1.9 mm OD)**
catheter and the imaging characteristics measured from the P_035_PointScatter
captures (focus ~19 mm, useful range ~3–25 mm, ring-down to ~3 mm). 12 wires
on a one-turn spiral, with denser sampling around the focal depth so the
Gaussian-beam fit can actually constrain focal_length_mm and element_radius_mm.

Cyst-phantom molds are sized for the **60 mm imaging diameter** reference
operating point of the Volcano s5i (was 16 mm in older versions of this
script; the larger mold ID leaves a ≥ 10 mm radial margin between the
30 mm imaging fan and the mold wall, eliminating wall ring-down inside
the imaging field). Two cyst variants are produced:
  * Canonical (`phantom_mold_cyst.stl` + `_lid.stl`): 4 mm PTFE / steel
    cyst rods at radii {10, 14, 14, 18} mm — per `A.3` of the IVUS
    calibration protocol.
  * Interim T1-E5* (`_interim.stl` + `_lid_interim.stl`): 8 mm rigid
    reusable plastic-straw cyst rods at radii {12, 16, 16, 20} mm — the
    shifted radii keep the innermost cyst's inner edge ≥ 8 mm from the
    catheter, clear of the ring-down zone.

Each cyst mold pairs with a printable **lid** that has matching cyst-rod
through-holes plus a rectangular **orientation key tab** projecting
downward from the lid's underside at 0°. The matching slot is cut into
the mold rim at 0° — the tab slip-fits into the slot when the lid is
correctly oriented, and the lid will sit visibly proud of the rim if
the orientation is wrong. With the lid correctly seated, the lid's rod
holes register directly above the mold's floor dimples, mechanically
forcing the rods vertical during the gel pour and set without any
external clamping. The mold's catheter rod registration is a 2 mm-deep
blind divot in the floor (analogous to the cyst-rod dimples) — the
bottom of the floor remains solid so the rod cannot poke through and
no gel can leak out during pouring.

All dimensions are in millimetres. Print orientation:
  * Discs: flat on bed (5 mm tall), 0.2 mm layer height, ≥ 3 perimeters,
    ≥ 30 % infill, 0.4 mm nozzle. The wire holes are at the FDM resolution
    floor — verify each one is open after printing and clear with a 0.5 mm
    drill bit if needed.
  * Standoffs: print 3 copies, vertically (long axis along Z) for best
    concentricity of the slip-fit pins; 0.2 mm layer height, ≥ 4 perimeters,
    ≥ 40 % infill. Or substitute the commercial M3-F/M3-F standoffs from the
    BoM (⌀5 mm hex × 50 mm long) with M3 × 6 mm pan-head screws.
  * Molds: flat on bed, 0.2 mm layer height, vase-mode walls if available,
    otherwise ≥ 3 perimeters and 20 % infill.
  * Cyst-mold lids: **print top-surface-down** (lid lip pointing UP) so
    the wider top plate sits on the bed and the narrower centering lip
    prints on top of it with no overhang. ≥ 3 perimeters, ≥ 30 % infill,
    0.2 mm layer height.

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

# Wire-spiral fixture parameters (must match the protocol).
# Sized for the Visions PV .035 (10 MHz, ⌀1.9 mm OD) catheter, focal depth
# ~19 mm. Wires span r ∈ [4, 26] mm so the Gaussian-beam fit has data both
# near-field, around focus, and in the far-field.
DISC_OD = 80.0          # mm — outer disc diameter (must contain wire array + standoff bolt-circle + 5 mm rim)
DISC_THICK = 5.0        # mm — disc thickness; 0.2 mm layer height gives 25 layers (rigid enough for wire tensioning)
DISC_CENTER_HOLE = 2.5  # mm — slip-fit on PV .035 catheter (1.9 mm OD) with 0.3 mm radial clearance per side to absorb FDM XY tolerance (~0.15–0.20 mm); was 2.0 mm in older prints and required field-drilling to 2.4 mm
WIRE_HOLE_DIA = 0.5     # mm — through-hole; sized for any sub-wavelength filament (25–127 µm); locked at the counterbore (see WIRE_CB_*). Default protocol material is ~75 µm nylon monofilament; alternates: 36 AWG copper magnet wire, 25 µm tungsten — see ivus_calibration_protocol.md Appendix A.1
WIRE_CB_DIA = 1.0       # mm — counterbore on top face for a small bead of cyanoacrylate (any wire) or a heat-melted ball (nylon) to lock the tensioned wire
WIRE_CB_DEPTH = 0.4     # mm — depth of lock-bead counterbore (below the top surface)

# Wire layout: 12 unique radii from 4 to 26 mm, denser around the focal zone
# (12-20 mm) where the lateral PSF is most informative. Azimuth increments
# 30° so consecutive wires are well-separated in arc-length even at r=4 mm
# (arc ≈ 2.1 mm >> ~1 mm lateral PSF).
WIRE_LAYOUT_MM: tuple[tuple[float, float], ...] = (
    (4.0,    0.0),   # near-field anchor (just outside ring-down zone)
    (6.0,   30.0),   # near-field
    (8.0,   60.0),
    (10.0,  90.0),   # entering focal zone
    (12.0, 120.0),
    (14.0, 150.0),
    (16.0, 180.0),   # near-focus (focal depth ~19 mm from E2 fit)
    (18.0, 210.0),
    (20.0, 240.0),   # past focus
    (22.0, 270.0),
    (24.0, 300.0),
    (26.0, 330.0),   # far-field anchor (still inside 60 mm-FOV imaging window)
)
N_WIRES = len(WIRE_LAYOUT_MM)

STANDOFF_HOLE_DIA = 3.2 # mm — clearance for M3 thread or for the printable shouldered standoff's tip pin
STANDOFF_BCD = 70.0     # mm — bolt circle diameter for the 3 standoffs (radius 35 mm — outside the wire array at 26 mm)
STANDOFF_ANGLES_DEG = (60.0, 180.0, 300.0)
STANDOFF_LEN = 50.0     # mm — standoff height (= clear distance between disc faces)
STANDOFF_OD = 5.0       # mm — for the visual assembly only (commercial M3 standoff hex flat-to-flat ≈ 5 mm)
# Printable shouldered standoff parameters
PRINT_STANDOFF_SHAFT_OD = 7.0  # mm — middle "shoulder" diameter (rests against disc face, prevents disc from sliding)
PRINT_STANDOFF_PIN_OD = 3.1    # mm — tip pin diameter (slip-fits into disc's ⌀3.2 mm STANDOFF_HOLE_DIA)
PRINT_STANDOFF_PIN_LEN = 5.0   # mm — tip pin length per end (= disc thickness, so the pin fully passes through)
PRINT_STANDOFF_LOCK_HOLE = 1.5 # mm — cross-bore at each pin tip for a split pin / zip-tie / safety wire


def _circle(cx: float, cy: float, r: float, n: int = 64) -> sg.Polygon:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return sg.Polygon(np.c_[cx + r * np.cos(th), cy + r * np.sin(th)])


def _wire_positions() -> list[tuple[float, float, float]]:
    """Return list of (x, y, theta_deg) for the 12 wire holes."""
    positions = []
    for r, th_deg in WIRE_LAYOUT_MM:
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
        holes.append(_vert_cyl(WIRE_CB_DIA / 2, WIRE_CB_DEPTH + 0.5,
                               z0=DISC_THICK - WIRE_CB_DEPTH,
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


def build_printable_standoff_mesh() -> trimesh.Trimesh:
    """3D-printable shouldered standoff (alternative to commercial M3-F/M3-F).

    Geometry:
      * middle shaft: ⌀ PRINT_STANDOFF_SHAFT_OD × STANDOFF_LEN — the shoulder
        that bears against the disc face, preventing the disc from sliding
        down toward the centre.
      * tip pins: ⌀ PRINT_STANDOFF_PIN_OD × PRINT_STANDOFF_PIN_LEN at each
        end — slip-fits into the disc's ⌀ STANDOFF_HOLE_DIA hole.
      * cross-bore at each pin tip (⌀ PRINT_STANDOFF_LOCK_HOLE) so the disc
        can be locked in place from the outside with a split pin, zip-tie,
        or safety-wire loop after both discs are seated.

    Print 3 copies vertically (long axis along Z) for best concentricity of
    the slip-fit pins. ≥ 4 perimeters and ≥ 40 % infill recommended.
    """
    parts: list[trimesh.Trimesh] = []
    shaft = _vert_cyl(PRINT_STANDOFF_SHAFT_OD / 2, STANDOFF_LEN,
                      z0=PRINT_STANDOFF_PIN_LEN, sections=48)
    parts.append(shaft)
    parts.append(_vert_cyl(PRINT_STANDOFF_PIN_OD / 2,
                           PRINT_STANDOFF_PIN_LEN, z0=0.0, sections=32))
    parts.append(_vert_cyl(PRINT_STANDOFF_PIN_OD / 2,
                           PRINT_STANDOFF_PIN_LEN,
                           z0=PRINT_STANDOFF_PIN_LEN + STANDOFF_LEN,
                           sections=32))
    body = _union(parts)

    # Cross-bores at each pin tip for a split-pin / zip-tie / safety-wire lock.
    # Bore axis = +Y, centred 1.5 mm in from the very end so it survives the
    # disc thickness when the disc is seated against the shoulder.
    lock_offset = 1.5
    bores: list[trimesh.Trimesh] = []
    for z_centre in (lock_offset,
                     2 * PRINT_STANDOFF_PIN_LEN + STANDOFF_LEN - lock_offset):
        bore = trimesh.creation.cylinder(
            radius=PRINT_STANDOFF_LOCK_HOLE / 2,
            height=PRINT_STANDOFF_PIN_OD + 1.0, sections=24)
        # Orient cylinder along +Y (default is +Z) and translate to z_centre.
        bore.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        bore.apply_translation([0, 0, z_centre])
        bores.append(bore)
    body = _diff(body, _union(bores))
    body.metadata["name"] = "wire_spiral_standoff"
    return body


def build_assembly_mesh() -> trimesh.Trimesh:
    """Visual reference: two discs + 3 printable shouldered standoffs.

    Stack-up (Z increases upward):
      0                          ← bottom disc lower face
      DISC_THICK                 ← bottom disc upper face / standoff shoulder
      DISC_THICK + STANDOFF_LEN  ← top disc lower face / standoff shoulder
      2*DISC_THICK + STANDOFF_LEN ← top disc upper face

    The standoff's tip pins (PRINT_STANDOFF_PIN_LEN = DISC_THICK = 5 mm) pass
    fully through each disc; the cross-bores end up just outside each disc's
    outer face for the safety-wire lock.
    """
    parts: list[trimesh.Trimesh] = []
    bottom = build_disc_mesh()
    parts.append(bottom)
    top = build_disc_mesh()
    top.apply_translation([0, 0, DISC_THICK + STANDOFF_LEN])
    parts.append(top)
    standoff = build_printable_standoff_mesh()
    for ang_deg in STANDOFF_ANGLES_DEG:
        ang = np.deg2rad(ang_deg)
        sx = STANDOFF_BCD / 2 * np.cos(ang)
        sy = STANDOFF_BCD / 2 * np.sin(ang)
        # build_printable_standoff_mesh() places its lower pin tip at z=0;
        # we want the lower pin tip to start at z=0 too, so the bottom disc
        # sits on a 5 mm pin and the shoulder bears against the disc's top
        # face at z=DISC_THICK.
        rod = standoff.copy()
        rod.apply_translation([sx, sy, 0.0])
        parts.append(rod)
    cath_len = STANDOFF_LEN + 2 * DISC_THICK + 30
    cath = trimesh.creation.cylinder(radius=DISC_CENTER_HOLE / 2 - 0.05,
                                     height=cath_len, sections=24)
    cath.apply_translation([0, 0, cath_len / 2 - 15])
    parts.append(cath)
    band = trimesh.creation.cylinder(radius=DISC_CENTER_HOLE / 2 + 0.05,
                                     height=2.0, sections=24)
    band.apply_translation([0, 0, DISC_THICK + STANDOFF_LEN / 2])
    parts.append(band)
    for x, y, _ in _wire_positions():
        wire_h = STANDOFF_LEN + 2 * DISC_THICK
        wire = trimesh.creation.cylinder(radius=0.064, height=wire_h, sections=8)
        wire.apply_translation([x, y, wire_h / 2])
        parts.append(wire)
    return trimesh.util.concatenate(parts)


# Phantom molds (E4 / E5)
# Sized for the **60 mm imaging diameter** reference operating point of the
# Volcano s5i (was 16 mm imaging diameter in older versions of this script).
# Inner radius 40 mm leaves a 10 mm safety margin beyond the 30 mm imaging
# radius so the mold wall does not produce ring-down artefacts inside the
# imaging field.
MOLD_OD = 90.0          # mm — outer diameter (5 mm wall; clean FDM print at 0.4 mm nozzle)
MOLD_ID = 80.0          # mm — inner diameter (40 mm internal radius vs 30 mm imaging radius)
MOLD_HEIGHT = 70.0      # mm — gives 60 mm of phantom + 10 mm headspace for lid lip and rod caps
MOLD_FLOOR = 3.0        # mm — closed bottom thickness
CATH_CHANNEL_DIA = 3.5  # mm — slip-fit on a 3 mm-class channel-forming rod (wooden skewer, 3 mm PTFE, 3 mm brass tubing, …) with 0.25 mm radial clearance to absorb FDM tolerance. The resulting 3 mm gel channel accepts the Visions PV .035 (1.9 mm OD) catheter with ~0.55 mm radial clearance per side; gel compression seals the gap for acoustic coupling. Was 2.0 mm in older revisions (sized for a 1.5 mm PTFE rod) — that produced a 1.5 mm gel channel which is too tight for the 1.9 mm catheter to enter without forcing.

# Cyst-forming rod parameters
# Canonical recipe (`A.3` of ivus_calibration_protocol.md):
#   4 mm PTFE or stainless rods at radii {10, 14, 14, 18} mm
CYST_ROD_DIA = 4.0
CYST_RADII_MM = (10.0, 14.0, 14.0, 18.0)
# Interim recipe (T1-E5* of interim_milk_phantom_sop.md):
#   8 mm rigid reusable plastic straws at shifted radii {12, 16, 16, 20} mm.
#   The +2 mm radial shift keeps the innermost cyst's inner edge ≥ 8 mm from
#   the catheter, clear of the 3 mm ring-down zone.
CYST_ROD_DIA_INTERIM = 8.0
CYST_RADII_MM_INTERIM = (12.0, 16.0, 16.0, 20.0)
CYST_ANGLES_DEG = (0.0, 90.0, 270.0, 180.0)

# Cyst-phantom registration lid parameters
# The lid sits on the mold rim with a centering lip slipped into the mold ID.
# Five through-holes register the catheter PTFE rod (centre) and the four cyst
# rods (at design radii). A **mechanical orientation key** — a downward-
# projecting rectangular tab on the lid's underside at 0°, slip-fitting into
# a matching rectangular slot in the mold rim at 0° — mechanically enforces
# the lid's rotational alignment (you cannot seat the lid flush on the rim
# unless the tab is over the slot). When properly seated the lid's cyst-rod
# holes register directly over the mold's floor dimples and the rods are
# constrained vertical without any external clamping.
LID_THICK = 4.0                       # mm — top plate thickness
LID_LIP_HEIGHT = 3.0                  # mm — height of centering lip into mold ID
LID_LIP_CLEARANCE = 0.4               # mm — diametric clearance of lip vs mold ID (0.2 mm radial)
LID_CYST_HOLE_RADIAL_CLEARANCE = 0.15 # mm — radial clearance on each side of cyst rod (0.3 mm dia)
LID_CATH_HOLE_DIA = CATH_CHANNEL_DIA  # mm — same diameter as the mold's catheter bore
KEY_W = 5.0                           # mm — width (azimuthal) of the lid's key tab; mold slot is this + 2 × KEY_AZI_CLEARANCE
KEY_H = 3.0                           # mm — height (vertical) of the lid's key tab; mold slot is this + KEY_VERT_CLEARANCE
KEY_AZI_CLEARANCE = 0.2               # mm — slot is wider than tab by this much on each azimuthal side (slip fit)
KEY_VERT_CLEARANCE = 0.3              # mm — slot is taller than tab by this much vertically (slip fit)


def _vert_cyl(radius: float, height: float, z0: float = 0.0,
              cx: float = 0.0, cy: float = 0.0,
              sections: int = 64) -> trimesh.Trimesh:
    """Vertical cylinder spanning [z0, z0+height] centered at (cx, cy)."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height,
                                    sections=sections)
    cyl.apply_translation([cx, cy, z0 + height / 2])
    return cyl


def _orientation_key_mesh(width: float, height: float, z0: float,
                          r_inner: float, r_outer: float) -> trimesh.Trimesh:
    """Rectangular orientation key block at 0° spanning radii [r_inner, r_outer].

    Used both as the lid's downward-projecting tab (added material at exact
    KEY_W / KEY_H dimensions) and as the mold's matching rim slot (subtracted
    material, slightly oversized for slip-fit clearance). The block is
    centred azimuthally on the +x axis (0°).
    """
    y_top = width / 2
    y_bot = -width / 2
    poly = sg.Polygon([
        (r_outer, y_top),
        (r_outer, y_bot),
        (r_inner, y_bot),
        (r_inner, y_top),
    ])
    mesh = extrude_polygon(poly, height=height)
    mesh.apply_translation([0.0, 0.0, z0])
    return mesh


def build_uniform_mold_mesh(
    with_cysts: bool = False,
    cyst_rod_dia: float = CYST_ROD_DIA,
    cyst_radii_mm: tuple[float, ...] = CYST_RADII_MM,
    add_orientation_key_slot: bool = False,
) -> trimesh.Trimesh:
    """Open-top cylindrical mold for casting tissue-mimicking material.

    Design rationale: a simple cup with a closed floor. The catheter rod
    drops into a 2 mm blind divot in the floor (analogous to the cyst-rod
    dimples) and is held vertical at its top by the registration lid's
    catheter through-hole; the bottom of the floor remains solid so no gel
    can leak out during pouring.

    Cyst variant adds blind dimples in the floor so cyst rods (4 mm PTFE or
    8 mm rigid plastic straws, depending on `cyst_rod_dia`) can be pressed
    in to stand vertically; they form the anechoic cylinders. When paired
    with `build_cyst_lid_mesh(cyst_rod_dia, cyst_radii_mm)`, every rod
    (catheter and cysts) is registered at both ends and held vertical
    without external clamping.

    When `add_orientation_key_slot=True`, a rectangular slot is cut into
    the mold rim at 0° (5.4 × 3.3 mm, full radial extent of the wall) so
    the matching tab on the printable lid mechanically keys into it; the
    lid cannot seat flush on the rim unless its tab is over the slot.
    """
    outer = _vert_cyl(MOLD_OD / 2, MOLD_HEIGHT)
    cup = _vert_cyl(MOLD_ID / 2, MOLD_HEIGHT - MOLD_FLOOR + 0.5,
                    z0=MOLD_FLOOR)
    # Catheter rod registration: blind divot in the floor (z=1.0 to MOLD_FLOOR)
    # + through-bore in the gel cavity (z=MOLD_FLOOR to MOLD_HEIGHT+0.5).
    # Bottom 1 mm of floor remains solid so no gel leaks during pouring.
    cath_bore = _vert_cyl(CATH_CHANNEL_DIA / 2, MOLD_HEIGHT - 0.5, z0=1.0)
    cavities: list[trimesh.Trimesh] = [cup, cath_bore]
    if add_orientation_key_slot:
        slot_w = KEY_W + 2 * KEY_AZI_CLEARANCE
        slot_h = KEY_H + KEY_VERT_CLEARANCE
        cavities.append(_orientation_key_mesh(
            width=slot_w,
            height=slot_h + 0.5,
            z0=MOLD_HEIGHT - slot_h,
            r_inner=MOLD_ID / 2 - 0.5,
            r_outer=MOLD_OD / 2 + 0.5,
        ))
    body = _diff(outer, _union(cavities))
    if with_cysts:
        dimple_h = MOLD_FLOOR - 1.0
        for r, ang_deg in zip(cyst_radii_mm, CYST_ANGLES_DEG):
            ang = np.deg2rad(ang_deg)
            sx = r * np.cos(ang)
            sy = r * np.sin(ang)
            dimple = _vert_cyl(cyst_rod_dia / 2 + 0.1, dimple_h,
                               z0=1.0, cx=sx, cy=sy, sections=32)
            body = _diff(body, dimple)
    body.metadata["name"] = "phantom_mold"
    return body


def build_cyst_lid_mesh(cyst_rod_dia: float,
                        cyst_radii_mm: tuple[float, ...]) -> trimesh.Trimesh:
    """Registration lid for the cyst-phantom mold.

    The lid is a flat disc with a centering lip on its underside that slip-
    fits into the mold ID, plus five through-holes (1 centre, 4 at the cyst
    design radii) sized to register the catheter rod and the cyst rods. A
    rectangular **orientation key tab** projects downward from the lid's
    underside at 0°, sized to slip-fit into the matching slot in the mold
    rim — mechanically enforcing the lid's rotational alignment so the rod
    holes register directly above the floor dimples.

    With both ends of each rod registered (lid hole at the top, floor
    divot/dimple at the bottom), the rods are mechanically forced vertical
    during the gel pour and set, with no external clamping needed.

    Print orientation: **top-surface-down** (lid lip and key tab pointing
    UP) so the wider top plate sits on the bed and the narrower features
    print on top of it with no overhang.
    """
    body = _vert_cyl(MOLD_OD / 2, LID_THICK, z0=0.0)
    lip = _vert_cyl((MOLD_ID - LID_LIP_CLEARANCE) / 2, LID_LIP_HEIGHT,
                    z0=-LID_LIP_HEIGHT)
    body = _union([body, lip])
    # Orientation key tab: downward-projecting rectangular block on the
    # underside at 0°, sized to slip-fit into the mold's rim slot.
    key_tab = _orientation_key_mesh(
        width=KEY_W,
        height=KEY_H,
        z0=-KEY_H,
        r_inner=MOLD_ID / 2,
        r_outer=MOLD_OD / 2,
    )
    body = _union([body, key_tab])
    holes: list[trimesh.Trimesh] = []
    holes.append(_vert_cyl(
        LID_CATH_HOLE_DIA / 2,
        LID_THICK + LID_LIP_HEIGHT + 2.0,
        z0=-LID_LIP_HEIGHT - 1.0,
        sections=24,
    ))
    cyst_hole_radius = cyst_rod_dia / 2 + LID_CYST_HOLE_RADIAL_CLEARANCE
    for r, ang_deg in zip(cyst_radii_mm, CYST_ANGLES_DEG):
        ang = np.deg2rad(ang_deg)
        sx = r * np.cos(ang)
        sy = r * np.sin(ang)
        holes.append(_vert_cyl(
            cyst_hole_radius,
            LID_THICK + LID_LIP_HEIGHT + 2.0,
            z0=-LID_LIP_HEIGHT - 1.0,
            cx=sx, cy=sy, sections=32,
        ))
    body = _diff(body, _union(holes))
    body.metadata["name"] = "cyst_phantom_lid"
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

    standoff = _finalize(build_printable_standoff_mesh())
    standoff_path = OUT / "wire_spiral_standoff.stl"
    standoff.export(standoff_path)
    print(f"  wrote {standoff_path}  ({standoff.vertices.shape[0]} verts, "
          f"{standoff.faces.shape[0]} faces)")

    asm = build_assembly_mesh()
    asm_path = OUT / "wire_spiral_assembly.stl"
    asm.export(asm_path)
    print(f"  wrote {asm_path}  (visual reference; do not print as one piece)")

    mold = _finalize(build_uniform_mold_mesh(with_cysts=False))
    mold_path = OUT / "phantom_mold_uniform.stl"
    mold.export(mold_path)
    print(f"  wrote {mold_path}")

    cyst = _finalize(build_uniform_mold_mesh(
        with_cysts=True,
        cyst_rod_dia=CYST_ROD_DIA,
        cyst_radii_mm=CYST_RADII_MM,
        add_orientation_key_slot=True,
    ))
    cyst_path = OUT / "phantom_mold_cyst.stl"
    cyst.export(cyst_path)
    print(f"  wrote {cyst_path}  (canonical: ⌀{CYST_ROD_DIA} mm cysts at "
          f"radii {tuple(CYST_RADII_MM)} mm)")

    cyst_lid = _finalize(build_cyst_lid_mesh(CYST_ROD_DIA, CYST_RADII_MM))
    cyst_lid_path = OUT / "phantom_mold_cyst_lid.stl"
    cyst_lid.export(cyst_lid_path)
    print(f"  wrote {cyst_lid_path}  (lid for canonical mold)")

    cyst_interim = _finalize(build_uniform_mold_mesh(
        with_cysts=True,
        cyst_rod_dia=CYST_ROD_DIA_INTERIM,
        cyst_radii_mm=CYST_RADII_MM_INTERIM,
        add_orientation_key_slot=True,
    ))
    cyst_interim_path = OUT / "phantom_mold_cyst_interim.stl"
    cyst_interim.export(cyst_interim_path)
    print(f"  wrote {cyst_interim_path}  (interim T1-E5*: "
          f"⌀{CYST_ROD_DIA_INTERIM} mm cysts at radii "
          f"{tuple(CYST_RADII_MM_INTERIM)} mm)")

    cyst_lid_interim = _finalize(build_cyst_lid_mesh(
        CYST_ROD_DIA_INTERIM, CYST_RADII_MM_INTERIM))
    cyst_lid_interim_path = OUT / "phantom_mold_cyst_lid_interim.stl"
    cyst_lid_interim.export(cyst_lid_interim_path)
    print(f"  wrote {cyst_lid_interim_path}  (lid for interim mold)")

    step = _finalize(build_step_mold_mesh())
    step_path = OUT / "phantom_step_mold.stl"
    step.export(step_path)
    print(f"  wrote {step_path}")

    print("\nWire-spiral fixture geometry (Visions PV .035 catheter):")
    print(f"  Disc OD: {DISC_OD} mm, thickness: {DISC_THICK} mm")
    print(f"  Center hole: ⌀{DISC_CENTER_HOLE} mm "
          f"(slip-fit on PV .035 catheter, OD = 1.9 mm)")
    print(f"  Wire holes: ⌀{WIRE_HOLE_DIA} mm × {N_WIRES} through-bores, "
          f"with ⌀{WIRE_CB_DIA} mm × {WIRE_CB_DEPTH} mm CA-cup counterbore "
          f"on top face")
    print(f"  Standoff holes: ⌀{STANDOFF_HOLE_DIA} mm × 3 on "
          f"⌀{STANDOFF_BCD} mm BCD at 60°/180°/300°")
    print(f"  Standoff option A (printable): 3× wire_spiral_standoff.stl")
    print(f"     shaft ⌀{PRINT_STANDOFF_SHAFT_OD} mm × {STANDOFF_LEN} mm + "
          f"tip pins ⌀{PRINT_STANDOFF_PIN_OD} mm × "
          f"{PRINT_STANDOFF_PIN_LEN} mm; ⌀{PRINT_STANDOFF_LOCK_HOLE} mm "
          f"cross-bore at each tip for safety-wire lock")
    print(f"  Standoff option B (commercial): 3× M3-F/M3-F standoffs, "
          f"⌀5 mm hex × {STANDOFF_LEN} mm + 6× M3 × 6 mm pan-head screws")
    print()
    print("Wire (r, θ) layout (12 wires, 1-turn spiral, focal-zone-dense):")
    for n, (x, y, th) in enumerate(_wire_positions()):
        r = np.hypot(x, y)
        print(f"  wire #{n+1:2d}:  r = {r:.2f} mm,  θ = {th:.0f}°")

    print()
    print("Cyst-phantom mold geometry (sized for 60 mm imaging diameter):")
    print(f"  Mold: OD {MOLD_OD} mm, ID {MOLD_ID} mm, height {MOLD_HEIGHT} mm, "
          f"floor {MOLD_FLOOR} mm")
    print(f"  Catheter channel: ⌀{CATH_CHANNEL_DIA} mm slip-fit on a 3 mm-"
          f"class channel-forming rod (wooden skewer, 3 mm PTFE, brass "
          f"tubing, ...); resulting ⌀3 mm gel channel accepts the Visions "
          f"PV .035 catheter (OD = 1.9 mm) with ~0.55 mm radial clearance")
    print(f"  Canonical cysts (PTFE / steel rods): "
          f"⌀{CYST_ROD_DIA} mm at radii {tuple(CYST_RADII_MM)} mm, "
          f"angles {tuple(CYST_ANGLES_DEG)}°")
    print(f"  Interim cysts (8 mm rigid plastic straws): "
          f"⌀{CYST_ROD_DIA_INTERIM} mm at radii "
          f"{tuple(CYST_RADII_MM_INTERIM)} mm, "
          f"angles {tuple(CYST_ANGLES_DEG)}°")
    print(f"  Lid: ⌀{MOLD_OD} mm × {LID_THICK} mm + ⌀"
          f"{MOLD_ID - LID_LIP_CLEARANCE:.1f} mm × {LID_LIP_HEIGHT} mm "
          f"centering lip; ⌀{LID_CATH_HOLE_DIA} mm catheter through-hole at "
          f"centre; rectangular orientation key tab ({KEY_W} × {KEY_H} mm) "
          f"on underside at 0° that slip-fits into matching slot in mold rim")


if __name__ == "__main__":
    main()
