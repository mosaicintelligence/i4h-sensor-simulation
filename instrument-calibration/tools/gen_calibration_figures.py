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

"""Generate schematic figures for the IVUS calibration / characterization protocol.

Produces one PNG per experiment in docs/calibration_figures/.

The Volcano s5i has a SOLID-STATE 64-element synthetic-aperture ring at the
catheter tip — the catheter does NOT rotate. Beams are emitted RADIALLY,
perpendicular to the catheter long axis, in a 360° plane. This drives the
geometry choices in these figures:

  * Cross-section views (looking down the catheter long axis): catheter
    appears as a small disc with a gold ring. Targets are placed at radial
    distances. Used by E1, E2, E4, E5, E6, E8.
  * Side views (catheter long axis horizontal): used by E3 because the
    elevation direction is along the catheter long axis.

Run from the repo root:

    PYTHONPATH=/tmp/calpkgs MPLCONFIGDIR=/tmp/mplcfg MPLBACKEND=Agg \\
        python3 tools/gen_calibration_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "calibration_figures"
OUT.mkdir(parents=True, exist_ok=True)

WATER = "#cfe8f3"
TANK = "#5b7a8a"
METAL = "#b6b6b6"
CATH = "#222222"
TISSUE = "#e8c9b8"
ANECHOIC = "#3a3a3a"
WIRE = "#ff8a3d"
ARROW = "#cc2a2a"
GOLD = "#d4a017"
LABEL_BG = dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5)


def _new_fig(title: str, w: float = 9.0, h: float = 5.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=12, weight="bold")
    return fig, ax


def _tank(ax, x0, y0, w, h, water_top=True):
    ax.add_patch(mp.Rectangle((x0, y0), w, h, fc=WATER, ec=TANK, lw=2))
    if water_top:
        ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=TANK, lw=2)
        for k in range(6):
            xk = x0 + 0.1 + k * (w - 0.2) / 5
            ax.plot([xk, xk + 0.15], [y0 + h - 0.05, y0 + h - 0.15],
                    color=TANK, lw=0.8)


def _ivus_catheter_xs(ax, x, y, R=0.18, label=None, tail_dir="left",
                      tail_len=2.0, show_fan=True, fan_radius=2.5):
    """Cross-section view of an IVUS catheter (looking down the long axis).

    Catheter body: dark disc. 64-element SA ring: gold annulus.
    Optional dashed tail indicates where the catheter shaft enters from
    off-screen (out of the imaging plane). Optional faint beam fan shows
    the 360° radial emission pattern.
    """
    if show_fan:
        for ang in np.linspace(0, 2 * np.pi, 64, endpoint=False):
            ax.plot([x, x + fan_radius * np.cos(ang)],
                    [y, y + fan_radius * np.sin(ang)],
                    color="#1f77b4", lw=0.3, alpha=0.18)
    if tail_dir == "left":
        ax.plot([x - tail_len, x - R], [y, y], color=CATH, lw=2,
                ls=(0, (3, 3)), solid_capstyle="round", alpha=0.5)
        ax.text(x - tail_len, y - 0.32, "catheter shaft\n(into page)",
                fontsize=7, ha="left", color="0.3", style="italic")
    ax.add_patch(mp.Circle((x, y), R, fc="#222222", ec="black", lw=0.8,
                           zorder=4))
    ax.add_patch(mp.Circle((x, y), R, fc="none", ec=GOLD, lw=2.4, zorder=5))
    if label:
        ax.text(x, y - R - 0.30, label, fontsize=8, ha="center", bbox=LABEL_BG)


def _ivus_catheter_side(ax, x_tip, y_axis, length=4.0, label=None):
    """Side view: catheter long axis horizontal, transducer band on the side.

    The imaging plane is a thin slab perpendicular to the page through the
    transducer band.
    """
    ax.plot([x_tip - length, x_tip], [y_axis, y_axis], color=CATH, lw=5,
            solid_capstyle="round", zorder=3)
    bx = x_tip - 0.55
    ax.add_patch(mp.Rectangle((bx - 0.30, y_axis - 0.18), 0.60, 0.36,
                              fc=GOLD, ec="black", lw=0.8, zorder=5))
    ax.add_patch(mp.FancyBboxPatch((bx - 0.65, y_axis - 1.65), 1.30, 0.10,
                                   boxstyle="round,pad=0.0",
                                   fc="#1f77b4", alpha=0.20, ec="none"))
    ax.add_patch(mp.FancyBboxPatch((bx - 0.65, y_axis + 1.55), 1.30, 0.10,
                                   boxstyle="round,pad=0.0",
                                   fc="#1f77b4", alpha=0.20, ec="none"))
    ax.plot([bx - 0.65, bx - 0.65], [y_axis - 1.55, y_axis + 1.55],
            color="#1f77b4", lw=0.8, ls=(0, (4, 3)), alpha=0.5)
    ax.plot([bx + 0.65, bx + 0.65], [y_axis - 1.55, y_axis + 1.55],
            color="#1f77b4", lw=0.8, ls=(0, (4, 3)), alpha=0.5)
    ax.text(bx, y_axis + 1.85, "imaging plane\n(⊥ to catheter axis)",
            fontsize=7, ha="center", color="#1f77b4", style="italic")
    if label:
        ax.text(x_tip - length + 0.1, y_axis + 0.40, label, fontsize=8,
                ha="left", bbox=LABEL_BG)


def _arrow(ax, x0, y0, x1, y1, color=ARROW, lw=1.4, text=None,
           text_offset=(0, 0.1), style="<->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    if text:
        ax.text((x0 + x1) / 2 + text_offset[0],
                (y0 + y1) / 2 + text_offset[1],
                text, fontsize=8, ha="center", va="center", color=color,
                bbox=LABEL_BG)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_e1_pulse_echo():
    fig, ax = _new_fig(
        "E1 — Pulse-Echo Capture (cross-section view; flat reflector parallel to catheter axis)",
        w=9.5, h=6.5,
    )
    _tank(ax, 0, 0, 9, 5.2)
    cx, cy = 3.0, 2.6
    _ivus_catheter_xs(ax, cx, cy, R=0.18, label="IVUS catheter\n(end-on)",
                      tail_dir="left", tail_len=2.5, show_fan=True,
                      fan_radius=3.0)
    d = 3.0
    plate_x = cx + d
    ax.add_patch(mp.Rectangle((plate_x - 0.05, cy - 1.6), 0.20, 3.2,
                              fc=METAL, ec="black", lw=1.5, zorder=6))
    ax.text(plate_x + 0.25, cy + 1.7,
            "Polished flat reflector\n(parallel to catheter long axis)\n"
            "λ/4 flat, 25 × 25 × 6 mm",
            fontsize=8, ha="left", bbox=LABEL_BG)
    _arrow(ax, cx + 0.18, cy, plate_x - 0.05, cy,
           text=f"d ≈ {d:.1f} mm (radial)\n±0.05 mm")
    for ang in np.deg2rad(np.linspace(-15, 15, 11)):
        x1 = cx + 2.95 * np.cos(ang)
        y1 = cy + 2.95 * np.sin(ang)
        ax.plot([cx, x1], [cy, y1], color="C3", lw=0.6, alpha=0.55)
    ax.add_patch(mp.Wedge((cx, cy), 2.95, -15, 15, fc="C3", alpha=0.07,
                          ec="none"))
    ax.text(cx + 1.5, cy - 1.7, "A-lines analyzed:\n±15° around plate normal",
            fontsize=8, ha="center", color="C3", bbox=LABEL_BG)
    ax.text(0.3, 0.4, "Degassed water, 22 ± 1 °C\n(c_water = 1488 m/s)",
            fontsize=8, bbox=LABEL_BG)
    ax.add_patch(mp.FancyBboxPatch((0.2, 5.5), 4.2, 0.8,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(2.3, 5.9,
            "Console: TGC flat • Acoustic Reference OFF\n"
            "Gain set so flat-plate echo ≈ 80% saturation",
            fontsize=8, ha="center", va="center")
    ax.add_patch(mp.FancyBboxPatch((4.7, 5.5), 4.6, 0.8,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(7.0, 5.9,
            "RF tap → DAQ (≥ 100 MS/s, ≥ 12-bit)\n"
            "or service-mode raw RF stream",
            fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.4, 6.5)
    _save(fig, "fig_e1_pulse_echo.png")


def fig_e2_wire_psf():
    fig, ax = _new_fig(
        "E2 — Spiral Wire-Phantom (cross-section view; samples PSF at multiple azimuths × radii)",
        w=10.0, h=7.5,
    )
    _tank(ax, 0, 0, 9, 6)
    cx, cy = 4.5, 3.0
    _ivus_catheter_xs(ax, cx, cy, R=0.18, label="IVUS catheter\n(end-on)",
                      tail_dir="left", tail_len=2.7, show_fan=True,
                      fan_radius=3.5)
    n_wires = 12
    radii = np.linspace(1.0, 3.8, n_wires)
    angles = np.linspace(0, 2 * np.pi * 1.6, n_wires)
    for r, a in zip(radii, angles):
        wx = cx + r * np.cos(a)
        wy = cy + r * np.sin(a)
        ax.add_patch(mp.Circle((wx, wy), 0.07, fc=WIRE, ec="black", lw=0.5,
                               zorder=6))
        ax.plot([cx, wx], [cy, wy], color=WIRE, lw=0.5, ls=":", alpha=0.5)
    spiral_a = np.linspace(0, 2 * np.pi * 1.6, 200)
    spiral_r = np.linspace(1.0, 3.8, 200)
    ax.plot(cx + spiral_r * np.cos(spiral_a),
            cy + spiral_r * np.sin(spiral_a),
            color=WIRE, lw=0.7, alpha=0.4)
    ax.text(cx + 4.1, cy + 0.7,
            "12 × 25 µm tungsten wires\nparallel to catheter long axis\n"
            "(wires go INTO the page)",
            fontsize=8, ha="left", color="0.2", bbox=LABEL_BG)
    a_call = angles[5]
    r_call = radii[5]
    wx = cx + r_call * np.cos(a_call)
    wy = cy + r_call * np.sin(a_call)
    ax.annotate(f"wire #{5+1}\nz={r_call:.2f} mm\nθ={np.degrees(a_call) % 360:.0f}°",
                xy=(wx, wy), xytext=(cx + 4.2, cy + 2.2),
                fontsize=7, ha="left", color="0.2", bbox=LABEL_BG,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7))
    ax.text(0.3, 0.4,
            "Spiral chosen so each wire occupies\na unique (range, azimuth) cell.\n"
            "One frame = 12 PSF samples.",
            fontsize=8, bbox=LABEL_BG)
    ax.add_patch(mp.FancyBboxPatch((0.2, 6.3), 4.0, 1.0,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(2.2, 6.8,
            "30 frames at static position →\nrepeat at 4 catheter rotations (0/90/180/270°)\n"
            "to average across all 64 elements",
            fontsize=8, ha="center", va="center")
    ax.add_patch(mp.FancyBboxPatch((4.4, 6.3), 5.4, 1.0,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(7.1, 6.8,
            "Per-wire 2D PSF → lateral_FWHM(z), axial_FWHM(z) →\n"
            "focal_length, element_radius (Gaussian-beam fit)\n"
            "+ depth-binned lateral-PSF table for the simulator",
            fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.4, 10.0)
    ax.set_ylim(-0.4, 7.4)
    _save(fig, "fig_e2_wire_psf.png")


def fig_e3_slice_thickness():
    fig, ax = _new_fig(
        "E3 — Slice-Thickness (side view; elevation = catheter long axis)",
        w=9.5, h=5.5,
    )
    ax.add_patch(mp.Rectangle((0, 0), 9, 4.6, fc=WATER, ec=TANK, lw=2))
    cy = 2.3
    _ivus_catheter_side(ax, x_tip=6.5, y_axis=cy, length=5.6,
                        label="IVUS catheter (long axis = elevation)")
    band_x = 6.5 - 0.55
    bead_y = cy + 1.4
    ax.add_patch(mp.Circle((band_x, bead_y), 0.10, fc=WIRE, ec="black",
                           lw=0.6, zorder=6))
    for dx in np.linspace(-1.2, 1.2, 11):
        if abs(dx) < 0.05:
            continue
        ax.add_patch(mp.Circle((band_x + dx, bead_y), 0.05, fc=WIRE,
                               ec="black", lw=0.4, alpha=0.55, zorder=5))
    ax.annotate("", xy=(band_x - 1.5, bead_y), xytext=(band_x + 1.5, bead_y),
                arrowprops=dict(arrowstyle="<->", color=ARROW, lw=1.4))
    ax.text(band_x, bead_y + 0.30,
            "Δy ∈ [−2, +2] mm along catheter axis\nstep 0.1 mm (3-axis stage)",
            fontsize=8, ha="center", color=ARROW, bbox=LABEL_BG)
    ax.text(band_x, bead_y - 0.50,
            "0.5 mm steel bead OR 25 µm wire\n(initial Δy = 0 = imaging plane)",
            fontsize=7, ha="center", bbox=LABEL_BG)
    _arrow(ax, band_x - 1.8, cy - 0.8, band_x - 1.8, cy + 0.8,
           text="imaging\nplane", color="#1f77b4",
           text_offset=(-0.7, 0), style="-|>")
    _arrow(ax, band_x - 0.5, cy - 1.1, band_x + 1.7, cy - 1.1,
           text="elevation\n(catheter long axis)", color=ARROW,
           text_offset=(0, -0.30), style="-|>")
    ax.text(0.3, 0.3,
            "Side view: imaging plane goes vertically through the transducer band;\n"
            "elevation extends horizontally along the catheter shaft.",
            fontsize=8, bbox=LABEL_BG)
    ax.add_patch(mp.FancyBboxPatch((0.2, 4.8), 8.6, 0.8,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(4.5, 5.2,
            "Record peak A-line amplitude at the bead's azimuth vs Δy → "
            "elevational FWHM = elevational_height_mm",
            fontsize=9, ha="center", va="center")
    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(-0.4, 5.7)
    _save(fig, "fig_e3_slice_thickness.png")


def fig_e4_attenuation_phantom():
    fig, ax = _new_fig(
        "E4 — Uniform Attenuation Phantom (cross-section view; TGC + α fit)",
        w=9.0, h=6.0,
    )
    ax.add_patch(mp.Rectangle((0, 0), 8, 5, fc=TISSUE, ec="0.3", lw=2))
    ax.text(0.3, 4.6, "CIRS 040GSE / ATS 539 (uniform, α known)",
            fontsize=8, bbox=LABEL_BG)
    cx, cy = 3.0, 2.5
    _ivus_catheter_xs(ax, cx, cy, R=0.16, label=None, tail_dir="left",
                      tail_len=2.4, show_fan=False)
    th = np.deg2rad(np.linspace(0, 360, 200))
    for r in (0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2):
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                color="#1f77b4", lw=0.5, alpha=0.4)
    ax.annotate("", xy=(cx + 4.4, cy), xytext=(cx + 0.2, cy),
                arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.2))
    ax.text(cx + 2.3, cy + 0.30, "radial depth z (mm)",
            fontsize=8, color=ARROW, bbox=LABEL_BG)
    inset = fig.add_axes([0.62, 0.18, 0.32, 0.30])
    z = np.linspace(0, 10, 200)
    raw = -1.2 * z
    flat = np.zeros_like(z)
    inset.plot(z, raw, "C3", lw=1.4, label="device output\n(no TGC)")
    inset.plot(z, flat, "C2", lw=1.4, label="target after TGC")
    inset.set_xlabel("depth z (mm)", fontsize=8)
    inset.set_ylabel("mean intensity (dB)", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.legend(fontsize=7, loc="lower left")
    inset.set_title("residual_dB(z) → TGC control points", fontsize=8)
    ax.add_patch(mp.FancyBboxPatch((0.2, 5.2), 7.6, 0.8,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(4.0, 5.6,
            "Console: gain = default • TGC sliders centered • Acoustic Ref ON",
            fontsize=9, ha="center", va="center")
    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(-0.3, 6.2)
    _save(fig, "fig_e4_attenuation_phantom.png")


def fig_e5_cyst_phantom():
    fig, ax = _new_fig(
        "E5 — Cyst Phantom (cross-section view; speckle, noise, reject)",
        w=9.5, h=6.0,
    )
    ax.add_patch(mp.Rectangle((0, 0), 9, 5, fc=TISSUE, ec="0.3", lw=2))
    cx, cy = 2.4, 2.5
    _ivus_catheter_xs(ax, cx, cy, R=0.16, label=None, tail_dir="left",
                      tail_len=2.0, show_fan=False)
    ax.add_patch(mp.Circle((6.0, 2.5), 0.9, fc=ANECHOIC, ec="0.2", lw=1))
    ax.text(6.0, 1.4, "anechoic cyst\n(ROI-A)", fontsize=8, ha="center",
            bbox=LABEL_BG)
    ax.add_patch(mp.Rectangle((3.4, 3.2), 1.0, 0.7, fc="none", ec="C2", lw=2,
                              ls="--"))
    ax.text(3.9, 4.05, "uniform tissue\n(ROI-T)", fontsize=8, ha="center",
            color="C2", bbox=LABEL_BG)
    th = np.deg2rad(np.linspace(0, 360, 200))
    for r in (0.5, 1.0, 1.5, 2.5, 3.5, 4.5):
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                color="#1f77b4", lw=0.4, alpha=0.3)
    ax.add_patch(mp.FancyBboxPatch((0.2, 5.2), 8.6, 0.8,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(4.5, 5.6,
            "ROI-A → noise σ, reject • ROI-T → speckle autocorr "
            "(scattering_resolution_mm), envelope hist (μ0/μ1/σ)",
            fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(-0.3, 6.2)
    _save(fig, "fig_e5_cyst_phantom.png")


def fig_e6_ringdown():
    fig, ax = _new_fig(
        "E6 — Ring-Down Capture (cross-section view; Acoustic Reference)",
        w=9.0, h=6.0,
    )
    _tank(ax, 0, 0, 8, 5)
    cx, cy = 4.0, 2.5
    _ivus_catheter_xs(ax, cx, cy, R=0.18, label=None, tail_dir="left",
                      tail_len=3.5, show_fan=True, fan_radius=3.0)
    ring_R = 1.1
    ax.add_patch(mp.Circle((cx, cy), ring_R, fc="#ffd6d6", ec="#cc2a2a",
                           lw=1.4, alpha=0.55, zorder=3))
    ax.text(cx, cy - 1.50, "near-field ring-down\nzone (~1–2 mm radial)",
            fontsize=8, ha="center", color="#cc2a2a", bbox=LABEL_BG)
    ax.add_patch(mp.Circle((cx, cy), 3.0, fc="none", ec=ARROW, lw=0.8,
                           ls=":"))
    ax.annotate("", xy=(cx + 3.0, cy - 0.05),
                xytext=(cx + 3.5, cy - 0.05),
                arrowprops=dict(arrowstyle="->", color=ARROW))
    ax.text(cx + 3.5, cy, "≥ 30 mm clear\nto any wall",
            fontsize=8, ha="left", color=ARROW, bbox=LABEL_BG)
    ax.add_patch(mp.FancyBboxPatch((0.2, 5.2), 7.6, 0.8,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(4.0, 5.6,
            "Console: Acoustic Reference OFF • gain = max (68) • TGC flat • 30 frames",
            fontsize=9, ha="center", va="center")
    ax.set_xlim(-0.3, 9.0)
    ax.set_ylim(-0.3, 6.2)
    _save(fig, "fig_e6_ringdown.png")


def fig_e7_grayscale():
    fig, ax = _new_fig("E7 — Grayscale / Compression Calibration", w=9.0,
                       h=5.5)
    ax.add_patch(mp.FancyBboxPatch((0.2, 1.8), 2.2, 1.6,
                                   boxstyle="round,pad=0.1",
                                   fc="white", ec="0.4"))
    ax.text(1.3, 3.0, "RF generator\n(amplitude sweep)", fontsize=8,
            ha="center")
    ax.text(1.3, 2.2, "−80…0 dB, 5 dB step", fontsize=7, ha="center",
            color="0.3")
    ax.add_patch(mp.FancyBboxPatch((3.0, 1.8), 2.2, 1.6,
                                   boxstyle="round,pad=0.1",
                                   fc="#fff3b0", ec="0.4"))
    ax.text(4.1, 3.0, "Volcano s5i\nfront-end RF input", fontsize=8,
            ha="center")
    ax.text(4.1, 2.2, "(or step-target phantom)", fontsize=7, ha="center",
            color="0.3")
    ax.add_patch(mp.FancyBboxPatch((5.8, 1.8), 2.2, 1.6,
                                   boxstyle="round,pad=0.1",
                                   fc="#d8f3dc", ec="0.4"))
    ax.text(6.9, 3.0, "Display capture\n(DVI/HDMI grabber)", fontsize=8,
            ha="center")
    ax.text(6.9, 2.2, "ROI mean grayscale", fontsize=7, ha="center",
            color="0.3")
    for x0, x1 in [(2.4, 3.0), (5.2, 5.8)]:
        ax.annotate("", xy=(x1, 2.6), xytext=(x0, 2.6),
                    arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.4))
    ax.add_patch(mp.FancyBboxPatch((0.2, 0.1), 7.8, 1.4,
                                   boxstyle="round,pad=0.1", fc="white",
                                   ec="0.4"))
    rf = np.linspace(-80, 0, 200)
    gs = np.clip(11 + (rf + 80) / 80 * 228, 0, 239)
    gs = np.where(rf < -75, 0, gs)
    inset = fig.add_axes([0.10, 0.10, 0.80, 0.20])
    inset.plot(rf, gs, "C0", lw=1.5)
    inset.set_xlabel("RF amplitude (dB)", fontsize=8)
    inset.set_ylabel("displayed grayscale", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_title("Output: compression LUT  →  log_multiplier, log_floor, "
                    "dynamic_range_db",
                    fontsize=8)
    ax.add_patch(mp.FancyBboxPatch((0.2, 4.0), 7.8, 0.8,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(4.1, 4.4,
            "Repeat for gain slider = 0, 20, 50, 68 → gain_db calibration curve",
            fontsize=9, ha="center", va="center")
    ax.set_xlim(0, 8.2)
    ax.set_ylim(-1.5, 5.2)
    _save(fig, "fig_e7_grayscale.png")


def fig_e8_tissue():
    fig, ax = _new_fig(
        "E8 — Tissue / Material Fit (cross-section view; ex vivo or layered phantom)",
        w=9.5, h=6.0,
    )
    ax.add_patch(mp.Rectangle((0, 0), 9, 5, fc="#e6f3fb", ec="0.3", lw=1.5))
    ax.text(0.3, 4.6, "Saline bath, 37 ± 1 °C", fontsize=8, bbox=LABEL_BG)
    cx, cy = 4.5, 2.5
    ax.add_patch(mp.Circle((cx, cy), 2.0, fc=TISSUE, ec="0.3", lw=1.5))
    ax.add_patch(mp.Circle((cx, cy), 1.6, fc="#d8a98c", ec="0.3", lw=1.0))
    ax.add_patch(mp.Circle((cx, cy), 0.9, fc="#f4e2cf", ec="0.3", lw=1.0))
    ax.add_patch(mp.Wedge((cx, cy), 1.4, 30, 80, fc="#cccccc", ec="0.3",
                          lw=0.8))
    ax.text(cx + 1.2, cy + 1.1, "calcium\nplaque", fontsize=7, ha="left",
            color="0.2", bbox=LABEL_BG)
    ax.add_patch(mp.Wedge((cx, cy), 1.3, 200, 240, fc="#f5d977", ec="0.3",
                          lw=0.8))
    ax.text(cx - 1.5, cy - 1.1, "lipid pool", fontsize=7, ha="right",
            color="0.4", bbox=LABEL_BG)
    ax.text(cx + 0.30, cy + 0.05, "lumen", fontsize=8, ha="center",
            color="0.3")
    ax.text(cx + 2.0, cy, "media /\nadventitia", fontsize=7, ha="left",
            color="0.3")
    _ivus_catheter_xs(ax, cx, cy, R=0.10, label=None, tail_dir="left",
                      tail_len=4.0, show_fan=False)
    ax.add_patch(mp.FancyBboxPatch((0.2, 5.2), 8.6, 0.8,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(4.5, 5.6,
            "Co-register IVUS RF with µCT / histology → "
            "per-material Z, c, α, μ0/μ1/σ, specularity",
            fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(-0.3, 6.2)
    _save(fig, "fig_e8_tissue.png")


def fig_e9_timing():
    fig, ax = _new_fig("E9 — Acquisition Timing Log (deferred layer)", w=9.0,
                       h=5.0)
    boxes = [
        (0.5, 2.0, "Volcano s5i\nback panel\n(sync / RF tap)", "#fff3b0"),
        (3.5, 2.0, "Trigger probe\n+ DAQ\n(≥ 1 GS/s)", "white"),
        (6.5, 2.0, "PC: timestamp\nlog (ns)", "#d8f3dc"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(mp.FancyBboxPatch((x, y), 2.2, 1.6,
                                       boxstyle="round,pad=0.1",
                                       fc=c, ec="0.4"))
        ax.text(x + 1.1, y + 0.8, t, fontsize=8, ha="center", va="center")
    for xa, xb in [(2.7, 3.5), (5.7, 6.5)]:
        ax.annotate("", xy=(xb, 2.8), xytext=(xa, 2.8),
                    arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.4))
    inset = fig.add_axes([0.10, 0.10, 0.80, 0.22])
    t = np.linspace(0, 5e-3, 1000)
    pulse = np.zeros_like(t)
    prf = 7700.0
    for k in range(int(prf * 5e-3)):
        pulse[(np.abs(t - k / prf)).argmin()] = 1
    inset.plot(t * 1e3, pulse, "C0", lw=1.0)
    inset.set_xlabel("time (ms)", fontsize=8)
    inset.set_ylabel("trigger", fontsize=8)
    inset.set_title("Δt between triggers → PRF, scanline rate, ADC fs",
                    fontsize=8)
    inset.tick_params(labelsize=7)
    ax.set_xlim(0, 9)
    ax.set_ylim(-1.5, 4.5)
    _save(fig, "fig_e9_timing.png")


MILK = "#fff3d6"
MILK_GEL = "#f5e8c8"


def fig_t1_e4i_milk_tank():
    """Interim T1-E4* — uniform-fluid (evaporated milk) stand-in for E4.

    Cross-section view (catheter long axis into page). The "tank" here is a
    tall glass beaker / jar filled with Carnation-type evaporated milk;
    the catheter is suspended vertically at the center.

    The figure shows the **Phase B1** configuration (E2 wire phantom
    submerged in **undiluted** milk for c refinement). The same setup
    is repeated in 1:1 diluted milk for Phase B2; Phase A1/A2 use the
    same fill with the wire phantom removed.
    """
    fig, ax = _new_fig(
        "T1-E4* — Uniform-fluid milk tank + wire-phantom c-cal "
        "(interim stand-in for E4; cross-section view, Phase B1 shown)",
        w=11.0, h=6.8,
    )
    bx, by, BR = 5.4, 2.7, 2.4
    ax.add_patch(mp.Circle((bx, by), BR + 0.18, fc="#dcdcdc", ec="0.35",
                           lw=2.0, zorder=1))
    ax.add_patch(mp.Circle((bx, by), BR, fc=MILK, ec="0.5", lw=1.0, zorder=2))
    ax.text(bx - BR - 0.20, by + BR + 0.08,
            "Glass beaker / jar\n(≥ 1 L; ID ≥ 90 mm)",
            fontsize=8, ha="left", bbox=LABEL_BG)
    ax.text(bx + 1.55, by - 1.55,
            "Phase A1/B1: undiluted\nevap milk (~25 % solids)\n"
            "Phase A2/B2: 1:1 diluted\nwith degassed water",
            fontsize=8, ha="left", color="0.25", bbox=LABEL_BG)
    _ivus_catheter_xs(ax, bx, by, R=0.18, label=None, tail_dir="left",
                      tail_len=3.8, show_fan=True, fan_radius=2.0)
    ax.text(bx - 0.05, by - 0.42, "IVUS catheter\n(end-on)", fontsize=8,
            ha="center", bbox=LABEL_BG)
    # E2 wire phantom: canonical 12-wire Archimedean spiral
    # at (r mm, θ°) per ivus_calibration_protocol.md § E2.
    # Scale: imaging fan radius = 2.0 fu = 30 mm imaging radius, so
    # 1 fu ≈ 15 mm and 1 mm ≈ 0.0667 fu.
    mm_per_fu = 15.0
    spiral_layout = [
        (4, 0),    (6, 30),   (8, 60),   (10, 90),
        (12, 120), (14, 150), (16, 180), (18, 210),
        (20, 240), (22, 270), (24, 300), (26, 330),
    ]
    for r_mm, a_deg in spiral_layout:
        r_fu = r_mm / mm_per_fu
        a_rad = np.deg2rad(a_deg)
        wx = bx + r_fu * np.cos(a_rad)
        wy = by + r_fu * np.sin(a_rad)
        ax.add_patch(mp.Circle((wx, wy), 0.05, fc=WIRE, ec="black",
                               lw=0.5, zorder=6))
        ax.plot([bx, wx], [by, wy], color=WIRE, lw=0.4, ls=":", alpha=0.4)
    # Single combined annotation for the wire phantom
    a_anchor = np.deg2rad(120)
    r_anchor = 12.0 / mm_per_fu
    ax.annotate(
        "E2 wire phantom (Phases B1, B2):\n"
        "canonical 12-wire Archimedean spiral,\n"
        "radii r ∈ {4, 6, 8, 10, 12, 14, 16, 18,\n"
        "20, 22, 24, 26} mm, 30° azimuthal step,\n"
        "all wires parallel to catheter long axis\n"
        "(into the page). Phase B gain operator-set\n"
        "per wire-visibility rule (~25–35 for tungsten,\n"
        "~50 for nylon).",
        xy=(bx + r_anchor * np.cos(a_anchor),
            by + r_anchor * np.sin(a_anchor)),
        xytext=(bx + 2.5, by + 1.5),
        fontsize=7, ha="left", color="0.2", bbox=LABEL_BG,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7),
    )
    _arrow(ax, bx + 0.18, by - 1.05, bx + BR - 0.05, by - 1.05,
           text="≥ 30 mm\nto wall")
    ax.text(0.3, 0.45,
            "Phase A1/B1 — undiluted Carnation evap milk @ 22 °C:\n"
            "  α ≈ 0.8 ± 0.15 dB/cm/MHz (Antoniou 2021),\n"
            "  c ≈ 1565 ± 10 m/s, Z ≈ 1.58 MRayl.\n"
            "Phase A2/B2 — 1:1 diluted with degassed distilled water:\n"
            "  α ≈ 0.4 ± 0.10 dB/cm/MHz (Farrer 2015 interp.),\n"
            "  c ≈ 1525 ± 10 m/s, Z ≈ 1.53 MRayl.\n"
            "Cross-check: α₁/α₂ ≈ 2.0; c₁−c₂ ≈ 40 m/s.",
            fontsize=7.5, bbox=LABEL_BG)
    ax.add_patch(mp.FancyBboxPatch((0.2, 5.7), 4.7, 1.0,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(2.55, 6.20,
            "Console: TGC sliders centered • Acoustic Reference ON\n"
            "• imaging diameter = 60 mm\n"
            "• Phase A gain slider stepped through {20, 35, 50, 68}\n"
            "• Phase B gain operator-set (~25–35 for tungsten)",
            fontsize=7.5, ha="center", va="center")
    ax.add_patch(mp.FancyBboxPatch((5.1, 5.45), 5.7, 1.25,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(7.95, 6.07,
            "Phase A1 (undiluted, no wire): 180 frames\n"
            "  (4 gains @ P1 + gain 50 @ P2, P3)\n"
            "Phase B1 (undiluted + wire phantom, shown): 30 frames\n"
            "  [fluid swap: drain → rinse → 250 mL milk + 250 mL water]\n"
            "Phase A2 (diluted, no wire): 180 frames (same as A1)\n"
            "Phase B2 (diluted + wire phantom): 30 frames (same as B1)",
            fontsize=7.0, ha="center", va="center")
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.3, 6.9)
    _save(fig, "fig_t1_e4i_milk_tank.png")


def fig_t1_e5i_milk_agar_cyst():
    """Interim T1-E5* — agar + evaporated milk gel with 4 cyst voids."""
    fig, ax = _new_fig(
        "T1-E5* — Agar + evaporated-milk cyst phantom (interim stand-in for E5; cross-section view)",
        w=10.5, h=7.4,
    )
    gel_x0, gel_y0, gel_w, gel_h = 0.6, 0.3, 9.3, 6.0
    ax.add_patch(mp.Rectangle((gel_x0, gel_y0), gel_w, gel_h, fc=MILK_GEL,
                              ec="0.35", lw=2))
    ax.text(gel_x0 + 0.25, gel_y0 + gel_h - 0.30,
            "Agar 3 % + evaporated milk 50 % (gel matrix is\n"
            "milky-translucent — cyst voids verifiable optically)",
            fontsize=8, bbox=LABEL_BG)
    cx, cy = 5.0, 3.3
    _ivus_catheter_xs(ax, cx, cy, R=0.14, label=None, tail_dir="left",
                      tail_len=3.8, show_fan=False)
    ax.text(cx, cy - 0.40, "IVUS catheter", fontsize=8, ha="center",
            bbox=LABEL_BG)
    scale = 0.13
    th = np.deg2rad(np.linspace(0, 360, 200))
    for r_mm in (5, 10, 15, 20, 25):
        r_fig = r_mm * scale
        ax.plot(cx + r_fig * np.cos(th), cy + r_fig * np.sin(th),
                color="#1f77b4", lw=0.35, alpha=0.22)
    cyst_R = 2.0 * scale
    cyst_targets = [
        (10.0, 0.0, "ROI-A", (cx + 1.0 * scale * 10 + 0.6,
                              cy + 0.9)),
        (14.0, 90.0, None, None),
        (14.0, 180.0, None, None),
        (18.0, 270.0, None, None),
    ]
    for r_mm, theta_deg, roi, label_xy in cyst_targets:
        r_fig = r_mm * scale
        ang = np.deg2rad(theta_deg)
        ccx = cx + r_fig * np.cos(ang)
        ccy = cy + r_fig * np.sin(ang)
        ax.add_patch(mp.Circle((ccx, ccy), cyst_R, fc=ANECHOIC, ec="0.2",
                               lw=1.0, zorder=4))
        ax.plot([cx, ccx], [cy, ccy], color="0.4", lw=0.6, ls=":", alpha=0.55)
        if theta_deg == 0:
            tx, ty = ccx + cyst_R + 0.10, ccy + 0.05
            ha = "left"
        elif theta_deg == 90:
            tx, ty = ccx, ccy + cyst_R + 0.30
            ha = "center"
        elif theta_deg == 180:
            tx, ty = ccx - cyst_R - 0.10, ccy + 0.05
            ha = "right"
        else:
            tx, ty = ccx, ccy - cyst_R - 0.32
            ha = "center"
        ax.text(tx, ty, f"⌀4 mm cyst\n(r = {r_mm:.0f} mm)",
                fontsize=7, ha=ha, va="center", color="0.15", bbox=LABEL_BG)
        if roi:
            ax.annotate(roi,
                        xy=(ccx + cyst_R * 0.4, ccy - cyst_R * 0.4),
                        xytext=(label_xy[0], label_xy[1]),
                        fontsize=8, ha="left", color="#cc2a2a",
                        bbox=LABEL_BG,
                        arrowprops=dict(arrowstyle="->", color="#cc2a2a",
                                        lw=0.8))
    roi_t_x = cx + 0.55 * scale * 10
    roi_t_y = cy + 0.55 * scale * 10
    ax.add_patch(mp.Rectangle((roi_t_x, roi_t_y), 0.8, 0.55, fc="none",
                              ec="C2", lw=1.6, ls="--"))
    ax.text(roi_t_x - 0.10, roi_t_y + 0.27, "ROI-T",
            fontsize=8, ha="right", va="center", color="C2", bbox=LABEL_BG)
    ax.text(gel_x0 + 0.25, gel_y0 + 0.30,
            "Cyst voids formed by PTFE / steel / dowel rods at\n"
            "design radii 10, 14, 14, 18 mm; rods withdrawn after set.",
            fontsize=8, bbox=LABEL_BG)
    ax.add_patch(mp.FancyBboxPatch((0.2, 6.55), 4.6, 0.75,
                                   boxstyle="round,pad=0.1", fc="#fff3b0",
                                   ec="0.4"))
    ax.text(2.5, 6.93,
            "Console: gain ∈ {20, 50, 68} • TGC centered\n"
            "Acoustic Reference ON • imaging diameter = 60 mm",
            fontsize=8, ha="center", va="center")
    ax.add_patch(mp.FancyBboxPatch((5.0, 6.55), 5.2, 0.75,
                                   boxstyle="round,pad=0.1", fc="#d8f3dc",
                                   ec="0.4"))
    ax.text(7.6, 6.93,
            "ROI-A → noise σ, reject  •  ROI-T → speckle autocorr\n"
            "(scattering_resolution_mm), envelope hist (μ0/μ1/σ)",
            fontsize=8, ha="center", va="center")
    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(-0.2, 7.4)
    _save(fig, "fig_t1_e5i_milk_agar_cyst.png")


if __name__ == "__main__":
    for fn in (fig_e1_pulse_echo, fig_e2_wire_psf, fig_e3_slice_thickness,
              fig_e4_attenuation_phantom, fig_e5_cyst_phantom,
              fig_e6_ringdown, fig_e7_grayscale, fig_e8_tissue,
              fig_e9_timing, fig_t1_e4i_milk_tank, fig_t1_e5i_milk_agar_cyst):
        fn()
        print(f"  wrote {fn.__name__}")
    print(f"All figures written to {OUT}")
