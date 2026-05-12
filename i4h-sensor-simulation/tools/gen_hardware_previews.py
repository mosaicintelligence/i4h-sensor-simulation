"""Render isometric preview PNGs of the printable phantom hardware."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1] / "docs" / "calibration_figures"
OUT.mkdir(parents=True, exist_ok=True)
HW = Path(__file__).resolve().parents[1] / "hardware"


def _render_disc_combo(out_path: Path) -> None:
    """Top view + 3D iso view of the wire-spiral disc, side by side."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_phantom_stl import (
        _wire_positions, DISC_OD, DISC_THICK, DISC_CENTER_HOLE,
        WIRE_HOLE_DIA, STANDOFF_HOLE_DIA, STANDOFF_BCD,
        STANDOFF_ANGLES_DEG, N_WIRES, WIRE_LAYOUT_MM,
    )
    wire_radii = [r for r, _ in WIRE_LAYOUT_MM]
    wire_thetas_deg = [th for _, th in WIRE_LAYOUT_MM]
    R_MIN_MM = min(wire_radii)
    R_MAX_MM = max(wire_radii)
    fig = plt.figure(figsize=(11, 5.0), dpi=170)

    ax1 = fig.add_subplot(121)
    ax1.set_aspect("equal")
    R = DISC_OD / 2
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.fill(R * np.cos(th), R * np.sin(th), facecolor="#e9dca0",
             edgecolor="#5a4a20", linewidth=1.4, zorder=1)
    ax1.fill((DISC_CENTER_HOLE / 2) * np.cos(th),
             (DISC_CENTER_HOLE / 2) * np.sin(th),
             facecolor="white", edgecolor="#5a4a20", linewidth=1.0, zorder=2)
    for x, y, _ in _wire_positions():
        ax1.fill(x + (WIRE_HOLE_DIA / 2) * np.cos(th),
                 y + (WIRE_HOLE_DIA / 2) * np.sin(th),
                 facecolor="black", zorder=4)
    for ang_deg in STANDOFF_ANGLES_DEG:
        ang = np.deg2rad(ang_deg)
        sx = STANDOFF_BCD / 2 * np.cos(ang)
        sy = STANDOFF_BCD / 2 * np.sin(ang)
        ax1.fill(sx + (STANDOFF_HOLE_DIA / 2) * np.cos(th),
                 sy + (STANDOFF_HOLE_DIA / 2) * np.sin(th),
                 facecolor="#888", edgecolor="#333", linewidth=0.8, zorder=4)
        ax1.text(sx * 0.78, sy * 0.78, "M3", fontsize=7, ha="center",
                 va="center", color="#333")
    ax1.text(0, -R - 2.5,
             f"⌀{DISC_OD:g} mm × {DISC_THICK:g} mm thick    "
             f"center ⌀{DISC_CENTER_HOLE:g} mm    "
             f"wire ⌀{WIRE_HOLE_DIA:g} mm × {N_WIRES}",
             ha="center", fontsize=8, color="0.25")
    ax1.set_xlim(-R - 4, R + 4)
    ax1.set_ylim(-R - 5, R + 4)
    ax1.set_axis_off()
    ax1.set_title("Top view (printed disc, full scale)", fontsize=10)

    rect_extent = R_MAX_MM + 1.0
    ax1.add_patch(plt.Rectangle((-rect_extent, -rect_extent),
                                 2 * rect_extent, 2 * rect_extent,
                                 fill=False, edgecolor="#cc4444",
                                 linewidth=0.8, linestyle="--", zorder=5))
    ax1.annotate("see inset →",
                 xy=(rect_extent, 0), xytext=(R - 6, -1),
                 fontsize=7, color="#cc4444",
                 arrowprops=dict(arrowstyle="->", color="#cc4444", lw=0.8))

    ax2 = fig.add_subplot(122)
    ax2.set_aspect("equal")
    spiral_th_dense = np.deg2rad(np.linspace(wire_thetas_deg[0],
                                             wire_thetas_deg[-1], 400))
    spiral_r_dense = np.linspace(R_MIN_MM, R_MAX_MM, 400)
    ax2.plot(spiral_r_dense * np.cos(spiral_th_dense),
             spiral_r_dense * np.sin(spiral_th_dense),
             color="#1f77b4", lw=1.5, ls="--", alpha=0.6, zorder=2,
             label=f"Spiral pattern\nr ∈ [{R_MIN_MM:g}, {R_MAX_MM:g}] mm, "
                   f"θ-step 30°")
    th2 = np.linspace(0, 2 * np.pi, 200)
    ax2.fill((DISC_CENTER_HOLE / 2) * np.cos(th2),
             (DISC_CENTER_HOLE / 2) * np.sin(th2),
             facecolor="white", edgecolor="#5a4a20", linewidth=1.0, zorder=3)
    for n, (x, y, _) in enumerate(_wire_positions()):
        ax2.fill(x + (WIRE_HOLE_DIA / 2) * np.cos(th2),
                 y + (WIRE_HOLE_DIA / 2) * np.sin(th2),
                 facecolor="black", zorder=5)
        r_lab = np.hypot(x, y) + 0.45
        ang = np.arctan2(y, x)
        ax2.text(r_lab * np.cos(ang), r_lab * np.sin(ang), str(n + 1),
                 fontsize=8, ha="center", va="center", color="#444")
    ax2.set_xlim(-rect_extent, rect_extent)
    ax2.set_ylim(-rect_extent, rect_extent)
    ax2.set_axis_off()
    ax2.set_title("Spiral pattern detail (zoomed)", fontsize=10)
    ax2.legend(loc="lower center", fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Wire-spiral disc — 12 holes, 1-turn spiral, "
                 f"r ∈ [{R_MIN_MM:g}, {R_MAX_MM:g}] mm",
                 fontsize=12, y=0.99)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _render_into(ax, mesh: trimesh.Trimesh, elev: float, azim: float,
                 face_color: tuple = (0.85, 0.78, 0.55),
                 cull_back: bool = True) -> None:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    az_rad = np.deg2rad(azim)
    el_rad = np.deg2rad(elev)
    view_dir = np.array([np.cos(el_rad) * np.cos(az_rad),
                         np.cos(el_rad) * np.sin(az_rad),
                         np.sin(el_rad)])
    light_dir = view_dir / np.linalg.norm(view_dir)
    normals = np.asarray(mesh.face_normals)
    if cull_back:
        keep = (normals @ view_dir) > -0.05
        faces = faces[keep]
        normals = normals[keep]
    centroids = verts[faces].mean(axis=1)
    intensity = np.clip(normals @ light_dir, 0, 1)
    intensity = 0.35 + 0.65 * intensity
    base = np.array(face_color)
    fc = np.clip(np.tile(base, (len(faces), 1)) * intensity[:, None], 0, 1)
    order = np.argsort(centroids @ view_dir)
    polys = verts[faces[order]]
    coll = Poly3DCollection(polys, linewidths=0.0, alpha=1.0)
    coll.set_facecolor(fc[order])
    coll.set_edgecolor((0, 0, 0, 0))
    ax.add_collection3d(coll)
    span = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
    cx, cy, cz = (verts.max(axis=0) + verts.min(axis=0)) / 2
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(cz - span, cz + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _render_solo(stl: str, title: str, out_path: Path,
                 elev: float = 25.0, azim: float = 35.0) -> None:
    mesh = trimesh.load(HW / stl)
    fig = plt.figure(figsize=(6.5, 5.0), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    _render_into(ax, mesh, elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, pad=4)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    _render_disc_combo(OUT / "fig_hw_wire_disc.png")
    _render_solo("wire_spiral_assembly.stl",
                 "Wire-spiral fixture (assembled, with 12 strung wires)",
                 OUT / "fig_hw_wire_assembly.png", elev=14, azim=30)
    _render_solo("phantom_mold_uniform.stl",
                 "Uniform-attenuation phantom mold (E4)",
                 OUT / "fig_hw_mold_uniform.png", elev=42, azim=35)
    _render_solo("phantom_mold_cyst.stl",
                 "Cyst phantom mold (E5)\nDimples in floor for 4 PTFE rods (4 mm OD)",
                 OUT / "fig_hw_mold_cyst.png", elev=58, azim=40)
    _render_solo("phantom_step_mold.stl",
                 "Step-attenuation phantom mold (E7)\n6 concentric chambers, 0.6 mm walls",
                 OUT / "fig_hw_mold_step.png", elev=55, azim=40)


if __name__ == "__main__":
    main()
