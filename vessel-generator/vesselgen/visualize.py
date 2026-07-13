"""Quick QC plots for generated vessels.

The intent is verification, not publication-grade rendering. Each preview
shows:

- a 3D wireframe of the lumen (and outer if requested) with branch
  centerlines overlaid;
- a small gallery of cross-sections at evenly-spaced arclengths along the
  parent branch, with the lumen and outer contours;
- a thumbnail of any sampled poses (overlay arrows on the 3D plot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vesselgen.sampling import GroundTruth, PoseSample
from vesselgen.vessel import Vessel


def _decimate_faces(mesh, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    n_faces = len(mesh.faces)
    if n_faces <= max_faces:
        return mesh.vertices, mesh.faces
    stride = max(1, int(np.ceil(n_faces / max_faces)))
    faces = mesh.faces[::stride]
    return mesh.vertices, faces


_LAYER_FACECOLOR = {
    "intima": "#ffd166",
    "media": "#ef476f",
    "adventitia": "#118ab2",
    "vessel_wall": "#7f7fff",
    "extravascular": "#cdb4db",
}

_LESION_FACECOLOR = {
    "calcified_plaque": "#f8f9fa",
    "lipid_pool": "#fdcdac",
    "fibrous_plaque": "#cbd5e8",
    "thrombus": "#b3cde3",
}


def preview_vessel(
    vessel: Vessel,
    out_path: str | Path,
    poses: Optional[Sequence[PoseSample]] = None,
    max_faces: int = 4000,
) -> Path:
    """Save a 3D + cross-section gallery preview of a vessel to ``out_path``.

    The 3D panel draws the outer wall, lumen, every interior layer
    interface (when present), every lesion mesh coloured by kind, and
    the guidewire as an opaque grey cylinder.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")

    verts, faces = _decimate_faces(vessel.outer_mesh, max_faces)
    poly = Poly3DCollection(
        verts[faces], alpha=0.08, facecolor="#7f7fff", edgecolor="#3030a0", linewidth=0.1
    )
    ax3.add_collection3d(poly)

    # Interior layer interfaces only — outer_mesh is surfaces[-1] and is
    # drawn separately below.
    outer_name = vessel.surfaces[-1].name
    interior_surfaces = [s for s in vessel.surfaces if s.name not in ("lumen", outer_name)]
    for s in interior_surfaces:
        verts, faces = _decimate_faces(s.mesh, max_faces)
        color = _LAYER_FACECOLOR.get(s.material_name, "#a3b18a")
        poly = Poly3DCollection(
            verts[faces], alpha=0.12, facecolor=color, edgecolor=color, linewidth=0.05
        )
        ax3.add_collection3d(poly)

    verts, faces = _decimate_faces(vessel.lumen_mesh, max_faces)
    poly = Poly3DCollection(
        verts[faces], alpha=0.18, facecolor="#ff6060", edgecolor="#a02020", linewidth=0.1
    )
    ax3.add_collection3d(poly)

    # Lesion meshes, opaque so they're easy to spot.
    for lesion in vessel.lesions:
        verts, faces = _decimate_faces(lesion.mesh, max_faces)
        color = _LESION_FACECOLOR.get(lesion.material_name, "#444444")
        poly = Poly3DCollection(
            verts[faces], alpha=0.55, facecolor=color, edgecolor="#222222", linewidth=0.2
        )
        ax3.add_collection3d(poly)

    if vessel.guidewire is not None:
        verts, faces = _decimate_faces(vessel.guidewire.mesh, max_faces)
        poly = Poly3DCollection(
            verts[faces], alpha=0.8, facecolor="#333333", edgecolor="#000000", linewidth=0.2
        )
        ax3.add_collection3d(poly)

    for b in vessel.branches:
        cl = b.centerline.positions
        ax3.plot(
            cl[:, 0],
            cl[:, 1],
            cl[:, 2],
            "-",
            lw=1.5,
            color=("k" if b.is_parent else "tab:orange"),
            label=b.name,
        )

    if poses:
        for pose in poses:
            p = pose.position
            d = pose.probe_axis_world
            ax3.plot([p[0]], [p[1]], [p[2]], "o", color="lime", markersize=4)
            ax3.plot(
                [p[0] - 1.5 * d[0], p[0] + 1.5 * d[0]],
                [p[1] - 1.5 * d[1], p[1] + 1.5 * d[1]],
                [p[2] - 1.5 * d[2], p[2] + 1.5 * d[2]],
                "-",
                color="lime",
                lw=1.5,
            )

    bbox_min = vessel.outer_mesh.vertices.min(axis=0)
    bbox_max = vessel.outer_mesh.vertices.max(axis=0)
    span = (bbox_max - bbox_min).max()
    centre = 0.5 * (bbox_min + bbox_max)
    for axis, set_lim in zip((0, 1, 2), (ax3.set_xlim, ax3.set_ylim, ax3.set_zlim)):
        set_lim(centre[axis] - 0.55 * span, centre[axis] + 0.55 * span)
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("y (mm)")
    ax3.set_zlabel("z (mm)")
    ax3.set_title(f"{vessel.config.name} (seed={vessel.config.seed})")
    ax3.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    parent = vessel.parent_branch
    n = parent.lumen_field.n_stations
    sample_indices = np.linspace(0, n - 1, 6).astype(int)
    cmap = plt.get_cmap("viridis", len(sample_indices))
    for k, i in enumerate(sample_indices):
        lumen = parent.lumen_field.contour(i)
        outer = lumen + (parent.wall_field.thicknesses[i, :, None]) * (
            lumen / np.maximum(parent.lumen_field.radii[i, :, None], 1e-6)
        )
        s_mm = i * parent.centerline.length_mm / max(1, n - 1)
        col = cmap(k)
        ax2.plot(
            np.append(outer[:, 0], outer[0, 0]),
            np.append(outer[:, 1], outer[0, 1]),
            "--",
            color=col,
            lw=1.0,
            label=f"s={s_mm:.1f} mm" if k in (0, len(sample_indices) - 1) else None,
        )
        ax2.plot(
            np.append(lumen[:, 0], lumen[0, 0]),
            np.append(lumen[:, 1], lumen[0, 1]),
            "-",
            color=col,
            lw=1.5,
        )
    ax2.set_aspect("equal")
    ax2.set_title("Parent branch cross-sections (lumen solid, outer dashed)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# Catheter ring-down artifact (probe-centered). Everything inside
# ``RING_DOWN_INNER_MM`` reads as a black dead zone; the bright ring-down
# echo spans out to ``RING_DOWN_OUTER_MM``. Defaults match the PV .035 /
# Volcano s5i profile (black < ~1 mm, bright ring ~1-2.8 mm). These are
# QA/analysis constants only -- ring-down is applied at render time by the
# simulator and is never baked into geometry.
RING_DOWN_INNER_MM = 1.0
RING_DOWN_OUTER_MM = 2.8


def _draw_ring_down_annulus(
    ax,
    ring_down_outer_mm: float,
    ring_down_inner_mm: float,
) -> None:
    """Shade the probe-centered ring-down annulus on a cross-section axis.

    The bright ring-down band (inner->outer) is drawn as a translucent
    orange annulus and the central dead zone as a translucent dark disc, so
    the wall contours underneath stay visible.
    """
    from matplotlib.patches import Circle

    # Bright ring-down band: filled outer disc minus the dead-zone disc,
    # both translucent so the lumen/outer contours remain readable.
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            ring_down_outer_mm,
            facecolor="tab:orange",
            edgecolor="tab:orange",
            alpha=0.18,
            lw=1.0,
            zorder=0,
            label=f"ring-down ({ring_down_inner_mm:.1f}-{ring_down_outer_mm:.1f} mm)",
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            ring_down_inner_mm,
            facecolor="black",
            edgecolor="none",
            alpha=0.30,
            zorder=1,
        )
    )


def preview_ground_truth(
    pose: PoseSample,
    gt: GroundTruth,
    out_path: str | Path,
    fov_radius_mm: Optional[float] = None,
    ring_down_outer_mm: Optional[float] = None,
    ring_down_inner_mm: float = RING_DOWN_INNER_MM,
) -> Path:
    """Save an overlay showing the pose's lumen + outer contours and per-angle
    distance arrays.

    When ``ring_down_outer_mm`` is provided, a translucent probe-centered
    ring-down annulus is overlaid on both panels so it is visually obvious
    which A-lines fall inside the ring-down zone (obscured wall) versus
    outside it (visible wall the model can use to guess the true wall).

    When ``fov_radius_mm`` is given, the imaging field of view is drawn as a
    dashed circle (left panel) and a dashed reference line (right panel).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    if ring_down_outer_mm is not None:
        _draw_ring_down_annulus(ax, ring_down_outer_mm, ring_down_inner_mm)
    if fov_radius_mm is not None:
        fov_thetas_rad = np.linspace(0.0, 2.0 * np.pi, 200)
        ax.plot(
            fov_radius_mm * np.cos(fov_thetas_rad),
            fov_radius_mm * np.sin(fov_thetas_rad),
            ":",
            color="tab:green",
            lw=1.0,
            label=f"FOV r={fov_radius_mm:.1f} mm",
        )
    for poly in gt.outer_contour_polygons:
        ax.plot(
            np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]),
            "--",
            color="tab:blue",
            lw=1.0,
        )
    for poly in gt.lumen_contour_polygons:
        ax.plot(
            np.append(poly[:, 0], poly[0, 0]),
            np.append(poly[:, 1], poly[0, 1]),
            "-",
            color="tab:red",
            lw=1.5,
        )
    ax.plot(0, 0, "o", color="black", markersize=6, label="probe origin")
    ax.set_aspect("equal")
    ax.set_xlabel("imaging-plane x (mm)")
    ax.set_ylabel("imaging-plane y (mm)")
    ax.set_title(
        f"branch={pose.branch_name}  s={pose.arclength_mm:.1f} mm  tilt={pose.tilt_deg:.1f}°"
    )
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    th_deg = np.degrees(gt.thetas_rad)
    if ring_down_outer_mm is not None:
        ax.axhspan(
            0.0,
            ring_down_outer_mm,
            color="tab:orange",
            alpha=0.15,
            zorder=0,
            label=f"ring-down (<{ring_down_outer_mm:.1f} mm)",
        )
        ax.axhspan(0.0, ring_down_inner_mm, color="black", alpha=0.25, zorder=1)
    if fov_radius_mm is not None:
        ax.axhline(
            fov_radius_mm,
            ls=":",
            color="tab:green",
            lw=1.0,
            label=f"FOV {fov_radius_mm:.1f} mm",
        )
    ax.plot(th_deg, gt.distance_to_lumen_wall_mm, "-", color="tab:red", label="lumen")
    ax.plot(th_deg, gt.distance_to_outer_wall_mm, "--", color="tab:blue", label="outer")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("distance from probe (mm)")
    ax.set_title(
        f"CSA={gt.lumen_csa_mm2:.2f} mm²  D_eq={gt.equivalent_lumen_diameter_mm:.2f} mm  "
        f"branches visible={gt.branch_ids_visible}"
    )
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
