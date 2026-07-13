"""Generate paired IVUS images and acoustic-boundary segmentations from vesselgen.

Coordinate convention (read once):
  * Raysim IVUS scanline ``a`` fires along probe-local ``(sin a, 0, cos a)``.
  * Vesselgen GT polygons live in the probe imaging plane as ``(probe_x, probe_z)``.
  * ``tier1_evaluation.polar_to_cart_display`` maps polar B-mode to Cartesian
    display pixels with ``Theta = arctan2(X, Y)``, i.e. ``X = r sin a``,
    ``Y = r cos a`` -- the same numbers as ``(probe_x, probe_z)``.
  * Overlays therefore rasterize GT directly at display coordinates ``(X, Y)``.

Every mesh from ``vessel.json`` is loaded with its declared simulator material
(lumen, every wall layer interface, every in-wall lesion, the guidewire) so the
paired segmentation distinguishes every modeled structure
(see :mod:`vesselgen.labels`).

Run from the workspace root on a machine with the built raysim CUDA extension::

    conda activate ultrasound
    pip install -e vessel-generator
    python vessel-generator/examples/render_paired_dataset.py \\
        --out vessel-generator/out/paired_dataset_100 --n 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
VISIONS_DIR = REPO_ROOT / "instrument-calibration" / "p035_visions"
SIM_PKG_ROOT = REPO_ROOT / "i4h-sensor-simulation" / "ultrasound-raytracing"
sys.path.insert(0, str(VISIONS_DIR))
sys.path.insert(0, str(SIM_PKG_ROOT))

from tier1_evaluation import (  # noqa: E402
    b_mode_to_theta_r,
    load_calibrated_config,
    make_probe_at,
    polar_axes,
    polar_to_cart_display,
    slider_to_db,
)
from vessel_evaluation import acoustic_boundary_offset_mm  # noqa: E402

from vesselgen.config import GenerationConfig  # noqa: E402
from vesselgen.io import save_vessel  # noqa: E402
from vesselgen.labels import (  # noqa: E402
    LABEL_BACKGROUND,
    LABEL_GUIDEWIRE,
    LABEL_PERI_ADVENTITIA,
    LABEL_LUMEN,
    LABEL_RGBA,
    LABEL_VESSEL_WALL,
    LABEL_WALL_LIKE,
    manifest_label_dict,
    material_to_label,
)
from vesselgen.library import iter_dataset  # noqa: E402
from vesselgen.sampling import (  # noqa: E402
    GroundTruth,
    PoseSample,
    ground_truth_is_valid_pose,
    side_branch_imaging_sector_mask,
)
from vesselgen.sim_randomization import (  # noqa: E402
    MaterialDraw,
    SimRandomizationConfig,
    apply_frame_sim_params,
    apply_material_draw,
    apply_vessel_sim_draw,
    copy_tgc_control_points,
    deep_tgc_gain_db,
    load_ringdown_waveforms,
    saturation_fraction,
)
from vesselgen.vessel import Vessel  # noqa: E402

PoseSlot = Literal["random", "side_branch", "parent_ostium"]


# ---------------------------------------------------------------------------
# World construction from the vessel manifest
# ---------------------------------------------------------------------------


def _meshes_from_manifest(manifest: dict, vessel_dir: Path) -> list[tuple[Path, str]]:
    """Return ``[(obj_path, material_name)]`` for every mesh the simulator
    should load, in the order ``surfaces, lesions, guidewire``.
    """
    entries: list[tuple[Path, str]] = []
    for s in manifest.get("surfaces", []):
        p = vessel_dir / s["obj"]
        if p.exists():
            entries.append((p, str(s["material"])))
    for lesion in manifest.get("lesions", []):
        p = vessel_dir / lesion["obj"]
        if p.exists():
            entries.append((p, str(lesion["material"])))
    gw = manifest.get("guidewire")
    if gw is not None:
        p = vessel_dir / gw["obj"]
        if p.exists():
            entries.append((p, str(gw["material"])))
    return entries


def build_vessel_world(vessel_dir: Path, materials):
    """Load every mesh listed in ``vessel.json`` into a raysim World.

    The world's background material follows the manifest (default
    ``"lumen"`` -- the probe sits inside the lumen, so rays start there
    before crossing any closed surface). Each :class:`SurfaceEntry` is
    loaded as an inward-facing closed mesh whose ``material_name`` is
    the material a ray enters when it crosses the surface outward;
    nothing is loaded past the outermost emitted surface (rays stay in
    the outer-most material until the FOV).

    Falls back to the pre-manifest two-file layout
    (``lumen.obj`` + ``outer.obj``) only if no ``vessel.json`` is
    present, for compatibility with vessels saved before the manifest
    format existed.
    """

    import raysim as rs
    from raysim.ray_sim_python import Mesh

    manifest_path = vessel_dir / "vessel.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        bg = str(manifest.get("world", {}).get("background_material", "lumen"))
        world = rs.World(bg)
        for obj_path, material_name in _meshes_from_manifest(manifest, vessel_dir):
            world.add(Mesh(str(obj_path), materials.get_index(material_name)))
        return world

    # Pre-manifest layout fallback (no surfaces list on disk).
    world = rs.World("lumen")
    world.add(Mesh(str(vessel_dir / "lumen.obj"), materials.get_index("vessel_wall")))
    if (vessel_dir / "outer.obj").is_file():
        world.add(Mesh(str(vessel_dir / "outer.obj"), materials.get_index("extravascular")))
    return world


# ---------------------------------------------------------------------------
# Cartesian rasterisation of the imaging-plane polygons
# ---------------------------------------------------------------------------


def _offset_polygon_radially(poly: np.ndarray, offset_mm: float) -> np.ndarray:
    """Push polygon vertices outward from the probe origin by ``offset_mm``."""
    if offset_mm == 0.0:
        return poly
    radii = np.linalg.norm(poly, axis=1, keepdims=True)
    radii = np.maximum(radii, 1e-6)
    return poly * (radii + offset_mm) / radii


def _display_grid(t_far: float, cart_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-t_far, t_far, cart_size)
    ys = np.linspace(-t_far, t_far, cart_size)
    return np.meshgrid(xs, ys)


def _polygon_mask(polys, plane_pts: np.ndarray, *, offset_mm: float) -> np.ndarray:
    """OR together a list of (M, 2) closed polygons into a boolean mask
    over ``plane_pts`` (N, 2)."""
    from matplotlib.path import Path as MplPath

    mask = np.zeros(len(plane_pts), dtype=bool)
    for poly in polys:
        if poly is None or len(poly) < 3:
            continue
        offset_poly = _offset_polygon_radially(poly, offset_mm)
        mask |= MplPath(offset_poly).contains_points(plane_pts)
    return mask


def _cartesian_label_grid(
    gt: GroundTruth,
    *,
    acoustic_offset_mm: float,
    t_far: float,
    cart_size: int,
) -> np.ndarray:
    """Rasterize per-class labels on the ``polar_to_cart_display`` grid.

    Each entry in ``gt.surface_polygons`` is ``(name, material_name,
    polys)`` listed innermost-to-outermost. Following raysim's nested
    closed-mesh convention, ``material_name`` is the material a ray
    enters when crossing that mesh **outward** -- equivalently, the
    material of the shell *just outside* the surface (between this
    mesh and the next-outer mesh, or extending to the FOV when no
    outer mesh exists).

    Painting order:

      1. Background everywhere outside the imaging FOV.
      2. Anything inside the FOV defaults to the material *just
         outside* the outermost surface, derived from
         ``surfaces[-1].material_name``. For legacy 4-mesh manifests
         this is ``peri_adventitia`` (formerly ``extravascular``);
         for the new 3-mesh layout it is ``adventitia``.
      3. For each interior surface from outermost-to-inner (skipping
         the innermost lumen mesh), paint the interior of its polygon
         with the material of the shell *just inside* that surface
         = ``material_name`` of the next-inner surface in the chain.
         Subsequent inner iterations overwrite their shells with the
         correct materials.
      4. Each lesion overwrites whatever wall layer it sits in.
      5. The lumen mesh carves out the blood pool.
      6. The guidewire (if present) overwrites the lumen.

    For legacy two-mesh manifests (no surfaces list) this collapses to
    background / lumen / wall / peri_adventitia.
    """

    X, Y = _display_grid(t_far, cart_size)
    plane_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    labels = np.full(len(plane_pts), LABEL_BACKGROUND, dtype=np.uint8)
    in_fov = np.linalg.norm(plane_pts, axis=1) <= t_far

    surfaces = list(gt.surface_polygons)
    n = len(surfaces)

    if n >= 1:
        # Initial in-FOV fill = material of shell just outside the
        # outermost surface (same value raysim sees when a ray exits
        # everything).
        outermost_mat = surfaces[-1][1]
        labels[in_fov] = material_to_label.get(outermost_mat, LABEL_PERI_ADVENTITIA)
    else:
        labels[in_fov] = LABEL_PERI_ADVENTITIA

    if n >= 2:
        # Outer-to-inner walk. surfaces[i].polys is the polygon of the
        # i-th surface in nested order; everything inside that polygon
        # is in the shell *just inside* surface i, plus deeper shells.
        # We paint that whole interior with the material of the shell
        # just-inside surface i, which is surfaces[i-1].material_name.
        # The next (more-inward) iteration overwrites its smaller
        # interior with the correct deeper material, and so on.
        for i in range(n - 1, 0, -1):
            _name_i, _material_i, polys = surfaces[i]
            if not polys:
                continue
            inner_material = surfaces[i - 1][1]
            label = material_to_label.get(inner_material, LABEL_VESSEL_WALL)
            mask = _polygon_mask(polys, plane_pts, offset_mm=acoustic_offset_mm)
            labels[mask] = label
    elif n == 0:
        # Legacy manifest with no surfaces list -- fall back to the
        # old single-slab vessel_wall painting via outer-contour polygons.
        outer_mask = _polygon_mask(
            gt.outer_contour_polygons, plane_pts, offset_mm=acoustic_offset_mm
        )
        labels[outer_mask] = LABEL_VESSEL_WALL

    for _name, material_name, polys in gt.lesion_polygons:
        if not polys:
            continue
        mask = _polygon_mask(polys, plane_pts, offset_mm=acoustic_offset_mm)
        if not mask.any():
            continue
        labels[mask] = material_to_label.get(material_name, LABEL_BACKGROUND)

    lumen_mask = _polygon_mask(gt.lumen_contour_polygons, plane_pts, offset_mm=acoustic_offset_mm)
    labels[lumen_mask] = LABEL_LUMEN

    if gt.guidewire_polygons:
        gw_mask = _polygon_mask(gt.guidewire_polygons, plane_pts, offset_mm=0.0)
        labels[gw_mask] = LABEL_GUIDEWIRE

    return labels.reshape(cart_size, cart_size)


def _sample_cart_mask_on_polar(cart_mask: np.ndarray, cfg, *, cart_size: int) -> np.ndarray:
    """Sample a Cartesian label image onto the simulator's native polar grid."""
    sim_theta_deg, r_mm, _, _ = polar_axes(cfg)
    a_rad = np.radians(sim_theta_deg)
    probe_x = r_mm[np.newaxis, :] * np.sin(a_rad)[:, np.newaxis]
    probe_z = r_mm[np.newaxis, :] * np.cos(a_rad)[:, np.newaxis]

    t_far = float(cfg.sim.t_far_mm)
    xi = np.clip(
        ((probe_x + t_far) / (2.0 * t_far) * (cart_size - 1)).astype(np.int64),
        0,
        cart_size - 1,
    )
    yi = np.clip(
        ((probe_z + t_far) / (2.0 * t_far) * (cart_size - 1)).astype(np.int64),
        0,
        cart_size - 1,
    )
    return cart_mask[yi, xi]


def acoustic_segmentation_mask(gt: GroundTruth, cfg, *, acoustic_offset_mm: float) -> np.ndarray:
    """Acoustic-boundary labels on the simulator's native ``(theta, r)`` grid."""
    cart_size = 512
    t_far = float(cfg.sim.t_far_mm)
    cart_mask = _cartesian_label_grid(
        gt, acoustic_offset_mm=acoustic_offset_mm, t_far=t_far, cart_size=cart_size
    )
    return _sample_cart_mask_on_polar(cart_mask, cfg, cart_size=cart_size)


# ---------------------------------------------------------------------------
# Pose validation -- legacy "wall is missing on the side-branch sector"
# ---------------------------------------------------------------------------


def _segmentation_side_branch_sector_missing_wall(
    seg: np.ndarray,
    sector_mask: np.ndarray,
    *,
    min_lumen_radius_frac: float = 0.08,
    max_bad_fraction: float = 0.05,
) -> bool:
    sector_idxs = np.flatnonzero(sector_mask)
    if len(sector_idxs) == 0:
        return False
    n_r = seg.shape[1]
    min_lumen_r = int(n_r * min_lumen_radius_frac)
    wall_set = np.asarray(LABEL_WALL_LIKE, dtype=np.uint8)
    bad = 0
    for i in sector_idxs:
        lumen_rs = np.where(seg[i] == LABEL_LUMEN)[0]
        if len(lumen_rs) == 0:
            continue
        max_l = int(lumen_rs.max())
        if max_l < min_lumen_r:
            continue
        wall_rs = np.where(np.isin(seg[i], wall_set))[0]
        if len(wall_rs) == 0 or not np.any(wall_rs > max_l):
            bad += 1
    return (bad / len(sector_idxs)) > max_bad_fraction


def _daughter_branch_names(vessel: Vessel) -> list[str]:
    return [b.name for b in vessel.branches if not b.is_parent]


def _build_bifurcation_pose_schedule(
    frames_per_vessel: int,
    rng: np.random.Generator,
    *,
    min_side_branch_frames: int,
    min_parent_ostium_frames: int,
) -> list[PoseSlot]:
    schedule: list[PoseSlot] = ["random"] * frames_per_vessel
    n_side = min(min_side_branch_frames, frames_per_vessel)
    n_ostium = min(min_parent_ostium_frames, max(0, frames_per_vessel - n_side))
    for i, slot in enumerate(["side_branch"] * n_side + ["parent_ostium"] * n_ostium):
        schedule[i] = slot
    rng.shuffle(schedule)
    return schedule


def _sample_pose_for_slot(
    vessel: Vessel,
    slot: PoseSlot,
    rng: np.random.Generator,
    *,
    max_tilt_deg: float,
    edge_margin_mm: float,
    side_branch_ostium_bias_prob: float,
    side_branch_ostium_arclength_frac: float,
    neighbor_bias_prob: float = 0.0,
) -> PoseSample:
    daughters = [b for b in vessel.branches if not b.is_parent]
    if slot == "random" or not daughters:
        return vessel.sample_pose(
            rng,
            max_tilt_deg=max_tilt_deg,
            edge_margin_mm=edge_margin_mm,
            side_branch_ostium_bias_prob=side_branch_ostium_bias_prob,
            side_branch_ostium_arclength_frac=side_branch_ostium_arclength_frac,
            neighbor_bias_prob=neighbor_bias_prob,
        )
    side = daughters[0]
    # Tilt-cone reach that the imaging plane sweeps along the centerline.
    # Any cap disc within this reach can flag wall ~ 0 and reject the
    # pose, so the bifurcation pose slots stay outside this margin from
    # the daughter ostium and the daughter end caps.
    cap_clearance_mm = 2.5
    if slot == "side_branch":
        side_length = side.centerline.length_mm
        ostium_extent = max(side_length * side_branch_ostium_arclength_frac, 2.0)
        lo = cap_clearance_mm
        hi = max(lo + 1e-3, min(ostium_extent, side_length - cap_clearance_mm))
        if hi <= lo:
            # Daughter too short to leave clearance at both ends; fall
            # back to the branch midpoint.
            s = float(side_length / 2.0)
        else:
            s = float(rng.uniform(lo, hi))
        return vessel.sample_pose_in_branch(
            side.name,
            rng,
            arclength_mm=s,
            max_tilt_deg=max_tilt_deg,
            edge_margin_mm=edge_margin_mm,
        )
    attach = float(
        side.parent_attachment_arclength_mm
        if side.parent_attachment_arclength_mm is not None
        else side.centerline.length_mm * 0.5
    )
    parent = vessel.branches[0]
    parent_len = parent.centerline.length_mm
    window = min(10.0, parent_len * 0.15)
    margin = max(edge_margin_mm * 2.0, cap_clearance_mm)
    branch_lo = margin
    branch_hi = parent_len - margin
    # Sample from one of the two sides of the bifurcation joint, leaving
    # ``cap_clearance_mm`` of axial space between the imaging plane and
    # the daughter cap so the daughter end disc isn't visible in the
    # parent's imaging plane.
    side_lo_a = max(attach - window, branch_lo)
    side_hi_a = max(side_lo_a, attach - cap_clearance_mm)
    side_lo_b = min(attach + cap_clearance_mm, branch_hi)
    side_hi_b = min(attach + window, branch_hi)
    valid_below = side_hi_a > side_lo_a
    valid_above = side_hi_b > side_lo_b
    if valid_below and valid_above:
        if rng.random() < 0.5:
            s = float(rng.uniform(side_lo_a, side_hi_a))
        else:
            s = float(rng.uniform(side_lo_b, side_hi_b))
    elif valid_below:
        s = float(rng.uniform(side_lo_a, side_hi_a))
    elif valid_above:
        s = float(rng.uniform(side_lo_b, side_hi_b))
    else:
        # Parent too short for two-sided clearance; fall back to the
        # closest valid offset.
        s = float(np.clip(attach, branch_lo, branch_hi))
    return vessel.sample_pose_in_branch(
        parent.name,
        rng,
        arclength_mm=s,
        max_tilt_deg=max_tilt_deg,
        edge_margin_mm=edge_margin_mm,
    )


def _pose_is_valid(
    vessel: Vessel,
    pose: PoseSample,
    gt: GroundTruth,
    seg: np.ndarray | None = None,
) -> bool:
    if not ground_truth_is_valid_pose(vessel, pose, gt):
        return False
    if seg is not None and pose.branch_name != "parent":
        sector = side_branch_imaging_sector_mask(vessel, pose, gt.thetas_rad, gt)
        if _segmentation_side_branch_sector_missing_wall(seg, sector):
            return False
    return True


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class VesselRenderer:
    """Reuse world + simulator across frames from the same vessel mesh."""

    def __init__(self, cfg, materials, vessel_dir: Path) -> None:
        import raysim as rs

        self.cfg = cfg
        self.materials = materials
        self.vessel_dir = vessel_dir
        self.world = build_vessel_world(vessel_dir, materials)
        self.sim = rs.RaytracingUltrasoundSimulator(self.world, materials)

    def render(self, sim_params, pose: PoseSample) -> np.ndarray:
        rotation_rad = np.radians(pose.rotation_euler_deg).astype(np.float32)
        probe = make_probe_at(
            self.cfg,
            position_mm=pose.position.astype(float),
            rotation_rad=rotation_rad.astype(float),
        )
        b_mode = self.sim.simulate(probe, sim_params)
        return b_mode_to_theta_r(np.asarray(b_mode), self.cfg)


# ---------------------------------------------------------------------------
# Image / segmentation / overlay PNG writers
# ---------------------------------------------------------------------------


def save_segmentation_png(mask: np.ndarray, out_path: Path) -> None:
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for label_id, rgba in LABEL_RGBA.items():
        sel = mask == label_id
        if sel.any():
            rgb[sel] = rgba[:3]
    plt.imsave(out_path, rgb)


def save_overlay_png(
    image: np.ndarray,
    gt: GroundTruth,
    cfg,
    out_path: Path,
    *,
    acoustic_offset_mm: float,
) -> None:
    t_far = float(cfg.sim.t_far_mm)
    cart_size = 512
    cart_img = polar_to_cart_display(image, t_far_mm=t_far, cart_size=cart_size)
    cart_mask = _cartesian_label_grid(
        gt, acoustic_offset_mm=acoustic_offset_mm, t_far=t_far, cart_size=cart_size
    )
    extent = (-t_far, t_far, t_far, -t_far)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        np.ma.masked_invalid(cart_img),
        cmap="gray",
        vmin=0.0,
        vmax=255.0,
        extent=extent,
    )
    for label_id, rgba in LABEL_RGBA.items():
        if label_id == LABEL_PERI_ADVENTITIA:
            # Peri-adventitia / legacy extravascular fill -- skip the
            # bulk fill so the in-frame structures (lumen, wall layers,
            # lesions, guidewire) stay readable on the overlay.
            continue
        if not (cart_mask == label_id).any():
            continue
        layer = np.zeros((*cart_mask.shape, 4), dtype=np.float32)
        layer[..., :3] = rgba[:3]
        layer[..., 3] = np.where(cart_mask == label_id, rgba[3], 0.0)
        ax.imshow(layer, extent=extent)
    ax.set_xlim(-t_far, t_far)
    ax.set_ylim(t_far, -t_far)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Segmentation overlay (acoustic boundaries)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_image_png(image: np.ndarray, cfg, out_path: Path) -> None:
    t_far = float(cfg.sim.t_far_mm)
    cart = polar_to_cart_display(image, t_far_mm=t_far, cart_size=512)
    plt.imsave(out_path, np.ma.masked_invalid(cart), cmap="gray", vmin=0.0, vmax=255.0)


# ---------------------------------------------------------------------------
# Per-frame metadata helpers
# ---------------------------------------------------------------------------


def _record_metadata(
    frame_idx: int,
    vessel_name: str,
    pose: PoseSample,
    gt: GroundTruth,
    acoustic_offset_mm: float,
    *,
    material_draw: MaterialDraw | None = None,
    sim_parameters: dict | None = None,
) -> dict:
    meta = {
        "index": frame_idx,
        "vessel": vessel_name,
        "acoustic_boundary_offset_mm": acoustic_offset_mm,
        "segmentation_labels": manifest_label_dict(),
        "pose": {
            "position_mm": pose.position.tolist(),
            "rotation_euler_deg_xyz": pose.rotation_euler_deg.tolist(),
            "probe_axis_world": pose.probe_axis_world.tolist(),
            "branch_id": pose.branch_id,
            "branch_name": pose.branch_name,
            "arclength_mm": pose.arclength_mm,
            "centerline_offset_mm": pose.centerline_offset_mm,
            "tilt_deg": pose.tilt_deg,
        },
        "ground_truth_geometric": {
            "n_angles": gt.n_angles,
            "distance_to_lumen_wall_mm": gt.distance_to_lumen_wall_mm.tolist(),
            "distance_to_outer_wall_mm": gt.distance_to_outer_wall_mm.tolist(),
            "lumen_csa_mm2": gt.lumen_csa_mm2,
            "equivalent_lumen_diameter_mm": gt.equivalent_lumen_diameter_mm,
        },
        "structures_in_frame": {
            "wall_layers": [
                {"name": name, "material": material}
                for name, material, polys in gt.surface_polygons
                if polys and name not in ("lumen", "outer")
            ],
            "lesions": [
                {"name": name, "material": material}
                for name, material, polys in gt.lesion_polygons
                if polys
            ],
            "guidewire_visible": bool(gt.guidewire_polygons),
        },
    }
    if sim_parameters is not None:
        meta["sim_parameters"] = sim_parameters
    elif material_draw is not None:
        meta["sim_parameters"] = {"materials": material_draw.to_metadata()}
    return meta


def _pose_from_metadata(meta: dict) -> PoseSample:
    """Rebuild the exact pose used when the frame was rendered."""
    from vesselgen.sampling import _build_probe_rotation

    probe_axis = np.array(meta["pose"]["probe_axis_world"], dtype=float)
    return PoseSample(
        position=np.array(meta["pose"]["position_mm"], dtype=float),
        rotation_euler_deg=np.array(meta["pose"]["rotation_euler_deg_xyz"], dtype=float),
        rotation_matrix=_build_probe_rotation(probe_axis),
        probe_axis_world=probe_axis,
        branch_id=int(meta["pose"]["branch_id"]),
        branch_name=str(meta["pose"]["branch_name"]),
        arclength_mm=float(meta["pose"]["arclength_mm"]),
        centerline_offset_mm=float(meta["pose"]["centerline_offset_mm"]),
        tilt_deg=float(meta["pose"]["tilt_deg"]),
    )


# ---------------------------------------------------------------------------
# Dataset generation loop
# ---------------------------------------------------------------------------


def _scan_existing_for_resume(
    frames_dir: Path,
    vessels_dir: Path,
    frames_per_vessel: int,
) -> tuple[list[dict], int]:
    """Scan an in-progress dataset to compute resume state.

    Returns ``(kept_meta_records, next_vessel_index)``:

    * ``kept_meta_records`` is the list of frame-level metadata dicts
      from every fully-completed vessel (one whose ``frames_per_vessel``
      frames are all present and whose ``vessels/<name>/`` dir exists).
    * ``next_vessel_index`` is the smallest ``i`` such that
      ``vessel_{i:04d}`` is *not* yet on disk -- the value to pass to
      ``iter_dataset(start_index=...)``.

    Frames belonging to partial / orphaned vessels are deleted along
    with their vessel folder so the next run can re-emit them cleanly.
    """

    if not frames_dir.is_dir():
        return [], 0

    # Group existing frame metadata by vessel name.
    by_vessel: dict[str, list[tuple[Path, dict]]] = {}
    for fd in sorted(frames_dir.glob("frame_*")):
        meta_path = fd / "metadata.json"
        if not meta_path.is_file():
            shutil.rmtree(fd, ignore_errors=True)
            continue
        try:
            with meta_path.open() as f:
                meta = json.load(f)
        except Exception:
            shutil.rmtree(fd, ignore_errors=True)
            continue
        v_name = meta.get("vessel")
        if not v_name:
            shutil.rmtree(fd, ignore_errors=True)
            continue
        by_vessel.setdefault(v_name, []).append((fd, meta))

    kept_meta: list[dict] = []
    completed_indices: list[int] = []
    for v_name, entries in by_vessel.items():
        vessel_dir = vessels_dir / v_name
        is_complete = len(entries) == frames_per_vessel and (vessel_dir / "vessel.json").is_file()
        if is_complete:
            for _, meta in sorted(entries, key=lambda e: e[1].get("frame_id", 0)):
                kept_meta.append(meta)
            try:
                idx = int(v_name.split("_")[-1])
                completed_indices.append(idx)
            except ValueError:
                pass
            continue
        for fd, _ in entries:
            shutil.rmtree(fd, ignore_errors=True)
        if vessel_dir.exists():
            shutil.rmtree(vessel_dir, ignore_errors=True)

    # Also wipe vessel folders that have no frames at all (e.g. the run
    # crashed after save_vessel but before any frame landed).
    for vd in sorted(vessels_dir.glob("vessel_*")):
        if vd.name in by_vessel:
            continue
        shutil.rmtree(vd, ignore_errors=True)

    next_index = (max(completed_indices) + 1) if completed_indices else 0
    return kept_meta, next_index


def generate_paired_dataset(
    out_dir: Path,
    n_frames: int,
    *,
    base_seed: int = 42,
    max_tilt_deg: float = 15.0,
    edge_margin_mm: float = 0.15,
    frames_per_vessel: int = 12,
    max_pose_attempts: int = 48,
    min_finite_fraction: float = 0.85,
    rand_cfg: SimRandomizationConfig | None = None,
    gen_cfg: GenerationConfig | None = None,
    max_gain_resamples: int = 8,
    require_side_branch: bool = True,
    min_side_branch_frames: int = 4,
    min_parent_ostium_frames: int = 3,
    skip_overlay: bool = False,
    resume: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vessels_dir = out_dir / "vessels"
    frames_dir = out_dir / "frames"
    vessels_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)

    resume_meta: list[dict] = []
    resume_start_index = 0
    if resume:
        resume_meta, resume_start_index = _scan_existing_for_resume(
            frames_dir,
            vessels_dir,
            frames_per_vessel,
        )
        if resume_meta:
            print(
                f"[resume] kept {len(resume_meta)} frames from "
                f"{resume_start_index} completed vessels; continuing from "
                f"vessel_{resume_start_index:04d}, frame_idx="
                f"{len(resume_meta):05d}"
            )
        else:
            print("[resume] no completed vessels found; starting from scratch")

    rand_cfg = rand_cfg or SimRandomizationConfig()
    cfg, materials, _base_sim_params = load_calibrated_config()
    base_gain_db = float(cfg.processing.gain_db)
    base_ring_down_amplitude = float(cfg.processing.ring_down.amplitude)
    base_t_far_mm = float(cfg.sim.t_far_mm)
    base_scattering_resolution_mm = float(cfg.processing.scattering_resolution_mm)
    base_tgc_deep_gain_db = deep_tgc_gain_db(cfg, deep_depth_cm=rand_cfg.tgc_deep_depth_cm)
    base_tgc_control_points = copy_tgc_control_points(_base_sim_params.tgc_control_points)
    acoustic_offset = acoustic_boundary_offset_mm(cfg)
    n_theta = int(cfg.sim.b_mode_size[0])

    if gen_cfg is None:
        # 50/50 mix of bifurcation vessels and straight vessels for
        # geometric variety. Both carry the full trilaminar
        # (intima / media / adventitia) wall now that
        # ``attach_side_branch_layered`` does per-layer boolean unions
        # at the ostium -- previously bifurcation parents were forced
        # back to a single-slab wall and never showed the three-band
        # IVUS appearance.
        #
        # Widen the intima fraction floor so the bright innermost
        # layer is always at least one PSF wide; pin
        # ``layered_wall_probability`` to 1.0 and ``n_layers_weights``
        # to (0, 0, 1) so every vessel emits the full 3-layer
        # trilaminar wall.
        gen_cfg = GenerationConfig(
            side_branch_probability=0.5,
            layered_wall_probability=1.0,
            n_layers_weights=(0.0, 0.0, 1.0),
            intima_thickness_frac_range=(0.18, 0.30),
            media_thickness_frac_range=(0.40, 0.55),
        )
    manifest_frames: list[dict] = list(resume_meta)
    frame_idx = len(resume_meta)
    if frame_idx >= n_frames:
        print(
            f"[resume] target {n_frames} already met by {frame_idx} kept "
            "frames; nothing to render."
        )
        return

    vessel_iter = iter_dataset(
        n=max(n_frames, 20),
        config=gen_cfg,
        base_seed=base_seed,
        name_prefix="vessel",
        require_side_branch=require_side_branch,
        start_index=resume_start_index,
    )

    pbar = tqdm(
        total=n_frames,
        initial=frame_idx,
        desc="Rendering paired frames",
    )
    for _vessel_cfg, vessel in vessel_iter:
        if frame_idx >= n_frames:
            break

        vessel_dir = vessels_dir / vessel.config.name
        save_vessel(vessel, vessel_dir)

        rng = np.random.default_rng(base_seed + frame_idx * 17 + vessel.config.seed)
        material_draw = rand_cfg.sample_vessel_materials(rng)
        vessel_sim_draw = rand_cfg.sample_vessel_sim(
            rng,
            base_ring_down_amplitude=base_ring_down_amplitude,
            base_t_far_mm=base_t_far_mm,
            base_scattering_resolution_mm=base_scattering_resolution_mm,
        )
        apply_material_draw(materials, material_draw)
        cfg.sim.t_far_mm = float(vessel_sim_draw.t_far_mm)
        ringdown = load_ringdown_waveforms(cfg, rand_cfg)
        renderer = VesselRenderer(cfg, materials, vessel_dir)
        vessel_sim_params = cfg.to_sim_params()
        apply_vessel_sim_draw(cfg, vessel_sim_params, vessel_sim_draw)

        n_ar_on = (frames_per_vessel + 1) // 2
        ar_schedule = ["on"] * n_ar_on + ["off"] * (frames_per_vessel - n_ar_on)
        rng.shuffle(ar_schedule)

        has_bifurcation = bool(_daughter_branch_names(vessel))
        pose_schedule = (
            _build_bifurcation_pose_schedule(
                frames_per_vessel,
                rng,
                min_side_branch_frames=min_side_branch_frames,
                min_parent_ostium_frames=min_parent_ostium_frames,
            )
            if has_bifurcation
            else ["random"] * frames_per_vessel
        )

        poses_collected = 0
        pose_attempts = 0

        while poses_collected < frames_per_vessel and frame_idx < n_frames:
            if pose_attempts >= max_pose_attempts:
                break
            pose_attempts += 1

            pose_slot: PoseSlot = pose_schedule[min(poses_collected, len(pose_schedule) - 1)]
            pose = _sample_pose_for_slot(
                vessel,
                pose_slot,
                rng,
                max_tilt_deg=max_tilt_deg,
                edge_margin_mm=edge_margin_mm,
                side_branch_ostium_bias_prob=rand_cfg.side_branch_ostium_bias_prob,
                side_branch_ostium_arclength_frac=rand_cfg.side_branch_ostium_arclength_frac,
                neighbor_bias_prob=gen_cfg.adjacent_vessel_pose_bias_prob,
            )
            gt = vessel.ground_truth_at(
                pose,
                n_angles=n_theta,
                max_distance_mm=float(vessel_sim_draw.t_far_mm),
            )
            # Hot-path optimisation: ~96% of pose attempts used to get
            # rejected, and the segmentation mask costs ~5x the cheap
            # geometric validity check. Reject early on the cheap check,
            # then compute seg only when needed.
            if not ground_truth_is_valid_pose(
                vessel, pose, gt, min_finite_fraction=min_finite_fraction
            ):
                continue
            seg: np.ndarray | None = None
            # The "side-branch sector missing wall" check rejects poses
            # whose imaging plane sees lumen with no wall behind it on
            # the parent-pointing side -- which is exactly the
            # bifurcation-ostium view we WANT to include in the dataset.
            # Skip it for the "side_branch" pose slot (which targets
            # those views by design) and only run it on "random" poses
            # that landed on a daughter branch, with a much looser
            # max_bad_fraction so genuine ostium transparency is
            # accepted while truly broken meshes are still caught.
            if pose.branch_name != "parent" and pose_slot == "random":
                seg = acoustic_segmentation_mask(gt, cfg, acoustic_offset_mm=acoustic_offset)
                sector = side_branch_imaging_sector_mask(vessel, pose, gt.thetas_rad, gt)
                if _segmentation_side_branch_sector_missing_wall(seg, sector, max_bad_fraction=0.5):
                    continue

            frame_draw = None
            image = None
            sat_frac = 1.0
            for _ in range(max_gain_resamples):
                frame_draw = rand_cfg.sample_frame_sim(
                    rng,
                    base_gain_db=base_gain_db,
                    frame_seed=frame_idx + 1,
                    noise_seed=frame_idx + 1,
                    slider_to_db=slider_to_db,
                    base_tgc_deep_gain_db=base_tgc_deep_gain_db,
                    artifact_reduction=ar_schedule[poses_collected],
                )
                apply_frame_sim_params(
                    vessel_sim_params,
                    frame_draw,
                    ringdown=ringdown,
                    base_tgc_control_points=base_tgc_control_points,
                    tgc_deep_depth_cm=rand_cfg.tgc_deep_depth_cm,
                )
                image = renderer.render(vessel_sim_params, pose)
                sat_frac = saturation_fraction(
                    image, saturation_palette=rand_cfg.saturation_palette
                )
                if sat_frac <= rand_cfg.max_saturation_fraction:
                    break

            assert frame_draw is not None and image is not None
            if sat_frac > rand_cfg.max_saturation_fraction:
                continue

            # Compute seg for parent-branch poses (skipped during
            # the validity check above for the parent path).
            if seg is None:
                seg = acoustic_segmentation_mask(gt, cfg, acoustic_offset_mm=acoustic_offset)

            frame_name = f"frame_{frame_idx:05d}"
            frame_dir = frames_dir / frame_name
            frame_dir.mkdir(exist_ok=True)

            np.save(frame_dir / "image.npy", image)
            np.save(frame_dir / "segmentation.npy", seg)
            save_image_png(image, cfg, frame_dir / "image.png")
            save_segmentation_png(seg, frame_dir / "segmentation.png")
            if not skip_overlay:
                save_overlay_png(
                    image,
                    gt,
                    cfg,
                    frame_dir / "overlay.png",
                    acoustic_offset_mm=acoustic_offset,
                )

            sim_parameters = {
                **material_draw.to_metadata(),
                **vessel_sim_draw.to_metadata(),
                **frame_draw.to_metadata(),
                "pose_slot": pose_slot,
                "ring_down_enabled": bool(vessel_sim_params.ring_down.enabled),
                "qc": {
                    "saturation_fraction": sat_frac,
                    "excessive_saturation": sat_frac > rand_cfg.max_saturation_fraction,
                },
            }
            meta = _record_metadata(
                frame_idx,
                vessel.config.name,
                pose,
                gt,
                acoustic_offset,
                sim_parameters=sim_parameters,
            )
            with (frame_dir / "metadata.json").open("w") as f:
                json.dump(meta, f, indent=2)

            manifest_frames.append(meta)
            frame_idx += 1
            poses_collected += 1
            pbar.update(1)

    pbar.close()

    if frame_idx < n_frames:
        raise RuntimeError(
            f"Only generated {frame_idx}/{n_frames} frames; "
            "try lowering edge_margin_mm or increasing vessel count."
        )

    manifest = {
        "n_frames": frame_idx,
        "acoustic_boundary_offset_mm": acoustic_offset,
        "boundary_type": "acoustic",
        "sim_config": str(VISIONS_DIR / "volcano_s5i.yaml"),
        "randomization": {
            "gain_slider_range": list(rand_cfg.gain_slider_range),
            "ar_on_probability": rand_cfg.ar_on_probability,
            "side_branch_ostium_bias_prob": rand_cfg.side_branch_ostium_bias_prob,
            "require_side_branch": require_side_branch,
            "min_side_branch_frames_per_vessel": min_side_branch_frames,
            "min_parent_ostium_frames_per_vessel": min_parent_ostium_frames,
            "max_saturation_fraction": rand_cfg.max_saturation_fraction,
            "min_finite_fraction": min_finite_fraction,
            "tier2_enabled": rand_cfg.enable_tier2,
            "tgc_deep_gain_scale_range": list(rand_cfg.tgc_deep_gain_scale_range),
            "ring_down_amplitude_scale_range": list(rand_cfg.ring_down_amplitude_scale_range),
            "scattering_resolution_mm_range": list(rand_cfg.scattering_resolution_mm_range),
            "t_far_mm_choices": list(rand_cfg.t_far_mm_choices),
        },
        "vessel_generation": {
            "side_branch_probability": gen_cfg.side_branch_probability,
            "layered_wall_probability": gen_cfg.layered_wall_probability,
            "n_layers_weights": list(gen_cfg.n_layers_weights),
            "calcification_probability": gen_cfg.calcification_probability,
            "calc_kind_weights": dict(gen_cfg.calc_kind_weights),
            "lesion_arc_extent_deg_range": list(gen_cfg.lesion_arc_extent_deg_range),
            "lesion_axial_extent_mm_range": list(gen_cfg.lesion_axial_extent_mm_range),
            "guidewire_probability": gen_cfg.guidewire_probability,
            "guidewire_diameter_choices_mm": list(gen_cfg.guidewire_diameter_choices_mm),
            "guidewire_diameter_weights": list(gen_cfg.guidewire_diameter_weights),
        },
        "segmentation_labels": manifest_label_dict(),
        "frames": manifest_frames,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {frame_idx} paired frames to {out_dir}")


# ---------------------------------------------------------------------------
# Re-derivation of segmentations from saved frames (no re-render)
# ---------------------------------------------------------------------------


def regenerate_segmentations(out_dir: Path) -> None:
    """Recompute segmentation masks and overlays from saved images + metadata."""
    cfg, _, _ = load_calibrated_config()
    acoustic_offset = acoustic_boundary_offset_mm(cfg)
    frames_dir = out_dir / "frames"
    vessels_dir = out_dir / "vessels"

    vessel_cache: dict[str, Vessel] = {}
    for frame_dir in sorted(frames_dir.glob("frame_*")):
        meta_path = frame_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        with meta_path.open() as f:
            meta = json.load(f)

        vessel_name = meta["vessel"]
        if vessel_name not in vessel_cache:
            vessel_cache[vessel_name] = Vessel.load(vessels_dir / vessel_name)
        vessel = vessel_cache[vessel_name]

        pose = _pose_from_metadata(meta)
        t_far_mm = float(meta.get("sim_parameters", {}).get("t_far_mm", cfg.sim.t_far_mm))
        cfg.sim.t_far_mm = t_far_mm
        gt = vessel.ground_truth_at(
            pose,
            n_angles=int(cfg.sim.b_mode_size[0]),
            max_distance_mm=t_far_mm,
        )
        seg = acoustic_segmentation_mask(gt, cfg, acoustic_offset_mm=acoustic_offset)
        image = np.load(frame_dir / "image.npy")

        np.save(frame_dir / "segmentation.npy", seg)
        save_segmentation_png(seg, frame_dir / "segmentation.png")
        save_overlay_png(
            image,
            gt,
            cfg,
            frame_dir / "overlay.png",
            acoustic_offset_mm=acoustic_offset,
        )

    print(f"Regenerated segmentations in {frames_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "vessel-generator" / "out" / "paired_dataset_100",
    )
    p.add_argument("--n", type=int, default=100, help="Number of paired frames to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tilt-deg", type=float, default=15.0)
    p.add_argument("--edge-margin-mm", type=float, default=0.15)
    p.add_argument("--frames-per-vessel", type=int, default=12)
    p.add_argument(
        "--min-finite-fraction",
        type=float,
        default=0.85,
        help=(
            "Minimum fraction of A-lines whose lumen AND outer wall both fall "
            "inside the FOV for a pose to be accepted. The 0.85 default rejects "
            "beyond-FOV anatomy by construction: a large vessel whose far wall "
            "runs past t_far has no outer hit on those sectors, which is the "
            "case we want to render, not discard. Lower it (e.g. 0.50) when "
            "generating a beyond-FOV shard."
        ),
    )
    p.add_argument(
        "--no-require-side-branch",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,  # legacy alias; the new default is to allow a mix
    )
    p.add_argument(
        "--require-side-branch",
        action="store_true",
        help=(
            "Only render vessels that have a side-branch bifurcation. "
            "Trilaminar walls are supported at ostia (per-layer boolean "
            "union). Default: a mix of bifurcation and straight vessels."
        ),
    )
    p.add_argument(
        "--min-side-branch-frames",
        type=int,
        default=4,
        help="Per bifurcation vessel: frames posed in the side branch near the ostium",
    )
    p.add_argument(
        "--min-parent-ostium-frames",
        type=int,
        default=3,
        help="Per bifurcation vessel: frames on the parent centered on the bifurcation",
    )
    p.add_argument(
        "--regenerate-segmentations-only",
        action="store_true",
        help="Recompute masks/overlays from saved images (no re-render)",
    )
    p.add_argument(
        "--skip-overlay",
        action="store_true",
        help=(
            "Skip the per-frame matplotlib overlay PNG. Saves ~0.7 s/frame "
            "(~2 hours over 10 K frames). The polar B-mode and segmentation "
            "PNGs are still written; rerun with --regenerate-segmentations-only "
            "later if overlays are needed."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a previously-interrupted run pointed at --out. Frames "
            "from fully-completed vessels are kept; frames from any "
            "partially-rendered vessel (and its vessel folder) are deleted "
            "so that vessel can be re-emitted cleanly. The seed schedule is "
            "deterministic per vessel index, so resumed runs reproduce the "
            "exact configs the original run would have generated."
        ),
    )
    args = p.parse_args()

    if args.regenerate_segmentations_only:
        regenerate_segmentations(args.out)
        return

    generate_paired_dataset(
        args.out,
        args.n,
        base_seed=args.seed,
        max_tilt_deg=args.max_tilt_deg,
        edge_margin_mm=args.edge_margin_mm,
        frames_per_vessel=args.frames_per_vessel,
        min_finite_fraction=args.min_finite_fraction,
        require_side_branch=bool(args.require_side_branch),
        min_side_branch_frames=args.min_side_branch_frames,
        min_parent_ostium_frames=args.min_parent_ostium_frames,
        skip_overlay=args.skip_overlay,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
