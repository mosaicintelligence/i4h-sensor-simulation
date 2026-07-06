"""Pose sampling and ground-truth extraction.

This module is the public surface the DL training pipeline calls every
frame. The two operations are:

1. :func:`sample_pose` -- draw a random catheter pose inside the vessel
   lumen. The sampler picks a branch (weighted by its arclength), an
   arclength along that branch, a 2D position inside the lumen contour at
   that arclength (rejection-sampled), and a small probe-axis tilt
   relative to the local vessel tangent. The result is a :class:`PoseSample`
   with a 3D position and an Euler-XYZ rotation (degrees) suitable for
   passing directly to ``rs.Pose(position=..., rotation=...)``.

2. :func:`ground_truth_at` -- compute per-frame ground truth for a given
   pose. The imaging plane (perpendicular to the probe long axis, through
   the probe origin) is intersected with the vessel's lumen and outer
   meshes; the intersection polygons are converted to polar coordinates
   centred on the probe to produce per-angle distance-to-wall arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vesselgen.vessel import BranchHandle, Vessel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PoseSample:
    """A sampled catheter pose, ready for the simulator."""

    position: np.ndarray  # (3,) world-space mm
    rotation_euler_deg: np.ndarray  # (3,) Euler XYZ degrees, suitable for rs.Pose
    rotation_matrix: np.ndarray  # (3, 3), probe-local-frame -> world
    probe_axis_world: np.ndarray  # (3,), unit vector of the probe long axis in world

    branch_id: int
    branch_name: str
    arclength_mm: float
    """Arclength along the branch's centerline at which this pose was sampled."""

    centerline_offset_mm: float
    """Distance from the sampled position to the branch's centerline (eccentricity)."""

    tilt_deg: float
    """Angle between the probe long axis and the local vessel tangent."""

    seed_used: Optional[int] = None


@dataclass
class GroundTruth:
    """Per-frame ground truth at one pose.

    Attributes:
        n_angles:            number of polar angle bins.
        thetas_rad:          (n_angles,) angle bin centres in radians.
        distance_to_lumen_wall_mm:
            (n_angles,) distance from probe to the nearest lumen-surface
            crossing along the ray at angle theta. ``np.nan`` for any
            ray that misses the lumen surface within the imaging window
            (e.g. probe-against-wall geometry).
        distance_to_outer_wall_mm:
            (n_angles,) same for the outer (adventitia) surface.
        wall_thickness_mm:
            (n_angles,) outer minus lumen.
        lumen_contour_polygons:
            list of (M, 2) closed polygons in the imaging plane (xz-like
            local coordinates, x along probe binormal, y along normal).
            Multiple polygons mean the imaging plane crosses an ostium and
            the model sees more than one lumen pocket.
        outer_contour_polygons:
            same, for the outer mesh.
        lumen_csa_mm2:
            sum of areas of the lumen polygons in the plane.
        outer_csa_mm2:
            sum of areas of the outer polygons in the plane.
        branch_ids_visible:
            branch IDs whose centerline is closer to the probe than its
            mean radius (heuristic for "this branch is visible in this
            frame"). Always includes the sampled branch.
        equivalent_lumen_diameter_mm:
            2 * sqrt(lumen_csa / pi). Convenient scalar for diameter MAE.
    """

    n_angles: int
    thetas_rad: np.ndarray
    distance_to_lumen_wall_mm: np.ndarray
    distance_to_outer_wall_mm: np.ndarray
    wall_thickness_mm: np.ndarray
    lumen_contour_polygons: list[np.ndarray]
    outer_contour_polygons: list[np.ndarray]
    lumen_csa_mm2: float
    outer_csa_mm2: float
    branch_ids_visible: list[int] = field(default_factory=list)
    equivalent_lumen_diameter_mm: float = 0.0

    # Per-mesh polygons in the imaging plane, ready to rasterize directly
    # into the same (probe_x, probe_z) display grid the renderer uses.
    # Each entry is ``(name, material_name, [polygon, ...])`` and a single
    # mesh can produce zero (mesh missed the plane), one, or multiple
    # polygons (e.g. a lesion sliced near its tip).
    surface_polygons: list[tuple[str, str, list[np.ndarray]]] = field(default_factory=list)
    """Wall-layer interfaces (lumen, intima/media boundary, ..., outer)."""

    lesion_polygons: list[tuple[str, str, list[np.ndarray]]] = field(default_factory=list)
    """In-wall lesions (calcified, lipid pool, fibrous, thrombus)."""

    guidewire_polygons: list[np.ndarray] = field(default_factory=list)
    """Guidewire cross-section, if present in this imaging plane."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lumen_contour_polygon(branch: BranchHandle, station_idx: int) -> np.ndarray:
    """Closed (M+1, 2) lumen contour polygon in the cross-section local frame."""
    contour = branch.lumen_field.contour(station_idx)
    return np.vstack([contour, contour[:1]])


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Standard even-odd point-in-polygon test for a convex-ish closed polygon.

    Polygon is (K, 2) and may be open (last vertex != first). Closed
    polygons also work.
    """
    x, y = float(point[0]), float(point[1])
    inside = False
    K = len(polygon)
    j = K - 1
    for i in range(K):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi:
            inside = not inside
        j = i
    return inside


def _euler_xyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """Recover Euler XYZ angles (degrees) from a rotation matrix.

    Convention matches scipy's intrinsic 'XYZ' Euler decomposition: R = Rx(a) Ry(b) Rz(c).
    """
    sy = -R[2, 0]
    cy = float(np.sqrt(max(0.0, 1.0 - sy * sy)))
    if cy > 1e-6:
        x = float(np.arctan2(R[2, 1], R[2, 2]))
        y = float(np.arctan2(sy, cy))
        z = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        x = float(np.arctan2(-R[1, 2], R[1, 1]))
        y = float(np.arctan2(sy, cy))
        z = 0.0
    return np.degrees(np.array([x, y, z]))


def _build_probe_rotation(probe_axis_world: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix that maps probe-local axes to world axes.

    Probe-local convention (matches the simulator's IVUSProbe at zero
    rotation): probe long axis is the local +Y axis, imaging plane is the
    local xz plane.
    """
    target_y = probe_axis_world / (np.linalg.norm(probe_axis_world) + 1e-12)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(target_y, candidate)) > 0.95:
        candidate = np.array([0.0, 0.0, 1.0])
    target_x = candidate - np.dot(candidate, target_y) * target_y
    target_x /= np.linalg.norm(target_x) + 1e-12
    target_z = np.cross(target_x, target_y)
    R = np.column_stack([target_x, target_y, target_z])
    return R


def _ray_polygon_intersection_distance(
    polygons: list[np.ndarray], theta_rad: float, max_distance_mm: float
) -> float:
    """Distance from origin (0,0) to the nearest intersection of a ray at
    angle ``theta_rad`` with any of the closed polygons.

    The ray is parameterised as ``P = t * D`` with ``t >= 0`` and
    ``D = (cos theta, sin theta)``. Each polygon segment ``A + s (B - A)``
    is intersected with the ray by solving the 2x2 system

        [ Dx   -(Bx - Ax) ] [ t ]   [ Ax ]
        [ Dy   -(By - Ay) ] [ s ] = [ Ay ]

    keeping the smallest ``t > 0`` with ``0 <= s <= 1``. Returns NaN if no
    intersection is found within ``max_distance_mm``.
    """
    dx = float(np.cos(theta_rad))
    dy = float(np.sin(theta_rad))
    best = np.inf
    for poly in polygons:
        K = len(poly) - 1 if np.allclose(poly[0], poly[-1]) else len(poly)
        for i in range(K):
            ax, ay = float(poly[i, 0]), float(poly[i, 1])
            bx, by = float(poly[(i + 1) % K, 0]), float(poly[(i + 1) % K, 1])
            ex, ey = bx - ax, by - ay
            det = dx * (-ey) - dy * (-ex)
            if abs(det) < 1e-12:
                continue
            t = (ax * (-ey) - ay * (-ex)) / det
            s = (dx * ay - dy * ax) / det
            if t < 0.0 or s < 0.0 or s > 1.0:
                continue
            if t < best:
                best = t
    if best <= max_distance_mm:
        return float(best)
    return float("nan")


# ---------------------------------------------------------------------------
# Pose sampling
# ---------------------------------------------------------------------------


def _pick_branch(vessel: Vessel, rng: np.random.Generator) -> BranchHandle:
    weights = np.asarray([b.centerline.length_mm for b in vessel.branches], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(vessel.branches), p=weights))
    return vessel.branches[idx]


def _interpolate_lumen_contour(
    branch: BranchHandle, arclength_mm: float
) -> np.ndarray:
    """Linearly interpolate the lumen contour at a non-grid arclength.

    Returns a (M, 2) contour in the local cross-section plane (x along
    branch.normal, y along branch.binormal).
    """
    s_grid = np.linspace(0.0, branch.centerline.length_mm, branch.lumen_field.n_stations)
    s = float(np.clip(arclength_mm, s_grid[0], s_grid[-1]))
    if s <= s_grid[0]:
        return branch.lumen_field.contour(0)
    if s >= s_grid[-1]:
        return branch.lumen_field.contour(-1)
    j = int(np.searchsorted(s_grid, s) - 1)
    j = max(0, min(branch.lumen_field.n_stations - 2, j))
    t = (s - s_grid[j]) / (s_grid[j + 1] - s_grid[j])
    a = branch.lumen_field.contour(j)
    b = branch.lumen_field.contour(j + 1)
    return (1.0 - t) * a + t * b


def _sample_branch_arclength(
    branch: BranchHandle,
    rng: np.random.Generator,
    *,
    side_branch_ostium_bias_prob: float = 0.0,
    side_branch_ostium_arclength_frac: float = 0.25,
    endcap_clearance_mm: float = 2.5,
) -> float:
    """Sample arclength along ``branch``, optionally biasing side branches to the ostium.

    ``endcap_clearance_mm`` keeps the sampled position away from the cap
    discs at each end of the branch. With tilted imaging planes the
    catheter's rays sweep a cone of axial reach ``sin(tilt) * t_far`` --
    any cap within that reach makes ``wall_thickness ~ 0`` along some
    azimuth and the GT validity check rejects the pose. A 2.5 mm
    interior margin matches the typical 8 deg tilt + 17.5 mm t_far
    operating point and dramatically improves the pose-acceptance rate
    (96% rejection -> ~30% rejection in the dataset profile).

    Branches shorter than ``2 * endcap_clearance_mm`` fall back to the
    branch's middle third.
    """
    length = branch.centerline.length_mm
    if length <= 2.0 * endcap_clearance_mm:
        return float(rng.uniform(length / 3.0, 2.0 * length / 3.0))

    lo = endcap_clearance_mm
    hi = length - endcap_clearance_mm
    if (
        branch.name != "parent"
        and side_branch_ostium_bias_prob > 0.0
        and rng.random() < side_branch_ostium_bias_prob
    ):
        ostium_extent = max(length * side_branch_ostium_arclength_frac, lo + 1e-3)
        return float(rng.uniform(lo, min(hi, ostium_extent)))
    return float(rng.uniform(lo, hi))


def sample_pose(
    vessel: Vessel,
    rng: np.random.Generator,
    max_tilt_deg: float = 15.0,
    edge_margin_mm: float = 0.2,
    max_attempts: int = 64,
    side_branch_ostium_bias_prob: float = 0.0,
    side_branch_ostium_arclength_frac: float = 0.25,
    wall_contact_probability: float = 0.0,
    wall_contact_margin_mm: float = 0.05,
    guidewire_clearance_mm: float = 0.15,
) -> PoseSample:
    """Sample a random catheter pose inside the vessel lumen.

    The sampler picks a branch (weighted by arclength), an arclength along
    the branch, a 2D position inside the lumen contour at that arclength
    (rejection-sampled inside the contour, with at least ``edge_margin_mm``
    clearance from the wall), then samples a small probe-axis tilt.

    Parameters
    ----------
    max_tilt_deg:
        Maximum angle between the probe long axis and the local vessel
        tangent. Tilt is sampled uniformly in [0, max_tilt_deg] with a
        random azimuth around the tangent.
    edge_margin_mm:
        Minimum clearance from the lumen wall, in the cross-section plane.
        Set to 0 to allow probe-against-wall poses (useful for training the
        contact-signal head).
    max_attempts:
        Maximum rejection-sampling attempts before giving up; raises
        :class:`RuntimeError` if exceeded.
    wall_contact_probability:
        Probability of forcing the probe to sit against the lumen wall
        (creating the bright contact rim + tangential shadow real probes
        produce on side-resting frames). When this fires, the candidate
        is rejection-sampled at the lumen boundary instead of inside it
        and ``edge_margin_mm`` is ignored.
    wall_contact_margin_mm:
        How close to the wall a "wall-contact" sample is forced; <=0
        means flush against the wall.
    guidewire_clearance_mm:
        Minimum in-plane clearance the probe centre must maintain from
        the guidewire surface. Ignored when the vessel has no wire.
    """
    branch = _pick_branch(vessel, rng)
    s = _sample_branch_arclength(
        branch,
        rng,
        side_branch_ostium_bias_prob=side_branch_ostium_bias_prob,
        side_branch_ostium_arclength_frac=side_branch_ostium_arclength_frac,
    )
    contour_local = _interpolate_lumen_contour(branch, s)

    bbox_min = contour_local.min(axis=0)
    bbox_max = contour_local.max(axis=0)
    margin = max(edge_margin_mm, 0.0)

    wall_contact = (
        wall_contact_probability > 0.0
        and branch.is_parent
        and rng.random() < wall_contact_probability
    )
    guidewire_cfg = vessel.config.guidewire if branch.is_parent else None

    chosen_local = None
    eccentricity = 0.0
    for _ in range(max_attempts):
        if wall_contact:
            candidate = _sample_wall_contact_point(contour_local, rng, wall_contact_margin_mm)
        else:
            candidate = rng.uniform(bbox_min, bbox_max)
            if not _point_in_polygon(candidate, contour_local):
                continue
            if margin > 0.0:
                min_dist = _min_distance_to_polygon_edge(candidate, contour_local)
                if min_dist < margin:
                    continue
        if guidewire_cfg is not None:
            from vesselgen.guidewire import guidewire_clearance_mm as _wire_clear
            if _wire_clear(candidate, guidewire_cfg) < float(guidewire_clearance_mm):
                continue
        chosen_local = candidate
        eccentricity = float(np.linalg.norm(candidate))
        break
    if chosen_local is None:
        raise RuntimeError(
            f"sample_pose: could not find an in-lumen point in branch '{branch.name}'"
            f" within {max_attempts} attempts (try lowering edge_margin_mm)"
        )

    frame = branch.centerline.frame(s)
    position_world = (
        frame.position
        + chosen_local[0] * frame.normal
        + chosen_local[1] * frame.binormal
    )

    tilt_deg = float(rng.uniform(0.0, max_tilt_deg))
    tilt_azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    probe_axis = (
        np.cos(np.radians(tilt_deg)) * frame.tangent
        + np.sin(np.radians(tilt_deg)) * (
            np.cos(tilt_azimuth) * frame.normal + np.sin(tilt_azimuth) * frame.binormal
        )
    )
    probe_axis /= np.linalg.norm(probe_axis) + 1e-12

    R = _build_probe_rotation(probe_axis)
    euler_deg = _euler_xyz_from_matrix(R)

    return PoseSample(
        position=position_world,
        rotation_euler_deg=euler_deg,
        rotation_matrix=R,
        probe_axis_world=probe_axis,
        branch_id=branch.branch_id,
        branch_name=branch.name,
        arclength_mm=s,
        centerline_offset_mm=eccentricity,
        tilt_deg=tilt_deg,
    )


def pose_at(
    vessel: "Vessel",
    position: np.ndarray,
    probe_axis_world: np.ndarray | None = None,
    require_inside_lumen: bool = True,
) -> PoseSample:
    """Build a :class:`PoseSample` at a user-specified 3D position.

    Use this when you want the probe at a *specific* location rather than a
    random sample. The probe long axis defaults to the local tangent of the
    nearest branch's centerline at the projection of ``position`` onto that
    centerline (so the probe is roughly aligned with the vessel even when
    you only supply a position).

    Parameters
    ----------
    vessel:
        The vessel to query.
    position:
        World-space (3,) position in mm. Must lie inside the lumen if
        ``require_inside_lumen`` is True.
    probe_axis_world:
        Optional (3,) probe long-axis direction in world coordinates. If
        ``None``, falls back to the nearest branch's local tangent.
    require_inside_lumen:
        When True (default), raises :class:`ValueError` if ``position`` is
        not inside ``vessel.lumen_mesh``.
    """
    position = np.asarray(position, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"position must be shape (3,), got {position.shape}")
    if require_inside_lumen and not bool(vessel.lumen_mesh.contains([position])[0]):
        raise ValueError(f"position {position.tolist()} is not inside the lumen")

    # Pick the branch whose centerline the position is closest to, and take
    # its local tangent as the default probe axis.
    nearest_branch = vessel.branches[0]
    nearest_offset = float("inf")
    nearest_arclength = 0.0
    for b in vessel.branches:
        s, off = b.centerline.project(position)
        d = float(np.linalg.norm(off))
        if d < nearest_offset:
            nearest_offset = d
            nearest_branch = b
            nearest_arclength = s

    if probe_axis_world is None:
        probe_axis = nearest_branch.centerline.tangent(nearest_arclength)
    else:
        probe_axis = np.asarray(probe_axis_world, dtype=float)
        n = float(np.linalg.norm(probe_axis))
        if n < 1e-9:
            raise ValueError("probe_axis_world must be a non-zero vector")
        probe_axis = probe_axis / n

    tilt = float(np.degrees(np.arccos(
        np.clip(float(np.dot(probe_axis, nearest_branch.centerline.tangent(nearest_arclength))),
                -1.0, 1.0)
    )))
    R = _build_probe_rotation(probe_axis)
    return PoseSample(
        position=position.copy(),
        rotation_euler_deg=_euler_xyz_from_matrix(R),
        rotation_matrix=R,
        probe_axis_world=probe_axis,
        branch_id=nearest_branch.branch_id,
        branch_name=nearest_branch.name,
        arclength_mm=nearest_arclength,
        centerline_offset_mm=nearest_offset,
        tilt_deg=tilt,
    )


def sample_pose_in_branch(
    vessel: "Vessel",
    branch_name: str,
    rng: np.random.Generator,
    arclength_mm: float | None = None,
    max_tilt_deg: float = 15.0,
    edge_margin_mm: float = 0.2,
    max_attempts: int = 64,
) -> PoseSample:
    """Sample a pose constrained to a specific branch.

    ``arclength_mm`` pins the sample to a particular axial station along
    the branch (useful for stratified sampling along the vessel); if None,
    the arclength is sampled uniformly along the branch like
    :func:`sample_pose`.
    """
    target = None
    for b in vessel.branches:
        if b.name == branch_name:
            target = b
            break
    if target is None:
        raise KeyError(f"no branch named '{branch_name}'")

    s = (
        float(rng.uniform(0.0, target.centerline.length_mm))
        if arclength_mm is None
        else float(np.clip(arclength_mm, 0.0, target.centerline.length_mm))
    )
    contour_local = _interpolate_lumen_contour(target, s)
    bbox_min = contour_local.min(axis=0)
    bbox_max = contour_local.max(axis=0)
    margin = max(edge_margin_mm, 0.0)
    chosen_local = None
    eccentricity = 0.0
    for _ in range(max_attempts):
        candidate = rng.uniform(bbox_min, bbox_max)
        if not _point_in_polygon(candidate, contour_local):
            continue
        if margin > 0.0:
            if _min_distance_to_polygon_edge(candidate, contour_local) < margin:
                continue
        chosen_local = candidate
        eccentricity = float(np.linalg.norm(candidate))
        break
    if chosen_local is None:
        raise RuntimeError(
            f"sample_pose_in_branch: could not find an in-lumen point in '{branch_name}'"
            f" at s={s:.2f} mm within {max_attempts} attempts"
        )
    frame = target.centerline.frame(s)
    position_world = (
        frame.position
        + chosen_local[0] * frame.normal
        + chosen_local[1] * frame.binormal
    )
    tilt_deg = float(rng.uniform(0.0, max_tilt_deg))
    tilt_azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    probe_axis = (
        np.cos(np.radians(tilt_deg)) * frame.tangent
        + np.sin(np.radians(tilt_deg)) * (
            np.cos(tilt_azimuth) * frame.normal + np.sin(tilt_azimuth) * frame.binormal
        )
    )
    probe_axis /= np.linalg.norm(probe_axis) + 1e-12
    R = _build_probe_rotation(probe_axis)
    return PoseSample(
        position=position_world,
        rotation_euler_deg=_euler_xyz_from_matrix(R),
        rotation_matrix=R,
        probe_axis_world=probe_axis,
        branch_id=target.branch_id,
        branch_name=target.name,
        arclength_mm=s,
        centerline_offset_mm=eccentricity,
        tilt_deg=tilt_deg,
    )


def _sample_wall_contact_point(
    polygon: np.ndarray,
    rng: np.random.Generator,
    inset_mm: float,
) -> np.ndarray:
    """Sample a point arbitrarily close to (but inside) the lumen boundary.

    Picks a random polygon edge weighted by length, draws a point on
    the edge, then walks inward along the inward normal by ``inset_mm``
    (clamped to 0). The result simulates the probe resting against the
    vessel wall.
    """

    K = len(polygon)
    edges = polygon[1:] if np.allclose(polygon[0], polygon[-1]) else polygon
    K_edges = len(edges)
    a_idx = rng.integers(0, K_edges)
    a = polygon[a_idx]
    b = polygon[(a_idx + 1) % K_edges]
    t = float(rng.uniform(0.05, 0.95))
    edge_pt = a + t * (b - a)
    edge_dir = b - a
    edge_len = float(np.linalg.norm(edge_dir))
    if edge_len < 1e-6:
        return edge_pt
    edge_dir /= edge_len
    inward = np.array([-edge_dir[1], edge_dir[0]])
    polygon_centroid = polygon.mean(axis=0)
    if np.dot(polygon_centroid - edge_pt, inward) < 0:
        inward = -inward
    return edge_pt + max(inset_mm, 0.0) * inward


def _min_distance_to_polygon_edge(point: np.ndarray, polygon: np.ndarray) -> float:
    """Minimum distance from ``point`` to any edge of the closed polygon."""
    K = len(polygon)
    best = np.inf
    for i in range(K):
        a = polygon[i]
        b = polygon[(i + 1) % K]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            continue
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(point - proj))
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------


def _section_to_polygons(section, plane_origin: np.ndarray,
                          probe_x: np.ndarray, probe_z: np.ndarray) -> list[np.ndarray]:
    """Project a trimesh Path3D section into 2D polygons in the imaging plane.

    Each discrete polyline of the 3D path is projected onto the
    (probe_x, probe_z) basis spanning the plane, with ``plane_origin``
    placed at (0, 0). The probe long axis (probe_y / plane normal) is
    perpendicular to this 2D basis so the projection is exact.
    """
    if section is None:
        return []
    polygons: list[np.ndarray] = []
    for entity_pts in section.discrete:
        pts3 = np.asarray(entity_pts, dtype=float)
        if len(pts3) < 2:
            continue
        rel = pts3 - plane_origin[None, :]
        xs = rel @ probe_x
        ys = rel @ probe_z
        polygons.append(np.column_stack([xs, ys]))
    return polygons


def _polygon_signed_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    if _polygon_signed_area(polygon) < 0:
        return polygon[::-1].copy()
    return polygon


def ground_truth_at(
    vessel: Vessel,
    pose: PoseSample,
    n_angles: int = 360,
    max_distance_mm: float = 30.0,
) -> GroundTruth:
    """Compute per-frame ground truth at the given pose.

    Slices the lumen and outer meshes with the imaging plane (perpendicular
    to ``pose.probe_axis_world``, through ``pose.position``) and converts
    the resulting polygons to polar (per-angle distance) ground truth.
    """
    plane_origin = pose.position
    plane_normal = pose.probe_axis_world
    probe_x = pose.rotation_matrix[:, 0]
    probe_z = pose.rotation_matrix[:, 2]

    lumen_section = vessel.lumen_mesh.section(plane_origin=plane_origin,
                                                plane_normal=plane_normal)
    outer_section = vessel.outer_mesh.section(plane_origin=plane_origin,
                                                plane_normal=plane_normal)

    lumen_polys = _section_to_polygons(lumen_section, plane_origin, probe_x, probe_z)
    outer_polys = _section_to_polygons(outer_section, plane_origin, probe_x, probe_z)
    lumen_polys = [_ensure_ccw(p) for p in lumen_polys if len(p) >= 3]
    outer_polys = [_ensure_ccw(p) for p in outer_polys if len(p) >= 3]

    thetas = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    d_lumen = np.full(n_angles, np.nan)
    d_outer = np.full(n_angles, np.nan)
    for i, th in enumerate(thetas):
        d_lumen[i] = _ray_polygon_intersection_distance(lumen_polys, th, max_distance_mm)
        d_outer[i] = _ray_polygon_intersection_distance(outer_polys, th, max_distance_mm)
    wall_thickness = d_outer - d_lumen

    lumen_csa = float(sum(abs(_polygon_signed_area(p)) for p in lumen_polys))
    outer_csa = float(sum(abs(_polygon_signed_area(p)) for p in outer_polys))

    visible: list[int] = [pose.branch_id]
    for b in vessel.branches:
        if b.branch_id == pose.branch_id:
            continue
        s_proj, offset = b.centerline.project(pose.position)
        if np.linalg.norm(offset) <= b.lumen_field.mean_radius.max() * 1.5:
            visible.append(b.branch_id)

    eq_diam = 2.0 * float(np.sqrt(max(0.0, lumen_csa) / np.pi))

    surface_polygons: list[tuple[str, str, list[np.ndarray]]] = []
    for surface in vessel.surfaces:
        polys = _section_to_polygons(
            surface.mesh.section(plane_origin=plane_origin, plane_normal=plane_normal),
            plane_origin, probe_x, probe_z,
        )
        polys = [_ensure_ccw(p) for p in polys if len(p) >= 3]
        surface_polygons.append((surface.name, surface.material_name, polys))

    lesion_polygons: list[tuple[str, str, list[np.ndarray]]] = []
    for lesion in vessel.lesions:
        polys = _section_to_polygons(
            lesion.mesh.section(plane_origin=plane_origin, plane_normal=plane_normal),
            plane_origin, probe_x, probe_z,
        )
        polys = [_ensure_ccw(p) for p in polys if len(p) >= 3]
        lesion_polygons.append((lesion.name, lesion.material_name, polys))

    guidewire_polygons: list[np.ndarray] = []
    if vessel.guidewire is not None:
        polys = _section_to_polygons(
            vessel.guidewire.mesh.section(
                plane_origin=plane_origin, plane_normal=plane_normal
            ),
            plane_origin, probe_x, probe_z,
        )
        guidewire_polygons = [_ensure_ccw(p) for p in polys if len(p) >= 3]

    return GroundTruth(
        n_angles=n_angles,
        thetas_rad=thetas,
        distance_to_lumen_wall_mm=d_lumen,
        distance_to_outer_wall_mm=d_outer,
        wall_thickness_mm=wall_thickness,
        lumen_contour_polygons=lumen_polys,
        outer_contour_polygons=outer_polys,
        lumen_csa_mm2=lumen_csa,
        outer_csa_mm2=outer_csa,
        branch_ids_visible=visible,
        equivalent_lumen_diameter_mm=eq_diam,
        surface_polygons=surface_polygons,
        lesion_polygons=lesion_polygons,
        guidewire_polygons=guidewire_polygons,
    )


def _imaging_ray_directions_world(
    pose: PoseSample, thetas_rad: np.ndarray
) -> np.ndarray:
    """Unit ray directions in world space for each imaging-plane angle."""
    probe_x = pose.rotation_matrix[:, 0]
    probe_z = pose.rotation_matrix[:, 2]
    dirs = np.stack([np.cos(thetas_rad), np.sin(thetas_rad)], axis=1)
    dirs_w = dirs[:, 0:1] * probe_x + dirs[:, 1:2] * probe_z
    dirs_w /= np.linalg.norm(dirs_w, axis=1, keepdims=True) + 1e-12
    return dirs_w


def _project_to_imaging_plane(vector: np.ndarray, probe_axis: np.ndarray) -> np.ndarray:
    """Remove probe-axis component so ``vector`` lies in the imaging plane."""
    axis = probe_axis / (np.linalg.norm(probe_axis) + 1e-12)
    vector = np.asarray(vector, dtype=float)
    return vector - np.dot(vector, axis) * axis


def _sector_around_planar_direction(
    pose: PoseSample,
    thetas_rad: np.ndarray,
    direction_world: np.ndarray,
    *,
    half_angle_deg: float,
) -> np.ndarray:
    """Angles whose rays fall within a cone around ``direction_world`` (in-plane)."""
    in_plane = _project_to_imaging_plane(direction_world, pose.probe_axis_world)
    norm = float(np.linalg.norm(in_plane))
    if norm < 1e-6:
        return np.zeros(len(thetas_rad), dtype=bool)
    dir_u = in_plane / norm
    cos_angle = (_imaging_ray_directions_world(pose, thetas_rad) @ dir_u).ravel()
    return cos_angle >= float(np.cos(np.radians(half_angle_deg)))


def _branch_mean_lumen_radius_mm(branch: BranchHandle, arclength_mm: float) -> float:
    """Typical in-plane lumen radius at ``arclength_mm`` (mm)."""
    contour = _interpolate_lumen_contour(branch, arclength_mm)
    return float(np.mean(np.linalg.norm(contour, axis=1)))


def side_branch_imaging_sector_mask(
    vessel: Vessel,
    pose: PoseSample,
    thetas_rad: np.ndarray,
    gt: GroundTruth | None = None,
    *,
    half_angle_deg: float = 45.0,
    local_lumen_median_frac: float = 0.75,
    local_lumen_max_mm: float | None = None,
) -> np.ndarray:
    """Angles that view the sampled side branch toward its caps or ostium.

    When the probe is aligned with the branch, rays are nearly perpendicular
    to the centerline tangent, so a tangent cone is empty.  Wedges aimed at
    the side branch distal cap, proximal cap, and parent ostium are combined
    with rays whose nearest lumen hit is much closer than the pose median
    (local side-branch lumen vs parent lumen in the same frame).
    """
    branch = None
    for b in vessel.branches:
        if b.branch_id == pose.branch_id:
            branch = b
            break
    if branch is None or branch.is_parent:
        return np.zeros(len(thetas_rad), dtype=bool)

    sector = np.zeros(len(thetas_rad), dtype=bool)
    sector |= _sector_around_planar_direction(
        pose,
        thetas_rad,
        branch.centerline.positions[-1] - pose.position,
        half_angle_deg=half_angle_deg,
    )
    sector |= _sector_around_planar_direction(
        pose,
        thetas_rad,
        branch.centerline.positions[0] - pose.position,
        half_angle_deg=half_angle_deg,
    )
    if branch.parent_attachment_arclength_mm is not None:
        parent = vessel.branches[0]
        attach_pt = parent.centerline.position(branch.parent_attachment_arclength_mm)
        sector |= _sector_around_planar_direction(
            pose,
            thetas_rad,
            attach_pt - pose.position,
            half_angle_deg=half_angle_deg,
        )

    if gt is not None:
        d_lumen = np.asarray(gt.distance_to_lumen_wall_mm, dtype=float)
        finite = np.isfinite(d_lumen)
        if finite.any():
            median_d = float(np.nanmedian(d_lumen[finite]))
            if local_lumen_max_mm is None:
                local_lumen_max_mm = max(
                    8.0, 1.5 * _branch_mean_lumen_radius_mm(branch, pose.arclength_mm)
                )
            cutoff = min(median_d * local_lumen_median_frac, local_lumen_max_mm)
            sector |= finite & (d_lumen <= cutoff)

    return sector


def ground_truth_side_branch_sector_invalid(
    vessel: Vessel,
    pose: PoseSample,
    gt: GroundTruth,
    *,
    endcap_wall_mm: float = 0.08,
    half_angle_deg: float = 45.0,
) -> bool:
    """True when the side-branch angular sector shows a mesh end cap."""
    if pose.branch_name == "parent":
        return False
    sector = side_branch_imaging_sector_mask(
        vessel, pose, gt.thetas_rad, gt, half_angle_deg=half_angle_deg
    )
    if not sector.any():
        return False
    wall = np.asarray(gt.wall_thickness_mm, dtype=float)
    finite = sector & np.isfinite(wall)
    if not finite.any():
        return True
    return bool(np.any(wall[finite] < endcap_wall_mm))


def ground_truth_shows_endcap(
    gt: GroundTruth,
    *,
    min_wall_mm: float = 0.08,
) -> bool:
    """Return True when any ray sees coincident lumen/outer hits (mesh end cap).

    Branch meshes are closed with centroid fan caps. When the imaging plane
    intersects a cap disc, lumen and outer surfaces coincide and the computed
    wall thickness collapses to ~0. This is a GT artifact, not real anatomy.
    """
    wall = np.asarray(gt.wall_thickness_mm, dtype=float)
    finite = np.isfinite(wall)
    if not finite.any():
        return True
    return bool(np.any(wall[finite] < min_wall_mm))


def ground_truth_is_valid_pose(
    vessel: Vessel,
    pose: PoseSample,
    gt: GroundTruth,
    *,
    min_finite_fraction: float = 0.85,
    endcap_wall_mm: float = 0.08,
    side_branch_cone_half_angle_deg: float = 45.0,
) -> bool:
    """Reject poses whose GT is incomplete or includes mesh cap artifacts."""
    finite = np.isfinite(gt.distance_to_lumen_wall_mm) & np.isfinite(
        gt.distance_to_outer_wall_mm
    )
    if finite.mean() < min_finite_fraction or gt.lumen_csa_mm2 <= 0.0:
        return False
    if pose.branch_name != "parent":
        return not ground_truth_side_branch_sector_invalid(
            vessel,
            pose,
            gt,
            endcap_wall_mm=endcap_wall_mm,
            half_angle_deg=side_branch_cone_half_angle_deg,
        )
    return not ground_truth_shows_endcap(gt, min_wall_mm=endcap_wall_mm)
