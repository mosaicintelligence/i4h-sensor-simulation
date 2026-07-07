"""Guidewire mesh + sampling helpers.

The guidewire is modelled as a thin cylinder running along the parent
branch's centerline with a constant lateral offset in the cross-section
plane. Because the (current 2026 v1) centerline is straight, the wire
is a simple offset cylinder; the same code falls out for curved
centerlines once they exist because vertices are placed in each
station's local frame.

The wire material is ``tungsten`` (already in the simulator's material
table). The impedance contrast between tungsten and blood is so large
that the wire produces a bright proximal reflection and a near-complete
acoustic shadow behind it -- exactly the artifact deep-learning models
trained on real IVUS data expect to see.

Catheter pose sampling needs to avoid the wire so the probe doesn't
end up inside the metal. :func:`guidewire_clearance_mm` returns the
in-plane distance from a candidate cross-section position to the wire's
in-plane position; :func:`pose_avoids_guidewire` is the convenience
boolean rejection used by :func:`vesselgen.sampling.sample_pose`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import GuidewireConfig


# A guidewire never has to look like a perfect cylinder -- IVUS rays
# only see the silhouette. Keep the tessellation cheap.
_GUIDEWIRE_RADIAL_SEGMENTS = 16
# Sample the wire mesh more finely than the parent centerline so the
# wire's caps end up well past the parent centerline endpoints.
_GUIDEWIRE_END_OVERHANG_MM = 3.0


@dataclass
class GuidewireArtifacts:
    """All bookkeeping the rest of the pipeline needs about the wire."""

    config: GuidewireConfig
    mesh: trimesh.Trimesh
    material_name: str
    # In-plane position of the wire centre in the parent branch's local
    # cross-section frame (x along normal, y along binormal). Constant
    # along the wire for a straight parent centerline.
    in_plane_position_mm: np.ndarray
    # 3D positions of the wire's central axis at every station (N, 3)
    # used for any per-station collision checks.
    axis_positions_world: np.ndarray


def _local_offset(config: GuidewireConfig) -> np.ndarray:
    az = float(np.deg2rad(config.offset_azimuth_deg))
    return np.array(
        [config.lateral_offset_mm * np.cos(az), config.lateral_offset_mm * np.sin(az)],
        dtype=float,
    )


def build_guidewire(
    config: GuidewireConfig,
    parent_centerline: Centerline,
) -> GuidewireArtifacts:
    """Build a thin closed cylinder representing the guidewire.

    The wire follows the parent centerline at a constant in-plane
    offset, with small end overhangs so its caps fall outside the
    parent vessel mesh (catheter-eye views never see the cap).
    """

    in_plane = _local_offset(config)
    n_radial = _GUIDEWIRE_RADIAL_SEGMENTS
    radius = 0.5 * float(config.diameter_mm)

    # Extend the wire slightly past both endpoints so its caps live
    # outside the parent vessel mesh.
    overhang = float(_GUIDEWIRE_END_OVERHANG_MM)
    length_mm = parent_centerline.length_mm + 2.0 * overhang
    n_stations = max(parent_centerline.stations.shape[0], 16)
    s_grid = np.linspace(-overhang, parent_centerline.length_mm + overhang, n_stations)

    # Axis positions: for each station, position along centerline + lateral offset.
    direction = parent_centerline.direction
    normal = parent_centerline.normals[0]
    binormal = parent_centerline.binormals[0]
    origin = parent_centerline.origin

    axis_positions = (
        origin[None, :]
        + s_grid[:, None] * direction[None, :]
        + in_plane[0] * normal[None, :]
        + in_plane[1] * binormal[None, :]
    )

    thetas = np.linspace(0.0, 2.0 * np.pi, n_radial, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    ring = radius * (cos_t[:, None] * normal[None, :] + sin_t[:, None] * binormal[None, :])

    vertices = np.empty((n_stations, n_radial, 3), dtype=float)
    for i in range(n_stations):
        vertices[i] = axis_positions[i][None, :] + ring

    flat = vertices.reshape(-1, 3)
    faces: list[list[int]] = []
    for i in range(n_stations - 1):
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v00 = i * n_radial + j
            v01 = i * n_radial + j_next
            v10 = (i + 1) * n_radial + j
            v11 = (i + 1) * n_radial + j_next
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])

    proximal_centroid = axis_positions[0]
    distal_centroid = axis_positions[-1]
    extended = np.vstack([flat, proximal_centroid[None, :], distal_centroid[None, :]])
    proximal_idx = len(flat)
    distal_idx = len(flat) + 1
    for j in range(n_radial):
        j_next = (j + 1) % n_radial
        faces.append([proximal_idx, j_next, j])
    for j in range(n_radial):
        j_next = (j + 1) % n_radial
        a = (n_stations - 1) * n_radial + j
        b = (n_stations - 1) * n_radial + j_next
        faces.append([distal_idx, a, b])

    mesh = trimesh.Trimesh(vertices=extended, faces=np.asarray(faces, dtype=np.int64), process=True)
    if mesh.volume < 0.0:
        mesh.invert()

    return GuidewireArtifacts(
        config=config,
        mesh=mesh,
        material_name=config.material_name,
        in_plane_position_mm=in_plane,
        axis_positions_world=axis_positions,
    )


# ---------------------------------------------------------------------------
# Pose rejection helpers
# ---------------------------------------------------------------------------


def guidewire_in_plane_position(config: GuidewireConfig) -> np.ndarray:
    """In-plane (x, y) wire centre in the parent branch's local frame."""

    return _local_offset(config)


def guidewire_clearance_mm(
    candidate_local_mm: np.ndarray,
    config: GuidewireConfig,
) -> float:
    """Distance from a candidate cross-section position to the wire surface.

    Negative when the candidate lies *inside* the wire. ``candidate_local_mm``
    is the (x, y) probe position in the same local cross-section frame as
    the wire (i.e. x along parent normal, y along parent binormal).
    """

    wire_xy = _local_offset(config)
    dist_to_centre = float(np.linalg.norm(np.asarray(candidate_local_mm) - wire_xy))
    return dist_to_centre - 0.5 * float(config.diameter_mm)


def pose_avoids_guidewire(
    pose_local_xy_mm: np.ndarray,
    config: GuidewireConfig,
    *,
    clearance_mm: float = 0.15,
) -> bool:
    """True iff the probe centre clears the wire by at least ``clearance_mm``.

    Real catheters cannot physically intersect the guidewire so the
    sampler should reject candidates that violate this clearance.
    """

    return guidewire_clearance_mm(pose_local_xy_mm, config) >= float(clearance_mm)
