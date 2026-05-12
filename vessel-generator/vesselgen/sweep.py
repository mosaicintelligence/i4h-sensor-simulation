"""Sweep cross-sections along a centerline to produce closed trimeshes.

For one branch the input is:
- a :class:`Centerline` with ``N`` stations and a frame at each station
- a :class:`CrossSectionField` with the lumen radii at every (station, angle)
- a :class:`WallField` with the per-angle wall thickness

The output is:
- a closed (capped, watertight) lumen mesh
- a closed outer (adventitia) mesh that nests around the lumen

Both meshes are constructed with **outward-facing** normals because that
is the convention required by trimesh's ``contains`` test and by the
manifold3d boolean engine used in :mod:`vesselgen.bifurcation`. The OBJ
files written by :mod:`vesselgen.io` are inverted at export time so the
simulator (which expects inward normals on phantom surfaces -- see the
``utils/phantom_maker.py`` cylinder convention in the simulator
repository) sees the right orientation.

Caps are added by fanning each end's contour to a centroid vertex. This
keeps the meshes watertight so boolean union produces a clean ostium.
"""

from __future__ import annotations

import numpy as np
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.cross_section import CrossSectionField
from vesselgen.wall import WallField


def _build_tube_vertices(centerline: Centerline, radii: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """3D vertex grid (N_stations, M_angles, 3) for one swept surface."""
    n_stations, n_angles = radii.shape
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    vertices = np.empty((n_stations, n_angles, 3), dtype=np.float64)
    for i in range(n_stations):
        pos = centerline.positions[i]
        normal = centerline.normals[i]
        binormal = centerline.binormals[i]
        local_x = (radii[i] * cos_t)[:, None] * normal[None, :]
        local_y = (radii[i] * sin_t)[:, None] * binormal[None, :]
        vertices[i] = pos[None, :] + local_x + local_y
    return vertices


def _faces_with_outward_normals(n_stations: int, n_angles: int) -> np.ndarray:
    """Triangle faces for a tube with outward-pointing normals.

    The internal trimesh objects use the standard convention of outward
    normals (so trimesh.contains, trimesh.boolean.union, etc. all behave
    correctly: manifold3d treats the surface as the boundary of a solid).
    The OBJ files written for the simulator are inverted at export time
    so the simulator sees inward-facing normals.

    Vertices are flattened in (station, angle) order; global index of
    (i, j) is ``i * n_angles + j``. With tangent = +Y, normal = +X,
    binormal = -Z, the winding below puts the right-hand rule normal
    pointing radially outward.
    """
    faces = []
    for i in range(n_stations - 1):
        for j in range(n_angles):
            j_next = (j + 1) % n_angles
            v00 = i * n_angles + j
            v01 = i * n_angles + j_next
            v10 = (i + 1) * n_angles + j
            v11 = (i + 1) * n_angles + j_next
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])
    return np.asarray(faces, dtype=np.int64)


def _add_caps_with_outward_normals(
    vertices: np.ndarray,
    faces: list[list[int]],
    n_stations: int,
    n_angles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append two cap fans (proximal and distal) to a tube mesh.

    Cap winding matches the wall winding so the resulting closed mesh has
    consistent outward-facing normals throughout. With the default basis,
    the proximal cap (at the smaller-Y end) needs an outward normal of -Y
    and the distal cap (at the larger-Y end) needs +Y.
    """
    flat = vertices.reshape(-1, 3)
    proximal_centroid = flat[:n_angles].mean(axis=0)
    distal_centroid = flat[(n_stations - 1) * n_angles : n_stations * n_angles].mean(axis=0)

    extended = np.vstack([flat, proximal_centroid[None, :], distal_centroid[None, :]])
    proximal_idx = len(flat)
    distal_idx = len(flat) + 1

    new_faces = list(faces)
    for j in range(n_angles):
        j_next = (j + 1) % n_angles
        new_faces.append([proximal_idx, j_next, j])
    for j in range(n_angles):
        j_next = (j + 1) % n_angles
        a = (n_stations - 1) * n_angles + j
        b = (n_stations - 1) * n_angles + j_next
        new_faces.append([distal_idx, a, b])

    return extended, np.asarray(new_faces, dtype=np.int64)


def _build_closed_mesh(
    centerline: Centerline, radii: np.ndarray, thetas: np.ndarray
) -> trimesh.Trimesh:
    n_stations, n_angles = radii.shape
    grid = _build_tube_vertices(centerline, radii, thetas)
    faces_only_walls = _faces_with_outward_normals(n_stations, n_angles).tolist()
    vertices, faces = _add_caps_with_outward_normals(
        grid, faces_only_walls, n_stations, n_angles
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh


def sweep_branch(
    centerline: Centerline,
    lumen: CrossSectionField,
    wall: WallField,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Build the closed lumen and outer meshes for one branch.

    Returns (lumen_mesh, outer_mesh). Both are closed, watertight (after
    trimesh's processing), and have inward-facing normals on the walls and
    inward-facing caps.
    """
    if lumen.radii.shape != wall.thicknesses.shape:
        raise ValueError(
            f"lumen and wall fields must share grid shape, got "
            f"{lumen.radii.shape} and {wall.thicknesses.shape}"
        )

    lumen_mesh = _build_closed_mesh(centerline, lumen.radii, lumen.thetas)
    outer_radii = lumen.radii + wall.thicknesses
    outer_mesh = _build_closed_mesh(centerline, outer_radii, lumen.thetas)
    return lumen_mesh, outer_mesh
