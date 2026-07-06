"""Tests for in-wall lesion mesh construction."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import (
    CalcificationLesionConfig,
    CenterlineConfig,
    CrossSectionConfig,
    LESION_KIND_TO_MATERIAL,
    LayeredWallConfig,
)
from vesselgen.cross_section import build_cross_sections
from vesselgen.inclusions import build_all_lesions, build_lesion_mesh
from vesselgen.wall import build_layered_wall


@pytest.fixture()
def fields():
    rng = np.random.default_rng(42)
    centerline_cfg = CenterlineConfig(length_mm=50.0, n_stations=40)
    xs_cfg = CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.3, n_angles=72)
    centerline = Centerline(centerline_cfg)
    lumen = build_cross_sections(xs_cfg, centerline_cfg.length_mm, centerline_cfg.n_stations, rng)
    wall_cfg = LayeredWallConfig.trilaminar(total_thickness_mm=0.9)
    wall = build_layered_wall(wall_cfg, lumen, length_mm=centerline_cfg.length_mm, rng=rng)
    return centerline, lumen, wall


def test_lesion_mesh_is_watertight_and_inside_wall(fields):
    centerline, lumen, wall = fields
    cfg = CalcificationLesionConfig(
        arclength_frac=0.5,
        azimuth_deg=90.0,
        kind="hard",
        arc_extent_deg=60.0,
        axial_extent_mm=8.0,
        inner_offset_frac=0.05,
        outer_offset_frac=0.85,
        seed=0,
    )
    mesh = build_lesion_mesh(cfg, centerline, lumen, wall)

    assert mesh.is_watertight, "lesion mesh should be watertight"
    assert mesh.volume > 0.0

    lumen_mesh = trimesh.Trimesh(
        vertices=_tube_points_with_caps(centerline, lumen.radii, lumen.thetas),
        faces=_tube_faces(*lumen.radii.shape),
        process=True,
    )
    outer_mesh = trimesh.Trimesh(
        vertices=_tube_points_with_caps(centerline, wall.interface_radii[-1], lumen.thetas),
        faces=_tube_faces(*lumen.radii.shape),
        process=True,
    )

    centroid = mesh.centroid
    assert outer_mesh.contains([centroid])[0], "lesion centroid must lie inside outer wall"
    assert not lumen_mesh.contains([centroid])[0], "lesion centroid must lie outside the lumen"


def test_lesion_volume_grows_with_radial_extent(fields):
    centerline, lumen, wall = fields
    small = CalcificationLesionConfig(
        arclength_frac=0.5, azimuth_deg=0.0, kind="hard",
        arc_extent_deg=60.0, axial_extent_mm=8.0,
        inner_offset_frac=0.3, outer_offset_frac=0.5, seed=1,
    )
    big = CalcificationLesionConfig(
        arclength_frac=0.5, azimuth_deg=0.0, kind="hard",
        arc_extent_deg=60.0, axial_extent_mm=8.0,
        inner_offset_frac=0.05, outer_offset_frac=0.95, seed=1,
    )
    v_small = build_lesion_mesh(small, centerline, lumen, wall).volume
    v_big = build_lesion_mesh(big, centerline, lumen, wall).volume
    assert v_big > v_small * 2.0


@pytest.mark.parametrize("kind", list(LESION_KIND_TO_MATERIAL))
def test_all_lesion_kinds_build(fields, kind):
    centerline, lumen, wall = fields
    cfg = CalcificationLesionConfig(
        arclength_frac=0.5, azimuth_deg=0.0, kind=kind,
        arc_extent_deg=60.0, axial_extent_mm=6.0, seed=2,
    )
    mesh = build_lesion_mesh(cfg, centerline, lumen, wall)
    assert mesh.is_watertight
    assert mesh.volume > 0.0


def test_build_all_lesions_attaches_correct_materials(fields):
    centerline, lumen, wall = fields
    configs = [
        CalcificationLesionConfig(
            arclength_frac=0.4, azimuth_deg=45.0, kind="hard", seed=10,
        ),
        CalcificationLesionConfig(
            arclength_frac=0.6, azimuth_deg=120.0, kind="soft_lipid", seed=11,
        ),
        CalcificationLesionConfig(
            arclength_frac=0.7, azimuth_deg=200.0, kind="fibrous", seed=12,
        ),
    ]
    out = build_all_lesions(configs, centerline, lumen, wall)
    assert [o.material_name for o in out] == [
        "calcified_plaque", "lipid_pool", "fibrous_plaque",
    ]
    for o in out:
        assert o.mesh.is_watertight
        assert o.mesh.volume > 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tube_points(centerline, radii, thetas):
    n_stations, n_angles = radii.shape
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    verts = np.empty((n_stations, n_angles, 3))
    for i in range(n_stations):
        pos = centerline.positions[i]
        nrm = centerline.normals[i]
        bin_ = centerline.binormals[i]
        verts[i] = pos[None, :] + (radii[i] * cos_t)[:, None] * nrm[None, :] + \
                   (radii[i] * sin_t)[:, None] * bin_[None, :]
    return verts.reshape(-1, 3)


def _tube_faces(n_stations, n_angles):
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
    proximal_centroid_idx = n_stations * n_angles
    distal_centroid_idx = proximal_centroid_idx + 1
    for j in range(n_angles):
        j_next = (j + 1) % n_angles
        faces.append([proximal_centroid_idx, j_next, j])
    for j in range(n_angles):
        j_next = (j + 1) % n_angles
        a = (n_stations - 1) * n_angles + j
        b = (n_stations - 1) * n_angles + j_next
        faces.append([distal_centroid_idx, a, b])
    return np.asarray(faces)


def _tube_points_with_caps(centerline, radii, thetas):
    verts = _tube_points(centerline, radii, thetas)
    n_stations, n_angles = radii.shape
    proximal_centroid = verts[:n_angles].mean(axis=0)
    distal_centroid = verts[(n_stations - 1) * n_angles : n_stations * n_angles].mean(axis=0)
    return np.vstack([verts, proximal_centroid[None, :], distal_centroid[None, :]])
