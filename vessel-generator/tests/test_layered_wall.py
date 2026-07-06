"""Tests for :class:`LayeredWallConfig` -> mesh pipeline."""

from __future__ import annotations

import numpy as np

from vesselgen.centerline import Centerline
from vesselgen.config import (
    CenterlineConfig,
    CrossSectionConfig,
    LayeredWallConfig,
    WallConfig,
)
from vesselgen.cross_section import build_cross_sections
from vesselgen.sweep import sweep_layered_branch
from vesselgen.wall import build_layered_wall, layered_wall_from_wall_field


def _make_lumen(rng: np.random.Generator):
    centerline_cfg = CenterlineConfig(length_mm=40.0, n_stations=32)
    xs_cfg = CrossSectionConfig(
        mean_radius_mm=4.5,
        distal_radius_mm=4.2,
        n_angles=72,
    )
    centerline = Centerline(centerline_cfg)
    lumen = build_cross_sections(xs_cfg, centerline_cfg.length_mm, centerline_cfg.n_stations, rng)
    return centerline, lumen


def test_layered_wall_three_layers_nested_no_crossings():
    rng = np.random.default_rng(0)
    centerline, lumen = _make_lumen(rng)
    cfg = LayeredWallConfig.trilaminar(
        total_thickness_mm=0.9,
        intima_frac=0.18,
        media_frac=0.55,
        adventitia_frac=0.27,
    )
    field = build_layered_wall(cfg, lumen, length_mm=40.0, rng=rng)

    assert field.n_layers == 3
    assert len(field.interface_radii) == 4
    for k in range(len(field.interface_radii) - 1):
        delta = field.interface_radii[k + 1] - field.interface_radii[k]
        assert np.all(delta > 0.0), f"interfaces {k} and {k+1} crossed"
    np.testing.assert_allclose(field.interface_radii[0], lumen.radii)


def test_layered_wall_two_layers_nested():
    rng = np.random.default_rng(1)
    centerline, lumen = _make_lumen(rng)
    cfg = LayeredWallConfig.media_adventitia(
        total_thickness_mm=0.85, media_frac=0.65, adventitia_frac=0.35
    )
    field = build_layered_wall(cfg, lumen, length_mm=40.0, rng=rng)

    assert field.n_layers == 2
    assert len(field.interface_radii) == 3
    for k in range(2):
        delta = field.interface_radii[k + 1] - field.interface_radii[k]
        assert np.all(delta > 0.0)


def test_layered_wall_single_layer_matches_legacy():
    rng = np.random.default_rng(2)
    centerline, lumen = _make_lumen(rng)
    cfg = LayeredWallConfig.single_layer(
        material_name="vessel_wall",
        total_thickness_mm=0.7,
        max_perturbation_frac=0.5,
    )
    field = build_layered_wall(cfg, lumen, length_mm=40.0, rng=rng)
    assert field.n_layers == 1
    assert len(field.interface_radii) == 2
    delta = field.interface_radii[1] - field.interface_radii[0]
    assert np.all(delta > 0.0)


def test_sweep_layered_branch_produces_closed_meshes():
    rng = np.random.default_rng(3)
    centerline, lumen = _make_lumen(rng)
    cfg = LayeredWallConfig.trilaminar(total_thickness_mm=0.9)
    field = build_layered_wall(cfg, lumen, length_mm=40.0, rng=rng)
    meshes = sweep_layered_branch(centerline, lumen, field)
    assert len(meshes) == 4
    for mesh in meshes:
        assert mesh.is_watertight, "layered surface mesh should be watertight"

    # Volumes monotonically increase from lumen to outer.
    volumes = [mesh.volume for mesh in meshes]
    assert all(volumes[k] < volumes[k + 1] for k in range(len(volumes) - 1))


def test_layered_wall_from_wall_field_roundtrip():
    rng = np.random.default_rng(4)
    centerline, lumen = _make_lumen(rng)
    from vesselgen.wall import build_wall

    wall = build_wall(
        WallConfig(mean_thickness_mm=0.8, max_perturbation_frac=0.5),
        lumen,
        length_mm=40.0,
        rng=rng,
    )
    wrapped = layered_wall_from_wall_field(wall, lumen, material_name="vessel_wall")
    assert wrapped.n_layers == 1
    assert wrapped.layer_specs[0].material_name == "vessel_wall"
    np.testing.assert_allclose(wrapped.interface_radii[0], lumen.radii)
    np.testing.assert_allclose(
        wrapped.interface_radii[1] - wrapped.interface_radii[0],
        wall.thicknesses,
    )
