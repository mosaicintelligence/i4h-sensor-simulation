"""Tests for small-vessel sampling and geometry validity."""

from __future__ import annotations

import numpy as np

from vesselgen.config import GenerationConfig
from vesselgen.sampling import ground_truth_is_valid_pose
from vesselgen.vessel import Vessel


def test_small_vessel_forced_draw_respects_radius_and_wall_ranges():
    cfg = GenerationConfig(
        small_vessel_probability=1.0,
        aortic_scale_probability=0.0,
        side_branch_probability=0.0,
        layered_wall_probability=0.0,
    )
    rng = np.random.default_rng(0)
    radii_mm: list[float] = []
    wall_thicknesses_mm: list[float] = []
    for i in range(200):
        v = cfg.sample(rng, seed=i)
        radii_mm.append(v.parent.cross_section.mean_radius_mm)
        wall_thicknesses_mm.append(v.parent.wall.mean_thickness_mm)

    radius_lo_mm, radius_hi_mm = cfg.small_vessel_radius_mm_range
    wall_lo_mm, wall_hi_mm = cfg.small_vessel_wall_thickness_mm_range
    assert min(radii_mm) >= radius_lo_mm
    assert max(radii_mm) <= radius_hi_mm
    assert min(wall_thicknesses_mm) >= wall_lo_mm
    assert max(wall_thicknesses_mm) <= wall_hi_mm


def test_small_vessel_empirical_rate_default_config():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    n_draws = 500
    small_radius_hi_mm = cfg.small_vessel_radius_mm_range[1]
    n_small = 0

    for i in range(n_draws):
        v = cfg.sample(rng, seed=i)
        if v.parent.cross_section.mean_radius_mm <= small_radius_hi_mm:
            n_small += 1

    small_rate = n_small / n_draws
    assert 0.05 <= small_rate <= 0.15


def test_small_vessel_geometry_valid_and_pose_inside_lumen():
    cfg = GenerationConfig(
        small_vessel_probability=1.0,
        aortic_scale_probability=0.0,
        side_branch_probability=0.0,
        layered_wall_probability=0.0,
        calcification_probability=0.0,
        guidewire_probability=0.0,
    )
    rng_cfg = np.random.default_rng(2)
    rng_pose = np.random.default_rng(3)

    for i in range(3):
        vessel_cfg = cfg.sample(rng_cfg, seed=i)
        vessel = Vessel.from_config(vessel_cfg)
        assert vessel.lumen_mesh.is_watertight
        assert vessel.outer_mesh.is_watertight

        pose = vessel.sample_pose(rng_pose, max_tilt_deg=10.0, edge_margin_mm=0.1)
        assert vessel.lumen_mesh.contains([pose.position])[0]

        gt = vessel.ground_truth_at(pose, n_angles=180)
        finite_lumen_distances_mm = gt.distance_to_lumen_wall_mm[
            np.isfinite(gt.distance_to_lumen_wall_mm)
        ]
        assert finite_lumen_distances_mm.size > 0
        assert (finite_lumen_distances_mm > 0).all()
        assert ground_truth_is_valid_pose(vessel, pose, gt)
