"""Tests for large-vessel (beyond-FOV) sampling and A-line NaN behavior."""

from __future__ import annotations

import numpy as np

from vesselgen.config import GenerationConfig
from vesselgen.vessel import Vessel


def test_large_vessel_forced_draw_respects_radius_and_wall_ranges():
    cfg = GenerationConfig(
        large_vessel_beyond_fov_probability=1.0,
        small_vessel_probability=0.0,
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

    radius_lo_mm, radius_hi_mm = cfg.large_vessel_radius_mm_range
    wall_lo_mm, wall_hi_mm = cfg.large_vessel_wall_thickness_mm_range
    assert min(radii_mm) >= radius_lo_mm
    assert max(radii_mm) <= radius_hi_mm
    assert min(wall_thicknesses_mm) >= wall_lo_mm
    assert max(wall_thicknesses_mm) <= wall_hi_mm


def test_large_vessel_empirical_rate_default_config():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    n_draws = 500
    large_radius_lo_mm = cfg.large_vessel_radius_mm_range[0]
    n_large = 0

    for i in range(n_draws):
        v = cfg.sample(rng, seed=i)
        if v.parent.cross_section.mean_radius_mm >= large_radius_lo_mm:
            n_large += 1

    large_rate = n_large / n_draws
    rate_tolerance = 0.06
    assert abs(large_rate - cfg.large_vessel_beyond_fov_probability) <= rate_tolerance


def test_large_vessel_beyond_fov_produces_nan_alines():
    cfg = GenerationConfig(
        large_vessel_beyond_fov_probability=1.0,
        small_vessel_probability=0.0,
        aortic_scale_probability=0.0,
        side_branch_probability=0.0,
        layered_wall_probability=0.0,
        calcification_probability=0.0,
        guidewire_probability=0.0,
    )
    fov_radius_mm = 17.5
    rng_cfg = np.random.default_rng(2)
    rng_pose = np.random.default_rng(3)

    n_vessels = 20
    poses_per_vessel = 10
    n_poses = 0
    n_poses_with_nan_lumen = 0
    for i in range(n_vessels):
        vessel_cfg = cfg.sample(rng_cfg, seed=i)
        vessel = Vessel.from_config(vessel_cfg)
        assert vessel.lumen_mesh.is_watertight
        assert vessel.outer_mesh.is_watertight

        for _ in range(poses_per_vessel):
            pose = vessel.sample_pose(rng_pose, max_tilt_deg=10.0, edge_margin_mm=0.2)
            gt = vessel.ground_truth_at(pose, n_angles=180, max_distance_mm=fov_radius_mm)
            lumen_distances_mm = gt.distance_to_lumen_wall_mm
            outer_distances_mm = gt.distance_to_outer_wall_mm

            n_poses += 1
            nan_lumen = np.isnan(lumen_distances_mm)
            if nan_lumen.any():
                n_poses_with_nan_lumen += 1
                # An "open sector", not a fully out-of-FOV ring: some wall
                # must remain visible on the near side.
                assert np.isfinite(lumen_distances_mm).mean() > 0.2
            # Wherever the lumen wall is beyond FOV, the outer wall (always
            # farther along the same ray) is beyond FOV too.
            assert np.all(np.isnan(outer_distances_mm[nan_lumen]))

    beyond_fov_rate = n_poses_with_nan_lumen / n_poses
    assert beyond_fov_rate >= 0.8


def test_large_vessel_open_sector_rate_meets_threshold_by_fov():
    """Fraction of poses with a beyond-FOV sector (>=1 NaN lumen A-line) meets
    a per-FOV floor at the smaller FOVs, while on average most of the wall
    stays visible (guards the degenerate "whole ring beyond FOV" case).

    Thresholds have wide margin against sampling noise.
    """
    cfg = GenerationConfig(
        large_vessel_beyond_fov_probability=1.0,
        small_vessel_probability=0.0,
        aortic_scale_probability=0.0,
        side_branch_probability=0.0,
        layered_wall_probability=0.0,
        calcification_probability=0.0,
        guidewire_probability=0.0,
    )
    min_open_sector_rate_by_fov_mm = {17.5: 0.70, 20.0: 0.50}
    min_mean_finite_fraction = 0.40
    fov_radii_mm = sorted(min_open_sector_rate_by_fov_mm)

    rng_cfg = np.random.default_rng(4)
    rng_pose = np.random.default_rng(5)
    n_vessels = 12
    poses_per_vessel = 12

    n_poses = 0
    n_open_by_fov_mm = {fov_mm: 0 for fov_mm in fov_radii_mm}
    finite_fraction_sum_by_fov_mm = {fov_mm: 0.0 for fov_mm in fov_radii_mm}
    for i in range(n_vessels):
        vessel = Vessel.from_config(cfg.sample(rng_cfg, seed=i))
        for _ in range(poses_per_vessel):
            pose = vessel.sample_pose(rng_pose, max_tilt_deg=10.0, edge_margin_mm=0.2)
            n_poses += 1
            for fov_mm in fov_radii_mm:
                gt = vessel.ground_truth_at(pose, n_angles=180, max_distance_mm=fov_mm)
                lumen_distances_mm = gt.distance_to_lumen_wall_mm
                if np.isnan(lumen_distances_mm).any():
                    n_open_by_fov_mm[fov_mm] += 1
                finite_fraction_sum_by_fov_mm[fov_mm] += float(
                    np.isfinite(lumen_distances_mm).mean()
                )

    for fov_mm, min_rate in min_open_sector_rate_by_fov_mm.items():
        open_sector_rate = n_open_by_fov_mm[fov_mm] / n_poses
        assert open_sector_rate >= min_rate, (
            f"open-sector rate {open_sector_rate:.2f} at FOV {fov_mm} mm "
            f"is below the required minimum {min_rate:.2f}"
        )
        mean_finite_fraction = finite_fraction_sum_by_fov_mm[fov_mm] / n_poses
        assert mean_finite_fraction >= min_mean_finite_fraction, (
            f"mean visible-wall fraction {mean_finite_fraction:.2f} at FOV {fov_mm} mm "
            f"is below the required minimum {min_mean_finite_fraction:.2f} "
            "(vessels may be too large, blacking out the whole ring)"
        )
