"""Tests for GenerationConfig scale sampling."""

from __future__ import annotations

import numpy as np
import pytest

from vesselgen.config import GenerationConfig, clamp_default_aortic_scale_probability


def test_generation_config_peripheral_radii():
    cfg = GenerationConfig()
    rng = np.random.default_rng(0)
    n_draws = 500
    radii_mm: list[float] = []
    for i in range(n_draws):
        v = cfg.sample(rng, seed=i)
        radii_mm.append(v.parent.cross_section.mean_radius_mm)

    small_radius_lo_mm, small_radius_hi_mm = cfg.small_vessel_radius_mm_range
    typical_radius_lo_mm, _ = cfg.parent_radius_mm_range
    aortic_radius_lo_mm, aortic_radius_hi_mm = cfg.aortic_radius_mm_range
    large_radius_lo_mm, large_radius_hi_mm = cfg.large_vessel_radius_mm_range

    small_radii_mm = [radius_mm for radius_mm in radii_mm if radius_mm <= small_radius_hi_mm]
    aortic_radii_mm = [
        radius_mm
        for radius_mm in radii_mm
        if aortic_radius_lo_mm <= radius_mm <= aortic_radius_hi_mm
    ]
    large_radii_mm = [radius_mm for radius_mm in radii_mm if radius_mm >= large_radius_lo_mm]
    typical_radii_mm = [
        radius_mm for radius_mm in radii_mm if small_radius_hi_mm < radius_mm < aortic_radius_lo_mm
    ]

    assert small_radii_mm
    assert typical_radii_mm
    assert aortic_radii_mm
    assert large_radii_mm
    assert min(small_radii_mm) >= small_radius_lo_mm
    assert max(small_radii_mm) <= small_radius_hi_mm
    assert min(typical_radii_mm) >= typical_radius_lo_mm
    assert min(aortic_radii_mm) >= aortic_radius_lo_mm
    assert max(aortic_radii_mm) <= aortic_radius_hi_mm
    assert min(large_radii_mm) >= large_radius_lo_mm
    assert max(large_radii_mm) <= large_radius_hi_mm

    small_rate = len(small_radii_mm) / n_draws
    aortic_rate = len(aortic_radii_mm) / n_draws
    large_rate = len(large_radii_mm) / n_draws
    rate_tolerance = 0.06
    assert abs(small_rate - cfg.small_vessel_probability) <= rate_tolerance
    assert abs(aortic_rate - cfg.aortic_scale_probability) <= rate_tolerance
    assert abs(large_rate - cfg.large_vessel_beyond_fov_probability) <= rate_tolerance
    assert np.median(typical_radii_mm) >= 4.5


def test_generation_config_wall_and_length():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    min_expected_wall_thickness_mm = min(
        cfg.parent_wall_thickness_mm_range[0],
        cfg.aortic_wall_thickness_mm_range[0],
        cfg.small_vessel_wall_thickness_mm_range[0],
        cfg.large_vessel_wall_thickness_mm_range[0],
    )
    for i in range(100):
        v = cfg.sample(rng, seed=i)
        assert v.parent.centerline.length_mm >= 45.0
        assert v.parent.wall.mean_thickness_mm >= min_expected_wall_thickness_mm
        assert v.parent.cross_section.mean_radius_mm >= 1.8


def test_clamp_default_aortic_scale_probability_keeps_default_when_valid():
    assert clamp_default_aortic_scale_probability(0.6, 0.18) == 0.18


def test_clamp_default_aortic_scale_probability_caps_to_remaining_share():
    assert np.isclose(clamp_default_aortic_scale_probability(0.9, 0.18), 0.1)
    assert np.isclose(clamp_default_aortic_scale_probability(1.0, 0.18), 0.0)


def test_force_all_small_vessel_probability_with_clamped_default_aortic():
    aortic_p = clamp_default_aortic_scale_probability(1.0, 0.18)
    cfg = GenerationConfig(small_vessel_probability=1.0, aortic_scale_probability=aortic_p)
    assert cfg.small_vessel_probability == 1.0
    assert cfg.aortic_scale_probability == 0.0


def test_force_large_scale_probability_overrides_default_small_share():
    cfg = GenerationConfig(large_vessel_beyond_fov_probability=1.0, aortic_scale_probability=0.0)
    rng = np.random.default_rng(11)

    radius_lo_mm, radius_hi_mm = cfg.large_vessel_radius_mm_range
    for i in range(200):
        radius_mm = cfg.sample(rng, seed=i).parent.cross_section.mean_radius_mm
        assert radius_lo_mm <= radius_mm <= radius_hi_mm


def test_force_aortic_scale_probability_overrides_other_default_shares():
    cfg = GenerationConfig(aortic_scale_probability=1.0)
    rng = np.random.default_rng(12)

    radius_lo_mm, radius_hi_mm = cfg.aortic_radius_mm_range
    for i in range(200):
        radius_mm = cfg.sample(rng, seed=i).parent.cross_section.mean_radius_mm
        assert radius_lo_mm <= radius_mm <= radius_hi_mm


def test_invalid_three_way_scale_partition_raises():
    with pytest.raises(ValueError):
        GenerationConfig(
            small_vessel_probability=0.45,
            aortic_scale_probability=0.45,
            large_vessel_beyond_fov_probability=0.20,
        )
