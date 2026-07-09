"""Tests for large-peripheral GenerationConfig scale."""

from __future__ import annotations

import numpy as np

from vesselgen.config import GenerationConfig


def test_generation_config_peripheral_radii():
    cfg = GenerationConfig()
    rng = np.random.default_rng(0)
    n_draws = 500
    radii_mm = []
    for i in range(n_draws):
        v = cfg.sample(rng, seed=i)
        radii_mm.append(v.parent.cross_section.mean_radius_mm)

    typical_radius_lo_mm, _ = cfg.parent_radius_mm_range
    aortic_radius_lo_mm, aortic_radius_hi_mm = cfg.aortic_radius_mm_range
    large_radius_lo_mm, large_radius_hi_mm = cfg.large_vessel_radius_mm_range

    aortic_radii_mm = [
        radius_mm
        for radius_mm in radii_mm
        if aortic_radius_lo_mm <= radius_mm <= aortic_radius_hi_mm
    ]
    large_radii_mm = [radius_mm for radius_mm in radii_mm if radius_mm >= large_radius_lo_mm]
    typical_radii_mm = [radius_mm for radius_mm in radii_mm if radius_mm < aortic_radius_lo_mm]

    assert typical_radii_mm
    assert aortic_radii_mm
    assert large_radii_mm
    assert min(typical_radii_mm) >= typical_radius_lo_mm
    assert min(aortic_radii_mm) >= aortic_radius_lo_mm
    assert max(aortic_radii_mm) <= aortic_radius_hi_mm
    assert min(large_radii_mm) >= large_radius_lo_mm
    assert max(large_radii_mm) <= large_radius_hi_mm

    aortic_rate = len(aortic_radii_mm) / n_draws
    large_rate = len(large_radii_mm) / n_draws
    rate_tolerance = 0.06
    assert abs(aortic_rate - cfg.aortic_scale_probability) <= rate_tolerance
    assert abs(large_rate - cfg.large_vessel_beyond_fov_probability) <= rate_tolerance
    assert np.median(typical_radii_mm) >= 4.5


def test_generation_config_wall_and_length():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    v = cfg.sample(rng, seed=1)
    assert v.parent.centerline.length_mm >= 45.0
    assert v.parent.wall.mean_thickness_mm >= 0.6
    assert v.parent.cross_section.mean_radius_mm * 2 >= 8.0
