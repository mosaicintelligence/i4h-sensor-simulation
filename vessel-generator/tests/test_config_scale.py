"""Tests for large-peripheral GenerationConfig scale."""

from __future__ import annotations

import numpy as np

from vesselgen.config import GenerationConfig


def test_generation_config_peripheral_radii():
    cfg = GenerationConfig()
    rng = np.random.default_rng(0)
    radii = []
    for i in range(200):
        v = cfg.sample(rng, seed=i)
        radii.append(v.parent.cross_section.mean_radius_mm)
    assert min(radii) >= 3.9
    assert np.median(radii) >= 4.5
    assert max(radii) >= 8.0


def test_generation_config_wall_and_length():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    v = cfg.sample(rng, seed=1)
    assert v.parent.centerline.length_mm >= 45.0
    assert v.parent.wall.mean_thickness_mm >= 0.6
    assert v.parent.cross_section.mean_radius_mm * 2 >= 8.0
