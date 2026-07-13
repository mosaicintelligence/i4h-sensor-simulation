"""Tests for GenerationConfig scale sampling."""

from __future__ import annotations

import numpy as np
import pytest

from vesselgen.config import GenerationConfig, clamp_default_aortic_scale_probability


def test_generation_config_peripheral_radii():
    cfg = GenerationConfig()
    rng = np.random.default_rng(0)
    radii = []
    adjacent_radii = []
    for i in range(200):
        v = cfg.sample(rng, seed=i)
        r = v.parent.cross_section.mean_radius_mm
        if v.adjacent_vessels:
            adjacent_radii.append(r)
        else:
            radii.append(r)
    # Non-adjacent draws keep the large-peripheral scale ladder.
    assert min(radii) >= 3.9
    assert np.median(radii) >= 4.5
    assert max(radii) >= 8.0
    # Adjacent-vessels draws use the dedicated adjacent primary radius range.
    lo, hi = cfg.adjacent_parent_radius_mm_range
    assert adjacent_radii, "expected some adjacent-vessels draws at the default rate"
    assert all(lo <= r <= hi for r in adjacent_radii)


def test_generation_config_wall_and_length():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    v = cfg.sample(rng, seed=1)
    while v.adjacent_vessels:  # scale assertions target the non-adjacent ladder
        v = cfg.sample(rng, seed=1)
    assert v.parent.centerline.length_mm >= 45.0
    assert v.parent.wall.mean_thickness_mm >= 0.6
    assert v.parent.cross_section.mean_radius_mm * 2 >= 8.0
