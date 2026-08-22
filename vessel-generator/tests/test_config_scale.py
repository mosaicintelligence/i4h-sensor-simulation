"""Tests for GenerationConfig scale sampling: the small / aortic / large /
adjacent / typical partition."""

from __future__ import annotations

import numpy as np
import pytest

from vesselgen.config import GenerationConfig, VesselConfig

BUCKETS = ("small", "aortic", "large", "adjacent", "typical")


def _bucket(cfg: GenerationConfig, v: VesselConfig) -> str:
    """Infer which scale bucket a sampled vessel came from. The radius ranges
    are disjoint except small vs adjacent, which the neighbor list resolves."""
    if v.adjacent_vessels:
        return "adjacent"
    r = v.parent.cross_section.mean_radius_mm
    if r <= cfg.small_vessel_radius_mm_range[1]:
        return "small"
    if r >= cfg.large_vessel_radius_mm_range[0]:
        return "large"
    if r >= cfg.aortic_radius_mm_range[0]:
        return "aortic"
    return "typical"


def test_generation_config_bucket_rates_match_defaults():
    cfg = GenerationConfig()
    rng = np.random.default_rng(0)
    n_draws = 500
    counts = {b: 0 for b in BUCKETS}
    for i in range(n_draws):
        counts[_bucket(cfg, cfg.sample(rng, seed=i))] += 1
    expected = {
        "small": cfg.small_vessel_probability,
        "aortic": cfg.aortic_scale_probability,
        "large": cfg.large_vessel_beyond_fov_probability,
        "adjacent": cfg.adjacent_vessel_probability,
    }
    expected["typical"] = 1.0 - sum(expected.values())
    for bucket, p in expected.items():
        assert abs(counts[bucket] / n_draws - p) <= 0.06, (bucket, counts)


def test_generation_config_wall_and_length():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    min_wall_mm = min(
        cfg.parent_wall_thickness_mm_range[0],
        cfg.aortic_wall_thickness_mm_range[0],
        cfg.small_vessel_wall_thickness_mm_range[0],
        cfg.large_vessel_wall_thickness_mm_range[0],
        cfg.adjacent_parent_wall_thickness_mm_range[0],
    )
    for i in range(100):
        v = cfg.sample(rng, seed=i)
        assert v.parent.centerline.length_mm >= 45.0
        assert v.parent.wall.mean_thickness_mm >= min_wall_mm
        assert v.parent.cross_section.mean_radius_mm >= cfg.small_vessel_radius_mm_range[0]


@pytest.mark.parametrize(
    "field_name, bucket",
    [
        ("small_vessel_probability", "small"),
        ("aortic_scale_probability", "aortic"),
        ("large_vessel_beyond_fov_probability", "large"),
        ("adjacent_vessel_probability", "adjacent"),
    ],
)
def test_forcing_one_bucket_to_one_wins_over_other_defaults(field_name, bucket):
    """``<bucket>_probability=1.0`` yields only that case even though the other
    buckets keep their non-zero defaults (single-case dataset generation)."""
    cfg = GenerationConfig(**{field_name: 1.0})
    rng = np.random.default_rng(2)
    for i in range(40):
        assert _bucket(cfg, cfg.sample(rng, seed=i)) == bucket


def test_invalid_scale_partitions_raise():
    with pytest.raises(ValueError):  # sum > 1 without a forced bucket
        GenerationConfig(
            small_vessel_probability=0.45,
            aortic_scale_probability=0.45,
            large_vessel_beyond_fov_probability=0.20,
        )
    with pytest.raises(ValueError):  # two forced buckets
        GenerationConfig(small_vessel_probability=1.0, large_vessel_beyond_fov_probability=1.0)
    with pytest.raises(ValueError):
        GenerationConfig(large_vessel_beyond_fov_probability=-0.1)
