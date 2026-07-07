"""Tests for simulator domain-randomization sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VISIONS_DIR = REPO_ROOT / "instrument-calibration" / "p035_visions"
sys.path.insert(0, str(VISIONS_DIR))

from tier1_evaluation import load_calibrated_config, slider_to_db  # noqa: E402

from vesselgen.config import GenerationConfig  # noqa: E402
from vesselgen.sampling import sample_pose  # noqa: E402
from vesselgen.sim_randomization import (  # noqa: E402
    SimRandomizationConfig,
    apply_vessel_sim_draw,
    deep_tgc_gain_db,
    load_ringdown_waveforms,
    palette_template_to_envelope,
    saturation_fraction,
)
from vesselgen.vessel import Vessel  # noqa: E402


def _sample_vessel_with_side_branch(seed: int) -> Vessel | None:
    cfg_gen = GenerationConfig(side_branch_probability=1.0)
    rng = np.random.default_rng(seed)
    for attempt in range(20):
        cfg = cfg_gen.sample(rng, seed=seed + attempt, name=f"test_{seed}")
        try:
            vessel = Vessel.from_config(cfg)
        except Exception:
            continue
        if any(b.name != "parent" for b in vessel.branches):
            return vessel
    return None


def test_material_draw_presets_sum_to_one():
    cfg = SimRandomizationConfig()
    rng = np.random.default_rng(0)
    counts = {"fat_like": 0, "muscle_like": 0, "mixed": 0}
    for _ in range(300):
        draw = cfg.sample_vessel_materials(rng)
        counts[draw.extravascular_preset] += 1
    assert all(v > 0 for v in counts.values())


def test_frame_gain_within_slider_range():
    cfg, _, _ = load_calibrated_config()
    rand_cfg = SimRandomizationConfig()
    rng = np.random.default_rng(1)
    base_gain_db = float(cfg.processing.gain_db)
    draw = rand_cfg.sample_frame_sim(
        rng,
        base_gain_db=base_gain_db,
        frame_seed=7,
        noise_seed=8,
        slider_to_db=slider_to_db,
        base_tgc_deep_gain_db=deep_tgc_gain_db(cfg),
    )
    assert rand_cfg.gain_slider_range[0] <= draw.gain_slider <= rand_cfg.gain_slider_range[1]
    assert draw.artifact_reduction in ("on", "off")
    assert draw.ring_down_template in ("raw", "disabled")
    assert draw.ring_down_enabled == (draw.artifact_reduction == "on")
    assert (
        rand_cfg.tgc_deep_gain_scale_range[0]
        <= draw.tgc_deep_gain_scale
        <= rand_cfg.tgc_deep_gain_scale_range[1]
    )


def test_tier2_vessel_draw_ranges():
    cfg, _, sim_params = load_calibrated_config()
    rand_cfg = SimRandomizationConfig()
    rng = np.random.default_rng(2)
    draw = rand_cfg.sample_vessel_sim(
        rng,
        base_ring_down_amplitude=float(cfg.processing.ring_down.amplitude),
        base_t_far_mm=float(cfg.sim.t_far_mm),
        base_scattering_resolution_mm=float(cfg.processing.scattering_resolution_mm),
    )
    assert draw.t_far_mm in rand_cfg.t_far_mm_choices
    lo, hi = rand_cfg.scattering_resolution_mm_range
    assert lo <= draw.scattering_resolution_mm <= hi
    base = float(cfg.processing.ring_down.amplitude)
    assert abs(draw.ring_down_amplitude - base * draw.ring_down_amplitude_scale) < 1e-9
    apply_vessel_sim_draw(cfg, sim_params, draw)
    assert float(sim_params.t_far) == draw.t_far_mm
    assert abs(float(sim_params.scattering_resolution_mm) - draw.scattering_resolution_mm) < 1e-5
    assert abs(float(sim_params.ring_down.amplitude) - draw.ring_down_amplitude) < 1e-9


def test_ringdown_waveforms_load():
    cfg, _, _ = load_calibrated_config()
    rand_cfg = SimRandomizationConfig()
    ringdown = load_ringdown_waveforms(cfg, rand_cfg)
    assert ringdown.ar_off.ndim == 1
    assert ringdown.ar_on.ndim == 1
    assert len(ringdown.ar_off) > 0
    assert float(ringdown.raw.max()) > 0.5
    assert float(ringdown.suppressed.max()) < float(ringdown.raw.max())


def test_palette_to_envelope_is_non_negative():
    palette = np.array([10.0, 50.0, 120.0], dtype=np.float32)
    env = palette_template_to_envelope(palette, log_multiplier=20.0, speckle_floor_palette=28.5)
    assert env.min() >= 0.0
    assert env[0] == 0.0


def test_saturation_fraction():
    img = np.array([[100.0, 238.0], [239.0, np.nan]])
    frac = saturation_fraction(img, saturation_palette=239.0)
    assert abs(frac - (1.0 / 3.0)) < 1e-6


def test_side_branch_ostium_bias():
    vessel = _sample_vessel_with_side_branch(99)
    if vessel is None:
        return
    side_names = {b.name for b in vessel.branches if b.name != "parent"}
    if not side_names:
        return
    rng = np.random.default_rng(42)
    ostium_samples = []
    for _ in range(200):
        pose = sample_pose(
            vessel,
            rng,
            edge_margin_mm=0.0,
            side_branch_ostium_bias_prob=1.0,
            side_branch_ostium_arclength_frac=0.25,
        )
        if pose.branch_name in side_names:
            branch = next(b for b in vessel.branches if b.name == pose.branch_name)
            ostium_samples.append(pose.arclength_mm / branch.centerline.length_mm)
    assert ostium_samples
    assert max(ostium_samples) <= 0.25 + 1e-6
