#!/usr/bin/env python3
"""Analytical re-derivation of ``processing.gain_db`` for the PV .035 calibration.

The calibration sheet treats the bench's envelope amplitude *at the reference
gain* (slider 54 for the PV .035) as the absolute reference scale (in
arbitrary linear units). The simulator's renderer produces envelope amplitudes
in its own arbitrary linear units; the ``processing.gain_db`` knob is the
calibrated scalar that converts between the two:

.. math::

    \\text{amp}_\\text{bench}(g_\\text{ref}) = \\text{amp}_\\text{sim}
        \\cdot 10^{(g_\\text{db} / 20)}

This script re-derives ``g_db`` from first principles by:

1. Rendering the bench wire phantom (5 wires at r ∈ {5, 10, 15, 20, 25} mm
   in a water bath, 30 frames) with the simulator in **raw-envelope mode**:
   ``log_floor = 1.0`` (calibration anchor; with K2v2 this means
   ``pixel = log10(amp/1.0) = log10(amp)``), ``log_multiplier = 1.0``,
   ``ring_down.enabled = false``,
   ``reject_palette = saturation_palette = 0`` (display window off),
   ``median_clip_filter = false``, ``gain_db = 0``. The post-log palette
   therefore equals ``log10(amp)`` and the per-wire amp is recovered as
   ``10**palette``. The K2v2 kernel's epsilon clamp keeps the math finite
   for amp == 0 (-> pixel ≈ -30) but never affects realistic amplitudes.

2. Reading the bench's per-wire peak palettes from
   ``derived/psf/psf_fit.json`` (the calibrated PSF table) and converting
   each to envelope amplitude via the calibration sheet's mapping
   ``amp = 10^(palette / log_multiplier)``. We exclude wires at or near
   palette saturation (≥ 220) because their underlying amplitude is no
   longer recoverable.

3. Computing the wire-by-wire required gain:

       gain_db_per_wire = 20 · log10(amp_bench / amp_sim)

   and reporting the median (used as the calibration value), the IQR (used
   as the uncertainty), and per-wire diagnostics.

The output JSON sits at ``derived/gain_lut/gain_db_derivation.json`` so it
is colocated with the existing ``gain_lut.json`` and can be cited from
``volcano_s5i.yaml``.

Usage::

    python instrument-calibration/p035_visions/derive_gain_db.py \\
        --n-frames 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths + simulator import
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
SIM_ROOT = WORKSPACE_ROOT / "i4h-sensor-simulation" / "ultrasound-raytracing"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import raysim as rs  # noqa: E402
from raysim import IvusSimConfig, RaytracingUltrasoundSimulator  # noqa: E402
from raysim.ray_sim_python import Sphere  # noqa: E402

YAML_PATH = HERE / "volcano_s5i.yaml"
BENCH_ROOT = WORKSPACE_ROOT / "P_035_PointScatter" / "derived"
PSF_FIT_PATH = BENCH_ROOT / "psf" / "psf_fit.json"
GAIN_LUT_PATH = BENCH_ROOT / "gain_lut" / "gain_lut.json"
OUTPUT_PATH = BENCH_ROOT / "gain_lut" / "gain_db_derivation.json"

WIRE_RADII_MM = (5.0, 10.0, 15.0, 20.0, 25.0)
WIRE_DIAMETER_MM = 0.127
WIRE_SPHERE_RADIUS_MM = WIRE_DIAMETER_MM / 2.0

# Bench rejection criteria for "saturated" wires whose peak palette no longer
# carries amplitude information.
BENCH_PEAK_SATURATION_PALETTE = 220.0


# ---------------------------------------------------------------------------
# Phantom + render
# ---------------------------------------------------------------------------
def build_wire_world(materials) -> tuple[Any, list[tuple[float, float]]]:
    """Same wire phantom as ``tier1_evaluation.py`` so radii line up exactly."""
    world = rs.World("lumen")
    wire_mat = materials.get_index("bone")
    positions: list[tuple[float, float]] = []
    for i, r in enumerate(WIRE_RADII_MM):
        theta = i * 2.0 * math.pi / len(WIRE_RADII_MM)
        x = r * math.sin(theta)
        z = r * math.cos(theta)
        world.add(Sphere(np.array([x, 0.0, z], dtype=np.float32),
                         WIRE_SPHERE_RADIUS_MM, wire_mat))
        positions.append((r, math.degrees(theta) % 360.0))
    return world, positions


def b_mode_to_theta_r(frame: np.ndarray, cfg: IvusSimConfig) -> np.ndarray:
    """Mirror of tier1_evaluation.b_mode_to_theta_r (kept here for self-contained run)."""
    n_theta = int(cfg.sim.b_mode_size[0])
    n_r = int(cfg.sim.b_mode_size[1])
    if frame.shape == (n_theta, n_r):
        return frame
    if frame.shape == (n_r, n_theta):
        return frame.T
    raise RuntimeError(
        f"Unexpected b_mode shape {frame.shape}; expected (theta, r)="
        f"({n_theta}, {n_r}) or its transpose."
    )


def measure_wire_amp(theta_r_frames: np.ndarray, *, t_far_mm: float,
                     positions: list[tuple[float, float]]) -> dict[float, float]:
    """For each wire, return the median peak post-log palette across frames.

    The simulator was run with ``log_multiplier = 1.0`` and ``log_floor = 1.0``
    (K2v2 calibration anchor)
    so the post-log palette equals ``log10(envelope_amp)`` (i.e. amp =
    10**palette). We measure the peak in a small window around the expected
    (theta, r) of each wire and take the median across frames.
    """
    n_frames, n_theta, n_r = theta_r_frames.shape
    dr = t_far_mm / n_r
    out: dict[float, float] = {}
    for r_mm, theta_deg in positions:
        th_idx = int(math.radians(theta_deg) / (2.0 * math.pi) * n_theta) % n_theta
        r_idx = int(r_mm / dr)
        th_lo, th_hi = max(0, th_idx - 30), min(n_theta, th_idx + 30)
        r_lo, r_hi = max(0, r_idx - 10), min(n_r, r_idx + 10)
        peaks = []
        for f in range(n_frames):
            patch = theta_r_frames[f, th_lo:th_hi, r_lo:r_hi]
            peaks.append(float(patch.max()))
        out[r_mm] = float(np.median(peaks))
    return out


def measure_water_bg_amp(theta_r_frames: np.ndarray, *, t_far_mm: float,
                         positions: list[tuple[float, float]],
                         r_mm_min: float = 6.0,
                         r_mm_max: float = 25.0) -> dict[str, float]:
    """Measure the water-scatter background palette in regions that exclude the wires.

    Returns a dict with ``mean_palette`` (post-log) and the corresponding
    ``mean_amp = 10**mean_palette`` (assuming log_mult = 1, log_floor = 1.0).
    We sample the radial band ``[r_mm_min, r_mm_max]`` (which contains the
    wire echoes) and exclude small windows around each wire so we are looking
    at pure water scatter.
    """
    n_frames, n_theta, n_r = theta_r_frames.shape
    dr = t_far_mm / n_r
    r_lo = max(0, int(r_mm_min / dr))
    r_hi = min(n_r, int(r_mm_max / dr))
    mask = np.ones((n_theta, r_hi - r_lo), dtype=bool)
    for r_mm, theta_deg in positions:
        th_idx = int(math.radians(theta_deg) / (2.0 * math.pi) * n_theta) % n_theta
        r_idx = int(r_mm / dr)
        th_lo, th_hi = max(0, th_idx - 40), min(n_theta, th_idx + 40)
        r_w_lo = max(r_lo, r_idx - 15) - r_lo
        r_w_hi = min(r_hi, r_idx + 15) - r_lo
        if r_w_hi > r_w_lo:
            mask[th_lo:th_hi, r_w_lo:r_w_hi] = False
    band = theta_r_frames[:, :, r_lo:r_hi]
    masked_values = band[:, mask]  # shape (n_frames, n_clean_pixels)
    mean_palette = float(masked_values.mean())
    median_palette = float(np.median(masked_values))
    return {
        "mean_palette_log10amp": mean_palette,
        "median_palette_log10amp": median_palette,
        "mean_envelope_amp": float(10.0 ** mean_palette),
        "median_envelope_amp": float(10.0 ** median_palette),
        "n_pixels_per_frame": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Bench data
# ---------------------------------------------------------------------------
def load_bench_wire_amps(*, log_multiplier: float,
                         saturation_palette: float = BENCH_PEAK_SATURATION_PALETTE) -> list[dict]:
    """Read bench per-wire peak palettes and convert to envelope amplitudes.

    The calibration sheet defines ``pixel = log_multiplier * log10(amp / log_floor)``
    with ``log_floor = 1.0``; equivalently ``amp = 10^(palette / log_multiplier)``.
    Wires at or above ``saturation_palette`` are dropped because their
    underlying amplitude is no longer recoverable from the palette value.
    """
    if not PSF_FIT_PATH.exists():
        raise FileNotFoundError(f"missing bench PSF fit: {PSF_FIT_PATH}")
    psf = json.load(open(PSF_FIT_PATH))
    rows = []
    for r in psf.get("axial_fwhm_mm_per_wire", []):
        peak = float(r.get("peak_palette", 0.0))
        if peak >= saturation_palette:
            continue
        amp = 10.0 ** (peak / log_multiplier)
        rows.append({
            "wire": int(r.get("wire", 0)),
            "r_mm": float(r.get("r_mm", float("nan"))),
            "peak_palette": peak,
            "envelope_amp": amp,
        })
    return rows


# ---------------------------------------------------------------------------
# Iterative refinement against the calibrated render
# ---------------------------------------------------------------------------
def _measure_calibrated_bg_palette(*, cfg, materials, base_params,
                                   gain_db: float, n_frames: int) -> float:
    """Render an anechoic lumen with the FULL calibrated pipeline at this
    gain_db and return the mean palette in the clean radial band.

    "Clean" = past the ring-down extent. The reject / saturation clamp is on,
    so this is the post-clamp mean palette an operator would see in the
    rendered frame.
    """
    params = cfg.to_sim_params()
    params.gain_db = float(gain_db)
    # OptiX needs >=1 primitive; place a tiny dummy sphere far outside the FOV.
    world = rs.World("lumen")
    world.add(Sphere(np.array([0.0, 1000.0, 0.0], dtype=np.float32), 0.001,
                     materials.get_index("lumen")))
    sim = RaytracingUltrasoundSimulator(world, materials)
    probe = cfg.to_probe()
    rd_extent_mm = float(cfg.processing.ring_down.extent_mm)
    r_lo_mm = max(rd_extent_mm + 1.0, 5.0)
    r_hi_mm = min(0.9 * float(cfg.sim.t_far_mm), 25.0)
    band_means = []
    for _ in range(n_frames):
        out = sim.simulate(probe, params)
        f = b_mode_to_theta_r(np.array(out, copy=True), cfg)
        n_r = f.shape[1]
        dr = float(cfg.sim.t_far_mm) / n_r
        r_lo = max(0, int(r_lo_mm / dr))
        r_hi = min(n_r, int(r_hi_mm / dr))
        band_means.append(float(f[:, r_lo:r_hi].mean()))
    return float(np.mean(band_means))


def _refine_gain_db_against_bg(*, cfg, materials, base_params,
                               target_palette: float, seed_gain_db: float,
                               n_frames: int = 4,
                               max_iter: int = 25,
                               tol_palette: float = 0.5) -> float:
    """Bisection on gain_db so the post-clamp bg palette matches target_palette.

    The post-clamp mean palette is monotonic-increasing in gain_db (within the
    relevant range), so a bracketing search converges quickly. We seed the
    bracket from the pure-amplitude derivation (which over-shoots) and walk it
    down until the rendered bg matches the bench reference within
    tol_palette.
    """
    log_mult = float(base_params.log_multiplier)
    # Bracket: high seed (over-shoots target due to clamp lift) + low seed
    # (no clamp activity, pure amplitude regime where the formula is exact).
    g_hi = float(seed_gain_db) + 0.0
    g_lo = float(seed_gain_db) - 30.0  # 30 dB below pure-amp guess
    p_hi = _measure_calibrated_bg_palette(cfg=cfg, materials=materials,
                                          base_params=base_params,
                                          gain_db=g_hi, n_frames=n_frames)
    p_lo = _measure_calibrated_bg_palette(cfg=cfg, materials=materials,
                                          base_params=base_params,
                                          gain_db=g_lo, n_frames=n_frames)
    print(f"[derive_gain_db] bracket: gain_db = {g_lo:+.2f} -> palette "
          f"{p_lo:.1f}; gain_db = {g_hi:+.2f} -> palette {p_hi:.1f}; "
          f"target = {target_palette:.1f}")
    # Expand bracket if needed.
    expand = 0
    while p_hi < target_palette and expand < 4:
        g_hi += 5.0
        p_hi = _measure_calibrated_bg_palette(cfg=cfg, materials=materials,
                                              base_params=base_params,
                                              gain_db=g_hi, n_frames=n_frames)
        expand += 1
    while p_lo > target_palette and expand < 8:
        g_lo -= 5.0
        p_lo = _measure_calibrated_bg_palette(cfg=cfg, materials=materials,
                                              base_params=base_params,
                                              gain_db=g_lo, n_frames=n_frames)
        expand += 1
    if not (p_lo <= target_palette <= p_hi):
        print(f"[derive_gain_db] WARNING: could not bracket target "
              f"({p_lo:.1f}, {p_hi:.1f}); falling back to pure-amplitude derivation")
        return float(seed_gain_db)
    for it in range(max_iter):
        g_mid = 0.5 * (g_lo + g_hi)
        p_mid = _measure_calibrated_bg_palette(cfg=cfg, materials=materials,
                                               base_params=base_params,
                                               gain_db=g_mid, n_frames=n_frames)
        print(f"[derive_gain_db]   it={it} gain_db={g_mid:+.3f} -> palette {p_mid:.2f}")
        if abs(p_mid - target_palette) <= tol_palette:
            return float(g_mid)
        if p_mid > target_palette:
            g_hi, p_hi = g_mid, p_mid
        else:
            g_lo, p_lo = g_mid, p_mid
    return float(g_mid)


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------
def derive_gain_db(n_frames: int = 30) -> dict[str, Any]:
    cfg = IvusSimConfig.from_yaml(YAML_PATH)

    # Diagnostic SimParams: post-log palette = log10(amp).
    base_params = cfg.to_sim_params()
    materials = rs.Materials()
    diag_params = cfg.to_sim_params()
    # K2v2: pixel = log_multiplier * log10(amp / log_floor). To recover
    # log10(amp) directly we set log_multiplier = 1 and log_floor = 1.0
    # (so the kernel computes log10(amp / 1) = log10(amp)). The kernel's
    # tiny epsilon clamp keeps the math finite for amp == 0 (-> pixel ≈
    # -30) but doesn't affect realistic amplitudes.
    diag_params.log_floor = 1.0
    diag_params.log_multiplier = 1.0
    diag_params.ring_down.enabled = False
    diag_params.reject_palette = 0.0
    diag_params.saturation_palette = 0.0
    diag_params.median_clip_filter = False
    diag_params.gain_db = 0.0  # we are deriving this from a zero-gain render
    # Pass 6: zero out the additive RF noise during the diagnostic measurement.
    # If left at the YAML value (sigma ~2.27 RF amp units = envelope mean ~2.84),
    # the noise envelope dominates the bg in the diagnostic-mode render and the
    # 60x20-pixel "wire peak" max-search picks up noise spikes instead of the
    # wire echo, biasing sim_envelope_amp_median_wire HIGH (we measured 7.57
    # this way; the underlying wire-only amplitude is much smaller). gain_db
    # then under-shoots and outer wires fall below the reject palette in the
    # final render. Forcing noise off here gives a clean wire-only amp.
    diag_params.noise_sigma = 0.0

    world, positions = build_wire_world(materials)
    probe = cfg.to_probe()
    sim = RaytracingUltrasoundSimulator(world, materials)

    print(f"[derive_gain_db] rendering {n_frames} wire-phantom frames "
          "(log_floor=1.0, log_mult=1.0, ring-down OFF, no display window)...")
    frames = []
    for _ in range(n_frames):
        out = sim.simulate(probe, diag_params)
        frames.append(np.array(out, copy=True))
    theta_r = np.stack([b_mode_to_theta_r(f, cfg) for f in frames])

    sim_palette_per_wire = measure_wire_amp(theta_r, t_far_mm=float(cfg.sim.t_far_mm),
                                            positions=positions)
    # Convert sim palette -> envelope amp (palette = log10(amp) since log_mult=1).
    sim_amp_per_wire = {r: float(10.0 ** p) for r, p in sim_palette_per_wire.items()}
    sim_amp_median = float(np.median(list(sim_amp_per_wire.values())))

    # Sim water-scatter background (excludes wires).
    sim_bg = measure_water_bg_amp(theta_r, t_far_mm=float(cfg.sim.t_far_mm),
                                  positions=positions)
    sim_bg_amp = sim_bg["mean_envelope_amp"]

    # Bench amps (calibrated, slider 54).
    log_multiplier = float(base_params.log_multiplier)
    bench_rows = load_bench_wire_amps(log_multiplier=log_multiplier)
    if not bench_rows:
        raise RuntimeError(
            "No unsaturated bench wires available for derivation "
            f"(saturation cutoff = {BENCH_PEAK_SATURATION_PALETTE} palette)."
        )
    bench_amp_median = float(np.median([r["envelope_amp"] for r in bench_rows]))

    # Bench water-scatter background at slider 54: from gain_lut.json's noise
    # row (E5 anechoic-ROI palette histograms). Mean palette = 46.2 at slider
    # 54 (clean log-Rayleigh shape, no reject clipping).
    BENCH_BG_PALETTE_AT_54 = 46.2
    bench_bg_amp = 10.0 ** (BENCH_BG_PALETTE_AT_54 / log_multiplier)

    # Two candidate calibrations of gain_db against the bench reference:
    #   * wire-peak target — match the median unsaturated bench wire
    #     amplitude. Saturates the bench's water bg if the simulator's
    #     wire/bg amplitude contrast is smaller than the bench's.
    #   * background target — match the bench's water-scatter floor in the
    #     final calibrated render. This is the metric the operator sees,
    #     so we calibrate against the **post-pipeline mean palette** (i.e.
    #     including the reject_palette / saturation_palette clamp) rather
    #     than the underlying geometric-mean envelope amplitude. Doing a
    #     pure-amplitude derivation (`20*log10(bench_bg_amp / sim_bg_amp)`)
    #     systematically over-shoots when the simulator's bg distribution
    #     is wider than the bench's — the pre-clamp mean lands at 46 but
    #     the clamp lifts the post-clamp mean by ~33 palette because
    #     Rayleigh's long left tail in log space crashes through the floor.
    gain_db_wire = 20.0 * math.log10(bench_amp_median / sim_amp_median)
    gain_db_bg_amp_pure = 20.0 * math.log10(bench_bg_amp / sim_bg_amp)

    # Pre-Pass-6 we bisected gain_db so the post-clamp anechoic mean palette
    # matched the bench. That worked when scatter was the only thing
    # contributing to the bg (noise.sigma = 0). Pass 6 re-anchors the bg
    # floor on additive Gaussian RF noise (calibrated separately by
    # derive_noise_sigma.py) -- so the bg palette is now noise-set and is
    # essentially independent of gain_db. We therefore pivot the gain_db
    # calibration onto the **wire-peak** target, which still scales with
    # gain_db. Keep the bg-bracket bisection running as a diagnostic so we
    # report the residual scatter contribution under the calibrated gain.
    try:
        gain_db_bg = _refine_gain_db_against_bg(
            cfg=cfg, materials=materials, base_params=base_params,
            target_palette=BENCH_BG_PALETTE_AT_54,
            seed_gain_db=gain_db_bg_amp_pure,
            n_frames=max(4, n_frames // 4),
        )
    except Exception as e:
        print(f"[derive_gain_db] bg bisection failed (expected with noise on): {e}")
        gain_db_bg = float("nan")

    # Contrast diagnostic.
    sim_contrast_db = 20.0 * math.log10(sim_amp_median / sim_bg_amp)
    bench_contrast_db = 20.0 * math.log10(bench_amp_median / bench_bg_amp)
    contrast_gap_db = bench_contrast_db - sim_contrast_db  # positive => sim under-contrasted

    # Pass 6 calibration: use the wire-peak target. With additive RF noise
    # anchoring the anechoic floor at the bench level, the wire-peak target
    # is the right anchor for gain_db (the bg is already correct by
    # construction). The contrast gap (bench_contrast_db - sim_contrast_db)
    # is reported below as a diagnostic of residual scattering-strength
    # mismatch the gain_db scalar cannot fix.
    gain_db_calibrated = gain_db_wire

    # Per-wire breakdown (sim wires only — the bench has more wires than the sim,
    # and bench wires are at different angles, so we report the renderer's gain
    # against each bench wire individually for transparency).
    wire_table = []
    for r_mm, sim_p in sim_palette_per_wire.items():
        sim_a = sim_amp_per_wire[r_mm]
        wire_table.append({
            "r_mm": r_mm,
            "sim_palette_log10amp": sim_p,
            "sim_envelope_amp": sim_a,
            "log10_amp_diff_to_bench_median": math.log10(bench_amp_median) - sim_p,
            "gain_db_required_to_match_bench_median": 20.0 * math.log10(bench_amp_median / sim_a),
        })

    # Renderer-vs-bench distribution (gain delta per bench wire if we use the
    # sim median wire amplitude). This is informational; the calibration value
    # is the median.
    bench_gain_db_distribution = [
        20.0 * math.log10(r["envelope_amp"] / sim_amp_median) for r in bench_rows
    ]

    out = {
        "method_summary": (
            "Analytical re-derivation of processing.gain_db for the PV .035 / "
            "Volcano s5i. Renders the wire phantom in raw-envelope mode "
            "(log_multiplier=1, log_floor=1.0, ring-down OFF, display window OFF, "
            "median clip OFF, gain_db=0) so post-log palette = log10(envelope amp). "
            "Two candidate calibrations are reported: matching the median bench "
            "wire amplitude (gain_db_wire) and matching the bench water-scatter "
            "background (gain_db_bg). Pass 6 (additive noise floor) shifted the "
            "anechoic floor anchor onto noise.sigma (calibrated independently "
            "via derive_noise_sigma.py), so the YAML now uses the **wire-peak** "
            "calibration. Any residual wire-vs-bg contrast gap is a scattering-"
            "strength problem the gain_db scalar cannot fix."
        ),
        "n_frames": n_frames,
        "log_multiplier_used": log_multiplier,
        "bench_saturation_cutoff_palette": BENCH_PEAK_SATURATION_PALETTE,
        "bench_unsaturated_wire_count": len(bench_rows),
        "sim_envelope_amp_median_wire": sim_amp_median,
        "sim_envelope_amp_water_bg_mean": sim_bg_amp,
        "bench_envelope_amp_median_wire": bench_amp_median,
        "bench_envelope_amp_water_bg_at_slider_54": bench_bg_amp,
        "bench_water_bg_palette_at_slider_54": BENCH_BG_PALETTE_AT_54,
        "gain_db_wire_target": gain_db_wire,
        "gain_db_bg_target": gain_db_bg,
        "gain_db_bg_amp_pure": gain_db_bg_amp_pure,
        "gain_db_calibrated": gain_db_calibrated,
        "calibration_target": "wire_peak_palette_median",
        "wire_vs_bg_contrast_db": {
            "sim": sim_contrast_db,
            "bench": bench_contrast_db,
            "gap_dB_bench_minus_sim": contrast_gap_db,
            "interpretation": (
                "Positive gap means the bench has more wire-vs-background "
                "contrast than the simulator. With the background-calibrated "
                "gain_db, sim wires will undershoot the bench wire palette by "
                f"~{contrast_gap_db:.1f} dB ({contrast_gap_db * log_multiplier / 20:.1f} "
                "palette). Closing this gap requires increasing the wire's "
                "scattering strength (e.g. switching the sphere material to "
                "one with higher impedance contrast) or increasing the wire "
                "size — the gain_db scalar cannot."
            ),
        },
        "gain_db_distribution_dB": {
            "median": float(np.median(bench_gain_db_distribution)),
            "p25": float(np.percentile(bench_gain_db_distribution, 25)),
            "p75": float(np.percentile(bench_gain_db_distribution, 75)),
            "p10": float(np.percentile(bench_gain_db_distribution, 10)),
            "p90": float(np.percentile(bench_gain_db_distribution, 90)),
            "n_bench_wires": len(bench_rows),
        },
        "per_wire": {
            "sim": wire_table,
            "bench_unsaturated": bench_rows,
        },
        "sim_water_bg_diagnostic": sim_bg,
        "calibration_recipe": (
            "Set processing.gain_db in volcano_s5i.yaml to the value reported "
            "as gain_db_calibrated. To extend to a different slider, add "
            "(slider - 54) dB per the bench gain LUT (gain_lut.json -> "
            "gain_db_table)."
        ),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-frames", type=int, default=30,
                   help="number of wire-phantom frames to average over (default 30)")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH,
                   help=f"output JSON path (default {OUTPUT_PATH})")
    args = p.parse_args(argv)
    result = derive_gain_db(n_frames=args.n_frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Calibrated processing.gain_db = {result['gain_db_calibrated']:+.2f} dB "
          f"(target = {result['calibration_target']})")
    print(f"  alt. wire-target gain_db = {result['gain_db_wire_target']:+.2f} dB")
    print(f"  alt. pure-amplitude gain_db = {result['gain_db_bg_amp_pure']:+.2f} dB "
          "(over-shoots due to clamp lift)")
    print()
    print("Amplitudes:")
    print(f"  sim   wire amp median  = {result['sim_envelope_amp_median_wire']:.3g}")
    print(f"  sim   water bg amp     = {result['sim_envelope_amp_water_bg_mean']:.3g}")
    print(f"  bench wire amp median  = {result['bench_envelope_amp_median_wire']:.3f}")
    print(f"  bench water bg amp     = {result['bench_envelope_amp_water_bg_at_slider_54']:.3f}")
    print()
    print("Wire-vs-background contrast (dB):")
    print(f"  sim   = {result['wire_vs_bg_contrast_db']['sim']:+.2f} dB")
    print(f"  bench = {result['wire_vs_bg_contrast_db']['bench']:+.2f} dB")
    print(f"  gap   = {result['wire_vs_bg_contrast_db']['gap_dB_bench_minus_sim']:+.2f} dB "
          "(positive = bench has more contrast)")
    print()
    print(f"Wrote derivation to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
