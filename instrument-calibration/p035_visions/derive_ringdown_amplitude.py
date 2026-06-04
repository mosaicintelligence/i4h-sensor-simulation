#!/usr/bin/env python3
"""Derive ``processing.ring_down.amplitude`` against the E6 AR-off bench peak.

Calibration approach:

  An earlier iteration analytically computed ``ring_down.amplitude``
  from the P_035 wedge-masked bench peak_palette_excess (187.1 at
  slider 54) under ``log_multiplier = 137.4``, giving amplitude = 23.0
    envelope-amp at slider 54.  Anchor source was a saturated P_035
    template (peak = 232 palette, near the 239 saturation ceiling), so
    the envelope-amp number was inferred from a clamped measurement.

  * Pass post-P_035 migration (THIS SCRIPT): bisect
    ``ring_down.amplitude`` so the sim's anechoic mean A-line peak
    palette matches the E6 AR-off bench peak at slider 50 (= 222 palette,
    median across the (g50, D30) and (g50, D60) paired captures).  Slider
    50 is the highest E6 capture where the bench peak is NOT saturated
    (g40 D60 has peak 152.75 -- well below saturation; g50 has 222 --
    well below saturation; no g54 or higher captures exist in the E6
    set).  The sim is rendered at slider 50 by bumping ``gain_db`` by
    ``(50 - 54) = -4`` dB; the calibrated amplitude is the slider-54
    reference value used directly in the YAML.

  * Why slider 50 and not slider 54: the YAML's reference is slider 54,
    but the bench at slider 54 will saturate the inner-zone ring-down
    palette ceiling (extrapolated peak ~ 250 palette > 239 saturation),
    so we calibrate at the bench's native operating point and let the
    amplitude scaling formula do the slider conversion implicitly.

  * Joint cascade: this script depends on ``gain_db`` and ``noise.sigma``
    already being calibrated (via ``derive_gain_db.py`` and
    ``derive_noise_sigma.py``).  After running this script, re-run
    ``derive_noise_sigma.py`` to confirm the noise floor hasn't shifted
    materially (it shouldn't -- ring-down is r<3 mm and noise.sigma is
    measured at r>4 mm).

Default target = 222.0 palette at slider 50 (median across (g50, D30) and
(g50, D60) AR-off bench captures from ``ringdown_summary_v2.json``).

Usage::

    python instrument-calibration/p035_visions/derive_ringdown_amplitude.py \\
        --target-peak 222.0 --n-frames 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
SIM_ROOT = WORKSPACE_ROOT / "i4h-sensor-simulation" / "ultrasound-raytracing"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import raysim as rs  # noqa: E402
from raysim import IvusSimConfig, RaytracingUltrasoundSimulator  # noqa: E402
from raysim.ray_sim_python import Sphere  # noqa: E402

YAML_PATH = HERE / "volcano_s5i.yaml"
E6_SUMMARY_PATH = (
    WORKSPACE_ROOT / "ivus_test_0508" / "raw" / "c_take2_water"
    / "derived" / "ringdown" / "ringdown_summary_v2.json"
)
OUTPUT_PATH = (
    WORKSPACE_ROOT / "ivus_test_0508" / "raw" / "c_take2_water" / "derived"
    / "ringdown" / "ringdown_amplitude_derivation.json"
)


def _load_e6_peak_target(*, ref_slider: float) -> dict:
    """Average AR-off bench peak palette across all E6 pairs at ``ref_slider``.

    The slider-50 pairs (g50, D30) and (g50, D60) give peaks 222.25 and
    222.0 palette respectively -- the bench is essentially diameter-
    independent at this gain, so averaging is just for noise reduction.
    Returns ``{ "target_peak": float, "ref_slider": float, "pairs":
    [...] }``.
    """
    if not E6_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"E6 ringdown summary missing: {E6_SUMMARY_PATH}. "
            "Stage `ivus_test_0508/raw/c_take2_water/derived/ringdown/` "
            "first (run `derive_ringdown_v2.py`)."
        )
    summary = json.loads(E6_SUMMARY_PATH.read_text())
    pairs_at_slider = []
    for p in summary.get("pairs", []):
        if abs(float(p["gain"]) - ref_slider) > 0.5:
            continue
        pairs_at_slider.append({
            "gain": float(p["gain"]),
            "diameter_mm": float(p["diameter_mm"]),
            "peak_palette_off": float(p["peak_palette_off"]),
            "peak_depth_mm": float(p["peak_depth_mm"]),
            "extent_mm": float(p["extent_mm"]),
        })
    if not pairs_at_slider:
        raise RuntimeError(
            f"No E6 pairs at slider {ref_slider}.  Available pairs: "
            + ", ".join(f"g{p['gain']:.0f} D{p['diameter_mm']:.0f}"
                        for p in summary.get("pairs", []))
        )
    target_peak = float(np.median([p["peak_palette_off"]
                                   for p in pairs_at_slider]))
    return {
        "target_peak": target_peak,
        "ref_slider": float(ref_slider),
        "pairs": pairs_at_slider,
        "source_summary": str(E6_SUMMARY_PATH.relative_to(WORKSPACE_ROOT)),
    }


def _b_mode_to_theta_r(frame: np.ndarray, cfg: IvusSimConfig) -> np.ndarray:
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


def _measure_ringdown_peak(*, cfg, materials, amplitude: float,
                            n_frames: int,
                            gain_db_override: float | None) -> dict[str, float]:
    """Render an anechoic-lumen scene with the FULL calibrated pipeline,
    overriding ``ring_down.amplitude`` (and optionally ``gain_db``), and
    return the mean-A-line peak palette + depth + extent.

    The "ring-down zone" is r in [0, 4] mm (inside the calibrated dead-
    zone is masked off automatically by the sim).
    """
    params = cfg.to_sim_params()
    params.ring_down.amplitude = float(amplitude)
    if gain_db_override is not None:
        params.gain_db = float(gain_db_override)
    # E6 AR-off is pure water (no scatterers).  Mirror that by rendering
    # against rs.World("water") with a dummy far-out sphere (OptiX needs
    # >= 1 primitive).  Using "lumen" would inject blood-like speckle on
    # top of the ring-down which biases the peak measurement.
    world = rs.World("water")
    world.add(Sphere(np.array([0.0, 1000.0, 0.0], dtype=np.float32),
                     0.001, materials.get_index("water")))
    sim = RaytracingUltrasoundSimulator(world, materials)
    probe = cfg.to_probe()
    alines = []
    for k in range(n_frames):
        params.frame_seed = int(k + 1)
        out = sim.simulate(probe, params)
        f = _b_mode_to_theta_r(np.array(out, copy=True), cfg)
        alines.append(f.mean(axis=0))  # mean over theta
    aline = np.stack(alines).mean(axis=0)  # mean over frames

    n_r = aline.size
    dr_mm = float(cfg.sim.t_far_mm) / n_r
    r_mm = (np.arange(n_r) + 0.5) * dr_mm

    # Search for the peak in the ring-down zone (r in [0, 4] mm).
    rd_mask = (r_mm >= 0.0) & (r_mm <= 4.0)
    peak_idx = int(np.argmax(aline[rd_mask]))
    peak_idx_global = int(np.where(rd_mask)[0][peak_idx])
    peak_palette = float(aline[peak_idx_global])
    peak_depth_mm = float(r_mm[peak_idx_global])

    # Extent = first r > peak where palette drops below 5% of peak.
    threshold = 0.05 * peak_palette
    extent_idx = peak_idx_global
    for i in range(peak_idx_global, n_r):
        if aline[i] < threshold:
            extent_idx = i
            break
    else:
        extent_idx = n_r - 1
    extent_mm = float(r_mm[extent_idx])

    return {
        "peak_palette": peak_palette,
        "peak_depth_mm": peak_depth_mm,
        "extent_mm": extent_mm,
    }


def _bisect_amplitude_on_peak(*, cfg, materials, target_peak: float,
                               ref_slider: float = 50.0,
                               n_frames: int = 8, max_iter: int = 25,
                               tol_palette: float = 1.0,
                               a_lo: float = 1e-4,
                               a_hi: float = 200.0) -> tuple[float, list[dict]]:
    """Bisect ``ring_down.amplitude`` so the sim's anechoic ring-down peak
    palette matches ``target_peak`` at ``ref_slider``.

    Sim is rendered at ``ref_slider`` by bumping ``params.gain_db`` by
    ``(ref_slider - 54)`` dB so the sim and bench operate at the same
    point.  The returned amplitude is the slider-54 reference (the YAML
    convention -- the amplitude scaling with slider is applied implicitly
    by the gain_db delta).

    KNOWN GAIN-RESPONSE LIMITATION: at high gain_db the sim's ring-down
    peak saturates against the palette ceiling (239), so the bisection
    range above ~ palette 230 will return amplitudes that flatten the
    sim's response with slider.  In practice this hasn't been a problem
    for the slider-50 reference (bench peak 222 leaves headroom), but
    cross-slider validation should be done in tier1 after the
    calibration lands.
    """
    history: list[dict] = []
    yaml_gain_db = float(cfg.processing.gain_db)
    gain_bump = float(ref_slider) - 54.0
    sim_gain_db = yaml_gain_db + gain_bump
    print(f"[derive_ringdown_amplitude] target peak = {target_peak:.2f} palette, "
          f"ref_slider = {ref_slider:.0f}, sim gain_db = {sim_gain_db:+.2f} dB")

    def _probe(a):
        return _measure_ringdown_peak(
            cfg=cfg, materials=materials, amplitude=a, n_frames=n_frames,
            gain_db_override=sim_gain_db,
        )

    p_lo, p_hi = _probe(a_lo), _probe(a_hi)
    history.append({"amplitude": a_lo, **p_lo})
    history.append({"amplitude": a_hi, **p_hi})
    print(f"[derive_ringdown_amplitude] bracket: amp={a_lo:.5g} -> peak "
          f"{p_lo['peak_palette']:.2f}; amp={a_hi:.5g} -> peak "
          f"{p_hi['peak_palette']:.2f}; target={target_peak:.2f}")
    expand = 0
    while p_hi["peak_palette"] < target_peak and expand < 8:
        a_hi *= 2.0
        p_hi = _probe(a_hi)
        history.append({"amplitude": a_hi, **p_hi})
        expand += 1
    expand = 0
    while p_lo["peak_palette"] > target_peak and expand < 8:
        a_lo *= 0.5
        p_lo = _probe(a_lo)
        history.append({"amplitude": a_lo, **p_lo})
        expand += 1
    if not (p_lo["peak_palette"] <= target_peak <= p_hi["peak_palette"]):
        raise RuntimeError(
            f"could not bracket target PEAK {target_peak}: "
            f"amplitude=[{a_lo}, {a_hi}] -> peak=["
            f"{p_lo['peak_palette']:.2f}, {p_hi['peak_palette']:.2f}].  "
            "Sim ring-down peak may be saturating against the palette "
            "ceiling at the upper bracket -- try lowering noise.sigma or "
            "gain_db first."
        )

    a_mid = 0.5 * (a_lo + a_hi)
    p_mid = p_lo
    for it in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        p_mid = _probe(a_mid)
        history.append({"amplitude": a_mid, **p_mid})
        print(f"[derive_ringdown_amplitude] it={it} amp={a_mid:.5g} -> "
              f"peak {p_mid['peak_palette']:.2f} extent "
              f"{p_mid['extent_mm']:.2f} mm (target peak={target_peak:.2f})")
        if abs(p_mid["peak_palette"] - target_peak) <= tol_palette:
            return float(a_mid), history
        if p_mid["peak_palette"] > target_peak:
            a_hi = a_mid
        else:
            a_lo = a_mid
    return float(a_mid), history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Frames per bisection step (independent seeds).")
    parser.add_argument("--target-peak", type=float, default=None,
                        help=("Target bench ring-down peak palette.  "
                              "Defaults to the median AR-off peak across "
                              "all E6 pairs at the reference slider."))
    parser.add_argument("--ref-slider", type=float, default=50.0,
                        help=("Reference slider at which the sim is "
                              "rendered for the bisection.  Defaults to "
                              "50 (E6 anchor)."))
    parser.add_argument("--tol-palette", type=float, default=1.0,
                        help="Bisection convergence tolerance, palette units.")
    args = parser.parse_args()

    cfg = IvusSimConfig.from_yaml(YAML_PATH)
    materials = rs.Materials()

    e6_target = _load_e6_peak_target(ref_slider=args.ref_slider)
    target_peak = (args.target_peak
                   if args.target_peak is not None
                   else e6_target["target_peak"])
    print(f"[derive_ringdown_amplitude] E6 anchor pairs at slider "
          f"{args.ref_slider:.0f}: "
          + ", ".join(f"g{p['gain']:.0f}_D{p['diameter_mm']:.0f}"
                      f"(peak={p['peak_palette_off']:.1f})"
                      for p in e6_target["pairs"]))
    print(f"[derive_ringdown_amplitude] target peak = {target_peak:.2f} palette")
    print(f"[derive_ringdown_amplitude] current YAML "
          f"ring_down.amplitude = {float(cfg.processing.ring_down.amplitude):.3f}")
    print(f"[derive_ringdown_amplitude] current YAML "
          f"gain_db = {float(cfg.processing.gain_db):.2f} dB")
    print(f"[derive_ringdown_amplitude] current YAML "
          f"noise.sigma = {float(cfg.processing.noise.sigma):.6f}")

    amplitude_calibrated, history = _bisect_amplitude_on_peak(
        cfg=cfg, materials=materials, target_peak=target_peak,
        ref_slider=args.ref_slider, n_frames=args.n_frames,
        tol_palette=args.tol_palette,
    )

    # Final verification at slider 54 (YAML reference) and the target
    # slider (50) so the user can see the cross-slider response.
    yaml_gain_db = float(cfg.processing.gain_db)
    verify_at_target = _measure_ringdown_peak(
        cfg=cfg, materials=materials,
        amplitude=amplitude_calibrated, n_frames=args.n_frames,
        gain_db_override=yaml_gain_db + (args.ref_slider - 54.0),
    )
    verify_at_54 = _measure_ringdown_peak(
        cfg=cfg, materials=materials,
        amplitude=amplitude_calibrated, n_frames=args.n_frames,
        gain_db_override=yaml_gain_db,
    )

    print()
    print("=== Calibration result ===")
    print(f"ring_down.amplitude = {amplitude_calibrated:.3f} envelope-amp "
          f"(slider 54 reference)")
    print(f"  was: {float(cfg.processing.ring_down.amplitude):.3f}")
    print(f"  ratio: {amplitude_calibrated / float(cfg.processing.ring_down.amplitude):.3f}x "
          f"({20 * math.log10(max(amplitude_calibrated / float(cfg.processing.ring_down.amplitude), 1e-6)):+.2f} dB)")
    print()
    print(f"Verification at slider {args.ref_slider:.0f} (calibration target):")
    print(f"  sim peak = {verify_at_target['peak_palette']:.2f} palette "
          f"(target {target_peak:.2f}; "
          f"err {verify_at_target['peak_palette'] - target_peak:+.2f})")
    print(f"  sim extent = {verify_at_target['extent_mm']:.2f} mm")
    print(f"  sim peak depth = {verify_at_target['peak_depth_mm']:.2f} mm")
    print(f"Verification at slider 54 (YAML reference):")
    print(f"  sim peak = {verify_at_54['peak_palette']:.2f} palette "
          f"(unconstrained; bench at slider 54 unavailable -- "
          "saturation risk)")
    print(f"  sim extent = {verify_at_54['extent_mm']:.2f} mm")

    out = {
        "method": ("Sim-in-the-loop bisection of ring_down.amplitude "
                   "against E6 AR-off bench peak palette at slider 50.  "
                   "Sim rendered with the full calibrated pipeline (gain_db, "
                   "noise.sigma, all materials) and only ring_down.amplitude "
                   "overridden per bisection step."),
        "yaml_path": str(YAML_PATH.relative_to(WORKSPACE_ROOT)),
        "yaml_gain_db_at_calibration": yaml_gain_db,
        "yaml_noise_sigma_at_calibration": float(cfg.processing.noise.sigma),
        "yaml_ring_down_amplitude_before": float(
            cfg.processing.ring_down.amplitude),
        "calibrated_ring_down_amplitude": float(amplitude_calibrated),
        "amplitude_change_dB": float(
            20 * math.log10(max(amplitude_calibrated
                                / float(cfg.processing.ring_down.amplitude),
                                1e-6))
        ),
        "target_peak_palette": float(target_peak),
        "ref_slider": float(args.ref_slider),
        "e6_anchor_pairs": e6_target["pairs"],
        "verification_at_ref_slider": verify_at_target,
        "verification_at_slider_54": verify_at_54,
        "n_frames_per_step": int(args.n_frames),
        "tol_palette": float(args.tol_palette),
        "history": history,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"\nUpdate volcano_s5i.yaml processing.ring_down.amplitude to "
          f"{amplitude_calibrated:.3f}")


if __name__ == "__main__":
    main()
