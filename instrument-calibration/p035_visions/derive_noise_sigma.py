#!/usr/bin/env python3
"""Derive ``processing.noise.sigma`` from the protocol-correct E6 anechoic
AR-on deep-tail STD.

Calibration history:

  * Pass 6 (legacy): the bench reference was the P_035 wedge-masked
    anechoic ROI at slider 54 -- mean palette 46.2, std palette 20.27
    (E5, n=19 frames). The script bisected ``noise.sigma`` so the
    simulator's anechoic post-clamp MEAN palette matched 46.2.  But the
    P_035 ROI was the inter-wire wedge mask on a 5-wire wire phantom and
    wire-PSF sidelobes leak into the mask, inflating the apparent
    "noise" std by ~6 x (true device noise std at the same slider is
    ~3.27 palette).  Calibrating against the mean side-stepped the
    sidelobe pollution but is still anchored on a non-robust single take.

  * Pass post-P_035 migration: protocol-correct anchor is the E6
    anechoic AR-on deep tail (``ivus_test_0508/raw/c_take2_water/``,
    paired AR-on / AR-off at multiple operating points, no scatterers
    in the field).  We bisect ``noise.sigma`` so the simulator's pure-
    water render (``rs.World("water")``, no scatter) MATCHES the E6
    AR-on deep tail MEAN palette at the bench's native slider (avg of
    g50 D30 + g50 D60 -> 37.26 palette at slider 50).

    Why MEAN (and not STD): mean is monotonic-increasing in sigma until
    palette saturation, so a stable bisection is possible.  STD is non-
    monotonic (rises, peaks ~ 30 palette at sigma ~ 1e-2, then falls
    as palette saturates near 239), so std-anchored bisection has two
    solutions and is unreliable.  The bench-vs-sim STD comparison is
    reported as a DIAGNOSTIC: Gaussian RF noise -> Rayleigh envelope ->
    log-palette gives a coefficient-of-variation (std/mean) ~ 0.5; the
    bench's AR-on processing flattens the noise distribution to CV
    ~ 0.07, a shape that single-scalar Gaussian RF noise cannot
    reproduce.  Test F (Noise floor sigma) is expected to fail on the
    std criterion until a Pass 8 noise-shape calibration is added (e.g.,
    post-clamp Gaussian smoothing or a fundamentally different noise
    distribution).

Default target (E6 g50 D60 + D30 average mean) = 37.26 palette at
slider 50.

Usage::

    python instrument-calibration/p035_visions/derive_noise_sigma.py \\
        --target-mean 37.26 --n-frames 16
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
# Legacy P_035 paths kept only for back-compat; the new derivation reads
# from E6 (``ivus_test_0508/raw/c_take2_water/``).
BENCH_ROOT = WORKSPACE_ROOT / "P_035_PointScatter" / "derived"
LEGACY_OUTPUT_PATH = BENCH_ROOT / "noise" / "noise_sigma_derivation.json"
E6_SUMMARY_PATH = (
    WORKSPACE_ROOT / "ivus_test_0508" / "raw" / "c_take2_water"
    / "derived" / "ringdown" / "ringdown_summary_v2.json"
)
OUTPUT_PATH = (
    WORKSPACE_ROOT / "ivus_test_0508" / "raw" / "c_take2_water" / "derived"
    / "ringdown" / "noise_sigma_derivation.json"
)


def _load_e6_target(*, gain: float | None = None,
                    diameter_mm: float | None = None,
                    floor_palette: float = 11.0,
                    std_eps: float = 0.5) -> dict:
    """Average the AR-on deep-tail std across ALL usable E6 pairs.

    A pair is "usable" if its AR-on tail is NOT clipped to the device's
    reject floor (mean within ``std_eps`` of ``floor_palette`` AND std
    <= ``std_eps``).  If `gain` / `diameter_mm` are passed, only that
    specific pair is returned.

    Returns ``{ "target_std": float, "target_mean": float, "ref_slider":
    float, "pairs": [...] }``.
    """
    if not E6_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"E6 ringdown summary missing: {E6_SUMMARY_PATH}. "
            "Stage `ivus_test_0508/raw/c_take2_water/derived/ringdown/` "
            "first (run `derive_ringdown_v2.py`)."
        )
    summary = json.loads(E6_SUMMARY_PATH.read_text())
    usable = []
    for p in summary.get("pairs", []):
        on_mean = float(p["noise_floor_palette_mean_from_palette_on"])
        on_std = float(p["noise_floor_palette_std_from_palette_on"])
        at_floor = (abs(on_mean - floor_palette) <= std_eps
                    and on_std <= std_eps)
        if at_floor:
            continue
        if gain is not None and abs(float(p["gain"]) - gain) > 0.5:
            continue
        if (diameter_mm is not None and
                abs(float(p["diameter_mm"]) - diameter_mm) > 0.5):
            continue
        usable.append({
            "gain": float(p["gain"]),
            "diameter_mm": float(p["diameter_mm"]),
            "on_mean_palette": on_mean,
            "on_std_palette": on_std,
            "on_p50_palette": float(p["noise_floor_palette_p50_from_palette_on"]),
            "band_r_mm": p["noise_floor_band_r_mm"],
        })
    if not usable:
        raise RuntimeError(
            f"No usable E6 pairs (gain={gain}, D={diameter_mm}).  All AR-on "
            "tails are at the device reject floor; need higher-gain E6 "
            "captures.")
    target_std = float(np.mean([u["on_std_palette"] for u in usable]))
    target_mean = float(np.mean([u["on_mean_palette"] for u in usable]))
    ref_slider = float(np.mean([u["gain"] for u in usable]))
    return {
        "target_std": target_std,
        "target_mean": target_mean,
        "ref_slider": ref_slider,
        "pairs": usable,
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


def _measure_anechoic_bg_palette(*, cfg, materials, sigma: float,
                                 n_frames: int,
                                 gain_db_override: float | None = None
                                 ) -> dict[str, float]:
    """Render an anechoic-lumen scene with the FULL calibrated pipeline,
    overriding only ``noise_sigma`` (and optionally ``gain_db``), and
    return bg statistics in the clean radial band (past the ring-down
    extent, inside the calibrated FOV).
    """
    params = cfg.to_sim_params()
    params.noise_sigma = float(sigma)
    if gain_db_override is not None:
        params.gain_db = float(gain_db_override)
    # The bench E5 anechoic ROI was recorded in pure water (no blood/cells), so
    # the bench bg palette of 46.2 reflects pure electronic noise with no
    # scatter contribution. We mirror that here by using the ``water`` material
    # (mu0 = mu1 = sigma = 0) for the world background. Using ``lumen`` (which
    # has blood-like speckle, mu0 = mu1 = sigma = 0.1) would inflate the bg
    # palette and bias sigma low. OptiX needs >= 1 primitive; place a tiny
    # dummy sphere far outside the FOV so it cannot perturb the measurement.
    world = rs.World("water")
    world.add(Sphere(np.array([0.0, 1000.0, 0.0], dtype=np.float32),
                     0.001, materials.get_index("water")))
    sim = RaytracingUltrasoundSimulator(world, materials)
    probe = cfg.to_probe()
    rd_extent_mm = float(cfg.processing.ring_down.extent_mm)
    r_lo_mm = max(rd_extent_mm + 1.0, 5.0)
    r_hi_mm = min(0.9 * float(cfg.sim.t_far_mm), 25.0)
    band = []
    for k in range(n_frames):
        params.frame_seed = int(k + 1)
        out = sim.simulate(probe, params)
        f = _b_mode_to_theta_r(np.array(out, copy=True), cfg)
        n_r = f.shape[1]
        dr = float(cfg.sim.t_far_mm) / n_r
        r_lo = max(0, int(r_lo_mm / dr))
        r_hi = min(n_r, int(r_hi_mm / dr))
        band.append(f[:, r_lo:r_hi].copy())
    band = np.stack(band)
    return {
        "mean_palette": float(band.mean()),
        "median_palette": float(np.median(band)),
        "std_palette": float(band.std()),
        "p05_palette": float(np.percentile(band, 5)),
        "p95_palette": float(np.percentile(band, 95)),
        "n_pixels": int(band.size),
    }


def _bisect_sigma_on_mean(*, cfg, materials, target_mean: float,
                          ref_slider: float = 54.0,
                          n_frames: int = 8, max_iter: int = 25,
                          tol_palette: float = 0.5,
                          s_lo: float = 1e-6,
                          s_hi: float = 5e-3) -> tuple[float, list[dict]]:
    """Bisect ``noise_sigma`` so the post-clamp anechoic bg MEAN palette
    matches ``target_mean`` at ``ref_slider``.

    MEAN is monotonic-increasing in sigma on the rising side (until palette
    saturates near 239).  STD, by contrast, is non-monotonic (rises, peaks
    around sigma ~ 1e-2 at gain_db ~ 70 dB, then falls) so std-anchored
    bisection is unreliable.  We anchor on MEAN as the primary calibration
    knob (matches the operator's perception of anechoic brightness) and
    surface the resulting STD as a diagnostic.

    The sim is rendered at ``ref_slider`` -- bench reference (E6 captures
    are at slider 50, YAML default is slider 54).  We adjust
    ``params.gain_db`` to render at the bench operating point so we can
    compare without slider-step extrapolation.

    KNOWN SHAPE MISMATCH: at this gain_db the sim's noise palette CV
    (std/mean) is ~0.5 (Rayleigh-like; Gaussian RF noise -> Rayleigh
    envelope -> log_palette spread scales with log_multiplier).  The
    bench's CV at E6 AR-on is ~0.07 (much narrower).  This is a noise-
    MODEL limitation, not a parameter one: bench AR processing flattens
    the noise distribution in a way that single-scalar Gaussian RF noise
    cannot match.  The diagnostic STD value will therefore not match the
    bench's STD -- Test F (Noise floor sigma) is expected to fail until
    a Pass 8 noise-shape calibration is added (e.g., post-clamp Gaussian
    smoothing, or a fundamentally different noise shape).
    """
    history: list[dict] = []
    yaml_gain_db = float(cfg.processing.gain_db)
    gain_bump = float(ref_slider) - 54.0
    sim_gain_db = yaml_gain_db + gain_bump
    print(f"[derive_noise_sigma] target MEAN = {target_mean:.2f} palette, "
          f"ref_slider = {ref_slider:.0f}, sim gain_db = {sim_gain_db:+.2f} dB")

    def _probe(s):
        return _measure_anechoic_bg_palette(
            cfg=cfg, materials=materials, sigma=s, n_frames=n_frames,
            gain_db_override=sim_gain_db,
        )

    p_lo, p_hi = _probe(s_lo), _probe(s_hi)
    history.append({"sigma": s_lo, **p_lo})
    history.append({"sigma": s_hi, **p_hi})
    print(f"[derive_noise_sigma] bracket: sigma={s_lo:.2e} -> mean "
          f"{p_lo['mean_palette']:.2f}; sigma={s_hi:.2e} -> mean "
          f"{p_hi['mean_palette']:.2f}; target={target_mean:.2f}")
    # Expand the upper bracket if needed (target may exceed the initial s_hi
    # mean; mean is monotonic-increasing on the rising side).
    expand = 0
    while p_hi["mean_palette"] < target_mean and expand < 4:
        s_hi *= 2.0
        p_hi = _probe(s_hi)
        history.append({"sigma": s_hi, **p_hi})
        expand += 1
    # Expand lower bracket if needed (target may be below the noise floor;
    # unlikely with the E6 anchor at ~37, but handle it gracefully).
    expand = 0
    while p_lo["mean_palette"] > target_mean and expand < 4:
        s_lo *= 0.5
        p_lo = _probe(s_lo)
        history.append({"sigma": s_lo, **p_lo})
        expand += 1
    if not (p_lo["mean_palette"] <= target_mean <= p_hi["mean_palette"]):
        raise RuntimeError(
            f"could not bracket target MEAN {target_mean}: sigma=[{s_lo}, "
            f"{s_hi}] -> mean=[{p_lo['mean_palette']:.2f}, "
            f"{p_hi['mean_palette']:.2f}]"
        )
    s_mid = 0.5 * (s_lo + s_hi)
    p_mid = p_lo
    for it in range(max_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        p_mid = _probe(s_mid)
        history.append({"sigma": s_mid, **p_mid})
        print(f"[derive_noise_sigma] it={it} sigma={s_mid:.4e} -> "
              f"mean {p_mid['mean_palette']:.2f} std "
              f"{p_mid['std_palette']:.2f} (target mean={target_mean:.2f})")
        if abs(p_mid["mean_palette"] - target_mean) <= tol_palette:
            return float(s_mid), history
        if p_mid["mean_palette"] > target_mean:
            s_hi = s_mid
        else:
            s_lo = s_mid
    return float(s_mid), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Frames per bisection step (independent seeds).")
    parser.add_argument("--target-mean", type=float, default=None,
                        help=("Target bench anechoic mean palette.  Defaults "
                              "to the average AR-on deep-tail mean across all "
                              "usable E6 pairs at their native slider."))
    parser.add_argument("--ref-slider", type=float, default=None,
                        help=("Reference slider at which the sim is rendered "
                              "for the bisection.  Defaults to the average "
                              "slider across usable E6 pairs (50 for the "
                              "current capture).  The YAML's slider 54 "
                              "gain_db value is bumped to this slider so the "
                              "sim and bench operate at the same point."))
    parser.add_argument("--tol-palette", type=float, default=0.5,
                        help="Bisection convergence tolerance, palette units.")
    args = parser.parse_args()

    cfg = IvusSimConfig.from_yaml(YAML_PATH)
    materials = rs.Materials()

    e6_target = _load_e6_target()
    target_mean = args.target_mean if args.target_mean is not None else e6_target["target_mean"]
    target_std = e6_target["target_std"]
    ref_slider = args.ref_slider if args.ref_slider is not None else e6_target["ref_slider"]

    print(f"[derive_noise_sigma] E6 anchor: {len(e6_target['pairs'])} "
          f"usable pairs, avg target mean = {e6_target['target_mean']:.2f}, "
          f"avg target std = {e6_target['target_std']:.2f}, "
          f"avg ref slider = {e6_target['ref_slider']:.1f}")
    print(f"[derive_noise_sigma] current YAML noise.sigma = "
          f"{cfg.processing.noise.sigma:.4g}")

    sigma, history = _bisect_sigma_on_mean(
        cfg=cfg, materials=materials,
        target_mean=target_mean,
        ref_slider=ref_slider,
        n_frames=args.n_frames,
        tol_palette=args.tol_palette,
    )
    final_stats = history[-1]
    cv_sim = (final_stats["std_palette"] / final_stats["mean_palette"]
              if final_stats["mean_palette"] > 0 else float("nan"))
    cv_bench = target_std / target_mean if target_mean > 0 else float("nan")
    print()
    print(f"Calibrated processing.noise.sigma = {sigma:.4g}")
    print(f"  sim mean palette = {final_stats['mean_palette']:.2f}  "
          f"(target {target_mean:.2f})")
    print(f"  sim std palette  = {final_stats['std_palette']:.2f}  "
          f"(bench E6 AR-on std {target_std:.2f} -- KNOWN GAP, see method)")
    print(f"  sim median       = {final_stats['median_palette']:.2f}")
    print(f"  Coefficient-of-variation gap: sim CV={cv_sim:.2f} vs "
          f"bench CV={cv_bench:.2f}  -> sim distribution is ~"
          f"{cv_sim/max(cv_bench,1e-6):.1f}x wider in palette space "
          "(noise-shape model limitation, Pass 8 deferred).")

    out: dict[str, Any] = {
        "method_summary": (
            "Bisection of processing.noise.sigma so the simulator's anechoic "
            "post-clamp MEAN palette matches the protocol-correct E6 AR-on "
            "deep-tail MEAN (averaged across usable E6 pairs at their native "
            "slider, pure water bath, no scatterers, AR suppresses catheter "
            "ringdown -- the cleanest device-noise reference).  The sim is "
            "rendered AT THE BENCH SLIDER (gain_db bumped by ref_slider-54 "
            "from the YAML's slider-54 default) so MEAN is compared "
            "without slider-step extrapolation.  The legacy P_035 wedge-"
            "masked anechoic ROI (mean 46.2 at slider 54) is deprecated "
            "because wire-PSF sidelobe leakage into the ROI biased the "
            "measurement away from the true device-noise floor.\n\n"
            "Why MEAN (and not STD): palette MEAN is monotonic-increasing "
            "in sigma (until palette saturation) so a stable bisection is "
            "possible.  STD is non-monotonic (rises, peaks ~ palette 30 at "
            "sigma ~ 1e-2, then falls as palette saturates at 239), so "
            "std-anchored bisection has two solutions and is unreliable.  "
            "The bench-vs-sim STD comparison is reported as a DIAGNOSTIC: "
            "Gaussian RF noise -> Rayleigh envelope -> log_palette gives "
            "CV ~ 0.5; bench AR-on processing flattens the noise distribution "
            "to CV ~ 0.07, a shape that single-scalar Gaussian RF noise "
            "cannot reproduce.  Test F (Noise floor sigma) is expected to "
            "fail until a Pass 8 noise-shape calibration is added (e.g., "
            "post-clamp Gaussian smoothing, or a fundamentally different "
            "noise distribution)."
        ),
        "target_mean": target_mean,
        "target_std_bench_diagnostic": target_std,
        "target_ref_slider": ref_slider,
        "n_frames": args.n_frames,
        "tol_palette": args.tol_palette,
        "calibrated_sigma": sigma,
        "previous_yaml_sigma": float(cfg.processing.noise.sigma),
        "final_stats": final_stats,
        "cv_sim_palette": cv_sim,
        "cv_bench_palette": cv_bench,
        "cv_ratio_sim_over_bench": cv_sim / max(cv_bench, 1e-6),
        "history": history,
        "e6_anchor": e6_target,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote derivation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
