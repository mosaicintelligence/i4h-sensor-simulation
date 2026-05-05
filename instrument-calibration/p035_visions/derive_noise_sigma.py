#!/usr/bin/env python3
"""Derive ``processing.noise.sigma`` from the bench anechoic-water mean palette.

Until Pass 6 the simulator was noiseless (sigma = 0) and the anechoic floor was
produced entirely by water-scatter speckle that mostly fell below the device's
reject palette (the median rendered palette in lumen was ~11 = reject floor).
The bench's gain-54 anechoic ROI sits at mean palette 46.2 (E5, n=19 frames,
800k+ samples) -- the device's electronic noise floor, not scatter, sets that
level. Pass 6 introduces additive Gaussian RF noise to reproduce that floor.

The YAML's old sigma value (2.6347) was derived from the gain-64 mean palette
back-stepped by an *assumed* 10 dB slider step. That value is too large in
practice: rendering with sigma = 2.6347 alone (no scatter, no ring-down) drives
the post-clamp mean palette to ~58, well above the bench target.

This script bisects ``processing.noise.sigma`` so the simulator's anechoic
post-clamp mean palette (rendered with the FULL calibrated pipeline -- TGC,
gain_db, log compression, display window, median clip) matches the bench's
gain-54 reference. The result is written to
``derived/noise/noise_sigma_derivation.json`` so it is colocated with the
other calibration artifacts and can be cited from ``volcano_s5i.yaml``.

Usage::

    python instrument-calibration/p035_visions/derive_noise_sigma.py \\
        --target-palette 46.2 --n-frames 16
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
BENCH_ROOT = WORKSPACE_ROOT / "P_035_PointScatter" / "derived"
OUTPUT_PATH = BENCH_ROOT / "noise" / "noise_sigma_derivation.json"


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
                                 n_frames: int) -> dict[str, float]:
    """Render an anechoic-lumen scene with the FULL calibrated pipeline,
    overriding only ``noise_sigma``, and return bg statistics in the clean
    radial band (past the ring-down extent, inside the calibrated FOV).
    """
    params = cfg.to_sim_params()
    params.noise_sigma = float(sigma)
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


def _bisect_sigma(*, cfg, materials, target_palette: float,
                  n_frames: int = 8, max_iter: int = 25,
                  tol_palette: float = 0.3) -> tuple[float, list[dict]]:
    """Bisect ``noise_sigma`` so the post-clamp anechoic bg mean palette
    matches ``target_palette``. Returns (sigma, history)."""
    history: list[dict] = []
    # Bracket the target. The bg palette is monotonic-increasing in sigma in
    # the noise-dominated regime. Start with a wide bracket spanning the
    # plausible range.
    # Bracket: in Pass 6 v1 (noise post-gain) the calibrated sigma was ~2.3.
    # In Pass 6 v2 (noise pre-PSF), the noise is amplified by gain_db (large
    # for the wire-target calibration ~+74 dB) and smoothed by the PSF (which
    # reduces per-pixel std by sqrt(sum(kernel^2))). The net effect can
    # require a much smaller input sigma -- start with a wide bracket and
    # let the expansion logic close in.
    s_lo, s_hi = 1e-5, 4.0
    p_lo = _measure_anechoic_bg_palette(cfg=cfg, materials=materials,
                                        sigma=s_lo, n_frames=n_frames)
    p_hi = _measure_anechoic_bg_palette(cfg=cfg, materials=materials,
                                        sigma=s_hi, n_frames=n_frames)
    history.append({"sigma": s_lo, **p_lo})
    history.append({"sigma": s_hi, **p_hi})
    print(f"[derive_noise_sigma] bracket: sigma={s_lo:.4f} -> palette "
          f"{p_lo['mean_palette']:.2f}; sigma={s_hi:.4f} -> palette "
          f"{p_hi['mean_palette']:.2f}; target={target_palette:.2f}")
    if p_hi["mean_palette"] < target_palette:
        # Need a larger upper bracket.
        for _ in range(4):
            s_hi *= 2.0
            p_hi = _measure_anechoic_bg_palette(cfg=cfg, materials=materials,
                                                sigma=s_hi, n_frames=n_frames)
            history.append({"sigma": s_hi, **p_hi})
            if p_hi["mean_palette"] >= target_palette:
                break
    if p_lo["mean_palette"] > target_palette:
        # Need a smaller lower bracket.
        for _ in range(4):
            s_lo *= 0.5
            p_lo = _measure_anechoic_bg_palette(cfg=cfg, materials=materials,
                                                sigma=s_lo, n_frames=n_frames)
            history.append({"sigma": s_lo, **p_lo})
            if p_lo["mean_palette"] <= target_palette:
                break
    if not (p_lo["mean_palette"] <= target_palette <= p_hi["mean_palette"]):
        raise RuntimeError(
            f"could not bracket target {target_palette}: "
            f"sigma=[{s_lo}, {s_hi}] -> palette=[{p_lo['mean_palette']:.2f}, "
            f"{p_hi['mean_palette']:.2f}]"
        )
    s_mid = 0.5 * (s_lo + s_hi)
    p_mid = p_lo
    for it in range(max_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        p_mid = _measure_anechoic_bg_palette(cfg=cfg, materials=materials,
                                             sigma=s_mid, n_frames=n_frames)
        history.append({"sigma": s_mid, **p_mid})
        print(f"[derive_noise_sigma] it={it} sigma={s_mid:.4f} -> palette "
              f"{p_mid['mean_palette']:.2f} (target={target_palette:.2f})")
        if abs(p_mid["mean_palette"] - target_palette) <= tol_palette:
            return float(s_mid), history
        if p_mid["mean_palette"] > target_palette:
            s_hi = s_mid
        else:
            s_lo = s_mid
    return float(s_mid), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Frames per bisection step (independent seeds).")
    parser.add_argument("--target-palette", type=float, default=46.2,
                        help="Bench gain-54 anechoic mean palette (E5).")
    parser.add_argument("--tol-palette", type=float, default=0.3,
                        help="Bisection convergence tolerance, palette units.")
    args = parser.parse_args()

    cfg = IvusSimConfig.from_yaml(YAML_PATH)
    materials = rs.Materials()
    print(f"[derive_noise_sigma] target bench palette = {args.target_palette}")
    print(f"[derive_noise_sigma] current YAML noise.sigma = "
          f"{cfg.processing.noise.sigma}")

    sigma, history = _bisect_sigma(
        cfg=cfg, materials=materials,
        target_palette=args.target_palette,
        n_frames=args.n_frames,
        tol_palette=args.tol_palette,
    )
    final_stats = history[-1]
    print()
    print(f"Calibrated processing.noise.sigma = {sigma:.4f}")
    print(f"  bg mean palette  = {final_stats['mean_palette']:.2f}  "
          f"(target {args.target_palette:.2f})")
    print(f"  bg median palette= {final_stats['median_palette']:.2f}  "
          f"(bench p50 ~ 44 from skew=1.6)")
    print(f"  bg std palette   = {final_stats['std_palette']:.2f}  "
          f"(bench std=20.3)")

    out: dict[str, Any] = {
        "method_summary": (
            "Bisection of processing.noise.sigma so the simulator's anechoic "
            "post-clamp mean palette matches the bench gain-54 reference "
            "(E5: anechoic ROI mean palette 46.2, n=19 frames). The YAML's "
            "old value (2.6347, derived from gain-64 back-stepped 10 dB) "
            "was too large because the 54->64 slider step is wider than "
            "10 dB and because the bench's 'noise floor' subsumes both "
            "electronic noise and any residual scatter; we calibrate against "
            "the *observed bench palette* directly to side-step both."
        ),
        "target_palette": args.target_palette,
        "n_frames": args.n_frames,
        "tol_palette": args.tol_palette,
        "calibrated_sigma": sigma,
        "previous_yaml_sigma": float(cfg.processing.noise.sigma),
        "final_stats": final_stats,
        "history": history,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote derivation to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
