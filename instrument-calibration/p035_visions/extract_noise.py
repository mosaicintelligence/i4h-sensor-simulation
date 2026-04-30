#!/usr/bin/env python3
"""E5 (partial) — Noise characterization from the P_035_PointScatter polar arrays.

Picks an anechoic ROI per frame (water between wires, past the ring-down
extent, away from the reject floor) and characterizes the residual
displayed-pixel distribution. Pools by gain slider so we get one set of
stats per gain setting.

Important caveat
----------------
The pixel values are POST log-compression and POST quantization, and the
device clips to a hard reject floor at palette ~10. So the histogram we see
is NOT the raw RF noise distribution -- it is the noise as it appears on
the displayed image. The simulator's `processing.noise.{type, sigma}`
parameters are defined on the pre-log RF samples; converting our palette
sigma to RF sigma requires the E7 log-compression LUT.

Until E7 lands, we report:
  * the empirical palette mean / std / skew / kurtosis per gain group,
  * the best-fit Gaussian sigma in palette units,
  * a Rayleigh-vs-Gaussian Anderson-Darling-like preference summary,
  * histograms saved as PNG + numpy for downstream re-analysis.

Outputs
-------
  derived/noise/per_frame_stats.csv     -- mean, std, skew, kurt, n_samples per frame
  derived/noise/per_gain_stats.json     -- pooled stats + Gaussian sigma per gain
  derived/noise/per_gain_histograms.npy -- (n_gain_groups, 256) raw palette histograms
  derived/noise/noise_overview.png      -- diagnostic plot
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import sys
sys.path.insert(0, str(Path(__file__).parent))
from polar_utils import (  # noqa: E402
    PolarFrame,
    build_anechoic_mask,
    load_polar_dataset,
)
from extract_calibration_inputs import parse_scattering_box_dxf  # noqa: E402


def stats_of(samples: np.ndarray) -> dict[str, float]:
    if samples.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    skew=float("nan"), kurt=float("nan"),
                    p05=float("nan"), p50=float("nan"), p95=float("nan"))
    s = samples.astype(np.float64)
    mu = float(s.mean())
    sd = float(s.std())
    if sd > 0:
        z = (s - mu) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)
    else:
        skew = kurt = float("nan")
    return dict(
        n=int(s.size), mean=mu, std=sd, skew=skew, kurt=kurt,
        p05=float(np.percentile(s, 5)),
        p50=float(np.percentile(s, 50)),
        p95=float(np.percentile(s, 95)),
    )


def fit_rayleigh_sigma(samples: np.ndarray) -> float:
    """ML estimator for Rayleigh sigma: sqrt(mean(x^2) / 2)."""
    if samples.size == 0:
        return float("nan")
    s = samples.astype(np.float64)
    return float(np.sqrt(np.mean(s * s) / 2.0))


def neg_log_likelihood_gauss(samples: np.ndarray, mu: float, sigma: float) -> float:
    if sigma <= 0 or samples.size == 0:
        return float("inf")
    s = samples.astype(np.float64)
    return float(0.5 * np.sum(((s - mu) / sigma) ** 2)
                 + samples.size * (0.5 * np.log(2 * np.pi) + np.log(sigma)))


def neg_log_likelihood_rayleigh(samples: np.ndarray, sigma: float) -> float:
    if sigma <= 0 or samples.size == 0:
        return float("inf")
    s = samples.astype(np.float64)
    return float(np.sum(0.5 * (s / sigma) ** 2)
                 - np.sum(np.log(s / (sigma ** 2))))


def render_overview_png(
    per_gain_samples: dict[float, np.ndarray],
    per_gain_stats: dict[float, dict],
    out_path: Path,
):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, 256, 1)
    cmap = {44.0: "#1f77b4", 54.0: "#ff7f0e", 64.0: "#d62728"}
    for gain in sorted(per_gain_samples):
        s = per_gain_samples[gain]
        c = cmap.get(gain, "#444")
        st = per_gain_stats[gain]
        ax.hist(s, bins=bins, density=True, alpha=0.4,
                color=c, label=f"g={gain:.0f}  n={s.size}  mu={st['mean']:.1f}  sigma={st['std']:.2f}")
        # Gaussian fit overlay
        x = np.linspace(0, 255, 1024)
        y = np.exp(-0.5 * ((x - st['mean']) / st['std']) ** 2) / (st['std'] * np.sqrt(2 * np.pi))
        ax.plot(x, y, color=c, linestyle="-")
    ax.set_xlim(0, 200)
    ax.set_xlabel("Palette value")
    ax.set_ylabel("Density")
    ax.set_title("E5: Anechoic-ROI palette distribution per gain (Gaussian fit overlay)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--polar-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/polar"))
    p.add_argument("--align-csv", type=Path,
                   default=Path("P_035_PointScatter/derived/alignment_fit.csv"))
    p.add_argument("--meta-csv", type=Path,
                   default=Path("P_035_PointScatter/derived/frames_meta.csv"))
    p.add_argument("--dxf", type=Path,
                   default=Path("P_035_PointScatter/IVUS Scattering Box - Sketch 1.dxf"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/noise"))
    p.add_argument("--exclude-angle-deg", type=float, default=15.0)
    p.add_argument("--inner-radial-mm", type=float, default=5.0,
                   help="Exclude r < this (ringdown zone)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_polar_dataset(args.polar_dir, args.align_csv, args.meta_csv)
    wires, _ = parse_scattering_box_dxf(args.dxf)

    # ---- per-frame: pick ROI, dump stats ----
    per_frame_rows = []
    per_gain_samples: dict[float, list[np.ndarray]] = defaultdict(list)
    for fr in frames:
        mask = build_anechoic_mask(
            fr, wires,
            exclude_angle_deg=args.exclude_angle_deg,
            inner_radial_mm=args.inner_radial_mm,
        )
        samples = fr.arr[mask]
        st = stats_of(samples)
        per_frame_rows.append({
            "file": fr.file,
            "gain_slider": fr.gain_slider,
            "diameter_mm": fr.diameter_mm,
            **{k: v for k, v in st.items()},
        })
        per_gain_samples[fr.gain_slider].append(samples)

    with (args.out_dir / "per_frame_stats.csv").open("w", newline="") as f:
        cols = list(per_frame_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_frame_rows)

    # ---- per-gain pooled ----
    per_gain_pooled: dict[float, np.ndarray] = {
        g: np.concatenate(parts) for g, parts in per_gain_samples.items()
    }
    per_gain_stats: dict[float, dict] = {}
    histograms = []
    bin_edges = np.arange(0, 257, 1)
    for gain in sorted(per_gain_pooled):
        s = per_gain_pooled[gain]
        st = stats_of(s)
        sig_g = st["std"]
        nll_g = neg_log_likelihood_gauss(s, st["mean"], sig_g)
        sig_r = fit_rayleigh_sigma(s - max(0.0, s.min() - 1.0))  # shift origin so Rayleigh has support >0
        nll_r = neg_log_likelihood_rayleigh(np.clip(s - max(0.0, s.min() - 1.0), 1e-6, None), sig_r)
        per_gain_stats[gain] = {
            **st,
            "n_frames": int(len(per_gain_samples[gain])),
            "gauss_sigma": sig_g,
            "gauss_nll_per_sample": nll_g / s.size,
            "rayleigh_sigma_shifted": sig_r,
            "rayleigh_nll_per_sample": nll_r / s.size,
            "preferred_distribution": (
                "gaussian" if nll_g <= nll_r else "rayleigh"
            ),
        }
        h, _ = np.histogram(s, bins=bin_edges)
        histograms.append(h.astype(np.int64))

    np.save(args.out_dir / "per_gain_histograms.npy", np.stack(histograms, axis=0))

    out_json = {
        "method": ("anechoic ROI per frame: r in [inner_radial_mm, depth - 1 mm], "
                   "exclude +/- exclude_angle_deg around each wire's design theta, "
                   "exclude palette <= 11 (device reject)."),
        "units": "palette (0..255); convert to RF sigma after E7 log-compression fit.",
        "exclude_angle_deg": args.exclude_angle_deg,
        "inner_radial_mm": args.inner_radial_mm,
        "per_gain": {f"{g:.0f}": v for g, v in per_gain_stats.items()},
    }
    with (args.out_dir / "per_gain_stats.json").open("w") as f:
        json.dump(out_json, f, indent=2)

    if plt is not None:
        render_overview_png(per_gain_pooled, per_gain_stats,
                            args.out_dir / "noise_overview.png")

    # Console summary
    print(f"{'gain':>4s} {'frames':>6s} {'n_samp':>8s} "
          f"{'mean':>7s} {'std':>6s} {'skew':>6s} {'kurt':>6s} "
          f"{'gauss_sigma':>11s} {'pref':>9s}")
    for g, st in per_gain_stats.items():
        print(f"{g:>4.0f} {st['n_frames']:>6d} {st['n']:>8d} "
              f"{st['mean']:>7.2f} {st['std']:>6.2f} "
              f"{st['skew']:>6.2f} {st['kurt']:>6.2f} "
              f"{st['gauss_sigma']:>11.3f} {st['preferred_distribution']:>9s}")
    print(f"\nWrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
