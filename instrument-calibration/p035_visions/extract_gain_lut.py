#!/usr/bin/env python3
"""E7 (partial) -- Grayscale / log-compression / gain-LUT calibration.

Without a calibrated reflector at multiple gains (the proper E7 procedure),
we recover the compression slope from internal redundancy in the dataset:
the same physical signal imaged at two different gain sliders.

Estimators of log_multiplier (palette per log10(amplitude))
-----------------------------------------------------------

(1) RING-DOWN PEAK cross-gain (best -- single physical source, highest SNR).
    The catheter ring-down is identical from frame to frame; only the gain
    knob changes the displayed peak. For two captures with the same
    physical ringdown but different gains:

        pixel_g2 - pixel_g1 = log_multiplier * (g2 - g1) / 20

    (this assumes 1 slider step = 1 dB; we sanity-check this below.)
    From ringdown_fit.json, gain 44 -> 54 differences at D = 40 and D = 60
    both give log_multiplier ~ 112.

(2) NOISE STD at gain 64 (least-clipped Rayleigh).
    For envelope of complex Gaussian noise (Rayleigh-distributed),
    standard deviation in dB is independent of mean amplitude:
        std_dB(Rayleigh envelope) = 20 / ln(10) * sqrt(pi^2 / 24)
                                  = 5.57 dB
    Therefore log_multiplier = std_palette * 20 / 5.57. Gain 64 gives
    std_palette = 35.0 -> log_multiplier ~ 126.

(3) MID-RANGE radial-profile cross-gain.
    Median palette in the wire-free, post-ringdown, mid-depth region
    (5-15 mm) at gain 64 vs gain 54 (gain 44 is reject-clipped):
        log_multiplier = (P_64(r) - P_54(r)) * 20 / (g_64 - g_54)

    We compute this radius-by-radius and take the median across r.

(4) RING-DOWN SPECKLE-FLOOR cross-gain (sanity check; clipping-biased
    at low gain).
    Same as (1) but using the speckle-floor palette (mean of the
    speckle-only region behind the ring-down). At gain 44 the floor is
    almost touching the device's reject palette (~11) so this estimator
    is biased.

Derived parameters
------------------

    log_floor          -- chosen by convention so that amplitude = 1 maps
                          to pixel 0 in the simulator (log_floor = 1.0).
                          The actual reject palette (~11) is reproduced
                          by setting noise.sigma at the right level.
    dynamic_range_db   -- (255_palette_ceiling - reject_palette) * 20 /
                          log_multiplier. With our log_multiplier ~ 112
                          and palette range 11..239, DR ~ 41 dB.
    gain_db(slider)    -- {44, 54, 64} -> {-10, 0, +10} dB by convention
                          (slider 54 is our canonical reference, used in
                          E2/E4/E5/E6). Backed up by direct measurement
                          on ring-down peaks (44 -> 54 = +10 dB confirmed
                          to within 0.2 dB).
    noise.sigma        -- in RF amplitude units, referenced to gain 54.
                          Derived from the gain-64 envelope mean (cleanest
                          Rayleigh measurement) divided by the gain shift.
    ring_down.amplitude -- envelope-amplitude EXCESS above the speckle
                          floor, at gain 54 reference, in the same units
                          as noise.sigma.

Calibration convention
----------------------

    pixel = log_multiplier * log10(max(envelope_amp, log_floor))
    log_floor = 1.0  (amp=1 -> pixel=0)

    The simulator's gain knob shifts the displayed pixel by:
        pixel(g) = pixel(g_ref) + log_multiplier * (g - g_ref) / 20

    Output values are AT THE SIMULATOR REFERENCE GAIN = 54. To produce a
    different gain, scale all amplitudes (noise.sigma, ring_down.amplitude,
    scatter intensities) by 10^((g - 54) / 20).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, str(Path(__file__).parent))
from polar_utils import load_polar_dataset, build_anechoic_mask  # noqa: E402
from extract_calibration_inputs import parse_scattering_box_dxf  # noqa: E402


# Constant: std of 20*log10(Rayleigh envelope) is sqrt(pi^2/24) * 20/ln(10).
RAYLEIGH_LOG_STD_DB = 20.0 / math.log(10.0) * math.pi / (2.0 * math.sqrt(6.0))
# = 5.572 dB

REJECT_PALETTE = 11.0
SATURATION_PALETTE = 239.0
SLIDER_TO_DB_HYPOTHESIS = 1.0  # 1 slider step = 1 dB (working assumption)
REFERENCE_GAIN_SLIDER = 54.0


# ----- Method 1: ringdown peak cross-gain -----


def estimate_logmult_from_ringdown_peaks(ringdown_fit: dict) -> list[dict]:
    """For each pair of (same diameter, two gains), compute log_multiplier.

    Saturated peaks are dropped (ringdown peak is biased low when clipped).
    """
    rows = ringdown_fit["groups"]
    by_d_gain: dict[float, dict[float, dict]] = {}
    for r in rows:
        by_d_gain.setdefault(r["diameter_mm"], {})[r["gain_slider"]] = r

    out = []
    for d, by_g in by_d_gain.items():
        gains = sorted(by_g.keys())
        for i in range(len(gains)):
            for j in range(i + 1, len(gains)):
                g1, g2 = gains[i], gains[j]
                r1, r2 = by_g[g1], by_g[g2]
                if r1["saturated"] or r2["saturated"]:
                    continue
                dpx = r2["peak_palette"] - r1["peak_palette"]
                dg_db = (g2 - g1) * SLIDER_TO_DB_HYPOTHESIS
                if dg_db == 0.0:
                    continue
                lm = dpx * 20.0 / dg_db
                out.append(dict(
                    method="ringdown_peak",
                    diameter_mm=d,
                    g1=g1, g2=g2,
                    palette_g1=r1["peak_palette"],
                    palette_g2=r2["peak_palette"],
                    delta_palette=dpx,
                    delta_db=dg_db,
                    log_multiplier=lm,
                ))
    return out


# ----- Method 2: noise std at high gain (cleanest Rayleigh) -----


def estimate_logmult_from_noise_std(noise_stats: dict) -> list[dict]:
    out = []
    for slider_str, stats in noise_stats["per_gain"].items():
        slider = float(slider_str)
        sd_palette = stats["std"]
        # Skip gains with significant low-tail clipping. We use the test
        # mean - 3*sigma > REJECT_PALETTE.
        if stats["mean"] - 3 * sd_palette < REJECT_PALETTE:
            clipped = True
        else:
            clipped = False
        lm = sd_palette * 20.0 / (RAYLEIGH_LOG_STD_DB * 20.0 / math.log(10.0))
        # ^ above: lm = sd_palette / (5.57 / 20) ?  Let me redo:
        # std_dB(Rayleigh envelope) = 5.572 dB. A pixel diff of `sd_palette`
        # corresponds to a dB diff of (sd_palette * 20 / log_mult). Setting
        # this equal to 5.572:
        #     log_mult = sd_palette * 20 / 5.572
        lm = sd_palette * 20.0 / RAYLEIGH_LOG_STD_DB
        out.append(dict(
            method="noise_std",
            gain_slider=slider,
            std_palette=sd_palette,
            log_multiplier=lm,
            reject_clipped=clipped,
        ))
    return out


# ----- Method 3: mid-range radial-profile cross-gain -----


def median_radial_profiles_per_gain(
    frames, wires, r_lo_mm: float, r_hi_mm: float,
    exclude_angle_deg: float = 15.0,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return {gain_slider: (r_grid_mm, median_palette_per_r)} pooled across
    frames at a single diameter (chosen as the most-populated diameter at
    each gain).
    """
    # Group by (gain, diameter)
    by_g_d: dict[tuple[float, float], list] = {}
    for fr in frames:
        by_g_d.setdefault((fr.gain_slider, fr.diameter_mm), []).append(fr)

    # For each gain, pick the most-populated diameter (so radii are aligned).
    gains = sorted({fr.gain_slider for fr in frames})
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for g in gains:
        # Pick D=60 if available (largest, most-populated for our dataset),
        # otherwise the most-populated diameter at this gain.
        ds = [d for (gg, d) in by_g_d if gg == g]
        if 60.0 in ds:
            d_pick = 60.0
        else:
            d_pick = max(ds, key=lambda d: len(by_g_d[(g, d)]))
        sel = by_g_d[(g, d_pick)]

        # Resample all frames to a common r-grid (use the smallest dr).
        dr = min(fr.dr_mm for fr in sel)
        n_r = int(round((r_hi_mm + 1.0) / dr))
        r_grid = (np.arange(n_r) + 0.5) * dr

        # Collect masked palette samples per (frame, r-bin)
        per_r_samples: list[list[float]] = [[] for _ in range(n_r)]
        for fr in sel:
            mask = build_anechoic_mask(
                fr, wires,
                exclude_angle_deg=exclude_angle_deg,
                inner_radial_mm=r_lo_mm,
                outer_radial_margin_mm=1.0,
            )
            arr = fr.arr  # (num_theta, num_r)
            r_native = (np.arange(arr.shape[1]) + 0.5) * fr.dr_mm
            for j_native, r_val in enumerate(r_native):
                if r_val < r_lo_mm or r_val > r_hi_mm:
                    continue
                j_grid = int(round((r_val - 0.5 * dr) / dr))
                if not (0 <= j_grid < n_r):
                    continue
                col_pixels = arr[mask[:, j_native], j_native]
                if col_pixels.size:
                    per_r_samples[j_grid].extend(col_pixels.tolist())

        med = np.array([
            float(np.median(s)) if s else np.nan
            for s in per_r_samples
        ])
        out[g] = (r_grid, med)
    return out


def estimate_logmult_from_radial_profiles(
    profiles: dict[float, tuple[np.ndarray, np.ndarray]],
    r_lo_mm: float, r_hi_mm: float,
) -> list[dict]:
    out = []
    gains = sorted(profiles.keys())
    for i in range(len(gains)):
        for j in range(i + 1, len(gains)):
            g1, g2 = gains[i], gains[j]
            r1, p1 = profiles[g1]
            r2, p2 = profiles[g2]
            # Resample p1 onto r2 grid via linear interpolation
            p1_on_r2 = np.interp(r2, r1, p1)
            in_range = (r2 >= r_lo_mm) & (r2 <= r_hi_mm)
            valid = in_range & np.isfinite(p1_on_r2) & np.isfinite(p2)
            # Drop reject-clipped bins (palette <= 13 = reject + small margin)
            valid &= (p1_on_r2 > 13.0) & (p2 > 13.0)
            # Drop saturated bins (palette >= 235)
            valid &= (p1_on_r2 < 235.0) & (p2 < 235.0)
            if valid.sum() < 5:
                continue
            dpx = p2[valid] - p1_on_r2[valid]
            dg_db = (g2 - g1) * SLIDER_TO_DB_HYPOTHESIS
            lm_per_r = dpx * 20.0 / dg_db
            out.append(dict(
                method="mid_range_profile",
                g1=g1, g2=g2,
                n_radial_bins=int(valid.sum()),
                delta_palette_median=float(np.median(dpx)),
                delta_palette_iqr=float(np.subtract(*np.percentile(dpx, [75, 25]))),
                log_multiplier=float(np.median(lm_per_r)),
                log_multiplier_iqr=float(np.subtract(*np.percentile(lm_per_r, [75, 25]))),
            ))
    return out


# ----- Method 4: ringdown speckle-floor cross-gain -----


def estimate_logmult_from_speckle_floor(ringdown_fit: dict) -> list[dict]:
    rows = ringdown_fit["groups"]
    by_d_gain: dict[float, dict[float, dict]] = {}
    for r in rows:
        by_d_gain.setdefault(r["diameter_mm"], {})[r["gain_slider"]] = r

    out = []
    for d, by_g in by_d_gain.items():
        gains = sorted(by_g.keys())
        for i in range(len(gains)):
            for j in range(i + 1, len(gains)):
                g1, g2 = gains[i], gains[j]
                r1, r2 = by_g[g1], by_g[g2]
                f1 = r1["speckle_floor_palette"]
                f2 = r2["speckle_floor_palette"]
                # Reject-clipped if floor is within 5 palette of reject
                clipped = (f1 - REJECT_PALETTE < 5.0) or (f2 - REJECT_PALETTE < 5.0)
                dpx = f2 - f1
                dg_db = (g2 - g1) * SLIDER_TO_DB_HYPOTHESIS
                if dg_db == 0.0:
                    continue
                lm = dpx * 20.0 / dg_db
                out.append(dict(
                    method="speckle_floor",
                    diameter_mm=d,
                    g1=g1, g2=g2,
                    floor_g1=f1,
                    floor_g2=f2,
                    delta_palette=dpx,
                    delta_db=dg_db,
                    log_multiplier=lm,
                    reject_clipped=clipped,
                ))
    return out


# ----- main -----


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--polar-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/polar"))
    p.add_argument("--align-csv", type=Path,
                   default=Path("P_035_PointScatter/derived/alignment_fit.csv"))
    p.add_argument("--meta-csv", type=Path,
                   default=Path("P_035_PointScatter/derived/frames_meta.csv"))
    p.add_argument("--dxf", type=Path,
                   default=Path("P_035_PointScatter/IVUS Scattering Box - Sketch 1.dxf"))
    p.add_argument("--ringdown-fit", type=Path,
                   default=Path("P_035_PointScatter/derived/ringdown/ringdown_fit.json"))
    p.add_argument("--noise-stats", type=Path,
                   default=Path("P_035_PointScatter/derived/noise/per_gain_stats.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/gain_lut"))
    p.add_argument("--mid-range-r-lo-mm", type=float, default=5.0,
                   help="Lower r for the mid-range radial-profile method")
    p.add_argument("--mid-range-r-hi-mm", type=float, default=15.0,
                   help="Upper r for the mid-range radial-profile method")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.ringdown_fit.open() as f:
        ringdown_fit = json.load(f)
    with args.noise_stats.open() as f:
        noise_stats = json.load(f)
    frames = load_polar_dataset(args.polar_dir, args.align_csv, args.meta_csv)
    wires, _ = parse_scattering_box_dxf(args.dxf)

    # ---- Apply all four estimators ----
    est_rd_peak = estimate_logmult_from_ringdown_peaks(ringdown_fit)
    est_noise = estimate_logmult_from_noise_std(noise_stats)
    profiles = median_radial_profiles_per_gain(
        frames, wires,
        r_lo_mm=args.mid_range_r_lo_mm,
        r_hi_mm=args.mid_range_r_hi_mm,
    )
    est_profile = estimate_logmult_from_radial_profiles(
        profiles,
        r_lo_mm=args.mid_range_r_lo_mm,
        r_hi_mm=args.mid_range_r_hi_mm,
    )
    est_floor = estimate_logmult_from_speckle_floor(ringdown_fit)

    # ---- Pick canonical log_multiplier ----
    # The ring-down PEAK method is our primary: same physical signal, high
    # SNR, no clipping (we filter saturated peaks). Take its mean.
    if est_rd_peak:
        canonical = float(np.mean([e["log_multiplier"] for e in est_rd_peak]))
        log_mult_uncert = float(np.std([e["log_multiplier"] for e in est_rd_peak]))
    else:
        # Fallback to noise-std method, gain 64
        clean_noise = [e for e in est_noise if not e["reject_clipped"]]
        if clean_noise:
            canonical = float(np.mean([e["log_multiplier"] for e in clean_noise]))
            log_mult_uncert = 15.0  # broad assumed uncertainty
        else:
            canonical = 60.0
            log_mult_uncert = 30.0

    # ---- Derived parameters ----
    log_floor = 1.0  # convention
    dynamic_range_db = (SATURATION_PALETTE - REJECT_PALETTE) * 20.0 / canonical

    # gain_db mapping: hypothesis 1 step = 1 dB, with reference at slider 54.
    sliders = sorted({fr.gain_slider for fr in frames})
    gain_db_table = {
        s: (s - REFERENCE_GAIN_SLIDER) * SLIDER_TO_DB_HYPOTHESIS
        for s in sliders
    }

    # Convert noise.sigma. Use the cleanest noise (gain 64).
    g64 = noise_stats["per_gain"].get("64")
    if g64 is None:
        sys.exit("noise stats missing gain 64")
    # envelope_mean_palette = log_mult * log10(envelope_mean_amp / log_floor)
    # ANCHOR: pixel = log_mult * log10(amp), so amp = 10^(pixel/log_mult).
    envelope_mean_amp_g64 = 10.0 ** (g64["mean"] / canonical)
    sigma_complex_g64 = envelope_mean_amp_g64 / math.sqrt(math.pi / 2.0)
    # Reference to gain 54 (-10 dB shift):
    sigma_complex_ref = sigma_complex_g64 / 10.0 ** ((64.0 - 54.0) / 20.0)
    noise_sigma_at_ref = sigma_complex_ref

    # Convert ring_down.amplitude. Use gain 54, D=60 (canonical) and the
    # peak palette EXCESS over speckle floor.
    rd_g54 = next(
        r for r in ringdown_fit["groups"]
        if r["gain_slider"] == 54.0 and r["diameter_mm"] == 60.0
    )
    rd_excess_palette = rd_g54["peak_palette_excess"]
    rd_amp_at_ref = 10.0 ** (rd_excess_palette / canonical)

    # ---- Output JSON ----
    out_json = {
        "method_summary": __doc__.strip().splitlines()[0],
        "calibration_convention": (
            "pixel = log_multiplier * log10(envelope_amp); log_floor = 1.0; "
            "amp at reference gain (slider 54) is in arbitrary linear units; "
            "gain shift: pixel(g) = pixel(54) + log_multiplier * (g - 54) / 20."
        ),
        "constants": {
            "rayleigh_log_std_db": RAYLEIGH_LOG_STD_DB,
            "reject_palette": REJECT_PALETTE,
            "saturation_palette": SATURATION_PALETTE,
            "slider_to_db_hypothesis": SLIDER_TO_DB_HYPOTHESIS,
            "reference_gain_slider": REFERENCE_GAIN_SLIDER,
        },
        "estimators": {
            "ringdown_peak": est_rd_peak,
            "noise_std": est_noise,
            "mid_range_profile": est_profile,
            "speckle_floor": est_floor,
        },
        "canonical_log_multiplier": {
            "value": canonical,
            "uncertainty_palette_per_log10amp": log_mult_uncert,
            "source": "mean of ringdown_peak estimators",
        },
        "derived": {
            "log_multiplier": canonical,
            "log_floor": log_floor,
            "dynamic_range_db": dynamic_range_db,
            "gain_db_table": gain_db_table,
            "noise.sigma_at_gain_54": noise_sigma_at_ref,
            "ring_down.amplitude_at_gain_54": rd_amp_at_ref,
            "noise.sigma_derivation": (
                f"envelope_mean_palette(g64)={g64['mean']:.1f} -> "
                f"envelope_mean_amp={envelope_mean_amp_g64:.3f} -> "
                f"sigma_complex(g64)={sigma_complex_g64:.3f} -> "
                f"sigma_complex(g54)={sigma_complex_ref:.4f}"
            ),
            "ring_down.amplitude_derivation": (
                f"peak_excess_palette(g54,D60)={rd_excess_palette:.1f} -> "
                f"envelope_amp={rd_amp_at_ref:.3f} (at gain 54 reference)"
            ),
        },
    }
    out_path = args.out_dir / "gain_lut.json"
    with out_path.open("w") as f:
        json.dump(out_json, f, indent=2)

    # ---- Plot ----
    if plt is not None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        # Each estimator as scatter with method label
        method_colors = {
            "ringdown_peak": "C0",
            "noise_std": "C1",
            "mid_range_profile": "C2",
            "speckle_floor": "C3",
        }
        x_pos = 0
        labels = []
        for method, lst in [
            ("ringdown_peak", est_rd_peak),
            ("noise_std", est_noise),
            ("mid_range_profile", est_profile),
            ("speckle_floor", est_floor),
        ]:
            for e in lst:
                lm = e["log_multiplier"]
                clipped = e.get("reject_clipped", False)
                marker = "x" if clipped else "o"
                ax.scatter([x_pos], [lm], c=method_colors[method],
                           marker=marker, s=80, alpha=0.85)
                if method == "ringdown_peak":
                    lbl = f"RD-pk D={e['diameter_mm']:.0f} g{e['g1']:.0f}->g{e['g2']:.0f}"
                elif method == "noise_std":
                    lbl = f"noise-std g{e['gain_slider']:.0f}"
                elif method == "mid_range_profile":
                    lbl = f"mid-prof g{e['g1']:.0f}->g{e['g2']:.0f}"
                else:
                    lbl = f"floor D={e['diameter_mm']:.0f} g{e['g1']:.0f}->g{e['g2']:.0f}"
                labels.append(lbl)
                x_pos += 1
        ax.axhline(canonical, color="k", linestyle="--",
                   label=f"canonical = {canonical:.1f} (mean of RD-pk)")
        ax.fill_between([-0.5, x_pos - 0.5],
                        canonical - log_mult_uncert,
                        canonical + log_mult_uncert,
                        color="k", alpha=0.1, label="+/- 1 sigma")
        ax.set_xticks(range(x_pos))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("log_multiplier (palette per log10(amplitude))")
        ax.set_title("Cross-gain estimators of log_multiplier "
                     "(x = clipped/biased)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out_dir / "gain_lut_overview.png", dpi=140)
        plt.close(fig)

        # Plot the radial profiles per gain
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for g, (r, p) in sorted(profiles.items()):
            ax.plot(r, p, label=f"gain {g:.0f}", lw=1.5)
        ax.axvspan(args.mid_range_r_lo_mm, args.mid_range_r_hi_mm,
                   alpha=0.1, color="green", label="mid-range fit window")
        ax.axhline(REJECT_PALETTE, color="gray", linestyle=":",
                   label=f"reject = {REJECT_PALETTE}")
        ax.axhline(SATURATION_PALETTE, color="gray", linestyle=":",
                   label=f"saturation = {SATURATION_PALETTE}")
        ax.set_xlabel("Depth from device center (mm)")
        ax.set_ylabel("Median wire-free palette")
        ax.set_title("Per-gain median wire-free radial profile (D=60 mm)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "radial_profiles_per_gain.png", dpi=140)
        plt.close(fig)

    # ---- Console report ----
    print()
    print("=" * 68)
    print("E7 (partial) -- gain LUT calibration")
    print("=" * 68)
    print()
    print("Estimators of log_multiplier (palette per log10(amplitude)):")
    print()
    for e in est_rd_peak:
        print(f"  ringdown peak  D={e['diameter_mm']:.0f}  "
              f"g{e['g1']:.0f}->g{e['g2']:.0f} "
              f"(dP={e['delta_palette']:+.1f}, dG={e['delta_db']:+.0f} dB):"
              f"  log_mult = {e['log_multiplier']:.1f}")
    for e in est_noise:
        flag = " [reject-clipped]" if e["reject_clipped"] else ""
        print(f"  noise std      g{e['gain_slider']:.0f} "
              f"(sigma={e['std_palette']:.1f}):  "
              f"log_mult = {e['log_multiplier']:.1f}{flag}")
    for e in est_profile:
        print(f"  mid-prof       g{e['g1']:.0f}->g{e['g2']:.0f} "
              f"(n={e['n_radial_bins']} bins, dP_med={e['delta_palette_median']:+.1f}):"
              f"  log_mult = {e['log_multiplier']:.1f} (IQR {e['log_multiplier_iqr']:.1f})")
    for e in est_floor:
        flag = " [reject-clipped]" if e["reject_clipped"] else ""
        print(f"  speckle floor  D={e['diameter_mm']:.0f}  "
              f"g{e['g1']:.0f}->g{e['g2']:.0f} "
              f"(dP={e['delta_palette']:+.1f}):  "
              f"log_mult = {e['log_multiplier']:.1f}{flag}")
    print()
    print(f"Canonical log_multiplier = {canonical:.1f} +/- {log_mult_uncert:.1f}")
    print(f"  (mean of {len(est_rd_peak)} ringdown-peak measurements)")
    print()
    print("Derived parameters (in simulator's amp = 10^(pixel/log_mult) units):")
    print(f"  log_floor              = {log_floor}")
    print(f"  dynamic_range_db       = {dynamic_range_db:.1f}")
    print(f"  gain_db(slider)        = {gain_db_table}")
    print(f"  noise.sigma at g54     = {noise_sigma_at_ref:.4f}")
    print(f"  ring_down.amplitude    = {rd_amp_at_ref:.2f}  (at g54 reference)")
    print()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
