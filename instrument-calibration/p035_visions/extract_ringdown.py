#!/usr/bin/env python3
"""E6 — Ring-down extraction from the polar P_035_PointScatter arrays.

Goal: populate the `processing.ring_down.*` block of the simulator
config from this dataset.

AR-mode caveat (corrected 2026-05-12 from ivus_test_0508 paired-capture
analysis):

  Private tag 0x00291006 was originally read as the AR-on/off flag. It
  is in fact a *capability* flag (=1 on every frame regardless of AR
  state). The actual AR-on/off state lives in 0x00291007 ("ar_state" in
  the new extract_metadata.py). 18 of 19 P_035 frames have ar_state = 0
  (AR-OFF, raw ringdown visible); only FILE0013 has ar_state = 1
  (AR-ON, ringdown subtracted).

  So the ringdown profile measured here is the RAW ringdown (AR-OFF) for
  every frame except FILE0013 -- not the AR residual as originally
  documented. The simulator's matching mode is therefore
  `subtract_reference: false` against this template; flipping to `true`
  also requires a separately-derived AR-on residual template (to be
  measured from CASE0000 in ivus_test_0508 in the upcoming re-fit).

Method
------
For each polar frame we take the median over theta. The wires are
localized in theta and median is robust to them, so the radial profile
is dominated by:

  * a hard reject floor (v == 10) inside ~1 mm where the device blacks
    out the catheter cross-section,
  * the bright ring-down dome peaking at r ~ catheter OD (~1.9 mm),
  * a decaying tail blending into the speckle floor by r ~ 4-5 mm,
  * the speckle floor (rises with gain).

We frame the ring-down as

    ringdown_excess(r) = profile(r) - speckle_floor

and characterize it per gain group with:
  * peak amplitude (palette units), depth of peak,
  * extent_mm (depth at which excess drops below 5% of peak),
  * exponential decay constant tau_mm fit to the post-peak tail
    (in palette units; converting to linear-RF amplitude requires
    the E7 log-compression calibration -- left to the next step).

We also save the per-group mean profile as an .npy template so the
simulator can use `decay: measured` with `waveform_path` if the
exponential approximation is not tight enough.

Outputs
-------
  derived/ringdown/per_frame_profiles.npy   float32 (n_frames, num_r)
  derived/ringdown/per_frame_meta.csv       gain, diameter, dr_mm
  derived/ringdown/group_profiles.npy       float32 (n_groups, num_r) mean profile per (gain, diameter)
  derived/ringdown/ringdown_fit.json        per-group fit parameters
  derived/ringdown/ringdown_overview.png    visualization
  derived/ringdown/ringdown_template_g{G}_d{D}.npy   canonical 1D template per group
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Optional matplotlib import; only needed for the overview PNG.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ----- helpers -----

REJECT_FLOOR = 10  # palette index s5 paints inside the catheter cross-section


@dataclass
class GroupFit:
    """Per-group ring-down fit, in palette units.

    The post-peak decay is fit as LINEAR in depth, not exponential,
    because the s5's display is log-compressed: an exp(-r/tau) decay
    in RF amplitude maps to (palette = const - slope * r) on the
    screen, with slope = log_multiplier / (tau_rf * ln 10). The
    palette slope is therefore directly proportional to the true RF
    decay rate -- conversion to physical tau_rf needs the E7 log
    fit, but the slope itself is meaningful right now.
    """
    gain_slider: float
    diameter_mm: float
    n_frames: int
    dr_mm: float

    speckle_floor_palette: float        # median palette value in the wire-free far band
    peak_palette: float                 # peak value in the median-over-theta profile
    peak_palette_excess: float          # peak - speckle_floor
    peak_depth_mm: float                # depth at which peak occurs
    extent_mm: float                    # depth at which excess drops below 5% of peak excess
    extent_palette_threshold: float     # absolute palette value at the extent boundary

    # Linear-in-depth post-peak fit (palette units / mm).
    palette_slope_per_mm: float | None
    palette_intercept_at_peak: float | None
    linear_fit_r2: float | None
    linear_fit_window_mm: tuple[float, float] | None

    saturated: bool                     # True if peak hit the palette ceiling (~239)


def load_polar_frames(polar_dir: Path, manifest_csv: Path):
    rows = list(csv.DictReader(manifest_csv.open()))
    profiles = []
    metas = []
    for r in rows:
        arr = np.load(polar_dir / f"{r['file']}.npy")
        # Robust median over theta, in palette units (0..255).
        prof = np.median(arr, axis=0).astype(np.float64)
        profiles.append(prof)
        metas.append(r)
    return profiles, metas


def estimate_speckle_floor(prof: np.ndarray, dr_mm: float,
                           depth_band_mm: tuple[float, float] = (18.0, 25.0)) -> float:
    """Median palette value in a far depth band (mostly speckle, no wires after median-over-theta).

    The 18-25 mm band is past every wire's radial half-width and well into
    the noise floor at every gain in this dataset.
    """
    i0 = int(round(depth_band_mm[0] / dr_mm))
    i1 = int(round(depth_band_mm[1] / dr_mm))
    i1 = min(i1, len(prof))
    if i1 - i0 < 5:
        # Diameter 35 mm has depth=17mm; fall back to the outer 30%.
        i0 = int(0.7 * len(prof))
        i1 = len(prof)
    return float(np.median(prof[i0:i1]))


def fit_post_peak_linear_palette(
    profile: np.ndarray, dr_mm: float, peak_idx: int,
    floor: float, min_frac: float = 0.10,
) -> tuple[float | None, float | None, float | None, tuple[float, float] | None]:
    """Fit palette(r) = a - slope * (r - r_peak) on the post-peak decay.

    Window: from peak out to where (palette - floor) drops below
    `min_frac * peak_excess`. Past that we are in speckle-floor
    territory and the slope estimate is dominated by noise.

    Returns (slope_per_mm, intercept_at_peak, R^2, window_mm).
    Slope is positive for a true decay (palette decreases with depth).
    """
    peak_excess = float(profile[peak_idx]) - floor
    if peak_excess <= 0:
        return None, None, None, None
    threshold = floor + min_frac * peak_excess

    # Find first index past the peak where palette drops below threshold.
    end_idx = peak_idx
    for i in range(peak_idx + 1, len(profile)):
        if profile[i] < threshold:
            end_idx = i
            break
    else:
        end_idx = len(profile) - 1
    if end_idx - peak_idx < 3:
        return None, None, None, None

    r = (np.arange(peak_idx, end_idx + 1, dtype=np.float64)) * dr_mm
    y = profile[peak_idx:end_idx + 1].astype(np.float64)
    r_rel = r - r[0]

    A = np.vstack([np.ones_like(r_rel), r_rel]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope_signed = sol  # palette = intercept + slope_signed * r_rel
    slope_per_mm = -float(slope_signed)  # positive = decay
    y_hat = A @ sol
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    window = (float(r[0]), float(r[-1]))
    return slope_per_mm, float(intercept), r2, window


def find_extent(prof_excess: np.ndarray, dr_mm: float, peak_idx: int,
                threshold_frac: float = 0.05) -> tuple[float, float]:
    """First depth past the peak where excess drops below `threshold_frac`*peak."""
    peak_v = float(prof_excess[peak_idx])
    threshold = peak_v * threshold_frac
    for i in range(peak_idx, len(prof_excess)):
        if prof_excess[i] < threshold:
            return i * dr_mm, threshold
    return (len(prof_excess) - 1) * dr_mm, threshold


def fit_group(
    profs: list[np.ndarray], dr_mm: float, gain: float, diameter: float
) -> tuple[GroupFit, np.ndarray]:
    """Fit ring-down for one (gain, diameter) group; return GroupFit + mean profile."""
    L = min(len(p) for p in profs)
    stack = np.stack([p[:L] for p in profs], axis=0)
    mean_prof = stack.mean(axis=0)

    floor = estimate_speckle_floor(mean_prof, dr_mm)
    excess = np.clip(mean_prof - floor, 0.0, None)

    # Search for the peak past the reject region (~1 mm).
    search_start = int(round(1.0 / dr_mm))
    peak_idx = int(np.argmax(excess[search_start:]) + search_start)
    peak_v = float(mean_prof[peak_idx])
    peak_depth = peak_idx * dr_mm

    extent_mm, extent_thr = find_extent(excess, dr_mm, peak_idx, threshold_frac=0.05)
    slope, intercept, r2, window = fit_post_peak_linear_palette(
        mean_prof, dr_mm, peak_idx, floor
    )

    saturated = peak_v >= 235.0  # palette ceiling on the s5 is 239

    fit = GroupFit(
        gain_slider=gain,
        diameter_mm=diameter,
        n_frames=stack.shape[0],
        dr_mm=dr_mm,
        speckle_floor_palette=floor,
        peak_palette=peak_v,
        peak_palette_excess=float(excess[peak_idx]),
        peak_depth_mm=peak_depth,
        extent_mm=extent_mm,
        extent_palette_threshold=extent_thr,
        palette_slope_per_mm=slope,
        palette_intercept_at_peak=intercept,
        linear_fit_r2=r2,
        linear_fit_window_mm=window,
        saturated=saturated,
    )
    return fit, mean_prof


def render_overview_png(
    fits: list[GroupFit], group_profiles: list[np.ndarray],
    out_path: Path, catheter_radius_mm: float,
):
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    cmap = {44.0: "#1f77b4", 54.0: "#ff7f0e", 64.0: "#d62728"}
    style_diam = {60.0: "-", 40.0: "--", 35.0: ":"}

    for fit, prof in zip(fits, group_profiles):
        r = np.arange(len(prof)) * fit.dr_mm
        c = cmap.get(fit.gain_slider, "#444")
        ls = style_diam.get(fit.diameter_mm, "-.")
        label = f"g={fit.gain_slider:.0f} D={fit.diameter_mm:.0f}  n={fit.n_frames}"
        axes[0].plot(r, prof, color=c, linestyle=ls, label=label)
        axes[0].axhline(fit.speckle_floor_palette, color=c, linestyle=ls, alpha=0.2)

        excess = np.clip(prof - fit.speckle_floor_palette, 0.0, None)
        axes[1].plot(r, excess, color=c, linestyle=ls, label=label)
        if fit.palette_slope_per_mm is not None and fit.linear_fit_window_mm:
            r0, r1 = fit.linear_fit_window_mm
            r_fit = np.linspace(r0, r1, 64)
            y_fit_palette = (
                fit.palette_intercept_at_peak
                - fit.palette_slope_per_mm * (r_fit - r0)
            )
            y_fit_excess = np.clip(y_fit_palette - fit.speckle_floor_palette, 1e-3, None)
            axes[1].plot(r_fit, y_fit_excess, color=c, linestyle=":", alpha=0.7)

    for ax in axes:
        ax.axvline(catheter_radius_mm, color="cyan", alpha=0.5, label="catheter OD")
        ax.set_xlim(0, max(len(p) for p in group_profiles) * fits[0].dr_mm)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Palette value (0..255)")
    axes[0].set_title("Median-over-theta radial profile (mean of frames in each group)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-1, 256)
    axes[1].set_ylabel("Profile - speckle floor (palette units, log)")
    axes[1].set_xlabel("Depth from device center (mm)")
    axes[1].set_title("Ring-down excess + exponential fits (dotted)")

    fig.suptitle("E6: Ring-down extraction from P_035_PointScatter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----- main -----


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--polar-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/polar"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/ringdown"))
    p.add_argument("--catheter-radius-mm", type=float, default=1.905,
                   help="PV .035 catheter outer radius (DXF: 1.905 mm)")
    args = p.parse_args(argv)

    manifest = args.polar_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing {manifest}; run p035_unwrap.py first.")

    profiles, metas = load_polar_frames(args.polar_dir, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame outputs
    L = max(len(p) for p in profiles)
    pad = np.full((len(profiles), L), np.nan, dtype=np.float32)
    for i, pr in enumerate(profiles):
        pad[i, :len(pr)] = pr
    np.save(args.out_dir / "per_frame_profiles.npy", pad)
    with (args.out_dir / "per_frame_meta.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "gain_slider", "diameter_mm", "dr_mm", "num_r",
                    "speckle_floor", "peak_palette", "peak_depth_mm"])
        for pr, m in zip(profiles, metas):
            dr = float(m["dr_mm"])
            floor = estimate_speckle_floor(pr, dr)
            search_start = int(round(1.0 / dr))
            excess = np.clip(pr - floor, 0.0, None)
            pi_ = int(np.argmax(excess[search_start:]) + search_start)
            w.writerow([m["file"], m["gain_slider"], m["diameter_mm"],
                        f"{dr:.6f}", len(pr),
                        f"{floor:.3f}", f"{pr[pi_]:.3f}", f"{pi_ * dr:.3f}"])

    # Group by (gain, diameter) and fit each
    groups: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    group_dr: dict[tuple[float, float], float] = {}
    for pr, m in zip(profiles, metas):
        key = (float(m["gain_slider"]), float(m["diameter_mm"]))
        groups[key].append(pr)
        group_dr[key] = float(m["dr_mm"])

    fits: list[GroupFit] = []
    group_profiles: list[np.ndarray] = []
    template_paths: dict[str, str] = {}
    for key in sorted(groups.keys()):
        gain, diam = key
        fit, mean_prof = fit_group(groups[key], group_dr[key], gain, diam)
        fits.append(fit)
        group_profiles.append(mean_prof)
        # Save the canonical 1D template for use with `decay: measured`.
        tname = f"ringdown_template_g{int(gain)}_d{int(diam)}.npy"
        np.save(args.out_dir / tname, mean_prof.astype(np.float32))
        template_paths[f"g{int(gain)}_d{int(diam)}"] = tname

    # Pad group profiles to common length for stacking
    Lg = max(len(p) for p in group_profiles)
    stacked = np.full((len(group_profiles), Lg), np.nan, dtype=np.float32)
    for i, pr in enumerate(group_profiles):
        stacked[i, :len(pr)] = pr
    np.save(args.out_dir / "group_profiles.npy", stacked)

    # JSON summary
    out_json = {
        "method": "median-over-theta radial profile; speckle floor from r in [18, 25] mm; "
                  "post-peak exponential fit; extent at 5% of peak excess",
        "units": "palette (0..255); convert to RF amplitude after E7 log-compression fit",
        "catheter_radius_mm": args.catheter_radius_mm,
        "n_frames_total": len(profiles),
        "groups": [asdict(f) for f in fits],
        "templates": template_paths,
    }
    with (args.out_dir / "ringdown_fit.json").open("w") as f:
        json.dump(out_json, f, indent=2)

    if plt is not None:
        render_overview_png(fits, group_profiles,
                            args.out_dir / "ringdown_overview.png",
                            args.catheter_radius_mm)

    # Console summary
    print(f"{'gain':>4s} {'D_mm':>5s} {'n':>2s} {'floor':>6s} "
          f"{'peak':>6s} {'excess':>7s} {'r_pk':>5s} {'extent':>6s} "
          f"{'slope':>7s} {'R^2':>5s} {'win_mm':>11s} {'sat':>4s}")
    for f in fits:
        slope = f"{f.palette_slope_per_mm:6.2f}" if f.palette_slope_per_mm else "  --  "
        r2 = f"{f.linear_fit_r2:.2f}" if f.linear_fit_r2 else " -- "
        win = (f"{f.linear_fit_window_mm[0]:.2f}-{f.linear_fit_window_mm[1]:.2f}"
               if f.linear_fit_window_mm else "    -    ")
        print(f"{f.gain_slider:>4.0f} {f.diameter_mm:>5.0f} {f.n_frames:>2d} "
              f"{f.speckle_floor_palette:>6.2f} "
              f"{f.peak_palette:>6.1f} {f.peak_palette_excess:>7.1f} "
              f"{f.peak_depth_mm:>5.2f} {f.extent_mm:>6.2f} "
              f"{slope:>7s} {r2:>5s} {win:>11s} "
              f"{('sat' if f.saturated else '   '):>4s}")
    print(f"\nWrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
