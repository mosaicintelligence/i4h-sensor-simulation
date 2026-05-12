#!/usr/bin/env python3
"""E4 (in-water variant) — TGC extraction from the polar arrays.

Math
----
In a wire-free, ringdown-free, unsaturated, non-rejected region of pure
water, the displayed pixel value as a function of depth is

    pixel(r) = log_multiplier * log10(|noise(r) * g_tgc(r)|) + offset
             = log_multiplier * log10(|noise|)
               + log_multiplier * log10(g_tgc(r)) + offset

In water at 10 MHz the attenuation is ~0.0022 dB/cm (orders of magnitude
below TGC), so |noise(r)| is depth-independent and any depth dependence
of pixel(r) is the device's TGC. With the standard IVUS log_multiplier
of 20:

    pixel(r) - pixel(r_ref) = tgc_db(r) - tgc_db(r_ref)

So the SHAPE of the median-over-theta palette profile in wire-free
sectors IS the TGC curve, in dB, up to the choice of reference depth.

Cross-gain consistency check
----------------------------
If TGC is gain-independent (which it should be) the cross-gain step
pixel(r, g_hi) - pixel(r, g_lo) MUST be depth-independent. We measure:
   54 - 44 step:  ~25 palette, depth-independent (good)
   64 - 54 step:  62 -> 78 palette ramp from 5 mm to 28 mm depth (BAD)

The gain-64 ramp comes from box-wall and inter-wire reverberations
that only become visible at high gain (the phantom sits in a finite
container at r ~ 50 mm). It is NOT a real TGC effect, so we exclude
gain 64 from the canonical fit and use only the gain-54 D=60 data
(n=5 frames, well above reject, well below saturation) as the
reference. Gain 44 is also discarded because the inner-mm noise is
clipped against the palette reject at 10.

Outputs
-------
  derived/tgc/per_frame_profiles.npy        float32 (n_frames, num_r)
  derived/tgc/per_gain_profiles.json        per-gain mean profile + ref-depth values
  derived/tgc/tgc_control_points.json       fitted piecewise-linear breakpoints in dB
  derived/tgc/tgc_overview.png              diagnostic plot
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

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


def median_wire_free_radial(
    fr: PolarFrame, wires, exclude_angle_deg: float, inner_radial_mm: float
) -> np.ndarray:
    """Median palette value per radial bin over wire-free angular sectors.

    Returns a length-num_r profile. Bins where everything is masked
    (e.g. inside the ringdown / reject zone) are nan.
    """
    mask = build_anechoic_mask(
        fr, wires,
        exclude_angle_deg=exclude_angle_deg,
        inner_radial_mm=inner_radial_mm,
    )
    arr = fr.arr.astype(np.float64).copy()
    arr[~mask] = np.nan
    # Per radial bin median ignoring nan:
    with np.errstate(all="ignore"):
        prof = np.nanmedian(arr, axis=0)
    return prof


def piecewise_linear_fit(
    r_mm: np.ndarray, y_db: np.ndarray, n_segments: int = 3,
    smooth_window_mm: float = 1.0,
) -> list[tuple[float, float]]:
    """Approximate y_db(r) by a piecewise-linear curve with `n_segments` segments.

    Smooths the input with a `smooth_window_mm` rolling median so that
    speckle noise in the wire-free pixels does not bias the breakpoint
    values. Then samples the smoothed curve at evenly spaced
    breakpoints over the valid depth range. Linearly extrapolates back
    to depth=0 using the first segment's slope so the simulator can use
    the points as-is without special-casing the near field.

    Returns [(depth_cm, gain_db), ...].
    """
    valid = ~np.isnan(y_db)
    r = r_mm[valid]
    y = y_db[valid]
    if len(r) < 4:
        return [(0.0, 0.0), (float(r_mm[-1]) / 10.0, 0.0)]

    if len(r) > 1:
        dr = float(r[1] - r[0])
        win = max(3, int(round(smooth_window_mm / dr)) | 1)  # odd
        half = win // 2
        y_smooth = np.array([
            float(np.median(y[max(0, i - half): i + half + 1]))
            for i in range(len(y))
        ])
    else:
        y_smooth = y

    br = np.linspace(r[0], r[-1], n_segments + 1)
    pts = []
    for b in br:
        win_mask = (r >= b - 0.5) & (r <= b + 0.5)
        if win_mask.sum() < 1:
            pts.append((b, float(np.interp(b, r, y_smooth))))
        else:
            pts.append((b, float(np.median(y_smooth[win_mask]))))

    (b0, y0), (b1, y1) = pts[0], pts[1]
    if b1 - b0 > 0:
        slope = (y1 - y0) / (b1 - b0)
        y_at_0 = y0 - slope * b0
    else:
        y_at_0 = y0
    pts.insert(0, (0.0, y_at_0))

    return [(float(b) / 10.0, float(y)) for b, y in pts]


def render_overview_png(
    fits: list[dict],
    profiles_per_gain: dict[float, np.ndarray],
    dr_per_gain: dict[float, float],
    out_path: Path,
    inner_radial_mm: float,
    log_multiplier_assumed: float,
):
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cmap = {44.0: "#1f77b4", 54.0: "#ff7f0e", 64.0: "#d62728"}
    for gain, prof in profiles_per_gain.items():
        dr = dr_per_gain[gain]
        r = np.arange(len(prof)) * dr
        c = cmap.get(gain, "#444")
        axes[0].plot(r, prof, color=c, linestyle="-",
                     label=f"g={gain:.0f}")
    axes[0].axvspan(0, inner_radial_mm, color="grey", alpha=0.2,
                    label=f"r < {inner_radial_mm} mm (ringdown excl.)")
    axes[0].set_ylabel("Median wire-free palette (0..255)")
    axes[0].set_title("E4: median-over-theta in wire-free sectors")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for fit in fits:
        gain = fit["gain_slider"]
        c = cmap.get(gain, "#444")
        r_db = np.array(fit["depth_mm"])
        y_db = np.array(fit["tgc_db"])
        axes[1].plot(r_db, y_db, color=c, alpha=0.5, linestyle="-",
                     label=f"g={gain:.0f} measured")
        ctrl = np.array(fit["tgc_control_points_db"])
        axes[1].plot(ctrl[:, 0] * 10.0, ctrl[:, 1], color=c, marker="o",
                     linestyle="--", label=f"g={gain:.0f} piecewise fit")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("Depth from device center (mm)")
    axes[1].set_ylabel(f"TGC (dB, ref={fits[0]['ref_depth_mm']:.1f} mm); "
                       f"assuming log_multiplier={log_multiplier_assumed:.1f}")
    axes[1].set_title("Inferred TGC curve per gain (should agree across gains)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
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
                   default=Path("P_035_PointScatter/derived/tgc"))
    p.add_argument("--exclude-angle-deg", type=float, default=15.0)
    p.add_argument("--inner-radial-mm", type=float, default=7.0,
                   help="Exclude r < this. Default 7 mm clears the ringdown "
                        "extent (3 mm) plus a comfortable margin so the "
                        "reference depth is not biased by ringdown tail.")
    p.add_argument("--ref-depth-mm", type=float, default=8.0,
                   help="Depth at which TGC is defined to be 0 dB. "
                        "Should be just past inner-radial-mm.")
    p.add_argument("--log-multiplier", type=float, default=20.0,
                   help="Assumed log_multiplier for palette->dB conversion. "
                        "20 is the IVUS default; will be re-confirmed by E7.")
    p.add_argument("--n-segments", type=int, default=3)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_polar_dataset(args.polar_dir, args.align_csv, args.meta_csv)
    wires, _ = parse_scattering_box_dxf(args.dxf)

    # Per-frame wire-free median radial profiles
    profiles: list[np.ndarray] = []
    drs: list[float] = []
    metas: list[tuple[str, float, float, float]] = []
    L = 0
    for fr in frames:
        prof = median_wire_free_radial(fr, wires, args.exclude_angle_deg,
                                       args.inner_radial_mm)
        profiles.append(prof)
        drs.append(fr.dr_mm)
        metas.append((fr.file, fr.gain_slider, fr.diameter_mm, fr.dr_mm))
        L = max(L, len(prof))

    pad = np.full((len(profiles), L), np.nan, dtype=np.float32)
    for i, pr in enumerate(profiles):
        pad[i, :len(pr)] = pr
    np.save(args.out_dir / "per_frame_profiles.npy", pad)

    # Pool by gain. Different diameters have different dr_mm; for the TGC fit
    # we pool only frames with the SAME dr_mm, so the pooled profile is in
    # consistent depth bins. Use D=60 mm as canonical (most frames per gain).
    by_gain_diam: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    by_gain_diam_dr: dict[tuple[float, float], float] = {}
    for prof, (name, g, d, dr) in zip(profiles, metas):
        by_gain_diam[(g, d)].append(prof)
        by_gain_diam_dr[(g, d)] = dr

    # We want one TGC per gain, fit on D=60 mm where we have the most data.
    profiles_per_gain: dict[float, np.ndarray] = {}
    dr_per_gain: dict[float, float] = {}
    for (g, d), parts in by_gain_diam.items():
        if d != 60.0:
            continue
        Lg = min(len(p) for p in parts)
        stacked = np.stack([p[:Lg] for p in parts], axis=0)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stacked, axis=0)
        profiles_per_gain[g] = mean
        dr_per_gain[g] = by_gain_diam_dr[(g, d)]

    # Convert to TGC in dB relative to ref_depth, then fit piecewise linear.
    fits = []
    for gain in sorted(profiles_per_gain):
        mean = profiles_per_gain[gain]
        dr = dr_per_gain[gain]
        r_mm = np.arange(len(mean)) * dr
        # Mask below inner_radial_mm and any nan pixels.
        valid = (r_mm >= args.inner_radial_mm) & ~np.isnan(mean)
        if not valid.any():
            continue
        # palette -> dB via assumed log_multiplier (1 palette ~ 1 dB if mult=20)
        y_db_raw = (mean - np.nanmin(mean[valid])) * (20.0 / args.log_multiplier)
        # Reference: median palette value in +-0.5 mm window around ref_depth_mm.
        ref_window = (r_mm >= args.ref_depth_mm - 0.5) & (r_mm <= args.ref_depth_mm + 0.5)
        ref_palette = float(np.nanmedian(mean[ref_window])) if ref_window.any() else float(mean[valid][0])
        y_db = (mean - ref_palette) * (20.0 / args.log_multiplier)

        ctrl_points = piecewise_linear_fit(r_mm[valid], y_db[valid],
                                           n_segments=args.n_segments)
        fits.append({
            "gain_slider": gain,
            "diameter_mm": 60.0,
            "ref_depth_mm": args.ref_depth_mm,
            "ref_palette": ref_palette,
            "depth_mm": r_mm[valid].tolist(),
            "tgc_db": y_db[valid].tolist(),
            "tgc_control_points_db": ctrl_points,  # [(depth_cm, gain_db), ...]
        })

    # Canonical control points: gain 54 D=60 ONLY.
    # Cross-gain check (run separately) showed:
    #   * gain 44 is reject-clipped in the inner band -> biased high
    #   * gain 64 sees box-wall/wire reverberations that ramp the deep
    #     end -> biased high at depth
    # Gain 54 is well above reject and well below saturation across
    # the whole depth range, so it is the only trustworthy measurement.
    canonical = None
    canonical_source = None
    for f in fits:
        if f["gain_slider"] == 54.0:
            canonical = f["tgc_control_points_db"]
            canonical_source = "gain_slider=54, diameter_mm=60 mm"
            break
    if canonical is None and fits:
        canonical = fits[0]["tgc_control_points_db"]
        canonical_source = f"gain_slider={fits[0]['gain_slider']}, diameter_mm=60 mm"

    out_json = {
        "method": __doc__.strip().splitlines()[0],
        "ref_depth_mm": args.ref_depth_mm,
        "log_multiplier_assumed": args.log_multiplier,
        "inner_radial_mm": args.inner_radial_mm,
        "exclude_angle_deg": args.exclude_angle_deg,
        "per_gain_fits": fits,
        "canonical_tgc_control_points_db": canonical,
        "canonical_source": canonical_source,
        "uncertainty_db": 3.0,
        "notes": [
            "Canonical fit uses gain 54 D=60 mm only; gain 44 is reject-clipped, "
            "gain 64 is contaminated by box-wall reverberations.",
            "Assumes log_multiplier = 20 (1 palette = 1 dB). E7 will refine.",
            "TGC was tuned for tissue, not water. In water the curve is nearly "
            "flat with a small ramp (~3-5 dB/cm in the outer half). For tissue "
            "imaging the TGC should be re-fit from a tissue-mimicking phantom (E4 in tissue).",
        ],
    }
    with (args.out_dir / "tgc_control_points.json").open("w") as f:
        json.dump(out_json, f, indent=2)

    if plt is not None:
        render_overview_png(fits, profiles_per_gain, dr_per_gain,
                            args.out_dir / "tgc_overview.png",
                            args.inner_radial_mm, args.log_multiplier)

    # Console summary
    print(f"{'gain':>4s} {'D_mm':>5s} {'ref_pal':>8s}  TGC control points (depth_cm, dB)")
    for f in fits:
        cp = f["tgc_control_points_db"]
        print(f"{f['gain_slider']:>4.0f} {f['diameter_mm']:>5.0f} "
              f"{f['ref_palette']:>8.2f}  "
              + " ".join(f"({d:.2f}, {db:+.2f})" for d, db in cp))
    print(f"\nCanonical TGC ({canonical_source}):")
    if canonical:
        for d_cm, db in canonical:
            print(f"  ({d_cm:.2f} cm, {db:+.2f} dB)")
    print(f"\nWrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
