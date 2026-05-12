#!/usr/bin/env python3
"""E7 N-point gain curve fit — replaces the 3-point cross-gain inference.

Designed to run on a `<dataset>/raw/<sweep>` folder containing FILE####.dcm
symlinks for an N-step gain sweep at a single (apparatus, diameter) point;
i.e. the layout produced by `scripts/stage_ivus_test_0508.py`.

Two anchor modes are supported:

  --anchor=global  (default; works without annotation/polar arrays)
        For each frame, take the 99.5th-percentile palette over the entire
        polar-disc area (excluding the inner ringdown band r < 0.12 R and
        outside the disc). This tracks the BRIGHTEST scatterer in the
        sweep — at low gain, the inner wires; at higher gain, those wires
        saturate and the next-brightest dominates. The shape of
        anchor_palette(slider) is therefore a faithful image of the
        device's compression curve up to the point of saturation; once
        the brightest pixel hits 239 the curve plateaus by construction.

  --anchor=wire    (requires polar/, alignment_fit.csv, and a chosen wire
        index; gives the cleanest per-wire fit. Implementation pending —
        produces a `--anchor=wire is not yet implemented` error and points
        the user at the polar pipeline.)

Both modes feed the same downstream fit:

  1. Reject sliders where anchor < REJECT_PALETTE + 1 (noise-only) and
     sliders where anchor >= SATURATION_PALETTE - 1 (saturated).
  2. Fit a monotonic cubic spline to (slider, anchor) on the unsaturated
     range. PCHIP — preserves monotonicity, doesn't oscillate.
  3. From the fitted curve, compute:
        - local d(anchor)/d(slider) -> per-slider effective log_multiplier
          (the device's compression slope, in palette per slider step;
          if 1 step = 1 dB then this equals log_multiplier / 20).
        - inverted curve: anchor -> slider -> dB shift, gives the
          compression LUT (palette -> envelope_amp_dB at the reference
          gain).
        - dynamic range: largest measured anchor - REJECT_PALETTE,
          converted via the local log_multiplier.
        - reject / saturation palettes: lowest anchor at noise-only
          sliders (= reject_palette) and highest anchor at saturated
          sliders (= saturation_palette).

Output:

  <out>/gain_curve.json
      Full fit details (per-slider anchor, splined curve, compression
      LUT, derived parameters, fit residuals).
  <out>/gain_curve.csv
      Tidy table (gain_slider, anchor_palette, used_in_fit, fitted_anchor,
      effective_log_multiplier, db_shift_from_reference).
  <out>/gain_curve.png
      Plot: anchor_palette vs slider, fit curve, slope (local
      log_multiplier) vs slider on a twin axis.

This is the *amplitude-only* part of E7. The pulse-shape, focal-zone,
and per-wire physics still come from the polar pipeline (extract_psf.py
etc.); when those land for the new dataset the per-wire anchor mode
will replace this one's --anchor=global fit.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pydicom

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
    load_polar_dataset,
    predict_wire_polar_positions,
)
from extract_calibration_inputs import parse_scattering_box_dxf, Wire  # noqa: E402

REJECT_PALETTE = 11.0
SATURATION_PALETTE = 239.0

# Wire-anchor search window. The s5 lateral PSF at the focal zone is
# ~1-2 deg wide (3-4 SA bins out of 1024); axial PSF is ~0.3 mm wide
# (2-3 r-bins). We use generous windows so that mild apparatus-vs-fit
# residuals (RMS ~ 0.5 mm) don't slip the peak out of view.
WIRE_WINDOW_THETA_DEG = 8.0
WIRE_WINDOW_R_MM = 2.0
# Reject in-field design wires whose nominal r is too close to ring-down
# or too close to the FOV edge -- their peak will be confounded.
WIRE_MIN_R_MM = 3.5
WIRE_FOV_MARGIN_MM = 2.0


@dataclass
class FrameRow:
    file: str
    gain_slider: float
    diameter_mm: float
    ar_state: int
    anchor_palette: float
    rows: int
    cols: int
    # Per-wire details when --anchor=wire; empty list otherwise.
    wire_peaks: list[dict] | None = None
    n_wires_used: int = 0
    theta0_deg: float | None = None


def read_anchor(path: Path) -> tuple[float, dict]:
    """Compute the global 99.5th-percentile palette outside the inner
    ringdown disc and outside the sector corners. Returns (anchor, meta)."""
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr.mean(axis=-1)
    # Use the middle frame for stability.
    f0 = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr
    H, W = f0.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)
    rmax = min(H, W) / 2.0
    mask = (r >= 0.12 * rmax) & (r <= rmax)
    pix = f0[mask]
    anchor = float(np.percentile(pix, 99.5))
    meta = {
        "rows": int(H),
        "cols": int(W),
        "gain_slider": float(ds.get((0x0029, 0x1001)).value
                              if (0x0029, 0x1001) in ds else float("nan")),
        "diameter_mm": float(ds.get((0x0029, 0x1003)).value
                              if (0x0029, 0x1003) in ds else float("nan")),
        "ar_state": int(ds.get((0x0029, 0x1007)).value
                          if (0x0029, 0x1007) in ds else 0),
    }
    return anchor, meta


def wire_anchor_for_frame(
    frame: PolarFrame, wires: list[Wire]
) -> tuple[float, list[dict], int]:
    """Compute the per-frame wire anchor: median peak palette across
    visible design wires.

    For each design wire whose predicted polar position falls inside
    [WIRE_MIN_R_MM, depth_mm - WIRE_FOV_MARGIN_MM], we crop a
    (WIRE_WINDOW_THETA_DEG x WIRE_WINDOW_R_MM) patch around the
    prediction in the frame's polar array and take its max as the
    per-wire peak. Returns (median_peak, per_wire_records, n_used).

    The median-across-wires absorbs the SA-orientation brightness
    variability discussed in calibration_delta -- the same physical
    catheter at the same gain produces somewhat different peaks for
    wires at different design theta because the SA element direction
    rotates with respect to each wire's range vector.
    """
    arr = frame.arr
    num_theta = frame.num_theta
    num_r = frame.num_r
    dr = frame.dr_mm
    dtheta = 2.0 * math.pi / num_theta

    half_t = max(1, int(round(math.radians(WIRE_WINDOW_THETA_DEG) / dtheta)))
    half_r = max(1, int(round(WIRE_WINDOW_R_MM / dr)))

    preds = predict_wire_polar_positions(frame, wires)
    records: list[dict] = []
    peaks: list[float] = []
    r_outer = frame.depth_mm - WIRE_FOV_MARGIN_MM

    for wi, theta_d, r_w in preds:
        if r_w < WIRE_MIN_R_MM or r_w > r_outer:
            continue
        i_theta = int(round(theta_d / dtheta)) % num_theta
        i_r = int(round(r_w / dr))
        if i_r < 0 or i_r >= num_r:
            continue
        i_lo = i_theta - half_t
        i_hi = i_theta + half_t + 1
        if i_lo >= 0 and i_hi <= num_theta:
            patch = arr[i_lo:i_hi, max(0, i_r - half_r):min(num_r, i_r + half_r + 1)]
        else:
            idx = [(i_lo + k) % num_theta for k in range(i_hi - i_lo)]
            patch = arr[np.array(idx), :][:, max(0, i_r - half_r):min(num_r, i_r + half_r + 1)]
        peak = float(np.nanmax(patch)) if patch.size else float("nan")
        # Track which sub-pixel inside the window had the peak.
        if patch.size and np.isfinite(peak):
            local_idx = np.unravel_index(int(np.nanargmax(patch)), patch.shape)
            peak_r_mm = (max(0, i_r - half_r) + local_idx[1] + 0.5) * dr
            peak_theta_deg = math.degrees((i_lo + local_idx[0]) * dtheta) % 360.0
        else:
            peak_r_mm = float("nan"); peak_theta_deg = float("nan")
        records.append({
            "wire_index": int(wi),
            "r_design_mm": float(r_w),
            "theta_design_deg": float(math.degrees(theta_d) % 360.0),
            "peak_palette": peak,
            "peak_r_mm": peak_r_mm,
            "peak_theta_deg": peak_theta_deg,
        })
        if np.isfinite(peak):
            peaks.append(peak)

    if not peaks:
        return float("nan"), records, 0
    return float(np.median(peaks)), records, len(peaks)


def pchip_eval(x_obs: np.ndarray, y_obs: np.ndarray, x_eval: np.ndarray):
    """Monotone-preserving Hermite cubic interpolation (PCHIP).

    Returns (y_eval, dy_eval) on x_eval. Implements the Fritsch-Carlson
    monotone-PCHIP construction so we don't pull in scipy as a hard dep.
    """
    n = len(x_obs)
    if n < 2:
        raise ValueError("need >= 2 points for PCHIP")
    h = np.diff(x_obs)
    delta = np.diff(y_obs) / h
    d = np.zeros(n)
    d[0] = delta[0]
    d[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    y_eval = np.empty_like(x_eval, dtype=float)
    dy_eval = np.empty_like(x_eval, dtype=float)
    for k, x in enumerate(x_eval):
        if x <= x_obs[0]:
            i = 0
        elif x >= x_obs[-1]:
            i = n - 2
        else:
            i = int(np.searchsorted(x_obs, x, side="right") - 1)
        t = (x - x_obs[i]) / h[i]
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t * t * (3 - 2 * t)
        h11 = t * t * (t - 1)
        y_eval[k] = (h00 * y_obs[i] + h10 * h[i] * d[i]
                     + h01 * y_obs[i + 1] + h11 * h[i] * d[i + 1])
        dh00 = 6 * t * (t - 1) / h[i]
        dh10 = (3 * t * t - 4 * t + 1)
        dh01 = 6 * t * (1 - t) / h[i]
        dh11 = (3 * t * t - 2 * t)
        dy_eval[k] = (dh00 * y_obs[i] + dh10 * d[i]
                      + dh01 * y_obs[i + 1] + dh11 * d[i + 1])
    return y_eval, dy_eval


def fit_curve(rows: list[FrameRow], reference_slider: float | None) -> dict:
    rows_sorted = sorted(rows, key=lambda r: r.gain_slider)
    sliders = np.array([r.gain_slider for r in rows_sorted])
    anchors = np.array([r.anchor_palette for r in rows_sorted])

    # Mark each point as noise-only / saturated / usable.
    noise_only = anchors <= REJECT_PALETTE + 1.0
    saturated = anchors >= SATURATION_PALETTE - 1.0
    usable = ~noise_only & ~saturated

    fit_mask = usable.copy()
    if fit_mask.sum() < 3:
        return {
            "error": "fewer than 3 usable points",
            "n_total": int(len(rows_sorted)),
            "n_noise_only": int(noise_only.sum()),
            "n_saturated": int(saturated.sum()),
            "n_usable": int(usable.sum()),
        }

    s_fit = sliders[fit_mask]
    a_fit = anchors[fit_mask]

    # PCHIP fit on the usable points, evaluated on the full slider grid.
    s_eval = sliders.astype(float)
    a_eval, da_eval = pchip_eval(s_fit, a_fit, s_eval)

    # Slope da/dslider at usable points (= log_multiplier / 20 if 1 step = 1 dB).
    slope = da_eval
    log_mult = slope * 20.0  # palette per log10(amp), assuming 1 step = 1 dB

    # Reference gain: pick the slider closest to the median usable slider
    # if not specified. Default for the s5 corpus is 54 if it exists.
    if reference_slider is None:
        if 54.0 in sliders:
            reference_slider = 54.0
        else:
            reference_slider = float(np.median(s_fit))

    # dB shift from reference: each slider step = 1 dB (working hypothesis).
    db_shift = sliders - reference_slider

    # Compression LUT: at the reference gain, what envelope_dB does each
    # palette correspond to? Build by inverting the fitted curve. We get
    # palette(slider) at the reference; assume slider step ↔ amp dB step,
    # so amp_dB_at_palette = (slider_at_palette - reference_slider).
    palette_grid = np.arange(int(REJECT_PALETTE), int(SATURATION_PALETTE) + 1)
    # Build (palette, slider) pairs from the fit on a fine slider grid.
    s_fine = np.linspace(s_fit.min(), s_fit.max(), 257)
    a_fine, _ = pchip_eval(s_fit, a_fit, s_fine)
    # Make a_fine monotone (PCHIP guarantees this if the input is, but
    # handle any tiny numerical roughness).
    order = np.argsort(a_fine)
    a_fine_sorted = a_fine[order]
    s_fine_sorted = s_fine[order]
    palette_to_slider = np.interp(palette_grid, a_fine_sorted, s_fine_sorted,
                                   left=np.nan, right=np.nan)
    palette_to_db_shift = palette_to_slider - reference_slider

    # Canonical log_multiplier: median slope across the usable interval,
    # excluding the bottom and top 10% (where the curve flattens).
    in_interior = ((sliders > np.percentile(s_fit, 10))
                    & (sliders < np.percentile(s_fit, 90))
                    & usable)
    if in_interior.sum() >= 2:
        log_mult_canonical = float(np.median(log_mult[in_interior]))
        log_mult_iqr = float(np.subtract(*np.percentile(log_mult[in_interior],
                                                         [75, 25])))
    else:
        log_mult_canonical = float(np.median(log_mult[usable]))
        log_mult_iqr = float(np.subtract(*np.percentile(log_mult[usable],
                                                         [75, 25])))

    # Reject / saturation palettes: empirical anchors.
    reject_observed = float(anchors[noise_only].max() if noise_only.any()
                              else REJECT_PALETTE)
    saturation_observed = float(anchors[saturated].min() if saturated.any()
                                  else SATURATION_PALETTE)

    # Dynamic range: top usable anchor - reject, in dB at the local slope.
    dr_palette = anchors[usable].max() - reject_observed
    log_mult_for_dr = log_mult_canonical
    dynamic_range_db = float(dr_palette * 20.0 / log_mult_for_dr) if log_mult_for_dr > 0 else float("nan")

    return {
        "reference_slider": reference_slider,
        "n_total": int(len(rows_sorted)),
        "n_noise_only": int(noise_only.sum()),
        "n_saturated": int(saturated.sum()),
        "n_usable": int(usable.sum()),
        "sliders": sliders.tolist(),
        "anchors": anchors.tolist(),
        "fit_mask": fit_mask.tolist(),
        "fit_anchor": a_eval.tolist(),
        "slope_palette_per_step": slope.tolist(),
        "effective_log_multiplier": log_mult.tolist(),
        "db_shift_from_reference": db_shift.tolist(),
        "log_multiplier_canonical": log_mult_canonical,
        "log_multiplier_iqr": log_mult_iqr,
        "reject_palette_observed": reject_observed,
        "saturation_palette_observed": saturation_observed,
        "dynamic_range_db": dynamic_range_db,
        "compression_lut_palette": palette_grid.tolist(),
        "compression_lut_db_shift_from_reference": palette_to_db_shift.tolist(),
        "compression_lut_slider_at_palette": palette_to_slider.tolist(),
    }


def write_csv(rows: list[FrameRow], fit: dict, out_csv: Path):
    import csv
    fields = [
        "file", "gain_slider", "diameter_mm", "ar_state", "anchor_palette",
        "used_in_fit", "fit_anchor", "slope_palette_per_step",
        "effective_log_multiplier", "db_shift_from_reference",
    ]
    sliders = np.array(fit["sliders"])
    out_rows = []
    for fr in sorted(rows, key=lambda r: r.gain_slider):
        idx = int(np.where(sliders == fr.gain_slider)[0][0])
        out_rows.append({
            "file": fr.file,
            "gain_slider": fr.gain_slider,
            "diameter_mm": fr.diameter_mm,
            "ar_state": fr.ar_state,
            "anchor_palette": fr.anchor_palette,
            "used_in_fit": int(fit["fit_mask"][idx]),
            "fit_anchor": fit["fit_anchor"][idx],
            "slope_palette_per_step": fit["slope_palette_per_step"][idx],
            "effective_log_multiplier": fit["effective_log_multiplier"][idx],
            "db_shift_from_reference": fit["db_shift_from_reference"][idx],
        })
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)


def write_plot(fit: dict, out_png: Path, label: str):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sliders = np.array(fit["sliders"])
    anchors = np.array(fit["anchors"])
    fit_mask = np.array(fit["fit_mask"])
    fit_anchor = np.array(fit["fit_anchor"])
    ax.scatter(sliders[fit_mask], anchors[fit_mask], color="C0", s=40,
               label="anchor (used)")
    ax.scatter(sliders[~fit_mask], anchors[~fit_mask], color="lightgray",
               s=40, label="anchor (rejected: noise/sat)")
    ax.plot(sliders, fit_anchor, "C0--", lw=1.5, label="PCHIP fit")
    ax.axhline(REJECT_PALETTE, color="red", lw=0.8, ls=":")
    ax.axhline(SATURATION_PALETTE, color="red", lw=0.8, ls=":")
    ax.set_xlabel("slider gain")
    ax.set_ylabel("anchor palette (p99.5 of disc)")
    ax.set_title(f"E7 N-point gain curve: {label}")
    ax2 = ax.twinx()
    log_mult = np.array(fit["effective_log_multiplier"])
    ax2.plot(sliders[fit_mask], log_mult[fit_mask], "C3-", lw=1.0,
             label="local log_mult (palette per dB / step)")
    ax2.set_ylabel("effective log_multiplier (palette per dB)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def collect_wire_anchored_rows(
    dataset: Path,
    polar_dir: Path,
    align_csv: Path,
    meta_csv: Path,
    dxf: Path,
) -> list[FrameRow]:
    """Return one FrameRow per clean-aligned polar frame, anchor = median
    peak across visible design wires."""
    frames = load_polar_dataset(polar_dir, align_csv, meta_csv)
    wires, _ = parse_scattering_box_dxf(dxf)
    rows: list[FrameRow] = []
    for fr in frames:
        anchor, records, n_used = wire_anchor_for_frame(fr, wires)
        rows.append(FrameRow(
            file=fr.file,
            gain_slider=float(fr.gain_slider),
            diameter_mm=float(fr.diameter_mm),
            ar_state=0,  # AR state is not in the polar manifest; all sweeps are AR-OFF
            anchor_palette=anchor,
            rows=int(fr.rows), cols=int(fr.cols),
            wire_peaks=records,
            n_wires_used=int(n_used),
            theta0_deg=math.degrees(fr.theta0_rad),
        ))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True,
                   help="Folder of FILE####.dcm sweep files (single-diameter)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Folder to write gain_curve.{json,csv,png}")
    p.add_argument("--anchor", choices=["global", "wire"], default="global",
                   help="Which anchor to use as the per-frame brightness "
                        "proxy (default: global p99.5; wire uses median "
                        "peak palette across in-field design wires)")
    p.add_argument("--polar-dir", type=Path, default=None,
                   help="(--anchor=wire) defaults to <dataset>/derived/polar")
    p.add_argument("--align-csv", type=Path, default=None,
                   help="(--anchor=wire) defaults to <dataset>/derived/alignment_fit_clean.csv")
    p.add_argument("--meta-csv", type=Path, default=None,
                   help="(--anchor=wire) defaults to <dataset>/derived/frames_meta.csv")
    p.add_argument("--dxf", type=Path, default=None,
                   help="(--anchor=wire) DXF file with the wire layout; "
                        "defaults to whichever .dxf is in <dataset>")
    p.add_argument("--reference-slider", type=float, default=None,
                   help="Reference slider for the dB-shift map (default: "
                        "54 if present, else median of usable sliders)")
    p.add_argument("--label", type=str, default=None,
                   help="Label written into the plot title; default = "
                        "dataset folder name")
    args = p.parse_args(argv)

    if args.anchor == "wire":
        polar_dir = args.polar_dir or (args.dataset / "derived" / "polar")
        align_csv = args.align_csv or (args.dataset / "derived" / "alignment_fit_clean.csv")
        if not align_csv.exists():
            align_csv = args.dataset / "derived" / "alignment_fit.csv"
        meta_csv = args.meta_csv or (args.dataset / "derived" / "frames_meta.csv")
        dxf = args.dxf
        if dxf is None:
            dxf_candidates = sorted(args.dataset.glob("*.dxf"))
            if not dxf_candidates:
                print(f"--anchor=wire: no .dxf found in {args.dataset}; pass --dxf")
                return 2
            dxf = dxf_candidates[0]
        for pth, label in [(polar_dir, "polar dir"), (align_csv, "align csv"),
                            (meta_csv, "meta csv"), (dxf, "dxf")]:
            if not pth.exists():
                print(f"--anchor=wire missing input ({label}): {pth}")
                return 2
        rows = collect_wire_anchored_rows(
            args.dataset, polar_dir, align_csv, meta_csv, dxf
        )
        if not rows:
            print(f"--anchor=wire: no polar frames in {polar_dir}")
            return 1
    else:
        files = sorted(args.dataset.glob("FILE*.dcm"))
        if not files:
            print(f"no FILE*.dcm in {args.dataset}")
            return 1

        rows = []
        for f in files:
            anchor, meta = read_anchor(f)
            rows.append(FrameRow(
                file=f.name, gain_slider=meta["gain_slider"],
                diameter_mm=meta["diameter_mm"], ar_state=meta["ar_state"],
                anchor_palette=anchor, rows=meta["rows"], cols=meta["cols"],
            ))
    fit = fit_curve(rows, args.reference_slider)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "gain_curve.json"
    out_csv = args.out_dir / "gain_curve.csv"
    out_png = args.out_dir / "gain_curve.png"

    payload = {
        "dataset": str(args.dataset),
        "anchor_method": args.anchor,
        "frames": [asdict(r) for r in rows],
        "fit": fit,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    write_csv(rows, fit, out_csv)
    write_plot(fit, out_png, label=args.label or args.dataset.name)

    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")
    if "error" not in fit:
        print(f"  log_multiplier (canonical) : {fit['log_multiplier_canonical']:.1f}  "
              f"+/- IQR {fit['log_multiplier_iqr']:.1f}")
        print(f"  reject_palette  (observed) : {fit['reject_palette_observed']:.1f}")
        print(f"  saturation_palette         : {fit['saturation_palette_observed']:.1f}")
        print(f"  dynamic_range_db           : {fit['dynamic_range_db']:.1f}")
        print(f"  n usable / total           : {fit['n_usable']} / {fit['n_total']}")
    else:
        print(f"  fit error: {fit['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
