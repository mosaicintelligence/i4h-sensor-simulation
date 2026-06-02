#!/usr/bin/env python3
"""T1-E5* — speckle / cyst statistics extraction from milk-agar-glycerin
phantom DICOMs (`ivus_test_0515/raw/e5_milk_cyst_take<N>`).

Produces:

  * per-frame envelope CoV (palette std / palette mean) over the uniform-
    texture background ROI (all gel, cyst discs excluded)
  * 2D spatial autocorrelation -> 1/e correlation lengths in
    (radial, lateral) directions
  * per-cyst anechoic contrast (mean_inside_cyst vs mean_background_same_r)
    using the manual cyst_alignment_fit.json annotations

Inputs
------
  raw cartesian DICOMs in --capture-dir, plus:
    derived/frames_meta.csv         produced by extract_metadata.py
    derived/cyst_alignment_fit.json produced by the manual annotation tool

If cyst_alignment_fit.json is absent, the script falls back to a
blind cyst detector (the original behaviour) — but cyst contrast will
not be accurate.

Outputs
-------
  derived/speckle/speckle_summary.json        per-frame + aggregated stats
  derived/speckle/speckle_overview.png        polar ROIs + autocorr panel

Methodology
-----------
* Each frame is polar-resampled around the device center (= image center
  for the s5 scanner).
* The "background" ROI is r ∈ [inner_r_mm, outer_r_mm] (default [3, 25] mm),
  all azimuths, with each annotated cyst disc excluded (disc radius + 1 mm
  safety margin).  This ROI captures the full milk-agar-glycerin gel signal.
* Cyst contrast: for each design cyst in the alignment fit, we compare
    mean palette inside the cyst disc (r_cyst ± cyst_r_mm, minus a
    0.5 mm inner margin to avoid the bright ring) vs the local background
    (same radial band, outside all cyst discs).  Reported in dB.
* Speckle CoV: computed on the background ROI in both log (palette) and
  linear-envelope domains.  Rayleigh target CoV_linear = 0.5227.
* Spatial autocorrelation: on the background polar patch, summed over
  azimuth (radial direction) and over radius (lateral direction).

Coordinate math (internal)
--------------------------
polar_resample convention (from extract_calibration_inputs.py):
  x_px = cx + r_mm * px_per_mm * cos(az)
  y_px = cy - r_mm * px_per_mm * sin(az)    ← y-down, so minus

fit_alignment convention:
  x_pred = apx + (r_d / px) * scale * cos(chir * theta_d + theta0)
  y_pred = apy - (r_d / px) * scale * sin(chir * theta_d + theta0)

For a design cyst at (r_d_mm, theta_d_deg):
  theta_img = radians(chirality * theta_d_deg + theta0_deg)
  x_cyst_px  = apparatus_cx + r_d_mm / pixel_spacing_mm * scale * cos(theta_img)
  y_cyst_px  = apparatus_cy - r_d_mm / pixel_spacing_mm * scale * sin(theta_img)

Cyst centre in mm from device centre (for mask computation):
  dx_mm = (x_cyst_px - device_cx) * pixel_spacing_mm
  dy_mm = (y_cyst_px - device_cy) * pixel_spacing_mm   [y-down in px, y-down in mm]

Polar-grid Cartesian (same y-down convention):
  dx_grid_mm = r_j * cos(az_i)
  dy_grid_mm = -r_j * sin(az_i)      ← because y_px = cy - r*sin(az)

Distance check: sqrt((dx_grid - dx_cyst)^2 + (dy_grid - dy_cyst)^2) < cyst_r
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
from extract_calibration_inputs import (  # noqa: E402
    estimate_catheter_center_px,
    load_frame,
    polar_resample,
)


# ---------------------------------------------------------------------------
# Helper: load annotated cyst positions for one frame
# ---------------------------------------------------------------------------

def load_cyst_fit(capture_dir: Path) -> dict[str, list[dict]]:
    """Load cyst_alignment_fit.json and return {filename: [cyst_info_dict, ...]}.

    Each cyst_info_dict has:
      dx_mm, dy_mm   – cyst centre offset from device centre (mm, y-down)
      diameter_mm    – design diameter (mm)
      design_index   – cyst index from the phantom design
      r_design_mm    – design radius
    """
    fit_path = capture_dir / "derived" / "cyst_alignment_fit.json"
    if not fit_path.is_file():
        return {}

    fit = json.loads(fit_path.read_text())
    design = {c["cyst_index"]: c for c in fit.get("cyst_design", [])}

    result: dict[str, list[dict]] = {}
    for frame in fit.get("frames", []):
        fname = frame["file"]
        if frame.get("unreadable"):
            result[fname] = []
            continue

        px = frame["pixel_spacing_mm"]
        apx, apy = frame["apparatus_center_px"]
        dcx, dcy = frame["device_center_px"]
        theta0 = frame["theta0_deg"]
        chirality = frame["chirality"]
        scale = frame["radial_scale"]

        cysts: list[dict] = []
        for idx in frame.get("matched_design_indices", []):
            if idx not in design:
                continue
            cd = design[idx]
            r_d = cd["r_mm"]
            theta_d = cd["theta_deg"]
            diam = cd["diameter_mm"]

            theta_img = math.radians(chirality * theta_d + theta0)
            x_cyst_px = apx + (r_d / px) * scale * math.cos(theta_img)
            y_cyst_px = apy - (r_d / px) * scale * math.sin(theta_img)

            # mm offset from device centre (y-down)
            dx_mm = (x_cyst_px - dcx) * px
            dy_mm = (y_cyst_px - dcy) * px

            cysts.append({
                "design_index": idx,
                "r_design_mm": r_d,
                "theta_design_deg": theta_d,
                "dx_mm": dx_mm,
                "dy_mm": dy_mm,
                "diameter_mm": diam,
            })
        result[fname] = cysts
    return result


# ---------------------------------------------------------------------------
# Polar-grid cyst masks
# ---------------------------------------------------------------------------

def build_cyst_masks(
    rs_mm: np.ndarray, azs_rad: np.ndarray,
    cysts: list[dict],
    excl_margin_mm: float = 1.0,
    inside_margin_mm: float = 0.5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (excl_mask, inside_masks).

    excl_mask (n_r, n_az) bool – True where any cyst disc (+ margin) lies.
    inside_masks list of (n_r, n_az) bool – True inside each cyst disc (- margin).

    Polar grid Cartesian (y-down, matching polar_resample):
      dx_grid = r * cos(az)
      dy_grid = -r * sin(az)
    """
    rg, ag = np.meshgrid(rs_mm, azs_rad, indexing="ij")  # (n_r, n_az)
    dx_grid = rg * np.cos(ag)
    dy_grid = -rg * np.sin(ag)

    excl_mask = np.zeros(rg.shape, dtype=bool)
    inside_masks: list[np.ndarray] = []
    for c in cysts:
        cyst_r = c["diameter_mm"] / 2.0
        dist2 = (dx_grid - c["dx_mm"]) ** 2 + (dy_grid - c["dy_mm"]) ** 2
        excl_mask |= dist2 < (cyst_r + excl_margin_mm) ** 2
        inside_masks.append(dist2 < (cyst_r - inside_margin_mm) ** 2)
    return excl_mask, inside_masks


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def palette_to_linear(arr_palette: np.ndarray, log_multiplier: float = 20.0) -> np.ndarray:
    """Invert canonical log scaling; result is linear envelope (normalised).

    A constant log offset (unknown display offset) becomes a multiplicative
    constant in linear space, so CoV = std/mean is invariant.
    """
    return np.power(10.0, arr_palette.astype(np.float64) / log_multiplier)


def autocorr_2d_radial_lateral(roi: np.ndarray, max_shift: int = 30) -> dict:
    """1D radial + lateral autocorrelation 1/e widths.

    `roi` is (n_r, n_az) in polar coords.
    """
    if roi.size == 0:
        return {"radial_corr_bins": float("nan"), "lateral_corr_bins": float("nan")}
    centered = roi - roi.mean()
    var = centered.var()
    if var <= 0.0:
        return {"radial_corr_bins": float("nan"), "lateral_corr_bins": float("nan")}

    n_r, n_az = roi.shape
    max_shift_r = min(max_shift, n_r - 1)
    radial = np.zeros(max_shift_r + 1)
    for k in range(max_shift_r + 1):
        a = centered[: n_r - k, :]
        b = centered[k:, :]
        radial[k] = float((a * b).mean()) / var
    max_shift_l = min(max_shift, n_az - 1)
    lateral = np.zeros(max_shift_l + 1)
    for k in range(max_shift_l + 1):
        a = centered[:, : n_az - k]
        b = centered[:, k:]
        lateral[k] = float((a * b).mean()) / var

    one_over_e = 1.0 / math.e

    def corr_width(arr: np.ndarray) -> float:
        for k in range(1, len(arr)):
            if arr[k] < one_over_e:
                a_val, b_val = arr[k - 1], arr[k]
                return float(k - 1 + (a_val - one_over_e) / max(a_val - b_val, 1e-9))
        return float(len(arr) - 1)

    return {
        "radial_corr_bins": corr_width(radial),
        "lateral_corr_bins": corr_width(lateral),
        "radial_curve": radial.tolist(),
        "lateral_curve": lateral.tolist(),
    }


# ---------------------------------------------------------------------------
# Fallback blind cyst detector (used when no alignment fit is available)
# ---------------------------------------------------------------------------

def detect_cysts(
    polar_palette: np.ndarray, rs_mm: np.ndarray, azs_rad: np.ndarray,
    bg_drop_db: float = 25.0, min_arc_deg: float = 30.0,
    r_min_mm: float = 4.0, r_max_mm: float = 25.0,
    n_cysts_max: int = 4,
) -> list[dict]:
    """Detect anechoic cyst regions by azimuthal deficit scan (fallback)."""
    n_r, n_az = polar_palette.shape
    dtheta_deg = math.degrees(azs_rad[1] - azs_rad[0]) if n_az > 1 else 1.0
    bg_per_r = np.median(polar_palette, axis=1)
    deficit = bg_per_r[:, None] - polar_palette  # (n_r, n_az), positive=darker

    r_mask = (rs_mm >= r_min_mm) & (rs_mm <= r_max_mm)
    cysts = []
    used_az = np.zeros(n_az, dtype=bool)
    for r_idx in np.where(r_mask)[0]:
        if len(cysts) >= n_cysts_max:
            break
        col = deficit[r_idx, :]
        is_dark = (col >= bg_drop_db) & (~used_az)
        if not is_dark.any():
            continue
        runs = []
        i = 0
        while i < n_az:
            if is_dark[i]:
                j = i
                while j < n_az and is_dark[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        if not runs:
            continue
        i0, i1 = max(runs, key=lambda x: x[1] - x[0])
        arc_deg = (i1 - i0) * dtheta_deg
        if arc_deg < min_arc_deg:
            continue
        center_az_deg = ((i0 + i1) / 2) * dtheta_deg
        cysts.append({
            "r_mm": float(rs_mm[r_idx]),
            "az_center_deg": float(center_az_deg % 360),
            "arc_deg": float(arc_deg),
            "contrast_db": float(deficit[r_idx, i0:i1].mean()),
        })
        used_az[i0:i1] = True
    cysts.sort(key=lambda c: c["r_mm"])
    return cysts


# ---------------------------------------------------------------------------
# Per-frame measurement
# ---------------------------------------------------------------------------

def measure_one_frame(
    frame,
    cysts: list[dict] | None = None,
    log_multiplier: float = 20.0,
    inner_r_mm: float = 3.0,
    outer_r_mm: float = 25.0,
    reject_palette: float = 11.0,
    n_radial: int = 256,
    n_az: int = 720,
    excl_margin_mm: float = 1.0,
    inside_margin_mm: float = 0.5,
) -> dict:
    """Measure speckle statistics and cyst contrast for one frame.

    Parameters
    ----------
    frame      : loaded DICOM frame object (from load_frame)
    cysts      : list of cyst dicts from load_cyst_fit (None → fallback detect)
    inner_r_mm : inner edge of background speckle ROI
    outer_r_mm : outer edge of background speckle ROI
    reject_palette : pixels <= this are the device reject floor (excluded)
    excl_margin_mm : extra margin beyond cyst radius for background exclusion
    inside_margin_mm : safety margin inset from cyst edge for inside measurement
    """
    # Use image centre as device polar origin (s5 scanner convention)
    cx = frame.cols / 2.0
    cy = frame.rows / 2.0
    px_per_mm = 1.0 / max(frame.pixel_spacing_mm, 1e-6)
    r_max_mm = (frame.cols / 2.0) * frame.pixel_spacing_mm

    polar, rs_mm, azs_rad = polar_resample(
        frame.array, cx, cy, px_per_mm, r_max_mm, n_radial, n_az,
    )
    # polar is (n_radial, n_az); rs_mm (n_radial,); azs_rad (n_az,)

    # Reject-floor mask
    above_floor = polar > reject_palette

    # Radial band for background ROI
    r_band = (rs_mm >= inner_r_mm) & (rs_mm <= outer_r_mm)  # (n_r,)
    r_band_2d = r_band[:, None]  # broadcast over az

    # --- Cyst masks ---
    if cysts is not None and len(cysts) > 0:
        excl_mask, inside_masks = build_cyst_masks(
            rs_mm, azs_rad, cysts, excl_margin_mm, inside_margin_mm,
        )
        used_annotations = True
    else:
        excl_mask = np.zeros(polar.shape, dtype=bool)
        inside_masks = []
        used_annotations = False

    # Background mask: in r-band, above floor, not in any cyst
    bg_mask = r_band_2d & above_floor & (~excl_mask)
    bg_vals = polar[bg_mask]

    palette_mean = float(bg_vals.mean()) if bg_vals.size > 0 else float("nan")
    palette_std = float(bg_vals.std()) if bg_vals.size > 0 else float("nan")
    cov_log = palette_std / max(palette_mean, 1e-6)

    linear = palette_to_linear(bg_vals, log_multiplier)
    lin_mean = float(linear.mean()) if linear.size > 0 else float("nan")
    lin_std = float(linear.std()) if linear.size > 0 else float("nan")
    cov_linear = lin_std / max(lin_mean, 1e-12)

    # Depth-normalised CoV_linear: compute per radial bin, average.
    # Only include bins where >= 50% of (non-cyst) azimuthal samples are
    # above the reject floor — otherwise the bin is floor-dominated and
    # its CoV underestimates the true speckle spread.
    per_r_cov: list[float] = []
    for r_idx in np.where(r_band)[0]:
        non_cyst_az = ~excl_mask[r_idx, :]
        n_non_cyst = non_cyst_az.sum()
        if n_non_cyst < 8:
            continue
        frac_above = (above_floor[r_idx, :] & non_cyst_az).sum() / n_non_cyst
        if frac_above < 0.5:
            continue  # floor-dominated bin — skip
        col_mask = above_floor[r_idx, :] & non_cyst_az
        vals_lin = palette_to_linear(polar[r_idx, col_mask], log_multiplier)
        if vals_lin.size >= 8:
            cov_r = float(vals_lin.std() / max(vals_lin.mean(), 1e-12))
            per_r_cov.append(cov_r)
    cov_linear_depth_norm = float(np.median(per_r_cov)) if per_r_cov else float("nan")

    # Autocorrelation on the good-signal polar patch.
    # Only use radial bins where >= 50% of (non-cyst) azimuthal samples are
    # above the reject floor; the rest would introduce NaN-dominated rows
    # that distort the autocorrelation length estimate.
    frac_above_per_r = np.zeros(len(rs_mm))
    for r_idx in range(len(rs_mm)):
        nc = (~excl_mask[r_idx, :]).sum()
        if nc > 0:
            frac_above_per_r[r_idx] = (above_floor[r_idx, :] & ~excl_mask[r_idx, :]).sum() / nc
    good_r_band = r_band & (frac_above_per_r >= 0.5)
    good_r_sel = np.where(good_r_band)[0]
    if good_r_sel.size >= 4:
        roi_patch = np.where(bg_mask, polar, np.nan)[good_r_sel[0]: good_r_sel[-1] + 1, :]
        row_means = np.nanmean(roi_patch, axis=1, keepdims=True)
        # Fill NaN (cyst-excluded azimuths) with row mean so autocorr isn't biased
        roi_patch_filled = np.where(np.isnan(roi_patch), row_means, roi_patch)
        autocorr = autocorr_2d_radial_lateral(roi_patch_filled)
    else:
        autocorr = {"radial_corr_bins": float("nan"), "lateral_corr_bins": float("nan")}

    dr_mm = float(rs_mm[1] - rs_mm[0]) if len(rs_mm) > 1 else 1.0
    dtheta_rad = float(azs_rad[1] - azs_rad[0]) if len(azs_rad) > 1 else 1.0
    # Use the good-signal band midpoint for lateral arc -> mm conversion
    if good_r_sel.size >= 2:
        r_mid_mm = float(rs_mm[[good_r_sel[0], good_r_sel[-1]]].mean())
    else:
        r_mid_mm = (inner_r_mm + outer_r_mm) / 2.0
    radial_corr_mm = autocorr["radial_corr_bins"] * dr_mm
    lateral_corr_arc_mm = autocorr["lateral_corr_bins"] * dtheta_rad * r_mid_mm

    # --- Per-cyst contrast (annotation-based) ---
    cyst_results: list[dict] = []
    if used_annotations:
        for c, in_mask in zip(cysts, inside_masks):
            # Use ALL pixels inside the cyst (including reject-floor pixels,
            # which IS the signal level in an anechoic cyst).
            inside_vals = polar[in_mask]
            # Local background: same r-band zone around the cyst, outside all cysts
            r_c = c["diameter_mm"] / 2.0
            r_inner_band = max(inner_r_mm,
                               math.sqrt(c["dx_mm"] ** 2 + c["dy_mm"] ** 2) - r_c - 2.0)
            r_outer_band = (math.sqrt(c["dx_mm"] ** 2 + c["dy_mm"] ** 2) + r_c + 2.0)
            local_r_band = (rs_mm >= r_inner_band) & (rs_mm <= r_outer_band)
            local_bg_mask = local_r_band[:, None] & above_floor & (~excl_mask)
            local_bg_vals = polar[local_bg_mask]

            if inside_vals.size < 4 or local_bg_vals.size < 4:
                contrast_db = float("nan")
            else:
                contrast_db = float(local_bg_vals.mean() - inside_vals.mean())

            cyst_results.append({
                "design_index": c["design_index"],
                "r_design_mm": c["r_design_mm"],
                "theta_design_deg": c["theta_design_deg"],
                "dx_mm": c["dx_mm"],
                "dy_mm": c["dy_mm"],
                "n_inside_pixels": int(inside_vals.size),
                "n_bg_pixels": int(local_bg_vals.size),
                "palette_inside": float(inside_vals.mean()) if inside_vals.size else float("nan"),
                "palette_bg": float(local_bg_vals.mean()) if local_bg_vals.size else float("nan"),
                "contrast_db": contrast_db,
            })
    else:
        # Fallback blind detection
        blind_cysts = detect_cysts(polar, rs_mm, azs_rad)
        for bc in blind_cysts:
            cyst_results.append({
                "design_index": None,
                "r_design_mm": bc["r_mm"],
                "contrast_db": bc["contrast_db"],
                "_blind_detect": True,
            })

    contrasts = [c["contrast_db"] for c in cyst_results
                 if not math.isnan(c.get("contrast_db", float("nan")))]
    cyst_contrast_db = float(np.median(contrasts)) if contrasts else float("nan")

    return {
        "file": frame.path.name,
        "instance_number": frame.instance_number,
        "used_cyst_annotations": used_annotations,
        "n_cysts": len(cyst_results),
        "palette_mean": palette_mean,
        "palette_std": palette_std,
        "cov_log_palette": cov_log,
        "linear_mean": lin_mean,
        "linear_std": lin_std,
        "cov_linear_envelope": cov_linear,
        "cov_linear_depth_norm": cov_linear_depth_norm,
        "radial_corr_mm": radial_corr_mm,
        "lateral_corr_arc_mm": lateral_corr_arc_mm,
        "cyst_contrast_db_median": cyst_contrast_db,
        "cyst_contrast_db": cyst_contrast_db,   # backward-compat alias
        "cysts": cyst_results,
        "_polar": polar,
        "_rs_mm": rs_mm,
        "_azs_rad": azs_rad,
        "_roi_extent": (inner_r_mm, outer_r_mm),
        "_excl_mask": excl_mask,
    }


# ---------------------------------------------------------------------------
# Overview render
# ---------------------------------------------------------------------------

def render_overview(per_file: list[dict], out_path: Path) -> None:
    if plt is None or not per_file:
        return
    n = min(6, len(per_file))
    samples = per_file[:: max(1, len(per_file) // n)][:n]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for col, m in enumerate(samples):
        polar = m["_polar"]
        rs_mm = m["_rs_mm"]
        excl = m["_excl_mask"]
        ax = axes[0, col]
        # Show polar image with cyst exclusion zones highlighted
        display = polar.copy().astype(float)
        display[excl] = 0  # zero out exclusion zones so they show clearly
        ax.imshow(display.T, aspect="auto", origin="lower",
                  extent=(rs_mm[0], rs_mm[-1], 0, 360),
                  cmap="gray", vmin=0, vmax=255)
        ax.axvline(m["_roi_extent"][0], color="C1", lw=0.7, label="ROI")
        ax.axvline(m["_roi_extent"][1], color="C1", lw=0.7)
        for c in m["cysts"]:
            if c.get("design_index") is not None:
                r_c = math.sqrt(c["dx_mm"] ** 2 + c["dy_mm"] ** 2)
                ax.scatter([r_c], [180], s=20, c="C3", marker="x")
        ax.set_title(f"{m['file']}\nCoV_lin={m['cov_linear_envelope']:.3f}\n"
                     f"cyst_contrast={m['cyst_contrast_db_median']:.1f} dB",
                     fontsize=7)
        ax.set_xlabel("r (mm)")
        ax.set_ylabel("az (deg)")

        ax = axes[1, col]
        r_sel = (rs_mm >= m["_roi_extent"][0]) & (rs_mm <= m["_roi_extent"][1])
        roi_lin = palette_to_linear(
            polar[r_sel, :][~m["_excl_mask"][r_sel, :]].ravel()
        )
        if roi_lin.size:
            ax.hist(roi_lin, bins=80, color="C0", alpha=0.7, density=True)
        ax.set_title(f"linear-envelope hist\nCoV={m['cov_linear_envelope']:.3f}",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <capture-dir>/derived/speckle")
    p.add_argument("--gain-slider", type=float, default=50.0,
                   help="Restrict to this gain step.")
    p.add_argument("--diameter-mm", type=float, default=60.0,
                   help="Restrict to this imaging diameter.")
    p.add_argument("--inner-r-mm", type=float, default=3.0,
                   help="Inner radius of background speckle ROI (mm). "
                        "Default 3.0 skips the ringdown core only.")
    p.add_argument("--outer-r-mm", type=float, default=25.0,
                   help="Outer radius of background speckle ROI (mm). "
                        "Default 25.0 captures the full gel signal extent.")
    p.add_argument("--log-multiplier", type=float, default=137.4,
                   help="Device log compression scale: palette = log_multiplier * "
                        "log10(envelope) + offset.  Default 137.4 is the calibrated "
                        "value for this scanner (volcano_s5 / p035 pipeline). "
                        "Use 20.0 only if comparing with an uncalibrated pipeline.")
    p.add_argument("--n-frames-max", type=int, default=8,
                   help="Cap the number of DICOMs processed.")
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.capture_dir / "derived" / "speckle")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = args.capture_dir / "derived" / "frames_meta.csv"
    if not meta_csv.is_file():
        print(f"missing frames_meta.csv at {meta_csv}", file=sys.stderr)
        return 2

    with meta_csv.open() as f:
        meta_rows = list(csv.DictReader(f))
    select = [r for r in meta_rows
              if abs(float(r["gain_slider"]) - args.gain_slider) < 0.5
              and abs(float(r["diameter_mm"]) - args.diameter_mm) < 0.5]
    if not select:
        print(f"no frames at gain={args.gain_slider}, "
              f"diam={args.diameter_mm} in {args.capture_dir}",
              file=sys.stderr)
        avail = sorted({(r["gain_slider"], r["diameter_mm"]) for r in meta_rows})
        print(f"  available (gain, diameter) pairs: {avail}", file=sys.stderr)
        return 3

    # Load cyst alignment fit (per-frame cyst positions from manual annotation)
    cyst_fit = load_cyst_fit(args.capture_dir)
    if cyst_fit:
        print(f"  loaded cyst_alignment_fit for {len(cyst_fit)} frames")
    else:
        print("  WARNING: no cyst_alignment_fit.json found — using blind cyst detector",
              file=sys.stderr)

    select = select[: args.n_frames_max]
    per_file = []
    for row in select:
        fname = row["file"]
        path = args.capture_dir / fname
        try:
            frame = load_frame(path)
        except Exception as exc:
            print(f"  skip {fname}: {exc}", file=sys.stderr)
            continue

        frame_cysts = cyst_fit.get(fname)  # None if not in fit (use fallback)
        m = measure_one_frame(
            frame, cysts=frame_cysts,
            log_multiplier=args.log_multiplier,
            inner_r_mm=args.inner_r_mm, outer_r_mm=args.outer_r_mm,
        )
        per_file.append(m)
        n_c = m["n_cysts"]
        contrast = m["cyst_contrast_db_median"]
        print(f"  {fname:<14} g={row['gain_slider']:>4} d={row['diameter_mm']:>4}  "
              f"CoV_log={m['cov_log_palette']:.3f}  CoV_lin={m['cov_linear_envelope']:.3f}  "
              f"r_corr={m['radial_corr_mm']:.3f} mm  lat_corr={m['lateral_corr_arc_mm']:.3f} mm  "
              f"n_cysts={n_c}  contrast={contrast:.1f} dB")

    if not per_file:
        return 4

    # Aggregate
    agg_keys = ["cov_log_palette", "cov_linear_envelope", "cov_linear_depth_norm",
                "radial_corr_mm", "lateral_corr_arc_mm", "cyst_contrast_db_median"]
    agg = {k: float(np.nanmedian([m[k] for m in per_file])) for k in agg_keys}
    agg["cyst_contrast_db"] = agg["cyst_contrast_db_median"]  # backward compat
    agg["n_frames"] = len(per_file)
    agg["gain_slider"] = args.gain_slider
    agg["diameter_mm"] = args.diameter_mm
    agg["inner_r_mm"] = args.inner_r_mm
    agg["outer_r_mm"] = args.outer_r_mm
    agg["log_multiplier"] = args.log_multiplier
    agg["used_cyst_annotations"] = bool(cyst_fit)

    out = {
        "summary": agg,
        "per_frame": [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in per_file
        ],
        "rayleigh_target_cov_linear": 0.5227,
        "notes": (
            "cov_linear_depth_norm is the depth-normalised CoV: per-radial-bin "
            "linear CoV averaged (median) across r — this is the Rayleigh-comparable "
            "metric (target 0.5227). cov_linear_envelope is the raw CoV over the "
            "full ROI depth and will be inflated by TGC slope for wide ROIs. "
            "cyst_contrast_db_median is palette_bg - palette_inside in dB "
            "(anechoic interior vs background at same depth), "
            "measured from annotated cyst positions (cyst_alignment_fit.json). "
            "Background ROI excludes cyst discs + 1 mm safety margin."
        ),
    }
    out_json = out_dir / "speckle_summary.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  wrote {out_json}")

    render_overview(per_file, out_dir / "speckle_overview.png")
    print(f"  wrote {out_dir / 'speckle_overview.png'}")

    print("\n  AGGREGATED:")
    for k, v in agg.items():
        print(f"    {k:>30s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
