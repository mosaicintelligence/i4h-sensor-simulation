"""Generate provenance / evidence figures for every bench-derived value
the tier-1 evaluation consumes.

Each function in this module produces a single composite PNG showing:

  1. The raw bench frame (Cartesian and / or polar) used to derive the
     value, with the ROI / mask / annotation overlay highlighted.
  2. The intermediate per-radius profile, histogram, autocorrelation
     curve, etc. that the value is extracted from.
  3. The final scalar(s) drawn on top of (2), so a reviewer can
     immediately verify the number.

The intent is that a reader of ``tier1_results.md`` can look at each
evidence figure and either say "yes, that number is correct" or
"wait, what happened there?" without leaving the report.

All figures are saved into ``tier1_results/figures/``.  Functions are
idempotent and skip silently if the bench input is missing locally
(e.g. when the polar arrays were gitignored on the dev machine).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:  # pragma: no cover
    plt = None
    mpatches = None


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))


def _rel(p: Path) -> str:
    """Best-effort workspace-relative path for figure annotations.

    Falls back to the absolute path if `p` is outside the workspace
    root or the comparison fails (e.g. when `p` was passed as a
    relative path).
    """
    try:
        return str(Path(p).resolve().relative_to(WORKSPACE_ROOT))
    except (ValueError, OSError):
        return str(p)

from unwrap import (  # noqa: E402
    load_alignment,
    read_dicom_array,
    polar_resample,
)


# Default polar grid for on-the-fly unwrap of bench DICOMs whose
# `derived/polar` arrays are not staged locally.  These match the
# defaults `unwrap.py` uses everywhere else in the pipeline.
DEFAULT_NUM_THETA = 720
DEFAULT_NUM_R = 320
DEFAULT_DR_MM = 0.10


@dataclass(frozen=True)
class PolarFrame:
    """Polar-unwrapped bench frame + its metadata."""
    file: str
    polar: np.ndarray   # (num_theta, num_r) uint8 palette values
    cart: np.ndarray    # raw DICOM palette image
    cx_px: float
    cy_px: float
    pixel_spacing_mm: float
    dr_mm: float
    num_theta: int
    num_r: int
    theta0_rad: float
    chirality: int
    gain_slider: float
    diameter_mm: float
    rows: int
    cols: int


def _load_align_meta(align_csv: Path, meta_csv: Path,
                     file_name: str) -> dict | None:
    """Lightweight loader that doesn't assume wire-specific columns.

    Returns the merged align/meta row for `file_name` (matched by stem
    so callers can pass either `FILE0001` or `FILE0001.dcm`).
    """
    if not align_csv.is_file() or not meta_csv.is_file():
        return None
    stem = file_name.replace(".dcm", "")
    targets = {file_name, stem, stem + ".dcm"}

    def _find(rows: list[dict]) -> dict | None:
        for row in rows:
            if row.get("file") in targets:
                return row
        return None

    with align_csv.open() as f:
        align_rows = list(csv.DictReader(f))
    with meta_csv.open() as f:
        meta_rows = list(csv.DictReader(f))
    a, m = _find(align_rows), _find(meta_rows)
    if a is None or m is None:
        return None
    return {**m, **a, "_aligned_file": a.get("file", file_name)}


def _load_bench_polar(dataset_root: Path, file_name: str,
                      num_theta: int = DEFAULT_NUM_THETA,
                      num_r: int = DEFAULT_NUM_R,
                      dr_mm: float = DEFAULT_DR_MM,
                      device_origin: bool = True,
                      align_csv: Path | None = None,
                      meta_csv: Path | None = None) -> PolarFrame | None:
    """Polar-unwrap one bench DICOM on the fly.

    `dataset_root` should contain the raw DICOMs at the top level
    (the way P_035_PointScatter and ivus_test_0515/raw/<sub>/ are laid
    out) plus a ``derived/alignment_fit.csv`` (or `cyst_alignment_fit.csv`
    for E5) + ``derived/frames_meta.csv``.
    """
    if align_csv is None:
        align_csv = dataset_root / "derived" / "alignment_fit.csv"
        if not align_csv.is_file():
            align_csv = dataset_root / "derived" / "cyst_alignment_fit.csv"
    if meta_csv is None:
        meta_csv = dataset_root / "derived" / "frames_meta.csv"
    if not align_csv.is_file() or not meta_csv.is_file():
        return None

    row = _load_align_meta(align_csv, meta_csv, file_name)
    if row is None:
        return None
    try:
        pixel_spacing_mm = float(row["pixel_spacing_mm"])
        rows_n = int(row["rows"])
        cols_n = int(row["cols"])
        apparatus_cx_px = float(row.get("apparatus_cx_px",
                                        row.get("apparatus_center_px", "0").split(";")[0]
                                        if "apparatus_center_px" in row else cols_n / 2.0))
        apparatus_cy_px = float(row.get("apparatus_cy_px",
                                        cols_n / 2.0))
        theta0_rad = math.radians(float(row["theta0_deg"]))
        chirality = int(row["chirality"])
        radial_scale = float(row.get("radial_scale", 1.0))
        gain_slider = float(row.get("gain_slider", 0.0))
        diameter_mm = float(row.get("diameter_mm", 60.0))
    except (KeyError, ValueError, TypeError):
        return None

    fname_resolved = row["_aligned_file"]
    candidates = [
        dataset_root / fname_resolved,
        dataset_root / f"{fname_resolved}.dcm",
        dataset_root / fname_resolved.replace(".dcm", ""),
    ]
    dicom_path = next((p for p in candidates if p.is_file()), None)
    if dicom_path is None:
        return None
    cart = read_dicom_array(dicom_path)
    if device_origin:
        cx_px, cy_px = cols_n / 2.0, rows_n / 2.0
    else:
        cx_px, cy_px = apparatus_cx_px, apparatus_cy_px
    polar = polar_resample(
        cart, cx_px, cy_px, pixel_spacing_mm,
        theta0_rad, chirality, num_theta, num_r, dr_mm,
    )
    return PolarFrame(
        file=fname_resolved,
        polar=polar.astype(np.float32),
        cart=cart.astype(np.float32),
        cx_px=cx_px, cy_px=cy_px,
        pixel_spacing_mm=pixel_spacing_mm,
        dr_mm=dr_mm,
        num_theta=num_theta,
        num_r=num_r,
        theta0_rad=theta0_rad,
        chirality=chirality,
        gain_slider=gain_slider,
        diameter_mm=diameter_mm,
        rows=rows_n,
        cols=cols_n,
    )


def _r_axis(fr: PolarFrame) -> np.ndarray:
    return (np.arange(fr.num_r) + 0.5) * fr.dr_mm


def _theta_axis_deg(fr: PolarFrame) -> np.ndarray:
    return (np.arange(fr.num_theta) + 0.5) * (360.0 / fr.num_theta)


def _draw_circle_on_cart(ax, cx_px, cy_px, r_mm, pixel_spacing_mm,
                         color="#00ff66", lw=1.0, ls="-",
                         label=None) -> None:
    rr_px = r_mm / pixel_spacing_mm
    circ = mpatches.Circle((cx_px, cy_px), rr_px,
                           edgecolor=color, facecolor="none",
                           linewidth=lw, linestyle=ls, label=label)
    ax.add_patch(circ)


# ---------------------------------------------------------------------------
# Ring-down: peak_palette, extent_mm, fall_off_db_per_mm
# ---------------------------------------------------------------------------

def render_ringdown_evidence(out_path: Path,
                             ringdown_fit_path: Path,
                             template_path: Path,
                             dataset_root: Path,
                             ref_frame: str = "FILE0000",
                             group_label: str = "g54_d60",
                             log_multiplier: float = 137.4) -> bool:
    """Show how `ring_down.peak`, `ring_down.extent_mm`, and the
    fall-off slope are pulled out of the bench mean A-line.

    Panels: (a) raw cartesian frame with catheter dead-zone + ringdown
    ROI overlays, (b) polar frame with the same ROI band, (c) mean
    A-line with peak / extent / linear-fit lines drawn on top so the
    reviewer can verify the extraction.
    """
    if plt is None:
        return False
    if not ringdown_fit_path.is_file() or not template_path.is_file():
        return False
    fit = json.loads(ringdown_fit_path.read_text())
    groups = {g["gain_slider"]: g for g in fit.get("groups", [])
              if g.get("diameter_mm") in (60.0, 40.0, 30.0)}
    target = next((g for g in fit.get("groups", [])
                   if f"g{int(g['gain_slider'])}_d{int(g['diameter_mm'])}" == group_label),
                  None)
    if target is None:
        # Fall back to the first group; that's better than nothing.
        target = fit["groups"][0] if fit.get("groups") else None
    if target is None:
        return False

    template = np.load(template_path).astype(np.float32)
    r_mm = (np.arange(template.size) + 0.5) * float(target.get("dr_mm", 0.12))

    fr = _load_bench_polar(dataset_root, ref_frame)
    if fr is None:
        return False

    cath_r_mm = float(fit.get("catheter_radius_mm", 1.4))
    peak_r = float(target["peak_depth_mm"])
    peak_p = float(target["peak_palette"])
    extent = float(target["extent_mm"])
    floor = float(target["speckle_floor_palette"])
    slope = float(target["palette_slope_per_mm"])
    intercept = float(target["palette_intercept_at_peak"])
    excess = peak_p - floor

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.6],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.86, bottom=0.10)

    # (a) cartesian DICOM with catheter dead-zone + ringdown ROI
    ax_cart = fig.add_subplot(gs[0, 0])
    ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, cath_r_mm,
                         fr.pixel_spacing_mm,
                         color="#ff5500", ls="--",
                         label=f"catheter dead-zone (r ≤ {cath_r_mm:.2f} mm)")
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, peak_r,
                         fr.pixel_spacing_mm,
                         color="#ffd400", ls="-",
                         label=f"ring-down peak (r = {peak_r:.2f} mm)")
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, peak_r + extent,
                         fr.pixel_spacing_mm,
                         color="#00ff66", ls="-",
                         label=f"ring-down extent (r = {peak_r + extent:.2f} mm)")
    ax_cart.set_title(f"Bench {fr.file} — Cartesian", fontsize=10)
    ax_cart.set_xticks([]); ax_cart.set_yticks([])
    ax_cart.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # (b) polar with ringdown band
    ax_polar = fig.add_subplot(gs[0, 1])
    polar_r = _r_axis(fr)
    polar_theta = _theta_axis_deg(fr)
    rmax_show = min(8.0, float(polar_r.max()))
    r_mask = polar_r <= rmax_show
    extent_box = (polar_r[r_mask][0], polar_r[r_mask][-1],
                  polar_theta[-1], polar_theta[0])
    ax_polar.imshow(fr.polar[:, r_mask], cmap="gray", aspect="auto",
                    vmin=0, vmax=255, extent=extent_box)
    ax_polar.axvline(cath_r_mm, color="#ff5500", lw=1.0, ls="--",
                     label="catheter dead-zone")
    ax_polar.axvline(peak_r, color="#ffd400", lw=1.0,
                     label="peak r")
    ax_polar.axvline(peak_r + extent, color="#00ff66", lw=1.0,
                     label="extent end r")
    ax_polar.set_xlabel("r (mm)")
    ax_polar.set_ylabel("θ (deg)")
    ax_polar.set_title("Polar (inner 8 mm) — ring-down band", fontsize=10)
    ax_polar.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # (c) mean A-line with peak / extent / linear-fit lines
    ax_a = fig.add_subplot(gs[0, 2])
    ax_a.plot(r_mm, template, color="#1f78b4", lw=1.2,
              label=f"bench template ({group_label})")
    ax_a.axvline(cath_r_mm, color="#ff5500", lw=1.0, ls="--",
                 alpha=0.8, label=f"dead-zone (r ≤ {cath_r_mm:.2f} mm)")
    ax_a.axhline(floor, color="#888888", lw=0.8, ls=":",
                 label=f"speckle floor = {floor:.1f} palette")
    ax_a.axvline(peak_r, color="#ffd400", lw=1.0,
                 label=f"peak r = {peak_r:.2f} mm")
    ax_a.scatter([peak_r], [peak_p], s=44, color="#ffd400",
                 edgecolors="black", zorder=5,
                 label=f"peak = {peak_p:.1f} palette")
    extent_thresh = floor + 0.05 * excess
    ax_a.axhline(extent_thresh, color="#00ff66", lw=0.8, ls=":",
                 label=(f"5%-of-excess threshold = "
                        f"{extent_thresh:.1f} palette"))
    ax_a.axvline(peak_r + extent, color="#00ff66", lw=1.0,
                 label=f"extent end r = {peak_r + extent:.2f} mm "
                       f"(Δr = {extent:.2f} mm)")
    # Linear-fit window (peak → extent end)
    fit_window = target.get("linear_fit_window_mm", [peak_r, peak_r + extent])
    fit_r0, fit_r1 = float(fit_window[0]), float(fit_window[1])
    fit_x = np.linspace(fit_r0, fit_r1, 64)
    fit_y = intercept - slope * (fit_x - peak_r)
    ax_a.plot(fit_x, fit_y, color="#e31a1c", lw=1.2, ls="--",
              label=f"linear fit (slope = {slope:.1f} palette/mm, "
                    f"R² = {float(target.get('linear_fit_r2', 0.0)):.3f})")
    ax_a.set_xlim(0, max(r_mm.max(), 6.0))
    ax_a.set_ylim(-5, max(peak_p + 20, 255))
    ax_a.set_xlabel("r (mm)")
    ax_a.set_ylabel("palette")
    ax_a.set_title(
        f"Ring-down extraction — bench mean A-line ({group_label})\n"
        f"peak_palette = {peak_p:.1f}  extent_mm = {extent:.2f}  "
        f"slope = {slope:.1f} pal/mm (≈ "
        f"{slope * 20.0 / log_multiplier:.2f} dB/mm)",
        fontsize=9)
    ax_a.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_a.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"Bench-derived ring-down anchors  ({target['n_frames']} frames pooled, "
        f"group `{group_label}`)\n"
        f"Source: `{_rel(ringdown_fit_path)}`",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Noise floor σ + gain alignment background
# ---------------------------------------------------------------------------

def render_noise_evidence(out_path: Path,
                          noise_stats_path: Path,
                          dataset_root: Path,
                          ref_frame: str = "FILE0000",
                          gain_label: str = "54",
                          log_multiplier: float = 137.4,
                          wire_positions_csv: Path | None = None) -> bool:
    """Show how `bench_std` / `bench_mean` / `bench_bg_palette` are
    extracted from the bench anechoic ROI.

    Panels: (a) cartesian DICOM with the wedge-exclusion + inner-radial
    + outer-radial ROI overlaid, (b) polar frame with the ROI band as a
    shaded box, (c) palette histogram of the ROI samples with the
    extracted mean / std drawn on top + the Rayleigh fit.
    """
    if plt is None:
        return False
    if not noise_stats_path.is_file():
        return False
    stats = json.loads(noise_stats_path.read_text())
    per_gain = stats.get("per_gain", {}).get(gain_label)
    if per_gain is None:
        return False
    exclude_angle_deg = float(stats.get("exclude_angle_deg", 30.0))
    inner_r_mm = float(stats.get("inner_radial_mm", 4.0))

    fr = _load_bench_polar(dataset_root, ref_frame)
    if fr is None:
        return False

    polar_r = _r_axis(fr)
    polar_theta = _theta_axis_deg(fr)
    # Recreate the ROI mask exactly the way the bench extractor builds it.
    outer_r_mm = float(fr.diameter_mm) / 2.0 * 0.85
    r_mask = (polar_r >= inner_r_mm) & (polar_r <= outer_r_mm)
    # Wedge exclusion: drop ± exclude_angle_deg/2 around every wire
    # azimuth.  Wire azimuths come from `wire_positions.csv` (design
    # frame); the polar unwrap already lives in design frame so the
    # theta values can be used directly.
    wire_centres: list[float] = []
    if wire_positions_csv is None:
        wire_positions_csv = dataset_root / "derived" / "wire_positions.csv"
    if Path(wire_positions_csv).is_file():
        with open(wire_positions_csv) as f:
            for row in csv.DictReader(f):
                try:
                    wire_centres.append(float(row["theta_deg"]) % 360.0)
                except (KeyError, ValueError):
                    continue
    if not wire_centres:
        wire_centres = [0.0, 90.0, 180.0, 270.0]
    theta_mask = np.ones(fr.num_theta, dtype=bool)
    wedge_half = exclude_angle_deg / 2.0
    for centre in wire_centres:
        theta_mask &= ~(np.abs(((polar_theta - centre + 180.0) %
                                 360.0) - 180.0) <= wedge_half)
    roi_samples = fr.polar[np.ix_(theta_mask, r_mask)].ravel()

    # Cartesian ROI mask for the left panel: build (y, x) → (r, theta) map.
    rows, cols = fr.rows, fr.cols
    yy, xx = np.mgrid[0:rows, 0:cols]
    dx = (xx - fr.cx_px) * fr.pixel_spacing_mm
    dy = (fr.cy_px - yy) * fr.pixel_spacing_mm
    r_pix = np.sqrt(dx**2 + dy**2)
    theta_pix = np.degrees(np.arctan2(dy, dx)) % 360.0
    roi_mask_cart = ((r_pix >= inner_r_mm) & (r_pix <= outer_r_mm))
    wedge_cart = np.zeros_like(roi_mask_cart)
    for centre in wire_centres:
        wedge_cart |= (np.abs(((theta_pix - centre + 180.0) %
                                360.0) - 180.0) <= wedge_half)
    roi_mask_cart &= ~wedge_cart

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.6],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.86, bottom=0.10)

    ax_cart = fig.add_subplot(gs[0, 0])
    ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*fr.cart.shape, 4), dtype=np.float32)
    overlay[roi_mask_cart] = (0.0, 1.0, 0.3, 0.30)
    ax_cart.imshow(overlay)
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, inner_r_mm,
                         fr.pixel_spacing_mm, color="#00ff66", lw=0.8)
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, outer_r_mm,
                         fr.pixel_spacing_mm, color="#00ff66", lw=0.8)
    ax_cart.set_title(f"Bench {fr.file} — ROI overlay (green)\n"
                      f"r ∈ [{inner_r_mm:.1f}, {outer_r_mm:.1f}] mm, "
                      f"wedge-excluded ±{wedge_half:.0f}°",
                      fontsize=9)
    ax_cart.set_xticks([]); ax_cart.set_yticks([])

    ax_pol = fig.add_subplot(gs[0, 1])
    extent_box = (polar_r[0], polar_r[-1], polar_theta[-1], polar_theta[0])
    ax_pol.imshow(fr.polar, cmap="gray", aspect="auto",
                  vmin=0, vmax=255, extent=extent_box)
    ax_pol.axvspan(inner_r_mm, outer_r_mm,
                   color="#00ff66", alpha=0.15, label="radial band")
    for centre in wire_centres:
        ax_pol.axhspan(centre - wedge_half, centre + wedge_half,
                       color="#ff5500", alpha=0.20)
    ax_pol.set_xlabel("r (mm)")
    ax_pol.set_ylabel("θ (deg)")
    ax_pol.set_title(f"Polar — anechoic ROI ({roi_samples.size:,} samples)",
                     fontsize=9)
    ax_pol.legend(loc="upper right", fontsize=7, framealpha=0.85)

    ax_h = fig.add_subplot(gs[0, 2])
    n_bins = 80
    bench_mean = float(per_gain["mean"])
    bench_std = float(per_gain["std"])
    p05 = float(per_gain.get("p05", np.percentile(roi_samples, 5)))
    p95 = float(per_gain.get("p95", np.percentile(roi_samples, 95)))
    ax_h.hist(roi_samples, bins=n_bins, range=(0, 200),
              color="#1f78b4", alpha=0.55, label="ROI samples (this frame)")
    ax_h.axvline(bench_mean, color="#e31a1c", lw=1.5,
                 label=f"bench mean = {bench_mean:.2f}")
    ax_h.axvspan(bench_mean - bench_std, bench_mean + bench_std,
                 color="#e31a1c", alpha=0.12,
                 label=f"±1σ = {bench_std:.2f}")
    ax_h.axvline(p05, color="#888888", lw=0.8, ls=":",
                 label=f"p05 = {p05:.1f}")
    ax_h.axvline(p95, color="#888888", lw=0.8, ls=":",
                 label=f"p95 = {p95:.1f}")
    ax_h.set_xlabel("palette value")
    ax_h.set_ylabel("count")
    pref = per_gain.get("preferred_distribution", "?")
    n_full = int(per_gain.get("n", roi_samples.size))
    n_frames_pool = int(per_gain.get("n_frames", 1))
    ax_h.set_title(
        f"Anechoic ROI palette histogram — slider {gain_label}\n"
        f"std = {bench_std:.2f}  mean = {bench_mean:.2f}  "
        f"(n_samples_full = {n_full:,} across {n_frames_pool} frames, "
        f"preferred dist = {pref})",
        fontsize=9)
    ax_h.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_h.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"Bench-derived anechoic noise stats  (gain slider `{gain_label}`)\n"
        f"Source: `{_rel(noise_stats_path)}`  —  same ROI also feeds the "
        f"gain-alignment anchor (bench_bg_palette_at_slider_54 = {bench_mean:.2f})",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Depth uniformity — per-radius mean palette + wire-mask
# ---------------------------------------------------------------------------

def render_depth_uniformity_evidence(out_path: Path,
                                     bench_polar_dir: Path,
                                     bench_frames_meta: Path,
                                     dataset_root: Path | None = None,
                                     ref_frame: str = "FILE0000",
                                     ref_gain: float = 54.0,
                                     ref_diameter_mm: float = 60.0) -> bool:
    """Reproduce the wire-mask + per-radius-mean profile used in Test I.

    The bench depth profile is computed as the column-wise (per-r)
    p70-clipped mean across the bench polar stack — this masks out the
    bright wire columns so the remaining signal is the device's
    water-scatter / ring-down floor.  This figure shows the mask on
    one example frame plus the resulting per-radius mean curve.
    """
    if plt is None:
        return False
    fr = None
    polar_stack = None
    n_frames_used = 0
    # Path A: load pre-staged polar arrays.
    if bench_polar_dir.is_dir():
        with open(bench_frames_meta) as f:
            meta = list(csv.DictReader(f))
        clean = [m for m in meta
                 if abs(float(m["gain_slider"]) - ref_gain) < 1e-3
                 and abs(float(m["diameter_mm"]) - ref_diameter_mm) < 1e-3]
        if not clean:
            return False
        arrs = []
        for m in clean:
            p = bench_polar_dir / f"{m['file']}.npy"
            if p.is_file():
                arrs.append(np.load(p).astype(np.float32))
        if not arrs:
            return False
        polar_stack = np.stack(arrs, axis=0)
        polar_demo = polar_stack[0]
        dr_mm = float(clean[0].get("pixel_spacing_mm", DEFAULT_DR_MM))
        n_frames_used = len(arrs)
    elif dataset_root is not None:
        # On-the-fly path.  We unwrap every clean frame in frames_meta.csv
        # at gain × diameter == (ref_gain, ref_diameter_mm) so the per-r p70
        # mask gets a proper population to threshold against.
        with open(bench_frames_meta) as f:
            meta = list(csv.DictReader(f))
        clean = [m for m in meta
                 if abs(float(m["gain_slider"]) - ref_gain) < 1e-3
                 and abs(float(m["diameter_mm"]) - ref_diameter_mm) < 1e-3]
        if not clean:
            return False
        arrs = []
        for m in clean:
            f1 = _load_bench_polar(dataset_root, m["file"])
            if f1 is not None:
                arrs.append(f1.polar)
        if not arrs:
            return False
        polar_stack = np.stack(arrs, axis=0)
        polar_demo = polar_stack[0]
        # Pull a representative metadata frame for the demo overlay.
        fr = _load_bench_polar(dataset_root, ref_frame)
        if fr is None:
            fr = _load_bench_polar(dataset_root, clean[0]["file"])
        dr_mm = fr.dr_mm if fr is not None else DEFAULT_DR_MM
        n_frames_used = len(arrs)
    else:
        return False

    # Wire mask = per-r p70 across the (stack × theta) population.
    # We treat anything above that threshold as a wire pixel.
    flat_per_r = polar_stack.reshape(-1, polar_stack.shape[-1])
    p70 = np.percentile(flat_per_r, 70.0, axis=0)
    p70 = np.maximum.accumulate(p70[::-1])[::-1]  # monotone smoothing
    mask = polar_demo > p70[None, :]
    masked = polar_stack.copy()
    masked[masked > p70[None, None, :]] = np.nan
    per_r_mean = np.nanmean(masked, axis=(0, 1))
    per_r_raw = polar_stack.mean(axis=(0, 1))

    n_r = polar_stack.shape[-1]
    r_axis = (np.arange(n_r) + 0.5) * (fr.dr_mm if fr is not None else dr_mm)
    theta_axis_demo = np.linspace(0.0, 360.0, polar_demo.shape[0])

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.86, bottom=0.10)

    if fr is not None:
        ax_cart = fig.add_subplot(gs[0, 0])
        ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
        ax_cart.set_title(f"Bench {ref_frame} — cartesian",
                          fontsize=9)
        ax_cart.set_xticks([]); ax_cart.set_yticks([])
    else:
        ax_cart = fig.add_subplot(gs[0, 0])
        ax_cart.imshow(polar_demo, cmap="gray", aspect="auto",
                       vmin=0, vmax=255,
                       extent=(r_axis[0], r_axis[-1], 360.0, 0.0))
        ax_cart.set_title("Bench polar (first clean frame)",
                          fontsize=9)
        ax_cart.set_xlabel("r (mm)"); ax_cart.set_ylabel("θ (deg)")

    ax_mask = fig.add_subplot(gs[0, 1])
    ax_mask.imshow(polar_demo, cmap="gray", aspect="auto",
                   vmin=0, vmax=255,
                   extent=(r_axis[0], r_axis[-1], theta_axis_demo[-1], 0.0))
    # overlay wire mask in red
    wire_overlay = np.zeros((*polar_demo.shape, 4), dtype=np.float32)
    wire_overlay[mask] = (1.0, 0.2, 0.2, 0.50)
    ax_mask.imshow(wire_overlay, aspect="auto",
                   extent=(r_axis[0], r_axis[-1], theta_axis_demo[-1], 0.0))
    ax_mask.set_xlabel("r (mm)")
    ax_mask.set_ylabel("θ (deg)")
    ax_mask.set_title(
        "Polar + per-r p70 wire-exclusion mask\n(red overlay = excluded as wire)",
        fontsize=9)

    ax_curve = fig.add_subplot(gs[0, 2])
    ax_curve.plot(r_axis, per_r_raw, color="#888888", lw=0.9,
                  label="raw per-r mean (with wires)")
    ax_curve.plot(r_axis, per_r_mean, color="#1f78b4", lw=1.4,
                  label="wire-masked per-r mean (Test I anchor)")
    ax_curve.plot(r_axis, p70, color="#e31a1c", lw=0.8, ls="--",
                  label="per-r p70 threshold")
    ax_curve.set_xlabel("r (mm)")
    ax_curve.set_ylabel("palette")
    ax_curve.set_title(
        f"Per-radius mean palette\n"
        f"({n_frames_used} frames, gain {ref_gain:g} / D = "
        f"{ref_diameter_mm:g} mm)  —  wire-masked curve = Test I anchor",
        fontsize=9)
    ax_curve.set_xlim(0, r_axis[-1])
    ax_curve.set_ylim(0, max(60, per_r_raw.max() * 1.05))
    ax_curve.grid(True, lw=0.3, alpha=0.5)
    ax_curve.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.suptitle(
        "Bench-derived depth-uniformity anchor  (Test I)\n"
        "How the per-radius mean palette is wire-masked (per-r p70 threshold) "
        "from the gain-54 / D=60 polar stack.",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# E5 speckle anchor (CoV_log, autocorr, cyst contrast)
# ---------------------------------------------------------------------------

def render_speckle_evidence(out_path: Path,
                            dataset_root: Path,
                            speckle_summary_paths: list[Path],
                            ref_frame: str = "FILE0001",
                            log_multiplier: float = 137.4,
                            align_csv: Path | None = None,
                            meta_csv: Path | None = None) -> bool:
    """Show how `cov_log_palette`, `cov_linear_depth_norm`, the
    radial / lateral correlation lengths, and the cyst contrast are
    derived from the Wave 0 E5 captures.

    Panels: (a) cartesian E5 frame with the anechoic ROI band +
    annotated cyst discs overlaid, (b) polar with the ROI box, (c)
    per-frame summary table (medians across the 3 takes) and a small
    histogram of the ROI palette samples.
    """
    if plt is None or mpatches is None:
        return False
    if not speckle_summary_paths:
        return False
    # Pool the summaries (one JSON per take).
    summaries = []
    for p in speckle_summary_paths:
        if p.is_file():
            summaries.append(json.loads(p.read_text())["summary"])
    if not summaries:
        return False
    medians = {
        k: float(np.median([s[k] for s in summaries if k in s and s[k] is not None]))
        for k in (
            "cov_log_palette", "cov_linear_envelope",
            "cov_linear_depth_norm", "radial_corr_mm",
            "lateral_corr_arc_mm", "cyst_contrast_db",
        )
    }
    rayleigh_target = float(json.loads(
        speckle_summary_paths[0].read_text()).get("rayleigh_target_cov_linear", 0.523))

    fr = _load_bench_polar(dataset_root, ref_frame,
                           align_csv=align_csv, meta_csv=meta_csv)
    if fr is None:
        return False

    inner_r_mm = float(summaries[0].get("inner_r_mm", 3.0))
    outer_r_mm = float(summaries[0].get("outer_r_mm", 25.0))
    polar_r = _r_axis(fr)
    polar_theta = _theta_axis_deg(fr)
    r_mask = (polar_r >= inner_r_mm) & (polar_r <= outer_r_mm)

    # Cyst annotations: load the design + the aligned (apparatus) frame
    # for this DICOM.  We draw the design discs in the apparatus frame
    # (i.e. the post-alignment positions); they are what the cyst-aware
    # ROI exclusion uses.
    cyst_fit_path = dataset_root / "derived" / "cyst_alignment_fit.json"
    cyst_discs: list[tuple[float, float, float]] = []  # (x_mm, y_mm, diam_mm)
    if cyst_fit_path.is_file():
        cf = json.loads(cyst_fit_path.read_text())
        design = cf.get("cyst_design", [])
        frame_entry = next((f for f in cf.get("frames", [])
                            if f.get("file") == ref_frame + ".dcm"
                            or f.get("file") == ref_frame), None)
        if frame_entry is not None:
            theta0 = math.radians(frame_entry.get("theta0_deg", 0.0))
            chir = int(frame_entry.get("chirality", 1))
            radial_scale = float(frame_entry.get("radial_scale", 1.0))
            apparatus_cx_px, apparatus_cy_px = frame_entry.get(
                "apparatus_center_px", [fr.cx_px, fr.cy_px])
            apparatus_dx_mm = (apparatus_cx_px - fr.cx_px) * fr.pixel_spacing_mm
            apparatus_dy_mm = (fr.cy_px - apparatus_cy_px) * fr.pixel_spacing_mm
            for c in design:
                r_mm_design = float(c["r_mm"]) * radial_scale
                theta_deg_design = float(c["theta_deg"])
                a = theta0 + chir * math.radians(theta_deg_design)
                x_mm = apparatus_dx_mm + r_mm_design * math.cos(a)
                y_mm = apparatus_dy_mm + r_mm_design * math.sin(a)
                cyst_discs.append((x_mm, y_mm, float(c["diameter_mm"])))

    # Layout: 3 panels.  (a) cart + cysts (b) polar + ROI (c) summary stat box.
    fig = plt.figure(figsize=(15, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.86, bottom=0.10)

    ax_cart = fig.add_subplot(gs[0, 0])
    ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, inner_r_mm,
                         fr.pixel_spacing_mm, color="#00ff66", lw=0.8)
    _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, outer_r_mm,
                         fr.pixel_spacing_mm, color="#00ff66", lw=0.8)
    for (x_mm, y_mm, diam_mm) in cyst_discs:
        cx = fr.cx_px + x_mm / fr.pixel_spacing_mm
        cy = fr.cy_px - y_mm / fr.pixel_spacing_mm
        ax_cart.add_patch(mpatches.Circle(
            (cx, cy), diam_mm / 2.0 / fr.pixel_spacing_mm,
            edgecolor="#ffd400", facecolor="none", lw=1.0))
    ax_cart.set_title(
        f"Bench {fr.file}\nanechoic ROI (green) + cyst discs (yellow)",
        fontsize=9)
    ax_cart.set_xticks([]); ax_cart.set_yticks([])

    ax_pol = fig.add_subplot(gs[0, 1])
    extent_box = (polar_r[0], polar_r[-1], polar_theta[-1], polar_theta[0])
    ax_pol.imshow(fr.polar, cmap="gray", aspect="auto",
                  vmin=0, vmax=255, extent=extent_box)
    ax_pol.axvspan(inner_r_mm, outer_r_mm, color="#00ff66", alpha=0.15,
                   label=f"speckle ROI: r ∈ [{inner_r_mm:.1f}, {outer_r_mm:.1f}] mm")
    ax_pol.set_xlabel("r (mm)")
    ax_pol.set_ylabel("θ (deg)")
    ax_pol.set_title("Polar — speckle ROI band\n"
                     "(cyst discs excluded by cyst_alignment_fit.json)",
                     fontsize=9)
    ax_pol.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # Histogram of ROI palette (single frame demo).
    roi_samples = fr.polar[:, r_mask].ravel()
    ax_h = fig.add_subplot(gs[0, 2])
    ax_h.hist(roi_samples, bins=60, range=(0, 200),
              color="#1f78b4", alpha=0.55,
              label=f"ROI palette (this frame, n={roi_samples.size:,})")
    # Overlay the medians from the 3-take pool.
    txt = (
        f"Pooled across n={len(summaries)} takes:\n"
        f"  CoV_log_palette           = {medians['cov_log_palette']:.3f}\n"
        f"  CoV_linear_envelope       = {medians['cov_linear_envelope']:.3f}\n"
        f"  CoV_linear_depth_norm     = {medians['cov_linear_depth_norm']:.3f}\n"
        f"      (Rayleigh target ≈ {rayleigh_target:.3f})\n"
        f"  radial_corr_mm            = {medians['radial_corr_mm']:.3f}\n"
        f"  lateral_corr_arc_mm       = {medians['lateral_corr_arc_mm']:.3f}\n"
        f"  cyst_contrast_db (median) = {medians['cyst_contrast_db']:.2f}"
    )
    ax_h.text(0.98, 0.97, txt, transform=ax_h.transAxes,
              fontsize=8, va="top", ha="right",
              family="monospace",
              bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))
    ax_h.set_xlabel("palette value")
    ax_h.set_ylabel("count")
    ax_h.set_title("Speckle anchors — derived from the pooled E5 takes",
                   fontsize=9)
    ax_h.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"Bench-derived E5 speckle anchors  ({len(summaries)} takes pooled)\n"
        f"Source: `{_rel(speckle_summary_paths[0])}` (+ siblings) — cyst discs are "
        "the annotation-based exclusion mask used by `extract_speckle.py`.",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# TGC schedule evidence — Wave 0 E4 profiles + YAML LUT
# ---------------------------------------------------------------------------

def render_tgc_evidence(out_path: Path,
                        e4_subfolder_roots: list[Path],
                        yaml_path: Path,
                        primary_subfolder: str | None = None,
                        gains_to_plot: tuple[float, ...] = (50.0, 57.0, 68.0),
                        ref_diameter_mm: float = 60.0) -> bool:
    """Show the bench TGC anchor: per-gain median palette vs r in milk
    captures (Wave 0 E4) alongside the YAML's processing.tgc_control_points
    schedule so the report reader can see how the LUT relates to bench data.

    Left panel:  pick one representative E4 capture and overlay several
    gain slider curves to show how the post-TGC palette varies with
    gain across the imaging range.
    Right panel: YAML `processing.tgc_control_points` (depths in cm,
    dB values) — the actual sim input for Test H.
    """
    if plt is None:
        return False
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    if primary_subfolder is None:
        primary = e4_subfolder_roots[0] if e4_subfolder_roots else None
    else:
        primary = next(
            (r for r in e4_subfolder_roots if r.name == primary_subfolder),
            e4_subfolder_roots[0] if e4_subfolder_roots else None,
        )
    if primary is None:
        return False
    prof_path = primary / "derived" / "tgc" / "per_gain_profiles.json"
    if not prof_path.is_file():
        return False
    prof = json.loads(prof_path.read_text())
    rs_mm = np.asarray(prof.get("rs_mm", []), dtype=np.float32)
    per_gain_curves = prof.get("per_gain_palette_vs_r", {})
    if rs_mm.size == 0 or not per_gain_curves:
        return False

    # Build the curve list: one per requested gain.
    curves: list[dict] = []
    available_keys = sorted(per_gain_curves.keys(),
                            key=lambda k: float(k))
    for target in gains_to_plot:
        chosen_key = None
        chosen_delta = math.inf
        for k in available_keys:
            try:
                delta = abs(float(k) - float(target))
            except ValueError:
                continue
            if delta < chosen_delta:
                chosen_delta = delta
                chosen_key = k
        if chosen_key is None or chosen_delta > 1.0:
            continue
        palette = np.asarray(per_gain_curves[chosen_key], dtype=np.float32)
        if palette.size != rs_mm.size:
            continue
        curves.append({
            "gain_slider": float(chosen_key),
            "palette": palette,
        })
    if not curves:
        return False

    # YAML tgc_control_points: list of [depth_cm, gain_db] pairs.
    yaml_curve = None
    if _yaml is not None and yaml_path.is_file():
        try:
            ycfg = _yaml.safe_load(yaml_path.read_text())
            cps = ycfg["processing"]["tgc_control_points"]
            xs_cm = np.asarray([float(c[0]) for c in cps])
            ys_db = np.asarray([float(c[1]) for c in cps])
            yaml_curve = (xs_cm * 10.0, ys_db)  # cm → mm
        except (KeyError, ValueError, TypeError):
            yaml_curve = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.84,
                        bottom=0.12, wspace=0.24)
    ax_b, ax_y = axes
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(curves):
        col = cmap(i / max(len(curves) - 1, 1))
        valid = np.isfinite(c["palette"])
        ax_b.plot(rs_mm[valid], c["palette"][valid], lw=1.4, color=col,
                  label=f"slider {c['gain_slider']:.0f}")
    ax_b.set_xlabel("r (mm)")
    ax_b.set_ylabel("median palette (after-reject)")
    ax_b.set_title(
        f"Bench E4 `{primary.name}` per-gain palette vs depth\n"
        f"D = {ref_diameter_mm:g} mm. NaN ⇒ < 50 % of θ-samples > reject floor.",
        fontsize=9)
    palette_max = max(np.nanmax(c["palette"]) for c in curves
                      if np.isfinite(c["palette"]).any())
    ax_b.set_ylim(0, max(150, palette_max * 1.05))
    ax_b.set_xlim(0, float(rs_mm.max()))
    ax_b.grid(True, lw=0.3, alpha=0.5)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.85)

    if yaml_curve is not None:
        xs_mm, ys_db = yaml_curve
        ax_y.plot(xs_mm, ys_db, color="#1f78b4", lw=1.4, marker="o", ms=5,
                  label="processing.tgc_control_points\n(volcano_s5i.yaml)")
        for x, y in zip(xs_mm, ys_db):
            ax_y.annotate(f"{y:.2f} dB @ {x:.0f} mm",
                          xy=(x, y), xytext=(4, 4),
                          textcoords="offset points",
                          fontsize=7)
        ax_y.set_xlabel("r (mm)")
        ax_y.set_ylabel("TGC gain (dB)")
        ax_y.set_title("YAML TGC schedule (Test H input)", fontsize=9)
        ax_y.set_xlim(0, max(float(xs_mm.max()), 30.0))
        ax_y.set_ylim(-0.5, max(float(ys_db.max()) * 1.2, 3.0))
        ax_y.grid(True, lw=0.3, alpha=0.5)
        ax_y.legend(loc="upper left", fontsize=8, framealpha=0.85)
    else:
        ax_y.text(0.5, 0.5, "YAML TGC schedule unavailable",
                  ha="center", va="center", transform=ax_y.transAxes)
        ax_y.set_axis_off()

    fig.suptitle(
        "Bench-derived TGC evidence  —  multi-gain palette vs depth in milk + "
        "YAML schedule\n"
        "Left: bench per-gain curves show the slider step ↔ palette mapping; "
        "right: YAML control points sim renders against in Test H.",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Anechoic-anchor figures (V2): source the right protocol-correct bench
# data — ivus_test_0508/raw/c_take2_water (E6 paired AR-OFF + AR-ON
# water captures, NO wires to mask) — for ringdown / noise / depth
# uniformity.  These supersede the wire-phantom variants above.
# ---------------------------------------------------------------------------

E6_RINGDOWN_DIR = (
    WORKSPACE_ROOT
    / "ivus_test_0508"
    / "raw"
    / "c_take2_water"
    / "derived"
    / "ringdown"
)
E6_DATASET_ROOT = WORKSPACE_ROOT / "ivus_test_0508" / "raw" / "c_take2_water"


def _load_e6_summary() -> dict | None:
    p = E6_RINGDOWN_DIR / "ringdown_summary_v2.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _pick_e6_pair(summary: dict, gain: float, diameter_mm: float) -> dict | None:
    for p in summary.get("pairs", []):
        if (abs(float(p["gain"]) - gain) < 1e-3
                and abs(float(p["diameter_mm"]) - diameter_mm) < 1e-3):
            return p
    return None


def _e6_r_axis(summary: dict, n: int) -> np.ndarray:
    """Bin-edge r-axis aligned with the .npy A-lines.

    The summary JSON documents `r_grid_convention_npy = bin EDGES;
    r[i] = i * pixel_spacing_mm`, so this returns the same array.
    """
    pitch = float(summary.get("pixel_spacing_mm", 0.12))
    return np.arange(n) * pitch


def _load_e6_cart(pair: dict, summary: dict, off: bool = True) -> PolarFrame | None:
    """Best-effort: load the cartesian DICOM for an E6 pair for context.

    Falls back to a minimal PolarFrame containing only the cartesian
    array if alignment metadata is unavailable in the local repo.
    """
    fname = pair["file_off"] if off else pair["file_on"]
    cand = list(E6_DATASET_ROOT.glob(fname))
    if not cand:
        return None
    try:
        cart = read_dicom_array(cand[0]).astype(np.float32)
    except Exception:
        return None
    return PolarFrame(
        file=fname, polar=np.zeros((1, 1), np.float32),
        cart=cart, cx_px=cart.shape[1] / 2.0, cy_px=cart.shape[0] / 2.0,
        pixel_spacing_mm=float(summary["pixel_spacing_mm"]),
        dr_mm=float(summary["pixel_spacing_mm"]),
        num_theta=1, num_r=1, theta0_rad=0.0, chirality=1,
        gain_slider=float(pair["gain"]),
        diameter_mm=float(pair["diameter_mm"]),
        rows=cart.shape[0], cols=cart.shape[1],
    )


def render_ringdown_evidence_v2(out_path: Path,
                                ref_gain: float = 50.0,
                                ref_diameter_mm: float = 60.0,
                                log_multiplier: float = 137.4) -> bool:
    """Ringdown evidence sourced from the E6 anechoic capture.

    The protocol-correct ringdown anchor comes from a *water-only* AR
    paired capture (ivus_test_0508 c_take2_water), not the wire phantom.
    Plots AR-OFF, AR-ON, and AR-OFF − AR-ON residual A-lines; marks the
    catheter dead-zone, peak, extent, and floor; reads the linear
    fall-off in palette/mm and dB/mm using `log_multiplier`.
    """
    if plt is None:
        return False
    summary = _load_e6_summary()
    if summary is None:
        return False
    pair = _pick_e6_pair(summary, ref_gain, ref_diameter_mm)
    if pair is None:
        return False
    off_p = E6_RINGDOWN_DIR / Path(pair["files"]["off_npy"]).name
    on_p = E6_RINGDOWN_DIR / Path(pair["files"]["on_npy"]).name
    res_p = E6_RINGDOWN_DIR / Path(pair["files"]["residual_npy"]).name
    if not (off_p.is_file() and on_p.is_file() and res_p.is_file()):
        return False
    a_off = np.load(off_p).astype(np.float32)
    a_on = np.load(on_p).astype(np.float32)
    a_res = np.load(res_p).astype(np.float32)
    r_mm = _e6_r_axis(summary, a_off.size)

    fr = _load_e6_cart(pair, summary, off=True)
    peak_p_off = float(pair["peak_palette_off"])
    peak_p_res = float(pair["peak_palette_diff"])
    peak_r = float(pair["peak_depth_mm"])
    # `extent_mm` is the ABSOLUTE radius at which the AR residual first
    # drops below 5 % of its peak (see derive_ringdown_v2._summarize_pair).
    # The decay window we fit + display is therefore [peak_r, extent_r].
    extent_r = float(pair["extent_mm"])
    extent_delta = max(extent_r - peak_r, 0.0)
    floor_on = float(pair["speckle_floor_on"])
    cath_r_mm = 1.32

    fit_mask = (r_mm >= peak_r) & (r_mm <= extent_r) & (a_off > 0)
    if fit_mask.sum() >= 3:
        z = np.polyfit(r_mm[fit_mask], a_off[fit_mask], 1)
        slope_palette_per_mm = -float(z[0])
        intercept_at_peak = float(np.polyval(z, peak_r))
    else:
        slope_palette_per_mm = float("nan")
        intercept_at_peak = peak_p_off
    db_per_mm = (slope_palette_per_mm * 20.0 / log_multiplier
                 if math.isfinite(slope_palette_per_mm) else float("nan"))

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.6],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.84, bottom=0.10)

    ax_cart = fig.add_subplot(gs[0, 0])
    if fr is not None:
        ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, cath_r_mm,
                             fr.pixel_spacing_mm,
                             color="#ff5500", ls="--",
                             label=f"catheter dead-zone (r ≤ {cath_r_mm:.2f} mm)")
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, peak_r,
                             fr.pixel_spacing_mm,
                             color="#ffd400", ls="-",
                             label=f"ring-down peak (r = {peak_r:.2f} mm)")
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, extent_r,
                             fr.pixel_spacing_mm,
                             color="#00ff66", ls="-",
                             label=f"extent (r = {extent_r:.2f} mm)")
        ax_cart.set_title(
            f"E6 anechoic capture — {pair['file_off']} (AR-OFF)",
            fontsize=10)
        ax_cart.legend(loc="upper right", fontsize=7, framealpha=0.8)
    else:
        ax_cart.text(0.5, 0.5,
                     "(cartesian DICOM not available locally)",
                     ha="center", va="center", transform=ax_cart.transAxes)
        ax_cart.set_axis_off()
    ax_cart.set_xticks([]); ax_cart.set_yticks([])

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(r_mm, a_off, color="#1f78b4", lw=1.4,
              label=f"AR-OFF  (peak = {peak_p_off:.1f})")
    ax_b.plot(r_mm, a_on, color="#33a02c", lw=1.2,
              label=f"AR-ON   (peak = {float(a_on.max()):.1f})")
    ax_b.plot(r_mm, a_res, color="#e31a1c", lw=1.0, ls="--",
              label=f"residual = OFF − ON  (peak = {peak_p_res:.1f})")
    ax_b.axvline(cath_r_mm, color="#ff5500", lw=0.8, ls=":")
    ax_b.set_xlabel("r (mm)")
    ax_b.set_ylabel("palette")
    ax_b.set_xlim(0, 8.0)
    ax_b.set_ylim(-5, 255)
    ax_b.set_title(
        f"E6 paired A-lines (g{int(ref_gain)}, D = {ref_diameter_mm:g} mm)\n"
        "OFF = device + catheter ring-down; ON = device only (AR removes\n"
        "the catheter ring-down); residual = pure catheter ring-down.",
        fontsize=9)
    ax_b.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_b.grid(True, lw=0.3, alpha=0.5)

    ax_a = fig.add_subplot(gs[0, 2])
    ax_a.plot(r_mm, a_off, color="#1f78b4", lw=1.4,
              label=f"AR-OFF A-line (g{int(ref_gain)}, D={ref_diameter_mm:g} mm)")
    ax_a.axvline(cath_r_mm, color="#ff5500", lw=0.8, ls="--",
                 label=f"dead-zone (r ≤ {cath_r_mm:.2f} mm)")
    ax_a.axhline(floor_on, color="#888888", lw=0.8, ls=":",
                 label=f"AR-ON noise floor = {floor_on:.1f} palette")
    ax_a.axvline(peak_r, color="#ffd400", lw=1.0,
                 label=f"peak r = {peak_r:.2f} mm")
    ax_a.scatter([peak_r], [peak_p_off], s=44, color="#ffd400",
                 edgecolors="black", zorder=5,
                 label=f"peak = {peak_p_off:.1f} palette")
    excess = peak_p_off - floor_on
    extent_thresh = floor_on + 0.05 * excess
    ax_a.axhline(extent_thresh, color="#00ff66", lw=0.8, ls=":",
                 label=(f"5%-of-excess threshold = "
                        f"{extent_thresh:.1f} palette"))
    ax_a.axvline(extent_r, color="#00ff66", lw=1.0,
                 label=f"extent end r = {extent_r:.2f} mm "
                       f"(Δr from peak = {extent_delta:.2f} mm)")
    if math.isfinite(slope_palette_per_mm):
        fit_x = np.linspace(peak_r, extent_r, 64)
        fit_y = intercept_at_peak - slope_palette_per_mm * (fit_x - peak_r)
        ax_a.plot(fit_x, fit_y, color="#e31a1c", lw=1.2, ls="--",
                  label=(f"linear fit (peak→extent): slope = "
                         f"{slope_palette_per_mm:.1f} pal/mm "
                         f"≈ {db_per_mm:.1f} dB/mm"))
    ax_a.set_xlim(0, 8.0)
    ax_a.set_ylim(-5, 255)
    ax_a.set_xlabel("r (mm)")
    ax_a.set_ylabel("palette")
    ax_a.set_title(
        f"Ring-down extraction on E6 AR-OFF A-line  ({pair['file_off']})\n"
        f"peak r = {peak_r:.2f} mm  extent r = {extent_r:.2f} mm  "
        f"(Δr = {extent_delta:.2f} mm)  slope = {slope_palette_per_mm:.1f} "
        f"pal/mm ({db_per_mm:.1f} dB/mm)",
        fontsize=9)
    ax_a.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_a.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"Bench-derived ring-down anchor — E6 paired water captures  "
        f"(g{int(ref_gain)}, D = {ref_diameter_mm:g} mm)\n"
        f"Source: `{_rel(E6_RINGDOWN_DIR / 'ringdown_summary_v2.json')}` — "
        "no wires in the field, no exclusion mask required.",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def render_noise_evidence_v2(out_path: Path,
                             ref_gain: float = 50.0,
                             ref_diameter_mm: float = 60.0,
                             tail_r0_mm: float = 4.0,
                             tail_r1_mm: float = 9.84) -> bool:
    """Noise-floor evidence from the AR-ON deep-radius tail of the E6
    paired capture.

    With AR on, the catheter ring-down is removed; in water there are
    no scatterers, so deep-radius samples report the device's electronic
    noise.  Histogram, mean, and std of those samples are the
    protocol-correct noise anchor.
    """
    if plt is None:
        return False
    summary = _load_e6_summary()
    if summary is None:
        return False
    pair = _pick_e6_pair(summary, ref_gain, ref_diameter_mm)
    if pair is None:
        return False
    on_p = E6_RINGDOWN_DIR / Path(pair["files"]["on_npy"]).name
    if not on_p.is_file():
        return False
    a_on = np.load(on_p).astype(np.float32)
    r_mm = _e6_r_axis(summary, a_on.size)
    tail_mask = (r_mm >= tail_r0_mm) & (r_mm <= tail_r1_mm)
    tail_samples = a_on[tail_mask]
    mean_p = float(tail_samples.mean())
    std_p = float(tail_samples.std(ddof=0))
    p05 = float(np.percentile(tail_samples, 5))
    p95 = float(np.percentile(tail_samples, 95))

    fr = _load_e6_cart(pair, summary, off=False)

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.6],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.84, bottom=0.10)

    ax_cart = fig.add_subplot(gs[0, 0])
    if fr is not None:
        ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, tail_r0_mm,
                             fr.pixel_spacing_mm, color="#00ff66", lw=0.9)
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, tail_r1_mm,
                             fr.pixel_spacing_mm, color="#00ff66", lw=0.9)
        ax_cart.set_title(
            f"E6 anechoic AR-ON — {pair['file_on']}\n"
            f"green band = deep-radius noise ROI ({tail_r0_mm:.1f}–{tail_r1_mm:.1f} mm)",
            fontsize=9)
        ax_cart.set_xticks([]); ax_cart.set_yticks([])
    else:
        ax_cart.text(0.5, 0.5,
                     "(cartesian DICOM not available locally)",
                     ha="center", va="center", transform=ax_cart.transAxes)
        ax_cart.set_axis_off()

    ax_p = fig.add_subplot(gs[0, 1])
    ax_p.plot(r_mm, a_on, color="#33a02c", lw=1.3,
              label="AR-ON mean A-line")
    ax_p.axvspan(tail_r0_mm, tail_r1_mm, color="#00ff66", alpha=0.18,
                 label="deep-radius noise ROI")
    ax_p.axhline(mean_p, color="#e31a1c", lw=1.0,
                 label=f"mean = {mean_p:.2f}")
    ax_p.axhspan(mean_p - std_p, mean_p + std_p, color="#e31a1c",
                 alpha=0.10, label=f"±1σ = {std_p:.2f}")
    ax_p.set_xlim(0, float(r_mm.max()))
    ax_p.set_ylim(0, 100)
    ax_p.set_xlabel("r (mm)")
    ax_p.set_ylabel("palette")
    ax_p.set_title(
        "E6 AR-ON A-line — deep-radius tail dominated by\n"
        "device electronic noise (no scatterers, no ring-down).",
        fontsize=9)
    ax_p.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_p.grid(True, lw=0.3, alpha=0.5)

    ax_h = fig.add_subplot(gs[0, 2])
    ax_h.hist(tail_samples, bins=12, range=(20, 80),
              color="#1f78b4", alpha=0.55,
              label=f"deep-radius samples (n = {tail_samples.size})")
    ax_h.axvline(mean_p, color="#e31a1c", lw=1.4,
                 label=f"mean = {mean_p:.2f}")
    ax_h.axvspan(mean_p - std_p, mean_p + std_p, color="#e31a1c",
                 alpha=0.12, label=f"±1σ = {std_p:.2f}")
    ax_h.axvline(p05, color="#888888", lw=0.8, ls=":",
                 label=f"p05 = {p05:.1f}")
    ax_h.axvline(p95, color="#888888", lw=0.8, ls=":",
                 label=f"p95 = {p95:.1f}")
    ax_h.set_xlabel("palette value")
    ax_h.set_ylabel("count")
    ax_h.set_title(
        f"AR-ON noise histogram, r ∈ [{tail_r0_mm:.1f}, {tail_r1_mm:.1f}] mm\n"
        f"mean = {mean_p:.2f}  std = {std_p:.2f}  "
        f"(p05 = {p05:.1f}, p95 = {p95:.1f})",
        fontsize=9)
    ax_h.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_h.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"Bench-derived noise-floor anchor — E6 anechoic AR-ON  "
        f"(g{int(ref_gain)}, D = {ref_diameter_mm:g} mm)\n"
        f"Source: `{_rel(on_p)}` — no wires in field, no wedge-exclusion needed.",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def _load_uniform_medium_polar(dataset_root: Path,
                               file_name: str,
                               meta_row: dict,
                               num_theta: int = DEFAULT_NUM_THETA,
                               num_r: int = DEFAULT_NUM_R,
                               dr_mm: float = DEFAULT_DR_MM,
                               ) -> PolarFrame | None:
    """Polar-unwrap a uniform-medium DICOM that has no alignment fit.

    E4a milk captures don't have wires to align against, so
    `frames_meta.csv` leaves the alignment fields blank.  For
    depth-uniformity purposes the theta-zero phase doesn't matter
    (every per-r statistic is theta-averaged) so we just unwrap
    around the device centre with theta0 = 0, chirality = +1.
    """
    if not (dataset_root / file_name).is_file():
        return None
    try:
        pixel_spacing_mm = float(meta_row["pixel_spacing_mm"])
        rows_n = int(meta_row["rows"])
        cols_n = int(meta_row["cols"])
        gain_slider = float(meta_row.get("gain_slider", 0.0))
        diameter_mm = float(meta_row.get("diameter_mm", 60.0))
    except (KeyError, ValueError, TypeError):
        return None
    cart = read_dicom_array(dataset_root / file_name)
    cx_px, cy_px = cols_n / 2.0, rows_n / 2.0
    polar = polar_resample(cart, cx_px, cy_px, pixel_spacing_mm,
                           0.0, 1, num_theta, num_r, dr_mm)
    return PolarFrame(
        file=file_name, polar=polar.astype(np.float32),
        cart=cart.astype(np.float32),
        cx_px=cx_px, cy_px=cy_px,
        pixel_spacing_mm=pixel_spacing_mm,
        dr_mm=dr_mm, num_theta=num_theta, num_r=num_r,
        theta0_rad=0.0, chirality=1,
        gain_slider=gain_slider, diameter_mm=diameter_mm,
        rows=rows_n, cols=cols_n,
    )


def render_depth_uniformity_evidence_e4(out_path: Path,
                                        e4_subfolder_roots: list[Path],
                                        ref_gain: float = 68.0,
                                        ref_diameter_mm: float = 60.0,
                                        fit_r0_mm: float = 5.0,
                                        fit_r1_mm: float = 20.0,
                                        log_multiplier: float = 137.4,
                                        ) -> bool:
    """Depth-uniformity evidence from the E4a uniform-milk captures.

    Milk is a real attenuating + scattering medium, so the per-r mean
    palette in milk exercises TGC against actual tissue-like
    attenuation -- exactly what Test I is meant to validate.
    Pooling across the three E4a takes (`e4a_milk_gain_p1/p2/p3`)
    at the same (slider, diameter) gives a clean uniform-medium
    depth profile.
    """
    if plt is None:
        return False
    # Discover and pool E4a frames at the requested (gain, diameter)
    # across the three takes.
    pooled_polars = []
    primary_frame: PolarFrame | None = None
    sources: list[str] = []
    for root in e4_subfolder_roots:
        meta_path = root / "derived" / "frames_meta.csv"
        if not meta_path.is_file():
            continue
        with meta_path.open() as f:
            rows = list(csv.DictReader(f))
        match = next(
            (r for r in rows
             if abs(float(r.get("gain_slider") or 0) - ref_gain) < 0.5
             and abs(float(r.get("diameter_mm") or 0) - ref_diameter_mm) < 0.5),
            None,
        )
        if match is None:
            continue
        fr = _load_uniform_medium_polar(root, match["file"], match)
        if fr is None:
            continue
        pooled_polars.append(fr.polar)
        sources.append(f"{root.name}/{match['file']}")
        if primary_frame is None:
            primary_frame = fr
    if primary_frame is None or not pooled_polars:
        return False
    polar_stack = np.stack(pooled_polars, axis=0)
    n_takes = polar_stack.shape[0]
    r_axis = (np.arange(primary_frame.num_r) + 0.5) * primary_frame.dr_mm

    # Per-r mean + median across all (takes x theta).
    flat = polar_stack.reshape(-1, polar_stack.shape[-1])
    per_r_mean = flat.mean(axis=0)
    per_r_median = np.median(flat, axis=0)
    per_r_p10 = np.percentile(flat, 10, axis=0)
    per_r_p90 = np.percentile(flat, 90, axis=0)

    fit_mask = (r_axis >= fit_r0_mm) & (r_axis <= fit_r1_mm)
    rs_b = r_axis[fit_mask]
    p_b = per_r_mean[fit_mask]
    slope = float("nan"); intercept = float("nan")
    if fit_mask.sum() >= 3:
        z = np.polyfit(rs_b, p_b, 1)
        slope = float(z[0]); intercept = float(z[1])
    db_per_mm = (slope * 20.0 / log_multiplier
                 if math.isfinite(slope) else float("nan"))
    band_mean = float(p_b.mean())
    band_std = float(p_b.std(ddof=0))
    band_p2t = float(p_b.max() - p_b.min())

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.4],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.82, bottom=0.10)

    # (a) Cartesian E4a frame with the fit-band ROI overlaid.
    ax_cart = fig.add_subplot(gs[0, 0])
    ax_cart.imshow(primary_frame.cart, cmap="gray", vmin=0, vmax=255)
    _draw_circle_on_cart(ax_cart, primary_frame.cx_px, primary_frame.cy_px,
                         fit_r0_mm, primary_frame.pixel_spacing_mm,
                         color="#00ff66", lw=0.9)
    _draw_circle_on_cart(ax_cart, primary_frame.cx_px, primary_frame.cy_px,
                         fit_r1_mm, primary_frame.pixel_spacing_mm,
                         color="#00ff66", lw=0.9)
    ax_cart.set_title(
        f"E4a milk -- {primary_frame.file} (slider {int(ref_gain)})\n"
        f"green band = depth-uniformity fit window "
        f"({fit_r0_mm:.1f}-{fit_r1_mm:.1f} mm)",
        fontsize=9)
    ax_cart.set_xticks([]); ax_cart.set_yticks([])

    # (b) Per-r mean palette with the linear fit.
    ax_p = fig.add_subplot(gs[0, 1])
    ax_p.fill_between(r_axis, per_r_p10, per_r_p90,
                      color="#1f78b4", alpha=0.18,
                      label="per-r p10-p90 spread")
    ax_p.plot(r_axis, per_r_mean, color="#1f78b4", lw=1.4,
              label=f"per-r mean palette ({n_takes} take(s) pooled)")
    ax_p.plot(r_axis, per_r_median, color="#33a02c", lw=0.9, ls=":",
              label="per-r median palette")
    ax_p.axvspan(fit_r0_mm, fit_r1_mm, color="#00ff66", alpha=0.12)
    if math.isfinite(slope):
        fit_x = np.linspace(fit_r0_mm, fit_r1_mm, 64)
        fit_y = intercept + slope * fit_x
        ax_p.plot(fit_x, fit_y, color="#e31a1c", lw=1.2, ls="--",
                  label=f"linear fit: {slope:+.2f} pal/mm "
                        f"({db_per_mm:+.2f} dB/mm)")
    ax_p.set_xlim(0, float(r_axis.max()))
    ax_p.set_ylim(0, max(180, float(per_r_p90[np.isfinite(per_r_p90)].max()) * 1.05))
    ax_p.set_xlabel("r (mm)")
    ax_p.set_ylabel("palette")
    ax_p.set_title(
        f"E4a milk per-r palette  (slider {int(ref_gain)}, D = "
        f"{ref_diameter_mm:g} mm)\n"
        f"in-band mean = {band_mean:.1f}, std = {band_std:.1f}, "
        f"peak-to-trough = {band_p2t:.1f}",
        fontsize=9)
    ax_p.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_p.grid(True, lw=0.3, alpha=0.5)

    # (c) Summary text box.
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.set_axis_off()
    src_block = "\n".join(f"  - {s}" for s in sources)
    note = (
        "Depth-uniformity extraction (E4a milk)\n"
        "----------------------------------------\n"
        f"Gain slider     : {ref_gain:g}\n"
        f"Diameter        : {ref_diameter_mm:g} mm\n"
        f"Takes pooled    : {n_takes} ({'+'.join(s.split('/')[0] for s in sources)})\n"
        f"Fit window      : r in [{fit_r0_mm:.1f}, {fit_r1_mm:.1f}] mm\n"
        f"Band mean       : {band_mean:.1f} palette\n"
        f"Band std        : {band_std:.1f} palette\n"
        f"Peak-to-trough  : {band_p2t:.1f} palette\n"
        f"Slope           : {slope:+.2f} pal/mm\n"
        f"                  ({db_per_mm:+.2f} dB/mm)\n"
        "\n"
        "Interpretation\n"
        "----------------------------------------\n"
        "  - Milk is a real attenuating, scattering medium so\n"
        "    the per-r mean exercises TGC vs ~0.5 dB/cm of\n"
        "    speckle attenuation (the Test I target).\n"
        "  - A small NEGATIVE slope (~ -0.1 to -0.2 dB/mm)\n"
        "    means TGC slightly under-compensates milk: deeper\n"
        "    speckle samples appear a few palette darker.\n"
        "  - A POSITIVE slope > 0.2 dB/mm means TGC over-\n"
        "    compensates and would bloom deep tissue too bright.\n"
    )
    ax_txt.text(0.0, 1.0, note, transform=ax_txt.transAxes,
                fontsize=8.5, va="top", ha="left",
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

    fig.suptitle(
        f"Bench-derived depth-uniformity anchor -- E4a uniform milk "
        f"(slider {int(ref_gain)}, D = {ref_diameter_mm:g} mm)\n"
        "Pooled across the three E4a takes.  Replaces both the P_035 "
        "wire-mask extraction (wire-skirt leakage) and the E6 anechoic "
        "version (no attenuation, only noise-floor depth-invariance test).",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def render_depth_uniformity_evidence_v2(out_path: Path,
                                        ref_gain: float = 50.0,
                                        ref_diameter_mm: float = 60.0,
                                        log_multiplier: float = 137.4
                                        ) -> bool:
    """Depth-uniformity evidence from the E6 AR-ON mean A-line.

    With AR on + no scatterers, the per-radius mean palette directly
    reports how the device's TGC-conditioned electronic noise / gain
    varies with depth.  No wires to mask, no p70 trick — the
    extraction is just `palette[r] = mean across θ`.
    """
    if plt is None:
        return False
    summary = _load_e6_summary()
    if summary is None:
        return False
    pair = _pick_e6_pair(summary, ref_gain, ref_diameter_mm)
    if pair is None:
        return False
    on_p = E6_RINGDOWN_DIR / Path(pair["files"]["on_npy"]).name
    if not on_p.is_file():
        return False
    a_on = np.load(on_p).astype(np.float32)
    r_mm = _e6_r_axis(summary, a_on.size)

    fit_r0, fit_r1 = 4.0, float(r_mm.max())
    fit_mask = (r_mm >= fit_r0) & (r_mm <= fit_r1)
    slope = float("nan"); intercept = float("nan")
    if fit_mask.sum() >= 3:
        z = np.polyfit(r_mm[fit_mask], a_on[fit_mask], 1)
        slope = float(z[0])
        intercept = float(z[1])
    db_per_mm = (slope * 20.0 / log_multiplier
                 if math.isfinite(slope) else float("nan"))
    std_in_band = float(np.std(a_on[fit_mask], ddof=0)) if fit_mask.sum() else float("nan")
    mean_in_band = float(np.mean(a_on[fit_mask])) if fit_mask.sum() else float("nan")

    fr = _load_e6_cart(pair, summary, off=False)

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4],
                          wspace=0.30, left=0.04, right=0.98,
                          top=0.84, bottom=0.10)

    ax_cart = fig.add_subplot(gs[0, 0])
    if fr is not None:
        ax_cart.imshow(fr.cart, cmap="gray", vmin=0, vmax=255)
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, fit_r0,
                             fr.pixel_spacing_mm, color="#00ff66", lw=0.9)
        _draw_circle_on_cart(ax_cart, fr.cx_px, fr.cy_px, fit_r1,
                             fr.pixel_spacing_mm, color="#00ff66", lw=0.9)
        ax_cart.set_title(
            f"E6 anechoic AR-ON — {pair['file_on']}\n"
            f"green band = depth-uniformity ROI ({fit_r0:.1f}–{fit_r1:.1f} mm)",
            fontsize=9)
        ax_cart.set_xticks([]); ax_cart.set_yticks([])
    else:
        ax_cart.text(0.5, 0.5,
                     "(cartesian DICOM not available locally)",
                     ha="center", va="center", transform=ax_cart.transAxes)
        ax_cart.set_axis_off()

    ax_p = fig.add_subplot(gs[0, 1])
    ax_p.plot(r_mm, a_on, color="#33a02c", lw=1.4,
              label="AR-ON mean A-line (per-r mean)")
    ax_p.axvspan(fit_r0, fit_r1, color="#00ff66", alpha=0.15,
                 label="uniformity fit window")
    if math.isfinite(slope):
        fit_x = np.linspace(fit_r0, fit_r1, 64)
        fit_y = intercept + slope * fit_x
        ax_p.plot(fit_x, fit_y, color="#e31a1c", lw=1.1, ls="--",
                  label=f"linear fit: slope = {slope:+.2f} pal/mm "
                        f"({db_per_mm:+.2f} dB/mm)")
    ax_p.axhline(mean_in_band, color="#888888", lw=0.8, ls=":",
                 label=f"band mean = {mean_in_band:.2f}")
    ax_p.set_xlim(0, float(r_mm.max()))
    ax_p.set_ylim(0, 80)
    ax_p.set_xlabel("r (mm)")
    ax_p.set_ylabel("palette")
    ax_p.set_title(
        "Per-r mean palette  (no wire mask — anechoic capture)\n"
        f"in-band std = {std_in_band:.2f} palette",
        fontsize=9)
    ax_p.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax_p.grid(True, lw=0.3, alpha=0.5)

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.set_axis_off()
    note = (
        "Depth-uniformity extraction (E6 anechoic)\n"
        "----------------------------------------\n"
        f"Gain slider     : {ref_gain:g}\n"
        f"Diameter        : {ref_diameter_mm:g} mm\n"
        f"Fit window      : r ∈ [{fit_r0:.1f}, {fit_r1:.1f}] mm\n"
        f"Band mean       : {mean_in_band:.2f} palette\n"
        f"Band std        : {std_in_band:.2f} palette\n"
        f"Slope           : {slope:+.2f} pal/mm\n"
        f"                  ({db_per_mm:+.2f} dB/mm)\n"
        "\n"
        "Interpretation\n"
        "----------------------------------------\n"
        "  - A flat in-band curve means the device's TGC keeps\n"
        "    the noise floor depth-invariant after compression.\n"
        "  - Any persistent slope ⇒ residual TGC mismatch and\n"
        "    will look like depth-dependent gain in Test I.\n"
        "  - Replaces the wire-phantom p70-mask approach, which\n"
        "    suffered from wire-signal bleed into the masked ROI."
    )
    ax_txt.text(0.0, 1.0, note, transform=ax_txt.transAxes,
                fontsize=8.5, va="top", ha="left",
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

    fig.suptitle(
        f"Bench-derived depth-uniformity anchor — E6 anechoic AR-ON  "
        f"(g{int(ref_gain)}, D = {ref_diameter_mm:g} mm)\n"
        f"Source: `{_rel(on_p)}` — protocol-correct replacement for the "
        "P_035 wire-mask extraction.",
        fontsize=10, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# log_multiplier validation — bench gain-step + -6 dB threshold demo
# ---------------------------------------------------------------------------

def render_log_multiplier_evidence(out_path: Path,
                                   yaml_log_multiplier: float = 137.4,
                                   wire_polar_path: Path | None = None,
                                   wire_theta_idx: int = 789,
                                   wire_r_idx: int = 58,
                                   wire_window: int = 40,
                                   dr_mm: float = 0.10) -> bool:
    """Validate ``log_multiplier`` against bench gain-step data and
    demonstrate why the −6 dB FWHM threshold sits ~41 palette below
    the peak (not at half-peak) on the log-compressed display.

    Panel (a): bench gain-step calibration — fit AR-OFF ringdown peak
        palette across slider g40 → g50 from the E6 paired captures,
        slope × 20 ⇒ implied ``log_multiplier``.
    Panel (b): one real wire axial profile with the −6 dB FWHM
        threshold drawn for several ``log_multiplier`` candidates;
        only ~137 puts the threshold at the half-amplitude point.
    Panel (c): same profile back-projected to linear amplitude with
        ``yaml_log_multiplier`` — −6 dB now visually crosses at half
        peak, validating the envelope-FWHM convention.
    """
    if plt is None:
        return False
    summary = _load_e6_summary()
    if summary is None:
        return False
    pairs_d60 = sorted(
        (p for p in summary.get("pairs", []) if abs(p["diameter_mm"] - 60.0) < 1e-3),
        key=lambda p: float(p["gain"]),
    )
    gains = np.array([float(p["gain"]) for p in pairs_d60])
    peaks = np.array([float(p["peak_palette_off"]) for p in pairs_d60])
    if gains.size < 2:
        return False
    z = np.polyfit(gains, peaks, 1)
    slope_pal_per_slider = float(z[0])
    intercept_pal = float(z[1])
    implied_log_mult = 20.0 * slope_pal_per_slider

    if wire_polar_path is None:
        wire_polar_path = (WORKSPACE_ROOT
                           / "ivus_test_0515" / "raw" / "b2_w_wire_p1"
                           / "derived" / "polar" / "FILE0000.dcm.npy")
    if not wire_polar_path.is_file():
        return False
    pol = np.load(wire_polar_path).astype(np.float32)
    r_lo = max(0, wire_r_idx - wire_window // 2)
    r_hi = min(pol.shape[1], wire_r_idx + wire_window // 2)
    profile = pol[wire_theta_idx, r_lo:r_hi].copy()
    r_axis = (np.arange(r_lo, r_hi)) * dr_mm
    peak_pal = float(profile.max())
    peak_r = float(r_axis[int(profile.argmax())])

    log_mult_candidates = [20.0, 50.0, 100.0, yaml_log_multiplier]
    linear_amp = np.power(10.0, profile / yaml_log_multiplier)
    linear_amp_peak = float(linear_amp.max())

    fig = plt.figure(figsize=(16.0, 6.4), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.4],
                          wspace=0.28, left=0.05, right=0.98,
                          top=0.76, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(gains, peaks, s=72, color="#1f78b4",
                 edgecolors="black", zorder=5,
                 label="E6 AR-OFF ringdown peak\n(D=60, c_take2_water)")
    xs = np.linspace(gains.min() - 2, gains.max() + 2, 64)
    ax_a.plot(xs, intercept_pal + slope_pal_per_slider * xs,
              color="#e31a1c", lw=1.4, ls="--",
              label=f"linear fit: slope = {slope_pal_per_slider:.2f} pal/slider")
    ax_a.set_xlabel("gain slider")
    ax_a.set_ylabel("AR-OFF ringdown peak palette")
    ax_a.set_xlim(min(gains) - 5, max(gains) + 5)
    ax_a.set_ylim(0, 255)
    ax_a.set_title(
        "Gain-step calibration\n"
        f"Δpalette / Δslider = {slope_pal_per_slider:.2f}\n"
        f"⇒ implied log_multiplier = 20 × {slope_pal_per_slider:.2f} = "
        f"{implied_log_mult:.1f}\n"
        f"YAML value: {yaml_log_multiplier:.1f}",
        fontsize=9)
    ax_a.legend(loc="lower right", fontsize=7, framealpha=0.85)
    ax_a.grid(True, lw=0.3, alpha=0.5)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(r_axis, profile, color="#1f78b4", lw=1.6,
              label="bench wire radial profile (palette)")
    ax_b.axvline(peak_r, color="#ffd400", lw=1.0, alpha=0.7)
    ax_b.scatter([peak_r], [peak_pal], s=60, color="#ffd400",
                 edgecolors="black", zorder=6,
                 label=f"peak = {peak_pal:.1f} palette")
    cmap = plt.get_cmap("plasma")
    for i, lm in enumerate(log_mult_candidates):
        col = cmap(0.15 + 0.7 * i / max(len(log_mult_candidates) - 1, 1))
        thresh = peak_pal - lm * math.log10(2.0)
        is_yaml = math.isclose(lm, yaml_log_multiplier, rel_tol=1e-3)
        label = (f"log_mult = {lm:.1f} ⇒ "
                 f"−6 dB at peak − {lm * math.log10(2.0):.1f} palette")
        if is_yaml:
            label += "  ← YAML"
        ax_b.axhline(thresh, color=col,
                     lw=1.5 if is_yaml else 1.0,
                     ls="-" if is_yaml else "--",
                     label=label)
    ax_b.set_xlabel("r (mm)")
    ax_b.set_ylabel("palette")
    ax_b.set_ylim(0, max(peak_pal + 20, 255))
    ax_b.set_title(
        "Where does the −6 dB FWHM threshold sit on a wire profile?\n"
        "On the device's log-compressed palette, the threshold sits\n"
        "Δpalette = log_mult × log10(2) ≈ 41.4 below the peak — NOT half-peak.",
        fontsize=9)
    ax_b.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    ax_b.grid(True, lw=0.3, alpha=0.5)

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(r_axis, linear_amp / linear_amp_peak, color="#1f78b4",
              lw=1.6,
              label=("bench profile, back-projected to linear amplitude\n"
                     f"(log_mult = {yaml_log_multiplier:.1f}, normalised)"))
    ax_c.axhline(0.5, color="#e31a1c", lw=1.5, ls="--",
                 label="−6 dB envelope threshold = peak / 2")
    ax_c.axvline(peak_r, color="#ffd400", lw=1.0, alpha=0.7)
    ax_c.scatter([peak_r], [1.0], s=60, color="#ffd400",
                 edgecolors="black", zorder=6,
                 label="peak (normalised = 1)")
    lin = linear_amp / linear_amp_peak
    crossings = np.where(np.diff(np.sign(lin - 0.5)))[0]
    for i_cross in crossings:
        r0 = r_axis[i_cross]
        r1 = r_axis[i_cross + 1]
        v0 = lin[i_cross]; v1 = lin[i_cross + 1]
        if v1 == v0:
            continue
        r_x = r0 + (0.5 - v0) / (v1 - v0) * (r1 - r0)
        ax_c.axvline(r_x, color="#e31a1c", lw=0.8, ls=":", alpha=0.6)
    ax_c.set_xlabel("r (mm)")
    ax_c.set_ylabel("amplitude (normalised)")
    ax_c.set_ylim(0.0, 1.1)
    ax_c.set_title(
        "Same profile in linear amplitude space\n"
        "−6 dB threshold visually crosses at half-peak, validating the\n"
        "envelope-FWHM convention used everywhere in the pipeline.",
        fontsize=9)
    ax_c.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_c.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        f"log_multiplier validation  —  bench gain-step ⇒ {implied_log_mult:.1f}, "
        f"YAML = {yaml_log_multiplier:.1f}\n"
        f"(agreement within {100 * abs(implied_log_mult - yaml_log_multiplier) / yaml_log_multiplier:.1f}%)  —  "
        "Why the −6 dB threshold looks 'high' on palette plots: palette is\n"
        "log-compressed; the linear back-projection (right panel) shows it IS at half-peak.",
        fontsize=11, y=0.97)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# PSF anchor: thin wrapper that just records the existing wave0/p035
# evidence figures.  We don't regenerate them here -- they already
# exist on disk under derived_aggregate/psf_b2_tungsten_water/.
# ---------------------------------------------------------------------------

def render_all_default_evidence(out_dir: Path,
                                psf_anchor: str = "wave0") -> dict[str, bool]:
    """Render every default bench-evidence figure into ``out_dir``.

    Returns a dict of name → bool indicating which figures were
    produced.  Missing inputs (e.g. polar arrays gitignored, no E5
    captures) silently skip — the corresponding entry returns False.
    """
    if plt is None:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = WORKSPACE_ROOT
    status: dict[str, bool] = {}

    status["log_multiplier"] = render_log_multiplier_evidence(
        out_path=out_dir / "bench_evidence_log_multiplier.png",
    )

    status["ringdown"] = render_ringdown_evidence_v2(
        out_path=out_dir / "bench_evidence_ringdown.png",
        ref_gain=50.0, ref_diameter_mm=60.0,
    )

    status["noise"] = render_noise_evidence_v2(
        out_path=out_dir / "bench_evidence_noise.png",
        ref_gain=50.0, ref_diameter_mm=60.0,
    )

    # Depth uniformity: E4a uniform milk is the canonical Test I bench
    # (uniform attenuating + scattering medium).  See tier1_results.md
    # bench-anchor section D for why E4a is the right source.  The E6
    # anechoic figure is retained as a secondary diagnostic for the
    # ringdown-zone behaviour at the same operating point.
    e4a_roots = [ws / f"ivus_test_0515/raw/e4a_milk_gain_p{i}" for i in (1, 2, 3)]
    e4a_roots = [r for r in e4a_roots if r.is_dir()]
    status["depth_uniformity"] = render_depth_uniformity_evidence_e4(
        out_path=out_dir / "bench_evidence_depth_uniformity.png",
        e4_subfolder_roots=e4a_roots,
        ref_gain=68.0, ref_diameter_mm=60.0,
    )
    status["depth_uniformity_water_diag"] = render_depth_uniformity_evidence_v2(
        out_path=out_dir / "bench_evidence_depth_uniformity_water_diag.png",
        ref_gain=50.0, ref_diameter_mm=60.0,
    )

    e5_takes = [ws / f"ivus_test_0515/raw/e5_milk_cyst_take{i}" for i in (1, 2, 3)]
    e5_summaries = [t / "derived/speckle/speckle_summary.json" for t in e5_takes
                    if (t / "derived/speckle/speckle_summary.json").is_file()]
    primary_take = next((t for t in e5_takes if (t / "FILE0001.dcm").is_file()), None)
    if primary_take is not None and e5_summaries:
        status["speckle"] = render_speckle_evidence(
            out_path=out_dir / "bench_evidence_speckle.png",
            dataset_root=primary_take,
            speckle_summary_paths=e5_summaries,
            ref_frame="FILE0001.dcm",
            align_csv=primary_take / "derived/cyst_alignment_fit.csv",
            meta_csv=primary_take / "derived/frames_meta.csv",
        )
    else:
        status["speckle"] = False

    e4_roots = [ws / f"ivus_test_0515/raw/{name}"
                for name in ("e4a_milk_gain_p1", "e4a_milk_gain_p2", "e4a_milk_gain_p3",
                             "e4c_milk_water_gain_p1", "e4c_milk_water_gain_p2",
                             "e4c_milk_water_gain_p3")]
    e4_roots = [r for r in e4_roots if r.is_dir()]
    status["tgc"] = render_tgc_evidence(
        out_path=out_dir / "bench_evidence_tgc.png",
        e4_subfolder_roots=e4_roots,
        yaml_path=ws / "instrument-calibration/p035_visions/volcano_s5i.yaml",
        primary_subfolder="e4c_milk_water_gain_p1",
        gains_to_plot=(35.0, 50.0, 57.0, 68.0),
    )

    # PSF anchor figures already exist on disk; we only verify the links.
    psf_links = collect_psf_evidence_links(psf_anchor)
    status["psf"] = all(p.exists() for _, p in psf_links)

    return status


def collect_psf_evidence_links(anchor: str) -> list[tuple[str, Path]]:
    """Return [(caption, path)] for the per-anchor PSF evidence figures."""
    if anchor == "wave0":
        base = WORKSPACE_ROOT / "ivus_test_0515" / "derived_aggregate" / "psf_b2_tungsten_water"
        return [
            ("Per-wire 2D patches with −6 dB contour (magenta) overlaid on the polar patch around each detected wire peak — visual sanity check that the FWHM walkout is locking onto the wire and not a side-lobe / ringdown artefact.",
             base / "per_wire_patches.png"),
            ("Per-wire axial radial-line profiles with FWHM crossings (dotted red).  Each row is one wire; the threshold line (dashed grey) sits at peak − 41.2 palette (= −6 dB amplitude at log_mult=137.4), confirming the FWHM walkout is genuinely at half-envelope.",
             base / "per_wire_axial_profiles.png"),
            ("Per-wire lateral azimuthal-line profiles with FWHM crossings.  Same threshold convention as the axial profile — this is where the arc-FWHM column of `per_wire_psf.csv` comes from.",
             base / "per_wire_lateral_profiles.png"),
            ("Lateral FWHM vs depth + Gaussian-beam fit (red curve).  This is the fit that yields the `w0_mm`, `z_f_mm`, and `lateral_fwhm_at_focus_mm` anchor values used in Test D.",
             base / "gaussian_beam_fit.png"),
            ("Axial-FWHM distribution histogram across all 147 clean wires pooled across the 4 B2 positions.  Median (red line) is the axial-PSF anchor used in Test C; the dashed grey line shows the Pass-16 sim's spec.",
             base / "axial_fwhm_histogram.png"),
        ]
    if anchor == "p035":
        base = WORKSPACE_ROOT / "P_035_PointScatter" / "derived" / "psf"
        return [
            ("P_035 per-wire patches with the −6 dB contour overlaid (legacy single-take wire phantom).  Generated by `extract_psf.py` — same FWHM walkout / Gaussian-beam fit as the Wave 0 anchor.",
             base / "psf_patches.png"),
            ("P_035 per-wire FWHM + Gaussian-beam fit.  This is the source of the P_035 anchor's `axial_fwhm_mm_median`, `z_f_mm`, and `lateral_fwhm_at_focus_mm`.",
             base / "psf_overview.png"),
        ]
    return []


def _main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path,
                   default=HERE / "tier1_results" / "figures",
                   help="Where to write the bench_evidence_*.png figures.")
    p.add_argument("--psf-anchor", choices=("p035", "wave0"), default="wave0",
                   help="Which PSF anchor to verify (for the report links).")
    args = p.parse_args(argv)
    status = render_all_default_evidence(args.out_dir, psf_anchor=args.psf_anchor)
    for name, ok in status.items():
        flag = "OK " if ok else "SKIP"
        print(f"  [{flag}] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
