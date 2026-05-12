#!/usr/bin/env python3
"""E2 — PSF extraction from per-wire echoes in the polar arrays.

Goal: derive
  probe.pulse_duration_cycles  -- from axial -6 dB FWHM
  probe.element_radius_mm      -- from beam-waist size at focus
  probe.focal_length_mm        -- from depth at minimum lateral FWHM

Per-wire measurement (in polar)
-------------------------------
For each wire predicted at (theta_design_rad, r_polar_mm) by the
alignment fit, we extract a small (theta x r) patch around the
prediction and:

  1. Search for the actual peak palette pixel within +/- search_radius_mm
     in r and +/- search_angle_deg in theta (residual fit error <= 1 mm).
  2. Estimate a local background from the patch border (10th percentile).
  3. Compute the -6 dB extent (palette > peak - 6) along the peak's row
     (axial, in mm) and column (lateral, in arc-length mm = r_w * dtheta).
  4. Discard wires whose peak excess over background is < 6 dB (can't
     measure -6 dB FWHM cleanly), or whose peak is saturated (>= 235).

We use 1 palette unit = 1 dB (assumes log_multiplier = 20). E7 will
refine; the FWHM measurement is tolerant to a moderate change in
log_multiplier because the -6 dB threshold scales with it.

Pulse duration
--------------
For an N-cycle pulse at frequency f in medium with sound speed c
(round-trip):

    FWHM_axial = N * lambda / 2     =>     N = 2 * FWHM_axial / lambda

We average axial FWHM across wires and gain it 6 dB for the conventional
"echo round-trip" definition; the per-wire scatter sets the uncertainty.

Gaussian-beam model
-------------------
Lateral 1/e amplitude radius w(z) of a focused circular aperture:

    w(z) = w_0 * sqrt(1 + ((z - z_f) / z_R)^2)
    z_R  = pi * w_0^2 / lambda
    w_0  = lambda * F / (2 * a)         (diffraction-limited at focus)

We fit (w_0, z_f) from the per-wire (z, w) cloud with z = r_polar_mm
and w = FWHM_lateral_arc_mm / (2 * sqrt(ln 2)) -- the conversion from
amplitude-FWHM to 1/e amplitude radius. Then

    focal_length_mm  = z_f
    element_radius_mm = lambda * z_f / (2 * w_0)

Outputs
-------
  derived/psf/per_wire_psf.csv      one row per (frame, wire)
  derived/psf/psf_fit.json          axial / lateral fit summary
  derived/psf/psf_overview.png      axial-FWHM(z) + lateral-FWHM(z) + Gaussian-beam fit
  derived/psf/psf_patches.png       per-wire patch montage with -6 dB contour
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

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
from extract_calibration_inputs import parse_scattering_box_dxf  # noqa: E402


SQRT_LN2 = math.sqrt(math.log(2.0))
TWO_SQRT_LN2 = 2.0 * SQRT_LN2  # FWHM of amplitude Gaussian = 2 * w * sqrt(ln 2)


# ----- per-wire patch extraction -----


def extract_patch(
    fr: PolarFrame, theta_d_rad: float, r_mm: float,
    search_angle_deg: float = 8.0, search_radius_mm: float = 1.0,
    patch_angle_deg: float = 20.0, patch_radius_mm: float = 2.5,
):
    """Return (patch, theta_offsets_deg, r_offsets_mm, peak_local_idx, peak_global_theta_bin, peak_global_r_bin).

    patch is a 2D ndarray (n_theta_bins, n_r_bins) excised from fr.arr.
    theta_offsets_deg / r_offsets_mm give the patch axes (relative to
    the prediction). Peak indices are local to the patch.
    """
    num_theta = fr.num_theta
    dtheta_rad = 2.0 * math.pi / num_theta

    # Predicted bin
    i_theta_pred = int(round(theta_d_rad / dtheta_rad)) % num_theta
    j_r_pred = int(round(r_mm / fr.dr_mm))

    # Patch half-widths
    half_theta_patch = int(round(math.radians(patch_angle_deg) / dtheta_rad / 2))
    half_r_patch = int(round(patch_radius_mm / fr.dr_mm))

    # Build wrapped theta indices
    theta_idx = [(i_theta_pred + d) % num_theta
                 for d in range(-half_theta_patch, half_theta_patch + 1)]
    j0 = max(0, j_r_pred - half_r_patch)
    j1 = min(fr.num_r, j_r_pred + half_r_patch + 1)
    if j1 - j0 < 5:
        return None
    patch = fr.arr[theta_idx, :][:, j0:j1].astype(np.float32)

    # Within the patch, restrict search to inner [search_*]
    half_theta_search = int(round(math.radians(search_angle_deg) / dtheta_rad / 2))
    half_r_search = int(round(search_radius_mm / fr.dr_mm))
    sr_t0 = max(0, half_theta_patch - half_theta_search)
    sr_t1 = min(patch.shape[0], half_theta_patch + half_theta_search + 1)
    sr_r0 = max(0, (j_r_pred - j0) - half_r_search)
    sr_r1 = min(patch.shape[1], (j_r_pred - j0) + half_r_search + 1)

    sub = patch[sr_t0:sr_t1, sr_r0:sr_r1]
    if sub.size == 0:
        return None
    pk_local = np.unravel_index(np.argmax(sub), sub.shape)
    pk_t_in_patch = sr_t0 + pk_local[0]
    pk_r_in_patch = sr_r0 + pk_local[1]

    # Patch axes (offsets in deg / mm relative to prediction)
    theta_offsets_deg = (np.arange(patch.shape[0])
                         - half_theta_patch) * math.degrees(dtheta_rad)
    r_offsets_mm = (np.arange(patch.shape[1]) + j0 - j_r_pred) * fr.dr_mm

    return dict(
        patch=patch,
        theta_offsets_deg=theta_offsets_deg,
        r_offsets_mm=r_offsets_mm,
        peak_local_t=pk_t_in_patch,
        peak_local_r=pk_r_in_patch,
        peak_global_theta_bin=theta_idx[pk_t_in_patch],
        peak_global_r_bin=j0 + pk_r_in_patch,
        j0=j0,
        theta_idx=theta_idx,
    )


def fwhm_walkout_bins(profile: np.ndarray, peak_idx: int,
                      threshold: float) -> tuple[float, bool] | None:
    """Direct -<thr> FWHM with sub-bin linear interpolation at the crossings.

    Walks outward from `peak_idx` in `profile` until the first bin that
    drops below `threshold` on each side, then linearly interpolates the
    crossing between the last-above and first-below bins. Returns
    (fwhm_in_bins, border_clipped). border_clipped=True if the profile
    never crossed `threshold` on one or both sides (FWHM is a lower
    bound).

    This is log_multiplier-independent IF `threshold = peak - 0.301 *
    log_multiplier` (= where amplitude drops to half). The caller
    chooses `threshold` based on its log_multiplier assumption (or
    just `threshold = peak - 6` for log_multiplier = 20).
    """
    n = len(profile)
    if profile[peak_idx] <= threshold:
        return None

    border_clipped = False

    # Right side
    i = peak_idx
    while i < n - 1 and profile[i + 1] > threshold:
        i += 1
    if i == n - 1:
        right_x = float(n - 1) - peak_idx
        border_clipped = True
    else:
        a, b = float(profile[i]), float(profile[i + 1])
        # crossing fraction within bin [i, i+1]
        frac = (a - threshold) / max(a - b, 1e-9)
        right_x = (i + frac) - peak_idx

    # Left side
    j = peak_idx
    while j > 0 and profile[j - 1] > threshold:
        j -= 1
    if j == 0:
        left_x = peak_idx - 0.0
        border_clipped = True
    else:
        a, b = float(profile[j]), float(profile[j - 1])
        frac = (a - threshold) / max(a - b, 1e-9)
        left_x = peak_idx - (j - frac)

    return left_x + right_x, border_clipped


def measure_psf_fwhm(
    patch: np.ndarray, peak_t: int, peak_r: int, dr_mm: float,
    arc_per_bin_mm: float, peak_excess_required_db: float = 6.0,
    bg_percentile: float = 10.0, threshold_db: float = 6.0,
    secondary_peak_max_db: float = 6.0,
):
    """Compute -threshold_db FWHM along axial and lateral cross-sections.

    * Direct -<threshold_db> walk-out with sub-bin linear interpolation
      at the crossings (no parabolic extrapolation, which would
      under-report FWHM for wide PSFs).
    * Quality flag `multilobed` if any pixel outside the primary lobe
      exceeds peak - `secondary_peak_max_db` (indicates a competing
      reflector or reverberation contaminating the patch).

    Returns dict or None if the peak excess is below
    `peak_excess_required_db`.
    """
    peak = float(patch[peak_t, peak_r])
    bg = float(np.percentile(patch, bg_percentile))
    excess_db = peak - bg
    if excess_db < peak_excess_required_db:
        return None
    if peak >= 235.0:
        # Hard saturation against the palette ceiling.
        return dict(saturated=True, peak=peak, bg=bg, excess_db=excess_db)

    threshold = peak - threshold_db

    # Direct -threshold_db FWHM with linear sub-bin interpolation.
    axial = fwhm_walkout_bins(patch[peak_t, :], peak_r, threshold)
    lateral = fwhm_walkout_bins(patch[:, peak_r], peak_t, threshold)
    if axial is None or lateral is None:
        return None
    axial_bins, axial_clipped = axial
    lateral_bins, lateral_clipped = lateral
    border_clipped = axial_clipped or lateral_clipped

    # Primary lobe extent (integer bins) for multilobe check.
    t_lo, t_hi = peak_t, peak_t
    while t_lo > 0 and patch[t_lo - 1, peak_r] > threshold:
        t_lo -= 1
    while t_hi < patch.shape[0] - 1 and patch[t_hi + 1, peak_r] > threshold:
        t_hi += 1
    r_lo, r_hi = peak_r, peak_r
    while r_lo > 0 and patch[peak_t, r_lo - 1] > threshold:
        r_lo -= 1
    while r_hi < patch.shape[1] - 1 and patch[peak_t, r_hi + 1] > threshold:
        r_hi += 1

    outside = patch.copy()
    outside[t_lo:t_hi + 1, r_lo:r_hi + 1] = bg
    second_peak = float(outside.max())
    multilobed = (peak - second_peak) < secondary_peak_max_db

    return dict(
        saturated=False,
        peak=peak,
        bg=bg,
        excess_db=excess_db,
        axial_fwhm_mm=axial_bins * dr_mm,
        lateral_fwhm_arc_mm=lateral_bins * arc_per_bin_mm,
        border_clipped=border_clipped,
        multilobed=multilobed,
        primary_lobe_db_above_secondary=peak - second_peak,
        t_lo=t_lo, t_hi=t_hi, r_lo=r_lo, r_hi=r_hi,
    )


# ----- Gaussian beam fit -----


def gaussian_beam_w(z, w_0, z_f, lam):
    z_R = math.pi * w_0 ** 2 / lam
    return w_0 * np.sqrt(1.0 + ((z - z_f) / z_R) ** 2)


def fit_gaussian_beam(
    z_mm: np.ndarray, w_mm: np.ndarray, lam_mm: float,
):
    p0 = [float(np.min(w_mm)), float(z_mm[np.argmin(w_mm)])]
    bounds = ([0.05, 0.5], [3.0, 30.0])
    popt, pcov = curve_fit(
        lambda z, w0, zf: gaussian_beam_w(z, w0, zf, lam_mm),
        z_mm, w_mm, p0=p0, bounds=bounds,
    )
    w_0, z_f = float(popt[0]), float(popt[1])
    perr = np.sqrt(np.diag(pcov))
    return w_0, z_f, float(perr[0]), float(perr[1])


# ----- visualization -----


def render_overview(
    rows: list[dict], w0: float, zf: float, lam: float,
    out_path: Path, freq_mhz: float, c_mm_per_us: float,
    lateral_peak_max: float = 230.0, lateral_min_r: float = 10.0,
):
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    z_all = np.array([r["r_actual_mm"] for r in rows])
    ax_fwhm = np.array([r["axial_fwhm_mm"] for r in rows])
    lat_fwhm = np.array([r["lateral_fwhm_arc_mm"] for r in rows])
    peak_pal = np.array([r["peak_palette"] for r in rows])
    used_in_lateral = np.array([
        not r["border_clipped"] and not r["multilobed"]
        and r["peak_palette"] <= lateral_peak_max
        and r["r_actual_mm"] >= lateral_min_r
        for r in rows
    ])
    cmap = np.where(used_in_lateral, "C1", "#888")

    axes[0].scatter(z_all, ax_fwhm, s=30, c="C0", alpha=0.7,
                    label=f"per-wire axial FWHM (n={len(z_all)})")
    axes[0].axhline(np.median(ax_fwhm), color="C0", linestyle="--",
                    label=f"median = {np.median(ax_fwhm):.3f} mm")
    n_cycles = 2.0 * np.median(ax_fwhm) * freq_mhz / c_mm_per_us
    axes[0].set_ylabel("Axial -6 dB FWHM (mm)")
    axes[0].set_title(f"Axial PSF: median FWHM = {np.median(ax_fwhm):.3f} mm  "
                      f"=>  N_cycles = {n_cycles:.2f}")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].scatter(z_all, lat_fwhm, s=30, c=cmap, alpha=0.7,
                    label=f"per-wire (orange = used in fit, gray = excluded)")
    z_curve = np.linspace(0.5, max(30.0, z_all.max() + 2), 256)
    w_curve = gaussian_beam_w(z_curve, w0, zf, lam)
    fwhm_curve = w_curve * TWO_SQRT_LN2
    axes[1].plot(z_curve, fwhm_curve, color="C3",
                 label=f"Gaussian beam fit: w_0={w0:.3f} mm, z_f={zf:.2f} mm")
    axes[1].axvline(zf, color="C3", linestyle="--", alpha=0.5)
    a_mm = lam * zf / (2.0 * w0)
    axes[1].set_xlabel("Depth from device center (mm)")
    axes[1].set_ylabel("Lateral -6 dB FWHM (arc-length mm)")
    axes[1].set_title(f"Lateral PSF: focal_length = {zf:.2f} mm,  "
                      f"element_radius = {a_mm:.3f} mm")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_patch_montage(rows: list[dict], out_path: Path):
    if plt is None:
        return
    n = len(rows)
    if n == 0:
        return
    cols = min(5, n)
    rows_per = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_per, cols, figsize=(2.4 * cols, 2.4 * rows_per),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, r in zip(axes.flat, rows):
        patch = r["_patch"]
        peak = r["peak_palette"]
        ax.imshow(patch, cmap="gray", aspect="auto",
                  extent=(r["_r_offsets"][0], r["_r_offsets"][-1],
                          r["_theta_offsets"][-1], r["_theta_offsets"][0]),
                  vmin=r["bg_palette"], vmax=peak)
        ax.contour(r["_r_offsets"], r["_theta_offsets"], patch,
                   levels=[peak - 6.0], colors=["#ff00ff"], linewidths=0.8)
        ax.scatter([0], [0], s=12, c="#00ff00", marker="+",
                   label="prediction")
        ax.scatter([(r["peak_local_r"] - len(r["_r_offsets"]) // 2) * r["_dr"]],
                   [(r["peak_local_t"] - len(r["_theta_offsets"]) // 2)
                    * (r["_theta_offsets"][1] - r["_theta_offsets"][0])],
                   s=12, c="#00ffff", marker="x")
        ax.set_axis_on()
        ax.set_title(f"{r['file']} w{r['wire_index']}\n"
                     f"r={r['r_actual_mm']:.2f} mm",
                     fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("Per-wire PSF patches — -6 dB contour (magenta), prediction (+), peak (x)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ----- main -----


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
                   default=Path("P_035_PointScatter/derived/psf"))
    p.add_argument("--gain-sliders", type=float, nargs="+", default=[44.0, 54.0],
                   help="Gain groups used for the PSF fit. Default 44+54: gain 44 "
                        "gives clean inner wires (gain 54 saturates them) and gain 54 "
                        "gives clean deep wires (gain 44's outer wires have low SNR).")
    p.add_argument("--diameter-mm", type=float, default=60.0)
    p.add_argument("--frequency-mhz", type=float, default=10.0,
                   help="Visions PV .035 family is 10 MHz")
    p.add_argument("--sound-speed-mm-per-us", type=float, default=1.54,
                   help="s5 device convention (Service Manual)")
    p.add_argument("--search-angle-deg", type=float, default=8.0)
    p.add_argument("--search-radius-mm", type=float, default=1.0)
    p.add_argument("--patch-angle-deg", type=float, default=24.0)
    p.add_argument("--patch-radius-mm", type=float, default=3.0)
    p.add_argument("--lateral-peak-max-palette", type=float, default=230.0,
                   help="Drop wires with peak > this from the lateral fit "
                        "(near-saturation flattens the top of the PSF and "
                        "biases the -6 dB FWHM low).")
    p.add_argument("--lateral-min-r-mm", type=float, default=10.0,
                   help="Drop wires with r < this from the lateral fit. The "
                        "inner wires (r ~ 4-5 mm) are virtually always near "
                        "saturation at any usable gain on this dataset, and "
                        "their displayed PSF reflects the clipped top, not "
                        "the true beam profile.")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_polar_dataset(args.polar_dir, args.align_csv, args.meta_csv)
    wires, _ = parse_scattering_box_dxf(args.dxf)

    lam_mm = args.sound_speed_mm_per_us / args.frequency_mhz

    # Filter to canonical groups
    sel = [fr for fr in frames
           if fr.gain_slider in args.gain_sliders
           and fr.diameter_mm == args.diameter_mm]
    if not sel:
        print(f"No frames at gain in {args.gain_sliders}, D={args.diameter_mm}",
              file=sys.stderr)
        return 1
    print(f"Using {len(sel)} frames at gain in {args.gain_sliders}, "
          f"D={args.diameter_mm:.0f} mm")

    rows = []
    skipped = []
    for fr in sel:
        preds = predict_wire_polar_positions(fr, wires)
        for w_idx, theta_d, r_polar in preds:
            if r_polar > fr.depth_mm - args.patch_radius_mm:
                continue  # wire out of FOV
            ext = extract_patch(
                fr, theta_d, r_polar,
                search_angle_deg=args.search_angle_deg,
                search_radius_mm=args.search_radius_mm,
                patch_angle_deg=args.patch_angle_deg,
                patch_radius_mm=args.patch_radius_mm,
            )
            if ext is None:
                continue
            patch = ext["patch"]
            arc_per_bin_mm = (2.0 * math.pi * r_polar) / fr.num_theta
            psf = measure_psf_fwhm(
                patch, ext["peak_local_t"], ext["peak_local_r"],
                fr.dr_mm, arc_per_bin_mm,
            )
            if psf is None:
                skipped.append((fr.file, w_idx, "low excess"))
                continue
            if psf.get("saturated"):
                skipped.append((fr.file, w_idx, "saturated"))
                continue
            r_actual_mm = ext["peak_global_r_bin"] * fr.dr_mm
            row = {
                "file": fr.file,
                "gain_slider": fr.gain_slider,
                "diameter_mm": fr.diameter_mm,
                "wire_index": w_idx,
                "r_pred_mm": round(r_polar, 4),
                "r_actual_mm": round(r_actual_mm, 4),
                "theta_pred_deg": round(math.degrees(theta_d), 3),
                "peak_palette": round(psf["peak"], 2),
                "bg_palette": round(psf["bg"], 2),
                "excess_db": round(psf["excess_db"], 2),
                "axial_fwhm_mm": round(psf["axial_fwhm_mm"], 4),
                "lateral_fwhm_arc_mm": round(psf["lateral_fwhm_arc_mm"], 4),
                "border_clipped": int(psf["border_clipped"]),
                "multilobed": int(psf["multilobed"]),
                "primary_db_above_secondary": round(psf["primary_lobe_db_above_secondary"], 2),
                # not written to CSV (start with _)
                "_patch": patch,
                "_theta_offsets": ext["theta_offsets_deg"],
                "_r_offsets": ext["r_offsets_mm"],
                "_dr": fr.dr_mm,
                "peak_local_t": ext["peak_local_t"],
                "peak_local_r": ext["peak_local_r"],
            }
            rows.append(row)

    print(f"Extracted {len(rows)} clean wire PSFs (skipped {len(skipped)}).")
    if skipped:
        for s in skipped:
            print(f"  skipped {s}")

    # Write per-wire CSV (drop underscore-prefixed fields)
    with (args.out_dir / "per_wire_psf.csv").open("w", newline="") as f:
        cols = [k for k in rows[0].keys() if not k.startswith("_")
                and k not in ("peak_local_t", "peak_local_r")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # ----- pulse duration from axial FWHM (clean wires only) -----
    ax_fwhm_all = np.array([r["axial_fwhm_mm"] for r in rows])
    keep_axial = np.array([
        not r["border_clipped"]
        and r["peak_palette"] <= args.lateral_peak_max_palette
        for r in rows
    ])
    if keep_axial.sum() < 2:
        keep_axial = np.ones_like(keep_axial)
    ax_fwhm_clean = ax_fwhm_all[keep_axial]
    n_cycles = 2.0 * np.median(ax_fwhm_clean) / lam_mm

    # ----- Gaussian beam fit on lateral FWHM -----
    z = np.array([r["r_actual_mm"] for r in rows])
    lat_fwhm_arc = np.array([r["lateral_fwhm_arc_mm"] for r in rows])
    # Convert amplitude FWHM -> 1/e amplitude radius (= w in Gaussian beam)
    w_arc = lat_fwhm_arc / TWO_SQRT_LN2

    # Drop wires that are unsuitable for the LATERAL Gaussian beam fit:
    #   - near-saturation peak (clipped top biases -6 dB FWHM low)
    #   - inner wires (always saturation-affected on this dataset)
    #   - border-clipped patches (FWHM is a lower bound)
    #   - multilobed patches (competing reflectors confuse the fit)
    keep = np.array([
        not r["border_clipped"]
        and not r["multilobed"]
        and r["peak_palette"] <= args.lateral_peak_max_palette
        and r["r_actual_mm"] >= args.lateral_min_r_mm
        for r in rows
    ])
    if keep.sum() < 4:
        print(f"Too few clean wires "
              f"(peak <= {args.lateral_peak_max_palette}, "
              f"r >= {args.lateral_min_r_mm} mm) "
              f"to fit Gaussian beam; relaxing filters.", file=sys.stderr)
        keep = np.array([
            not r["border_clipped"] and not r["multilobed"]
            and r["r_actual_mm"] >= args.lateral_min_r_mm
            for r in rows
        ])

    w0, zf, w0_err, zf_err = fit_gaussian_beam(z[keep], w_arc[keep], lam_mm)
    a_mm = lam_mm * zf / (2.0 * w0)

    out_json = {
        "gain_sliders": list(args.gain_sliders),
        "diameter_mm": args.diameter_mm,
        "frequency_mhz": args.frequency_mhz,
        "sound_speed_mm_per_us": args.sound_speed_mm_per_us,
        "wavelength_mm": lam_mm,
        "n_wires_used": int(len(rows)),
        "n_wires_for_lateral_fit": int(keep.sum()),
        "axial_fwhm_mm_median": float(np.median(ax_fwhm_clean)),
        "axial_fwhm_mm_std": float(np.std(ax_fwhm_clean)),
        "axial_fwhm_n_clean_wires": int(keep_axial.sum()),
        "axial_fwhm_mm_per_wire": [
            {"wire": r["wire_index"], "r_mm": r["r_actual_mm"],
             "axial_fwhm_mm": r["axial_fwhm_mm"],
             "peak_palette": r["peak_palette"]}
            for r in rows
        ],
        "pulse_duration_cycles": float(n_cycles),
        "gaussian_beam_fit": {
            "w0_mm": w0, "w0_err_mm": w0_err,
            "z_f_mm": zf, "z_f_err_mm": zf_err,
            "z_R_mm": float(math.pi * w0 ** 2 / lam_mm),
            "lateral_fwhm_at_focus_mm": float(w0 * TWO_SQRT_LN2),
        },
        "derived": {
            "focal_length_mm": zf,
            "element_radius_mm": a_mm,
            "pulse_duration_cycles": float(n_cycles),
        },
        "method": __doc__.strip().splitlines()[0],
    }
    with (args.out_dir / "psf_fit.json").open("w") as f:
        json.dump(out_json, f, indent=2)

    if plt is not None:
        render_overview(rows, w0, zf, lam_mm,
                        args.out_dir / "psf_overview.png",
                        args.frequency_mhz, args.sound_speed_mm_per_us,
                        args.lateral_peak_max_palette,
                        args.lateral_min_r_mm)
        render_patch_montage(rows, args.out_dir / "psf_patches.png")

    # Console summary
    print()
    print(f"Wavelength @ {args.frequency_mhz} MHz: lambda = {lam_mm * 1000:.1f} um")
    print(f"Axial PSF (-6 dB, n={keep_axial.sum()}/{len(rows)} clean): "
          f"median FWHM = {np.median(ax_fwhm_clean)*1000:.0f} um  "
          f"(std={np.std(ax_fwhm_clean)*1000:.0f} um)")
    print(f"  => pulse_duration_cycles = {n_cycles:.2f}")
    print(f"Lateral PSF (Gaussian beam fit, n={keep.sum()}/{len(rows)}):")
    print(f"  w_0 = {w0:.3f} +/- {w0_err:.3f} mm  "
          f"(FWHM at focus = {w0 * TWO_SQRT_LN2:.3f} mm)")
    print(f"  z_f = {zf:.2f} +/- {zf_err:.2f} mm  (focal_length_mm)")
    print(f"  a   = {a_mm:.3f} mm                 (element_radius_mm)")
    print(f"\nWrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
