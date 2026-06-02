#!/usr/bin/env python3
"""Wave 0 PSF visual-verification figures.

Loads the per-wire extractions from `ivus_test_0515/derived_aggregate/
psf_b2_tungsten_water/per_wire_psf.csv` and produces four sanity-check
figures used by the tier-1 evaluation report:

1. ``per_wire_axial_profiles.png``  — axial radial-line profile through
   each pooled wire's measured peak, with the −6 dB threshold and FWHM
   crossings drawn explicitly.  Visual proof that the per-wire axial
   FWHM extractions are correct.
2. ``per_wire_lateral_profiles.png`` — azimuthal-line profile through
   each pooled wire's measured peak, with the −6 dB threshold and arc-
   FWHM crossings drawn explicitly.
3. ``per_wire_patches.png``         — 2D polar patches around each wire,
   with the −6 dB contour, the prediction crosshair (green) and the
   measured peak (cyan), and a small label of the axial/lateral FWHM.
4. ``gaussian_beam_fit.png``        — pooled (r_actual, lateral_FWHM)
   scatter coloured by subfolder + the Gaussian-beam fit curve, with
   the focal-window and Rayleigh-range annotations.  Visual proof
   that w0 = 0.504 mm, z_f = 17.85 mm is the correct anchor.
5. ``axial_fwhm_histogram.png``     — distribution of all per-wire
   axial FWHMs vs the pooled median + the Pass-16 sim spec.

The "per_wire_*" montages sample N wires evenly across the (r=4-26 mm)
span so the figure stays compact (default N=10).

Outputs land in the same directory the input CSV lives in (by default
``ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/``).
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
from extract_psf import (  # noqa: E402
    extract_patch,
    fwhm_walkout_bins,
    db_to_palette_delta,
)
from polar_utils import (  # noqa: E402
    load_polar_dataset,
    predict_wire_polar_positions,
)
from extract_calibration_inputs import parse_scattering_box_dxf  # noqa: E402


ROOT = Path("/home/jocelynbarker/i4h-sensor-simulation")
B2_SUBFOLDERS = ("b2_w_wire_p1", "b2_w_wire_p2", "b2_w_wire_p3", "b2_w_wire_p4")
# Each B2 capture directory carries its own wire-spiral DXF (12 wires,
# r ∈ {4, 6, ... 26} mm) — different from the P_035 5-wire phantom.
DXF_FILENAME = "wire_spiral_v1.dxf"
# Device-calibrated log compression scale: palette = log_multiplier *
# log10(envelope) + offset. Same default extract_psf.py / extract_speckle.py
# use; threshold lines / FWHM widths in the figures depend on it.
DEFAULT_LOG_MULTIPLIER = 137.4
DEFAULT_THRESHOLD_DB = 6.0
SQRT_LN2 = math.sqrt(math.log(2.0))
TWO_SQRT_LN2 = 2.0 * SQRT_LN2


def _load_polar_for(subfolder: str):
    cap = ROOT / "ivus_test_0515" / "raw" / subfolder
    polar_dir = cap / "derived" / "polar"
    align_csv = cap / "derived" / "alignment_fit.csv"
    meta_csv = cap / "derived" / "frames_meta.csv"
    if not polar_dir.is_dir() or not align_csv.is_file() or not meta_csv.is_file():
        return {}
    return {fr.file: fr for fr in load_polar_dataset(polar_dir, align_csv, meta_csv)}


def _patch_for_row(row, polar_cache, wires_by_sub):
    """Re-extract the same patch + peak that extract_psf.py used for this row.

    `wires_by_sub` is a dict mapping subfolder name → list[Wire] parsed
    from that subfolder's wire_spiral_v1.dxf.
    """
    fr = polar_cache.get(row["source_subfolder"], {}).get(row["file"])
    if fr is None:
        return None
    wires_dxf = wires_by_sub.get(row["source_subfolder"], [])
    if not wires_dxf:
        return None
    preds = predict_wire_polar_positions(fr, wires_dxf)
    pred = next((p for p in preds if p[0] == row["wire_index"]), None)
    if pred is None:
        return None
    _, theta_d_rad, r_pred_mm = pred
    ext = extract_patch(
        fr, theta_d_rad, r_pred_mm,
        search_angle_deg=8.0, search_radius_mm=1.0,
        patch_angle_deg=24.0, patch_radius_mm=3.0,
    )
    if ext is None:
        return None
    ext["frame"] = fr
    ext["r_pred_mm"] = r_pred_mm
    ext["theta_d_rad"] = theta_d_rad
    return ext


def _crossings(profile: np.ndarray, peak_idx: int, threshold: float):
    """Return (left_bin_crossing, right_bin_crossing) or None."""
    res = fwhm_walkout_bins(profile, peak_idx, threshold)
    if res is None:
        return None
    fwhm_bins, _clip = res
    # Re-derive sub-bin crossings (cleaner than reimplementing the walk-out).
    n = len(profile)
    # right
    i = peak_idx
    while i < n - 1 and profile[i + 1] > threshold:
        i += 1
    if i == n - 1:
        right = float(n - 1)
    else:
        a, b = float(profile[i]), float(profile[i + 1])
        right = i + (a - threshold) / max(a - b, 1e-9)
    # left
    j = peak_idx
    while j > 0 and profile[j - 1] > threshold:
        j -= 1
    if j == 0:
        left = 0.0
    else:
        a, b = float(profile[j]), float(profile[j - 1])
        left = j - (a - threshold) / max(a - b, 1e-9)
    return left, right


def render_axial_profiles(rows_subset, polar_cache, wires_by_sub, out_path: Path,
                          log_multiplier: float = DEFAULT_LOG_MULTIPLIER,
                          threshold_db: float = DEFAULT_THRESHOLD_DB):
    """Per-wire axial radial-line profile + measured FWHM markers."""
    n = len(rows_subset)
    cols = min(5, n)
    rows_per = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_per, cols, figsize=(2.6 * cols, 2.1 * rows_per),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    threshold_palette = db_to_palette_delta(threshold_db, log_multiplier)
    for ax, r in zip(axes.flat, rows_subset):
        ext = _patch_for_row(r, polar_cache, wires_by_sub)
        if ext is None:
            continue
        patch = ext["patch"]
        peak_t = ext["peak_local_t"]
        peak_r = ext["peak_local_r"]
        r_offsets_mm = ext["r_offsets_mm"]
        fr = ext["frame"]
        profile = patch[peak_t, :].astype(np.float32)
        peak_val = float(profile[peak_r])
        threshold = peak_val - threshold_palette
        cross = _crossings(profile, peak_r, threshold)
        ax.plot(r_offsets_mm, profile, color="C0", lw=1.2)
        ax.axhline(threshold, color="grey", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(peak_val, color="C0", lw=0.6, ls=":", alpha=0.5)
        if cross is not None:
            left_mm = (cross[0] - peak_r) * fr.dr_mm + r_offsets_mm[peak_r]
            right_mm = (cross[1] - peak_r) * fr.dr_mm + r_offsets_mm[peak_r]
            ax.axvline(left_mm, color="C3", lw=0.8, ls=":")
            ax.axvline(right_mm, color="C3", lw=0.8, ls=":")
            ax.fill_betweenx([threshold, peak_val], left_mm, right_mm,
                             color="C3", alpha=0.18)
        ax.set_axis_on()
        ax.set_title(f"{r['source_subfolder']}/{r['file']}\n"
                     f"w{r['wire_index']}  r={r['r_actual_mm']:.2f}mm  "
                     f"FWHM={r['axial_fwhm_mm']*1000:.0f}µm  pk={r['peak_palette']:.0f}",
                     fontsize=7)
        ax.set_xlabel("Δr from prediction (mm)", fontsize=7)
        ax.set_ylabel("palette", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Wave 0 B2 — axial radial-line profile through each wire's measured peak  "
        f"(−{threshold_db:g} dB threshold = peak − {threshold_palette:.1f} palette @ "
        f"log_mult={log_multiplier:g}; FWHM crossings = dotted red)",
        fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_lateral_profiles(rows_subset, polar_cache, wires_by_sub, out_path: Path,
                            log_multiplier: float = DEFAULT_LOG_MULTIPLIER,
                            threshold_db: float = DEFAULT_THRESHOLD_DB):
    """Per-wire lateral azimuthal-line profile + measured arc-FWHM markers."""
    n = len(rows_subset)
    cols = min(5, n)
    rows_per = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_per, cols, figsize=(2.6 * cols, 2.1 * rows_per),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    threshold_palette = db_to_palette_delta(threshold_db, log_multiplier)
    for ax, r in zip(axes.flat, rows_subset):
        ext = _patch_for_row(r, polar_cache, wires_by_sub)
        if ext is None:
            continue
        patch = ext["patch"]
        peak_t = ext["peak_local_t"]
        peak_r = ext["peak_local_r"]
        theta_offsets_deg = ext["theta_offsets_deg"]
        fr = ext["frame"]
        dtheta_rad = 2.0 * math.pi / fr.num_theta
        arc_per_bin_mm = r["r_actual_mm"] * dtheta_rad
        profile = patch[:, peak_r].astype(np.float32)
        peak_val = float(profile[peak_t])
        threshold = peak_val - threshold_palette
        cross = _crossings(profile, peak_t, threshold)
        theta_offsets_mm = (np.arange(len(profile)) - peak_t) * arc_per_bin_mm
        ax.plot(theta_offsets_mm, profile, color="C0", lw=1.2)
        ax.axhline(threshold, color="grey", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(peak_val, color="C0", lw=0.6, ls=":", alpha=0.5)
        if cross is not None:
            left_mm = (cross[0] - peak_t) * arc_per_bin_mm
            right_mm = (cross[1] - peak_t) * arc_per_bin_mm
            ax.axvline(left_mm, color="C3", lw=0.8, ls=":")
            ax.axvline(right_mm, color="C3", lw=0.8, ls=":")
            ax.fill_betweenx([threshold, peak_val], left_mm, right_mm,
                             color="C3", alpha=0.18)
        ax.set_axis_on()
        ax.set_title(f"{r['source_subfolder']}/{r['file']}\n"
                     f"w{r['wire_index']}  r={r['r_actual_mm']:.2f}mm  "
                     f"arc-FWHM={r['lateral_fwhm_arc_mm']:.2f}mm  pk={r['peak_palette']:.0f}",
                     fontsize=7)
        ax.set_xlabel("Δaz arc (mm)", fontsize=7)
        ax.set_ylabel("palette", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Wave 0 B2 — lateral azimuthal-line profile through each wire's measured peak  "
        f"(−{threshold_db:g} dB threshold = peak − {threshold_palette:.1f} palette @ "
        f"log_mult={log_multiplier:g}; FWHM crossings = dotted red)",
        fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_2d_patches(rows_subset, polar_cache, wires_by_sub, out_path: Path,
                      log_multiplier: float = DEFAULT_LOG_MULTIPLIER,
                      threshold_db: float = DEFAULT_THRESHOLD_DB):
    """Per-wire 2D polar patch with -6dB contour + crosshair labels."""
    n = len(rows_subset)
    cols = min(5, n)
    rows_per = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_per, cols, figsize=(2.6 * cols, 2.5 * rows_per),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, r in zip(axes.flat, rows_subset):
        ext = _patch_for_row(r, polar_cache, wires_by_sub)
        if ext is None:
            continue
        patch = ext["patch"]
        peak_t = ext["peak_local_t"]
        peak_r = ext["peak_local_r"]
        theta_offsets_deg = ext["theta_offsets_deg"]
        r_offsets_mm = ext["r_offsets_mm"]
        peak_val = float(patch[peak_t, peak_r])
        threshold_palette = db_to_palette_delta(threshold_db, log_multiplier)
        vmin = float(np.percentile(patch, 5))
        vmax = max(peak_val, vmin + 1.0)
        ax.imshow(patch, cmap="gray", aspect="auto",
                  extent=(r_offsets_mm[0], r_offsets_mm[-1],
                          theta_offsets_deg[-1], theta_offsets_deg[0]),
                  vmin=vmin, vmax=vmax)
        ax.contour(r_offsets_mm, theta_offsets_deg, patch,
                   levels=[peak_val - threshold_palette],
                   colors=["#ff00ff"], linewidths=1.0)
        ax.scatter([0], [0], s=24, c="#00ff00", marker="+", label="pred")
        ax.scatter([r_offsets_mm[peak_r]], [theta_offsets_deg[peak_t]],
                   s=24, c="#00ffff", marker="x", label="peak")
        ax.set_axis_on()
        ax.set_title(f"{r['source_subfolder']}/{r['file']}\n"
                     f"w{r['wire_index']}  r={r['r_actual_mm']:.2f}mm\n"
                     f"ax={r['axial_fwhm_mm']*1000:.0f}µm  "
                     f"lat={r['lateral_fwhm_arc_mm']:.2f}mm",
                     fontsize=7)
        ax.set_xlabel("Δr (mm)", fontsize=7)
        ax.set_ylabel("Δaz (deg)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(
        f"Wave 0 B2 — per-wire 2D patches  (magenta = −{threshold_db:g} dB contour, "
        f"+ = prediction, × = measured peak; threshold = peak − {threshold_palette:.1f} palette)",
        fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_gaussian_fit(per_wire_rows, fit_json: dict, out_path: Path):
    """Pooled scatter + Gaussian-beam fit curve."""
    fit = fit_json["gaussian_beam_fit"]
    lam_mm = fit_json["wavelength_mm"]
    w0 = fit["w0_mm"]
    zf = fit["z_f_mm"]
    a_mm = fit_json["derived"]["element_radius_mm"]
    peak_max = 230.0
    r_min = 10.0
    rows_in = [r for r in per_wire_rows
               if (not r["border_clipped"]) and r["peak_palette"] <= peak_max
               and r["r_actual_mm"] >= r_min]
    if len(rows_in) < 4:
        rows_in = [r for r in per_wire_rows
                   if r["peak_palette"] <= peak_max
                   and r["r_actual_mm"] >= r_min]
    rows_out = [r for r in per_wire_rows if r not in rows_in]

    rs_for_curve = [r["r_actual_mm"] for r in per_wire_rows]
    z_curve = np.linspace(0.5, max(28.0, max(rs_for_curve) + 2), 200)
    z_R = math.pi * w0 ** 2 / lam_mm
    w_curve = w0 * np.sqrt(1.0 + ((z_curve - zf) / z_R) ** 2)
    fwhm_curve = w_curve * TWO_SQRT_LN2

    colours = {"b2_w_wire_p1": "C0", "b2_w_wire_p2": "C2",
               "b2_w_wire_p3": "C4", "b2_w_wire_p4": "C5"}
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for sub, col in colours.items():
        sub_in = [r for r in rows_in if r["source_subfolder"] == sub]
        sub_out = [r for r in rows_out if r["source_subfolder"] == sub]
        if sub_in:
            ax.scatter([r["r_actual_mm"] for r in sub_in],
                       [r["lateral_fwhm_arc_mm"] for r in sub_in],
                       s=26, color=col, alpha=0.85,
                       label=f"{sub}  (n={len(sub_in)} in fit)")
        if sub_out:
            ax.scatter([r["r_actual_mm"] for r in sub_out],
                       [r["lateral_fwhm_arc_mm"] for r in sub_out],
                       s=16, color=col, alpha=0.30, marker="x")
    ax.plot(z_curve, fwhm_curve, color="C3", lw=1.8,
            label=(f"Gaussian-beam fit\n"
                   f"  w0={w0:.3f} mm  z_f={zf:.2f} mm\n"
                   f"  a={a_mm:.2f} mm  z_R={z_R:.2f} mm"))
    ax.axvline(zf, color="C3", ls="--", alpha=0.5)
    ax.axvspan(zf - z_R, zf + z_R, color="C3", alpha=0.08,
               label=f"focal zone (z_f ± z_R)")
    ax.set_xlabel("Depth from device centre (mm)")
    ax.set_ylabel("Lateral −6 dB FWHM, arc length (mm)")
    ax.set_title("Wave 0 B2 — lateral FWHM vs depth, Gaussian-beam fit  "
                 f"(n_in_fit={len(rows_in)} of {len(per_wire_rows)} pooled wires)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_xlim(0, max(28.0, max(rs_for_curve) + 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_axial_distribution(per_wire_rows, fit_json: dict, out_path: Path):
    """Histogram of per-wire axial FWHM, with the median + Pass-16 sim value."""
    fwhms_um = [r["axial_fwhm_mm"] * 1000 for r in per_wire_rows
                if not r["border_clipped"] and r["peak_palette"] <= 230.0]
    med = fit_json["axial_fwhm_mm_median"] * 1000
    n_cycles = fit_json["pulse_duration_cycles"]
    p16_um = 0.149 * 1000  # current Pass 16 sim spec
    p16_cycles = 1.93
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.hist(fwhms_um, bins=24, color="C0", alpha=0.7, edgecolor="white")
    ax.axvline(med, color="C3", lw=2.2,
               label=f"Wave 0 B2 median = {med:.0f} µm  ({n_cycles:.2f} cyc)")
    ax.axvline(p16_um, color="grey", lw=1.6, ls="--",
               label=f"Pass 16 sim spec = {p16_um:.0f} µm  ({p16_cycles:.2f} cyc)")
    ax.set_xlabel("Axial −6 dB FWHM (µm)")
    ax.set_ylabel("Number of wires")
    ax.set_title(f"Wave 0 B2 — axial FWHM distribution  "
                 f"(n={len(fwhms_um)} clean wires, pooled across 4 positions)")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def pick_montage_rows(rows, n_target=10):
    clean = [r for r in rows
             if not r["border_clipped"] and not r["multilobed"]
             and r["peak_palette"] <= 230.0]
    clean.sort(key=lambda r: r["r_actual_mm"])
    if len(clean) <= n_target:
        return clean
    idxs = np.linspace(0, len(clean) - 1, n_target).astype(int)
    return [clean[i] for i in idxs]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchor-dir", type=Path,
                   default=ROOT / "ivus_test_0515" / "derived_aggregate" / "psf_b2_tungsten_water")
    p.add_argument("--n-montage", type=int, default=10,
                   help="Number of per-wire panels in each montage figure.")
    args = p.parse_args()

    if plt is None:
        print("matplotlib not available", flush=True)
        return 1

    per_wire_csv = args.anchor_dir / "per_wire_psf.csv"
    fit_json_path = args.anchor_dir / "psf_fit.json"
    if not per_wire_csv.is_file() or not fit_json_path.is_file():
        print(f"missing inputs at {args.anchor_dir}", flush=True)
        return 2

    fit_json = json.loads(fit_json_path.read_text())
    rows = []
    with per_wire_csv.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "source_subfolder": r["source_subfolder"],
                "file": r["file"],
                "gain_slider": float(r["gain_slider"]),
                "diameter_mm": float(r["diameter_mm"]),
                "wire_index": int(r["wire_index"]),
                "r_actual_mm": float(r["r_actual_mm"]),
                "axial_fwhm_mm": float(r["axial_fwhm_mm"]),
                "lateral_fwhm_arc_mm": float(r["lateral_fwhm_arc_mm"]),
                "peak_palette": float(r["peak_palette"]),
                "excess_db": float(r["excess_db"]),
                "border_clipped": int(r["border_clipped"]),
                "multilobed": int(r["multilobed"]),
            })
    print(f"loaded {len(rows)} per-wire rows from {per_wire_csv}")

    # Per-subfolder DXF + polar cache.  Each B2 capture has its own
    # `wire_spiral_v1.dxf` (12 wires, r ∈ {4, 6, … 26} mm).
    wires_by_sub: dict[str, list] = {}
    polar_cache: dict[str, dict] = {}
    for sub in B2_SUBFOLDERS:
        dxf_path = ROOT / "ivus_test_0515" / "raw" / sub / DXF_FILENAME
        if dxf_path.is_file():
            wires_by_sub[sub], _ = parse_scattering_box_dxf(dxf_path)
        else:
            wires_by_sub[sub] = []
        polar_cache[sub] = _load_polar_for(sub)
        print(f"  {sub}: {len(polar_cache[sub])} polar frames, "
              f"{len(wires_by_sub[sub])} wires in DXF")

    montage_rows = pick_montage_rows(rows, args.n_montage)
    print(f"  montage rows ({len(montage_rows)}):  "
          + ", ".join(f"r={r['r_actual_mm']:.1f}" for r in montage_rows))

    render_axial_profiles(montage_rows, polar_cache, wires_by_sub,
                          args.anchor_dir / "per_wire_axial_profiles.png")
    print(f"wrote {args.anchor_dir / 'per_wire_axial_profiles.png'}")

    render_lateral_profiles(montage_rows, polar_cache, wires_by_sub,
                            args.anchor_dir / "per_wire_lateral_profiles.png")
    print(f"wrote {args.anchor_dir / 'per_wire_lateral_profiles.png'}")

    render_2d_patches(montage_rows, polar_cache, wires_by_sub,
                      args.anchor_dir / "per_wire_patches.png")
    print(f"wrote {args.anchor_dir / 'per_wire_patches.png'}")

    render_gaussian_fit(rows, fit_json,
                        args.anchor_dir / "gaussian_beam_fit.png")
    print(f"wrote {args.anchor_dir / 'gaussian_beam_fit.png'}")

    render_axial_distribution(rows, fit_json,
                              args.anchor_dir / "axial_fwhm_histogram.png")
    print(f"wrote {args.anchor_dir / 'axial_fwhm_histogram.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
