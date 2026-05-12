#!/usr/bin/env python3
"""Derive the Acoustic-Reference (AR) subtraction template from the
ivus_test_0508 Capture-C pairs.

Inputs:
  --dataset      Path to a folder of staged Capture-C DICOMs (FILE####.dcm
                 symlinks; ar_state alternates 0/1 in pairs at the same
                 gain/diameter). Typically:
                   ivus_test_0508/raw/c_take2_water/      (6 frames, 3 pairs)
                   ivus_test_0508/raw/c_pilot_water/      (2 frames, 1 pair)

Outputs: <out>/
  ar_template.json             Per-pair summary (gain, diameter, peak,
                                extent_mm, area_under_excess, etc.)
  ar_template_by_pair.npy      float32 (n_pairs, n_r) palette-domain AR
                                templates, one row per pair
  ar_template_canonical.npy    float32 (n_r,) canonical template (median
                                across the gain-50 pairs after scaling to
                                a common gain)
  ar_template_overview.png     Diagnostic plot: AR-off, AR-on, difference,
                                and the canonical template.
  ar_template_pair_<i>.csv     Per-pair (r_mm, palette_off, palette_on,
                                palette_diff) for cross-checks.

The AR template is the device's running estimate of the catheter's
ringdown-only signal. It is captured in displayed-palette space (the
DICOM contains a post-log-compressed image), so the simulator should
convert it to amplitude space via the calibrated log_multiplier before
applying it; the YAML's `processing.ring_down.waveform_path` and
`subtract_reference` flag govern that conversion.

Method:
  1. For each pair (gain, diameter) where ar_state flips 0->1:
     - load AR-OFF (mode_flag=0) and AR-ON (mode_flag=1) frames
     - middle-frame radial profile (median over theta about image
       center, in palette units)
     - delta = profile_OFF - profile_ON
  2. The delta is the AR template's palette contribution at that gain.
  3. Per-pair, characterise the template: peak, peak depth, 5%-of-peak
     extent.
  4. Build a canonical template by median-pooling the gain-50 pairs
     (after light radial alignment; these are the same physical
     catheter so the template shape is gain-invariant by construction).

This script does NOT require the polar pipeline (alignment fit /
unwrap.py) -- the displayed B-mode is already radially symmetric around
the catheter centre, so the median-over-theta of the scan-converted
display gives the radial profile directly. The catheter sits at the
DICOM image centre.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pydicom

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
Path("/tmp/mpl-cache").mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


REJECT_PALETTE = 11.0


@dataclass
class FrameMeta:
    file: str
    gain: float
    diameter_mm: float
    ar_state: int
    pixel_spacing_mm: float
    rows: int
    cols: int


def load_frame(path: Path) -> tuple[np.ndarray, FrameMeta]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr.mean(axis=-1)
    f0 = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr
    region = ds.SequenceOfUltrasoundRegions[0]
    pix_mm = float(region.PhysicalDeltaX) * 10.0
    meta = FrameMeta(
        file=path.name,
        gain=float(ds.get((0x0029, 0x1001)).value),
        diameter_mm=float(ds.get((0x0029, 0x1003)).value),
        ar_state=int(ds.get((0x0029, 0x1007)).value),
        pixel_spacing_mm=pix_mm,
        rows=int(ds.Rows),
        cols=int(ds.Columns),
    )
    return f0.astype(np.float32), meta


def radial_profile(image: np.ndarray, pix_mm: float, r_max_mm: float,
                   stat: str = "median") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-over-theta palette profile vs radius (mm) about the image centre.

    Returns (r_mm, profile_palette, n_samples_per_bin).
    """
    H, W = image.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r_pix = np.hypot(yy - cy, xx - cx)
    r_mm = r_pix * pix_mm
    n_bins = int(round(r_max_mm / pix_mm))
    r_grid = (np.arange(n_bins) + 0.5) * pix_mm
    profile = np.full(n_bins, np.nan, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        r_lo = i * pix_mm
        r_hi = (i + 1) * pix_mm
        mask = (r_mm >= r_lo) & (r_mm < r_hi)
        if mask.any():
            pix = image[mask]
            if stat == "median":
                profile[i] = float(np.median(pix))
            else:
                profile[i] = float(pix.mean())
            counts[i] = int(mask.sum())
    return r_grid, profile, counts


def find_extent_mm(r_grid: np.ndarray, excess: np.ndarray,
                   threshold_frac: float = 0.05) -> float:
    """First radius past the peak at which excess falls below
    threshold_frac * peak. Returns NaN if not crossed in the window."""
    if excess.size == 0:
        return float("nan")
    peak_idx = int(np.nanargmax(excess))
    peak = float(excess[peak_idx])
    if peak <= 0:
        return float("nan")
    thresh = threshold_frac * peak
    for i in range(peak_idx + 1, len(excess)):
        if excess[i] <= thresh:
            return float(r_grid[i])
    return float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True,
                   help="Folder of Capture-C FILE####.dcm files (ar_state pairs)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output folder for ar_template.* artifacts")
    p.add_argument("--r-max-mm", type=float, default=10.0,
                   help="Max radius for the template (mm); default 10 (well past "
                        "the ringdown extent of 3 mm)")
    p.add_argument("--canonical-gain", type=float, default=50.0,
                   help="Gain to use for the canonical template (averages "
                        "across diameters at that gain). Default 50.")
    args = p.parse_args(argv)

    files = sorted(args.dataset.glob("FILE*.dcm"))
    if not files:
        print(f"no FILE*.dcm in {args.dataset}", file=os.sys.stderr)
        return 1

    # Load all frames
    loaded: list[tuple[np.ndarray, FrameMeta]] = []
    for f in files:
        loaded.append(load_frame(f))

    # Pair by (gain, diameter)
    by_key: dict[tuple[float, float], dict[int, tuple[np.ndarray, FrameMeta]]] = {}
    for img, meta in loaded:
        key = (meta.gain, meta.diameter_mm)
        by_key.setdefault(key, {})[meta.ar_state] = (img, meta)

    pairs: list[tuple[tuple[float, float], tuple[np.ndarray, FrameMeta],
                       tuple[np.ndarray, FrameMeta]]] = []
    for key, by_state in by_key.items():
        if 0 in by_state and 1 in by_state:
            pairs.append((key, by_state[0], by_state[1]))

    if not pairs:
        print("no AR-off/AR-on pairs found in dataset", file=os.sys.stderr)
        return 1
    pairs.sort(key=lambda x: x[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair profiles + per-pair CSVs
    per_pair_records = []
    profiles_palette: list[np.ndarray] = []
    pix_mm_ref = None
    r_grid_ref = None
    for i, (key, (img_off, m_off), (img_on, m_on)) in enumerate(pairs):
        gain, diam = key
        pix_mm = m_off.pixel_spacing_mm
        if pix_mm_ref is None:
            pix_mm_ref = pix_mm
            r_grid_ref, _, _ = radial_profile(img_off, pix_mm, args.r_max_mm)
        r_grid_pair, prof_off, n_off = radial_profile(img_off, pix_mm, args.r_max_mm)
        _,           prof_on,  _    = radial_profile(img_on,  pix_mm, args.r_max_mm)
        # Resample to the reference grid if pixel pitch differs
        if pix_mm != pix_mm_ref:
            prof_off_res = np.interp(r_grid_ref, r_grid_pair, prof_off,
                                       left=np.nan, right=np.nan)
            prof_on_res = np.interp(r_grid_ref, r_grid_pair, prof_on,
                                      left=np.nan, right=np.nan)
        else:
            prof_off_res = prof_off
            prof_on_res = prof_on
        # Speckle floor for AR-on (palette near the far field)
        far_mask = (r_grid_ref >= r_grid_ref[-1] - 1.0)
        speckle_floor_on = float(np.nanmedian(prof_on_res[far_mask]))
        far_mask_off = (r_grid_ref >= r_grid_ref[-1] - 1.0)
        speckle_floor_off = float(np.nanmedian(prof_off_res[far_mask_off]))
        # AR template = OFF - ON (palette space; positive where AR subtracts).
        ar_template = prof_off_res - prof_on_res
        # Peak + extent
        peak_idx = int(np.nanargmax(ar_template))
        peak_palette = float(ar_template[peak_idx])
        peak_r_mm = float(r_grid_ref[peak_idx])
        extent_mm = find_extent_mm(r_grid_ref, ar_template, threshold_frac=0.05)
        # Area under the excess (palette·mm) — proxy for total subtracted signal
        valid = np.isfinite(ar_template) & (ar_template > 0)
        area_palette_mm = float(np.nansum(ar_template[valid]) * pix_mm_ref)
        # Save per-pair CSV
        csv_path = args.out_dir / f"ar_template_pair_{i:02d}_g{int(gain):02d}_d{int(diam):02d}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["r_mm", "palette_off", "palette_on", "palette_diff"])
            for j in range(len(r_grid_ref)):
                w.writerow([f"{r_grid_ref[j]:.4f}",
                             f"{prof_off_res[j]:.3f}",
                             f"{prof_on_res[j]:.3f}",
                             f"{ar_template[j]:.3f}"])
        per_pair_records.append({
            "pair_index": i,
            "gain": gain,
            "diameter_mm": diam,
            "file_off": m_off.file,
            "file_on": m_on.file,
            "pixel_spacing_mm": pix_mm_ref,
            "peak_palette": peak_palette,
            "peak_r_mm": peak_r_mm,
            "extent_mm": extent_mm,
            "area_palette_mm": area_palette_mm,
            "speckle_floor_off": speckle_floor_off,
            "speckle_floor_on": speckle_floor_on,
            "csv": csv_path.name,
        })
        profiles_palette.append(ar_template)

    profiles_arr = np.stack(profiles_palette).astype(np.float32)
    np.save(args.out_dir / "ar_template_by_pair.npy", profiles_arr)

    # Build a canonical AR template: median of the pairs at canonical_gain
    canonical_pairs = [i for i, rec in enumerate(per_pair_records)
                        if abs(rec["gain"] - args.canonical_gain) < 0.5]
    if canonical_pairs:
        canonical_template = np.nanmedian(profiles_arr[canonical_pairs], axis=0)
    else:
        canonical_template = np.nanmedian(profiles_arr, axis=0)
    np.save(args.out_dir / "ar_template_canonical.npy",
            canonical_template.astype(np.float32))

    # Summary JSON
    payload = {
        "dataset": str(args.dataset),
        "r_max_mm": args.r_max_mm,
        "canonical_gain": args.canonical_gain,
        "r_grid_mm": r_grid_ref.tolist(),
        "pixel_spacing_mm": float(pix_mm_ref),
        "pairs": per_pair_records,
    }
    with (args.out_dir / "ar_template.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # Diagnostic plot
    if plt is not None:
        n = len(pairs)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        colors = plt.cm.viridis(np.linspace(0.0, 0.85, n))
        # Top: AR-off and AR-on per pair
        for i, (key, (img_off, m_off), (img_on, m_on)) in enumerate(pairs):
            gain, diam = key
            pix_mm = m_off.pixel_spacing_mm
            r_grid_pair, prof_off, _ = radial_profile(img_off, pix_mm, args.r_max_mm)
            _,            prof_on,  _ = radial_profile(img_on,  pix_mm, args.r_max_mm)
            label_off = f"OFF g{int(gain)} D{int(diam)}"
            label_on = f"ON  g{int(gain)} D{int(diam)}"
            axes[0].plot(r_grid_pair, prof_off, "-", color=colors[i],
                         alpha=0.9, lw=1.5, label=label_off)
            axes[0].plot(r_grid_pair, prof_on, "--", color=colors[i],
                         alpha=0.7, lw=1.0, label=label_on)
        axes[0].axhline(REJECT_PALETTE, color="red", lw=0.8, ls=":")
        axes[0].set_ylabel("palette (median over theta)")
        axes[0].set_title("AR-off vs AR-on radial profiles (Capture C pairs)")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        axes[0].grid(alpha=0.3)
        # Bottom: AR template per pair + canonical
        for i, rec in enumerate(per_pair_records):
            axes[1].plot(r_grid_ref, profiles_arr[i], "-",
                         color=colors[i], lw=1.2,
                         label=f"AR delta g{int(rec['gain'])} D{int(rec['diameter_mm'])}")
        axes[1].plot(r_grid_ref, canonical_template, "k-", lw=2.0,
                     label="canonical (median of canonical gain)")
        axes[1].set_xlabel("radial distance from catheter centre (mm)")
        axes[1].set_ylabel("AR template = OFF - ON  (palette)")
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "ar_template_overview.png", dpi=140)
        plt.close(fig)

    # Print a tidy summary
    print(f"wrote {args.out_dir / 'ar_template.json'}")
    print(f"wrote {args.out_dir / 'ar_template_by_pair.npy'}  shape={profiles_arr.shape}")
    print(f"wrote {args.out_dir / 'ar_template_canonical.npy'}  shape={canonical_template.shape}")
    if plt is not None:
        print(f"wrote {args.out_dir / 'ar_template_overview.png'}")
    print(f"\nPer-pair summary:")
    print(f"  {'i':>2}  {'gain':>5}  {'diam':>5}  {'peak_p':>7}  {'peak_r':>7}  "
          f"{'extent':>7}  {'area':>9}  {'fl_off':>7}  {'fl_on':>7}")
    for rec in per_pair_records:
        print(f"  {rec['pair_index']:>2}  {rec['gain']:>5.0f}  {rec['diameter_mm']:>5.0f}  "
              f"{rec['peak_palette']:>7.1f}  {rec['peak_r_mm']:>7.2f}  "
              f"{rec['extent_mm']:>7.2f}  {rec['area_palette_mm']:>9.1f}  "
              f"{rec['speckle_floor_off']:>7.1f}  {rec['speckle_floor_on']:>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
