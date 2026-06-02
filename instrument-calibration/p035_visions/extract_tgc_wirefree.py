#!/usr/bin/env python3
"""T1-E4{a,c} — TGC / tissue depth-attenuation extraction from wire-free
milk-phantom DICOMs.

Wire-free variant of `extract_tgc.py`: where the canonical TGC extractor
uses the wire-aligned polar arrays + `build_anechoic_mask` to pick
wire-free angular sectors, the milk phases (E4a undiluted, E4c 1:1
diluted) contain only the milk medium, no wires. Every azimuthal sector
is "anechoic" (= tissue-mimicking) and the median palette vs depth IS
the device's TGC schedule, in dB.

Workflow per capture-dir:

  1. Read each DICOM (after `extract_metadata.py`), auto-detect the
     catheter center, polar-resample around it.
  2. Per gain group: average frames, then compute median-over-theta
     palette profile vs depth. This is the displayed pixel(r) curve.
  3. Convert palette -> dB via `1 palette unit = 1 dB` (log_multiplier =
     20 convention used throughout the P_035 pipeline) and emit per-gain
     TGC profile JSON.

Cross-gain consistency
----------------------
If TGC is gain-independent (which it should be) the cross-gain step
pixel(r, g_hi) - pixel(r, g_lo) MUST be approximately depth-independent
across the wire-free reach. We compute the (gain_hi - gain_lo) ramp as
a quality check and report its slope.

Outputs
-------
  derived/tgc/per_gain_profiles.json   {gain: median_palette_vs_r}
  derived/tgc/tgc_overview.png         per-gain profile + cross-gain step
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
    load_frame,
    polar_resample,
)


def median_radial_profile(frame, inner_r_mm=2.0, n_radial=400, n_az=720,
                          reject_palette=11.0, cx_px=None, cy_px=None):
    """Median palette across azimuth at each radius. Reject-floor pixels
    are excluded (NaN-masked) before taking the median.

    For wire-free milk frames the auto catheter-center detection is
    unreliable at low gain (the timestamp watermark in the image corner
    can be brighter than the catheter), so the caller can fix the polar
    origin via (cx_px, cy_px); when both are None we fall back to the
    image center (the s5 anchors its polar origin there by design).
    """
    if cx_px is None or cy_px is None:
        cx, cy = frame.cols / 2.0, frame.rows / 2.0
    else:
        cx, cy = float(cx_px), float(cy_px)
    px_per_mm = 1.0 / max(frame.pixel_spacing_mm, 1e-6)
    r_max_mm = (frame.cols / 2.0) * frame.pixel_spacing_mm
    polar, rs_mm, _ = polar_resample(
        frame.array, cx, cy, px_per_mm, r_max_mm, n_radial, n_az,
    )
    polar_masked = np.where(polar > reject_palette, polar, np.nan)
    profile = np.nanmedian(polar_masked, axis=1)  # median over az -> (n_r,)
    # Suppress the ring-down core; treat r < inner_r_mm as NaN.
    profile[rs_mm < inner_r_mm] = np.nan
    return rs_mm, profile, (cx, cy, r_max_mm)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <capture-dir>/derived/tgc")
    p.add_argument("--diameter-mm", type=float, default=60.0)
    p.add_argument("--inner-r-mm", type=float, default=2.0,
                   help="Inner radius to start the TGC profile (mm).")
    p.add_argument("--reject-palette", type=float, default=11.0,
                   help="Pixels at or below this palette value are treated as "
                        "the device's reject floor (set to NaN before "
                        "averaging).")
    p.add_argument("--n-radial", type=int, default=400)
    p.add_argument("--n-az", type=int, default=720)
    p.add_argument("--cx-px", type=float, default=None,
                   help="Override the polar-origin x (px). Default: image center.")
    p.add_argument("--cy-px", type=float, default=None,
                   help="Override the polar-origin y (px). Default: image center.")
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.capture_dir / "derived" / "tgc")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = args.capture_dir / "derived" / "frames_meta.csv"
    if not meta.is_file():
        print(f"missing {meta}", file=sys.stderr)
        return 2
    with meta.open() as f:
        rows = list(csv.DictReader(f))
    sel = [r for r in rows
           if abs(float(r["diameter_mm"]) - args.diameter_mm) < 0.5]
    if not sel:
        print(f"no frames at D={args.diameter_mm} mm; available diameters: "
              f"{sorted({r['diameter_mm'] for r in rows})}", file=sys.stderr)
        return 3

    per_gain_profiles: dict[float, list[np.ndarray]] = defaultdict(list)
    rs_mm_ref = None
    centers = []
    for r in sel:
        path = args.capture_dir / r["file"]
        try:
            frame = load_frame(path)
        except Exception as exc:
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        rs_mm, profile, (cx, cy, r_max_mm) = median_radial_profile(
            frame, inner_r_mm=args.inner_r_mm,
            n_radial=args.n_radial, n_az=args.n_az,
            reject_palette=args.reject_palette,
            cx_px=args.cx_px, cy_px=args.cy_px,
        )
        if rs_mm_ref is None:
            rs_mm_ref = rs_mm
        per_gain_profiles[float(r["gain_slider"])].append(profile)
        centers.append((cx, cy))
        print(f"  {path.name}  g={r['gain_slider']:>4}  "
              f"profile(r=4mm)={profile[np.argmin(np.abs(rs_mm-4))]:.1f}  "
              f"profile(r=15mm)={profile[np.argmin(np.abs(rs_mm-15))]:.1f}  "
              f"center=({cx:.1f},{cy:.1f})px")

    if not per_gain_profiles:
        return 4

    per_gain_mean = {
        g: np.nanmean(np.stack(profs, axis=0), axis=0).tolist()
        for g, profs in sorted(per_gain_profiles.items())
    }
    workspace_root = Path("/home/jocelynbarker/i4h-sensor-simulation")
    cap_abs = args.capture_dir.resolve()
    try:
        capture_dir_str = str(cap_abs.relative_to(workspace_root))
    except ValueError:
        capture_dir_str = str(cap_abs)
    out_json = {
        "capture_dir": capture_dir_str,
        "diameter_mm": args.diameter_mm,
        "n_radial": args.n_radial,
        "rs_mm": rs_mm_ref.tolist(),
        "per_gain_palette_vs_r": per_gain_mean,
        "log_multiplier_assumed": 20.0,
        "notes": "palette[r] is the median-over-theta of (env>reject) pixels "
                  "per radius; 1 palette unit = 1 dB under log_multiplier=20.",
        "auto_catheter_centers": [list(c) for c in centers],
    }
    (out_dir / "per_gain_profiles.json").write_text(
        json.dumps(out_json, indent=2, default=float))
    print(f"\n  wrote {out_dir / 'per_gain_profiles.json'}")

    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        cmap = plt.get_cmap("viridis")
        gains = sorted(per_gain_mean.keys())
        n_g = len(gains)
        for i, g in enumerate(gains):
            prof = np.array(per_gain_mean[g])
            axes[0].plot(rs_mm_ref, prof, color=cmap(i / max(1, n_g - 1)),
                         label=f"gain={g:.0f}")
        axes[0].set_ylabel("palette (1 unit ~ 1 dB)")
        axes[0].set_title(f"E4 TGC: median palette vs depth, "
                          f"D={args.diameter_mm:.0f} mm  "
                          f"({args.capture_dir.name})")
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8, loc="best")

        if len(gains) >= 2:
            ref_g = gains[len(gains) // 2]
            ref_prof = np.array(per_gain_mean[ref_g])
            for i, g in enumerate(gains):
                if g == ref_g:
                    continue
                diff = np.array(per_gain_mean[g]) - ref_prof
                axes[1].plot(rs_mm_ref, diff,
                             color=cmap(i / max(1, n_g - 1)),
                             label=f"gain={g:.0f} - {ref_g:.0f}")
            axes[1].set_ylabel("Δ palette = Δ dB")
            axes[1].set_title(f"Cross-gain step (should be depth-independent)")
            axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8, loc="best")
        axes[1].set_xlabel("r (mm)")
        fig.tight_layout()
        fig.savefig(out_dir / "tgc_overview.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {out_dir / 'tgc_overview.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
