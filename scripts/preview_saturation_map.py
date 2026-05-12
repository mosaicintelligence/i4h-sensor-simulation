#!/usr/bin/env python3
"""Preview the saturation map per sweep: for each gain step, report the
99.5th-percentile palette and the fraction of pixels >= 235 (saturated) on
a mid-frame, plus the median palette inside the brightest local maximum
(proxy for the brightest wire's peak). Helps decide:

  - Lowest gain at which signal is comfortably above the reject floor (~11)
  - Highest gain below which the inner wires are still resolvable (peak <230)

This is a quick read of the captured data; the proper E7 fit will replace
these heuristics with a real log_multiplier + slider->dB curve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

ROOT = Path("/Users/jocelynbarker/ivus-sim/ivus_test_0508/raw")
SWEEPS = ["sweep1_orig_60mm", "sweep2_new_60mm", "sweep3_new_30mm"]


def palette_stats(ds: pydicom.Dataset) -> dict:
    arr = ds.pixel_array
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr.mean(axis=-1)
    f0 = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr
    H, W = f0.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)
    rmax = min(H, W) / 2.0
    # Mask: ignore inner ringdown (r < 0.10 rmax) and the corners (r > rmax).
    mask = (r >= 0.12 * rmax) & (r <= rmax)
    pix = f0[mask]
    p995 = float(np.percentile(pix, 99.5))
    sat = float((pix >= 235).mean())
    p50 = float(np.percentile(pix, 50))
    p99 = float(np.percentile(pix, 99))
    # Inner-wire band (5-10 mm physical), outer (20-25 mm).
    return {
        "median": p50,
        "p99": p99,
        "p99.5": p995,
        "frac_saturated": sat,
        "global_mean": float(f0.mean()),
    }


def main() -> int:
    print(f"{'sweep':<22} {'gain':>5} {'diam':>5} {'med':>6} {'p99':>6} "
          f"{'p99.5':>6} {'sat%':>6} {'verdict'}")
    print("-" * 78)
    for sweep in SWEEPS:
        sweep_dir = ROOT / sweep
        files = sorted(sweep_dir.glob("FILE*.dcm"))
        for f in files:
            ds = pydicom.dcmread(str(f), force=True)
            gain = ds.get((0x0029, 0x1001)).value if (0x0029, 0x1001) in ds else None
            diam = ds.get((0x0029, 0x1003)).value if (0x0029, 0x1003) in ds else None
            s = palette_stats(ds)
            verdict_parts = []
            if s["p99.5"] >= 235:
                verdict_parts.append("SAT (inner wires clipped)")
            elif s["p99.5"] <= 30:
                verdict_parts.append("noise-dom (no signal)")
            else:
                verdict_parts.append(f"clean p99.5={s['p99.5']:.0f}")
            verdict = " ".join(verdict_parts)
            print(f"{sweep:<22} {gain!s:>5} {diam!s:>5} "
                  f"{s['median']:>6.1f} {s['p99']:>6.1f} {s['p99.5']:>6.1f} "
                  f"{100*s['frac_saturated']:>5.2f}%  {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
