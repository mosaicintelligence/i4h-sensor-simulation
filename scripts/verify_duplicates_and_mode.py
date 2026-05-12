#!/usr/bin/env python3
"""Sanity-check two things on ivus_test_0508:

(A) Are CASE0000 and CASE0001 byte-identical pixel data (just two exports)?
    Hash the PixelData payload of each FILE pair and compare.

(B) When mode_flag (0x0029,0x1007) flips 0 -> 1 in a paired capture
    (same gain, same diameter, taken seconds apart), how does the displayed
    image differ? Compare global mean + ringdown-region mean to see if
    mode_flag=1 looks like AR-OFF (more energy near r=0) or just a re-take.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path("/Users/jocelynbarker/ivus-sim/ivus_test_0508")


def pixel_hash(ds: pydicom.Dataset) -> str:
    arr = ds.pixel_array
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def stats(ds: pydicom.Dataset) -> dict[str, float]:
    arr = ds.pixel_array  # shape (frames, H, W) for multi-frame
    if arr.ndim == 4:
        # palette-color: take Y plane via mean of RGB or first channel
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr.mean(axis=-1)
    if arr.ndim == 3:
        f0 = arr[arr.shape[0] // 2]  # mid frame
    else:
        f0 = arr
    H, W = f0.shape
    cy, cx = H // 2, W // 2
    # Inner-disc mean (catheter ringdown region, r < 0.10 * H/2):
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)
    rmax = min(H, W) / 2.0
    inner = float(f0[r < 0.10 * rmax].mean())
    mid = float(f0[(r >= 0.30 * rmax) & (r < 0.45 * rmax)].mean())
    outer = float(f0[(r >= 0.70 * rmax) & (r < 0.85 * rmax)].mean())
    return {"global": float(f0.mean()), "inner": inner, "mid": mid, "outer": outer}


def section_a():
    print("\n## (A) CASE0000 vs CASE0001 pixel-data hash")
    case0 = sorted((ROOT / "CASE0000").glob("FILE*"))
    case1 = sorted((ROOT / "CASE0001").glob("FILE*"))
    for f0, f1 in zip(case0, case1):
        ds0 = pydicom.dcmread(str(f0), force=True)
        ds1 = pydicom.dcmread(str(f1), force=True)
        h0, h1 = pixel_hash(ds0), pixel_hash(ds1)
        ok = "EQUAL " if h0 == h1 else "DIFFER"
        print(f"  {f0.name}  {h0} vs {h1}  {ok}")


def section_b():
    print("\n## (B) mode_flag-paired captures: ringdown-region intensity")
    print(f"  {'case/file':<22} {'gain':>5} {'diam':>5} {'mode':>4} "
          f"{'global':>7} {'inner':>7} {'mid':>7} {'outer':>7}")
    targets = [
        ("CASE0000", "FILE0000", "FILE0001"),  # D=30, g=50, mode 0->1
        ("CASE0000", "FILE0002", "FILE0003"),  # D=60, g=50, mode 0->1
        ("CASE0000", "FILE0004", "FILE0005"),  # D=60, g=40, mode 0->1
        ("CASE0007", "FILE0000", "FILE0001"),  # D=60, g=44, mode 0->1
    ]
    for case, f_a, f_b in targets:
        for f_name in (f_a, f_b):
            p = ROOT / case / f_name
            ds = pydicom.dcmread(str(p), force=True)
            gain = ds.get((0x0029, 0x1001)).value if (0x0029, 0x1001) in ds else ""
            diam = ds.get((0x0029, 0x1003)).value if (0x0029, 0x1003) in ds else ""
            mode = ds.get((0x0029, 0x1007)).value if (0x0029, 0x1007) in ds else ""
            s = stats(ds)
            print(f"  {case}/{f_name:<10}  {gain:>5} {diam:>5} {mode!s:>4} "
                  f"{s['global']:>7.2f} {s['inner']:>7.2f} {s['mid']:>7.2f} {s['outer']:>7.2f}")
        print()


if __name__ == "__main__":
    section_a()
    section_b()
