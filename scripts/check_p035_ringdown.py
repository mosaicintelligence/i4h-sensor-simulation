#!/usr/bin/env python3
"""Compare the ringdown-region (inner-disc) brightness in:

  (1) every file of the original P_035_PointScatter dataset
  (2) the ivus_test_0508 AR-on / AR-off pairs

If P_035's frames look like the new dataset's mode_flag=0 (AR-OFF, bright
inner) we have a problem with the YAML's `subtract_reference: true`.
If they look like mode_flag=1 (AR-ON, near-noise inner) the YAML is fine
and we just need to re-document mode_flag's meaning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

P035 = Path("/Users/jocelynbarker/ivus-sim/P_035_PointScatter")
NEW = Path("/Users/jocelynbarker/ivus-sim/ivus_test_0508")


def stats(path: Path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr.mean(axis=-1)
    if arr.ndim == 3:
        f0 = arr[arr.shape[0] // 2]
    else:
        f0 = arr
    H, W = f0.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)
    rmax = min(H, W) / 2.0
    inner = float(f0[r < 0.10 * rmax].mean())
    mid = float(f0[(r >= 0.30 * rmax) & (r < 0.45 * rmax)].mean())
    outer = float(f0[(r >= 0.70 * rmax) & (r < 0.85 * rmax)].mean())
    gain = ds.get((0x0029, 0x1001)).value if (0x0029, 0x1001) in ds else ""
    diam = ds.get((0x0029, 0x1003)).value if (0x0029, 0x1003) in ds else ""
    mode = ds.get((0x0029, 0x1007)).value if (0x0029, 0x1007) in ds else ""
    return gain, diam, mode, inner, mid, outer


def main():
    print(f"{'file':<40} {'gain':>5} {'diam':>5} {'mode':>4} "
          f"{'inner':>7} {'mid':>7} {'outer':>7}  inner/mid")
    print(f"{'':-<40} {'':->5} {'':->5} {'':->4} {'':->7} {'':->7} {'':->7}")
    print("# P_035_PointScatter")
    for f in sorted(P035.glob("FILE*")):
        gain, diam, mode, inner, mid, outer = stats(f)
        ratio = inner / mid if mid > 0 else 0.0
        print(f"{f.name:<40} {gain!s:>5} {diam!s:>5} {mode!s:>4} "
              f"{inner:>7.2f} {mid:>7.2f} {outer:>7.2f}  {ratio:>5.2f}x")
    print("\n# ivus_test_0508 paired Capture-C frames (reference)")
    for case, fname in [
        ("CASE0000", "FILE0000"),  # mode=0
        ("CASE0000", "FILE0001"),  # mode=1
        ("CASE0000", "FILE0002"),  # mode=0
        ("CASE0000", "FILE0003"),  # mode=1
        ("CASE0007", "FILE0000"),  # mode=0
        ("CASE0007", "FILE0001"),  # mode=1
    ]:
        gain, diam, mode, inner, mid, outer = stats(NEW / case / fname)
        ratio = inner / mid if mid > 0 else 0.0
        print(f"{case+'/'+fname:<40} {gain!s:>5} {diam!s:>5} {mode!s:>4} "
              f"{inner:>7.2f} {mid:>7.2f} {outer:>7.2f}  {ratio:>5.2f}x")


if __name__ == "__main__":
    main()
