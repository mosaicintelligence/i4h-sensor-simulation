#!/usr/bin/env python3
"""Emit a DXF of the wire-spiral apparatus geometry that the existing
P_035 calibration tools (parse_scattering_box_dxf in
extract_calibration_inputs.py) can consume without changes.

The DXF contains exactly the entities the reader expects:

  * 1 outer disc circle  (radius = DISC_OD / 2, at origin)
  * 1 catheter circle    (radius = DISC_CENTER_HOLE / 2, at origin)
  * N wire-position circles (radius = WIRE_HOLE_DIA / 2, one per wire)

Run from the workspace root or anywhere -- writes to
`instrument-calibration/hardware/wire_spiral_v1.dxf`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import ezdxf

# Import the canonical layout from gen_phantom_stl.py so this DXF and the
# STL stay in lockstep. We re-import the constants (no need to import
# trimesh / shapely just to read constants).
TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))


def _load_layout():
    # Read constants without importing trimesh by parsing the file.
    src = (TOOL_DIR / "gen_phantom_stl.py").read_text()
    ns: dict = {}
    # Whitelist the small subset of constants we want.
    for line in src.splitlines():
        line = line.strip()
        if line.startswith(("DISC_OD ", "DISC_THICK ", "DISC_CENTER_HOLE ",
                             "WIRE_HOLE_DIA ", "WIRE_CB_DIA ", "WIRE_CB_DEPTH ",
                             "WIRE_LAYOUT_MM ", "WIRE_LAYOUT_MM:")):
            try:
                exec(line, ns)
            except Exception:
                pass
    if "WIRE_LAYOUT_MM" not in ns:
        # Multi-line tuple parse — find the WIRE_LAYOUT_MM block.
        i_start = src.index("WIRE_LAYOUT_MM")
        i_end = src.index(")\nN_WIRES")
        chunk = src[i_start:i_end + 1]
        chunk = chunk.replace(":", "=", 1) if "WIRE_LAYOUT_MM:" in chunk else chunk
        chunk = chunk.split("=", 1)[1].strip()
        # Strip the type annotation
        chunk = chunk.lstrip(": tuple[tuple[float, float], ...]").lstrip("=").strip()
        ns["WIRE_LAYOUT_MM"] = eval(chunk)
    return ns


def main() -> int:
    ns = _load_layout()
    DISC_OD = ns["DISC_OD"]
    DISC_CENTER_HOLE = ns["DISC_CENTER_HOLE"]
    WIRE_HOLE_DIA = ns["WIRE_HOLE_DIA"]
    WIRE_LAYOUT_MM = ns["WIRE_LAYOUT_MM"]

    out_path = TOOL_DIR.parents[0] / "hardware" / "wire_spiral_v1.dxf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new(dxfversion="R2010", units=ezdxf.units.MM)
    msp = doc.modelspace()
    msp.add_circle((0.0, 0.0), DISC_OD / 2.0)
    msp.add_circle((0.0, 0.0), DISC_CENTER_HOLE / 2.0)
    import math
    for r_mm, theta_deg in WIRE_LAYOUT_MM:
        th = math.radians(theta_deg)
        x = r_mm * math.cos(th)
        y = r_mm * math.sin(th)
        msp.add_circle((x, y), WIRE_HOLE_DIA / 2.0)

    doc.saveas(out_path)
    print(f"wrote {out_path}")
    print(f"  outer disc circle  : r = {DISC_OD/2:.3f} mm at (0, 0)")
    print(f"  catheter circle    : r = {DISC_CENTER_HOLE/2:.3f} mm at (0, 0)")
    print(f"  {len(WIRE_LAYOUT_MM)} wire circles  : r = {WIRE_HOLE_DIA/2:.3f} mm each")
    for i, (r, th) in enumerate(WIRE_LAYOUT_MM):
        print(f"    wire #{i+1:2d}: r = {r:5.2f} mm, theta = {th:5.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
