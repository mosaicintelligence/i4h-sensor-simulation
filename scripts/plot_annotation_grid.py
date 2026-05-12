#!/usr/bin/env python3
"""Per-sweep composite plot: every annotated frame with clicks + fitted wires.

For each of the three E2 sweeps, draws one matplotlib figure with N subplots
(one per fitted frame), each showing:
    - the median-collapsed B-mode image (greyscale)
    - the user's clicks (yellow circles with labels)
    - the predicted (FITTED) wire positions (magenta crosses with w# labels)
    - the apparatus center (orange) and device center (cyan)
    - the FOV circle (green)

Title per panel: FILE####  gain=X  theta0=Y°  scale=Z  rms=W mm  n_clicks=K
Saves the figure to <capture-dir>/derived/annotation_grid.png.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
Path("/tmp/mpl-cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "instrument-calibration" / "p035_visions"))
from extract_calibration_inputs import (  # noqa: E402
    load_frame,
    parse_scattering_box_dxf,
)


SWEEPS = [
    ("sweep1_orig_60mm", "IVUS Scattering Box - Sketch 1.dxf"),
    ("sweep2_new_60mm",  "wire_spiral_v1.dxf"),
    ("sweep3_new_30mm",  "wire_spiral_v1.dxf"),
]


def load_fits(align_csv: Path) -> dict[str, dict]:
    out = {}
    with align_csv.open() as f:
        for row in csv.DictReader(f):
            out[row["file"]] = row
    return out


def render_sweep(capture_dir: Path, dxf_name: str, out_path: Path) -> None:
    wires, geom = parse_scattering_box_dxf(capture_dir / dxf_name)
    ann_path = capture_dir / "derived" / "wire_annotations.json"
    fit_path = capture_dir / "derived" / "alignment_fit.csv"
    annotations = json.load(ann_path.open())["frames"]
    fits = load_fits(fit_path)

    # Show all frames the user touched (clicked OR marked unreadable, OR fit exists).
    panels = []
    for a in sorted(annotations, key=lambda x: x["file"]):
        name = a["file"]
        clicks = a.get("clicks") or []
        if not clicks and not fits.get(name):
            continue
        panels.append((name, a, fits.get(name)))

    n = len(panels)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.0 * nrow))
    if nrow == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, (name, ann, fit) in enumerate(panels):
        ax = axes_flat[i]
        fr = load_frame(capture_dir / name)
        ax.imshow(fr.array, cmap="gray", vmin=0, vmax=255, origin="upper")

        cx_dev, cy_dev = fr.cols / 2.0, fr.rows / 2.0
        px_per_mm = 1.0 / fr.pixel_spacing_mm

        # Device center + catheter outline (cyan)
        ax.plot(cx_dev, cy_dev, "+", color="cyan", markersize=10, mew=1.2)
        circ = patches.Circle((cx_dev, cy_dev),
                              geom.catheter_radius_mm * px_per_mm,
                              fill=False, edgecolor="cyan", lw=0.8)
        ax.add_patch(circ)

        # FOV circle (green)
        fov_r_mm = min(geom.outer_radius_mm, fr.cols * fr.pixel_spacing_mm / 2.0)
        circ = patches.Circle((cx_dev, cy_dev), fov_r_mm * px_per_mm,
                              fill=False, edgecolor="lime", lw=0.6, ls="--")
        ax.add_patch(circ)

        title_extra = "  (no fit)"
        if fit is not None:
            apx, apy = float(fit["apparatus_cx_px"]), float(fit["apparatus_cy_px"])
            theta0 = float(fit["theta0_deg"])
            chir = int(fit["chirality"])
            scale = float(fit["radial_scale"])
            rms = float(fit["rms_residual_mm"])
            gain = 4 * int(name.replace("FILE", "").replace(".dcm", ""))

            # Apparatus center (orange X)
            ax.plot(apx, apy, "x", color="orange", markersize=8, mew=1.4)

            # Predicted (FITTED) wire positions (magenta crosses)
            for w in wires:
                a_world = math.radians(chir * w.theta_deg + theta0)
                px = apx + w.r_mm * scale * px_per_mm * math.cos(a_world)
                py = apy - w.r_mm * scale * px_per_mm * math.sin(a_world)
                if not (0 <= px < fr.cols and 0 <= py < fr.rows):
                    continue
                ax.plot(px, py, "+", color="magenta", markersize=9, mew=1.2)
                ax.text(px + 4, py - 4, f"w{w.index}", color="magenta",
                        fontsize=6.5)
            title_extra = (f"  th0={theta0:+.1f}  s={scale:.3f}"
                           f"  rms={rms:.2f}mm")
            if rms > 1.5:
                title_extra += "  DROP"
        else:
            gain_idx = int(name.replace("FILE", "").replace(".dcm", ""))
            gain = 4 * gain_idx

        # User clicks (yellow circles with index labels)
        for c in clicks:
            wi = c.get("wire_index")
            if wi is None:
                continue
            ax.plot(c["x_px"], c["y_px"], "o", mfc="none", mec="yellow",
                    mew=1.2, ms=10)
            ax.text(c["x_px"] + 5, c["y_px"] + 7, f"u{wi}",
                    color="yellow", fontsize=6.5)

        used_clicks = sum(1 for c in clicks if c.get("wire_index") is not None)
        unread_flag = " UNREAD" if ann.get("unreadable") else ""
        ax.set_title(f"{name}  g={gain}  n={used_clicks}{unread_flag}{title_extra}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{capture_dir.name}  ({dxf_name})", fontsize=11, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    base = REPO / "ivus_test_0508" / "raw"
    for sub, dxf_name in SWEEPS:
        cap_dir = base / sub
        render_sweep(cap_dir, dxf_name,
                     cap_dir / "derived" / "annotation_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
