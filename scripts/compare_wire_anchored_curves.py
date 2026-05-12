#!/usr/bin/env python3
"""Cross-sweep comparison of the WIRE-ANCHORED E7 gain curves, plus a
pooled fit that combines Sweep #1 + Sweep #3 (single-orientation each).

Sweep #2 is shown but excluded from the pooled fit because its mid-sweep
catheter rotation introduces a non-monotonic step (the ~77 deg rotation
between gain 28 and gain 32 drops the median wire peak by ~4 dB-
equivalent -- a real-but-orthogonal effect to the device's gain
response).

Outputs:
    ivus_test_0508/gain_curve_wire_comparison.png
    ivus_test_0508/gain_curve_wire_summary.csv
    ivus_test_0508/gain_curve_wire_pooled.json    (canonical pooled fit)
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
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "instrument-calibration" / "p035_visions"))
from extract_gain_curve import (  # noqa: E402
    REJECT_PALETTE, SATURATION_PALETTE, fit_curve, FrameRow,
)


SWEEPS = [
    ("sweep1_orig_60mm", "Sweep #1 (orig, D=60, theta0=288 deg)", "C0", False),
    ("sweep2_new_60mm",  "Sweep #2 (new, D=60, 142 deg + 65 deg)", "C1", False),
    ("sweep3_new_30mm",  "Sweep #3 (new, D=30, theta0=144 deg)",   "C2", True),
]


def load_curves() -> dict:
    out = {}
    for sub, label, color, in_pool in SWEEPS:
        p = REPO / "ivus_test_0508" / "raw" / sub / "derived" / \
            "gain_curve_wire" / "gain_curve.json"
        d = json.load(p.open())
        out[sub] = {"label": label, "color": color, "in_pool": in_pool,
                    "frames": d["frames"], "fit": d["fit"]}
    return out


def make_pool(curves: dict) -> list[FrameRow]:
    pool: list[FrameRow] = []
    for sub, meta in curves.items():
        if not meta["in_pool"]:
            continue
        for fr in meta["frames"]:
            pool.append(FrameRow(
                file=f"{sub}/{fr['file']}",
                gain_slider=fr["gain_slider"],
                diameter_mm=fr["diameter_mm"],
                ar_state=int(fr.get("ar_state") or 0),
                anchor_palette=fr["anchor_palette"],
                rows=fr.get("rows", 0), cols=fr.get("cols", 0),
                wire_peaks=fr.get("wire_peaks"),
                n_wires_used=int(fr.get("n_wires_used") or 0),
                theta0_deg=fr.get("theta0_deg"),
            ))
    return pool


def write_summary_csv(curves: dict, pool_fit: dict, out_csv: Path) -> None:
    fields = ["sweep", "anchor_method", "n_total", "n_usable",
              "log_multiplier_canonical", "log_multiplier_iqr",
              "reject_palette_observed", "saturation_palette_observed",
              "dynamic_range_db", "reference_slider"]
    rows = []
    for sub, meta in curves.items():
        f = meta["fit"]
        rows.append({
            "sweep": sub,
            "anchor_method": "wire",
            "n_total": f["n_total"],
            "n_usable": f["n_usable"],
            "log_multiplier_canonical": round(f["log_multiplier_canonical"], 2),
            "log_multiplier_iqr": round(f["log_multiplier_iqr"], 2),
            "reject_palette_observed": round(f["reject_palette_observed"], 1),
            "saturation_palette_observed": round(f["saturation_palette_observed"], 1),
            "dynamic_range_db": round(f["dynamic_range_db"], 2),
            "reference_slider": round(f["reference_slider"], 1),
        })
    rows.append({
        "sweep": "POOLED(sweep1+sweep3)",
        "anchor_method": "wire",
        "n_total": pool_fit["n_total"],
        "n_usable": pool_fit["n_usable"],
        "log_multiplier_canonical": round(pool_fit["log_multiplier_canonical"], 2),
        "log_multiplier_iqr": round(pool_fit["log_multiplier_iqr"], 2),
        "reject_palette_observed": round(pool_fit["reject_palette_observed"], 1),
        "saturation_palette_observed": round(pool_fit["saturation_palette_observed"], 1),
        "dynamic_range_db": round(pool_fit["dynamic_range_db"], 2),
        "reference_slider": round(pool_fit["reference_slider"], 1),
    })
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def plot(curves: dict, pool_rows: list[FrameRow], pool_fit: dict,
         out_png: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: anchor vs slider per sweep + pooled fit
    for sub, meta in curves.items():
        slid = np.array([fr["gain_slider"] for fr in meta["frames"]])
        anc = np.array([fr["anchor_palette"] for fr in meta["frames"]])
        marker = "o" if meta["in_pool"] else "s"
        ax_a.scatter(slid, anc, s=46, marker=marker, color=meta["color"],
                     label=meta["label"], edgecolor="black", linewidth=0.5)
        # Per-sweep PCHIP curve (faint)
        if "fit_anchor" in meta["fit"]:
            ax_a.plot(slid, meta["fit"]["fit_anchor"], color=meta["color"],
                      ls=":", lw=1.0, alpha=0.55)

    # Pooled fit overlay
    pool_sliders = np.array([r.gain_slider for r in pool_rows])
    pool_anchors = np.array([r.anchor_palette for r in pool_rows])
    if "fit_anchor" in pool_fit:
        order = np.argsort(pool_sliders)
        ax_a.plot(pool_sliders[order], np.array(pool_fit["fit_anchor"])[order],
                  "k-", lw=1.8, label="pooled (sweep1+sweep3) PCHIP")

    ax_a.axhline(REJECT_PALETTE, color="red", lw=0.7, ls=":")
    ax_a.axhline(SATURATION_PALETTE, color="red", lw=0.7, ls=":")
    ax_a.set_xlabel("slider gain")
    ax_a.set_ylabel("median wire-peak palette  (anchor)")
    ax_a.set_title("E7 wire-anchored gain curve")
    ax_a.legend(loc="upper left", fontsize=9)
    ax_a.grid(alpha=0.3)

    # Right: local log_multiplier vs slider (slope) per sweep + pooled
    for sub, meta in curves.items():
        slid = np.array([fr["gain_slider"] for fr in meta["frames"]])
        fit = meta["fit"]
        if "effective_log_multiplier" not in fit:
            continue
        lm = np.array(fit["effective_log_multiplier"])
        mask = np.array(fit["fit_mask"], dtype=bool)
        ax_b.plot(slid[mask], lm[mask], marker="o", color=meta["color"],
                  lw=1.0, label=meta["label"])
    if "effective_log_multiplier" in pool_fit:
        slid = np.sort(np.unique(pool_sliders))
        # Re-map pool lm to plot — it's stored on the union slider grid
        lm = np.array(pool_fit["effective_log_multiplier"])
        mask = np.array(pool_fit["fit_mask"], dtype=bool)
        s_grid = np.array(pool_fit["sliders"])
        ax_b.plot(s_grid[mask], lm[mask], "k-", lw=1.8,
                  marker="x", label="pooled")
    ax_b.set_xlabel("slider gain")
    ax_b.set_ylabel("effective log_multiplier (palette per dB)")
    ax_b.set_title("Local compression slope")
    ax_b.legend(loc="best", fontsize=9)
    ax_b.grid(alpha=0.3)

    fig.suptitle("Wire-anchored E7: cross-sweep comparison "
                 "(Sweep #2 excluded from pool due to mid-sweep rotation)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_dir = REPO / "ivus_test_0508"
    curves = load_curves()
    pool_rows = make_pool(curves)
    pool_fit = fit_curve(pool_rows, reference_slider=None)
    pool_payload = {
        "anchor_method": "wire",
        "pool_sources": [s for s, _, _, in_pool in SWEEPS if in_pool],
        "exclude_sources": [s for s, _, _, in_pool in SWEEPS if not in_pool],
        "exclude_reason": (
            "Canonical curve uses Sweep #3 alone: most frames (12), "
            "widest gain coverage (8-52), single orientation (144 deg), "
            "and the new apparatus we're modeling. Sweep #1 is plotted "
            "as orig-apparatus cross-check but its baseline palette is "
            "offset (different apparatus geometry and wire set, so "
            "absolute median peak palette differs at the same slider). "
            "Sweep #2 has a mid-sweep ~77 deg rotation that introduces "
            "a non-monotonic step (~4-5 dB-equivalent), quantified "
            "separately."),
        "n_frames_pooled": len(pool_rows),
        "fit": pool_fit,
    }
    (out_dir / "gain_curve_wire_pooled.json").write_text(
        json.dumps(pool_payload, indent=2) + "\n")
    write_summary_csv(curves, pool_fit, out_dir / "gain_curve_wire_summary.csv")
    plot(curves, pool_rows, pool_fit, out_dir / "gain_curve_wire_comparison.png")
    print(f"wrote {out_dir / 'gain_curve_wire_comparison.png'}")
    print(f"wrote {out_dir / 'gain_curve_wire_summary.csv'}")
    print(f"wrote {out_dir / 'gain_curve_wire_pooled.json'}")
    print()
    print("=== Pooled (Sweep #1 + Sweep #3) wire-anchored fit ===")
    print(f"  n usable / total       : {pool_fit['n_usable']} / {pool_fit['n_total']}")
    print(f"  log_multiplier (canon) : {pool_fit['log_multiplier_canonical']:.1f} "
          f"+/- IQR {pool_fit['log_multiplier_iqr']:.1f}")
    print(f"  reject_palette         : {pool_fit['reject_palette_observed']:.1f}")
    print(f"  saturation_palette     : {pool_fit['saturation_palette_observed']:.1f}")
    print(f"  dynamic_range_db       : {pool_fit['dynamic_range_db']:.1f}")
    print(f"  reference_slider       : {pool_fit['reference_slider']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
