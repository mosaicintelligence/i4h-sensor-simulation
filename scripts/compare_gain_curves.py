#!/usr/bin/env python3
"""Compare the gain_curve.json outputs from the three Capture-A sweeps and
write a single multi-panel comparison plot + a tidy summary CSV.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
Path("/tmp/mpl-cache").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/jocelynbarker/ivus-sim")
SWEEPS = [
    ("sweep1_orig_60mm", "Sweep 1: ORIGINAL apparatus, D=60 (asc)"),
    ("sweep2_new_60mm",  "Sweep 2: NEW apparatus, D=60 (desc)"),
    ("sweep3_new_30mm",  "Sweep 3: NEW apparatus, D=30 (asc)"),
]


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    summary_rows = []
    colors = ["C0", "C1", "C2"]
    for color, (sweep, label) in zip(colors, SWEEPS):
        path = ROOT / "ivus_test_0508" / "raw" / sweep / "derived" / "gain_curve" / "gain_curve.json"
        with path.open() as f:
            payload = json.load(f)
        fit = payload["fit"]
        sliders = np.array(fit["sliders"])
        anchors = np.array(fit["anchors"])
        used = np.array(fit["fit_mask"])
        log_mult = np.array(fit["effective_log_multiplier"])
        # Top: compression curve
        axes[0].plot(sliders, anchors, "-o", color=color, lw=1.0, ms=4, label=label)
        axes[0].plot(sliders[used], anchors[used], "o", color=color, ms=8,
                     mfc="none", mew=2)
        # Bottom: local log_multiplier
        axes[1].plot(sliders[used], log_mult[used], "-o", color=color,
                     lw=1.0, ms=4, label=label)
        # Compute first and last clean slider for the table.
        used_sliders = sliders[used]
        summary_rows.append({
            "sweep": sweep,
            "n_total": fit["n_total"],
            "n_usable": fit["n_usable"],
            "g_clean_min": int(used_sliders.min()) if used.any() else None,
            "g_clean_max": int(used_sliders.max()) if used.any() else None,
            "log_mult_canonical": round(fit["log_multiplier_canonical"], 1),
            "log_mult_iqr": round(fit["log_multiplier_iqr"], 1),
            "log_mult_min": round(float(np.min(log_mult[used])), 1) if used.any() else None,
            "log_mult_max": round(float(np.max(log_mult[used])), 1) if used.any() else None,
            "log_mult_max_at_g": int(used_sliders[np.argmax(log_mult[used])]) if used.any() else None,
            "reject_palette_observed": fit["reject_palette_observed"],
            "saturation_palette_observed": fit["saturation_palette_observed"],
            "dynamic_range_db": round(fit["dynamic_range_db"], 1),
        })

    axes[0].axhline(11, color="red", lw=0.8, ls=":", alpha=0.6)
    axes[0].axhline(239, color="red", lw=0.8, ls=":", alpha=0.6)
    axes[0].set_ylabel("anchor palette (p99.5 of disc)")
    axes[0].set_title("E7 N-point gain curves on ivus_test_0508\n"
                       "(open circles = used in fit; closed = rejected as noise/sat)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("slider gain")
    axes[1].set_ylabel("local log_multiplier (palette per dB)")
    axes[1].axhline(112.3, color="black", lw=0.8, ls=":",
                     label="P_035 3-point estimate (=112.3)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    out_png = ROOT / "ivus_test_0508" / "gain_curve_comparison.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    out_csv = ROOT / "ivus_test_0508" / "gain_curve_summary.csv"
    with out_csv.open("w", newline="") as f:
        cols = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"wrote {out_png}")
    print(f"wrote {out_csv}\n")
    # Pretty print summary
    print(f"{'sweep':<22} {'n_us':>4} {'g_clean':<10} "
          f"{'log_m_canon':>12} {'log_m_max':>10} {'@g':>4} "
          f"{'DR_dB':>7} {'rej':>5} {'sat':>5}")
    for r in summary_rows:
        print(f"{r['sweep']:<22} {r['n_usable']:>4} "
              f"{str(r['g_clean_min'])+'-'+str(r['g_clean_max']):<10} "
              f"{r['log_mult_canonical']:>12} "
              f"{r['log_mult_max']:>10} {str(r['log_mult_max_at_g']):>4} "
              f"{r['dynamic_range_db']:>7} "
              f"{r['reject_palette_observed']:>5} "
              f"{r['saturation_palette_observed']:>5}")


if __name__ == "__main__":
    main()
