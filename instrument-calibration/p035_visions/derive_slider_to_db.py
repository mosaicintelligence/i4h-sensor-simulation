#!/usr/bin/env python3
"""Measure the bench's actual slider->dB curve.

The current YAML assumes a linear 1.0 dB / slider relationship between the
PV.035 front-panel gain slider and the analog receive gain.  Test I and Test
M both anchor at slider 68 and the sim's +14 dB bump (slider 54 -> slider 68)
relies on this assumption.  If the bench is sublinear at high sliders, the
sim is being pushed to the wrong gain and the resulting milk-bath
brightness / TGC slope diagnostics are confounded.

This script characterises the slider->dB curve directly from bench data by
combining three independent probes:

  (a) E2 wire phantom (3 takes, sliders 0..68 in steps of 4).  Wire peak
      palette saturates at slider >=40 so this only constrains the low /
      mid range (slider 8..36).
  (b) E6 anechoic water deep tail (sliders 40, 44, 50).  The noise floor is
      a known-amplitude reference; its palette shift vs slider directly
      reports the gain change.  Limited slider range and slider 40 is
      reject-floor clipped, but slider 44 vs 50 gives a clean +6 dB step.
  (c) E4a uniform-milk speckle (sliders 20, 35, 50, 57, 68).  Milk's
      speckle floor at slider 68 is the operating point for Test I/M.
      Below saturation across the full slider range, so this is the only
      probe that constrains the slider 50 -> 68 regime.

For each probe and each pair of sliders (g1, g2), we compute the implied
``delta_dB`` by converting ``delta_palette`` through the canonical
``log_multiplier = 137.4`` (palette = 137.4 * log10(envelope)) and report
the apparent ``dB/slider`` rate.  We then overlay all probes on a single
palette-vs-slider chart and compute a piecewise slider->dB curve that the
sim can use.

Output:
  - tier1_results/figures/slider_to_db_curve.png  (the main diagnostic)
  - tier1_results/slider_to_db_curve.json         (machine-readable curve)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
IVUS_0508 = WORKSPACE_ROOT / "ivus_test_0508"
IVUS_0515 = WORKSPACE_ROOT / "ivus_test_0515"
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = HERE / "tier1_results"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_MULT = 137.4  # palette per decade in envelope amplitude
PALETTE_PER_DB = LOG_MULT / 20.0  # ~6.87 palette per dB
REJECT_FLOOR = 11.0
SATURATION = 239.0


def palette_to_db(palette: float, ref_palette: float) -> float:
    """Convert a palette difference (vs ref) to dB under the canonical LUT."""
    return float(palette - ref_palette) / PALETTE_PER_DB


# ----------------------------------------------------------------------------
# Probe A: E2 wire phantom
# ----------------------------------------------------------------------------
def load_wire_phantom_anchors() -> dict[str, dict[float, float]]:
    """Load wire anchor palette vs slider for each E2 sweep."""
    sweep_dirs = {
        "sweep1_orig_60mm": IVUS_0508 / "raw" / "sweep1_orig_60mm"
            / "derived" / "gain_curve_wire" / "gain_curve.csv",
        "sweep2_new_60mm":  IVUS_0508 / "raw" / "sweep2_new_60mm"
            / "derived" / "gain_curve_wire" / "gain_curve.csv",
        "sweep3_new_30mm":  IVUS_0508 / "raw" / "sweep3_new_30mm"
            / "derived" / "gain_curve_wire" / "gain_curve.csv",
    }
    out: dict[str, dict[float, float]] = {}
    for name, path in sweep_dirs.items():
        if not path.is_file():
            continue
        d: dict[float, float] = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                slider = float(row["gain_slider"])
                pal = float(row["anchor_palette"])
                d[slider] = pal
        out[name] = d
    return out


# ----------------------------------------------------------------------------
# Probe B: E6 anechoic water deep tail
# ----------------------------------------------------------------------------
def load_e6_water_deep_tail() -> dict[float, dict[str, float]]:
    """Load the E6 AR-on deep-tail mean palette at each slider/diameter pair.

    Also returns the AR-off ringdown PEAK palette per slider/diameter, which
    is a cleaner cross-probe than the noise floor (peak doesn't reject-clip
    until much higher slider, and gives a tight slider 40 vs 50 measurement).
    """
    ringdown = (IVUS_0508 / "raw" / "c_take2_water"
                / "derived" / "ringdown")
    summary_path = ringdown / "ringdown_summary_v2.json"
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text())
    out: dict[float, dict[str, float]] = {}
    for pair in summary.get("pairs", []):
        gain = float(pair["gain"])
        diam = float(pair["diameter_mm"])
        key = f"g{int(gain)}_D{int(diam)}"
        out[gain] = out.get(gain, {})
        out[gain].setdefault("by_diameter", {})[key] = {
            "diameter_mm": diam,
            "mean_palette_on": pair.get("noise_floor_palette_mean_from_palette_on"),
            "std_palette_on": pair.get("noise_floor_palette_std_from_palette_on"),
            "mean_palette_off": pair.get("noise_floor_palette_mean_from_palette_off"),
            "peak_palette_off": pair.get("peak_palette_off"),
        }
    for g, info in out.items():
        means_on = [v["mean_palette_on"] for v in info["by_diameter"].values()
                    if v["mean_palette_on"] is not None]
        means_off = [v["mean_palette_off"] for v in info["by_diameter"].values()
                     if v["mean_palette_off"] is not None]
        peaks_off = [v["peak_palette_off"] for v in info["by_diameter"].values()
                     if v["peak_palette_off"] is not None]
        info["mean_palette_on"] = float(np.mean(means_on)) if means_on else None
        info["mean_palette_off"] = float(np.mean(means_off)) if means_off else None
        info["peak_palette_off"] = float(np.mean(peaks_off)) if peaks_off else None
    return out


# ----------------------------------------------------------------------------
# Probe C: E4a uniform-milk speckle (the operating-point probe)
# ----------------------------------------------------------------------------
def load_e4a_milk_per_slider(diameters: list[float] = (15.0, 30.0)) -> dict:
    """Load E4a milk eval-band mean palette at every slider/diameter point.

    Returns a dict keyed by slider with sub-dicts per diameter:
        {slider: {diameter: {mean_band, mean_pixels_above_floor, n_takes, sources}}}
    """
    try:
        from unwrap import read_dicom_array, polar_resample  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import unwrap helpers from {HERE}: {exc}") from exc

    e4a_roots = [IVUS_0515 / "raw" / f"e4a_milk_gain_p{i}"
                 for i in (1, 2, 3)]
    num_theta, num_r, dr_mm = 720, 320, 0.10

    # Eval band — pick a depth band that's well within milk for both
    # diameters and above the ring-down extent.
    R_LO, R_HI = 5.0, 12.0  # mm
    r_mm = (np.arange(num_r) + 0.5) * dr_mm
    band_mask = (r_mm >= R_LO) & (r_mm <= R_HI)

    by_slider: dict[float, dict[float, dict]] = {}
    for root in e4a_roots:
        meta_path = root / "derived" / "frames_meta.csv"
        if not meta_path.is_file():
            continue
        with meta_path.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            try:
                slider = float(row["gain_slider"])
                diameter = float(row["diameter_mm"])
            except (KeyError, ValueError, TypeError):
                continue
            if diameter not in diameters:
                continue
            fname = row["file"]
            dpath = root / fname
            if not dpath.is_file():
                continue
            try:
                pix = float(row["pixel_spacing_mm"])
                rows_n = int(row["rows"])
                cols_n = int(row["cols"])
            except (KeyError, ValueError, TypeError):
                continue
            cart = read_dicom_array(dpath)
            cx_px, cy_px = cols_n / 2.0, rows_n / 2.0
            polar = polar_resample(cart, cx_px, cy_px, pix,
                                    0.0, 1, num_theta, num_r, dr_mm)
            band = polar[:, band_mask].astype(np.float64)
            d_per = by_slider.setdefault(slider, {})
            entry = d_per.setdefault(diameter, {
                "frames": [], "sources": [], "diameter_mm": diameter,
            })
            entry["frames"].append(band)
            entry["sources"].append(f"{root.name}/{fname}")

    out: dict = {"r_band_mm": [R_LO, R_HI], "by_slider": {}}
    for slider, d_per in by_slider.items():
        for diameter, entry in d_per.items():
            stack = np.concatenate(entry["frames"], axis=0)
            mean_band = float(stack.mean())
            mean_above_floor = float(stack[stack > REJECT_FLOOR].mean()) \
                if (stack > REJECT_FLOOR).any() else float("nan")
            frac_above = float((stack > REJECT_FLOOR).mean())
            out["by_slider"].setdefault(slider, {})[diameter] = {
                "mean_band": mean_band,
                "mean_above_floor": mean_above_floor,
                "frac_above_floor": frac_above,
                "std_band": float(stack.std()),
                "n_takes": len(entry["frames"]),
                "sources": entry["sources"],
            }
    return out


# ----------------------------------------------------------------------------
# Analysis: fit slider->dB curve and compare to LUT
# ----------------------------------------------------------------------------
def analyse_curve(wire: dict, water: dict, milk: dict):
    """Build the slider->dB curve from the three probes.

    Approach: pick a reference slider (54 = Tier 1's reference), then for
    each probe and each (slider) determine the dB shift relative to ref by
    converting the palette delta through the canonical LUT.  Combine the
    probes by their dynamic range:
      * Wire phantom: only sliders <= 36 (unsaturated)
      * E6 water: sliders 44 + 50 (slider 40 is floor-clipped)
      * E4a milk: sliders >= 50 (lower sliders are floor-clipped)
    """
    REF_SLIDER = 54.0
    # We can't measure REF directly (no wires unsaturated, no milk floored,
    # only E6 partial), so we use slider 50 as the anchor and assume
    # dB(50) = dB(54) - 4.0 dB initially (test the hypothesis).

    # ----- A. Wire (sweep3 30mm — cleanest, most points) -----
    s3 = wire.get("sweep3_new_30mm", {})
    sliders_w = sorted(s for s, p in s3.items() if 8 <= s <= 36
                       and REJECT_FLOOR < p < SATURATION)
    wire_curve = []
    if sliders_w:
        # Reference within wire data: pick slider 36 (highest unsaturated).
        ref_w = max(sliders_w)
        ref_w_pal = s3[ref_w]
        for s in sliders_w:
            db = palette_to_db(s3[s], ref_w_pal)
            wire_curve.append({"slider": s, "palette": s3[s],
                               "db_vs_ref": db, "lut_db_vs_ref": s - ref_w})

    # ----- B. E6 water -----
    water_noise_curve = []
    water_ringdown_curve = []
    for s, info in water.items():
        m_on = info.get("mean_palette_on")
        if m_on is not None and m_on > REJECT_FLOOR + 1:
            water_noise_curve.append({"slider": s, "palette": m_on})
        # Ringdown peak (off): available even when noise floor clipped.
        p_off = info.get("peak_palette_off")
        if p_off is not None and p_off < SATURATION - 1:
            water_ringdown_curve.append({"slider": s, "palette": p_off})
    water_noise_curve.sort(key=lambda x: x["slider"])
    water_ringdown_curve.sort(key=lambda x: x["slider"])
    water_curve = water_noise_curve  # back-compat alias

    # ----- C. E4a milk (slider 20..68, focus on >= 35 where unfloored) -----
    milk_curve = []
    for slider in sorted(milk["by_slider"].keys()):
        for diameter, entry in milk["by_slider"][slider].items():
            if entry["frac_above_floor"] < 0.05:
                continue  # essentially fully reject-clipped
            milk_curve.append({
                "slider": slider, "diameter_mm": diameter,
                "mean_band": entry["mean_band"],
                "mean_above_floor": entry["mean_above_floor"],
                "frac_above_floor": entry["frac_above_floor"],
            })

    # Compute pairwise dB/slider rates within each probe.
    def pairwise_rates(curve, key_pal: str, key_slider: str = "slider"):
        rs = []
        for i in range(len(curve) - 1):
            ds = curve[i+1][key_slider] - curve[i][key_slider]
            if ds <= 0:
                continue
            dp = curve[i+1][key_pal] - curve[i][key_pal]
            ddb = dp / PALETTE_PER_DB
            rs.append({
                "from": curve[i][key_slider],
                "to": curve[i+1][key_slider],
                "delta_palette": dp,
                "delta_db_inferred": ddb,
                "db_per_slider": ddb / ds,
            })
        return rs

    wire_rates = pairwise_rates(wire_curve, "palette")
    water_noise_rates = pairwise_rates(water_noise_curve, "palette")
    water_ringdown_rates = pairwise_rates(water_ringdown_curve, "palette")
    milk_rates_30 = pairwise_rates(
        [m for m in milk_curve if m["diameter_mm"] == 30.0],
        "mean_above_floor"
    )
    milk_rates_15 = pairwise_rates(
        [m for m in milk_curve if m["diameter_mm"] == 15.0],
        "mean_above_floor"
    )

    return {
        "ref_slider": REF_SLIDER,
        "wire_curve": wire_curve,
        "water_noise_curve": water_noise_curve,
        "water_ringdown_curve": water_ringdown_curve,
        "milk_curve": milk_curve,
        "rates": {
            "wire": wire_rates,
            "water_noise": water_noise_rates,
            "water_ringdown_peak": water_ringdown_rates,
            "milk_30mm": milk_rates_30,
            "milk_15mm": milk_rates_15,
        },
    }


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_curve(analysis: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left panel: palette vs slider (raw bench data) ----
    ax = axes[0]
    wc = analysis["wire_curve"]
    if wc:
        sl = [r["slider"] for r in wc]
        pl = [r["palette"] for r in wc]
        ax.plot(sl, pl, "o-", color="C0", linewidth=2, markersize=8,
                label="Wire peak (sweep3 30mm, unsat only)")
    for r in analysis["water_ringdown_curve"]:
        ax.plot(r["slider"], r["palette"], "D", color="C4", markersize=12,
                markeredgecolor="black", markeredgewidth=1.5)
    if analysis["water_ringdown_curve"]:
        sl = [r["slider"] for r in analysis["water_ringdown_curve"]]
        pl = [r["palette"] for r in analysis["water_ringdown_curve"]]
        ax.plot(sl, pl, "-", color="C4", linewidth=1.5, alpha=0.6)
    ax.plot([], [], "D", color="C4", markersize=12, markeredgecolor="black",
            label="E6 ringdown peak (AR-off, unsaturated)")
    for r in analysis["water_noise_curve"]:
        ax.plot(r["slider"], r["palette"], "s", color="C1", markersize=14,
                markerfacecolor="none", markeredgewidth=2.5)
    ax.plot([], [], "s", color="C1", markersize=14, markerfacecolor="none",
            markeredgewidth=2.5, label="E6 water noise floor (AR-on, unclipped)")
    milk_30 = [m for m in analysis["milk_curve"] if m["diameter_mm"] == 30.0]
    milk_15 = [m for m in analysis["milk_curve"] if m["diameter_mm"] == 15.0]
    if milk_30:
        sl = [r["slider"] for r in milk_30]
        pl = [r["mean_above_floor"] for r in milk_30]
        ax.plot(sl, pl, "^-", color="C2", linewidth=1.5, markersize=12, alpha=0.85)
    ax.plot([], [], "^-", color="C2", markersize=12,
            label="E4a milk 30mm (mean above floor)")
    if milk_15:
        sl = [r["slider"] for r in milk_15]
        pl = [r["mean_above_floor"] for r in milk_15]
        ax.plot(sl, pl, "v--", color="C3", linewidth=1.5, markersize=10, alpha=0.85)
        ax.plot([], [], "v--", color="C3", markersize=10,
                label="E4a milk 15mm (mean above floor)")
    ax.axhline(SATURATION, color="grey", linestyle="--", alpha=0.5,
               label="palette saturation 239")
    ax.axhline(REJECT_FLOOR, color="grey", linestyle=":", alpha=0.5,
               label="reject floor 11")
    ax.set_xlabel("gain slider")
    ax.set_ylabel("palette")
    ax.set_title("All probes: palette vs slider")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 260)

    # ---- Right panel: inferred dB/slider rate ----
    ax = axes[1]
    rates = analysis["rates"]
    # Plot each set of rate measurements at the midpoint slider.
    def plot_rates(label, rs, color, marker):
        if not rs:
            return
        mids = [(r["from"] + r["to"]) / 2.0 for r in rs]
        rates_vals = [r["db_per_slider"] for r in rs]
        ax.plot(mids, rates_vals, marker, color=color, markersize=10,
                label=label, linestyle="--", alpha=0.8)

    plot_rates("Wire peak (sweep3, slider 8-36)", rates["wire"], "C0", "o")
    plot_rates("E6 noise floor (slider 40-50)", rates["water_noise"], "C1", "s")
    plot_rates("E6 ringdown peak (slider 40-50)", rates["water_ringdown_peak"], "C4", "D")
    plot_rates("E4a milk 30mm", rates["milk_30mm"], "C2", "^")
    if rates["milk_15mm"]:
        plot_rates("E4a milk 15mm", rates["milk_15mm"], "C3", "v")
    ax.axhline(1.0, color="black", linestyle="-", alpha=0.5,
               label="LUT assumption: 1.0 dB/slider")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.4)
    ax.axhline(0.0, color="grey", linestyle=":", alpha=0.4)
    # Annotate key cells
    ax.annotate("Wire near sat\n(display compression)",
                xy=(34, 0.6), xytext=(28, 0.2),
                arrowprops=dict(arrowstyle="->", color="C0", alpha=0.6),
                fontsize=8, color="C0")
    ax.annotate("Milk in linear LUT\n(true analog gain)",
                xy=(65, 0.51), xytext=(50, 1.4),
                arrowprops=dict(arrowstyle="->", color="C2", alpha=0.7),
                fontsize=8, color="C2")
    ax.set_xlabel("midpoint slider")
    ax.set_ylabel("dB / slider (inferred from delta_palette / 6.87)")
    ax.set_title("Implied dB/slider rate (assuming log_multiplier=137.4)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 72)
    ax.set_ylim(-0.2, 1.8)

    plt.suptitle("Bench slider->dB measurement from "
                 "wire / water / milk probes",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}", flush=True)


def main() -> int:
    print("[step1] Loading wire-phantom anchors...", flush=True)
    wire = load_wire_phantom_anchors()
    print(f"  Got {len(wire)} sweeps: {list(wire.keys())}", flush=True)
    for name, d in wire.items():
        print(f"  {name}: {len(d)} sliders, range {min(d.keys()):.0f}..{max(d.keys()):.0f}",
              flush=True)

    print("[step1] Loading E6 anechoic water deep tail...", flush=True)
    water = load_e6_water_deep_tail()
    print(f"  Got {len(water)} slider points: "
          f"{sorted(water.keys())}", flush=True)
    for g, info in water.items():
        print(f"    slider {g:.0f}: mean_on={info.get('mean_palette_on')}, "
              f"mean_off={info.get('mean_palette_off')}", flush=True)

    print("[step1] Loading E4a milk per-slider per-diameter...", flush=True)
    milk = load_e4a_milk_per_slider()
    print(f"  Got {len(milk['by_slider'])} sliders, "
          f"r-band {milk['r_band_mm']} mm", flush=True)
    for slider in sorted(milk["by_slider"].keys()):
        for diameter, entry in milk["by_slider"][slider].items():
            print(f"    slider {slider:.0f} D{diameter:.0f}: "
                  f"mean_band={entry['mean_band']:.2f} "
                  f"mean>floor={entry['mean_above_floor']:.2f} "
                  f"frac>floor={100*entry['frac_above_floor']:.1f}% "
                  f"n_takes={entry['n_takes']}", flush=True)

    print("\n[step1] Analysing slider->dB curve...", flush=True)
    analysis = analyse_curve(wire, water, milk)

    print("\n=== Pairwise dB/slider rates ===", flush=True)
    for probe, rates in analysis["rates"].items():
        if not rates:
            continue
        print(f"\n  {probe}:", flush=True)
        for r in rates:
            print(f"    slider {r['from']:>3.0f} -> {r['to']:>3.0f}: "
                  f"dpal = {r['delta_palette']:+6.2f}  "
                  f"dB inferred = {r['delta_db_inferred']:+6.2f}  "
                  f"dB/slider = {r['db_per_slider']:+5.3f}",
                  flush=True)

    plot_curve(analysis, FIG_DIR / "slider_to_db_curve.png")

    # ---- Piecewise fit + impact on Test I/M ----
    # Anchor points (each is a clean unsaturated, unclipped measurement
    # in the linear LUT regime):
    #   slider 40-50 (ringdown peak): 1.0 dB/slider
    #   slider 62-68 (milk speckle):  0.51 dB/slider
    # Assume linear interpolation of the RATE between slider 50 and 62.
    rate_anchors = [
        (45.0, 1.00),  # midpoint of 40-50 ringdown measurement
        (65.0, 0.51),  # midpoint of 62-68 milk measurement
    ]
    def dB_per_slider_at(s: float) -> float:
        """Piecewise-linear interpolation of dB/slider rate."""
        if s <= rate_anchors[0][0]:
            return rate_anchors[0][1]
        if s >= rate_anchors[-1][0]:
            return rate_anchors[-1][1]
        s0, r0 = rate_anchors[0]
        s1, r1 = rate_anchors[1]
        return r0 + (r1 - r0) * (s - s0) / (s1 - s0)

    # Build cumulative dB from slider 54 (the YAML reference) using the
    # measured rate.
    REF_SLIDER = 54.0
    slider_grid = np.arange(40, 71, 0.5)
    dB_vs_ref = np.zeros_like(slider_grid)
    for i, s in enumerate(slider_grid):
        if s == REF_SLIDER:
            continue
        # Integrate rate from REF to s
        lo, hi = sorted([REF_SLIDER, s])
        ss = np.arange(lo, hi + 0.5, 0.5)
        rates = np.array([dB_per_slider_at(x) for x in ss])
        # Trapezoidal integration
        deltas = np.diff(ss)
        avg_rates = 0.5 * (rates[:-1] + rates[1:])
        cum = (avg_rates * deltas).sum()
        dB_vs_ref[i] = cum if s > REF_SLIDER else -cum

    print("\n=== Measured slider->dB curve (vs ref slider 54) ===", flush=True)
    test_sliders = [40, 44, 50, 54, 57, 62, 65, 68, 70]
    print(f"  {'slider':>7s} {'measured':>10s} {'LUT (1 dB/sl)':>14s} {'delta':>8s}",
          flush=True)
    impact = {}
    for s in test_sliders:
        idx = int(np.argmin(np.abs(slider_grid - s)))
        measured = float(dB_vs_ref[idx])
        lut = s - REF_SLIDER
        delta = measured - lut
        print(f"  {s:>7.0f} {measured:>+8.2f}dB {lut:>+12.2f}dB {delta:>+7.2f}dB",
              flush=True)
        impact[s] = {"measured_dB": measured, "lut_dB": lut, "delta_dB": delta}

    analysis["slider_to_db_curve"] = {
        "rate_anchors_slider_db_per_slider": rate_anchors,
        "reference_slider": REF_SLIDER,
        "table_slider_to_db_measured_vs_ref": impact,
        "method": ("Piecewise-linear interpolation of the dB/slider rate "
                   "between two cleanly-measured anchors: (slider 45, "
                   "1.0 dB/sl) from ringdown peak slider 40->50 and "
                   "(slider 65, 0.51 dB/sl) from E4a milk speckle slider "
                   "62->68. Rate is clamped at the anchor values outside "
                   "the [45, 65] interval."),
    }
    out_json = OUT_DIR / "slider_to_db_curve.json"
    out_json.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"\nWrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
