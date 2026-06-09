#!/usr/bin/env python3
"""Axial PSF tooling: expose pulse_duration_cycles as a knob.

The bench Wave 0 B2 aggregate
(``ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/psf_fit.json``)
records median axial FWHM = 0.3032 mm across 154 wires (range 0.05-1.5 mm
depending on r and SNR).  Analytically the same JSON derives
``pulse_duration_cycles = 2 * ax_med / lambda = 3.94`` from the
Hanning-windowed-cosine first-order relation.  But the sim has a *causal*
Hanning kernel (only the left half is non-zero) -- the -6 dB walkout on
that kernel is narrower than the symmetric case, so the right
pulse_duration_cycles to recover bench median FWHM is something we must
**measure in the loop**, not just read off the analytic formula.

This script:

1. Reads the current YAML.
2. Sweeps ``probe.pulse_duration_cycles`` across a configurable set
   (default = [1.93, 2.5, 3.0, 3.94, 5.0, 7.0, 10.0]).
3. For each value, renders the same 5-sphere wire phantom used by Test C
   (r = 5/10/15/20/25 mm) with the diagnostic settings (log_floor =
   1e-19, ring-down OFF, saturation cutoff suppressed) so the
   underlying kernel FWHM is recoverable regardless of clamp behaviour.
4. Extracts per-wire axial FWHM using the same ``measure_wire`` walkout
   that tier1_evaluation.py uses.
5. Plots sim per-wire FWHM curves (one curve per pulse_duration value)
   alongside the bench per-wire FWHM scatter (154 wires) and the bench
   median + IQR.
6. Reports the pulse_duration_cycles whose sim median FWHM best matches
   the bench median; that's the recommended YAML value.

Writes:

- ``axial_psf_derivation.json`` next to the bench aggregate, with the
  full sweep table + recommended value + provenance.
- ``figures/axial_psf_pulse_duration_sweep.png`` showing sim vs bench.

Usage::

    python instrument-calibration/p035_visions/derive_axial_psf_pulse_duration.py
    python instrument-calibration/p035_visions/derive_axial_psf_pulse_duration.py \\
        --n-frames 8 --values 1.93 3.0 3.94 5.0 7.0

Limitations.  The simulator has **one** axial-PSF knob today
(``pulse_duration_cycles``); there is no receive-side bandpass filter
or separate SIR stage exposed.  If the sweep shows that no single
``pulse_duration_cycles`` recovers the bench shape across r (e.g. sim
FWHM is independent of r but bench FWHM grows with r), that gates a
sim-physics change to add a receive-SIR stage (analogous to the
constant-angular lateral kernel switch). This script reports that
diagnosis if it applies.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
SIM_ROOT = WORKSPACE_ROOT / "i4h-sensor-simulation" / "ultrasound-raytracing"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import raysim as rs  # noqa: E402
from raysim import IvusSimConfig  # noqa: E402
from raysim.ray_sim_python import Sphere  # noqa: E402

from tier1_evaluation import (  # noqa: E402
    WIRE_RADII_MM,
    WIRE_SPHERE_RADIUS_MM,
    b_mode_to_theta_r,
    find_wire_peak,
    measure_wire,
    polar_axes,
    render_frames,
)

YAML_PATH = HERE / "volcano_s5i.yaml"
WAVE0_ROOT = WORKSPACE_ROOT / "ivus_test_0515"
PSF_FIT_PATH = (
    WAVE0_ROOT / "derived_aggregate" / "psf_b2_tungsten_water" / "psf_fit.json"
)
OUTPUT_JSON = (
    WAVE0_ROOT
    / "derived_aggregate"
    / "psf_b2_tungsten_water"
    / "axial_psf_derivation.json"
)
FIG_DIR = HERE / "tier1_results" / "figures"
FIG_PATH = FIG_DIR / "axial_psf_pulse_duration_sweep.png"


def build_wire_world(materials):
    world = rs.World("lumen")
    wire_mat = materials.get_index("tungsten")
    positions: list[tuple[float, float]] = []
    for i, r in enumerate(WIRE_RADII_MM):
        theta = i * 2.0 * math.pi / len(WIRE_RADII_MM)
        x = r * math.sin(theta)
        z = r * math.cos(theta)
        world.add(Sphere(np.array([x, 0.0, z], dtype=np.float32),
                         WIRE_SPHERE_RADIUS_MM, wire_mat))
        positions.append((r, math.degrees(theta) % 360.0))
    return world, positions


def render_and_measure(cfg: IvusSimConfig, n_frames: int) -> list[dict]:
    """Render with diagnostic settings, return per-wire FWHM rows."""
    sim_params = cfg.to_sim_params()
    sim_params.log_floor = 1e-19
    sim_params.ring_down.enabled = False
    sim_params.reject_palette = 0.0
    sim_params.saturation_palette = 0.0
    sim_params.median_clip_filter = False

    materials = rs.Materials()
    world, positions = build_wire_world(materials)
    frames = render_frames(cfg, world, materials, n_frames, sim_params)

    theta_deg, r_mm, _, _ = polar_axes(cfg)
    rows: list[dict] = []
    for fi in range(n_frames):
        frame = b_mode_to_theta_r(frames[fi], cfg)
        for wi, (r_pred, theta_pred) in enumerate(positions):
            pk = find_wire_peak(frame, theta_deg, r_mm, theta_pred, r_pred)
            if pk is None:
                continue
            pk_t, pk_r = pk
            m = measure_wire(
                frame, theta_deg, r_mm, pk_t, pk_r,
                log_multiplier=float(cfg.processing.log_multiplier),
                saturation_palette=float("inf"),
            )
            if m is None or m["axial_fwhm_mm"] is None:
                continue
            rows.append({
                "wire_idx": wi + 1,
                "r_pred_mm": float(r_pred),
                "axial_fwhm_mm": float(m["axial_fwhm_mm"]),
                "lateral_fwhm_arc_mm": float(m["lateral_fwhm_arc_mm"]),
                "peak_palette": float(m["peak"]),
            })
    return rows


def median_per_wire(rows: list[dict]) -> dict[int, float]:
    by_wire: dict[int, list[float]] = {}
    for row in rows:
        by_wire.setdefault(row["wire_idx"], []).append(row["axial_fwhm_mm"])
    return {wi: float(np.median(v)) for wi, v in by_wire.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep pulse_duration_cycles and compare sim axial FWHM to bench."
    )
    ap.add_argument("--values", type=float, nargs="+",
                    default=[1.93, 2.5, 3.0, 3.94, 5.0, 7.0, 10.0],
                    help="pulse_duration_cycles values to sweep.")
    ap.add_argument("--n-frames", type=int, default=8,
                    help="Frames per sweep step.")
    ap.add_argument("--out-json", type=Path, default=OUTPUT_JSON)
    ap.add_argument("--out-fig", type=Path, default=FIG_PATH)
    ap.add_argument("--psf-fit", type=Path, default=PSF_FIT_PATH)
    args = ap.parse_args()

    if not args.psf_fit.is_file():
        raise FileNotFoundError(f"Missing bench PSF fit: {args.psf_fit}")
    bench = json.load(open(args.psf_fit))
    bench_per_wire = bench["axial_fwhm_mm_per_wire"]
    bench_ax = np.array([float(r["axial_fwhm_mm"]) for r in bench_per_wire])
    bench_r = np.array([float(r["r_mm"]) for r in bench_per_wire])
    bench_peak = np.array([float(r["peak_palette"]) for r in bench_per_wire])
    keep = (bench_peak <= 230.0) & np.isfinite(bench_ax) & (bench_ax > 0)
    bench_ax_kept = bench_ax[keep]
    bench_r_kept = bench_r[keep]
    bench_med = float(np.median(bench_ax_kept))
    bench_p25 = float(np.percentile(bench_ax_kept, 25))
    bench_p75 = float(np.percentile(bench_ax_kept, 75))
    bench_analytic_n_cycles = float(bench.get("pulse_duration_cycles", float("nan")))

    print(f"[axial_psf] bench: n={int(keep.sum())} / {len(bench_per_wire)} wires kept; "
          f"median axial FWHM = {bench_med*1000:.0f} um "
          f"(IQR {bench_p25*1000:.0f}-{bench_p75*1000:.0f} um)")
    print(f"[axial_psf] bench analytic pulse_duration_cycles = {bench_analytic_n_cycles:.3f} "
          f"(from 2 * ax_med / lambda)")

    cfg_base = IvusSimConfig.from_yaml(YAML_PATH)

    sweep: list[dict] = []
    for n_cycles in args.values:
        print(f"[axial_psf] rendering at pulse_duration_cycles = {n_cycles:.3f} ...")
        cfg = IvusSimConfig.from_yaml(YAML_PATH)
        cfg.probe.pulse_duration_cycles = float(n_cycles)
        rows = render_and_measure(cfg, args.n_frames)
        med_by_wire = median_per_wire(rows)
        all_fwhm = [row["axial_fwhm_mm"] for row in rows]
        med_overall = float(np.median(all_fwhm)) if all_fwhm else float("nan")
        sweep.append({
            "pulse_duration_cycles": float(n_cycles),
            "median_fwhm_mm": med_overall,
            "per_wire_median_fwhm_mm": {str(k): float(v) for k, v in med_by_wire.items()},
            "n_frames": args.n_frames,
            "n_wires_measured": int(len(med_by_wire)),
        })
        delta_med = med_overall - bench_med if med_overall == med_overall else float("nan")
        print(f"[axial_psf]   sim median FWHM = {med_overall*1000:.0f} um "
              f"(delta vs bench {delta_med*1000:+.0f} um)")

    valid = [s for s in sweep
             if s["median_fwhm_mm"] == s["median_fwhm_mm"]]
    if not valid:
        raise RuntimeError("No sweep step produced a measurable FWHM.")
    best = min(valid, key=lambda s: abs(s["median_fwhm_mm"] - bench_med))
    print(f"\n[axial_psf] best match: pulse_duration_cycles = "
          f"{best['pulse_duration_cycles']:.3f} "
          f"(sim median FWHM {best['median_fwhm_mm']*1000:.0f} um "
          f"vs bench {bench_med*1000:.0f} um, "
          f"delta {(best['median_fwhm_mm']-bench_med)*1000:+.0f} um)")

    payload = {
        "bench_source": str(args.psf_fit.relative_to(WORKSPACE_ROOT)),
        "bench": {
            "n_wires_total": len(bench_per_wire),
            "n_wires_kept": int(keep.sum()),
            "median_fwhm_mm": bench_med,
            "p25_fwhm_mm": bench_p25,
            "p75_fwhm_mm": bench_p75,
            "wavelength_mm": float(bench.get("wavelength_mm", float("nan"))),
            "analytic_pulse_duration_cycles": bench_analytic_n_cycles,
        },
        "sim_yaml_pulse_duration_cycles": float(cfg_base.probe.pulse_duration_cycles),
        "sweep": sweep,
        "best_match": {
            "pulse_duration_cycles": best["pulse_duration_cycles"],
            "sim_median_fwhm_mm": best["median_fwhm_mm"],
            "bench_median_fwhm_mm": bench_med,
            "delta_mm": best["median_fwhm_mm"] - bench_med,
        },
        "yaml_field": "probe.pulse_duration_cycles",
        "diagnosis": _diagnose(sweep, bench_med),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[axial_psf] wrote {args.out_json}")

    try:
        _make_figure(sweep, bench_r_kept, bench_ax_kept,
                     bench_med, bench_p25, bench_p75,
                     args.out_fig)
        print(f"[axial_psf] wrote {args.out_fig}")
    except Exception as exc:  # pragma: no cover
        print(f"[axial_psf] figure render failed: {exc}")

    print("\nTo use, set in volcano_s5i.yaml:")
    print("  probe:")
    print(f"    pulse_duration_cycles: {best['pulse_duration_cycles']:.4f}")
    return 0


def _diagnose(sweep: list[dict], bench_med: float) -> dict:
    """Heuristic: can we match the bench with this single knob, or is there a shape issue?"""
    valid = [s for s in sweep if s["median_fwhm_mm"] == s["median_fwhm_mm"]]
    if not valid:
        return {"verdict": "no measurable sim FWHM"}
    lo = min(valid, key=lambda s: s["median_fwhm_mm"])
    hi = max(valid, key=lambda s: s["median_fwhm_mm"])
    spans_bench = lo["median_fwhm_mm"] <= bench_med <= hi["median_fwhm_mm"]
    monotonic = all(
        valid[i]["median_fwhm_mm"] <= valid[i+1]["median_fwhm_mm"]
        for i in range(len(valid) - 1)
        if valid[i]["pulse_duration_cycles"] <= valid[i+1]["pulse_duration_cycles"]
    )
    return {
        "spans_bench_median": spans_bench,
        "monotonic_in_pulse_duration": monotonic,
        "min_sweep_fwhm_mm": lo["median_fwhm_mm"],
        "max_sweep_fwhm_mm": hi["median_fwhm_mm"],
        "verdict": (
            "knob alone is sufficient -- pick the matching pulse_duration_cycles"
            if spans_bench else
            "knob alone is INsufficient -- bench shape is outside the sweep range, "
            "consider adding a receive-SIR / bandpass stage"
        ),
    }


def _make_figure(sweep, bench_r, bench_ax,
                 bench_med, bench_p25, bench_p75, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.scatter(bench_r, bench_ax * 1000, s=14, c="C0", alpha=0.45,
               label=f"bench (n={len(bench_ax)} wires)")
    ax.axhline(bench_med * 1000, color="C0", ls=":", lw=1.5,
               label=f"bench median {bench_med*1000:.0f} um")
    ax.axhspan(bench_p25 * 1000, bench_p75 * 1000, color="C0", alpha=0.08,
               label="bench IQR")

    cmap = plt.get_cmap("plasma")
    for i, s in enumerate(sweep):
        if s["median_fwhm_mm"] != s["median_fwhm_mm"]:
            continue
        pw = s["per_wire_median_fwhm_mm"]
        rs_mm = [float(WIRE_RADII_MM[int(wi)-1]) for wi in pw.keys()]
        fwhm = [float(v) * 1000 for v in pw.values()]
        c = cmap(i / max(len(sweep) - 1, 1))
        ax.plot(rs_mm, fwhm, "o-", color=c, ms=8, lw=1.5,
                label=f"sim n_cycles={s['pulse_duration_cycles']:.2f} "
                      f"(med {s['median_fwhm_mm']*1000:.0f} um)")

    ax.set_xlabel("radius r (mm)")
    ax.set_ylabel("axial FWHM (um)")
    ax.set_title("Sim axial PSF vs bench -- pulse_duration_cycles sweep")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
