#!/usr/bin/env python3
"""Pass 5d — derive ``processing.lateral_psf_kernel.sigma_theta_rad`` from bench.

The Pass 5d constant-angular Gaussian lateral PSF kernel uses one global
parameter, ``sigma_theta_rad`` (the angular standard deviation of the
lateral PSF, in radians), applied uniformly across depth.  The bench
anchor is the **median angular FWHM** of unsaturated tungsten wires from
the Wave 0 B2 aggregate (148 wires pooled across 4 spiral-fixture takes
in water), converted to a Gaussian sigma via

    sigma_theta_rad = (median_angular_fwhm_deg / FWHM_to_sigma) * (pi / 180)

with ``FWHM_to_sigma = 2 * sqrt(2 * ln 2) ≈ 2.3548``.

This script:

1. Loads ``ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/psf_fit.json``
   (built by ``aggregate_psf_0515.py`` from the per-wire walkout fits in
   ``derived/psf_d60/`` of every wave-0 B2 subfolder).
2. Drops wires with peak palette >= ``--peak-saturation-palette`` (default
   230) because their lateral FWHM extraction is no longer reliable.
3. Optionally restricts the radial range with ``--r-min-mm`` / ``--r-max-mm``
   (default 4.0 / 26.0 mm — the wires inside r = 4 mm sit in the ring-down
   zone where the angular FWHM extraction blows up).
4. Computes per-wire angular FWHM via ``ang_deg = degrees(arc_fwhm_mm / r_mm)``.
5. Reports median / mean / IQR of the angular-FWHM distribution and converts
   the median to ``sigma_theta_rad``.
6. Writes a derivation JSON next to the input ``psf_fit.json`` so the
   provenance lives with the bench data, and copies the summary to
   ``derived_aggregate/psf_b2_tungsten_water/sigma_theta_derivation.json``.

The YAML field ``processing.lateral_psf_kernel.sigma_theta_rad`` should be
updated by hand from the script's printout (or the JSON's
``sigma_theta_rad`` field).

Usage::

    python instrument-calibration/p035_visions/derive_lateral_psf_sigma_theta.py
    python instrument-calibration/p035_visions/derive_lateral_psf_sigma_theta.py \\
        --r-min-mm 6 --r-max-mm 22 --peak-saturation-palette 220
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PSF_FIT = (
    REPO_ROOT
    / "ivus_test_0515"
    / "derived_aggregate"
    / "psf_b2_tungsten_water"
    / "psf_fit.json"
)
FWHM_TO_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _compute_angular_fwhm(rows: list[dict],
                           peak_sat_palette: float,
                           r_min_mm: float,
                           r_max_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.array([row["r_mm"] for row in rows], dtype=float)
    lat = np.array([row["lateral_fwhm_arc_mm"] for row in rows], dtype=float)
    peak = np.array([row["peak_palette"] for row in rows], dtype=float)
    keep = (
        (peak <= peak_sat_palette)
        & (r >= r_min_mm)
        & (r <= r_max_mm)
        & np.isfinite(lat)
        & (lat > 0)
        & (r > 0)
    )
    r_k = r[keep]
    lat_k = lat[keep]
    ang_deg = np.degrees(lat_k / r_k)
    return r_k, lat_k, ang_deg


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Derive constant-angular Gaussian lateral PSF sigma_theta_rad "
            "from the Wave 0 B2 bench aggregate."
        )
    )
    ap.add_argument("--psf-fit", type=Path, default=DEFAULT_PSF_FIT,
                    help="Path to aggregate psf_fit.json (Wave 0 B2 default).")
    ap.add_argument("--peak-saturation-palette", type=float, default=230.0,
                    help="Drop wires with peak palette >= this (default 230).")
    ap.add_argument("--r-min-mm", type=float, default=4.0,
                    help="Lower r cut-off in mm (default 4.0).")
    ap.add_argument("--r-max-mm", type=float, default=26.0,
                    help="Upper r cut-off in mm (default 26.0).")
    ap.add_argument("--out", type=Path, default=None,
                    help=("Optional output JSON path; default writes "
                          "sigma_theta_derivation.json next to --psf-fit."))
    args = ap.parse_args()

    if not args.psf_fit.is_file():
        raise FileNotFoundError(f"psf_fit.json not found: {args.psf_fit}")
    with open(args.psf_fit) as f:
        fit = json.load(f)
    rows = fit.get("axial_fwhm_mm_per_wire") or []
    if not rows:
        raise RuntimeError(
            f"No per-wire rows in {args.psf_fit} -- expected key "
            "'axial_fwhm_mm_per_wire'"
        )

    r_k, lat_k, ang_deg = _compute_angular_fwhm(
        rows,
        peak_sat_palette=args.peak_saturation_palette,
        r_min_mm=args.r_min_mm,
        r_max_mm=args.r_max_mm,
    )
    n = int(len(r_k))
    if n < 10:
        raise RuntimeError(
            f"Only {n} wires survive the cuts (peak<={args.peak_saturation_palette}, "
            f"r in [{args.r_min_mm}, {args.r_max_mm}] mm) -- bench data may be "
            "saturated or shifted; widen the cuts and re-run."
        )

    median_deg = float(np.median(ang_deg))
    mean_deg = float(np.mean(ang_deg))
    p25_deg = float(np.percentile(ang_deg, 25))
    p75_deg = float(np.percentile(ang_deg, 75))
    sigma_theta_deg = median_deg / FWHM_TO_SIGMA
    sigma_theta_rad = math.radians(sigma_theta_deg)

    out_payload = {
        "bench_source": str(args.psf_fit.relative_to(REPO_ROOT)),
        "subfolders": fit.get("subfolders"),
        "wavelength_mm": fit.get("wavelength_mm"),
        "frequency_mhz": fit.get("frequency_mhz"),
        "cuts": {
            "peak_saturation_palette": args.peak_saturation_palette,
            "r_min_mm": args.r_min_mm,
            "r_max_mm": args.r_max_mm,
        },
        "n_wires_total": len(rows),
        "n_wires_kept": n,
        "angular_fwhm_deg": {
            "median": median_deg,
            "mean": mean_deg,
            "p25": p25_deg,
            "p75": p75_deg,
            "min": float(np.min(ang_deg)),
            "max": float(np.max(ang_deg)),
        },
        "r_range_kept_mm": [float(r_k.min()), float(r_k.max())],
        "sigma_theta_deg": sigma_theta_deg,
        "sigma_theta_rad": sigma_theta_rad,
        "fwhm_to_sigma": FWHM_TO_SIGMA,
        "yaml_field": "processing.lateral_psf_kernel.sigma_theta_rad",
    }

    out_path = args.out or (args.psf_fit.parent / "sigma_theta_derivation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"[derive_sigma_theta] read {args.psf_fit}")
    print(f"[derive_sigma_theta] n_wires kept = {n} / {len(rows)} "
          f"(peak<={args.peak_saturation_palette}, "
          f"r in [{args.r_min_mm}, {args.r_max_mm}] mm)")
    print(f"[derive_sigma_theta] angular FWHM deg: "
          f"median={median_deg:.4f}, mean={mean_deg:.4f}, "
          f"IQR=[{p25_deg:.4f}, {p75_deg:.4f}]")
    print(f"[derive_sigma_theta] sigma_theta_deg = {sigma_theta_deg:.6f} deg")
    print(f"[derive_sigma_theta] sigma_theta_rad = {sigma_theta_rad:.8f}")
    print(f"[derive_sigma_theta] wrote {out_path}")
    print("\nTo use, set in volcano_s5i.yaml:")
    print("  processing:")
    print("    lateral_psf_kernel:")
    print("      type: constant_angular")
    print(f"      sigma_theta_rad: {sigma_theta_rad:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
