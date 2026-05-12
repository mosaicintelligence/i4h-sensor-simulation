#!/usr/bin/env python3
"""Stage ivus_test_0508 captures into per-experiment subfolders that the
existing P_035 calibration pipeline (annotate_wires.py, fit_alignment.py,
unwrap.py, extract_*.py) can consume without changes.

Reads the curated `ivus_test_0508/manifest.csv` and creates symlinks under
`ivus_test_0508/raw/<experiment>/FILE####.dcm`, plus copies the wire-phantom
DXF (and a per-experiment manifest copy) into each Capture-A subfolder so
the alignment GUI can find them on the default --capture-dir.

Layout produced (no copies, only symlinks):

  ivus_test_0508/raw/
    sweep1_orig_60mm/    # Capture A on original apparatus, D=60mm, asc 0..68
      FILE0000.dcm -> ../../CASE0007/FILE0002    (gain  0)
      FILE0001.dcm -> ../../CASE0007/FILE0003    (gain  4)
      ...
      FILE0017.dcm -> ../../CASE0006/FILE0009    (gain 68)
      IVUS Scattering Box - Sketch 1.dxf -> ../../../P_035_PointScatter/...
      manifest_subset.csv
    sweep2_new_60mm/     # Capture A new apparatus widest, D=60mm, desc 68..0
    sweep3_new_30mm/     # Capture A new apparatus narrow,  D=30mm, asc 0..68
    c_take2_water/       # Capture C take 2 multi-grid water only (CASE0000)
    c_pilot_water/       # Capture C pilot AR pair (CASE0007 F0000-F0001)
    c_take1_water/       # Capture C take 1 partial (CASE0008)

The dropped cases (CASE0001 duplicate, CASE0009 onboarding) are not staged.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/Users/jocelynbarker/ivus-sim")
DATASET = ROOT / "ivus_test_0508"
MANIFEST = DATASET / "manifest.csv"

# Per-apparatus wire-phantom DXF.
#   ORIGINAL apparatus = the P_035 5x5 wire grid in
#       P_035_PointScatter/IVUS Scattering Box - Sketch 1.dxf
#   NEW apparatus      = the 12-wire spiral in
#       instrument-calibration/hardware/wire_spiral_v1.dxf
#       (generated from instrument-calibration/tools/gen_phantom_stl.py
#       via instrument-calibration/tools/gen_phantom_dxf.py).
DXF_BY_APPARATUS = {
    "original": ROOT / "P_035_PointScatter" / "IVUS Scattering Box - Sketch 1.dxf",
    "new": ROOT / "instrument-calibration" / "hardware" / "wire_spiral_v1.dxf",
}

# experiment_label  -> subfolder + sort key + (gain ascending bool)
# Use sweep_role / take to figure out which experiment a row belongs to.

EXPERIMENTS = {
    "sweep1_orig_60mm": {
        "predicate": lambda row: (
            row["experiment"] == "E2" and row["take"] == "1"
        ),
        "sort_key": lambda row: int(row["gain_slider"]),
        "dxf_apparatus": "original",
    },
    "sweep2_new_60mm": {
        "predicate": lambda row: (
            row["experiment"] == "E2" and row["take"] == "2"
        ),
        "sort_key": lambda row: int(row["gain_slider"]),
        "dxf_apparatus": "new",
    },
    "sweep3_new_30mm": {
        "predicate": lambda row: (
            row["experiment"] == "E2" and row["take"] == "3"
        ),
        "sort_key": lambda row: int(row["gain_slider"]),
        "dxf_apparatus": "new",
    },
    "c_take2_water": {
        "predicate": lambda row: (
            row["experiment"] == "E6" and row["take"] == "2"
        ),
        "sort_key": lambda row: (int(row["diameter_mm"]), int(row["gain_slider"]),
                                  0 if row["mode_flag"] == "0" else 1),
        "dxf_apparatus": None,
    },
    "c_pilot_water": {
        "predicate": lambda row: (
            row["experiment"] == "E6" and row["take"] == "pilot"
        ),
        "sort_key": lambda row: 0 if row["mode_flag"] == "0" else 1,
        "dxf_apparatus": None,
    },
    "c_take1_water": {
        "predicate": lambda row: (
            row["experiment"] == "E6" and row["take"] == "1"
        ),
        "sort_key": lambda row: 0,
        "dxf_apparatus": None,
    },
}


def load_rows() -> list[dict]:
    with MANIFEST.open() as f:
        return [r for r in csv.DictReader(f)]


def stage_one(name: str, cfg: dict, rows: list[dict]) -> tuple[Path, int]:
    out_dir = DATASET / "raw" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [r for r in rows if cfg["predicate"](r)]
    keep.sort(key=cfg["sort_key"])
    # Wipe any prior FILE*.dcm symlinks AND any stale DXF symlinks so rerunning
    # cleanly transitions apparatuses (e.g. the prior staging pointed every
    # E2 sweep at the P_035 DXF; the new staging points the new-apparatus
    # sweeps at wire_spiral_v1.dxf instead).
    for prior in out_dir.glob("FILE*.dcm"):
        if prior.is_symlink() or prior.is_file():
            prior.unlink()
    for prior in out_dir.glob("*.dxf"):
        if prior.is_symlink() or prior.is_file():
            prior.unlink()
    n = 0
    sub_rows = []
    for i, r in enumerate(keep):
        src = DATASET / r["case"] / r["file"]
        dst = out_dir / f"FILE{i:04d}.dcm"
        rel_src = Path("..") / ".." / src.relative_to(DATASET)
        dst.symlink_to(rel_src)
        n += 1
        sub_rows.append({**r, "staged_file": dst.name})
    sub_csv = out_dir / "manifest_subset.csv"
    if sub_rows:
        with sub_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sub_rows[0].keys()))
            w.writeheader()
            w.writerows(sub_rows)
    apparatus = cfg.get("dxf_apparatus")
    if apparatus is not None:
        dxf_src = DXF_BY_APPARATUS[apparatus]
        dxf_dst = out_dir / dxf_src.name
        if dxf_dst.exists() or dxf_dst.is_symlink():
            dxf_dst.unlink()
        rel_dxf = Path("..") / ".." / ".." / dxf_src.relative_to(ROOT)
        dxf_dst.symlink_to(rel_dxf)
    return out_dir, n


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 1
    rows = load_rows()
    print(f"# stage_ivus_test_0508 — read {len(rows)} manifest rows")
    for name, cfg in EXPERIMENTS.items():
        out_dir, n = stage_one(name, cfg, rows)
        print(f"  staged {n:3d} files -> {out_dir.relative_to(ROOT)}")
    print("\nDone. Each subfolder has FILE####.dcm symlinks ready for the existing pipeline.")
    print("Hint: run from workspace root, e.g.:")
    print("  python3 instrument-calibration/p035_visions/extract_metadata.py \\")
    print("      --dataset ivus_test_0508/raw/sweep3_new_30mm \\")
    print("      --out     ivus_test_0508/raw/sweep3_new_30mm/derived/frames_meta.csv \\")
    print("      --align-csv ivus_test_0508/raw/sweep3_new_30mm/derived/alignment_fit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
