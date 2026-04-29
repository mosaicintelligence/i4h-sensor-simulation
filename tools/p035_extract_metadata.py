#!/usr/bin/env python3
"""Extract per-frame metadata from the P_035_PointScatter DICOMs.

Reads every `FILExxxx` file in the dataset and pulls the device settings we
need to (a) stratify the calibration fits and (b) pin per-frame parameters
in the simulator config. Writes a tidy `frames_meta.csv` next to the dataset.

Tag map (verified empirically across the 19 captures):

  Public DICOM
    DepthOfScanField     (0018,5050)  -- imaging radius in mm
    PhysicalDeltaX/Y     (0018,602C)  -- displayed pixel spacing in cm/px
    Rows / Columns                    -- image grid
    TransducerData       (0018,5010)  -- catheter family
    ImageComments        (0020,4000)  -- "FxxLive:" capture index

  Volcano private (group 0029, "VOLCANO-PCDE 1.0")
    (0029,1001) FD       gain_slider     -- {44, 54, 64} in this dataset
    (0029,1002) US       constant 3      -- unknown (display fmt?), kept for completeness
    (0029,1003) FD       diameter_mm     -- {35, 40, 60} in this dataset
    (0029,1006) US       ar_enabled?     -- =1 in all frames; presumed Acoustic-Ref ON
    (0029,1007) US       mode_flag?      -- =0 except FILE0013 = 1; flagged as suspect
    (0029,1008) US       =5 in all       -- TGC slot? unverified
    (0029,1015) US       frame_index     -- 1..19, matches ImageComments

If `derived/alignment_fit.csv` exists, the alignment fit columns
(theta0_deg, chirality, radial_scale, apparatus offsets, wiggle, RMS) are
joined onto each row so downstream scripts can read everything from one CSV.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pydicom

# Volcano private group; the strings are the empirically derived meanings,
# carried in the CSV header and yaml provenance comments.
PRIVATE_TAGS: list[tuple[tuple[int, int], str]] = [
    ((0x0029, 0x1001), "gain_slider"),
    ((0x0029, 0x1002), "priv_1002"),
    ((0x0029, 0x1003), "diameter_mm"),
    ((0x0029, 0x1006), "ar_enabled"),
    ((0x0029, 0x1007), "mode_flag"),
    ((0x0029, 0x1008), "priv_1008"),
    ((0x0029, 0x1009), "priv_1009"),
    ((0x0029, 0x1015), "frame_index"),
    ((0x0029, 0x1030), "priv_1030"),
]


def extract_one(path: Path) -> dict[str, str]:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    region = ds.SequenceOfUltrasoundRegions[0]

    row: dict[str, str] = {}
    row["file"] = path.name
    row["catheter"] = ", ".join(getattr(ds, "TransducerData", []) or []) or ""
    row["model"] = str(getattr(ds, "ManufacturerModelName", ""))
    row["depth_mm"] = str(getattr(ds, "DepthOfScanField", ""))
    row["rows"] = str(ds.Rows)
    row["cols"] = str(ds.Columns)
    # PhysicalDeltaX is in cm/px -> convert to mm/px for downstream sanity.
    row["pixel_spacing_mm"] = f"{float(region.PhysicalDeltaX) * 10.0:.6f}"
    row["comment"] = str(getattr(ds, "ImageComments", "")).strip()

    for tag, name in PRIVATE_TAGS:
        if tag in ds:
            v = ds[tag].value
            row[name] = str(v)
        else:
            row[name] = ""

    # FILE0013 is the only frame with mode_flag=1 in this dataset; mark it
    # so downstream stats can choose to exclude it from group-level fits.
    row["mode_flag_anomaly"] = "1" if row.get("mode_flag", "") not in ("", "0") else "0"
    return row


def join_alignment(rows: list[dict], align_csv: Path) -> list[dict]:
    """Augment each row with alignment-fit columns (best-effort)."""
    if not align_csv.exists():
        print(f"  (no alignment fit CSV at {align_csv}; skipping join)")
        return rows
    fits: dict[str, dict[str, str]] = {}
    with align_csv.open() as f:
        for entry in csv.DictReader(f):
            fits[entry["file"]] = entry

    keep_cols = [
        "theta0_deg", "chirality", "radial_scale", "wire_offset",
        "apparatus_cx_mm_off_image_center", "apparatus_cy_mm_off_image_center",
        "apparatus_minus_device_x_mm", "apparatus_minus_device_y_mm",
        "apparatus_minus_device_mag_mm",
        "rms_residual_mm", "max_residual_mm",
        "n_clicks_used", "excluded_user_wire_indices",
    ]
    for row in rows:
        fit = fits.get(row["file"], {})
        for c in keep_cols:
            row[c] = fit.get(c, "")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("P_035_PointScatter"),
        help="Folder containing FILExxxx DICOMs",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("P_035_PointScatter/derived/frames_meta.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--align-csv",
        type=Path,
        default=Path("P_035_PointScatter/derived/alignment_fit.csv"),
        help="Optional alignment fit CSV to left-join onto each frame",
    )
    args = p.parse_args(argv)

    files = sorted(args.dataset.glob("FILE*"))
    if not files:
        print(f"No DICOMs found in {args.dataset}", file=sys.stderr)
        return 1

    rows = [extract_one(f) for f in files]
    rows = join_alignment(rows, args.align_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Quick stratification summary so the user can sanity check.
    from collections import Counter
    by_gain = Counter(r["gain_slider"] for r in rows)
    by_diam = Counter(r["diameter_mm"] for r in rows)
    by_pair = Counter((r["gain_slider"], r["diameter_mm"]) for r in rows)
    print(f"Wrote: {args.out}  ({len(rows)} frames)")
    print(f"  by gain  : {dict(sorted(by_gain.items()))}")
    print(f"  by diam  : {dict(sorted(by_diam.items()))}")
    print(f"  by (g,D) : {dict(sorted(by_pair.items()))}")
    anomalies = [r["file"] for r in rows if r["mode_flag_anomaly"] == "1"]
    if anomalies:
        print(f"  mode_flag anomalies (priv 0x1007 != 0): {anomalies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
