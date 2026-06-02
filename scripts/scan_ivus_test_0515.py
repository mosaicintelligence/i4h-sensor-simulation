#!/usr/bin/env python3
"""Scan ivus_test_0515 DICOMs and dump per-case/per-file metadata.

Forked from scripts/scan_ivus_test_0508.py with the workspace-root path
hard-corrected to the current machine and the dataset switched to the
2026-05-15 drop. Reads every FILExxxx in every CASExxxx directory and
prints metadata so we can identify which case corresponds to which
capture (B2 tungsten wire phantom; E4 evaporated-milk TGC; E5
evaporated-milk speckle / cyst phantom).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pydicom

ROOT = Path("/home/jocelynbarker/i4h-sensor-simulation/ivus_test_0515")

PRIVATE_TAGS = [
    ((0x0029, 0x1001), "gain"),
    ((0x0029, 0x1003), "diam_mm"),
    ((0x0029, 0x1006), "ar"),
    ((0x0029, 0x1007), "mode"),
    ((0x0029, 0x1008), "p1008"),
    ((0x0029, 0x1015), "frame_idx"),
]


def summarize(path: Path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:
        return {"file": path.name, "error": str(exc)}
    row = {"file": str(path.relative_to(ROOT))}
    row["depth_mm"] = str(getattr(ds, "DepthOfScanField", ""))
    row["rows"] = str(getattr(ds, "Rows", ""))
    row["cols"] = str(getattr(ds, "Columns", ""))
    n_frames = getattr(ds, "NumberOfFrames", 1)
    row["nframes"] = str(n_frames)
    try:
        region = ds.SequenceOfUltrasoundRegions[0]
        row["pix_mm"] = f"{float(region.PhysicalDeltaX) * 10.0:.4f}"
    except Exception:
        row["pix_mm"] = ""
    row["catheter"] = ", ".join(getattr(ds, "TransducerData", []) or []) or ""
    row["model"] = str(getattr(ds, "ManufacturerModelName", ""))
    row["comment"] = str(getattr(ds, "ImageComments", "")).strip()
    row["study_date"] = str(getattr(ds, "StudyDate", ""))
    row["acq_time"] = str(getattr(ds, "AcquisitionDateTime", "") or getattr(ds, "AcquisitionTime", ""))
    row["content_time"] = str(getattr(ds, "ContentTime", ""))
    row["series_desc"] = str(getattr(ds, "SeriesDescription", ""))
    row["study_desc"] = str(getattr(ds, "StudyDescription", ""))
    row["series_uid"] = str(getattr(ds, "SeriesInstanceUID", ""))
    row["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
    for tag, name in PRIVATE_TAGS:
        if tag in ds:
            v = ds[tag].value
            try:
                if hasattr(v, "__len__") and not isinstance(v, str):
                    v = ",".join(str(x) for x in v)
            except Exception:
                pass
            row[name] = str(v)
        else:
            row[name] = ""
    return row


def main():
    cases = sorted(ROOT.glob("CASE*"))
    all_rows = []
    for case in cases:
        files = sorted(case.glob("FILE*"))
        for f in files:
            row = summarize(f)
            row["case"] = case.name
            all_rows.append(row)

    print(f"# Scan of {ROOT}")
    print(f"# {len(cases)} cases, {len(all_rows)} files\n")
    for case in cases:
        case_rows = [r for r in all_rows if r["case"] == case.name]
        if not case_rows:
            print(f"## {case.name}  (no files)\n")
            continue
        print(f"## {case.name}  ({len(case_rows)} files)")
        first = case_rows[0]
        print(f"  catheter     : {first.get('catheter', '')}")
        print(f"  model        : {first.get('model', '')}")
        print(f"  study_date   : {first.get('study_date', '')}")
        print(f"  study_desc   : {first.get('study_desc', '')}")
        print(f"  series_desc  : {first.get('series_desc', '')}")
        print(f"  comment      : {first.get('comment', '')}")
        print(f"  study_uid    : {first.get('study_uid', '')}")
        print(f"  {'file':<22} {'depth':>5} {'pix_mm':>7} {'rows':>5} {'cols':>5} "
              f"{'nfr':>4} {'gain':>5} {'diam':>5} {'ar':>3} {'mode':>4} {'frm_i':>5} "
              f"{'time':<14} {'comment'}")
        for r in case_rows:
            print(f"  {r['file']:<22} "
                  f"{r.get('depth_mm','') or '':>5} "
                  f"{r.get('pix_mm','') or '':>7} "
                  f"{r.get('rows','') or '':>5} "
                  f"{r.get('cols','') or '':>5} "
                  f"{r.get('nframes','') or '':>4} "
                  f"{r.get('gain','') or '':>5} "
                  f"{r.get('diam_mm','') or '':>5} "
                  f"{r.get('ar','') or '':>3} "
                  f"{r.get('mode','') or '':>4} "
                  f"{r.get('frame_idx','') or '':>5} "
                  f"{(r.get('acq_time','') or r.get('content_time',''))[:14]:<14} "
                  f"{r.get('comment','')}")
        print()

    print("## Cross-case grouping by StudyInstanceUID")
    by_study = defaultdict(list)
    for r in all_rows:
        by_study[r.get("study_uid", "")].append(r)
    for uid, rows in by_study.items():
        cases_in_study = sorted(set(r["case"] for r in rows))
        print(f"  {uid}  ->  cases: {cases_in_study}  ({len(rows)} files)")

    print("\n## Gain values seen per case")
    for case in cases:
        case_rows = [r for r in all_rows if r["case"] == case.name]
        gains = [r.get("gain", "") for r in case_rows]
        diams = [r.get("diam_mm", "") for r in case_rows]
        print(f"  {case.name}: gains={gains}  diams={diams}")


if __name__ == "__main__":
    sys.exit(main() or 0)
