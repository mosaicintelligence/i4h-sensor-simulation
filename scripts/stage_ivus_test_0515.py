#!/usr/bin/env python3
"""Stage ivus_test_0515 captures into per-experiment subfolders consumable
by the existing P_035 calibration pipeline (extract_metadata.py,
annotate_wires.py, fit_alignment.py, unwrap.py, extract_*.py).

Forked from `scripts/stage_ivus_test_0508.py` but with an entirely DICOM-
driven layout: the bench operator encoded the experiment label and
phantom position in the DICOM PatientName / PatientID fields on the
Volcano console.

Cross-reference with the SOP at
`instrument-calibration/docs/interim_milk_phantom_sop.md` —
T1-E4* (milk variants) is a four-phase capture run:

  Phase A1: undiluted milk, multi-gain @ P1,P2,P3   <- T1E4a (this drop)
  Phase B1: undiluted milk + WIRE phantom (c-cal)   <- T1E4b (this drop)
  Phase A2: 1:1 diluted milk, multi-gain @ P1,P2,P3 <- T1E4c (this drop)
  Phase B2: 1:1 diluted milk + WIRE phantom (c-cal) <- T1E4d (this drop)

T1-E5* is a single-session 2% milk-agar-glycerin cyst gel capture; the
DICOM PatientID position labels (P3 / P4 / P1-P2) are KNOWN to be
unreliable for E5 (per bench notes), so we bucket by acquisition CASE
rather than by PID.

For all milk experiments (T1E4*, T1E5) the bench operator encoded
gel/fluid temperature (e.g. "23C", "23.5C") into the DICOM PatientID
slot — that's the temperature column we record for sound-speed
correction, NOT a phantom-position index. For T1E2 (water + tungsten
wire phantom), PatientID *is* the phantom-position index (P1-P4).

PatientName -> experiment subfolder mapping
-------------------------------------------

  PatientName prefix    PatientID    Experiment subfolder
  --------------------- ------------ --------------------------
  T1E2 (W-Wire-Pn)      Pn           b2_w_wire_p<n>            (4 takes)
  T1E5 (Milk-Agar-cyst) (unreliable) e5_milk_cyst_take<i>      (3 takes)
  T1E4a (milk-gain-     "<temp>C"    e4a_milk_gain_p<n>        (3 takes)
         sweep-p<n>)
  T1E4b (milk-wire-     "<temp>C"    e4b_milk_undil_wire       (1 take)
         phantom)
  T1E4c (milk-water-    "<temp>C"    e4c_milk_water_gain_p<n>  (3 takes)
         gain-p<n>)
  T1E4d (milk-water-    "<temp>C"    e4d_milk_water_wire       (1 take)
         wire)

The three workstream-relevant aggregates are:

  B2  (tungsten wire, sub-Mie-resonance)  -> b2_w_wire_p[1-4]
  E4  (evap-milk TGC variants, 4 phases)  -> e4{a,b,c,d}_*
  E5  (milk-agar-glycerin cyst, 2% agar)  -> e5_milk_cyst_*

NOTE on wire phantoms: the bench wire phantom appears in *three* sets
of subfolders — b2_w_wire_p[1-4] (clean water), e4b_milk_undil_wire
(undiluted milk for c-cal), and e4d_milk_water_wire (1:1 diluted milk
for c-cal). All three sets need wire annotation + alignment fitting
before extract_psf.py can run.

Each subfolder gets a manifest_subset.csv + FILE####.dcm symlinks.
A combined `ivus_test_0515/manifest.csv` is also written for parity
with the 0508 staging.

Wire phantoms (T1E2 + T1E4{b,d}) get a DXF symlink for the wire layout
matching the apparatus that was on the bench. The 0515 captures all
used the new (12-wire spiral) apparatus so we point at
`instrument-calibration/hardware/wire_spiral_v1.dxf`.
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pydicom

ROOT = Path("/home/jocelynbarker/i4h-sensor-simulation")
DATASET = ROOT / "ivus_test_0515"
MANIFEST = DATASET / "manifest.csv"

# All 0515 captures use the new (12-wire spiral) wire phantom apparatus.
DXF_NEW_APPARATUS = ROOT / "instrument-calibration" / "hardware" / "wire_spiral_v1.dxf"

PRIVATE_TAGS = {
    "gain_slider": (0x0029, 0x1001),
    "diameter_mm": (0x0029, 0x1003),
    "ar_capability": (0x0029, 0x1006),
    "ar_state": (0x0029, 0x1007),
    "frame_index": (0x0029, 0x1015),
}

PATIENT_NAME_RE = re.compile(
    r"^T1(?P<exp>E2|E4[abcd]|E5)\^(?P<sublabel>[^^]*)\^?",
    re.IGNORECASE,
)


@dataclass
class FileRow:
    case: str
    file: str
    patient_name: str
    patient_id: str
    experiment_label: str   # e.g. "E2", "E4a", "E5"
    sublabel: str           # e.g. "W-Wire-P4", "Milk-Agar-cyst"
    depth_mm: int
    gain_slider: float
    ar_state: int
    diameter_mm: float
    frames: int
    acq_time: str

    @property
    def phantom_position(self) -> str:
        """Phantom-axial-position index Pn (or P1-P2 for combined scans).

        For T1E2 (water + W-wire) the PatientID encodes Pn directly
        (this is reliable). For all milk experiments the PatientID is
        TEMPERATURE — never a position. For T1E4{a,c} we read the
        position from the PatientName ``-p<n>`` suffix instead.
        T1E5 PatientID is documented-unreliable so we never trust it.
        """
        exp_lo = self.experiment_label.lower()
        if exp_lo == "e2":
            pid = self.patient_id.upper()
            m = re.search(r"\bP(\d)(?:-P(\d))?\b", pid)
            if m:
                return ("p%s_p%s" % (m.group(1), m.group(2))) \
                    if m.group(2) else ("p%s" % m.group(1))
        if exp_lo in ("e4a", "e4c"):
            m = re.search(r"-?p(\d)$", self.sublabel.lower())
            if m:
                return "p%s" % m.group(1)
        return ""

    @property
    def temperature_c(self) -> str:
        """Temperature recorded in the DICOM PatientID for milk experiments.

        Returns the raw string (e.g. "23C", "23.5C") or "" if not parseable.
        """
        if self.experiment_label.lower() == "e2":
            return ""
        pid = self.patient_id.strip()
        # Accept "23C", "23.5C", "22 C", etc.
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*C\b", pid, re.IGNORECASE)
        if m:
            return f"{m.group(1)}C"
        return ""


def read_one(case_dir: Path, file_path: Path) -> FileRow | None:
    try:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
    except Exception as exc:
        print(f"  err {file_path}: {exc}", file=sys.stderr)
        return None
    pn = str(getattr(ds, "PatientName", ""))
    pid = str(getattr(ds, "PatientID", ""))
    m = PATIENT_NAME_RE.match(pn)
    if not m:
        print(f"  warn {file_path}: unparsable PatientName={pn!r}", file=sys.stderr)
        exp = "UNKNOWN"
        sublabel = pn
    else:
        exp = m.group("exp").upper().replace("E4A", "E4a").replace("E4B", "E4b") \
            .replace("E4C", "E4c").replace("E4D", "E4d")
        sublabel = m.group("sublabel")

    def _priv_float(tag):
        try:
            return float(ds[tag].value)
        except Exception:
            return float("nan")

    def _priv_int(tag):
        try:
            return int(ds[tag].value)
        except Exception:
            return -1

    depth = int(getattr(ds, "DepthOfScanField", 0) or 0)
    gain = _priv_float(PRIVATE_TAGS["gain_slider"])
    ar = _priv_int(PRIVATE_TAGS["ar_state"])
    diam = _priv_float(PRIVATE_TAGS["diameter_mm"])
    nframes = int(getattr(ds, "NumberOfFrames", 1) or 1)
    acq = str(getattr(ds, "AcquisitionDateTime", "") or
              getattr(ds, "AcquisitionTime", "") or
              getattr(ds, "ContentTime", ""))[:14]
    return FileRow(
        case=case_dir.name,
        file=file_path.name,
        patient_name=pn,
        patient_id=pid,
        experiment_label=exp,
        sublabel=sublabel,
        depth_mm=depth,
        gain_slider=gain,
        ar_state=ar,
        diameter_mm=diam,
        frames=nframes,
        acq_time=acq,
    )


# T1-E5 PatientID positions are unreliable per bench notes; bucket E5 by
# acquisition CASE so each independent acquisition session ends up in its
# own subfolder, regardless of what the operator typed in PatientID.
E5_CASE_TO_TAKE = {
    # Filled in dynamically below — first encountered E5 case is take1, etc.
}


def experiment_subfolder(row: FileRow) -> str:
    """Map (experiment_label, sublabel, phantom_position) to a subfolder name."""
    exp = row.experiment_label.lower()
    pos = row.phantom_position
    if exp == "e2":
        return f"b2_w_wire_{pos}" if pos else "b2_w_wire"
    if exp == "e5":
        # E5 PatientID position is documented-unreliable; bucket by case.
        if row.case not in E5_CASE_TO_TAKE:
            E5_CASE_TO_TAKE[row.case] = f"take{len(E5_CASE_TO_TAKE) + 1}"
        return f"e5_milk_cyst_{E5_CASE_TO_TAKE[row.case]}"
    if exp.startswith("e4"):
        # e4a/e4b/e4c/e4d each get their own bucket; phantom-position appended
        # only for the multi-position phases (e4a = A1, e4c = A2). The two
        # wire-phantom phases (e4b = B1, e4d = B2) are single-take so no
        # position suffix.
        base = {
            "e4a": "e4a_milk_gain",
            "e4b": "e4b_milk_undil_wire",
            "e4c": "e4c_milk_water_gain",
            "e4d": "e4d_milk_water_wire",
        }[exp]
        return f"{base}_{pos}" if pos and base.endswith(("gain",)) else base
    return f"unknown_{exp}"


def needs_wire_dxf(subfolder: str) -> bool:
    return subfolder.startswith((
        "b2_w_wire",
        "e4b_milk_undil_wire",
        "e4d_milk_water_wire",
    ))


def main() -> int:
    cases = sorted(DATASET.glob("CASE*"))
    rows: list[FileRow] = []
    for case in cases:
        for fp in sorted(case.glob("FILE*")):
            r = read_one(case, fp)
            if r is not None:
                rows.append(r)

    print(f"# stage_ivus_test_0515 — read {len(rows)} files across {len(cases)} cases")

    # E5 takes are bucketed in chronological order, so resolve case->take by
    # earliest acq_time per E5 case before bucketing.
    e5_cases_by_time = sorted(
        {r.case for r in rows if r.experiment_label.lower() == "e5"},
        key=lambda c: min((r.acq_time for r in rows if r.case == c), default=""),
    )
    for i, c in enumerate(e5_cases_by_time):
        E5_CASE_TO_TAKE[c] = f"take{i + 1}"

    # Group by subfolder
    by_sub: dict[str, list[FileRow]] = {}
    for r in rows:
        sub = experiment_subfolder(r)
        by_sub.setdefault(sub, []).append(r)

    # Stage each subfolder
    raw_root = DATASET / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    all_manifest_rows = []
    for sub, sub_rows in sorted(by_sub.items()):
        out_dir = raw_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any prior FILE*.dcm + DXF symlinks
        for prior in out_dir.glob("FILE*.dcm"):
            if prior.is_symlink() or prior.is_file():
                prior.unlink()
        for prior in out_dir.glob("*.dxf"):
            if prior.is_symlink() or prior.is_file():
                prior.unlink()
        # Sort within subfolder: depth desc (60/45/30/22/15), gain asc, then case+file lex.
        sub_rows.sort(key=lambda r: (-r.depth_mm, r.gain_slider, r.case, r.file))
        staged_rows = []
        for i, r in enumerate(sub_rows):
            src = DATASET / r.case / r.file
            dst = out_dir / f"FILE{i:04d}.dcm"
            rel_src = Path("..") / ".." / src.relative_to(DATASET)
            dst.symlink_to(rel_src)
            staged_rows.append({
                "case": r.case,
                "file": r.file,
                "experiment": r.experiment_label,
                "sop_phase": _sop_phase(r.experiment_label),
                "take": r.sublabel,
                "apparatus": "new",
                "phantom": _phantom_kind(sub),
                "diameter_mm": int(r.depth_mm),
                "gain_slider": int(round(r.gain_slider)) if r.gain_slider == r.gain_slider else "",
                "mode_flag": r.ar_state,
                "ar_state": "on" if r.ar_state == 1 else ("off" if r.ar_state == 0 else ""),
                "sweep_role": _sweep_role(r),
                "frames": r.frames,
                "patient_name": r.patient_name,
                "patient_id": r.patient_id,
                "phantom_position": r.phantom_position,
                "temperature_c": r.temperature_c,
                "acq_time": r.acq_time,
                "notes": f"{sub} {'30mm' if r.depth_mm == 30 else f'{r.depth_mm}mm'} g{int(round(r.gain_slider))}",
                "staged_file": dst.name,
            })
        # Write per-subfolder manifest
        sub_csv = out_dir / "manifest_subset.csv"
        if staged_rows:
            with sub_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(staged_rows[0].keys()))
                w.writeheader()
                w.writerows(staged_rows)
        # Wire DXF for wire phantoms
        if needs_wire_dxf(sub):
            dxf_dst = out_dir / DXF_NEW_APPARATUS.name
            if dxf_dst.exists() or dxf_dst.is_symlink():
                dxf_dst.unlink()
            rel_dxf = Path("..") / ".." / ".." / DXF_NEW_APPARATUS.relative_to(ROOT)
            dxf_dst.symlink_to(rel_dxf)
        all_manifest_rows.extend(staged_rows)
        print(f"  staged {len(staged_rows):3d} files -> {out_dir.relative_to(ROOT)}")

    # Top-level combined manifest
    if all_manifest_rows:
        with MANIFEST.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(all_manifest_rows)
        print(f"\n  wrote combined manifest: {MANIFEST.relative_to(ROOT)} "
              f"({len(all_manifest_rows)} rows)")

    print("\nDone. Run extract_metadata.py per subfolder, e.g.:")
    print("  python instrument-calibration/p035_visions/extract_metadata.py \\")
    print("      --dataset ivus_test_0515/raw/b2_w_wire_p4 \\")
    print("      --out     ivus_test_0515/raw/b2_w_wire_p4/derived/frames_meta.csv")
    return 0


def _phantom_kind(subfolder: str) -> str:
    if subfolder.startswith("b2_w_wire"):
        return "tungsten_wire_phantom_water"
    if subfolder.startswith("e5_milk_cyst"):
        return "milk_agar_glycerin_cyst_2pct_agar"
    if subfolder.startswith("e4a_milk_gain"):
        return "evap_milk_undiluted"
    if subfolder.startswith("e4b_milk_undil_wire"):
        return "evap_milk_undiluted_with_wire_phantom"
    if subfolder.startswith("e4c_milk_water_gain"):
        return "evap_milk_diluted_1to1"
    if subfolder.startswith("e4d_milk_water_wire"):
        return "evap_milk_diluted_1to1_with_wire_phantom"
    return "unknown"


def _sop_phase(experiment_label: str) -> str:
    """Map our experiment label to the canonical SOP phase letter."""
    return {
        "E2": "T1-E2 (water + W-wire phantom)",
        "E4a": "T1-E4* phase A1 (undiluted milk multi-gain)",
        "E4b": "T1-E4* phase B1 (undiluted milk + wire c-cal)",
        "E4c": "T1-E4* phase A2 (1:1 diluted milk multi-gain)",
        "E4d": "T1-E4* phase B2 (1:1 diluted milk + wire c-cal)",
        "E5": "T1-E5* (milk-agar-glycerin cyst, 2% agar)",
    }.get(experiment_label, experiment_label)


def _sweep_role(r: FileRow) -> str:
    """Subdivide rows by depth band so existing extractors can filter."""
    return f"{r.depth_mm}mm_gain_sweep"


if __name__ == "__main__":
    raise SystemExit(main())
