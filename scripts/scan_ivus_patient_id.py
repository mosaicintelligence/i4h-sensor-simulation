#!/usr/bin/env python3
"""Print PatientName / PatientID for every DICOM in ivus_test_0508."""
from __future__ import annotations

from pathlib import Path

import pydicom

ROOT = Path("/Users/jocelynbarker/ivus-sim/ivus_test_0508")


def main() -> int:
    cases = sorted(ROOT.glob("CASE*"))
    print(f"# {ROOT}")
    print(f"{'case':<10} {'patient_name':<48} {'patient_id':<24} {'study_date':<10} {'gain':>5} {'diam':>5} {'mode':>4} {'nfr':>4}")
    for case in cases:
        files = sorted(case.glob("FILE*"))
        for f in files:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            pn = str(getattr(ds, "PatientName", ""))
            pid = str(getattr(ds, "PatientID", ""))
            sd = str(getattr(ds, "StudyDate", ""))
            gain = ds.get((0x0029, 0x1001)).value if (0x0029, 0x1001) in ds else ""
            diam = ds.get((0x0029, 0x1003)).value if (0x0029, 0x1003) in ds else ""
            mode = ds.get((0x0029, 0x1007)).value if (0x0029, 0x1007) in ds else ""
            nfr = getattr(ds, "NumberOfFrames", 1)
            label = f"{case.name}/{f.name}"
            print(f"{label:<22} {pn:<48} {pid:<24} {sd:<10} {gain:>5} {diam:>5} {mode!s:>4} {nfr:>4}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
