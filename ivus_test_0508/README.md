# ivus_test_0508 — bench dataset for s5i / Visions PV .035 calibration

Captured 2026-05-09 at the bench, intended to follow the calibration plan
laid out in `instrument-calibration/p035_visions/calibration_delta.md`
(Capture A gain sweep, Capture C AR-on/AR-off cross-check; Capture B
not done — copper magnet wire was kept).

The DICOMDIR organizes everything as anonymous "CASExxxx" with no
human-readable label, so the labelling here was reverse-engineered from
the **PatientName** / **PatientID** fields (which the operator used as
free-text experiment labels) plus the per-file private tags
`(0029,1001)` (gain slider), `(0029,1003)` (imaging diameter mm), and
`(0029,1007)` (AR flag — see "Private-tag interpretation" below).

## Per-experiment summary

| Experiment | Cases | Apparatus | Phantom | Diameter | # vids | Gain order | Notes |
|---|---|---|---|---:|---:|---|---|
| **E2 sweep #1** (Capture A) | `CASE0007/FILE0002-9` + `CASE0006` | **ORIGINAL apparatus** | wire phantom | 60 mm | 18 | asc 0→68 step 4 | First sweep of the day, 20:40-20:50 |
| **E2 sweep #2** (Capture A) | `CASE0005` + `CASE0004` | **NEW apparatus, widest** | wire phantom | 60 mm | 18 | desc 68→0 step 4 | 21:11-21:20 |
| **E2 sweep #3** (Capture A) | `CASE0003` + `CASE0002` | **NEW apparatus, narrow** | wire phantom | **30 mm** | 18 | asc 0→68 step 4 | 21:23-21:32 — best axial sampling (0.06 mm/px) |
| **E6 pilot** (Capture C) | `CASE0007/FILE0000-1` | n/a | **degassed water only** | 60 mm | 2 | g=44, AR-off + AR-on | Single-gain AR pair the operator did first |
| **E6 take 2** (Capture C) | `CASE0000` | n/a | **degassed water only** | 30 + 60 mm | 6 | g=50/D=30, g=50/D=60, g=40/D=60 each AR-off + AR-on | Multi-(gain,diameter) grid AR cross-check |
| **E6 take 1, partial** | `CASE0008` | n/a | **degassed water only** | 60 mm | 1 | g=44 AR-off only | Incomplete; superseded by Take 2 |
| ~~CASE0001~~ | — | — | — | — | — | — | **Drop**: byte-identical pixel-data duplicate of CASE0000 (verified by SHA1 hashes; differs only in StudyInstanceUID) |
| ~~CASE0009~~ | — | — | — | — | — | — | **Drop**: PatientID `7595`, PatientName `vbgy v` — onboarding/test typing |

Total usable: **63 videos** (54 wire-phantom + 9 water-only).

The full per-file mapping (gain, diameter, AR state, frames, acquisition
time, patient label, sweep role) lives in `manifest.csv` next to this
README. Downstream scripts should read that CSV rather than re-parsing
the patient labels.

## Private-tag interpretation (this firmware)

Verified by paired-capture inner-disc intensity (CASE0000 and
CASE0007/FILE0000-1 — same gain/diameter, taken seconds apart, only the
private tag flips):

| Tag | Name | Meaning on this dataset |
|---|---|---|
| `(0029,1001)` FD | gain slider | 0..68 in step-4 increments (Capture A); 40, 44, 50 (Capture C) |
| `(0029,1003)` FD | imaging diameter mm | 30 or 60 |
| `(0029,1006)` US | "ar capability"? | =1 in every frame in this dataset; **not** the AR-on/off flag |
| `(0029,1007)` US | **AR-on/off flag** | 0 = AR-OFF (raw ringdown visible); 1 = AR-ON (ringdown subtracted). See "Evidence" below. |
| `(0029,1015)` US | frame index | matches FILExxxx ordering |

### Evidence for `(0029,1007)` ↔ AR

In every paired Capture-C clip taken seconds apart at identical
gain/diameter, when `(0029,1007)` flips 0 → 1 the **inner-disc**
(catheter ringdown) mean drops dramatically while the **mid-disc**
and **outer-disc** means are unchanged:

```
CASE0000/F0  g=50 D=30 mode=0  inner=38.7  mid=31.9  outer=34.7
CASE0000/F1  g=50 D=30 mode=1  inner=10.1  mid=37.7  outer=36.4
CASE0000/F2  g=50 D=60 mode=0  inner=139.6 mid=38.0  outer=47.3
CASE0000/F3  g=50 D=60 mode=1  inner=13.2  mid=37.1  outer=46.1
CASE0000/F4  g=40 D=60 mode=0  inner=80.6  mid=13.3  outer=16.1
CASE0000/F5  g=40 D=60 mode=1  inner=10.8  mid=13.2  outer=16.1
CASE0007/F0  g=44 D=60 mode=0  inner=105.4 mid=13.4  outer=15.1
CASE0007/F1  g=44 D=60 mode=1  inner=10.7  mid=14.0  outer=15.0
```

Mode=0 reproduces the raw ringdown signature; mode=1 reproduces the
AR-subtracted appearance. So mode_flag=1 is the standard clinical mode
and mode_flag=0 is the AR-OFF cross-check.

> **Open follow-up:** the existing P_035 dataset has `mode_flag=0`
> on every frame except FILE0013, which under this interpretation
> would mean P_035 was captured in AR-OFF mode — contradicting the
> current `processing.ring_down.subtract_reference: true` setting in
> `volcano_s5i.yaml`. Resolve before applying the new gain LUT.

## Pixel pitch by diameter

From `(0018,602C)` PhysicalDeltaX (cm/px → mm/px):

| Diameter | Pixel pitch | Comment |
|---:|---:|---|
| **30 mm** (new on this catheter) | **0.060 mm/px** | 2× better axial sampling than the original P_035 dataset |
| 60 mm | 0.120 mm/px | Same as P_035 |

The previous calibration_delta plan assumed the smallest available
diameter would be 35 mm at 0.07 mm/px; we got 30 mm at 0.06 mm/px,
which is even better.

## Files of interest

- `manifest.csv` — case → experiment → apparatus → diameter → gain → AR
- `CASE000{2..9}/FILE*` — DICOM data (do not modify)
- `DICOMDIR`, `Show_Studies.exe`, `Viewer/` — Volcano viewer payload (ignore)
- `VH0000..VH0009` — empty placeholder dirs from the export

## Scripts that read this dataset

- `scripts/scan_ivus_test_0508.py` — print the gain/diameter/mode
  parameters for every file
- `scripts/scan_ivus_patient_id.py` — print the PatientName / PatientID
  used as experiment labels
- `scripts/verify_duplicates_and_mode.py` — sanity check that
  CASE0000 == CASE0001 by pixel hash, and that mode_flag flips look like AR
