# Instrument calibration

This folder holds the per-instrument calibration code that fits the parameters
of a specific ultrasound device (catheter + console pair) from bench-capture
data, and writes the canonical YAML configuration that the simulator in
`../i4h-sensor-simulation/ultrasound-raytracing/` consumes at runtime.

The general (instrument-agnostic) calibration **protocol** lives with the
simulator at `../i4h-sensor-simulation/docs/ivus_calibration_protocol.md`.
The code here implements that protocol against a particular dataset.

## Layout

```
instrument-calibration/
├── README.md                ← (this file)
├── pyproject.toml           ← dependencies for the fitting code
└── <instrument-id>/         ← one subfolder per device under calibration
    ├── README.md
    ├── calibration_delta.md ← what has been fit, what is still uncertain,
    │                          and what to ask the bench team for
    ├── parameter_sheet.csv  ← working notes / param-tracking spreadsheet
    ├── <yaml-output>.yaml   ← the canonical OUTPUT of the calibration —
    │                          consumed by the simulator
    ├── extract_*.py         ← per-stage fitting scripts (E2 / E4 / E5 / E6 / E7 / …)
    ├── fit_alignment.py     ← upstream alignment of the catheter to the data
    ├── annotate_wires.py    ← optional manual GUI for wire annotation
    ├── unwrap.py            ← scan-conversion / un-scan-conversion utilities
    └── polar_utils.py       ← shared helpers used by several extract scripts
```

## Currently calibrated instruments

| Folder | Catheter | Console | Frequency | Source data |
|--------|----------|---------|-----------|-------------|
| `p035_visions/` | Visions PV .035 (10 MHz peripheral IVUS) | Volcano s5 / s5i | 10 MHz | `../P_035_PointScatter/` |

## Where the YAML is consumed

The simulator reads its config via `IvusSimConfig.from_yaml(path)` (see
`../i4h-sensor-simulation/ultrasound-raytracing/raysim/config.py`). Point it
at the YAML produced by the relevant subfolder here, e.g.

```bash
cd /path/to/ivus-sim
python3 -m raysim ... --config instrument-calibration/p035_visions/volcano_s5i.yaml
```

## Running the fitting pipeline

All scripts default to relative paths from the workspace root
(`/path/to/ivus-sim`), so run them from there:

```bash
cd /path/to/ivus-sim
python3 instrument-calibration/p035_visions/extract_metadata.py
python3 instrument-calibration/p035_visions/fit_alignment.py
python3 instrument-calibration/p035_visions/unwrap.py
# … then the per-stage extracts (ringdown, noise, tgc, psf, gain_lut)
```

See each instrument folder's `README.md` for the per-instrument run order and
the `calibration_delta.md` for the current calibration status.
