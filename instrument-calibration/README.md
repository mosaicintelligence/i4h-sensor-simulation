# Instrument calibration

This folder holds the per-instrument calibration code that fits the parameters
of a specific ultrasound device (catheter + console pair) from bench-capture
data, and writes the canonical YAML configuration that the simulator in
`../i4h-sensor-simulation/ultrasound-raytracing/` consumes at runtime.

## Documentation (start here)

| Doc | Role |
|-----|------|
| [`docs/probe_onboarding_protocol.md`](docs/probe_onboarding_protocol.md) | **Start here for a new probe** — lessons-learned O1–O3 workflow |
| [`docs/ivus_calibration_protocol.md`](docs/ivus_calibration_protocol.md) | Encyclopedic E1–E9 reference |
| [`docs/interim_milk_phantom_sop.md`](docs/interim_milk_phantom_sop.md) | Milk-bath recipe when no commercial phantom is on hand |
| [`docs/tier2_acceptance_protocol.md`](docs/tier2_acceptance_protocol.md) | Sim-vs-real Tier 2 acceptance |

The code here implements those protocols against a particular dataset.

## Layout

```
instrument-calibration/
├── README.md                ← (this file)
├── docs/                    ← instrument-agnostic protocols
├── pyproject.toml           ← dependencies for the fitting code
└── <instrument-id>/         ← one subfolder per device under calibration
    ├── README.md
    ├── calibration_delta.md ← lab notebook / historical follow-ups
    ├── parameter_sheet.csv  ← working notes / param-tracking spreadsheet
    ├── <yaml-output>.yaml   ← the canonical OUTPUT of the calibration —
    │                          consumed by the simulator
    ├── extract_*.py         ← per-stage fitting scripts (E2 / E4 / E5 / E6 / E7 / …)
    ├── derive_*.py          ← re-derive individual YAML fields from staged outputs
    ├── fit_alignment.py     ← upstream alignment of the catheter to the data
    ├── annotate_wires.py    ← optional manual GUI for wire annotation
    ├── unwrap.py            ← scan-conversion / un-scan-conversion utilities
    └── polar_utils.py       ← shared helpers used by several extract scripts
```

## Currently calibrated instruments

| Folder | Catheter | Console | Frequency | Source data |
|--------|----------|---------|-----------|-------------|
| `p035_visions/` | Visions PV .035 (10 MHz peripheral IVUS) | Volcano s5 / s5i | 10 MHz | Live anchors: `ivus_test_0508/`, `ivus_test_0515/` (raw DICOMs out of band). Historical first pass: `P_035_PointScatter/` |

Shipping PV .035 config uses an uncalibrated elevational default (`elevational_height_mm: 1.5`, `num_elevational_samples: 8`). See [`p035_visions/tier1_elevational_waiver.md`](p035_visions/tier1_elevational_waiver.md) — checked-in Tier 1 results are the last **2D** catalog until a 2.5D re-run.

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

See each instrument folder's `README.md` for the per-instrument run order.
For live status prefer the YAML comments, `tier1_results/`, and (for PV .035)
`tier1_elevational_waiver.md`; treat `calibration_delta.md` as a lab notebook.
