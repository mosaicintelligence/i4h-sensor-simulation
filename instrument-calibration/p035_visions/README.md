# PV .035 / Volcano s5i calibration

Fits the simulator parameters for the **Visions PV .035 (10 MHz peripheral
IVUS catheter)** running on a **Volcano s5 / s5i console**.

Primary bench anchors (raw DICOMs out of band; not committed to git):

- `ivus_test_0508/` — E2 gain sweeps + E6 AR-on/off water captures
- `ivus_test_0515/` — Wave-0 B2 tungsten wire PSF, E4a/E4c milk TGC/speckle/noise

Historical first pass: `P_035_PointScatter/` (partial gain × diameter grid).

The output is `volcano_s5i.yaml` (this folder), which the simulator at
`../../i4h-sensor-simulation/ultrasound-raytracing/` consumes.

## Shipping defaults (read first)

| Field | Value | Notes |
|-------|------:|-------|
| `probe.elevational_height_mm` | **1.5** | Product default, **not** an E3-measured FWHM |
| `probe.num_elevational_samples` | **8** | Planes averaged by `mean_planes` (~8× OptiX vs 2D) |
| `processing.gain_db` | **58.09** | From `derive_gain_db.py` (see YAML comments) |

Checked-in `tier1_results/` describe the last **2D** (`N = 1`) gate. Until a
2.5D re-run, read them with [`tier1_elevational_waiver.md`](tier1_elevational_waiver.md).

**Live status:** YAML comments + waiver + [`tier1_results/tier1_results.md`](tier1_results/tier1_results.md).
Treat [`calibration_delta.md`](calibration_delta.md) as a lab notebook / history.

## Files

| File | Purpose |
|------|---------|
| `volcano_s5i.yaml` | **Canonical output** — the simulator config produced by this calibration. Comments in the file document which stage produced each parameter and its uncertainty. |
| `tier1_elevational_waiver.md` | 2.5D product default vs checked-in 2D Tier 1 catalog |
| `calibration_delta.md` | Lab notebook: historical fits, bench inventories, follow-up asks |
| `parameter_sheet.csv` | Working spreadsheet that tracks every parameter's value, source experiment, and confidence. |

### Fitting scripts

Run from the workspace root (`ivus-sim/`). Order matters — later stages
depend on earlier ones.

| # | Script | Stage | What it produces |
|---|--------|-------|-------------------|
| 0 | `extract_metadata.py` | metadata | `derived/frames_meta.csv` (per-frame DICOM tags) |
| 1 | `extract_calibration_inputs.py` | DXF + auto-align | `derived/wire_positions.csv`, `derived/box_geometry.json` |
| 1a | `annotate_wires.py` | manual annotation | `derived/wire_annotations.json` (only if the auto-align is off) |
| 2 | `fit_alignment.py` | catheter ↔ image alignment | `derived/alignment_fit.{csv,json}` (theta0, chirality, radial_scale) |
| 3 | `unwrap.py` | scan-conversion | `derived/polar/*.npy` (per-frame polar arrays) |
| 4 | `extract_ringdown.py` | E6 | `derived/ringdown/*` → `processing.ring_down.*` |
| 5 | `extract_noise.py` | E5 | `derived/noise/*` → `processing.noise.*` |
| 6 | `extract_tgc.py` | E4 | `derived/tgc/*` → `processing.tgc_control_points` |
| 7 | `extract_psf.py` | E2 | `derived/psf/*` → `probe.{element_radius_mm,focal_length_mm,pulse_duration_cycles}` |
| 8 | `extract_gain_lut.py` | E7 | `derived/gain_lut/*` → `processing.{log_multiplier,log_floor,dynamic_range_db,gain_db,…}` |

Extended extracts (0515 bench): `extract_speckle.py`, `extract_tgc_wirefree.py`,
`annotate_cysts.py`.

### YAML population (`derive_*.py`)

Re-derive individual `volcano_s5i.yaml` fields from staged bench outputs:
`derive_gain_db.py`, `derive_noise_sigma.py`, `derive_slider_to_db.py`,
`derive_axial_psf_pulse_duration.py`, `derive_lateral_psf_sigma_theta.py`,
`derive_ringdown_amplitude.py`.

Most `derived/*` outputs land under the relevant `ivus_test_*/raw/*/derived/`
or `P_035_PointScatter/derived/` trees (see each script's header).

### Evaluation

| Script | Output |
|--------|--------|
| `tier1_evaluation.py` | `tier1_results/tier1_results.md` — Tier 1 physical-fidelity gate |
| `tier1_self_validate.py` | `tier1_results/self_validate/` — metric self-validation harness |
| `bench_evidence.py` | Bench-anchor figures embedded in the tier1 report |
| `vessel_evaluation.py` | `vessel_evaluation_output/VESSEL_EVALUATION_REPORT.md` |
| `rerender_tier1.py` | Re-render tier1 figures without a full re-eval |

Diagnostics: `visualize_wave0_psf.py`, `visualize_b2_diagnostic.py`.

### Shared helpers

| File | What |
|------|------|
| `polar_utils.py` | Common polar-domain helpers (`load_polar_dataset`, `build_anechoic_mask`, `predict_wire_polar_positions`) used by `extract_noise.py`, `extract_tgc.py`, `extract_psf.py`, `extract_gain_lut.py`. |

## Quick re-run after new bench data arrives

See `calibration_delta.md` § "What re-runs to do when the new data arrives"
for the historical sequence. For elevational calibration, run E3 and refresh
Tier 1 under the 1.5 mm / 8-plane default (closes the waiver).
