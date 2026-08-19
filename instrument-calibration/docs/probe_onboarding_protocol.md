# Probe Onboarding Protocol — streamlined IVUS / ultrasound calibration

This is the **lessons-learned onboarding protocol** distilled from the
Visions PV .035 / Volcano s5i calibration. It supersedes the canonical
nine-experiment protocol (`ivus_calibration_protocol.md`) for the
*onboarding workflow*: it lists only the experiments that actually
produced parameters in the shipping `volcano_s5i.yaml`, and it folds in
the improvements that the PV .035 calibration *wished it had*.

The canonical protocol remains the encyclopedic reference (it documents
the full E1–E9 sweep, including the experiments we now skip on
onboarding). The interim milk-based SOP (`interim_milk_phantom_sop.md`)
remains the "no commercial phantom on hand" fall-back recipe for O3.

## Why a new protocol?

Three lessons came out of the PV .035 calibration:

1. **The displayed-image gain-sweep is the single most informative
   capture.** Every YAML field that depends on `log_multiplier` —
   `gain_db`, `noise.sigma`, `envelope_noise`, `ring_down.amplitude`,
   `tgc_control_points` in dB — was eventually anchored against a fine-
   step gain sweep at the smallest available imaging diameter. The
   original P_035 dataset (3 gain points × 1 diameter, every wire
   saturated) was insufficient and forced a second bench session
   (`ivus_test_0508`) before any of those fields could move past
   "preliminary".

2. **The uniform attenuating phantom replaces the cyst phantom for
   speckle.** The canonical E5 specifies a cyst-bearing phantom because
   it gives both a clean anechoic `roi_A` (noise floor) and a uniform
   `roi_T` (speckle). In practice the bench's three milk-agar cyst
   takes (`ivus_test_0515/raw/e5_milk_cyst_take{1,2,3}/`) were
   *deprecated* during Tier 1 evaluation: the cyst inclusions break
   uniformity over the speckle ROI, while a *fully uniform* phantom
   (`ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}/`) is uniform everywhere
   and is the cleanest speckle target by design. The deep-r tail of the
   uniform phantom also doubles as a noise-floor anchor once the gain
   sweep crosses into the LUT-floored regime. So one phantom does both
   jobs and the cyst phantom can be skipped on onboarding.

3. **The anechoic-water capture is the cleanest device-noise reference,
   and it needs more gain points than we collected.** The current
   `ring_down.amplitude` value is anchored on two slider settings
   (40 and 50 at D = 60 mm) plus one more at D = 30 mm — three
   `(slider, diameter)` triples total. That was enough to verify the
   ring-down peak location and template shape, but the user explicitly
   called out that **a full slider sweep AR-OFF (and matched AR-ON)**
   would have let us read the slider→dB curve directly off the ring-
   down peak, *independent of the wire-phantom anchor*. We add this as
   a required capture in O2.

## What we keep, what we drop

| Canonical | Onboarding | Disposition |
|---|---|---|
| E1 — Flat-reflector pulse-echo | *Bonus path only* | Requires RF tap; never performed on PV .035; impulse_response_path remains `null`. Skipped unless the console exposes RF. |
| **E2 — Spiral wire-phantom 2D PSF** | **O1 (mandatory)** | Workhorse. Drives `element_radius_mm`, `focal_length_mm`, `pulse_duration_cycles`, `lateral_psf_kernel`, `log_multiplier`, `gain_db`, slider→dB curve, dynamic range. |
| E3 — Slice-thickness sweep | Estimated in YAML; E3 still required before treating `elevational_height_mm` as calibrated | Canonical YAML ships `elevational_height_mm = 1.5` / `num_elevational_samples = 8` (~8× OptiX vs 2D). That is a placeholder, not a measured FWHM. Tier 1 thresholds tuned in strict 2D may need a waiver or re-derivation. |
| E4 — Uniform attenuation phantom | **O3 (mandatory)** | Drives `tgc_control_points`, `noise.sigma` / `envelope_noise.sigma`, speckle correlation length, milk-material scattering parameters. Doubles as the E5 replacement (see below). |
| E5 — Cyst phantom | *Dropped on onboarding* | Three takes were performed (`ivus_test_0515/raw/e5_milk_cyst_take*/`) but later marked `WAVE0_SPECKLE_PATHS_E5_LEGACY` in `tier1_evaluation.py`: a uniform phantom is a strictly better speckle anchor and is also a strictly better anechoic-tail anchor once the slider crosses the LUT-floored regime. Keep the cyst hardware for clinical-image dynamic-range demonstrations, not for parameter fitting. |
| **E6 — Ring-down (anechoic water)** | **O2 (mandatory) — expanded** | Drives `ring_down.amplitude` / `extent_mm` / `decay` / `waveform_path` / `subtract_reference`, `catheter.dead_zone_mm`. **Onboarding upgrade: capture a full slider sweep (not just 2–3 anchors) AR-OFF + matched AR-ON at each slider, at the smallest available diameter.** Gives the slider→dB curve from ring-down peak directly, independent of O1, and a multi-gain AR-residual template. |
| E7 — Grayscale / compression LUT | *Folded into O1* | The wire-phantom gain sweep at fine slider steps is the calibration. A standalone RF-injection E7 was not performed on the PV .035 and `compression_lut` remains `null` in the YAML — the sigmoidal cross-gain fit from O1 is what populates `log_multiplier`. Keep RF-injection E7 as a *bonus path* if the device exposes RF. |
| E8 — Tissue / material fit | *Out of onboarding scope* | Per-material `mu0`/`mu1`/`sigma`/`impedance` for vessel wall, calcium, lipid, etc. are a separate ex-vivo / clinical session. Onboarding produces a working catheter+console pair; the material catalog is updated later from tissue data. |
| E9 — Timing / PRF log | *Out of per-frame raysim scope* | Lives in the deferred motion / acquisition layer; not part of probe onboarding. |

## Onboarding deliverables

A successful onboarding session ships:

1. A populated `<instrument-id>/volcano_<console>.yaml` with every
   `processing.*` and `probe.*` field derived from bench data.
2. A `<instrument-id>/calibration_delta.md` documenting per-field
   uncertainty and any deferred follow-ups.
3. A `<instrument-id>/parameter_sheet.csv` with one row per YAML field
   and a `Source / Experiment` column pointing to the bench capture
   that produced it.
4. A passing Tier-1 report (`<instrument-id>/tier1_results/tier1_results.md`).

The three onboarding experiments (O1, O2, O3) together produce
**every** parameter required for items (1) and (4); items (2) and (3)
are templated against the PV .035 versions of these documents.

> All experiments use the same console, the same catheter, and the same
> **water/phantom temperature (22 ± 1 °C)** unless otherwise noted.
> Speed of sound in degassed water at 22 °C is taken as
> **c_water = 1488 m/s** (Marczak 1997).

## Common equipment

(Inherited verbatim from the canonical protocol's "Common equipment"
table — see `ivus_calibration_protocol.md`. Highlights below.)

| Item | Spec | Why it matters for onboarding |
|------|------|-------------------------------|
| Console + catheter under test | service-mode access, same S/N for all O1–O3 captures | Apparatus-offset properties (Sweep #1 vs #2 in `calibration_delta.md`) get absorbed into per-instrument YAML if and only if the same fixture is used end-to-end. |
| Water tank | ≥ 300 × 200 × 150 mm, non-reflective lining, ≥ 30 mm catheter-to-wall clearance | O1 and O2 both run in this tank; multipath off a near wall is the dominant systematic at low gain. |
| Degassed deionized water | dissolved O₂ ≤ 4 ppm | ≥ 24 h sit after degassing; T = 22 ± 1 °C. The degassing matters most for O2 — micro-bubbles in the inner 5 mm pull `ring_down.amplitude` off by a measurable amount. |
| RF capture (optional) | DAQ ≥ 100 MS/s, ≥ 12-bit | If available, enables the *bonus path* E1 and an absolute (rather than displayed) compression LUT. The PV .035 calibration ran entirely on displayed-image DICOM and shipped fine, so RF is a "nice to have", not a blocker. |
| Frame grabber / DICOM export | the console's standard DICOM cine export is sufficient | All three onboarding experiments analyze post-scan-conversion polar B-mode via the same `extract_*.py` pipeline that the PV .035 used. |

## O1 — Wire-phantom gain sweep at smallest diameter (mandatory)

**Replaces canonical E2 + most of E7. Produces:**
`probe.pulse_duration_cycles`, `probe.element_radius_mm`,
`probe.focal_length_mm`, `processing.lateral_psf_kernel.sigma_theta_rad`,
`processing.log_multiplier`, `processing.gain_db`,
`processing.dynamic_range_db`, the slider→dB curve, and the
azimuthal-uniformity QC.

### Hardware

The Archimedean **12-wire spiral fixture** from canonical Appendix A.1
(STLs in `hardware/wire_spiral_disc.stl` + `wire_spiral_standoff.stl`).
Critical onboarding deltas from canonical E2:

- **Wire material:** *nylon monofilament, 70–100 µm* is the default for
  10 MHz probes. The PV .035 calibration started with 36 AWG copper
  magnet wire and lost most of the low-gain frames to saturation; 30 µm
  tungsten works at 20 MHz but saturates at any usable PV .035 gain.
  Saturation is **the** dominant onboarding-time risk for the PSF /
  log_multiplier fits — pick the wire material that does *not* saturate
  on the deepest unsaturated slider you'll capture.
- **Apparatus check before the first capture.** Image the empty fixture
  in water and verify the catheter sits on the spiral centre to within
  ±0.1 mm (the PV .035 calibration saw a 7.6 % radial-scale offset on
  the original apparatus that propagated all the way to the
  `alignment_fit.csv` step — see `calibration_delta.md` § "Alignment
  fit summary (post-cleanup)"). Reject the apparatus if the catheter
  is not concentric with the spiral.

### Capture matrix

Three captures (P1, P2, P3, P4 — four catheter positions / rotations).
The PV .035 used 4 catheter positions × 12 wires/position = 48 wire
samples per gain (after rejecting the saturated ones), which is what
the Wave 0 / B2 anchor in
`ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/` ships against.

For each catheter position:

- **Imaging diameter:** the smallest the catheter supports. PV .035
  delivered 30 mm at 0.06 mm/pixel; this is what enables the axial PSF
  to be sampled at ≥ 2 pixels per wavelength. The wire-phantom diameter
  at 60 mm sampling (0.12 mm/pixel) was insufficient for `pulse_duration_cycles`.
- **Slider sweep:** every 4 slider units from 0 to 68 (18 captures per
  position). This is the level of granularity needed to fit a
  sigmoidal `compression_lut` and to detect non-linearity in the
  slider→dB curve. The PV .035 dataset originally had 3 slider points
  (44, 54, 64) — that was not enough; the 18-point sweep added in
  `ivus_test_0508` was decisive.
- **30 frames per slider** for cross-frame averaging at the per-wire
  peak.

A second diameter (~60 mm) is a useful cross-check (apparatus-offset
detection, gain-curve invariance across diameter) but is **not
required** — it costs ~10 minutes of additional bench time per
position. Run it if you have time; skip it if you don't.

### Console settings

| Setting | Value |
|---------|-------|
| Imaging mode | the device's B-mode default |
| Diameter | smallest supported (e.g. 30 mm for PV .035) |
| Gain (slider) | swept 0 → 68 step 4 |
| TGC sliders | all centered (flat / disabled) |
| Acoustic Reference | OFF for the calibration captures (ring-down visible). If you also want an AR-ON rendering target, capture a matched ON subset as an auxiliary pair at the same settings. |
| Speckle reduction / compounding / smoothing | OFF |

### Capture protocol (per position)

1. Centre the spiral fixture on the catheter; verify the catheter axis
   is on the spiral centre to ±0.1 mm.
2. Capture 30 frames at each slider setting, ascending from 0 to 68 in
   steps of 4. **Use ascending sweeps only**: the PV .035 descending
   sweep (`ivus_test_0508` Sweep #2) showed a non-monotonic palette
   between slider 28 and 32 that was traced to console-settle lag and
   forced us to deprecate that sweep for the canonical fit.
3. Rotate the catheter about its long axis by ~90° between positions
   P1 → P2 → P3 → P4. The 12-wire spiral fixture stays still. This
   re-images each wire with a different SA element subset, averaging
   over the ~±2–3 dB azimuthal-lobe variability characterised in
   `calibration_delta.md` § "Wire-orientation brightness variability".
4. Log per-position metadata: water temperature, room temperature,
   catheter rotation angle (estimate to ±10° from the connector
   marking — the exact angle doesn't matter, what matters is that the
   four positions are clearly distinct).

### Outputs

For each position, run:

```bash
python3 instrument-calibration/<instrument>/extract_metadata.py --raw-dir <NEW>/raw/<position>
python3 instrument-calibration/<instrument>/annotate_wires.py --raw-dir <NEW>/raw/<position>   # interactive
python3 instrument-calibration/<instrument>/fit_alignment.py --raw-dir <NEW>/raw/<position>
python3 instrument-calibration/<instrument>/unwrap.py --raw-dir <NEW>/raw/<position>
```

Then run the per-stage extracts driven by the staged polar arrays:

| Script | Stage | YAML fields produced |
|--------|-------|----------------------|
| `extract_psf.py` | E2 axial + lateral PSF | `probe.element_radius_mm`, `probe.focal_length_mm`, `probe.pulse_duration_cycles`, `processing.lateral_psf_kernel.sigma_theta_rad`, lateral PSF lookup |
| `extract_gain_curve.py --anchor=wire` | per-wire compression curve | `processing.log_multiplier`, `processing.dynamic_range_db`, `processing.compression_lut`, slider→dB curve |

Pool the per-position outputs into a single `psf_fit.json` /
`gain_curve_wire_pooled.json` via the aggregator scripts (see
`aggregate_psf_0515.py`-style examples in
`ivus_test_0515/derived_aggregate/`).

### Acceptance criteria

Direct port of canonical E2 with onboarding-specific tightening:

- **No wire echo saturated at the median analysis slider.** If the
  inner wires saturate at slider 36 on the recommended wire material,
  re-string with a weaker scatterer (next bullet point) before
  proceeding.
- Per-wire echo SNR ≥ 30 dB at the median analysis slider, averaged
  across positions.
- Per-position `lateral_FWHM(z)` consistency: σ across the four
  positions ≤ 10 % of the mean at every wire.
- Gaussian-beam fit residual ≤ 5 % across the 12 wires.
- Apparatus radial-scale consistency: σ of the per-position
  `radial_scale` across the four positions ≤ 1 %. (PV .035 original
  apparatus failed this at 7.6 %; new apparatus passed at 0.2 %.)

### Onboarding pre-flight checklist

Before declaring O1 complete, the operator should confirm:

- [ ] The smallest available imaging diameter has been used.
- [ ] The slider sweep covers 0 → 68 step 4 (18 captures per position).
- [ ] All four catheter positions imaged.
- [ ] At least one slider in `[20, 36]` is *not* saturated at the
      innermost wire — this is the slider that anchors `log_multiplier`.
- [ ] At least one slider above 56 *is* saturated at the brightest
      wire — this is what anchors `dynamic_range_db`.

## O2 — Anechoic water gain sweep (mandatory)

**Replaces canonical E6. Produces:**
`processing.ring_down.amplitude` (with proper gain scaling),
`ring_down.extent_mm`, `ring_down.decay`, `ring_down.waveform_path`,
`ring_down.subtract_reference` policy, `processing.catheter.dead_zone_mm`,
and a slider→dB curve cross-check that anchors the slider mapping
*independently* of O1.

### Hardware

The catheter alone, suspended in the centre of the degassed water
tank, ≥ 30 mm clearance to every wall and to the surface in the
imaging plane. No targets in the field.

### Capture matrix — *upgraded from the PV .035 baseline*

The PV .035 calibration captured three `(slider, diameter)` triples
(g50/D30, g50/D60, g40/D60) and used a linear extrapolation to back
out how the ring-down peak scales with gain. The user's bench-team
feedback during this calibration was that "more gain values for the
anechoic data so the ring down scales with gain" would have been
strictly more informative — the slider→dB curve from the ring-down
peak is then a direct measurement, not an extrapolation.

The onboarding ask is therefore a **full slider sweep** (the same
0 → 68 step 4 cadence as O1), at the **smallest available imaging
diameter** (so the ring-down peak is sampled at the finest pixel
pitch and the AR-residual estimate is least quantised), and with
**matched AR-OFF / AR-ON pairs at every slider**.

| Capture | Slider | Diameter | AR | Frames | Purpose |
|---------|-------:|---------:|----|-------:|---------|
| C1.1 | 0 | smallest | OFF | 30 | reject floor |
| C1.2 | 4 | smallest | OFF | 30 | low-gain noise floor |
| … | every 4 | smallest | OFF | 30 | full ring-down vs gain curve |
| C1.18 | 68 | smallest | OFF | 30 | high-gain saturation diagnostic |
| C2.* | every 4 | smallest | ON | 30 | matched AR-residual template |
| C3.1–3 | 50 | second largest available diameter | OFF | 30 | diameter-invariance cross-check |

Total: 36 + 3 = 39 captures, ~25 minutes of bench time.

### Console settings

| Setting | Value |
|---------|-------|
| Diameter | smallest (per matrix above) |
| Gain | swept (per matrix) |
| TGC | all sliders centered |
| Acoustic Reference | OFF *and* ON (per matrix) |
| Speckle reduction / compounding | OFF |

### Capture protocol

1. Suspend the catheter centrally in the tank; wait ≥ 60 s for tank
   vibrations / bubbles to settle.
2. Run the AR-OFF slider sweep first (C1.1 → C1.18). Do not adjust
   the catheter position between captures — log the operator's hand
   on the console as the only thing that moves between captures.
3. Without moving the catheter, run the AR-ON slider sweep (C2.*).
4. Run the second-diameter cross-check (C3.1–3) at slider 50, both AR
   states.
5. If the console exposes a service-mode export of the on-board
   "Acoustic Reference" template (Volcano s5i does, p.190 of the
   manual), save it as `e6_device_acoustic_reference.bin` — useful for
   cross-validation of the simulator's AR subtraction policy.

### Outputs

```bash
python3 instrument-calibration/<instrument>/extract_metadata.py --raw-dir <NEW>/raw/o2_anechoic
python3 instrument-calibration/<instrument>/unwrap.py --raw-dir <NEW>/raw/o2_anechoic
python3 instrument-calibration/<instrument>/extract_ringdown.py --raw-dir <NEW>/raw/o2_anechoic
```

`extract_ringdown.py` outputs:

- `ringdown_template_<slider>_<D>.npy` — one raw-AR-OFF template per
  slider (the simulator's `waveform_path` consumes the canonical mid-
  gain one, see `volcano_s5i.yaml` for the chosen anchor).
- `ar_template_canonical.npy` — median of the slider-resolved AR-on /
  AR-off pair *residuals*, used when `subtract_reference: true`.
- `ringdown_amplitude_derivation.json` — fitted
  `ring_down.amplitude` × slider curve. The slope in palette/slider-
  step is the **independent** slider→dB anchor (matches O1's wire-
  anchored slider→dB within 5 % or the calibration is incoherent).
- `ringdown_fit.json` — `peak_depth_mm`, `extent_mm`, decay form
  (`exponential` vs `hanning`), and the per-slider peak palette.

### Acceptance criteria

- Per-frame A-line variation in the inner 2 mm ≤ 5 % at every slider
  (the bias is deterministic; if not, increase the tank-wall clearance
  or check for micro-bubbles).
- Validation step (AR-on minus AR-off vs the device's exported AR
  template, if available) matches within ±10 %.
- Independent slider→dB curve from O2's ring-down peak agrees with
  O1's wire-anchored slider→dB to within ±5 % across the unsaturated
  range. This is the **cross-check that the calibration is internally
  consistent** — it does not appear in the PV .035 dataset because
  the ring-down captures were too sparsely sampled to support it.
- Ring-down `peak_depth_mm` matches the catheter OD + bandwidth (PV
  .035: 2.16 mm = 1.905 mm catheter OD + 0.25 mm pulse extent).
- `extent_mm` is diameter-invariant within ±0.1 mm (cross-check from
  C3.1–3 at the second diameter).

### Onboarding pre-flight checklist

- [ ] The full slider sweep (0 → 68 step 4) AR-OFF was captured at
      the smallest diameter — *the user-flagged onboarding upgrade
      vs the PV .035 baseline*.
- [ ] Matched AR-ON sweep at every slider, same diameter.
- [ ] One slider–pair at a second diameter (diameter-invariance check).
- [ ] Independent slider→dB curve from ring-down peak matches O1's
      wire-anchored slider→dB to within ±5 %.

## O3 — Uniform attenuating phantom (mandatory)

**Replaces canonical E4 + canonical E5 + most of E8's pipeline rehearsal
on a controlled medium. Produces:**
`processing.tgc_control_points`, `processing.envelope_noise.sigma`,
`processing.envelope_noise.mean`,
`processing.scattering_resolution_mm`, `processing.scatter_integral_scale`,
the milk-material (or graphite-agar-material) `mu0` / `mu1` / `sigma`
priors, refined `probe.speed_of_sound_mm_per_us`,
`processing.reject_palette_softness`, and a slider→dB E2E cross-check
in attenuating medium.

### Hardware

Pick **one** of the following, in order of preference. All three
produce the same simulator output fields; the per-material backscatter
coefficients vary but they are calibrated against the chosen substrate
post-hoc.

1. **Commercial calibrated uniform phantom** (CIRS 040GSE-PA, CIRS 049,
   ATS 539). Manufacturer-calibrated α to ±0.05 dB/cm/MHz; sets the
   gold-standard onboarding output. Run with **the same fixture** as
   O1 / O2 so the catheter position is reproducible.
2. **Custom agar–graphite phantom** (canonical Appendix A.2 — see
   `ivus_calibration_protocol.md`). Lower-cost. α verified to
   ±0.10 dB/cm/MHz by transmission-substitution (Appendix A.6).
3. **Interim milk phantom** (`interim_milk_phantom_sop.md`) — the
   fastest "no commercial phantom on hand" option. The PV .035 ran
   this for its O3-equivalent (`ivus_test_0515/raw/e4a_milk_gain_p*/`
   for the uniform run and `e4c_milk_water_gain_p*/` for the diluted
   cross-check). α prior is loose (±0.15 dB/cm/MHz) but the analysis
   pipeline is identical — every Tier-1 field in `volcano_s5i.yaml`
   that the canonical E4 / E5 would have produced is in fact
   populated from the milk captures and passes Tier-1.

Plus, optionally, **the O1 wire-phantom fixture** submerged in the
uniform medium. This gives a free in-medium PSF / speed-of-sound
cross-check (the wire echoes appear at radii that shift by the bulk-c
ratio between water and the phantom medium). The PV .035 ran this as
`b2_w_wire_p1` (water) + `e4b_milk_undil_wire` (undiluted milk) +
`e4d_milk_water_wire` (diluted milk); see
`ivus_test_0515/derived/wire_fit_summary/wire_fit_summary.csv` for the
output shape.

### Capture matrix

The PV .035 onboarding settled on this matrix; we recommend it as the
template:

| Sub-experiment | Substrate | Captures | Purpose |
|---|---|---|---|
| **O3-A** | uniform phantom (chosen above) | 4 sliders × 3 positions = 12 captures + 1 extra slider at P1 for the noise-floor anchor (= 13 captures) | TGC schedule, α, gain_db E2E, multi-position replicate stability |
| **O3-B** | uniform phantom + O1 wire fixture submerged | 1 operator-set slider × 1 position = 1 capture | in-medium c refinement, in-medium PSF cross-check |
| **O3-C** (cross-validation, optional) | same phantom recipe at a *different* concentration / dilution | 4 sliders × 1 position = 4 captures + 1 wire capture in the diluted medium | pipeline self-consistency: α ratio across dilutions, slider→dB substrate-invariance, c agreement at the new bulk-c |

Total: 13 + 1 + (5 if O3-C run) = 14–19 captures. ~30 min of bench
time without O3-C; ~50 min with.

The **O3-C cross-validation** is the *strongly recommended* PV .035
innovation. It catches systematic α-extraction biases that a single-
substrate run cannot. The milk SOP's two-dilution structure (`<sid_und>`
+ `<sid_dil>` in `interim_milk_phantom_sop.md`) is exactly this idea.

### Console settings

| Setting | Value |
|---------|-------|
| Diameter | smallest supported (consistent with O1 / O2 — this is what makes the TGC schedule directly applicable in the simulator without re-running) |
| Gain (slider) | sweep `{20, 35, 50, 68}` (4-point: low noise floor / mid / canonical / high saturation) plus one extra at 30 frames per slider; for the cross-validation O3-C run, just the same 4 sliders |
| TGC sliders | all centered |
| Acoustic Reference | OFF for calibration consistency with O1/O2 raw-characterization captures. If an AR-ON target is required for deployment, add a matched ON subset at the same settings and process it as a secondary target. |
| Speckle reduction / compounding | OFF |

### Capture protocol

Follow canonical E4 § "Procedure" for the bulk capture, with the
following onboarding-specific deltas:

1. **Use the same fixture** as O1 / O2 so the catheter position is
   reproducible.
2. **Capture sub-experiment O3-B before draining the phantom.** It is
   ~3 minutes of bench time and gives you the in-medium c refinement
   for free. If you can only do it once, do it after the 4-gain O3-A
   sweep at P1.
3. **If running O3-C cross-validation**, drain the phantom medium
   without moving the catheter, rinse, refill with the
   second-concentration medium, and re-run O3-A's slider sweep at P1
   only. This is the structural cross-check that catches α-extraction
   pipeline bugs (see `interim_milk_phantom_sop.md` § "Pipeline
   self-consistency" for the full diagnostic table).

### Outputs

```bash
python3 instrument-calibration/<instrument>/extract_metadata.py --raw-dir <NEW>/raw/o3a
python3 instrument-calibration/<instrument>/unwrap.py --raw-dir <NEW>/raw/o3a
python3 instrument-calibration/<instrument>/extract_tgc.py --raw-dir <NEW>/raw/o3a
python3 instrument-calibration/<instrument>/extract_noise.py --raw-dir <NEW>/raw/o3a
python3 instrument-calibration/<instrument>/extract_speckle.py --raw-dir <NEW>/raw/o3a
```

`extract_tgc.py` produces `tgc_control_points`; `extract_noise.py`
produces `envelope_noise.sigma` and the noise distribution shape;
`extract_speckle.py` produces `scattering_resolution_mm` and the
per-material `mu0` / `mu1` / `sigma` priors.

If sub-experiment O3-B (wire-in-medium) was run, also run:

```bash
python3 instrument-calibration/<instrument>/fit_alignment.py --raw-dir <NEW>/raw/o3b
```

This gives the in-medium `radial_scale`, which combined with the
in-water `radial_scale` from O1 directly yields the medium's bulk-c
ratio (see `wire_fit_summary.csv` for the PV .035 numbers — undiluted
milk c ≈ 1538 m/s, diluted milk c ≈ 1515 m/s, water c ≈ 1465–1480 m/s
under the device's 1540-m/s display convention).

### Acceptance criteria

- Replicate-to-replicate variation in `I_dB(z)` ≤ 1 dB at every depth
  across the three O3-A positions (looser ±3 palette for milk per the
  interim SOP).
- α extracted from `I_dB(z)` slope agrees with the substrate spec to
  within the substrate's stated tolerance (±0.05 dB/cm/MHz for
  CIRS-class commercial; ±0.10 dB/cm/MHz for canonical agar-graphite;
  ±0.15 dB/cm/MHz for interim milk per the milk SOP).
- Gain-linearity `Δ_dB(slider=68) − Δ_dB(slider=50)` falls in
  `[8, 22] dB` (matches O1's slider→dB cross-check to within ±1.5 dB).
- Speckle autocorrelation length consistent across the four sliders
  (CV ≤ 15 %).
- If O3-B run: in-medium `radial_scale` from the wire fit agrees with
  the literature/manufacturer c value of the substrate to within
  ±15 m/s.
- If O3-C run: α ratio between concentrations matches the recipe
  ratio to within 20 % (the milk SOP's `α_undiluted / α_diluted ≈ 2`
  check); TGC curve recovery agrees across concentrations to within
  ±2 dB at every depth.

### Onboarding pre-flight checklist

- [ ] Same fixture as O1 / O2.
- [ ] Same imaging diameter as O1 / O2.
- [ ] Wire-in-medium capture (O3-B) collected.
- [ ] Substrate concentration / dilution cross-check (O3-C) collected
      — strongly recommended; mandatory if interim milk is the substrate.
- [ ] α extracted matches the substrate spec.

## Bonus paths (run only if applicable)

These are not required for onboarding but should be run if the
console / scope makes them cheap:

| Bonus | When to run | Adds |
|---|---|---|
| **E1 (RF impulse response)** | Console exposes service-mode RF tap | populates `probe.impulse_response_path`, validates `pulse_duration_cycles` from FFT BW, validates `sim.sampling_freq_mhz` |
| **E3 (slice-thickness sweep)** | Sim is being upgraded to ≥ 2.5D | populates `probe.elevational_height_mm`, sets the elevational PSF profile |
| **E7 standalone with RF injection** | Console exposes RF tap *and* you have a programmable RF generator | populates the full 256-entry `compression_lut` |
| **E8 (tissue / material fit)** | Separate clinical / ex-vivo session, NOT part of onboarding | populates `materials[].*` for vessel_wall, calcium, fibrous, lipid_pool, extravascular |
| **E9 (timing / PRF)** | Required only for the deferred motion / acquisition layer | populates the per-procedure motion config — not in the per-frame raysim YAML |

## End-to-end onboarding bench session

Total bench time, no bonus paths: **~2.5 hours** of console time, plus
phantom prep (~30 min for commercial; ~5 h for canonical agar-graphite
including 4 h cure time; ~1 h for interim milk).

Suggested order (this is what the PV .035 effectively followed across
its two bench sessions):

1. **O2 first** (anechoic water). Fastest, no targets to align,
   warms up the operator and verifies the console is at a sane
   reference operating point.
2. **O1** (wire phantom in water). Re-uses the O2 tank. The wire
   fixture is loaded onto the same catheter mount.
3. **O3-A** (uniform phantom). Drain the water tank, mount the
   phantom in the same fixture; do not move the catheter.
4. **O3-B** (wire-in-phantom). The wire fixture is re-introduced into
   the still-mounted phantom; ~3 minutes additional bench time.
5. **O3-C** (cross-validation; optional but strongly recommended for
   interim-milk runs). Drain, rinse, refill with the second
   concentration, re-run O3-A at P1 only and O3-B once.

A session that runs all of O1 / O2 / O3-A / O3-B / O3-C in one bench
day produces a fully populated YAML by end of day plus the bench
captures needed to populate `calibration_delta.md` § "Recommended
follow-up experiments" with the appropriate next-tier requests
(typically: re-shoot at a deliberately chosen second orientation if O1
detected azimuthal lobing > 5 dB peak-to-peak, or run E8 if the
device is destined for in-vivo deployment).

## Mapping back to the canonical protocol's `parameter_sheet.csv`

For traceability, here is the field-by-field map from the canonical
`parameter_sheet.csv` to the onboarding experiment that now produces it.

| YAML field | Canonical source | Onboarding source |
|---|---|---|
| `probe.frequency_mhz` | Manual | Manual |
| `probe.pulse_duration_cycles` | E1 + E2 axial PSF | **O1** axial PSF |
| `probe.element_radius_mm` | E2 Gaussian-beam fit | **O1** |
| `probe.focal_length_mm` | E2 depth-of-minimum-lateral-FWHM | **O1** |
| `probe.elevational_height_mm` | E3 | Estimated 1.5 mm in YAML; E3 still required before treating as calibrated |
| `probe.speed_of_sound_mm_per_us` | E4 (TOF refinement) | **O3-B** wire-in-medium |
| `probe.impulse_response_path` | E1 | *Bonus path* (E1 if RF available) |
| `processing.tgc_control_points` | E4 | **O3-A** |
| `processing.log_multiplier` | E7 | **O1** wire-anchored gain curve |
| `processing.dynamic_range_db` | E7 | **O1** |
| `processing.compression_lut` | E7 (RF injection) | **O1** sigmoidal fit (full LUT requires E7 bonus path) |
| `processing.gain_db` | E7 + sim-in-loop bisection | **O1** wire-anchored + cross-checked against **O2** ring-down peak |
| `processing.lateral_psf_kernel.sigma_theta_rad` | E2 | **O1** |
| `processing.envelope_noise.sigma` | E5 + E7 conversion | **O3-A** at slider 68 in the chosen substrate |
| `processing.envelope_noise.mean` | (not in canonical) | **O3-A** noise-floor anchor |
| `processing.reject_palette` | E7 | **O1** |
| `processing.reject_palette_softness` | (not in canonical) | **O3-A** multi-slider noise floor |
| `processing.saturation_palette` | E7 | **O1** |
| `processing.noise.sigma` | E5 + E7 | currently disabled in favour of `envelope_noise.sigma` (PV .035 paradigm shift, see YAML comment block) |
| `processing.noise.type` | E5 KS test | **O3-A** speckle histogram |
| `processing.scattering_resolution_mm` | E5 | **O3-A** speckle autocorrelation |
| `processing.scatter_integral_scale` | E5 | **O3-A** sim-in-loop |
| `processing.catheter.dead_zone_mm` | (not in canonical) | **O2** dead-zone observation |
| `processing.ring_down.amplitude` | E6 + E7 conversion | **O2** with full slider sweep |
| `processing.ring_down.extent_mm` | E6 | **O2** |
| `processing.ring_down.decay` | E6 | **O2** |
| `processing.ring_down.waveform_path` | E6 | **O2** raw template |
| `processing.ring_down.subtract_reference` | manual | **O2** AR-on/off cross-check |
| `materials[].impedance_mrayl` | E8 | *Out of onboarding scope* |
| `materials[].speed_of_sound_m_per_s` | E8 | *Out of onboarding scope* (substrate c refined by O3-B is logged on the substrate row, not on the in-body material rows) |
| `materials[].attenuation_db_per_cm_mhz` | E4 + E8 | **O3-A** for the substrate; *Out of scope* for in-body materials |
| `materials[].mu0` / `mu1` / `sigma` | E5 + E8 | **O3-A** for the substrate; *Out of scope* for in-body materials |
| `materials[].specularity` | E8 | *Out of onboarding scope* |

## Closing — when to upgrade the onboarding

Re-open this protocol when any of the following becomes true:

1. **The sim adds elevational rendering.** Promote E3 from a bonus
   path to a required experiment in O1's capture matrix (or as a
   standalone O4).
2. **The first AR-residual rendering bug is reported in clinical
   deployment.** The current `ar_residual_waveform_path` is documented
   in the YAML but not wired through `RingDownConfig`; that wiring +
   the corresponding O2 capture of slider-resolved AR-residual
   templates becomes a Tier-1-gating output.
3. **A second probe / console pair fails O1's apparatus radial-scale
   acceptance criterion.** This is the canary for a per-apparatus
   correction being needed in the YAML (not yet a YAML field, but
   foreshadowed by the PV .035 7.6 % original-apparatus offset). At
   that point, promote `apparatus_radial_scale` from a derived
   alignment artefact to a first-class YAML field.
4. **Synthetic-aperture beamforming is enabled in raysim.** The
   `probe.synthetic_aperture: true` path will change the per-azimuth
   PSF interpretation; O1's per-position replicate count likely
   needs to grow from 4 to 8, and the AR template in O2 may become
   element-subset-dependent.

Until then, this is the *necessary and sufficient* onboarding protocol
for any probe destined for the per-frame raysim simulator.
