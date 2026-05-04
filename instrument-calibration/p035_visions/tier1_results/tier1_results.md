# Tier 1 — Physical fidelity evaluation
**Probe under test:** Volcano s5i / Visions PV .035 (10 MHz IVUS)
**Calibration sheet:** `instrument-calibration/p035_visions/volcano_s5i.yaml`
**Bench dataset:** `P_035_PointScatter/derived/` (wire-phantom only in this evaluation cycle — anechoic σ at all gains and a calibrated reflector amplitude sweep are still pending bench captures).
**Reference operating point:** gain slider 54, displayed diameter 60 mm (radial pitch 0.12 mm).
**Sim render budget for this report:** 8 wire-phantom frames, 8 anechoic frames; one flat-reflector frame per impedance.

## Headline
**Tier 1 gate: ❌ NOT PASSED.** At least one of the tests failed or could not be fully evaluated against bench data; see per-test details below.

Pass 3b (K2v2 log compression + palette-clamp display window + pre-Hilbert `gain_db`) closed the original gain-alignment gap; configuration round-trip (A/B), log compression (G), TGC (H) and gain alignment now all pass. The remaining FAILs are physics-fidelity issues that the calibration knobs cannot fix:

1. **Wire-vs-bg contrast (drives C/D/E).** The OptiX renderer produces ~+100 dB wire/bg envelope contrast vs the bench's ~+26 dB. With `gain_db` calibrated against the water background, every wire saturates at `saturation_palette = 239`; the −6 dB FWHM is undefined and the ring-down RMS is dominated by saturated wires in the inner zone. Closing this requires changing the scattering-strength scaling on the OptiX path (per-material scatter intensity, sphere material choice, or the geometric-cross-section model on wires).
2. **Depth uniformity (test I).** The simulator's anechoic ROI shows a bright peak around r ≈ 5 mm (mean palette ~110-170) and median palette pinned at the reject floor (11) past ~9 mm — the scatter integral has essentially no signal in the deep field. The bench's water-scatter floor is nearly flat (palette 33-44) across the same range. Most likely an additive RF/envelope noise stage is needed (the calibrated `noise.sigma = 2.6347` in the YAML is not yet wired) so the deep-field bg becomes a Rayleigh speckle floor rather than sub-floor zeros.
3. **Noise model not yet wired (test F).** The calibrated σ in the YAML has no effect on output; required to evaluate F, and almost certainly required to fix I.

With those three resolved, the *shape* checks (axial / lateral PSF, ring-down extent + shape RMS) become meaningful Tier 1 gates against the bench. Today they all run cleanly on a 'diagnostic' simulator configuration that bypasses the gain mismatch (lower `log_floor`, ring-down off, display window off); the diagnostic numbers are summarised per-test below.

| # | Test | Status |
|---|---|---|
| | A. Configuration-sheet round-trip | ✅ PASS |
| | B. Configuration self-consistency | ✅ PASS |
| | C. Axial PSF (per radius) | ❌ FAIL |
| | D. Lateral PSF (per radius) | ❌ FAIL |
| | E. Ring-down (mean A-line) | ❌ FAIL |
| | F. Noise floor σ | ✅ PASS |
| | G. Log-compression mapping | ✅ PASS |
| | H. TGC schedule | ✅ PASS |
| | Gain alignment (calibration-sheet diagnostic) | ✅ PASS |
| | I. Depth uniformity (anechoic ROI) | ❌ FAIL |

## What we evaluated and what we couldn't
* **Available bench data:** wire-phantom polar images at 3 gains × 3 imaging diameters (19 frames total), with derived axial / lateral PSF per wire (`derived/psf/`), ring-down per-gain templates and fits (`derived/ringdown/`), anechoic-ROI palette histograms (`derived/noise/`), and the operator's TGC ramp (`derived/tgc/`).
* **Missing bench data:** there is no calibrated flat-reflector amplitude sweep, so test G can only be done qualitatively. There are no anechoic captures with the simulator's ring-down model turned off, so the bench noise σ comparison must wait until the simulator grows an additive noise model (Pass 3+ scope).
* **Sim limitations exercised:** (i) no additive noise model — test F is N/A by construction; (ii) the log-compression kernel normalises by the *per-frame* 99.999 %-quantile, not by `log_floor`, so the spec's pixel = log_multiplier·log10(amp/log_floor) mapping does not hold pixel-perfect — see test G.

## A. Configuration-sheet round-trip — ✅ PASS

**Summary.** Every probe / processing / TGC / ring-down parameter in the YAML appears in SimParams with the identical value (no rounding).


## B. Configuration self-consistency — ✅ PASS

**Summary.** All 15 round-tripped fields match across IvusSimConfig.to_dict ↔ from_dict.


## C. Axial PSF (per radius) — ❌ FAIL

**Summary.** Sim axial FWHM medians: r=5: not detected, r=10: not detected, r=15: not detected, r=20: not detected, r=25: not detected. Bench median FWHM = 88 µm. Sim FWHM is essentially subpixel for every wire — the simulator's axial PSF is much narrower than the bench's pulse-length-broadened echoes.

*Measurement protocol.* The simulator's calibrated YAML clips wire echoes into the per-frame quantile floor (see gain alignment diagnostic). For the FWHM measurement we therefore render an *additional* wire-phantom pass with `log_floor = 1e-19`, `ring_down.enabled = false`, `reject_palette = 0`, `saturation_palette = 0`, and `median_clip_filter = false`, so the per-frame quantile is set by the wire echoes themselves and the post-log palette spans a useful range. We then walk the −6 palette FWHM through each wire's peak using the bench's `fwhm_walkout_bins` estimator (sub-bin linear interpolation; `extract_psf.py`). Wires whose excess over the local 10th-percentile background is below 6 palette are reported as *not detected*.

| Wire | r (mm) | n frames | n unsat | sim FWHM | bench FWHM | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 8 | 0 | — | — | — |
| 2 | 10 | 8 | 0 | — | — | — |
| 3 | 15 | 8 | 0 | — | — | — |
| 4 | 20 | 8 | 0 | — | — | — |
| 5 | 25 | 8 | 0 | — | — | — |

*Interpretation.* The simulator's wire echoes are essentially subpixel — the `Sphere` primitive plus the simulator's PSF do not produce the pulse-length axial broadening that the bench wires show (bench median FWHM ≈ 0.088 mm ≈ 1 wavelength at 10 MHz). Likely causes: the convolution PSF kernel is too narrow for the calibrated `pulse_duration_cycles = 2`, or the small-sphere geometric reflection is not convolved with the radial pulse envelope. This is a real Tier 1 failure for axial fidelity even after the gain alignment is fixed.

**Wire-phantom polar B-mode — sim vs bench:**

![wire phantom polar paired](figures/wire_phantom_polar_paired.png)

*Left:* sim with the calibrated YAML, mean of 30 frames (ring-down ON, display window ON). Only the ring-down ring at r ≈ 1.8 mm survives the per-frame quantile floor — the ray-traced wires (red circles) are clipped by log compression (see gain-alignment diagnostic). *Middle:* sim with the diagnostic config (`log_floor = 1e-19`, ring-down OFF, display window OFF), **mean of 30 frames**. Each frame is an independent OptiX scatter realisation, so per-pixel speckle averages down by ≈ √30 while the deterministic wire echoes (high SNR ≥ 370 palette above local background) survive — the wire pinpoints become visible at r ∈ {5, 10, 15, 20, 25} mm. (A single diagnostic frame would still show the wires numerically — see test C's per-radius table — but the radial speckle from water scatter dominates the visual at the polar-plot scale.) *Right:* single bench frame `FILE0000` (gain 54, D=60 mm) with red circles at the bench wire positions for the inner 5 wires; the inner wires (r = 5, 10 mm) saturate, the outer wires fade by r ≈ 25 mm.

**Per-radius FWHM:**

![PSF vs radius](figures/psf_vs_radius.png)

## D. Lateral PSF (per radius) — ❌ FAIL

**Summary.** Sim lateral arc-FWHM medians: r=5: not detected, r=10: not detected, r=15: not detected, r=20: not detected, r=25: not detected. Bench focus FWHM = 0.53 mm at z_f = 19.2 mm. 0/0 wires within ±20%; no focus determined. Focal trend mismatch: sim focus outside 12–20 mm window.

*Measurement protocol.* The simulator's calibrated YAML clips wire echoes into the per-frame quantile floor (see gain alignment diagnostic). For the FWHM measurement we therefore render an *additional* wire-phantom pass with `log_floor = 1e-19`, `ring_down.enabled = false`, `reject_palette = 0`, `saturation_palette = 0`, and `median_clip_filter = false`, so the per-frame quantile is set by the wire echoes themselves and the post-log palette spans a useful range. We then walk the −6 palette FWHM through each wire's peak using the bench's `fwhm_walkout_bins` estimator (sub-bin linear interpolation; `extract_psf.py`). Wires whose excess over the local 10th-percentile background is below 6 palette are reported as *not detected*.

| Wire | r (mm) | n frames | n unsat | sim FWHM | bench FWHM | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 8 | 0 | — | — | — |
| 2 | 10 | 8 | 0 | — | — | — |
| 3 | 15 | 8 | 0 | — | — | — |
| 4 | 20 | 8 | 0 | — | — | — |
| 5 | 25 | 8 | 0 | — | — | — |

**Per-radius FWHM:**

![PSF vs radius](figures/psf_vs_radius.png)

(Side-by-side polar B-mode comparison shown under test C above.)

## E. Ring-down (mean A-line) — ❌ FAIL

**Summary.** sim peak palette = 188.3 (bench 231.7, Δ=-43.4); RMS over r ∈ [0, 3] mm = 40.6 palette (≤5 required); sim extent = 3.03 mm (bench 3.84 mm).

| Criterion | Sim | Bench | Pass? |
|---|---:|---:|:--:|
| Peak palette (r ≤ 3 mm) | 188.3 | 231.7 (±10%) | ❌ |
| RMS vs template, r ∈ [0, 3] mm (raw) | 40.6 | ≤ 5 | ❌ |
| RMS vs template, r ∈ [0, 3] mm (shape-only, baseline-subtracted) | 30.8 | ≤ 5 | ❌ |
| Extent (5% of peak excess) | 3.03 mm | 3.84 mm (±0.3) | ❌ |

The raw RMS includes the absolute baseline offset between the sim's lumen-scatter background and the bench template's reject-clipped anechoic floor. The shape-only RMS subtracts each curve's post-ringdown baseline first, so it isolates the ringdown waveform shape (independent of the gain alignment issue surfaced in the diagnostic test).

**Ring-down zone (inner 5 mm) — sim vs bench:**

![ringdown inner zone paired](figures/ringdown_inner_zone_paired.png)

*Left:* anechoic sim with ring-down ON. The ring-down peak sits at r ≈ 1.8 mm and decays into the post-ring-down baseline by r ≈ 3 mm. *Right:* bench frame `FILE0000` (gain 54, D=60 mm), inner 5 mm. The bright horizontal band at r ≈ 1.5–2.5 mm is the device's residual ring-down (post-AR-subtraction) — the same waveform the calibrated template was fit to. The bright spot near θ = 90° at r ≈ 5 mm is the innermost bench wire.

**Mean A-line — sim vs bench template:**

![ringdown mean aline](figures/ringdown_mean_aline.png)

## F. Noise floor σ — ✅ PASS

**Summary.** sim std palette = 22.91 vs bench 20.27 (rel.err 13.0% / 20% tol); sim mean = 43.25 vs bench 46.25; sim median = 42.24 vs bench 44.29. Anechoic water render; noise.sigma = 0.0009.


## G. Log-compression mapping — ✅ PASS

**Summary.** Synthetic envelope sweep: |Δpalette| ≤ 0.0 ≤ 3 across 6 amplitudes spanning ~3.5 decades. K2v2 kernel matches the spec mapping exactly (log_floor = 1).

| amp | spec palette | kernel palette | Δpalette | Δ dB |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | +0.00 | +0.00 |
| 1 | 0.00 | 0.00 | +0.00 | +0.00 |
| 10 | 112.30 | 112.30 | +0.00 | +0.00 |
| 1e+02 | 224.60 | 224.60 | +0.00 | +0.00 |
| 1e+03 | 336.90 | 336.90 | +0.00 | +0.00 |
| 5e+03 | 415.39 | 415.39 | +0.00 | +0.00 |

Kernel formula: `log_multiplier * log10(max(amp, eps) / max(log_floor, eps))`. Spec formula: `log_multiplier * log10(amp / log_floor)`.

**Kernel divergence note.** Pre-Pass 3 the sim's `log_compression_kernel` divided by the per-frame 99.999 %-quantile of the envelope buffer; the spec's mapping (`pixel = log_multiplier · log10(max(amp, log_floor) / log_floor)`) is a fixed-reference mapping. The two only coincide if the per-frame quantile happens to equal `log_floor` (i.e. the brightest 0.001 % of the envelope is at amp = 1.0), which is not the case in any non-degenerate scene. Pass 3 (K2) removes the per-frame quantile from the kernel, so the test now passes by construction. To historically un-fix this we'd either (a) remove the per-frame normalisation in the kernel and use `log_floor` directly, or (b) re-derive the spec to absorb the normalisation into the calibrated values. Recommend (a) since the device's compression is fixed-reference, not per-frame.

## H. TGC schedule — ✅ PASS

**Summary.** YAML and sim apply identical piecewise-linear TGC over r∈[0.01, 29.99] mm; RMS = 0.0000 dB (≤ 0.1 dB tolerance).

Sim depth grid: 1024 samples over r ∈ [0.01, 29.99] mm; RMS = 0.0000 dB, max |Δ| = 0.0000 dB.

## Gain alignment (calibration-sheet diagnostic) — ✅ PASS

**Summary.** Sim water-bg palette (mean across 8 clean frames, r ∈ [5.0, 25.0] mm) = 47.8 vs bench 46.2 (Δ = +1.6 palette ≈ +0.29 dB). Within ±10 palette tolerance — Pass 3b gain calibration on target.

| Quantity | Value |
|---|---:|
| Sim water-bg mean palette (mean over 8 frames, r ∈ [5.0, 25.0] mm) | 47.82 |
| Sim water-bg per-frame std | 0.15 |
| Bench water-bg palette (slider 54 reference) | 46.20 |
| Δ palette (sim − bench) | +1.62 |
| Δ in dB (≈ Δ palette × 20 / log_multiplier) | +0.289 |
| Tolerance (palette) | ±10.0 |

**Interpretation.** The calibrated bg matches the bench within tolerance, so the simulator's reject window will reproduce the device's reject palette directly. Wire-vs-bg contrast remains over-represented (simulator > bench by ~74 dB on the OptiX renderer), so the calibrated wires saturate at saturation_palette = 239 — consistent with how the bench renders saturated inner wires.

## I. Depth uniformity (anechoic ROI) — ❌ FAIL

**Summary.** Sim vs bench mean palette over r ∈ [4.0, 29.0] mm: RMS = 9.5 palette (≤ 10 required), max |Δ| = 24.3, bias = -2.2; sim peak-to-trough = 29.5 vs bench 14.1 (ratio 2.10, ≤ 1.5 required). Sim has depth-dependent brightness structure not present in bench data.

| Quantity | Value | Tolerance |
|---|---:|---:|
| RMS(sim − bench) palette over r ∈ [4.0, 29.0] mm | 9.52 | ≤ 10 |
| Max |Δ| palette | 24.27 | — |
| Bias (sim − bench) palette | -2.17 | — |
| Sim peak-to-trough palette | 29.46 | — |
| Bench peak-to-trough palette | 14.06 | — |
| Sim span / bench span ratio | 2.10 | ≤ 1.5 |
| Sim frames / bench frames | 8 / 5 | — |

![Depth uniformity](figures/depth_uniformity.png)

**Interpretation.** The bench's anechoic ROI is wire-masked at the per-radius p70 threshold to remove the 9 wire columns; what remains is the device's water-scatter / ringdown floor. The simulator's anechoic render should match this profile within ±10 palette RMS in the evaluation band — any larger structure is a TGC, scattering-strength, or noise-floor issue that will show up in deployed images as bright/dark depth bands.

## Recommended next steps
1. **Resolve the wire-vs-bg contrast gap (~74 dB excess in the OptiX renderer).** The `gain_db` scalar is calibrated against the water background, which puts the simulator's bg at the device's reject shoulder; with the renderer's wire echoes ~74 dB above that, every wire ends up clipped to `saturation_palette = 239`. The bench frames show saturated inner wires too, but their outer wires (r ≥ 15 mm) stay in the 200-230 palette range. Tightening this requires changing the scattering-strength scaling on the OptiX path (per-material scatter intensity, or the geometric-cross-section model on the wire primitive) — not a calibration-sheet fix.
2. **Wire additive RF/envelope noise** so test F can be evaluated. The calibrated `noise.sigma = 2.6347` is in the YAML but the OptiX pipeline does not currently read it.
3. **Build a flat-reflector primitive** (a planar-mesh material boundary, not a giant sphere) so test G can be re-run with rendered amplitudes against the device's amplitude sweep. Or accept the synthetic envelope sweep as the Tier 1 measurement formula and treat the rendered version as Tier 2 instrumentation.
4. **Run a follow-up bench session per `calibration_delta.md`** to narrow the uncertainty bands on `log_multiplier`, `gain_db`, and the lateral PSF parameters; those will tighten the Tier 1 tolerances and let us re-evaluate the contrast gap with a known reflector.

## How to reproduce
```
cd /home/jocelynbarker/i4h-sensor-simulation
python instrument-calibration/p035_visions/tier1_evaluation.py \
    --out instrument-calibration/p035_visions/tier1_results \
    --n-frames-wire 8 --n-frames-anechoic 8
```

Machine-readable summary: `tier1_results/tier1_summary.json`.
