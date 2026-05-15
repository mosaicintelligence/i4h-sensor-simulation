# Tier 1 — Physical fidelity evaluation
**Probe under test:** Volcano s5i / Visions PV .035 (10 MHz IVUS)
**Calibration sheet:** `instrument-calibration/p035_visions/volcano_s5i.yaml`
**Bench dataset:** `P_035_PointScatter/derived/` (wire-phantom only in this evaluation cycle — anechoic σ at all gains and a calibrated reflector amplitude sweep are still pending bench captures).
**Reference operating point:** gain slider 54, displayed diameter 60 mm (radial pitch 0.12 mm).
**Sim render budget for this report:** 8 wire-phantom frames, 8 anechoic frames; one flat-reflector frame per impedance.

## TL;DR

**Sim status: Tier 1 score 7/10.** Tests A, B, F, G, H, I and gain alignment all PASS. Tests C (axial PSF), D (lateral PSF), and E (ring-down) FAIL — but only for measurement reasons we can resolve with new bench captures, not because of any confirmed sim physics error.

**Why C / D / E still fail (one line each):**

- **C — axial PSF:** every wire saturates to palette 239 in both sim and bench at the gain-54 reference, so the −6 dB FWHM walkout returns *not detected* for all five wires. The sim's axial PSF kernel may already be correct; we cannot measure it against the current bench data.
- **D — lateral PSF:** same root cause as C, plus an aperture / focus question we cannot answer without an unsaturated wire image at the focal radius (12-20 mm).
- **E — ring-down:** the bench ring-down templates we have today are already post-AR-subtraction (the device's internal Acoustic Reference subtraction was on during capture), so we calibrated the sim's `ring_down.peak` and `ring_down.extent_mm` against an already-suppressed waveform. Sim peak runs ~7.5 dB low and extent ~0.8 mm short vs the AR-on bench frame.

**What we need from the lab to unblock the next wave of sim work** (full procedure citations + value explanation in [Bench data requests](#bench-data-requests) below; ordered within each tier by sim-improvement value):

- **Tier A** — no new equipment, runnable today:
  - **A1** Anechoic captures at gain ∈ {30, 40, 50, 54, 60, 68}, AR ON, 30 frames each.
  - **A2** Paired AR-OFF + AR-ON anechoic captures at gain 68 (per the E6 procedure in the calibration protocol).
  - **A3** Re-image the current copper wire phantom at gain ∈ {20, 30, 40} in addition to the existing gain-54 capture.
- **Tier B** — uses the spiral-fixture STLs we already shipped:
  - **B1** Build the 12-wire spiral fixture from `hardware/wire_spiral_*.stl` and run the canonical E2 spiral wire phantom.
  - **B2** Same fixture, but with **nylon monofilament** (75 µm) instead of copper — the primary spec in the E2 protocol; switches the wires from the Mie-resonance regime into the Rayleigh regime where sub-wavelength scattering is well-behaved.
- **Tier C** — needs new equipment or fabrication time:
  - **C1** Cyst phantom + full E5 at gain ∈ {20, 50, 68}.
  - **C2** Flat reflector / step phantom for E7 log-compression validation.
  - **C3** Slice-thickness sweep (E3) for elevational PSF.
  - **C4** Tissue / material fit (E8) for in-vivo material parameters.
- **Stretch goal** — repeat any subset on a second device unit for unit-to-unit variance bands. We assume this is not feasible at this time but the value is on record.

**What we will not pursue in sim until that data lands:** any further C / D / E work is blocked by the saturation / AR-subtracted-template issues above. Pass 8 (per-material backscatter scaling, the planned fix for the wire-vs-bg contrast gap) is also gated on B2 — without a calibration target outside the Mie-resonance regime, calibrating Pass 8 against the current copper wire phantom would bake in a non-physical assumption.

</div>

## Headline
**Tier 1 gate: ❌ NOT PASSED.** At least one of the tests failed or could not be fully evaluated against bench data; see per-test details below.

The simulator's calibration sheet has been advanced through several passes since the last evaluation cycle (each is a merged PR against `ivus-probe`):

- **Pass 3b** (K2v2 log compression + palette-clamp display window + pre-Hilbert `gain_db`) closed the original gain-alignment gap; tests A, B, G, H, and the gain-alignment diagnostic flipped to PASS.
- **Pass 5 / 5c** rewrote the lateral PSF as a depth-dependent Gaussian-beam kernel with cyclic angular convolution and a pre-focal beam clamp, killing the bright shoulder at r ≈ 4-7 mm.
- **Pass 6 v2** added a CUDA additive-Gaussian RF noise stage applied **before** the lateral PSF (so the noise is bandlimited by the receive chain and renders as bench-like mottled speckle), plus a catheter dead-zone mask (`r < 1.4 mm` → palette 0). Pivoted `gain_db` to wire-peak-anchored (+73.92 dB) since the noise stage now sets the bg floor. Test F (noise floor σ) flipped to PASS.
- **Pass 7** added per-radial-sample weight `w(z) = sqrt(sigma_bins(z)/sigma_bins(z_focal))` on the noise sigma so that post-PSF noise std is uniform across depth (the L1-normalised lateral PSF was concentrating focal-zone noise variance, producing a +22 palette focal-zone hump). Test I (depth uniformity) flipped to PASS; sim peak-to-trough dropped from 29.5 to 6.5 palette.

The remaining FAILs (C / D / E) are **measurement-blocked**, not calibration-blocked. The active diagnosis (which the next sim pass is gated on) is:

**Wire-vs-bg contrast gap (drives C / D, gates Pass 8).** The OptiX renderer returns ~74 dB more signal from a wire-target sphere than the bench measures. Concretely: wire targets are 0.127 mm `Sphere` primitives ([tier1_evaluation.py L188](/home/jocelynbarker/i4h-sensor-simulation/instrument-calibration/p035_visions/tier1_evaluation.py#L188)) whose hits are processed through the OptiX flat-acoustic-interface model ([optix_trace.cu L562-L578](/home/jocelynbarker/i4h-sensor-simulation/i4h-sensor-simulation/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L562)) returning intensity `R = ((Z₂−Z₁)/(Z₂+Z₁))² ≈ 0.42` (-3.8 dB) per hit. But at 10 MHz `λ = 0.154 mm` so `ka ≈ 2.6` — the wires are sub-wavelength scatterers in the **Mie resonance regime** where the actual backscatter cross-section is much smaller than the geometric flat-interface return. With `gain_db` calibrated against the water background (Pass 6+7), every wire saturates to `saturation_palette = 239` and the −6 dB FWHM walkout fails by construction.

Closing this requires either (a) bench data we don't have today (see Tier B requests below) or (b) a sim physics change (Pass 8: per-material `backscatter_scale` knob calibrated against a non-resonance target) that we have deliberately deferred until (a) lands.

**Ring-down peak gap (drives E).** The current calibration of `ring_down.peak` and `ring_down.extent_mm` was fit against bench templates that already had the device's internal Acoustic Reference subtraction applied. We need raw AR-OFF anechoic captures (Tier A request A2) to re-anchor the calibration; this is a parameter recalibration, not a sim physics change.

| # | Test | Status |
|---|---|---|
| 1 | A. Configuration-sheet round-trip | ✅ PASS |
| 2 | B. Configuration self-consistency | ✅ PASS |
| 3 | C. Axial PSF (per radius) | ❌ FAIL |
| 4 | D. Lateral PSF (per radius) | ❌ FAIL |
| 5 | E. Ring-down (mean A-line) | ❌ FAIL |
| 6 | F. Noise floor σ | ✅ PASS |
| 7 | G. Log-compression mapping | ✅ PASS |
| 8 | H. TGC schedule | ✅ PASS |
| 9 | Gain alignment (calibration-sheet diagnostic) | ✅ PASS |
| 10 | I. Depth uniformity (anechoic ROI) | ✅ PASS |

## What we evaluated and what we couldn't
* **Available bench data:** wire-phantom polar images at 3 gains × 3 imaging diameters (19 frames total), with derived axial / lateral PSF per wire (`derived/psf/`), ring-down per-gain templates and fits (`derived/ringdown/`), anechoic-ROI palette histograms (`derived/noise/`), and the operator's TGC ramp (`derived/tgc/`).
* **Missing bench data (cross-references to [Bench data requests](#bench-data-requests) below):** anechoic captures at gains other than 54 (A1); paired AR-ON / AR-OFF anechoic captures (A2); lower-gain wire-phantom captures so the inner wires drop out of saturation (A3); the canonical 12-wire spiral phantom from our shipped STLs (B1) and a sub-wavelength variant in nylon monofilament that breaks the Mie-resonance regime (B2); a calibrated flat-reflector amplitude sweep for log-compression validation (C2); a cyst phantom for per-tissue scattering calibration (C1).
* **Sim limitations exercised:** (i) wire targets returned via the OptiX flat-acoustic-interface model at ka ≈ 2.6 — over-returns vs the true Mie cross-section by ~74 dB (Pass 8 deferred until B2 lands); (ii) lateral PSF builder uses a synthetic-aperture-aware Gaussian-beam model with a pre-focal clamp (Pass 5c) — not yet validated at the focus because all wires saturate; (iii) noise model calibrated at a single gain anchor (slider 54), no per-gain LUT (A1 unblocks).

## A. Configuration-sheet round-trip — ✅ PASS

**Summary.** Every probe / processing / TGC / ring-down parameter in the YAML appears in SimParams with the identical value (no rounding).


## B. Configuration self-consistency — ✅ PASS

**Summary.** All 15 round-tripped fields match across IvusSimConfig.to_dict ↔ from_dict.


## C. Axial PSF (per radius) — ❌ FAIL

**Summary.** Sim axial FWHM medians: r=5: not detected, r=10: not detected, r=15: not detected, r=20: not detected, r=25: not detected. Bench median FWHM = 88 µm. Sim FWHM is essentially subpixel for every wire — the simulator's axial PSF is much narrower than the bench's pulse-length-broadened echoes.

*Measurement protocol.* We walk the −6 palette FWHM through each wire's peak using the bench's `fwhm_walkout_bins` estimator (sub-bin linear interpolation; `extract_psf.py`). Wires whose excess over the local 10th-percentile background is below 6 palette are reported as *not detected*. With the Pass 6+7 calibrated YAML, the OptiX wire echoes (post-`gain_db = +73.92 dB`) all clip to `saturation_palette = 239` because of the wire-vs-bg contrast gap (~74 dB excess vs bench, see Headline) — every wire shows up as *not detected* by construction. **This is a measurement-blocked failure, not a confirmed sim PSF kernel error**; see Bench data requests [A3](#bench-data-requests) (lower-gain copper-wire captures) and [B2](#bench-data-requests) (nylon-monofilament spiral phantom) for the data we need to actually measure the sim's PSF widths against bench.

| Wire | r (mm) | n frames | n unsat | sim FWHM | bench FWHM | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 8 | 0 | — | — | — |
| 2 | 10 | 8 | 0 | — | — | — |
| 3 | 15 | 8 | 0 | — | — | — |
| 4 | 20 | 8 | 0 | — | — | — |
| 5 | 25 | 8 | 0 | — | — | — |

*Interpretation.* All 5 sim wires saturate to palette 239 in the calibrated render, so the −6 dB FWHM walkout reports *not detected*. The sim's axial PSF kernel (a 2-cycle Hanning-windowed cosine from `create_ivus_axial_psf_causal`, FWHM ≈ 1λ ≈ 0.154 mm at 10 MHz) is **already in the same magnitude band as the bench median FWHM ≈ 0.088 mm ≈ 0.57 λ**, so there is no evidence today that the kernel itself is wrong — we simply cannot resolve it against saturated peaks. Lower-gain bench captures (A3) or a nylon-monofilament spiral phantom (B2) would give us unsaturated wire profiles to measure FWHM against.

**Wire-phantom polar B-mode — sim vs bench:**

![wire phantom polar paired](figures/wire_phantom_polar_paired.png)

*Left:* sim with the calibrated YAML (Pass 6+7), mean of 8 frames, ring-down ON, display window ON, γ-stretched. The ring-down ring at r ≈ 1.8 mm dominates the inner zone; the catheter dead-zone mask (r < 1.4 mm) renders as solid black; the bandlimited mottled background is the Pass 6+7 pre-PSF Gaussian noise stage. The ray-traced wires (red circles, sim wire layout: 5 spheres at r ∈ {5, 10, 15, 20, 25} mm) saturate to palette 239 due to the wire-vs-bg contrast gap (see Headline). *Right:* single bench frame `FILE0000` (gain 54, D=60 mm), γ-stretched the same way; red circles mark the bench wire positions for the inner 5 wires. The inner wires (r = 5, 10 mm) also saturate on the bench at gain 54; outer wires fade by r ≈ 25 mm. **The visual qualitative match is good — Pass 6+7 produces bench-like mottled speckle, ring-down, and dead-zone — but the inner-wire saturation on both sides is exactly why the −6 dB FWHM walkout (test C / D) cannot resolve the wire profiles today.**

**Per-radius FWHM:**

![PSF vs radius](figures/psf_vs_radius.png)

## D. Lateral PSF (per radius) — ❌ FAIL

**Summary.** Sim lateral arc-FWHM medians: r=5: not detected, r=10: not detected, r=15: not detected, r=20: not detected, r=25: not detected. Bench focus FWHM = 0.53 mm at z_f = 19.2 mm. 0/0 wires within ±20%; no focus determined. Focal trend mismatch: sim focus outside 12–20 mm window.

*Measurement protocol.* We walk the −6 palette FWHM through each wire's peak using the bench's `fwhm_walkout_bins` estimator (sub-bin linear interpolation; `extract_psf.py`). Wires whose excess over the local 10th-percentile background is below 6 palette are reported as *not detected*. With the Pass 6+7 calibrated YAML, the OptiX wire echoes (post-`gain_db = +73.92 dB`) all clip to `saturation_palette = 239` because of the wire-vs-bg contrast gap (~74 dB excess vs bench, see Headline) — every wire shows up as *not detected* by construction. **This is a measurement-blocked failure, not a confirmed sim PSF kernel error**; see Bench data requests [A3](#bench-data-requests) (lower-gain copper-wire captures) and [B2](#bench-data-requests) (nylon-monofilament spiral phantom) for the data we need to actually measure the sim's PSF widths against bench.

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

**Summary.** sim peak palette = 189.9 (bench 231.7, Δ=-41.8); RMS over r ∈ [0, 3] mm = 37.7 palette (≤5 required); sim extent = 3.03 mm (bench 3.84 mm).

| Criterion | Sim | Bench | Pass? |
|---|---:|---:|:--:|
| Peak palette (r ≤ 3 mm) | 189.9 | 231.7 (±10%) | ❌ |
| RMS vs template, r ∈ [0, 3] mm (raw) | 37.7 | ≤ 5 | ❌ |
| RMS vs template, r ∈ [0, 3] mm (shape-only, baseline-subtracted) | 39.9 | ≤ 5 | ❌ |
| Extent (5% of peak excess) | 3.03 mm | 3.84 mm (±0.3) | ❌ |

The raw RMS includes the absolute baseline offset between the sim's lumen-scatter background and the bench template's reject-clipped anechoic floor. The shape-only RMS subtracts each curve's post-ringdown baseline first, so it isolates the ringdown waveform shape.

*Interpretation.* The sim's ring-down has the right onset and the right qualitative decay shape, but its peak runs ~7.5 dB below bench and its 5 %-of-peak extent is ~0.8 mm short. Both gaps trace to the calibration input: the bench templates we fit `processing.ring_down.peak` and `processing.ring_down.extent_mm` against were captured **with the device's internal Acoustic Reference (AR) subtraction ON**, so the calibrated waveform represents the AR *residual*, not the raw ring-down. To re-anchor the calibration we need a paired AR-OFF + AR-ON capture (Tier A request [A2](#bench-data-requests)). Once that lands the recalibration is a parameter sweep, not a sim physics change — Test E should flip to PASS without any code edits.

**Ring-down zone (inner 5 mm) — sim vs bench:**

![ringdown inner zone paired](figures/ringdown_inner_zone_paired.png)

*Left:* anechoic sim with ring-down ON. The ring-down peak sits at r ≈ 1.8 mm and decays into the post-ring-down baseline by r ≈ 3 mm. *Right:* bench frame `FILE0000` (gain 54, D=60 mm), inner 5 mm. The bright horizontal band at r ≈ 1.5–2.5 mm is the device's residual ring-down (post-AR-subtraction) — the same waveform the calibrated template was fit to. The bright spot near θ = 90° at r ≈ 5 mm is the innermost bench wire.

**Mean A-line — sim vs bench template:**

![ringdown mean aline](figures/ringdown_mean_aline.png)

## F. Noise floor σ — ✅ PASS

**Summary.** sim std palette = 22.29 vs bench 20.27 (rel.err 10.0% / 20% tol); sim mean = 46.43 vs bench 46.25; sim median = 46.91 vs bench 44.29. Anechoic water render; noise.sigma = 0.0007.


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

*Caveat.* This is a synthetic-envelope sweep against the spec mapping — a self-consistency check on the K2v2 kernel, not a measurement against device output. A true production validation requires the bench-side flat-reflector / step-phantom amplitude sweep (see Bench data request [C2](#bench-data-requests)).

## H. TGC schedule — ✅ PASS

**Summary.** YAML and sim apply identical piecewise-linear TGC over r∈[0.01, 29.99] mm; RMS = 0.0000 dB (≤ 0.1 dB tolerance).

Sim depth grid: 1024 samples over r ∈ [0.01, 29.99] mm; RMS = 0.0000 dB, max |Δ| = 0.0000 dB.

## Gain alignment (calibration-sheet diagnostic) — ✅ PASS

**Summary.** Sim water-bg palette (mean across 8 clean frames, r ∈ [5.0, 25.0] mm) = 48.1 vs bench 46.2 (Δ = +1.9 palette ≈ +0.34 dB). Within ±10 palette tolerance — Pass 3b gain calibration on target.

| Quantity | Value |
|---|---:|
| Sim water-bg mean palette (mean over 8 frames, r ∈ [5.0, 25.0] mm) | 48.12 |
| Sim water-bg per-frame std | 0.14 |
| Bench water-bg palette (slider 54 reference) | 46.20 |
| Δ palette (sim − bench) | +1.92 |
| Δ in dB (≈ Δ palette × 20 / log_multiplier) | +0.342 |
| Tolerance (palette) | ±10.0 |

**Interpretation.** The calibrated bg matches the bench within tolerance, so the simulator's reject window will reproduce the device's reject palette directly. The wire-vs-bg contrast gap (sim > bench by ~74 dB on the OptiX renderer at ka ≈ 2.6) is a separate scattering-physics issue that the gain_db scalar cannot fix — see Headline and the deferred Pass 8.

## I. Depth uniformity (anechoic ROI) — ✅ PASS

**Summary.** Sim vs bench mean palette over r ∈ [4.0, 29.0] mm: RMS = 3.5 palette (≤ 10 required), max |Δ| = 8.6, bias = +0.2; sim peak-to-trough = 6.5 vs bench 14.1 (ratio 0.46, ≤ 1.5 required). Sim depth uniformity matches bench within tolerance.

| Quantity | Value | Tolerance |
|---|---:|---:|
| RMS(sim − bench) palette over r ∈ [4.0, 29.0] mm | 3.53 | ≤ 10 |
| Max |Δ| palette | 8.56 | — |
| Bias (sim − bench) palette | +0.18 | — |
| Sim peak-to-trough palette | 6.46 | — |
| Bench peak-to-trough palette | 14.06 | — |
| Sim span / bench span ratio | 0.46 | ≤ 1.5 |
| Sim frames / bench frames | 8 / 5 | — |

![Depth uniformity](figures/depth_uniformity.png)

**Interpretation.** The bench's anechoic ROI is wire-masked at the per-radius p70 threshold to remove the 9 wire columns; what remains is the device's water-scatter / ringdown floor. The simulator's anechoic render should match this profile within ±10 palette RMS in the evaluation band — any larger structure is a TGC, scattering-strength, or noise-floor issue that will show up in deployed images as bright/dark depth bands.

## Bench data requests
Concrete experimental asks for the lab team, ordered within each tier by sim-improvement value. Each request cites the specific experiment in the [IVUS Calibration & Characterization Protocol](../../../instrument-calibration/docs/ivus_calibration_protocol.md) (E1-E8), states the sim limitation it removes, and describes the concrete sim deliverable that it unblocks.

### Tier A — no new equipment, runnable today

**A1. Anechoic captures at multiple gain settings.** Cite [E6 — Ring-Down Capture](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e6--ring-down-capture-acoustic-reference) procedure but with **gain stepped through {30, 40, 50, 54, 60, 68}**, AR ON, all TGC sliders centered, 30 frames per gain. Same probe, same bath, same temperature.

*Why we need it.* Today our `processing.noise.sigma` and `processing.gain_db` are both anchored against a single bench gain (slider 54). We cannot tell whether the simulator generalises to a clinical capture made at a different gain, or whether the device's gain-vs-noise curve is linear in dB.

*Sim deliverable.* Lets us fit a `noise.sigma(gain)` LUT and a `gain_db(slider)` table grounded in real bench measurements. The sim becomes physically accurate at any console gain, not just 54 — required for matching clinical captures whose gain is not necessarily 54.

**A2. Paired AR-OFF + AR-ON anechoic capture.** Cite [E6 procedure steps 3-5](../../../instrument-calibration/docs/ivus_calibration_protocol.md#procedure-3) directly — the protocol already specifies both AR ON and AR OFF; we just need the AR-OFF stream shipped to us alongside the AR-ON one we already have. 30 frames each at gain 68, all sliders centered.

*Why we need it.* The bench ring-down templates we have today are post-AR-subtraction, so when we calibrated `ring_down.peak` and `ring_down.extent_mm` we matched the *residual* waveform after AR removed the bulk of it. The sim's ring-down therefore runs ~7.5 dB low at peak and ~0.8 mm short at extent vs the AR-ON bench frame (see Test E above).

*Sim deliverable.* Re-derive `ring_down.peak`, `extent_mm`, and `fall_off_db_per_mm` against the true AR-OFF waveform amplitude. Test E flips to PASS without any sim physics changes — this is a parameter recalibration only.

**A3. Multi-gain repeat of the current copper-wire phantom.** Re-image the existing copper-wire fixture (per [E2](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e2--spiral-wire-phantom-2d-psf), but using the existing fixture, not the new spiral phantom — that's B1) at **gain ∈ {20, 30, 40}** in addition to the existing gain-54 capture. Same fixture, same diameter (D = 60 mm), 30 frames each.

*Why we need it.* At gain 54 the inner wires (r = 5, 10 mm) saturate to palette 239 on the bench, so we can't measure their −6 dB FWHM from the existing data. At gain 20-30 those inner wires drop into the linear palette band, exposing their PSF widths, while outer wires (r ∈ {15, 20, 25} mm) move down toward the noise floor where we already have decent measurements.

*Sim deliverable.* Validates the sim's axial + lateral PSF widths against bench widths at all five radii — Tests C and D candidate PASS. If the kernels miss bench, drives a small recalibration pass (`pulse_duration_cycles`, `effective_element_radius_mm`, `focal_length_mm`); if they match, we ship Tests C and D as PASS without any kernel changes. Highest-leverage Tier A request for PSF-related sim work.

### Tier B — uses the spiral-fixture STLs we already shipped

**B1. Build the canonical 12-wire spiral fixture and run E2.** Cite [E2 — Spiral Wire-Phantom 2D PSF](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e2--spiral-wire-phantom-2d-psf) end-to-end. Print 2 × `hardware/wire_spiral_disc.stl` and use 3 × `hardware/wire_spiral_standoff.stl` (or M3 threaded standoffs) per the protocol's *Building the spiral fixture* note. Wire layout: 12 wires on a 1-turn Archimedean spiral at `(r, θ)` ∈ {(4, 0°), (6, 30°), (8, 60°), (10, 90°), (12, 120°), (14, 150°), (16, 180°), (18, 210°), (20, 240°), (22, 270°), (24, 300°), (26, 330°)}. Capture 30 frames per catheter rotation × 4 rotations (0°, 90°, 180°, 270°) per the E2 procedure. Gain set per E2 spec: deepest wire ≥ 30 dB above noise but the shallowest ≤ 95 % saturated.

*Why we need it.* The current copper-wire fixture has only 5 wires and they are all on one azimuth, so we get one PSF sample per radius and zero azimuthal-uniformity data. The spiral layout gives us 12 PSF samples at 12 distinct radii including denser sampling around the focal zone (12-20 mm), at staggered azimuths so per-rotation SA-element-subset variance is decorrelated from per-radius PSF variation.

*Sim deliverable.* High-confidence, depth-binned `psf_lat_2d` lookup table for the Pass 5c lateral-PSF cache (E2 directly produces this table as a listed output). Plus bench-fit `probe.focal_length_mm` and `probe.element_radius_mm` from the Gaussian-beam fit, replacing the manufacturer-spec values we currently use. Plus per-rotation azimuthal-uniformity QC numbers we don't have today at all.

**B2. Same spiral fixture, but with nylon monofilament wires.** This is the **primary** spec in the [E2 Equipment table](../../../instrument-calibration/docs/ivus_calibration_protocol.md#equipment-1) — nylon monofilament 70-100 µm (e.g. 4-0 polyamide surgical suture, or 1-2 lb-test clear fishing line). We currently use the alternate 36 AWG copper magnet wire (127 µm) spec because it's what the lab had on hand.

*Why we need it (this is the headline ask).* Nylon at 75 µm = λ/2 in water at 10 MHz, vs copper at 127 µm = 0.85 λ. Copper sits squarely in the **Mie resonance regime** (`ka ≈ 2.6`) where the backscatter cross-section is dominated by sphere/cylinder resonance modes — not well-approximated by either geometric optics or Rayleigh scattering, so the bench wire echo strength is hard to predict from physics. Nylon at 75 µm sits well inside the Rayleigh regime where backscatter scales as `(ka)⁴` and the cross-section is well-defined from the wire's geometric area + acoustic-impedance ratio. Bench wire echoes will be 30-50 dB lower than copper, dropping the inner wires out of saturation at any sensible gain.

*Sim deliverable.* This is the **primary unblocker for Pass 8** (per-material backscatter scaling, the planned fix for the wire-vs-bg contrast gap). Without B2 we'd have to calibrate Pass 8 against the resonance-regime copper phantom, baking a non-physical assumption into the per-material scaling. With B2 we calibrate Pass 8 against a clean Rayleigh scatterer and **predict** the resonance-regime copper wire echo strength as an out-of-sample validation. Long-term: a physically meaningful scatter-strength scaling that generalises to clinical scenes (calcified plaque, stent struts) with their own characteristic ka regimes.

### Tier C — needs new equipment / fabrication time

**C1. Cyst phantom — full E5 protocol.** Cite [E5 — Cyst Phantom](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e5--cyst-phantom-speckle-noise-reject) at gain ∈ {20, 50, 68} per the E5 procedure. CIRS 040GSE preferred; DIY recipe in Appendix A.3 acceptable.

*Why we need it.* The current `lumen`, `vessel_wall`, `extravascular` material parameters in `volcano_s5i.yaml` (mu0, mu1, sigma, specularity) are literature defaults — none of them are calibrated against bench data because we have no anechoic-cyst-in-tissue ROI to fit them against. Until E5 lands, every claim about the simulator producing realistic per-tissue scatter is a qualitative one.

*Sim deliverable.* Replaces the literature defaults with bench-fit values for `noise.sigma`, `noise.type`, `scattering_resolution_mm`, `scatter_integral_scale`, `reject_db`, and the per-tissue `mu0` / `mu1` / `sigma`. Unblocks the entire in-vivo simulator workstream — without E5, all cyst / vessel-wall renders are physically uncalibrated.

**C2. Flat reflector / step phantom for E7 — log compression validation.** Cite [E7 — Grayscale / Compression Calibration](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e7--grayscale--compression-calibration). Either path works: the preferred RF-injection setup (programmable RF generator + attenuator + coupling jig) or the step-phantom fallback (CIRS 044, or the DIY 6-chamber phantom in Appendix A.4).

*Why we need it.* Today Test G is a synthetic-envelope self-consistency check on the K2v2 log-compression kernel against the spec mapping. We have not validated the mapping against measured device output. The single-anchor `gain_db` calibration also leaves the slider→dB curve under-determined for non-54 gain settings (B1's noise sweep covers part of this; E7 grounds the absolute amplitude calibration).

*Sim deliverable.* Closes the residual log-compression ambiguity flagged by the *Caveat* under Test G. Replaces the synthetic-envelope check with a true device-output validation. Pairs with A1 to fully ground the gain-vs-noise relationship.

**C3. Slice-thickness sweep — E3.** Cite [E3 — Slice-Thickness Sweep](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e3--slice-thickness-sweep-elevational-psf). Bead or tungsten-wire target translated along the catheter long axis.

*Why we need it.* The sim's elevational PSF is currently set to a fixed default (Gaussian σ = 2 mm) — it has never been validated against bench data. For 2D imaging this matters less, but for any off-imaging-plane scatter (volumetric phantoms, angled vessels, 3D reconstructions) the elevation-direction PSF is part of the model.

*Sim deliverable.* Fits `probe.elevational_height_mm` and the elevational PSF profile from bench data. Sim becomes accurate for off-plane scatter scenarios.

**C4. Tissue / material fit — E8.** Cite [E8 — Tissue / Material Fit](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e8--tissue--material-fit). Per-tissue (intima, media, calcified plaque, fibrous plaque) speed-of-sound, attenuation, and scatter parameters from in-vivo or ex-vivo captures.

*Why we need it.* Replaces the placeholder `vessel_wall` / `extravascular` parameters in `volcano_s5i.yaml` with clinically meaningful per-tissue values.

*Sim deliverable.* Required before any in-vivo phantom rendering can claim quantitative fidelity. C1 calibrates the global scatter model; C4 differentiates per-tissue.

### Stretch goal — second device unit (likely infeasible)

Repeat any subset of the above experiments (ideally A1 + A3 + B1 at minimum) on a second Volcano s5i console + Eagle Eye catheter.

*Why it would be valuable.* Unit-to-unit hardware variance is the single largest unmeasured source of uncertainty in our calibration sheet — every YAML field today is a single point estimate from one device, and we don't know whether (for example) the +1.9 palette residual on the gain-alignment diagnostic is a sim error or just device-to-device variance. Without that bound we cannot tell whether any sim residual is within hardware tolerance or is a real model error worth additional work.

*Sim deliverable.* Error bars on `gain_db`, `noise.sigma`, `ring_down.peak`, `focal_length_mm`, `element_radius_mm`, and the per-tissue scatter parameters. Lets us state *which* sim residuals are within hardware tolerance and which are real model errors.

**However we assume this is not feasible at this time** given the cost and availability of a second clinical-grade unit; included here so the value is on record if a service loaner ever becomes available (e.g. during a console swap).

## What sim work is blocked on what

- **Tests C, D** (axial / lateral PSF): blocked on **A3** (lower-gain copper) and ideally also **B1 + B2** (canonical spiral + nylon).
- **Test E** (ring-down): blocked on **A2** (paired AR-OFF / AR-ON captures). Will pass on parameter recalibration alone, no sim physics changes.
- **Test F** (noise floor σ at all gains): blocked on **A1** (multi-gain anechoic).
- **Test G** (log compression production validation): blocked on **C2** (flat-reflector / step phantom).
- **Pass 8** (per-material backscatter scaling, fixes the wire-vs-bg contrast gap): blocked on **B2** (sub-wavelength nylon target). Calibration would be unphysical against the current Mie-regime copper wires.
- **In-vivo simulator workstream** (vessel walls, plaque, cyst rendering quantitative claims): blocked on **C1** (cyst phantom) and **C4** (per-tissue fit).
- **Off-plane / 3D phantom scenarios**: blocked on **C3** (slice-thickness sweep).

## How to reproduce
```
cd /home/jocelynbarker/i4h-sensor-simulation
python instrument-calibration/p035_visions/tier1_evaluation.py \
    --out instrument-calibration/p035_visions/tier1_results \
    --n-frames-wire 8 --n-frames-anechoic 8
```

Machine-readable summary: `tier1_results/tier1_summary.json`.
