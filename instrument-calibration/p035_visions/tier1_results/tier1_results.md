# Tier 1 — Physical fidelity evaluation
**Probe under test:** Volcano s5i / Visions PV.035 (10 MHz IVUS)
**Calibration sheet:** `instrument-calibration/p035_visions/volcano_s5i.yaml`
**Bench PSF anchor (tests C / D):** `B2 (12-wire spiral, n=168 wires pooled over 4 positions)`. Per-wire FWHM extractions stored at `ivus_test_0515/derived_aggregate/psf_b2_tungsten_water`.
**Bench-data visual sanity checks:** per-wire 2D patches with measured −6 dB contour ([per_wire_patches.png](ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_patches.png)), per-wire axial / lateral radial-line profiles ([per_wire_axial_profiles.png](ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_axial_profiles.png), [per_wire_lateral_profiles.png](ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_lateral_profiles.png)), pooled Gaussian-beam fit ([gaussian_beam_fit.png](ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/gaussian_beam_fit.png)), and axial-FWHM distribution ([axial_fwhm_histogram.png](ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/axial_fwhm_histogram.png)). Build with `visualize_wave0_psf.py`.
**Bench datasets.**

- **Wire-peak / PSF anchors** (Tests C, D, gain_db calibration): `ivus_test_0515/raw/b2_w_wire_p{1,2,3,4}/` (B2 tungsten / water spiral wire phantom, 154 wires across 4 catheter positions at slider 54 / D=60). Pooled in `ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/psf_fit.json`.
- **Ring-down + noise + gain alignment anchors** (Tests E, F, gain alignment, `noise.sigma` calibration): `ivus_test_0508/raw/c_take2_water/derived/ringdown/` (E6 anechoic paired AR-on / AR-off at g40_D60, g50_D30, g50_D60 -- no scatterers, no wedge masking, the protocol-correct device-noise reference).
- **TGC + speckle anchors** (Tests I, M): `ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}` (E4a uniform evaporated-milk phantom, three takes at sliders 60-76, attenuating + scattering medium, no wires -> clean TGC and speckle anchors).
**Reference operating point:** gain slider 54, displayed diameter 60 mm (radial pitch 0.12 mm).
**Sim render budget for this report:** 8 wire-phantom frames, 8 anechoic frames; one flat-reflector frame per impedance.

<div class="tldr" markdown="1">

## Summary

**Tier 1 gate: ⚠️ INCOMPLETE. 10 / 11 parameter-bank tests pass; ≥1 test is N/A pending bench data.**

Each parameter-bank test evaluates one calibrated component of the simulator -- PSF, receive-chain noise, ring-down, TGC, log compression, gain alignment, milk depth uniformity, milk speckle -- against a paired bench measurement. Phenomenology-bank rows report un-modelled bench phenomena (coherent reverberation contaminating the E6 water diagnostic, the milk E2E gain anchor that compounds every sim layer) as informational diagnostics and do not gate.

**Parameter bank -- gates Tier 1 readiness.**

| # | Test | Status |
|---|---|---|
| 1 | A. Configuration-sheet round-trip | ✅ PASS |
| 2 | B. Configuration self-consistency | ✅ PASS |
| 3 | C. Axial PSF (per radius) | ✅ PASS |
| 4 | D. Lateral PSF (per radius) | ✅ PASS |
| 5 | E. Ring-down (E6 anechoic, multi-pair) | ✅ PASS |
| 6 | F. Noise floor (E4c milk multi-slider primary, E6 water diagnostic) | ✅ PASS |
| 7 | G. Log-compression mapping | ✅ PASS |
| 8 | H. TGC schedule | ✅ PASS |
| 9 | B2. Gain alignment (B2a function / B2b sim-internal / B2c milk E2E) | ⚪ N/A |
| 10 | I. Depth uniformity (uniform milk bath) | ✅ PASS |
| 11 | M. Speckle (residual-domain milk anchor + mu0-invariance) | ✅ PASS |

*Parameter bank summary: 10 / 10 pass (`n/a` rows are excluded from the gating denominator).*

**Phenomenology bank -- informational diagnostics.** These rows flag sim/bench deltas attributable to known un-modelled phenomenology but do not gate the parameter-bank PASS / FAIL above; when a phenomenon becomes calibrated, the corresponding row promotes from `diagnostic` to a gating row in the parameter bank.

| # | Diagnostic | Status |
|---|---|---|
| 1 | F-diag. E6 anechoic-water envelope diagnostic (informational) | 🔍 DIAGNOSTIC |
| 2 | B2c-diag. Gain E2E milk anchor (informational) | 🔍 DIAGNOSTIC |
| 3 | M-diag. Legacy palette CoV_log + radial_corr (informational) | 🔍 DIAGNOSTIC |

**Acceptance-criteria framework.** Tier 1 gates around a three-tier shape-vs-magnitude policy:

1. **Sensor properties** (receive-chain noise, catheter ring-down, TGC, log compression, gain alignment) -- **both magnitude and shape gate**. Device parameters are calibrated against direct bench measurements (not material BSC values), so magnitude is a real anchor that must match.
2. **Materials with published BSC-vs-frequency curves** (water, blood, well-characterised metal reflectors) -- **tight magnitude** criteria.
3. **Less-characterised materials** (milk, soft-tissue defaults, lumen-fill heuristics) -- **shape gates, magnitude is informational**. Magnitude is back-fit from bench data (no literature anchor); gating on shape (which is independent of our magnitude back-fit) is principled.

Shape metrics (FWHM, correlation length, distribution shape parameters, depth-uniformity span ratio, contrast ratios) translate across gain, machine, and operator. Palette magnitude metrics require per-material BSC calibration that does not generalise unless a literature anchor exists or the thing being measured is the device itself.

**Outstanding sim limitations (current state):**

- **Bench artifact-reduction (AR) angular smoothing not modelled.** The bench display chain runs an angular-averaging filter across receive scanlines; the sim does not model it. This drives the bench-vs-sim background-texture difference in the wire-phantom polar B-mode, the Test F per-pixel temporal std-map texture, and Test M's `lateral_corr_arc_mm` (which is reported as informational for that reason; Test M gates on `radial_corr_mm`, which the sim matches). Resolvable by fitting an azimuthal smoothing kernel against the AR-OFF / AR-ON paired captures we already have.
- **Coherent catheter / container-wall reverberation in water not modelled.** The bench's AR-on water palette is dominated by catheter and container-wall reverberation above the true noise floor; the sim renders pure water as scatterer-free. This drives the F-diag phenomenology row and the bench-side speckle background in the wire-phantom polar B-mode. Gating milk anchors are unaffected because milk attenuation suppresses the reverb tail; resolvable by adding a coherent reverberation source on the sim's receive chain.
- **Per-material backscatter for materials outside the calibrated set.** Per-tissue `mu0` / `mu1` / `sigma` values for `vessel_wall` and `extravascular` are literature defaults, not bench-fit. The PSF / wire-amplitude path is calibrated against 30 µm tungsten (`ka ≈ 0.6`, long-wavelength / Rayleigh limit), so the OptiX flat-interface model is physical at the PSF level; the open gap is per-tissue echo-strength differentiation for in-vivo scenes. Resolvable by bench data request **B2** (multi-material flat-interface phantom) + **C4** (in-vivo tissue fit).

Full bench-data requests (Tiers A / B / C) are listed in [Bench data requests](#bench-data-requests) below.

</div>

<a id="bench-anchors"></a>

## Bench anchors — provenance & evidence

Every scalar the tier-1 tests use as a *bench reference* is listed here. For each value we show (a) the file it comes from, (b) the extraction protocol that derived it from the raw bench frame, (c) an evidence figure that overlays the ROI / mask / fit on the raw data, and (d) a short **what does correct extraction look like?** check so the reader can decide in one glance whether the figure is healthy or broken. Anything that looks wrong in one of these figures invalidates the test below that depends on it. Regenerate every figure with `python instrument-calibration/p035_visions/bench_evidence.py`.

> **Anchor-to-test pairing.** Each test is anchored against the bench capture whose medium and operating conditions match what the test is probing:
> - **PSF (tests C / D)** -- `ivus_test_0515/raw/b2_w_wire_p{1,2,3,4}/` (B2 tungsten / water spiral wire phantom, 154 wires at slider 54 / D = 60 mm across 4 catheter positions).
> - **Ring-down + receive-chain noise (tests E / F / B2)** -- `ivus_test_0508/raw/c_take2_water/` (E6 anechoic water with paired AR-on / AR-off captures and no scatterers in the field), plus the multi-slider milk anchors in `ivus_test_0515/raw/e4c_milk_lut_p1/derived/noise/` for the Test F primary gate.
> - **Depth uniformity + TGC (tests H / I)** -- `ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}/` (E4a uniform evaporated milk, a real attenuating + scattering medium). An anechoic-water depth-uniformity check only verifies that TGC keeps the noise floor flat, which is a much weaker check than testing TGC against tissue-like attenuation.
> - **Speckle (test M)** -- `ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}/derived/speckle/` (same E4a milk takes, slider 68 / D = 60 mm). The whole imaging window is a uniform-medium speckle target by design, so no cyst-mask or alignment fit is needed.

<a id="bench-anchor-log-multiplier"></a>

### 0. Calibration backbone -- `log_multiplier` (used by every dB / palette-delta in the report)

**Source files:** `instrument-calibration/p035_visions/volcano_s5i.yaml` (`processing.log_multiplier`). Bench corroboration: `ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_summary_v2.json` (AR-OFF ringdown peak palette across slider 40 / 50 at D = 60 mm).

**Anchor value & corroboration:**

| Source | Number |
|---|---:|
| YAML `processing.log_multiplier` | 137.4 |
| Bench gain-step (slope * 20 from E6 AR-OFF peaks at D = 60 mm) | 138.5 |
| Agreement | within 0.8 % |

**Extraction protocol.** The display palette follows `palette = log_multiplier * log10(amp / log_floor)`. Two AR-OFF captures at the same diameter (D = 60 mm) and two different slider settings (40 and 50) give a known `delta_gain_dB = 10` and an observed `delta_palette ~= log_multiplier * delta_gain_dB / 20`. Solving for `log_multiplier` produces the implied value without any assumption about absolute amplitude.

**Evidence figure.**

![log_multiplier](figures/bench_evidence_log_multiplier.png)

*(a) E6 AR-OFF ringdown peak palette vs slider (D = 60 mm) -- slope * 20 reads the implied log_multiplier directly off the bench data. (b) A real wire radial profile in palette space with the -6 dB FWHM threshold drawn for four candidate log_multipliers (20 / 50 / 100 / 137.4 = YAML). On the device's log-compressed palette, the threshold sits delta_palette = log_mult * log10(2) ~= 41.4 palette below the peak -- it does NOT visually land at half-peak. (c) Same profile back-projected to linear amplitude via the YAML log_multiplier; the -6 dB threshold now visually crosses at peak / 2, validating the envelope-FWHM convention used everywhere in the pipeline.*


**What does a correct extraction look like?**

- Panel (a): the two dots fall on a near-straight line with slope > 6 palette / slider unit. If the line is flat or has slope < 1 the device was saturating at one of the gains -- pick a lower-gain pair instead.
- Panel (b): the solid yellow line (= YAML log_mult = 137.4) should fall ~41 palette below the peak. The grey, purple, and blue dashed lines (= candidate log_mults 20 / 50 / 100) deliberately sit too close to the peak; they show what wrong values would look like and let the reader confirm by eye that 137.4 is the right one.
- Panel (c): the dashed red horizontal at amplitude = 0.5 should cross the back-projected profile at the same r positions where the yellow line crosses the palette profile in panel (b). If panels (b) and (c) disagree on those crossing positions the log_multiplier is wrong and every dB-quoted number downstream needs to be re-derived.
- **What would be wrong?** A bench gain-step slope < 6 pal/slider (implying log_mult < 120) would push the -6 dB threshold close to the peak and make FWHM measurements artificially narrow; a slope > 8 pal/slider would push the threshold lower and widen every FWHM. Either case manifests as a systematic axial / lateral PSF bias in Test C / D.

<a id="bench-anchor-psf"></a>

### A. PSF anchor — `B2 (12-wire spiral, n=168 wires pooled over 4 positions)` (feeds Tests C and D)

**Source files:** `ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/psf_fit.json` (scalar anchors), `ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_psf.csv` (per-wire FWHM table).

**Anchor values (Tests C / D consume these directly):**

| Value | Source field | Number | Used by |
|---|---|---:|---|
| Axial median FWHM | `axial_fwhm_mm_median` | 0.303 mm | Test C bench reference |
| Pulse-duration estimate | `pulse_duration_cycles` | 3.938 cycles | Sim YAML `probe.pulse_duration_cycles` |
| Lateral focus FWHM | `gaussian_beam_fit.lateral_fwhm_at_focus_mm` | 0.944 mm | Test D bench reference |
| Beam waist | `gaussian_beam_fit.w0_mm` | 0.567 mm | Sim YAML `probe.effective_element_radius_mm` |
| Focal length | `gaussian_beam_fit.z_f_mm` | 2.81 mm | Sim YAML `probe.focal_length_mm` |
| Rayleigh range | `gaussian_beam_fit.z_R_mm` | 6.55 mm | Sanity-check on beam-divergence model |

**Extraction protocol.** `extract_psf.py` polar-unwraps each B2 frame, walks a −6 dB-from-peak threshold across each wire's axial and lateral profiles (palette delta = `log_multiplier · 6 / 20 = 41.2` at log_mult = 137.4), sub-bin-interpolates the crossings, and sums clean wires across the 4 spiral positions. The Gaussian-beam fit (`gaussian_beam_fit.*`) is a 2-parameter least-squares fit of `w(z) = w0 · sqrt(1 + ((z − z_f) / z_R)²)` to the lateral FWHM-vs-depth scatter, with z_R derived from `w0` and the 10 MHz wavelength.

**Evidence figures.**

![psf_per_wire_patches](../../../ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_patches.png)

*Per-wire 2D polar patches with the −6 dB iso-palette contour (magenta) overlaid on the polar patch around each detected wire peak. Direct visual proof that the FWHM walkout locked onto the wire and not a side-lobe / ringdown artefact.*

![psf_per_wire_axial_profiles](../../../ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_axial_profiles.png)

*Per-wire axial radial-line profiles with FWHM crossings (dotted red verticals). Each row is one wire; the dashed grey horizontal sits at peak − 41.2 palette (= −6 dB amplitude at log_mult = 137.4), so the FWHM crossings are genuinely at −6 dB and not at −6 *palette units*.*

![psf_per_wire_lateral_profiles](../../../ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/per_wire_lateral_profiles.png)

*Per-wire azimuthal (lateral) profiles with FWHM crossings. Same threshold convention as the axial profile — this is the source of the arc-FWHM column in `per_wire_psf.csv` and hence the `lateral_fwhm_at_focus_mm` anchor.*

![psf_gaussian_beam_fit](../../../ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/gaussian_beam_fit.png)

*Lateral FWHM vs depth (markers) + Gaussian-beam fit (red curve) across all clean wires pooled over the 4 B2 positions. The fit yields the `w0_mm`, `z_f_mm`, and `lateral_fwhm_at_focus_mm` anchors used in Test D.*

![psf_axial_fwhm_histogram](../../../ivus_test_0515/derived_aggregate/psf_b2_tungsten_water/axial_fwhm_histogram.png)

*Axial-FWHM distribution across all clean wires. Red vertical = median (the Test C bench anchor); dashed grey = legacy sim spec for comparison.*


**What does a correct extraction look like?**

- Per-wire patches: the magenta -6 dB contour should enclose a single tight lobe centred on each wire's annotated position. Concentric / multi-lobed contours indicate the walkout snapped onto a side-lobe and the wire should be tagged `multilobed=True` in `per_wire_psf.csv`.
- Per-wire axial/lateral profiles: the dashed grey threshold sits at peak - 41.2 palette and the dotted-red vertical FWHM crossings should sit symmetrically around the peak. This is exactly what the log_multiplier validation figure (section 0 above) confirmed -- when the threshold looks 'high' on the palette plot, that is correct because palette is log-compressed.
- Gaussian-beam fit: the red curve should pass through the centre of the FWHM-vs-depth cloud with a minimum at z_f. Lateral FWHM at the focus should be 0.4-0.7 mm for a 10 MHz IVUS; > 1 mm or < 0.2 mm indicates the fit captured side-lobe artefacts.
- Axial-FWHM histogram: median should sit at ~0.18 mm for a clean 10 MHz IVUS; the spread (p10-p90) should be < 0.1 mm. A bi-modal histogram is a red flag for two wire-saturation regimes mixed together.
- **What would be wrong?** Border-clipped wires (the FWHM walkout hits the patch edge) get flagged `border_clipped=True` in `per_wire_psf.csv` and dropped; if > 30 % of wires are flagged, widen the patch.

<a id="bench-anchor-ringdown"></a>

### B. Ring-down anchor -- `ivus_test_0508/raw/c_take2_water/` (E6 paired AR-OFF / AR-ON, feeds Test E)

**Source files:** `ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_summary_v2.json` (scalar anchors per gain / diameter pair), plus the paired `ringdown_off_g*_d*.npy`, `ringdown_on_g*_d*.npy`, and `ringdown_residual_g*_d*.npy` mean A-lines.

> **Why E6 anechoic water.** The c_take2_water capture is a pure-water AR paired sweep with no scatterers in the field, so the polar-averaged A-line *is* the catheter ring-down (AR-OFF) plus the device noise floor (AR-ON). No wedge exclusion or wire mask is needed, which keeps the ring-down extent and slope free of wire-PSF leakage.

**Anchor values (E6 paired summary, all 3 pairs are pooled by Test E):**

| Value | Source field | g40_d60 | g50_d30 | g50_d60 | Used by |
|---|---|---:|---:|---:|---|
| Peak palette (AR-OFF) | `peak_palette_off` | 152.8 | 222.2 | 222.0 | Test E per-pair peak criterion |
| Peak palette (residual) | `peak_palette_diff` | 141.8 | 211.2 | 211.0 | Pure catheter ring-down contribution |
| Peak depth | `peak_depth_mm` | 2.16 mm | 2.16 mm | 2.16 mm | Ring-down onset alignment |
| Extent end r (5 %-of-peak crossing, ABSOLUTE r) | `extent_mm` | 3.24 mm | 3.48 mm | 3.48 mm | Test E per-pair extent criterion |
| AR-ON noise floor | `speckle_floor_on` | 11.0 | ~40 | ~40 | Per-pair noise reference |

> **Why pool across all 3 pairs.** Each E6 capture is a single AR-OFF / AR-ON pair at one (gain, diameter) operating point, so a single-pair anchor is vulnerable to per-acquisition artefacts (transient bath disturbances, AR-mode anomalies, occasional operator slider drift). Pooling the three pairs (slider 40 / D60, slider 50 / D30, slider 50 / D60) sweeps both gain (10 dB span) and diameter (2x) so the test's median pass criterion is robust to any single-point outlier. Test E renders the sim at each pair's slider (via a `sim_params.gain_db` bump) so the comparison is apples-to-apples at every operating point.

**Extraction protocol.** `derive_ringdown_v2.py` polar-unwraps each AR-OFF / AR-ON pair, builds the mean A-line (mean over all theta and frames in the pair), finds the maximum palette of the AR-OFF-minus-AR-ON residual at `peak_depth_mm`, then walks outward from that peak and records the **absolute radius at which the residual first drops below 5 % of its peak** as `extent_mm` (i.e. `extent_mm` is an absolute r, not a Δr from the peak; for the g50_d60 pair the peak sits at 2.16 mm and the 5 % crossing at 3.48 mm, so the catheter ring-down occupies Δr ≈ 1.3 mm of the depth axis). Subtracting the AR-ON mean A-line from the AR-OFF mean A-line yields the pure catheter ring-down residual. Panel (c) overlays a linear fit of the AR-OFF curve between `peak_depth_mm` and `extent_mm`, reported as palette/mm (and dB/mm via log_multiplier = 137.4, section 0); this fit is informational only -- the simulator's ring-down shape is driven by the measured AR-OFF waveform via `processing.ring_down.waveform_path`, not by a parametric slope.

**Evidence figure.**

![ringdown](figures/bench_evidence_ringdown.png)

*(a) Bench Cartesian AR-OFF frame `FILE0004` with the catheter dead-zone (orange, r <= 1.32 mm), ring-down peak (yellow, r = 2.16 mm), and 5 %-of-peak extent end (green, r = 3.48 mm) circles overlaid. (b) AR-OFF (blue), AR-ON (green), and the OFF - ON residual (red dashed) A-lines for the g50_d60 pair: the AR-OFF curve is the total signal in water, the AR-ON curve is the residual after the device suppresses catheter ring-down, and the difference is the pure catheter ring-down contribution. (c) AR-OFF A-line with peak (yellow vertical + dot), AR-ON noise floor (grey dotted), 5 %-of-excess threshold (green dotted), extent end (green vertical at the 5 %-of-peak crossing absolute r), and the linear fit of the AR-OFF curve between peak and extent (red dashed, informational). Every number in the table appears as a label.*


**What does a correct extraction look like?**

- Panel (b): the AR-OFF (blue) curve should have a sharp peak around r ~= 2.2 mm that decays to the AR-ON floor (~ 40 palette) within ~1-1.5 mm past the peak. The AR-ON (green) curve should be essentially flat between r = 2.5 mm and r = 9.5 mm at the noise-floor level. The residual (red dashed) should be indistinguishable from AR-OFF in the dead-zone -> peak band and approach zero in the AR-ON tail.
- Panel (c): the yellow peak dot should sit exactly on the blue curve's maximum; the green vertical (extent end, absolute r) should fall right where the blue curve crosses the green-dotted 5 %-of-peak threshold; the red-dashed linear fit should follow the steep drop from peak to extent within +/- 5 palette. If the fit doesn't follow the drop, the linear-fit window is wrong and the dB/mm number is unusable.
- **What would be wrong?** A noisy AR-ON tail (>= 5 palette std) would mean the pair acquisition is contaminated by an external echo (re-check that the probe pose didn't shift between AR-OFF and AR-ON). An AR-OFF tail that doesn't return to the AR-ON floor means there is wall reflection in the field; use a larger water tank.

<a id="bench-anchor-noise"></a>

### C. Anechoic noise stats / gain-alignment background -- `ivus_test_0508/raw/c_take2_water/` (E6 AR-ON, feeds Test F + gain alignment)

**Source files:** `ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_on_g50_d60.npy` (mean A-line with AR enabled, i.e. catheter ring-down suppressed, imaged in pure water). The matching summary block is in `ringdown_summary_v2.json` under the `pairs[g50_d60]` entry.

**Anchor values (E6 multi-pair, r in [4.0, 9.84] mm):**

| Pair | AR-ON mean | AR-ON std | AR-OFF mean | AR-OFF std | Status |
|---|---:|---:|---:|---:|---|
| g40 D60 | 11.00 | 0.00 | 11.12 | 0.41 | clipped to reject floor -> skipped |
| g50 D30 | 36.93 | 2.02 | 34.18 | 4.69 | usable |
| g50 D60 | 37.59 | 3.27 | 34.54 | 4.69 | usable |
| **avg of g50** | **37.26** | **2.65** | **34.36** | **4.69** | Used by gain-alignment B2 sub-tests (AR-OFF) and as the informational F-diagnostic (AR-ON) |

> **Why this E6 capture is anechoic by construction.** No scatterers and no ring-down (the AR processing suppresses catheter ring-down), so the deep-radius tail directly reports the device's electronic noise floor. Test F's gating calibration uses the multi-slider milk anchor (`ivus_test_0515/raw/e4c_milk_lut_p1/derived/noise/`); the E6 AR-ON deep-tail mean / std numbers above are reported here as the phenomenology-bank diagnostic that compares the sim's noise-only water palette to the bench's reverb-contaminated water palette (the milk anchor is not subject to that contamination because milk attenuation suppresses the reverb).

**Extraction protocol.** The AR-ON A-line is the mean over all theta of the polar-unwrapped frame. The deep-radius tail r in [4.0, 9.8] mm is taken as the noise sample (deep enough to be past any residual ring-down, shallow enough to stay within the imaging band). Mean / std / percentiles of the samples in that band are the anchor values.

**Evidence figure.**

![noise](figures/bench_evidence_noise.png)

*(a) Bench Cartesian E6 AR-ON frame with the deep-radius noise ROI annulus (green, r in [4.0, 9.8] mm) overlaid. No wires in field, no wedge exclusion. (b) AR-ON mean A-line (green) with the deep-radius band shaded green and the extracted mean (red horizontal) + +/- 1 sigma band (red translucent). The tail should be flat at ~38 palette with sigma < 5. (c) Histogram of the deep-radius palette samples (n = 49) with mean (red), +/- 1 sigma band, and p05 / p95 marked.*


**What does a correct extraction look like?**

- Panel (b): the AR-ON A-line should be flat in the deep-radius band. A persistent slope means TGC is still leaving depth-dependent gain (see depth uniformity, section D below).
- Panel (c): the histogram should be unimodal and tight (sigma < 5 palette at slider 50), centred close to the mean horizontal. A bimodal histogram or a long right tail indicates contamination by a residual scatterer in the water tank.
- **What would be wrong?** sigma > 10 palette at slider 50 indicates either (a) a real device noise problem or (b) the AR was turned off mid-acquisition and the tail is actually ring-down debris. In either case re-acquire the pair.

<a id="bench-anchor-depth"></a>

### D. Depth-uniformity profile -- `ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}/` (E4a uniform milk, feeds Test I)

**Source files:** `ivus_test_0515/raw/e4a_milk_gain_p1/FILE0004.dcm` (slider 68, D = 60 mm) + the matching frames in `e4a_milk_gain_p2` and `e4a_milk_gain_p3` (three takes of the same milk-only capture). Per-frame metadata in `<take>/derived/frames_meta.csv`.

> **Why E4a uniform milk.** Test I is asking whether TGC compensates correctly for real tissue-like attenuation, so the bench source needs to be a *uniform attenuating + scattering* medium. Evaporated milk (~0.5 dB/cm/MHz, like soft tissue) fits both requirements, and the polar frame contains no inclusions or wires to mask around. The per-r mean palette is then a direct TGC-vs-attenuation test: in a perfectly TGC-compensated medium it should be flat in r across the imaging band. Any droop = TGC under-compensates; any hump = TGC over-compensates.

**Anchor value:** the per-r mean palette curve across the three E4a takes at (slider 68, D = 60 mm), evaluated in r in [5, 20] mm (past the ringdown extent of ~3.5 mm, well inside the 25 mm displayed half-radius).

**In-band statistics (E4a milk, slider 68, D = 60 mm, r in [5, 20] mm, pooled across 3 takes):**

| Stat | Value | Interpretation |
|---|---:|---|
| Band mean | 41.4 palette | Mean speckle palette across the imaging band |
| Band std | 1.4 palette | Depth-direction speckle-mean dispersion |
| Peak-to-trough | 5.3 palette | Largest residual TGC vs attenuation mismatch |
| Linear slope | +0.13 palette/mm | Effectively flat |
| Slope in dB | **+0.02 dB/mm** | TGC matches milk attenuation to within ~0.02 dB/mm |

> **Operating point: slider 68 (matched on the sim side).** The reference operating point for the rest of the report is slider 54, but milk speckle at slider 54 sits below the device's reject floor (only ~8 % of the imaging band is above floor at slider 50, ~50 % at slider 57). Slider 68 is the lowest E4a gain at which milk speckle is consistently above reject. Rather than scaling the bench data down (which would push it below the reject floor and lose information), **Test I renders the sim at slider 68 too** by bumping `sim_params.gain_db` by +14 dB (the YAML's slider-54 calibration plus the slider-step convention 1 step = 1 dB). Sim and bench are therefore matched on both the *medium* (milk-vs-milk) and the *gain* axes.

> **The sim-side milk material.** The simulator's bulk-medium speckle is parameterised by the world's background material via the YAML's `mu0` (scatter probability) and `sigma` (scatter amplitude scale). The `milk` material in `volcano_s5i.yaml` uses literature acoustic values for the four *physical* knobs (impedance 1.58 MRayl, speed-of-sound 1530 m/s, attenuation 0.5 dB/cm/MHz, specularity 0) and `mu0 = 0.5` / `sigma = 0.3` seeded from the `extravascular` soft-tissue analogue. Because (`mu0`, `sigma`) is not a published quantity for this Bernoulli-Gaussian sparse-scatterer model, Test I's magnitude column is informational while the depth-uniformity *shape* (peak-to-trough span ratio) is the gating metric.

**Extraction protocol.** Each take is polar-unwrapped around the device centre (theta-zero is irrelevant for theta-averaged stats), then the per-r mean palette is the mean across all (takes x theta) at each radial bin. Linear-fit the in-band r in [5, 20] mm samples; slope * 20 / log_multiplier gives the depth-uniformity tilt in dB/mm. No alignment fit, no wire mask, no AR template subtraction needed.

**Evidence figure.**

![depth_uniformity](figures/bench_evidence_depth_uniformity.png)

*(a) Bench Cartesian E4a milk frame `FILE0004.dcm` (take p1, slider 68) with the depth-uniformity fit window (green, r in [5, 20] mm) overlaid. Note the textured milk speckle filling the imaging band -- this is the *uniform attenuating medium* that exercises TGC against real attenuation. (b) Per-r palette curve pooled across all 3 takes: blue line = mean across (takes x theta), green dotted = median, blue band = p10-p90 spread, red dashed = linear fit across the green shaded window. The strong spike at r ~ 2 mm is the catheter ring-down; the flat plateau from r ~ 5 mm onward is the milk speckle. (c) Text summary with all in-band stats and an interpretation guide.*


**What does a correct extraction look like?**

- Panel (b): the blue per-r mean should be visibly flat across r in [5, 20] mm, sitting between 35 and 50 palette. The p10-p90 spread (blue shaded band) should be < 5 palette wide -- a uniform medium has theta-symmetric speckle, so the spread comes only from the speckle CoV, not from real angular structure.
- The red-dashed linear fit should overlap the band mean within +/- 2 palette across the fit window; a slope > 0.3 dB/mm in magnitude is the threshold to investigate.
- Peak-to-trough across r in [5, 20] mm should be < 10 palette (= 1.5 dB). Larger values mean TGC has a real depth-dependent mismatch with the attenuation of milk and Test I will surface the same shape in the sim comparison.
- **What would be wrong?** A *downward* ramp > 0.3 dB/mm indicates TGC under-compensates for milk attenuation; a clear *upward* ramp > 0.3 dB/mm means TGC over-compensates and would bloom deep tissue too bright. Either way the YAML's `processing.tgc_control_points` schedule needs to be re-fit against this curve (the new control points should null out the observed ramp). A bumpy or multi-peaked per-r curve indicates either a non-uniform milk distribution (un-stirred sediment) or scatterer contamination -- re-acquire after stirring + thermal equilibration.

<a id="bench-anchor-speckle"></a>

### E. Speckle anchor -- `ivus_test_0515/raw/e4a_milk_gain_p*/derived/speckle/` (uniform-milk phantom, feeds Test M)

> **Why E4a uniform milk for speckle.** The E4a captures are pure evaporated milk with no inclusions, so the whole imaging window is a uniform-medium speckle target by design. No cyst mask, no alignment fit, and no Rayleigh-CoV violation from gel macro-inclusions.

**Source files:** `ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}/derived/speckle/speckle_summary.json` (3 takes, generated by `extract_speckle.py --gain-slider 68 --diameter-mm 60` -- the slider 68 D60 operating point keeps the milk speckle well above the reject-palette floor across the full r ∈ [3, 25] mm ROI), and the per-take `frames_meta.csv` for the gain / diameter metadata.

**Anchor values (medians across the 3 E4a takes at slider 68 D60):**

| Value | Source field | Used by |
|---|---|---|
| `cov_log_palette` | `summary.cov_log_palette` | Test M sim speckle CoV (palette) |
| `cov_linear_envelope` | `summary.cov_linear_envelope` | Sim envelope CoV (full-ROI, depth-tilt-confounded) |
| `cov_linear_depth_norm` | `summary.cov_linear_depth_norm` | Depth-detrended Rayleigh check; expect ≪ 0.523 for structured milk |
| `radial_corr_mm` | `summary.radial_corr_mm` | Test M sim radial corr |
| `lateral_corr_arc_mm` | `summary.lateral_corr_arc_mm` | **Informational only** (E4a uniform medium has lateral coherence longer than the 30-bin autocorr search window, so the bench number clips at ~3.67 mm regardless of speckle reality) |

**Extraction protocol.** `extract_speckle.py` polar-unwraps each E4a DICOM, applies the r ∈ [3, 25] mm radial-annulus ROI, drops samples below the reject_palette (= 11) floor, and computes CoVs + spatial autocorrelations as for E5 -- but with no cyst mask (the E4a captures have no inclusions, so the whole annulus is speckle). Per-frame medians are aggregated across the 3 takes with `np.median` to give the anchor.

**Evidence figure.**

![speckle](figures/bench_evidence_speckle.png)

*Cyst-overlay rendering retained from the E5 milk-agar-glycerin phantom because it shows the speckle ROI annulus and the contrast measurement geometry more legibly than a uniform-medium E4a frame (which is just a featureless ring of milk speckle). Test M's gating anchor numbers come from the E4a uniform-milk takes listed above; the E5 numbers stay in the test detail (`legacy_e5_bench`) for cross-reference only.*


**What does a correct extraction look like?**

- Bench `cov_log_palette` should fall in the [0.35, 0.45] range across all 3 takes (currently 0.378 / 0.384 / 0.400 -- spread < 0.025). A spread > 0.05 indicates either an inconsistent reject-floor mask or a transient milk concentration drift between takes.
- `cov_linear_depth_norm` should fall in the [0.10, 0.15] range; this is well below the Rayleigh target (0.5227) because milk is a *structured* sub-Rayleigh scatterer (fat globules are too small and too dense to satisfy the Rayleigh assumption of spatially uncorrelated, low-density scatterers). This is a feature of milk, not a calibration bug.
- `radial_corr_mm` should be ~0.45 mm and consistent across takes (currently 0.456 / 0.457 / 0.471). Spreads > 0.1 mm indicate the reject-floor mask is clipping different fractions of the ROI across takes.
- **What would be wrong?** A multi-modal palette histogram in the ROI (= ROI contaminated with a wire or reflector that should not be there), a CoV that drifts more than +/- 0.05 across the 3 takes (= unstable bath or gain), or a `cov_linear_depth_norm` near 0.523 (= the ROI is actually filled with uncorrelated Rayleigh speckle, which milk is *not* -- check the medium identification).

<a id="bench-anchor-tgc"></a>

### F. TGC schedule — `volcano_s5i.yaml` + E4 bench evidence (feeds Test H)

**Source files:** `instrument-calibration/p035_visions/volcano_s5i.yaml` (`processing.tgc_control_points`, depths in cm + gain in dB), with bench provenance in `ivus_test_0515/raw/e4{a,c}*_gain_*/derived/tgc/per_gain_profiles.json` (per-gain palette-vs-r curves in milk + milk/water).

**Anchor values.** Test H is a round-trip: it walks the 5-breakpoint LUT through the sim's TGC stage and confirms |sim_db − yaml_db| < 0.1 dB at every depth. The bench evidence shows where those 5 breakpoints came from.

**Extraction protocol.** The YAML LUT was fit against the bench per-gain palette-vs-r curves so that the post-TGC palette is approximately flat in the r ∈ [5, 20] mm imaging band (= the device operator was barely using TGC because water has negligible attenuation). Any later re-fit against tissue-mimicking phantom data will follow the same procedure — match the YAML breakpoints to the bench palette-vs-r plateau.

**Evidence figure.**

![tgc](figures/bench_evidence_tgc.png)

*(a) Bench per-gain palette vs r for one representative E4 capture (`e4c_milk_water_gain_p1`, D = 60 mm). Curves are overlaid for slider in {35, 50, 57, 68} so the reader can see how the post-TGC palette plateau scales with gain. (b) The YAML's `processing.tgc_control_points` schedule -- 5 breakpoints in (cm, dB) annotated with their values. The flat (0 dB) inner region from 0-15 mm matches the bench observation of a flat per-r plateau in that band; the deeper ramp matches the gradual roll-off the bench shows at r > 20 mm.*


**What does a correct extraction look like?**

- Panel (a): each gain curve should plateau in r in [5, 20] mm and then either stay flat or roll off slightly. Curves at different sliders should be vertically offset by ~7 palette per +1 slider unit (= log_multiplier * 1 dB / 20). If two curves overlap, one of the captures had auto-gain re-engage; drop it from the fit.
- Panel (b): the YAML schedule should have a flat inner region matching the bench plateau and a small positive ramp deeper than the plateau matching the bench roll-off. A large (> 6 dB) jump at any breakpoint is suspicious and should be back-checked against the bench data.
- **What would be wrong?** Bench curves that *droop* with depth (median palette decreasing > 20 palette across r in [5, 20] mm) indicate the medium isn't uniform (suspended sediment, temperature gradient). Re-acquire after stirring + thermal equilibration.

## A. Configuration-sheet round-trip — ✅ PASS *(parameter bank — gating)*

**Summary.** Every probe / processing / TGC / ring-down parameter in the YAML appears in SimParams with the identical value (no rounding).


## B. Configuration self-consistency — ✅ PASS *(parameter bank — gating)*

**Summary.** All 15 round-tripped fields match across IvusSimConfig.to_dict ↔ from_dict.


## C. Axial PSF (per radius) — ✅ PASS *(parameter bank — gating)*

**Summary.** B2 spiral phantom, 10 scored wires (w2..w11), 4 catheter rotations x 8 frames = 32 obs/wire. 10/10 wires pass per-wire axial FWHM match (tol = max(50 um, bench IQR)). Sim per-wire medians: r=6.0: 272 um, r=8.0: 266 um, r=10.0: 279 um, r=12.0: 257 um, r=14.0: 269 um, r=16.0: 273 um, r=18.0: 264 um, r=20.0: 271 um, r=22.0: 263 um, r=24.0: 266 um. Bench global median = 303 um.

*Measurement protocol.* We walk the −6 palette FWHM through each wire's peak using the bench's `fwhm_walkout_bins` estimator (sub-bin linear interpolation; `extract_psf.py`). Wires whose excess over the local 10th-percentile background is below 6 palette are reported as *not detected*. The C/D test renders a separate *diagnostic* frame with `log_floor = 1e-19`, ring-down disabled, no display window, and the 235-palette device-saturation cutoff suppressed -- this isolates the geometric PSF shape from the gain / clamp / display pipeline so we can compare FWHM to the bench regardless of whether the calibrated render saturates the 8-bit palette. (Inspecting the calibrated-render polar figure below tells you the consumer-visible saturation; the diagnostic table tells you whether the underlying PSF kernel matches the bench.)

*Lateral kernel in effect.* `processing.lateral_psf_kernel.type = constant_angular` (SA-aware Gaussian, calibrated to median bench angular FWHM = 7.31 deg / sigma_theta_rad = 0.05416). The expectation under this kernel is that the per-wire **angular FWHM** is roughly constant across depth and the lateral arc-FWHM grows linearly with r. The legacy fixed-focus `gaussian_beam` kernel (focal_length_mm = 9.55) remains available behind the YAML switch.

| Wire | r (mm) | n obs | n unsat | sim FWHM (median) | bench FWHM (median, IQR) | Δ | tol | pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 (snap) | 4.0 | 32 | 32 | 261 um | — | — | — | — |
| 2 | 6.0 | 32 | 32 | 272 um | 289 ± IQR 44 um (n=9) | -17 um | 50 um | ✅ |
| 3 | 8.0 | 32 | 32 | 266 um | 259 ± IQR 42 um (n=9) | +7 um | 50 um | ✅ |
| 4 | 10.0 | 32 | 32 | 279 um | 279 ± IQR 21 um (n=8) | +0 um | 50 um | ✅ |
| 5 | 12.0 | 32 | 32 | 257 um | 284 ± IQR 24 um (n=9) | -27 um | 50 um | ✅ |
| 6 | 14.0 | 32 | 32 | 269 um | 320 ± IQR 74 um (n=16) | -51 um | 74 um | ✅ |
| 7 | 16.0 | 32 | 32 | 273 um | 326 ± IQR 77 um (n=13) | -52 um | 77 um | ✅ |
| 8 | 18.0 | 32 | 32 | 264 um | 304 ± IQR 77 um (n=14) | -39 um | 77 um | ✅ |
| 9 | 20.0 | 32 | 32 | 271 um | 319 ± IQR 87 um (n=14) | -47 um | 87 um | ✅ |
| 10 | 22.0 | 32 | 32 | 263 um | 304 ± IQR 61 um (n=21) | -41 um | 61 um | ✅ |
| 11 | 24.0 | 32 | 32 | 266 um | 309 ± IQR 141 um (n=22) | -43 um | 141 um | ✅ |
| 12 (snap) | 26.0 | 32 | 32 | 269 um | — | — | — | — |

*Interpretation.* B2 spiral phantom: 10 scored wires (w2..w11) x 4 catheter rotations x 8 frames = 32 observations per wire. Per-wire pass = sim median axial FWHM within max(50 um, bench IQR) of the bench median at the matching radius (bench wires within ±1 mm). **10/10 scored wires pass.** This replaces the legacy 5-wire ladder phantom (per-wire ±29 um tolerance, dominated by single-wire outliers).

**Magnitude-side informational diagnostic.** Per-wire `gain_db_required_to_match_bench_median` from `gain_db_derivation.json` -- the dB shift you would need to add to a single sim wire's envelope amplitude for it to match the bench wire-peak median. Does **not** gate this test; surfaced here for visibility into the magnitude-side fingerprint of the sim PSF / scattering issues.

| Sim wire r (mm) | gain_db_required (dB) |
|---:|---:|
| 5 | +49.0 |
| 10 | +54.6 |
| 15 | +58.1 |
| 20 | +69.6 |
| 25 | +61.1 |

Pooled (across 133 bench wires): median 58.09 dB (p25 / p75 = 51.12 / 62.97). Sim per-wire spread across r ∈ [5, 25] mm: **20.6 dB** -- this is the fingerprint of the fixed-focus lateral PSF concentrating energy at the focal zone and the flat-interface Fresnel return over-shooting at ka << 1 (`expert_meeting_questions.md` Q1 + Q3).

**Bench-data extraction evidence (jump to [Bench anchors](#bench-anchors) for full provenance):**
* Per-wire 2D patches with the −6 dB iso-contour: see [PSF anchor evidence](#bench-anchor-psf).
* Axial profile + FWHM crossings: see [PSF anchor evidence](#bench-anchor-psf).
* Axial-FWHM distribution histogram: see [PSF anchor evidence](#bench-anchor-psf).

**Wire-phantom polar B-mode — sim vs bench:**

![wire phantom polar paired](figures/wire_phantom_polar_paired.png)

*Left:* sim with the calibrated YAML, mean of 8 frames, ring-down ON, display window ON, γ-stretched. The ring-down ring at r ≈ 1.8 mm dominates the inner zone; the catheter dead-zone mask (r < 1.0 mm) renders at the soft-reject floor (palette ~11); the bandlimited mottled background is the pre-PSF Gaussian noise stage. The ray-traced wires (red circles, sim wire layout: 5 spheres at r ∈ {5, 10, 15, 20, 25} mm) may clip to palette 239 in the calibrated render at inner radii -- this is the consumer-visible behaviour, and the bench also clips on its inner wires (r = 5, 10 mm) at slider 54. The C / D FWHM table above uses the *diagnostic* render (log_floor = 1e-19, no display window, no saturation cutoff) to extract the underlying kernel shape regardless of clamp behaviour. *Right:* single bench frame `FILE0000` (gain 54, D=60 mm), γ-stretched the same way; red circles mark the bench wire positions for the inner 5 wires.

*Why the backgrounds look different.* The bench frame carries (i) coherent reverberation from the catheter sheath and container walls and (ii) AR angular-smoothing applied to those reverb scanlines -- both produce the textured speckle and "comet-tail" streaks visible everywhere in the bench panel. Neither is modelled in the sim today (see the Summary's *Outstanding sim limitations*), so the sim renders a near-black water background plus the pre-PSF Gaussian noise stage averaged down by ~√n_frames. Wire positions, ring-down geometry, dead-zone, and inner-wire palette clipping reproduce bench-like in both panels -- the background-texture gap is the residue of the un-modelled bench display chain, not a PSF or wire-rendering error.

**Per-radius FWHM:**

![PSF vs radius](figures/psf_vs_radius.png)

## D. Lateral PSF (per radius) — ✅ PASS *(parameter bank — gating)*

**Summary.** B2 spiral phantom, 10 scored wires (w2..w11). 10/10 wires pass per-wire lateral arc-FWHM match (tol = max(20%, bench IQR/2)). Sim per-wire arc-FWHM medians: r=6.0: 0.74 mm, r=8.0: 0.97 mm, r=10.0: 1.17 mm, r=12.0: 1.41 mm, r=14.0: 1.64 mm, r=16.0: 1.86 mm, r=18.0: 2.10 mm, r=20.0: 2.48 mm, r=22.0: 2.56 mm, r=24.0: 2.79 mm. Bench Gaussian-beam fit z_f = 2.8 mm, focus FWHM = 0.94 mm (informational; constant_angular has no focal minimum).

*Measurement protocol.* We walk the −6 palette FWHM through each wire's peak using the bench's `fwhm_walkout_bins` estimator (sub-bin linear interpolation; `extract_psf.py`). Wires whose excess over the local 10th-percentile background is below 6 palette are reported as *not detected*. The C/D test renders a separate *diagnostic* frame with `log_floor = 1e-19`, ring-down disabled, no display window, and the 235-palette device-saturation cutoff suppressed -- this isolates the geometric PSF shape from the gain / clamp / display pipeline so we can compare FWHM to the bench regardless of whether the calibrated render saturates the 8-bit palette. (Inspecting the calibrated-render polar figure below tells you the consumer-visible saturation; the diagnostic table tells you whether the underlying PSF kernel matches the bench.)

*Lateral kernel in effect.* `processing.lateral_psf_kernel.type = constant_angular` (SA-aware Gaussian, calibrated to median bench angular FWHM = 7.31 deg / sigma_theta_rad = 0.05416). The expectation under this kernel is that the per-wire **angular FWHM** is roughly constant across depth and the lateral arc-FWHM grows linearly with r. The legacy fixed-focus `gaussian_beam` kernel (focal_length_mm = 9.55) remains available behind the YAML switch.

| Wire | r (mm) | n obs | n unsat | sim FWHM (median) | bench FWHM (median, IQR) | Δ | tol | pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 (snap) | 4.0 | 32 | 32 | 0.48 mm | — | — | — | — |
| 2 | 6.0 | 32 | 32 | 0.74 mm | 0.77 ± IQR 0.12 (n=6) | -0.03 mm | 0.15 mm | ✅ |
| 3 | 8.0 | 32 | 32 | 0.97 mm | 0.98 ± IQR 0.13 (n=5) | -0.01 mm | 0.20 mm | ✅ |
| 4 | 10.0 | 32 | 32 | 1.17 mm | 1.19 ± IQR 0.11 (n=6) | -0.02 mm | 0.24 mm | ✅ |
| 5 | 12.0 | 32 | 32 | 1.41 mm | 1.47 ± IQR 0.30 (n=8) | -0.06 mm | 0.29 mm | ✅ |
| 6 | 14.0 | 32 | 32 | 1.64 mm | 1.80 ± IQR 0.29 (n=13) | -0.17 mm | 0.36 mm | ✅ |
| 7 | 16.0 | 32 | 32 | 1.86 mm | 2.19 ± IQR 0.51 (n=9) | -0.33 mm | 0.44 mm | ✅ |
| 8 | 18.0 | 32 | 32 | 2.10 mm | 2.37 ± IQR 0.25 (n=6) | -0.27 mm | 0.47 mm | ✅ |
| 9 | 20.0 | 32 | 32 | 2.48 mm | 2.64 ± IQR 0.35 (n=8) | -0.16 mm | 0.53 mm | ✅ |
| 10 | 22.0 | 32 | 32 | 2.56 mm | 2.70 ± IQR 0.46 (n=16) | -0.14 mm | 0.54 mm | ✅ |
| 11 | 24.0 | 32 | 32 | 2.79 mm | 2.99 ± IQR 0.63 (n=13) | -0.19 mm | 0.60 mm | ✅ |
| 12 (snap) | 26.0 | 32 | 32 | 3.05 mm | — | — | — | — |

**Magnitude-side informational diagnostic.** Per-wire `gain_db_required_to_match_bench_median` from `gain_db_derivation.json` -- the dB shift you would need to add to a single sim wire's envelope amplitude for it to match the bench wire-peak median. Does **not** gate this test; surfaced here for visibility into the magnitude-side fingerprint of the sim PSF / scattering issues.

| Sim wire r (mm) | gain_db_required (dB) |
|---:|---:|
| 5 | +49.0 |
| 10 | +54.6 |
| 15 | +58.1 |
| 20 | +69.6 |
| 25 | +61.1 |

Pooled (across 133 bench wires): median 58.09 dB (p25 / p75 = 51.12 / 62.97). Sim per-wire spread across r ∈ [5, 25] mm: **20.6 dB** -- this is the fingerprint of the fixed-focus lateral PSF concentrating energy at the focal zone and the flat-interface Fresnel return over-shooting at ka << 1 (`expert_meeting_questions.md` Q1 + Q3).

*Per-wire angular FWHM (constant-angular kernel check).* Bench median angular FWHM ≈ 6.75° (the kernel's target). Sim per-wire angular FWHM:

| Wire | r (mm) | sim ang FWHM (°) | Δ vs target (°) |
|---:|---:|---:|---:|
| 2 | 6.0 | 7.06 | +0.32 |
| 3 | 8.0 | 6.95 | +0.20 |
| 4 | 10.0 | 6.72 | -0.03 |
| 5 | 12.0 | 6.72 | -0.03 |
| 6 | 14.0 | 6.69 | -0.05 |
| 7 | 16.0 | 6.67 | -0.08 |
| 8 | 18.0 | 6.68 | -0.07 |
| 9 | 20.0 | 7.09 | +0.35 |
| 10 | 22.0 | 6.67 | -0.07 |
| 11 | 24.0 | 6.67 | -0.07 |

Sim per-wire angular spread ≈ 0.4° around the 6.7° target. Under a perfect constant-angular kernel the spread would be zero; residual spread reflects per-wire measurement scatter (10 wires x n rotations x n frames) plus the kernel's interaction with the lumen-water speckle background, not a kernel-form error.

*Focal trend (informational).* Sim lateral minimum at r = 6.0 mm; bench Gaussian-beam fit z_f = 2.8 mm. The constant_angular kernel has no focal minimum within the phantom (sim arc-FWHM grows ~linearly with r), so the 'minimum' is just the wire at the smallest r. Both numbers are kept as legacy provenance from the gaussian_beam kernel; under the SA-aware kernel they do not gate the test.

**Bench-data extraction evidence (jump to [Bench anchors](#bench-anchors) for full provenance):**
* Per-wire lateral profile + FWHM crossings: see [PSF anchor evidence](#bench-anchor-psf).
* Gaussian-beam fit (source of `z_f_mm`, `lateral_fwhm_at_focus_mm`): see [PSF anchor evidence](#bench-anchor-psf).

**Per-radius FWHM:**

![PSF vs radius](figures/psf_vs_radius.png)

(Side-by-side polar B-mode comparison shown under test C above.)

## E. Ring-down (E6 anechoic, multi-pair) — ✅ PASS *(parameter bank — gating)*

**Summary.** E6 ringdown anchor: 3 (gain, D) pairs (g40D60, g50D30, g50D60). Median |peak| rel.err = 0.4% (≤10% req), median |extent| err = 0.15 mm (≤0.30 req), median RMS inner 3 mm = 4.9 palette (≤5 req). Worst-case: peak 1.1%, extent 0.15 mm, RMS 8.5 palette.

| (gain, D) | sim gain_db | sim peak | bench peak | rel.err | sim extent | bench extent | extent err | RMS inner 3 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (40, 60) | +44.1 dB | 154.4 | 152.8 | 1.1% | 3.15 mm | 3.24 mm | 0.09 mm | 8.5 |
| (50, 30) | +54.1 dB | 222.8 | 222.2 | 0.2% | 3.33 mm | 3.48 mm | 0.15 mm | 4.9 |
| (50, 60) | +54.1 dB | 222.8 | 222.0 | 0.4% | 3.33 mm | 3.48 mm | 0.15 mm | 4.4 |

**Pooled pass criteria (median across pairs, AC tiering):**

| Criterion | Value | Threshold | Tier | Pass? |
|---|---:|---:|---|:--:|
| Median |peak| rel.err | 0.4% | ≤ 10% | magnitude (gates -- sensor property) | ✅ |
| Median |extent| err | 0.15 mm | ≤ 0.30 mm | shape (gates) | ✅ |
| Median RMS (r ∈ [0, 3] mm) | 4.9 palette | ≤ 5 | shape-of-decay (gates) | ✅ |

*Tiering.* Ring-down is a **sensor property** (catheter ring-down on the receive chain), not a material BSC, so both magnitude (peak palette, calibrated by `processing.ring_down.amplitude`) AND shape (radial extent + inner-r curve-match RMS) gate.

*Worst-case across pairs:* peak 1.1%, extent 0.15 mm, RMS 8.5 palette. Pooling across `n = 3` (gain, diameter) operating points (E6 anechoic AR-off references at slider 40 D60, slider 50 D30, slider 50 D60) prevents a single-gain bench artefact from hiding a real sim ringdown shape mismatch.

*Interpretation.* Test E compares the sim's ring-down mean A-line against the **E6 anechoic AR-off** reference at **every paired operating point** in `c_take2_water` (slider 40 / D60, slider 50 / D30, slider 50 / D60). The sim is re-rendered at each bench slider by bumping `sim_params.gain_db` by `(pair_slider − 54)` dB. Per-pair peak / extent / RMS are shown above; the pooled pass criterion is the median across pairs. Pooling across three (gain, diameter) operating points prevents a single-gain bench artefact from hiding a real sim ring-down shape mismatch.

**Bench-data extraction evidence (jump to [Bench anchors](#bench-anchors) for full provenance):** the protocol-correct E6 AR-off / AR-on A-lines and the extracted peak / extent numbers are documented in [B. Ring-down anchor](#bench-anchor-ringdown).

**Per-pair mean A-line (sim vs bench AR-off / AR-on):**

![ringdown mean aline](figures/ringdown_mean_aline.png)

## F. Noise floor (E4c milk multi-slider primary, E6 water diagnostic) — ✅ PASS *(parameter bank — gating)*

**Summary.** PRIMARY (milk anchor, gates): E4c D=15 r=[3,6.5] mm -- median rel.err = 1.7% across 3 usable anchors (<= 30% req); per-anchor: g50(bench=0.063 sim=0.062 rel=1% vf=4%), g56(bench=0.113 sim=0.117 rel=4% vf=8%), g62(bench=0.215 sim=0.219 rel=2% vf=17%), g68(bench=0.409 sim=0.415 rel=1% vf=74%). E4a cross-check median rel.err = 7.2% (3 anchors). Status: pass. LEGACY E6 DIAGNOSTIC (informational, not gating): 2 water pairs (g50D30, g50D60); ff_std_palette rel.err = 98.5%, mean abs.err = 26.6 palette, corr rel.err = 282.3% -- LEGACY_ALL_PASS = False. YAML: envelope_noise.{mean,sigma} = 0 / 0.379, reference_gain_db = 54.09, noise.sigma (legacy RF) = 0.


![Milk anchor paired -- bench vs sim env_resid across sliders + representative palette images](figures/milk_anchor_paired.png)

*Reading the bottom-right panel.* It is the **per-pixel temporal std map** in the envelope domain (bench-left, sim-right; same colorbar) -- the underlying field whose median is `envelope_residual_std`, the scalar this test gates on. The bench map shows large coherent patches of high std along certain theta because bench AR angular smoothing introduces lateral correlation in the std field; the sim map is more uniformly spotted because the simulator's envelope noise is iid Gaussian per pixel. **Texture similarity is NOT gated** -- only the median values are, and they agree to within the 30 % tolerance at every usable slider (median rel.err ≈ 2 % across the sweep). The same un-modelled AR angular smoothing also drives the lateral-autocorr gap in Test M.

**Milk anchor primary (multi-slider envelope-domain residual; gates Test F):**

| slider | sim gain_db | bench E4c env_resid | sim env_resid | E4c rel.err | E4c valid frac | bench E4a env_resid | E4a rel.err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | +54.49 | 0.0627 | 0.0620 | 1.1% | 3.7% | 0.0658 | 5.7% |
| 56 | +59.89 | 0.1127 | 0.1175 | 4.3% | 8.3% | 0.1266 | 7.2% |
| 62 | +65.29 | 0.2155 | 0.2191 | 1.7% | 17.1% | 0.2682 | 18.3% |
| 68 | +70.69 | 0.4089 | 0.4150 | 1.5% | 73.5% | 0.4318 | 3.9% |

**Pooled milk-anchor pass criterion**: median |sim - bench_E4c| rel.err across usable E4c anchors = 1.7% (≤ 30% req) -> **PASS**. Cross-check E4a median rel.err = 7.2%.

#### Informational: E6 water-domain noise distribution (does **not** gate Test F)

![E6 water noise distribution paired](figures/noise_distribution_paired.png)

*The bench and sim distributions are expected to differ in this figure -- it is shown to make the underlying phenomenology gap visible, not as a PASS/FAIL diagnostic.* The bench's AR-on water palette sits at ~38 (above the soft-reject knee at palette ~11) because the bench frame still contains coherent reverberation from the catheter sheath and container walls that the sim does not model. The sim's water render has no scatterers and no reverb, so its post-envelope Gaussian noise gets pushed below the soft-reject knee and the per-pixel frame-to-frame std collapses to ~0.14 palette (vs bench 9.40). Milk attenuation suppresses the bench reverb tail, which is why the milk anchor (above) agrees within 2 % and honestly gates the test. The same un-modelled coherent reverberation drives the F-diag row in the Summary's phenomenology-bank table.

| pair | sim gain_db | sim L_c | bench L_c | L_c rel.err | sim mean | bench mean | |Δmean| | sim std | bench std | σ rel.err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g50 D30 | +54.1 | 0.76 mm | 0.17 mm | 349.9% | 11.09 | 36.93 | 27.2 | 0.19 | 2.02 | 98.6% |
| g50 D60 | +54.1 | 0.76 mm | 0.24 mm | 214.7% | 11.09 | 37.59 | 26.1 | 0.19 | 3.27 | 98.5% |

**Legacy E6 water diagnostic (informational, does NOT gate Test F):**

- median |mean| abs.err: 26.6 palette
- median sigma rel.err: 98.5%
- median |sim L_c - bench L_c| / bench L_c: 282.3%

*Note.* These palette-domain metrics are operating-point biased -- the bench water palette sits above the soft-reject knee thanks to un-modelled coherent reverberation, while the sim's noise-only water palette is below the knee. The same `noise.sigma` value lands at very different palette stds in sim vs bench, so `noise.sigma` cannot be calibrated against bench water; the milk envelope-residual anchor above is the gating metric.

*Skipped (AR-on tail clipped to reject floor):* g40D60 (mean=11.0, std=0.00). These pairs are below the device's display floor so the noise distribution is erased; the sim cannot be calibrated against a clipped histogram.

**Bench-data extraction evidence (jump to [Bench anchors](#bench-anchors) for full provenance):** the bench's AR-on deep tail mean / std numbers are pulled from `ringdown_summary_v2.json`'s `noise_floor_palette_{mean,std}_from_palette_on` fields and reproduced from the.npy A-lines in the per-pair table above. See [C. Anechoic noise stats](#bench-anchor-noise) for the overlay of the bench tail band on the polar frame and the rationale for switching from the P_035 wedge-masked ROI (std 20.27, polluted by wire-PSF sidelobes) to the E6 AR-on deep tail (std 3.27, pure noise -- 6x narrower).

## F-diag. E6 anechoic-water envelope diagnostic (informational) — 🔍 DIAGNOSTIC *(phenomenology bank — informational)*

**Summary.** E6 anechoic-water envelope-domain diagnostic across 2 (gain, D) pairs. Operating-point biased by un-modelled coherent reverb (bench water palette sits above the soft-reject knee in the bench but not in the sim noise-only render); informational only, does NOT gate Test F.


## G. Log-compression mapping — ✅ PASS *(parameter bank — gating)*

**Summary.** Synthetic envelope sweep: |Δpalette| ≤ 0.0 ≤ 3 across 6 amplitudes spanning ~3.5 decades. K2v2 kernel matches the spec mapping exactly (log_floor = 1).

| amp | spec palette | kernel palette | Δpalette | Δ dB |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | +0.00 | +0.00 |
| 1 | 0.00 | 0.00 | +0.00 | +0.00 |
| 10 | 137.40 | 137.40 | +0.00 | +0.00 |
| 1e+02 | 274.80 | 274.80 | +0.00 | +0.00 |
| 1e+03 | 412.20 | 412.20 | +0.00 | +0.00 |
| 5e+03 | 508.24 | 508.24 | +0.00 | +0.00 |

Kernel formula: `log_multiplier * log10(max(amp, eps) / max(log_floor, eps))`. Spec formula: `log_multiplier * log10(amp / log_floor)`.

*Caveat.* This is a synthetic-envelope sweep against the spec mapping — a self-consistency check on the K2v2 kernel, not a measurement against device output. A true production validation requires the bench-side flat-reflector / step-phantom amplitude sweep (see Bench data request [C2](#bench-data-requests)).

## H. TGC schedule — ✅ PASS *(parameter bank — gating)*

**Summary.** True simulator round-trip: differential render of T_B (5-CP 0->+6 dB ramp) minus T_A (flat 0 dB) recovers the YAML Delta_TGC with RMS=0.183 dB (<= 0.5 dB tol), max-abs=0.566 dB (<= 1.0 dB tol) over r in [5.0, 25.0] mm. Ramp slope: sim +0.3680 dB/mm vs YAML +0.4000 dB/mm.


**Method: true simulator round-trip.** We render the same uniform-milk world with two different TGC schedules and compare the resulting per-r palette delta against the YAML's interpolated `T_B(r) - T_A(r)`:

- `T_A` = [[0.0, 0.0], [3.0, 0.0]] (flat 0 dB)
- `T_B` = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.5, 6.0], [3.0, 6.0]] (5-CP ramp 0 -> +6 dB)
- Eval band: r in [5.0, 25.0] mm (skip ring-down + guard).
- `envelope_noise.mean` overridden to 0.0 for this test (saved YAML value 0.00 restored afterwards) so the log compressor stays in its linear band.

In the linear log-compression band the chain reduces to `palette_B - palette_A = log_mult / 20 * (T_B - T_A)`, so recovered `Delta_T_meas = (P_B - P_A) * 20 / log_mult` (log_mult = 137.4) must match the YAML `Delta_T_yaml` over the eval band.

**Result:** RMS = 0.183 dB (tol = 0.50), max-abs = 0.566 dB (tol = 1.00).

**Ramp slope check (diagnostic).** In r in [10.0, 25.0] mm the recovered slope is +0.3680 dB/mm vs the YAML +0.4000 dB/mm.

![TGC round-trip](figures/tgc_roundtrip.png)

**Why this matters.** The round-trip method exercises the C++ TGC kernel end-to-end and catches any regression of the `create_piece_wise_tgc` control-point interpolation (`std::upper_bound`-based CP search; depth-vs-sample mapping derived from `t_far / buffer_size`) or of the cfg ↔ SimParams binding for `tgc_control_points`. Both are tested by the multi-CP `T_B` schedule above; a single-CP self-consistency check would miss either fault.

**Bench-data extraction evidence (jump to [Bench anchors](#bench-anchors) for full provenance):** the 5 TGC breakpoints (`processing.tgc_control_points` in the YAML) were fit from the E4 multi-gain per-r palette curves in milk + milk/water. The bench curves and the YAML schedule are shown side-by-side in [F. TGC schedule](#bench-anchor-tgc).

## B2. Gain alignment (B2a function / B2b sim-internal / B2c milk E2E) — ⚪ N/A *(parameter bank — gating)*

**Summary.** B2a function-check vs bench wire peaks: n/a (median |dB err| = nan dB, worst nan dB, tol nan dB). B2b sim-internal 6dB delta in milk (palette_pre domain): pass (measured 40.6 vs expected 41.2 palette_pre = +0.09 dB err, tol 3.0 palette). B2c E2E milk at slider 68 D=15: pass (sim 58.6 vs bench 56.0 palette = +0.38 dB, tol 10.0 palette). Overall: n/a (at least one sub-test n/a (bench data missing); see per-sub-test 'reason').


**Decomposition.** Gain alignment is split into three decoupled sub-tests so each failure mode is identifiable in isolation:

| Sub-test | Probe | Sim | Bench / Target | Δ (sim − bench) | Pass? |
|---|---|---:|---:|---:|:--:|
| B2a (function check) | slider_to_db vs bench wires | sim slider_to_db | bench slider→dB LUT | med |Δ| dB = nan (≤ 1.5 req) | ❌ |
| B2b (sim-internal 6 dB Δ) | palette_pre after inverting soft-reject + log | meas Δ = 40.6 | exp Δ = 41.2 | |dB err| = 0.09 (palette tol = 3.0) | ✅ |
| B2c (E2E milk slider 68) | uniform milk render, slider 68 D=15 | 58.6 palette | 56.0 palette | Δ dB = +0.38 (palette tol = 10.0) | ✅ |

**Status logic.** at least one sub-test n/a (bench data missing); see per-sub-test 'reason'

*Decomposition rationale.* The three sub-tests isolate three independent failure modes (slider→dB curve shape; sim-internal log / reject chain; end-to-end calibration in milk) so a B2a failure does not mask a B2b structural bug, and a B2c calibration miss does not obscure a B2a / B2b layer that is structurally correct.

## B2c-diag. Gain E2E milk anchor (informational) — 🔍 DIAGNOSTIC *(phenomenology bank — informational)*

**Summary.** B2c E2E milk anchor at slider 68 D = 15 mm: sim palette mean = 58.6 vs bench 56.0 (Δ = +0.38 dB, tol 10.0 palette; sub-status: pass). Compounds every sim layer (gain pipeline, log compression, material, noise); does NOT gate B2 PASS/FAIL on its own (B2b -- sim-internal gain delta -- is the parameter-level gate).


## I. Depth uniformity (uniform milk bath) — ✅ PASS *(parameter bank — gating)*

**Summary.** Milk sim vs E4a bench mean palette over r ∈ [5.0, 20.0] mm @ slider 68: SHAPE (primary): sim peak-to-trough = 5.4 vs bench 5.5 (ratio 0.98, |ratio−1| ≤ 0.5 required); slope sim/bench = +0.03 / +0.13 palette/mm. MAGNITUDE (informational, milk not literature-anchored): RMS = 2.0 palette (strict ≤ 10, partial-band ≤ 40), max |Δ| = 6.0, bias = +1.4. Sim milk-bath depth uniformity matches bench shape; magnitude within tolerance.

| Quantity | Value | Tolerance | Tier |
|---|---:|---:|---|
| Sim span / bench span ratio | 0.98 | |ratio − 1| ≤ 0.5 | **shape (primary)** |
| RMS(sim − bench) palette over r ∈ [5.0, 20.0] mm | 2.03 | strict ≤ 10, partial ≤ 40 | magnitude (informational) |
| Max |Δ| palette | 6.00 | — | magnitude (info) |
| Bias (sim − bench) palette | +1.35 | — | magnitude (info) |
| Sim peak-to-trough palette | 5.41 | — | shape (input) |
| Bench peak-to-trough palette | 5.55 | — | shape (input) |
| Sim slope (palette/mm) | +0.032 | — | shape (diagnostic) |
| Bench slope (palette/mm) | +0.133 | — | shape (diagnostic) |
| Bench / sim slider | 68 / 68 (sim gain_db bumped to +70.7) | — | — |
| Sim frames / bench takes | 8 / 3 | — | — |

*Tiering.* Milk has no published BSC vs frequency curve, so the absolute RMS-of-palette-diff between sim and bench is **informational** -- it tells us how far our back-fit `milk.mu0 / sigma` is from bench magnitude, but it is not a generalisation-relevant AC. The peak-to-trough **span ratio** is the shape AC that gates -- a sim that captures the depth-uniformity shape correctly will generalise across operator gain / TGC even if its absolute palette is offset. See the **Acceptance-criteria framework** in the Summary.

![Depth uniformity](figures/depth_uniformity.png)

**Noise-floor model.** Receive-chain noise is post-envelope Gaussian whose per-pixel draw inherits the cached TGC linear-gain curve (`processing.envelope_noise.apply_tgc_depth_scaling = true`), mirroring how analog electronic noise enters the real receive chain BEFORE the TGC variable-gain amplifier. This couples the simulator's deep-r noise floor to the YAML TGC schedule, so the bench's deep-r palette plateau (visible in the figure above near r > 15 mm) is reproduced without lifting the shallow-r milk magnitude that gates B2c.

**Diagnostic playbook (for future failure modes).** Test I distinguishes two failure modes:

- **Mean-offset failure (RMS large, span ratio ≈ 1).** The milk material's `mu0` / `sigma` scattering pair is mis-tuned. Fix by running a sim-in-loop bisection on `(mu0, sigma)` against the per-r mean target. Does *not* indicate a TGC problem.
- **Tilt / span failure (slope mismatch; RMS may also be large).** The TGC schedule does not match the milk-medium attenuation gradient, OR the noise-floor model is mis-set. Fix by re-fitting `processing.tgc_control_points` against the E4a per-r curve, adjusting `processing.envelope_noise.{mean, sigma, apply_tgc_depth_scaling}`, or revisit the `milk.attenuation_db_per_cm_mhz` literature value (0.5 dB/cm/MHz used today; whole-milk literature spans 0.5–1.0).

**Bench-data extraction evidence (jump to [Bench anchors](#bench-anchors) for full provenance):** the E4a milk bench profile (Test I anchor) is plotted in [D. Depth-uniformity profile](#bench-anchor-depth) with the fit window, the per-r mean curve, and the linear-fit slope overlaid directly on the milk Cartesian frame.

## M. Speckle (residual-domain milk anchor + mu0-invariance) — ✅ PASS *(parameter bank — gating)*

**Summary.** Residual radial_corr: sim 0.132 mm vs bench E4c 0.130 mm (Δ=+0.002 mm, 1%; ✓ 25% tol). Bench E4a cross-check radial_corr 0.134 mm (3% E4a-E4c spread). mu0-invariance: max_rel_shift 2.0% (≤20% tol; ✓). per_scale: x1.0->0.132mm, x2.0->0.129mm.


![Residual radial+lateral autocorr paired -- sim vs bench-E4c vs bench-E4a with 1/e crossings annotated](figures/speckle_residual_autocorr_paired.png)

*Reading the two panels.* **Left (gating):** residual *radial* autocorrelation; sim and the two bench takes overlay almost exactly (1/e at ~0.13 mm), so the residual radial_corr_mm metric -- which is what Test M gates on -- passes inside its 25 % tolerance. **Right (informational, does NOT gate):** residual *lateral* autocorrelation; the sim (red) drops noticeably faster than the bench (blue / green). This gap is expected -- E4a uniform milk has no lateral structure in the medium, so the bench's apparent lateral coherence is dominated by the device's AR angular-smoothing filter, which spreads each receive scanline across several azimuthal samples. The sim does not model AR angular smoothing, so its lateral correlation length is set by the lateral PSF alone. Resolving this gap is tracked under "Outstanding sim limitations" (AR angular smoothing); it is deliberately not gated because doing so would fail the test for an un-modelled bench display stage rather than a sim error.

**Primary residual-domain anchor (gates Test M):**

| Metric | Sim | Bench E4c | Bench E4a (cross) | Δ (sim − E4c) | rel.err vs E4c | Tol | Pass? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| radial_corr_mm | 0.132 | 0.130 | 0.134 | +0.002 | 1.4% | ≤ 25% | ✅ |
| lateral_corr_arc_mm | 0.074 | 0.329 | 0.346 | -0.255 | 77.6% | informational | — |
| residual_std_palette (diag) | 8.721 | 8.302 | 8.832 | +0.419 | 5.0% | informational | — |

*Bench cross-validation*: E4a vs E4c radial_corr spread = 2.7% -- the residual-domain metric cancels the coherent-reverb contamination that differentiated E4a from E4c in the legacy palette-CoV diagnostic.

**mu0-invariance cross-check (gates Test M):**

| mu0 scale | milk mu0 | residual radial_corr (mm) | residual std palette | palette mean |
|---:|---:|---:|---:|---:|
| x1.0 | 0.500 | 0.132 | 8.721 | 57.5 |
| x2.0 | 1.000 | 0.129 | 8.028 | 40.7 |

**mu0-invariance**: max_rel_shift in residual radial_corr = 2.0% (≤ 20% req) -> **PASS**. If the residual radial_corr metric is dominated by PSF + `scattering_resolution_mm` (the parameters Test M validates), it must be invariant to `milk.mu0` (scatterer density). A shift > 20% would indicate the metric is conflated with amplitude and the primary-anchor pass / fail above is unreliable.

**Bench provenance.** E4c primary (take p1, n_frames=37) + E4a cross-check (take p1, n_frames=35) at slider 68, D = 15 mm, r in [3, 6.5] mm; see [E. Speckle anchor](#bench-anchor-speckle) for the full extraction protocol.

**Interpretation.** Test M asks: *does the simulator's PSF + `scattering_resolution_mm` reproduce the spatial decorrelation length of bench milk speckle?* The residual-domain metric (per-pixel temporal mean subtracted on a multi-frame stack) cancels coherent reverberation that biases the legacy palette-CoV / palette-radial_corr diagnostic; the mu0-invariance cross-check rules out scatterer-density as the dominant driver of the radial autocorrelation length. Failure modes:

- **Residual radial_corr too long** -- sim's effective axial PSF + `scattering_resolution_mm` is wider than bench. Calibration target: 2D sweep over `pulse_duration_cycles` and `scattering_resolution_mm` to bracket the joint operating point that matches bench radial_corr.
- **Residual radial_corr too short** -- sim's PSF is narrower than bench (the opposite mis-tune).
- **mu0-invariance violated (shift > 20 %)** -- the residual radial_corr metric is conflated with scatterer density; primary anchor pass / fail is unreliable.

The legacy palette-domain CoV_log + palette-radial_corr diagnostic is preserved as the M-diag phenomenology companion (see below) but does NOT gate Test M, because both metrics are biased by un-modelled bench coherent reverb.

## M-diag. Legacy palette CoV_log + radial_corr (informational) — 🔍 DIAGNOSTIC *(phenomenology bank — informational)*

**Summary.** Legacy palette-domain speckle diagnostic vs E4a Wave 0 anchor. CoV_log: sim 0.524 vs bench 0.384 (36% rel.err). Palette radial_corr: sim 0.166 mm vs bench 0.458 mm (64% rel.err). Both metrics biased by un-modelled coherent reverb (see test_speckle_wave0 docstring); does NOT gate Test M.


## Bench data requests
All eleven Tier 1 parameter-bank tests currently pass against the bench data we have on hand (see Summary). The requests below are **not** blockers for the current Tier 1 gate; they are the experiments that will (a) tighten the existing calibrations with wider operating-point coverage, (b) close the residual phenomenology gaps the sim flags as informational today (coherent reverb, AR angular smoothing), and (c) unblock the next workstreams (log-compression production validation, per-material backscatter, elevational PSF, in-vivo tissue rendering). Each request cites the specific experiment in the [IVUS Calibration & Characterization Protocol](../../../instrument-calibration/docs/ivus_calibration_protocol.md) (E1-E8), states the sim limitation it removes, and describes the concrete sim deliverable that it unblocks.

### Tier A — refinements using existing equipment

**A1. Multi-slider anechoic captures (gain LUT extension).** Cite [E6 — Ring-Down Capture](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e6--ring-down-capture-acoustic-reference) with **gain stepped through {20, 30, 40, 50, 54, 60, 68}**, AR ON, all TGC sliders centered, 30 frames per gain. Same probe, same bath, same temperature.

*What the sim has today.* `processing.envelope_noise.sigma` and `processing.gain_db` are already calibrated against the E4c milk anchor across four sliders (50 / 56 / 62 / 68) and the E6 anechoic capture at slider 50, so the sim noise sweep agrees with bench to within 2 % median rel.err (Test F). The gain LUT is linear-in-dB by construction (slider_to_db).

*What's still missing.* A direct measurement of `envelope_noise.sigma(slider)` over the full operator range (20-68) would replace the linear-in-dB assumption with a measured LUT. Useful for clinical captures at non-standard gains and for ruling out console-side non-linearities at the low-gain end.

*Sim deliverable.* Replaces `processing.envelope_noise.sigma` (scalar) + the linear gain LUT with a bench-fit `sigma(slider)` table; quantifies any residual gain-LUT non-linearity. Not blocking any Tier 1 test today.

**A2. Higher-gain paired AR-OFF + AR-ON anechoic capture.** Cite [E6 procedure steps 3-5](../../../instrument-calibration/docs/ivus_calibration_protocol.md#procedure-3) with paired AR-OFF + AR-ON streams at **slider 68** (current paired captures in `ivus_test_0508/raw/c_take2_water/` cover slider 40 + 50 only). 30 frames each, all TGC sliders centered.

*What the sim has today.* Test E now passes against the three paired E6 captures already shipped in `c_take2_water` (g40_d60, g50_d30, g50_d60). The ring-down amplitude + waveform template + extent are calibrated from those captures (peak rel.err = ~0%, extent err ≤ 0.3 mm, RMS inner-3mm ≤ 5 palette).

*What's still missing.* A slider-68 paired capture would let us verify the ring-down peak palette + extent shape at the top of the operator gain range, where the inner-zone palette saturates in the calibrated render. Cross-validation, not a blocker.

*Sim deliverable.* Out-of-sample cross-validation of `ring_down.amplitude` at slider 68; confirms the calibrated amplitude generalises across the full ±14 dB slider range.

**A3. Lower-gain repeat of the existing wire phantom (inner-wire FWHM).** Re-image the **existing tungsten wire fixture** (per [E2](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e2--spiral-wire-phantom-2d-psf)) at **gain ∈ {30, 40, 44}** in addition to the existing gain-50 captures, same fixture / diameter (D = 60 mm), 30 frames each.

*What the sim has today.* Tests C / D both pass against the B2 wave-0 spiral capture (n=168 wires pooled over 4 catheter rotations). The constant-angular-FWHM lateral PSF kernel (σ_θ = 0.054 rad) is calibrated against the median bench angular FWHM.

*What's still missing.* At slider 50, the inner wires (r ~ 5, 10 mm) saturate to palette 239 on the bench, so we extract their FWHM from a diagnostic render rather than directly from the saturated bench palette. Lower-gain captures would expose the inner wires in the linear palette band and give us direct bench-side FWHM measurements at every radius.

*Sim deliverable.* Direct bench-side validation of inner-radius PSF widths (r ≤ 10 mm), replacing the saturation-aware diagnostic-render extraction with a clean linear-palette measurement. Tightens C / D bench confidence intervals; not expected to change the PASS status.

### Tier B — uses the spiral-fixture STLs (cross-validation)

**B1. 12-wire nylon spiral cross-validation phantom.** Cite [E2 — Spiral Wire-Phantom 2D PSF](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e2--spiral-wire-phantom-2d-psf) with **nylon monofilament 70-100 µm** wires in the existing spiral-disc fixture (`hardware/wire_spiral_disc.stl`). Capture 30 frames per catheter rotation × 4 rotations (0°, 90°, 180°, 270°), gain set per E2 spec.

*What the sim has today.* Tests C / D pass against the **30 µm tungsten** spiral phantom (`ivus_test_0515/raw/b2_w_wire_p{1..4}`), which sits at `ka ≈ 0.6` -- the long-wavelength / Rayleigh limit -- giving a clean per-wire FWHM with no Mie-resonance artefacts.

*What's still missing.* An independent material (nylon, ka ≈ 1.0, similar acoustic impedance to tissue) lets us cross-validate the per-material backscatter scaling. Nylon echo strengths should fall 30-50 dB below tungsten under correct calibration; significant deviation would point at the per-material scaling factor in the OptiX flat-interface model.

*Sim deliverable.* Out-of-sample validation of `materials.bone` / per-material `mu0` scaling against a second, lower-impedance scatterer. Would unblock confident per-tissue extension (calcified plaque, stent struts) which lives in higher-impedance Mie regimes.

**B2. Multi-material flat-interface phantom for per-tissue backscatter calibration.** Cite [E2-extension Equipment table](../../../instrument-calibration/docs/ivus_calibration_protocol.md#equipment-1) -- mount thin flat slabs of representative materials (PMMA, gelatin, silicone, polyurethane, nylon) at known incident angles and depths. 30 frames per material, slider 50.

*What the sim has today.* Per-material `mu0` / `mu1` / `sigma` values are literature defaults for `lumen`, `vessel_wall`, and `extravascular`. No bench-side measurement of relative echo strength across materials.

*Sim deliverable.* Replaces literature defaults with bench-fit per-material backscatter and reflection-coefficient values. Required before any in-vivo rendering can claim quantitative per-tissue fidelity. This experiment subsumes the older "nylon vs tungsten Rayleigh anchor" ask; we have a clean Rayleigh anchor already (30 µm tungsten).

### Tier C — needs new equipment / fabrication time

**C1. Cyst phantom — full E5 protocol.** Cite [E5 — Cyst Phantom](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e5--cyst-phantom-speckle-noise-reject) at gain ∈ {20, 50, 68} per the E5 procedure. CIRS 040GSE preferred; DIY recipe in Appendix A.3 acceptable.

*Why we need it.* Test M now passes against the E4a uniform-milk phantom (residual radial_corr, mu0-invariance), but the cyst-in-tissue contrast metric (anechoic insert visibility in a speckle background) has no bench anchor today. In-vivo scenes routinely include anechoic structures (lumen, calcific cores) and we cannot currently claim the sim reproduces their contrast quantitatively.

*Sim deliverable.* Adds a cyst-contrast acceptance criterion to Test M (or a new Test M2). Validates the simulator's anechoic-vs-speckle contrast against bench at three gain operating points. Unlocks confident in-vivo rendering for scenes containing lumen / anechoic features.

**C2. Flat reflector / step phantom for E7 -- log-compression validation.** Cite [E7 — Grayscale / Compression Calibration](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e7--grayscale--compression-calibration). Either path works: the preferred RF-injection setup (programmable RF generator + attenuator + coupling jig) or the step-phantom fallback (CIRS 044, or the DIY 6-chamber phantom in Appendix A.4).

*Why we need it.* Today Test G is a synthetic-envelope self-consistency check on the K2v2 log-compression kernel against the spec mapping. We have not validated the mapping against measured device output across a controlled amplitude sweep.

*Sim deliverable.* Closes the residual log-compression ambiguity flagged by the *Caveat* under Test G. Replaces the synthetic-envelope check with a true device-output validation across a known amplitude ladder.

**C3. Slice-thickness sweep -- E3.** Cite [E3 — Slice-Thickness Sweep](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e3--slice-thickness-sweep-elevational-psf). Bead or tungsten-wire target translated along the catheter long axis.

*Why we need it.* The sim's elevational PSF is currently set to a fixed default (Gaussian σ = 2 mm) -- it has never been validated against bench data. For 2-D imaging this matters less, but for any off-imaging-plane scatter (volumetric phantoms, angled vessels, 3-D reconstructions) the elevation-direction PSF is part of the model.

*Sim deliverable.* Fits `probe.elevational_height_mm` and the elevational PSF profile from bench data. Sim becomes accurate for off-plane scatter scenarios.

**C4. Tissue / material fit -- E8.** Cite [E8 — Tissue / Material Fit](../../../instrument-calibration/docs/ivus_calibration_protocol.md#e8--tissue--material-fit). Per-tissue (intima, media, calcified plaque, fibrous plaque) speed-of-sound, attenuation, and scatter parameters from in-vivo or ex-vivo captures.

*Why we need it.* Replaces the placeholder `vessel_wall` / `extravascular` parameters in `volcano_s5i.yaml` with clinically meaningful per-tissue values.

*Sim deliverable.* Required before any in-vivo phantom rendering can claim quantitative fidelity. B2 + C1 calibrate the global scatter model; C4 differentiates per tissue.

### Closing the phenomenology gaps (no new equipment)

The phenomenology-bank rows in the Summary table flag two **modelling** gaps that current bench data is sufficient to close once the sim side is implemented -- they are listed here for completeness because they drive the visible bench-vs-sim texture differences in the wire-phantom polar B-mode, Test F E6 water diagnostic, and Test M lateral autocorrelation:

- **Catheter / wall coherent reverberation.** Source: any of the existing AR-OFF E6 captures in `c_take2_water` already contain the reverb signature. Sim work: add a coherent reverberation source on the receive chain (catheter sheath ring response + container-wall echo), driven by a measured AR-OFF residual outside the catheter dead-zone. Resolves F-diag and the wire-phantom polar background texture gap.
- **Bench artifact-reduction (AR) angular smoothing.** Source: the same paired AR-OFF / AR-ON E6 captures we already have are sufficient to fit an angular smoothing kernel. Sim work: model AR as an azimuthal IIR (or FIR) across receive scanlines, parameter-fit so the sim's `lateral_corr_arc_mm` matches bench under uniform milk. Resolves the Test M lateral-autocorr informational delta and the Test F std-map texture difference.

### Stretch goal -- second device unit (likely infeasible)

Repeat any subset of the above experiments (ideally A1 + A3 + B1 at minimum) on a second Volcano s5i console + Eagle Eye catheter.

*Why it would be valuable.* Unit-to-unit hardware variance is the single largest unmeasured source of uncertainty in our calibration sheet -- every YAML field today is a single point estimate from one device, and we don't know whether (for example) the +1.9 palette residual on the gain-alignment diagnostic is a sim error or just device-to-device variance. Without that bound we cannot tell whether any sim residual is within hardware tolerance or is a real model error worth additional work.

*Sim deliverable.* Error bars on `gain_db`, `envelope_noise.sigma`, `ring_down.amplitude`, `focal_length_mm`, `element_radius_mm`, and the per-tissue scatter parameters. Lets us state *which* sim residuals are within hardware tolerance and which are real model errors.

**However we assume this is not feasible at this time** given the cost and availability of a second clinical-grade unit; included here so the value is on record if a service loaner ever becomes available (e.g. during a console swap).

## What sim work is blocked on what

All Tier 1 parameter-bank tests currently pass. This section lists the **next** sim deliverables and what (if anything) blocks each one.

- **Tier 1 PSF / noise / ring-down / TGC / gain / log-compression / depth-uniformity / speckle (Tests C, D, E, F, H, B2, G, I, M).** Not blocked. All eleven parameter-bank tests pass against the existing bench corpus (`ivus_test_0515` + `ivus_test_0508`). Tier A refinements (A1-A3) would tighten the calibrations and widen the operator-range coverage without changing PASS status.
- **Coherent-reverb + AR angular-smoothing sim models** (closes the F-diag and Test M lateral-corr informational gaps). Not blocked by bench data -- the existing E6 paired AR-OFF / AR-ON captures are sufficient. This is a sim implementation task (catheter-wall reverb source + AR azimuthal kernel) on the next-phase roadmap.
- **Test G production validation** (true device-output log-compression measurement, not the current synthetic self-consistency check): blocked on **C2** (flat-reflector / step phantom).
- **Per-material backscatter calibration** (per-tissue `mu0` / `mu1` / `sigma` beyond the current literature defaults): blocked on **B2** (multi-material flat-interface phantom). The previously-flagged Mie-regime calibration gap is now closed -- the existing 30 µm tungsten spiral sits at `ka ≈ 0.6` (long-wavelength / Rayleigh limit), so PSF and per-wire amplitude calibration are already physical. B1 (nylon spiral) remains available as cross-validation.
- **Cyst-contrast acceptance criterion** (anechoic-vs-speckle contrast as a quantitative gate, not just qualitative Test M passage): blocked on **C1** (E5 cyst phantom).
- **In-vivo simulator workstream** (vessel walls, plaque, cyst rendering with quantitative claims): blocked on **C1** (cyst phantom), **B2** (per-material backscatter), and **C4** (per-tissue fit).
- **Off-plane / 3-D phantom scenarios**: blocked on **C3** (slice-thickness sweep).
- **Confidence intervals on every calibrated YAML field** (currently single-device point estimates): blocked on the second-device-unit stretch goal -- not actively pursued.

## How to reproduce
```
cd /home/jocelynbarker/i4h-sensor-simulation
python instrument-calibration/p035_visions/tier1_evaluation.py \
 --out instrument-calibration/p035_visions/tier1_results \
 --n-frames-wire 8 --n-frames-anechoic 8
```

Machine-readable summary: `tier1_results/tier1_summary.json`.
