# Ultrasound Simulation Evaluation Report

This report describes test scenarios used to validate the ultrasound simulation. The scenarios are aligned with **validation practices reported in the literature**: geometric accuracy, attenuation with depth, resolution (point spread), contrast, scan uniformity, and reproducibility. For each scenario we state the validation aim, what to expect if the model is accurate, and what to verify in the generated images.

## Quick reference

| Scenario | Key image(s) |
|----------|--------------|
| Geometric accuracy: single boundary (cen | 01_centered_single_wall.png |
| Attenuation with depth & contrast (two-l | 02_centered_thick_wall.png |
| Geometric accuracy: known offset (eccent | 03_eccentric_probe.png |
| Resolution & geometry: point reflectors | 04_point_reflectors.png |
| Temporal/positional consistency (pullbac | 05_pullback_frame_00_z-1.00.png, 05_pullback_frame_01_z-0.50.png, 05_pullback_frame_02_z0.00.png, 05_pullback_frame_03_z0.50.png, 05_pullback_frame_04_z1.00.png |
| Frequency dependence (20 vs 40 MHz) | 06_lower_frequency_20MHz.png |
| Scan uniformity (angular) | 07_angular_uniformity.png |

## Validation methodology (literature)

Ultrasound simulation models are typically validated by:

1. **Geometric accuracy** – Compare reflector positions (depth, lateral/angular position) in the image to known phantom geometry or to reference solutions. Tissue-mimicking phantoms with targets of known size/position are standard (e.g. gel wax, IEC-style phantoms; distance/diameter measurements).

2. **Attenuation with depth** – Deeper echoes should not be brighter than shallower ones for the same reflectivity; intensity decay with depth is routinely checked in phantom and simulation studies.

3. **Resolution (PSF)** – Point or line targets yield a point/line spread function; axial and lateral resolution (e.g. FWHM) are compared to measurements or to reference simulators (Field II, FOCUS, Rayleigh–Sommerfeld).

4. **Contrast** – Regions with different acoustic properties should show plausible contrast in the image (e.g. hypoechoic spheres, layer boundaries); contrast-to-noise and similar metrics are used in IEC and QA standards.

5. **Scan uniformity** – Sensitivity should not exhibit systematic bands aligned with the scan pattern; uniformity over the field is a standard QA check.

6. **Reproducibility / consistency** – Same phantom at different positions or frames should give consistent geometry and no position-dependent artifacts.

7. **Reference comparisons** – Where available, comparison to analytical solutions (e.g. Rayleigh–Sommerfeld), or to established simulators (Field II, FOCUS), or to experimental phantom data provides a strong validation.

---

## Scenario: Geometric accuracy: single boundary (centered)

**Validation type:** Geometric accuracy

**Purpose:** Known geometry: circular boundary at fixed radius. Validates that reflector depth and angular position are correctly rendered (cf. phantom-based geometric accuracy in IEC/QA practice and simulation–phantom comparison).

**Expected if the simulation is accurate:**

Single bright interface at constant depth (~4 mm) over 360°. Weakly scattering interior (dark). Depth and angular symmetry match the phantom geometry; no systematic depth error or angular dropout.

**What to verify in the output:**

Ring at ~4 mm depth; symmetric in angle; dark interior; no axial streak artifacts or wrong depth scale.

**Generated image:** `01_centered_single_wall.png`

---

## Scenario: Attenuation with depth & contrast (two-layer)

**Validation type:** Attenuation; contrast

**Purpose:** Two interfaces at known depths test depth-dependent attenuation (deeper echoes should not be brighter than shallower ones) and contrast between layers (cf. contrast resolution and TMM phantoms in validation literature).

**Expected if the simulation is accurate:**

Two concentric bright interfaces at ~3.5 mm and ~4 mm. The deeper interface should be no brighter than the shallower one (attenuation with depth). Interior remains dark.

**What to verify in the output:**

Two distinct rings at correct depths; outer ring not brighter than inner; continuous in angle; plausible intensity ordering with depth.

**Generated image:** `02_centered_thick_wall.png`

---

## Scenario: Geometric accuracy: known offset (eccentric)

**Validation type:** Geometric accuracy

**Purpose:** Transducer offset from symmetry axis: depth-of-interface must vary with angle in a predictable way (geometry). Validates that ray/geometry and coordinate mapping are correct (cf. geometric accuracy in phantom validation).

**Expected if the simulation is accurate:**

Interface depth varies with angle: minimum depth where probe is nearest the boundary (~2.8 mm), maximum on the opposite side (~5.2 mm). Closed ring; no artificial gaps or dropouts.

**What to verify in the output:**

Depth varies sinusoidally with angle; min ~2.8 mm, max ~5.2 mm; continuous interface; no spurious dropout or wrong depth scale.

**Generated image:** `03_eccentric_probe.png`

---

## Scenario: Resolution & geometry: point reflectors

**Validation type:** Resolution (PSF); geometric accuracy

**Purpose:** Discrete point-like reflectors at known positions: validate geometric placement (angle, depth) and effective resolution (spread of the echo). Standard in phantom validation (wire targets, point spread, IEC-style resolution tests).

**Expected if the simulation is accurate:**

Bright spots at known (angle, depth) corresponding to reflector positions (e.g. ~(1.5, 2) and ~(-2, 3) mm in imaging plane). Ring from boundary at ~4 mm. Limited spread of each spot indicates reasonable axial/lateral resolution.

**What to verify in the output:**

Reflector positions match known geometry; no ghost echoes or wrong depths; point-like appearance (not excessively blurred) indicates adequate resolution.

**Generated image:** `04_point_reflectors.png`

---

## Scenario: Temporal/positional consistency (pullback)

**Validation type:** Reproducibility; geometric consistency

**Purpose:** Same phantom at different probe positions: cross-section geometry should remain consistent (reproducibility and absence of position-dependent artifacts). Common in multi-frame and 3D validation.

**Expected if the simulation is accurate:**

Each frame shows the same interface depth (~4 mm) and shape. No systematic drift or new artifacts at specific z; only end effects if phantom is finite.

**What to verify in the output:**

Interface depth and shape consistent across frames; no z-dependent artifacts or spurious bands.

**Generated images:**
- `05_pullback_frame_00_z-1.00.png`
- `05_pullback_frame_01_z-0.50.png`
- `05_pullback_frame_02_z0.00.png`
- `05_pullback_frame_03_z0.50.png`
- `05_pullback_frame_04_z1.00.png`

---

## Scenario: Frequency dependence (20 vs 40 MHz)

**Validation type:** Physical consistency; resolution vs frequency

**Purpose:** Same geometry at different center frequency: depth should be unchanged (geometry); texture/resolution may coarsen at lower frequency. Validates that frequency is applied consistently (attenuation, PSF) without breaking geometry.

**Expected if the simulation is accurate:**

Interface at same depth (~4 mm) as baseline. Coarser speckle at 20 MHz is plausible; no spurious depth shift or loss of interface.

**What to verify in the output:**

Depth unchanged vs scenario 1; optionally coarser texture at 20 MHz; no unexpected artifacts or depth error.

**Generated image:** `06_lower_frequency_20MHz.png`

---

## Scenario: Scan uniformity (angular)

**Validation type:** Image uniformity; artifact check

**Purpose:** Uniform phantom: sensitivity should not show systematic bands aligned with the scan pattern (ray alignment, mesh aliasing). Standard uniformity check in QA and simulation validation.

**Expected if the simulation is accurate:**

Uniform response around 360° aside from real geometry. No fixed periodic bright/dark bands at scan angles (e.g. 0°, 90°) from sampling or alignment.

**What to verify in the output:**

No periodic bands aligned with angular sampling; ring intensity smooth around circumference aside from speckle.

**Generated image:** `07_angular_uniformity.png`

---

## Summary

Use this report with the generated images to check geometric accuracy, depth-dependent attenuation, resolution/contrast, and scan uniformity. For quantitative validation, compare to phantom data, known scatterer positions (e.g. scenario 4), or to reference simulators where available.

## References (validation methodology)

- Phantom-based geometric accuracy and resolution: tissue-mimicking phantoms, target dimensions vs. imaging (e.g. CT vs. US); IEC TS 62791, 62736 (sphere phantoms, contrast, resolution).
- Reference solutions: Field II, FOCUS; comparison to Rayleigh–Sommerfeld integral for pressure/phase; FNM as reference (e.g. FOCUS validation pages).
- Speckle and PSF: first/second-order statistics; PSF convolution for image formation; FWHM for resolution (e.g. computer speckle models, image-based PSF estimation).
- Benchmark problems: standardized benchmarks for intercomparison (e.g. transcranial ultrasound benchmark: focus position/size, field metrics).
- Ray-based validation: Simsonic and similar ray-based tools validated against experimental A-scans/B-scans; UltraRay/UltraScatter for full-path and scattering validation.
