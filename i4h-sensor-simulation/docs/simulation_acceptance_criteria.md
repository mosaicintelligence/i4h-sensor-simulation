# IVUS Probe Simulation — Acceptance Criteria

**Status:** Porcine-lab gate.
**Owner:** Jocelyn Barker
**Scope:** Defines when the IVUS probe simulator is "good enough" to use as
training data for the porcine-lab deep-learning navigation model. 

---

## 1. Purpose and scope

### 1.1 What the simulator is for

The IVUS probe simulator exists to produce a large, parametrically diverse
dataset of synthetic IVUS frames with **known ground truth** (lumen
contour, probe position relative to lumen, vessel geometry, …) suitable as
training data for a deep-learning navigation model. There is no
probe-matched real training data available at this time, so the simulator
is the only source of probe-matched training data the DL team has.

The probe under test ("the probe") is the **Visions PV .035** intravascular
ultrasound catheter, a 10 MHz peripheral-vascular IVUS device. All tests
below assume the simulator is configured to model this probe; if a
different probe is to be qualified, the reference data inventory in §2 has
to be re-collected for that probe and the same tests re-run.

Acceptance does **not** assume any particular DL model architecture. The
working comparator for Tier 3 evaluation is **ACE-Net**
([Farola Barata et al. 2025](https://link.springer.com/article/10.1007/s11517-025-03323-z),
[github.com/bfarolabarata/ace-net](https://github.com/bfarolabarata/ace-net)).

### 1.2 What it is NOT for

- **Not a replacement for clinical validation.** Clinical and animal-test
  data programs proceed in parallel and remain the source of truth for
  model performance.
- **Not a high-fidelity acoustic simulator.** Ray-traced approximation with
  calibrated PSF, TGC, log compression, scatter, and ring-down. Full-wave
  propagation, transducer impulse response, and reverberation are not
  modeled.
- **Not a substitute for diversity in vessel anatomy.** Vessel geometry
  generation is a separate model with its own acceptance criteria; this
  document does not gate vessel-anatomy diversity.

---

## 2. Reference data inventory

The acceptance tests rely on three categories of reference data. Any
simulator under evaluation must be testable against the same references;
if the reference data changes, the gates have to be re-run.

### 2.1 Bench measurements of the probe

A characterisation of the probe's physical and signal-processing
parameters acquired on a known phantom in a controlled bench setting.

The minimum bench captures used for the gates below are:

| Capture | Phantom | Operating conditions | What it measures |
|---|---|---|---|
| Wire-phantom acquisition | 5 columns of 36 AWG copper magnet wire (⌀ 127 µm) at radii r ∈ {5, 10, 15, 20, 25} mm in a water bath, probe at the center | Gain slider 54, imaging diameter 60 mm (0.12 mm/pixel display pitch), ≥ 30 frames | Axial PSF FWHM, lateral PSF FWHM, log compression mapping, ring-down envelope |
| Anechoic-region capture | Same water tank; sample pixels far from any reflector | Same gain | Noise floor σ |
| TGC capture | Operator-set TGC schedule, water bath | Same gain | TGC dB-vs-depth control points |

If these captures do not yet exist for the probe under test, Tier 1 cannot
be evaluated and the gate is blocked until they do.

### 2.2 Reference acquisitions for sim-vs-real comparison

The Tier 2 metrics decompose into three physical questions — *how does the
probe represent the lumen medium, the wall material (including the
interfaces at its inner and outer surfaces), and bulk tissue beyond the
wall* — plus a whole-frame domain-similarity check (Tier 3). Each question
is best answered by the **simplest phantom that isolates it**. Most Tier 2
rows therefore do **not** require a vessel pullback or per-frame
annotations; only Tier 3 needs a vessel-with-surroundings acquisition.

The reference acquisitions below are the minimum set the lab must collect
for sim-vs-real comparison. Each is acquired with the probe at the
**reference operating point** (gain, imaging diameter, TGC) defined in the
calibration sheet (§A.1.1). T2-E1–T2-E3 use **known phantom geometry** or
post-acquisition image segmentation for ROI definition and do not require
per-frame annotation. T2-E4 Phase A is the only acquisition with
annotated lumen and intima contours; T2-E4 Phase B is unannotated and
used only as the cross-probe FID anchor.

| ID | Phantom | What it characterises | Annotations | Feeds |
|---|---|---|---|---|
| **T2-E1** | Uniform lumen-medium tank (Ramnarine or CIRS BMF) — probe immersed, no scatterer at depth | System mapping (TGC, log compression, scatter floor) on the **lumen medium** | None (depth band by phantom geometry) | A.2.3 lumen rows: histogram, Rayleigh σ, lumen-mean vs depth |
| **T2-E2** | Wall-material **tube** in lumen medium (custom Syndaver vessel material is primary; butcher artery is the fallback) | System mapping, texture, and soft-tissue interface reflectivity (leading edge at lumen↔wall, trailing edge at wall↔lumen) on the wall material; wall round-trip attenuation derived from the inner-vs-outer-wall amplitude ratio | None (wall depth band, catheter offset, and near-normal-incidence azimuth all recovered post-acquisition by image segmentation) | A.2.3 wall rows: histogram, texture, leading/trailing-edge palette, edge axial spread |
| **T2-E3** | Bulk-tissue slab away from any vessel (Syndaver muscular pad bulk imaged away from the vessel, or butcher skeletal muscle) | System mapping and texture on **bulk tissue** at far depth; bulk-tissue acoustic impedance from the slab front-face reflection | None | A.2.3 bulk-tissue rows: histogram, texture; and the through-wall consistency check below |
| **T2-E4** | Vessel-with-surroundings pullback through a Syndaver muscular vessel pad with a BMF-filled lumen, run in **two phases** on the **same mounted phantom**: **Phase A** with the Visions PV .035 catheter (mandatory) and **Phase B** with a cross-probe replicate (Boston Scientific IVUS or a different-size Volcano IVUS catheter) on the same vessel section (optional but strongly recommended). | Whole-frame deployment-shaped scenes for embedding similarity | **Lumen + intima contours** per frame on the Tier 3 split of Phase A. Phase B is not annotated. | A.3 Tier 3 (FID + inference parity); A.2.3 visual review; Phase B feeds the primary Tier 3 cross-probe anchor (§A.3.3) |

Three previously-considered experiments (cadaver wall T2-E5, saline-pair
T2-E6, soft-interface slab T2-E7, stacked-layer phantom T2-E8) have been
**dropped** from the inventory: the metrics they were targeted at are
either redundant (T2-E6 with the Tier 1 anechoic capture in §2.1; T2-E7
with the inner/outer-wall echoes in T2-E2; T2-E8 with the analysis-side
through-wall consistency check below) or unattainable in this lab
(T2-E5 cadaver wall material is not available).

**Through-wall consistency check (no extra capture).** The original
T2-E8 metrics — through-wall attenuation and post-interface bulk-tissue
speckle — are computed analytically from data already collected in
T2-E2 + T2-E3 + T2-E4 Phase A: take the wall round-trip attenuation
derived from T2-E2 (inner-vs-outer-wall amplitude ratio), apply it to
the T2-E3 free-fluid bulk-tissue palette and texture statistics, and
compare to the T2-E4 Phase A perivascular-ROI palette and texture
statistics measured at the matching radial depth and wall thickness.
Pass criteria are listed in §A.2.3.

Each acquisition records ≥ 60 frames at the reference operating point
(more for T2-E4; see §A.3.2). The expected output format for every
acquisition is **DICOM**, matching the existing
[`P_035_PointScatter/`](../../P_035_PointScatter/) dataset (raw RF or
vendor polar `.npy` exports are welcome in addition to DICOM but are not
required). For each session, the lab records: substrate identity and
batch ID, fluid composition and temperature, mounting geometry with
photos, console settings, frame count and motion type (static, manual
pullback, or motorised pullback with velocity), and any deviations from
the protocol. T2-E4 Phase A's frame budget is governed by the FID
sample-size requirement in §B.5 (≥ 1000 frames after the Tier 2 / Tier 3
split, i.e. ≥ 2200 frames captured before quality filtering).

#### 2.2.1 Tier 2 / Tier 3 split for T2-E4

Only **T2-E4 Phase A** is split into two non-overlapping halves before
any acceptance test is run, and the splits must not be mixed across
tests:

| Split | Approximate share | Used in |
|---|---|---|
| Tier 2 visual-review reference | First ~50 % of Phase A | A.2.3 visual-review row only (T2-E1–T2-E3 satisfy the rest of Tier 2) |
| Tier 3 reference | Remaining ~50 % of Phase A | A.3 encoder feature-distance and inference-parity computations against the Visions PV .035 reference |

**T2-E4 Phase B** (cross-probe replicate, if collected) is **not** split.
The entire clean Phase B frame set is used as a single distribution —
the primary Tier 3 cross-probe anchor in §A.3.3.

T2-E1, T2-E2, and T2-E3 are not split; the entire acquisition is used
in the metric(s) it feeds.

### 2.3 Cross-distribution anchors for Tier 3

Tier 3 requires a "different probe" reference so that "sim-vs-real"
distance can be calibrated against "sim-vs-different-probe" distance.
None of these references are used for training, fidelity matching, or
to set Tier 4 thresholds.

| Source | Role | Notes |
|---|---|---|
| **T2-E4 Phase B** (cross-probe replicate — Boston Scientific IVUS or different-size Volcano IVUS catheter on the **same** Syndaver vessel pad and **same** vessel section as Phase A) | **Primary** Tier 3 cross-probe anchor | Same phantom, same anatomy, only the probe differs — cleanest possible isolation of "different probe". Collected per Phase B of the T2-E4 protocol. |
| [Roboflow IVUS-2](https://universe.roboflow.com/computer-vision-buu/ivus-2) | **Secondary** Tier 3 cross-probe anchor (sanity check on more diverse anatomy) | 863 IVUS images (different probe and different anatomy). The labelled subset is reserved for the model team's separate AC and **must not** be used in any sim-acceptance computation. |
| [ACE-Net pretrained weights](https://seafile.unistra.fr/d/0160d5182a1941c68e5a/) | Tier 3 encoder (one of several allowable choices, see §A.3.1) | Not a reference dataset; a model checkpoint. |

If T2-E4 Phase B is not collected, only the secondary (Roboflow-based)
anchor is available; the Tier 3 inequality in §A.3.3 falls back to that
anchor with the caveat that "different probe" is confounded with
"different anatomy".

---

## Section A — Porcine-lab gate

The porcine lab needs a working DL model packaged for delivery. The DL
team's separate acceptance document defines what "working" means
quantitatively (mirrored in §A.4). This document defines what the simulator
must deliver to make those targets achievable.

### A.1 Tier 1 — Physical fidelity (sim reproduces the bench)

**Question:** when the simulator is run on a phantom that mirrors the bench
wire-phantom acquisition (§2.1), does it produce output whose extracted
parameters match the bench-measured parameters of the probe?

This is the most rigorous form of "sim matches calibration": same input
geometry, same measurement formulae, compare extracted numbers.

#### A.1.1 What "calibration sheet" means

The **calibration sheet** is the set of bench-measured values for the probe
that the simulator's runtime configuration is derived from. The canonical
calibration sheet for the probe under test is

- [`instrument-calibration/p035_visions/volcano_s5i.yaml`](../../instrument-calibration/p035_visions/volcano_s5i.yaml) — the canonical YAML config consumed by the simulator;
- [`instrument-calibration/p035_visions/parameter_sheet.csv`](../../instrument-calibration/p035_visions/parameter_sheet.csv) — per-parameter provenance (which bench experiment produced each value, with uncertainty);
- [`instrument-calibration/p035_visions/calibration_delta.md`](../../instrument-calibration/p035_visions/calibration_delta.md) — current calibration status and outstanding bench requests.

To support the acceptance tests below, the calibration sheet must contain
at minimum, for the probe under test and at the reference operating point:

| Group | Parameter | Form |
|---|---|---|
| Probe physical | Centre frequency | scalar (MHz) |
| Probe physical | Pulse duration | scalar (cycles) |
| Probe physical | Effective element radius | scalar (mm) |
| Probe physical | Focal length | scalar (mm) |
| Probe physical | Speed of sound assumption | scalar (mm/µs) |
| TGC | Time-gain control schedule | list of (depth_cm, gain_dB) control points |
| Log compression | Log multiplier | scalar (palette per log10 of envelope amplitude) |
| Log compression | Log floor | scalar (envelope amplitude) |
| Log compression | Dynamic range | scalar (dB) |
| Log compression | Gain-slider mapping | scalar (dB per slider step) or table |
| Noise | Noise floor σ at reference gain | scalar (RF amplitude units) |
| Ring-down | Peak amplitude at reference gain | scalar (RF amplitude units) |
| Ring-down | Extent | scalar (mm), defined as the radial distance at which the ring-down excess drops below 5 % of peak |
| Ring-down | Decay shape | string ("exponential" | "hanning" | "measured") |
| Ring-down | Measured A-line template | vector (palette units vs radial sample) at the reference gain, if decay = "measured" |
| Ring-down | Acoustic-reference subtraction state | boolean (whether the device subtracts an internal reference) |

Each parameter must be paired with a citation to the bench experiment
that produced it (capture name + measurement protocol + uncertainty band).

#### A.1.2 Test protocol

1. **Construct the bench-phantom geometry in the simulator.** A scene
   matching §2.1's wire-phantom acquisition: 5 wire columns of 36 AWG
   copper magnet wire (⌀ 127 µm, geometric-scattering regime at 10 MHz)
   at radii r ∈ {5, 10, 15, 20, 25} mm in a water bath, probe at the
   centre of the columns, gain slider 54, imaging diameter 60 mm
   (0.12 mm/pixel display pitch). Render ≥ 30 frames.
2. **Construct an anechoic-region scene:** uniform-medium water bath, no
   scatterers, large enough to sample ≥ 10⁴ pixels far from any boundary.
   Render ≥ 30 frames at the same gain.
3. **Construct a flat-reflector scene:** a single high-impedance plane at
   a known depth (e.g. 5 mm), rendered at ≥ 4 amplitudes spanning at least
   3 decades of envelope amplitude.
4. **Compute the parameters defined in §A.1.3 from the simulator output**
   using the formulae specified there.
5. **Compare each computed value to the corresponding value in the
   calibration sheet** within the tolerance specified.

The same measurement formulae used to compute the values listed in the
calibration sheet (§A.1.1) from the original bench captures are used to
compute them from the simulator output. This is the critical property: the
gate is a direct numerical comparison between the simulator output and the
calibration sheet, computed by identical analysis.

#### A.1.3 Quantitative pass criteria

In what follows, "wire i" denotes the wire at radius r_i ∈ {5, 10, 15, 20,
25} mm. "Reference gain" is gain slider 54 throughout.

**Configuration round-trip (clerical):**

| Test | Method | Pass criterion |
|---|---|---|
| Calibration-sheet round-trip | For every parameter in §A.1.1, compare the value in the simulator's runtime configuration to the value in the calibration sheet. | All values equal within the documented uncertainty band of the calibration sheet entry. |
| Configuration self-consistency | Load the simulator's runtime configuration; run a no-op simulation; export the effective parameter set; diff against the loaded configuration. | All values round-trip identically. |

**Axial PSF (per radius):**

- *Method:* On the rendered wire-phantom frames, isolate the patch
  surrounding wire i. Compute the axial profile (intensity vs radial
  sample) through the wire's centre. Fit/measure the −6 dB full width at
  half maximum (FWHM) in mm.
- *Pass:* For every wire i where the bench measurement is unsaturated
  (peak palette < 240 in the bench data), |FWHM_sim(r_i) − FWHM_bench(r_i)|
  ≤ 1 display pixel (≈ 0.12 mm at the bench display pitch).

**Lateral PSF (per radius):**

- *Method:* On the same patches, compute the lateral profile (intensity
  vs angular sample) at the wire's depth. Measure the −6 dB FWHM. Convert
  angular FWHM to mm at the wire's depth via FWHM_mm = FWHM_rad · r_i.
- *Pass:* For every wire i where the bench measurement is unsaturated,
  |FWHM_sim(r_i) − FWHM_bench(r_i)| / FWHM_bench(r_i) ≤ 20 %.
- *Additional pass:* The lateral FWHM as a function of radius must have
  its minimum within the focal window (12–20 mm from the probe centre)
  in the simulator output, matching the sign of the depth trend in the
  bench data.

**Ring-down (single A-line, mean over angles):**

- *Method:* On the wire-phantom frames, compute the mean A-line as the
  angular average of the polar B-mode at the reference gain. Truncate to
  the inner 0 ≤ r ≤ extent + 1 mm.
- *Pass — peak amplitude:* The maximum palette value of the simulator's
  mean A-line in r ≤ 3 mm is within ±10 % of the calibration sheet's
  ring-down peak palette at the reference gain.
- *Pass — envelope shape:* The element-wise RMS difference between the
  simulator's mean A-line and the calibration sheet's ring-down template
  over 0 ≤ r ≤ 3 mm is ≤ 5 palette units.
- *Pass — extent:* The radial distance at which the simulator's mean
  A-line excess (over the local speckle floor) drops below 5 % of its
  peak excess is within ±0.3 mm of the calibration sheet's ring-down
  extent.

**Noise floor σ:**

- *Method:* On the anechoic-region scene, sample ≥ 10⁴ pixels well away
  from any boundary or ring-down zone. Compute the standard deviation in
  RF amplitude units (or palette units, divided by log_multiplier and
  un-log-compressed if measured in palette).
- *Pass:* |σ_sim − σ_bench| / σ_bench ≤ 20 % at the reference gain.

**Log compression mapping:**

- *Method:* On the flat-reflector scenes at envelope amplitude A, read
  the displayed palette value at the reflector depth. Predict the
  expected palette value as
  `pixel_predicted = log_multiplier · log10(A / log_floor)`,
  using the calibration sheet's log_multiplier and log_floor.
- *Pass:* |pixel_sim − pixel_predicted| ≤ 3 palette units at every test
  amplitude (recommend ≥ 4 amplitudes spanning ≥ 3 decades).

**TGC schedule:**

- *Method:* On a uniform-scattering scene (or via whatever diagnostic
  output the simulator exposes for its effective TGC curve), interpolate
  the calibration sheet's TGC control points piecewise-linearly to the
  same depth grid the simulator uses. Compare RMS-difference in dB
  across 0 ≤ r ≤ 3 cm.
- *Pass:* RMS difference ≤ 0.1 dB across 0 ≤ r ≤ 3 cm.

The Tier 1 gate passes when **every test above passes**. Failures point
at which calibration-sheet group is mismatched; the simulator vendor or
developer is responsible for resolving discrepancies.

#### A.1.4 Phantoms required

The simulator must be able to construct three scene types:

- **Wire phantom** matching §A.1.2 step 1.
- **Anechoic-region scene** matching §A.1.2 step 2.
- **Flat-reflector scene** matching §A.1.2 step 3.

If the simulator cannot construct any of these, that is itself a Tier 1
blocker.

### A.2 Tier 2 — Image-level fidelity in polar domain

**Question:** does the simulator reproduce, in polar coordinates, the
probe's representation of (a) the lumen medium, (b) the wall material,
(c) bulk tissue beyond the wall, and (d) the soft-tissue interfaces
between them?

Tier 2 is decomposed into three per-substrate experiments
T2-E1 (lumen medium), T2-E2 (wall material — including soft-tissue
interfaces at the inner and outer wall surfaces), and T2-E3 (bulk
tissue) plus an analysis-side through-wall consistency check that
combines T2-E2 + T2-E3 + T2-E4 Phase A data (§2.2). Each row of A.2.3
is fed by the experiment whose phantom isolates the physical question
that row asks. No DL model is involved.

The ring-down profile on the inner ~3 mm of an A-line is a probe-level
parameter; it is gated by Tier 1 (§A.1.3) and is not duplicated as a
Tier 2 row.

#### A.2.1 Substrate parameter sheet

The simulator must consume a per-substrate parameter set in addition to
the probe-level calibration sheet (§A.1.1). To support the experiments in
§2.2, the substrate parameter sheet contains, for each substrate the
simulator is asked to render:

| Group | Parameter | Form | Source |
|---|---|---|---|
| Lumen medium | Identity | string (e.g. `bmf-ramnarine-batch-<YYYYMMDD>`, `bmf-cirs-046-lot-<###>`, `saline-09pct`) | T2-E1 metadata |
| Lumen medium | Speed of sound | scalar (mm/µs) | Recipe / commercial source |
| Lumen medium | Acoustic attenuation | scalar (dB/cm) at 10 MHz, or coefficient + frequency exponent | T2-E1 substitution measurement (Appendix A of the Tier 2 protocol) or recipe / commercial source |
| Lumen medium | Backscatter coefficient (or scatter density) | scalar in the simulator's native scatter-density units | Recipe / commercial source, or fit from T2-E1 Rayleigh σ |
| Wall material | Identity | string (e.g. `syndaver-customvessel-<part>-lot<###>`, `butcher-<species>-<vessel>-batch-<YYYYMMDD>`) | T2-E2 metadata |
| Wall material | Speed of sound | scalar (mm/µs) | Recipe / vendor / literature for the material at 10 MHz |
| Wall material | Acoustic attenuation | scalar (dB/cm) at 10 MHz | T2-E2 inner-vs-outer-wall amplitude ratio (round-trip attenuation through the segmented wall thickness) |
| Wall material | Backscatter / scatter density | scalar (sim units) | Fit from T2-E2 wall histogram and texture |
| Wall material | Acoustic impedance | scalar (Mrayl) | T2-E2 leading-edge palette inverted through the predicted-reflection formula, with the lumen-medium impedance from this sheet |
| Bulk tissue | Identity | string (e.g. `syndaver-pad-bulk-lot<###>`, `butcher-<species>-skeletal-batch-<YYYYMMDD>`, `tmm-madsen-recipe-<short>-batch-<YYYYMMDD>`) | T2-E3 metadata |
| Bulk tissue | Speed of sound | scalar (mm/µs) | Recipe / vendor / literature |
| Bulk tissue | Acoustic attenuation | scalar (dB/cm) at 10 MHz | T2-E3 substitution measurement on a thin replicate, or literature |
| Bulk tissue | Backscatter / scatter density | scalar (sim units) | Fit from T2-E3 bulk-tissue histogram and texture |
| Bulk tissue | Acoustic impedance | scalar (Mrayl) | T2-E3 front-face leading-edge palette inverted through the predicted-reflection formula |

Each parameter must be paired with a citation to the bench experiment
(or recipe / batch ID + measurement protocol) that produced it. For
substrates whose properties are taken from published literature rather
than measured in-house, the literature source must be cited and the
uncertainty band recorded.

#### A.2.2 Test protocol

For each Tier 2 experiment **E**:

1. Acquire the real reference per §2.2 at the reference operating point.
   ROI extraction follows §B.1, which defines the ROI for T2-E1, T2-E2,
   and T2-E3 by phantom geometry plus post-acquisition image
   segmentation, and for T2-E4 by per-frame lumen / intima contours.
2. Generate the matching sim frames. The simulator must be configured to
   render the same substrate(s) as the experiment, using the substrate
   parameter sheet (§A.2.1) and the calibration sheet (§A.1.1). Sample
   sizes per experiment:
   - T2-E1, T2-E2, T2-E3: ≥ 60 frames real and ≥ 60 frames sim each.
   - T2-E4 Phase A (mandatory): ≥ 1000 frames real on the Tier 3 split
     (visual review reuses the Tier 2 split) and ≥ 1000 frames sim with
     vessel geometries distributionally matched to the Phase A
     reference (vessel diameter range, eccentricity range,
     catheter-offset range).
   - T2-E4 Phase B (cross-probe replicate, if collected): ≥ 1000
     frames from the second probe on the same vessel section as
     Phase A. Phase B contributes the primary Tier 3 cross-distribution
     anchor (§A.3.3); no matching sim frames are rendered for Phase B.
3. Apply the same ROI extraction and the same measurement formulae to
   sim and real. Compute the metrics in §A.2.3 from each experiment's
   data.

#### A.2.3 Quantitative pass criteria

| Experiment | Test | Real source | Sim source | Pass criterion |
|---|---|---|---|---|
| **T2-E1** | Lumen-medium palette histogram | Pixels in lumen-medium ROI (§B.1), all T2-E1 frames | Pixels in equivalent sim ROI, all sim frames | Two-sample KS statistic D ≤ 0.15 |
| **T2-E1** | Speckle envelope (Rayleigh fit) in lumen medium | Rayleigh σ from un-log-compressed lumen-medium pixels (§B.3) | Same | \|σ_sim − σ_real\| / σ_real ≤ 25 % |
| **T2-E1** | Lumen-medium mean palette vs depth | Mean palette in 1 mm depth bins inside lumen-medium ROI | Same | RMS over depth ≤ 15 palette units |
| **T2-E2** | Wall palette histogram | Pixels in wall ROI (§B.1), all T2-E2 frames | Equivalent sim ROI | KS statistic D ≤ 0.15 |
| **T2-E2** | Wall texture — lateral autocorrelation length | Mean across depth bins inside wall ROI (§B.4) | Same | Within ±30 % of real |
| **T2-E2** | Wall texture — GLCM statistics | Per-patch GLCM contrast, homogeneity, energy, correlation (§B.4.2) | Same | KS distance D ≤ 0.20 on each statistic distribution |
| **T2-E2** | Soft-interface leading-edge palette (lumen↔wall) | Peak palette at the segmented inner-wall depth in the ±15° near-normal-incidence window, mean over angles and frames | Same | Within ±10 % of value predicted from impedance contrast and the calibration log mapping |
| **T2-E2** | Soft-interface trailing-edge palette (wall↔lumen) | Peak palette at the segmented outer-wall depth in the same window | Same | Within ±10 % of impedance-predicted value |
| **T2-E2** | Soft-interface edge axial spread | Axial −6 dB FWHM of the leading-edge bright band on the A-line trace | Same | Within ±20 % of the Tier 1 axial PSF FWHM at the same depth |
| **T2-E3** | Bulk-tissue palette histogram | Pixels in bulk-tissue ROI (§B.1), all T2-E3 frames | Equivalent sim ROI | KS statistic D ≤ 0.20 (looser; further from probe and less well controlled by calibration) |
| **T2-E3** | Bulk-tissue texture — lateral autocorrelation length | Mean across depth bins inside bulk-tissue ROI | Same | Within ±30 % of real |
| **T2-E3** | Bulk-tissue texture — GLCM statistics | Same as T2-E2 GLCM, computed on bulk-tissue patches | Same | KS distance D ≤ 0.20 on each statistic distribution |
| **T2-E2 + T2-E3 + T2-E4 Phase A** | Through-wall consistency check | Apply the wall round-trip attenuation derived from T2-E2 (inner-vs-outer-wall ratio) to the T2-E3 free-fluid bulk-tissue palette and texture statistics, then compare to the T2-E4 Phase A perivascular ROI palette and texture statistics at the matching radial depth and segmented wall thickness | Same | Predicted vs measured perivascular palette within ±2 dB; perivascular speckle statistics (autocorrelation, GLCM) within ±30 % of T2-E3 after attenuation correction. |
| **T2-E4 Phase A** | Visual review (informal) | 20 frames (10 sim, 10 real) drawn from the Phase A Tier 2 visual-review split (§2.2.1), shuffled; one or more reviewers flag the synthetic ones | — | Logged, not gated. ≤ 75 % accuracy is a positive signal; ≥ 90 % means an obvious tell exists and should be investigated. |

The Tier 2 gate passes when every quantitative row above passes. Failure
modes:

- KS > 0.30 in a palette histogram → log-compression mapping or TGC
  issue, or substrate parameter mismatch (re-check §A.2.1 entries).
- Rayleigh σ off > 50 % → scatter-scale issue in the simulator's
  lumen-medium configuration.
- Wall or bulk-tissue texture KS > 0.30 → scatter-density or PSF
  mismatch at the relevant depth.
- T2-E2 leading-edge palette off by > 30 % of the predicted value →
  impedance-contrast / specular-reflection model mismatch.
- Through-wall consistency check off by > 4 dB → wall-attenuation entry
  in §A.2.1 is wrong, or simulator does not apply per-substrate
  attenuation.

### A.3 Tier 3 — Feature-level fidelity (encoder distance)

**Question:** do sim and real frames live in the same feature space as far
as a domain-relevant encoder is concerned?

Tier 3 is a **cheap predictor for Tier 4**: it requires no model training,
only inference. If sim and real are far apart in encoder feature space,
Tier 4 (downstream task) is unlikely to pass and the simulator needs work
first.

Tier 3 is the only tier that compares **whole frames**, so it is sensitive
to everything in the field of view — including regions that Tier 2 treats
as separate ROIs and whether outer-field regions contain bulk tissue or
tank fluid. The only acquisition that satisfies Tier 3 is **T2-E4** (vessel
with surroundings; §2.2), which is run in two phases on the same mounted
phantom: **Phase A** with the Visions PV .035 (the Tier 3 reference) and
**Phase B** with a cross-probe replicate (the cross-distribution anchor;
optional but strongly recommended). Naked-vessel acquisitions (T2-E2) do
**not** satisfy Tier 3 because the absence of perivascular tissue is a
distribution shift relative to the simulator's training distribution and
relative to the deployment scene; they remain valid for Tier 2 wall rows
only.

The cross-distribution anchor is **T2-E4 Phase B** (Boston Scientific
IVUS or different-size Volcano IVUS catheter on the same Syndaver vessel
pad — see §2.3) when collected. If Phase B is not available, the
Roboflow IVUS-2 dataset is used as the secondary anchor with the caveat
that it confounds "different probe" with "different anatomy".

#### A.3.1 Encoder choices

The test is run with **at least one** encoder. The DL team picks a
"primary" Tier 3 anchor based on what is selected for the lab; the others
remain available as cross-checks.

| Encoder | Source | Why it is suitable |
|---|---|---|
| ACE-Net backbone | Pretrained checkpoint | A-line based encoder trained on carotid IVUS; native polar-domain input matches the simulator's output format |
| Generic ImageNet ResNet50 | torchvision | Always available, domain-agnostic; useful as a sanity floor for FID computation |
| DL-team-selected porcine-lab encoder | Whatever the team selects | Tightest correspondence to Tier 4 if available |

All inputs to all encoders must be polar-domain at 256×256 with the
encoder's expected normalization. The same preprocessing must be applied
to all distributions under comparison (sim, T2-E4 Phase A, T2-E4 Phase B
if available, Roboflow IVUS-2).

#### A.3.2 Test protocol

1. Run the chosen encoder on every frame in the Tier 3 reference split
   of **T2-E4 Phase A** (§2.2.1). Save the layer-of-interest activations
   (default: the encoder's penultimate feature map; for ACE-Net
   specifically, use the backbone feature map at the output of its
   first downsampling stage).
2. Run the same encoder on ≥ 1000 simulator frames, with the same
   normalization and resolution. The simulator must be configured to
   render scenes whose substrate composition matches T2-E4 Phase A
   (lumen medium, wall material, bulk tissue beyond the wall) using
   the substrate parameter sheet (§A.2.1).
3. Run the same encoder on the cross-distribution anchors (§2.3):
   - Primary: ≥ 1000 frames from **T2-E4 Phase B** (cross-probe
     replicate on the same Syndaver vessel pad and same vessel section
     as Phase A) if collected.
   - Secondary: ≥ 1000 frames from the Roboflow IVUS-2 public dataset.
   Preprocess all distributions to the same polar 256×256 normalization.
   If a source is in Cartesian form, apply the same Cartesian→polar
   conversion the DL team would apply at deployment time.
4. Compute the Fréchet Inception Distance (FID) on the per-frame feature
   vectors (spatially averaged feature maps), per the standard formula.
5. If the encoder has a head with usable outputs on the chosen
   normalization (e.g. a segmenter with a lumen-region output, or
   ACE-Net's presence vector and lumen coordinates), run inference and
   record per-frame outputs.

Implementation details for FID, sample sizes, and bootstrap CIs are in
§B.5.

#### A.3.3 Quantitative pass criteria

| Test | Pass criterion |
|---|---|
| FID(sim, T2-E4 Phase A) using primary encoder, vs the **primary** cross-probe anchor | ≤ FID(sim, T2-E4 Phase B). The simulator must look more like the Visions PV .035 than like a different IVUS probe imaging the **same phantom**. (Applies whenever Phase B has been collected.) |
| FID(sim, T2-E4 Phase A) using primary encoder, vs the **secondary** cross-probe anchor | ≤ FID(sim, Roboflow IVUS-2). Sanity check on more diverse anatomy; passes whenever the primary anchor passes, but logged independently because Roboflow confounds probe and anatomy. Becomes the load-bearing Tier 3 anchor only when no T2-E4 Phase B replicate is available. |
| FID(sim, T2-E4 Phase A) absolute value | Logged; no absolute threshold. Used as a trend metric across iterations. |
| Off-the-shelf inference parity (segmenter heads only): lumen-region Dice on sim vs T2-E4 Phase A | \|Dice_sim − Dice_real\| / Dice_real ≤ 20 % |
| Off-the-shelf inference parity: ACE-Net presence-vector statistics (when ACE-Net is the encoder) | Two-sample KS distance between sim and T2-E4 Phase A presence-probability distributions D ≤ 0.20 |

The Tier 3 gate passes when at least one encoder satisfies all applicable
rows. Failure modes: cross-distribution test fails (sim looks more like
a different probe than like the Visions PV .035 on the same phantom) →
the simulator is generic-IVUS-like but not probe-specific; ring-down or
log-compression are usual culprits. Inference parity fails wildly → the
chosen encoder may be unusable for this probe regardless of sim quality,
in which case Tier 3 is downgraded to FID-only and Tier 4 becomes the
load-bearing gate.

### A.4 Tier 4 — Downstream task fidelity

**Question:** does a deep-learning model trained or fine-tuned with the
simulator's data meet the model-level acceptance targets when evaluated on
real probe data?

Tier 4 is the gate that actually matters. **Training the DL model is out
of scope for this document** — the DL team owns the training pipeline,
model selection, and downstream evaluation. The metrics below mirror the
DL team's separate acceptance document so the simulator team knows what
the simulator data is contributing to.

#### A.4.1 Model-level acceptance metrics (cross-listed)

These are reproduced from the DL team's "Acceptance Criterion Framework
(Animal-Test Development Phase) First Pass." If the DL team's document
changes, the source there is canonical and this list is the mirror.

| Metric | Target | Notes |
|---|---|---|
| Lumen segmentation pass rate | ≥ 95 % of relevant frames pass | A frame passes when HD95 ≤ 2.5 mm |
| Mean Dice (lumen) | ≥ 0.90 overall, ≥ 0.88 in every diversity stratum | — |
| Lumen diameter MAE | ≤ 1.5 mm | Defined by a separate diameter-extraction functional process |
| Pullback-level lumen CSA / volume agreement | ICC ≥ 0.90 | — |
| Absolute lumen CSA error | ≤ 10 % of nominal | Low priority |
| Contact signal F1 | ≥ 0.9 | On ground-truth dataset |
| Catastrophic contour failures | < 1 % of analyzable frames across diversity strata | "Catastrophic" = breaking 2-σ error bounds |
| End-to-end segmentation latency (p95) | ≤ 25 % of B-mode single-plane processing budget (~125 ms given ~500 ms / 12 fps) | No sustained frame backlog |

#### A.4.2 What the simulator must deliver to enable Tier 4

The simulator's deliverables for Tier 4 are:

1. **Sim-data corpus** large enough and diverse enough to support model
   training. The DL team sets the volume and diversity targets; typical
   published numbers for similar tasks are O(10⁴) frames spanning the
   geometry parameter ranges of the porcine-lab evaluation set.
2. **Reproducibility:** the sim corpus must be regeneratable from the
   simulator's runtime configuration plus a documented vessel-geometry
   seed list, so that the DL team can re-train when the configuration or
   vessel generator changes.
3. **Per-frame ground truth** at the geometric precision of the simulator
   (lumen contour, probe position, vessel ID, plaque/calcium presence
   masks if requested) saved in the format the DL team consumes.

The simulator team's responsibility ends at delivering the corpus and the
ground truth. The model team is responsible for choosing the architecture,
training, and reporting whether §A.4.1 targets were met.

#### A.4.3 Failure-mode escalation

If Tier 1–3 all pass but the model team reports a Tier 4 failure (model
trained on sim does not meet §A.4.1 on real probe evaluation data), the
simulator team's investigation order is:

1. Re-run Tier 3 with the encoder the DL team actually used for the lab
   model. A passing Tier 3 with one encoder does not guarantee passing
   Tier 3 with another.
2. Re-check coverage: did the sim corpus span the geometries the DL team's
   evaluation set contains?
3. Re-check Tier 1 and Tier 2 on the failure-mode frames specifically (not
   just on the aggregate).

If after these checks the simulator gates still pass and the model fails,
the gap is in the DL pipeline (architecture, training schedule, label
quality), not in the simulator.

---

## 3. Out-of-scope / known limitations

Things this acceptance framework explicitly does not claim:

- **Full-wave acoustic accuracy.** Reverberation, multiple scattering, and
  transducer-impulse-response effects are not modeled. Tier 2 metrics
  catch the visible consequences; no acoustic-level fidelity claim is
  made.
- **Vessel-anatomy diversity.** Vessel geometry generation is a separate
  model with its own AC. This document gates the *probe* simulation, not
  the anatomy.
- **Real-time deployed safety.** Acceptance is for *training data quality*,
  not for any deployed model's real-time clinical safety. That is a
  separate V&V activity.
- **Patient population coverage.** Tier 3 reference (T2-E4) is a single
  vessel-with-surroundings phantom acquisition; Tier 2 references
  (T2-E1–T2-E3) characterise specific substrates rather than the
  population. Population coverage is a future concern when clinical
  data arrives.
- **Final clinical model targets.** The thresholds here are for
  porcine-lab readiness, not for clinical use.

---

## 4. Re-validation triggers

The gates must be re-run when any of the following changes:

- Probe firmware or hardware revision (the calibration sheet is
  per-instrument).
- Probe pre-processing pipeline changes (new polar conversion, new
  normalization, new bench extraction protocol that produces different
  calibration-sheet values).
- Encoder change in Tier 3 (re-run §A.3 with the new encoder).
- Substrate change: any modification to the lumen-medium recipe, the
  wall-material source, or the bulk-tissue source between sessions
  invalidates the corresponding §A.2.1 substrate parameter sheet entry
  and the Tier 2 rows fed by that experiment must be re-run.
- Vessel-geometry generator change (re-validate Tier 2 / Tier 3 T2-E4
  geometry distributions on the new geometries).
- Updates to the DL team's model-level AC (re-mirror §A.4.1).

---

## Appendix B — Measurement details

### B.1 Tier 2 ROI definitions

The Tier 2 ROI is "the polar region occupied by the substrate under
test." For T2-E1, T2-E2, and T2-E3 it is defined by known phantom
geometry (T2-E1) or by **post-acquisition image segmentation** of
substrate boundaries in the polar B-mode (T2-E2, T2-E3); per-frame
annotation by a human annotator is not required. For T2-E4 Phase A it
is defined by per-frame lumen and intima contours; T2-E4 Phase B has
no ROI-level analysis (only whole-frame FID). On sim frames the
equivalent ROI is computed from the simulator's known ground-truth
geometry.

**T2-E1 — lumen-medium ROI.** Polar pixels (θ, r) with r in the band
`[r_inner, r_outer]`, where `r_inner` is just outside the ring-down
extent (calibration sheet §A.1.1) and `r_outer` is the deepest sample
free of tank-wall or boundary returns. The band must span at least 5 mm
of unperturbed lumen-medium response. No vessel is present in T2-E1.

**T2-E2 — wall + interface ROIs.** All ROIs are derived from the
segmented inner-wall and outer-wall echoes of the wall tube. The lab
provides only the tube's `R_in` and `R_out` from physical callipering;
the catheter offset and near-normal-incidence azimuth are recovered
from the polar image:

- *Geometry recovery.* Segment the inner-wall echo as a function of
  azimuth in each polar frame. The azimuth at which the inner-wall
  depth is minimum gives the near-normal-incidence azimuth `θ₀`; that
  minimum depth is `r_min`. The catheter offset is `d = R_in − r_min`.
  The outer-wall depth at `θ₀` is `r_max = r_min + (R_out − R_in)`.
- **Wall ROI.** Annular polar region between the segmented inner-wall
  and outer-wall echoes, restricted to angles within ±15° of `θ₀`.
- **Leading-edge ROI.** Narrow band `[r_min − w, r_min + w]` at the
  inner-wall echo, restricted to angles within ±15° of `θ₀`, with `w ≥ 1.5×`
  the Tier 1 axial PSF FWHM at `r_min`.
- **Trailing-edge ROI.** Narrow band `[r_max − w, r_max + w]` at the
  outer-wall echo, restricted to the same angular window.

**T2-E3 — bulk-tissue ROI.** Polar pixels (θ, r) with r in the band
`[D + δ, D + T − δ]`, where `D` is the segmented depth of the slab
front face in the polar image, `T` is the slab thickness (≥ 15 mm), and
δ ≥ 0.5 mm is a buffer excluding the slab's front and back interfaces.
`D` is recovered post-acquisition from the front-face echo and does not
need to be physically pre-measured.

**T2-E4 Phase A — vessel-with-surroundings ROIs (per-frame contour).**

- **Lumen ROI:** all polar pixels (θ, r) where r is between the catheter
  surface (the inner edge of the polar image, r = 0 in the catheter
  frame) and the lumen contour at angle θ.
- **Wall ROI:** annular polar region between the lumen contour and the
  intima contour, per angle.
- **Perivascular ROI:** all polar pixels with r ≥ intima contour, capped
  at the simulator's maximum radial range (`t_far` or equivalent).

### B.2 KS test convention

- Histogram bin: 1 palette unit (256 bins covering the displayed range,
  for visualization).
- Two-sample KS test: compute `scipy.stats.ks_2samp` (or equivalent) on
  the unbinned pixel arrays. The 0.15 / 0.20 thresholds in §A.2.3 refer to
  the KS statistic D, not the p-value. (Sample sizes will be large enough
  that p-values are uninformatively small.)

### B.3 Rayleigh fit

In the lumen-medium ROI of T2-E1, the dominant signal is speckle from the
medium and from acoustic-reference residuals (the simulator's analogue
is its calibrated ring-down + scatter floor combined with the
substrate-parameter scatter density of §A.2.1). Fit a Rayleigh
distribution to the ROI pixel envelope in **un-log-compressed RF
amplitude units**:

- If samples are in palette units, invert: `A = log_floor · 10^(palette / log_multiplier)`.
- Fit: `scipy.stats.rayleigh.fit` (or equivalent) on the inverted samples.
- Report σ in RF amplitude units. The ±25 % tolerance in §A.2.3 is on σ.

### B.4 Texture metrics

#### B.4.1 Lateral autocorrelation length

In the wall ROI (T2-E2) and bulk-tissue ROI (T2-E3), compute the lateral
autocorrelation length of polar A-lines at fixed depth bins:

1. For each depth bin r, take the sequence of palette values across all
   angles inside the ROI.
2. Compute the normalised autocorrelation; find the first lag at which it
   crosses 1/e.
3. Convert the lag (in angular bins) to mm at that depth via
   `lag · (2π · r / N_angles)`, where `N_angles` is the number of
   angular samples in the polar image.
4. Report the mean autocorrelation length across all depth bins inside
   the ROI.

The ±30 % tolerance is on this mean.

#### B.4.2 GLCM statistics

Lateral autocorrelation length captures one spatial scale; the GLCM
(grey-level co-occurrence matrix) statistics capture the broader texture
distribution and discriminate scatter-density and PSF mismatches that
share an autocorrelation length. Compute on the same wall and
bulk-tissue ROIs.

1. For each ROI, extract non-overlapping polar patches of
   approximately 32 angular samples × 16 radial samples, all entirely
   inside the ROI. Patches that cross an interface are excluded.
2. Quantise each patch's palette values to 32 grey levels covering the
   palette range used in the simulator's display configuration.
3. For each patch, compute the GLCM with offsets {(1, 0), (0, 1),
   (1, 1), (1, −1)} (lateral, radial, and the two diagonals) at unit
   pixel distance. Average the four GLCMs into a single rotation-invariant
   GLCM per patch.
4. From the averaged per-patch GLCM, extract the four standard Haralick
   statistics: **contrast**, **homogeneity**, **energy**, and
   **correlation**.
5. For each statistic, accumulate the per-patch values into a
   distribution per (sim, real) and per ROI. Compare distributions with
   a two-sample KS test (B.2 convention).

The ±0.20 KS-distance tolerance in §A.2.3 is per statistic.

### B.5 FID computation

- **Activation source:** the encoder's penultimate feature map, spatially
  averaged to a per-frame feature vector. For ACE-Net specifically, use
  the backbone feature map at the output of its first downsampling stage.
- **Sample sizes:** ≥ 1000 frames per distribution.
- **Statistic:** standard FID
  `FID = ||μ₁ − μ₂||² + Tr(Σ₁ + Σ₂ − 2(Σ₁ Σ₂)^(1/2))`
  where (μ_i, Σ_i) are the mean and covariance of the per-frame feature
  vectors for distribution i.
- **Confidence intervals:** bootstrap 95 % CIs over the frames in each
  distribution to make the relative comparisons (sim-vs-real vs
  sim-vs-public) defensible.
- **Implementation:** standard `pytorch_fid` package or equivalent.
