# P_035 calibration — lab notebook & historical follow-ups

> **Current status (as of PR #21, 2026-08-19).** Live shipping values are in
> [`volcano_s5i.yaml`](volcano_s5i.yaml) comments, not in the historical
> sections below: `processing.gain_db: 58.09`,
> `probe.elevational_height_mm: 1.5`, `probe.num_elevational_samples: 8`
> (uncalibrated 2.5D product default; E3 still required for a measured
> FWHM). Checked-in [`tier1_results/`](tier1_results/tier1_results.md) is
> the last **2D** gate — see
> [`tier1_elevational_waiver.md`](tier1_elevational_waiver.md). For live
> truth prefer the YAML + waiver + Tier 1 report; treat everything below
> as a lab notebook / chronology (including mid-doc notes that still cite
> older `gain_db` figures such as 73.92).

> **Update 2026-05-12:** the requested follow-up bench session has landed
> in `ivus_test_0508/`. See **"Bench session received (2026-05-08)"**
> below for the inventory and the parameters that change before the
> re-fit pipeline is re-run. Everything below that section is the
> pre-bench plan and remains accurate as historical context.

## Bench session received (2026-05-08)

Captured at the bench on 2026-05-08, exported as
`ivus_test_0508/CASE0000` … `CASE0009`. The full per-file mapping
(case → experiment → apparatus → diameter → gain → AR-flag) lives in
`ivus_test_0508/manifest.csv` and `ivus_test_0508/README.md`; the data
is also staged into per-experiment subfolders under
`ivus_test_0508/raw/<experiment>/FILE####.dcm` (symlinks; no copies)
so the existing analysis pipeline can run on it without changes.

### Inventory delivered

| Experiment | Cases | Apparatus | Phantom | Diameter | Vids | Gain order |
|---|---|---|---|---:|---:|---|
| **E2 sweep #1** (Capture A) | `CASE0007/F2-9` + `CASE0006` | **ORIGINAL apparatus** | 36 AWG copper magnet wire | 60 mm | 18 | asc 0→68 step 4 |
| **E2 sweep #2** (Capture A) | `CASE0005` + `CASE0004` | **NEW apparatus, widest** | same | 60 mm | 18 | desc 68→0 step 4 |
| **E2 sweep #3** (Capture A) | `CASE0003` + `CASE0002` | **NEW apparatus, narrow** | same | **30 mm** | 18 | asc 0→68 step 4 |
| **E6 pilot** (Capture C) | `CASE0007/F0-1` | n/a — degassed water only | water only | 60 mm | 2 | g=44, AR-off + AR-on |
| **E6 take 2** (Capture C) | `CASE0000` | n/a — degassed water only | water only | 30 + 60 mm | 6 | (g=50/D=30, g=50/D=60, g=40/D=60) × AR-off + AR-on |
| **E6 take 1, partial** | `CASE0008` | n/a — degassed water only | water only | 60 mm | 1 | g=44 AR-off only — superseded by Take 2 |

Dropped: `CASE0001` (verified byte-identical duplicate of `CASE0000` by
SHA1 of pixel data) and `CASE0009` (PatientID `7595` /
PatientName `vbgy v` — onboarding/keyboard-mash entries from earlier
in the day).

### Bonuses vs. the pre-bench plan

1. **Smaller imaging diameter is available than expected.** The plan
   assumed the smallest available D was 35 mm at 0.07 mm/pixel; the
   bench delivered D = 30 mm at **0.06 mm/pixel** — 2× better axial
   sampling than P_035 and finer than the protocol target.
2. **Three full Capture-A sweeps (one per apparatus / diameter combo)
   instead of one.** The intent was a single 18-step sweep at 35 mm;
   we got 18 steps × 3 sweeps = 54 unsaturated and saturated wire-
   phantom captures. This buys us an apparatus cross-check and a
   redundancy across two diameters that the plan didn't ask for.
3. **Capture C delivered as both a single-gain pilot and a 3-(gain,
   diameter)-grid Take 2** (CASE0007/F0-1 and CASE0000), each with
   matched AR-off / AR-on pairs. This is the cleanest AR
   characterisation we've ever had.
4. **Single fixed orientation per apparatus** (per the operator: probe
   was *not* rotated within a sweep, but was rotated between
   apparatus). The orientation-averaging part of the original E2
   protocol is therefore deferred. The operator has offered to
   re-shoot at a single preferred gain at multiple orientations if
   needed once the fits land.

### Findings that propagate back to the existing P_035 analysis

#### AR-on/off flag is `(0029,1007)`, not `(0029,1006)`

The current `volcano_s5i.yaml` and `extract_metadata.py` both label
private tag `(0029,1006)` as the AR-enable flag. The new dataset shows
`(0029,1006) = 1` in every frame regardless of AR mode (capability flag,
not state flag). The actual AR-on/off flag is `(0029,1007)`,
provisionally named `mode_flag` in the existing extractor.

Verified by paired-capture inner-disc intensity: when `(0029,1007)`
flips 0 → 1 the catheter ringdown disc collapses from a bright bright
ring (palette 80-140) to the surrounding speckle floor (palette 10-13)
while mid- and outer-disc means do not change. So:

- `(0029,1007) = 0` → AR-OFF (raw ringdown visible)
- `(0029,1007) = 1` → AR-ON (ringdown subtracted)

Cross-checking P_035 with this interpretation reveals the original
dataset is **AR-OFF on every frame except `FILE0013`** (which alone
has `mode_flag=1` and the clean inner-disc signature). Every E6 fit
behind today's YAML values was therefore made on raw ringdown, not on
the AR residual. **The `processing.ring_down.subtract_reference: true`
setting in `volcano_s5i.yaml` is mislabelled** — the bench template at
`P_035_PointScatter/derived/ringdown/ringdown_template_g54_d60.npy` is
the raw ringdown, and the simulator's match-mode policy needs to be
restated:

- If we want simulator output to match P_035 + the new E2 sweeps
  (which are also AR-OFF): set `subtract_reference: false` and
  add the raw template once.
- If we want simulator output to match the clinical default
  (AR-ON, FILE0013 + `CASE0000` mode=1 frames + `CASE0007/FILE0001`):
  the simulator should add the raw template and then subtract a stored
  reference, as it does today, but the *reference* template needs to
  be re-derived from the new AR-on / AR-off pairs (the old
  "subtracted" template was never actually subtracted).

Recommendation: support both modes; calibrate primarily against
AR-OFF (54 wire-phantom frames + 9 water-only frames), then derive the
AR-on residual from `CASE0000` AR-pair differences.

#### Compression curve is sigmoidal, not pure-log

A no-cost saturation preview on Sweep #3 (D=30) shows the inner-wire
peak palette as a function of slider gain has a clearly non-linear
slope:

```
g 24→28 : Δp99.5 = 22 palette  (slope 5.5 palette/step  ≡ log_mult ≈ 110)
g 28→32 : Δp99.5 = 29           (slope 7.3              ≡ log_mult ≈ 146)
g 32→36 : Δp99.5 = 38           (slope 9.5              ≡ log_mult ≈ 190)  steepest
g 36→40 : Δp99.5 = 34           (slope 8.5              ≡ log_mult ≈ 170)
g 40→44 : Δp99.5 = 28           (slope 7.0              ≡ log_mult ≈ 140)
```

A pure-log compression with the current `log_multiplier = 112.3`
predicts a constant ~22 palette per 4 gain steps. The data is steeper
in the middle and shallower at the extremes, exactly the signature
the original 3-point E7 analysis flagged as untestable. The new sweeps
let us replace the current scalar `log_multiplier` with a per-palette
`compression_lut` (currently `null` in YAML) and a non-linear slider→dB
mapping.

#### New apparatus is ~4 dB more efficient than the original

At equal gain on D = 60 mm, Sweep #2 (new apparatus) sees ~4 dB more
signal than Sweep #1 (original apparatus): for example at g = 20,
Sweep #2's p99.5 = 43 palette vs Sweep #1's p99.5 = 20 palette. Whatever
changed in the new apparatus (geometry / coupling / centre alignment)
adds ~4 dB of effective sensitivity. We should keep this offset
labelled per-apparatus rather than collapsing it into a single
`gain_db` term so that future P_035-style data on the original rig
remains self-consistent.

### Preferred gain if a single-orientation re-shoot is wanted

Based on the saturation map alone (the proper E7 fit will refine):

| Diameter | Preferred gain | Rationale |
|---:|---:|---|
| **30 mm** (new apparatus) | **32-36** | inner-wire p99.5 = 140-174 (~6-12 dB below saturation), outer-wire signal well above noise, slider sits in the steepest segment of the compression curve |
| 60 mm (either apparatus) | 44-48 | same logic at the wider depth |

If picking one combo, **g = 36, D = 30 mm, new apparatus** simultaneously
gives the cleanest lateral-PSF anchor for E2 and the cleanest
amplitude reference for E7.

### Re-fit order

Same as the prospective plan in §"What re-runs to do when the new data
arrives" below, with these specifics:

1. `extract_metadata.py --dataset ivus_test_0508/raw/<sweep>` — runs
   cleanly against the staged layout. Done for all three E2 sweeps and
   the multi-(gain,diameter) Capture-C subset, output at
   `ivus_test_0508/raw/<sweep>/derived/frames_meta.csv`.
2. `extract_gain_curve.py` (new) — runs the **N-point compression-curve
   fit directly on the DICOM display** without needing polar/alignment.
   Done for all three E2 sweeps, output at
   `ivus_test_0508/raw/<sweep>/derived/gain_curve/gain_curve.{json,csv,png}`
   plus a cross-sweep comparison at
   `ivus_test_0508/gain_curve_comparison.png`. **This is a preview**:
   the per-frame anchor it uses (99.5th-percentile palette over the
   disc) is a fast proxy for the brightest wire and is good enough to
   recover the shape of the compression curve, but the proper version
   wants the per-wire palette anchor that comes out of step 4 below.
3. `annotate_wires.py` (interactive) on each E2 sweep at the
   *unsaturated* gains the saturation preview identified — not all 18.
4. `fit_alignment.py` per apparatus (Sweep #1 vs Sweeps #2+#3) so the
   apparatus offset isn't absorbed into a single fit.
5. `unwrap.py` — convert all 18 frames per sweep to polar.
6. `extract_gain_curve.py --anchor=wire` (TODO) — re-run the same fit
   with the per-wire anchor instead of the global p99.5; replaces the
   current preview numbers below with the canonical fit. Will also feed
   per-bin `compression_lut` updates to the YAML.
7. `extract_psf.py` on the unsaturated low-gain frames — drops
   `--lateral-min-r-mm` to 2 and removes the `--lateral-peak-max-palette`
   filter.
8. `extract_ringdown.py` driven from CASE0000 paired AR-off / AR-on
   captures — gives a true AR-on residual template for the first time
   (and validates that the AR-off raw template at
   `P_035_PointScatter/derived/ringdown/ringdown_template_g54_d60.npy`
   matches the new AR-off frames).
9. Re-derive `noise.sigma`, `tgc_control_points`, AR policy.

### Preview fit results (step 2; global-anchor preview)

The N-point compression-curve fit on each sweep:

| Sweep | Apparatus | D | usable sliders | clean range | log_mult median | log_mult peak (palette/dB) | @ gain | DR (dB) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| #1 ascending | original | 60 | 9 / 18 | g 20-52 | 124.8 | **179.4** | 40 | 34.1 |
| #2 descending | new | 60 | 11 / 18 | g 12-52 | 111.5 | 175.0 | 40 | 38.4 |
| #3 ascending | new | 30 | 9 / 18 | g 12-44 | 125.1 | **179.4** | 32 | 33.9 |

Three independent observations come straight out:

1. **The compression curve is sigmoidal.** All three sweeps show the
   local log_multiplier rising from ~70 at the noise-floor end to a
   peak of ~175-180 around palette 140-200 (gain 32-40) and then
   falling back to ~110 just before saturation. The current YAML's
   scalar `log_multiplier = 112.3` is the *median across the curve*,
   not the *peak slope*; for high-amplitude rendering (wires, vessel
   walls) the simulator should use ~175, for low-amplitude rendering
   (speckle floor, anechoic background) it should use ~70.
2. **Sweep #2 is contaminated by descending-sweep settle lag.** The
   anchor palette is non-monotonic between gain 32 (palette 73) and
   gain 28 (palette 89); the ascending sweeps don't have this. For
   the canonical fit we should weight the two ascending sweeps and
   use Sweep #2 only as a coarse cross-check.
3. **The new apparatus is more sensitive at the noise-floor end** of
   the curve (Sweep #2 has 11 usable points vs 9 for Sweep #1), but
   the *steep* segment of the curve sits at the same gain (32-40) on
   all three sweeps. So the compression slope is a property of the
   console, not of the apparatus — the apparatus offset is a pure
   gain-shift, not a curve-shape change.

A canonical answer for the YAML (preview, will tighten with the
per-wire anchor in step 6):

| Field | Old (P_035 3-point) | New (preview, this dataset) |
|---|---:|---:|
| `processing.log_multiplier` (canonical / scalar) | 112.3 | ≈ 125 (median of the steep segment) |
| `processing.log_multiplier_peak` (new field) | — | ≈ 178 (palette per dB at gain 32-40) |
| `processing.compression_lut` | null | populated 18-point curve in `gain_curve.json` |
| `processing.dynamic_range_db` | 40.6 | 34-38 (depending on sweep) |
| `processing.reject_palette` | 11 | 11-12 (consistent) |
| `processing.saturation_palette` | 239 | 239 (unchanged) |

### Annotation, alignment, and wire-anchored E7/PSF fits (delivered 2026-05-12)

The bench operator annotated 30 frames across the three sweeps. The
fits converged cleanly on **31/31 usable frames** (all of Sweep #1, all
of Sweep #2, and 12 of 13 attempted Sweep #3 frames -- FILE0014 was
the only frame the rigid-body fit could not reconcile).

#### Alignment fit summary (post-cleanup)

| Sweep | Apparatus | D | clean frames / annotated | median theta0 | median radial_scale | implied SoS (m/s) | median wiggle |
|---|---|---:|---:|---:|---:|---:|---:|
| #1 | original | 60 | 9 / 9 | +287.9 deg | 1.076 | 1656 | 2.16 mm |
| #2 pre-rot | new | 60 | 4 / 4 (FILE0004-7) | +142.2 deg | 1.043 | 1485 | 0.79 mm |
| #2 post-rot | new | 60 | 6 / 6 (FILE0008-13) | +65.3 deg | 1.052 | 1465 | 0.43 mm |
| #3 | new | 30 | 12 / 13 (drop FILE0014) | +144.1 deg | 1.002 | 1537 | 0.61 mm |

Three things changed our understanding of the dataset relative to the
preview:

1. **Sweep #2 has a mid-sweep ~77 deg catheter rotation** between
   FILE0007 (gain 28) and FILE0009 (gain 36). FILE0008 (gain 32) is a
   transient captured during the rotation -- its 10 clean clicks (the
   user's clicks `u10`/`u11` were stray speckle; `u12` lands on
   physical wire `w12`) cannot be reconciled by any single rigid-body
   fit because different angular sectors of the polar B-mode frame
   came from different theta0 values. We drop FILE0008 only; the
   pre- and post-rotation halves remain usable and *give us a free
   second orientation on the new apparatus* (142 deg + 65 deg, ~77
   deg apart). This is exactly the orientation diversity that the
   pre-bench plan asked for as item (2) of "Recommended follow-up".
2. **Original apparatus's wires sit ~7-8% farther from center than
   the DXF design.** Sweep #1's radial_scale converges to 1.076 with
   std 0.005 across 9 independent gain frames -- a real apparatus
   property. The new apparatus's wires fit the DXF to ~0.2% (scale =
   1.002 in Sweep #3). The simulator's match against original-
   apparatus data should compensate for this 7.6% radial offset.
3. **Catheter wiggle is much smaller on the new apparatus.** Sweep #1
   shows 2.0-2.4 mm of catheter-vs-apparatus offset (the catheter
   wedged off-center in the orig fixture). Sweep #3 shows 0.5-0.7 mm
   (much more centered). The new apparatus is mechanically cleaner.

#### Wire-anchored E7 (replaces the global-anchor preview)

Median peak palette across in-field design wires, per frame. The
median absorbs the SA-orientation brightness scatter; PCHIP gives the
compression curve on the unsaturated band.

| Sweep | n_usable / n_total | log_multiplier (canonical) | IQR | DR (dB) | reference slider |
|---|---:|---:|---:|---:|---:|
| #1 (orig, 288 deg)             | 5 / 9  | **133.2** | 26.6  | 33.7 | 32 |
| #2 (new, 142 deg + 65 deg)     | 7 / 10 | 110.7     | 170.4 | 39.7 | 28 |
| #3 (new, 144 deg) — **canonical** | 8 / 12 | **137.4** | 42.7  | 33.0 | 22 |
| pooled (Sweep #1 + #3 are different apparatus baselines, can't be naively pooled) | — | — | — | — | — |

- **`log_multiplier = 137 +/- 27`** is the canonical value (Sweep #3,
  with Sweep #1's 133 as orig-apparatus cross-check -- agreement to
  3%).
- **The 25% upward shift from the preview (~125 → 137)** is because
  the global-anchor (p99.5 of disc) picks up speckle, which has a
  flatter compression than the wires themselves; the wire-anchored
  number is the correct compression slope for modelling scattering.
- **Sweep #2's IQR explosion (170) is real and useful**: it directly
  measures the wire-orientation effect. Between gain 28 (orient
  142 deg, anchor 188.2) and gain 32 (orient 65 deg, anchor 185.5)
  the median wire peak drops despite the gain step adding nominally
  +4 dB. The orientation flip cost roughly **4-5 dB of effective
  signal** -- consistent with the +/-2-3 dB SA-pitch lobing
  estimate, and worth budgeting into the simulator's lateral PSF.

Outputs:

- `ivus_test_0508/raw/<sweep>/derived/gain_curve_wire/gain_curve.{json,csv,png}`
- `ivus_test_0508/gain_curve_wire_comparison.png` (cross-sweep plot)
- `ivus_test_0508/gain_curve_wire_summary.csv`
- `ivus_test_0508/gain_curve_wire_pooled.json` (canonical = Sweep #3)

#### PSF fits (axial pulse / lateral beam Gaussian fit)

`extract_psf.py` outputs at `ivus_test_0508/raw/<sweep>/derived/psf/`.

| Sweep | gains used | n_axial_clean | axial median FWHM | n_cycles | w_0 (mm) | z_f (mm) | a (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| #1 (orig, D=60)                | 32-40    | 9 / 12  | 58 um  | 0.75 | 0.487 | 16.67 | 2.635 |
| #2 (new, D=60, both orient)    | 24-40    | 46 / 47 | 116 um | 1.51 | 0.496 | 18.08 | 2.807 |
| #3 (new, D=30) — **canonical** | 20-32    | 20 / 20 | 149 um | **1.93** | **0.264** | **9.55** | 2.787 |

Three observations:

1. **`element_radius_mm = 2.74 +/- 0.07 mm`** -- agrees across all
   three sweeps and both apparatus (highest-confidence number we have).
2. **`focal_length_mm = 9.55 mm`** comes from Sweep #3 only. Sweeps
   #1/#2 fit z_f ~ 17-18 mm but their wires (after the r >= 10 mm
   filter) sit entirely in the far field of the focal zone, so their
   z_f is essentially an extrapolation. Sweep #3's wires at
   r = 4-14 mm straddle the focal zone, which is why the fit is
   well-conditioned (sigma_z_f = 0.35 mm vs 1.2-2.0 mm on the others).
3. **`pulse_duration_cycles = 1.93`** -- only Sweep #3 has the axial
   sampling resolution (0.06 mm/pixel) to resolve the axial PSF
   properly (lambda at 10 MHz is 154 um). Sweep #1/#2 at 0.12 mm/pixel
   are under-sampled axially, giving artificially narrow axial FWHM
   measurements and a too-low n_cycles. **Sweep #3's 1.93 cycles is
   the canonical value; the YAML's current 1.0 cycles is too low.**

Canonical YAML deltas after the wire-anchored fits:

| Field | Old (P_035 3-point) | Global-anchor preview | New canonical (wire-anchored, Sweep #3) |
|---|---:|---:|---:|
| `processing.log_multiplier` | 112.3 | ~125 (median) | **137** (canonical) |
| `processing.dynamic_range_db` | 40.6 | 34-38 | **33** |
| `processing.reject_palette` | 11 | 11-12 | **11** |
| `processing.saturation_palette` | 239 | 239 | **239** |
| `transducer.element_radius_mm` | (placeholder) | — | **2.79** |
| `transducer.focal_length_mm`   | (placeholder) | — | **9.55** |
| `transducer.pulse_duration_cycles` | 1.0 | — | **1.93** |

These are the numbers to write into `volcano_s5i.yaml` for the
canonical (new-apparatus, AR-OFF) match against the bench dataset.
The orig-apparatus geometric offset (radial_scale = 1.076) is *not*
a YAML field -- it's an apparatus property; downstream P_035 match
plots should rescale wire radii by 1.076 before comparison.

### Suggested gain (and apparatus) for any orientation re-shoot

`g = 36, D = 30 mm, new apparatus` — sits in the steepest region of
the compression curve on the cleanest sweep, gives 6-12 dB headroom
below saturation, and produces the highest signal density per
displayed pixel.

**Update 2026-05-12**: the orientation diversity Sweep #2's mid-sweep
rotation gave us (142 deg + 65 deg, 77 deg apart) is already enough to
quantify the orientation-induced gain modulation at ~4-5 dB
peak-to-peak. The Gaussian-beam lateral fit on Sweep #2 (using both
orientations across 16 clean wire-peaks) agrees with Sweep #3
(`element_radius_mm` 2.81 vs 2.79; `w_0` 0.50 vs 0.26 -- the difference
is the sampling resolution at the focal zone, not an orientation
effect). So a deliberate orientation re-shoot is now **no longer on
the critical path**; the bench operator already gave us the data we
needed via the mid-sweep rotation.

### Wire-orientation brightness variability

Bench observation during the 2026-05-08 session: rotating the
catheter relative to the wire phantom made individual wires get
brighter / dimmer with no other change. This is consistent with the
PV .035 catheter's **synthetic-aperture beamforming + 64-element
azimuthal sampling**:

- 64 elements on the 1.9 mm OD circumference -> 5.6 deg / element.
- SA reconstructs the lateral image; the reconstructed lateral PSF
  has azimuthal lobes at the element-pitch frequency.
- A wire that lands at a sub-element angular offset of 0 (centred on
  an element) images ~2-3 dB brighter than one at offset 2.8 deg
  (halfway between elements).

The operator selected the lowest-background orientation, which biases
the per-wire amplitude in a fixed but unknown direction.

Impact on this calibration:

| Impact | Severity |
|---|---|
| Compression-curve slope (gain LUT, log_multiplier) | mild -- median across N wires averages out the wire-to-wire offsets; the new 12-wire spiral at 30 deg spacing is well sampled |
| Lateral-PSF -6 dB FWHM at a single wire (E2 fit) | ~+/- 2-3 dB amplitude noise -> ~+/- 0.05 mm FWHM bias per wire |
| Focal-length / element-radius (Gaussian-beam fit) | mild -- the multi-wire fit averages the orientation effect across radii, especially with the 12-wire spiral |
| AR template (E6) | none -- AR is at r ~= 2 mm where the element-pitch lobes don't apply |
| Saturation / reject palettes (E7) | none -- these are device thresholds |

Anomaly in the Sweep #2 descending curve at gain 28 -> 32
(palette 89 -> 73, non-monotonic) is most likely the global-anchor
following different wires at the two gains rather than a settle-lag
artefact. The wire-anchored re-fit will smooth this out.

### Recommended follow-up experiments

In priority order:

1. **Multi-wire median anchored gain curve** -- already enabled by the
   12-wire spiral. Comes with the in-flight wire annotation; will be
   driven by `extract_gain_curve.py --anchor=wire` when annotation is
   done. No new bench time required. **Resolves the orientation noise
   at the gain-LUT level.**
2. **Orientation-averaged E2 at the preferred gain** (g=36, D=30 mm,
   new apparatus). 4-8 captures at different probe rotations, all at
   the same gain, ~5 min of bench time. Resolves the orientation
   noise at the per-wire / lateral-PSF level.
3. **E4 uniform-attenuation phantom**. The mold STL is already
   designed in `instrument-calibration/tools/gen_phantom_stl.py`
   (`phantom_mold_uniform.stl`). One pour of a graphite-agar or PVA
   gel + 24 h cure + one sweep gives a wire-free, orientation-free
   in-tissue gain LUT, TGC, and speckle-statistics measurement. This
   is also the only way to refit `tgc_control_points` for tissue
   imaging (the current values are in-water only). Highest ROI of any
   future bench session.
4. **E3 slice-thickness sweep.** The sim now defaults to
   `elevational_height_mm = 1.5` with 8 elevational samples (uniform
   top-hat mean). That is an uncalibrated placeholder, not a measured
   FWHM / elevational PSF. E3 is what replaces the estimate.

For this dataset's calibration we proceed with (1) and document (2)
+ (3) + (4) as future improvements.

### AR template (E6) -- delivered

`ivus_test_0508/raw/c_take2_water/derived/ar_template/`

- `ar_template_canonical.npy` -- 83-sample (0-5 mm at 0.06 mm/px)
  canonical AR template in palette units, median of the gain-50
  D=30 and D=60 pairs.
- `ar_template.json` -- per-pair summary.
- `ar_template_overview.png` -- AR-off / AR-on / template plot.

Three independent pairs (g=50/D=30, g=50/D=60, g=40/D=60) give:

- peak depth = 2.22 mm in all three (matches catheter OD = 1.9 mm
  + bandwidth)
- extent (5% of peak threshold) = 3.30-3.42 mm in all three
- gain-50 peak palette = 212.5 (D=30) vs 212.0 (D=60) -- **diameter-
  invariant** to within 0.5 palette; confirms the AR template is a
  property of the catheter, not the display
- gain-40 / gain-50 peak ratio = 142.5 / 212.5 = 0.67 -> implied
  log_multiplier = 140 palette/dB, **matching the local slope of
  the new sigmoidal gain-curve fit at gain 40-50** independently

The canonical template is the input the simulator's
`processing.ring_down.waveform_path` should consume when
`subtract_reference: true` (clinical AR-ON mode). When
`subtract_reference: false` (the new default, matching P_035 and the
E2 wire-phantom frames), the simulator should instead consume the
raw template at
`P_035_PointScatter/derived/ringdown/ringdown_template_g54_d60.npy`
-- both modes are now characterised.

The bench session is therefore *complete* for the table at the top of
this doc: every "yes" in the *Likely to move?* column will move once
the steps above are run. The remaining uncertainty bullets at lines
44-53 are now resolvable.

### YAML update applied 2026-05-12 (`volcano_s5i.yaml`)

The wire-anchored E7 and PSF results have been written into
`volcano_s5i.yaml`. Summary of diffs:

| Field | Old | New | Source |
|---|---|---|---|
| `probe.pulse_duration_cycles` | 2 | **1.93** | Sweep #3 axial FWHM = 149 um at lambda = 154 um |
| `probe.element_radius_mm` | 4.6 | **2.79** | Sweep #3 Gaussian-beam lateral fit; cross-checks 2.64 / 2.81 on Sweeps #1 / #2 |
| `probe.focal_length_mm` | 19.2 | **9.55** | Sweep #3 focus; first dataset with wires straddling the focal zone |
| `processing.log_multiplier` | 112.3 | **137.4** | Wire-anchored E7 canonical (Sweep #3); 3% cross-check from Sweep #1 |
| `processing.ring_down.amplitude` | 46.37 | **23.0** | analytically recomputed from peak_palette_excess = 187.1 under the new log_multiplier |
| `processing.gain_db` | 73.92 | 73.92 (**STALE**) | needs derive_gain_db.py re-run; first-order linearised target ~ 72.4 dB |
| `processing.noise.sigma` | 0.000712 | 0.000712 (**STALE**) | needs derive_noise_sigma.py re-run *after* gain_db |

The AR-residual canonical (`ar_template_canonical.npy`) is documented
in the YAML's ring-down comment block as a swap-in waveform when
`subtract_reference: true`; the actual wiring requires a new
`ar_residual_waveform_path` field on `raysim/config.py::RingDownConfig`
which is left for the simulator side.

### Tier 1 status after the YAML update

Run `scripts/tier1_analytical_subset.py` for the four CPU-only tests
that don't require the CUDA backend:

```
A. YAML round-trip                 -> PASS
B. Configuration self-consistency  -> PASS
G. Log-compression mapping         -> PASS
H. TGC schedule                    -> PASS
```

The simulator-rendering tests (C/D/E/F/I + gain alignment) need to run
on a CUDA host with `raysim.ray_sim_python` available. Predicted
qualitative impact at the *current* (stale) gain_db / noise.sigma:

- **C/D Axial/Lateral PSF.** Both bench medians are within the
  previous fit's CI, but the sim renders at the OLD aperture (a=4.6)
  produced wider lateral PSFs than the bench. The new a=2.79 + z_f
  =9.55 should *tighten* the sim lateral PSF and shift the focus
  shallower; we expect Test D to move from FAIL toward PASS once the
  geometry change is exercised.
- **E. Ring-down.** Amplitude was rescaled deterministically, so
  if the simulator was matching shape but missing magnitude before
  it should now be at the right palette excess (peak ~189 above
  the speckle floor at gain 54). Shape unchanged.
- **F/I Noise floor + depth uniformity.** Will *regress* in the
  short term: with log_multiplier 137.4 and unchanged noise.sigma
  the anechoic palette mean is predicted to land at ~56 (vs the
  bench target 46.2). Re-running derive_noise_sigma.py on the CUDA
  host (and then derive_gain_db.py) will restore Pass-7 behaviour.
- **G/H** -- analytical, already PASS.

### Outstanding YAML follow-ups (in order)

1. **Re-derive `gain_db`** on a CUDA host:
   `derive_gain_db.py` will bisect gain_db to match the bench's
   anechoic mean palette (46.2 @ slider 54). Predicted target ~72.4 dB
   (linearised). Wire-target IQR + focal shift may push it +/- 6 dB.
2. **Re-derive `noise.sigma`** (depends on step 1):
   `derive_noise_sigma.py` bisects sigma so the post-pipeline anechoic
   palette has the right mean *and* std. Predicted scale factor 0.840
   on the current sigma; needs the new PSF + gain_db to verify.
3. **Wire `ar_residual_waveform_path`** through the simulator if and
   when AR-on rendering is needed (low priority while
   `subtract_reference: false` matches all current bench data).

---

## TL;DR

We populated `ultrasound-raytracing/configs/volcano_s5i.yaml` from a partial,
improvised version of the `ivus_calibration_protocol.md` experiments using
the `P_035_PointScatter` DICOM dataset (post-scan-converted, log-compressed
B-mode images at 3 gain settings × 3 imaging diameters; 19 frames total;
no RF data; AR always ON). Every entry now has a real measurement behind
it, but **5 of them are bracketed by limitations of the displayed-image
data** and will move when a follow-up bench session captures the same
phantom at lower gain settings (and ideally a smaller imaging diameter).

The Volcano s5i clinical port likely **does not expose raw RF**, so the
primary follow-up plan below is built around displayed-image data; an
"if RF can be obtained" alternative path is at the end.

## What we have right now (and how confident we are in it)

| YAML field | Current value | Source experiment | Confidence | Likely to move? |
|---|---:|---|---|---|
| `probe.frequency_mhz` | 10.0 | Manual + Visions PV .035 family | high | no |
| `probe.pulse_duration_cycles` | 2 | E2 axial PSF, sampling-limited | low | yes — could be 1 or 2 |
| `probe.element_radius_mm` | 4.6 | E2 Gaussian-beam fit | low | yes — non-physical, SA-fit artifact |
| `probe.focal_length_mm` | 19.2 | E2 Gaussian-beam fit | medium | yes — could be 12-20 |
| `probe.speed_of_sound_mm_per_us` | 1.54 | Manual (s5 device convention) | high | no |
| `processing.tgc_control_points` | `[[0,0],[0.7,0],[1.5,0],[2.2,0.98],[2.9,2.05]]` | E4 | medium | maybe — operator-set, water-only |
| `processing.log_multiplier` | 112.3 | E7 partial (cross-gain ringdown) | medium | yes — fit assumes pure log + 1 step = 1 dB |
| `processing.log_floor` | 1.0 | Convention | high (def.) | no |
| `processing.dynamic_range_db` | 40.6 | Derived from log_multiplier | medium | yes — moves with log_multiplier |
| `processing.gain_db(slider)` | 1 step = 1 dB | Working hypothesis | low | yes — slider may be non-linear |
| `processing.noise.type` | gaussian | E5 (Rayleigh-on-display = Gaussian-RF) | high | no |
| `processing.noise.sigma` | 2.6347 | E5 + E7 conversion | medium | yes — moves with log_multiplier |
| `processing.ring_down.amplitude` | 46.37 | E6 + E7 conversion | medium | yes — moves with log_multiplier |
| `processing.ring_down.extent_mm` | 3.0 | E6 (5% of peak excess) | high | no |
| `processing.ring_down.decay` | measured | E6 template captured | high | no |
| `processing.ring_down.subtract_reference` | true | DICOM private tag confirmed | high | no |

The four "low" / "medium" entries that are most likely to change are:

1. `probe.pulse_duration_cycles` -- currently constrained by a 0.12 mm
   pixel pitch; true PSF is at or below 1 sample.
2. `probe.element_radius_mm` -- 4.6 mm is mathematically what makes a
   Gaussian beam reproduce the *deep-wire* lateral FWHM; the catheter's
   physical element pitch is ~0.09 mm. The "right" simulator value
   depends on whether the simulator ever models real synthetic-aperture
   beamforming.
3. `probe.focal_length_mm` -- ±0.8 mm formal uncertainty, but the data
   is also consistent with anything in 12-19 mm.
4. `processing.log_multiplier` -- agrees within 12% across our two
   estimators, but the third (mid-range profile) differs by 2-3x
   depending on which gain pair you look at -- strong evidence for
   either non-linear slider or non-pure-log compression.

## Root causes of the uncertainty

### What the dataset is

The phantom in P_035_PointScatter uses **36 AWG magnet wire** (≈ 127 µm
diameter copper with thin enamel insulation) as the point scatterers,
arranged in 5 radial columns at r ≈ 5, 10, 15, 20, 25 mm in a water tank.
This is actually the right *kind* of target -- the wire diameter is
≈ λ at 10 MHz (λ = 154 µm in water), so each wire is in the geometric-
scattering regime and behaves as a reasonably idealized line scatterer.
The protocol's E2 calls for 25 µm tungsten which is similar (thinner →
weaker → less saturation, but otherwise the same physics).

### What's wrong with the dataset for our purposes

The catheter was operated at **gain ∈ {44, 54, 64}** -- the lowest
available was 44 on the slider, and 44 still saturates every wire
within ≈ 15 mm. The dataset also contains **only displayed B-mode
images** (post-scan-conversion, post-log-compression, post-reject;
PALETTE COLOR uint8). Combined, these give us four limitations:

1. **Axial sampling resolution.** The s5 acquires at ~40 MHz internally
   (= 19 µm axial sample pitch in water) but the displayed images are
   at the device's screen-pixel pitch, which depends on the imaging
   diameter setting:
   - Diameter 35 mm -> 0.07 mm/pixel  (best)
   - Diameter 40 mm -> 0.08 mm/pixel
   - Diameter 60 mm -> 0.12 mm/pixel  (most of our data)
   At 0.12 mm/pixel the axial PSF is at or below 1 sample and we can
   only put a *bound* on `pulse_duration_cycles`.

2. **Linear amplitude.** Every measurement is in palette units (uint8
   grayscale post-log-compression), not RF amplitude. We recovered
   `log_multiplier` through cross-gain redundancy, but with only 3
   gain points and ubiquitous saturation we can't constrain the
   compression curve shape (linear-in-log vs sigmoidal).

3. **Saturation everywhere.** Every wire echo within ~15 mm hits or
   exceeds 200 palette at gain 44 (and worse at 54/64). Saturation
   distorts the -6 dB FWHM and contaminates the lateral-PSF fit. With
   the existing slider settings we can't escape it.

4. **No sub-saturation gain setting.** The slider in the dataset is
   {44, 54, 64} -- the protocol's E1 / E2 want gain set so the
   *deepest* wire is ≥ 30 dB above noise but the *shallowest* is
   ≤ 95 % saturated, which for these wires is probably gain ~ 20-30.
   We don't have any data below 44.

## What the bench session needs to deliver

Two captures using the existing P_035 phantom, no new hardware.

### Capture A (mandatory) -- gain sweep, smallest available diameter

Same phantom (36 AWG magnet wire columns), same water tank, no AR
toggle changes (leave AR ON if that's the default). Acquire:

- **Imaging diameter = 35 mm** (the smallest the catheter supports
  in this dataset), so the displayed pixel pitch is 0.07 mm and the
  axial PSF is sampled by ~2 pixels per wavelength instead of < 1.
- **Gain swept in fine steps** across the full slider range. Target:
  one capture every 4 slider units from 0 to 68, i.e. gain ∈ {0, 4,
  8, ..., 64, 68} -- 18 captures.
- **30 frames per gain setting** (so each individual frame's noise
  is averaged out for the cleanest per-wire peak measurement).

Total time: ≈ 20 minutes of console time (mostly waiting for the
console to settle between gain changes).

What this fixes in the YAML:

| YAML field | How |
|---|---|
| `processing.log_multiplier` | At each gain step, the same wire produces a different displayed peak. Fitting `pixel(g) = log_multiplier * g_dB / 20 + offset` across 18 gain points (instead of our current 3) directly gives `log_multiplier` and confirms whether 1 slider step = 1 dB. **Replaces our cross-gain inference with a direct measurement.** |
| `processing.gain_db(slider)` | The 18-point slider→displayed-pixel curve directly gives the slider→dB mapping, including any non-linearity in the slider. Replaces our 3-point estimate. |
| `processing.dynamic_range_db` | Direct read from the gain values that produce palette = 11 (just above reject) and palette = 239 (saturation) on a known reflector. |
| `processing.compression_lut` | Reconstructable as a 256-entry palette↔dB-relative-to-noise table (still in arbitrary-amplitude units without RF, but absolute palette↔dB is what the simulator actually needs). |
| `probe.pulse_duration_cycles` | At gain steps where the wires are *not* saturated (the new low-gain frames), the displayed axial -6 dB FWHM at 0.07 mm/pixel measures the pulse duration to within ~1 sample. We expect to distinguish 1.0 vs 1.5 vs 2.0 cycles cleanly. |
| `probe.element_radius_mm` | At low gain the inner wires (r = 5 mm) are no longer saturated, so the lateral PSF measurement at the focus -- the most sensitive depth -- becomes reliable. The Gaussian-beam fit gets a real focal-region anchor instead of extrapolating from r ≥ 15 mm. |
| `probe.focal_length_mm` | Same as above; the fit picks up real depth-dependent FWHM curvature on the inner side of the focus, not just the outer side. |
| `processing.noise.sigma` | Re-derived from the new `log_multiplier`. |
| `processing.ring_down.amplitude` | Re-derived from the new `log_multiplier`. |
| `processing.tgc_control_points` | dB values rescale to the new `log_multiplier` (E4 measured the palette difference; only the palette-to-dB conversion changes). |

### Capture B (nice-to-have) -- weaker reflectors

If it's easy to re-string the phantom, switching to a **lower-impedance,
sub-wavelength filament** would directly fix the saturation problem and
also give a cleaner sub-λ scatterer for the axial PSF. In ranked order
of preference for the PV .035 (10 MHz):

1. **Nylon monofilament, ⌀ 70–100 µm** (e.g. 4-0 polyamide surgical
   suture, or 1–2 lb-test clear nylon fishing line). Single-surface
   reflection is ~6 dB (vs ~30 dB for copper wire), so the inner wires
   stop saturating at all reasonable gains, and the 75 µm diameter is
   ≈ λ/2 at 10 MHz so the axial PSF is no longer extended by the
   wire's own diameter (36 AWG copper at 127 µm is ≈ 0.85 λ — borderline).
   Cost ≈ $5; standard AIUM/Pinto-phantom material.
2. Thinner copper magnet wire (40 AWG ≈ 80 µm, 38 AWG ≈ 100 µm) at the
   inner radii while keeping 36 AWG at the outer radii. Keeps the
   "metallic wire" experimental setup but only partially addresses
   saturation (the inner wires drop ~6 dB but stay specular and high-Z).
3. 25–50 µm tungsten wire if a spool is on hand. Smallest sub-λ and
   the canonical AIUM target, but the highest impedance contrast — will
   saturate at any practically useful gain on the PV .035, so this is
   only worth doing if the captures will be done at very low gain
   (slider ≤ 25) or with an attenuator inline.

Option 1 (nylon) is the clean fix and is what we'd recommend if the
phantom is being re-strung anyway. Capture B is still optional —
Capture A's gain sweep alone is sufficient to fix everything in the
table above — but a nylon re-string would give the lateral-PSF fit
the cleanest possible inner-wire anchor at every gain step.

### What stays uncertain even with both captures

Without RF data, two YAML fields cannot be improved:

- `probe.impulse_response_path` -- needs raw A-line RF samples for the
  simulator's pulse-shape lookup. Stays `null`.
- `sim.sampling_freq_mhz` -- the displayed-image data tells us nothing
  about the device's RF sample rate. Stays at the documented 40 MHz.
- *Absolute* RF amplitude calibration -- the noise.sigma and
  ring_down.amplitude values are in arbitrary "amp = 1" units defined
  by `log_floor = 1.0`. This is fine for the simulator (it just sets
  the unit of the rendered image's intensity) and matches whatever
  convention the rest of the pipeline uses, but it means there is no
  traceable mapping back to mPa or W/m².

## What we DON'T need from the bench session

- **E4 in tissue** -- the in-water TGC we have is correct for the
  P_035 dataset; tissue TGC is a separate question for a separate
  phantom.
- **E5 (cyst phantom)** -- noise.sigma is already constrained by E5
  + E7 conversion; the cyst phantom adds reject_db and per-tissue
  scattering priors which are E8-flavored and not blocking.
- **E6 alone** -- ringdown is already cleanly characterized.
  Re-running E6 with AR OFF (if the console allows it) would let us
  cross-check the AR-residual hypothesis; useful but not required.
- **E7 standalone with RF injection or step phantom** -- Capture A
  delivers the same calibration with no extra hardware.
- **E8** -- per-material parameters are not driven by P_035; this is
  a separate clinical / ex-vivo session.
- **E9** -- timing is not in the per-frame raysim config.

So the practical ask is **one ~20-minute capture session** (Capture A),
with optional weaker-wire swap (Capture B) and optional AR-OFF
single-frame (Capture C, see below).

### Optional Capture C -- AR OFF cross-check

If the console allows toggling Acoustic Reference subtraction off (some
service-mode menus expose this), capture **30 frames at gain 50, D = 60
mm, AR OFF**, in the same water tank. This gives the *raw* (unsubtracted)
ringdown for cross-checking our AR-residual measurement and for
validating that the simulator's `subtract_reference: true` path produces
the right output when run against the AR-off raw.

## If RF capture *is* possible (the bonus path)

If the console exposes a service-mode RF stream (some Volcano consoles
do via the back-panel research port; depends on firmware), then the
proper protocol E1 + E2 from `ivus_calibration_protocol.md` becomes
feasible. Specifically:

- **E1 with the gain-sweep extension** I sketched in an earlier
  revision of this doc (flat plate at d = 3 mm, sweep gain). RF gives
  us `probe.impulse_response_path`, `sim.sampling_freq_mhz`, the
  *signed* compression LUT (palette ↔ absolute dB), and a redundant
  measurement of `pulse_duration_cycles` directly from the FFT
  bandwidth.
- **E2 spiral fixture** from Appendix A.1 (now sized for the PV .035
  catheter: ⌀80 mm discs, ⌀2.0 mm catheter hole, 12 wires at
  r = 4 → 26 mm with extra density around the 12–20 mm focal zone). The
  protocol now lists nylon monofilament (~75 µm) as the primary wire
  material with copper / tungsten as labelled alternatives — see the
  "Wire material trade-offs" table in Appendix A.1. RF makes the
  lateral-PSF measurement physical-amplitude rather than
  displayed-grayscale, which removes the saturation problem entirely and
  gives an absolute focal_length / element_radius fit even with the
  high-impedance metallic wires.

If RF is available, this is preferred over Capture A above. If only
one path is available, Capture A above gives ~80 % of the value of
the full RF-based protocol.

## What re-runs to do when the new data arrives

The analysis pipeline is set up so the new data slots in cleanly. The
order matters because some scripts depend on outputs of others.

1. **Ingest.** Save the new DICOM (or RF) data to
   `<NEW_DATASET>/raw/` following the same convention as
   `P_035_PointScatter/`.

2. **Alignment + un-scan-conversion** for any new diameter / gain
   combos (run from the workspace root):
   ```
   python3 instrument-calibration/p035_visions/annotate_wires.py \
       --raw-dir <NEW_DATASET>/raw
   python3 instrument-calibration/p035_visions/fit_alignment.py \
       --raw-dir <NEW_DATASET>/raw
   python3 instrument-calibration/p035_visions/unwrap.py \
       --raw-dir <NEW_DATASET>/raw
   ```

3. **Refit log_multiplier and gain LUT (E7 proper)** from Capture A's
   ~18-point gain sweep. The current
   `extract_gain_lut.py` does the cross-gain inference
   from 3 gain points; for the proper N-point fit, add an
   `--n-point-mode` flag (≈ 30 min of work) that:
   - Picks one un-saturated wire per gain step
   - Fits `pixel(g) = log_multiplier * (g - g0) / 20 + p0`
     by linear regression
   - Reports the slider→dB curve, log_multiplier with proper
     uncertainty, dynamic_range_db, and the displayed-grayscale
     compression LUT
   - Updates YAML's `processing.log_multiplier`, `log_floor`,
     `dynamic_range_db`, `gain_db`, `compression_lut`.

4. **Refit lateral PSF (E2)** with the un-saturated low-gain frames
   (run from the workspace root):
   ```
   python3 instrument-calibration/p035_visions/extract_psf.py \
       --polar-dir <NEW_DATASET>/derived/polar \
       --align-csv <NEW_DATASET>/derived/alignment_fit.csv \
       --meta-csv <NEW_DATASET>/derived/frames_meta.csv \
       --dxf <NEW_DATASET>/<wire_geometry>.dxf \
       --out-dir <NEW_DATASET>/derived/psf \
       --gain-sliders <pick the gains where inner wires don't saturate> \
       --diameter-mm 35
   ```
   With un-saturated inner wires, `--lateral-peak-max-palette` can be
   raised to ~230 (or removed) and `--lateral-min-r-mm` can drop to 2.
   Update YAML's `probe.element_radius_mm`, `probe.focal_length_mm`,
   `probe.pulse_duration_cycles`.

5. **Re-derive noise.sigma and ring_down.amplitude** with the new
   log_multiplier (rerun `extract_gain_lut.py`; it does the
   conversions automatically and the YAML comments tell you the
   exact formulas).

6. **Re-scale tgc_control_points** with the new log_multiplier:
   ```
   new_dB = old_palette_diff * 20 / new_log_multiplier
   ```
   Trivial Python one-liner; update the YAML.

7. **Update this doc** to reflect the new uncertainties.

## Where the artifacts live

For traceability after the new data lands:

- Current YAML: `ultrasound-raytracing/configs/volcano_s5i.yaml`
- Current per-experiment results:
  - `P_035_PointScatter/derived/alignment_fit.csv` (E0)
  - `P_035_PointScatter/derived/polar/` (un-scan-converted polar)
  - `P_035_PointScatter/derived/psf/` (E2)
  - `P_035_PointScatter/derived/tgc/` (E4)
  - `P_035_PointScatter/derived/noise/` (E5)
  - `P_035_PointScatter/derived/ringdown/` (E6)
  - `P_035_PointScatter/derived/gain_lut/` (E7 partial)
- Analysis tools (all under `instrument-calibration/p035_visions/`):
  - `unwrap.py`
  - `fit_alignment.py`
  - `extract_psf.py`
  - `extract_tgc.py`
  - `extract_noise.py`
  - `extract_ringdown.py`
  - `extract_gain_lut.py`
  - `polar_utils.py` (shared helpers)
- Protocol document: `../docs/ivus_calibration_protocol.md`
- Calibrated YAML output (this folder): `volcano_s5i.yaml`

## One-line bottom line

The current YAML is correct in shape and order-of-magnitude for every
field. The **fine values for axial pulse, lateral PSF, log multiplier,
and everything that derives from log multiplier (noise sigma, ringdown
amplitude, dB-valued TGC, dynamic range)** will tighten once the bench
delivers a fine-step gain sweep on the same 36 AWG wire phantom at
the smallest available imaging diameter -- a ≈ 20-minute console session
with no new hardware required.
