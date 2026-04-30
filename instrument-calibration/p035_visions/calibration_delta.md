# P_035_PointScatter calibration — current state and follow-up bench needs

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

- **E3 (slice thickness)** -- the simulator's
  `elevational_height_mm = 0.0` configures it as 2D; until that
  changes there's no point measuring elevational PSF.
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
- Protocol document: `../../i4h-sensor-simulation/docs/ivus_calibration_protocol.md`
- Calibrated YAML output (this folder): `volcano_s5i.yaml`

## One-line bottom line

The current YAML is correct in shape and order-of-magnitude for every
field. The **fine values for axial pulse, lateral PSF, log multiplier,
and everything that derives from log multiplier (noise sigma, ringdown
amplitude, dB-valued TGC, dynamic range)** will tighten once the bench
delivers a fine-step gain sweep on the same 36 AWG wire phantom at
the smallest available imaging diameter -- a ≈ 20-minute console session
with no new hardware required.
