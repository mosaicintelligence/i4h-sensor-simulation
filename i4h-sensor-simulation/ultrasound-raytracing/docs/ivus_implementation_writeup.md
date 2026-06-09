# IVUS Implementation Notes

This document describes the IVUS-specific implementation in the `ultrasound-raytracing` package: the probe class, the IVUS-specific physics, the RF / B-mode processing pipeline, and the Python configuration layer. It complements the probe-agnostic [Technical Guide](../../docs/ultrasound_simulator_technical_guide.md) and the [Quick Start](quick_start.md).

All file paths are relative to `ultrasound-raytracing/`. The canonical calibrated configuration for the Volcano PV .035 / s5i probe lives in [`instrument-calibration/p035_visions/volcano_s5i.yaml`](../../../instrument-calibration/p035_visions/volcano_s5i.yaml); this document describes the simulator runtime that consumes it.

---

## 1. Architecture overview

IVUS imaging uses a small transducer at the centre of a vessel to acquire a 360° radial cross-section. Depth is radial (lumen → wall → perivascular tissue); the "lateral" dimension is angular. The simulator models this end-to-end on the GPU; the 4-stage pipeline diagram in the technical guide is the canonical reference and applies here as well.

What is IVUS-specific in this package:

1. A dedicated probe class (`IVUSProbe`) and `ProbeType::PROBE_TYPE_IVUS` enum value.
2. OptiX ray generation that produces one ray per angular sample from a single origin (catheter centre), sweeping 0 → 2π in the probe's xz-plane.
3. Vascular materials (`lumen`, `vessel_wall`, `extravascular`) plus a `tungsten` material for the bench wire-spiral phantom.
4. A causal, Hanning-windowed axial PSF and a depth-dependent lateral PSF based on a Gaussian-beam model.
5. Probe-type-specific TGC (mm-scale depth range and gain) cached separately from abdominal/curvilinear TGC.
6. An unwrapped scan conversion (x = angle 0–360°, y = depth 0–`t_far` mm).
7. Finer scattering texture resolution and pipeline-parameter overrides via `SimParams`.
8. A processing chain that adds the bench-calibrated RF-domain stages (RF noise floor with depth weighting, reference gain, ring-down injection, palette clamp, catheter dead-zone) needed to match clinical IVUS device output.
9. A Python configuration layer (`raysim.config.IvusSimConfig`) that maps a YAML schema directly onto `SimParams` and the probe.

The remainder of this document walks through each of these components.

---

## 2. Probe type and `IVUSProbe` class

**Files:** [`include/raysim/core/probe_types.hpp`](../include/raysim/core/probe_types.hpp), [`include/raysim/core/probe.hpp`](../include/raysim/core/probe.hpp), [`include/raysim/core/ivus_probe.hpp`](../include/raysim/core/ivus_probe.hpp).

### 2.1 `ProbeType` enum

`ProbeType::PROBE_TYPE_IVUS = 3` joins curvilinear, linear-array, and phased-array. The simulator branches on probe type to select IVUS-specific physics (causal axial PSF, depth-dependent lateral PSF, IVUS TGC, unwrapped scan conversion).

### 2.2 `BaseProbe` accessors

`BaseProbe` exposes two virtual accessors used by the depth-dependent lateral PSF:

- `get_element_radius_mm()` — element radius in mm (focused single-element probes).
- `get_focal_length_mm()` — focal length in mm.

Both default to `0` so non-IVUS probes are unaffected. When both are positive the simulator builds a 2D depth-dependent lateral PSF (Gaussian-beam model; see §5.2).

### 2.3 `IVUSProbe`

`IVUSProbe` is header-only and extends `BaseProbe`:

| Field | Value |
|---|---|
| `get_local_element_position()` | `(0, 0, 0)` (catheter centre) |
| `get_local_element_direction(idx)` | unit vector in the xz-plane at angle `2π · idx / num_elements`, `+z` at angle 0 |
| Sector angle | 360° |
| Width / radius | 0 (point source) |
| Probe type | `PROBE_TYPE_IVUS` |

Construction parameters: `num_angular_rays` (stored as `num_elements_x_`), `frequency` (MHz; typical IVUS values 20–40 MHz), `elevational_height` (often 0), `element_radius_mm` (typical 0.6), `focal_length_mm` (typical 4), and the standard `Pose`. The finite physical aperture is represented downstream through the depth-dependent lateral PSF rather than via the ray geometry.

---

## 3. Materials

**File:** [`csrc/core/material.cpp`](../csrc/core/material.cpp). `Materials` is the registry; `Materials::get_index(name)` returns the index used when adding geometry to a `World`.

### 3.1 Tissue materials

| Material | Z (MRayl) | α (dB·cm⁻¹·MHz⁻¹) | c (m/s) | Notes |
|---|---:|---:|---:|---|
| `water` | 1.48 | 0.002 | 1480 | reference background |
| `blood` / `lumen` | 1.68 | 0.2 | 1584 | literature values; `lumen` is the IVUS alias |
| `vessel_wall` | 1.82 | ≈ 1.0 | 1571 | intima/media at vascular 50 MHz |
| `extravascular` | 1.62 | 0.7 | 1547 | muscle-like perivascular tissue |
| `fat`, `liver`, `muscle`, `bone` | — | — | — | abdominal materials |

The vascular materials are sized so that (a) the Fresnel coefficients at the lumen↔wall and wall↔extravascular interfaces match published vascular ultrasound data and (b) Beer–Lambert attenuation produces the correct depth dependence at 40 MHz IVUS frequencies.

Citations in the source comments: Goss et al. compilations, PMC3570716 (Ultrasound Med Biol 2013), PMC5126009 (J Ultrasound 2016), Lockwood et al. UMB 17(7) 1991.

### 3.2 Phantom hardware

`tungsten`: Z ≈ 101 MRayl, c = 5200 m/s, α = 15 dB·cm⁻¹·MHz⁻¹, `mu0_ = sigma_ = specularity_ = 0.05`. Used for the 30 µm tungsten wires in the bench wire-spiral phantom. The Fresnel coefficient against water/milk is ≈ 0.96, which is what the calibration pipeline anchors against. The very high attenuation is harmless because the bulk propagation path through a 30 µm wire is negligible; it only affects grazing rays that travel along the wire surface.

---

## 4. OptiX raytracing

**Files:** [`csrc/cuda/optix_trace.cu`](../csrc/cuda/optix_trace.cu), [`include/raysim/cuda/optix_trace.hpp`](../include/raysim/cuda/optix_trace.hpp).

### 4.1 IVUS ray generation

`generate_ivus_probe_ray_local` maps the launch dimension `d_x` to an angle in `[0, 2π]`, sets the ray origin to `(0, 0, 0)` in local coordinates, and emits a unit direction `(sin(angle), 0, cos(angle))` in the xz-plane (matching `IVUSProbe::get_local_element_direction`). The raygen entry point switches on `probe_type` and dispatches to this function for `PROBE_TYPE_IVUS`. Elevation handling is shared with the other probe types and is typically zero for 2D IVUS.

`ray.t_ancestors` is explicitly initialised to 0 in the raygen so every ray starts with a correct cumulative path length, which the scatter and hit stages use to compute depth bins.

### 4.2 Scattering

`get_scattering_value(pos, material)` samples the world's `float2` 3D scattering texture in WRAP addressing mode at `pos / params.scattering_resolution_mm`. The resolution is set per run from `SimParams::scattering_resolution_mm` (10 mm for IVUS, 50 mm for abdominal — auto-selected by probe type when the field is left at its sentinel 0). The first channel is a uniform-distribution density gate against `material.mu0_`; the second is a Gaussian-distributed amplitude scaled by `material.sigma_`.

**Per-scanline angular decorrelation.** At small radii the angular sampling rate is sub-voxel (arc length ≈ 0.025 mm per scanline at r = 1 mm with 256 scanlines), so adjacent scanlines would otherwise sample correlated points in the scattering texture. The simulator adds a per-scanline pseudo-random offset to the texture coordinate before lookup:

```cpp
if (params.scatter_angular_decorrelate) {
  const uint32_t base = ray_index * 3u + params.frame_seed * 2654435761u;
  pos.x += pcg_to_unit_float(pcg_hash(base + 0u)) * 4096.f;
  pos.y += pcg_to_unit_float(pcg_hash(base + 1u)) * 4096.f;
  pos.z += pcg_to_unit_float(pcg_hash(base + 2u)) * 4096.f;
}
```

The hash is a standard PCG mix (Jarzynski & Olano 2020; O'Neill 2014). All depth samples *along* a single scanline share the same offset, so the axial scatter integral remains coherent (preserving wire/sphere PSFs); only the angular dimension is decorrelated. The `4096×` scaling places adjacent scanlines into well-separated regions of the wrapped texture. `frame_seed` is mixed in so successive frames draw independent speckle realisations — required for temporal averaging in the calibration evaluations.

**Scatter accumulation** (`sample_intensities`):

```text
bin   = get_intensity_offset(t_ancestors + t_val)
scanline[bin] += segment_weight * scatter_val
```

Indexing by true depth (`t_ancestors + t_val`) rather than by a step counter is essential — step-indexed writes produce axial streaks because the write index decouples from the round-trip depth. `params.scatter_integral_scale` (default 40) multiplies the line integral so reflection-dominated phantoms (wire) and scatter-dominated phantoms (tissue) both render in a displayable range. `params.disable_scatter` is an early-out for diagnostic renders.

### 4.3 Reflection and refraction

**Oblique-incidence Fresnel coefficient:**

\[R = \left(\frac{Z_2/\cos\theta_t - Z_1/\cos\theta_i}{Z_2/\cos\theta_t + Z_1/\cos\theta_i}\right)^2\]

with Snell's law `sin θ_t = (v_1/v_2) sin θ_i` providing `cos θ_t`. Grazing rays and total internal reflection (`cos ≤ 1e-6`) return `R = 1`. The pressure form (rather than normal-incidence Z·cos θ) is required for IVUS rays hitting the vessel wall at varying angles.

**Hit contribution at an interface.** Both the physical reflection and the Mattausch-2016 empirical directivity term are accumulated into the same depth bin:

```cpp
const uint32_t hit_bin = get_intensity_offset(ray.t_ancestors + t);
scanline[hit_bin] += reflected_intensity;             // R · I
scanline[hit_bin] += 2.f * specular_reflection * R;   // empirical directivity, Fresnel-scaled
```

The `cos^n` term represents the angular **distribution** of energy at a non-mirror interface, not an independent intensity channel. Multiplying it by `R` ties the empirical contribution to the physical Fresnel envelope so the total reflected intensity at any interface is bounded by R. This is essential for soft-tissue interfaces (`R ≈ 0.002` at lumen↔vessel_wall) where an un-scaled empirical term would dominate the physical echo by orders of magnitude and saturate every tissue boundary. The remaining `2.f` factor is an empirical brightness scale; lower it (toward 1.0) for a more diffuse interface look or raise it for stronger highlights.

**Refracted ray.** The refracted-ray origin is `back_start` (just inside the second medium) and `t_min = 1e-3 mm` is used to avoid self-intersection with the same interface. Always using `back_start` (rather than picking `front_start` or `back_start` from refraction direction) guarantees the refracted ray propagates in the correct medium.

### 4.4 OptiX `Params` struct

The kernel `Params` struct ([`include/raysim/cuda/optix_trace.hpp`](../include/raysim/cuda/optix_trace.hpp)) carries:

- `scattering_resolution_mm` — voxel size for the scattering texture lookup.
- `disable_scatter` — non-zero disables scatter accumulation.
- `scatter_integral_scale` — multiplier on the scatter line integral (`0` = strict integration; `40` = displayable tissue background).
- `scatter_angular_decorrelate` — toggles the per-scanline jitter described in §4.2.
- `frame_seed` — used for both the scatter decorrelation hash and the additive-noise PCG seed.
- `source_frequency`, `contact_epsilon`, and the OptiX traversable handle.

All numeric fields are populated from `SimParams` (with probe-type-based fallbacks for the sentinel `0.0` values).

---

## 5. RF / B-mode processing pipeline

**Files:** [`csrc/core/raytracing_ultrasound_simulator.cpp`](../csrc/core/raytracing_ultrasound_simulator.cpp), [`include/raysim/core/raytracing_ultrasound_simulator.hpp`](../include/raysim/core/raytracing_ultrasound_simulator.hpp), [`csrc/cuda/cuda_algorithms.cu`](../csrc/cuda/cuda_algorithms.cu).

`RaytracingUltrasoundSimulator::simulate()` runs the stages below in the listed order. Each stage is a no-op when its toggle / parameter is at its default; default `SimParams()` reproduces a probe-agnostic baseline pipeline byte-for-byte.

### 5.1 Pre-PSF additive RF noise (depth-weighted)

Stage: between scattering accumulation and PSF convolution.

When `noise_sigma > 0` a CUDA kernel (`add_gaussian_noise_depth_weighted` / `add_gaussian_noise`) adds Gaussian noise to the raw RF buffer. The noise is drawn from `N(0, sigma²)` via Box–Muller seeded with a PCG hash of `(row, col, frame_seed)`. Pre-PSF placement means the same axial + lateral PSF convolutions that shape the scatter signal also shape the noise, which matches the bench's bandlimited receiver noise (mottled speckle) rather than per-pixel static.

The noise is **depth-weighted** so the post-PSF noise standard deviation is uniform across depth. Without weighting, the L1-normalised depth-dependent lateral PSF would otherwise amplify focal-zone noise variance, producing a focal-zone hump in the anechoic-region depth profile. The weight is

\[w(z) = \sqrt{\frac{\sigma_\text{bins}(z)}{\sigma_\text{bins}(z_\text{focal})}}\]

where `σ_bins(z)` is the angular sigma of the lateral PSF at depth `z`. With this weight, σ is specified in **input-RF units at the focal depth** (pre-gain, pre-TGC, pre-PSF); the calibration script `derive_noise_sigma.py` bisects against the bench gain-54 anechoic palette mean/std to land on the calibrated value (≈ 7.1 × 10⁻⁴ for PV .035).

### 5.2 PSF convolution

**Axial.** For IVUS, `create_ivus_axial_psf_causal` builds a causal, Hanning-windowed one-sided kernel with extent derived from pulse duration and wavelength. The causal kernel ensures a strong wall echo only smears deeper (later in time) and never backwards into the lumen, preserving the anechoic lumen appearance. For other probe types `create_gaussian_psf` produces a symmetric Gaussian kernel. The axial PSF is cached and invalidated when probe type or frequency changes.

**Lateral.** When `element_radius_mm > 0` and `focal_length_mm > 0`, the simulator builds a 2D depth-dependent lateral kernel (depth_bins × kernel_len) using a Gaussian-beam model:

- Beam waist at focus: `w_0 = λ · F / (2 a)` (with `λ` the centre wavelength, `F` the focal length, `a` the element radius).
- Rayleigh length: `z_R = π w_0² / λ`.
- Beam radius at depth `z`: `w(z) = w_0 · √(1 + (z / z_R)²)`.

Two corrections are applied to the textbook model so it matches synthetic-aperture IVUS behaviour:

1. **Pre-focal clamp.** The textbook Gaussian-beam radius is symmetric about the focus, predicting a wide beam at depths `r ≪ focal_length` (≈ 2.8 mm beam radius at r = 1 mm with the PV .035 geometry). Bench imagery shows the opposite — angular FWHM is roughly constant with depth at depths shallower than the focus because a rotating-element IVUS coherently sums over a narrow beam bounded by element directivity. The simulator clamps the pre-focal contribution to zero:

   ```cpp
   const float z_post = (z > 0.f) ? z : 0.f;
   const float sigma_mm = w0_mm * std::sqrt(1.f + (z_post * z_post) / (z_R_mm * z_R_mm));
   ```

   so `σ_mm == w_0` for any depth `r ≤ focal_length` and the textbook expansion only applies post-focal where it is meaningful.

2. **Cyclic angular convolution, wide kernel.** IVUS angles wrap 360° (= `num_scanlines`), so the lateral convolution kernel wraps source row indices via `((iy + k) % N + N) % N` rather than truncating at the edges. The kernel radius is `num_scanlines / 2` so the kernel can span the full half-circumference at any depth without truncation. L1 normalisation (`sum = 1`) is used; L2 normalisation amplifies the residual sub-voxel scatter correlation at small radii and breaks the established `gain_db` calibration.

When `element_radius_mm` and `focal_length_mm` are both zero, the 2D path is bypassed and a 1D lateral kernel with a safe fallback width is used.

**Elevational.** Standard Gaussian PSF across the elevational planes followed by averaging. For 2D IVUS `num_el_samples = 1` and this stage is a no-op.

The lateral PSF is invalidated when the probe frequency, element radius, focal length, `buffer_size`, or `t_far` change.

The CUDA implementation is in `convolve_columns_depth_dependent_kernel` (lateral 2D) and `convolve_columns` (lateral 1D); the axial convolution is `convolve_rows`.

### 5.3 Time-gain compensation

A piecewise-linear gain curve indexed by depth. Two paths:

- **Probe-type default** (`SimParams.tgc_control_points` empty): IVUS uses `{(0, 0), (1, tgc_dB_per_cm)}` with `tgc_dB_per_cm = 2 dB/cm`; abdominal uses `{(0, 0), (40, 28)}`. Cached per probe type so switching probes rebuilds the curve.
- **User-supplied** (list non-empty): the simulator rebuilds the curve every frame from the supplied `(depth_cm, gain_db)` pairs. The probe-type cache is invalidated so a later frame that reverts to the empty path re-builds the default schedule.

The calibrated PV .035 YAML uses five control points spanning the 0–3 cm IVUS range; see `instrument-calibration/p035_visions/volcano_s5i.yaml`.

### 5.4 Reference gain (`gain_db`)

Applied pre-Hilbert (between TGC and ring-down injection) as `rf ← rf · 10^(gain_db / 20)`. The CUDA wrapper short-circuits on `gain_db == 0` so default callers pay no kernel-launch cost.

`gain_db` lumps two physically distinct effects into one calibrated scalar:

1. The bench's **slider-gain offset** (the device's user-facing gain control; for the PV .035, slider 54 maps to 0 dB).
2. A **renderer-specific reference-amplitude offset** that puts the simulator's raw envelope amplitudes onto the bench's calibrated scale (the bench reference amplitude is "amp at slider 54").

Pre-Hilbert placement is required because the ring-down template (§5.5) is specified in bench-calibrated envelope units; applying `gain_db` after ring-down would double-scale the ring-down by `10^(gain_db/20)`. By linearity of the Hilbert transform (`|H(s · rf)| = s · |H(rf)|`), scaling RF pre-Hilbert is mathematically equivalent to scaling envelope post-Hilbert for the scattering signal, so the pure-amplitude derivation in `derive_gain_db.py` is unaffected by the placement.

The PV .035 calibration anchors `gain_db` against the bench wire-peak target; see `instrument-calibration/p035_visions/derive_gain_db.py`.

### 5.5 Ring-down injection (post-Hilbert)

The catheter sheath produces a near-field ring-down signature that the simulator emits as a post-Hilbert envelope template. The stage runs between envelope detection and log compression so the cuFFTDx Hilbert transform — which is a length-`buffer_size` cyclic FFT — never sees the ring-down transient. (If the ring-down were added pre-Hilbert, spectral side lobes would smear it across the whole buffer.)

`SimParams::ring_down` carries a `RingDownParams` struct:

| Field | Meaning |
|---|---|
| `enabled` | toggle |
| `amplitude` | overall scale (envelope-amp units) |
| `extent_mm` | radial truncation (samples beyond this are untouched) |
| `decay` | `"measured"` (use the supplied waveform verbatim) or `"exponential"` |
| `waveform` | 1D numpy float32 array, envelope amplitudes |

The host stores the resolved template as `std::vector<float>` in envelope-amp units, uploads it once per change, and adds it row-wise to the envelope buffer with `add_row` (broadcast vector add along depth). The Python loader (`raysim.config.IvusSimConfig`) resolves the `.npy` template relative to the workspace root, subtracts the bench speckle floor, converts palette → envelope amplitude via `log_multiplier`, and resamples from the device's display pitch (0.12 mm/sample for PV .035) onto the simulator's `t_far / buffer_size` pitch. Cache invalidation keys on `(decay, amplitude, extent_mm, buffer_size, host_pointer, host_size)` so callers can swap templates frame-to-frame.

The OptiX raygen maps `[0, t_far]` mm onto `buffer_size` samples (`offset = round(t / t_far · (buffer_size - 1))`), so the spatial pitch in the scanlines is `t_far / buffer_size` mm/sample — *not* `c / SAMPLING_FREQ / 2`. Both the C++ ring-down stage and the Python loader's resampler use this convention.

### 5.6 Post-envelope Gaussian noise

A small additive noise term applied to the envelope after Hilbert (and after ring-down injection) and before log compression. Models residual sensor noise that survives envelope detection in real systems. Configured via `SimParams::envelope_noise` (`EnvelopeNoiseConfig`: `mean`, `sigma`, `gain_db_ref`, `frame_seed`, `scale_with_gain`).

When `scale_with_gain == true`, both `mean` and `sigma` track the receive-chain gain by scaling with `10^((gain_db - gain_db_ref) / 20)`. This matches how bench-measured noise references at slider 54 and then scales with the device's gain knob.

### 5.7 Post-Hilbert radial low-pass

Optional Gaussian low-pass filter along the radial axis (`SimParams::post_hilbert_lpf`). Smooths residual high-frequency ringing left in the envelope signal after Hilbert.

### 5.8 Log compression

`20 · log10(max(envelope, log_floor)) · log_multiplier / 20` — i.e., a fixed-reference mapping from envelope amplitude to palette value:

\[\text{palette} = \log_{10}\left(\frac{\max(\text{amp}, \text{log\_floor})}{1}\right) \cdot \text{log\_multiplier}\]

`amp == log_floor` lands at palette 0 by construction. `amp < log_floor` produces *negative* palette values that the display-window stage (§5.9) clamps to the device's reject palette. Two epsilon clamps inside the kernel make `amp == 0` / `log_floor == 0` produce a finite, very-negative palette value (`~ -30 · log_multiplier`) instead of NaN/-∞.

The default `log_floor = 1.0` and `log_multiplier = 20.0` put the post-log palette in roughly `[-60, 0]` for envelope amplitudes in `[1e-3, 1]` — the range the default `MIN_VAL/MAX_VAL` in `examples/ivus_example.py` expects.

### 5.9 Display window — palette clamp

`apply_display_window` clamps each sample to `[reject_palette, saturation_palette]`:

```cpp
buffer[offset] = fminf(fmaxf(buffer[offset], reject_palette), saturation_palette);
```

The kernel is skipped when `saturation_palette <= reject_palette` (the default `0/0` configuration). The PV .035 YAML reads these values straight out of the device's `gain_lut.json` (`reject_palette: 11.0`, `saturation_palette: 239.0`).

A softplus-smoothed variant (`apply_display_window_softplus`) is also available, controlled by `SimParams::display_softplus_scale`. When the scale is positive the reject floor is smoothed by a softplus so the transition from "noise pixels below reject" to "noise pixels at reject" is differentiable rather than discontinuous.

### 5.10 Median clip

A 5×1 median filter on the post-display-window image, clamped to `[median_clip_d_min_db, median_clip_d_max_db]` (defaults `-60`/`0`). Reduces salt-and-pepper artefacts from the log/display chain while preserving structure.

### 5.11 Catheter dead-zone

A radial zeroing kernel (`zero_inner_radial_kernel`) sets every sample at `r < catheter_dead_zone_mm` to palette 0. Applied at the very end of the pipeline (post log-compression / display window / median clip), so the dead-zone is solid black (palette 0) rather than reject-floor (palette 11) — matching the bench's solid-black inner zone. Calibrated to 1.4 mm for PV .035 from `tier1_results/figures/ringdown_inner_zone_paired.png`.

### 5.12 Scan conversion and display bounds

IVUS scan conversion (`scan_convert_ivus`) takes the (depth_norm, angle_norm) scanline buffer through a CUDA texture and resamples to an unwrapped 2D buffer with horizontal axis = angle, vertical axis = depth. The simulator exposes the display bounds via `get_min_x()` / `get_max_x()` / `get_min_z()` / `get_max_z()`; for IVUS these are `(0, 360, 0, t_far_mm)`.

For polar display (clinical IVUS view) the unwrapped image is resampled by the client (see `examples/ivus_example.py::save_polar_frame` for a reference implementation).

---

## 6. Python bindings and configuration layer

**Files:** [`csrc/python/raysim_bindings.cpp`](../csrc/python/raysim_bindings.cpp), [`raysim/__init__.py`](../raysim/__init__.py), [`raysim/cuda/__init__.py`](../raysim/cuda/__init__.py), [`raysim/config.py`](../raysim/config.py).

### 6.1 pybind11 classes

| Class | Purpose |
|---|---|
| `IVUSProbe` | Bound directly; defaults `num_angular_rays=256`, `frequency=40`, `element_radius_mm=0.6`, `focal_length_mm=4`. |
| `TgcControlPoint` | `(depth_cm, gain_db)`; constructible from a 2-tuple. |
| `RingDownParams` | All ring-down fields including a numpy-backed `waveform`. |
| `Materials` | Updated docstring lists the IVUS materials. |
| `SimParams` | All RF/B-mode knobs exposed via `def_readwrite`. |

`SimParams` exposes (in addition to the legacy fields): `tgc_control_points`, `log_multiplier`, `log_floor`, `median_clip_size`, `median_clip_d_min_db`, `median_clip_d_max_db`, `scattering_resolution_mm`, `scatter_integral_scale`, `disable_scatter`, `scatter_angular_decorrelate`, `frame_seed`, `noise_sigma`, `noise_focal_depth_mm`, `catheter_dead_zone_mm`, `envelope_noise`, `display_softplus_scale`, `ring_down`, `reject_palette`, `saturation_palette`, `gain_db`, `lateral_psf_kernel_type`, `lateral_psf_constant_sigma_rad`, `ivus_angular_subsampling`.

### 6.2 `raysim.config.IvusSimConfig`

`IvusSimConfig.from_yaml(path)` loads the calibrated PV .035 YAML and returns a config dataclass. `to_sim_params()` produces a `SimParams` instance with every field populated; `build_probe(pose)` constructs an `IVUSProbe` from the probe block; `materials()` returns the matching `Materials` registry.

The YAML schema mirrors `SimParams` field-by-field:

| YAML key | `SimParams` field |
|---|---|
| `processing.gain_db` | `gain_db` |
| `processing.tgc_control_points` | `tgc_control_points` (list of `TgcControlPoint`) |
| `processing.log_multiplier` / `log_floor` | `log_multiplier` / `log_floor` |
| `processing.median_clip.{size, d_min_db, d_max_db}` | `median_clip_size` / `median_clip_d_min_db` / `median_clip_d_max_db` |
| `processing.scattering_resolution_mm` | `scattering_resolution_mm` (`0.0` ⇒ probe-type auto) |
| `processing.scatter_integral_scale` | `scatter_integral_scale` |
| `processing.scatter_angular_decorrelate` / `frame_seed` | `scatter_angular_decorrelate` / `frame_seed` |
| `processing.noise.{type, sigma}` | `noise_sigma` (`type=gaussian` wired; `none` ⇒ 0) |
| `processing.noise.focal_depth_mm` | `noise_focal_depth_mm` |
| `processing.envelope_noise.*` | `envelope_noise` |
| `processing.ring_down.*` | `ring_down` (`waveform` resolved from `waveform_path`) |
| `processing.catheter.dead_zone_mm` | `catheter_dead_zone_mm` |
| `display.{reject_palette, saturation_palette}` | `reject_palette` / `saturation_palette` |
| `display.softplus_scale` | `display_softplus_scale` |
| `processing.lateral_psf.kernel_type` / `constant_sigma_rad` | `lateral_psf_kernel_type` / `lateral_psf_constant_sigma_rad` |
| `probe.{frequency, element_radius_mm, focal_length_mm, num_angular_rays, ...}` | `IVUSProbe` constructor args |

YAML values that are absent or `null` fall back to the corresponding `SimParams` default. Setting `scattering_resolution_mm: 0.0` keeps the C++ probe-type sentinel (10 mm for IVUS).

---

## 7. Vessel phantom utilities

**File:** [`utils/phantom_maker.py`](../utils/phantom_maker.py).

`generate_cylinder_mesh` writes an open cylinder OBJ (no caps), axis along Y, cross-section in xz; optional **inward normals** so rays from the lumen hit the front face. Default 129 segments to avoid aliasing with 256 IVUS scanlines.

`generate_cylinder_thick_mesh` writes `Cylinder_inner.obj` and `Cylinder_outer.obj` for a thick vessel wall (same segment count and inward normals).

CLI: `python utils/phantom_maker.py cylinder --output mesh [--cylinder-thick]`. Both files ship pre-generated in `mesh/`.

---

## 8. Evaluation

Three example scripts exercise the IVUS implementation end-to-end:

- [`examples/ivus_example.py`](../examples/ivus_example.py) — thick-walled cylinder phantom; tests geometry, interface echoes, and attenuation. Expected unwrapped image: two concentric bright rings at the lumen/wall and wall/extravascular interfaces.
- [`examples/wire_phantom_evaluation.py`](../examples/wire_phantom_evaluation.py) — five 30 µm tungsten wires at 1–5 mm in a spiral; tests resolution and geometric accuracy. Expected unwrapped image: five bright spots in a spiral pattern.
- [`examples/cystic_resolution_phantom_evaluation.py`](../examples/cystic_resolution_phantom_evaluation.py) — tissue background with fluid cysts; tests contrast and scatter/TGC. Expected unwrapped image: speckled tissue with five darker cyst regions.

For per-instrument calibration the canonical evaluations live alongside the `p035_visions` calibration pipeline:

- **Tier 1 (parameter-bank acceptance):** [`instrument-calibration/p035_visions/tier1_results/tier1_results.md`](../../../instrument-calibration/p035_visions/tier1_results/tier1_results.md) — 10 quantitative gates regenerated by `tier1_evaluation.py`.
- **Vessel scenarios:** [`instrument-calibration/p035_visions/vessel_evaluation_output/VESSEL_EVALUATION_REPORT.md`](../../../instrument-calibration/p035_visions/vessel_evaluation_output/VESSEL_EVALUATION_REPORT.md) — seven canonical vessel cases rendered by `vessel_evaluation.py`.

---

## 9. Known limitations / not modelled

### 9.1 Frequency dependence of scattering

Scatter strength is modulated by `material.sigma_` and path-length attenuation but has **no explicit frequency dependence** (e.g. f⁴ for Rayleigh). Changing centre frequency changes attenuation and beam width but not the inherent scattering vs frequency. Relevant when comparing 20 vs 40 MHz acquisitions or matching multi-frequency clinical data. *Possible follow-up:* add a frequency-dependent term `σ(f) ∝ f^k` per material or globally.

### 9.2 Wire-vs-bg contrast in the OptiX scatter integral

The bench's wire-vs-bg amplitude contrast (~26 dB) is much smaller than the simulator's (~60–70 dB). `gain_db` anchored on the bench background plus the calibrated noise floor reproduces the device's anechoic palette mean/std and reject behaviour, but bench frames with stronger wire-vs-bg contrast (mid-radius wires at palette ~150 against bg ~46) saturate against `saturation_palette = 239` in the simulator. Closing the gap is a scattering-strength problem (sphere-as-wire primitive, sub-wavelength target representation) tracked in `instrument-calibration/p035_visions/calibration_delta.md`.

### 9.3 Element directivity at transmit

Ray intensity starts at 1.0; **element directivity is only applied in the lateral PSF** (receive-side blur). Transmit directivity (e.g. angular sensitivity of the single element) is not weighted onto the ray contribution. For a rotating single element this affects angular uniformity of sensitivity. *Possible follow-up:* apply an angular weighting (from element size and frequency) to the transmit ray contribution so both transmit and receive directivity are represented; the depth-weighted pre-PSF noise (§5.1) would then need to track the receive-only aperture rather than the combined TX·RX product.

### 9.4 Other electronic / acquisition effects

- **Rotation and motion:** the simulator renders "all angles at once". Real IVUS uses a rotating element, so rotation blur and per-frame motion artefacts are not represented. Adequate for static phantoms; relevant for moving vessels or pullback validation.
- **Speed-of-sound heterogeneity:** refraction at *interfaces* is correct, but rays are straight between interfaces. Smooth `c` variations would bend rays. Usually a second-order effect for small vessels.
- **Multiple scattering:** only single scattering is modelled in the line integral. Adequate for most vascular imaging; relevant for very heterogeneous tissue.
- **Reverberation / multipath / mode conversion:** none of these are modelled. Can add clutter in real IVUS but rarely changes coarse interpretation.

These are tracked as candidates for future work; none of them is required for the PV .035 calibration to reach its current state.

---

## 10. File map

| Area | Files |
|---|---|
| Probe type | [`include/raysim/core/probe_types.hpp`](../include/raysim/core/probe_types.hpp), [`include/raysim/core/probe.hpp`](../include/raysim/core/probe.hpp), [`include/raysim/core/ivus_probe.hpp`](../include/raysim/core/ivus_probe.hpp) |
| Materials | [`csrc/core/material.cpp`](../csrc/core/material.cpp) |
| Ray / scattering | [`csrc/cuda/optix_trace.cu`](../csrc/cuda/optix_trace.cu), [`include/raysim/cuda/optix_trace.hpp`](../include/raysim/cuda/optix_trace.hpp) |
| Simulator / PSF / TGC / pipeline | [`csrc/core/raytracing_ultrasound_simulator.cpp`](../csrc/core/raytracing_ultrasound_simulator.cpp), [`include/raysim/core/raytracing_ultrasound_simulator.hpp`](../include/raysim/core/raytracing_ultrasound_simulator.hpp) |
| CUDA kernels (PSF, noise, log, display, scan conversion) | [`csrc/cuda/cuda_algorithms.cu`](../csrc/cuda/cuda_algorithms.cu), [`include/raysim/cuda/cuda_algorithms.hpp`](../include/raysim/cuda/cuda_algorithms.hpp) |
| Python bindings & exports | [`csrc/python/raysim_bindings.cpp`](../csrc/python/raysim_bindings.cpp), [`raysim/__init__.py`](../raysim/__init__.py), [`raysim/cuda/__init__.py`](../raysim/cuda/__init__.py) |
| Python configuration layer | [`raysim/config.py`](../raysim/config.py) |
| Vessel phantom utilities | [`utils/phantom_maker.py`](../utils/phantom_maker.py) |
| Per-instrument calibrated config | [`instrument-calibration/p035_visions/volcano_s5i.yaml`](../../../instrument-calibration/p035_visions/volcano_s5i.yaml) |
