/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPP_RAYTRACING_ULTRASOUND_SIMULATOR
#define CPP_RAYTRACING_ULTRASOUND_SIMULATOR

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "raysim/core/probe_types.hpp"
#include "raysim/cuda/cuda_helper.hpp"
#include "raysim/cuda/optix_helper.hpp"

namespace raysim {

class World;
class Materials;
class BaseProbe;
class CUDAAlgorithms;

/// Public-API control point for the piece-wise linear TGC curve. Mirrors the file-local
/// `ControlPoint` used inside the simulator implementation, but lives in the public
/// namespace so callers (Python bindings, host code) can build a TGC schedule from a
/// config file. Empty `SimParams::tgc_control_points` => use the simulator's probe-type
/// default (preserves backward compatibility).
struct TgcControlPoint {
  float depth_cm = 0.f;
  float gain_db = 0.f;
};

/// Calibrated ring-down injection knobs.
///
/// Models the residual ring-down signal that survives the device's Acoustic Reference
/// subtraction. When `enabled == false` the simulator emits no ring-down at all
/// (default; matches the "truly quiet lumen" semantics from the calibration handoff).
/// When `enabled == true` a depth-only waveform is added to every angular A-line
/// before envelope detection so the lumen looks like a real PV .035 frame.
///
/// `decay`:
///   * "exponential": per-sample envelope = amplitude * exp(-r/decay_length); decay_length
///     defaults to extent_mm/3 so amplitude has dropped to ~5% by `extent_mm`.
///   * "hanning":     a half-cosine window of length extent_mm scaled by amplitude.
///   * "measured":    use `waveform` directly (already in envelope-amplitude units; the
///                    YAML loader converts the palette template via
///                    amp = 10^(palette / log_multiplier)).
///
/// In all cases the waveform is truncated to zero past `extent_mm` so deeper structures
/// are unaffected. See `ivus_implementation_writeup.md` §11 for the calibration
/// provenance and the device-side AR subtraction modeling decisions.
struct RingDownParams {
  bool enabled = false;
  float amplitude = 0.f;            // envelope amplitude at simulator reference gain
  float extent_mm = 0.5f;           // hard cutoff in radial mm
  std::string decay = "exponential";  // exponential | hanning | measured
  std::vector<float> waveform;      // populated by host code from .npy when decay == "measured"
};

class RaytracingUltrasoundSimulator {
 public:
  /**
   * Construct a new Raytracing Unltrasound Simulator object
   *
   * @param world World object containing scene geometry
   * @param materials Materials
   */
  explicit RaytracingUltrasoundSimulator(World* world, const Materials* materials);
  RaytracingUltrasoundSimulator() = delete;

  /// Simulation parameters
  struct SimParams {
    float t_far = 180.f;              // Maximum ray distance [mm]
    uint32_t buffer_size = 4096;      // Samples per line
    uint32_t max_depth = 15;          // Maximum reflection depth
    float min_intensity = 1e-3f;      // Minimum intensity threshold
    bool use_scattering = true;       // Enable scattering simulation
    bool conv_psf = false;            // Convolve the raw hits with the PSF
    bool median_clip_filter = false;  // Enable median clip filter for speckle reduction
    uint2 b_mode_size = make_uint2(500, 500);
    cudaStream_t stream = cudaStreamPerThread;  // CUDA stream
    bool enable_cuda_timing = false;            // Print timing of CUDA operations
    bool write_debug_images = false;            // Write debug images to `debug_images` directory
    float contact_epsilon = 0.0f;               // Maximum distance for element activation [mm]

    // -------------------------------------------------------------------------
    // RF / B-mode processing parameters. Defaults are chosen so callers that
    // construct `SimParams()` and only set the legacy fields produce a
    // baseline pipeline unchanged from the probe-agnostic original:
    //   * empty tgc_control_points => probe-type default schedule
    //   * scattering_resolution_mm == 0 => probe-type default (10 IVUS / 50 other)
    //   * remaining defaults match the literals previously baked into the kernels.
    // -------------------------------------------------------------------------
    std::vector<TgcControlPoint> tgc_control_points;  // (depth_cm, gain_db); empty = auto

    // Log compression: out = log_multiplier * log10(amp / log_floor).
    // `log_floor` is the calibration anchor (amp == log_floor maps to palette 0).
    // Default `log_floor = 1.f` makes the mapping reduce to
    // `log_multiplier * log10(amp)` for amp >= 1, which matches the baseline
    // `examples/ivus_example.py` MIN_VAL/MAX_VAL = (-60, 0) display window
    // when `log_multiplier = 20`. Calibrated YAMLs override this with the
    // bench's anchor (e.g. PV .035 uses log_floor = 1.0, log_multiplier = 112.3).
    float log_multiplier = 20.f;
    float log_floor = 1.f;

    // Median clip filter (only used when median_clip_filter == true)
    uint32_t median_clip_size = 5;        // square kernel side
    float median_clip_d_min_db = -60.f;   // clamp floor (post log compression)
    float median_clip_d_max_db = 0.f;     // clamp ceiling

    // Scatter pipeline params (mirrored into raysim::Params for the OptiX kernel).
    // 0.f means "auto from probe type" so this knob is purely additive.
    float scattering_resolution_mm = 0.f;
    float scatter_integral_scale = 40.f;
    bool disable_scatter = false;

    // Per-scanline angular decorrelation of the scatter texture lookup.
    //
    // `scatter_angular_decorrelate == true`: each scanline (`ray_index`) adds a
    // pseudo-random offset to the scatter texture coordinate before sampling
    // (see `optix_trace.cu::get_scattering_value`). Adjacent angular bins
    // therefore sample independent regions of the 256³ wrap-mode texture
    // instead of a single trilinearly-interpolated voxel. This is the fix for
    // the near-field bright shoulder produced by the depth-dependent lateral
    // PSF (Tier 1 test I, see ivus_implementation_writeup.md §11.x). For
    // spherical / mesh phantoms the deterministic targets are unaffected
    // (the offset only changes which patch of speckle is sampled).
    //
    // `frame_seed`: per-frame seed mixed into the decorrelation hash so
    // successive `simulate()` calls draw independent speckle realizations.
    // Callers that want frame-to-frame stable speckle can keep
    // `frame_seed == 0`. Tier 1 test I increments this per frame to converge
    // on the bench's depth-flat anechoic statistics under temporal averaging.
    //
    // Scatter-realization seeding (this `frame_seed`) is SEPARATE from
    // electronic-noise seeding (`noise_seed` below). The bench's phased-array
    // IVUS imaging a static phantom has HIGHLY CORRELATED scatterer realizations
    // between
    // frames (the catheter doesn't rotate; milk fat globules are static at
    // 30 fps) but INDEPENDENT electronic noise per frame.  Hold `frame_seed`
    // constant and vary `noise_seed` per frame to match this bench reality.
    bool scatter_angular_decorrelate = true;
    uint32_t frame_seed = 0u;

    // Per-frame noise seed, decoupled from `frame_seed`.
    //
    // All three additive-Gaussian noise kernels (pre-PSF RF noise, depth-
    // weighted RF noise, post-envelope noise) derive their seed
    // from `frame_seed`.  That made the noise realization track the
    // scatter realization frame-to-frame: incrementing `frame_seed` re-rolls
    // both scatterers AND noise; keeping `frame_seed` constant freezes both.
    // Neither matched the bench, which has correlated scatterers + independent
    // noise.  `noise_seed` now provides separate control:
    //
    //   * Caller sets `noise_seed` to a different value per frame to get
    //     bench-like behaviour (correlated scatterers, independent noise).
    //   * Default value 0 falls through to `frame_seed` (legacy behaviour:
    //     noise tracks scatter seed; suitable for tests that explicitly
    //     want independent realizations per frame, e.g. Test C/D wire-PSF
    //     aggregation across 4 catheter rotations).
    //
    // Salt constants in the CUDA wrappers keep the three noise kernels'
    // realizations independent of each other (and of the scatter
    // realization) even when both seeds collide; see `cuda_algorithms.cu`.
    uint32_t noise_seed = 0u;

    // Additive Gaussian RF noise floor.
    //
    // `noise_sigma > 0` adds N(0, noise_sigma^2) to every element of the
    // post-gain RF buffer (just before Hilbert envelope detection), so the
    // post-Hilbert envelope of a no-scatter region becomes Rayleigh-
    // distributed with mean `sigma * sqrt(pi/2)`. After log compression
    // and the post-log display window, this lifts low-envelope pixels
    // (currently clamped at `reject_palette = 11`) up to a bench-like
    // mean palette ~30-50 in a band-pass-saturated band, eliminating the
    // bimodal sim distribution that the user observed in the wire-phantom
    // ringdown zone.
    //
    // Units: RF amplitude at the reference gain (i.e. AFTER the calibrated
    // `gain_db` scalar has been applied to the OptiX RF chain). The PV .035
    // calibration sheet (`processing.noise.{type, sigma}` in the YAML)
    // provides `sigma = 2.6347` in these units.
    //
    // Default `noise_sigma == 0.f` is a no-op; the host wrapper short-
    // circuits on `sigma <= 0` so existing callers see no overhead.
    //
    // The frame seed for this stage is derived from `frame_seed` (the same
    // field used by the scatter decorrelation), with a different
    // domain-separation salt mixed in inside the CUDA host wrapper, so a
    // single `frame_seed` increment per frame draws independent
    // realizations of *both* the scatter pattern and the noise floor.
    float noise_sigma = 0.f;

    // Post-envelope noise.
    //
    // `envelope_noise_sigma > 0` adds N(envelope_noise_mean, envelope_noise_sigma^2)
    // per pixel to the envelope buffer right after Hilbert and BEFORE the
    // post-Hilbert low-pass (`psf_env_lp_`). The LPF then convolves the noise
    // with a ~1-wavelength radial kernel, giving the noise spatial correlation
    // that matches the bench's visible coarse-grained speckle texture.
    //
    // Motivation (vs the pre-PSF `noise_sigma` knob): the pre-PSF additive RF
    // noise stage produces Rayleigh-distributed envelope statistics (CV ~ 0.52)
    // regardless of sigma. The bench's anechoic CV is ~0.35 with a softer
    // unimodal distribution, which cannot be reproduced by tuning a Rayleigh
    // model. Adding noise directly in envelope-amplitude space allows the
    // distribution shape and CV to be tuned independently by `mean` / `sigma`.
    //
    // The mean term seeds a baseline noise floor in materials with little or
    // no acoustic scatter (e.g. water): `env_new = env_existing + N(mean, sigma)`,
    // so anechoic regions get a calibrated mean envelope amplitude that maps
    // through log compression to the bench's mean palette (~38-42). For
    // tissue / scatterers, env_existing is non-zero and the additive term acts
    // as a noise floor sitting on top of the deterministic signal.
    //
    // Units: envelope amplitude at the reference gain (i.e. AFTER the calibrated
    // `gain_db` scalar has been applied to the post-Hilbert envelope). The
    // analytic post-envelope prototype against E6 anechoic water gives
    // `mean ~ 2.0`, `sigma ~ 1.23` for `processing.log_floor / log_multiplier`
    // currently in the YAML. The calibration script
    // `derive_envelope_noise.py` bisects sigma against the bench CV target
    // and solves mean against the bench mean palette analytically.
    //
    // Default `envelope_noise_sigma == 0.f` is a no-op; the host wrapper
    // short-circuits when both fields are zero so existing callers pay no cost.
    //
    // The frame seed mixes `frame_seed` with its own salt (distinct from the
    // pre-PSF RF noise salt) so the two stages draw independent realizations
    // even when both are active.
    float envelope_noise_mean = 0.f;
    float envelope_noise_sigma = 0.f;

    // Envelope-noise gain scaling.
    //
    // The bench's noise floor scales with the receive-chain gain (slider /
    // gain_db): each +1 dB of gain amplifies the post-receive noise envelope
    // by ~10^(1/20). Because `envelope_noise_*` is injected POST-Hilbert
    // (after the calibrated `gain_db` scalar has been applied to the
    // simulated signal), the noise term does NOT pick up the gain
    // amplification by default. To match the bench's gain-dependent noise
    // floor we scale the effective mean / sigma at simulate-time by
    //
    //   scale = 10^((sim_params.gain_db - envelope_noise_reference_gain_db) / 20)
    //
    // The reference value is the `gain_db` at which `envelope_noise_*` were
    // calibrated (i.e. the value of `gain_db` when the calibration script
    // measured the bench's noise std). For the PV .035 YAML this is the
    // sim's `gain_db` at the bench slider used for the noise calibration
    // (slider 50, so `gain_db = base_gain_db - 4 dB`).
    //
    // Default `envelope_noise_reference_gain_db == 0.f` plus a non-zero
    // `gain_db` would over-amplify the noise; the host code short-circuits
    // the scaling (scale = 1) when both fields are zero so legacy callers
    // see no change in behaviour.
    float envelope_noise_reference_gain_db = 0.f;

    // Depth-gain (TGC) scaling of the post-Hilbert envelope noise.
    //
    // When `false` (default), `envelope_noise_mean` / `_sigma` are added as
    // a depth-FLAT N(mean, sigma^2) per pixel. This matches a noise floor
    // that has been injected AFTER the TGC stage and therefore does NOT
    // inherit any TGC depth-dependent gain.
    //
    // When `true`, both the additive mean and the per-pixel Gaussian draw
    // are multiplied per sample by `tgc_curve_[r]` (the linear gain curve
    // built by `create_piece_wise_tgc`, normalised to 1.0 at r = 0). This
    // models the physically-correct picture of bench analog electronic
    // noise that has been TGC-amplified in the receive chain: the same TGC
    // schedule that lifts the deep-r signal also lifts the deep-r noise
    // floor. Net effect on the per-r palette:
    //   - At shallow r (TGC = 0 dB on a typical schedule), no change.
    //   - At deep r (TGC > 0 dB), the noise floor envelope is lifted by
    //     ~10^(TGC_dB / 20), so the deep-r palette in milk / soft-tissue
    //     stays elevated above the bench's reject floor without needing
    //     a depth-flat DC lift in `envelope_noise_mean`.
    //
    // The CUDA kernels use the same hash salt as the un-scaled variant,
    // so flipping this flag changes only the per-sample noise amplitude,
    // not the underlying noise realization -- regression tests anchored on
    // `noise_seed` remain stable apart from the desired depth-shape change.
    bool envelope_noise_apply_tgc_depth_scaling = false;

    // Soft reject-floor compression.
    //
    // When `reject_palette_softness > 0`, the post-log display-window kernel
    // applies a smooth (softplus-style) approach to the reject floor instead of
    // a hard `max(palette, reject_palette)` clamp:
    //
    //   palette = reject + softness * log1p(exp((palette - reject) / softness))
    //
    // This removes the spurious histogram spike at `reject_palette` that the
    // hard clamp creates when the envelope-noise distribution has tails below
    // the floor: with envelope noise N(2.0, 1.23) and log_floor ~0.69 the
    // lower 5-10% of pixels fall under `reject_palette = 11` and pile up as
    // a single histogram bar. The softplus blend lifts them smoothly into
    // a continuous lower shoulder that matches the bench's softer fade-out.
    //
    // Units: palette units. Setting `reject_palette_softness == 0` falls back
    // to the original hard clamp behaviour (default for existing YAMLs).
    float reject_palette_softness = 0.f;

    // Catheter sheath dead-zone mask.
    //
    // `catheter_dead_zone_mm > 0` zeros the inner radial samples of the
    // *final* (post log-compression, post display window) palette buffer
    // for any sample with `r < catheter_dead_zone_mm`. On the bench, the
    // catheter wall (OD ≈ 1.4-1.9 mm depending on probe) blocks any
    // acquired signal, so the inner zone renders as solid black (palette 0
    // -- *deeper* than the device's reject_palette = 11). Without this
    // mask the additive-noise stage fills the dead zone with the
    // calibrated noise floor, and any leaked ring-down energy raises it
    // further -- both visible vs the bench's solid-black inner zone.
    //
    // Default `catheter_dead_zone_mm == 0.f` is a no-op (the kernel
    // wrapper short-circuits on `dead_zone_samples == 0`).
    float catheter_dead_zone_mm = 0.f;

    // -------------------------------------------------------------------------
    // Ring-down injection.
    // -------------------------------------------------------------------------

    /// Calibrated ring-down injection (off by default; see `RingDownParams`).
    /// Added to the envelope buffer between Hilbert and log compression so the
    /// cyclic Hilbert FFT never sees the near-field transient.
    RingDownParams ring_down;

    /// Display window in palette units.
    ///
    /// After log compression (which uses the calibrated `log_multiplier` /
    /// `log_floor`, so its output is in absolute palette units) the post-log
    /// buffer is clamped to ``[reject_palette, saturation_palette]``. This
    /// reproduces the device's hard reject floor and saturation ceiling
    /// directly (cf. PV .035: `reject_palette = 11`, `saturation_palette =
    /// 239` from `gain_lut.json`).
    ///
    /// Disabled (skipped) when `saturation_palette <= reject_palette`. Both
    /// default to 0.f so default callers keep the pure-log mapping;
    /// calibrated YAMLs set both to non-zero.
    float reject_palette = 0.f;
    float saturation_palette = 0.f;

    // -------------------------------------------------------------------------
    // Reference gain.
    // -------------------------------------------------------------------------

    /// Single-knob reference gain applied to the RF buffer between TGC and
    /// ring-down injection as ``rf <- rf * 10^(gain_db / 20)``. Lumps two
    /// physically distinct
    /// effects into one calibrated scalar:
    ///
    ///   1. The bench's *slider gain* offset (the device's gain control:
    ///      slider 54 maps to 0 dB by convention; a 10-step change is +/- 10 dB
    ///      on the bench gain LUT).
    ///   2. A renderer-specific *reference-amplitude offset* — the simulator's
    ///      raw envelope amplitudes are not on the same linear scale as the
    ///      bench's calibrated amplitudes. The calibration sheet treats the
    ///      bench's "amp at slider 54" as the reference (in arbitrary linear
    ///      units), so this offset is the constant that puts the simulator's
    ///      output onto that scale. See
    ///      `instrument-calibration/p035_visions/calibration_delta.md` for
    ///      how it's measured (analytically, from a known-target render).
    ///
    /// Default ``gain_db = 0.f`` is a no-op so callers that don't set it see
    /// no change in behaviour. For the calibrated PV .035 YAML the value is
    /// derived per-renderer; ~+27 dB for the current OptiX/CUDA backend
    /// against the PV .035 bench at slider 54.
    float gain_db = 0.f;

    // -------------------------------------------------------------------------
    // Lateral-PSF kernel-type switch.
    // -------------------------------------------------------------------------
    //
    // The default kernel models the lateral PSF as a fixed-focus Gaussian
    // beam: sigma_mm(r) = w0 * sqrt(1 + ((r - z_f) / z_R)^2) with a
    // pre-focal clamp. Bench data on the s5i synthetic-aperture probe shows
    // a *constant* angular FWHM (~7.3 deg across r in [4, 26] mm), which
    // the Gaussian-beam kernel cannot reproduce: it has a focal minimum
    // (with z_f ~ 2.8 mm in the bench fit) and a smooth post-focal
    // expansion. To support an SA-aware kernel without a behavioural
    // regression, this enum + companion field switches between the two:
    //
    //   0 (GAUSSIAN_BEAM):     legacy fixed-focus Gaussian-beam kernel
    //                          (sigma_mm(r) above). Cache invalidation is
    //                          keyed on probe geometry as before.
    //   1 (CONSTANT_ANGULAR):  every depth bin uses the same angular spread
    //                          sigma_theta_rad, i.e. sigma_bins(r) =
    //                          sigma_theta_rad * num_angular_rays / (2*pi),
    //                          constant in r. Calibrate against the bench
    //                          median angular FWHM via
    //                          instrument-calibration/p035_visions/derive_lateral_psf_sigma_theta.py.
    //
    // Default GAUSSIAN_BEAM keeps existing YAMLs / examples unchanged.
    // For the calibrated PV .035, the constant-angular value derived from
    // the Wave 0 B2 bench (147 unsaturated wires) is
    // sigma_theta_rad ~ 0.05416 (FWHM ~ 7.31 deg).
    int lateral_psf_kernel_type = 0;
    float lateral_psf_sigma_theta_rad = 0.054164779787691845f;  // ~ 3.103 deg

    // IVUS angular ray super-sampling.
    //
    // Fire K sub-rays per scanline at deterministic sub-bin angular offsets
    // (uniform across the scanline's angular pitch) and accumulate into the
    // same scanline buffer with weight 1/K.  This forward-models the
    // bench's finite beam width at the raycasting stage, eliminating the
    // bimodal hit/miss behaviour that a single ray per scanline produces
    // for sub-wavelength point scatterers.
    //
    // Default `ivus_rays_per_scanline = 1` is backward-compatible (one
    // OptiX trace per scanline).  Recommended K = 8 for sub-wavelength
    // wire phantoms.  Speckle / continuous-scatter renders are not
    // sensitive to K (the integral over a uniform medium is invariant in
    // K), so the only cost is render time (linear in K for the OptiX
    // raygen stage).
    uint32_t ivus_rays_per_scanline = 1u;
  };

  /// Simulation results
  struct SimResult {
    std::unique_ptr<CudaMemory> rf_data;
    std::unique_ptr<CudaMemory> b_mode;
  };

  /**
   * Generate a single B-mode ultrasound frame
   *
   * @param probe BaseProbe object
   * @param sim_params Simulation parameters
   *
   * @returns Dictionary containing simulation results
   */
  SimResult simulate(const BaseProbe* probe, const SimParams& sim_params);

  /**
   * Get the minimum x value of the simulated region
   *
   * @returns Minimum x value [mm]
   */
  float get_min_x() const { return min_x_; }

  /**
   * Get the maximum x value of the simulated region
   *
   * @returns Maximum x value [mm]
   */
  float get_max_x() const { return max_x_; }

  /**
   * Get the minimum z value of the simulated region
   *
   * @returns Minimum z value [mm]
   */
  float get_min_z() const { return min_z_; }

  /**
   * Get the maximum z value of the simulated region
   *
   * @returns Maximum z value [mm]
   */
  float get_max_z() const { return max_z_; }

 private:
  World* const world_;
  const Materials* const materials_;
  const float SAMPLING_FREQ = 40e6f;

  // Boundary values of the simulated region
  float min_x_ = 0.f;
  float max_x_ = 0.f;
  float min_z_ = 0.f;
  float max_z_ = 0.f;

  std::shared_ptr<OptixDeviceContext_t> context_;
  std::shared_ptr<OptixPipeline_t> pipeline_;               ///< OptiX pipeline
  std::shared_ptr<OptixProgramGroup_t> raygen_prog_group_;  ///< OptiX ray gen program group
  std::shared_ptr<OptixProgramGroup_t> miss_prog_group_;    ///< OptiX miss program group
  std::shared_ptr<OptixProgramGroup_t>
      hitgroup_prog_group_sphere_;  ///< OptiX hit group program group for spheres
  std::shared_ptr<OptixProgramGroup_t>
      hitgroup_prog_group_triangles_;               ///< OptiX hit group program group for triangles
  OptixShaderBindingTable shader_binding_table_{};  ///< OptiX shader binding table

  std::shared_ptr<CUDAAlgorithms> cuda_algorithms_;

  CudaMemory raygen_record_;
  CudaMemory miss_record_;
  CudaMemory hitgroup_record_;

  CudaMemory pipeline_params_;

  float probe_frequency_ = 0.f;
  std::optional<ProbeType> psf_ax_probe_type_;  ///< Probe type for which psf_ax_ was built (IVUS uses causal kernel)
  std::unique_ptr<CudaMemory> psf_ax_;
  /// Post-Hilbert envelope-detection low-pass kernel.
  /// The Hilbert transform produces an analytic-signal magnitude that
  /// still contains intra-cycle carrier ripple at 2x the carrier
  /// frequency; the textbook envelope-detection step convolves
  /// |analytic_signal| with a low-pass kernel of bandwidth ~ 1
  /// wavelength to suppress the ripple while preserving the slow
  /// envelope.  Without this stage the sim's envelope FWHM is
  /// effectively a single-sample spike (see
  /// instrument-calibration/p035_visions/debug_sim_axial_envelope.py)
  /// and the bench's broader (~2 lambda) axial PSF cannot be recovered
  /// by any setting of pulse_duration_cycles.
  std::unique_ptr<CudaMemory> psf_env_lp_;
  /// Cached probe frequency used when psf_env_lp_ was last built; rebuild
  /// when the probe carrier frequency changes (which sets the low-pass FWHM
  /// = 1 wavelength = c / freq by default).
  float psf_env_lp_freq_cached_ = 0.f;
  float probe_element_spacing_ = 0.f;
  std::unique_ptr<CudaMemory> psf_lat_;
  CudaMemory psf_tmp_;

  /// Depth-dependent lateral PSF for IVUS (focused element): one kernel per depth bin
  std::unique_ptr<CudaMemory> psf_lat_2d_;
  uint32_t psf_lat_2d_depth_bins_ = 0;
  uint32_t psf_lat_2d_kernel_radius_ = 0;
  float psf_lat_2d_element_radius_ = 0.f;
  float psf_lat_2d_focal_length_ = 0.f;
  uint32_t psf_lat_2d_buffer_size_ = 0;
  float psf_lat_2d_t_far_ = 0.f;
  // Cache keys for the lateral-PSF kernel-type switch: invalidate when the
  // caller changes kernel type or moves the constant-angular sigma. Initial
  // values (-1 / 0.f) force a (re)build on the first call regardless of the
  // SimParams default.
  int psf_lat_2d_kernel_type_ = -1;
  float psf_lat_2d_sigma_theta_rad_ = 0.f;

  // Per-depth additive-noise weight = sqrt(sigma_bins(z) / sigma_bins(z_focal)).
  // Built alongside the depth-dependent lateral PSF (same probe params drive
  // both) and consumed by `add_gaussian_noise_depth_weighted` so the post-PSF
  // noise standard deviation is uniform across depth (matching bench's flat
  // anechoic profile). Cached fields mirror psf_lat_2d_'s invalidation keys.
  std::unique_ptr<CudaMemory> noise_depth_weight_;
  uint32_t noise_depth_weight_size_ = 0;
  float noise_depth_weight_element_radius_ = 0.f;
  float noise_depth_weight_focal_length_ = 0.f;
  uint32_t noise_depth_weight_buffer_size_ = 0;
  float noise_depth_weight_t_far_ = 0.f;
  uint32_t noise_depth_weight_num_angular_rays_ = 0;
  float noise_depth_weight_lambda_mm_ = 0.f;
  // Kernel type also drives the noise-depth-weight build (constant-
  // angular kernel => uniform post-PSF noise variance => weight == 1).
  int noise_depth_weight_kernel_type_ = -1;

  std::unique_ptr<CudaMemory> tgc_curve_;
  std::optional<ProbeType> tgc_probe_type_;  ///< Probe type used to build current TGC (for cache invalidation)

  // Cached ring-down waveform on device. Rebuilt whenever the host-side
  // RingDownParams that feed it change (decay shape, amplitude, extent_mm, the
  // measured waveform, sampling buffer size, or sampling frequency / SoS conversion).
  std::unique_ptr<CudaMemory> ring_down_waveform_;
  uint32_t ring_down_sample_count_ = 0;
  std::string ring_down_decay_cached_;
  float ring_down_amplitude_cached_ = 0.f;
  float ring_down_extent_mm_cached_ = 0.f;
  uint32_t ring_down_buffer_size_cached_ = 0;
  size_t ring_down_waveform_data_size_cached_ = 0;
  const float* ring_down_waveform_data_cached_ = nullptr;

  void update_psfs(const BaseProbe* probe, cudaStream_t stream, uint32_t buffer_size, float t_far,
                   int lateral_psf_kernel_type, float lateral_psf_sigma_theta_rad);
};

}  // namespace raysim

#endif /* CPP_RAYTRACING_ULTRASOUND_SIMULATOR */
