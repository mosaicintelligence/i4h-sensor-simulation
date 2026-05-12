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

/// Pass 2 — calibrated ring-down injection knobs.
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
    // Processing parameters (Pass 1 plumbing). All defaults are chosen to
    // exactly reproduce the previously hard-coded behavior in simulate():
    //   * empty tgc_control_points => probe-type default schedule
    //   * scattering_resolution_mm == 0 => probe-type default (10 IVUS / 50 other)
    //   * remaining defaults match the literals previously baked into the kernels.
    // -------------------------------------------------------------------------
    std::vector<TgcControlPoint> tgc_control_points;  // (depth_cm, gain_db); empty = auto

    // Log compression (Pass 3b / K2v2): out = log_multiplier * log10(amp / log_floor).
    // `log_floor` is the calibration anchor (amp == log_floor maps to palette 0).
    // Default `log_floor = 1.f` makes the mapping reduce to
    // `log_multiplier * log10(amp)` for amp >= 1, which matches the legacy
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

    // Pass 5b — per-scanline angular decorrelation of the scatter texture lookup.
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
    bool scatter_angular_decorrelate = true;
    uint32_t frame_seed = 0u;

    // Pass 6 — additive Gaussian RF noise floor.
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
    // field used by the Pass 5b scatter decorrelation), with a different
    // domain-separation salt mixed in inside the CUDA host wrapper, so a
    // single `frame_seed` increment per frame draws independent
    // realizations of *both* the scatter pattern and the noise floor.
    float noise_sigma = 0.f;

    // Pass 6 v2 — catheter sheath dead-zone mask.
    //
    // `catheter_dead_zone_mm > 0` zeros the inner radial samples of the
    // *final* (post log-compression, post display window) palette buffer
    // for any sample with `r < catheter_dead_zone_mm`. On the bench, the
    // catheter wall (OD ≈ 1.4-1.9 mm depending on probe) blocks any
    // acquired signal, so the inner zone renders as solid black (palette 0
    // -- *deeper* than the device's reject_palette = 11). Without this
    // mask the Pass 6 additive-noise stage fills the dead zone with the
    // calibrated noise floor, and any leaked ring-down energy raises it
    // further -- both visible vs the bench's solid-black inner zone.
    //
    // Default `catheter_dead_zone_mm == 0.f` is a no-op (the kernel
    // wrapper short-circuits on `dead_zone_samples == 0`).
    float catheter_dead_zone_mm = 0.f;

    // -------------------------------------------------------------------------
    // Pass 2 plumbing.
    // -------------------------------------------------------------------------

    /// Calibrated ring-down injection (off by default; see `RingDownParams`).
    /// Added to each A-line between TGC and envelope detection so it goes
    /// through the same Hilbert + log-compression pipeline as the scatter signal.
    RingDownParams ring_down;

    /// Pass 3b — display window in palette units.
    ///
    /// After log compression (which uses the calibrated `log_multiplier` /
    /// `log_floor`, so its output is in absolute palette units) the post-log
    /// buffer is clamped to ``[reject_palette, saturation_palette]``. This
    /// reproduces the device's hard reject floor and saturation ceiling
    /// directly (cf. PV .035: `reject_palette = 11`, `saturation_palette =
    /// 239` from `gain_lut.json`).
    ///
    /// Disabled (skipped) when `saturation_palette <= reject_palette`. Both
    /// default to 0.f so default callers keep the historical pure-log
    /// mapping; calibrated YAMLs set both to non-zero.
    ///
    /// Note: this replaces the previous Pass 2 ``dynamic_range_db`` /
    /// ``reject_db`` knobs, which both clamped *and* re-zeroed the palette
    /// (shifting `reject_db` to palette 0). That re-zeroing was incorrect:
    /// it interacted with K2v2's negative-palette outputs (which represent
    /// `amp < log_floor`) by shifting them up to the saturation ceiling, so
    /// the device's reject floor was never actually displayed. The
    /// palette-anchor formulation removes that bug and matches the
    /// calibration sheet's semantics directly.
    float reject_palette = 0.f;
    float saturation_palette = 0.f;

    // -------------------------------------------------------------------------
    // Pass 3 plumbing.
    // -------------------------------------------------------------------------

    /// Pass 3 — single-knob reference gain applied to the envelope buffer
    /// between Hilbert (stage 2) and log compression (stage 3) as
    /// ``amp <- amp * 10^(gain_db / 20)``. This lumps two physically distinct
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
  float probe_element_spacing_ = 0.f;
  std::unique_ptr<CudaMemory> psf_lat_;
  float probe_elevational_height_ = 0.f;
  std::unique_ptr<CudaMemory> psf_elev_;
  CudaMemory psf_tmp_;

  /// Depth-dependent lateral PSF for IVUS (focused element): one kernel per depth bin
  std::unique_ptr<CudaMemory> psf_lat_2d_;
  uint32_t psf_lat_2d_depth_bins_ = 0;
  uint32_t psf_lat_2d_kernel_radius_ = 0;
  float psf_lat_2d_element_radius_ = 0.f;
  float psf_lat_2d_focal_length_ = 0.f;
  uint32_t psf_lat_2d_buffer_size_ = 0;
  float psf_lat_2d_t_far_ = 0.f;

  // Pass 7 — per-depth additive-noise weight = sqrt(sigma_bins(z) / sigma_bins(z_focal)).
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

  std::unique_ptr<CudaMemory> tgc_curve_;
  std::optional<ProbeType> tgc_probe_type_;  ///< Probe type used to build current TGC (for cache invalidation)

  // Pass 2 — cached ring-down waveform on device. Rebuilt whenever the host-side
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

  void update_psfs(const BaseProbe* probe, cudaStream_t stream, uint32_t buffer_size, float t_far);
};

}  // namespace raysim

#endif /* CPP_RAYTRACING_ULTRASOUND_SIMULATOR */
