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

#include "raysim/core/raytracing_ultrasound_simulator.hpp"

#include <spdlog/fmt/fmt.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iterator>

#include "raysim/core/probe.hpp"
#include "raysim/core/world.hpp"
#include "raysim/core/write_image.hpp"
#include "raysim/cuda/cuda_algorithms.hpp"
#include "raysim/cuda/optix_trace.hpp"

namespace raysim {

template <typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

/**
 * @brief Create Gaussian PSF kernel, optionally modulated by cosine.
 *
 * @param stream CUDA stream
 * @param width Standard deviation of Gaussian representation of point spread function [mm]
 * @param k Sampling frequency [1/mm] or [MHz]
 * @param freq Optional frequency for cosine modulation [MHz] or [1/mm]
 *
 * @returns 1D PSF kernel array
 */
static std::unique_ptr<CudaMemory> create_gaussian_psf(cudaStream_t stream, float width, float k,
                                                       float freq = 0.f) {
  const float c = 1.54f;  // [mm/us]
  const float dx = c / k;
  const uint32_t size =
      uint32_t(std::ceil(width / dx)) * 10 + 1;  // convolution filter size is 10x sigma
  std::vector<float> gaussian(size);

  const float t = -1.f / (2.f * std::pow(width, 2.f));

  float amplitude_integral = 0.f;
  for (int index = 0; index < size; ++index) {
    const float x = float(index - float(size / 2)) * dx;
    const float value = std::exp((x * x) * t);
    amplitude_integral += value;
    gaussian[index] = value;
  }
  // Normalize
  if (amplitude_integral != 0.f) {
    amplitude_integral = 1.f / amplitude_integral;
    for (int index = 0; index < size; ++index) { gaussian[index] *= amplitude_integral; }
  }

  if (freq != 0.f) {
    const float f = freq / c;  // spatial frequency of pulse 1/mm 1/wl
    for (int index = 0; index < size; ++index) {
      const float x = float(index - float(size / 2)) * dx;
      // Modulate Gaussian with cosine at given frequency
      gaussian[index] *= -std::cos(2.f * M_PI * f * x);
    }
  }

  auto buffer = std::make_unique<CudaMemory>(size * sizeof(float), stream);
  buffer->upload(gaussian.data(), stream);
  return buffer;
}

/**
 * Create a causal, Hanning-windowed axial PSF for IVUS so the wall echo does not leak into the lumen.
 * Convolution: output[d] = sum_k source[d+k] * kernel[k+radius]. For a delta at depth d,
 * kernel[radius+j] for j>0 adds to output[d-j] (shallower). So zero the right half of the
 * kernel (indices radius+1 .. 2*radius) to prevent the strong wall echo from smearing shallower.
 * Non-zero part: indices 0..radius (main lobe + pre-cursor); echo then only smears deeper.
 *
 * @param stream CUDA stream
 * @param extent_mm Total extent of the pulse in mm (e.g. n_cycles * wavelength)
 * @param k Sampling spatial frequency [1/us] -> dx = c/k in mm
 * @param freq Center frequency in MHz for cosine modulation
 * @param c Speed of sound mm/us
 * @returns 1D kernel, odd length, causal (right half zero so no backward leak)
 */
static std::unique_ptr<CudaMemory> create_ivus_axial_psf_causal(cudaStream_t stream,
                                                                float extent_mm, float k, float freq,
                                                                float c = 1.54f) {
  const float dx = c / k;
  const int half_samples = static_cast<int>(std::ceil(extent_mm / dx));
  const uint32_t kernel_size = 2 * half_samples + 1;
  const int radius = static_cast<int>(kernel_size / 2);
  std::vector<float> kernel(kernel_size, 0.f);

  // Non-zero only on left half + center (indices 0..radius). Hanning * cos so the echo
  // at the wall smears only deeper (no energy in kernel[radius+1..2*radius]).
  const int n = half_samples;
  float amplitude_integral = 0.f;
  for (int i = 0; i <= half_samples; ++i) {
    const int idx = radius - i;  // from center (radius) down to 0
    if (idx < 0) { break; }
    const float x = static_cast<float>(i) * dx;
    const float hanning = (n > 0) ? 0.5f * (1.f - std::cos(2.f * M_PI * static_cast<float>(i) / static_cast<float>(n)))
                                  : 1.f;
    const float carrier = (freq != 0.f) ? -std::cos(2.f * M_PI * (freq / c) * x) : 1.f;
    const float value = hanning * carrier;
    kernel[idx] = value;
    amplitude_integral += value * value;
  }
  amplitude_integral = std::sqrt(amplitude_integral);
  if (amplitude_integral > 0.f) {
    const float scale = 1.f / amplitude_integral;
    for (int i = 0; i <= half_samples && (radius - i) >= 0; ++i) { kernel[radius - i] *= scale; }
  }

  auto buffer = std::make_unique<CudaMemory>(kernel_size * sizeof(float), stream);
  buffer->upload(kernel.data(), stream);
  return buffer;
}

/**
 * Build a symmetric Gaussian low-pass kernel for post-Hilbert envelope detection.
 *
 * The Hilbert transform |analytic_signal| contains intra-cycle carrier
 * ripple at 2x the carrier frequency; a textbook envelope detector
 * convolves the magnitude with a low-pass kernel of bandwidth ~ 1
 * wavelength to suppress the ripple while preserving the slow envelope.
 * We use a Gaussian of FWHM = 1 wavelength (= c / freq) by default; the
 * kernel is symmetric (no causal trimming) since the envelope-detection
 * smoothing is acausal in r.
 *
 * @param stream CUDA stream
 * @param freq Probe carrier frequency in MHz (1 wavelength = c / freq mm)
 * @param k Sampling spatial frequency [1/us] -> dx = c/k in mm
 * @param fwhm_wavelengths Low-pass FWHM in wavelengths (default 1.0)
 * @param c Speed of sound mm/us
 * @returns L1-normalized 1D Gaussian kernel, odd length, extent +/- 3 sigma
 */
static std::unique_ptr<CudaMemory> create_envelope_lowpass_kernel(
    cudaStream_t stream,
    float freq,
    float dx_mm,  // sample pitch in the d_scanlines buffer (mm/sample)
    float fwhm_wavelengths = 1.0f,
    float c = 1.54f) {
  const float dx = dx_mm;
  const float wavelength_mm = c / freq;
  const float fwhm_mm = fwhm_wavelengths * wavelength_mm;
  const float sigma_mm = fwhm_mm / (2.f * std::sqrt(2.f * std::log(2.f)));
  // Extent: +/- 3 sigma is enough that the kernel's tails are < 1.1%.
  const int half_samples =
      std::max(1, static_cast<int>(std::ceil(3.f * sigma_mm / dx)));
  const uint32_t kernel_size = 2 * half_samples + 1;
  const int radius = static_cast<int>(kernel_size / 2);
  std::vector<float> kernel(kernel_size, 0.f);
  float sum = 0.f;
  const float inv_two_sigma_sq = 1.f / (2.f * sigma_mm * sigma_mm);
  for (uint32_t i = 0; i < kernel_size; ++i) {
    const float x = (static_cast<float>(i) - static_cast<float>(radius)) * dx;
    const float value = std::exp(-(x * x) * inv_two_sigma_sq);
    kernel[i] = value;
    sum += value;
  }
  // L1 normalize so the convolution preserves the input's mean amplitude
  // (envelope detection should not change the DC level).
  if (sum > 0.f) {
    const float inv_sum = 1.f / sum;
    for (uint32_t i = 0; i < kernel_size; ++i) {
      kernel[i] *= inv_sum;
    }
  }
  auto buffer = std::make_unique<CudaMemory>(kernel_size * sizeof(float), stream);
  buffer->upload(kernel.data(), stream);
  return buffer;
}

struct ControlPoint {
  float depth;  // cm
  float amp;    // dB
};

/**
 * Create a piece-wise linear TGC curve from control points.
 *
 * Control points must be sorted in ASCENDING `depth` order.  For each
 * query depth the curve is interpolated linearly between the bracketing
 * control points; queries before the first CP are clamped to the first
 * CP's amplitude, queries after the last CP are clamped to the last
 * CP's amplitude.  The output curve is normalised so that index 0
 * (depth = 0) has linear gain 1.0, matching the historical contract
 * (`derive_gain_db.py` etc. derive `gain_db` against this normalised
 * curve).
 *
 * The control points are walked with std::upper_bound for O(log N)
 * lookup per sample.
 *
 * The sample-index to physical-depth mapping uses the scanline buffer
 * pitch `t_far_mm / depth_samples` (matching the raygen offset
 * computation in `optix_trace.cu`). Callers pass `t_far_mm` directly;
 * the legacy `c` and `fs` arguments are retained as ignored parameters
 * to preserve the call-site signature.
 *
 * @param depth_samples Number of samples along depth
 * @param control_points Vector of (depth_cm, gain_db) pairs, sorted ascending
 * @param t_far_mm Physical depth at the last sample (mm) -- buffer pitch is t_far_mm/depth_samples
 * @param c (Ignored; retained for ABI compatibility.)
 * @param fs (Ignored; retained for ABI compatibility.)
 * @return TGC curve interpolated from control points
 */
static std::unique_ptr<CudaMemory> create_piece_wise_tgc(
    cudaStream_t stream, uint32_t depth_samples, const std::vector<ControlPoint>& control_points,
    float t_far_mm, float c = 1540.f, float fs = 50e6f) {
  (void)c;
  (void)fs;
  std::vector<float> tgc_curve(depth_samples);
  assert(!control_points.empty());
  assert(depth_samples > 0u);
  assert(t_far_mm > 0.f);

  const float dr_cm = (t_far_mm / static_cast<float>(depth_samples)) / 10.f;
  float first_value = 1.f;
  for (uint32_t i = 0; i < depth_samples; ++i) {
    const float depth = static_cast<float>(i) * dr_cm;  // depth in cm

    // Find the first CP whose depth is strictly greater than the query
    // depth (`upper`).  The lower CP is then `upper - 1`.
    auto upper = std::upper_bound(
        control_points.begin(), control_points.end(), depth,
        [](float d, const ControlPoint& cp) { return d < cp.depth; });

    float value_dB;
    if (upper == control_points.begin()) {
      // Query depth is at or before the first CP -- clamp to first amp.
      value_dB = control_points.front().amp;
    } else if (upper == control_points.end()) {
      // Query depth is past the last CP -- clamp to last amp.
      value_dB = control_points.back().amp;
    } else {
      auto lower = std::prev(upper);
      const float dep_lo = lower->depth, amp_lo = lower->amp;
      const float dep_hi = upper->depth, amp_hi = upper->amp;
      const float span = dep_hi - dep_lo;
      const float frac = (span > 0.f) ? ((depth - dep_lo) / span) : 0.f;
      value_dB = amp_lo + (amp_hi - amp_lo) * frac;
    }

    float value = std::pow(10.f, value_dB / 20.f);
    if (i == 0) {
      first_value = value;
      value = 1.f;
    } else {
      value = value / first_value;
    }
    tgc_curve[i] = value;
  }

  auto d_tgc_curve = std::make_unique<CudaMemory>(depth_samples * sizeof(float), stream);
  d_tgc_curve->upload(tgc_curve.data(), stream);

  return std::move(d_tgc_curve);
}

RaytracingUltrasoundSimulator::RaytracingUltrasoundSimulator(World* world,
                                                             const Materials* materials)
    : world_(world), materials_(materials) {
  // Initialize OptiX
  context_ = optix_init();

  // Create the OptiX pipeline
  optix_create_pipeline(context_.get(),
                        pipeline_,
                        raygen_prog_group_,
                        miss_prog_group_,
                        hitgroup_prog_group_sphere_,
                        hitgroup_prog_group_triangles_);

  const cudaStream_t stream = cudaStreamPerThread;

  // Build the acceleration structure
  world_->build(context_.get(), stream);

  // Set up the shader binding table
  const size_t raygen_record_size = sizeof(RayGenSbtRecord);
  raygen_record_.resize(raygen_record_size);
  shader_binding_table_.raygenRecord = raygen_record_.get_device_ptr(stream);

  MissSbtRecord ms_sbt;
  OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_.get(), &ms_sbt));
  const size_t miss_record_size = sizeof(MissSbtRecord);
  miss_record_.resize(miss_record_size);
  miss_record_.upload(&ms_sbt, stream);
  shader_binding_table_.missRecordBase = miss_record_.get_device_ptr(stream);
  shader_binding_table_.missRecordStrideInBytes = sizeof(MissSbtRecord);
  shader_binding_table_.missRecordCount = 1;

  const std::vector<OptixBuildInput>& build_input = world_->get_build_input();
  const std::vector<HitGroupData>& hit_group_data = world_->get_hit_group_data();
  std::vector<HitGroupSbtRecord> hg_sbt(hit_group_data.size());
  for (size_t index = 0; index < hit_group_data.size(); ++index) {
    if (build_input[index].type == OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_SPHERES) {
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_sphere_.get(), &hg_sbt[index]));
    } else if (build_input[index].type == OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES) {
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_triangles_.get(), &hg_sbt[index]));
    } else {
      throw std::runtime_error("Unhandled OptiX build input type");
    }
    hg_sbt[index].data = hit_group_data[index];
  }

  const size_t hitgroup_record_size = hg_sbt.size() * sizeof(HitGroupSbtRecord);
  hitgroup_record_.resize(hitgroup_record_size);
  hitgroup_record_.upload(&hg_sbt[0], stream);

  shader_binding_table_.hitgroupRecordBase = hitgroup_record_.get_device_ptr(stream);
  shader_binding_table_.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  shader_binding_table_.hitgroupRecordCount = hit_group_data.size();

  pipeline_params_.resize(sizeof(Params));

  cuda_algorithms_ = std::make_shared<CUDAAlgorithms>();
}

void RaytracingUltrasoundSimulator::update_psfs(const BaseProbe* probe, cudaStream_t stream,
                                                uint32_t buffer_size, float t_far,
                                                int lateral_psf_kernel_type,
                                                float lateral_psf_sigma_theta_rad) {
  if (probe_frequency_ != probe->get_frequency()) {
    probe_frequency_ = probe->get_frequency();
    psf_ax_.reset();
    psf_lat_.reset();
    psf_lat_2d_.reset();
    noise_depth_weight_.reset();
  }

  const ProbeType pt = probe->get_probe_type();
  if (psf_ax_probe_type_ != pt) {
    psf_ax_probe_type_ = pt;
    psf_ax_.reset();
  }

  if (probe_element_spacing_ != probe->get_element_spacing()) {
    probe_element_spacing_ = probe->get_element_spacing();
    psf_lat_.reset();
  }

  if (probe_elevational_height_ != probe->get_elevational_height()) {
    probe_elevational_height_ = probe->get_elevational_height();
    psf_elev_.reset();
  }

  const float el_radius = probe->get_element_radius_mm();
  const float focal_mm = probe->get_focal_length_mm();
  if (pt == ProbeType::PROBE_TYPE_IVUS && el_radius > 0.f && focal_mm > 0.f) {
    if (!psf_lat_2d_ || psf_lat_2d_element_radius_ != el_radius ||
        psf_lat_2d_focal_length_ != focal_mm || psf_lat_2d_buffer_size_ != buffer_size ||
        psf_lat_2d_t_far_ != t_far ||
        psf_lat_2d_kernel_type_ != lateral_psf_kernel_type ||
        psf_lat_2d_sigma_theta_rad_ != lateral_psf_sigma_theta_rad) {
      psf_lat_2d_element_radius_ = el_radius;
      psf_lat_2d_focal_length_ = focal_mm;
      psf_lat_2d_buffer_size_ = buffer_size;
      psf_lat_2d_t_far_ = t_far;
      psf_lat_2d_kernel_type_ = lateral_psf_kernel_type;
      psf_lat_2d_sigma_theta_rad_ = lateral_psf_sigma_theta_rad;
      constexpr uint32_t depth_bins = 64;
      // kernel_radius scales with the angular array so the near-field
      // beam (sigma ~ 100+ angular bins at r ≈ 1 mm with the PV .035 geometry)
      // can fit inside the kernel window without truncation, and the cyclic
      // angular convolution (see `convolve_columns_depth_dependent_kernel`)
      // covers the full half-circumference.
      const uint32_t num_angular_rays = probe->get_num_elements();
      const uint32_t kernel_radius = std::max(1u, num_angular_rays / 2u);
      const uint32_t kernel_len = 2 * kernel_radius + 1;
      const float lambda_mm = probe->get_wave_length();
      // Gaussian beam model for depth-dependent lateral PSF (see e.g. Siegman "Lasers", Ch. 17;
      // Wikipedia "Rayleigh length"; rp-photonics "Gaussian beams"). Beam waist at focus (diffraction-
      // limited for circular aperture): w0 = lambda*F/(2*a) [Goodman "Introduction to Fourier
      // Optics"; -3 dB beam width = F#*lambda with F# = F/(2a)]. Rayleigh length z_R = pi*w0^2/lambda;
      // beam radius vs depth: w(z) = w0*sqrt(1 + (z/z_R)^2) with z = depth - focal_length.
      const float w0_mm = 0.5f * lambda_mm * focal_mm / el_radius;  // beam waist radius at focus
      const float z_R_mm = (lambda_mm > 0.f)
                               ? (static_cast<float>(M_PI) * w0_mm * w0_mm / lambda_mm)
                               : w0_mm;  // Rayleigh length; avoid div-by-zero if lambda unset
      const float two_pi = 6.28318530717958647692f;
      std::vector<float> k2d(depth_bins * kernel_len, 0.f);
      for (uint32_t b = 0; b < depth_bins; ++b) {
        const float depth_mm = (static_cast<float>(b) + 0.5f) / static_cast<float>(depth_bins) * t_far;
        const float z = depth_mm - focal_mm;  // distance from focus (mm)
        // Clamp the pre-focal beam expansion to its focal value.
        //
        // The textbook Gaussian beam w(z) = w0 * sqrt(1 + (z/z_R)^2) is symmetric
        // about the focus, so it predicts a *wide* mm-scale beam at depths
        // r << focal_length (e.g. ~2.8 mm beam radius at r = 1 mm with the
        // PV .035 geometry). Combined with the angular conversion
        // sigma_bins = sigma_mm * N / (2 * pi * r), the 1/r factor explodes
        // sigma_bins to >100 angular bins at r = 1 mm and the PSF averages
        // wires across nearly half the imaging circle in the near field.
        //
        // Bench imagery shows the opposite trend: in the unwrapped polar view
        // the angular FWHM of a wire is roughly constant with depth (so the
        // arc-length FWHM in mm in the polar-to-Cartesian mapping is
        // *narrower* for near wires, *wider* for far wires — see the
        // wire_phantom_polar_paired.png inset right panel). Physically this
        // matches the synthetic-aperture IVUS imaging chain: at depths
        // r < focal_length the rotating element only coherently sums over a
        // narrow beam (the SA aperture overlap is limited by the element
        // directivity), so the effective beam never expands above its focal
        // value. The depth-symmetric Gaussian beam model conflates this
        // single-element-rotated-SA geometry with a static focused circular
        // aperture and wrongly broadens the near-field beam.
        //
        // Pragmatic fix: zero out the pre-focal contribution to (z/z_R)^2,
        // so sigma_mm == w0 for any depth r <= focal_length and the textbook
        // expansion only applies post-focal where the model is meaningful.
        // This restores the bench-like wire shapes (tight pinpoints in the
        // polar view across all radii) and removes the residual bright-
        // center contribution from near-field wires being smeared across
        // many angular bins. The cyclic + wide-kernel + L1 normalization
        // still applies; the only change is the sigma_mm schedule per
        // depth_bin.
        // Branch on lateral PSF kernel type:
        //   0 = fixed-focus Gaussian beam (see comment above for the
        //       textbook + clamp derivation).
        //   1 = constant-angular Gaussian. Every depth bin uses the same
        //       angular sigma -- the SA-aware kernel motivated by the
        //       bench's constant ~7.3 deg angular FWHM across r.
        float sigma_bins;
        if (lateral_psf_kernel_type == 1) {
          // Constant-angular Gaussian: sigma_bins independent of depth.
          sigma_bins = lateral_psf_sigma_theta_rad
                       * static_cast<float>(num_angular_rays) / two_pi;
        } else {
          const float z_post = (z > 0.f) ? z : 0.f;
          const float sigma_mm = w0_mm * std::sqrt(1.f + (z_post * z_post) / (z_R_mm * z_R_mm));
          const float depth_safe = std::max(depth_mm, 0.5f);
          sigma_bins = sigma_mm * static_cast<float>(num_angular_rays) / (two_pi * depth_safe);
        }
        float* row = k2d.data() + b * kernel_len;
        float sum = 0.f;
        for (uint32_t i = 0; i < kernel_len; ++i) {
          const float x = static_cast<float>(static_cast<int>(i) - static_cast<int>(kernel_radius));
          const float v = std::exp(-0.5f * (x * x) / (sigma_bins * sigma_bins));
          row[i] = v;
          sum += v;
        }
        // L1 normalization (sum = 1).
        //
        // Empirical comparison vs L2 normalization (sqrt(sum_sq) = 1):
        //   * L2 norm preserves envelope amplitude for *uncorrelated* random-
        //     scatter input (the textbook expectation for distributed point
        //     scatterers).
        //   * Our scatter sampling (`scattering_resolution_mm` ~ 10 mm voxel
        //     texture, trilinearly interpolated) is *positively correlated*
        //     across angular bins at small radii where the arc length per
        //     scanline (2*pi*r/N) is much smaller than the texture voxel.
        //     With L2 normalization the cyclic Gaussian convolution then
        //     amplifies the correlated near-field signal by ~sqrt(sigma_bins)
        //     (sum_k a_k for fully-correlated input vs sqrt(sum_k a_k^2) for
        //     uncorrelated), producing a +50-80 palette unit bright shoulder
        //     at r ≈ 4-5 mm that overshoots even the saturation_palette.
        //   * L1 normalization gives the depth-dependent envelope behaviour
        //     baked into the bench TGC schedule, which the calibration
        //     `gain_db` was derived against and which produces the closest
        //     match to bench mean palette across the 8-29 mm window.
        //
        // The bright-shoulder near-field artifact (r ≈ 4-7 mm, +50-100
        // palette excess vs bench) is NOT fully cured by this pass; see
        // `ivus_implementation_writeup.md` §11.x for the deferred follow-up
        // (additive noise floor, scatter-texture decorrelation, or a
        // physics-based near-field beam cap).
        if (sum > 0.f) {
          for (uint32_t i = 0; i < kernel_len; ++i) { row[i] /= sum; }
        }
      }
      psf_lat_2d_ = std::make_unique<CudaMemory>(k2d.size() * sizeof(float), stream);
      psf_lat_2d_->upload(k2d.data(), stream);
      psf_lat_2d_depth_bins_ = depth_bins;
      psf_lat_2d_kernel_radius_ = kernel_radius;
    }

    // Per-depth additive-noise weight cache.
    //
    // We multiply the additive-RF-noise sigma by a depth-dependent weight so
    // that after the L1-normalised depth-dependent lateral PSF concentrates
    // the focal-zone noise, the post-PSF noise standard deviation is uniform
    // across depth (matching the bench's flat anechoic profile -- see
    // `tier1_results/figures/depth_uniformity.png` and the regression
    // analysis in `tier1_results.md` Test I).
    //
    // Derivation. For a Gaussian beam of std sigma_bins(z) (in angular-bin
    // units) that is L1-normalised (sum_k psf_k = 1), the discrete kernel
    // satisfies sum_k psf_k^2 ≈ 1 / (2*sqrt(pi)*sigma_bins) (continuous
    // Gaussian L2 squared norm). After convolution with i.i.d. Gaussian
    // input noise of std sigma_pre, the per-output-bin noise variance is
    //   var_post(z) = sigma_pre(z)^2 * sum_k psf_k(z)^2
    //              ≈ sigma_pre(z)^2 / (2*sqrt(pi)*sigma_bins(z))
    // To make var_post(z) uniform we need sigma_pre(z) ∝ sqrt(sigma_bins(z)).
    // Choosing the focal length (where sigma_bins is minimum) as the
    // reference depth keeps the calibrated `noise.sigma` in the YAML
    // bench-anchored:
    //   weight(z) = sqrt(sigma_bins(z) / sigma_bins(z_focal))
    // At z = z_focal the weight is 1; pre-focal (where the angular 1/r
    // factor inflates sigma_bins) and post-focal (where the Gaussian beam
    // expansion inflates sigma_mm) the weight grows.
    if (!noise_depth_weight_ || noise_depth_weight_element_radius_ != el_radius ||
        noise_depth_weight_focal_length_ != focal_mm ||
        noise_depth_weight_buffer_size_ != buffer_size ||
        noise_depth_weight_t_far_ != t_far ||
        noise_depth_weight_num_angular_rays_ != probe->get_num_elements() ||
        noise_depth_weight_lambda_mm_ != probe->get_wave_length() ||
        noise_depth_weight_kernel_type_ != lateral_psf_kernel_type) {
      noise_depth_weight_element_radius_ = el_radius;
      noise_depth_weight_focal_length_ = focal_mm;
      noise_depth_weight_buffer_size_ = buffer_size;
      noise_depth_weight_t_far_ = t_far;
      noise_depth_weight_num_angular_rays_ = probe->get_num_elements();
      noise_depth_weight_lambda_mm_ = probe->get_wave_length();
      noise_depth_weight_kernel_type_ = lateral_psf_kernel_type;

      std::vector<float> w(buffer_size, 1.f);
      if (lateral_psf_kernel_type == 1) {
        // Constant-angular Gaussian -> sigma_bins is uniform in r,
        // so the L1-normalised cyclic convolution produces uniform post-PSF
        // noise variance, and the depth weight collapses to 1. The
        // calibrated `noise.sigma` therefore feeds the additive RF noise
        // unscaled across depth.
        // (vector already initialised to 1.f above)
      } else {
        const uint32_t num_angular_rays = probe->get_num_elements();
        const float lambda_mm = probe->get_wave_length();
        const float w0_mm = 0.5f * lambda_mm * focal_mm / el_radius;
        const float z_R_mm = (lambda_mm > 0.f)
                                 ? (static_cast<float>(M_PI) * w0_mm * w0_mm / lambda_mm)
                                 : w0_mm;
        const float two_pi = 6.28318530717958647692f;
        const float dr_mm = t_far / static_cast<float>(buffer_size);
        // sigma_bins at the focal length (minimum across depth):
        const float sigma_bins_focal =
            w0_mm * static_cast<float>(num_angular_rays) / (two_pi * std::max(focal_mm, 0.5f));
        for (uint32_t i = 0; i < buffer_size; ++i) {
          const float depth_mm = (static_cast<float>(i) + 0.5f) * dr_mm;
          // Same pre-focal clamp + post-focal expansion as the lateral PSF
          // builder, so the noise weight tracks the actual kernel widths.
          const float z_post = (depth_mm > focal_mm) ? (depth_mm - focal_mm) : 0.f;
          const float sigma_mm = w0_mm * std::sqrt(1.f + (z_post * z_post) / (z_R_mm * z_R_mm));
          const float depth_safe = std::max(depth_mm, 0.5f);
          const float sigma_bins =
              sigma_mm * static_cast<float>(num_angular_rays) / (two_pi * depth_safe);
          const float ratio = (sigma_bins_focal > 0.f) ? (sigma_bins / sigma_bins_focal) : 1.f;
          w[i] = std::sqrt(std::max(ratio, 0.f));
        }
      }
      noise_depth_weight_ =
          std::make_unique<CudaMemory>(w.size() * sizeof(float), stream);
      noise_depth_weight_->upload(w.data(), stream);
      noise_depth_weight_size_ = static_cast<uint32_t>(w.size());
    }
  } else {
    psf_lat_2d_.reset();
    psf_lat_2d_depth_bins_ = 0;
    psf_lat_2d_kernel_radius_ = 0;
    noise_depth_weight_.reset();
    noise_depth_weight_size_ = 0;
  }

  if (!psf_ax_) {
    const float k = SAMPLING_FREQ * 1e-6;  // [1/us]
    float axial_width = probe->get_axial_resolution();
    if (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) {
      // Causal axial PSF with one-sided window (peak at true depth, no central null)
      // so wire phantoms show a single spot and wall echo does not leak into lumen.
      const float n_cycles = static_cast<float>(probe->get_pulse_duration());
      const float extent_mm = n_cycles * probe->get_wave_length();
      psf_ax_ = create_ivus_axial_psf_causal(stream, extent_mm, k, probe->get_frequency(), 1.54f);
    } else {
      psf_ax_ = create_gaussian_psf(stream, axial_width, k, probe->get_frequency());
    }
  }

  // Post-Hilbert envelope-detection low-pass kernel. Built once
  // per (probe frequency) change; FWHM = 1 wavelength is the textbook
  // envelope-detector smoothing for a Hilbert-magnitude envelope (suppresses
  // the 2x carrier-frequency ripple while preserving the slow envelope).
  //
  // IMPORTANT: the scanline buffer pitch is t_far/buffer_size mm/sample, NOT
  // c/SAMPLING_FREQ.  Build the kernel against the actual buffer pitch so the
  // FWHM-in-wavelengths spec is honoured.
  if (!psf_env_lp_ || psf_env_lp_freq_cached_ != probe->get_frequency()) {
    if (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) {
      const float dx_buffer_mm = (buffer_size > 0u)
                                     ? (t_far / static_cast<float>(buffer_size))
                                     : (1.54f / (SAMPLING_FREQ * 1e-6f));
      psf_env_lp_ = create_envelope_lowpass_kernel(
          stream, probe->get_frequency(), dx_buffer_mm,
          1.0f /* fwhm in wavelengths */,
          1.54f /* speed of sound mm/us */);
      psf_env_lp_freq_cached_ = probe->get_frequency();
    } else {
      psf_env_lp_.reset();
      psf_env_lp_freq_cached_ = 0.f;
    }
  }

  if (!psf_lat_) {
    // For point-source probes (e.g. IVUS) element_spacing is 0; use fallback to avoid div-by-zero
    float inv_spacing = probe->get_element_spacing() > 0.f
                           ? 1.f / probe->get_element_spacing()
                           : 1.f;
    float lat_width = probe->get_lateral_resolution();
    // IVUS: single small element → beam diverges with depth. Use beam width at mid-range
    // (typical_depth / aperture) so lateral PSF is finite. ~0.5 mm aperture, 5 mm depth → ~10*lambda.
    if (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) {
      constexpr float ivus_typical_depth_mm = 5.f;
      constexpr float ivus_typical_aperture_mm = 0.5f;
      lat_width = probe->get_wave_length() * (ivus_typical_depth_mm / ivus_typical_aperture_mm);
      inv_spacing = 1.f;  // kernel in mm; applied along angle bins
    }
    psf_lat_ = create_gaussian_psf(stream, lat_width, inv_spacing);
  }

  // Note: psf_elev_ is no longer built. The elevational integration is now
  // a uniform top-hat mean over the sampled ray planes (see mean_planes
  // in simulate(), which runs whenever num_el_samples > 1). Pending E3
  // there is no calibrated Gaussian elevational beam profile. The
  // historical hard-coded `(width = 2 mm, k = freq/c)` gave a ~91-tap
  // kernel that was wrongly scaled and far too wide, silently dimming
  // images when num_el_samples > 1. The psf_elev_ member is left in
  // place for a follow-up that either deletes it or rebuilds a
  // pitch-correct kernel from E3.
  if (psf_elev_) { psf_elev_.reset(); }
}

RaytracingUltrasoundSimulator::SimResult RaytracingUltrasoundSimulator::simulate(
    const BaseProbe* probe, const SimParams& sim_params) {
  CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Simulation", sim_params.stream);

  // Update the ray gen record
  {
    RayGenSbtRecord rg_sbt{};

    rg_sbt.data.probe_type = static_cast<int>(probe->get_probe_type());

    rg_sbt.data.sector_angle = probe->get_sector_angle();
    rg_sbt.data.elevational_height =
        probe->get_num_el_samples() ? probe->get_elevational_height() : 0.f;
    rg_sbt.data.radius = probe->get_radius();

    rg_sbt.data.width = probe->get_width();
    rg_sbt.data.position = probe->get_pose().position_;
    rg_sbt.data.rotation_matrix = probe->get_pose().rotation_matrix_;

    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_.get(), &rg_sbt));
    raygen_record_.upload(&rg_sbt, sim_params.stream);
  }

  auto d_scanlines =
      std::make_unique<CudaMemory>(sim_params.buffer_size * probe->get_num_elements() *
                                       probe->get_num_el_samples() * sizeof(float),
                                   sim_params.stream);
  // Initialize scan lines to zero, the algorithm adds reflected and refracted rays in place
  CUDA_CHECK(cudaMemsetAsync(
      d_scanlines->get_ptr(sim_params.stream), 0, d_scanlines->get_size(), sim_params.stream));

  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "OptiX", sim_params.stream);

    //
    // launch
    //
    Params params{};
    params.scanlines = reinterpret_cast<float*>(d_scanlines->get_ptr(sim_params.stream));
    params.buffer_size = sim_params.buffer_size;
    params.t_far = sim_params.t_far;
    params.min_intensity = sim_params.min_intensity;
    params.max_depth = sim_params.max_depth;
    params.materials =
        reinterpret_cast<Material*>(materials_->get_material_data()->get_ptr(sim_params.stream));
    params.background_material_id = materials_->get_index(world_->get_background_material());
    params.scattering_texture = world_->get_scattering_texture();
    // scattering_resolution_mm: 0.f in SimParams means "auto from probe type" (preserves the
    // historical 10 mm IVUS / 50 mm general default). A positive override comes straight
    // from the YAML config.
    params.scattering_resolution_mm =
        (sim_params.scattering_resolution_mm > 0.f)
            ? sim_params.scattering_resolution_mm
            : ((probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) ? 10.f : 50.f);
    params.handle = world_->get_gas_handle();
    params.source_frequency = probe->get_frequency();
    params.contact_epsilon = sim_params.contact_epsilon;
    params.disable_scatter = sim_params.disable_scatter ? 1u : 0u;
    // Scale scatter integral so vascular/cystic phantoms have visible background; wire phantom
    // remains valid (reflections dominate). 0 = strict integral (dark); ~40 gives usable range.
    //
    // Elevational compensation is a 2D-calibration shim, not a physical
    // aperture model. `mean_planes` always runs when num_el_samples > 1
    // (after the optional in-plane PSF). If scatter_angular_decorrelate is
    // on, OptiX hashes `ray_index = idx.y * dim.x + idx.x`, so elevational
    // planes are independent speckle draws; averaging them would drop
    // speckle RMS by ~sqrt(N) relative to the 2D-calibrated
    // scatter_integral_scale. Pre-multiply by sqrt(N) so post-mean speckle
    // matches that calibration. Do not apply the boost when decorrelate is
    // off: world-y steps (~height/N) are much smaller than
    // scattering_resolution_mm, so the planes are highly correlated and
    // mean_planes barely reduces RMS. Coherent echoes from geometry that
    // varies along y *are* changed by the mean; "Fresnel echoes unchanged"
    // only holds for extruded (y-invariant) phantoms.
    {
      const uint32_t n_el = probe->get_num_el_samples();
      const bool compensate_speckle =
          (n_el > 1u) && sim_params.scatter_angular_decorrelate;
      const float n_el_speckle_compensation =
          compensate_speckle ? std::sqrt(static_cast<float>(n_el)) : 1.f;
      params.scatter_integral_scale =
          sim_params.scatter_integral_scale * n_el_speckle_compensation;
    }
    // Per-scanline scatter decorrelation (see SimParams).
    params.scatter_angular_decorrelate = sim_params.scatter_angular_decorrelate ? 1u : 0u;
    params.frame_seed = sim_params.frame_seed;
    params.ivus_rays_per_scanline =
        (sim_params.ivus_rays_per_scanline >= 1u) ? sim_params.ivus_rays_per_scanline : 1u;

    pipeline_params_.upload(&params, sim_params.stream);

    OPTIX_CHECK(optixLaunch(pipeline_.get(),
                            sim_params.stream,
                            pipeline_params_.get_device_ptr(sim_params.stream),
                            pipeline_params_.get_size(),
                            &shader_binding_table_,
                            probe->get_num_elements(),
                            probe->get_num_el_samples(),
                            /*depth=*/1));
    CUDA_CHECK(cudaPeekAtLastError());
  }

  const uint2 plane_size = make_uint2(sim_params.buffer_size, probe->get_num_elements());
  // Mutable: drops to z = 1 once mean_planes collapses the elevational
  // stack so subsequent 3D-launcher calls (post-Hilbert axial envelope
  // low-pass) operate on the actual 2D buffer extent.
  uint3 size = make_uint3(plane_size.x, plane_size.y, probe->get_num_el_samples());

  if (sim_params.write_debug_images) {
    std::filesystem::create_directory("debug_images");
    for (uint32_t plane = 0; plane < size.z; ++plane) {
      write_image(d_scanlines.get(),
                  plane_size,
                  fmt::format("debug_images/0_scanlines{0:03}.png", plane),
                  nullptr /*min_max*/,
                  plane * plane_size.x * plane_size.y * sizeof(float));
    }
  }

  // Process "RF data" into B - mode image

  // PSF/noise-weight cache update -- runs outside the conv_psf gate so the
  // per-depth additive-noise weight built alongside the lateral PSF is
  // available to stage 0.9 below. update_psfs() is idempotent: it only
  // rebuilds when probe params (or the lateral-PSF kernel type /
  // sigma_theta) change, so unconditional invocation has zero cost on the
  // steady-state hot path.
  update_psfs(probe, sim_params.stream, sim_params.buffer_size, sim_params.t_far,
              sim_params.lateral_psf_kernel_type, sim_params.lateral_psf_sigma_theta_rad);

  // 0.9 Pre-PSF additive Gaussian RF noise (depth-weighted)
  //
  // Adds N(0, (noise_sigma * w(z))^2) per RF sample to the raw post-
  // raytracing buffer, BEFORE the PSF convolutions. The depth weight w(z)
  // (cached as `noise_depth_weight_`, see update_psfs) equals
  // sqrt(sigma_bins(z) / sigma_bins(z_focal)) so that after the L1-
  // normalised depth-dependent lateral PSF concentrates the focal-zone
  // noise, the post-PSF noise standard deviation is uniform across depth.
  // This matches the bench's flat anechoic depth profile (~35 palette,
  // peak-to-trough 14 over r in [4, 29] mm; see Test I in tier1_results.md).
  //
  // Calibration consequence: noise_sigma in the YAML now refers to the
  // input-RF stage AT THE FOCAL DEPTH (where w(z) = 1). The downstream
  // pipeline (PSF + TGC + gain_db + Hilbert + log) is linear up to the
  // Hilbert envelope, so derive_noise_sigma.py remains pipeline-aware: it
  // bisects sigma against the bench gain-54 anechoic palette directly.
  //
  // For non-IVUS probes (no depth-dependent lateral PSF), or when the
  // probe's focal/element params don't define a Gaussian-beam model, the
  // weight buffer is null and we fall back to the unweighted variant.
  // Default (`noise_sigma == 0.f`) is a no-op; both wrappers short-circuit
  // on `sigma <= 0` so existing callers pay no overhead.
  //
  // The noise launchers are 2D (`uint2 plane_size`) and therefore only
  // write plane 0 of a still-3D elevational stack. Harmless while
  // `noise_sigma == 0` (the YAML default). A 3D / per-plane launcher is
  // follow-up if pre-PSF RF noise is re-enabled with num_el_samples > 1.
  if (sim_params.noise_sigma > 0.f) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Additive RF noise", sim_params.stream);
    // noise_seed is decoupled from frame_seed.  When noise_seed is 0 we
    // fall back to frame_seed so the noise tracks the scatter realization.
    // When set
    // non-zero by the caller, the noise realization is decoupled from
    // scatterer seeding entirely.
    const uint32_t base_seed =
        (sim_params.noise_seed != 0u) ? sim_params.noise_seed : sim_params.frame_seed;
    const uint32_t noise_seed = base_seed * 2246822519u + 1u;
    if (noise_depth_weight_ && noise_depth_weight_size_ == sim_params.buffer_size) {
      cuda_algorithms_->add_gaussian_noise_depth_weighted(
          d_scanlines.get(), plane_size, sim_params.noise_sigma,
          noise_depth_weight_.get(), noise_seed, sim_params.stream);
    } else {
      cuda_algorithms_->add_gaussian_noise(d_scanlines.get(), plane_size, sim_params.noise_sigma,
                                           noise_seed, sim_params.stream);
    }
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/0a_additive_noise_pre_psf.png");
    }
  }

  // 1. PSF Convolution
  if (sim_params.conv_psf) {
    {
      CudaTiming cuda_timing(sim_params.enable_cuda_timing, "PSF Convolution", sim_params.stream);
      // (update_psfs already called above; the cache hit makes this cheap.)
      update_psfs(probe, sim_params.stream, sim_params.buffer_size, sim_params.t_far,
                  sim_params.lateral_psf_kernel_type, sim_params.lateral_psf_sigma_theta_rad);

      psf_tmp_.resize(d_scanlines->get_size(), sim_params.stream);
      cuda_algorithms_->convolve_rows(
          d_scanlines.get(), size, &psf_tmp_, psf_ax_.get(), sim_params.stream);
      if (psf_lat_2d_) {
        cuda_algorithms_->convolve_columns_depth_dependent(
            &psf_tmp_, size, d_scanlines.get(), psf_lat_2d_.get(),
            psf_lat_2d_depth_bins_, psf_lat_2d_kernel_radius_, sim_params.stream);
      } else {
        cuda_algorithms_->convolve_columns(
            &psf_tmp_, size, d_scanlines.get(), psf_lat_.get(), sim_params.stream);
      }
    }

    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/1_psf.png");
    }
  }

  // Elevational contract: when num_el_samples > 1 the OptiX buffer is a
  // stack of ray planes. Collapse it to 2D *unconditionally* (not only
  // when conv_psf is on) so TGC / Hilbert / the post-Hilbert axial
  // envelope low-pass always see size.z == 1. Leaving size.z == N after
  // shrinking the buffer to one plane is the historical
  // cudaErrorIllegalAddress. In-plane PSF above still runs on the 3D
  // stack when conv_psf is true; the elevational "PSF" is a uniform
  // top-hat mean until E3 provides a calibrated beam profile.
  if (probe->get_num_el_samples() > 1 && size.z > 1u) {
    auto d_plane = std::make_unique<CudaMemory>(
        sim_params.buffer_size * probe->get_num_elements() * sizeof(float), sim_params.stream);
    cuda_algorithms_->mean_planes(d_scanlines.get(), size, d_plane.get(), sim_params.stream);
    d_scanlines = std::move(d_plane);
    size.z = 1u;
  }

  // 1.5 Time-Gain-Compensation
  {
    CudaTiming cuda_timing(
        sim_params.enable_cuda_timing, "Time-Gain-Compensation", sim_params.stream);

    const bool tgc_size_ok = tgc_curve_ && (tgc_curve_->get_size() / sizeof(float) == sim_params.buffer_size);
    const bool tgc_probe_match = tgc_probe_type_.has_value() && (*tgc_probe_type_ == probe->get_probe_type());
    const bool user_tgc = !sim_params.tgc_control_points.empty();
    // When the caller supplies their own control points we rebuild every frame (no caching),
    // which is the simplest way to honor frame-to-frame changes without bookkeeping the last
    // schedule. The probe-type fallback path keeps its existing cache.
    if (user_tgc || !tgc_curve_ || !tgc_size_ok || !tgc_probe_match) {
      std::vector<ControlPoint> control_points;
      if (user_tgc) {
        control_points.reserve(sim_params.tgc_control_points.size());
        for (const auto& cp : sim_params.tgc_control_points) {
          control_points.push_back({cp.depth_cm, cp.gain_db});
        }
      } else if (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) {
        // IVUS: TGC ~ compensates for tissue (α≈1 dB/(cm·MHz)); avoid over-compensation so
        // wire phantom (lumen) and cystic phantom (tissue) both show correct depth dependence.
        const float tgc_dB_per_cm = 2.f;  // ~α*f for typical IVUS tissue
        control_points = {{0.f, 0.f}, {1.f, tgc_dB_per_cm}};  // (depth [cm], gain [dB])
      } else {
        // Abdominal / general: 0–40 cm
        control_points = {{0.f, 0.f}, {40.f, 28.f}};
      }
      tgc_curve_ = create_piece_wise_tgc(
          sim_params.stream, sim_params.buffer_size, control_points,
          sim_params.t_far, 1540.f, SAMPLING_FREQ);
      // Invalidate the probe-type cache when the curve was built from user control points so
      // that switching back to the default path on a later frame triggers a rebuild.
      tgc_probe_type_ = user_tgc ? std::nullopt
                                 : std::optional<ProbeType>(probe->get_probe_type());
    }
    cuda_algorithms_->mul_row(d_scanlines.get(), plane_size, tgc_curve_.get(), sim_params.stream);
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/2_tgc.png");
  }

  // 1.55 Reference gain
  //
  // Apply the calibrated reference-gain scalar to the post-TGC RF buffer:
  //   rf <- rf * 10^(gain_db / 20)
  // Default `gain_db == 0.f` is a no-op; the CUDA helper short-circuits on
  // scale==1.f so default callers see no overhead.
  //
  // The pre-Hilbert location is mathematically equivalent to scaling the
  // post-Hilbert envelope by the same factor (linearity of
  // |H(s*x)| = s*|H(x)|), so the analytical derivation in
  // `derive_gain_db.py` is unaffected. Doing it here avoids a separate kernel
  // launch on the envelope buffer.
  //
  // This stage lumps two physically distinct effects (see SimParams::gain_db
  // doc): the bench's slider gain offset and the renderer-specific reference-
  // amplitude offset that puts the simulator's RF amplitudes onto the
  // bench's calibrated linear scale. Calibration provenance for the PV .035
  // lives in `instrument-calibration/p035_visions/calibration_delta.md`.
  if (sim_params.gain_db != 0.f) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Reference gain", sim_params.stream);
    const float scale = std::pow(10.f, sim_params.gain_db / 20.f);
    cuda_algorithms_->scale_buffer(d_scanlines.get(), plane_size, scale, sim_params.stream);
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/2a_reference_gain.png");
    }
  }

  // (Additive Gaussian RF noise is applied at stage 0.9 above, pre-PSF, so
  // the same PSF convolution that bandlimits the scatter signal also
  // bandlimits the noise.)

  // 2. Envelope detection
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Envelope detection", sim_params.stream);

    cuda_algorithms_->hilbert_row(d_scanlines.get(), plane_size, sim_params.stream);

    // 2a. Post-envelope additive Gaussian noise.
    //
    // Adds N(envelope_noise_mean, envelope_noise_sigma^2) per envelope pixel
    // BEFORE the post-Hilbert LPF (stage 2b below), so the LPF convolves the
    // noise with the same ~1-wavelength radial kernel that smooths the rest
    // of the envelope. This gives the noise spatial correlation that matches
    // the bench's visible coarse-grained speckle texture (see
    // `instrument-calibration/p035_visions/tier1_results/figures/noise_lpf_effect.png`)
    // while the per-pixel CV is tunable via `envelope_noise_sigma` (independent
    // of the Rayleigh fixed-point of the pre-PSF RF noise stage).
    //
    // The mean term seeds a baseline noise floor for anechoic materials such
    // as water (where the upstream envelope is ~0), so anechoic regions render
    // with a calibrated mean palette that matches the bench (~38-42).
    if (sim_params.envelope_noise_sigma > 0.f || sim_params.envelope_noise_mean != 0.f) {
      // Gain-scale the calibrated envelope_noise so the noise floor tracks
      // the simulated slider (see SimParams docstring above). The reference
      // is the gain_db at which envelope_noise_mean/sigma were
      // measured; when sim_params.envelope_noise_reference_gain_db == 0.f
      // AND gain_db == 0.f the scale is 1 (legacy no-op path).
      float noise_gain_scale = 1.f;
      if (sim_params.envelope_noise_reference_gain_db != 0.f
          || sim_params.gain_db != 0.f) {
        const float dgain_db =
            sim_params.gain_db - sim_params.envelope_noise_reference_gain_db;
        noise_gain_scale = std::pow(10.f, dgain_db / 20.f);
      }
      const float scaled_mean  = sim_params.envelope_noise_mean  * noise_gain_scale;
      const float scaled_sigma = sim_params.envelope_noise_sigma * noise_gain_scale;
      // noise_seed is decoupled from frame_seed (see the pre-PSF noise
      // stage comment above). noise_seed = 0 falls back to frame_seed.
      const uint32_t base_seed =
          (sim_params.noise_seed != 0u) ? sim_params.noise_seed : sim_params.frame_seed;
      // When the TGC depth-scaling flag is set, multiply the post-Hilbert
      // envelope noise by the cached `tgc_curve_` (the same per-sample
      // linear gain that scaled the signal at the TGC stage,
      // normalised to 1.0 at r = 0).  This makes the post-Hilbert noise
      // floor physically correct (analog electronic noise that has been
      // TGC-amplified in the receive chain) and lets the YAML drive the
      // deep-r palette plateau via a real TGC ramp rather than via a
      // depth-flat DC offset in `envelope_noise_mean`.  When the flag is
      // unset OR `tgc_curve_` was not built for the current plane (size
      // mismatch -- can happen on the very first frame before the cache
      // is populated), we silently fall back to the legacy depth-flat
      // path so existing callers keep working.
      const bool can_depth_scale =
          sim_params.envelope_noise_apply_tgc_depth_scaling &&
          tgc_curve_ &&
          (tgc_curve_->get_size() / sizeof(float) == sim_params.buffer_size);
      if (can_depth_scale) {
        cuda_algorithms_->add_gaussian_noise_offset_depth_scaled(
            d_scanlines.get(), plane_size, scaled_mean,
            scaled_sigma, tgc_curve_.get(), base_seed, sim_params.stream);
      } else {
        cuda_algorithms_->add_gaussian_noise_offset(
            d_scanlines.get(), plane_size, scaled_mean,
            scaled_sigma, base_seed, sim_params.stream);
      }
    }

    // 2b. Post-Hilbert low-pass to suppress the 2x-carrier ripple in
    // |analytic_signal| and expose the slow envelope.
    //
    // Without this stage the sim's effective envelope FWHM is a single
    // sample (~30 um) regardless of the kernel's actual envelope width,
    // because the Hilbert-magnitude of a windowed cosine is the windowed
    // cosine modulated by the carrier (peaks at each anti-node, zeros
    // each node).  The textbook envelope detector adds a Gaussian-shaped
    // low-pass of bandwidth ~ 1 wavelength to suppress the carrier
    // ripple while preserving the slow envelope -- this is what real
    // device demodulators do.  We do this in r (the same axis as the
    // axial PSF) right after the Hilbert; psf_env_lp_ is built by
    // update_psfs() with FWHM = 1 wavelength.
    //
    // Cross-references:
    //  - instrument-calibration/p035_visions/debug_sim_axial_envelope.py
    //    -- analytic kernel envelope vs sim rendered envelope showing
    //    the missing low-pass.
    //  - instrument-calibration/p035_visions/diagnose_bench_axial_psf.py
    //    -- bench ensemble axial PSF shape (1/cosh-bandpass, R2=0.98 at
    //    FWHM ~ 300 um = ~2 wavelengths).
    if (psf_env_lp_) {
      // Ensure scratch is sized for the current scanline buffer (psf_tmp_
      // may not have been touched yet on this frame if the lateral kernel
      // path skipped the temp ping-pong).
      psf_tmp_.resize(d_scanlines->get_size(), sim_params.stream);
      // Convolve d_scanlines (envelope magnitude) -> psf_tmp_ (smoothed),
      // then memcpy back into d_scanlines so downstream stages see the
      // low-pass-filtered envelope.  See header / comments above for why
      // this stage is necessary.
      cuda_algorithms_->convolve_rows(
          d_scanlines.get(), size, &psf_tmp_, psf_env_lp_.get(), sim_params.stream);
      CUDA_CHECK(cudaMemcpyAsync(
          d_scanlines->get_ptr(sim_params.stream),
          psf_tmp_.get_ptr(sim_params.stream),
          d_scanlines->get_size(),
          cudaMemcpyDeviceToDevice,
          sim_params.stream));
    }
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/3_envelope_detection.png");
  }

  // 2.5 Ring-down injection
  //
  // Adds the calibrated catheter ring-down residual on top of the envelope
  // buffer (i.e. POST-Hilbert). Off by default (`ring_down.enabled == false`
  // => no signal added, i.e. silent lumen). The waveform is truncated past
  // `extent_mm` so it contributes to the inner zone only.
  //
  // Why post-Hilbert: the calibrated bench template is delivered in
  // envelope-amp units (the YAML loader applies the palette->envelope-amp
  // conversion via
  // amp = 10^(palette/log_mult) - 1 with the speckle floor subtracted), so
  // adding it directly to the envelope buffer is the literal mathematical
  // operation we want — "the catheter contributes this envelope on top of
  // the scattering envelope". Adding it pre-Hilbert would be a category
  // error: cuFFTDx's Hilbert is a length-N cyclic FFT, which would smear
  // any inner-zone transient across the *entire* buffer via spectral
  // side lobes. For the PV .035 ring-down (peak envelope ~45 over 410
  // samples ≈ 3 mm) the cyclic-Hilbert wraparound contributes ~+50 palette
  // at r ≈ 29 mm even though `extent_mm = 3` should bound the influence to
  // the inner 3 mm. Test I (depth uniformity) caught this directly.
  //
  // Provenance of the calibration numbers consumed here is in
  // `instrument-calibration/p035_visions/volcano_s5i.yaml` (E6 / E7).
  //
  // AR-mode caveat (corrected 2026-05-12): the bench template was
  // originally documented as "the AR residual" because private tag
  // 0x00291006 was thought to be the AR-on flag. It is in fact a
  // capability flag (always 1); the AR-on/off state lives in 0x00291007
  // and is 0 (AR-OFF) on every P_035 frame except FILE0013. So the
  // template is the RAW ring-down. Set
  // `sim_params.ring_down.subtract_reference = false` to match P_035
  // and the ivus_test_0508 wire-phantom frames; flip to true (with an
  // AR-on residual template at waveform_path) to render the clinical
  // AR-on default.
  if (sim_params.ring_down.enabled) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Ring-down", sim_params.stream);
    const auto& rd = sim_params.ring_down;

    // Each scanline has `buffer_size` samples covering the radial range
    // [0, t_far_mm] (cf. OptiX raygen offset computation in optix_trace.cu:
    //   `offset = round(t / t_far * (buffer_size - 1))`).
    // So the spatial pitch in the scanlines is `t_far / buffer_size` mm/sample
    // — NOT `c / SAMPLING_FREQ / 2`. Using the wrong convention misaligns the
    // injected ring-down with where it ends up after scan conversion, which
    // matters because the ring-down stage must reproduce the calibrated
    // 1.8 mm peak depth.
    const float samples_per_mm = (sim_params.t_far > 0.f)
                                     ? static_cast<float>(sim_params.buffer_size) / sim_params.t_far
                                     : 0.f;
    uint32_t extent_samples =
        static_cast<uint32_t>(std::min<float>(std::ceil(rd.extent_mm * samples_per_mm),
                                              static_cast<float>(sim_params.buffer_size)));

    // Cache invalidation: rebuild whenever the host-side waveform inputs change.
    // For decay == "measured" we additionally key on the data pointer + byte size
    // so callers that swap the waveform get a fresh upload.
    const float* waveform_ptr = rd.waveform.empty() ? nullptr : rd.waveform.data();
    const size_t waveform_data_size = rd.waveform.size() * sizeof(float);
    const bool stale = !ring_down_waveform_ ||
                       ring_down_decay_cached_ != rd.decay ||
                       ring_down_amplitude_cached_ != rd.amplitude ||
                       ring_down_extent_mm_cached_ != rd.extent_mm ||
                       ring_down_buffer_size_cached_ != sim_params.buffer_size ||
                       ring_down_sample_count_ != extent_samples ||
                       (rd.decay == std::string("measured") &&
                        (ring_down_waveform_data_cached_ != waveform_ptr ||
                         ring_down_waveform_data_size_cached_ != waveform_data_size));

    if (stale && extent_samples > 0) {
      std::vector<float> wf(extent_samples, 0.f);
      if (rd.decay == std::string("exponential")) {
        // amp(r) = amplitude * exp(-r / decay_length); decay_length = extent_mm/3
        // so amp drops to ~5% of peak by `extent_mm`.
        const float decay_length_mm = rd.extent_mm / 3.f;
        for (uint32_t i = 0; i < extent_samples; ++i) {
          const float r_mm = static_cast<float>(i) / samples_per_mm;
          wf[i] = rd.amplitude * std::exp(-r_mm / std::max(decay_length_mm, 1e-6f));
        }
      } else if (rd.decay == std::string("hanning")) {
        // Half-cosine window: amp(0) = amplitude, amp(extent_mm) = 0.
        for (uint32_t i = 0; i < extent_samples; ++i) {
          const float t = static_cast<float>(i) / static_cast<float>(extent_samples);
          wf[i] = rd.amplitude * 0.5f * (1.f + std::cos(static_cast<float>(M_PI) * t));
        }
      } else if (rd.decay == std::string("measured")) {
        // Caller-supplied envelope template, already in envelope-amp units (the
        // YAML loader does the palette->amp conversion via 10^(palette/log_mult)).
        // Truncated to extent_samples; padded with zeros if shorter.
        const uint32_t copy_n = std::min<uint32_t>(extent_samples,
                                                   static_cast<uint32_t>(rd.waveform.size()));
        for (uint32_t i = 0; i < copy_n; ++i) { wf[i] = rd.waveform[i]; }
        // For "measured" the YAML pre-scales the template into envelope amplitude;
        // amplitude is then a multiplicative override in the same units so a value
        // of 1.0 keeps the calibration as-fit. Default amplitude == 0 in the
        // schema, but YAML configs typically set both fields explicitly.
        if (rd.amplitude != 0.f && rd.amplitude != 1.f) {
          // Renormalize so the peak of the supplied template equals `amplitude`.
          float peak = 0.f;
          for (uint32_t i = 0; i < copy_n; ++i) { peak = std::max(peak, std::fabs(wf[i])); }
          if (peak > 0.f) {
            const float scale = rd.amplitude / peak;
            for (uint32_t i = 0; i < copy_n; ++i) { wf[i] *= scale; }
          }
        }
      } else {
        throw std::runtime_error(
            std::string("Ring-down: unknown decay shape '") + rd.decay +
            "' (expected one of: exponential, hanning, measured)");
      }
      ring_down_waveform_ = std::make_unique<CudaMemory>(extent_samples * sizeof(float),
                                                         sim_params.stream);
      ring_down_waveform_->upload(wf.data(), sim_params.stream);
      ring_down_sample_count_ = extent_samples;
      ring_down_decay_cached_ = rd.decay;
      ring_down_amplitude_cached_ = rd.amplitude;
      ring_down_extent_mm_cached_ = rd.extent_mm;
      ring_down_buffer_size_cached_ = sim_params.buffer_size;
      ring_down_waveform_data_cached_ = waveform_ptr;
      ring_down_waveform_data_size_cached_ = waveform_data_size;
    }

    if (extent_samples > 0 && ring_down_waveform_) {
      // Scale the ring-down envelope by the same gain factor that was
      // applied to the scatter signal pre-Hilbert (stage 1.55). The bench's
      // catheter ring-down rides on the receive chain, so it scales
      // linearly with the receiver gain (i.e. with `gain_db`).  Before this
      // fix the ring-down was injected at a fixed envelope amplitude
      // independent of `gain_db`, so at sliders below the calibration
      // anchor the sim ring-down stayed saturated while the bench scaled
      // down -- caught in Test E's (g40, D60) pair where the sim peak was
      // 221.9 palette vs bench 152.8 (45.3 % gap).  Multiplying by
      // 10^(gain_db / 20) here brings the ring-down onto the same
      // receive-gain axis as the scatter envelope.  The calibration of
      // `ring_down.amplitude` itself must be re-derived against the new
      // pipeline (see `derive_ringdown_amplitude.py` -- it runs at the
      // calibration anchor gain so the bisection still converges, just to
      // a value reduced by exactly the new factor).
      const float ringdown_gain_scale = std::pow(10.f, sim_params.gain_db / 20.f);
      cuda_algorithms_->add_row(d_scanlines.get(), plane_size, ring_down_waveform_.get(),
                                ring_down_sample_count_, /*scale=*/ringdown_gain_scale,
                                sim_params.stream);
    }
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/3b_ringdown.png");
    }
  }

  // 3. Log compression
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Log compression", sim_params.stream);

    cuda_algorithms_->log_compression(
        d_scanlines.get(), plane_size,
        sim_params.log_multiplier, sim_params.log_floor,
        sim_params.stream);
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/4_log_compression.png");
  }

  // 3.5 Display window
  //
  // Direct palette clamp to [reject_palette, saturation_palette]. With both
  // at 0.f (default) this stage is a no-op so default callers see the
  // historical pure-log output. Calibrated PV .035 settings
  // (reject_palette=11, saturation_palette=239 from gain_lut.json) reproduce
  // the device's reject floor and saturation ceiling directly.
  //
  // When `reject_palette_softness > 0` the reject floor uses a softplus
  // blend instead of a hard `max(palette, reject)` clamp. This removes the
  // spurious histogram spike at the floor that the hard clamp
  // creates under post-envelope Gaussian noise with tails below the floor.
  if (sim_params.saturation_palette > sim_params.reject_palette) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Display window", sim_params.stream);
    if (sim_params.reject_palette_softness > 0.f) {
      cuda_algorithms_->apply_display_window_soft(
          d_scanlines.get(), plane_size,
          sim_params.reject_palette, sim_params.saturation_palette,
          sim_params.reject_palette_softness, sim_params.stream);
    } else {
      cuda_algorithms_->apply_display_window(
          d_scanlines.get(), plane_size,
          sim_params.reject_palette, sim_params.saturation_palette,
          sim_params.stream);
    }
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/4b_display_window.png");
    }
  }

  // 4. Median clip filter for speckle noise reduction
  // Speckle noise is a type of salt-and-pepper noise, where the noise is randomly distributed
  // across the image. The median filter is a good way to reduce speckle noise.
  // https://en.wikipedia.org/wiki/Salt-and-pepper_noise
  if (sim_params.median_clip_filter) {
    {
      CudaTiming cuda_timing(
          sim_params.enable_cuda_timing, "Median clip filter", sim_params.stream);

      // Create temporary buffer for filter output
      auto d_filtered = std::make_unique<CudaMemory>(d_scanlines->get_size(), sim_params.stream);

      // Median clip filter parameters are now sourced from SimParams; defaults match the
      // previous literals (kernel=5, dMin=-60 dB, dMax=0 dB) so behavior is unchanged unless
      // a YAML overrides them.
      cuda_algorithms_->median_clip_filter(
          d_scanlines.get(), plane_size, d_filtered.get(),
          sim_params.median_clip_size,
          sim_params.median_clip_d_min_db, sim_params.median_clip_d_max_db,
          sim_params.stream);

      // Replace original with filtered data
      d_scanlines = std::move(d_filtered);
    }
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/5_median_clip.png");
    }
  }

  // 4.5 Catheter sheath dead-zone mask
  //
  // Overwrite the inner radial samples of the final palette buffer to
  // reproduce the bench's catheter zone.  The bench's catheter wall blocks
  // any acquired signal for r < ~1.4 mm, so the device renders that region
  // at a uniform floor.  Without this mask the additive noise stage fills
  // the dead zone with the calibrated noise floor (which varies with gain),
  // which differs visibly from the bench's flat inner zone.  We apply the
  // mask AFTER log compression / display window / median clip so the masked
  // palette is independent of the display-window blending used elsewhere
  // in the pipeline.
  //
  // The c_take2_water E6 corpus shows the bench's catheter zone sitting at
  // palette ~ 11 (= reject_palette / soft-reject floor), gain-independent,
  // NOT at literal palette 0 as the original P_035 anchor
  // assumed.  The caller now passes `reject_palette` as the dead-zone fill
  // value (was hard-coded 0.f) so the simulator matches the bench's
  // observed inner-zone palette and Test E `RMS_inner_3mm` drops from the
  // ~21 floor caused by the (sim 0 vs bench 11) mismatch.
  //
  // dr_mm = t_far / buffer_size; dead_zone_samples = floor(dead_zone_mm / dr_mm).
  // Default `catheter_dead_zone_mm == 0.f` is a no-op (host wrapper short-
  // circuits on `dead_zone_samples == 0`).
  if (sim_params.catheter_dead_zone_mm > 0.f) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Catheter dead-zone", sim_params.stream);
    const float dr_mm = sim_params.t_far / static_cast<float>(sim_params.buffer_size);
    const uint32_t dead_zone_samples = (dr_mm > 0.f)
        ? static_cast<uint32_t>(sim_params.catheter_dead_zone_mm / dr_mm)
        : 0u;
    cuda_algorithms_->zero_inner_radial(d_scanlines.get(), plane_size, dead_zone_samples,
                                        sim_params.reject_palette,
                                        sim_params.stream);
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/5b_catheter_deadzone.png");
    }
  }

  // 5. Scan conversion - based on probe type
  std::unique_ptr<CudaMemory> b_mode;
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Scan conversion", sim_params.stream);

    switch (probe->get_probe_type()) {
      case ProbeType::PROBE_TYPE_PHASED_ARRAY:
        b_mode = cuda_algorithms_->scan_convert_phased(d_scanlines.get(),
                                                       plane_size,
                                                       probe->get_sector_angle(),
                                                       sim_params.t_far,
                                                       sim_params.b_mode_size,
                                                       sim_params.stream);
        break;
      case ProbeType::PROBE_TYPE_LINEAR_ARRAY:
        b_mode = cuda_algorithms_->scan_convert_linear(d_scanlines.get(),
                                                       plane_size,
                                                       probe->get_width(),
                                                       sim_params.t_far,
                                                       sim_params.b_mode_size,
                                                       sim_params.stream);
        break;
      case ProbeType::PROBE_TYPE_CURVILINEAR:
        b_mode = cuda_algorithms_->scan_convert_curvilinear(d_scanlines.get(),
                                                            plane_size,
                                                            probe->get_sector_angle(),
                                                            probe->get_radius(),
                                                            sim_params.t_far + probe->get_radius(),
                                                            sim_params.b_mode_size,
                                                            sim_params.stream);
        break;
      case ProbeType::PROBE_TYPE_IVUS:
        b_mode = cuda_algorithms_->scan_convert_ivus(d_scanlines.get(),
                                                     plane_size,
                                                     sim_params.b_mode_size,
                                                     sim_params.stream);
        break;
    }
  }

  // extract min and max x and z values based on probe type
  float min_x = 0.f;
  float max_x = 0.f;
  float min_z = 0.f;
  float max_z = 0.f;

  switch (probe->get_probe_type()) {
    case ProbeType::PROBE_TYPE_PHASED_ARRAY: {
      float sector_angle_rad = probe->get_sector_angle() * M_PI / 180.0f;
      min_x = sim_params.t_far * std::sin(-sector_angle_rad / 2.0f);
      max_x = sim_params.t_far * std::sin(sector_angle_rad / 2.0f);
      min_z = 0.f;  // Phased array image typically starts at depth 0 from the origin
      max_z = sim_params.t_far * std::cos(sector_angle_rad / 2.0f);
    } break;
    case ProbeType::PROBE_TYPE_LINEAR_ARRAY:
      min_x = -probe->get_width() / 2.0f;
      max_x = probe->get_width() / 2.0f;
      min_z = 0.f;
      max_z = sim_params.t_far;
      break;
    case ProbeType::PROBE_TYPE_IVUS:
      // Unwrapped display: x = angle (degrees 0..360), z = depth (mm, 0..t_far)
      min_x = 0.f;
      max_x = 360.f;
      min_z = 0.f;
      max_z = sim_params.t_far;
      break;
    case ProbeType::PROBE_TYPE_CURVILINEAR:
    default:  // Fallback for safety or new types
    {
      float sector_angle_in_rad = probe->get_sector_angle() * M_PI / 180.0f;  // Use generic getter
      float current_radius = probe->get_radius();                             // Use generic getter
      min_x = (current_radius + sim_params.t_far) * std::sin(-sector_angle_in_rad / 2.0f);
      max_x = (current_radius + sim_params.t_far) * std::sin(sector_angle_in_rad / 2.0f);
      float z_behind_image_origin = current_radius * (1 - std::cos(sector_angle_in_rad / 2.0f));
      min_z = -sim_params.t_far;  // Curvilinear depth can be negative relative to image origin if
                                  // radius is large
      max_z = z_behind_image_origin;
    } break;
  }

  // Update the class member variables
  min_x_ = min_x;
  max_x_ = max_x;
  min_z_ = min_z;
  max_z_ = max_z;

  SimResult result;
  result.rf_data = std::move(d_scanlines);
  result.b_mode = std::move(b_mode);

  return result;
}

}  // namespace raysim
