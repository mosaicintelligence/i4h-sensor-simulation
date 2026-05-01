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
#include <cmath>
#include <filesystem>

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

struct ControlPoint {
  float depth;  // cm
  float amp;    // dB
};

/**
 * Create a piece-wise linear TGC curve from control points.
 *
 * @param depth_samples Number of samples along depth
 * @param control_points Vector of (depth_cm, gain_db) pairs
 * @param c Speed of sound in tissue (m/s) (optional)
 * @param fs Sampling frequency (Hz) (optional)
 * @return TGC curve interpolated from control points
 */
static std::unique_ptr<CudaMemory> create_piece_wise_tgc(
    cudaStream_t stream, uint32_t depth_samples, const std::vector<ControlPoint>& control_points,
    float c = 1540.f, float fs = 50e6f) {
  std::vector<float> tgc_curve(depth_samples);

  float first_value;
  for (uint32_t i = 0; i < depth_samples; ++i) {
    // Time
    const float t = i / fs;
    // Depth in cm
    const float depth = (c * t / 2.f) * 100.f;

    // Interpolate between control points
    auto it = control_points.begin();
    while ((it != control_points.end()) && (depth < it->depth)) { ++it; }
    float value;
    if (it == control_points.end()) {
      value = control_points.back().amp;
    } else if (depth <= it->depth) {
      value = it->amp;
    } else {
      float depth_min = it->depth;
      float amp_min = it->amp;
      ++it;
      float depth_max = it->depth;
      float amp_max = it->amp;
      value = amp_min + (amp_max - amp_min) * ((depth - depth_min) / (depth_max - depth_min));
    }

    // Convert from dB to linear scale
    value = std::pow(10.f, value / 20.f);

    // Normalize to start at 1
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
                                                uint32_t buffer_size, float t_far) {
  if (probe_frequency_ != probe->get_frequency()) {
    probe_frequency_ = probe->get_frequency();
    psf_ax_.reset();
    psf_lat_.reset();
    psf_lat_2d_.reset();
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
        psf_lat_2d_t_far_ != t_far) {
      psf_lat_2d_element_radius_ = el_radius;
      psf_lat_2d_focal_length_ = focal_mm;
      psf_lat_2d_buffer_size_ = buffer_size;
      psf_lat_2d_t_far_ = t_far;
      constexpr uint32_t depth_bins = 64;
      constexpr uint32_t kernel_radius = 64;
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
      const uint32_t num_angular_rays = probe->get_num_elements();
      const float two_pi = 6.28318530717958647692f;
      std::vector<float> k2d(depth_bins * kernel_len, 0.f);
      for (uint32_t b = 0; b < depth_bins; ++b) {
        const float depth_mm = (static_cast<float>(b) + 0.5f) / static_cast<float>(depth_bins) * t_far;
        const float z = depth_mm - focal_mm;  // distance from focus (mm)
        const float sigma_mm = w0_mm * std::sqrt(1.f + (z * z) / (z_R_mm * z_R_mm));
        const float depth_safe = std::max(depth_mm, 0.5f);
        const float sigma_bins = sigma_mm * static_cast<float>(num_angular_rays) / (two_pi * depth_safe);
        float* row = k2d.data() + b * kernel_len;
        float sum = 0.f;
        for (uint32_t i = 0; i < kernel_len; ++i) {
          const float x = static_cast<float>(static_cast<int>(i) - static_cast<int>(kernel_radius));
          const float v = std::exp(-0.5f * (x * x) / (sigma_bins * sigma_bins));
          row[i] = v;
          sum += v;
        }
        if (sum > 0.f) {
          for (uint32_t i = 0; i < kernel_len; ++i) { row[i] /= sum; }
        }
      }
      psf_lat_2d_ = std::make_unique<CudaMemory>(k2d.size() * sizeof(float), stream);
      psf_lat_2d_->upload(k2d.data(), stream);
      psf_lat_2d_depth_bins_ = depth_bins;
      psf_lat_2d_kernel_radius_ = kernel_radius;
    }
  } else {
    psf_lat_2d_.reset();
    psf_lat_2d_depth_bins_ = 0;
    psf_lat_2d_kernel_radius_ = 0;
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

  if ((probe->get_num_el_samples() > 1) && !psf_elev_) {
    psf_elev_ = create_gaussian_psf(stream, 2.f, probe->get_elevational_spatial_frequency());
  }
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
    params.scatter_integral_scale = sim_params.scatter_integral_scale;

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
  const uint3 size = make_uint3(plane_size.x, plane_size.y, probe->get_num_el_samples());

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

  // 1. PSF Convolution
  if (sim_params.conv_psf) {
    {
      CudaTiming cuda_timing(sim_params.enable_cuda_timing, "PSF Convolution", sim_params.stream);

      update_psfs(probe, sim_params.stream, sim_params.buffer_size, sim_params.t_far);

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

      if (probe->get_num_el_samples() > 1) {
        cuda_algorithms_->convolve_planes(
            d_scanlines.get(), size, &psf_tmp_, psf_elev_.get(), sim_params.stream);

        auto d_plane = std::make_unique<CudaMemory>(
            sim_params.buffer_size * probe->get_num_elements() * sizeof(float), sim_params.stream);
        cuda_algorithms_->mean_planes(&psf_tmp_, size, d_plane.get(), sim_params.stream);

        d_scanlines = std::move(d_plane);
      }
    }

    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/1_psf.png");
    }
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
          sim_params.stream, sim_params.buffer_size, control_points, 1540.f, SAMPLING_FREQ);
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

  // 1.55 Reference gain (Pass 3b: applied PRE-ring-down)
  //
  // Apply the calibrated reference-gain scalar to the post-TGC RF buffer:
  //   rf <- rf * 10^(gain_db / 20)
  // Default `gain_db == 0.f` is a no-op; the CUDA helper short-circuits on
  // scale==1.f so default callers see no overhead.
  //
  // Order matters: this stage MUST run before ring-down injection. Ring-down
  // amplitude is specified in the YAML in **bench-calibrated envelope units**
  // (i.e. amp == 46.4 means peak displays at log10(46.4)*log_multiplier
  // palette in the final image), so the renderer has to scale the raytraced
  // scattering UP to bench scale BEFORE adding ring-down — otherwise the
  // gain_db scalar would also amplify the already-bench-scale ring-down and
  // saturate the entire image. Linearity of |Hilbert(s*x)| = s*|Hilbert(x)|
  // means scaling RF here is equivalent to scaling envelope post-Hilbert.
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

  // 1.6 Ring-down injection (Pass 2)
  //
  // Adds the calibrated catheter ring-down residual to every A-line between TGC
  // and envelope detection. Off by default (`sim_params.ring_down.enabled ==
  // false` => no signal at all, i.e. quiet lumen). When on, the waveform is
  // truncated past `extent_mm` (converted to sample count via the sampling
  // frequency and the round-trip speed of sound) and added pre-Hilbert so the
  // existing envelope detection + log compression handle the result naturally.
  //
  // Provenance of the calibration numbers consumed here is in
  // `instrument-calibration/p035_visions/volcano_s5i.yaml` (E6 / E7) and the
  // residual is what survives the device's Acoustic Reference subtraction
  // (private DICOM tag 0x00291006 = 1 in all PV .035 frames).
  if (sim_params.ring_down.enabled) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Ring-down", sim_params.stream);
    const auto& rd = sim_params.ring_down;

    // Each scanline has `buffer_size` samples covering the radial range
    // [0, t_far_mm] (cf. OptiX raygen offset computation in optix_trace.cu:
    //   `offset = round(t / t_far * (buffer_size - 1))`).
    // So the spatial pitch in the scanlines is `t_far / buffer_size` mm/sample
    // — NOT `c / SAMPLING_FREQ / 2`. Using the wrong convention misaligns the
    // injected ring-down with where it ends up after scan conversion, which
    // matters because Pass 2 must reproduce the calibrated 1.8 mm peak depth.
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
      cuda_algorithms_->add_row(d_scanlines.get(), plane_size, ring_down_waveform_.get(),
                                ring_down_sample_count_, /*scale=*/1.f, sim_params.stream);
    }
    if (sim_params.write_debug_images) {
      write_image(d_scanlines.get(), plane_size, "debug_images/2b_ringdown.png");
    }
  }

  // 2. Envelope detection
  {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Envelope detection", sim_params.stream);

    cuda_algorithms_->hilbert_row(d_scanlines.get(), plane_size, sim_params.stream);
  }
  if (sim_params.write_debug_images) {
    write_image(d_scanlines.get(), plane_size, "debug_images/3_envelope_detection.png");
  }

  // (Pass 3b: reference gain stage moved to step 1.55, before ring-down.
  //  See the long comment there for why ring-down has to be injected on the
  //  bench-scale RF buffer rather than on the raw raytracer output.)

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

  // 3.5 Display window (Pass 3b)
  //
  // Direct palette clamp to [reject_palette, saturation_palette]. With both
  // at 0.f (default) this stage is a no-op so default callers see the
  // historical pure-log output. Calibrated PV .035 settings
  // (reject_palette=11, saturation_palette=239 from gain_lut.json) reproduce
  // the device's reject floor and saturation ceiling directly.
  if (sim_params.saturation_palette > sim_params.reject_palette) {
    CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Display window", sim_params.stream);
    cuda_algorithms_->apply_display_window(
        d_scanlines.get(), plane_size,
        sim_params.reject_palette, sim_params.saturation_palette,
        sim_params.stream);
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
