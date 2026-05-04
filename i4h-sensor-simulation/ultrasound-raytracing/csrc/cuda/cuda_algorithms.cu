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

#include "raysim/cuda/cuda_algorithms.hpp"

#include <sutil/vec_math.h>
// cub/cub.cuh was previously included for the per-frame quantile sort in
// log_compression; Pass 3 (K2) removed that reduction so the include is no
// longer needed.
#include <cufftdx/cufftdx.hpp>

namespace raysim {

static __global__ void normalize_kernel(float* __restrict__ buffer, uint2 size,
                                        const float* __restrict__ min_max) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  buffer += index.y * size.x + index.x;

  *buffer = (*buffer - min_max[0]) / (min_max[1] - min_max[0]);
}

/// @todo the convolution kernels are using a naiive implementation, improve see
/// https://github.com/zchee/cuda-sample/tree/master/3_Imaging/convolutionSeparable

static __global__ void convolve_rows_kernel(const float* __restrict__ source, uint3 size,
                                            float* __restrict__ dst,
                                            const float* __restrict__ kernel,
                                            uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const int k_min = -min(index.x, kernel_radius);
  const int k_max = min(size.x - 1 - index.x, kernel_radius);
  const int offset = ((index.z * size.y) + index.y) * size.x + index.x;
  source += offset;

  float sum = 0.f;
  for (int k = k_min; k <= k_max; ++k) { sum += source[k] * kernel[k + kernel_radius]; }

  dst[offset] = sum;
}

static __global__ void convolve_columns_kernel(const float* __restrict__ source, uint3 size,
                                               float* __restrict__ dst,
                                               const float* __restrict__ kernel,
                                               uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const int k_min = -min(index.y, kernel_radius);
  const int k_max = min(size.y - 1 - index.y, kernel_radius);
  const int offset = ((index.z * size.y) + index.y) * size.x + index.x;
  source += offset + k_min * size.x;

  float sum = 0.f;
  for (int k = k_min; k <= k_max; ++k) {
    sum += *source * kernel[k + kernel_radius];
    source += size.x;
  }

  dst[offset] = sum;
}

// Pass 5: depth-dependent column convolution with CYCLIC angular wrap.
//
// IVUS images are intrinsically periodic in angle (360° = num_scanlines), so
// the lateral PSF convolution must wrap around the angular boundary instead
// of truncating at index.y == 0 / index.y == size.y - 1. Truncation produced
// two depth-uniformity artifacts that the host-side kernel build (Pass 5
// `update_psfs`) could not compensate for:
//   * Boundary darkening within ~kernel_radius bins of the seam.
//   * Loss of L1/L2 mass when the host-built kernel was wider than the
//     truncation window, breaking the depth-dependent normalization that
//     keeps the post-Hilbert envelope magnitude depth-invariant.
//
// The cyclic indexing here is paired with two host-side fixes (see
// `raytracing_ultrasound_simulator.cpp` §`update_psfs`):
//   1. `kernel_radius = num_scanlines / 2` so the kernel can span the full
//      half-circumference at the most divergent (near-field) depth.
//   2. L2 normalization (sum_sq = 1) so the post-Hilbert envelope mean
//      becomes depth-invariant for random-scatter input — preserving the
//      bench's flat in-water bg statistics.
//
// `convolve_columns_depth_dependent` is only called for IVUS today (the
// host build only sets `psf_lat_2d_` for `PROBE_TYPE_IVUS`), so making the
// indexing cyclic unconditionally is safe.
static __global__ void convolve_columns_depth_dependent_kernel(const float* __restrict__ source,
                                                               uint3 size,
                                                               float* __restrict__ dst,
                                                               const float* __restrict__ kernel_2d,
                                                               uint32_t depth_bins,
                                                               uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const uint32_t kernel_len = 2 * kernel_radius + 1;
  const uint32_t depth_bin = (size.x > 0)
                                 ? min((index.x * depth_bins) / size.x, depth_bins - 1u)
                                 : 0u;
  const float* kernel = kernel_2d + depth_bin * kernel_len;

  const int N = static_cast<int>(size.y);
  const int radius = static_cast<int>(kernel_radius);
  const int row_stride = static_cast<int>(size.x);
  const int plane_offset = static_cast<int>(index.z * size.y * size.x);
  const int col_offset = static_cast<int>(index.x);
  const int dst_offset = plane_offset + static_cast<int>(index.y) * row_stride + col_offset;

  float sum = 0.f;
  // Cyclic sum over k in [-radius, +radius]; wrap source row index modulo N.
  // The kernel is centred at k == 0 (kernel index == kernel_radius). The
  // wrap math `((iy + k) % N + N) % N` handles negative k correctly under
  // C/CUDA's truncated integer modulo semantics.
  const int iy = static_cast<int>(index.y);
  for (int k = -radius; k <= radius; ++k) {
    const int srcy = ((iy + k) % N + N) % N;
    const int src_offset = plane_offset + srcy * row_stride + col_offset;
    sum += source[src_offset] * kernel[k + radius];
  }

  dst[dst_offset] = sum;
}

static __global__ void convolve_planes_kernel(const float* __restrict__ source, uint3 size,
                                              float* __restrict__ dst,
                                              const float* __restrict__ kernel,
                                              uint32_t kernel_radius) {
  const uint3 index = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

  if ((index.x >= size.x) || (index.y >= size.y) || (index.z >= size.z)) { return; }

  const int k_min = -min(index.z, kernel_radius);
  const int k_max = min(size.z - 1 - index.z, kernel_radius);
  const int offset = ((index.z * size.y) + index.y) * size.x + index.x;
  source += offset + k_min * size.x * size.y;

  float sum = 0.f;
  for (int k = k_min; k <= k_max; ++k) {
    sum += *source * kernel[k + kernel_radius];
    source += size.x * size.y;
  }

  dst[offset] = sum;
}

static __global__ void mean_planes_kernel(const float* __restrict__ source, uint3 size,
                                          float* __restrict__ dst) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const int offset = index.y * size.x + index.x;
  source += offset;

  const int plane_size = size.x * size.y;

  float sum = 0.f;
  for (int plane = 0; plane < size.z; ++plane) {
    sum += *source;
    source += plane_size;
  }

  dst[offset] = sum / size.z;
}

// Pass 3 (K2v2): fixed-reference log compression.
//
// Implements the spec mapping
//   pixel = log_multiplier * log10(amp / log_floor)
// where `minimum == log_floor` is the calibration anchor: amp == log_floor
// maps to pixel == 0, amp == 10*log_floor maps to pixel == log_multiplier,
// amp < log_floor maps to a *negative* palette (which the post-log display
// window can then clamp to the device's reject palette). This is the spec
// mapping used by the calibration sheet (`gain_lut.json` /
// `volcano_s5i.yaml`).
//
// Two safety clamps:
//   * `floor_safe = max(log_floor, 1e-30)` so a degenerate `log_floor == 0`
//     in user-supplied SimParams does not divide by zero.
//   * `amp_safe = max(amp, 1e-30 * floor_safe)` so envelope amp == 0 (or any
//     denormal) maps to a finite, very-negative palette of about
//     `-30 * log_multiplier` instead of NaN/-inf. Downstream stages
//     (display window) then clamp this to `reject_palette`.
//
// Note: this changes the meaning of the historical Pass 3a K2 mapping
// `pixel = log_multiplier * log10(max(amp, log_floor))` by an additive
// offset of `-log_multiplier * log10(log_floor)`. The Pass 1 default was
// `log_floor = 1e-19`; that default is now `1.0` (see SimParams::log_floor)
// so default callers get a useful `[-60, 0]`-ish palette range that matches
// the existing `examples/ivus_example.py` `MIN_VAL/MAX_VAL` window.
static __global__ void log_compression_kernel(float* __restrict__ buffer, uint2 size,
                                              float mutliplicator, float minimum) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;

  const float floor_safe = fmaxf(minimum, 1e-30f);
  const float amp_safe = fmaxf(buffer[offset], 1e-30f * floor_safe);
  buffer[offset] = log10f(amp_safe / floor_safe) * mutliplicator;
}

static __global__ void mul_rows_kernel(float* __restrict__ buffer, uint2 size,
                                       const float* __restrict__ multiplicator) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  buffer[index.y * size.x + index.x] *= multiplicator[index.x];
}

// Pass 2: add a per-depth vector to every row of the buffer in place.
// `addend` has length `addend_size` <= size.x; samples past addend_size are untouched.
// Used by the ring-down injection stage.
static __global__ void add_row_kernel(float* __restrict__ buffer, uint2 size,
                                      const float* __restrict__ addend, uint32_t addend_size,
                                      float scale) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  if (index.x < addend_size) {
    buffer[index.y * size.x + index.x] += scale * addend[index.x];
  }
}

// Pass 3: in-place scalar multiply on every element of `buffer`.
//
// Used by the reference-gain stage to bring raytraced RF amplitudes onto the
// bench's calibrated linear scale before ring-down injection
// (`rf <- rf * 10^(gain_db / 20)`; equivalent to scaling envelope post-
// Hilbert by linearity of |Hilbert(s*x)| = s*|Hilbert(x)|). The host wrapper
// short-circuits on `scale == 1.f` so default callers don't even launch the
// kernel.
static __global__ void scale_buffer_kernel(float* __restrict__ buffer, uint2 size,
                                           float scale) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;
  buffer[offset] *= scale;
}

// Pass 6: additive Gaussian RF noise.
//
// Per the calibration sheet (volcano_s5i.yaml processing.noise.{type, sigma})
// the bench's measured noise is "complex Gaussian per quadrature" — i.e. the
// underlying RF noise (BEFORE envelope detection) is Gaussian with the
// calibrated sigma in RF-amplitude units at the reference gain. This kernel
// adds N(0, sigma^2) to every element of the post-gain RF buffer, just before
// Hilbert. The Hilbert transform of a Gaussian RF stream is itself Gaussian
// with the same variance, so the post-Hilbert envelope of pure noise becomes
// Rayleigh(sigma) with mean = sigma * sqrt(pi/2). After log compression the
// noise floor lifts the per-pixel envelope distribution off the lower
// rejection clamp (palette 11), which is the dominant contributor to the
// bimodal sim-vs-bench palette histogram (test I residual + test F not
// previously evaluated).
//
// Box-Muller transform on two PCG-hashed uniforms per buffer element (the
// same hash family used by Pass 5b's scatter decorrelation, see
// optix_trace.cu::pcg_hash; standard Jarzynski & Olano 2020 / O'Neill 2014
// PCG mix — adequate for per-sample additive-noise jitter, not for any
// security-sensitive use). u1 is clamped to (0, 1] by adding 1/2^24 so the
// log term cannot overflow.
//
// Disabled (skipped) when sigma <= 0; default callers pay no cost.
static __device__ uint32_t pcg_hash_noise(uint32_t x) {
  uint32_t state = x * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

static __device__ float pcg_to_unit_float_noise(uint32_t x) {
  return (x >> 8) * (1.f / 16777216.f);
}

static __global__ void add_gaussian_noise_kernel(float* __restrict__ buffer, uint2 size,
                                                 float sigma, uint32_t seed) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;
  const uint32_t base = offset * 2u + seed * 2654435761u;
  // u1 in (0, 1] (avoid 0 to keep -2 ln u1 finite); u2 in [0, 1).
  const float u1 = pcg_to_unit_float_noise(pcg_hash_noise(base + 0u)) + (1.f / 16777216.f);
  const float u2 = pcg_to_unit_float_noise(pcg_hash_noise(base + 1u));
  const float radius = sqrtf(-2.f * logf(u1));
  // Single Gaussian sample per element; the second Box-Muller output (sin
  // term) is discarded — we save one register and the work is dwarfed by
  // the surrounding Hilbert/log-compression stages.
  const float z = radius * __cosf(6.28318530717958647692f * u2);
  buffer[offset] += sigma * z;
}

// Pass 6 v2: catheter dead-zone mask. Zero the inner radial samples where
// the catheter sheath physically blocks any acquired signal -- on the bench
// this region renders as pure black (palette 0) for r < ~1.4 mm. Without
// this mask the additive noise stage (and any leaking ring-down energy)
// fills the dead zone with a noise floor, which differs visibly from the
// bench's solid-black inner zone.
//
// Applied at the very end of the pipeline (post log-compression, post
// display window) so the masked palette is exactly 0, deeper than the
// device's reject_palette (11). This matches the bench appearance.
//
// `dead_zone_samples` is the number of leading radial samples to zero.
// Default (dead_zone_samples == 0) is a no-op; the wrapper short-circuits.
static __global__ void zero_inner_radial_kernel(float* __restrict__ buffer, uint2 size,
                                                uint32_t dead_zone_samples) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if ((index.x >= size.x) || (index.y >= size.y)) { return; }
  if (index.x < dead_zone_samples) {
    const uint32_t offset = index.y * size.x + index.x;
    buffer[offset] = 0.f;
  }
}

// Pass 3b: post-log display window. Direct clamp to
// `[reject_palette, saturation_palette]` in palette units. This reproduces
// the device's reject / saturation palette behaviour: any amplitude whose
// post-log palette is below `reject_palette` is pushed up to the reject
// floor (no negative pixels leak through), and any amplitude above
// `saturation_palette` is clipped to the saturation ceiling.
//
// The previous Pass 2 kernel did two things at once: clamp the dB window AND
// re-zero the lower bound (subtract `reject_db`). The re-zero step shifted
// the entire palette so that `reject_db` mapped to palette 0 and the
// device's own `reject_palette` value (e.g. 11) was no longer reached;
// worse, the input palette of 0 (which K2v2's negative outputs *should*
// map to the reject floor) was instead shifted up to the saturation
// ceiling. This kernel matches the calibration sheet semantics: the post-
// log palette is *already* in absolute palette units (because
// `log_compression_kernel` uses the calibrated `log_multiplier` /
// `log_floor`), so the display window only has to enforce the device's
// hard floor and ceiling.
static __global__ void display_window_kernel(float* __restrict__ buffer, uint2 size,
                                             float reject_palette,
                                             float saturation_palette) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;
  buffer[offset] = fminf(fmaxf(buffer[offset], reject_palette), saturation_palette);
}

static __global__ void median_clip_kernel(const float* __restrict__ source, uint2 size,
                                          float* __restrict__ dst, uint32_t filter_size,
                                          float d_min, float d_max) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= size.x) || (index.y >= size.y)) { return; }

  const uint32_t offset = index.y * size.x + index.x;

  // Calculate median filter bounds
  const int half_size = filter_size / 2;
  const int y_min = max(0, (int)index.y - half_size);
  const int y_max = min((int)size.y - 1, (int)index.y + half_size);

  // Collect values for median calculation
  float values[11];  // Maximum filter size is 11
  int count = 0;

  for (int y = y_min; y <= y_max; ++y) { values[count++] = source[y * size.x + index.x]; }

  // Simple bubble sort for small arrays (efficient for small filter sizes)
  for (int i = 0; i < count - 1; ++i) {
    for (int j = 0; j < count - i - 1; ++j) {
      if (values[j] > values[j + 1]) {
        float temp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = temp;
      }
    }
  }

  // Get median value
  float median = values[count / 2];

  // Calculate bounds
  float lower_bound = fmaxf(d_min, median);
  float upper_bound = fminf(d_max, median);

  // Clamp original value to bounds
  float original_value = source[offset];
  dst[offset] = fmaxf(lower_bound, fminf(original_value, upper_bound));
}

// cuFFTDx needs to know the SM architecture, this is only known when compiling device code. Use the
// lowest supported arch for host code.
#ifdef __CUDA_ARCH__
#define CUFFTDX_ARCH __CUDA_ARCH__
#else
#define CUFFTDX_ARCH 700
#endif

// The FFT size needs to be known at compile time
static constexpr uint32_t HILBERT_FFT_SIZE = 4096;

// Define the forward FFT type
using HilbertForwardFFT = decltype(cufftdx::Size<HILBERT_FFT_SIZE>() + cufftdx::Precision<float>() +
                                   cufftdx::Type<cufftdx::fft_type::r2c>() + cufftdx::Block() +
                                   cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<1>() +
                                   cufftdx::SM<CUFFTDX_ARCH>());

// Define the inverse FFT type
using HilbertInverseFFT =
    decltype(cufftdx::Size<HilbertForwardFFT::input_length>() + cufftdx::Precision<float>() +
             cufftdx::Type<cufftdx::fft_type::c2c>() +
             cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::Block() +
             cufftdx::ElementsPerThread<HilbertForwardFFT::elements_per_thread>() +
             cufftdx::FFTsPerBlock<1>() + cufftdx::SM<CUFFTDX_ARCH>());

// Since we execute both forward and inverse in one kernel the parameters have to match
static_assert(HilbertForwardFFT::max_threads_per_block == HilbertInverseFFT::max_threads_per_block);
static_assert(HilbertForwardFFT::storage_size == HilbertInverseFFT::storage_size);
static_assert(HilbertForwardFFT::shared_memory_size == HilbertInverseFFT::shared_memory_size);

static __launch_bounds__(HilbertForwardFFT::max_threads_per_block) __global__
    void hilbert_kernel(float* __restrict__ buffer) {
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= gridDim.y) { return; }

  // Jump to current row
  buffer += row * HilbertForwardFFT::input_length;

  // If there are threads with partial work load then we need to check for bounds
  constexpr bool has_partial_load =
      ((HilbertForwardFFT::input_length % HilbertForwardFFT::elements_per_thread) != 0);

  // Local array for thread
  HilbertForwardFFT::value_type thread_data[HilbertForwardFFT::storage_size];

  // Load data from global memory to registers
  unsigned int index = threadIdx.x;
  for (unsigned int i = 0; i < HilbertForwardFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertForwardFFT::input_length)) {
      reinterpret_cast<float*>(thread_data)[i] = buffer[index];
      index += HilbertForwardFFT::stride;
    }
  }

  // Execute forward FFT
  extern __shared__ HilbertForwardFFT::value_type shared_mem[];
  HilbertForwardFFT().execute(thread_data, shared_mem);

  // Zero out negative frequencies and double positive frequencies (keep zero frequency) and copy
  // to output
  constexpr unsigned int half_size = HilbertForwardFFT::input_length >> 1;
  index = threadIdx.x;
  for (unsigned int i = 0; i < HilbertForwardFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertForwardFFT::input_length)) {
      thread_data[i] *= (index < half_size) ? 2.0f : (index > half_size) ? 0.f : 1.f;
    }
    index += HilbertForwardFFT::stride;
  }

  // Execute inverse FFT
  HilbertInverseFFT().execute(thread_data, shared_mem);

  // Convert to real, scale and copy to buffer
  index = threadIdx.x;
  for (unsigned int i = 0; i < HilbertInverseFFT::elements_per_thread; ++i) {
    if (!has_partial_load || (index < HilbertInverseFFT::input_length)) {
      const auto value = thread_data[i];
      buffer[index] = sqrtf(value.x * value.x + value.y * value.y) *
                      (1.f / float(HilbertInverseFFT::input_length));
    }
    index += HilbertForwardFFT::stride;
  }
}

static __global__ void scan_convert_curvilinear_kernel(cudaTextureObject_t input, uint2 input_size,
                                                       float* __restrict__ output,
                                                       uint2 output_size, float sector_angle,
                                                       float near, float far, float scale_x,
                                                       float offset_z) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  float2 coord = make_float2((float(index.x) / float(output_size.x)) * 2.f - 1.f,
                             float(index.y) / float(output_size.y));
  // Scale to fit
  coord.x *= scale_x;
  coord.y = (coord.y * (1.f - offset_z)) + offset_z;

  // Mask out near and far
  const float dist = sqrtf(coord.x * coord.x + coord.y * coord.y);
  if ((dist < near) || (dist > far)) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  // Mask out outside of opening angle
  const float angle = atanf(coord.x / coord.y);
  if (fabsf(angle) > sector_angle / 2.f) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  const float source_x = (dist - near) / (far - near);
  const float source_y = angle / sector_angle + 0.5f;

  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

static __global__ void scan_convert_linear_kernel(cudaTextureObject_t input, uint2 input_size,
                                                  float* __restrict__ output, uint2 output_size,
                                                  float width, float far) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  // Calculate physical aspect ratio of the linear probe's field of view
  const float physical_aspect = width / far;

  // Calculate the normalized coordinates in the output image space [0,1] x [0,1]
  const float normalized_x = float(index.x) / float(output_size.x - 1);
  const float normalized_y = float(index.y) / float(output_size.y - 1);

  // For x: Convert to centered coordinates in [-0.5, 0.5] range
  // For y: Keep y = 0 at the top of the image (probe surface)
  const float centered_x = normalized_x - 0.5f;

  // Calculate the region where the linear probe's field of view is displayed
  // For square output images, we need to adjust based on aspect ratio
  float scale_factor;
  if (physical_aspect < 1.0f) {
    // Width is smaller than depth - add black bars on sides
    scale_factor = physical_aspect;
  } else {
    // Width is larger than depth - use full width (rare for ultrasound)
    scale_factor = 1.0f;
  }

  // Scale the x coordinate to account for the aspect ratio
  const float scaled_x = centered_x / scale_factor;

  // Check if we're outside the valid image region (add black bars)
  if (fabsf(scaled_x) > 0.5f) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  // Map from normalized coordinates to physical coordinates
  const float px = scaled_x * width;    // Map to [-width/2, width/2]
  const float pz = normalized_y * far;  // Map to [0, far] with 0 at the top

  // Check if point is within the rectangular field of view
  if (fabsf(px) > width / 2.0f || pz < 0.0f || pz > far) {
    output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();
    return;
  }

  // Map to texture coordinates [0,1] x [0,1]
  // Linear arrays have scanlines running along elements (lateral dimension)
  // Each element's scanline represents depth data from that position
  const float source_y = (px + width / 2.0f) / width;  // Map lateral position to scanline index
  const float source_x = pz / far;                     // Map depth to position along scanline

  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

static __global__ void scan_convert_phased_kernel(cudaTextureObject_t input, uint2 input_size,
                                                  float* __restrict__ output, uint2 output_size,
                                                  float sector_angle, float far) {
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  // Black out pixels by default
  output[index.y * output_size.x + index.x] = std::numeric_limits<float>::lowest();

  // Convert sector angle to radians
  const float sector_angle_rad = (sector_angle / 180.0f) * M_PI;
  const float half_angle_rad = sector_angle_rad / 2.0f;

  // PROPER PHASED ARRAY SECTOR GEOMETRY:
  // 1. Define origin at the top center of the image
  const float origin_x = output_size.x / 2.0f;
  const float origin_y = 0.0f;

  // 2. Get current pixel coordinates
  const float px = static_cast<float>(index.x);
  const float py = static_cast<float>(index.y);

  // Skip if we're at the top (y=0) and outside the width of the probe
  // This ensures a clean flat interface
  if (py == 0.0f && fabsf(px - origin_x) > (output_size.x / 2.0f * 0.1f)) { return; }

  // Calculate the ray angle for this pixel
  // For points not at the origin, use atan2 to get the angle from vertical
  float theta;
  float depth;

  if (py == 0.0f) {
    // At the top interface, use a special case to ensure it's flat
    // All top-row pixels use the angle that corresponds to their x position
    theta = (px - origin_x) / origin_x * half_angle_rad;
    depth = 0.0f;
  } else {
    // For all other pixels, calculate properly from the origin
    theta = atan2f(px - origin_x, py - origin_y);

    // Calculate depth along the ray (distance from origin)
    depth = sqrtf((px - origin_x) * (px - origin_x) + (py - origin_y) * (py - origin_y));

    // Scale depth to be in physical units [0, far]
    depth = depth * far / output_size.y;
  }

  // Skip if outside the sector angle
  if (fabsf(theta) > half_angle_rad) { return; }

  // Skip if beyond max depth
  if (depth > far) { return; }

  // Map to texture coordinates
  // For angle: convert from [-half_angle, half_angle] to [0, 1]
  const float source_y = (theta + half_angle_rad) / sector_angle_rad;

  // For depth: normalize to [0, 1]
  const float source_x = depth / far;

  // Sample the scan line data
  output[index.y * output_size.x + index.x] = tex2D<float>(input, source_x, source_y);
}

// IVUS: unwrapped polar display (angle horizontal, depth vertical)
// Input texture: (s, t) = (depth_norm, angle_norm) with size (buffer_size, num_angular_rays)
static __global__ void scan_convert_ivus_kernel(cudaTextureObject_t input, uint2 input_size,
                                                float* __restrict__ output, uint2 output_size) {
  (void)input_size;
  const uint2 index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if ((index.x >= output_size.x) || (index.y >= output_size.y)) { return; }

  // Output: index.x = angle (0 .. width-1), index.y = depth (0 .. height-1)
  // Standard unwrapped IVUS: horizontal = angle, vertical = depth (probe at top)
  const float angle_norm = (float(index.x) + 0.5f) / float(output_size.x);
  const float depth_norm = (float(index.y) + 0.5f) / float(output_size.y);

  output[index.y * output_size.x + index.x] = tex2D<float>(input, depth_norm, angle_norm);
}

CUDAAlgorithms::CUDAAlgorithms()
    : normalize_launcher_((void*)&normalize_kernel),
      convolve_rows_launcher_((void*)&convolve_rows_kernel),
      convolve_columns_launcher_((void*)&convolve_columns_kernel),
      convolve_columns_depth_dependent_launcher_(
          (void*)&convolve_columns_depth_dependent_kernel),
      convolve_planes_launcher_((void*)&convolve_planes_kernel),
      mean_planes_launcher_((void*)&mean_planes_kernel),
      log_compression_launcher_((void*)&log_compression_kernel),
      mul_rows_launcher_((void*)&mul_rows_kernel),
      add_row_launcher_((void*)&add_row_kernel),
      scale_buffer_launcher_((void*)&scale_buffer_kernel),
      add_gaussian_noise_launcher_((void*)&add_gaussian_noise_kernel),
      zero_inner_radial_launcher_((void*)&zero_inner_radial_kernel),
      display_window_launcher_((void*)&display_window_kernel),
      median_clip_launcher_((void*)&median_clip_kernel),
      scan_convert_curvilinear_launcher_((void*)&scan_convert_curvilinear_kernel),
      scan_convert_linear_launcher_((void*)&scan_convert_linear_kernel),
      scan_convert_phased_launcher_((void*)&scan_convert_phased_kernel),
      scan_convert_ivus_launcher_((void*)&scan_convert_ivus_kernel) {
  CUDA_CHECK(cudaFuncSetAttribute(hilbert_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  HilbertForwardFFT::shared_memory_size));
  sub_event_.reset([] {
    CUevent event;
    CUDA_CHECK(cudaEventCreate(&event, CU_EVENT_BLOCKING_SYNC));
    return event;
  }());

  for (auto&& sub_stream : sub_streams_) {
    sub_stream.reset([] {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      return stream;
    }());
  }
}

void CUDAAlgorithms::normalize(CudaMemory* buffer, uint2 size, CudaMemory* buffer_min_max,
                               cudaStream_t stream) {
  normalize_launcher_.launch(size,
                             stream,
                             reinterpret_cast<float*>(buffer->get_ptr(stream)),
                             reinterpret_cast<const float*>(buffer_min_max->get_ptr(stream)));
}

void CUDAAlgorithms::convolve_rows(CudaMemory* source, uint3 size, CudaMemory* dst,
                                   CudaMemory* kernel, cudaStream_t stream) {
  if (!((kernel->get_size() / sizeof(float)) & 1)) {
    throw std::runtime_error("Expected uneven kernel size");
  }

  convolve_rows_launcher_.launch(size,
                                 stream,
                                 reinterpret_cast<const float*>(source->get_ptr(stream)),
                                 size,
                                 reinterpret_cast<float*>(dst->get_ptr(stream)),
                                 reinterpret_cast<const float*>(kernel->get_ptr(stream)),
                                 kernel->get_size() / sizeof(float) / 2);
}

void CUDAAlgorithms::convolve_columns(CudaMemory* source, uint3 size, CudaMemory* dst,
                                      CudaMemory* kernel, cudaStream_t stream) {
  if (!((kernel->get_size() / sizeof(float)) & 1)) {
    throw std::runtime_error("Expected uneven kernel size");
  }

  convolve_columns_launcher_.launch(size,
                                    stream,
                                    reinterpret_cast<const float*>(source->get_ptr(stream)),
                                    size,
                                    reinterpret_cast<float*>(dst->get_ptr(stream)),
                                    reinterpret_cast<const float*>(kernel->get_ptr(stream)),
                                    kernel->get_size() / sizeof(float) / 2);
}

void CUDAAlgorithms::convolve_columns_depth_dependent(CudaMemory* source, uint3 size,
                                                      CudaMemory* dst, CudaMemory* kernel_2d,
                                                      uint32_t depth_bins, uint32_t kernel_radius,
                                                      cudaStream_t stream) {
  convolve_columns_depth_dependent_launcher_.launch(
      size,
      stream,
      reinterpret_cast<const float*>(source->get_ptr(stream)),
      size,
      reinterpret_cast<float*>(dst->get_ptr(stream)),
      reinterpret_cast<const float*>(kernel_2d->get_ptr(stream)),
      depth_bins,
      kernel_radius);
}

void CUDAAlgorithms::convolve_planes(CudaMemory* source, uint3 size, CudaMemory* dst,
                                     CudaMemory* kernel, cudaStream_t stream) {
  if (!((kernel->get_size() / sizeof(float)) & 1)) {
    throw std::runtime_error("Expected uneven kernel size");
  }

  convolve_planes_launcher_.launch(size,
                                   stream,
                                   reinterpret_cast<const float*>(source->get_ptr(stream)),
                                   size,
                                   reinterpret_cast<float*>(dst->get_ptr(stream)),
                                   reinterpret_cast<const float*>(kernel->get_ptr(stream)),
                                   kernel->get_size() / sizeof(float) / 2);
}

void CUDAAlgorithms::mean_planes(CudaMemory* source, uint3 size, CudaMemory* dst,
                                 cudaStream_t stream) {
  mean_planes_launcher_.launch(make_uint2(size.x, size.y),
                               stream,
                               reinterpret_cast<const float*>(source->get_ptr(stream)),
                               size,
                               reinterpret_cast<float*>(dst->get_ptr(stream)));
}

void CUDAAlgorithms::log_compression(CudaMemory* buffer, uint2 size, float mutliplicator,
                                     float minimum, cudaStream_t stream) {
  // Pass 3 (K2): the kernel now uses the spec's fixed-reference mapping
  // `pixel = mutliplicator * log10(max(amp, minimum))`, no per-frame quantile.
  // The quantile-normalisation scratch buffers are no longer needed; they are
  // kept on the host as zero-size resize-able allocations so that downstream
  // bookkeeping (`CudaMemory` pool churn) is unchanged.
  float* const d_data = reinterpret_cast<float*>(buffer->get_ptr(stream));
  log_compression_launcher_.launch(size, stream, d_data, size, mutliplicator, minimum);
}

void CUDAAlgorithms::mul_row(CudaMemory* buffer, uint2 size, CudaMemory* multiplicator,
                             cudaStream_t stream) {
  if (size.x != multiplicator->get_size() / sizeof(float)) {
    throw std::runtime_error("Unexpected multiplicator buffer size");
  }

  mul_rows_launcher_.launch(size,
                            stream,
                            reinterpret_cast<float*>(buffer->get_ptr(stream)),
                            size,
                            reinterpret_cast<const float*>(multiplicator->get_ptr(stream)));
}

void CUDAAlgorithms::add_row(CudaMemory* buffer, uint2 size, CudaMemory* addend,
                             uint32_t addend_size, float scale, cudaStream_t stream) {
  if (addend_size > size.x) {
    throw std::runtime_error("add_row: addend_size larger than buffer row");
  }
  if (addend_size > addend->get_size() / sizeof(float)) {
    throw std::runtime_error("add_row: addend_size exceeds addend buffer length");
  }
  if (addend_size == 0) { return; }

  add_row_launcher_.launch(size,
                           stream,
                           reinterpret_cast<float*>(buffer->get_ptr(stream)),
                           size,
                           reinterpret_cast<const float*>(addend->get_ptr(stream)),
                           addend_size,
                           scale);
}

void CUDAAlgorithms::scale_buffer(CudaMemory* buffer, uint2 size, float scale,
                                  cudaStream_t stream) {
  if (scale == 1.f) { return; }  // no-op fast path

  scale_buffer_launcher_.launch(size,
                                stream,
                                reinterpret_cast<float*>(buffer->get_ptr(stream)),
                                size,
                                scale);
}

void CUDAAlgorithms::add_gaussian_noise(CudaMemory* buffer, uint2 size, float sigma,
                                        uint32_t seed, cudaStream_t stream) {
  // Disabled (skipped) when sigma <= 0; default callers (which leave
  // SimParams::noise_sigma == 0.f) pay no kernel-launch cost.
  if (!(sigma > 0.f)) { return; }

  add_gaussian_noise_launcher_.launch(size,
                                      stream,
                                      reinterpret_cast<float*>(buffer->get_ptr(stream)),
                                      size,
                                      sigma,
                                      seed);
}

void CUDAAlgorithms::zero_inner_radial(CudaMemory* buffer, uint2 size,
                                       uint32_t dead_zone_samples, cudaStream_t stream) {
  // No-op when there's no dead zone (default SimParams::catheter_dead_zone_mm
  // == 0.f maps to dead_zone_samples == 0); existing callers pay no cost.
  if (dead_zone_samples == 0u) { return; }

  zero_inner_radial_launcher_.launch(size,
                                     stream,
                                     reinterpret_cast<float*>(buffer->get_ptr(stream)),
                                     size,
                                     dead_zone_samples);
}

void CUDAAlgorithms::apply_display_window(CudaMemory* buffer, uint2 size, float reject_palette,
                                          float saturation_palette, cudaStream_t stream) {
  // Disabled when the ceiling is at or below the floor (default-constructed
  // SimParams leaves both at 0.f, so default callers skip the launch).
  if (!(saturation_palette > reject_palette)) { return; }

  display_window_launcher_.launch(size,
                                  stream,
                                  reinterpret_cast<float*>(buffer->get_ptr(stream)),
                                  size,
                                  reject_palette,
                                  saturation_palette);
}

void CUDAAlgorithms::hilbert_row(CudaMemory* buffer, uint2 size, cudaStream_t stream) {
  if (size.x != HILBERT_FFT_SIZE) {
    std::stringstream buf;
    buf << "Hilbert: row length of " << size.x << " does not match supported row length of "
        << HILBERT_FFT_SIZE << ". The row length is set at compile time.";
    throw std::runtime_error(buf.str().c_str());
  }

  const dim3 grid{1, size.y, 1};

  hilbert_kernel<<<grid,
                   HilbertForwardFFT::block_dim,
                   HilbertForwardFFT::shared_memory_size,
                   stream>>>(reinterpret_cast<float*>(buffer->get_ptr(stream)));
}

void CUDAAlgorithms::median_clip_filter(CudaMemory* source, uint2 size, CudaMemory* dst,
                                        uint32_t filter_size, float d_min, float d_max,
                                        cudaStream_t stream) {
  if (filter_size > 11 || filter_size % 2 == 0) {
    throw std::runtime_error("Filter size must be odd and <= 11");
  }

  median_clip_launcher_.launch(size,
                               stream,
                               reinterpret_cast<const float*>(source->get_ptr(stream)),
                               size,
                               reinterpret_cast<float*>(dst->get_ptr(stream)),
                               filter_size,
                               d_min,
                               d_max);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_curvilinear(CudaMemory* scan_lines,
                                                                     uint2 input_size,
                                                                     float sector_angle, float near,
                                                                     float far, uint2 output_size,
                                                                     cudaStream_t stream) {
  // Create the array and the texture
  if (scan_convert_curvilinear_array_ &&
      ((scan_convert_curvilinear_array_->get_size().width != input_size.x) ||
       (scan_convert_curvilinear_array_->get_size().height != input_size.y))) {
    scan_convert_curvilinear_array_.reset();
  }
  if (!scan_convert_curvilinear_array_) {
    scan_convert_curvilinear_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_curvilinear_texture_ = std::make_unique<CudaTexture>(
        scan_convert_curvilinear_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  // Upload scan lines
  scan_convert_curvilinear_array_->upload(scan_lines, stream);

  // Create the output memory
  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  // Calculate the image bounds
  const float sector_angle_rad = (sector_angle / 360.f) * 2.f * M_PI;
  const float max_x = std::sin(sector_angle_rad * 0.5f);                 // width / 2
  const float min_z = std::cos(sector_angle_rad * 0.5f) * (near / far);  // depth

  scan_convert_curvilinear_launcher_.launch(output_size,
                                            stream,
                                            scan_convert_curvilinear_texture_->get_texture().get(),
                                            input_size,
                                            reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                            output_size,
                                            sector_angle_rad,
                                            near / far,
                                            far / far,
                                            max_x,
                                            min_z);

  return std::move(grid_z);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_linear(CudaMemory* scan_lines,
                                                                uint2 input_size, float width,
                                                                float far, uint2 output_size,
                                                                cudaStream_t stream) {
  // Create the array and the texture
  if (scan_convert_linear_array_ &&
      ((scan_convert_linear_array_->get_size().width != input_size.x) ||
       (scan_convert_linear_array_->get_size().height != input_size.y))) {
    scan_convert_linear_array_.reset();
  }
  if (!scan_convert_linear_array_) {
    scan_convert_linear_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_linear_texture_ = std::make_unique<CudaTexture>(
        scan_convert_linear_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  // Upload scan lines
  scan_convert_linear_array_->upload(scan_lines, stream);

  // Create the output memory
  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  // For linear arrays, scan conversion is mostly a direct mapping
  scan_convert_linear_launcher_.launch(output_size,
                                       stream,
                                       scan_convert_linear_texture_->get_texture().get(),
                                       input_size,
                                       reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                       output_size,
                                       width,
                                       far);

  return std::move(grid_z);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_phased(CudaMemory* scan_lines,
                                                                uint2 input_size,
                                                                float sector_angle, float far,
                                                                uint2 output_size,
                                                                cudaStream_t stream) {
  // Create the array and the texture
  if (scan_convert_phased_array_ &&
      ((scan_convert_phased_array_->get_size().width != input_size.x) ||
       (scan_convert_phased_array_->get_size().height != input_size.y))) {
    scan_convert_phased_array_.reset();
  }
  if (!scan_convert_phased_array_) {
    scan_convert_phased_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_phased_texture_ = std::make_unique<CudaTexture>(
        scan_convert_phased_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  // Upload scan lines
  scan_convert_phased_array_->upload(scan_lines, stream);

  // Create the output memory
  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  // Convert from polar coordinates to Cartesian for display
  scan_convert_phased_launcher_.launch(output_size,
                                       stream,
                                       scan_convert_phased_texture_->get_texture().get(),
                                       input_size,
                                       reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                       output_size,
                                       sector_angle,
                                       far);

  return std::move(grid_z);
}

std::unique_ptr<CudaMemory> CUDAAlgorithms::scan_convert_ivus(CudaMemory* scan_lines,
                                                             uint2 input_size, uint2 output_size,
                                                             cudaStream_t stream) {
  if (scan_convert_ivus_array_ &&
      ((scan_convert_ivus_array_->get_size().width != input_size.x) ||
       (scan_convert_ivus_array_->get_size().height != input_size.y))) {
    scan_convert_ivus_array_.reset();
  }
  if (!scan_convert_ivus_array_) {
    scan_convert_ivus_array_ = std::make_shared<CudaArray>(
        cudaExtent({input_size.x, input_size.y, 0}), cudaChannelFormatKindFloat, sizeof(float));
    scan_convert_ivus_texture_ = std::make_unique<CudaTexture>(
        scan_convert_ivus_array_, cudaAddressModeClamp, cudaFilterModeLinear);
  }

  scan_convert_ivus_array_->upload(scan_lines, stream);

  auto grid_z = std::make_unique<CudaMemory>(output_size.x * output_size.y * sizeof(float), stream);

  scan_convert_ivus_launcher_.launch(output_size,
                                    stream,
                                    scan_convert_ivus_texture_->get_texture().get(),
                                    input_size,
                                    reinterpret_cast<float*>(grid_z->get_ptr(stream)),
                                    output_size);

  return std::move(grid_z);
}

}  // namespace raysim
