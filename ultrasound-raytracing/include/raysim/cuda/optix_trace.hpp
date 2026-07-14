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

#ifndef CPP_OPTIX_TRACE
#define CPP_OPTIX_TRACE

#include <optix.h>
#include <cstdint>

#include "raysim/core/material.hpp"
#include "raysim/core/probe_types.hpp"
#include "raysim/cuda/matrix.hpp"

namespace raysim {
struct Params {
  // Legacy scanline output: [num_el_samples * num_elements * buffer_size] floats.
  // If null, scanline writes are skipped (channel-capture-only mode).
  float* scanlines;

  uint32_t buffer_size;
  float t_far;
  float min_intensity;
  uint32_t max_depth;
  Material* materials;
  uint32_t background_material_id;
  cudaTextureObject_t scattering_texture;
  // Voxel size (mm) at which the scattering texture is sampled. Smaller -> finer speckle.
  float scattering_resolution_mm;
  OptixTraversableHandle handle;
  float source_frequency;
  float contact_epsilon;

  // If non-zero, sample_intensities returns immediately (debug switch).
  uint32_t disable_scatter;
  // Empirical scale factor on the volumetric scatter integral. 0 -> strict integral.
  float scatter_integral_scale;

  // ------------------------------------------------------------------------
  // Channel-capture (per-element RF) output. See CHANNEL_CAPTURE.md.
  //
  // When `channel_rf` is non-null, the kernel additionally splats every scatter
  // sample / specular hit to all RX elements via atomicAdd. The per-element
  // arrival time at receive element e for a scatter point p is
  //
  //   t_total = t_tx_path + |p - rx_positions[e]|
  //
  // and is rounded to a sample bin via the same get_intensity_offset as the
  // legacy scanline path. The resulting tensor has shape
  // [num_rx, num_rx, buffer_size] in row-major layout
  // (tx_index varies slowest, then rx index, then time bin).
  // ------------------------------------------------------------------------

  // [num_rx, num_rx, buffer_size] float buffer; null => channel-capture disabled.
  float* channel_rf;
  // World-space transducer element positions and outward normals, length num_rx.
  const float3* rx_positions;
  const float3* rx_normals;
  // Number of receive elements (also the leading TX dimension when in FMC mode).
  uint32_t num_rx;
  // Index of the currently-firing TX element (0..num_rx-1). Set per launch.
  uint32_t tx_index;
};

struct RayGenData {
  int probe_type;            // Type of probe (curvilinear, linear, phased)
  float sector_angle;        // Field of view in degrees (sector angle for phased array)
  float elevational_height;  // Height in elevational direction in mm
  float radius;              // Radius of curvature in mm (for curvilinear)
  float width;               // Width of linear/phased array in mm
  float3 position;           // Probe position in world coordinates
  float33 rotation_matrix;   // Probe orientation in world coordinates

  // Channel-capture: lateral offset (mm) of the firing TX element within the
  // probe's local frame. Zero in legacy scanline mode (origin = array center).
  float3 tx_origin_local;

  // Channel-capture (synthetic-aperture IVUS): center firing direction of the
  // currently-firing element in the probe's local frame (unit vector in the
  // xz imaging plane). Each TX event emits a fan of rays around this direction.
  float3 tx_dir_local;
  // Half-angle of the transmit fan in degrees. The launch dimension d_x in
  // [-0.5, 0.5] maps to [-tx_fan_half_deg, +tx_fan_half_deg] about tx_dir_local.
  float tx_fan_half_deg;
};

struct MissData {};

struct HitGroupData {
  uint32_t material_id;
  uint32_t* indices;
  float3* normals;
};

struct Payload {
  float intensity;
  uint32_t depth;
  float t_ancestors;
  // use 16 bit for object and material ID to save space
  uint16_t current_obj_id;
  uint16_t outter_obj_id;
  uint16_t current_material_id;
  uint16_t outter_material_id;
};

}  // namespace raysim

#endif /* CPP_OPTIX_TRACE */
