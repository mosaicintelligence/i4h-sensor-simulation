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

#ifndef CPP_IVUS_PROBE
#define CPP_IVUS_PROBE

#include "raysim/core/probe.hpp"

namespace raysim {

/**
 * IVUS (Intravascular Ultrasound) probe implementation.
 * Single rotating transducer at catheter center; rays emanate radially in 360°
 * for a cross-sectional image. Typical use: vessel lumen and wall imaging.
 */
class IVUSProbe : public BaseProbe {
 public:
  /**
   * Initialize IVUS probe parameters
   *
   * @param pose Probe pose (position and orientation; position = catheter center in vessel)
   * @param num_angular_rays Number of rays over 360° (angular samples per frame)
   * @param frequency Center frequency in MHz (typical IVUS: 20–40 MHz)
   * @param elevational_height Height in elevational direction in mm (slice thickness; often 0 for 2D)
   * @param num_el_samples Number of samples in elevational direction (typically 1)
   * @param f_num F-number (focal length / aperture) - unitless
   * @param speed_of_sound Speed of sound in tissue in mm/μs
   * @param pulse_duration Duration of excitation pulse in cycles
   */
  explicit IVUSProbe(const Pose& pose = Pose(make_float3(0.f, 0.f, 0.f),
                                             make_float3(0.f, 0.f, 0.f)),
                     uint32_t num_angular_rays = 256,
                     float frequency = 40.f,           // MHz (typical IVUS)
                     float elevational_height = 0.f,   // mm (2D cross-section)
                     uint32_t num_el_samples = 1,
                     float f_num = 1.0f,
                     float speed_of_sound = 1.54f,  // mm/us
                     float pulse_duration = 2.f)
      : BaseProbe(pose, num_angular_rays, frequency, elevational_height, num_el_samples, f_num,
                  speed_of_sound, pulse_duration, 0.f),  // width = 0 (point source)
        sector_angle_(360.f) {}

  /**
   * Get element position in local probe coordinates.
   * For IVUS all rays share the same origin (catheter center).
   */
  void get_local_element_position(uint32_t element_idx, float3& position) const override {
    (void)element_idx;
    position = make_float3(0.f, 0.f, 0.f);
  }

  /**
   * Get element ray direction in local probe coordinates.
   * Direction is radial outward; angle sweeps 0 to 2π over element indices.
   * Convention: +z is one reference direction; angle increases with element index.
   */
  void get_local_element_direction(uint32_t element_idx, float3& direction) const override {
    constexpr float two_pi = 6.28318530717958647692f;
    const float angle_rad =
        two_pi * static_cast<float>(element_idx) /
        static_cast<float>(num_elements_x_ > 1u ? num_elements_x_ : 1u);
    direction = make_float3(sinf(angle_rad), 0.f, cosf(angle_rad));
  }

  /// Sector angle is always 360° for full radial sweep
  float get_sector_angle() const override { return sector_angle_; }

  /// Radius of curvature: 0 (point source at catheter center)
  float get_radius() const override { return 0.f; }

  /// Width: 0 (no lateral aperture; single point source)
  float get_width() const override { return 0.f; }

  ProbeType get_probe_type() const override { return ProbeType::PROBE_TYPE_IVUS; }

 private:
  float sector_angle_;  ///< 360° for full circumferential sweep
};

}  // namespace raysim

#endif /* CPP_IVUS_PROBE */
