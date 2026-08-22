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

#include <optix.h>

#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "raysim/cuda/optix_trace.hpp"

#include <OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h>

namespace raysim {

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ Payload get_payload() {
  Payload payload;
  static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  reinterpret_cast<uint32_t*>(&payload)[0] = optixGetPayload_0();
  reinterpret_cast<uint32_t*>(&payload)[1] = optixGetPayload_1();
  reinterpret_cast<uint32_t*>(&payload)[2] = optixGetPayload_2();
  reinterpret_cast<uint32_t*>(&payload)[3] = optixGetPayload_3();
  reinterpret_cast<uint32_t*>(&payload)[4] = optixGetPayload_4();
  return payload;
}

// Per-scanline scatter decorrelation hash.
//
// PCG-style integer hash (Jarzynski & Olano 2020, "Hash Functions for GPU Rendering";
// Melissa O'Neill 2014, "PCG: A Family of Simple Fast Space-Efficient Statistically
// Good Algorithms for Random Number Generation"). Decorrelates 32-bit input keys
// to 32-bit output with avalanche-quality mixing — fully sufficient for
// per-scanline texture-coordinate jitter; we are not using this for any
// security-sensitive purpose.
static __device__ uint32_t pcg_hash(uint32_t x) {
  uint32_t state = x * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Map a 32-bit integer to a float in [0, 1) using the top 24 bits (mantissa
// width). Equivalent to (x >> 8) * 2^-24, which is exact in IEEE-754 single
// precision.
static __device__ float pcg_to_unit_float(uint32_t x) {
  return (x >> 8) * (1.f / 16777216.f);
}

static __device__ float get_scattering_value(float3 pos, const Material* material,
                                             uint32_t ray_index) {
  // Convert point to texture coordinates (resolution_mm sets speckle scale)
  const float resolution_mm = params.scattering_resolution_mm;
  pos /= resolution_mm;

  // Angular decorrelation of the scatter texture lookup.
  //
  // The scatter texture is 256³ voxels addressed in WRAP mode (see
  // World::generate_scattering_texture). Adding a large pseudo-random offset
  // in texture coordinates is equivalent to sampling an independent region
  // of the texture per ray, which (after wrap-around) gives uncorrelated
  // scatter values across angular bins for any ray_index spacing — including
  // adjacent scanlines. This addresses the near-field positive angular
  // correlation: at r=1 mm with num_scanlines=256, adjacent scanlines sample
  // world points 0.025 mm apart, well within one trilinearly-interpolated
  // texture voxel even at scattering_resolution_mm = 0.1 mm. Without
  // decorrelation, the depth-dependent lateral PSF (with sigma > 100
  // angular bins in the near field) sums these correlated samples up to a
  // bright shoulder at r ≈ 4-7 mm.
  //
  // The offset depends on ray_index (per-scanline decorrelation) and
  // frame_seed (so successive frames sample independent realizations,
  // enabling temporal averaging to converge on the bench's noise-floor
  // statistics). All depth samples ALONG a single scanline share the same
  // offset, so the axial scatter integral is still spatially coherent in
  // depth (preserving wire/sphere PSFs and the band-pass character that
  // the axial Hanning-windowed cosine PSF later filters).
  if (params.scatter_angular_decorrelate) {
    const uint32_t base = ray_index * 3u + params.frame_seed * 2654435761u;
    const float ox = pcg_to_unit_float(pcg_hash(base + 0u)) * 4096.f;
    const float oy = pcg_to_unit_float(pcg_hash(base + 1u)) * 4096.f;
    const float oz = pcg_to_unit_float(pcg_hash(base + 2u)) * 4096.f;
    pos.x += ox;
    pos.y += oy;
    pos.z += oz;
  }

  const float2 scatter_val = tex3D<float2>(params.scattering_texture, pos.x, pos.y, pos.z);

  // Apply material properties
  if (scatter_val.x <= material->mu0_) { return scatter_val.y * material->sigma_; }
  return 0.f;
}

/**
 * @return offset to intensity buffer at ray distance t
 */
static __device__ uint32_t get_intensity_offset(float t) {
  uint32_t offset = uint32_t((t / params.t_far) * (params.buffer_size - 1) + 0.5f);
  assert(offset < params.buffer_size);
  return offset;
}

/// Calculate intensity using Beer-Lambert Law
static __device__ float get_intensity_at_distance(float distance, float medium_attenuation) {
  // I = I₀ * 10^(-αfd/20)
  // where α is attenuation coefficient in dB/(cm⋅MHz)
  // f is frequency in MHz
  // d is distance in cm
  const float source_freq = params.source_frequency;                            // MHz
  const float distance_cm = distance * 0.1f;                                    // Convert to cm
  const float attenuation_db = medium_attenuation * source_freq * distance_cm;  // dB
  return __powf(10.f, -attenuation_db * 0.05f);
}

/**
 * Sample intensities
 *
 * @param origin ray origin in world space
 * @param dir ray direction in world space
 * @param t_ancestors
 * @param t_min
 * @param t_max
 * @param intensity
 * @param material
 * @param intensities
 */
static __device__ void sample_intensities(float3 origin, float3 dir, float t_ancestors, float t_min,
                                          float t_max, float intensity, const Material* material,
                                          float* scanline, uint32_t ray_index) {
  if (params.disable_scatter) { return; }
  if ((material->mu0_ <= 0.f) || (material->sigma_ == 0.f)) { return; }

  const float range = t_max - t_min;
  if (range <= 0.f) { return; }

  // Backscatter approximates the line integral int (sigma(s)*I(s)) ds.
  // Weight by (range/steps) for correct integral; scale by scatter_integral_scale so the
  // result is in a displayable range for both wire (lumen) and vascular/cystic (tissue) phantoms.
  const float integral_weight = (params.scatter_integral_scale > 0.f)
                                    ? (range * params.scatter_integral_scale)
                                    : range;

  // Dense integration: one sample per depth bin
  const uint32_t steps = (range / params.t_far) * params.buffer_size + 0.5f;
  const float t_step = (steps > 0) ? (range / steps) : 0.f;
  const float segment_weight = (steps > 0u) ? (integral_weight / static_cast<float>(steps)) : 0.f;
  for (uint32_t step = 0; step < steps; ++step) {
    const float t_val = t_min + (step + 0.5f) * t_step;
    const float depth = t_ancestors + t_val;
    const uint32_t bin = get_intensity_offset(depth);
    if (bin >= params.buffer_size) { continue; }
    const float3 pos_world = origin + (t_ancestors + t_val) * dir;
    const float scatter = get_scattering_value(pos_world, material, ray_index) * intensity *
                         get_intensity_at_distance(t_val - t_min, material->attenuation_);
    scanline[bin] += segment_weight * scatter;
  }
}

template <OptixPrimitiveType PRIM_TYPE>
static __device__ float3 get_normal(float3 ray_orig, float3 ray_dir, float t_hit, uint32_t hit_id,
                                    const HitGroupData* hit_group_data) {
  const unsigned int prim_idx = optixGetPrimitiveIndex();

  float3 normal;
  if constexpr (PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    uint32_t* const tri = &hit_group_data->indices[prim_idx * 3];
    const float3 N0 = hit_group_data->normals[tri[0]];
    const float3 N1 = hit_group_data->normals[tri[1]];
    const float3 N2 = hit_group_data->normals[tri[2]];

    const float2 barys = optixGetTriangleBarycentrics();

    normal = (1.f - barys.x - barys.y) * N0 + barys.x * N1 + barys.y * N2;
  } else {
    assert(PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE);
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData(gas, prim_idx, hit_id, 0.f, &q);

    const float3 raypos = ray_orig + t_hit * ray_dir;
    normal = (raypos - make_float3(q)) / q.w;
  }
  return normal;
}

// -----------------------------------------------------------------------------
// Surface interaction (reflection / refraction) — physics summary
// -----------------------------------------------------------------------------
// • Reflection direction: law of reflection (θ_r = θ_i), implemented as
//   r = d - 2(d·n)n with n pointing against d. Theoretically exact.
// • Refraction direction: Snell's law n1 sin(θ_i) = n2 sin(θ_t) with n = c/v;
//   sin_t = (v1/v2)*sin_i, refracted vector in plane of incidence. Theoretically exact.
// • Total internal reflection: when sin_t >= 1, no transmitted ray; R = 1.
// • Intensity R: oblique acoustic formula R_p = (Z2/cos(θ_t)-Z1/cos(θ_i))/(Z2/cos(θ_t)+Z1/cos(θ_i)),
//   R_I = R_p^2 (pressure continuity + normal velocity continuity at interface).
// • Intensity split: reflected = I*R, transmitted = I*(1-R); power-conserving.
// • Specular term: empirical (Mattausch2016); directivity-like cos^n; not from wave equation.
// -----------------------------------------------------------------------------

/**
 * Calculate reflected direction vector (law of reflection)
 */
static __device__ float3 calc_reflected_dir(float3 incident_dir, float3 normal) {
  // Ensure normal points against incident direction
  if (dot(incident_dir, normal) > 0.f) { normal = -normal; }
  return incident_dir - 2.f * dot(incident_dir, normal) * normal;
}

/**
 * Intensity reflection coefficient at an acoustic interface (oblique incidence).
 *
 * Physics: continuity of pressure and normal particle velocity at the interface gives
 * pressure reflection coefficient R_p = (Z2/cos(θ_t) - Z1/cos(θ_i)) / (Z2/cos(θ_t) + Z1/cos(θ_i))
 * (effective normal impedances Z/cos(θ)). Intensity R_I = R_p^2.
 * Snell: sin(θ_t) = (c1/c2) sin(θ_i) => cos(θ_t) = sqrt(1 - sin²(θ_t)).
 * Total internal reflection (sin(θ_t) >= 1) is handled by caller (no refracted ray).
 *
 * @param cos_theta_i cos(angle of incidence), in [0,1]
 * @param cos_theta_t cos(angle of transmission), in [0,1] (from Snell)
 * @param material1 incident medium (Z1)
 * @param material2 transmitted medium (Z2)
 */
static __device__ float calculate_reflection_coefficient(float cos_theta_i, float cos_theta_t,
                                                         const Material* material1,
                                                         const Material* material2) {
  float Z1 = material1->impedance_;
  float Z2 = material2->impedance_;
  if (cos_theta_i <= 1e-6f || cos_theta_t <= 1e-6f) { return 1.f; }  // grazing or TIR
  float R_p = (Z2 / cos_theta_t - Z1 / cos_theta_i) / (Z2 / cos_theta_t + Z1 / cos_theta_i);
  return R_p * R_p;
}

/**
 * Calculate refracted direction vector using Snell's law
 *
 * @returns true for total internal reflection
 */
static __device__ bool calc_refracted_dir(float3 incident_dir, float3 normal, float v1, float v2,
                                          float3* refracted_dir) {
  // Ensure normal points against incident direction
  if (dot(incident_dir, normal) > 0.f) { normal = -normal; }

  float cos_i = -dot(normal, incident_dir);  // Incidence
  float sin_i = sqrtf(1.f - cos_i * cos_i);  // Incidence
  float sin_t = (v1 / v2) * sin_i;           // Transmission

  // Check for total internal reflection
  if (sin_t >= 1) { return true; }

  float cos_t = sqrtf(1 - sin_t * sin_t);
  *refracted_dir = (v1 / v2) * incident_dir + ((v1 / v2) * cos_i - cos_t) * normal;
  return false;
}

/**
 * Calculate the reflection intensity Ir for ultrasound RF image
 *
 * Eq. 5 Mattausch2016Monte-Carlo
 *
 *     Ir = max(0, cos(angle_r)^n) + max(0, cos(angle_t)^n)
 *
 * where `n` is the per-material `specularity_` (directivity sharpness exponent).
 *
 * IMPORTANT semantic note: despite the name, `specularity_` is the EXPONENT,
 * not a scale factor.  Counterintuitively, n=1 gives a broad Lambertian-like
 * lobe, n>>1 gives a narrow mirror-like peak, and n=0 gives the maximum
 * constant intensity at every interface hit (cos^0 = 1 for any angle).
 * Setting specularity ~ 0 to mean "less mirror-like" is the OPPOSITE of the
 * intended behaviour -- it produces a constant maximum-brightness rind at
 * every interface.  See note on the Material constructor in material.cpp.
 *
 * @param V_r reflected ray direction vector
 * @param V_i refracted ray direction vector
 * @param total_internal_reflection
 * @param D vector from intersection point to transducer origin
 * @param n surface specularity parameter (exponent; default 1.0)
 * @return reflection intensity Ir
 */
static __device__ float calculate_specular_intensity(float3 V_r, float3 V_i,
                                                     bool total_internal_reflection, float3 D,
                                                     float n) {
  // Calculate angles using dot product
  float cos_reflected = dot(V_r, D) / (length(V_r) * length(D));
  float cos_refracted;
  if (!total_internal_reflection) {
    cos_refracted = dot(V_i, D) / (length(V_i) * length(D));
  } else {
    cos_refracted = 0.f;
  }

  // Calculate the two terms (reflection and refraction)
  float reflected_term = max(0.f, __powf(cos_reflected, n));
  float refracted_term = max(0.f, __powf(cos_refracted, n));

  // Total intensity is sum of both terms
  float Ir = reflected_term + refracted_term;

  return Ir;
}

/**
 * Self-intersection avoidance. Get the save front and back start points.
 * See https://github.com/NVIDIA/optix-toolkit/tree/master/ShaderUtil#self-intersection-avoidance.
 *
 * @param out_front_start [out] offset spawn point on the front of the surface, safe from self
 * intersection
 * @param out_back_start [out] offset spawn point on the back of the surface, safe from self
 * intersection
 * @param out_wld_norm [out] unit length spawn point normal in world space
 */
static __device__ void get_save_start_point(float3& out_front_start, float3& out_back_start,
                                            float3& out_wld_norm) {
  // Compute a surface point, normal and conservative offset in object-space.
  float3 obj_pos, obj_norm;
  float obj_offset;
  SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(obj_pos, obj_norm, obj_offset);
  // Transform the object-space position, normal and offset into world-space. The output world-space
  // offset includes the input object-space offset and accounts for the transformation.
  float3 wld_pos;
  float wld_offset;
  SelfIntersectionAvoidance::transformSafeSpawnOffset(
      wld_pos, out_wld_norm, wld_offset, obj_pos, obj_norm, obj_offset);

  // The offset is used to compute safe spawn points on the front and back of the surface.
  SelfIntersectionAvoidance::offsetSpawnPoint(
      out_front_start, out_back_start, wld_pos, out_wld_norm, wld_offset);
}

// Helper function to generate ray for curvilinear probe in local coordinates
static __forceinline__ __device__ void generate_curvilinear_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  // Convert normalized coordinates to lateral angle in radians
  const float lateral_angle = (ray_gen_data->sector_angle * d_x) * (M_PI / 180.f);

  // Calculate element position on probe surface in probe's local coordinate system
  // where (0,0,0) is at the probe face center
  out_origin =
      make_float3(ray_gen_data->radius * __sinf(lateral_angle),         // x = r * sin(θ)
                  0.f,                                                  // y (elevation added later)
                  ray_gen_data->radius * (__cosf(lateral_angle) - 1.f)  // z = r * (cos(θ) - 1)
      );

  // Calculate ray direction away from center of curvature
  // Center of curvature is at (0,0,-radius) in probe's local coordinate system
  out_direction = normalize(out_origin - make_float3(0.f, 0.f, -ray_gen_data->radius));
}

// Helper function to generate ray for linear array probe in local coordinates
static __forceinline__ __device__ void generate_linear_array_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  // For linear arrays, elements are positioned along a straight line
  // Map normalized coordinate to position along the width
  const float element_width = ray_gen_data->width;
  const float element_pos = element_width * d_x;

  // Element position in local coordinates
  out_origin = make_float3(element_pos,  // x position along array
                           0.f,          // y (elevation added later)
                           0.f           // z at surface (probe face)
  );

  // For linear arrays, rays travel perpendicular to the array
  out_direction = make_float3(0.f, 0.f, 1.f);
}

// Helper function to generate ray for phased array probe in local coordinates
static __forceinline__ __device__ void generate_phased_array_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  // Use full sector angle range
  // Map d_x from [-0.5, 0.5] directly to [-half_angle_rad, half_angle_rad]
  const float steering_angle = d_x * ray_gen_data->sector_angle;     // in degrees
  const float steering_angle_rad = steering_angle * (M_PI / 180.f);  // convert to radians

  // For phased arrays, all rays originate from a single virtual point (0,0,0)
  // This is the center of the transducer array face
  out_origin = make_float3(0.0f,  // Center of the array
                           0.f,   // y (elevation added later)
                           0.f    // z at the surface of the probe
  );

  // Direction determined by steering angle
  out_direction = make_float3(sinf(steering_angle_rad),  // x component based on steering angle
                              0.f,                       // y component (no elevation steering)
                              cosf(steering_angle_rad)   // z component (along central axis)
  );

  // Normalize direction to ensure unit length vector
  out_direction = normalize(out_direction);
}

// Helper function to generate ray for IVUS probe in local coordinates
static __forceinline__ __device__ void generate_ivus_probe_ray_local(
    const RayGenData* ray_gen_data, float d_x, float3& out_origin, float3& out_direction) {
  (void)ray_gen_data;
  // Map d_x from [-0.5, 0.5] to angle in [0, 2*pi] for full 360° radial sweep
  constexpr float two_pi = 6.28318530717958647692f;
  const float angle = (d_x + 0.5f) * two_pi;

  // Single origin at catheter center (point source)
  out_origin = make_float3(0.f, 0.f, 0.f);

  // Radial direction in xz plane: +z at angle 0, consistent with IVUSProbe::get_local_element_direction
  out_direction = make_float3(sinf(angle), 0.f, cosf(angle));
  out_direction = normalize(out_direction);
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const RayGenData* ray_gen_data = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

  // IVUS angular ray super-sampling.
  //
  // For IVUS only, fire K = `params.ivus_rays_per_scanline` sub-rays per
  // scanline at deterministic sub-bin angular offsets ((k+0.5)/K - 0.5 in
  // bin units, k = 0..K-1).  This forward-models the bench's finite beam
  // width at the raycasting stage so that any sub-pixel wire scatterer is
  // captured by at least one sub-ray.  All K sub-rays deposit into the
  // same scanline buffer; we normalise the accumulated result by 1/K
  // after the trace loop so the per-scanline integral is independent of K
  // for a uniform medium.
  //
  // Other probe types always fire K = 1 sub-ray (legacy behaviour).
  uint32_t K = 1u;
  if (ray_gen_data->probe_type == PROBE_TYPE_IVUS && params.ivus_rays_per_scanline > 1u) {
    K = params.ivus_rays_per_scanline;
  }

  for (uint32_t k = 0; k < K; ++k) {
    const float sub_offset =
        (K > 1u) ? ((static_cast<float>(k) + 0.5f) / static_cast<float>(K) - 0.5f) : 0.f;
    const float d_x =
        ((static_cast<float>(idx.x) + sub_offset) / static_cast<float>(dim.x)) - 0.5f;

    float3 origin;
    float3 direction;

    // Different ray generation based on probe type
    switch (ray_gen_data->probe_type) {
      case PROBE_TYPE_CURVILINEAR: {
        generate_curvilinear_probe_ray_local(ray_gen_data, d_x, origin, direction);
        break;
      }

      case PROBE_TYPE_LINEAR_ARRAY: {
        generate_linear_array_probe_ray_local(ray_gen_data, d_x, origin, direction);
        break;
      }

      case PROBE_TYPE_PHASED_ARRAY: {
        generate_phased_array_probe_ray_local(ray_gen_data, d_x, origin, direction);
        break;
      }

      case PROBE_TYPE_IVUS: {
        generate_ivus_probe_ray_local(ray_gen_data, d_x, origin, direction);
        break;
      }
    }

    // Add elevation in probe's local coordinate system (common for all probes).
    // For N>1, sample the full aperture [-H/2, H/2] (idx.y/(N-1) - 0.5).
    // The old idx.y/N - 0.5 left the last bin unsampled.
    // For N=1, d_y = 0 (mid-plane). Previously N=1 used 0/1 - 0.5 = -0.5,
    // so origin.y was -H/2. IVUS 2D (H=0) is unchanged. Default
    // curvilinear/linear/phased examples (H = 5–7 mm, N = 1) now fire on
    // the geometric mid-plane instead of H/2 below it.
    const float d_y = (dim.y <= 1u)
                          ? 0.f
                          : (static_cast<float>(idx.y) / static_cast<float>(dim.y - 1u)) - 0.5f;
    const float elevation = ray_gen_data->elevational_height * d_y;
    origin.y = elevation;

    // Transform from probe's local coordinate system to global coordinate system
    origin = ray_gen_data->rotation_matrix * origin;
    origin += ray_gen_data->position;

    direction = ray_gen_data->rotation_matrix * direction;

    Payload ray{};
    ray.intensity = 1.f;
    ray.depth = 0;
    ray.t_ancestors = 0.f;  // Required: miss shader uses this for depth bins; unset = wrong scatter
    ray.current_material_id = params.background_material_id;
    ray.outter_material_id = 0;
    ray.current_obj_id = static_cast<uint16_t>(-1);
    ray.outter_obj_id = static_cast<uint16_t>(-1);

    optixTrace(params.handle,
               origin,
               direction,
               0.f,           // tmin
               params.t_far,  // tmax
               0.f,           // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&ray)[0],
               reinterpret_cast<uint32_t*>(&ray)[1],
               reinterpret_cast<uint32_t*>(&ray)[2],
               reinterpret_cast<uint32_t*>(&ray)[3],
               reinterpret_cast<uint32_t*>(&ray)[4]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  }

  // Normalise the accumulated scanline by 1/K so the per-scanline integral
  // is K-invariant.  Only executes when K > 1 (cost = `buffer_size` writes
  // per thread); skipped entirely in the legacy K=1 path.
  if (K > 1u) {
    const uint32_t ray_index = idx.y * dim.x + idx.x;
    float* const scanline = &params.scanlines[ray_index * params.buffer_size];
    const float inv_K = 1.f / static_cast<float>(K);
    for (uint32_t b = 0; b < params.buffer_size; ++b) { scanline[b] *= inv_K; }
  }
}

extern "C" __global__ void __miss__ms() {
  const uint3 idx = optixGetLaunchIndex();
  const Payload ray = get_payload();

  // In contact check mode (epsilon > 0), if the initial ray (depth 0) misses all geometry,
  // terminate its path immediately. This blacks out elements that are not pointing towards the
  // phantom.
  if (params.contact_epsilon > 0.f && ray.depth == 0) { return; }

  // no hits, just do scattering up to t_far
  const uint32_t ray_index = idx.y * optixGetLaunchDimensions().x + idx.x;
  sample_intensities(
      optixGetWorldRayOrigin(),
      optixGetWorldRayDirection(),
      ray.t_ancestors,
      optixGetRayTmin(),
      optixGetRayTmax(),
      ray.intensity,
      &params.materials[ray.current_material_id],
      &params.scanlines[ray_index * params.buffer_size],
      ray_index);
}

template <OptixPrimitiveType PRIM_TYPE>
static __device__ void closest_hit() {
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();
  const float t_min = optixGetRayTmin();
  const float t = optixGetRayTmax();

  const Payload ray = get_payload();

  // In contact check mode (epsilon > 0), if the initial ray (depth 0) hits geometry
  // but the hit distance `t` is greater than the allowed epsilon, terminate the path.
  // This blacks out elements that are too far from the phantom to be considered in contact.
  if (params.contact_epsilon > 0.f && ray.depth == 0 && t > params.contact_epsilon) { return; }

  const uint32_t current_material_id = ray.current_material_id;
  const Material* current_material = &params.materials[current_material_id];
  const uint3 idx = optixGetLaunchIndex();
  const uint32_t ray_index = idx.y * optixGetLaunchDimensions().x + idx.x;
  float* const scanline = &params.scanlines[ray_index * params.buffer_size];

  // add scattering contribution up to hit
  sample_intensities(
      ray_orig, ray_dir, ray.t_ancestors, t_min, t, ray.intensity, current_material, scanline,
      ray_index);

  // Don't generate secondary rays is max depth is reached
  if (ray.depth + 1 >= params.max_depth) { return; }

  const uint32_t hit_id = optixGetSbtGASIndex();
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

  uint32_t next_material_id, next_obj_id;
  if (ray.current_obj_id == hit_id) {
    // Exiting an object
    next_material_id = ray.outter_material_id;
    next_obj_id = ray.outter_obj_id;
  } else {
    next_material_id = hit_group_data->material_id;
    next_obj_id = hit_id;
  }

  // Calculate final intensity
  const float final_intensity =
      ray.intensity * get_intensity_at_distance(t - t_min, current_material->attenuation_);

  // Add hit reflection contribution (oblique incidence: need cos(θ_i), cos(θ_t) for R)
  const Material* next_material = &params.materials[next_material_id];
  const float3 normal = get_normal<PRIM_TYPE>(ray_orig, ray_dir, t, hit_id, hit_group_data);
  const float cos_i = fabsf(dot(ray_dir, normal));  // cos(angle of incidence)
  const float sin_i = sqrtf(1.f - cos_i * cos_i);
  const float v1 = current_material->speed_of_sound_;
  const float v2 = next_material->speed_of_sound_;
  const float sin_t = (v1 / v2) * sin_i;
  const float cos_t = (sin_t < 1.f) ? sqrtf(1.f - sin_t * sin_t) : 0.f;  // 0 if TIR
  const float R = calculate_reflection_coefficient(cos_i, cos_t, current_material, next_material);

  float3 refracted_dir;
  const bool total_internal_reflection = calc_refracted_dir(ray_dir,
                                                            normal,
                                                            v1,
                                                            v2,
                                                            &refracted_dir);
  const float3 reflected_dir = calc_reflected_dir(ray_dir, normal);
  const float reflected_intensity = final_intensity * R;
  const float refracted_intensity = final_intensity * (1.f - R);

  // Echo at interface: (1) Intensity R from acoustic oblique formula (first-principles).
  // (2) Specular term: empirical (Mattausch et al. Monte Carlo); cos^n toward transducer
  // approximates directivity; not derivable from wave equation alone.
  const uint32_t hit_bin = get_intensity_offset(ray.t_ancestors + t);
  scanline[hit_bin] += reflected_intensity;

  const float ray_coherence_attenuation = __powf(0.3f, ray.depth);
  const float specular_reflection = calculate_specular_intensity(reflected_dir,
                                                                 refracted_dir,
                                                                 total_internal_reflection,
                                                                 ray_dir,
                                                                 next_material->specularity_) *
                                    ray_coherence_attenuation;
  // Physics-correct Fresnel-scaled specular contribution.
  //
  // The empirical Mattausch-2016 directivity term `cos^n` represents the
  // angular DISTRIBUTION of reflected energy at a non-mirror interface, not
  // an independent intensity contribution.  The TOTAL reflected intensity is
  // bounded by the Fresnel reflection coefficient R (computed above from the
  // impedance contrast).  So the empirical specular contribution must be
  // scaled by R to be physically consistent:
  //
  //     scanline += 2 * R * cos^n * coherence
  //
  // Previously the code added `2 * cos^n * coherence` without the R factor,
  // which produced a constant ~2.0 intensity at every interface hit
  // regardless of impedance.  That was OK for bone-wire phantoms (R ~ 0.44
  // for bone-milk, so a fixed 2.0 was a tolerable approximation) but
  // catastrophically wrong for soft-tissue interfaces (R ~ 0.002 for
  // lumen/vessel_wall): the bright empirical peak ~1000x dominated the
  // physically-correct Fresnel echo, painting an unrealistic saturated rim
  // wherever any tissue boundary appeared.
  //
  // Multiplying by R gives both bench-anchored bone-wire calibration
  // (relative differences preserved) and physics-correct soft-tissue
  // interfaces (vessel walls now appear as texture transitions).
  //
  // Bench calibration implication: bone-wire absolute brightness drops by
  // ~7 dB (factor 1/R ~ 2.3 for bone-milk), so `processing.gain_db` in
  // volcano_s5i.yaml must be re-derived against the wire phantom (Test E).
  scanline[hit_bin] += 2.f * specular_reflection * R;

  // Self-intersection avoidance
  float3 front_start, back_start, wld_norm;
  if (PRIM_TYPE == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    get_save_start_point(front_start, back_start, wld_norm);
  } else {
    wld_norm = normal;
    const float epsilon = 1e-5f;
    front_start = ray_orig + ray_dir * (t - epsilon);
    back_start = ray_orig + ray_dir * (t + epsilon);
  }

  bool reflection_on = false;
  // Create reflected ray
  if (reflection_on && (reflected_intensity > params.min_intensity)) {
    Payload reflected_ray{};
    reflected_ray.intensity = reflected_intensity;
    reflected_ray.depth = ray.depth + 1;
    reflected_ray.t_ancestors = ray.t_ancestors + t;
    reflected_ray.current_material_id = ray.current_material_id;
    reflected_ray.outter_material_id = ray.outter_material_id;
    reflected_ray.current_obj_id = ray.current_obj_id;
    reflected_ray.outter_obj_id = ray.outter_obj_id;

    // Secondary rays along the surface normal should use the generated front point as origin, while
    // rays pointing away from the normal should use the back point as origin.
    const float3 start = (dot(reflected_dir, wld_norm) > 0.f) ? front_start : back_start;
    optixTrace(params.handle,
               start,
               reflected_dir,
               0.f,                                       // tmin
               params.t_far - reflected_ray.t_ancestors,  // tmax
               0.f,                                       // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&reflected_ray)[0],
               reinterpret_cast<uint32_t*>(&reflected_ray)[1],
               reinterpret_cast<uint32_t*>(&reflected_ray)[2],
               reinterpret_cast<uint32_t*>(&reflected_ray)[3],
               reinterpret_cast<uint32_t*>(&reflected_ray)[4]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  }

  // Create refracted ray
  if ((refracted_intensity > params.min_intensity) && !total_internal_reflection) {
    Payload refracted_ray{};
    refracted_ray.intensity = refracted_intensity;
    refracted_ray.depth = ray.depth + 1;
    refracted_ray.t_ancestors = ray.t_ancestors + t;
    refracted_ray.current_material_id = next_material_id;
    refracted_ray.outter_material_id = ray.current_material_id;
    refracted_ray.current_obj_id = next_obj_id;
    refracted_ray.outter_obj_id = ray.current_obj_id;

    // Refracted ray: start just past the hit surface (back_start) so we are in the new material.
    // Use tmin > 0 to avoid re-hitting the same surface (vessel inner wall, mesh self-hit).
    const float t_min_refract = 1e-3f;  // mm; skip hits at same surface
    const float3 start = back_start;
    optixTrace(params.handle,
               start,
               refracted_dir,
               t_min_refract,                             // tmin: avoid self-intersection
               params.t_far - refracted_ray.t_ancestors,  // tmax
               0.f,                                       // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0,  // SBT offset
               1,  // SBT stride
               0,  // missSBTIndex
               reinterpret_cast<uint32_t*>(&refracted_ray)[0],
               reinterpret_cast<uint32_t*>(&refracted_ray)[1],
               reinterpret_cast<uint32_t*>(&refracted_ray)[2],
               reinterpret_cast<uint32_t*>(&refracted_ray)[3],
               reinterpret_cast<uint32_t*>(&refracted_ray)[4]);
    static_assert(sizeof(Payload) / sizeof(uint32_t) == 5);
  }
}

extern "C" __global__ void __closesthit__sphere() {
  closest_hit<OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE>();
}

extern "C" __global__ void __closesthit__triangle() {
  closest_hit<OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_TRIANGLE>();
}

}  // namespace raysim
