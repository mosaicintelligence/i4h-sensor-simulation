# IVUS Probe Implementation in ultrasound-raytracing

This document describes the work done to add **Intravascular Ultrasound (IVUS)** probe support to the ultrasound-raytracing package. It is based on a diff of the package against the `main` branch and focuses on changes to the **core ultrasound implementation** (C++/CUDA, materials, probes, pipeline). Example scripts and evaluation workflows are not covered in detail.

Implementation references link to the code at commit `3a00920723c7821b83b3fb6b400b006dbbc84e96` on [i4h-sensor-simulation](https://github.com/mosaicintelligence/i4h-sensor-simulation). When a function is **modified** (not added new), a short **Diff (vs main)** summarizes the changes.

---

## 1. High-level overview

### 1.1 Original ultrasound-raytracing package (main branch)

The package is a **GPU-accelerated ultrasound simulation** that uses **NVIDIA OptiX** for raytracing and CUDA for signal processing and image formation.

- **Probes**: Three probe types are supported—**curvilinear**, **linear array**, and **phased array**. Each has a distinct ray layout (element positions and directions) and scan geometry (sector or rectangular).
- **Acoustic model**: Rays are cast from the probe; they interact with the scene via **reflection** and **refraction** at interfaces (meshes, spheres). **Volumetric scattering** is accumulated along each ray in depth bins. Materials define impedance, attenuation, and scattering.
- **Materials**: A fixed set of tissue materials: water, blood, fat, liver, muscle, bone (impedance, attenuation in dB/(cm·MHz), speed of sound, scattering).
- **Pipeline**: After raytracing, scanlines are processed by **axial** and **lateral PSF convolution** (Gaussian kernels, depth-invariant lateral), **TGC**, **Hilbert envelope**, **log compression**, and **scan conversion** to a 2D B-mode image. Scan conversion is probe-specific (curvilinear sector, linear, phased sector).
- **Display**: The simulator reports coordinate bounds (min/max x, z) for the B-mode image so the client can label axes (e.g. depth, lateral).

The design is **probe-agnostic** in the sense that ray generation and scan conversion are driven by a `ProbeType` enum and virtual methods on `BaseProbe` (element position, direction, sector angle, etc.).

### 1.2 Modifications for IVUS

IVUS is **intravascular** imaging: a small rotating transducer at the center of a vessel acquires a **360° radial cross-section**. Depth is radial (lumen → wall → perivascular); the “lateral” dimension is **angle**. The implementation adds:

1. **New probe type and class**: `ProbeType::PROBE_TYPE_IVUS` and an **`IVUSProbe`** class—single point source at the catheter center, rays emitted radially over 360° in the imaging plane, with optional **element radius** and **focal length** for beam modeling.
2. **Ray generation**: In OptiX, a dedicated **IVUS ray generator** produces one ray per angular sample from a common origin, with direction sweeping 0→2π in the probe’s xz-plane.
3. **Materials**: **Blood** is updated to literature values; three **IVUS/vascular** materials are added: **lumen**, **vessel_wall**, and **extravascular** (with cited attenuation and impedance).
4. **PSF and TGC**:  
   - **Axial**: IVUS uses a **causal, Hanning-windowed axial PSF** so the strong wall echo does not smear backward into the lumen.  
   - **Lateral**: For IVUS, **depth-dependent lateral PSF** is introduced (Gaussian beam model: beam waist at focus, Rayleigh length, σ(depth)); when element radius and focal length are set, a 2D kernel (depth × angle) is built and applied via a new **depth-dependent column convolution**. For point-source probes, element spacing is 0; a safe fallback lateral width is used.  
   - **TGC** is made **probe-type-specific** (IVUS: short depth range, moderate gain per cm) and cached per probe type.
5. **Scan conversion and display**: A new **IVUS scan conversion** path produces an **unwrapped** display: horizontal = angle (0–360°), vertical = depth (mm). The simulator’s coordinate bounds for IVUS are set to (0–360°, 0–t_far mm).
6. **Pipeline and scattering**: Pipeline parameters are extended with **scattering resolution** (finer for IVUS), **scatter integral scale**, and a **disable_scatter** flag. Scattering logic is corrected so **depth-bin indexing** uses the true ray depth and avoids streaks.
7. **Physics**: **Oblique incidence** reflection (acoustic impedance formula with cos θ_i, cos θ_t from Snell’s law) and **refraction** with a small **t_min** for the refracted ray to avoid self-intersection at the vessel wall.
8. **Utilities and Python**: **Cylinder mesh generation** (single and thick-walled) for vessel phantoms; **Python bindings** for `IVUSProbe` and updated material/SimParams docs; **raysim** package exports `IVUSProbe`.

The following sections walk through these changes by component (probes, materials, raytracing, PSF/TGC, scan conversion, pipeline/scattering, Python/utils), without detailing example or evaluation scripts.

---

## 2. Probe type and IVUS probe class

**Implementation (probe types):** [probe_types.hpp#L28](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/probe_types.hpp#L28), [probe.hpp#L258-L268](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/probe.hpp#L258-L268), [ivus_probe.hpp#L1-L107](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/ivus_probe.hpp#L1-L107).

### 2.1 ProbeType enum

- **Implementation:** [probe_types.hpp#L28](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/probe_types.hpp#L28).
- New value: `PROBE_TYPE_IVUS = 3` (in addition to curvilinear, linear array, phased array).

**Justification:** The pipeline (ray gen, PSF choice, TGC, scan conversion, display bounds) is driven by **probe type**. Adding a distinct **PROBE_TYPE_IVUS** allows the simulator to branch on IVUS-specific physics (causal axial PSF, depth-dependent lateral PSF, IVUS TGC, unwrapped scan conversion) without affecting existing probes.

### 2.2 BaseProbe extensions

- **Implementation:** [probe.hpp#L258-L268](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/probe.hpp#L258-L268).
- New virtual accessors (default 0 for non-IVUS probes):
  - `get_element_radius_mm()` — element radius in mm (for focused single-element probes).
  - `get_focal_length_mm()` — focal length in mm.

These are used by the simulator to build the depth-dependent lateral PSF when both are positive (IVUS with focused element model).

**Justification:** IVUS uses a single small transducer (often ~0.6 mm radius, ~4 mm focal length) at the catheter center. Exposing **element_radius_mm** and **focal_length_mm** on the base probe allows the simulator to apply a **depth-dependent lateral beam model** (Gaussian beam: waist at focus, Rayleigh length) so the effective lateral resolution varies with depth as in real IVUS, instead of treating the probe as a pure point source with no aperture.

### 2.3 IVUSProbe class

- **Implementation:** [ivus_probe.hpp#L1-L107](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/ivus_probe.hpp#L1-L107) — **new file**.
- **IVUSProbe** extends `BaseProbe` with:
  - **Single origin**: `get_local_element_position()` always returns the origin (0,0,0) in local coordinates (catheter center).
  - **Radial directions**: `get_local_element_direction()` returns a unit vector in the xz-plane; angle = 2π × element_idx / num_elements, with +z as reference (consistent with OptiX IVUS ray gen).
  - **Sector angle**: 360°.
  - **Radius / width**: 0 (point source).
  - **Probe type**: `PROBE_TYPE_IVUS`.
  - **Parameters**: `num_angular_rays` (stored as `num_elements_x_`), frequency (e.g. 40 MHz), elevational_height (often 0), `element_radius_mm` (e.g. 0.6), `focal_length_mm` (e.g. 4). Constructor passes `width = 0` to the base.

No separate `.cpp`; the class is header-only.

**Justification:** **Single origin** and **360° radial directions** match the physical geometry of an IVUS catheter: the transducer sits at the center of the vessel and is rotated (or synthetically sampled) over 2π to form a cross-sectional image. **Width = 0** and **radius = 0** model a point-like source for ray casting; the finite aperture is represented later via the depth-dependent lateral PSF (element_radius_mm, focal_length_mm). **Sector angle 360°** ensures the scan-conversion and display bounds treat the image as a full circumferential sweep.

---

## 3. Materials

**Implementation:** [material.cpp#L40-L56](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/material.cpp#L40-L56).

- **Blood**: Updated to Z = 1.68 MRayl, α = 0.2 dB/(cm·MHz), c = 1584 m/s (literature: PMC3570716, PMC5126009).
- **New IVUS/vascular materials** (with comments citing literature and attenuation in dB/(cm·MHz)):
  - **lumen**: Same as updated blood (1.68, 0.2, 1584).
  - **vessel_wall**: c = 1571 m/s, Z = 1.82 MRayl, α ≈ 1.0 (vascular/coronary 50 MHz regime).
  - **extravascular**: Muscle-like c = 1547 m/s, Z = 1.62 MRayl, α = 0.7.

References in comments: Goss et al. compilations, PMC3570716 (Ultrasound Med Biol 2013), PMC5126009 (J Ultrasound 2016), Lockwood et al. UMB 17(7) 1991.

**Diff (vs main):** `Materials::Materials()` — blood entry and three new material entries added to `materials_` initializer; no signature change.

**Justification:** **Blood** was updated to **literature values** (Z = 1.68 MRayl, c = 1584 m/s, α = 0.2 dB/(cm·MHz) from PMC3570716, PMC5126009) so that reflection and attenuation at blood–tissue interfaces match published data; the original values were less aligned with vascular ultrasound references. **Lumen**, **vessel_wall**, and **extravascular** were added because IVUS scenes explicitly model (1) the blood-filled lumen, (2) the vessel wall (intima/media with impedance and attenuation from coronary/vascular 50 MHz data), and (3) perivascular tissue. Using correct Z and α is necessary for **physically correct reflection coefficients** at interfaces (lumen–wall, wall–extravascular) and for **depth-dependent attenuation** along each ray (Beer–Lambert).

---

## 4. OptiX raytracing (IVUS rays and physics)

**Implementation:** [optix_trace.cu](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu) (see subsections for line ranges).

### 4.1 Ray generation for IVUS

- **Implementation:** [optix_trace.cu#L342-L356](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L342-L356) (`generate_ivus_probe_ray_local`), [optix_trace.cu#L358-L390](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L358-L390) (raygen switch).
- New device function **`generate_ivus_probe_ray_local`**:
  - Maps launch dimension `d_x` to angle in [0, 2π].
  - **Origin**: (0, 0, 0) in local coordinates.
  - **Direction**: radial in xz-plane, `(sin(angle), 0, cos(angle))`, normalized; +z at angle 0 to match `IVUSProbe::get_local_element_direction`.

- In the **raygen** `__raygen__rg`, the switch on `probe_type` is extended with `PROBE_TYPE_IVUS` calling this function. Payload `ray.t_ancestors` is set to 0 (line 406). Elevation is applied afterward as for other probes.

**Justification:** IVUS rays must **emanate from a single point** (catheter center) and **sweep 360° in the imaging plane** (xz with +z as reference). This matches the physical acquisition: one transducer at the center, with A-lines acquired at evenly spaced angles. Mapping the launch dimension to angle in [0, 2π] and using a common origin is the minimal change to the raygen to support this geometry; elevation is kept for consistency with the shared pipeline but is typically zero for 2D IVUS.

### 4.2 Payload and scattering

- **Implementation:** [optix_trace.cu#L406](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L406) (payload), [optix_trace.cu#L44-L55](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L44-L55) (`get_scattering_value`), [optix_trace.cu#L89-L119](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L89-L119) (`sample_intensities`), [optix_trace.cu#L438-L450](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L438-L450) (miss), [optix_trace.cu#L468-L476](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L468-L476) (closest_hit).
- **Payload**: `ray.t_ancestors` is explicitly set to 0 in the raygen (needed for correct depth-bin indexing in scatter and hit).
- **Scattering**:
  - **`get_scattering_value`** (modified): No longer takes a fixed `resolution_mm` parameter; it uses **`params.scattering_resolution_mm`** so the pipeline can set a finer scale for IVUS (e.g. 10 mm vs 50 mm).
  - **`sample_intensities`** (modified): Now takes **scanline pointer and ray_index**; early-out if `params.disable_scatter`; depth-bin indexing via `get_intensity_offset(t_ancestors + t_val)` with bounds check; integral scaling via `scatter_integral_scale`; contributions written to `scanline[bin]` with `segment_weight`.

**Justification:**  
- **`ray.t_ancestors` initialized to 0:** Depth along the ray must include the full path from the probe (origin). If `t_ancestors` were left uninitialized, scatter and hit contributions would use wrong depths for binning, producing **streaks** or misplacement. Explicitly setting it to 0 in the raygen ensures every ray starts with correct cumulative path length.  
- **`get_scattering_value` using `params.scattering_resolution_mm`:** The scattering texture is sampled in world space divided by a resolution (voxel size). IVUS operates at **much smaller spatial scale** (mm, 1–10 mm depth) than abdominal imaging (cm). Using a **finer resolution** (e.g. 10 mm vs 50 mm) for IVUS gives **speckle at the appropriate scale** (smaller correlation length) so tissue texture looks plausible.  
- **Depth-bin indexing in `sample_intensities`:** The original code used `intensities += get_intensity_offset(t_ancestors + t_min)` and then wrote to `intensities[step]` with a **step index**, not a **depth-derived bin**. That decouples the write index from true propagation depth and causes **axial streaks**. Writing to `scanline[bin]` with `bin = get_intensity_offset(t_ancestors + t_val)` ensures scatter is placed at the **correct depth bin** for the round-trip time.  
- **`scatter_integral_scale`:** The line integral of scatter has no natural scale that matches display (0–1 or dB). Empirically, **strict integration** gives very dark tissue in vascular/cystic phantoms while wire phantoms (reflection-dominated) are fine. A scale factor (~40) brings the scatter contribution into a **displayable range** so both reflection-dominated (wire) and scatter-dominated (tissue) phantoms are usable without changing the underlying physics of the integral.  
  **Tuning:** Set in `raytracing_ultrasound_simulator.cpp` when filling `params.scatter_integral_scale`. **Increase** (e.g. 60–80) if tissue background remains too dark in vascular/cystic phantoms; **decrease** (e.g. 20–30) if tissue is too bright or reflections (e.g. wire targets) are drowned out. Use **0** for strict physics (no scaling); compare wire phantom (reflections should dominate) and cystic/tissue phantom (scatter visible) to balance. Probe-type-specific values could be used (e.g. one scale for IVUS, another for abdominal) if needed.

**Diff (vs main) — `get_scattering_value`:**
```diff
- static __device__ float get_scattering_value(float3 pos, const Material* material,
-                                              float resolution_mm = 50.f) {
-   // Convert point to texture coordinates
+ static __device__ float get_scattering_value(float3 pos, const Material* material) {
+   const float resolution_mm = params.scattering_resolution_mm;
    pos /= resolution_mm;
```

**Diff (vs main) — `sample_intensities`:**
```diff
  static __device__ void sample_intensities(float3 origin, float3 dir, float t_ancestors, float t_min,
                                            float t_max, float intensity, const Material* material,
-                                           float* intensities) {
-   // Early out for materials with zero scattering density or coefficient
+                                           float* scanline, uint32_t ray_index) {
+   if (params.disable_scatter) { return; }
    if ((material->mu0_ <= 0.f) || (material->sigma_ == 0.f)) { return; }
-
-   const uint32_t steps = ((t_max - t_min) / params.t_far) * params.buffer_size + 0.5f;
-   const float t_step = (t_max - t_min) / steps;
-   const float3 start = origin + t_min * dir;
-   intensities += get_intensity_offset(t_ancestors + t_min);
-   for (uint32_t step = 0; step < steps; ++step) {
-     const float distance = (step * t_step);
-     const float3 pos = start + distance * dir;
-     intensities[step] += get_scattering_value(pos, material) * intensity *
-                          get_intensity_at_distance(distance, material->attenuation_);
-   }
+   const float range = t_max - t_min;
+   ...
+   const float integral_weight = (params.scatter_integral_scale > 0.f) ? (range * params.scatter_integral_scale) : range;
+   const uint32_t steps = (range / params.t_far) * params.buffer_size + 0.5f;
+   ... (per-step: t_val, depth, bin = get_intensity_offset(depth), scanline[bin] += segment_weight * scatter);
  }
```

### 4.3 Reflection and refraction (oblique incidence)

- **Implementation:** [optix_trace.cu#L186-L194](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L186-L194) (`calculate_reflection_coefficient`), [optix_trace.cu#L455-L620](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L455-L620) (`closest_hit`), [optix_trace.cu#L588-L619](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu#L588-L619) (refracted ray t_min).
- **Reflection coefficient** (modified): Replaced normal-incidence formula by **oblique incidence**: `R_p = (Z2/cos(θ_t) - Z1/cos(θ_i)) / (Z2/cos(θ_t) + Z1/cos(θ_i))`, with `R_I = R_p²`. Angles come from Snell’s law: `sin(θ_t) = (v1/v2)*sin(θ_i)`, then `cos(θ_t)`. Grazing and total internal reflection (cos ≤ 0) yield R = 1.
- **Hit contribution**: Both the **acoustic reflection** and the **specular term** (Mattausch-style) are **added** into the same depth bin (`hit_bin`), instead of overwriting.
- **Refracted ray**: Origin is **back_start** (just inside the second medium); **t_min** for the refracted ray is set to a small constant (e.g. 1e-3 mm) to avoid self-intersection at the same interface.

**Justification:**  
- **Oblique-incidence reflection coefficient:** At an interface, **pressure and normal particle velocity** are continuous. For oblique incidence the effective impedances are **Z/cos(θ)** (normal component). The pressure reflection coefficient is then R_p = (Z2/cos(θ_t) − Z1/cos(θ_i)) / (Z2/cos(θ_t) + Z1/cos(θ_i)), with intensity R_I = R_p². The original formula used only **normal incidence** (single cos(θ)), which is incorrect when the ray is not perpendicular to the surface (e.g. IVUS rays hitting the vessel wall at various angles). Using **Snell’s law** to get θ_t from θ_i and the two cosines yields the **correct acoustic reflection** from first principles.  
- **Adding reflection and specular to the same bin:** The interface echo has two contributions: (1) the **acoustic reflection** (R × intensity) and (2) an **empirical specular term** (Mattausch-style, directivity-like). Both should contribute to the **same depth** (the interface). The original code **overwrote** the bin with only the specular term; now both are **added** so the total echo at the interface is physically consistent (reflected energy) plus the empirical term. The specular term is weighted by **2.f** in `closest_hit` (e.g. `scanline[hit_bin] += 2.f * specular_reflection`).  
  **Tuning:** The factor **2.f** is empirical. To tune: in `optix_trace.cu` search for `2.f * specular_reflection`; **increase** for stronger interface highlights (more “specular” appearance), **decrease** (or 1.f) for a more diffuse interface. Compare to real IVUS or reference sims if available.  
- **Refracted ray: back_start and t_min_refract:** Physically the refracted ray propagates **in the second medium**, so its origin must be **just inside** that medium (back_start). Using **front_start** when the refracted direction pointed away from the normal was a heuristic that could place the origin in the wrong medium. Always using **back_start** ensures we are in the transmitted medium. **t_min_refract = 1e-3 mm** avoids the refracted ray **immediately re-hitting the same surface** (e.g. inner vessel wall or mesh self-intersection), which is a **numerical robustness** fix rather than a change in physics.

**Diff (vs main) — `calculate_reflection_coefficient`:**
```diff
- static __device__ float calculate_reflection_coefficient(float incident_angle,
+ static __device__ float calculate_reflection_coefficient(float cos_theta_i, float cos_theta_t,
                                                           const Material* material1,
                                                           const Material* material2) {
    float Z1 = material1->impedance_;
    float Z2 = material2->impedance_;
-   float cos_theta = fabsf(__cosf(incident_angle));
-   float R = ((Z2 * cos_theta - Z1) / (Z2 * cos_theta + Z1));
-   return R * R;
+   if (cos_theta_i <= 1e-6f || cos_theta_t <= 1e-6f) { return 1.f; }
+   float R_p = (Z2 / cos_theta_t - Z1 / cos_theta_i) / (Z2 / cos_theta_t + Z1 / cos_theta_i);
+   return R_p * R_p;
  }
```

**Diff (vs main) — `closest_hit` (reflection + refracted ray):**
```diff
- const float incident_angle = acosf(fabsf(dot(ray_dir, normal)));
- const float R = calculate_reflection_coefficient(incident_angle, current_material, next_material);
+ const float cos_i = fabsf(dot(ray_dir, normal));
+ const float sin_i = sqrtf(1.f - cos_i * cos_i);
+ const float v1 = current_material->speed_of_sound_;  const float v2 = next_material->speed_of_sound_;
+ const float sin_t = (v1 / v2) * sin_i;
+ const float cos_t = (sin_t < 1.f) ? sqrtf(1.f - sin_t * sin_t) : 0.f;
+ const float R = calculate_reflection_coefficient(cos_i, cos_t, current_material, next_material);
  ...
- scanline[get_intensity_offset(ray.t_ancestors + t)] = 2.f * specular_reflection;
+ const uint32_t hit_bin = get_intensity_offset(ray.t_ancestors + t);
+ scanline[hit_bin] += reflected_intensity;
+ scanline[hit_bin] += 2.f * specular_reflection;
  ...
- const float3 start = (dot(refracted_dir, wld_norm) > 0.f) ? front_start : back_start;
+ const float t_min_refract = 1e-3f;
+ const float3 start = back_start;
  optixTrace(params.handle, start, refracted_dir,
-            0.f,   // tmin
+            t_min_refract,   // tmin: avoid self-intersection
             params.t_far - refracted_ray.t_ancestors, ...);
```

### 4.4 Pipeline parameters (OptiX)

- **Implementation:** [optix_trace.hpp#L29-L45](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/optix_trace.hpp#L29-L45) (struct `Params`).
- **Params** struct extended with:
  - `scattering_resolution_mm` — voxel scale for scattering texture (IVUS uses smaller value).
  - `disable_scatter` — if non-zero, scatter accumulation is skipped.
  - `scatter_integral_scale` — scale factor for the scatter integral (0 = strict; ~40 used for visible tissue background).

**Justification:** These pipeline parameters allow the **same raytracing kernel** to be used for both abdominal and IVUS without recompilation. **scattering_resolution_mm** is set per run (10 for IVUS, 50 for abdominal) so speckle scale matches the imaging geometry. **disable_scatter** is a switch for debugging or comparison. **scatter_integral_scale** is the empirical scale discussed in §4.2 so scatter contributes in a displayable range.  
**Tuning:** **scattering_resolution_mm:** Set in `raytracing_ultrasound_simulator.cpp` (e.g. 10 for IVUS, 50 for abdominal). **Smaller** values give **finer speckle** (smaller correlation length); **larger** values give **coarser speckle**. Tune to match target speckle size (e.g. from literature or reference images) or to desired texture. **scatter_integral_scale:** See §4.2; same tuning as above (set in same place).

**Pass 1 update (configuration-driven):** As of the calibration-driven refactor (§11), the three `Params` fields above are populated from `SimParams::scattering_resolution_mm`, `SimParams::disable_scatter`, and `SimParams::scatter_integral_scale` rather than being hard-coded in `simulate()`. `scattering_resolution_mm == 0.f` in `SimParams` is treated as a sentinel meaning *"auto from probe type"* and falls back to the historical 10 mm (IVUS) / 50 mm (abdominal) defaults; any positive value overrides them. The header itself was previously missing these three fields even though `optix_trace.cu` and the simulator already referenced them — see §11.1 for the (latent) header fix.

---

## 5. Simulator: PSF and TGC

**Implementation:** [raytracing_ultrasound_simulator.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp), [raytracing_ultrasound_simulator.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp) (see subsections).

### 5.1 Axial PSF

- **Implementation:** [raytracing_ultrasound_simulator.cpp#L89-L136](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L89-L136) (`create_ivus_axial_psf_causal`), [raytracing_ultrasound_simulator.cpp#L337-L351](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L337-L351) (axial PSF branch in `update_psfs`). Header: [raytracing_ultrasound_simulator.hpp#L135](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp#L135) (`psf_ax_probe_type_`).
- **IVUS**: New helper **`create_ivus_axial_psf_causal`** builds a causal, Hanning-windowed one-sided kernel (extent from pulse duration and wavelength). Used when `probe->get_probe_type() == PROBE_TYPE_IVUS`.
- **Other probes**: Unchanged Gaussian axial PSF via `create_gaussian_psf`.
- **Caching**: Axial PSF is invalidated when **probe type** or frequency changes (`psf_ax_probe_type_` in header).

**Justification:** A **symmetric** (two-sided) axial kernel smears energy both **shallower and deeper** than the true interface. For IVUS, the **vessel wall** is a strong reflector; convolution with a symmetric kernel would **smear the wall echo backward into the lumen**, making the lumen appear bright and destroying the anechoic blood appearance. A **causal** kernel (energy only at and “deeper” in index space, i.e. no right half) ensures that the wall echo **only smears deeper**, preserving a **dark lumen** and a single peak at the true wall depth. The **Hanning window** shapes the pulse to reduce sidelobes while keeping the causal constraint. This is a **physics-based** choice: causality in time (echo arrives after transmission) maps to one-sided convolution in depth.

### 5.2 Lateral PSF

- **Implementation:** [raytracing_ultrasound_simulator.cpp#L282-L368](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L282-L368) (depth-dependent 2D kernel build and fallback lateral in `update_psfs`); header [raytracing_ultrasound_simulator.hpp#L143-L150](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp#L143-L150) (`psf_lat_2d_` and related members).
- **Depth-dependent lateral (IVUS)**: When probe type is IVUS and `element_radius_mm` and `focal_length_mm` are both > 0, the simulator builds a **2D lateral kernel** (depth_bins × kernel_len) using Gaussian beam model (w0, z_R, w(z)). Stored in **`psf_lat_2d_`**; dimensions and parameters cached.
- **Convolution**: If `psf_lat_2d_` is present, **`convolve_columns_depth_dependent`** is called; otherwise **`convolve_columns`** with 1D lateral kernel.
- **Fallback for point-source**: When `element_spacing == 0` (IVUS), lateral resolution and inverse spacing set to safe defaults to avoid division by zero.

**Justification:** For a **focused circular aperture**, the **Gaussian beam model** (Siegman, Goodman) gives: beam waist at focus w0 = λF/(2a), Rayleigh length z_R = πw0²/λ, and beam radius w(z) = w0√(1 + (z/z_R)²) with z = depth − focal_length. Using a **depth-invariant** lateral kernel (as for linear arrays) would be wrong for IVUS, where the beam **narrows near the focus** and **widens** elsewhere. The **depth-dependent lateral PSF** applies the correct σ(depth) in angle-bin space so lateral blur matches the physical beam. **Fallback** when element_spacing is 0: the original code used 1/element_spacing for the lateral kernel; IVUS has no element array, so element_spacing is 0. Using a **finite lateral width** derived from typical depth and aperture (λ·depth/aperture) and inv_spacing = 1 avoids division by zero and gives a reasonable kernel when the full 2D depth-dependent kernel is not built (e.g. focal_length_mm or element_radius_mm not set).

### 5.3 update_psfs signature and TGC

- **Implementation:** [raytracing_ultrasound_simulator.cpp#L260-L376](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L260-L376) (`update_psfs`), [raytracing_ultrasound_simulator.cpp#L500-L520](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L500-L520) (TGC). Header: [raytracing_ultrasound_simulator.hpp#L154-L155](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp#L154-L155) (`update_psfs` declaration, `tgc_probe_type_`).

**Diff (vs main) — `update_psfs`:**
```diff
- void RaytracingUltrasoundSimulator::update_psfs(const BaseProbe* probe, cudaStream_t stream) {
+ void RaytracingUltrasoundSimulator::update_psfs(const BaseProbe* probe, cudaStream_t stream,
+                                                 uint32_t buffer_size, float t_far) {
    if (probe_frequency_ != probe->get_frequency()) {
      ...
+     psf_lat_2d_.reset();
    }
+   const ProbeType pt = probe->get_probe_type();
+   if (psf_ax_probe_type_ != pt) { psf_ax_probe_type_ = pt; psf_ax_.reset(); }
    ...
+   // Build psf_lat_2d_ for IVUS when element_radius_mm and focal_length_mm > 0 (Gaussian beam model)
+   // Axial: IVUS branch uses create_ivus_axial_psf_causal; else create_gaussian_psf
+   // Lateral: IVUS fallback lat_width and inv_spacing when element_spacing == 0
  }
```

**Diff (vs main) — TGC:**
```diff
-   if (!tgc_curve_ || (tgc_curve_->get_size() / sizeof(float) != sim_params.buffer_size)) {
-     std::vector<ControlPoint> control_points{{0.f, 0.f}, {40.f, 28.f}};
+   const bool tgc_size_ok = ...;
+   const bool tgc_probe_match = tgc_probe_type_.has_value() && (*tgc_probe_type_ == probe->get_probe_type());
+   if (!tgc_curve_ || !tgc_size_ok || !tgc_probe_match) {
+     std::vector<ControlPoint> control_points;
+     if (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) {
+       control_points = {{0.f, 0.f}, {1.f, tgc_dB_per_cm}};
+     } else {
+       control_points = {{0.f, 0.f}, {40.f, 28.f}};
+     }
      tgc_curve_ = create_piece_wise_tgc(...);
+     tgc_probe_type_ = probe->get_probe_type();
    }
```

**Justification:** **update_psfs(buffer_size, t_far):** The IVUS **depth-dependent lateral** kernel depends on the depth range (t_far) and the number of depth samples (buffer_size) to build the 2D kernel (depth_bins × kernel_len). Passing these in allows the simulator to build the correct kernel without assuming global state. **TGC probe-type-specific:** **Time-gain compensation** compensates for **attenuation with depth** (and sometimes diffraction). Abdominal imaging uses depths of order **tens of cm** and a TGC curve (e.g. 0–40 cm, ~28 dB at 40 cm). IVUS imaging depth is **millimeters** (e.g. 0–10 mm), and tissue attenuation at 40 MHz is ~α·f per cm. Using the **same** TGC curve for IVUS would over-compensate (huge gain at 1 cm) and distort depth dependence. A **separate** TGC for IVUS (e.g. 0–1 cm, ~2 dB/cm) matches the physical depth range and attenuation scale. **Caching by probe type** ensures switching between IVUS and another probe type rebuilds the TGC curve appropriately.  
**Tuning (IVUS TGC):** In `raytracing_ultrasound_simulator.cpp`, the IVUS TGC uses control points `{{0.f, 0.f}, {1.f, tgc_dB_per_cm}}` with **tgc_dB_per_cm = 2.f**. Approximate gain per cm from tissue attenuation is **α × f_MHz** (α in dB/(cm·MHz)). For vessel_wall α ≈ 1, 40 MHz → ~40 dB/m = **4 dB/cm**; 2 dB/cm is deliberately moderate so wire phantoms (lumen) and cystic phantoms (tissue) both show plausible depth dependence. **Increase** tgc_dB_per_cm if deeper tissue is too dark; **decrease** if near-field is over-gained or depth gradient looks wrong. Extend the control-point depth (e.g. 1.5 or 2 cm) if imaging beyond 1 cm. Adjust so that (1) wire echoes do not get over-amplified with depth and (2) tissue at 5–10 mm is visible without clipping.

**Pass 1 update (user-supplied TGC schedule):** The TGC block now also accepts a caller-supplied schedule via `SimParams::tgc_control_points` (a list of `(depth_cm, gain_db)` pairs). Behavior:

- **Empty list (default)** → preserves the legacy probe-type cached path (IVUS: `{0,0}, {1,2}`; abdominal: `{0,0}, {40,28}`) verbatim, so existing scripts that only set the previously exposed fields are byte-identical.
- **Non-empty list** → the curve is rebuilt every frame (no caching) so frame-to-frame schedule changes are honored, and the probe-type cache is invalidated so a later frame that goes back to the empty path re-builds the default schedule. This is the path the YAML calibration takes for the PV .035 (5 control points spanning the 0–3 cm IVUS range, see `instrument-calibration/p035_visions/volcano_s5i.yaml`).

The internal `ControlPoint` struct in the .cpp is unchanged; a new public `raysim::TgcControlPoint { depth_cm, gain_db }` lives in the simulator header so callers (Python bindings, host code) can build a schedule without touching simulator-private types. See §11.2 for the bindings.

---

## 6. CUDA algorithms: depth-dependent convolution and IVUS scan conversion

**Implementation:** [cuda_algorithms.cu](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu), [cuda_algorithms.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/cuda_algorithms.hpp).

### 6.1 Depth-dependent column convolution

- **Implementation:** [cuda_algorithms.cu#L86-L118](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu#L86-L118) (kernel), [cuda_algorithms.cu#L564-L577](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu#L564-L577) (host). Header: [cuda_algorithms.hpp#L68-L72](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/cuda_algorithms.hpp#L68-L72), [cuda_algorithms.hpp#L219](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/cuda_algorithms.hpp#L219).
- New kernel **`convolve_columns_depth_dependent_kernel`**: For each (depth, angle, plane) sample, depth index maps to a **depth_bin**; 1D kernel from **kernel_2d** (row-major). Convolution along columns (angle dimension).
- **`convolve_columns_depth_dependent`** host method launches this kernel; used by the simulator when `psf_lat_2d_` is set.

**Justification:** The **lateral** dimension in IVUS is **angle** (columns are angular samples). The beam width **varies with depth**, so a single 1D lateral kernel is incorrect. The depth-dependent kernel implements the **Gaussian beam** lateral PSF: at each depth bin we apply the kernel that corresponds to w(z) at that depth, so the convolution is **physically consistent** with the beam model.

### 6.2 IVUS scan conversion

- **Implementation:** [cuda_algorithms.cu#L477-L494](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu#L477-L494) (kernel), [cuda_algorithms.cu#L805-L831](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu#L805-L831) (host). Header: [cuda_algorithms.hpp#L205-L207](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/cuda_algorithms.hpp#L205-L207), [cuda_algorithms.hpp#L224](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/cuda_algorithms.hpp#L224).
- New kernel **`scan_convert_ivus_kernel`**: Input texture (depth_norm, angle_norm); output 2D buffer (angle pixels × depth pixels). Unwrapped IVUS display: angle horizontal, depth vertical (probe at top).
- **`scan_convert_ivus`**: Manages CudaArray/CudaTexture for scanlines, uploads, launches kernel, returns B-mode buffer. Caching invalidated when input size changes.

**Justification:** IVUS is conventionally displayed in **unwrapped** form: **horizontal = angle** (0–360°) and **vertical = depth** (probe at top). This matches clinical and research viewers and preserves the one-to-one mapping from (angle, depth) to (x, y) pixel. The kernel is a direct resample from polar (depth_norm, angle_norm) to this Cartesian layout; no sector geometry or masking is needed.

---

## 7. Simulator: pipeline wiring and display bounds

**Implementation:** [raytracing_ultrasound_simulator.cpp#L418-L433](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L418-L433) (pipeline params), [raytracing_ultrasound_simulator.cpp#L464-L477](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L464-L477) (PSF step), [raytracing_ultrasound_simulator.cpp#L591-L611](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L591-L611) (scan conversion switch), [raytracing_ultrasound_simulator.cpp#L625-L638](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp#L625-L638) (display bounds). Header: [raytracing_ultrasound_simulator.hpp#L83-L104](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp#L83-L104) (`get_min_x` / `get_max_x` / `get_min_z` / `get_max_z`).

- **Pipeline params**: **scattering_resolution_mm** = 10 for IVUS else 50; **disable_scatter** = 0; **scatter_integral_scale** = 40.
- **PSF step**: Calls **`update_psfs(probe, stream, buffer_size, t_far)`**; then either **convolve_columns_depth_dependent** (if `psf_lat_2d_`) or **convolve_columns**.
- **Scan conversion**: Switch on probe type; **`PROBE_TYPE_IVUS`** calls **`scan_convert_ivus`** (plane size and b_mode_size only).
- **Display bounds**: For **PROBE_TYPE_IVUS**, **get_min_x/get_max_x/get_min_z/get_max_z** return (0, 360, 0, t_far). Implementation: `simulate()` sets `min_x_`, `max_x_`, `min_z_`, `max_z_` in the switch (lines 631–638).

**Justification:** **Pipeline params:** Setting **scattering_resolution_mm** to 10 for IVUS (vs 50 for abdominal) matches the finer spatial scale of IVUS so the scattering texture is sampled at an appropriate voxel size (§4.2). **scatter_integral_scale = 40** is the empirical scale for displayable scatter. **PSF step:** Calling **update_psfs** with buffer_size and t_far is required to build the IVUS depth-dependent lateral kernel; choosing **convolve_columns_depth_dependent** when the 2D kernel exists applies the correct beam model. **Display bounds (0, 360, 0, t_far):** The unwrapped IVUS image has **angle in degrees** on the horizontal axis (0–360°) and **depth in mm** on the vertical axis (0 to t_far). Exposing these bounds lets the client (e.g. Python or C++) label axes correctly and set aspect ratio so the image is not stretched incorrectly.  
**Tuning:** Empirical pipeline values are set in `raytracing_ultrasound_simulator.cpp` (params block before `pipeline_params_.upload`). **scattering_resolution_mm:** 10 for IVUS, 50 for abdominal; tune as in §4.4 (finer → finer speckle). **scatter_integral_scale:** 40 by default; tune as in §4.2 (higher → brighter tissue, lower → darker tissue; 0 = strict).

---

## 8. Python bindings and package exports

**Implementation:** [raysim_bindings.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/python/raysim_bindings.cpp) (IVUSProbe class and SimParams/Materials docs); [raysim/__init__.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/raysim/__init__.py), [raysim/cuda/__init__.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/raysim/cuda/__init__.py) (exports).

- **IVUSProbe** bound as pybind11 class (constructors, readonly **element_radius_mm**, **focal_length_mm**). Defaults: num_angular_rays=256, frequency=40, element_radius_mm=0.6, focal_length_mm=4, etc.
- **Materials**: Docstring for `get_index` updated to list **lumen**, **vessel_wall**, **extravascular**.
- **SimParams**: Docstrings for **t_far**, **buffer_size**, **b_mode_size** clarified (mm, samples per ray, IVUS unwrapped angle×depth).
- **IVUSProbe** added to exports and `__all__` in `raysim/__init__.py` and `raysim/cuda/__init__.py`.

**Justification:** These are **API and usability** changes, not physics. Exposing **IVUSProbe** and the new materials (**lumen**, **vessel_wall**, **extravascular**) in Python allows users to build IVUS scenes and run simulations without touching C++. Documenting **t_far** (mm), **buffer_size** (samples per ray), and **b_mode_size** (angle × depth for IVUS) in SimParams reduces misuse (e.g. wrong units or expecting sector geometry for IVUS).

---

## 9. Utilities: vessel phantom meshes

**Implementation:** [phantom_maker.py#L327-L394](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/utils/phantom_maker.py#L327-L394) (`generate_cylinder_mesh`), [phantom_maker.py#L396-L458](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/utils/phantom_maker.py#L396-L458) (`generate_cylinder_thick_mesh`), [phantom_maker.py#L489-L541](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/utils/phantom_maker.py#L489-L541) (CLI cylinder).

- **`generate_cylinder_mesh`**: Writes an open cylinder OBJ (no caps), axis along Y, cross-section in xz; optional **inward normals** so rays from the lumen hit the front face. Default 129 segments to avoid alignment with 256 IVUS rays.
- **`generate_cylinder_thick_mesh`**: Writes **Cylinder_inner.obj** and **Cylinder_outer.obj** for a thick vessel wall (inner/outer radius, same segment count and inward normals).
- **CLI**: New phantom type **cylinder** with options **--cylinder-radius**, **--cylinder-length**, **--cylinder-segments**, **--cylinder-thick**, **--cylinder-inner-radius**, **--cylinder-outer-radius**.

**Justification:** IVUS validates against **vessel phantoms**: a lumen (blood) surrounded by a **cylindrical wall**. The cylinder axis is along **Y** (vessel axis); the **cross-section in xz** is the IVUS imaging plane. **Inward normals** ensure that rays cast **from the center** (probe) hit the **front face** of the mesh (OptiX back-face culling would otherwise hide the wall). **129 segments** avoids aligning mesh edges with 256 angular rays, which would cause **periodic intensity bands** (aliasing). The **thick-walled** variant (inner + outer cylinder) models a wall with finite thickness and two interfaces (lumen–wall, wall–extravascular) for attenuation and two-layer validation.

---

## 10. Evaluation (vessel, wire phantom, cystic phantom)

A separate document **[ivus_evaluation_writeup.md](ivus_evaluation_writeup.md)** describes the three evaluation setups used to validate the IVUS implementation:

- **IVUS vessel example** (`ivus_example.py`): thick-walled cylinder phantom; tests geometry, interface echoes, and attenuation; expected unwrapped image shows two concentric bright rings at ~3.5 and ~4 mm depth.
- **Wire phantom** (`wire_phantom_evaluation.py`): point targets (bone spheres) at 1–5 mm in a spiral; tests resolution and geometric accuracy; expected unwrapped image shows five bright spots in a spiral pattern.
- **Cystic-resolution phantom** (`cystic_resolution_phantom_evaluation.py`): tissue background with fluid cysts; tests contrast and scatter/TGC; expected unwrapped image shows speckled tissue with five darker cyst regions.

That writeup includes the **unwrapped B-mode images** from each simulation for review and the commands to regenerate them.

---

## 11. Configuration-driven processing pipeline (Pass 1)

This section describes the **Pass 1 plumbing** added to make the simulator dynamically configurable from a YAML config without recompiling. The motivating use case is the per-instrument calibration pipeline that lives in `instrument-calibration/p035_visions/` and emits a `volcano_s5i.yaml` consumed at runtime via `raysim.IvusSimConfig`.

The design constraint throughout was **bit-identical default behavior**: every new field has a default value chosen so that any existing example/script that constructs `SimParams()` and only sets the previously exposed knobs produces an unchanged pipeline. Calibration data only takes effect when the YAML overrides those defaults.

**Implementation pointers (Pass 1 surface area):**

- C++ public API: [raytracing_ultrasound_simulator.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp) (new `TgcControlPoint`, extended `SimParams`).
- C++ pipeline: [raytracing_ultrasound_simulator.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp) (`simulate()` reads the new `SimParams` fields).
- OptiX header fix: [optix_trace.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/include/raysim/cuda/optix_trace.hpp) (added the three `Params` fields the .cu/.cpp already used).
- Bindings: [raysim_bindings.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/python/raysim_bindings.cpp) (new `TgcControlPoint`, extended `SimParams` `def_readwrite` set).
- Python config: [raysim/config.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/raysim/config.py) (`IvusSimConfig.to_sim_params()` now sets every Pass 1 field; `_PARTIALLY_WIRED_PATHS` reduced to `()`).

### 11.1 OptiX `Params` header fix (latent bug)

`csrc/cuda/optix_trace.cu` and `csrc/core/raytracing_ultrasound_simulator.cpp` were both reading/writing `params.scattering_resolution_mm`, `params.disable_scatter`, and `params.scatter_integral_scale` (see §4.2 / §4.4 / §7), but the matching `struct Params` in `include/raysim/cuda/optix_trace.hpp` did not declare those fields. That is a **latent build bug**: the simulator could not have compiled against a strictly conforming `Params`. Pass 1 added the three fields at the end of the struct (preserves layout for fields that were already there).

**Diff (vs main) — `Params`:**
```diff
  struct Params {
    ...
    float source_frequency;
    float contact_epsilon;
+   // Pipeline parameters consumed by optix_trace.cu (also see §4):
+   float scattering_resolution_mm;  // voxel size used to sample scattering texture
+   uint32_t disable_scatter;        // non-zero disables scatter accumulation
+   float scatter_integral_scale;    // multiplier on the scatter line integral (0 = strict)
  };
```

**Justification:** The kernels needed these fields and were already reading them; the header just had to declare them so that any future `static_assert`/sizeof check or fresh build environment works. Behavior is unchanged because the .cpp continues to write the same values into them.

### 11.2 Public `TgcControlPoint` and extended `SimParams`

A new public struct **`raysim::TgcControlPoint { float depth_cm; float gain_db; }`** was added to `raytracing_ultrasound_simulator.hpp` so callers can build a piece-wise-linear TGC schedule without touching the file-local `ControlPoint` used inside `simulate()`. `SimParams` then carries:

| `SimParams` field | YAML path | Default | Behavior at default |
|---|---|---|---|
| `tgc_control_points` (`std::vector<TgcControlPoint>`) | `processing.tgc_control_points` | empty | use the existing probe-type schedule (IVUS: 2 dB/cm to 1 cm; abdo: 0–28 dB to 40 cm) and keep the per-probe-type cache |
| `log_multiplier` | `processing.log_multiplier` | `20.f` | matches the prior literal in `cuda_algorithms_->log_compression(...)` |
| `log_floor` | `processing.log_floor` | `1e-19f` | matches the prior literal |
| `median_clip_size` | `processing.median_clip.size` | `5` | matches the prior 5×1 kernel |
| `median_clip_d_min_db` | `processing.median_clip.d_min_db` | `-60.f` | matches the prior dMin |
| `median_clip_d_max_db` | `processing.median_clip.d_max_db` | `0.f` | matches the prior dMax |
| `scattering_resolution_mm` | `processing.scattering_resolution_mm` | `0.f` | sentinel ⇒ probe-type auto (10 IVUS / 50 other) |
| `scatter_integral_scale` | `processing.scatter_integral_scale` | `40.f` | matches the prior literal |
| `disable_scatter` | (not in YAML yet) | `false` | scatter on, as before |

**Justification:** Pass 1 deliberately chose **knobs that were already hard-coded in `simulate()`** (bucket B in the calibration plan) rather than introducing any new physics. Each entry is therefore a one-line replacement of a literal with the corresponding `sim_params.X`, plus a default that reproduces the literal exactly. This minimizes risk and lets the calibration pipeline drive the simulator immediately for the parameters we already extracted (TGC, log compression, median clip, scatter scale).

### 11.3 Pipeline wiring inside `simulate()`

The `simulate()` body picks up the new fields at the same places §4.4, §5.3, §7 already documented:

**Diff (vs main) — pipeline params before optixLaunch:**
```diff
- params.scattering_resolution_mm =
-     (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) ? 10.f : 50.f;
+ params.scattering_resolution_mm =
+     (sim_params.scattering_resolution_mm > 0.f)
+         ? sim_params.scattering_resolution_mm
+         : ((probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) ? 10.f : 50.f);
  ...
- params.disable_scatter = 0;
+ params.disable_scatter = sim_params.disable_scatter ? 1u : 0u;
- params.scatter_integral_scale = 40.f;
+ params.scatter_integral_scale = sim_params.scatter_integral_scale;
```

**Diff (vs main) — TGC block (extends the §5.3 caching to a user-supplied path):**
```diff
+ const bool user_tgc = !sim_params.tgc_control_points.empty();
- if (!tgc_curve_ || !tgc_size_ok || !tgc_probe_match) {
+ if (user_tgc || !tgc_curve_ || !tgc_size_ok || !tgc_probe_match) {
    std::vector<ControlPoint> control_points;
+   if (user_tgc) {
+     control_points.reserve(sim_params.tgc_control_points.size());
+     for (const auto& cp : sim_params.tgc_control_points) {
+       control_points.push_back({cp.depth_cm, cp.gain_db});
+     }
+   } else if (probe->get_probe_type() == ProbeType::PROBE_TYPE_IVUS) {
      control_points = {{0.f, 0.f}, {1.f, tgc_dB_per_cm}};
    } else {
      control_points = {{0.f, 0.f}, {40.f, 28.f}};
    }
    tgc_curve_ = create_piece_wise_tgc(...);
-   tgc_probe_type_ = probe->get_probe_type();
+   // Invalidate the probe-type cache when the curve was built from user control
+   // points so that switching back to the default path on a later frame triggers a rebuild.
+   tgc_probe_type_ = user_tgc ? std::nullopt
+                              : std::optional<ProbeType>(probe->get_probe_type());
  }
```

**Diff (vs main) — log compression and median clip:**
```diff
- cuda_algorithms_->log_compression(d_scanlines.get(), plane_size, 20.f, 1e-19f, sim_params.stream);
+ cuda_algorithms_->log_compression(
+     d_scanlines.get(), plane_size,
+     sim_params.log_multiplier, sim_params.log_floor,
+     sim_params.stream);
  ...
- cuda_algorithms_->median_clip_filter(
-     d_scanlines.get(), plane_size, d_filtered.get(), 5, -60.0f, 0.0f, sim_params.stream);
+ cuda_algorithms_->median_clip_filter(
+     d_scanlines.get(), plane_size, d_filtered.get(),
+     sim_params.median_clip_size,
+     sim_params.median_clip_d_min_db, sim_params.median_clip_d_max_db,
+     sim_params.stream);
```

The CUDA kernel signatures (`log_compression`, `median_clip_filter`, `mul_row`) were already templated on the right scalar/integer types, so no kernel-side changes were needed for Pass 1.

### 11.4 Python bindings and package exports

`raysim_bindings.cpp` exposes:

- **`TgcControlPoint`** as a class with the two-argument constructor `TgcControlPoint(depth_cm, gain_db)`, read/write properties for both fields, and a `__repr__` for debug printing.
- All new `SimParams` fields via `def_readwrite` (`tgc_control_points`, `log_multiplier`, `log_floor`, `median_clip_size`, `median_clip_d_min_db`, `median_clip_d_max_db`, `scattering_resolution_mm`, `scatter_integral_scale`, `disable_scatter`).
- The `SimParams` docstring was updated to document the new knobs.

`raysim/__init__.py` re-exports `TgcControlPoint` so the canonical user-facing import is `from raysim import SimParams, TgcControlPoint, IvusSimConfig`.

### 11.5 Python config layer (`IvusSimConfig.to_sim_params`)

`raysim.config.IvusSimConfig.to_sim_params()` was extended to set every Pass 1 field. The TGC list of `(depth_cm, gain_db)` tuples in YAML is converted into a `list[rs.TgcControlPoint]` before assignment. An empty YAML list keeps the simulator on its probe-type default schedule (preserves backward compat for configs that omit `processing.tgc_control_points`). `scattering_resolution_mm` in YAML defaults to `10.0` (the historical IVUS value) and is forwarded as-is; setting it to `0.0` falls back to the C++ probe-type auto.

The book-keeping registry `_PARTIALLY_WIRED_PATHS` — which previously listed every YAML path that was in the schema but still hard-coded in C++ — has been emptied since all bucket-B knobs are now plumbed. The list is kept (with a comment) so future schema additions can be flagged before their bindings land.

Verified locally (without a CUDA build) by loading `instrument-calibration/p035_visions/volcano_s5i.yaml` through the schema and checking the diagnostic registries:

```
_PARTIALLY_WIRED_PATHS = ()
partially_wired_fields() (should be empty after Pass 1):    <empty>
pending_fields() (Future, not yet wired):
  processing.dynamic_range_db = 40.6
  processing.reject_db = -40.6
  processing.noise.sigma = 2.6347
  processing.ring_down.amplitude = 46.37
  processing.ring_down.extent_mm = 3.0
  processing.ring_down.decay = measured
  processing.ring_down.waveform_path = P_035_PointScatter/derived/ringdown/...
```

### 11.6 What's still on the to-do list

Pass 1 deliberately did not touch any new physics. **Pass 2 (§11.7) lands the ring-down injection stage and the dynamic-range / reject display window**; only `gain_db`, `compression_lut`, and `noise.{type,sigma}` remain in `_FUTURE_PATHS`. Noise is intentionally deferred until after ring-down so we can measure noise on a ring-down-subtracted simulator rather than fitting a number that conflates noise with residual catheter signal. `_FUTURE_PATHS` in `raysim/config.py` is the live source of truth for what's still pending.

### 11.7 Configuration-driven processing pipeline (Pass 2)

Pass 2 adds the first batch of new physics on top of Pass 1's plumbing: a calibrated **ring-down injector** that reproduces the catheter's near-field signature, and a **post-log display window** that reproduces the device's reject / saturation palette. Both are off by default so default `SimParams()` continues to be byte-identical to pre-Pass-1 (validated end-to-end: `max abs diff = 0` against the pre-Pass-1 build of the same example).

**Implementation pointers (Pass 2 surface area):**

- C++ surface: [include/raysim/core/raytracing_ultrasound_simulator.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp) (new public `struct RingDownParams`; `SimParams::ring_down`, `SimParams::reject_palette`, `SimParams::saturation_palette` — see §11.9 for the Pass 3b rename of the display-window knobs).
- C++ pipeline: [csrc/core/raytracing_ultrasound_simulator.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp) (stage **1.6 Ring-down injection** between TGC and envelope detection; stage **3.5 Display window** after log compression).
- CUDA helpers: [csrc/cuda/cuda_algorithms.{hpp,cu}](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu) — new `add_row` (broadcast vector add along depth) and `apply_display_window` (Pass 3b: direct palette clamp; replaces the dB-shift formulation that landed in Pass 2).
- Python bindings: [csrc/python/raysim_bindings.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/python/raysim_bindings.cpp) (`RingDownParams` class with `def_readwrite` for every field including a numpy-backed `waveform`; `SimParams.ring_down / reject_palette / saturation_palette`).
- Python config: [raysim/config.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/raysim/config.py) (`RingDownConfig` gained `enabled`, `template_pitch_mm`, `template_speckle_floor_palette`; `to_sim_params` loads the .npy template, subtracts the speckle floor, converts palette → envelope amp, and resamples to the simulator's depth-sample pitch).
- Calibrated YAML: [`instrument-calibration/p035_visions/volcano_s5i.yaml`](../../../instrument-calibration/p035_visions/volcano_s5i.yaml) (`ring_down.enabled: true`, `template_pitch_mm: 0.12`, `template_speckle_floor_palette: 45.0`).

**Where things go in `simulate()`:** ring-down is added pre-Hilbert (between TGC at step 1.5 and envelope detection at step 2). The calibrated template is stored on the host as `std::vector<float>` in envelope-amplitude units, uploaded once per change, and added row-wise via `add_row` so the existing Hilbert + log-compression handle the resulting peak shape naturally. Cache invalidation keys on (`decay`, `amplitude`, `extent_mm`, `buffer_size`, and the host pointer + size of the measured waveform) so callers that swap templates frame-to-frame get a fresh upload. The display window is applied post-log-compression and pre-median-clip (so the median clip filter still operates on a reasonable local window). Both stages skip themselves when their toggle is off (`ring_down.enabled == false` and `saturation_palette <= reject_palette`), so default callers see no change.

**Sample-pitch convention.** The OptiX raygen maps the radial range `[0, t_far_mm]` onto `buffer_size` samples (`offset = round(t / t_far * (buffer_size - 1))` in `optix_trace.cu`), so the spatial pitch in the scanlines is `t_far / buffer_size` mm/sample — *not* `c / SAMPLING_FREQ / 2`. Both the C++ ring-down stage and the Python loader's resampler use this convention so a calibration template peak at 1.8 mm in the file lands at 1.8 mm in the simulator output (verified end-to-end: median-over-angle radial profile peaks at 1.787 mm for the PV .035 calibrated template, matching the calibrated 1.80 mm). The pre-existing `create_piece_wise_tgc` path uses the legacy `SAMPLING_FREQ`-based convention for its own depth → sample math and is left alone (a separate alignment cleanup).

**Acceptance check (PV .035 calibrated YAML against the cylinder phantom):**

| metric                                | expected (calibration / device)             | observed                                  |
|---------------------------------------|---------------------------------------------|-------------------------------------------|
| Inner ~3 mm peak depth (ring_down ON) | ~1.80 mm (E6, calibration_delta.md)         | **1.787 mm** (peak palette 227.9)         |
| Frame palette range (display window)  | reject ≈ 11, saturation ≈ 239 (E7)          | **[~11, 227.9]** with calibrated dr/reject |
| `default SimParams()` regression      | bit-identical to Pass 1                     | **max abs diff = 0** (vs Pass 1 reference) |

**What's intentionally not in Pass 2:** noise (deferred per the calibration plan — measurable only on a ring-down-subtracted simulator), `gain_db` (kept informational; per-frame gain is applied by scaling the ring-down `amplitude` and the speckle calibration externally), `compression_lut` (no calibrated LUT yet — full E7 sweep needed). All three remain in `_FUTURE_PATHS`.

### 11.8 Log-compression: fixed-reference mapping (Pass 3a → Pass 3b / K2v2)

Tier 1 evaluation (test G in `instrument-calibration/p035_visions/tier1_results/tier1_results.md`) surfaced a structural divergence between the calibration sheet and the simulator's `log_compression_kernel`:

* The sheet defines `pixel = log_multiplier · log10(amp / log_floor)` — an **absolute, fixed-reference** mapping between envelope amplitude and palette.
* The legacy kernel computed `pixel = log_multiplier · log10(max(amp, log_floor) / per_frame_quantile)` where `per_frame_quantile` was the per-frame 99.999 %-quantile of the envelope buffer. This made absolute palette values **frame-dependent** (every frame's brightest pixel landed at palette 0 regardless of absolute amplitude) and broke the round-trip with the calibration sheet for any non-degenerate scene.

**Pass 3a (K2)** removed the per-frame quantile and used `log10(max(amp, log_floor)) · log_multiplier`. **Pass 3b (K2v2)** further switches to the spec form so `amp == log_floor` lands at palette 0 (instead of palette `log_multiplier · log10(log_floor)`), and so `amp < log_floor` produces *negative* palette values that the display-window stage can clamp to the device's reject palette:

```diff
- buffer[offset] = log10f(max(buffer[offset], minimum)) * mutliplicator;
+ const float floor_safe = fmaxf(minimum, 1e-30f);
+ const float amp_safe   = fmaxf(buffer[offset], 1e-30f * floor_safe);
+ buffer[offset] = log10f(amp_safe / floor_safe) * mutliplicator;
```

The two epsilon clamps make `amp == 0` / `log_floor == 0` produce a finite, very-negative palette value (~`-30 · log_multiplier`) instead of NaN/-inf. The corresponding C++ caller (`CUDAAlgorithms::log_compression`) drops the `cub::DeviceRadixSort` reduction and the `log_compression_sorted_` / `temp_log_compression_` scratch buffers (Pass 3a). The host-side `<cub/cub.cuh>` include is also no longer needed by `cuda_algorithms.cu`.

**Default change (Pass 3b).** `SimParams::log_floor` default changes from `1e-19f` to `1.f`. With `log_multiplier == 20` (default) this puts the post-log palette in roughly `[-60, 0]` for envelope amplitudes in `[1e-3, 1]` — the same range the existing `examples/ivus_example.py` `MIN_VAL/MAX_VAL` window assumes. Default callers that constructed `SimParams()` see a palette shift of `+log_multiplier · log10(1e-19) = -380` cancelled out by the new default, so their displayed images look the same as before Pass 3a.

**Tier 1 acceptance after K2v2:** test G (log-compression mapping) passes by construction (kernel now mirrors the spec exactly). The display-window rewrite (Pass 3b, §11.9) and the calibrated `gain_db` stage (Pass 3a, §11.9.1) close out the rest of the gain-alignment story.

### 11.9 Display window + reference gain (Pass 3b)

Pass 3b lands two coupled fixes that together let the calibrated PV .035 YAML reproduce the device's reject behaviour and put the simulator's envelope amplitudes onto the bench's reference scale at slider 54.

#### 11.9.1 Reference gain stage (`gain_db`)

A new `SimParams::gain_db` is applied **between TGC (stage 1.5) and ring-down injection (stage 1.6)** as `rf ← rf · 10^(gain_db / 20)`. The CUDA wrapper short-circuits on `gain_db == 0` so default callers pay no kernel-launch cost.

The pre-Hilbert location is deliberate — `gain_db` has to scale the raytraced RF up to bench-calibrated amplitude *before* ring-down is added, because ring-down's `amplitude` field is specified in bench-calibrated envelope units. Putting `gain_db` after ring-down would double-scale the ring-down by `10^(gain_db/20) = 10^7.86 ≈ 7×10^7`, blowing out the inner ring and cascading saturation through the lateral PSF into the rest of the image. By linearity of the Hilbert transform (`|H(s·rf)| = s·|H(rf)|`), scaling RF pre-Hilbert is mathematically equivalent to scaling envelope post-Hilbert for the scattering signal, so the pure-amplitude derivation in `derive_gain_db.py` is unchanged by the move.

`gain_db` lumps two physically distinct effects into one calibrated scalar:

1. The bench's **slider-gain offset** (the device's gain control: slider 54 maps to 0 dB by convention; a 10-step change is ±10 dB on the bench gain LUT).
2. A **renderer-specific reference-amplitude offset** — the simulator's raw envelope amplitudes are not on the same linear scale as the bench's calibrated amplitudes. The calibration sheet treats the bench's "amp at slider 54" as the reference (in arbitrary linear units), so this offset is the constant that puts the simulator's output onto that scale. See `instrument-calibration/p035_visions/derive_gain_db.py` for the analytical derivation.

The PV .035 YAML calibrates `gain_db` against the **bench water-scatter background** (palette 46.2 at slider 54). `derive_gain_db.py` runs in two passes:

1. *Pure-amplitude seed.* Render the wire phantom in raw-envelope mode (`log_floor=1.0, log_multiplier=1.0, ring-down OFF, display window OFF, gain_db=0`) so the post-log palette equals `log10(envelope_amp)`. The geometric-mean water-bg amplitude is `sim_bg_amp = 10^(mean(log10(amp_i)))`; the bench reference is `bench_bg_amp = 10^(46.2 / log_multiplier)`; the seed is `gain_db_seed = 20·log10(bench_bg_amp / sim_bg_amp) = +157.20 dB`.
2. *Bisection refinement.* The seed systematically over-shoots because the simulator's rendered bg distribution is wider than the bench's (Rayleigh log10 std ≈ 31 palette vs the bench's narrower-than-Rayleigh 20.3 palette). The seeded mean palette would be 46 if no clipping occurred; in the calibrated render the long left tail of `log10(amp)` crashes through `reject_palette = 11` and the clamp lifts the post-clamp mean by ~33 palette. The script bisects `gain_db` against the *post-pipeline post-clamp mean palette* of an anechoic lumen render until the rendered bg matches the bench reference within 0.5 palette. For PV .035 this lands at **`gain_db = +132.83 dB`**.

Calibrating against the background (rather than the wire peaks) is deliberate: the bench's wire-vs-bg amplitude contrast (~26 dB) is much smaller than the simulator's (~100 dB on the current renderer), so a single `gain_db` scalar cannot align both. Anchoring at the background preserves the device's reject-palette behaviour; the wire echoes will then saturate at `saturation_palette = 239`, which matches how the bench frames already render their inner wires (palette ≥ 220 on the device's display) but over-saturates the bench's outer wires. Closing that contrast gap is a scattering-strength problem in the OptiX pipeline — orthogonal to the gain/log/display-window calibration and tracked in `instrument-calibration/p035_visions/calibration_delta.md`.

#### 11.9.2 Display window: direct palette clamp

The Pass 2 display window had two bugs:

* It used dB-space anchors (`reject_db`, `dynamic_range_db`) and remapped `[reject_db, reject_db + dr_db]` onto `[0, log_multiplier · dr_db / 20]`. With K2v2's *negative* palette outputs (which represent `amp < log_floor`), the kernel's input pixel of 0 was shifted *up* to the saturation ceiling instead of clamped *down* to the reject floor — exactly the opposite of the device's behaviour. So everything below the reject window came out white instead of black.
* The re-zeroing step meant the device's documented reject palette (e.g. 11 for the PV .035) was never actually displayed; the simulator's reject floor was always palette 0.

Pass 3b replaces the dB-shift formulation with a **direct palette clamp**:

```cpp
buffer[offset] = fminf(fmaxf(buffer[offset], reject_palette), saturation_palette);
```

`SimParams::reject_palette` and `SimParams::saturation_palette` replace the old `reject_db` / `dynamic_range_db` knobs. With both at 0 (default) the display-window stage is skipped entirely. The PV .035 YAML uses the values straight out of `gain_lut.json` (`reject_palette: 11.0`, `saturation_palette: 239.0`), so the device's reject floor and saturation ceiling reproduce by construction.

**Tier 1 acceptance after Pass 3b:** the Tier 1 evaluation script (`instrument-calibration/p035_visions/tier1_evaluation.py`) reports five PASSes (configuration round-trip A, configuration self-consistency B, log-compression mapping G with 0.0 palette error, TGC schedule H with 0.0 dB error, and the new gain-alignment diagnostic with bg landing at palette 47.3 vs the bench reference 46.2 — Δ = +0.19 dB). The remaining FAILs (axial PSF C, lateral PSF D, ring-down RMS E) are all attributable to the wire-vs-bg contrast gap: every wire saturates at `saturation_palette = 239`, so the −6 dB FWHM measurement is undefined and the PSF-ringdown shape RMS is dominated by saturation rather than ringdown shape. Test F (noise floor σ) remains N/A pending the additive-noise wiring deferred to Pass 4. The renderer-vs-bench wire-vs-bg contrast gap (~74 dB) is tracked separately in `instrument-calibration/p035_visions/calibration_delta.md` as the next blocking issue for closing C/D/E.

### 11.10 Ring-down post-Hilbert + depth-uniformity test (Pass 4)

Pass 3b shipped a calibrated PV .035 build that visually showed a "bright centre, dark middle, bright outer" radial pattern in anechoic regions. Pass 4 adds a quantitative diagnostic for that pattern (Tier 1 test I — depth uniformity in anechoic ROI) and uses it to root-cause two distinct effects:

1. **Mysterious far-field rise (r ≈ 20 → 29 mm).** The simulator's anechoic mean palette grew by ~+50 from r ≈ 20 mm to the buffer edge at r ≈ 29 mm, with the rise correlated with `ring_down.enabled = true` even though `extent_mm = 3.0`. Diagnostic with shared scatter texture isolated it: cuFFTDx's Hilbert is a length-N (= 4096) cyclic FFT, so any inner-zone transient gets smeared across the entire buffer via spectral side lobes. Predicted leakage envelope at sample 4000 (computed offline by FFT-Hilberting a zero-padded copy of the ring-down template): ~4.1 envelope-amp, equivalent to ~+68 palette at the calibrated `log_multiplier = 112.3`. Observed in-pipeline excess matched within a few palette.

   **Fix.** Move the ring-down add stage to **post-Hilbert** (`raytracing_ultrasound_simulator.cpp` §2.5, between envelope detection and log compression). The bench template is already in envelope-amp units (the YAML loader applies `amp = 10^(palette/log_mult) - 1` with the speckle-floor subtraction), so adding it to the envelope buffer is the literal mathematical operation we want — "the catheter contributes this envelope on top of the scattering envelope". The cuFFTDx Hilbert now sees only the broadband scatter signal and there is no transient to smear. The ring-down `extent_mm` truncation is honoured exactly (samples past `extent_samples` are untouched).

   Knock-on: the bisection in `derive_gain_db.py` was using a calibrated render that previously included the Hilbert-leakage bg lift; with that lift gone, the calibrated `gain_db` had to grow by **+12.19 dB** (132.83 → 145.02) to keep the post-clamp anechoic mean palette at the bench's 46.2 reference. Test I drops from RMS = 31.4 → 21.4 → ~30 palette across the chain (the third number is after the gain re-fit; max |Δ| now sits in the bright shoulder at r ≈ 4-7 mm rather than the spurious far-field rise).

2. **Bright shoulder at r ≈ 4-7 mm — root-caused but not yet fixed (Pass 5).** Pure-scatter renders (ring-down OFF, scatter ON) show a near-field palette peak (mean ≈ 100-200, vs bench ~33-44) that the bench does not. A controlled diagnostic comparing 2D-depth-dependent vs 1D-constant lateral PSF (probe `element_radius=0` falls back to 1D const) showed the **2D depth-dependent lateral PSF is the source**:

   - The 1D-constant lateral PSF gives mean palette ≈ 18-45 across all depths — consistent with the bench's 33-44 floor.
   - The 2D depth-dependent PSF (`update_psfs` Gaussian-beam model) produces wild depth-dependent oscillation (palette 14 → 200 over ~1 mm intervals in r ∈ [0, 7] mm) that converges to the 1D fallback past r ≈ 12 mm.

   Three sub-issues identified in the implementation (`csrc/cuda/cuda_algorithms.cu` `convolve_columns_depth_dependent_kernel` and `csrc/core/raytracing_ultrasound_simulator.cpp` `update_psfs`):

   1. **Non-cyclic angular convolution.** IVUS angles wrap 360° but the convolution truncates at `index.y = 0` and `index.y = num_scanlines - 1` (lines 106-107). Should be cyclic (`(index.y + k + size.y) % size.y`).
   2. **`kernel_radius = 64` too small for near-field beam.** At r = 1 mm the Gaussian σ = 113 angular bins (wider than the 64-bin half-window), so the kernel is severely truncated and the sum=1 normalization fails to renormalize against the lost mass.
   3. **L1-normalized kernel makes envelope amplitude depth-dependent.** For random scatter, the envelope mean of a kernel-convolved zero-mean RF is proportional to the kernel's L2 norm. With sum=1 normalization, L2 ∝ 1/√σ → small-σ depths see disproportionately larger envelope amplitude than large-σ depths. Should normalize by L2 to make envelope amp depth-invariant for random-bg scatter.

   Pass 5 should fix all three (one PR, requires re-running `derive_gain_db.py` since the bg statistics will shift again).

**Tier 1 test I — depth uniformity in anechoic ROI.** New diagnostic in `tier1_evaluation.py` that:
- Renders N anechoic frames with the calibrated YAML (ring-down ON since that's the deployed config).
- Computes per-radius mean / median / std palette across angles + frames.
- Reads bench polar frames (gain 54, D = 60 mm, 5 frames) and masks wires by per-radius p70 clip; takes mean across frames.
- Compares sim vs bench in the band `r ∈ [max(extent_mm + 1, 4), min(0.97 · t_far_mm, 29)]` mm.
- Pass criterion: RMS(sim − bench) ≤ 10 palette **AND** sim peak-to-trough span ≤ 1.5 × bench span.
- Saves figure `tier1_results/figures/depth_uniformity.png` and persists per-radius profiles to `tier1_results/arrays/`.

After Pass 4 with the re-derived `gain_db = +145.02`: bias is essentially zero (sim mean tracks bench mean within ±10 palette over r ∈ [10, 29] mm), but the bright shoulder at r ≈ 4-7 mm dominates the RMS (max |Δ| ≈ 100-120 palette there). Test I FAILS until the near-field scatter peak and the missing additive noise (test F) are addressed in Pass 5.

---

## 12. Potential next steps and modeling gaps

The following are **missing elements** that could explain mismatches between simulation and real IVUS, plus **suggested next steps** to enhance the model.

### 12.1 Frequency dependence of scattering

Scattering strength is modulated by material σ and attenuation along the path, but there is **no explicit frequency dependence** (e.g. f⁴ for Rayleigh). Changing center frequency changes attenuation and beam width but not the inherent scattering strength vs frequency. That can distort relative speckle vs frequency when comparing 20 vs 40 MHz or when matching to real IVUS.

**Next step:** Add a frequency-dependent scattering term (e.g. σ(f) ∝ f⁴ for Rayleigh, or a material-level exponent) so that scatter contribution scales correctly with probe frequency.

### 12.2 Catheter / ring-down

There is **no model of the catheter or sheath**: no near-field ring-down, guided waves, or fixed echo from the housing. Real IVUS has a dead zone and strong echo from the catheter; its absence can make the simulated lumen look “too clean” near the probe.

**Status (Pass 2 — implemented):** The ring-down injector in `simulate()` (stage 1.6, between TGC and envelope detection) now consumes the calibrated `processing.ring_down.{enabled, amplitude, extent_mm, decay, waveform_path, template_pitch_mm, template_speckle_floor_palette}` block. The `IvusSimConfig` loader resolves the `.npy` template relative to the workspace root, subtracts the speckle floor, converts palette → envelope amplitude using `log_multiplier`, and resamples from the device's display pitch (0.12 mm/sample for PV .035) onto the simulator's `t_far / buffer_size` pitch. With `enabled=false` the simulator emits no ring-down signal at all (silent lumen); with `enabled=true` it adds the residual that survives the device's Acoustic Reference subtraction. See §11.7 for the full implementation table and the acceptance numbers (peak at 1.787 mm vs the calibrated 1.80 mm).

**Outstanding for ring-down:** the `subtract_reference` field is informational only (the device already does the AR subtraction; we model the residual). A simple dead-zone mask for the inner few hundred microns is not yet wired and may not be needed once the calibrated waveform template is doing the work; revisit if the inner-most ~0.2 mm shows residual artifacts in the comparison frames.

### 12.3 Element directivity at transmit

Ray intensity starts at 1.0; **element directivity is only applied in the lateral PSF** (receive-side blur). Transmit directivity (e.g. angular sensitivity of the single element) is not applied to the ray weights. For a rotating single element this can affect angular uniformity of sensitivity.

**Next step:** Apply an angular weighting (e.g. from element size and frequency) to the **transmit** ray contribution (e.g. in the raytracing or in the RF accumulation) so that both transmit and receive directivity are represented.

### 12.4 Electronic / thermal noise

There is **no noise model**. For SNR, contrast resolution, or detector-limited studies, at least a simple noise model is needed (e.g. additive Gaussian, or noise figure).

**Schema status (Pass 1):** The YAML schema carries `processing.noise.{type, sigma}` (Gaussian / Rayleigh / none) and the calibration pipeline measures a baseline σ in palette units from anechoic ROIs. These are listed in `pending_fields()` because the C++ side is not yet implemented; per the Pass-2 plan, noise is intentionally deferred until after ring-down so that we can measure noise on a ring-down-subtracted simulator output rather than fitting a number that conflates noise with residual catheter signal.

**Next step (Pass 2 follow-up):** Add an optional noise stage (e.g. post–log-compression Gaussian, or pre-compression with a simple noise figure) honoring the YAML schema fields above.

### 12.5 Rotation and motion

The simulation is **“all angles at once”** (full 360° in one frame). Real IVUS uses a rotating element; rotation blur and motion artifacts are not represented. Acceptable for static phantoms; relevant for moving vessels or pullback validation.

**Next step:** Optionally model a finite rotation window per frame (e.g. angular sector and integration time) or add a simple motion-blur kernel for pullback studies.

### 12.6 Speed-of-sound heterogeneity and refraction

Refraction at **interfaces** is correct, but the ray is **straight between interfaces**. In reality, smooth variations in c would bend rays. For small vessels and relatively uniform lumen/wall, this is often a second-order effect.

**Next step:** For tissue with spatially varying c, consider ray bending (e.g. ray tracing in a graded index or layered c) if validation targets require it.

### 12.7 Multiple scattering

Only **single scattering** is modeled in the scatter integral. In dense, heterogeneous tissue, multiple scattering can affect speckle and attenuation; that’s a known limitation of ray-based methods.

**Next step:** Document as a known limitation; if needed for specific studies, consider hybrid or post-hoc corrections (e.g. extra attenuation term) rather than full multi-scatter raytracing.

### 12.8 Additional gaps (summary)

- **Near-field / beam formation:** The beam is represented via the lateral PSF and t_far; explicit near-field (Fresnel) beam evolution is not modeled. Fine for many validation cases; relevant if focal behavior or very short ranges are critical.
- **Angle-dependent reflection:** Reflection uses an intensity coefficient; full angular dependence (e.g. obliquity factor, mode conversion) is not included. Can matter for steep angles and shear waves.
- **System transfer function / calibration:** Real systems have gain curves, digitization, and bandpass; the sim assumes an ideal chain. For pixel-level matching to a specific scanner, a system TF or calibration step would help.
- **Reverberation and multipath:** Ringing in layers (e.g. wall–catheter–wall) and multipath are not modeled; they can add clutter in real IVUS.

Incorporating the items in §12.1–12.4 would address the most visible gaps (frequency scaling, catheter clutter, transmit directivity, noise); §12.5–12.8 are secondary for static phantom validation but matter for realism and clinical comparison.

---

## 13. Summary of main implementation files changed/added

| Area                                  | Files (core implementation) |
|---------------------------------------|-----------------------------|
| Probe type                            | [probe_types.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/probe_types.hpp), [probe.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/probe.hpp), [ivus_probe.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/core/ivus_probe.hpp) (new) |
| Materials                             | [material.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/core/material.cpp) |
| Ray / scattering                      | [optix_trace.cu](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/optix_trace.cu), [optix_trace.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/include/raysim/cuda/optix_trace.hpp) (Pass 1 header fix, §11.1) |
| Simulator / PSF                       | [raytracing_ultrasound_simulator.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/core/raytracing_ultrasound_simulator.cpp), [raytracing_ultrasound_simulator.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/include/raysim/core/raytracing_ultrasound_simulator.hpp) (Pass 1 `TgcControlPoint`, extended `SimParams`, §11.2–11.3) |
| CUDA algorithms                       | [cuda_algorithms.cu](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/csrc/cuda/cuda_algorithms.cu), [cuda_algorithms.hpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/include/raysim/cuda/cuda_algorithms.hpp) |
| Python bindings & exports             | [raysim_bindings.cpp](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/csrc/python/raysim_bindings.cpp), [raysim/__init__.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/raysim/__init__.py), [raysim/cuda/__init__.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/raysim/cuda/__init__.py) |
| Python config schema (Pass 1, §11.5)  | [raysim/config.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/main/ultrasound-raytracing/raysim/config.py) |
| Utils                                 | [phantom_maker.py](https://github.com/mosaicintelligence/i4h-sensor-simulation/blob/3a00920723c7821b83b3fb6b400b006dbbc84e96/ultrasound-raytracing/utils/phantom_maker.py) |
| Per-instrument calibration (consumer) | [`instrument-calibration/p035_visions/volcano_s5i.yaml`](../../instrument-calibration/p035_visions/volcano_s5i.yaml) |

Example and evaluation scripts (e.g. `ivus_example.py`, `ivus_evaluation.py`, `wire_phantom_evaluation.py`, `cystic_resolution_phantom_evaluation.py`) and the comparison doc **`docs/ivus_rotating_single_element_psf_comparison.md`** are not described step-by-step here, as requested.
