# Foundations of Ultrasound Simulation

This document teaches the core ideas behind the ultrasound raytracing simulator for an audience with strong math and programming background who are new to ultrasound and physics-based simulation. We first build **intuition** in plain language, then connect those ideas to **mathematics and code** in this project.

---

## Part 1: Intuition — What Are We Simulating?

### 1.1 The Big Picture

**Ultrasound imaging** works like this: a probe sends high-frequency sound into the body; the sound bounces off tissues and some of it returns to the probe. The machine measures “how much came back” and “when,” and turns that into a 2D image where brightness means “echo strength” and position means “where in the slice.”

A **simulator** tries to reproduce that process: given a virtual world (phantoms, organs, materials) and a virtual probe, it computes what the received signal would be, then turns it into an image that looks like real ultrasound.

So the core question is: **how does sound interact with the virtual scene, and how do we turn that into an image?**

### 1.2 From Waves to Rays (The Main Approximation)

Real ultrasound is a **wave**: it spreads, diffracts, and interferes. Simulating the full wave equation is expensive. This project uses a **ray** model instead.

**Intuition:** Think of the wavefront as a surface moving outward. At each point, the direction of motion is perpendicular to the wavefront. A **ray** is that direction of travel. So instead of tracking the whole wave, we track “beams” that move in straight lines and carry energy. When a ray hits a boundary between two tissues, we split it into a reflected ray and a refracted (transmitted) ray, and we scale their energy using acoustic reflection/transmission laws. Along the way, we reduce the ray’s energy with distance (attenuation) and add small contributions from scattering (speckle).

**Why this is reasonable:** After beamforming, each “scanline” effectively has a main propagation direction. Modeling that as a ray gives a good trade-off between physical fidelity and speed. Diffraction and fine interference are not in the ray model; we approximate their effect later with a **Point Spread Function (PSF)** convolution so the image has realistic resolution and blur.

### 1.3 What the Simulator Needs

To produce one frame, the simulator needs:

1. **A world** — What’s in the scene? (background material, spheres, meshes.)
2. **Materials** — For each tissue: how fast does sound travel, how much bounces at interfaces, how much is lost with distance, and how much is scattered?
3. **A probe** — Where is it, how is it oriented, and what’s its geometry (linear array, curvilinear, phased array, IVUS, etc.)?
4. **Simulation parameters** — How deep do we trace, how many samples per ray, do we apply PSF/TGC, etc.?

The pipeline is: **ray generation → ray tracing (hits, reflection/refraction, attenuation, scattering) → raw “scanline” data → PSF → TGC → envelope detection → log compression → scan conversion → B-mode image.**

---

## Part 2: Core Concepts in Plain Language

### 2.1 Acoustic Impedance and Reflection

**Intuition:** When sound hits a boundary between two materials (e.g., soft tissue and bone), part of the energy is reflected back and part is transmitted. The “mismatch” between the two materials determines how much reflects. That mismatch is captured by **acoustic impedance** \(Z = \rho c\) (density × speed of sound). Big difference in \(Z\) → strong echo (e.g., bone); similar \(Z\) → weak echo (e.g., muscle and liver).

So: **brightness of an interface in the image is driven by the impedance contrast**, not by the absolute impedance. The simulator assigns each object a material; at each hit it looks up both materials’ impedances and computes a reflection coefficient.

### 2.2 Attenuation (Beer–Lambert)

**Intuition:** As the wave travels through tissue, it loses energy (absorption, scattering). So the same reflector deeper in the body returns a weaker echo. We model this by multiplying the ray’s intensity by an exponential decay with distance. The decay rate depends on the material and on **frequency**: higher frequency is attenuated more. That’s why we use lower frequencies for deep imaging.

### 2.3 Scattering and Speckle

**Intuition:** Tissues aren’t just smooth mirrors. They contain many small structures (much smaller than the wavelength) that send a little energy back in many directions. The sum of many such tiny echoes produces the grainy “speckle” texture typical of ultrasound. In the simulator we don’t place millions of scatterers; we use a **3D texture** that, when sampled along the ray, gives a scattering strength. Combined with material parameters (how often and how strongly the tissue scatters), we add small contributions into the scanline as the ray travels. So speckle emerges from the ray path and material properties.

### 2.4 Reflection and Refraction Geometry

**Intuition:** At a smooth interface, the reflected ray obeys “angle of incidence = angle of reflection.” The transmitted ray bends (refraction) according to the ratio of sound speeds in the two media (Snell’s law). If the second medium is “slower,” the refracted ray can bend so much that all energy reflects (total internal reflection). The simulator implements reflection direction, Snell’s law for refraction, and total internal reflection.

### 2.5 Probes and Scanlines

**Intuition:** Different probes produce different image shapes because they emit rays in different patterns. A **linear array** sends many parallel rays (rectangular image). A **curvilinear** probe has elements on a curved surface, so rays diverge (sector image). A **phased array** uses a small footprint and steers rays in different angles (sector). **IVUS** rotates a single element to get a circular cross-section. In the simulator, “one ray per element (and per elevational sample)” is generated; each ray fills one column (or angular line) of raw data. **Scan conversion** later maps that acquisition geometry (e.g., polar or sector) to a rectangular display image.

### 2.6 From Raw Hits to B-Mode Image

**Intuition:** The ray tracer writes **echo amplitudes** into buffers indexed by ray index and depth (time). That’s “RF-like” data: one column per ray. Real systems then:

- **Blur** the data with the system’s resolution (we do this with a **PSF** convolution).
- **Boost** deeper echoes to compensate for attenuation (**Time Gain Compensation**, TGC).
- **Envelope detection** — take the “amplitude” of the oscillating signal (e.g., via Hilbert transform).
- **Log compress** — map the huge dynamic range to a displayable range (e.g., 20 log₁₀).
- **Scan convert** — resample from (ray index, depth) to (x, z) or (r, θ) for display.

The simulator follows this same chain so the final image looks like clinical ultrasound.

---

## Part 3: Mathematics and Implementation

Here we state the main formulas and point to where they appear in the codebase.

### 3.1 Acoustic Impedance and Reflection Coefficient

**Definitions:**

- **Acoustic impedance:** \(Z = \rho c\) (kg/m²·s). In code, often in MRayl (\(10^6\) kg/m²·s).
- **Reflection coefficient (power):** The fraction of incident intensity that is reflected. For normal incidence the classic form is \(R = \left(\frac{Z_2 - Z_1}{Z_2 + Z_1}\right)^2\). For oblique incidence we use the angle-dependent form.

**Implementation (oblique incidence):**  
In `optix_trace.cu`, `calculate_reflection_coefficient()` uses the acoustic form with cosine of the incident angle:

\[
R = \left( \frac{Z_2 \cos\theta - Z_1}{Z_2 \cos\theta + Z_1} \right)^2
\]

- \(Z_1, Z_2\): impedances of the two materials (incident and transmitted side).
- \(\theta\): angle of incidence (between ray and surface normal).

So the reflected intensity is scaled by \(R\); the transmitted (refracted) ray is scaled by \(1 - R\) (for energy conservation in the model).

**Code:** `csrc/cuda/optix_trace.cu` — `calculate_reflection_coefficient()`, and the hit shader that multiplies specular intensity by this factor.

### 3.2 Attenuation (Beer–Lambert)

**Model:** Intensity decreases exponentially with path length. In dB form, attenuation over distance \(d\) is \(\alpha f d\) (with \(d\) in cm, \(f\) in MHz, \(\alpha\) in dB/(cm·MHz)). In linear scale:

\[
I(d) = I_0 \cdot 10^{-\alpha f d / 20}
\]

**Implementation:**  
`get_intensity_at_distance(distance, medium_attenuation)` in `optix_trace.cu`:

- `params.source_frequency` is \(f\) (MHz).
- Distance is converted to cm; then attenuation in dB is \(\alpha \cdot f \cdot d_{\text{cm}}\).
- The factor `0.05f` in the exponent corresponds to \(1/20\) (dB to linear: \(10^{-\text{dB}/20}\)).

So as a ray travels, its intensity is multiplied by this factor; the same factor is used when adding scattering contributions (attenuation from probe to scatterer and back is approximated in the model).

**Code:** `csrc/cuda/optix_trace.cu` — `get_intensity_at_distance()`.

### 3.3 Scattering (Texture + Material)

**Model:** Scattering strength at a point is given by sampling a 3D texture, then applying material parameters. The texture has two channels: one for “density” (whether to scatter) and one for amplitude. Material parameters:

- **mu0:** threshold on the first channel (scatter only when texture value ≤ mu0).
- **sigma:** scale factor for the scattered amplitude.

So: `scatter_value = (texture_channel_2 * sigma)` when `texture_channel_1 <= mu0`, else 0.

**Implementation:**  
`get_scattering_value(pos, material)` in `optix_trace.cu` converts world position to texture coordinates (using `params.scattering_resolution_mm`), samples the 3D texture, then applies `mu0_` and `sigma_`. Contributions are accumulated in `sample_intensities()` along the ray, scaled by current ray intensity and by `get_intensity_at_distance()` for the segment.

**Code:** `csrc/cuda/optix_trace.cu` — `get_scattering_value()`, `sample_intensities()`; texture generation in `world.cpp` — `generate_scattering_texture()`.

### 3.4 Reflection and Refraction Directions

**Reflection:** \(\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}\), with \(\mathbf{n}\) pointing to the incident side (so the ray goes “back” from the surface).  
**Code:** `calc_reflected_dir()` in `optix_trace.cu`.

**Refraction (Snell’s law):** \(\frac{\sin\theta_1}{\sin\theta_2} = \frac{c_1}{c_2}\). The refracted direction is computed from the incident direction, the normal, and the two speeds of sound. If \(\sin\theta_2 \ge 1\), total internal reflection: no transmitted ray.  
**Code:** `calc_refracted_dir()` in `optix_trace.cu`.

**Specular intensity (directional weighting):** The hit shader also uses a specularity parameter and the angles between reflected/refracted directions and the direction back to the probe to weight the reflected intensity (Mattausch-style term).  
**Code:** `calculate_specular_intensity()` in `optix_trace.cu`.

### 3.5 Ray Generation (Probe Geometry)

Rays are generated in **local** probe coordinates then transformed to world space by the probe’s pose.

- **Linear array:** Origin at \((x, 0, 0)\) along the array width, direction \((0, 0, 1)\).
- **Curvilinear:** Origin on an arc: e.g. \((R\sin\theta, y, R(\cos\theta - 1))\), direction normal to the arc.
- **Phased array:** Origin near \((0, y, 0)\), direction \((\sin\theta, 0, \cos\theta)\) for steering angle \(\theta\).
- **IVUS:** Single rotating element; geometry and possibly depth-dependent lateral PSF are probe-specific.

Elevational dimension: multiple samples in \(y\) over the element height; after PSF convolution in elevation, planes are averaged to form the 2D image.

**Code:** `csrc/cuda/optix_trace.cu` — raygen and probe-specific functions (e.g. `generate_linear_array_probe_ray_local`, `generate_curvilinear_probe_ray_local`); `include/raysim/core/probe.hpp` and probe-specific headers for element positions and directions.

### 3.6 Depth (Time) Indexing

Raw data are stored per ray and per “depth bin.” The mapping from ray distance \(t\) to bin index is:

\[
\text{offset} = \left\lfloor \frac{t}{t_{\text{far}}} \cdot (N_{\text{buffer}} - 1) + 0.5 \right\rfloor
\]

where \(t_{\text{far}}\) is the maximum ray length and \(N_{\text{buffer}}\) is `buffer_size`. So \(t\) is proportional to time-of-flight and thus to depth.

**Code:** `get_intensity_offset()` in `optix_trace.cu`; `SimParams::buffer_size`, `SimParams::t_far` in `raytracing_ultrasound_simulator.hpp`.

### 3.7 PSF Convolution

The Point Spread Function models finite resolution (axial and lateral, and elevational if used). The axial kernel is typically a Gaussian modulated by a cosine (carrier at the center frequency); lateral (and elevational) kernels are Gaussians. Convolution is applied to the raw scanline data so that point reflectors and edges appear blurred as in real systems.

**Code:** `raytracing_ultrasound_simulator.cpp` — `create_gaussian_psf()`, `update_psfs()`, and the convolution step in `simulate()`; for IVUS, a causal axial PSF and depth-dependent lateral PSF are available.

### 3.8 TGC, Envelope, Log Compression, Scan Conversion

- **TGC:** Piecewise-linear gain curve in dB as a function of depth, converted to linear and applied per depth bin.  
  **Code:** `create_piece_wise_tgc()`, and in `simulate()` the application of `tgc_curve_` via `cuda_algorithms_->mul_row()`.

- **Envelope:** Hilbert transform (or equivalent) to get the analytic signal; then magnitude.  
  **Code:** `cuda_algorithms_->envelope_detection()` (called from `simulate()`).

- **Log compression:** e.g. \(20\log_{10}(\text{value})\) with clipping and optional normalization.  
  **Code:** `cuda_algorithms_->log_compression()`.

- **Scan conversion:** Resample from (ray index, depth) to (x, z) or (r, θ) depending on probe type (linear, curvilinear, phased, IVUS).  
  **Code:** `cuda_algorithms_->scan_convert_linear()`, `scan_convert_curvilinear()`, `scan_convert_phased()`, `scan_convert_ivus()` in `cuda_algorithms.cu` and used from `simulate()`.

### 3.9 Material and World

**Materials:** Each material has: `impedance_`, `attenuation_`, `speed_of_sound_`, `mu0_`, `mu1_`, `sigma_`, `specularity_`. Defined in `include/raysim/core/material.hpp` and initialized (e.g. water, fat, liver, muscle, bone) in `material.cpp`. The registry `Materials` stores them and uploads to GPU for the ray tracer.

**World:** Holds a list of `Hitable` objects (e.g. spheres, meshes), each with a material index; a background material name; and a 3D scattering texture. `World::build()` builds the OptiX acceleration structure and hit groups with material indices.  
**Code:** `include/raysim/core/world.hpp`, `csrc/core/world.cpp`.

---

## Part 4: How It Fits Together (Pipeline Summary)

1. **Setup:** World (geometry + materials), Materials registry, Probe, SimParams.
2. **Ray generation:** For each pixel of the launch grid (element × elevational sample), compute ray origin and direction in world space; pass to OptiX.
3. **Ray tracing (OptiX):** For each ray, find hits; at each hit, compute reflection/refraction, update intensity, write specular contribution to the scanline, spawn reflected/refracted rays up to max depth; along segments, sample scattering and add to scanline with attenuation.
4. **Post-process (CPU/GPU):** Optional elevational PSF and averaging; PSF convolution (axial, lateral, elevational); TGC; envelope detection; log compression; scan conversion.
5. **Output:** B-mode image (e.g. float per pixel, in dB) and optional RF data.

This pipeline is implemented in `RaytracingUltrasoundSimulator::simulate()` in `raytracing_ultrasound_simulator.cpp`, with the core ray logic in `optix_trace.cu` and the signal chain in `cuda_algorithms.cu`.

---

## Further Reading

- **Quick start (5 minutes):** [Quick Start Guide](../ultrasound-raytracing/docs/quick_start.md)
- **Hands-on tutorial:** [Ultrasound Simulator Tutorial](ultrasound_simulator_tutorial.md)
- **Technical reference (physics and implementation):** [Ultrasound Simulator Technical Guide](ultrasound_simulator_technical_guide.md).
- **Code:** `ultrasound-raytracing/` — `include/raysim/`, `csrc/core/`, `csrc/cuda/`, and `examples/` for usage patterns.
