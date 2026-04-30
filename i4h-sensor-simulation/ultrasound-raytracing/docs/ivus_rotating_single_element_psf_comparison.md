# IVUS: Comparison to `rotatingSingleElm_psf.ipynb`

This document compares the current raytracing IVUS implementation to the **rotating single-element IVUS** model in `rotatingSingleElm_psf.ipynb` (in this repo). The notebook uses **Field II** (via PyField, MATLAB in background) for RF simulation with **point scatterers only** and a **focused circular element**.

---

## 1. What the notebook does (from the source)

### Pipeline

1. **Input**: Scatterer map in **global artery coordinates** (positions + reflectivities).
2. **Transform**: Apply translation and rotation (probe pose) to get scatterers in **local transducer coordinates**. The probe frame has **x = catheter/rotation axis**, **z = transducer normal** (depth).
3. **Slice / visibility**: Mask scatterers by a **slice thickness** (`x_slice_thickness = 2 * elm_radius`) and by visibility: keep only scatterers with `norm(x,y) <= elm_radius` and `z > dead_r` in the **rotated** frame (so only scatterers “in front of” the element at that angle).
4. **Rotation loop**: For each rotation angle θ (e.g. 0° to 360°, 1° step):
  - Rotate the (masked) scatterers around the **x** axis by θ.
  - For the rotated scatterers, apply the visibility mask again: `norm(x,y) <= elm_radius`, `z > dead_r`.
  - Call **Field II** `field.calc_scat(Th, Th, scatterers_frame[:,:3], scatterers_frame[:,3])` to get one A-line (RF data + `t0`).
  - Interpolate RF to a common time base `t_int`.
5. **Display**: Stack A-lines → (r, θ) polar; Hilbert envelope → dB → normalized; plot as r–θ.

### Transducer (Field II)

- **Concave (focused) circular element**: `field.xdc_concave(elm_radius, elm_focal_radius, ele_size)`.
  - `elm_radius = 600e-6` m = **0.6 mm** (element radius).
  - `elm_focal_radius = 4e-3` m = **4 mm** (focal length / radius of curvature).
  - `ele_size = 200e-6` m (mathematical element size for Field II convergence).
- **Pulse**: `util.define_impulse_resp(fc, fbw, fs)` with **fc = 40 MHz**, **fbw = 0.4**; then `field.xdc_impulse(Th, h_acc)` and `field.xdc_excitation(Th, excitation)`.
- So the “PSF” is **implicit in Field II**: focused beam, pulse length, and element geometry.

### Other parameters

- **dead_r = 0.5e-3** m = **0.5 mm** (dead zone; no image reconstruction shallower than this).
- **imaging_depth** used in time base; scatterer phantom uses e.g. r from 1 mm to 5 mm.
- **c0 = 1540** m/s, **fs = 200e6** Hz; **attenuation in Field II = 0** (not yet set for blood/tissue).
- **Rotation**: Explicit step over angles (e.g. 1°); one A-line per angle.

### Tissue model

- **Point scatterers only**: positions (x,y,z) + reflectivity; no surfaces or meshes. Optional wire phantom: scatterers on rings at several r and θ.

---

## 2. Current implementation (this codebase)


| Aspect              | Current raytracing IVUS                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Transducer**      | **Point source** at catheter center (no aperture, no focal length). All rays share one origin; direction sweeps 360°.                                           |
| **Acoustic engine** | **OptiX raytracing**: rays hit meshes/spheres; specular (+ optional refraction); volumetric scatter along ray. |
| **Tissue**          | **Meshes (e.g. cylinder)** + **volumetric scatter** (3D texture + material). |
| **Rotation**        | **No explicit rotation**: all angles launched in one frame; no “current angle” or mechanical rotation.                                                          |
| **Axial PSF**       | **Causal Hanning** (IVUS) applied by 1D convolution along depth after raytracing.                                                                               |
| **Lateral**         | Single **depth-invariant** Gaussian lateral kernel (width from “typical” depth and aperture 0.5 mm).                                                            |
| **Dead zone**       | No explicit dead zone; causal axial PSF reduces near-field leak.                                                                                                |
| **Output**          | Unwrapped B-mode (angle × depth); scan-convert from polar scanlines.                                                                                            |


---

## 3. Key differences (notebook vs current code)


| Feature              | rotatingSingleElm_psf.ipynb                                         | Current implementation                                                                                |
| -------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Element geometry** | **Focused circular** (0.6 mm radius, 4 mm focal length)             | **Point source** (no size, no focus)                                                                  |
| **Beam shape**       | Focused (narrower near 4 mm); given by Field II                     | Diverging from point; lateral width ~ λ·r/a with fixed “typical” a                                    |
| **RF / PSF**         | **Field II** `calc_scat` (full Green’s function / impulse response) | Ray hits + scatter → scanline → **axial** (causal Hanning) + **lateral** (fixed Gaussian) convolution |
| **Tissue**           | **Point scatterers only** (list of positions + reflectivities)      | **Geometry (meshes)** + **volumetric scatter**                |
| **Rotation**         | **Explicit**: loop over θ, rotate scatterers, one A-line per angle  | **Implicit**: one launch, one ray per angle                                                           |
| **Dead zone**        | **dead_r = 0.5 mm** (no image shallower)                            | None (causal PSF only)                                                                                |
| **Slice**            | Slice thickness = 2× element radius; visibility mask                | 2D cross-section (no slice thickness)                                                                 |
| **Attenuation**      | Currently 0 in Field II                                             | Material attenuation along ray (Beer–Lambert)                                                         |


---

## 4. Changes needed to get similar properties

### 4.1 Focused circular element (biggest physical difference)

- **Notebook**: Focused concave element (0.6 mm radius, 4 mm focal length) → beam narrows near focus.
- **Current**: Point source → beam only widens with depth.
- **Change**: Introduce an **IVUS probe with aperture and focal length** (e.g. effective element radius and focal distance). That implies:
  - **Ray gen**: Either (a) keep one ray per angle but weight or taper by a **depth-dependent lateral beam profile** (narrower near focus), or (b) emit multiple rays per angle (e.g. over a disk) and sum, or (c) add a **spatially varying lateral PSF** that depends on depth (e.g. narrower at 4 mm, wider at 0 and 8 mm).
  - **Parameters**: Expose **element_radius_mm** (e.g. 0.6) and **focal_length_mm** (e.g. 4) on the probe or sim params; use them in lateral PSF and/or ray weighting.

### 4.2 Spatially varying lateral PSF (depth-dependent beam)

- **Notebook**: Field II gives a focused beam (lateral width varies with depth).
- **Current**: **Implemented.** Depth-dependent lateral PSF uses a **Gaussian beam model** from the literature:
  - **Beam waist at focus**: w0 = λ F / (2a) (diffraction-limited circular aperture; F = focal length, a = element radius). Refs: Goodman "Introduction to Fourier Optics"; -3 dB beam width = F#×λ with F# = F/(2a).
  - **Rayleigh length**: z_R = π w0²/λ (distance from waist where beam area doubles). Refs: Siegman "Lasers" Ch. 17; Wikipedia "Rayleigh length"; rp-photonics "Gaussian beams".
  - **Beam radius vs depth**: w(z) = w0 × sqrt(1 + (z/z_R)²) with z = depth − focal length. The lateral PSF at each depth bin uses a Gaussian kernel with σ = w(z). The depth scale is fully determined by z_R (no ad hoc constant).

### 4.3 Point-scatterer-only mode

- **Notebook**: Only point scatterers; no surfaces.
- **Current**: Meshes + volumetric scatter (dense integration per depth bin along each ray). No point-scatterer path in the implementation.
- **Change**: Either (a) add a **mode** that disables specular echoes and uses only scatter (tissue represented as scatter only), or (b) add a **separate path** that takes a list of point scatterers and uses an analytic or Field II–style model (no OptiX geometry). For (a): e.g. probe type or sim flag to skip writing specular in `closest_hit`; feed scatter from texture or from a point list.

### 4.4 Explicit rotation and visibility

- **Notebook**: For each θ, scatterers are rotated and then masked with `norm(x,y) <= elm_radius` and `z > dead_r`.
- **Current**: All angles at once; no rotation state; no “element radius” visibility.
- **Change**: Optional **explicit rotation**: e.g. “current angle” on the probe; ray gen emits only for a sector around that angle, or weights by angle. **Visibility**: if we add an element radius, mask or weight scatterers/rays by “within element footprint” at that depth/angle (e.g. cone or cylinder from element).

### 4.5 Dead zone

- **Notebook**: dead_r = 0.5 mm; no reconstruction shallower.
- **Current**: No explicit dead zone.
- **Change**: Add an optional **dead_zone_mm** (e.g. 0.5); zero or gate scanline samples for depth < dead_zone_mm before or after convolution.

### 4.6 Slice thickness

- **Notebook**: Keeps scatterers in a slice of thickness `2*elm_radius` (along x).
- **Current**: 2D cross-section (no thickness).
- **Change**: If moving to 3D scatterer lists, restrict to a slice (e.g. |x| ≤ half_thickness) for consistency with the notebook.

### 4.7 Parameter alignment

- **Element radius**: 0.6 mm in notebook; current uses 0.5 mm only for lateral PSF. Expose **element_radius_mm** (e.g. 0.6) and use in beam/PSF.
- **Focal length**: 4 mm in notebook; not present in current code. Add **focal_length_mm** and use it in depth-dependent lateral width (e.g. narrowest at 4 mm).
- **Pulse**: 40 MHz, fractional bandwidth 0.4 in notebook; current has frequency and pulse_duration (cycles). Keep 40 MHz; optionally match bandwidth (e.g. via pulse envelope).

---

## 5. Minimal set for “similar” behavior

1. **Focused element model**: Add **element_radius_mm** and **focal_length_mm**; implement **depth-dependent lateral beam/PSF** (narrower at focus, wider elsewhere).
2. **Point-scatterer-only option**: Either disable specular and use only scatter, or add a separate point-scatterer + PSF pipeline.
3. **Optional explicit rotation** and **visibility mask** (element radius, dead zone) for parity with the notebook loop.

---

## 6. Reference

- **Notebook**: `rotatingSingleElm_psf.ipynb` in this repository.
- **Field II**: Transducer and `calc_scat` define the notebook’s “PSF” and rotation-based A-line collection.

