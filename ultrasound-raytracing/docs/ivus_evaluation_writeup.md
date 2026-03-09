# IVUS Simulation Evaluation

This document describes the **evaluation setups** used to validate the IVUS implementation: the **IVUS vessel example**, the **wire phantom**, and the **cystic-resolution phantom**. For each setup we describe the phantom geometry, what the test validates, what image the simulation should produce, and show the **unwrapped** B-mode output for review.

All three use the same IVUS probe and simulation parameters (40 MHz, 256 angular rays, `t_far` = 10 mm, `b_mode_size` = 512×512). The unwrapped display has **horizontal axis = angle (0–360°)** and **vertical axis = depth (0–10 mm)**; the probe is at the top (depth 0).

---

## 1. IVUS vessel example (`ivus_example`)

### Setup

- **Script:** `examples/ivus_example.py` (single-frame run: `python examples/ivus_example.py` or `--single`).
- **Phantom:** Vessel with **thick-walled cylinder** (inner radius 3.5 mm, outer 4.0 mm, axis along Y). Inner surface: **vessel_wall**; outer: **extravascular** (so refracted rays in the wall see a second interface). Background (lumen): **water**. Probe at vessel center (origin); imaging plane is x–z.
- **Meshes:** `mesh/Cylinder_inner.obj`, `mesh/Cylinder_outer.obj` (generated with `python utils/phantom_maker.py cylinder --output mesh --cylinder-thick`).

### What it tests

- **Geometric accuracy:** The vessel wall should appear as **two concentric bright rings** at ~3.5 mm (inner) and ~4.0 mm (outer) depth, continuous over 360°.
- **Reflection and refraction:** Correct interface echoes at lumen–wall and wall–extravascular; no obvious streaks from wrong depth indexing.
- **Attenuation with depth:** Deeper (outer) interface should not be brighter than the inner one.
- **Pipeline:** Causal axial PSF (dark lumen, no wall echo leaking backward), depth-dependent lateral PSF, IVUS TGC, and unwrapped scan conversion.

### Expected image

- **Unwrapped:** Dark interior (lumen) from 0 to ~3.5 mm depth; **two bright horizontal bands** at ~3.5 mm and ~4.0 mm depth, extending across the full angular range (0–360°). Possible mild speckle in the wall region; no bright streaks in the lumen.

### Unwrapped output (for review)

![IVUS vessel example (unwrapped)](../ivus_example_output/ivus_frame.png)

*Unwrapped B-mode from the vessel phantom. Horizontal: angle (deg). Vertical: depth (mm).*

---

## 2. Wire phantom (`wire_phantom_evaluation`)

### Setup

- **Script:** `examples/wire_phantom_evaluation.py` (default output dir: `wire_phantom_output`).
- **Phantom:** **No vessel.** Background: **lumen** (blood). **Five point targets** (small spheres, **bone** material, radius 0.12 mm) on rings at 1, 2, 3, 4, and 5 mm radius, arranged in a **spiral** (one wire per ring at angles 0°, 72°, 144°, 216°, 288°) in the x–z plane (y = 0). Wires are high-impedance so reflection dominates over scatter.
- **Purpose:** Resolution and geometric validation; point targets at known (angle, depth) with minimal clutter.

### What it tests

- **Geometric accuracy:** Each wire should appear as a **bright spot** at the correct (angle, depth) in the unwrapped image: depths 1, 2, 3, 4, 5 mm and angles spaced by ~72° (spiral).
- **Axial/lateral resolution:** Spots should be **compact** (causal axial PSF avoids double peaks; depth-dependent lateral PSF gives finite spot size). No wall to create strong clutter; wire echoes should be clearly localized.
- **Reflection-dominated regime:** Lumen background is weakly scattering; wire echoes should dominate, validating that the scatter integral scale does not overwhelm point reflectors.

### Expected image

- **Unwrapped:** Largely **dark** (blood background). **Five bright spots** at depths 1, 2, 3, 4, 5 mm, spaced in angle so they form a spiral pattern (not aligned in one column). Spots should be **single** (no central null from symmetric PSF) and **localized** in angle and depth.

### Unwrapped output (for review)

![Wire phantom (unwrapped)](../wire_phantom_output/wire_phantom_unwrapped.png)

*Unwrapped B-mode from the wire phantom. Five point targets at 1–5 mm depth in a spiral pattern.*

---

## 3. Cystic-resolution phantom (`cystic_resolution_phantom_evaluation`)

### Setup

- **Script:** `examples/cystic_resolution_phantom_evaluation.py` (default output dir: `cystic_phantom_output`).
- **Phantom:** **Tissue-mimicking background** (**vessel_wall**); **five fluid-filled “cysts”** (spheres with **water** material) at known positions and sizes in the x–z plane. Cysts are (radius_mm, angle_deg, cyst_radius_mm): (2, 0°, 0.5), (3, 72°, 0.8), (4, 144°, 1.0), (5, 216°, 1.2), (3.5, 288°, 0.6). Background is scattering; cysts are fluid (low backscatter) so they appear **anechoic** relative to tissue.
- **Purpose:** Contrast resolution and cyst detection; scatter-dominated background with dark regions at known locations.

### What it tests

- **Contrast:** Cysts should appear **darker** than the surrounding tissue (vessel_wall speckle). No vessel ring; only tissue + cysts.
- **Scatter and TGC:** The **scatter_integral_scale** and **scattering_resolution_mm** should produce **visible speckle** in the background without over-gaining; cysts should remain clearly hypoechoic.
- **Geometric/depth consistency:** Cyst centers and sizes should match the phantom (2–5 mm depth range, angles 0°–288°). Cyst boundaries may be slightly blurred by the PSF but should be identifiable.

### Expected image

- **Unwrapped:** **Speckled background** (tissue) over the full (angle, depth) range. **Five darker regions** (cysts) at the specified (angle, depth) positions and approximate sizes. Deeper cysts may be slightly less visible if TGC is conservative; overall the image should show clear contrast between tissue and fluid regions.

### Unwrapped output (for review)

![Cystic-resolution phantom (unwrapped)](../cystic_phantom_output/cystic_phantom_unwrapped.png)

*Unwrapped B-mode from the cystic-resolution phantom. Speckled tissue background with five anechoic cysts at known positions.*

---

## Summary

| Evaluation        | Phantom                    | Main test focus              | Key unwrapped output                    |
|------------------|----------------------------|------------------------------|-----------------------------------------|
| IVUS vessel       | Thick cylinder (lumen + wall) | Geometry, interfaces, attenuation | Two concentric rings at ~3.5 and ~4 mm  |
| Wire phantom      | Point targets in lumen     | Resolution, geometry, reflections | Five bright spots in spiral (1–5 mm)    |
| Cystic phantom    | Tissue + fluid cysts       | Contrast, scatter, TGC        | Speckled background, five dark cysts  |

To regenerate the unwrapped images (from the `ultrasound-raytracing` directory):

```bash
# Vessel (requires mesh: python utils/phantom_maker.py cylinder --output mesh --cylinder-thick)
python examples/ivus_example.py --output-dir ivus_example_output

# Wire phantom
python examples/wire_phantom_evaluation.py --output-dir wire_phantom_output

# Cystic phantom
python examples/cystic_resolution_phantom_evaluation.py --output-dir cystic_phantom_output
```

Unwrapped files written:

- `ivus_example_output/ivus_frame.png`
- `wire_phantom_output/wire_phantom_unwrapped.png`
- `cystic_phantom_output/cystic_phantom_unwrapped.png`

---

## Potential next steps

For **modeling gaps** and **suggested enhancements** (e.g. frequency dependence of scattering, catheter/ring-down, transmit directivity, noise, rotation/motion, speed-of-sound heterogeneity, multiple scattering, and other limitations), see **[ivus_implementation_writeup.md](ivus_implementation_writeup.md)** §12 (Potential next steps and modeling gaps).
