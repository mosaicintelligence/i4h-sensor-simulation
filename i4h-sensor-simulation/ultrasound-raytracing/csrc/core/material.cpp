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

#include "raysim/core/material.hpp"

#include <algorithm>

#include "raysim/cuda/cuda_helper.hpp"

namespace raysim {

Material::Material(float impedance, float attenuation, float speed_of_sound, float mu0, float mu1,
                   float sigma, float specularity)
    : impedance_(impedance),
      attenuation_(attenuation),
      speed_of_sound_(speed_of_sound),
      mu0_(mu0),
      mu1_(mu1),
      sigma_(sigma),
      specularity_(specularity) {}

float Material::density() const {
  return impedance_ * 1e6f / speed_of_sound_;
}

Materials::Materials() {
  // Standard materials (Goss et al. compilations, tissue phantoms). Attenuation in dB/(cm·MHz).
  //
  // NOTE on `specularity` (Material constructor last arg, default 1.0):
  //
  // Despite the name, this is the EXPONENT `n` in the Mattausch-2016 specular
  // directivity model used in optix_trace.cu::calculate_specular_intensity:
  //
  //     I_specular = max(0, cos(theta_r)^n) + max(0, cos(theta_t)^n)
  //
  // where theta_r / theta_t are the angles of the reflected / refracted ray
  // with respect to the line back to the transducer.
  //
  //   n = 1.0  -> Lambertian-like broad lobe, cos-weighted (DEFAULT).
  //   n = 10   -> narrower lobe, weaker off-normal contribution.
  //   n = 100  -> sharp mirror-like peak only at near-normal incidence.
  //   n = 0    -> cos^0 = 1 for ANY angle => CONSTANT MAXIMUM intensity
  //               at every interface hit, regardless of geometry.  This is
  //               the OPPOSITE of "no specular component" -- it lights up
  //               every wall pixel to its maximum value.  Do NOT use n < 1.
  //
  // Historically vessel_wall, liver, fat, and milk were set to n ~ 0 with the
  // misunderstanding that "low specularity = less mirror-like".  In this
  // formula the relationship is inverted: HIGHER n = narrower / less
  // contribution off-normal.  For "diffuse-like" tissue the right value is
  // n = 1 (default).  For a true mirror, use n >> 1.
  //
  // The bug surfaced when rendering vessel cross-sections (vessel_evaluation.py,
  // 2026-05-27): every pixel along the cylindrical wall was saturated to white
  // because the constant n~0 specular term dominated the (small) Fresnel
  // intensity reflection.  Tier 1 milk-bath / wire-phantom tests did not
  // exercise this because milk is uniform (no interfaces sampling milk's
  // specularity) and the bone-wire interfaces correctly used bone's default
  // n = 1.  Cyst-phantom Tests F/M evaluate only bulk speckle and noise floor
  // statistics, not interface brightness.
  //
  materials_ = {{"water", Material(1.48f, 0.0022f, 1480.f, 0.f)},
                {"blood", Material(1.68f, 0.7f, 1584.f, 0.22f, 0.16f, 0.24f)},  // high-frequency IVUS-tuned default
                {"fat", Material(1.38f, 0.63f, 1450.f, 1.f, 0.f, 1.f)},  // specularity defaulted to 1.0 (was 0.0 -- bug, see note above)
                {"liver", Material(1.65f, 0.7f, 1550.f, 0.7f, 0.f, 0.3f)},  // specularity defaulted to 1.0 (was 1e-5 -- bug, see note above)
                {"muscle", Material(1.70f, 1.09f, 1580.f, 0.5f, 0.8f, 0.4f)},
                {"bone", Material(7.80f, 5.f, 4080.f, 0.8f, 0.9f, 0.5f)},
                /*
                 * Tungsten (B2 / PSF wire phantoms; ivus_test_0515 bench fixture)
                 * AND vesselgen guidewires (PV .035 nominal 0.89 mm diameter).
                 *   Z = 101 MRayl, c = 5200 m/s, rho = 19.4 g/cm^3 (Selfridge /
                 *   Goss compilations; WebElements c = 5174 m/s).
                 *   Replaces the historical "bone" hack for wire spheres -- bench
                 *   data is 30 um tungsten wires in water / milk, not bone.
                 *   Attenuation raised to 60 dB/cm/MHz from the original 15 dB/cm/MHz
                 *   bench-fixture value.  At IVUS frequencies (10 MHz) and a
                 *   nominal 0.89 mm peripheral guidewire diameter, the bulk
                 *   attenuation across 0.89 mm of tungsten is 60 * 0.89 / 10
                 *   = 5.3 dB/MHz * 10 MHz = 53 dB one-way (~106 dB round-trip),
                 *   which combined with the ~12 dB Fresnel transmission loss
                 *   (R = 0.937) reliably saturates the acoustic shadow behind
                 *   the wire even at the closer angles where the scanline only
                 *   grazes the cylinder. The 30-um bench wires (Test E) sit
                 *   far below this attenuation budget so the small per-bench
                 *   grazing-ray contribution is unchanged.  Scatter kept low
                 *   (specular reflector).
                 */
                {"tungsten", Material(101.f, 60.f, 5200.f, 0.05f, 0.05f, 0.05f)},
                /*
                 * IVUS / vascular materials (literature; attenuation in dB/(cm·MHz)):
                 * - lumen/blood: c 1584 m/s, Z 1.68 MRayl, α 0.7 with elevated
                 *   scatter terms (mu0=0.22, mu1=0.16, sigma=0.24). This default
                 *   follows high-frequency IVUS tuning and better visual realism in
                 *   pullback renders than legacy low-frequency blood attenuation.
                 * - vessel_wall: blood vessel c 1571 m/s, Z 1.82 MRayl (PMC5126009); α ~1 from 50 MHz coronary data (e.g. 4.99 dB/mm @ 50 MHz).
                 *   Specularity set to default 1.0 (Lambertian-like) -- previously 1e-5 which
                 *   gave a constant maximum specular intensity at every wall hit (saturated
                 *   rind in vessel renders).  See `specularity` note above.
                 * - extravascular: muscle-like c 1547 m/s, Z 1.62 MRayl (PMC5126009); α 0.7.
                 * Refs: Goss et al. JASA compilations; PMC3570716 (Ultrasound Med Biol 2013); PMC5126009 (J Ultrasound 2016); Lockwood et al. UMB 17(7) 1991 (35–65 MHz vascular).
                 */
                {"lumen", Material(1.68f, 0.7f, 1584.f, 0.22f, 0.16f, 0.24f)},
                {"vessel_wall", Material(1.82f, 1.0f, 1571.f, 0.5f, 0.6f, 0.35f)},
                {"extravascular", Material(1.62f, 0.7f, 1547.f, 0.5f, 0.4f, 0.3f)},
                /*
                 * Trilaminar IVUS wall layers (intima / media / adventitia).
                 *
                 * The "vessel_wall" entry above is the legacy single-slab
                 * placeholder used by the original vessel-generator output.
                 * Below are the three named layers that drive the
                 * canonical bright-dark-bright IVUS appearance described
                 * in every IVUS interpretation textbook (e.g. ACC Expert
                 * Consensus on IVUS, JACC 2001).
                 *
                 * Layer-dependent values are anchored to:
                 *   * Lockwood et al., Ultrasound Med Biol 17(6):653-666,
                 *     1991 -- arterial wall c=1579-1628 m/s, alpha rising
                 *     from 4 dB/mm @ 30 MHz to 10 dB/mm @ 60 MHz, with
                 *     the media as the darkest band in the radial view.
                 *   * Machado et al., Braz J Med Biol Res, 50-MHz acoustic
                 *     microscopy of human coronaries -- integrated
                 *     backscatter coefficient (IBC) per layer:
                 *         media          ~5.1  (sr*m)^-1   -- darkest
                 *         thickened intima ~17.4 (sr*m)^-1
                 *         adventitia     ~21.3 (sr*m)^-1   -- brightest
                 *     Reproduced as the relative `mu0`/`sigma` ordering
                 *     below: media `mu0`=0.25 (lowest), intima `mu0`=0.6,
                 *     adventitia `mu0`=0.7 with `sigma`=0.55 (highest).
                 *   * Hiro et al., Layer-Dependent Anisotropy of IBC,
                 *     Ultrasound Med Biol 2011 -- confirms the
                 *     bright-dark-bright pattern in side-looking IVUS.
                 *
                 * Speed of sound and impedance are within the Lockwood
                 * 1579-1628 m/s wall-as-a-whole window; we spread the
                 * three layers to match the histologically expected
                 * ordering (collagen-rich adventitia stiffest, muscular
                 * media intermediate, intima close to muscle).
                 *
                 * The vessel-generator `LayeredWallConfig` emits one
                 * mesh per layer boundary, each labelled with the
                 * material that lies *beyond* it, so a 2-layer wall
                 * uses (media, adventitia) and a 3-layer wall uses
                 * (intima, media, adventitia) in order from the lumen.
                 *
                 * Impedance recalibration (matches real IVUS look):
                 * an early draft used Z_intima=1.74, Z_adventitia=1.80
                 * which made the lumen->intima jump (R=3e-4) ~5x dimmer
                 * than the legacy lumen->vessel_wall echo (R=1.6e-3),
                 * and the rendered wall in the trilaminar demo
                 * disappeared into the noise floor. Real IVUS shows
                 * the LUMEN->INTIMA interface as the brightest in-frame
                 * feature; that requires a bigger impedance step.
                 * Updated values:
                 *   intima Z = 1.85 MRayl (Lockwood 1991, dense
                 *     subendothelial collagen rho~1170 kg/m^3, c~1600 m/s)
                 *   adventitia Z = 1.95 MRayl (dense collagen+elastin
                 *     mat with vasa vasorum, rho~1200, c~1620)
                 * Reflection coefficients R = ((Z2-Z1)/(Z2+Z1))^2:
                 *   blood(1.68) -> intima(1.85) : R = 2.3e-3
                 *   intima(1.85) -> media(1.66) : R = 2.9e-3
                 *   media(1.66)  -> adventitia(1.95) : R = 6.5e-3
                 *   adventitia(1.95) -> extra(1.62) : R = 8.4e-3
                 * This reproduces the canonical bright-dark-bright-bright
                 * IVUS appearance and lifts the wall out of the noise
                 * floor without breaking the legacy `vessel_wall`/
                 * `extravascular` calibration used by the CT pullbacks.
                 *
                 * Bulk-scatter recalibration v2: previous values gave
                 * only a ~2x bright-dim-bright ratio in the rendered
                 * frames (the media body brightness was ~70 vs ~140 at
                 * the intima peak), well below the ~3-5x clinical
                 * trilaminar contrast. Push intima/adventitia sigma up
                 * and media sigma down so the three bands are clearly
                 * resolvable post-TGC.  Impedance contrast is also
                 * widened slightly so the lumen->intima specular and
                 * the media->adventitia specular each appear as
                 * distinct bright thin lines.
                 *
                 *   intima   Z 1.85 -> 1.92 (lumen->intima R 2.3e-3 -> 3.6e-3)
                 *   adventitia Z 1.95 -> 2.05 (media->advent R 6.5e-3 -> 1.3e-2)
                 *   media sigma 0.12 -> 0.05 (4x bulk dim)
                 *   intima sigma 0.5 -> 0.75 (1.5x)
                 *   adventitia sigma 0.65 -> 0.95 (1.5x)
                 */
                {"intima", Material(1.92f, 1.1f, 1600.f, 0.85f, 0.7f, 0.75f)},
                {"media", Material(1.66f, 0.9f, 1571.f, 0.25f, 0.15f, 0.05f)},
                {"adventitia", Material(2.05f, 1.3f, 1620.f, 1.0f, 0.8f, 0.95f)},
                /*
                 * Atherosclerotic plaque components (in-wall inclusions).
                 *
                 * Impedance values are from Top et al., Celal Bayar Univ
                 * J Sci 2018, scanning acoustic microscopy at 80 MHz on
                 * fresh human carotid endarterectomy specimens:
                 *     calcified region:  Z = 2.23 +/- 0.09 MRayl
                 *     fibrous tissue:    Z = 2.02 +/- 0.06 MRayl
                 *     lipid pool:        Z = 1.70 +/- 0.07 MRayl
                 *
                 * Cross-validated by:
                 *   * de Korte et al., Circulation 2000 (in-vitro IVUS
                 *     elastography on human femoral and coronary
                 *     arteries) -- same ordering of plaque-component
                 *     stiffness and impedance.
                 *   * Nair et al., Circulation 2002 (VH-IVUS) and Nair
                 *     UMB 2007 -- spectral classifier separates
                 *     fibrous, fibrofatty, calcified, and necrotic-core
                 *     regions at 30 MHz with these same relative
                 *     acoustic signatures.
                 *   * Brewin et al., Ultrasonics 54(2):428-441, 2014 --
                 *     carotid plaque sound speed at 20 MHz, components
                 *     between 1450 (lipid) and 1620 (fibrous) m/s.
                 *
                 * Attenuation for calcified plaque (alpha = 20 dB/cm/MHz)
                 * sits in the upper half of the 6-30 dB/cm/MHz
                 * literature window (Lockwood 1991; Saijo UMB 2007
                 * carotid acoustic microscopy at 80 MHz extrapolated
                 * down to the IVUS band).
                 *
                 * High attenuation is critical for clinical realism:
                 *   * It buries the *back* interface specular below
                 *     the noise floor, so the calcium reads as a
                 *     bright proximal arc fading into shadow rather
                 *     than as a "ring" with both proximal and distal
                 *     bright lines (which is the closed-mesh
                 *     simulation artefact).
                 *   * It still produces a clean posterior shadow.
                 *
                 * One-way attenuation @ 30 MHz vs depth into plaque:
                 *   0.2 mm: 12 dB    0.4 mm: 24 dB    0.6 mm: 36 dB
                 *   0.8 mm: 48 dB    1.0 mm: 60 dB    1.5 mm: 90 dB
                 * The body brightness gradient is restored by the
                 * boosted scatter sigma below.
                 *
                 * Earlier iterations: 25 (back specular invisible but
                 * body too dim); 12 (bright body but visible back
                 * specular at thinner edges); 6 (no shadow at all).
                 */
                /*
                 * Calcified / lipid / fibrous mu0/sigma recalibration
                 * (2026-06-10): the demo renders showed the lesion bodies
                 * reading too dim at IVUS through 5+ mm of intervening
                 * blood + intima/media.
                 *
                 * Calcified plaque:
                 *   mu0 = 1.0 (saturated -- every voxel scatters; the
                 *   threshold has no effect above 1.0), sigma = 16.0
                 *   (~46x the vessel-wall sigma).  At sigma >> R the
                 *   body scatter dominates the back-interface specular
                 *   peak at every depth, so the lesion reads as a
                 *   bright filled blob fading smoothly into the
                 *   posterior shadow rather than two thin bright
                 *   lines with a dark middle.  The high sigma also
                 *   makes the body bright enough to remain visible
                 *   even at 0.5+ mm of plaque depth where 20 dB/cm/MHz
                 *   attenuation has already dropped the carrier
                 *   intensity by 30+ dB.
                 * Fibrous plaque:
                 *   mu0 = 1.5, mu1 = 1.0, sigma = 0.8 -- bright but
                 *   not as brilliant as calcium; visible cap that
                 *   sits between media and lipid in echogenicity.
                 * Lipid pool:
                 *   left low. Its identifying feature is being darker
                 *   than the surrounding wall, not bright.
                 */
                {"calcified_plaque", Material(2.23f, 20.0f, 2500.f, 1.0f, 1.5f, 16.0f)},
                {"lipid_pool", Material(1.70f, 0.4f, 1480.f, 0.12f, 0.08f, 0.05f)},
                {"fibrous_plaque", Material(2.02f, 1.4f, 1620.f, 1.5f, 1.0f, 0.8f)},
                /*
                 * Fresh intraluminal thrombus.
                 *
                 * Acoustically close to blood with slightly elevated
                 * impedance and backscatter (organised fibrin network):
                 *   * Wang & Shung, IEEE UFFC 41(6) 1994 -- blood
                 *     backscatter sigma rises with hematocrit and
                 *     organisation.
                 *   * Picano et al., Circulation 72(3):572-576, 1985 --
                 *     comparative backscatter levels of normal blood
                 *     vs organised thrombus vs fibrous tissue.
                 *
                 * Z=1.70 MRayl matches young thrombus measurements; the
                 * elevated `mu0`/`sigma` vs blood produces the
                 * "smoky" intraluminal echo seen on real IVUS frames of
                 * acute occlusion.
                 */
                {"thrombus", Material(1.70f, 0.4f, 1560.f, 0.55f, 0.40f, 0.30f)},
                /*
                 * Evaporated milk uniform phantom (Tests I + M bench medium).
                 *   * Physical params from literature (UNCHANGED):
                 *     - density 1030 kg/m^3, c = 1530 m/s -> Z = 1.58 MRayl
                 *       (Buckin & Smyth 1999; Schmid 1991).
                 *     - attenuation 0.5 dB/cm/MHz at 10 MHz (Goss et al. 1980,
                 *       whole milk; evaporated milk runs 0.5-1.0 due to ~2x
                 *       solids -- we use the conservative lower bound so
                 *       TGC mismatch surfaces rather than being absorbed).
                 *   * Scattering pair (mu0 = 0.5, sigma = 0.1):
                 *     - Calibrated against Wave 0 E4a uniform-milk bench data
                 *       (`ivus_test_0515/raw/e4a_milk_gain_p{1,2,3}/`).
                 *     - sigma = 0.1 lands Test I (depth uniformity) with
                 *       span_ratio = 0.98 (bench 5.5 vs sim 5.4 palette span)
                 *       PASS under the shape-primary AC; mu0 = 0.5 kept from
                 *       the literature-analogue seed (uniform scatterer
                 *       density).
                 *   * Test M (speckle CoV / radial correlation) FAILS on the
                 *     CoV axis: bench CoV_log = 0.384 is dominated by COHERENT
                 *     reverberation features (60-78% coherent variance), not
                 *     random speckle. The sim reproduces the bench's radial
                 *     correlation length within ~1% at (mu0=0.5, sigma=1.0)
                 *     but that combination breaks Test I span_ratio (4.58 ->
                 *     FAIL), so the trade-off favours Test I PASS + Test M
                 *     structural-limit FAIL. Closing Test M requires modelling
                 *     catheter / container reverberations (not implemented).
                 *   * The YAML `materials: - name: milk` block in
                 *     `instrument-calibration/p035_visions/volcano_s5i.yaml`
                 *     mirrors this entry (the YAML is decorative documentation;
                 *     this list is the actual source of truth read at runtime).
                 *   * For sim-in-loop sweeps over milk parameters, prefer the
                 *     runtime override `Materials::update_material("milk", ...)`
                 *     (Python: `materials.update_material("milk", mu0=..,
                 *     sigma=..)`) over editing this list, which requires a
                 *     C++ rebuild.
                 */
                {"milk", Material(1.58f, 0.5f, 1530.f, 0.5f, 0.4f, 0.1f)}};  // specularity defaulted to 1.0 (was 0.0 -- bug, see note above; latent because milk is used as uniform background)

  // Upload materials to device
  std::vector<Material> material_data;
  material_data.reserve(materials_.size());
  for (size_t index = 0; index < materials_.size(); ++index) {
    material_data.push_back(materials_[index].second);
  }
  const size_t material_data_size = materials_.size() * sizeof(Material);
  d_material_data_ = std::make_unique<CudaMemory>(material_data_size);
  d_material_data_->upload(&material_data[0], cudaStreamPerThread);
}

const std::unique_ptr<CudaMemory>& Materials::get_material_data() const {
  return Materials::d_material_data_;
}

uint32_t Materials::get_index(const std::string& name) const {
  const uint32_t index = std::distance(
      materials_.begin(),
      std::find_if(
          materials_.begin(),
          materials_.end(),
          [&name](const std::pair<std::string, Material>& item) { return item.first == name; }));
  if (index == materials_.size()) { throw std::runtime_error("Material not found"); }
  return index;
}

void Materials::update_material(const std::string& name,
                                float impedance,
                                float attenuation,
                                float speed_of_sound,
                                float mu0,
                                float mu1,
                                float sigma,
                                float specularity) {
  const uint32_t index = get_index(name);
  Material& mat = materials_[index].second;
  if (impedance      >= 0.f) { mat.impedance_      = impedance; }
  if (attenuation    >= 0.f) { mat.attenuation_    = attenuation; }
  if (speed_of_sound >= 0.f) { mat.speed_of_sound_ = speed_of_sound; }
  if (mu0            >= 0.f) { mat.mu0_            = mu0; }
  if (mu1            >= 0.f) { mat.mu1_            = mu1; }
  if (sigma          >= 0.f) { mat.sigma_          = sigma; }
  if (specularity    >= 0.f) { mat.specularity_    = specularity; }
  std::vector<Material> material_data;
  material_data.reserve(materials_.size());
  for (size_t i = 0; i < materials_.size(); ++i) {
    material_data.push_back(materials_[i].second);
  }
  d_material_data_->upload(&material_data[0], cudaStreamPerThread);
}

}  // namespace raysim
