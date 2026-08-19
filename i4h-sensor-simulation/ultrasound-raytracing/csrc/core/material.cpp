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
                 * Tungsten (B2 / PSF wire phantoms; ivus_test_0515 bench fixture).
                 *   Z = 101 MRayl, c = 5200 m/s, rho = 19.4 g/cm^3 (Selfridge /
                 *   Goss compilations; WebElements c = 5174 m/s).
                 *   Replaces the historical "bone" hack for wire spheres -- bench
                 *   data is 30 um tungsten wires in water / milk, not bone.
                 *   Attenuation set high (15 dB/cm/MHz) -- bulk path through the
                 *   wire is negligible at 30 um diameter; value mainly affects
                 *   grazing rays.  Scatter kept low (specular reflector).
                 */
                {"tungsten", Material(101.f, 15.f, 5200.f, 0.05f, 0.05f, 0.05f)},
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
                {"milk", Material(1.58f, 0.5f, 1530.f, 0.5f, 0.4f, 0.1f)},  // specularity defaulted to 1.0 (was 0.0 -- bug, see note above; latent because milk is used as uniform background)
                /*
                 * Trilaminar wall + plaque (vessel-generator REQUIRED_MATERIAL_NAMES).
                 *
                 * DESIGN: wall contrast is mostly BACKSCATTER (mu0 / sigma).
                 * Interlayer Z steps stay small so intima/media and media/
                 * adventitia Fresnel stay quiet; the lumen border is the one
                 * interface we deliberately lift. `specularity` is the
                 * Mattausch n exponent -- MUST stay >= 1 (see note above).
                 *
                 * intima: Z 1.87 vs lumen 1.68 → ~0.28% intensity reflection
                 *   (was 1.811 / ~0.14%; PR20 is 1.92 / ~0.44%).
                 * media: dark stripe -- low mu0 / sigma (unchanged).
                 * adventitia: outer bright band; mu0/sigma nudged up vs the
                 *   first reconstruction.
                 *
                 * Plaque (PR20 / 2026-06-10 visual recal):
                 *   fibrous / lipid / thrombus -- table as documented.
                 *   calcified_plaque -- BETWEEN fibrous and full calc for
                 *   now (full calc is Z=2.23, α=20, σ=16). α≈12 is the
                 *   documented "back specular still visible" stop; σ≈6 is
                 *   bright without the full 46× vessel-wall saturate.
                 */
                {"intima", Material(1.87f, 0.93f, 1600.f, 0.55f, 0.6f, 0.49f, 1.f)},
                {"media", Material(1.83f, 0.80f, 1570.f, 0.12f, 0.15f, 0.10f, 1.f)},
                {"adventitia", Material(1.86f, 1.10f, 1610.f, 0.95f, 0.75f, 0.85f, 1.f)},
                {"calcified_plaque", Material(2.12f, 12.0f, 2000.f, 1.25f, 1.25f, 6.0f, 1.f)},
                {"lipid_pool", Material(1.70f, 0.4f, 1480.f, 0.12f, 0.08f, 0.05f, 1.f)},
                {"fibrous_plaque", Material(2.02f, 1.4f, 1620.f, 1.5f, 1.0f, 0.8f, 1.f)},
                {"thrombus", Material(1.70f, 0.4f, 1560.f, 0.55f, 0.40f, 0.30f, 1.f)}};

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
