/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include <spdlog/spdlog.h>

#include "raysim/core/curvilinear_probe.hpp"
#include "raysim/core/hitable.hpp"
#include "raysim/core/ivus_probe.hpp"
#include "raysim/core/linear_array_probe.hpp"
#include "raysim/core/material.hpp"
#include "raysim/core/phased_array_probe.hpp"
#include "raysim/core/raytracing_ultrasound_simulator.hpp"
#include "raysim/core/world.hpp"
#include "raysim/core/write_image.hpp"
#include "raysim/cuda/cuda_helper.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to float3
float3 numpy_to_float3(py::array_t<float> array) {
  auto buf = array.request();
  if (buf.ndim != 1 || buf.shape[0] != 3) {
    throw std::runtime_error("numpy array must have shape (3,)");
  }
  float* ptr = static_cast<float*>(buf.ptr);
  return make_float3(ptr[0], ptr[1], ptr[2]);
}

// Helper function to convert float3 to numpy array
py::array_t<float> float3_to_numpy(float3 vec) {
  auto result = py::array_t<float>(3);
  auto buf = result.request();
  float* ptr = static_cast<float*>(buf.ptr);
  ptr[0] = vec.x;
  ptr[1] = vec.y;
  ptr[2] = vec.z;
  return result;
}

PYBIND11_MODULE(ray_sim_python, m) {
  m.doc() = R"pbdoc(
        Raytracing-based ultrasound simulator

        This module provides a CUDA-accelerated ultrasound simulation framework.
        It allows for realistic simulation of ultrasound imaging by raytracing through
        3D meshes with different material properties.

        Key Classes:
            - Materials: Manages acoustic material properties
            - World: Contains the 3D scene geometry
            - Pose: Represents 3D position and orientation
            - CurvilinearProbe, LinearArrayProbe, PhasedArrayProbe: Define ultrasound probe parameters
            - RaytracingUltrasoundSimulator: Main simulation engine
    )pbdoc";

  // Bind Pose class
  py::class_<raysim::Pose>(m, "Pose", R"pbdoc(
        Represents a 3D pose with position and rotation.

        Args:
            position (array-like): 3D position vector [x, y, z]. Can be a list or numpy array.
            rotation (array-like): 3D rotation vector [roll, pitch, yaw] in radians. Can be a list or numpy array.

        Examples:
            Using numpy arrays:
            >>> pose = Pose(position=np.array([0, 0, 0]), rotation=np.array([0, np.pi, 0]))

            Using lists:
            >>> pose = Pose([0, 0, 0], [0, np.pi, 0])

            Using keyword arguments with lists:
            >>> pose = Pose(position=[0, 0, 0], rotation=[0, np.pi, 0])
    )pbdoc")
      .def(py::init([](py::array_t<float> position, py::array_t<float> rotation) {
             return std::make_unique<raysim::Pose>(numpy_to_float3(position),
                                                   numpy_to_float3(rotation));
           }),
           py::arg("position"),
           py::arg("rotation"),
           R"pbdoc(
        Initialize a new Pose.

        Args:
            position (np.ndarray): 3D position vector [x, y, z]
            rotation (np.ndarray): 3D rotation vector [roll, pitch, yaw] in radians
      )pbdoc")
      .def_property(
          "position",
          [](const raysim::Pose& self) { return float3_to_numpy(self.position_); },
          [](raysim::Pose& self, py::array_t<float> position) {
            self.position_ = numpy_to_float3(position);
          },
          "3D position vector [x, y, z]")
      .def_property(
          "rotation",
          [](const raysim::Pose& self) { return float3_to_numpy(self.rotation_); },
          [](raysim::Pose& self, py::array_t<float> rotation) {
            self.rotation_ = numpy_to_float3(rotation);
          },
          "3D rotation vector [roll, pitch, yaw] in radians");

  // Bind Material class
  py::class_<raysim::Material>(m, "Material", R"pbdoc(
        Represents acoustic material properties for ultrasound simulation.

        Properties:
            - impedance: Acoustic impedance (MRayl)
            - attenuation: Attenuation coefficient (dB/cm/MHz)
            - speed_of_sound: Speed of sound in material (m/s)
            - mu0: Scattering coefficient
            - mu1: Scattering coefficient
            - sigma: Surface roughness
            - specularity: Specularity coefficient
    )pbdoc")
      .def(py::init<float, float, float, float, float, float, float>(),
           py::arg("impedance"),
           py::arg("attenuation"),
           py::arg("speed_of_sound"),
           py::arg("mu0") = 0.f,
           py::arg("mu1") = 0.f,
           py::arg("sigma") = 0.f,
           py::arg("specularity") = 1.f)
      .def_readwrite("impedance_", &raysim::Material::impedance_, "Acoustic impedance (MRayl)")
      .def_readwrite(
          "attenuation_", &raysim::Material::attenuation_, "Attenuation coefficient (dB/cm/MHz)")
      .def_readwrite("speed_of_sound_", &raysim::Material::speed_of_sound_, "Speed of sound (m/s)")
      .def_readwrite("mu0_", &raysim::Material::mu0_, "Scattering coefficient mu0")
      .def_readwrite("mu1_", &raysim::Material::mu1_, "Scattering coefficient mu1")
      .def_readwrite("sigma_", &raysim::Material::sigma_, "Surface roughness")
      .def_readwrite("specularity_", &raysim::Material::specularity_, "Specularity coefficient")
      .def("density", &raysim::Material::density, "Calculate material density from impedance");

  // Bind Materials class
  py::class_<raysim::Materials>(m, "Materials", R"pbdoc(
        Manages a collection of acoustic materials for simulation.

        Provides predefined materials like:
            - water
            - fat
            - muscle
            - liver
            - bone
            - blood
    )pbdoc")
      .def(py::init<>())
      .def("get_index", &raysim::Materials::get_index, R"pbdoc(
        Get the index of a predefined material.

        Args:
            name (str): Material name (e.g., "water", "fat", "muscle", "liver", "bone", "blood",
                "lumen", "vessel_wall", "extravascular")

        Returns:
            int: Material index for use in World objects
      )pbdoc");

  // Bind World class
  py::class_<raysim::World>(m, "World", R"pbdoc(
        Represents the 3D scene for ultrasound simulation.

        Contains meshes and their material properties. The world has a background
        material (typically "water") and can contain multiple objects with different
        materials.
    )pbdoc")
      .def(py::init([](const std::string& background_material) {
             spdlog::info("Creating World with background material: {}", background_material);
             try {
               auto world = std::make_unique<raysim::World>(background_material);
               spdlog::info("World created successfully");
               return world;
             } catch (const std::exception& e) {
               spdlog::error("Failed to create World: {}", e.what());
               throw;
             }
           }),
           py::arg("background_material") = "water",
           R"pbdoc(
        Initialize a new World.

        Args:
            background_material (str): Name of the background material (default: "water")
      )pbdoc")
      .def(
          "add",
          [](raysim::World& self, py::object obj) {
            spdlog::info("Adding object to world");
            try {
              auto* hitable_ptr = obj.cast<raysim::Hitable*>();
              auto unique_hitable = std::unique_ptr<raysim::Hitable>(hitable_ptr);
              self.add(std::move(unique_hitable));
              obj.release();
              spdlog::info("Object added successfully");
            } catch (const std::exception& e) {
              spdlog::error("Failed to add object to world: {}", e.what());
              throw;
            }
          },
          R"pbdoc(
        Add an object (Mesh or Sphere) to the world.
        Note: The world takes ownership of the object.
      )pbdoc")
      .def("get_background_material",
           &raysim::World::get_background_material,
           "Get the background material name")
      .def(
          "get_aabb_min",
          [](const raysim::World& self) {
            auto min = self.get_aabb_min();
            return float3_to_numpy(min);
          },
          "Get the minimum point of the world's bounding box")
      .def(
          "get_aabb_max",
          [](const raysim::World& self) {
            auto max = self.get_aabb_max();
            return float3_to_numpy(max);
          },
          "Get the maximum point of the world's bounding box");

  // Bind BaseProbe class and derived classes
  py::class_<raysim::BaseProbe>(m, "BaseProbe", R"pbdoc(
Base class for ultrasound probes.

All probe types derive from this class and inherit common functionality such as:
- Position and orientation (pose) handling
- Element position and ray direction calculation
- Basic acoustic parameters
- Support for 1D and 2D (matrix) arrays

Note: Pose orientation is specified in radians (roll, pitch, yaw).
)pbdoc")
      .def("set_pose", &raysim::BaseProbe::set_pose, "Set the probe's pose")
      .def("get_pose", &raysim::BaseProbe::get_pose, "Get the probe's current pose")
      .def("get_num_elements_x",
           &raysim::BaseProbe::get_num_elements_x,
           "Get number of transducer elements in lateral (x) direction")
      .def("set_num_elements_x",
           &raysim::BaseProbe::set_num_elements_x,
           "Set number of transducer elements in lateral (x) direction")
      .def("get_frequency", &raysim::BaseProbe::get_frequency, "Get probe frequency (MHz)")
      .def("set_frequency", &raysim::BaseProbe::set_frequency, "Set probe frequency (MHz)")
      .def("get_speed_of_sound",
           &raysim::BaseProbe::get_speed_of_sound,
           "Get speed of sound (mm/μs)")
      .def("set_speed_of_sound",
           &raysim::BaseProbe::set_speed_of_sound,
           "Set speed of sound (mm/μs)")
      .def("get_pulse_duration",
           &raysim::BaseProbe::get_pulse_duration,
           "Get pulse duration (cycles)")
      .def("set_pulse_duration",
           &raysim::BaseProbe::set_pulse_duration,
           "Set pulse duration (cycles)")
      .def("get_wave_length", &raysim::BaseProbe::get_wave_length, "Get wavelength")
      .def(
          "get_element_position",
          [](const raysim::BaseProbe& self, uint32_t element_idx) {
            float3 position;
            self.get_element_position(element_idx, position);
            return float3_to_numpy(position);
          },
          "Get position of an element by index")
      .def(
          "get_element_direction",
          [](const raysim::BaseProbe& self, uint32_t element_idx) {
            float3 direction;
            self.get_element_direction(element_idx, direction);
            return float3_to_numpy(direction);
          },
          "Get direction for an element by index");

  py::class_<raysim::CurvilinearProbe, raysim::BaseProbe>(m, "CurvilinearProbe", R"pbdoc(
Curvilinear ultrasound probe with elements positioned along a curved surface.
)pbdoc")
      .def(py::init<const raysim::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float>(),
           py::arg("pose") = raysim::Pose(),
           py::arg("num_elements_x") = 256,
           py::arg("sector_angle") = 73.0f,
           py::arg("radius") = 45.0f,
           py::arg("frequency") = 2.5f,
           py::arg("elevational_height") = 7.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 0.7f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f);

  py::class_<raysim::LinearArrayProbe, raysim::BaseProbe>(m, "LinearArrayProbe", R"pbdoc(
Linear array ultrasound probe with elements positioned in a straight line.
Elements emit parallel beams perpendicular to the face of the probe.
)pbdoc")
      .def(py::init<const raysim::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float>(),
           py::arg("pose") = raysim::Pose(),
           py::arg("num_elements_x") = 256,
           py::arg("width") = 60.0f,
           py::arg("frequency") = 5.0f,
           py::arg("elevational_height") = 5.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 1.0f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f);

  py::class_<raysim::PhasedArrayProbe, raysim::BaseProbe>(m, "PhasedArrayProbe", R"pbdoc(
Phased array ultrasound probe with elements positioned in a straight line.
Elements steer beams electronically to create a sector image from a small footprint.
)pbdoc")
      .def(py::init<const raysim::Pose&,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float>(),
           py::arg("pose") = raysim::Pose(),
           py::arg("num_elements_x") = 128,
           py::arg("width") = 20.0f,
           py::arg("sector_angle") = 90.0f,
           py::arg("frequency") = 3.5f,
           py::arg("elevational_height") = 5.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 1.0f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f);

  py::class_<raysim::IVUSProbe, raysim::BaseProbe>(m, "IVUSProbe", R"pbdoc(
IVUS (Intravascular Ultrasound) probe. Single rotating transducer at catheter center;
rays emanate radially over 360° for a cross-sectional vessel image.
)pbdoc")
      .def(py::init<const raysim::Pose&,
                    uint32_t,
                    float,
                    float,
                    uint32_t,
                    float,
                    float,
                    float,
                    float,
                    float>(),
           py::arg("pose") = raysim::Pose(),
           py::arg("num_angular_rays") = 256,
           py::arg("frequency") = 40.0f,
           py::arg("elevational_height") = 0.0f,
           py::arg("num_el_samples") = 1,
           py::arg("f_num") = 1.0f,
           py::arg("speed_of_sound") = 1.54f,
           py::arg("pulse_duration") = 2.0f,
           py::arg("element_radius_mm") = 0.6f,
           py::arg("focal_length_mm") = 4.0f)
      .def_property_readonly("element_radius_mm", &raysim::IVUSProbe::get_element_radius_mm)
      .def_property_readonly("focal_length_mm", &raysim::IVUSProbe::get_focal_length_mm);

  // Pass 2 — bind RingDownParams so callers can drive the ring-down injection
  // stage from Python (and from the YAML loader in raysim/config.py).
  py::class_<raysim::RingDownParams>(m, "RingDownParams", R"pbdoc(
        Calibrated catheter ring-down injection parameters.

        Attributes:
            enabled: Toggle. False (default) => no ring-down signal at all.
            amplitude: Envelope amplitude at the simulator reference gain. For
                decay == "measured", this is interpreted as the desired peak of
                the supplied waveform (the loader scales the template so its
                peak equals this amplitude); 0 keeps the template as-is.
            extent_mm: Hard cutoff in radial mm. Samples past this depth are
                untouched.
            decay: One of "exponential", "hanning", "measured".
            waveform: 1D float32 array of envelope-amplitude samples (only used
                when decay == "measured"). The simulator slices/zeros it to the
                computed sample count.
    )pbdoc")
      .def(py::init<>())
      .def_readwrite("enabled", &raysim::RingDownParams::enabled)
      .def_readwrite("amplitude", &raysim::RingDownParams::amplitude)
      .def_readwrite("extent_mm", &raysim::RingDownParams::extent_mm)
      .def_readwrite("decay", &raysim::RingDownParams::decay)
      .def_property(
          "waveform",
          [](const raysim::RingDownParams& self) {
            return py::array_t<float>(static_cast<py::ssize_t>(self.waveform.size()),
                                       self.waveform.data());
          },
          [](raysim::RingDownParams& self, py::array_t<float, py::array::c_style |
                                                                py::array::forcecast> array) {
            auto buf = array.request();
            if (buf.ndim != 1) {
              throw std::runtime_error("RingDownParams.waveform must be 1-D");
            }
            const float* ptr = static_cast<const float*>(buf.ptr);
            self.waveform.assign(ptr, ptr + buf.shape[0]);
          },
          "1D float32 array; envelope-amplitude samples for decay == 'measured'.")
      .def("__repr__", [](const raysim::RingDownParams& rd) {
        return std::string("RingDownParams(enabled=") + (rd.enabled ? "True" : "False") +
               ", amplitude=" + std::to_string(rd.amplitude) +
               ", extent_mm=" + std::to_string(rd.extent_mm) +
               ", decay='" + rd.decay +
               "', waveform.size=" + std::to_string(rd.waveform.size()) + ")";
      });

  // Bind TgcControlPoint (depth_cm, gain_db) so Python callers can build a TGC schedule
  // straight from a config file. A list/empty of these is exposed on SimParams.
  py::class_<raysim::TgcControlPoint>(m, "TgcControlPoint", R"pbdoc(
        Single piece-wise-linear TGC control point.

        Attributes:
            depth_cm: Depth of the control point [cm].
            gain_db: Gain at this depth [dB].
    )pbdoc")
      .def(py::init<>())
      .def(py::init([](float depth_cm, float gain_db) {
             return raysim::TgcControlPoint{depth_cm, gain_db};
           }),
           py::arg("depth_cm"),
           py::arg("gain_db"))
      .def_readwrite("depth_cm", &raysim::TgcControlPoint::depth_cm)
      .def_readwrite("gain_db", &raysim::TgcControlPoint::gain_db)
      .def("__repr__", [](const raysim::TgcControlPoint& cp) {
        return "TgcControlPoint(depth_cm=" + std::to_string(cp.depth_cm) +
               ", gain_db=" + std::to_string(cp.gain_db) + ")";
      });

  // Bind SimParams struct
  py::class_<raysim::RaytracingUltrasoundSimulator::SimParams>(m, "SimParams", R"pbdoc(
        Simulation parameters for ultrasound imaging.

        Parameters:
            - t_far: Maximum imaging depth [mm]. For IVUS, this is max radial depth (e.g. 5–10 mm).
            - buffer_size: Samples per ray (depth). Must match the build's Hilbert row length (e.g. 4096).
            - max_depth: Maximum ray reflection depth
            - min_intensity: Minimum ray intensity threshold
            - use_scattering: Enable scattering simulation
            - conv_psf: Enable PSF convolution
            - median_clip_filter: Enable median clip filter for speckle reduction
            - enable_cuda_timing: Enable CUDA timing measurements
            - write_debug_images: Enable debug image output
            - b_mode_size: B-mode image size (width, height). For IVUS unwrapped display: (angle pixels, depth pixels).
            - tgc_control_points: List[TgcControlPoint]; empty => use built-in probe-type default.
            - log_multiplier / log_floor: log_compression(out = mul * log10(max(in, floor))).
            - median_clip_size / median_clip_d_min_db / median_clip_d_max_db: median clip filter knobs.
            - scattering_resolution_mm: 0 => auto from probe type (10 mm IVUS / 50 mm general).
            - scatter_integral_scale: scale on the scatter line integral (0 = strict).
            - disable_scatter: skip scatter accumulation entirely.
    )pbdoc")
      .def(py::init<>())
      .def_readwrite("t_far",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::t_far,
                     "Maximum imaging depth [mm]. For IVUS: max radial depth.")
      .def_readwrite("buffer_size",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::buffer_size,
                     "Samples per ray (depth). Must match build's Hilbert row length (e.g. 4096).")
      .def_readwrite("max_depth",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::max_depth,
                     "Maximum ray reflection depth")
      .def_readwrite("min_intensity",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::min_intensity,
                     "Minimum ray intensity threshold")
      .def_readwrite("use_scattering",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::use_scattering,
                     "Enable scattering simulation")
      .def_readwrite("conv_psf",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::conv_psf,
                     "Enable PSF convolution")
      .def_readwrite("median_clip_filter",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::median_clip_filter,
                     "Enable median clip filter for speckle reduction")
      .def_readwrite("enable_cuda_timing",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::enable_cuda_timing,
                     "Enable CUDA timing measurements")
      .def_readwrite("write_debug_images",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::write_debug_images,
                     "Enable debug image output")
      .def_readwrite("contact_epsilon",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::contact_epsilon,
                     "Maximum distance for element activation [mm]")
      .def_readwrite("tgc_control_points",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::tgc_control_points,
                     "List of TgcControlPoint(depth_cm, gain_db); empty => probe-type default")
      .def_readwrite("log_multiplier",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::log_multiplier,
                     "log_compression multiplier (default 20.0)")
      .def_readwrite("log_floor",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::log_floor,
                     "log_compression floor (default 1e-19)")
      .def_readwrite("median_clip_size",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::median_clip_size,
                     "Median clip filter kernel size (default 5)")
      .def_readwrite("median_clip_d_min_db",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::median_clip_d_min_db,
                     "Median clip filter floor in dB (default -60)")
      .def_readwrite("median_clip_d_max_db",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::median_clip_d_max_db,
                     "Median clip filter ceiling in dB (default 0)")
      .def_readwrite("scattering_resolution_mm",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::scattering_resolution_mm,
                     "Scattering texture voxel size [mm]; 0 => auto by probe type")
      .def_readwrite("scatter_integral_scale",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::scatter_integral_scale,
                     "Scale factor on scatter line integral (default 40; 0 = strict)")
      .def_readwrite("disable_scatter",
                     &raysim::RaytracingUltrasoundSimulator::SimParams::disable_scatter,
                     "If true, skip scatter accumulation entirely")
      .def_readwrite(
          "ring_down",
          &raysim::RaytracingUltrasoundSimulator::SimParams::ring_down,
          "Pass 2: calibrated ring-down injection (RingDownParams). Enabled=False "
          "by default (truly silent lumen).")
      .def_readwrite(
          "dynamic_range_db",
          &raysim::RaytracingUltrasoundSimulator::SimParams::dynamic_range_db,
          "Pass 2: post-log display window width in dB. 0 (default) disables the "
          "stage so default callers keep the historical pure-log mapping.")
      .def_readwrite(
          "reject_db",
          &raysim::RaytracingUltrasoundSimulator::SimParams::reject_db,
          "Pass 2: reject floor in dB (display window). Inputs <= reject_db map to 0.")
      .def_property(
          "b_mode_size",
          [](raysim::RaytracingUltrasoundSimulator::SimParams& self) {
            return py::make_tuple(self.b_mode_size.x, self.b_mode_size.y);
          },
          [](raysim::RaytracingUltrasoundSimulator::SimParams& self, py::object size) {
            // Check if input is a numpy array
            if (py::isinstance<py::array>(size)) {
              auto array = size.cast<py::array>();
              if (array.ndim() != 1 || array.shape(0) != 2) {
                throw std::runtime_error("b_mode_size numpy array must have shape (2,)");
              }

              // Handle different numeric types in numpy arrays
              if (array.dtype().kind() == 'i') {
                auto buf = array.request();
                int* ptr = static_cast<int*>(buf.ptr);
                self.b_mode_size =
                    make_uint2(static_cast<uint32_t>(ptr[0]), static_cast<uint32_t>(ptr[1]));
              } else if (array.dtype().kind() == 'u') {
                auto buf = array.request();
                unsigned int* ptr = static_cast<unsigned int*>(buf.ptr);
                self.b_mode_size =
                    make_uint2(static_cast<uint32_t>(ptr[0]), static_cast<uint32_t>(ptr[1]));
              } else if (array.dtype().kind() == 'f') {
                auto buf = array.request();
                float* ptr = static_cast<float*>(buf.ptr);
                self.b_mode_size =
                    make_uint2(static_cast<uint32_t>(ptr[0]), static_cast<uint32_t>(ptr[1]));
              } else {
                throw std::runtime_error("Unsupported numpy array data type for b_mode_size");
              }
            }
            // Check if it's a tuple or list
            else if (py::isinstance<py::tuple>(size) || py::isinstance<py::list>(size)) {
              if (py::len(size) != 2) {
                throw std::runtime_error("b_mode_size tuple/list must have exactly 2 elements");
              }

              try {
                uint32_t width = py::cast<uint32_t>(size[py::int_(0)]);
                uint32_t height = py::cast<uint32_t>(size[py::int_(1)]);
                self.b_mode_size = make_uint2(width, height);
              } catch (const py::cast_error&) {
                throw std::runtime_error(
                    "b_mode_size elements must be convertible to unsigned integers");
              }
            } else {
              throw std::runtime_error(
                  "b_mode_size must be a tuple, list, or numpy array with 2 elements");
            }
          },
          "B-mode image size as (width, height), can be a tuple, list, or numpy array");

  // Bind RaytracingUltrasoundSimulator class
  py::class_<raysim::RaytracingUltrasoundSimulator>(m, "RaytracingUltrasoundSimulator", R"pbdoc(
        Main ultrasound simulation engine using raytracing.

        This class performs the actual ultrasound simulation by:
            1. Raytracing through the scene geometry
            2. Computing acoustic interactions
            3. Generating B-mode images
    )pbdoc")
      .def(py::init([](raysim::World* world, const raysim::Materials* materials) {
             try {
               spdlog::info("Creating RaytracingUltrasoundSimulator");
               auto simulator =
                   std::make_unique<raysim::RaytracingUltrasoundSimulator>(world, materials);
               spdlog::info("RaytracingUltrasoundSimulator created successfully");
               return simulator;
             } catch (const std::exception& e) {
               spdlog::error("Failed to create RaytracingUltrasoundSimulator: {}", e.what());
               throw;
             }
           }),
           R"pbdoc(
        Initialize a new simulator.

        Args:
            world: World instance containing the scene geometry
            materials: Materials instance with material definitions
      )pbdoc")
      .def("get_min_x",
           &raysim::RaytracingUltrasoundSimulator::get_min_x,
           "Get the minimum x value of the simulated region [mm]")
      .def("get_max_x",
           &raysim::RaytracingUltrasoundSimulator::get_max_x,
           "Get the maximum x value of the simulated region [mm]")
      .def("get_min_z",
           &raysim::RaytracingUltrasoundSimulator::get_min_z,
           "Get the minimum z value of the simulated region [mm]")
      .def("get_max_z",
           &raysim::RaytracingUltrasoundSimulator::get_max_z,
           "Get the maximum z value of the simulated region [mm]")
      .def(
          "simulate",
          [](raysim::RaytracingUltrasoundSimulator& self,
             const raysim::BaseProbe* probe,
             const raysim::RaytracingUltrasoundSimulator::SimParams& params) {
            try {
              spdlog::info("Starting simulation");
              auto result = self.simulate(probe, params);

              if (!result.b_mode) {
                spdlog::error("Simulation returned null b_mode");
                throw std::runtime_error("Simulation returned null b_mode");
              }

              const uint32_t elements = result.b_mode->get_size() / sizeof(float);
              auto host_data = std::unique_ptr<float[]>(new float[elements]);
              result.b_mode->download(host_data.get(), cudaStreamDefault);

              std::vector<ssize_t> shape = {static_cast<ssize_t>(params.b_mode_size.y),
                                            static_cast<ssize_t>(params.b_mode_size.x)};

              auto array = py::array_t<float>(shape, host_data.get());
              spdlog::info("Simulation completed successfully");
              return array;
            } catch (const std::exception& e) {
              spdlog::error("Exception in simulate: {}", e.what());
              throw;
            }
          },
          R"pbdoc(
        Perform ultrasound simulation.

        Args:
            probe: BaseProbe instance defining imaging parameters
            params: SimParams instance with simulation settings

        Returns:
            np.ndarray: B-mode ultrasound image
      )pbdoc");

  // Bind Hitable base class
  py::class_<raysim::Hitable>(m, "Hitable", R"pbdoc(
        Base class for objects that can be hit by rays.

        This is the parent class for Mesh and Sphere objects.
    )pbdoc")
      .def(
          "get_aabb_min",
          [](const raysim::Hitable& self) { return float3_to_numpy(self.get_aabb_min()); },
          "Get the minimum point of the object's bounding box")
      .def(
          "get_aabb_max",
          [](const raysim::Hitable& self) { return float3_to_numpy(self.get_aabb_max()); },
          "Get the maximum point of the object's bounding box");

  // Bind Mesh class
  py::class_<raysim::Mesh, raysim::Hitable>(m, "Mesh", R"pbdoc(
        Represents a 3D mesh object for ultrasound simulation.

        The mesh is loaded from an OBJ file and assigned a material.
    )pbdoc")
      .def(py::init([](const std::string& file_name, uint32_t material_id) {
             spdlog::info(
                 "Creating Mesh from file: {} with material_id: {}", file_name, material_id);
             try {
               auto mesh = std::make_unique<raysim::Mesh>(file_name, material_id);
               spdlog::info("Mesh created successfully");
               return mesh;
             } catch (const std::exception& e) {
               spdlog::error("Failed to create Mesh: {}", e.what());
               throw;
             }
           }),
           py::arg("file_name"),
           py::arg("material_id"),
           R"pbdoc(
        Initialize a new mesh.

        Args:
            file_name (str): Path to OBJ file
            material_id (int): Material index from Materials.get_index()
      )pbdoc");

  // Bind Sphere class
  py::class_<raysim::Sphere, raysim::Hitable>(m, "Sphere", R"pbdoc(
        Represents a sphere object for ultrasound simulation.

        A simple geometric primitive useful for testing and basic simulations.
    )pbdoc")
      .def(py::init([](py::array_t<float> center, float radius, uint32_t material_id) {
             // Convert NumPy array to float3
             float3 center_float3 = numpy_to_float3(center);
             return std::make_unique<raysim::Sphere>(center_float3, radius, material_id);
           }),
           py::arg("center"),
           py::arg("radius"),
           py::arg("material_id"),
           R"pbdoc(
        Initialize a new sphere.

        Args:
            center (np.ndarray): 3D center position [x, y, z]
            radius (float): Sphere radius
            material_id (int): Material index from Materials.get_index()
      )pbdoc");
}
