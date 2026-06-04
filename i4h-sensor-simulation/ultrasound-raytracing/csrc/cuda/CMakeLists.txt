# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Include GenHeaderFromTextFile module needed for processing PTX files
include(GenHeaderFromTextFile)

########################################
# Generate PTX code from OptiX kernels
########################################

add_library(ray_sim_optix OBJECT
    optix_trace.cu
    )

set_property(TARGET ray_sim_optix
    PROPERTY CUDA_PTX_COMPILATION ON
    )

# PTX output supports only a SINGLE virtual architecture, so override the
# project-wide CMAKE_CUDA_ARCHITECTURES (which may be a list when multi-arch
# binaries are requested, e.g. inside the production Docker image). PTX is
# forward-compatible: the OptiX runtime JIT-compiles it for the running GPU
# at first use, so pinning to a low virtual arch keeps the artifact portable.
# sm_70 is the lowest arch supported by cuFFTDx (the rest of the build), so
# we use compute_70 here for consistency.
set_target_properties(ray_sim_optix PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "70-virtual"
    )

target_link_libraries(ray_sim_optix
    PRIVATE
        optix_sdk
    )

target_include_directories(ray_sim_optix
    PRIVATE
        ${optix-toolkit_SOURCE_DIR}/ShaderUtil/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

target_compile_options(ray_sim_optix
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${RAY_SIM_CUDA_FLAGS}>
    )

################################################
# Build the CUDA components
################################################

add_library(ray_sim_cuda STATIC
    cuda_algorithms.cu
    cuda_helper.cpp
    optix_helper.cpp
    )

set_property(TARGET ray_sim_cuda
    PROPERTY POSITION_INDEPENDENT_CODE ON
    )

set_target_properties(ray_sim_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    )

target_compile_options(ray_sim_cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${RAY_SIM_CUDA_FLAGS}>
    )

add_dependencies(ray_sim_cuda ray_sim_optix)


set(ADD_STRING_LITERAL_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/add_string_literal.sh)

gen_header_from_text_file(TARGET ray_sim_cuda
    FILE_PATH ${CMAKE_BINARY_DIR}/csrc/cuda/CMakeFiles/ray_sim_optix.dir/optix_trace.ptx
    )

target_include_directories(ray_sim_cuda
    PRIVATE
        ${stb_SOURCE_DIR}
        ${optix-toolkit_SOURCE_DIR}/ShaderUtil/include
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

target_link_libraries(ray_sim_cuda
    PRIVATE
        mathdx::cufftdx
    PUBLIC
        optix_sdk
        CUDA::cudart_static
        spdlog::spdlog
    )
