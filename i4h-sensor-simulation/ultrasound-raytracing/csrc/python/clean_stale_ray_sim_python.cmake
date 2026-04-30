# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remove extension modules built for a different Python interpreter before linking.
# Invoked as: cmake -DRAYSIM_PY_DIR=<absolute-path-to-raysim-pkg> -P clean_stale_ray_sim_python.cmake
#
# Prevents multiple ray_sim_python.cpython-*.{so,pyd} files from coexisting in the source tree.
# scikit-build-core's editable install records one path at configure time; stale artifacts can
# point the linker at the wrong ABI (e.g. 3.10 vs 3.13).

if(NOT DEFINED RAYSIM_PY_DIR OR "${RAYSIM_PY_DIR}" STREQUAL "")
  message(FATAL_ERROR "clean_stale_ray_sim_python.cmake: RAYSIM_PY_DIR is not set")
endif()

file(GLOB stale_so "${RAYSIM_PY_DIR}/ray_sim_python.cpython*.so")
file(GLOB stale_pyd "${RAYSIM_PY_DIR}/ray_sim_python.cp*.pyd")
set(stale ${stale_so} ${stale_pyd})
foreach(f ${stale})
  message(STATUS "Removing stale ray_sim_python extension: ${f}")
  file(REMOVE "${f}")
endforeach()
