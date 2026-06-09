# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ray-Based Ultrasound Simulator

A GPU-accelerated ultrasound simulation package that uses ray-tracing for realistic acoustic behavior.

The native simulator (``raysim.cuda`` / the ``ray_sim_python`` extension)
requires a CUDA+OptiX build. Host-only modules — currently ``raysim.config``
and ``raysim.docker`` — are importable without it, so users can dispatch
simulations to the raysim Docker image from machines that don't have a GPU.
"""

from .config import IvusSimConfig

__version__ = "0.1.0"

__all__ = ["IvusSimConfig"]

try:
    from .ray_sim_python import (
        CurvilinearProbe,
        IVUSProbe,
        LinearArrayProbe,
        Materials,
        PhasedArrayProbe,
        Pose,
        RaytracingUltrasoundSimulator,
        RingDownParams,
        SimParams,
        TgcControlPoint,
        World,
    )

    __all__.extend([
        "CurvilinearProbe",
        "IVUSProbe",
        "LinearArrayProbe",
        "Materials",
        "PhasedArrayProbe",
        "Pose",
        "RaytracingUltrasoundSimulator",
        "RingDownParams",
        "SimParams",
        "TgcControlPoint",
        "World",
    ])
except ImportError as _native_import_error:  # pragma: no cover - host-only install
    import warnings as _warnings

    _warnings.warn(
        "raysim native extension (ray_sim_python) not available: "
        f"{_native_import_error}. Host-only modules (raysim.config, "
        "raysim.docker) remain importable; building/rendering requires "
        "the CUDA/OptiX simulator (use the raysim Docker image to render "
        "from a host without a local CUDA build).",
        stacklevel=2,
    )
