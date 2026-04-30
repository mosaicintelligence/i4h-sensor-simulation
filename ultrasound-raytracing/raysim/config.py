# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""
Config schema and loader for the IVUS / ultrasound raytracing simulator.

The schema mirrors `IVUS Simulation Parameters - Sheet1.csv`. It supports all
"Config" fields (currently wired into the simulator) and "Future" fields
(part of the schema but not yet plumbed into the C++ code — the loader records
them and warns when they are non-default).

Typical use:

    from raysim.config import IvusSimConfig
    cfg = IvusSimConfig.from_yaml("configs/volcano_s5i.yaml")
    probe = cfg.to_probe()
    sim_params = cfg.to_sim_params()
    pending = cfg.pending_fields()  # list of (path, value) for Future knobs

Loading does NOT import the C++ bindings, so the schema is usable for tooling
even on machines without a CUDA/OptiX build of `raysim`.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Probe
# -----------------------------------------------------------------------------


@dataclass
class PoseConfig:
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ProbeConfig:
    type: str = "ivus"  # ivus | curvilinear | linear_array | phased_array
    frequency_mhz: float = 20.0
    pulse_duration_cycles: float = 2.0
    num_scanlines: int = 256
    element_radius_mm: float = 0.6
    focal_length_mm: float = 4.0
    elevational_height_mm: float = 0.0
    num_elevational_samples: int = 1
    speed_of_sound_mm_per_us: float = 1.54
    f_num: float = 1.0
    width_mm: float = 0.0
    pose: PoseConfig = field(default_factory=PoseConfig)
    # --- Future ---
    impulse_response_path: Optional[str] = None
    synthetic_aperture: bool = False


# -----------------------------------------------------------------------------
# Sim
# -----------------------------------------------------------------------------


@dataclass
class SimConfig:
    t_far_mm: float = 5.0
    buffer_size: int = 4096
    b_mode_size: tuple[int, int] = (256, 1024)
    max_reflection_depth: int = 15
    min_intensity: float = 1.0e-3
    contact_epsilon_mm: float = 0.0
    conv_psf: bool = True
    median_clip_filter: bool = True
    # --- Future ---
    sampling_freq_mhz: float = 40.0


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------


@dataclass
class MedianClipConfig:
    size: int = 5
    d_min_db: float = -60.0
    d_max_db: float = 0.0


@dataclass
class NoiseConfig:
    type: str = "gaussian"  # gaussian | rayleigh | none
    sigma: float = 0.0


@dataclass
class RingDownConfig:
    amplitude: float = 0.0
    extent_mm: float = 0.5
    decay: str = "exponential"  # exponential | hanning | measured
    waveform_path: Optional[str] = None
    subtract_reference: bool = True


@dataclass
class ProcessingConfig:
    scattering_resolution_mm: float = 10.0
    scatter_integral_scale: float = 40.0
    tgc_control_points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 2.0)]
    )
    log_multiplier: float = 20.0
    log_floor: float = 1.0e-19
    median_clip: MedianClipConfig = field(default_factory=MedianClipConfig)
    # --- Future ---
    gain_db: float = 0.0
    dynamic_range_db: float = 60.0
    reject_db: float = -80.0
    compression_lut: Optional[str] = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    ring_down: RingDownConfig = field(default_factory=RingDownConfig)


# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------


@dataclass
class MaterialConfig:
    name: str
    impedance_mrayl: float
    speed_of_sound_m_per_s: float
    attenuation_db_per_cm_mhz: float
    mu0: float = 0.0
    mu1: float = 0.0
    sigma: float = 0.0
    specularity: float = 0.0


# -----------------------------------------------------------------------------
# Future-field registry: every entry is a path into IvusSimConfig that maps to a
# parameter that exists in the schema but is not yet consumed by the simulator.
# `pending_fields()` walks this list and returns the ones whose value differs
# from the default so callers know what is still TODO. Keep this in sync with
# the "Future" rows in `IVUS Simulation Parameters - Sheet1.csv`.
# -----------------------------------------------------------------------------

_FUTURE_PATHS: tuple[str, ...] = (
    "probe.impulse_response_path",
    "probe.synthetic_aperture",
    "sim.sampling_freq_mhz",
    "processing.gain_db",
    "processing.dynamic_range_db",
    "processing.reject_db",
    "processing.compression_lut",
    "processing.noise.type",
    "processing.noise.sigma",
    "processing.ring_down.amplitude",
    "processing.ring_down.extent_mm",
    "processing.ring_down.decay",
    "processing.ring_down.waveform_path",
    "processing.ring_down.subtract_reference",
)

# Likewise: Config rows that are in the schema but not yet exposed via SimParams
# bindings. The loader will use whatever is configured but log a one-time notice.
# As of Pass 1 of the simulator wiring all of the previously partially-wired
# processing parameters (TGC, log compression, median clip, scatter scale) are
# now plumbed straight through to SimParams, so this list is empty. Add new
# entries here whenever a Config-row knob lands in the YAML schema before its
# C++ binding does.
_PARTIALLY_WIRED_PATHS: tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# Top-level config
# -----------------------------------------------------------------------------


@dataclass
class IvusSimConfig:
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    materials: list[MaterialConfig] = field(default_factory=list)

    # ---- Constructors ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IvusSimConfig":
        probe = _build_probe(data.get("probe", {}))
        sim = _build_sim(data.get("sim", {}))
        processing = _build_processing(data.get("processing", {}))
        materials = [MaterialConfig(**m) for m in data.get("materials", [])]
        return cls(probe=probe, sim=sim, processing=processing, materials=materials)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IvusSimConfig":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for from_yaml(); pip install pyyaml or use from_json()."
            ) from exc
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "IvusSimConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # ---- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("PyYAML is required for to_yaml(); pip install pyyaml.") from exc
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    # ---- raysim object builders -------------------------------------------

    def to_probe(self):
        """Build the appropriate raysim.BaseProbe subclass from the probe config."""
        rs = _import_raysim()
        pose = rs.Pose(
            tuple(self.probe.pose.position_mm),
            tuple(self.probe.pose.rotation_rad),
        )
        kind = self.probe.type.lower()
        if kind == "ivus":
            return rs.IVUSProbe(
                pose,
                int(self.probe.num_scanlines),
                float(self.probe.frequency_mhz),
                float(self.probe.elevational_height_mm),
                int(self.probe.num_elevational_samples),
                float(self.probe.f_num),
                float(self.probe.speed_of_sound_mm_per_us),
                float(self.probe.pulse_duration_cycles),
                float(self.probe.element_radius_mm),
                float(self.probe.focal_length_mm),
            )
        if kind == "curvilinear":
            return rs.CurvilinearProbe(
                pose,
                int(self.probe.num_scanlines),
                float(self.probe.frequency_mhz),
                float(self.probe.elevational_height_mm),
                int(self.probe.num_elevational_samples),
                float(self.probe.f_num),
                float(self.probe.speed_of_sound_mm_per_us),
                float(self.probe.pulse_duration_cycles),
            )
        if kind == "linear_array":
            return rs.LinearArrayProbe(
                pose,
                int(self.probe.num_scanlines),
                float(self.probe.width_mm),
                float(self.probe.frequency_mhz),
                float(self.probe.elevational_height_mm),
                int(self.probe.num_elevational_samples),
                float(self.probe.f_num),
                float(self.probe.speed_of_sound_mm_per_us),
                float(self.probe.pulse_duration_cycles),
            )
        if kind == "phased_array":
            return rs.PhasedArrayProbe(
                pose,
                int(self.probe.num_scanlines),
                float(self.probe.width_mm),
                90.0,  # sector_angle (not in the unified schema; use phased default)
                float(self.probe.frequency_mhz),
                float(self.probe.elevational_height_mm),
                int(self.probe.num_elevational_samples),
                float(self.probe.f_num),
                float(self.probe.speed_of_sound_mm_per_us),
                float(self.probe.pulse_duration_cycles),
            )
        raise ValueError(f"Unknown probe type: {self.probe.type!r}")

    def to_sim_params(self):
        """Build SimParams from the YAML config.

        Pass 1 of the simulator wiring exposes the bucket-B processing knobs
        (TGC schedule, log compression, median clip filter, scatter scale) as
        SimParams fields. Anything still missing from the C++ pipeline is tracked
        by ``pending_fields()`` and surfaced via ``warn_about_unwired()``.
        """
        rs = _import_raysim()
        params = rs.SimParams()

        # ---- sim block ------------------------------------------------------
        params.t_far = float(self.sim.t_far_mm)
        params.buffer_size = int(self.sim.buffer_size)
        params.b_mode_size = (int(self.sim.b_mode_size[0]), int(self.sim.b_mode_size[1]))
        params.max_depth = int(self.sim.max_reflection_depth)
        params.min_intensity = float(self.sim.min_intensity)
        params.contact_epsilon = float(self.sim.contact_epsilon_mm)
        params.conv_psf = bool(self.sim.conv_psf)
        params.median_clip_filter = bool(self.sim.median_clip_filter)

        # ---- processing block ----------------------------------------------
        proc = self.processing

        # TGC: convert the [(depth_cm, gain_db), ...] tuples into the bound
        # TgcControlPoint type. An empty list keeps the simulator on its
        # probe-type default schedule.
        if proc.tgc_control_points:
            params.tgc_control_points = [
                rs.TgcControlPoint(float(d_cm), float(g_db))
                for d_cm, g_db in proc.tgc_control_points
            ]

        params.log_multiplier = float(proc.log_multiplier)
        params.log_floor = float(proc.log_floor)

        params.median_clip_size = int(proc.median_clip.size)
        params.median_clip_d_min_db = float(proc.median_clip.d_min_db)
        params.median_clip_d_max_db = float(proc.median_clip.d_max_db)

        # scattering_resolution_mm of 0 in SimParams means "auto from probe
        # type"; YAML defaults to 10.0 (the historical IVUS value), so we just
        # forward whatever the user asked for.
        params.scattering_resolution_mm = float(proc.scattering_resolution_mm)
        params.scatter_integral_scale = float(proc.scatter_integral_scale)

        return params

    # ---- Diagnostics -------------------------------------------------------

    def pending_fields(self) -> list[tuple[str, Any]]:
        """Return Future fields whose value differs from the default.

        These are configured but not yet honored by the simulator.
        """
        defaults = IvusSimConfig()
        return [(p, _get_path(self, p)) for p in _FUTURE_PATHS if _get_path(self, p) != _get_path(defaults, p)]

    def partially_wired_fields(self) -> list[tuple[str, Any]]:
        """Return Config fields that exist in the schema but are still hard-coded
        in the C++ pipeline (not yet exposed via SimParams).

        After Pass 1 of the simulator wiring this list is empty; it remains here
        so future schema additions can be flagged before their bindings land.
        """
        defaults = IvusSimConfig()
        return [
            (p, _get_path(self, p))
            for p in _PARTIALLY_WIRED_PATHS
            if _get_path(self, p) != _get_path(defaults, p)
        ]

    def warn_about_unwired(self) -> None:
        """Emit a single warning summarizing fields that are configured but not
        yet acted on by the simulator. Safe to call from example scripts."""
        pending = self.pending_fields()
        partial = self.partially_wired_fields()
        if pending:
            warnings.warn(
                "These Future config fields are set but NOT yet wired into the "
                "simulator (will be ignored): "
                + ", ".join(f"{p}={v!r}" for p, v in pending),
                stacklevel=2,
            )
        if partial:
            warnings.warn(
                "These Config fields are set but currently HARD-CODED in the C++ "
                "pipeline (override has no effect until the SimParams bindings "
                "are extended): "
                + ", ".join(f"{p}={v!r}" for p, v in partial),
                stacklevel=2,
            )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _build_probe(d: dict[str, Any]) -> ProbeConfig:
    pose = d.pop("pose", None) if isinstance(d, dict) else None
    pose_cfg = PoseConfig(**pose) if isinstance(pose, dict) else PoseConfig()
    return ProbeConfig(pose=pose_cfg, **d)


def _build_sim(d: dict[str, Any]) -> SimConfig:
    if "b_mode_size" in d:
        d["b_mode_size"] = tuple(d["b_mode_size"])
    return SimConfig(**d)


def _build_processing(d: dict[str, Any]) -> ProcessingConfig:
    median = d.pop("median_clip", None) if isinstance(d, dict) else None
    noise = d.pop("noise", None) if isinstance(d, dict) else None
    ring_down = d.pop("ring_down", None) if isinstance(d, dict) else None
    if "tgc_control_points" in d and d["tgc_control_points"] is not None:
        d["tgc_control_points"] = [tuple(pt) for pt in d["tgc_control_points"]]
    return ProcessingConfig(
        median_clip=MedianClipConfig(**median) if isinstance(median, dict) else MedianClipConfig(),
        noise=NoiseConfig(**noise) if isinstance(noise, dict) else NoiseConfig(),
        ring_down=RingDownConfig(**ring_down) if isinstance(ring_down, dict) else RingDownConfig(),
        **d,
    )


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _import_raysim():
    try:
        from . import ray_sim_python as rs  # type: ignore[attr-defined]
    except ImportError:
        try:
            import raysim as rs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "raysim C++ bindings not available. Build the package "
                "(see ultrasound-raytracing/README.md) before calling to_probe()/to_sim_params()."
            ) from exc
    return rs
