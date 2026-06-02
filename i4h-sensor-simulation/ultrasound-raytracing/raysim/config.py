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
    # Pass 5f -- IVUS angular ray super-sampling.
    #
    # When > 1, the IVUS raygen kernel fires K = ``ivus_rays_per_scanline``
    # sub-rays per scanline at deterministic sub-bin angular offsets and
    # accumulates into the same scanline buffer with weight 1/K. This
    # forward-models the bench's finite beam width at the raycasting stage
    # so sub-wavelength wire scatterers (e.g. the 0.0635 mm-radius B2
    # tungsten wires whose angular subtense is ~0.4° at r=10 mm, much
    # smaller than the 1.4° scanline pitch) are captured by at least one
    # sub-ray instead of being hit-or-missed by accidental alignment.
    #
    # K=1 (default) is the legacy single-ray-per-scanline behaviour;
    # K=8 is recommended for B2 wire-phantom Tier 1 PSF tests. Speckle
    # renders are insensitive to K (the integral over a uniform medium is
    # K-invariant) so the only cost is render time (~linear in K for the
    # OptiX raygen stage).
    ivus_rays_per_scanline: int = 1
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
class EnvelopeNoiseConfig:
    """Pass 20 — post-envelope additive Gaussian noise.

    Adds ``N(mean, sigma**2)`` per envelope pixel at the post-Hilbert /
    pre-LPF stage, so the post-Hilbert low-pass smooths the noise with a
    ~1-wavelength radial kernel and produces the bench's coarse-grained
    speckle texture. ``mean`` seeds a baseline noise floor for anechoic
    materials (e.g. water) where the upstream envelope is ~0; the post-
    log mean palette of an anechoic region is then determined entirely
    by ``mean`` and the calibrated ``log_floor`` / ``log_multiplier``.

    ``sigma`` controls per-pixel CV of the noise floor independently of
    the Rayleigh fixed-point that constrains the pre-PSF RF noise stage
    (``NoiseConfig.sigma``). Use both knobs together to match the bench
    distribution's mean and CV.

    Both default to 0 (no-op). When ``processing.noise.sigma`` is also 0
    the simulator's noise pipeline is fully disabled. The post-envelope
    stage is intended to *replace* the pre-PSF RF noise stage on the
    PV .035 calibration; the pre-PSF stage is kept for backward
    compatibility with other YAMLs.

    Calibrate via
    ``instrument-calibration/p035_visions/derive_envelope_noise.py``.

    ``reference_gain_db``: the ``processing.gain_db`` value at which
    ``mean`` / ``sigma`` were measured against the bench. The simulator
    multiplies the effective noise mean / sigma by
    ``10 ** ((sim_gain_db - reference_gain_db) / 20)`` at simulate-time so
    the noise floor tracks the receive-chain gain (Pass 20b). For the PV
    .035 calibration the noise was measured against a slider-50 bench
    capture, so ``reference_gain_db = processing.gain_db - 4.0`` (4 dB
    below the simulator's slider-54 reference). Default 0 keeps the
    legacy (un-scaled) behaviour for callers that don't set it.
    """
    mean: float = 0.0
    sigma: float = 0.0
    reference_gain_db: float = 0.0
    # Pass 28i -- when true, the C++ envelope-noise stage multiplies both
    # `mean` and `sigma` per sample by the cached TGC linear-gain curve
    # (normalised to 1.0 at r = 0).  This models bench analog electronic
    # noise being amplified by the receive chain's TGC schedule, so the
    # post-envelope noise floor inherits the TGC's depth-dependent gain.
    # Default false preserves the legacy depth-flat behaviour.  See
    # `SimParams::envelope_noise_apply_tgc_depth_scaling` and the
    # `volcano_s5i.yaml` envelope_noise block for the calibration rationale.
    apply_tgc_depth_scaling: bool = False


@dataclass
class LateralPsfKernelConfig:
    """Pass 5d — lateral-PSF kernel-type switch.

    ``type`` selects the depth-dependent lateral PSF used by the IVUS pipeline:

    * ``"gaussian_beam"`` (default) — legacy fixed-focus Gaussian beam,
      ``sigma_mm(r) = w0 * sqrt(1 + ((r - z_f) / z_R)^2)`` with a pre-focal
      clamp. Pre-Pass-5d behaviour; keeps existing calibrations unchanged.
    * ``"constant_angular"`` — every depth bin uses the same angular spread
      ``sigma_theta_rad`` (so ``sigma_bins`` is independent of depth).
      Motivated by the s5i synthetic-aperture probe's bench data showing
      a constant ~7.3° angular FWHM across r ∈ [4, 26] mm. Calibrate
      ``sigma_theta_rad`` against the median bench angular FWHM via
      ``instrument-calibration/p035_visions/derive_lateral_psf_sigma_theta.py``.
    """
    type: str = "gaussian_beam"  # gaussian_beam | constant_angular
    sigma_theta_rad: float = 0.054164779787691845  # ~3.103 deg; PV .035 calibrated


@dataclass
class CatheterConfig:
    """Catheter sheath geometry (Pass 6 v2).

    `dead_zone_mm` is the radial extent (mm) inside which any acquired signal
    is blocked by the catheter wall. The simulator zeros the final palette
    buffer for r < dead_zone_mm so the inner zone renders as solid black --
    matching the bench, where the catheter sheath physically blocks signal
    acquisition for the inner ~1.4-1.9 mm depending on probe.
    """
    dead_zone_mm: float = 0.0


@dataclass
class RingDownConfig:
    # When False (the default), the simulator emits no ring-down signal at all.
    # When True, it adds the catheter ring-down on top of scatter.
    #
    # AR-mode policy (corrected 2026-05-12 after ivus_test_0508 paired-
    # capture cross-check):
    #
    # The s5i AR-on/off state lives in private tag 0x00291007 (NOT
    # 0x00291006, which is a capability flag and is always 1). 18 of 19
    # P_035 frames are AR-OFF and the bench template
    # `ringdown_template_g54_d60.npy` is therefore the RAW ring-down, not
    # the AR residual. Set `subtract_reference = False` to match P_035
    # and the wire-phantom frames in ivus_test_0508; flip to True to
    # render a clinical-default AR-on image, in which case the template
    # at `waveform_path` should be an AR-on residual template (TBD; will
    # be derived from CASE0000 paired captures in ivus_test_0508).
    enabled: bool = False
    amplitude: float = 0.0
    extent_mm: float = 0.5
    decay: str = "exponential"  # exponential | hanning | measured
    waveform_path: Optional[str] = None
    subtract_reference: bool = False
    # Pitch (mm per sample) of the file at `waveform_path`. Calibration
    # templates are typically saved at the device's *display* pitch (e.g.
    # 0.12 mm/pixel for the PV .035), which is much coarser than the
    # simulator's RF rate (c / fs / 2 ≈ 0.0193 mm/sample at 40 MHz). The
    # loader linearly resamples the template from this pitch onto the sim
    # grid before handing it to SimParams. Set to None (default) to assume
    # the template is already at the simulator's sample pitch.
    template_pitch_mm: Optional[float] = None
    # Palette value to subtract from the loaded template before converting
    # palette → envelope amplitude. The calibrated PV .035 template is the
    # median over angle of a ring-down ROI, so the baseline (~speckle floor)
    # represents the device's no-signal value, not real ring-down energy.
    # Subtracting it makes the "absent ring-down" parts of the template map
    # to amp ≈ 0 instead of a constant background. Default 0.0 = no
    # subtraction (matches a synthetic template with a true zero baseline).
    template_speckle_floor_palette: float = 0.0


@dataclass
class ProcessingConfig:
    scattering_resolution_mm: float = 10.0
    scatter_integral_scale: float = 40.0
    tgc_control_points: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 2.0)]
    )
    log_multiplier: float = 20.0
    # Pass 3b: log_floor is the calibration anchor (amp == log_floor maps to
    # palette 0). Default 1.0 makes the K2v2 mapping reduce to
    # `log_multiplier * log10(amp)` for amp >= 1, which matches the legacy
    # `examples/ivus_example.py` MIN_VAL/MAX_VAL = (-60, 0) display window
    # at log_multiplier = 20. Calibrated YAMLs override.
    log_floor: float = 1.0
    median_clip: MedianClipConfig = field(default_factory=MedianClipConfig)
    # Pass 3a: reference gain applied to the envelope buffer between Hilbert
    # and log compression: amp <- amp * 10^(gain_db / 20). 0.0 = no-op
    # (default); calibrated YAMLs set this to the renderer-specific offset
    # that puts the simulator's envelope onto the bench's reference scale
    # (see SimParams::gain_db doc and calibration_delta.md).
    gain_db: float = 0.0
    # Pass 3b: post-log display-window palette anchors. With both at 0.0
    # (default) the display-window stage is skipped entirely, so YAMLs that
    # omit these fields keep the historical pure-log palette mapping.
    # Calibrated configs set both to non-zero (e.g. PV .035:
    # reject_palette=11, saturation_palette=239 from gain_lut.json).
    reject_palette: float = 0.0
    saturation_palette: float = 0.0
    # Pass 20 — softplus scale applied to the reject floor in the display-
    # window kernel. When > 0 the floor uses a smooth (softplus) blend
    # instead of a hard `max(palette, reject)` clamp, removing the
    # spurious histogram spike that the hard clamp creates under post-
    # envelope Gaussian noise with tails below the floor. Default 0 keeps
    # the hard-clamp behaviour.
    reject_palette_softness: float = 0.0
    # --- Future ---
    compression_lut: Optional[str] = None
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    # Pass 20 — post-envelope additive Gaussian noise; see EnvelopeNoiseConfig.
    envelope_noise: EnvelopeNoiseConfig = field(default_factory=EnvelopeNoiseConfig)
    catheter: CatheterConfig = field(default_factory=CatheterConfig)
    ring_down: RingDownConfig = field(default_factory=RingDownConfig)
    # Pass 5d — lateral-PSF kernel selector. Default keeps the pre-Pass-5d
    # gaussian_beam kernel so existing YAMLs that omit this block render
    # identically. Set ``type: constant_angular`` in the YAML to opt in to
    # the SA-aware constant-angular Gaussian kernel.
    lateral_psf_kernel: LateralPsfKernelConfig = field(
        default_factory=LateralPsfKernelConfig
    )


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
    "processing.compression_lut",
    # Pass 2 wired the ring-down stage. Pass 3a wired `processing.gain_db`
    # (reference gain) and Pass 3b replaced the dB-shift display window with
    # `processing.reject_palette` / `processing.saturation_palette` (direct
    # palette clamp). Pass 6 wired `processing.noise.{type, sigma}` (the
    # additive RF noise floor stage; see SimParams.noise_sigma).
    # `ring_down.subtract_reference` is informational only (the device
    # already does the subtraction; we model the residual) and stays out of
    # the wiring.
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
    # Path the config was loaded from (set by from_yaml/from_json); used to
    # resolve workspace-relative asset paths like ring_down.waveform_path.
    # Excluded from to_dict() / dataclass equality below.
    source_path: Optional[Path] = field(default=None, repr=False, compare=False)

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
        cfg = cls.from_dict(data)
        cfg.source_path = Path(path).resolve()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "IvusSimConfig":
        with open(path, "r") as f:
            cfg = cls.from_dict(json.load(f))
        cfg.source_path = Path(path).resolve()
        return cfg

    # ---- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # `source_path` is loader-only metadata; it should never be round-tripped
        # into a written config (and `Path` is not JSON-serializable).
        d.pop("source_path", None)
        return d

    # ---- Path resolution ---------------------------------------------------

    def _resolve_asset_path(self, p: str | Path) -> Path:
        """Resolve `p` (typically from a YAML field like ring_down.waveform_path).

        Resolution rules, in order:
          1. Absolute path -> returned unchanged.
          2. If the config was loaded from disk and the loader can find a
             "workspace root" by walking parents of `source_path` until a
             sibling directory named `instrument-calibration` exists, resolve
             relative to that workspace root. This is what calibrated YAMLs
             expect (paths like `P_035_PointScatter/derived/...`).
          3. Otherwise resolve relative to `source_path`'s parent.
          4. Last resort: relative to current working directory.
        """
        path = Path(p)
        if path.is_absolute():
            return path
        if self.source_path is not None:
            yaml_dir = self.source_path.parent
            for ancestor in (yaml_dir, *yaml_dir.parents):
                candidate = ancestor / "instrument-calibration"
                if candidate.is_dir():
                    return (ancestor / path).resolve()
            return (yaml_dir / path).resolve()
        return Path.cwd() / path

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
        params.ivus_rays_per_scanline = max(1, int(self.sim.ivus_rays_per_scanline))

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

        # Pass 6 — additive Gaussian RF noise floor.
        #
        # The bench's calibrated RF noise std (in RF amplitude units at the
        # reference gain) is forwarded directly; `SimParams.noise_sigma <= 0`
        # is the no-op default. The YAML `processing.noise.type` is currently
        # informational: the runtime equivalence between adding Gaussian RF
        # noise pre-Hilbert (what the simulator does today) and observing
        # Rayleigh noise on the post-Hilbert envelope (what the calibration
        # measured) means a single Gaussian-RF stage reproduces both bench
        # noise families with the given sigma. A future pass can add a
        # per-type dispatch if the calibration sheet starts distinguishing
        # them in a way that matters.
        if proc.noise.type.lower() not in {"gaussian", "rayleigh", "none", ""}:
            raise ValueError(
                f"Unsupported processing.noise.type {proc.noise.type!r}; "
                "expected one of 'gaussian', 'rayleigh', 'none'."
            )
        params.noise_sigma = float(proc.noise.sigma) if proc.noise.type.lower() != "none" else 0.0

        # Pass 20 — post-envelope additive Gaussian noise (mean + sigma).
        # See `EnvelopeNoiseConfig` for the model semantics. Both default to
        # 0 (no-op), preserving prior YAMLs that omit the block.
        params.envelope_noise_mean = float(proc.envelope_noise.mean)
        params.envelope_noise_sigma = float(proc.envelope_noise.sigma)
        # Pass 20b — gain-scaling reference (gain_db at which the noise was
        # calibrated). The simulator multiplies the effective mean / sigma
        # by 10^((sim_gain_db - reference_gain_db) / 20) so the noise floor
        # tracks the receive-chain gain.
        params.envelope_noise_reference_gain_db = float(
            proc.envelope_noise.reference_gain_db)
        params.envelope_noise_apply_tgc_depth_scaling = bool(
            proc.envelope_noise.apply_tgc_depth_scaling)

        # Pass 20 — softplus reject-floor softness (palette units). 0 = hard clamp.
        params.reject_palette_softness = float(proc.reject_palette_softness)

        # Pass 6 v2 — catheter sheath dead-zone mask.
        #
        # `processing.catheter.dead_zone_mm` (default 0) zeros the inner
        # radial samples of the final palette buffer so the catheter region
        # renders solid black, matching the bench (the catheter wall blocks
        # signal acquisition for ~1.4-1.9 mm depending on probe).
        params.catheter_dead_zone_mm = float(proc.catheter.dead_zone_mm)

        # ---- Pass 2: ring-down injection ------------------------------------
        # Off by default (RingDownConfig.enabled = False) => no signal at all.
        # When enabled and decay == "measured", load the palette template from
        # `waveform_path` and convert to envelope amplitude using the calibrated
        # log_multiplier (cf. volcano_s5i.yaml: `amp = 10^(palette / log_mult)`).
        rd = proc.ring_down
        params.ring_down.enabled = bool(rd.enabled)
        params.ring_down.amplitude = float(rd.amplitude)
        params.ring_down.extent_mm = float(rd.extent_mm)
        params.ring_down.decay = str(rd.decay)
        if rd.enabled and rd.decay == "measured" and rd.waveform_path:
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - numpy is a hard dep
                raise ImportError(
                    "numpy is required to load measured ring-down templates."
                ) from exc
            wf_path = self._resolve_asset_path(rd.waveform_path)
            if not wf_path.is_file():
                raise FileNotFoundError(
                    f"Ring-down waveform template not found: {wf_path} "
                    f"(from ring_down.waveform_path={rd.waveform_path!r})"
                )
            palette = np.load(wf_path).astype(np.float32, copy=False)
            if palette.ndim != 1:
                raise ValueError(
                    f"Ring-down waveform must be 1-D; got shape {palette.shape} from {wf_path}."
                )
            log_mult = float(proc.log_multiplier)
            if log_mult <= 0.0:
                raise ValueError(
                    "Ring-down: log_multiplier must be > 0 to convert palette template to amplitude."
                )

            # Subtract the speckle / reject floor so "no ring-down" samples map
            # to amp 0 instead of a constant background. Clip to >= 0 so floor-
            # clipped samples (which read below the device's reject palette)
            # don't produce phantom signal.
            palette_excess = np.maximum(
                palette - float(rd.template_speckle_floor_palette), 0.0
            ).astype(np.float32, copy=False)

            # Convert the palette excess to envelope-amplitude units. After this
            # conversion any sample where palette_excess == 0 (no ring-down at
            # this depth) produces amp = 10^0 = 1. We *want* those samples to
            # contribute zero, so subtract 1 and clip again — that gives a true
            # zero baseline matching the C++ side's "ring-down injects on top of
            # whatever else is in the scanline" semantics.
            envelope_amp = np.power(10.0, palette_excess / log_mult).astype(
                np.float32, copy=False
            )
            envelope_amp = np.maximum(envelope_amp - 1.0, 0.0).astype(
                np.float32, copy=False
            )

            # Resample the template from its source pitch onto the simulator's
            # depth-sample pitch so a template peak at 1.8 mm in the file lands
            # at 1.8 mm in the simulator output (and not at sample index 1.8).
            template_pitch = rd.template_pitch_mm
            if template_pitch is not None and template_pitch > 0.0:
                # Simulator depth-sample pitch in the scanline buffer:
                # OptiX raygen maps the radial range [0, t_far_mm] onto
                # `buffer_size` samples (see optix_trace.cu offset formula),
                # so pitch = t_far_mm / buffer_size mm/sample.
                # NOTE: this is *not* `c / sampling_freq_mhz / 2` — the
                # legacy `sampling_freq_mhz` field is informational only and
                # the OptiX path doesn't use it for sample placement.
                sim_pitch_mm = float(self.sim.t_far_mm) / float(self.sim.buffer_size)
                src_n = envelope_amp.shape[0]
                src_extent_mm = src_n * float(template_pitch)
                dst_n = int(round(src_extent_mm / sim_pitch_mm))
                if dst_n > 1 and src_n > 1:
                    src_grid = np.arange(src_n, dtype=np.float64) * float(template_pitch)
                    dst_grid = np.arange(dst_n, dtype=np.float64) * sim_pitch_mm
                    envelope_amp = np.interp(
                        dst_grid, src_grid, envelope_amp.astype(np.float64)
                    ).astype(np.float32, copy=False)
            params.ring_down.waveform = envelope_amp

        # ---- Pass 3b: display window (palette clamp) ------------------------
        # Calibrated PV .035 YAML sets reject_palette = 11 and
        # saturation_palette = 239 from gain_lut.json. With both at 0.0
        # (default) the SimParams kernel skips the display-window stage and
        # keeps the historical pure-log mapping.
        params.reject_palette = float(proc.reject_palette)
        params.saturation_palette = float(proc.saturation_palette)

        # ---- Pass 3: reference gain ----------------------------------------
        # Calibrated YAMLs set this to the renderer-specific scalar that puts
        # the simulator's envelope onto the bench's reference scale at the
        # bench's reference gain (slider 54 for the PV .035). Default 0.0 is
        # a no-op so YAMLs that omit the field keep the historical behaviour.
        params.gain_db = float(proc.gain_db)

        # ---- Pass 5d: lateral-PSF kernel-type switch -----------------------
        # 0 = gaussian_beam (legacy default), 1 = constant_angular (SA-aware).
        # Validate the string before mapping so a typo at the YAML layer is
        # caught here instead of silently falling back to the default.
        kernel_type_str = str(proc.lateral_psf_kernel.type).lower()
        kernel_type_map = {"gaussian_beam": 0, "constant_angular": 1}
        if kernel_type_str not in kernel_type_map:
            raise ValueError(
                f"Unsupported processing.lateral_psf_kernel.type "
                f"{proc.lateral_psf_kernel.type!r}; expected one of "
                f"{sorted(kernel_type_map)}."
            )
        params.lateral_psf_kernel_type = kernel_type_map[kernel_type_str]
        params.lateral_psf_sigma_theta_rad = float(
            proc.lateral_psf_kernel.sigma_theta_rad
        )

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
    envelope_noise = d.pop("envelope_noise", None) if isinstance(d, dict) else None
    catheter = d.pop("catheter", None) if isinstance(d, dict) else None
    ring_down = d.pop("ring_down", None) if isinstance(d, dict) else None
    lateral_psf_kernel = (
        d.pop("lateral_psf_kernel", None) if isinstance(d, dict) else None
    )
    if "tgc_control_points" in d and d["tgc_control_points"] is not None:
        d["tgc_control_points"] = [tuple(pt) for pt in d["tgc_control_points"]]
    return ProcessingConfig(
        median_clip=MedianClipConfig(**median) if isinstance(median, dict) else MedianClipConfig(),
        noise=NoiseConfig(**noise) if isinstance(noise, dict) else NoiseConfig(),
        envelope_noise=(
            EnvelopeNoiseConfig(**envelope_noise)
            if isinstance(envelope_noise, dict)
            else EnvelopeNoiseConfig()
        ),
        catheter=CatheterConfig(**catheter) if isinstance(catheter, dict) else CatheterConfig(),
        ring_down=RingDownConfig(**ring_down) if isinstance(ring_down, dict) else RingDownConfig(),
        lateral_psf_kernel=(
            LateralPsfKernelConfig(**lateral_psf_kernel)
            if isinstance(lateral_psf_kernel, dict)
            else LateralPsfKernelConfig()
        ),
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
