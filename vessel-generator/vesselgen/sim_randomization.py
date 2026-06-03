"""Domain-randomization parameters for IVUS simulation (ML training datasets).

Tier-1 knobs (gain, tissue acoustics, AR on/off) are sampled around the
calibrated operating point in ``instrument-calibration/p035_visions/volcano_s5i.yaml``.
PSF geometry, log compression, and image grid size stay fixed.

Tier-2 knobs (TGC deep scale, ring-down amplitude, scattering resolution,
``t_far_mm`` FOV) add secondary appearance diversity without changing PSF
geometry or the display LUT chain.

See the ML Dataset Parameter Randomization plan for rationale and ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

ArtifactReduction = Literal["on", "off"]
ExtravascularPreset = Literal["fat_like", "muscle_like", "mixed"]

# Literature / material.cpp reference presets (attenuation dB/cm/MHz).
_FAT = dict(
    impedance_mrayl=1.38,
    speed_of_sound_m_per_s=1450.0,
    attenuation_db_per_cm_mhz=0.63,
    mu0=1.0,
    mu1=0.0,
    sigma=1.0,
    specularity=1.0,
)
_MUSCLE = dict(
    impedance_mrayl=1.70,
    speed_of_sound_m_per_s=1580.0,
    attenuation_db_per_cm_mhz=1.09,
    mu0=0.5,
    mu1=0.8,
    sigma=0.4,
    specularity=1.0,
)
_WALL_DEFAULT = dict(
    impedance_mrayl=1.82,
    speed_of_sound_m_per_s=1571.0,
    attenuation_db_per_cm_mhz=1.0,
    mu0=0.5,
    mu1=0.6,
    sigma=0.35,
    specularity=1.0,
)
_LUMEN_DEFAULT = dict(
    impedance_mrayl=1.68,
    speed_of_sound_m_per_s=1584.0,
    attenuation_db_per_cm_mhz=0.2,
    mu0=0.1,
    mu1=0.1,
    sigma=0.1,
    specularity=0.0,
)


@dataclass
class UniformRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class MaterialDraw:
    """One sampled acoustic material configuration."""

    extravascular_preset: ExtravascularPreset
    vessel_wall: dict[str, float]
    extravascular: dict[str, float]
    lumen: dict[str, float]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tissue_preset": self.extravascular_preset,
            "materials": {
                "vessel_wall": dict(self.vessel_wall),
                "extravascular": dict(self.extravascular),
                "lumen": dict(self.lumen),
            },
        }


@dataclass
class VesselSimDraw:
    """Per-vessel simulator knobs (FOV, speckle grain, ring-down strength)."""

    t_far_mm: float
    scattering_resolution_mm: float
    ring_down_amplitude: float
    ring_down_amplitude_scale: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "t_far_mm": self.t_far_mm,
            "scattering_resolution_mm": self.scattering_resolution_mm,
            "ring_down_amplitude": self.ring_down_amplitude,
            "ring_down_amplitude_scale": self.ring_down_amplitude_scale,
        }


@dataclass
class FrameSimDraw:
    """Per-frame simulator knobs."""

    gain_slider: float
    gain_db: float
    gain_db_offset_from_calibrated: float
    artifact_reduction: ArtifactReduction
    ring_down_enabled: bool
    ring_down_template: str
    frame_seed: int
    noise_seed: int
    tgc_deep_gain_scale: float = 1.0
    tgc_deep_gain_db: float = 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "gain_slider": self.gain_slider,
            "gain_db": self.gain_db,
            "gain_db_offset_from_calibrated": self.gain_db_offset_from_calibrated,
            "artifact_reduction": self.artifact_reduction,
            "ring_down_enabled": self.ring_down_enabled,
            "ring_down_template": self.ring_down_template,
            "frame_seed": self.frame_seed,
            "noise_seed": self.noise_seed,
            "tgc_deep_gain_scale": self.tgc_deep_gain_scale,
            "tgc_deep_gain_db": self.tgc_deep_gain_db,
        }


@dataclass
class SimRandomizationConfig:
    """Sampling distributions for simulator domain randomization."""

    ref_gain_slider: float = 54.0
    gain_slider_range: tuple[float, float] = (44.0, 68.0)
    ar_on_probability: float = 0.5

    wall_attenuation: UniformRange = field(
        default_factory=lambda: UniformRange(0.6, 1.4)
    )
    wall_mu0: UniformRange = field(default_factory=lambda: UniformRange(0.25, 0.75))
    wall_sigma: UniformRange = field(default_factory=lambda: UniformRange(0.15, 0.55))
    wall_impedance: UniformRange = field(
        default_factory=lambda: UniformRange(1.75, 1.90)
    )
    wall_speed_of_sound: UniformRange = field(
        default_factory=lambda: UniformRange(1540.0, 1600.0)
    )
    wall_specularity: UniformRange = field(default_factory=lambda: UniformRange(0.5, 2.0))

    lumen_attenuation: UniformRange = field(
        default_factory=lambda: UniformRange(0.15, 0.35)
    )
    lumen_mu0: UniformRange = field(default_factory=lambda: UniformRange(0.05, 0.20))
    lumen_sigma: UniformRange = field(default_factory=lambda: UniformRange(0.05, 0.20))

    extravascular_fat_weight: float = 0.4
    extravascular_muscle_weight: float = 0.4
    extravascular_mixed_weight: float = 0.2

    side_branch_ostium_bias_prob: float = 0.35
    side_branch_ostium_arclength_frac: float = 0.25

    max_saturation_fraction: float = 0.35
    saturation_palette: float = 239.0

    ar_off_waveform_path: str = (
        "ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_off_g50_d60.npy"
    )
    ar_off_speckle_floor_palette: float = 28.5
    ar_on_waveform_path: str = (
        "ivus_test_0508/raw/c_take2_water/derived/ringdown/ringdown_on_g50_d60.npy"
    )
    ar_on_speckle_floor_palette: float = 11.0
    ar_template_path: str = (
        "ivus_test_0508/raw/c_take2_water/derived/ar_template/ar_template_canonical.npy"
    )
    ar_template_speckle_floor_palette: float = 11.0

    enable_tier2: bool = True
    tgc_deep_depth_cm: float = 3.0
    tgc_deep_gain_scale_range: tuple[float, float] = (0.8, 1.2)
    ring_down_amplitude_scale_range: tuple[float, float] = (0.85, 1.15)
    scattering_resolution_mm_range: tuple[float, float] = (8.0, 14.0)
    t_far_mm_choices: tuple[float, ...] = (17.5, 20.0, 30.0)

    def sample_vessel_sim(
        self,
        rng: np.random.Generator,
        *,
        base_ring_down_amplitude: float,
        base_t_far_mm: float,
        base_scattering_resolution_mm: float,
    ) -> VesselSimDraw:
        if not self.enable_tier2:
            return VesselSimDraw(
                t_far_mm=float(base_t_far_mm),
                scattering_resolution_mm=float(base_scattering_resolution_mm),
                ring_down_amplitude=float(base_ring_down_amplitude),
                ring_down_amplitude_scale=1.0,
            )
        t_far = float(rng.choice(self.t_far_mm_choices))
        scatter_res = float(rng.uniform(*self.scattering_resolution_mm_range))
        rd_scale = float(rng.uniform(*self.ring_down_amplitude_scale_range))
        return VesselSimDraw(
            t_far_mm=t_far,
            scattering_resolution_mm=scatter_res,
            ring_down_amplitude=float(base_ring_down_amplitude) * rd_scale,
            ring_down_amplitude_scale=rd_scale,
        )

    def sample_vessel_materials(self, rng: np.random.Generator) -> MaterialDraw:
        preset = self._sample_extravascular_preset(rng)
        extravascular = self._extravascular_from_preset(preset, rng)
        vessel_wall = {
            "impedance_mrayl": self.wall_impedance.sample(rng),
            "speed_of_sound_m_per_s": self.wall_speed_of_sound.sample(rng),
            "attenuation_db_per_cm_mhz": self.wall_attenuation.sample(rng),
            "mu0": self.wall_mu0.sample(rng),
            "mu1": _WALL_DEFAULT["mu1"],
            "sigma": self.wall_sigma.sample(rng),
            "specularity": self.wall_specularity.sample(rng),
        }
        lumen = {
            "impedance_mrayl": _LUMEN_DEFAULT["impedance_mrayl"],
            "speed_of_sound_m_per_s": _LUMEN_DEFAULT["speed_of_sound_m_per_s"],
            "attenuation_db_per_cm_mhz": self.lumen_attenuation.sample(rng),
            "mu0": self.lumen_mu0.sample(rng),
            "mu1": _LUMEN_DEFAULT["mu1"],
            "sigma": self.lumen_sigma.sample(rng),
            "specularity": _LUMEN_DEFAULT["specularity"],
        }
        return MaterialDraw(
            extravascular_preset=preset,
            vessel_wall=vessel_wall,
            extravascular=extravascular,
            lumen=lumen,
        )

    def _sample_extravascular_preset(self, rng: np.random.Generator) -> ExtravascularPreset:
        weights = np.array(
            [
                self.extravascular_fat_weight,
                self.extravascular_muscle_weight,
                self.extravascular_mixed_weight,
            ],
            dtype=float,
        )
        weights /= weights.sum()
        idx = int(rng.choice(3, p=weights))
        return ("fat_like", "muscle_like", "mixed")[idx]

    def _extravascular_from_preset(
        self, preset: ExtravascularPreset, rng: np.random.Generator
    ) -> dict[str, float]:
        if preset == "fat_like":
            return dict(_FAT)
        if preset == "muscle_like":
            return dict(_MUSCLE)
        t = float(rng.uniform(0.0, 1.0))
        mixed: dict[str, float] = {}
        for key in _FAT:
            mixed[key] = (1.0 - t) * _FAT[key] + t * _MUSCLE[key]
        return mixed

    def sample_frame_sim(
        self,
        rng: np.random.Generator,
        *,
        base_gain_db: float,
        frame_seed: int,
        noise_seed: int,
        slider_to_db,
        base_tgc_deep_gain_db: float,
        artifact_reduction: ArtifactReduction | None = None,
    ) -> FrameSimDraw:
        slider = float(rng.uniform(*self.gain_slider_range))
        offset_db = float(slider_to_db(slider, self.ref_gain_slider))
        gain_db = float(base_gain_db + offset_db)
        if artifact_reduction is None:
            ar_on = bool(rng.random() < self.ar_on_probability)
        else:
            ar_on = artifact_reduction == "on"
        if self.enable_tier2:
            tgc_scale = float(rng.uniform(*self.tgc_deep_gain_scale_range))
        else:
            tgc_scale = 1.0
        # ``artifact_reduction: "on"`` => bright catheter ring-down visible (raw template).
        # ``"off"`` => ring-down injection disabled (clinical AR-on inner zone).
        return FrameSimDraw(
            gain_slider=slider,
            gain_db=gain_db,
            gain_db_offset_from_calibrated=offset_db,
            artifact_reduction="on" if ar_on else "off",
            ring_down_enabled=ar_on,
            ring_down_template="raw" if ar_on else "disabled",
            frame_seed=frame_seed,
            noise_seed=noise_seed,
            tgc_deep_gain_scale=tgc_scale,
            tgc_deep_gain_db=float(base_tgc_deep_gain_db) * tgc_scale,
        )


def deep_tgc_gain_db(cfg, *, deep_depth_cm: float = 3.0) -> float:
    """Return the calibrated deep TGC control-point gain (dB)."""
    for depth_cm, gain_db in cfg.processing.tgc_control_points:
        if abs(float(depth_cm) - deep_depth_cm) < 1e-3:
            return float(gain_db)
    if cfg.processing.tgc_control_points:
        return float(cfg.processing.tgc_control_points[-1][1])
    return 13.0


def copy_tgc_control_points(control_points):
    """Copy TGC control points for per-frame scaling."""
    import raysim as rs

    return [
        rs.TgcControlPoint(float(cp.depth_cm), float(cp.gain_db))
        for cp in control_points
    ]


def scale_tgc_deep_control_point(
    control_points,
    *,
    deep_depth_cm: float,
    scale: float,
):
    """Scale only the deep TGC knot gain, leaving shallow points fixed."""
    import raysim as rs

    scaled = []
    for cp in control_points:
        depth = float(cp.depth_cm)
        gain = float(cp.gain_db)
        if abs(depth - deep_depth_cm) < 1e-3:
            gain *= float(scale)
        scaled.append(rs.TgcControlPoint(depth, gain))
    return scaled


def apply_vessel_sim_draw(cfg, sim_params, vessel_draw: VesselSimDraw) -> None:
    """Apply per-vessel Tier-2 knobs to config + SimParams."""
    cfg.sim.t_far_mm = float(vessel_draw.t_far_mm)
    sim_params.t_far = float(vessel_draw.t_far_mm)
    sim_params.scattering_resolution_mm = float(vessel_draw.scattering_resolution_mm)
    sim_params.ring_down.amplitude = float(vessel_draw.ring_down_amplitude)


def apply_material_draw(materials, draw: MaterialDraw) -> None:
    """Push sampled acoustics into a raysim Materials instance."""
    for name, params in (
        ("vessel_wall", draw.vessel_wall),
        ("extravascular", draw.extravascular),
        ("lumen", draw.lumen),
    ):
        materials.update_material(
            name,
            params["impedance_mrayl"],
            params["attenuation_db_per_cm_mhz"],
            params["speed_of_sound_m_per_s"],
            params["mu0"],
            params["mu1"],
            params["sigma"],
            params["specularity"],
        )


def palette_template_to_envelope(
    palette: np.ndarray,
    *,
    log_multiplier: float,
    speckle_floor_palette: float,
) -> np.ndarray:
    """Convert a bench palette-domain ring-down template to envelope amplitude."""
    palette = np.asarray(palette, dtype=np.float32).reshape(-1)
    if log_multiplier <= 0.0:
        raise ValueError("log_multiplier must be positive")
    palette_excess = np.maximum(palette - float(speckle_floor_palette), 0.0).astype(
        np.float32, copy=False
    )
    envelope_amp = np.power(10.0, palette_excess / log_multiplier).astype(np.float32)
    return np.maximum(envelope_amp - 1.0, 0.0).astype(np.float32, copy=False)


def resample_envelope_to_sim_grid(
    envelope_amp: np.ndarray,
    *,
    template_pitch_mm: float,
    t_far_mm: float,
    buffer_size: int,
) -> np.ndarray:
    """Resample a 1-D envelope template onto the simulator depth grid."""
    if template_pitch_mm <= 0.0 or len(envelope_amp) < 2:
        return envelope_amp.astype(np.float32, copy=False)
    sim_pitch_mm = float(t_far_mm) / float(buffer_size)
    src_extent_mm = len(envelope_amp) * float(template_pitch_mm)
    dst_n = int(round(src_extent_mm / sim_pitch_mm))
    if dst_n <= 1:
        return envelope_amp.astype(np.float32, copy=False)
    src_grid = np.arange(len(envelope_amp), dtype=np.float64) * float(template_pitch_mm)
    dst_grid = np.arange(dst_n, dtype=np.float64) * sim_pitch_mm
    return np.interp(dst_grid, src_grid, envelope_amp.astype(np.float64)).astype(
        np.float32
    )


@dataclass
class RingDownWaveforms:
    """Pre-built ring-down envelope waveforms.

    ``raw`` — bright catheter ring-down (``artifact_reduction: "on"``).
    ``suppressed`` — AR-subtracted residual (``artifact_reduction: "off"``).
    """

    raw: np.ndarray
    suppressed: np.ndarray
    suppressed_fallback_disabled: bool = False

    # Back-compat aliases
    @property
    def ar_off(self) -> np.ndarray:
        return self.raw

    @property
    def ar_on(self) -> np.ndarray:
        return self.suppressed

    @property
    def ar_on_fallback_disabled(self) -> bool:
        return self.suppressed_fallback_disabled


def load_ringdown_waveforms(cfg, rand_cfg: SimRandomizationConfig) -> RingDownWaveforms:
    """Load and resample bench ring-down templates for AR mode switching."""
    log_mult = float(cfg.processing.log_multiplier)
    template_pitch = float(cfg.processing.ring_down.template_pitch_mm or 0.12)
    t_far = float(cfg.sim.t_far_mm)
    buffer_size = int(cfg.sim.buffer_size)

    def _to_sim_envelope(palette: np.ndarray, *, floor: float) -> np.ndarray:
        env = palette_template_to_envelope(
            palette,
            log_multiplier=log_mult,
            speckle_floor_palette=floor,
        )
        return resample_envelope_to_sim_grid(
            env,
            template_pitch_mm=template_pitch,
            t_far_mm=t_far,
            buffer_size=buffer_size,
        )

    off_path = cfg._resolve_asset_path(rand_cfg.ar_off_waveform_path)
    off_palette = np.load(off_path).astype(np.float32, copy=False)
    ar_off = _to_sim_envelope(
        off_palette, floor=rand_cfg.ar_off_speckle_floor_palette
    )

    suppressed_fallback_disabled = False
    suppressed = None
    ar_template_path = cfg._resolve_asset_path(rand_cfg.ar_template_path)
    if ar_template_path.is_file():
        ar_palette = np.load(ar_template_path).astype(np.float32, copy=False)
        ar_template_env = _to_sim_envelope(
            ar_palette, floor=rand_cfg.ar_template_speckle_floor_palette
        )
        suppressed = np.maximum(ar_off - ar_template_env, 0.0).astype(np.float32, copy=False)
    else:
        on_path = cfg._resolve_asset_path(rand_cfg.ar_on_waveform_path)
        if on_path.is_file():
            on_palette = np.load(on_path).astype(np.float32, copy=False)
            suppressed = _to_sim_envelope(
                on_palette, floor=rand_cfg.ar_on_speckle_floor_palette
            )
        else:
            suppressed = np.zeros_like(ar_off)
            suppressed_fallback_disabled = True

    return RingDownWaveforms(
        raw=ar_off,
        suppressed=suppressed,
        suppressed_fallback_disabled=suppressed_fallback_disabled,
    )


def apply_frame_sim_params(
    sim_params,
    frame_draw: FrameSimDraw,
    *,
    ringdown: RingDownWaveforms,
    base_tgc_control_points,
    tgc_deep_depth_cm: float = 3.0,
) -> None:
    """Apply a per-frame draw onto a SimParams object (mutates in place)."""
    sim_params.gain_db = float(frame_draw.gain_db)
    sim_params.frame_seed = int(frame_draw.frame_seed)
    sim_params.noise_seed = int(frame_draw.noise_seed)
    sim_params.tgc_control_points = scale_tgc_deep_control_point(
        base_tgc_control_points,
        deep_depth_cm=tgc_deep_depth_cm,
        scale=frame_draw.tgc_deep_gain_scale,
    )
    if frame_draw.ring_down_enabled and frame_draw.artifact_reduction == "on":
        sim_params.ring_down.enabled = True
        sim_params.ring_down.waveform = ringdown.raw
    else:
        sim_params.ring_down.enabled = False
        sim_params.ring_down.waveform = np.zeros(1, dtype=np.float32)


def saturation_fraction(image: np.ndarray, *, saturation_palette: float) -> float:
    """Fraction of finite pixels at or above the display saturation level."""
    valid = np.isfinite(image)
    if not valid.any():
        return 1.0
    return float((image[valid] >= saturation_palette - 0.5).mean())
