"""Per-angle, per-arclength wall thickness model.

Adds a thickness ``t(theta, s)`` to a lumen contour to produce the outer
(adventitia) contour. Thickness varies smoothly around the circumference
and along the length, so walls clearly read as "thicker on one side than
another" at a given station and the asymmetry rotates slowly with depth.

Pass the same ``CrossSectionField`` (lumen) plus this wall field to the
sweep code; it will build a closed lumen surface and a closed outer
surface that nests around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vesselgen.config import WallConfig
from vesselgen.cross_section import CrossSectionField


@dataclass
class WallField:
    """Per-angle wall thickness on the same (station, angle) grid as the lumen."""

    thicknesses: np.ndarray  # (N, M)
    mean_thickness: float

    @property
    def n_stations(self) -> int:
        return int(self.thicknesses.shape[0])

    @property
    def n_angles(self) -> int:
        return int(self.thicknesses.shape[1])


def build_wall(
    config: WallConfig,
    lumen: CrossSectionField,
    length_mm: float,
    rng: np.random.Generator,
) -> WallField:
    """Sample a wall-thickness field on the lumen's (station, angle) grid."""
    modes = np.asarray(config.perturbation_modes, dtype=int)
    n_modes = len(modes)
    raw = 1.0 / np.power(modes, config.perturbation_decay)
    raw = raw / raw.sum()
    base = raw * config.max_perturbation_frac
    signs = rng.choice([-1.0, 1.0], size=n_modes)
    base = base * signs

    initial_phases = rng.uniform(0.0, 2 * np.pi, size=n_modes)
    drift = config.phase_drift_per_mm * (1.0 + 0.3 * (rng.random(n_modes) - 0.5))
    arclengths = np.linspace(0.0, length_mm, lumen.n_stations)
    phases = initial_phases[None, :] + drift[None, :] * arclengths[:, None]

    perturbation = np.zeros((lumen.n_stations, lumen.n_angles))
    for k_idx, k in enumerate(modes):
        ph = phases[:, k_idx][:, None]
        perturbation += base[k_idx] * np.cos(k * lumen.thetas[None, :] + ph)

    perturbation = np.clip(perturbation, -0.85, 0.85)
    thicknesses = config.mean_thickness_mm * (1.0 + perturbation)
    thicknesses = np.maximum(thicknesses, config.min_thickness_mm)
    return WallField(thicknesses=thicknesses, mean_thickness=config.mean_thickness_mm)
