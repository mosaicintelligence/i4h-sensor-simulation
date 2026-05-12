"""Non-circular lumen cross-sections.

The lumen radius at angle ``theta`` and station index ``i`` is::

    r(theta, i) = mean_radius(i) * (1 + sum_k a_k * cos(k * theta + phi_k(i)))

with two key properties:

1. The amplitudes ``a_k`` are scaled so that the *total* radial deviation,
   ``sum_k a_k``, never exceeds ``max_perturbation_frac``. This guarantees
   the contour never degenerates (no negative radii) regardless of mode
   superposition.
2. The phases ``phi_k(i)`` drift smoothly with arclength so that adjacent
   cross-sections are similar but never identical, producing
   axially-coherent shapes (lumen does not "spin" frame-to-frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vesselgen.config import CrossSectionConfig


@dataclass
class CrossSectionField:
    """Lumen radii on a (station, angle) grid for one branch.

    Attributes:
        thetas:       (M,) angular sample positions in radians, [0, 2pi).
        radii:        (N, M) lumen radius at each (station, angle).
        mean_radius:  (N,) mean radius along arclength (the taper).
    """

    thetas: np.ndarray
    radii: np.ndarray
    mean_radius: np.ndarray

    @property
    def n_stations(self) -> int:
        return int(self.radii.shape[0])

    @property
    def n_angles(self) -> int:
        return int(self.radii.shape[1])

    def contour(self, station_index: int) -> np.ndarray:
        """(M, 2) contour in the local cross-section plane (x = normal, y = binormal).

        Closed implicitly by repeating angles[0] at the end if the caller
        needs an explicitly closed polyline.
        """
        r = self.radii[station_index]
        x = r * np.cos(self.thetas)
        y = r * np.sin(self.thetas)
        return np.stack([x, y], axis=1)


def _mean_radius_profile(
    config: CrossSectionConfig, length_mm: float, n_stations: int
) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n_stations)
    r0 = float(config.mean_radius_mm)
    r1 = float(config.distal_radius_mm) if config.distal_radius_mm is not None else r0
    if config.taper_profile == "constant":
        return np.full(n_stations, 0.5 * (r0 + r1))
    if config.taper_profile == "linear":
        return r0 + (r1 - r0) * s
    # smooth: smoothstep blend
    smooth = s * s * (3 - 2 * s)
    return r0 + (r1 - r0) * smooth


def build_cross_sections(
    config: CrossSectionConfig,
    length_mm: float,
    n_stations: int,
    rng: np.random.Generator,
) -> CrossSectionField:
    """Sample a lumen cross-section field over ``n_stations`` along arclength.

    Parameters
    ----------
    config:
        Cross-section parameters.
    length_mm:
        Total arclength of the branch (used for phase-drift accumulation).
    n_stations:
        Number of axial stations (must match the parent centerline).
    rng:
        Per-branch random generator for reproducibility.
    """
    modes = np.asarray(config.perturbation_modes, dtype=int)
    n_modes = len(modes)

    raw_amplitudes = 1.0 / np.power(modes, config.perturbation_decay)
    raw_amplitudes = raw_amplitudes / raw_amplitudes.sum()
    base_amplitudes = raw_amplitudes * config.max_perturbation_frac

    sign_flips = rng.choice([-1.0, 1.0], size=n_modes)
    base_amplitudes = base_amplitudes * sign_flips

    initial_phases = rng.uniform(0.0, 2 * np.pi, size=n_modes)

    arclengths = np.linspace(0.0, length_mm, n_stations)
    drift_per_mode = config.phase_drift_per_mm * (1.0 + 0.3 * (rng.random(n_modes) - 0.5))
    phases = initial_phases[None, :] + drift_per_mode[None, :] * arclengths[:, None]

    thetas = np.linspace(0.0, 2.0 * np.pi, config.n_angles, endpoint=False)
    perturbation = np.zeros((n_stations, config.n_angles))
    for k_idx, k in enumerate(modes):
        a = base_amplitudes[k_idx]
        ph = phases[:, k_idx][:, None]
        perturbation += a * np.cos(k * thetas[None, :] + ph)

    perturbation = np.clip(perturbation, -0.45, 0.45)

    mean_radius = _mean_radius_profile(config, length_mm, n_stations)
    radii = mean_radius[:, None] * (1.0 + perturbation)
    radii = np.maximum(radii, 0.05 * mean_radius[:, None])

    return CrossSectionField(thetas=thetas, radii=radii, mean_radius=mean_radius)
