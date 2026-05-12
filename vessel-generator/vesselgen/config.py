"""Generation parameters for vessel segments.

These dataclasses are the public schema for everything the generator does.
Each field has a sensible default for a peripheral-vessel-sized segment
(typical lumen radius 2-5 mm, length 20-40 mm). For batch generation,
``GenerationConfig`` carries distributions over these parameters and
``GenerationConfig.sample()`` draws a concrete ``VesselConfig`` from them.

All distances are in millimetres. All angles are in degrees unless the
field name says otherwise. Coordinate convention matches the simulator:
vessel axis along Y, cross-sections in the xz plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Centerline
# ---------------------------------------------------------------------------


@dataclass
class CenterlineConfig:
    """Per-branch centerline geometry.

    For frame-level training the catheter is sampled inside the lumen at
    arbitrary positions, so the centerline is primarily used to define the
    local "vessel axis" (tangent direction) so that probe tilts can be
    sampled relative to it.

    A v1 branch centerline is a straight line of length ``length_mm`` with
    its origin at ``origin`` and direction ``direction`` (unit vector). The
    parametric curvature_amplitude_mm and curvature_period_mm fields are
    reserved for v2 (smooth 3D curvature); they are accepted but currently
    ignored by the sweep code.
    """

    length_mm: float = 30.0
    origin: tuple[float, float, float] = (0.0, -15.0, 0.0)
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    n_stations: int = 64

    curvature_amplitude_mm: float = 0.0  # reserved, v2
    curvature_period_mm: float = 0.0     # reserved, v2

    def __post_init__(self) -> None:
        if self.length_mm <= 0:
            raise ValueError("length_mm must be positive")
        if self.n_stations < 8:
            raise ValueError("n_stations must be >= 8 to support sweep meshing")
        d = np.asarray(self.direction, dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise ValueError("direction must be a non-zero vector")
        self.direction = tuple((d / n).tolist())


# ---------------------------------------------------------------------------
# Lumen cross-section
# ---------------------------------------------------------------------------


@dataclass
class CrossSectionConfig:
    """Non-circular lumen cross-section model.

    The lumen radius at angle ``theta`` and arclength ``s`` is

        r(theta, s) = r_mean(s) * (1 + sum_k a_k(s) cos(k theta + phi_k(s)))

    where the modes ``k`` come from ``perturbation_modes`` and the
    amplitudes ``a_k`` are drawn so that the maximum total perturbation
    stays within ``max_perturbation_frac`` of the mean radius. Phases drift
    slowly in arclength so adjacent cross-sections are similar but never
    identical.
    """

    mean_radius_mm: float = 3.0
    """Mean lumen radius at the proximal end of the branch."""

    distal_radius_mm: Optional[float] = None
    """Mean lumen radius at the distal end. ``None`` -> same as mean_radius_mm."""

    taper_profile: Literal["linear", "smooth", "constant"] = "smooth"
    """How mean radius varies between proximal and distal."""

    perturbation_modes: tuple[int, ...] = (2, 3, 4, 5)
    """Angular Fourier modes. k=2 -> ovality, k=3 -> trefoil, etc."""

    max_perturbation_frac: float = 0.18
    """Maximum total radial deviation as a fraction of mean radius."""

    perturbation_decay: float = 1.6
    """Higher-order modes scale by 1 / k**perturbation_decay."""

    phase_drift_per_mm: float = 0.10
    """Per-mode phase drift along arclength, in radians/mm. Smoothness."""

    n_angles: int = 96
    """Number of angular samples per cross-section. Higher -> smoother mesh."""

    def __post_init__(self) -> None:
        if self.mean_radius_mm <= 0:
            raise ValueError("mean_radius_mm must be positive")
        if self.distal_radius_mm is not None and self.distal_radius_mm <= 0:
            raise ValueError("distal_radius_mm must be positive when provided")
        if not 0.0 <= self.max_perturbation_frac < 0.5:
            raise ValueError("max_perturbation_frac must be in [0, 0.5)")
        if self.n_angles < 24:
            raise ValueError("n_angles must be >= 24 for a usable mesh")


# ---------------------------------------------------------------------------
# Wall thickness
# ---------------------------------------------------------------------------


@dataclass
class WallConfig:
    """Per-angle, per-arclength wall thickness model.

    Thickness at angle ``theta`` and arclength ``s`` is

        t(theta, s) = t_mean * (1 + sum_k b_k(s) cos(k theta + psi_k(s)))

    clamped to >= ``min_thickness_mm`` to avoid the outer wall crossing the
    lumen on a thin spot. Modes are kept low (k = 1..3) so the result reads
    as "thicker on one side than the other" rather than as fine speckle.
    """

    mean_thickness_mm: float = 0.7
    perturbation_modes: tuple[int, ...] = (1, 2, 3)
    max_perturbation_frac: float = 0.6
    """Allows >50% modulation so walls clearly vary thick/thin around the
    circumference, like real diseased peripheral vessels."""

    perturbation_decay: float = 1.2
    phase_drift_per_mm: float = 0.05
    min_thickness_mm: float = 0.15

    def __post_init__(self) -> None:
        if self.mean_thickness_mm <= 0:
            raise ValueError("mean_thickness_mm must be positive")
        if self.min_thickness_mm <= 0:
            raise ValueError("min_thickness_mm must be positive")
        if self.min_thickness_mm >= self.mean_thickness_mm:
            raise ValueError("min_thickness_mm must be < mean_thickness_mm")
        if not 0.0 <= self.max_perturbation_frac < 1.0:
            raise ValueError("max_perturbation_frac must be in [0, 1)")


# ---------------------------------------------------------------------------
# Branch (centerline + cross-section + wall)
# ---------------------------------------------------------------------------


@dataclass
class BranchConfig:
    """A single branch of a vessel: centerline + lumen + wall."""

    centerline: CenterlineConfig = field(default_factory=CenterlineConfig)
    cross_section: CrossSectionConfig = field(default_factory=CrossSectionConfig)
    wall: WallConfig = field(default_factory=WallConfig)
    name: str = "main"
    seed: Optional[int] = None
    """Per-branch RNG seed. None inherits from the parent vessel seed."""


# ---------------------------------------------------------------------------
# Bifurcations
# ---------------------------------------------------------------------------


@dataclass
class SideBranchConfig:
    """A daughter vessel that buds off a parent.

    The daughter centerline starts on the parent's centerline at the
    parent-arclength fraction ``parent_arclength_frac`` (0 = proximal,
    1 = distal end of the parent), exits in the direction defined by
    ``azimuth_deg`` (around the parent axis) and ``polar_deg`` (angle from
    the parent axis; 0 = along the parent, 90 = perpendicular), and
    continues for the daughter's own ``length_mm``.

    The catheter is expected to remain in the parent; from inside the
    parent the daughter appears as an opening in the wall and a short
    visible segment beyond.
    """

    parent_arclength_frac: float = 0.5
    azimuth_deg: float = 0.0
    polar_deg: float = 60.0
    branch: BranchConfig = field(default_factory=lambda: BranchConfig(
        centerline=CenterlineConfig(length_mm=15.0),
        cross_section=CrossSectionConfig(mean_radius_mm=1.6),
        wall=WallConfig(mean_thickness_mm=0.5),
        name="side_branch",
    ))

    def __post_init__(self) -> None:
        if not 0.05 <= self.parent_arclength_frac <= 0.95:
            raise ValueError("parent_arclength_frac must be in [0.05, 0.95]")
        if not 10.0 <= self.polar_deg <= 89.0:
            raise ValueError("polar_deg must be in [10, 89] degrees")


# ---------------------------------------------------------------------------
# Vessel
# ---------------------------------------------------------------------------


@dataclass
class VesselConfig:
    """A complete vessel: parent branch + zero or more side-branch bifurcations.

    Y-junction (parent split into two daughters) is intentionally not
    supported in v1; only side-branch ostia are modelled. See
    ``docs/design.md`` for the rationale.
    """

    parent: BranchConfig = field(default_factory=BranchConfig)
    side_branches: list[SideBranchConfig] = field(default_factory=list)
    seed: int = 0
    name: str = "vessel"

    def __post_init__(self) -> None:
        if self.parent.seed is None:
            self.parent.seed = self.seed
        for i, sb in enumerate(self.side_branches):
            if sb.branch.seed is None:
                sb.branch.seed = self.seed + 100 + i


# ---------------------------------------------------------------------------
# Batch / library generation
# ---------------------------------------------------------------------------


@dataclass
class _UniformRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class GenerationConfig:
    """Distributions over vessel parameters for batch generation.

    A call to :meth:`sample` returns one concrete :class:`VesselConfig`
    drawn from these distributions. The defaults span a peripheral-vessel
    range that covers the porcine-lab evaluation envelope (lumen radius
    1.2-4.5 mm, length 20-40 mm, mixed bifurcation states).
    """

    length_mm_range: tuple[float, float] = (20.0, 40.0)
    parent_radius_mm_range: tuple[float, float] = (1.5, 4.5)
    parent_radius_taper_frac_range: tuple[float, float] = (0.85, 1.05)
    """Distal radius as a fraction of proximal radius."""

    parent_wall_thickness_mm_range: tuple[float, float] = (0.4, 1.2)
    parent_wall_perturbation_frac_range: tuple[float, float] = (0.3, 0.7)
    parent_lumen_perturbation_frac_range: tuple[float, float] = (0.08, 0.22)

    side_branch_probability: float = 0.45
    side_branch_radius_frac_range: tuple[float, float] = (0.4, 0.75)
    side_branch_length_mm_range: tuple[float, float] = (10.0, 20.0)
    side_branch_polar_deg_range: tuple[float, float] = (35.0, 75.0)
    max_side_branches: int = 1

    def sample(self, rng: np.random.Generator, seed: int, name: str = "vessel") -> VesselConfig:
        """Draw one VesselConfig from the configured distributions."""
        length = _UniformRange(*self.length_mm_range).sample(rng)
        r_proximal = _UniformRange(*self.parent_radius_mm_range).sample(rng)
        taper = _UniformRange(*self.parent_radius_taper_frac_range).sample(rng)
        r_distal = r_proximal * taper

        parent = BranchConfig(
            centerline=CenterlineConfig(
                length_mm=length,
                origin=(0.0, -length / 2.0, 0.0),
                direction=(0.0, 1.0, 0.0),
                n_stations=int(max(32, min(128, round(length * 2)))),
            ),
            cross_section=CrossSectionConfig(
                mean_radius_mm=r_proximal,
                distal_radius_mm=r_distal,
                max_perturbation_frac=_UniformRange(
                    *self.parent_lumen_perturbation_frac_range
                ).sample(rng),
            ),
            wall=WallConfig(
                mean_thickness_mm=_UniformRange(
                    *self.parent_wall_thickness_mm_range
                ).sample(rng),
                max_perturbation_frac=_UniformRange(
                    *self.parent_wall_perturbation_frac_range
                ).sample(rng),
            ),
            name="parent",
            seed=seed,
        )

        side_branches: list[SideBranchConfig] = []
        n_side = 0
        if rng.random() < self.side_branch_probability:
            n_side = int(rng.integers(1, self.max_side_branches + 1))
        for i in range(n_side):
            sb_radius = r_proximal * _UniformRange(
                *self.side_branch_radius_frac_range
            ).sample(rng)
            sb_length = _UniformRange(*self.side_branch_length_mm_range).sample(rng)
            side_branches.append(
                SideBranchConfig(
                    parent_arclength_frac=float(rng.uniform(0.25, 0.8)),
                    azimuth_deg=float(rng.uniform(0.0, 360.0)),
                    polar_deg=_UniformRange(
                        *self.side_branch_polar_deg_range
                    ).sample(rng),
                    branch=BranchConfig(
                        centerline=CenterlineConfig(
                            length_mm=sb_length,
                            n_stations=int(max(24, min(64, round(sb_length * 2)))),
                        ),
                        cross_section=CrossSectionConfig(
                            mean_radius_mm=sb_radius,
                            distal_radius_mm=sb_radius * 0.85,
                            max_perturbation_frac=0.18,
                        ),
                        wall=WallConfig(
                            mean_thickness_mm=max(0.3, parent.wall.mean_thickness_mm * 0.8),
                            max_perturbation_frac=0.4,
                        ),
                        name=f"side_{i}",
                        seed=seed + 100 + i,
                    ),
                )
            )

        return VesselConfig(
            parent=parent,
            side_branches=side_branches,
            seed=seed,
            name=name,
        )
