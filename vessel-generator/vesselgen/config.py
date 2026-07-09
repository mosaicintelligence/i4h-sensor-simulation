"""Generation parameters for vessel segments.

These dataclasses are the public schema for everything the generator does.
Each field has a sensible default sized for **large peripheral vasculature**
as imaged with the PV .035 (femoral / iliac / renal vein / aortic segments),
not coronary scale. For batch generation, ``GenerationConfig`` carries
distributions over these parameters and ``GenerationConfig.sample()`` draws
a concrete ``VesselConfig`` from them.

Typical targets (lumen diameter ≈ 2 × mean_radius_mm):
  * Femoral / iliac artery: 8–14 mm lumen, wall ~0.7–1.2 mm
  * Renal vein / large vein: similar lumen, slightly thinner wall
  * Aortic segment (occasional draw): 16–24 mm lumen, wall ~1.0–1.5 mm

At 10 MHz the catheter ring-down occupies roughly r < 2–3.6 mm; lumen radii
≥ 4 mm place the vessel wall clearly outside the ring-down disc in the image.

All distances are in millimetres. All angles are in degrees unless the
field name says otherwise. Coordinate convention matches the simulator:
vessel axis along Y, cross-sections in the xz plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Union

import numpy as np

LesionKind = Literal["hard", "soft_lipid", "fibrous", "thrombus"]
"""Single-material lesion kinds. Maps 1:1 onto a simulator material:

* ``hard``       -> ``calcified_plaque`` (high impedance, high attenuation;
  bright leading edge + acoustic shadow).
* ``soft_lipid`` -> ``lipid_pool`` (low impedance, low backscatter;
  echo-poor necrotic-core appearance).
* ``fibrous``    -> ``fibrous_plaque`` (moderate impedance, moderate
  backscatter; brighter than media but no shadow).
* ``thrombus``   -> ``thrombus`` (close to blood, slightly elevated
  backscatter; "smoky" intraluminal echo on real frames).
"""

CompositeLesionKind = Literal["calcified", "vulnerable_plaque"]
"""Composite lesion kinds the generator can emit as one knob.

* ``calcified``         -> one ``hard`` (calcified_plaque) lesion.
* ``vulnerable_plaque`` -> a ``soft_lipid`` body covered on the lumen
  side by a thinner ``fibrous`` cap (thin-cap fibroatheroma).
"""

LESION_KIND_TO_MATERIAL: dict[str, str] = {
    "hard": "calcified_plaque",
    "soft_lipid": "lipid_pool",
    "fibrous": "fibrous_plaque",
    "thrombus": "thrombus",
}


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

    length_mm: float = 55.0
    origin: tuple[float, float, float] = (0.0, -27.5, 0.0)
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    n_stations: int = 64

    curvature_amplitude_mm: float = 0.0  # reserved, v2
    curvature_period_mm: float = 0.0  # reserved, v2

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

    mean_radius_mm: float = 5.0
    """Mean lumen radius at the proximal end (~10 mm lumen diameter)."""

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

    mean_thickness_mm: float = 0.85
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
# Multi-layered wall (intima / media / adventitia)
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    """One concentric tissue layer in a :class:`LayeredWallConfig`.

    Layers are listed from the lumen surface outward. The layer's
    thickness is ``thickness_frac * total_thickness_mm`` (modulo the
    per-angle Fourier modulation, which uses the same per-mode/decay
    machinery as :class:`WallConfig`).

    The ``material_name`` is consumed by the simulator at render time
    and must match a key in ``raysim::Materials``. The shipped
    convention for a 3-layer wall is
    ``("intima", "media", "adventitia")`` from inner to outer.
    """

    material_name: str
    thickness_frac: float
    perturbation_modes: tuple[int, ...] = (1, 2, 3)
    max_perturbation_frac: float = 0.4
    perturbation_decay: float = 1.2
    phase_drift_per_mm: float = 0.05

    def __post_init__(self) -> None:
        if not self.material_name:
            raise ValueError("material_name must be non-empty")
        if not 0.0 < self.thickness_frac <= 1.0:
            raise ValueError(f"thickness_frac must be in (0, 1], got {self.thickness_frac}")
        if not 0.0 <= self.max_perturbation_frac < 1.0:
            raise ValueError("max_perturbation_frac must be in [0, 1)")


@dataclass
class LayeredWallConfig:
    """A multi-layer wall built from one or more concentric :class:`LayerSpec`.

    The layered wall replaces :class:`WallConfig` when richer per-layer
    acoustics are needed. The legacy single-layer wall is recovered by
    passing a single :class:`LayerSpec` with ``material_name="vessel_wall"``
    and ``thickness_frac=1.0``.

    ``total_thickness_mm`` is the radial offset between the lumen
    surface and the outermost layer boundary (i.e. the same quantity as
    :attr:`WallConfig.mean_thickness_mm`). Layer fractions must sum to
    1.0.
    """

    layers: list[LayerSpec]
    total_thickness_mm: float = 0.85
    min_thickness_mm: float = 0.15
    """Floor on the *total* wall thickness at any (station, angle)."""

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("LayeredWallConfig requires at least one layer")
        if len(self.layers) > 3:
            raise ValueError(
                "LayeredWallConfig supports at most 3 layers (intima, media, adventitia)"
            )
        total_frac = sum(layer.thickness_frac for layer in self.layers)
        if abs(total_frac - 1.0) > 1e-6:
            raise ValueError(f"layer thickness_frac values must sum to 1.0, got {total_frac:.4f}")
        if self.total_thickness_mm <= 0:
            raise ValueError("total_thickness_mm must be positive")
        if self.min_thickness_mm <= 0:
            raise ValueError("min_thickness_mm must be positive")
        if self.min_thickness_mm >= self.total_thickness_mm:
            raise ValueError("min_thickness_mm must be < total_thickness_mm")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def mean_thickness_mm(self) -> float:
        """Backward-compatible alias for ``total_thickness_mm``.

        ``WallConfig.mean_thickness_mm`` is the legacy attribute name for
        the total radial wall thickness; consumers that worked with
        :class:`WallConfig` use ``wall.mean_thickness_mm`` so we expose the
        same name on the layered wall too.
        """

        return float(self.total_thickness_mm)

    @property
    def max_perturbation_frac(self) -> float:
        """Outermost layer's perturbation fraction (governs outer surface)."""

        return float(self.layers[-1].max_perturbation_frac)

    @classmethod
    def single_layer(
        cls,
        material_name: str = "vessel_wall",
        total_thickness_mm: float = 0.85,
        max_perturbation_frac: float = 0.6,
        min_thickness_mm: float = 0.15,
    ) -> "LayeredWallConfig":
        """Build a legacy single-layer wall (one concentric mesh + outer)."""
        return cls(
            layers=[
                LayerSpec(
                    material_name=material_name,
                    thickness_frac=1.0,
                    max_perturbation_frac=max_perturbation_frac,
                )
            ],
            total_thickness_mm=total_thickness_mm,
            min_thickness_mm=min_thickness_mm,
        )

    @classmethod
    def trilaminar(
        cls,
        total_thickness_mm: float = 0.85,
        intima_frac: float = 0.18,
        media_frac: float = 0.55,
        adventitia_frac: float = 0.27,
        min_thickness_mm: float = 0.15,
    ) -> "LayeredWallConfig":
        """Classic intima / media / adventitia trilaminar wall."""
        return cls(
            layers=[
                LayerSpec(
                    material_name="intima", thickness_frac=intima_frac, max_perturbation_frac=0.35
                ),
                LayerSpec(
                    material_name="media", thickness_frac=media_frac, max_perturbation_frac=0.25
                ),
                LayerSpec(
                    material_name="adventitia",
                    thickness_frac=adventitia_frac,
                    max_perturbation_frac=0.35,
                ),
            ],
            total_thickness_mm=total_thickness_mm,
            min_thickness_mm=min_thickness_mm,
        )

    @classmethod
    def media_adventitia(
        cls,
        total_thickness_mm: float = 0.85,
        media_frac: float = 0.65,
        adventitia_frac: float = 0.35,
        min_thickness_mm: float = 0.15,
    ) -> "LayeredWallConfig":
        """Two-layer wall: dark media + bright adventitia (no thin intima)."""
        return cls(
            layers=[
                LayerSpec(
                    material_name="media", thickness_frac=media_frac, max_perturbation_frac=0.3
                ),
                LayerSpec(
                    material_name="adventitia",
                    thickness_frac=adventitia_frac,
                    max_perturbation_frac=0.4,
                ),
            ],
            total_thickness_mm=total_thickness_mm,
            min_thickness_mm=min_thickness_mm,
        )


# ---------------------------------------------------------------------------
# Lesions / inclusions (calcifications, lipid pools, fibrous plaque, thrombus)
# ---------------------------------------------------------------------------


@dataclass
class CalcificationLesionConfig:
    """One in-wall lesion (calcium, lipid pool, fibrous plaque, thrombus).

    Geometry is a closed lens-shaped solid embedded between the lumen
    surface and the outermost layer boundary. The angular wedge spans
    ``arc_extent_deg`` centred on ``azimuth_deg``; the axial extent
    spans ``axial_extent_mm`` centred on
    ``arclength_frac * branch_length``.

    Radial placement is controlled by ``inner_offset_frac`` and
    ``outer_offset_frac``, both expressed as fractions of the local
    wall thickness:

    * 0.0 = at the lumen surface,
    * 1.0 = at the outermost layer boundary.

    The defaults straddle the intima-media junction (where calcium
    typically sits), but a lipid pool can be pushed deeper by raising
    ``inner_offset_frac`` (e.g. 0.3 -> covered by a thin intima cap).
    """

    arclength_frac: float
    azimuth_deg: float
    kind: LesionKind = "hard"
    arc_extent_deg: float = 50.0
    axial_extent_mm: float = 6.0
    inner_offset_frac: float = 0.0
    outer_offset_frac: float = 0.75
    n_angular_samples: int = 24
    n_axial_samples: int = 16
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in LESION_KIND_TO_MATERIAL:
            raise ValueError(
                f"kind must be one of {tuple(LESION_KIND_TO_MATERIAL)}, " f"got {self.kind!r}"
            )
        if not 0.0 <= self.arclength_frac <= 1.0:
            raise ValueError("arclength_frac must be in [0, 1]")
        if not 5.0 <= self.arc_extent_deg <= 270.0:
            raise ValueError("arc_extent_deg must be in [5, 270] degrees")
        if self.axial_extent_mm <= 0:
            raise ValueError("axial_extent_mm must be positive")
        if not 0.0 <= self.inner_offset_frac < self.outer_offset_frac <= 1.0:
            raise ValueError(
                "inner_offset_frac must satisfy " "0 <= inner_offset_frac < outer_offset_frac <= 1"
            )
        if self.n_angular_samples < 6:
            raise ValueError("n_angular_samples must be >= 6")
        if self.n_axial_samples < 4:
            raise ValueError("n_axial_samples must be >= 4")

    @property
    def material_name(self) -> str:
        return LESION_KIND_TO_MATERIAL[self.kind]


@dataclass
class DiseasedSectorConfig:
    """Per-vessel "diseased side" used to cluster lesion azimuths.

    A real atherosclerotic vessel is asymmetric: plaque sits
    preferentially on one side. The lesion sampler draws every lesion's
    ``azimuth_deg`` from a wrapped-uniform distribution centred on
    ``dominant_azimuth_deg`` with full extent ``sector_extent_deg``, so
    multiple lesions on the same vessel cluster on the same side.
    """

    dominant_azimuth_deg: float
    sector_extent_deg: float = 100.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.dominant_azimuth_deg < 360.0:
            self.dominant_azimuth_deg = float(self.dominant_azimuth_deg % 360.0)
        if not 20.0 <= self.sector_extent_deg <= 360.0:
            raise ValueError("sector_extent_deg must be in [20, 360] degrees")

    def sample_azimuth(self, rng: np.random.Generator) -> float:
        half = 0.5 * self.sector_extent_deg
        offset = float(rng.uniform(-half, half))
        return float((self.dominant_azimuth_deg + offset) % 360.0)


# ---------------------------------------------------------------------------
# Guidewire
# ---------------------------------------------------------------------------


# Commercial wire diameters (inches -> mm).
GUIDEWIRE_DIAMETERS_MM: dict[str, float] = {
    "0.014in": 0.014 * 25.4,  # 0.3556
    "0.018in": 0.018 * 25.4,  # 0.4572
    "0.035in": 0.035 * 25.4,  # 0.8890
}


@dataclass
class GuidewireConfig:
    """A guidewire sharing the lumen with the IVUS catheter.

    The wire is modelled as a closed cylinder parallel to the parent
    centerline, offset laterally by ``lateral_offset_mm`` along the
    ``offset_azimuth_deg`` direction (measured in the parent's local
    normal/binormal basis). Material is ``tungsten`` by default --
    impedance contrast vs blood is so large (Z ~ 101 vs 1.68 MRayl,
    R ~ 0.94) that the choice of metallic core (nitinol, 316L
    stainless, platinum) is acoustically indistinguishable at IVUS
    resolution, so a single high-Z material is faithful.
    """

    diameter_mm: float = GUIDEWIRE_DIAMETERS_MM["0.014in"]
    lateral_offset_mm: float = 0.0
    offset_azimuth_deg: float = 0.0
    material_name: str = "tungsten"

    def __post_init__(self) -> None:
        if self.diameter_mm <= 0:
            raise ValueError("diameter_mm must be positive")
        if self.lateral_offset_mm < 0:
            raise ValueError("lateral_offset_mm must be >= 0")
        if not self.material_name:
            raise ValueError("material_name must be non-empty")

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm


# ---------------------------------------------------------------------------
# Branch (centerline + cross-section + wall)
# ---------------------------------------------------------------------------


@dataclass
class BranchConfig:
    """A single branch of a vessel: centerline + lumen + wall.

    ``wall`` accepts either the legacy :class:`WallConfig` (single
    homogeneous slab) or a :class:`LayeredWallConfig` (one or more
    concentric tissue layers). The two are interchangeable through
    :func:`branch_wall_to_layered`; the rest of the pipeline always
    sees a :class:`LayeredWallConfig` after that conversion.
    """

    centerline: CenterlineConfig = field(default_factory=CenterlineConfig)
    cross_section: CrossSectionConfig = field(default_factory=CrossSectionConfig)
    wall: Union[WallConfig, LayeredWallConfig] = field(default_factory=WallConfig)
    name: str = "main"
    seed: Optional[int] = None
    """Per-branch RNG seed. None inherits from the parent vessel seed."""

    def layered_wall(self) -> LayeredWallConfig:
        """Return ``self.wall`` as a :class:`LayeredWallConfig`."""
        return branch_wall_to_layered(self.wall)


def branch_wall_to_layered(
    wall: Union[WallConfig, LayeredWallConfig],
) -> LayeredWallConfig:
    """Coerce a wall config into the layered form.

    Legacy :class:`WallConfig` becomes a single-layer wall with material
    ``vessel_wall``; :class:`LayeredWallConfig` passes through.
    """
    if isinstance(wall, LayeredWallConfig):
        return wall
    if isinstance(wall, WallConfig):
        return LayeredWallConfig(
            layers=[
                LayerSpec(
                    material_name="vessel_wall",
                    thickness_frac=1.0,
                    perturbation_modes=tuple(wall.perturbation_modes),
                    max_perturbation_frac=wall.max_perturbation_frac,
                    perturbation_decay=wall.perturbation_decay,
                    phase_drift_per_mm=wall.phase_drift_per_mm,
                )
            ],
            total_thickness_mm=wall.mean_thickness_mm,
            min_thickness_mm=wall.min_thickness_mm,
        )
    raise TypeError(f"wall must be WallConfig or LayeredWallConfig, got {type(wall)!r}")


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
    branch: BranchConfig = field(
        default_factory=lambda: BranchConfig(
            centerline=CenterlineConfig(length_mm=45.0),
            cross_section=CrossSectionConfig(mean_radius_mm=3.5),
            wall=WallConfig(mean_thickness_mm=0.7),
            name="side_branch",
        )
    )

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

    ``lesions``, ``diseased_sector`` and ``guidewire`` are optional and
    apply to the parent branch only. Side branches retain a simple
    single-layer wall geometry; combining layered walls with side
    branches is tracked as a follow-up because every concentric layer
    mesh would have to be unioned with the daughter independently.
    """

    parent: BranchConfig = field(default_factory=BranchConfig)
    side_branches: list[SideBranchConfig] = field(default_factory=list)
    seed: int = 0
    name: str = "vessel"
    lesions: list[CalcificationLesionConfig] = field(default_factory=list)
    diseased_sector: Optional[DiseasedSectorConfig] = None
    guidewire: Optional[GuidewireConfig] = None

    def __post_init__(self) -> None:
        if self.parent.seed is None:
            self.parent.seed = self.seed
        for i, sb in enumerate(self.side_branches):
            if sb.branch.seed is None:
                sb.branch.seed = self.seed + 100 + i
        # Side branches now flow through the per-layer
        # ``attach_side_branch_layered`` path, so multi-layer parent
        # walls are fully supported. The historical guard that forced
        # bifurcation parents back to a single-slab wall has been
        # removed; the only remaining requirement is that every side
        # branch's wall has the same number of layers as the parent
        # (enforced at union time inside bifurcation.py).
        for i, lesion in enumerate(self.lesions):
            if lesion.seed is None:
                lesion.seed = self.seed + 1000 + i

    def with_minimum_side_branch_lengths(self) -> "VesselConfig":
        """Ensure every side branch is at least as long as the parent."""
        parent_length = self.parent.centerline.length_mm
        updated: list[SideBranchConfig] = []
        changed = False
        for sb in self.side_branches:
            requested = sb.branch.centerline.length_mm
            min_length = minimum_side_branch_length_mm(parent_length, requested)
            if min_length > requested + 1e-9:
                changed = True
                branch = replace(
                    sb.branch,
                    centerline=replace(
                        sb.branch.centerline,
                        length_mm=min_length,
                        n_stations=side_branch_n_stations(min_length),
                    ),
                )
                updated.append(replace(sb, branch=branch))
            else:
                updated.append(sb)
        if not changed:
            return self
        return replace(self, side_branches=updated)


# ---------------------------------------------------------------------------
# Batch / library generation
# ---------------------------------------------------------------------------


def minimum_side_branch_length_mm(parent_length_mm: float, requested_length_mm: float) -> float:
    """Side branches must extend at least as far as the parent centerline."""
    return max(float(requested_length_mm), float(parent_length_mm))


def side_branch_n_stations(length_mm: float) -> int:
    return int(max(24, min(64, round(length_mm * 2))))


@dataclass
class _UniformRange:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class GenerationConfig:
    """Distributions over vessel parameters for batch generation.

      Defaults target large peripheral arteries and veins (femoral, iliac, renal,
    EVAR-scale aorta) for the PV .035 ICE catheter. Lumen radii place the wall
    outside the ring-down zone (r >~ 4 mm). ~18% of draws use aortic-scale lumina.
    """

    length_mm_range: tuple[float, float] = (45.0, 75.0)
    parent_radius_mm_range: tuple[float, float] = (4.0, 6.5)
    parent_radius_taper_frac_range: tuple[float, float] = (0.88, 1.02)
    """Distal radius as a fraction of proximal radius."""

    parent_wall_thickness_mm_range: tuple[float, float] = (0.65, 1.2)
    parent_wall_perturbation_frac_range: tuple[float, float] = (0.25, 0.65)
    parent_lumen_perturbation_frac_range: tuple[float, float] = (0.06, 0.18)

    aortic_scale_probability: float = 0.18
    aortic_radius_mm_range: tuple[float, float] = (8.0, 11.5)
    aortic_wall_thickness_mm_range: tuple[float, float] = (1.0, 1.5)

    # Large vessels whose wall runs past the imaging FOV on some angular
    # sectors for typical (naturally off-centre) poses, so those A-lines
    # have no wall echo. The lumen radius is drawn large enough that the
    # far wall exceeds the smaller ``t_far_mm`` FOVs (17.5 / 20 mm); at the
    # 30 mm FOV these vessels mostly stay in view. aortic + large
    # probabilities should sum to <= 1.0 (the remainder is the typical draw).
    large_vessel_beyond_fov_probability: float = 0.10
    large_vessel_radius_mm_range: tuple[float, float] = (12.0, 16.0)
    large_vessel_wall_thickness_mm_range: tuple[float, float] = (1.0, 1.5)

    side_branch_probability: float = 0.45
    side_branch_radius_frac_range: tuple[float, float] = (0.55, 0.80)
    side_branch_length_mm_range: tuple[float, float] = (40.0, 75.0)
    side_branch_polar_deg_range: tuple[float, float] = (35.0, 75.0)
    max_side_branches: int = 1

    # --- Wall layering ---------------------------------------------------
    layered_wall_probability: float = 0.7
    """Per-vessel probability of emitting a layered wall instead of the
    legacy single homogeneous slab. Side branches inherit the parent's
    layered structure so the per-layer boolean union at the ostium has
    matching layer counts on both sides."""

    n_layers_weights: tuple[float, float, float] = (0.30, 0.35, 0.35)
    """Probability weights for 1 / 2 / 3 wall layers when a layered wall
    is selected. ``(1-layer, media+adventitia, intima+media+adventitia)``."""

    intima_thickness_frac_range: tuple[float, float] = (0.10, 0.25)
    media_thickness_frac_range: tuple[float, float] = (0.40, 0.60)
    """Adventitia thickness is the residual (1 - intima - media)."""

    # --- Lesions ---------------------------------------------------------
    calcification_probability: float = 0.35
    """Per-vessel probability a lesion-bearing diseased segment is drawn
    (when the wall geometry can hold it). Bumped from 0.15 -> 0.35 so a
    typical 100-frame paired dataset has a healthy fraction of diseased
    cases for the segmentation head to learn from."""

    calcification_count_range: tuple[int, int] = (1, 3)
    calc_kind_weights: dict[str, float] = field(
        default_factory=lambda: {
            "hard": 0.35,
            "soft_lipid": 0.20,
            "fibrous": 0.20,
            "thrombus": 0.05,
            "vulnerable_plaque": 0.20,
        }
    )
    diseased_sector_extent_deg_range: tuple[float, float] = (60.0, 140.0)
    # Lesions read more clearly on the rendered B-mode when the arc and
    # axial extents are larger; the previous (30-90 deg, 3-12 mm) range
    # produced very subtle lesion echoes through 5+ mm of intervening
    # blood + wall. The widened ranges still cover thin focal lesions
    # (60 deg / 6 mm) without forcing every plaque to be huge.
    lesion_arc_extent_deg_range: tuple[float, float] = (60.0, 150.0)
    lesion_axial_extent_mm_range: tuple[float, float] = (6.0, 16.0)
    # Lesions sit flush against the lumen by default so the proximal
    # bright echo (calcium / fibrous cap) is visible under the catheter
    # ring-down zone. Raise the floor only slightly so a small subset of
    # samples still has a thin endothelial layer covering the lesion.
    lesion_inner_offset_frac_range: tuple[float, float] = (0.0, 0.10)
    lesion_outer_offset_frac_range: tuple[float, float] = (0.75, 1.0)

    # --- Guidewire -------------------------------------------------------
    guidewire_probability: float = 0.70
    guidewire_diameter_choices_mm: tuple[float, ...] = (
        GUIDEWIRE_DIAMETERS_MM["0.014in"],
        GUIDEWIRE_DIAMETERS_MM["0.018in"],
        GUIDEWIRE_DIAMETERS_MM["0.035in"],
    )
    guidewire_diameter_weights: tuple[float, ...] = (0.30, 0.30, 0.40)
    """Default skewed toward 0.035\" because the calibrated catheter
    (PV .035) typically images alongside a peripheral 0.035\" wire;
    keep the smaller diameters for coronary-style frames."""

    guidewire_lateral_offset_frac_range: tuple[float, float] = (0.4, 0.85)
    """Lateral offset as a fraction of the smallest lumen radius along
    the parent. 0 = on the catheter, 1 = touching the wall."""

    # --- Pose sampling ---------------------------------------------------
    wall_contact_probability: float = 0.12
    """Fraction of sampled poses with the catheter placed against the
    lumen wall (produces the bright contact rim + opposite-side shadow
    seen on real frames). Consumed by ``sampling.sample_pose`` when
    enabled via the per-vessel ``GenerationConfig`` knob."""

    def sample(
        self,
        rng: np.random.Generator,
        seed: int,
        name: str = "vessel",
        *,
        force_side_branch: bool = False,
    ) -> VesselConfig:
        """Draw one VesselConfig from the configured distributions."""
        length = _UniformRange(*self.length_mm_range).sample(rng)
        u_scale = rng.random()
        if u_scale < self.aortic_scale_probability:
            r_proximal = _UniformRange(*self.aortic_radius_mm_range).sample(rng)
            wall_lo, wall_hi = self.aortic_wall_thickness_mm_range
        elif u_scale < (self.aortic_scale_probability + self.large_vessel_beyond_fov_probability):
            r_proximal = _UniformRange(*self.large_vessel_radius_mm_range).sample(rng)
            wall_lo, wall_hi = self.large_vessel_wall_thickness_mm_range
        else:
            r_proximal = _UniformRange(*self.parent_radius_mm_range).sample(rng)
            wall_lo, wall_hi = self.parent_wall_thickness_mm_range
        taper = _UniformRange(*self.parent_radius_taper_frac_range).sample(rng)
        r_distal = r_proximal * taper

        wall_thickness_mm = _UniformRange(wall_lo, wall_hi).sample(rng)
        wall_perturbation_frac = _UniformRange(*self.parent_wall_perturbation_frac_range).sample(
            rng
        )

        # Decide side branches and wall layering independently;
        # ``attach_side_branch_layered`` does per-layer boolean unions
        # so both can be combined freely.
        n_side = 0
        if force_side_branch or rng.random() < self.side_branch_probability:
            n_side = int(rng.integers(1, self.max_side_branches + 1))

        parent_wall: Union[WallConfig, LayeredWallConfig]
        if rng.random() < self.layered_wall_probability:
            parent_wall = self._sample_layered_wall(
                rng,
                total_thickness_mm=wall_thickness_mm,
                max_perturbation_frac=wall_perturbation_frac,
            )
        else:
            parent_wall = WallConfig(
                mean_thickness_mm=wall_thickness_mm,
                max_perturbation_frac=wall_perturbation_frac,
            )

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
            wall=parent_wall,
            name="parent",
            seed=seed,
        )

        side_branches: list[SideBranchConfig] = []
        for i in range(n_side):
            sb_radius = r_proximal * _UniformRange(*self.side_branch_radius_frac_range).sample(rng)
            sb_length = minimum_side_branch_length_mm(
                length,
                _UniformRange(*self.side_branch_length_mm_range).sample(rng),
            )
            sb_wall_thickness = max(0.5, wall_thickness_mm * 0.85)
            sb_wall: Union[WallConfig, LayeredWallConfig]
            if isinstance(parent_wall, LayeredWallConfig):
                # Daughter inherits the parent's layer count + thickness
                # fractions so the per-layer boolean union at the
                # ostium has matching geometry on both sides. Total
                # thickness rescales to the daughter's own wall and
                # the perturbation amplitudes are toned down (smaller
                # daughter radius -> tighter relative sweep).
                sb_wall = LayeredWallConfig(
                    layers=[
                        LayerSpec(
                            material_name=ly.material_name,
                            thickness_frac=ly.thickness_frac,
                            max_perturbation_frac=min(ly.max_perturbation_frac, 0.4),
                        )
                        for ly in parent_wall.layers
                    ],
                    total_thickness_mm=sb_wall_thickness,
                    min_thickness_mm=parent_wall.min_thickness_mm,
                )
            else:
                sb_wall = WallConfig(
                    mean_thickness_mm=sb_wall_thickness,
                    max_perturbation_frac=0.4,
                )
            side_branches.append(
                SideBranchConfig(
                    parent_arclength_frac=float(rng.uniform(0.25, 0.8)),
                    azimuth_deg=float(rng.uniform(0.0, 360.0)),
                    polar_deg=_UniformRange(*self.side_branch_polar_deg_range).sample(rng),
                    branch=BranchConfig(
                        centerline=CenterlineConfig(
                            length_mm=sb_length,
                            n_stations=side_branch_n_stations(sb_length),
                        ),
                        cross_section=CrossSectionConfig(
                            mean_radius_mm=sb_radius,
                            distal_radius_mm=sb_radius * 0.85,
                            max_perturbation_frac=0.18,
                        ),
                        wall=sb_wall,
                        name=f"side_{i}",
                        seed=seed + 100 + i,
                    ),
                )
            )

        # Lesions only attach to the parent branch in v1 and require a
        # layered wall to nest the inclusions inside. We skip lesion
        # generation on bifurcation vessels because the
        # ``push_layered_interfaces_around_lesions`` smoothing step is
        # currently bypassed when the parent layer meshes have already
        # been boolean-unioned with a daughter (the smoothing would
        # need a re-union pass). Plain trilam vessels still get the
        # full lesion set.
        lesions: list[CalcificationLesionConfig] = []
        diseased_sector: Optional[DiseasedSectorConfig] = None
        if (
            n_side == 0
            and isinstance(parent_wall, LayeredWallConfig)
            and rng.random() < self.calcification_probability
        ):
            diseased_sector = DiseasedSectorConfig(
                dominant_azimuth_deg=float(rng.uniform(0.0, 360.0)),
                sector_extent_deg=_UniformRange(*self.diseased_sector_extent_deg_range).sample(rng),
            )
            lo, hi = self.calcification_count_range
            n_lesions = int(rng.integers(lo, hi + 1))
            lesions = self._sample_lesions(
                rng,
                n=n_lesions,
                diseased_sector=diseased_sector,
                vessel_seed=seed,
            )

        guidewire: Optional[GuidewireConfig] = None
        if n_side == 0 and rng.random() < self.guidewire_probability:
            guidewire = self._sample_guidewire(rng, parent_radius_mm=min(r_proximal, r_distal))

        return VesselConfig(
            parent=parent,
            side_branches=side_branches,
            seed=seed,
            name=name,
            lesions=lesions,
            diseased_sector=diseased_sector,
            guidewire=guidewire,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_layered_wall(
        self,
        rng: np.random.Generator,
        *,
        total_thickness_mm: float,
        max_perturbation_frac: float,
    ) -> LayeredWallConfig:
        weights = np.asarray(self.n_layers_weights, dtype=float)
        weights = weights / weights.sum()
        n_layers = int(rng.choice([1, 2, 3], p=weights))
        if n_layers == 1:
            return LayeredWallConfig.single_layer(
                material_name="vessel_wall",
                total_thickness_mm=total_thickness_mm,
                max_perturbation_frac=max_perturbation_frac,
            )
        if n_layers == 2:
            media_frac = float(rng.uniform(0.55, 0.75))
            return LayeredWallConfig.media_adventitia(
                total_thickness_mm=total_thickness_mm,
                media_frac=media_frac,
                adventitia_frac=1.0 - media_frac,
            )
        intima_frac = _UniformRange(*self.intima_thickness_frac_range).sample(rng)
        media_frac = _UniformRange(*self.media_thickness_frac_range).sample(rng)
        # Ensure adventitia gets at least 10 percent of the wall.
        if intima_frac + media_frac > 0.9:
            scale = 0.9 / (intima_frac + media_frac)
            intima_frac *= scale
            media_frac *= scale
        adventitia_frac = 1.0 - intima_frac - media_frac
        return LayeredWallConfig.trilaminar(
            total_thickness_mm=total_thickness_mm,
            intima_frac=intima_frac,
            media_frac=media_frac,
            adventitia_frac=adventitia_frac,
        )

    def _sample_lesions(
        self,
        rng: np.random.Generator,
        *,
        n: int,
        diseased_sector: DiseasedSectorConfig,
        vessel_seed: int,
    ) -> list[CalcificationLesionConfig]:
        kinds = list(self.calc_kind_weights.keys())
        weights = np.asarray([self.calc_kind_weights[k] for k in kinds], dtype=float)
        weights = weights / weights.sum()

        lesions: list[CalcificationLesionConfig] = []
        for i in range(n):
            chosen = str(rng.choice(kinds, p=weights))
            arclength_frac = float(rng.uniform(0.15, 0.85))
            azimuth = diseased_sector.sample_azimuth(rng)
            arc_extent = _UniformRange(*self.lesion_arc_extent_deg_range).sample(rng)
            axial_extent = _UniformRange(*self.lesion_axial_extent_mm_range).sample(rng)
            inner_off = _UniformRange(*self.lesion_inner_offset_frac_range).sample(rng)
            outer_off = _UniformRange(*self.lesion_outer_offset_frac_range).sample(rng)
            if outer_off <= inner_off + 0.05:
                outer_off = min(0.95, inner_off + 0.2)
            base_seed = vessel_seed + 1000 + i

            if chosen == "vulnerable_plaque":
                # Lipid core in the deeper half + a thin fibrous cap on
                # the lumen-facing side. Same arc / arclength window so
                # the two lesions read as a single composite plaque.
                cap_inner = max(0.0, inner_off)
                cap_outer = max(cap_inner + 0.1, min(0.45, inner_off + 0.25))
                core_inner = cap_outer
                core_outer = max(core_inner + 0.15, outer_off)
                # Cap covers a slightly wider arc than the core so it
                # actually "covers" the core on the lumen side.
                cap_arc = min(180.0, arc_extent * 1.15)
                lesions.append(
                    CalcificationLesionConfig(
                        arclength_frac=arclength_frac,
                        azimuth_deg=azimuth,
                        kind="fibrous",
                        arc_extent_deg=cap_arc,
                        axial_extent_mm=axial_extent,
                        inner_offset_frac=cap_inner,
                        outer_offset_frac=cap_outer,
                        seed=base_seed,
                    )
                )
                lesions.append(
                    CalcificationLesionConfig(
                        arclength_frac=arclength_frac,
                        azimuth_deg=azimuth,
                        kind="soft_lipid",
                        arc_extent_deg=arc_extent,
                        axial_extent_mm=axial_extent,
                        inner_offset_frac=core_inner,
                        outer_offset_frac=core_outer,
                        seed=base_seed + 500,
                    )
                )
            else:
                lesions.append(
                    CalcificationLesionConfig(
                        arclength_frac=arclength_frac,
                        azimuth_deg=azimuth,
                        kind=chosen,  # type: ignore[arg-type]
                        arc_extent_deg=arc_extent,
                        axial_extent_mm=axial_extent,
                        inner_offset_frac=inner_off,
                        outer_offset_frac=outer_off,
                        seed=base_seed,
                    )
                )
        return lesions

    def _sample_guidewire(
        self,
        rng: np.random.Generator,
        *,
        parent_radius_mm: float,
    ) -> GuidewireConfig:
        diam_weights = np.asarray(self.guidewire_diameter_weights, dtype=float)
        diam_weights = diam_weights / diam_weights.sum()
        diameter = float(rng.choice(self.guidewire_diameter_choices_mm, p=diam_weights))
        offset_frac = _UniformRange(*self.guidewire_lateral_offset_frac_range).sample(rng)
        # Lateral offset is bounded so the wire stays well inside the
        # narrowest lumen. Subtract one wire radius + a small margin so
        # the wire isn't flush against the wall.
        max_offset = max(0.0, parent_radius_mm - 0.5 * diameter - 0.05)
        lateral_offset_mm = float(np.clip(offset_frac * parent_radius_mm, 0.0, max_offset))
        return GuidewireConfig(
            diameter_mm=diameter,
            lateral_offset_mm=lateral_offset_mm,
            offset_azimuth_deg=float(rng.uniform(0.0, 360.0)),
            material_name="tungsten",
        )
