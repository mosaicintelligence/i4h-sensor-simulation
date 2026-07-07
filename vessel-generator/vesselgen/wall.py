"""Per-angle, per-arclength wall thickness model.

Adds a thickness ``t(theta, s)`` to a lumen contour to produce the outer
(adventitia) contour. Thickness varies smoothly around the circumference
and along the length, so walls clearly read as "thicker on one side than
another" at a given station and the asymmetry rotates slowly with depth.

Pass the same ``CrossSectionField`` (lumen) plus this wall field to the
sweep code; it will build a closed lumen surface and a closed outer
surface that nests around it.

Layered walls
-------------

For a richer model :func:`build_layered_wall` returns a
:class:`LayeredWallField` whose ``interface_radii`` describe ``n_layers
+ 1`` concentric surfaces: the lumen (innermost) plus one boundary per
layer. Adjacent interfaces are guaranteed not to cross thanks to the
fractional perturbation clamps in :func:`build_layered_wall`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vesselgen.config import LayeredWallConfig, LayerSpec, WallConfig
from vesselgen.cross_section import CrossSectionField

if TYPE_CHECKING:
    from vesselgen.centerline import Centerline
    from vesselgen.config import CalcificationLesionConfig


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


@dataclass
class LayeredWallField:
    """Per-layer interface radii on the lumen's (station, angle) grid.

    ``interface_radii`` holds ``n_layers + 1`` arrays (each shape
    ``(N_stations, M_angles)``) describing concentric surfaces:

    - ``interface_radii[0]`` is the lumen (innermost) surface.
    - ``interface_radii[k]`` for ``1 <= k < n_layers`` are interior
      interfaces (e.g. intima/media, media/adventitia).
    - ``interface_radii[n_layers]`` is the outer (adventitia) surface.

    By construction adjacent interface arrays satisfy
    ``interface_radii[k+1] >= interface_radii[k] + min_gap`` so no two
    surfaces ever cross.
    """

    interface_radii: list[np.ndarray]
    layer_specs: list[LayerSpec]
    thetas: np.ndarray
    total_thickness_mm: float
    min_thickness_mm: float

    @property
    def n_layers(self) -> int:
        return len(self.layer_specs)

    @property
    def n_stations(self) -> int:
        return int(self.interface_radii[0].shape[0])

    @property
    def n_angles(self) -> int:
        return int(self.interface_radii[0].shape[1])

    def to_outer_wall_field(self) -> WallField:
        """Compress to a single :class:`WallField` (lumen -> outermost)."""

        thicknesses = self.interface_radii[-1] - self.interface_radii[0]
        return WallField(
            thicknesses=thicknesses,
            mean_thickness=self.total_thickness_mm,
        )


def build_layered_wall(
    config: LayeredWallConfig,
    lumen: CrossSectionField,
    length_mm: float,
    rng: np.random.Generator,
) -> LayeredWallField:
    """Sample radii for each layer interface, nested inside the outer wall.

    The outer (adventitia) surface inherits its perturbation amplitude
    from the outermost layer's :attr:`LayerSpec.max_perturbation_frac`
    so the outermost contour matches what the legacy single-layer
    pipeline would have produced. Interior interfaces wobble more
    gently and are clamped so they never collide with their neighbours.
    """

    outer_spec = config.layers[-1]
    total_wall = build_wall(
        WallConfig(
            mean_thickness_mm=config.total_thickness_mm,
            min_thickness_mm=config.min_thickness_mm,
            max_perturbation_frac=outer_spec.max_perturbation_frac,
        ),
        lumen,
        length_mm,
        rng,
    )
    total = total_wall.thicknesses

    n_layers = config.n_layers
    fracs = np.array([layer.thickness_frac for layer in config.layers], dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(fracs)])  # length n_layers + 1

    interface_radii: list[np.ndarray] = [lumen.radii.copy()]

    if n_layers == 1:
        interface_radii.append(lumen.radii + total)
        return LayeredWallField(
            interface_radii=interface_radii,
            layer_specs=list(config.layers),
            thetas=lumen.thetas.copy(),
            total_thickness_mm=config.total_thickness_mm,
            min_thickness_mm=config.min_thickness_mm,
        )

    arclengths = np.linspace(0.0, length_mm, lumen.n_stations)
    # Smaller gap so thin layers (e.g. intima at ~10 percent) survive.
    min_gap_frac = 0.02

    for k in range(1, n_layers):
        nominal_k = float(cumulative[k])
        adj_frac = float(min(fracs[k - 1], fracs[k]))
        amp_per_layer = float(
            min(
                config.layers[k - 1].max_perturbation_frac,
                config.layers[k].max_perturbation_frac,
            )
        )
        amp = max(adj_frac * amp_per_layer * 0.5, 0.0)

        if amp <= 0.0:
            perturbation = np.zeros_like(total)
        else:
            modes = np.array([2, 3, 5], dtype=int)
            phases = rng.uniform(0.0, 2 * np.pi, size=len(modes))
            weights = 1.0 / np.power(modes, 1.3)
            weights = weights / weights.sum() * amp
            signs = rng.choice([-1.0, 1.0], size=len(modes))
            weights = weights * signs
            drift = 0.05 * (1.0 + 0.3 * (rng.random(len(modes)) - 0.5))

            perturbation = np.zeros_like(total)
            for m_idx, m in enumerate(modes):
                ph = phases[m_idx] + drift[m_idx] * arclengths[:, None]
                perturbation += weights[m_idx] * np.cos(m * lumen.thetas[None, :] + ph)

        frac_field = nominal_k + perturbation
        lower = float(cumulative[k - 1]) + min_gap_frac
        upper = float(cumulative[k + 1]) - min_gap_frac
        if upper <= lower:
            frac_field = np.full_like(total, 0.5 * (lower + upper))
        else:
            frac_field = np.clip(frac_field, lower, upper)
        interface_radii.append(lumen.radii + frac_field * total)

    interface_radii.append(lumen.radii + total)

    return LayeredWallField(
        interface_radii=interface_radii,
        layer_specs=list(config.layers),
        thetas=lumen.thetas.copy(),
        total_thickness_mm=config.total_thickness_mm,
        min_thickness_mm=config.min_thickness_mm,
    )


def push_layered_interfaces_around_lesions(
    layered_wall: LayeredWallField,
    centerline: "Centerline",
    lesion_configs: "list[CalcificationLesionConfig]",
    *,
    epsilon_mm: float = 0.02,
    bumpiness_margin_frac: float = 0.06,
) -> LayeredWallField:
    """Deform interior wall interfaces so they sit just outside every lesion.

    The simulator only tracks two materials per ray (current + outer), so
    if a lesion mesh and a wall interface mesh both pass through the same
    region, rays inside the lesion lose track of the calcified / lipid /
    fibrous / thrombus material as soon as they hit the interior interface
    -- which truncates the lesion's apparent radial thickness to a thin band
    near the lumen and produces a "ring" artifact (bright proximal arc +
    bright distal arc with a dark core, no real distal shadow).

    This routine post-processes a :class:`LayeredWallField` so every
    interior interface is pushed outward to at least ``lesion_outer_r +
    k*epsilon`` inside each lesion's (s, theta) footprint. The lumen
    surface (interface 0) and the outermost wall surface (interface -1)
    are left unchanged. The lesion ends up entirely nested inside the
    innermost wall layer (typically intima); behind the lesion the
    pushed interfaces stack into thin slabs of media / adventitia. From
    the simulator's point of view the lesion is now fully nested inside
    one material with no overlapping interfaces, so the calcified body
    accumulates its full attenuation budget and produces a real shadow.

    A ``bumpiness_margin_frac`` is added to the analytical ellipsoid
    radius to absorb the per-vertex bumps that ``inclusions.build_lesion_mesh``
    sprinkles on the icosphere surface.
    """

    if not lesion_configs or layered_wall.n_layers <= 1:
        return layered_wall

    n_stations = layered_wall.n_stations
    n_angles = layered_wall.n_angles
    thetas = layered_wall.thetas
    arclengths = np.linspace(0.0, centerline.length_mm, n_stations)

    inner_radii = layered_wall.interface_radii[0]
    outer_radii = layered_wall.interface_radii[-1]
    total_wall = np.maximum(outer_radii - inner_radii, 1e-6)

    lesion_outer_r = np.zeros((n_stations, n_angles), dtype=float)
    inside_lesion = np.zeros((n_stations, n_angles), dtype=bool)

    for cfg in lesion_configs:
        s_center = float(cfg.arclength_frac) * centerline.length_mm
        half_axial = max(0.5 * float(cfg.axial_extent_mm), 1e-6)
        half_arc = max(0.5 * float(np.deg2rad(cfg.arc_extent_deg)), 1e-6)
        azimuth = float(np.deg2rad(cfg.azimuth_deg))

        u = (arclengths - s_center) / half_axial
        d_theta = np.mod(thetas - azimuth + np.pi, 2.0 * np.pi) - np.pi
        v = d_theta / half_arc

        u2 = u[:, None] ** 2
        v2 = v[None, :] ** 2
        uv2 = u2 + v2
        footprint = uv2 <= 1.0
        if not footprint.any():
            continue

        w_max = np.sqrt(np.clip(1.0 - uv2, 0.0, 1.0))
        radial_mid = 0.5 * (cfg.inner_offset_frac + cfg.outer_offset_frac)
        radial_half = 0.5 * (cfg.outer_offset_frac - cfg.inner_offset_frac)
        outer_frac = radial_mid + radial_half * w_max
        outer_frac = outer_frac * (1.0 + bumpiness_margin_frac)
        outer_frac = np.clip(outer_frac, 0.0, 0.999)

        candidate = inner_radii + outer_frac * total_wall
        update = footprint & (candidate > lesion_outer_r)
        lesion_outer_r = np.where(update, candidate, lesion_outer_r)
        inside_lesion = inside_lesion | footprint

    if not inside_lesion.any():
        return layered_wall

    new_radii: list[np.ndarray] = [inner_radii.copy()]
    n_interior = layered_wall.n_layers - 1
    cap = outer_radii - epsilon_mm
    for k in range(1, layered_wall.n_layers):
        original = layered_wall.interface_radii[k]
        offset = epsilon_mm * (k - 1 + 1)
        target = lesion_outer_r + offset
        adventitia_cap = outer_radii - epsilon_mm * (n_interior - k + 1)
        target = np.minimum(target, adventitia_cap)
        target = np.minimum(target, cap)
        pushed = np.where(inside_lesion, np.maximum(original, target), original)
        new_radii.append(pushed)
    new_radii.append(outer_radii.copy())

    for k in range(1, len(new_radii)):
        new_radii[k] = np.maximum(new_radii[k], new_radii[k - 1] + 1e-6)
    for k in range(len(new_radii) - 2, -1, -1):
        new_radii[k] = np.minimum(new_radii[k], new_radii[k + 1] - 1e-6)

    return LayeredWallField(
        interface_radii=new_radii,
        layer_specs=list(layered_wall.layer_specs),
        thetas=layered_wall.thetas.copy(),
        total_thickness_mm=layered_wall.total_thickness_mm,
        min_thickness_mm=layered_wall.min_thickness_mm,
    )


def layered_wall_from_wall_field(
    wall: WallField,
    lumen: CrossSectionField,
    *,
    material_name: str = "vessel_wall",
    total_thickness_mm: float | None = None,
    min_thickness_mm: float = 0.15,
) -> LayeredWallField:
    """Wrap a legacy :class:`WallField` as a single-layer
    :class:`LayeredWallField` for downstream code that always expects
    the layered representation."""

    spec = LayerSpec(material_name=material_name, thickness_frac=1.0)
    return LayeredWallField(
        interface_radii=[lumen.radii.copy(), lumen.radii + wall.thicknesses],
        layer_specs=[spec],
        thetas=lumen.thetas.copy(),
        total_thickness_mm=(
            float(total_thickness_mm)
            if total_thickness_mm is not None
            else float(wall.mean_thickness)
        ),
        min_thickness_mm=float(min_thickness_mm),
    )
