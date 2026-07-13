"""Attach daughter branches to a parent vessel via boolean union.

A side-branch daughter is constructed as its own :class:`Centerline` +
:class:`CrossSectionField` + :class:`LayeredWallField`, swept into a
nested set of closed surface meshes (lumen, every interior layer
interface, outer adventitia boundary), and then **boolean-unioned**
layer-by-layer with the matching parent surface. Each union produces a
single watertight surface with a clean ostium, so simulator rays
passing through the opening from inside any wall layer of the parent
see the matching layer of the daughter (rather than a closed
boundary).

The number of layers must match between parent and daughter. The
:class:`vesselgen.config.SideBranchConfig` builder coerces the
daughter wall to layered form via
:meth:`vesselgen.config.BranchConfig.layered_wall`, mirroring the
parent's layer fractions.

Y-junctions (parent splits into two daughters at a single station) are
not implemented in v1; see ``docs/design.md`` for the rationale.

The boolean engine is ``manifold3d`` via trimesh's interface. If a union
fails for a particular geometric configuration (rare; happens when the
daughter's base does not penetrate the parent surface), the function
raises ``BifurcationError`` and the caller can re-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    LayeredWallConfig,
    SideBranchConfig,
    branch_wall_to_layered,
)
from vesselgen.cross_section import CrossSectionField, build_cross_sections
from vesselgen.sweep import sweep_branch, sweep_layered_branch
from vesselgen.wall import (
    LayeredWallField,
    WallField,
    build_layered_wall,
    build_wall,
)


class BifurcationError(RuntimeError):
    """Raised when a daughter branch cannot be cleanly attached."""


# ---------------------------------------------------------------------------
# Daughter centerline construction
# ---------------------------------------------------------------------------


def _spherical_to_unit(
    parent_tangent: np.ndarray,
    parent_normal: np.ndarray,
    parent_binormal: np.ndarray,
    polar_deg: float,
    azimuth_deg: float,
) -> np.ndarray:
    """Direction vector at (polar, azimuth) from a parent's local frame.

    polar_deg is the angle from the parent tangent (0 = along the parent,
    90 = perpendicular). azimuth_deg is the angle around the parent
    tangent, measured from the parent's normal toward the parent's
    binormal.
    """
    polar = np.radians(polar_deg)
    azimuth = np.radians(azimuth_deg)
    sp, cp = np.sin(polar), np.cos(polar)
    sa, ca = np.sin(azimuth), np.cos(azimuth)
    direction = cp * parent_tangent + sp * (ca * parent_normal + sa * parent_binormal)
    return direction / np.linalg.norm(direction)


def build_daughter_centerline(
    parent: Centerline,
    parent_arclength_frac: float,
    polar_deg: float,
    azimuth_deg: float,
    cfg: CenterlineConfig,
    parent_local_radius_mm: float,
    recess_frac_of_parent_radius: float = 0.5,
) -> Centerline:
    """Construct a daughter :class:`Centerline` from parent + emergence angles.

    The daughter origin is placed slightly *inside* the parent so the
    daughter mesh interpenetrates the parent mesh just enough for the
    boolean union to produce a clean ostium on the daughter's outgoing
    side, but does **not** poke out the opposite side of the parent.

    The recess is computed as ``recess_frac_of_parent_radius *
    parent_local_radius_mm`` and clamped to never exceed
    ``0.85 * parent_local_radius_mm``. This guarantees the daughter cap
    sits inside the parent regardless of the daughter's own radius or the
    daughter's emergence angle.

    The daughter's distal end stays at ``parent_attachment_point + length
    * direction`` (i.e. the user-requested length is measured outward from
    the parent centerline, unchanged by the recess).
    """
    s_origin = parent_arclength_frac * parent.length_mm
    parent_frame = parent.frame(s_origin)
    direction = _spherical_to_unit(
        parent_frame.tangent,
        parent_frame.normal,
        parent_frame.binormal,
        polar_deg,
        azimuth_deg,
    )
    safe_recess = min(
        max(0.1, recess_frac_of_parent_radius * parent_local_radius_mm),
        0.85 * parent_local_radius_mm,
    )
    origin = parent_frame.position - safe_recess * direction
    daughter_cfg = CenterlineConfig(
        length_mm=cfg.length_mm + safe_recess,
        origin=tuple(origin.tolist()),
        direction=tuple(direction.tolist()),
        n_stations=cfg.n_stations,
        curvature_amplitude_mm=cfg.curvature_amplitude_mm,
        curvature_period_mm=cfg.curvature_period_mm,
    )
    return Centerline(daughter_cfg)


def _parent_local_radius(
    parent_centerline: Centerline,
    parent_lumen_field,
    parent_arclength_frac: float,
) -> float:
    """Mean lumen radius of the parent at the attachment arclength."""
    n = parent_lumen_field.n_stations
    s_grid = np.linspace(0.0, 1.0, n)
    frac = float(np.clip(parent_arclength_frac, 0.0, 1.0))
    return float(np.interp(frac, s_grid, parent_lumen_field.mean_radius))


# ---------------------------------------------------------------------------
# Boolean attachment
# ---------------------------------------------------------------------------


def _safe_union(a: trimesh.Trimesh, b: trimesh.Trimesh, label: str) -> trimesh.Trimesh:
    try:
        out = trimesh.boolean.union([a, b], engine="manifold")
    except Exception as e:  # pragma: no cover - propagated as BifurcationError
        raise BifurcationError(f"boolean union failed for {label}: {e}") from e
    if out is None or out.is_empty or len(out.faces) == 0:
        raise BifurcationError(f"boolean union produced empty mesh for {label}")
    return out


@dataclass
class DaughterArtifacts:
    """All geometry produced for one attached daughter (kept for ground truth)."""

    name: str
    centerline: Centerline
    lumen_field: CrossSectionField
    wall_field: WallField
    parent_attachment_arclength_mm: float
    daughter_lumen_mesh: trimesh.Trimesh
    daughter_outer_mesh: trimesh.Trimesh
    layered_wall_field: Optional[LayeredWallField] = None
    """Per-layer wall radii so the lesion builder can query interior
    interfaces inside the daughter."""


def attach_side_branch_layered(
    parent_layer_meshes: list[trimesh.Trimesh],
    parent_centerline: Centerline,
    parent_lumen_field,
    side_branch: SideBranchConfig,
    rng: np.random.Generator,
) -> tuple[list[trimesh.Trimesh], DaughterArtifacts]:
    """Attach a side branch with layered walls and return updated layer meshes.

    ``parent_layer_meshes`` carries the parent's ``n_layers + 1`` nested
    surface meshes (lumen, every interior interface, outer adventitia
    boundary). The daughter is built with the same number of layers
    (its :class:`vesselgen.config.WallConfig` is coerced to a
    :class:`vesselgen.config.LayeredWallConfig` if necessary), swept
    into a matching nested mesh stack, and unioned layer-by-layer.

    Returns ``(new_parent_layer_meshes, DaughterArtifacts)``. The
    artifacts include the daughter's own pre-union meshes plus the
    layered wall field used to interrogate per-layer interfaces.
    """

    parent_local_radius = _parent_local_radius(
        parent_centerline,
        parent_lumen_field,
        side_branch.parent_arclength_frac,
    )
    daughter_centerline = build_daughter_centerline(
        parent_centerline,
        side_branch.parent_arclength_frac,
        side_branch.polar_deg,
        side_branch.azimuth_deg,
        side_branch.branch.centerline,
        parent_local_radius_mm=parent_local_radius,
    )

    branch_cfg = side_branch.branch
    daughter_seed = (
        branch_cfg.seed if branch_cfg.seed is not None else int(rng.integers(0, 2**31 - 1))
    )
    daughter_rng = np.random.default_rng(daughter_seed)

    lumen_field = build_cross_sections(
        branch_cfg.cross_section,
        daughter_centerline.length_mm,
        daughter_centerline.config.n_stations,
        daughter_rng,
    )

    daughter_layered_cfg = branch_wall_to_layered(branch_cfg.wall)
    daughter_layered_field = build_layered_wall(
        daughter_layered_cfg,
        lumen_field,
        daughter_centerline.length_mm,
        daughter_rng,
    )
    daughter_layer_meshes = sweep_layered_branch(
        daughter_centerline,
        lumen_field,
        daughter_layered_field,
    )

    if len(parent_layer_meshes) != len(daughter_layer_meshes):
        raise BifurcationError(
            f"layer count mismatch: parent has {len(parent_layer_meshes)} "
            f"surface meshes, daughter '{branch_cfg.name}' has "
            f"{len(daughter_layer_meshes)}. Both branches must specify the "
            "same number of wall layers."
        )

    new_layer_meshes: list[trimesh.Trimesh] = []
    for i, (p, d) in enumerate(zip(parent_layer_meshes, daughter_layer_meshes)):
        new_layer_meshes.append(_safe_union(p, d, label=f"{branch_cfg.name}/layer_{i:02d}"))

    daughter_outer_wall_field = daughter_layered_field.to_outer_wall_field()

    art = DaughterArtifacts(
        name=branch_cfg.name,
        centerline=daughter_centerline,
        lumen_field=lumen_field,
        wall_field=daughter_outer_wall_field,
        parent_attachment_arclength_mm=(
            side_branch.parent_arclength_frac * parent_centerline.length_mm
        ),
        daughter_lumen_mesh=daughter_layer_meshes[0],
        daughter_outer_mesh=daughter_layer_meshes[-1],
        layered_wall_field=daughter_layered_field,
    )
    return new_layer_meshes, art
