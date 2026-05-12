"""Attach daughter branches to a parent vessel via boolean union.

A side-branch daughter is constructed as its own :class:`Centerline` +
:class:`CrossSectionField` + :class:`WallField`, swept into closed lumen
and outer meshes, and then **boolean-unioned** with the parent's lumen
and outer meshes. The union produces a single watertight surface with a
clean ostium, so simulator rays passing through the opening from inside
the parent actually see the daughter interior (rather than hitting a
closed wall).

Y-junctions (parent splits into two daughters at a single station) are
not implemented in v1; see ``docs/design.md`` for the rationale.

The boolean engine is ``manifold3d`` via trimesh's interface. If a union
fails for a particular geometric configuration (rare; happens when the
daughter's base does not penetrate the parent surface), the function
raises ``BifurcationError`` and the caller can re-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    SideBranchConfig,
)
from vesselgen.cross_section import CrossSectionField, build_cross_sections
from vesselgen.sweep import sweep_branch
from vesselgen.wall import WallField, build_wall


class BifurcationError(RuntimeError):
    """Raised when a daughter branch cannot be cleanly attached."""


# ---------------------------------------------------------------------------
# Daughter centerline construction
# ---------------------------------------------------------------------------


def _spherical_to_unit(parent_tangent: np.ndarray, parent_normal: np.ndarray,
                      parent_binormal: np.ndarray, polar_deg: float, azimuth_deg: float) -> np.ndarray:
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
    direction = (
        cp * parent_tangent
        + sp * (ca * parent_normal + sa * parent_binormal)
    )
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
        parent_frame.tangent, parent_frame.normal, parent_frame.binormal,
        polar_deg, azimuth_deg,
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


def attach_side_branch(
    parent_lumen_mesh: trimesh.Trimesh,
    parent_outer_mesh: trimesh.Trimesh,
    parent_centerline: Centerline,
    parent_lumen_field,
    side_branch: SideBranchConfig,
    rng: np.random.Generator,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, DaughterArtifacts]:
    """Attach a side-branch and return updated (lumen, outer) meshes."""
    parent_local_radius = _parent_local_radius(
        parent_centerline, parent_lumen_field, side_branch.parent_arclength_frac,
    )
    daughter_centerline = build_daughter_centerline(
        parent_centerline,
        side_branch.parent_arclength_frac,
        side_branch.polar_deg,
        side_branch.azimuth_deg,
        side_branch.branch.centerline,
        parent_local_radius_mm=parent_local_radius,
    )
    return _attach_daughter_branch(
        parent_lumen_mesh, parent_outer_mesh, parent_centerline,
        daughter_centerline, side_branch.branch, rng, name=side_branch.branch.name,
        attachment_arclength_mm=side_branch.parent_arclength_frac * parent_centerline.length_mm,
    )


def _attach_daughter_branch(
    parent_lumen_mesh: trimesh.Trimesh,
    parent_outer_mesh: trimesh.Trimesh,
    parent_centerline: Centerline,
    daughter_centerline: Centerline,
    branch_cfg: BranchConfig,
    rng: np.random.Generator,
    name: str,
    attachment_arclength_mm: float,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, DaughterArtifacts]:
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
    wall_field = build_wall(
        branch_cfg.wall,
        lumen_field,
        daughter_centerline.length_mm,
        daughter_rng,
    )
    daughter_lumen_mesh, daughter_outer_mesh = sweep_branch(
        daughter_centerline, lumen_field, wall_field
    )

    new_lumen = _safe_union(parent_lumen_mesh, daughter_lumen_mesh, label=f"{name}/lumen")
    new_outer = _safe_union(parent_outer_mesh, daughter_outer_mesh, label=f"{name}/outer")

    return new_lumen, new_outer, DaughterArtifacts(
        name=name,
        centerline=daughter_centerline,
        lumen_field=lumen_field,
        wall_field=wall_field,
        parent_attachment_arclength_mm=attachment_arclength_mm,
        daughter_lumen_mesh=daughter_lumen_mesh,
        daughter_outer_mesh=daughter_outer_mesh,
    )
