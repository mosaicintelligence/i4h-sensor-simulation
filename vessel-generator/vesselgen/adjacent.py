"""Parallel neighbor vessels alongside the parent.

An adjacent vessel is a separate, non-touching tube offset laterally in the
parent's cross-section plane (the "artery next to a vein" case). Unlike
:mod:`vesselgen.bifurcation`, neighbors do not fuse into the parent wall;
their per-layer surface meshes are concatenated into the parent's at build
time, so they share materials/labels and flow through ground truth, the
rasterizer, previews and raysim unchanged.

The catheter stays in the parent lumen -- neighbors are not added to the
pose sampler's branch list.

Exports:

* :func:`place_adjacent_vessels` -- azimuth and center-to-center offset for
  1--2 neighbors with provably non-overlapping outer walls.
* :func:`measured_outer_radius_mm` -- exact outer radius of a branch's meshed
  geometry (via :func:`vesselgen.wall.build_branch_fields`).
* :func:`build_adjacent_neighbor` -- sweep one neighbor and offset its
  centerline from the parent.
* :func:`rescale_wall_to_thickness` -- copy a wall config at a new thickness
  while preserving layer structure (used when sizing neighbor walls).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

import numpy as np
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import (
    AdjacentVesselConfig,
    BranchConfig,
    LayeredWallConfig,
    LayerSpec,
    WallConfig,
)
from vesselgen.cross_section import CrossSectionField
from vesselgen.sweep import sweep_layered_branch
from vesselgen.wall import build_branch_fields


def measured_outer_radius_mm(branch: BranchConfig) -> float:
    """Exact maximum outer-wall radius (mm) of the mesh a branch will build.

    A branch's geometry is deterministic in its seed, so the outer surface
    can be reconstructed here (via :func:`vesselgen.wall.build_branch_fields`,
    the same draw the mesh builder uses) without building the mesh, and this
    returns the true maximum outer radius over every station and angle.

    Because it is the exact maximum (not a loose analytic bound that sums
    the independent worst-case lumen and wall perturbations, which peak at
    different angles), it lets placement seat neighbors as close as
    possible while staying provably non-overlapping: every swept vertex
    lies at or inside this radius from the branch axis, so two branches
    whose center-to-center distance is at least the sum of their measured
    radii cannot touch.
    """
    seed = branch.seed if branch.seed is not None else 0
    rng = np.random.default_rng(seed)
    _lumen, layered_wall_field = build_branch_fields(branch, rng)
    return float(layered_wall_field.interface_radii[-1].max())


def adjacent_min_separation_deg(
    center_offset_a_mm: float,
    center_offset_b_mm: float,
    min_center_distance_mm: float,
) -> float:
    """Minimum angular separation (deg) between two neighbors of the parent.

    Two neighbors sit at distances ``center_offset_a``/``center_offset_b``
    from the parent center. By the law of cosines their center-to-center
    distance is ``sqrt(a^2 + b^2 - 2 a b cos(delta))``. Requiring that to
    be ``>= min_center_distance_mm`` (the sum of their measured outer radii
    plus the desired gap) and solving for ``delta`` yields this minimum
    separation; any ``|delta| >= result`` keeps their walls clear. The
    ``clip`` handles the degenerate case where the neighbors clear at any
    angle (returns 0).
    """
    cos_delta = (
        center_offset_a_mm**2 + center_offset_b_mm**2 - min_center_distance_mm**2
    ) / (2.0 * center_offset_a_mm * center_offset_b_mm)
    cos_delta = float(np.clip(cos_delta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_delta)))


def rescale_wall_to_thickness(
    wall: Union[WallConfig, LayeredWallConfig],
    new_total_thickness_mm: float,
    *,
    max_perturbation_cap: float = 0.4,
) -> Union[WallConfig, LayeredWallConfig]:
    """Copy ``wall`` with its total thickness set to ``new_total_thickness_mm``.

    Preserves the layer count and per-layer thickness fractions (so an
    adjacent neighbor keeps the parent's wall structure) while rescaling
    the total thickness to the neighbor's own smaller wall and capping
    perturbation amplitudes (a smaller tube reads cleaner with a gentler
    sweep, matching the side-branch convention).
    """
    if isinstance(wall, LayeredWallConfig):
        return LayeredWallConfig(
            layers=[
                LayerSpec(
                    material_name=ly.material_name,
                    thickness_frac=ly.thickness_frac,
                    max_perturbation_frac=min(ly.max_perturbation_frac, max_perturbation_cap),
                )
                for ly in wall.layers
            ],
            total_thickness_mm=new_total_thickness_mm,
            min_thickness_mm=wall.min_thickness_mm,
        )
    return WallConfig(
        mean_thickness_mm=new_total_thickness_mm,
        max_perturbation_frac=min(wall.max_perturbation_frac, max_perturbation_cap),
    )


def place_adjacent_vessels(
    parent: BranchConfig,
    neighbor_branches: list[BranchConfig],
    gaps_mm: list[float],
    rng: np.random.Generator,
    *,
    first_azimuth_deg: Optional[float] = None,
) -> list[AdjacentVesselConfig]:
    """Assign each neighbor a lateral offset + azimuth around the parent.

    Placement is as tight as the geometry allows: the center-to-center
    offset for each neighbor is ``measured(parent) + measured(neighbor) +
    gap`` using :func:`measured_outer_radius_mm` (the exact maximum outer
    radius of the mesh each branch will build), so the realised
    edge-to-edge clearance is exactly ``gap`` and the outer walls are still
    guaranteed never to touch.

    The first neighbor's azimuth is ``first_azimuth_deg`` (or uniform-random
    when ``None``). A second neighbor is packed right next to the first, at
    the minimum angular separation that still clears it
    (:func:`adjacent_min_separation_deg`), so multiple neighbors form a
    tight cluster on one side of the parent rather than spreading around it.
    Clustering keeps the narrow tissue gaps between vessels concentrated in
    a small sector, which is where the catheter ring-down obscures the
    vessel-vessel boundary.

    Supports 1 or 2 neighbors (the configured maximum).
    """
    if len(neighbor_branches) != len(gaps_mm):
        raise ValueError("neighbor_branches and gaps_mm must have equal length")
    if len(neighbor_branches) > 2:
        raise ValueError("place_adjacent_vessels supports at most 2 neighbors")

    parent_r = measured_outer_radius_mm(parent)
    neighbor_r = [measured_outer_radius_mm(nb) for nb in neighbor_branches]
    offsets = [parent_r + nb_r + float(g) for nb_r, g in zip(neighbor_r, gaps_mm)]

    placed: list[AdjacentVesselConfig] = []
    azimuths: list[float] = []
    for k, (nb, offset) in enumerate(zip(neighbor_branches, offsets)):
        if k == 0:
            az = (
                float(rng.uniform(0.0, 360.0))
                if first_azimuth_deg is None
                else float(first_azimuth_deg)
            )
        else:
            min_center = neighbor_r[0] + neighbor_r[k] + float(gaps_mm[k])
            delta_min = adjacent_min_separation_deg(offsets[0], offset, min_center)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            az = azimuths[0] + sign * delta_min
        azimuths.append(az)
        placed.append(
            AdjacentVesselConfig(
                azimuth_deg=float(az % 360.0),
                center_offset_mm=float(offset),
                branch=nb,
            )
        )
    return placed


def build_adjacent_neighbor(
    adj: AdjacentVesselConfig,
    parent_centerline: Centerline,
) -> tuple[list[trimesh.Trimesh], Centerline, CrossSectionField]:
    """Build one neighbor's per-layer meshes, offset laterally from the parent.

    The neighbor runs parallel to the parent (same direction), shifted in
    the parent's cross-section plane by ``center_offset_mm`` along
    ``azimuth_deg`` (measured in the parent's local normal/binormal basis).
    Returns ``(layer_meshes, centerline, lumen_field)`` where
    ``layer_meshes`` has the same length as the parent's (lumen + interior
    interfaces + outer), so it can be concatenated index-wise.
    """
    normal = parent_centerline.normals[0]
    binormal = parent_centerline.binormals[0]
    az = np.radians(adj.azimuth_deg)
    offset = adj.center_offset_mm * (np.cos(az) * normal + np.sin(az) * binormal)
    base_origin = np.asarray(adj.branch.centerline.origin, dtype=float)
    neighbor_cl_cfg = replace(adj.branch.centerline, origin=tuple((base_origin + offset).tolist()))
    neighbor_centerline = Centerline(neighbor_cl_cfg)
    neighbor_rng = np.random.default_rng(adj.branch.seed if adj.branch.seed is not None else 0)
    neighbor_lumen, neighbor_layered_field = build_branch_fields(adj.branch, neighbor_rng)
    neighbor_meshes = sweep_layered_branch(
        neighbor_centerline, neighbor_lumen, neighbor_layered_field
    )
    return neighbor_meshes, neighbor_centerline, neighbor_lumen
