"""Top-level Vessel object.

A :class:`Vessel` holds:
- the parent :class:`Centerline`, lumen field, wall field and meshes
- per-daughter :class:`bifurcation.DaughterArtifacts` (one per attached
  side branch)
- the unified watertight lumen and outer meshes (after boolean union of
  parent + all daughters)
- a list of :class:`BranchHandle` objects that the sampling code uses to
  pick a branch, an arclength, and a local frame for pose sampling

A vessel is built from a :class:`VesselConfig` via :meth:`Vessel.from_config`,
saved with :meth:`save`, and reloaded with :meth:`load`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from vesselgen.bifurcation import (
    BifurcationError,
    DaughterArtifacts,
    attach_side_branch,
)
from vesselgen.centerline import Centerline
from vesselgen.config import BranchConfig, VesselConfig
from vesselgen.cross_section import CrossSectionField, build_cross_sections
from vesselgen.sweep import sweep_branch
from vesselgen.wall import WallField, build_wall


@dataclass
class BranchHandle:
    """Lightweight handle the sampler uses to pick a branch.

    Stores the centerline, the lumen radii field, and the branch metadata
    needed to compute ground truth at a sampled pose.
    """

    name: str
    branch_id: int
    centerline: Centerline
    lumen_field: CrossSectionField
    wall_field: WallField
    is_parent: bool = False
    parent_attachment_arclength_mm: Optional[float] = None
    """For daughters: arclength on the parent centerline at which this
    daughter was attached. None for the parent itself."""


@dataclass
class Vessel:
    """A complete vessel: branch tree + unified watertight meshes."""

    config: VesselConfig
    branches: list[BranchHandle]
    lumen_mesh: trimesh.Trimesh
    outer_mesh: trimesh.Trimesh
    parent_branch_id: int = 0
    daughter_artifacts: list[DaughterArtifacts] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: VesselConfig) -> "Vessel":
        rng_root = np.random.default_rng(cfg.seed)

        parent_centerline = Centerline(cfg.parent.centerline)
        parent_rng = np.random.default_rng(
            cfg.parent.seed if cfg.parent.seed is not None else cfg.seed
        )
        parent_lumen_field = build_cross_sections(
            cfg.parent.cross_section,
            parent_centerline.length_mm,
            cfg.parent.centerline.n_stations,
            parent_rng,
        )
        parent_wall_field = build_wall(
            cfg.parent.wall,
            parent_lumen_field,
            parent_centerline.length_mm,
            parent_rng,
        )
        parent_lumen_mesh, parent_outer_mesh = sweep_branch(
            parent_centerline, parent_lumen_field, parent_wall_field
        )

        branches: list[BranchHandle] = [
            BranchHandle(
                name=cfg.parent.name,
                branch_id=0,
                centerline=parent_centerline,
                lumen_field=parent_lumen_field,
                wall_field=parent_wall_field,
                is_parent=True,
                parent_attachment_arclength_mm=None,
            )
        ]

        cur_lumen = parent_lumen_mesh
        cur_outer = parent_outer_mesh
        all_artifacts: list[DaughterArtifacts] = []
        next_branch_id = 1

        for sb in cfg.side_branches:
            try:
                cur_lumen, cur_outer, art = attach_side_branch(
                    cur_lumen, cur_outer, parent_centerline, parent_lumen_field,
                    sb, rng_root,
                )
            except BifurcationError as e:
                raise BifurcationError(f"side branch '{sb.branch.name}': {e}") from e
            all_artifacts.append(art)
            branches.append(
                BranchHandle(
                    name=art.name,
                    branch_id=next_branch_id,
                    centerline=art.centerline,
                    lumen_field=art.lumen_field,
                    wall_field=art.wall_field,
                    is_parent=False,
                    parent_attachment_arclength_mm=art.parent_attachment_arclength_mm,
                )
            )
            next_branch_id += 1

        return cls(
            config=cfg,
            branches=branches,
            lumen_mesh=cur_lumen,
            outer_mesh=cur_outer,
            parent_branch_id=0,
            daughter_artifacts=all_artifacts,
        )

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    @property
    def parent_branch(self) -> BranchHandle:
        return self.branches[self.parent_branch_id]

    def branch_by_name(self, name: str) -> BranchHandle:
        for b in self.branches:
            if b.name == name:
                return b
        raise KeyError(name)

    def total_lumen_arclength_mm(self) -> float:
        return float(sum(b.centerline.length_mm for b in self.branches))

    # -----------------------------------------------------------------
    # Sampling (delegated to vesselgen.sampling for the hot loop)
    # -----------------------------------------------------------------

    def sample_pose(
        self,
        rng: np.random.Generator,
        max_tilt_deg: float = 15.0,
        edge_margin_mm: float = 0.2,
        max_attempts: int = 64,
    ):
        """Draw a fresh random pose anywhere inside the vessel lumen.

        Use this for arbitrary-pose simulator training. Every call returns
        a different (position, rotation) with no preselection: branch is
        sampled weighted by arclength, then arclength uniformly along that
        branch, then a 2D in-lumen position rejection-sampled, then a
        small probe-axis tilt.
        """
        from vesselgen.sampling import sample_pose
        return sample_pose(
            self, rng,
            max_tilt_deg=max_tilt_deg,
            edge_margin_mm=edge_margin_mm,
            max_attempts=max_attempts,
        )

    def sample_pose_in_branch(
        self,
        branch_name: str,
        rng: np.random.Generator,
        arclength_mm: float | None = None,
        max_tilt_deg: float = 15.0,
        edge_margin_mm: float = 0.2,
        max_attempts: int = 64,
    ):
        """Sample a pose constrained to one named branch (and optionally one
        arclength station)."""
        from vesselgen.sampling import sample_pose_in_branch
        return sample_pose_in_branch(
            self, branch_name, rng,
            arclength_mm=arclength_mm,
            max_tilt_deg=max_tilt_deg,
            edge_margin_mm=edge_margin_mm,
            max_attempts=max_attempts,
        )

    def pose_at(
        self,
        position,
        probe_axis_world=None,
        require_inside_lumen: bool = True,
    ):
        """Build a pose at a user-specified 3D position (no random sampling).

        Useful when you already know exactly where you want the probe (e.g.
        for replay, regression tests, or hand-crafted test cases). The
        probe long axis defaults to the local tangent of the nearest
        branch's centerline if not supplied.
        """
        from vesselgen.sampling import pose_at
        return pose_at(self, position, probe_axis_world=probe_axis_world,
                       require_inside_lumen=require_inside_lumen)

    def contains_point(self, position) -> bool:
        """True iff ``position`` (3,) lies inside the lumen."""
        return bool(self.lumen_mesh.contains([np.asarray(position, dtype=float)])[0])

    def ground_truth_at(self, pose, n_angles: int = 360):
        from vesselgen.sampling import ground_truth_at
        return ground_truth_at(self, pose, n_angles=n_angles)

    # -----------------------------------------------------------------
    # IO (full IO lives in vesselgen.io to keep this module light)
    # -----------------------------------------------------------------

    def save(self, out_dir: str | Path) -> Path:
        from vesselgen.io import save_vessel
        return save_vessel(self, out_dir)

    @classmethod
    def load(cls, in_dir: str | Path) -> "Vessel":
        from vesselgen.io import load_vessel
        return load_vessel(in_dir)

    # -----------------------------------------------------------------
    # Serialisation helpers (for io.py)
    # -----------------------------------------------------------------

    def manifest_dict(self) -> dict:
        cfg = self.config

        def branch_cfg_to_dict(b: BranchConfig) -> dict:
            return {
                "name": b.name,
                "seed": b.seed,
                "centerline": {
                    "length_mm": b.centerline.length_mm,
                    "origin": list(b.centerline.origin),
                    "direction": list(b.centerline.direction),
                    "n_stations": b.centerline.n_stations,
                },
                "cross_section": {
                    "mean_radius_mm": b.cross_section.mean_radius_mm,
                    "distal_radius_mm": b.cross_section.distal_radius_mm,
                    "taper_profile": b.cross_section.taper_profile,
                    "perturbation_modes": list(b.cross_section.perturbation_modes),
                    "max_perturbation_frac": b.cross_section.max_perturbation_frac,
                    "perturbation_decay": b.cross_section.perturbation_decay,
                    "phase_drift_per_mm": b.cross_section.phase_drift_per_mm,
                    "n_angles": b.cross_section.n_angles,
                },
                "wall": {
                    "mean_thickness_mm": b.wall.mean_thickness_mm,
                    "perturbation_modes": list(b.wall.perturbation_modes),
                    "max_perturbation_frac": b.wall.max_perturbation_frac,
                    "perturbation_decay": b.wall.perturbation_decay,
                    "phase_drift_per_mm": b.wall.phase_drift_per_mm,
                    "min_thickness_mm": b.wall.min_thickness_mm,
                },
            }

        return {
            "name": cfg.name,
            "seed": cfg.seed,
            "axis_convention": {"vessel_axis": "Y", "cross_section_plane": "xz", "units": "mm"},
            "parent": branch_cfg_to_dict(cfg.parent),
            "side_branches": [
                {
                    "parent_arclength_frac": sb.parent_arclength_frac,
                    "azimuth_deg": sb.azimuth_deg,
                    "polar_deg": sb.polar_deg,
                    "branch": branch_cfg_to_dict(sb.branch),
                }
                for sb in cfg.side_branches
            ],
            "branches": [
                {
                    "name": b.name,
                    "branch_id": b.branch_id,
                    "is_parent": b.is_parent,
                    "parent_attachment_arclength_mm": b.parent_attachment_arclength_mm,
                    "centerline": b.centerline.to_dict(),
                    "lumen": {
                        "n_angles": b.lumen_field.n_angles,
                        "n_stations": b.lumen_field.n_stations,
                        "mean_radius_mm": b.lumen_field.mean_radius.tolist(),
                    },
                    "wall": {
                        "mean_thickness_mm": b.wall_field.mean_thickness,
                    },
                }
                for b in self.branches
            ],
            "lumen_mesh": {"n_vertices": int(len(self.lumen_mesh.vertices)),
                            "n_faces": int(len(self.lumen_mesh.faces)),
                            "is_watertight": bool(self.lumen_mesh.is_watertight)},
            "outer_mesh": {"n_vertices": int(len(self.outer_mesh.vertices)),
                            "n_faces": int(len(self.outer_mesh.faces)),
                            "is_watertight": bool(self.outer_mesh.is_watertight)},
        }
