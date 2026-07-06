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
    attach_side_branch_layered,
)
from vesselgen.centerline import Centerline
from vesselgen.config import (
    BranchConfig,
    LayeredWallConfig,
    VesselConfig,
    WallConfig,
    branch_wall_to_layered,
)
from vesselgen.cross_section import CrossSectionField, build_cross_sections
from vesselgen.sweep import sweep_branch, sweep_layered_branch
from vesselgen.wall import (
    LayeredWallField,
    WallField,
    build_layered_wall,
    build_wall,
    layered_wall_from_wall_field,
    push_layered_interfaces_around_lesions,
)


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
    layered_wall_field: Optional[LayeredWallField] = None
    """Populated when the branch's wall is described by a
    :class:`LayeredWallConfig`. Lets the lesion builder query interior
    interface radii."""


@dataclass
class SurfaceEntry:
    """One closed mesh exported to the simulator with a single material.

    The simulator transitions a ray's current material to ``material_name``
    when the ray crosses this surface from the inside-of-the-volume side
    to the outside. In ``vesselgen``'s nested mesh convention each entry
    sits radially inside the next, so ``material_name`` is the material
    of the shell *just outside* this surface.

    For a 3-layer (intima/media/adventitia) wall the emitted entries are::

        ("lumen",       "intima",     lumen mesh)
        ("interface_01","media",      intima/media interface)
        ("interface_02","adventitia", media/adventitia interface)

    The simulator's world background is ``"lumen"``, so a ray fired from
    the probe starts in the lumen, crosses the lumen mesh into the
    intima, and so on. Beyond ``interface_02`` the ray stays in
    ``adventitia`` until it exits the FOV -- there is no closed outer
    boundary on the adventitia (its back is acoustically invisible).

    Legacy single-slab vessels (no per-layer breakdown) emit two
    entries instead -- ``("lumen", "vessel_wall", ...)`` and
    ``("outer", "extravascular", ...)`` -- so the historical CT-pullback
    calibration still loads cleanly.
    """

    name: str
    material_name: str
    mesh: trimesh.Trimesh
    obj_filename: str
    """Path relative to the vessel directory."""


@dataclass
class Vessel:
    """A complete vessel: branch tree + unified watertight meshes."""

    config: VesselConfig
    branches: list[BranchHandle]
    lumen_mesh: trimesh.Trimesh
    parent_branch_id: int = 0
    daughter_artifacts: list[DaughterArtifacts] = field(default_factory=list)
    surfaces: list[SurfaceEntry] = field(default_factory=list)
    """All meshes the simulator should load, in nested order from
    innermost (lumen) to outermost emitted surface. For a 3-layer wall
    that is ``[lumen, interface_01, interface_02]``; for legacy
    single-slab vessels it is ``[lumen, outer]``."""

    lesions: list = field(default_factory=list)
    """List of :class:`vesselgen.inclusions.LesionMesh` records."""

    guidewire: Optional[object] = None
    """Optional :class:`vesselgen.guidewire.GuidewireArtifacts`."""

    world_background_material: str = "lumen"
    """Material the simulator should use for rays outside any mesh
    (the lumen blood pool, since the probe sits inside the lumen)."""

    @property
    def outer_mesh(self) -> trimesh.Trimesh:
        """Outermost emitted surface mesh.

        For a layered wall this is the media/adventitia interface; for a
        legacy single-slab wall it is the adventitia outer boundary.
        Used by sampling (``ground_truth_at`` slices it for ``d_outer``)
        and by visualisation for bbox computation. Always equal to
        ``self.surfaces[-1].mesh``.
        """
        if not self.surfaces:
            raise AttributeError(
                "Vessel has no surfaces; cannot derive outer_mesh."
            )
        return self.surfaces[-1].mesh

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: VesselConfig) -> "Vessel":
        cfg = cfg.with_minimum_side_branch_lengths()
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

        parent_layered_field, parent_layer_meshes = _build_parent_wall_meshes(
            cfg.parent.wall, parent_lumen_field, parent_centerline, parent_rng
        )
        parent_wall_field = parent_layered_field.to_outer_wall_field()

        branches: list[BranchHandle] = [
            BranchHandle(
                name=cfg.parent.name,
                branch_id=0,
                centerline=parent_centerline,
                lumen_field=parent_lumen_field,
                wall_field=parent_wall_field,
                is_parent=True,
                parent_attachment_arclength_mm=None,
                layered_wall_field=parent_layered_field,
            )
        ]

        # ``parent_layer_meshes`` carries n_layers + 1 entries: the
        # innermost lumen mesh, every interior interface, and finally
        # the outer (adventitia outer) mesh. We thread *all* of them
        # through bifurcation (so per-layer unions stay nested) but
        # drop the outermost when we build the simulator-facing
        # ``surfaces`` list -- the new world model lets rays exit the
        # outermost emitted surface into adventitia without a hard
        # back-boundary echo.
        cur_layer_meshes = parent_layer_meshes
        all_artifacts: list[DaughterArtifacts] = []
        next_branch_id = 1

        for sb in cfg.side_branches:
            try:
                cur_layer_meshes, art = attach_side_branch_layered(
                    cur_layer_meshes,
                    parent_centerline,
                    parent_lumen_field,
                    sb,
                    rng_root,
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
                    layered_wall_field=art.layered_wall_field,
                )
            )
            next_branch_id += 1

        from vesselgen.inclusions import build_all_lesions
        lesions = build_all_lesions(
            list(cfg.lesions),
            parent_centerline,
            parent_lumen_field,
            parent_layered_field,
        )

        # Push every interior wall interface (intima/media, media/adventitia)
        # outboard of every lesion footprint before sweeping the surface
        # meshes. Without this the lesion mesh and the wall interface meshes
        # overlap, which the simulator's two-material ray bookkeeping cannot
        # represent -- producing the canonical "ring" artifact (bright
        # proximal arc + bright distal arc with a dark core, no shadow).
        # We only do this on bifurcation-free vessels because the boolean
        # union for side branches has already locked in the parent layer
        # geometry.
        if cfg.lesions and not cfg.side_branches:
            parent_layered_field = push_layered_interfaces_around_lesions(
                parent_layered_field,
                parent_centerline,
                list(cfg.lesions),
            )
            cur_layer_meshes = sweep_layered_branch(
                parent_centerline, parent_lumen_field, parent_layered_field
            )
            branches[0] = BranchHandle(
                name=branches[0].name,
                branch_id=branches[0].branch_id,
                centerline=branches[0].centerline,
                lumen_field=branches[0].lumen_field,
                wall_field=parent_layered_field.to_outer_wall_field(),
                is_parent=True,
                parent_attachment_arclength_mm=None,
                layered_wall_field=parent_layered_field,
            )

        surfaces = _build_emitted_surface_list(cur_layer_meshes, cfg.parent.wall)

        guidewire = None
        if cfg.guidewire is not None:
            from vesselgen.guidewire import build_guidewire
            guidewire = build_guidewire(cfg.guidewire, parent_centerline)

        return cls(
            config=cfg,
            branches=branches,
            lumen_mesh=surfaces[0].mesh,
            parent_branch_id=0,
            daughter_artifacts=all_artifacts,
            surfaces=surfaces,
            lesions=lesions,
            guidewire=guidewire,
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
        side_branch_ostium_bias_prob: float = 0.0,
        side_branch_ostium_arclength_frac: float = 0.25,
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
            side_branch_ostium_bias_prob=side_branch_ostium_bias_prob,
            side_branch_ostium_arclength_frac=side_branch_ostium_arclength_frac,
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

    def ground_truth_at(
        self,
        pose,
        n_angles: int = 360,
        max_distance_mm: float = 30.0,
    ):
        from vesselgen.sampling import ground_truth_at
        return ground_truth_at(
            self, pose, n_angles=n_angles, max_distance_mm=max_distance_mm
        )

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

    # -----------------------------------------------------------------
    # Mesh / material lookup
    # -----------------------------------------------------------------

    def all_mesh_records(self) -> list[dict]:
        """Flat list of dicts ``{name, material, mesh, kind}`` for every
        mesh the simulator should load. ``kind`` is ``"surface"``,
        ``"lesion"``, or ``"guidewire"``."""

        out: list[dict] = []
        for s in self.surfaces:
            out.append({
                "name": s.name,
                "material": s.material_name,
                "mesh": s.mesh,
                "obj_filename": s.obj_filename,
                "kind": "surface",
            })
        for lesion in self.lesions:
            out.append({
                "name": lesion.name,
                "material": lesion.material_name,
                "mesh": lesion.mesh,
                "obj_filename": f"lesions/{lesion.name}.obj",
                "kind": "lesion",
            })
        if self.guidewire is not None:
            out.append({
                "name": "guidewire",
                "material": self.guidewire.material_name,
                "mesh": self.guidewire.mesh,
                "obj_filename": "guidewire.obj",
                "kind": "guidewire",
            })
        return out

    def manifest_dict(self) -> dict:
        cfg = self.config

        def wall_cfg_to_dict(wall) -> dict:
            if isinstance(wall, LayeredWallConfig):
                return {
                    "kind": "layered",
                    "total_thickness_mm": wall.total_thickness_mm,
                    "min_thickness_mm": wall.min_thickness_mm,
                    "layers": [
                        {
                            "material_name": ly.material_name,
                            "thickness_frac": ly.thickness_frac,
                            "max_perturbation_frac": ly.max_perturbation_frac,
                        }
                        for ly in wall.layers
                    ],
                }
            return {
                "kind": "single",
                "mean_thickness_mm": wall.mean_thickness_mm,
                "perturbation_modes": list(wall.perturbation_modes),
                "max_perturbation_frac": wall.max_perturbation_frac,
                "perturbation_decay": wall.perturbation_decay,
                "phase_drift_per_mm": wall.phase_drift_per_mm,
                "min_thickness_mm": wall.min_thickness_mm,
            }

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
                "wall": wall_cfg_to_dict(b.wall),
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
            "outermost_mesh": {"n_vertices": int(len(self.outer_mesh.vertices)),
                                "n_faces": int(len(self.outer_mesh.faces)),
                                "is_watertight": bool(self.outer_mesh.is_watertight)},
            "world": {"background_material": self.world_background_material},
            "surfaces": [
                {
                    "name": s.name,
                    "material": s.material_name,
                    "obj": s.obj_filename,
                }
                for s in self.surfaces
            ],
            "lesions": [
                {
                    "name": lesion.name,
                    "material": lesion.material_name,
                    "obj": f"lesions/{lesion.name}.obj",
                    "kind": lesion.config.kind,
                    "config": {
                        "arclength_frac": lesion.config.arclength_frac,
                        "azimuth_deg": lesion.config.azimuth_deg,
                        "arc_extent_deg": lesion.config.arc_extent_deg,
                        "axial_extent_mm": lesion.config.axial_extent_mm,
                        "inner_offset_frac": lesion.config.inner_offset_frac,
                        "outer_offset_frac": lesion.config.outer_offset_frac,
                        "seed": lesion.config.seed,
                    },
                }
                for lesion in self.lesions
            ],
            "guidewire": (
                None if self.guidewire is None else {
                    "material": self.guidewire.material_name,
                    "obj": "guidewire.obj",
                    "config": {
                        "diameter_mm": self.guidewire.config.diameter_mm,
                        "lateral_offset_mm": self.guidewire.config.lateral_offset_mm,
                        "offset_azimuth_deg": self.guidewire.config.offset_azimuth_deg,
                        "material_name": self.guidewire.config.material_name,
                    },
                }
            ),
        }


# ---------------------------------------------------------------------------
# Internal helpers for Vessel.from_config
# ---------------------------------------------------------------------------


def _build_parent_wall_meshes(
    wall_cfg,
    lumen: CrossSectionField,
    centerline: Centerline,
    rng: np.random.Generator,
) -> tuple[LayeredWallField, list[trimesh.Trimesh]]:
    """Build the layered-wall field + its closed surface meshes.

    Returns the field and ``[lumen_surface, interior_interface_0, ...,
    outer_surface]`` meshes (``n_layers + 1`` entries). The outermost
    mesh is kept here because the bifurcation pipeline still needs
    something to union daughters against, but it gets dropped before
    being exported in :func:`_build_emitted_surface_list`. Legacy
    single-layer :class:`WallConfig` is wrapped as a one-layer layered
    field so the rest of the pipeline always sees the same shape.
    """

    if isinstance(wall_cfg, LayeredWallConfig):
        field = build_layered_wall(wall_cfg, lumen, centerline.length_mm, rng)
        meshes = sweep_layered_branch(centerline, lumen, field)
        return field, meshes

    wall_field = build_wall(wall_cfg, lumen, centerline.length_mm, rng)
    field = layered_wall_from_wall_field(
        wall_field,
        lumen,
        material_name="vessel_wall",
        total_thickness_mm=wall_cfg.mean_thickness_mm,
        min_thickness_mm=wall_cfg.min_thickness_mm,
    )
    lumen_mesh, outer_mesh = sweep_branch(centerline, lumen, wall_field)
    return field, [lumen_mesh, outer_mesh]


def _emitted_material_chain(wall_cfg) -> list[str]:
    """Material on the *outside* of each emitted closed surface, in nested order.

    The simulator transitions a ray's material to the chain entry when
    it crosses the matching surface from inside-of-the-volume to
    outside. For a 3-layer wall that yields ``["intima", "media",
    "adventitia"]`` paired with ``[lumen, interface_01, interface_02]``,
    so a ray fired from the probe (which sits inside the lumen mesh in
    the world-background ``"lumen"`` material) progresses lumen ->
    intima -> media -> adventitia, and stays in adventitia past the
    outermost emitted surface.

    Legacy single-slab walls still emit two surfaces (lumen, outer)
    paired with ``["vessel_wall", "extravascular"]`` so the historical
    CT-pullback calibration loads unchanged.
    """

    if isinstance(wall_cfg, LayeredWallConfig):
        return [ly.material_name for ly in wall_cfg.layers]
    return ["vessel_wall", "extravascular"]


def _build_emitted_surface_list(
    layer_meshes: list[trimesh.Trimesh], wall_cfg
) -> list[SurfaceEntry]:
    """Build the simulator-facing :class:`SurfaceEntry` list.

    ``layer_meshes`` is the full ``n_layers + 1`` list returned by
    :func:`_build_parent_wall_meshes` (or the analogous bifurcation
    output): innermost lumen mesh, each interior interface, and the
    outer adventitia boundary. For a layered wall we drop the
    outermost mesh -- nothing transitions a ray's material there
    anymore -- and emit ``n_layers`` surfaces.

    Legacy single-slab walls keep both meshes (``[lumen, outer]``) so
    raysim still sees the historical extravascular back-boundary.
    """

    chain = _emitted_material_chain(wall_cfg)
    n_emit = len(chain)
    if n_emit > len(layer_meshes):
        raise RuntimeError(
            "internal error: emitted material chain length "
            f"{n_emit} > number of layer meshes {len(layer_meshes)}"
        )
    emit_meshes = layer_meshes[:n_emit]
    is_layered = isinstance(wall_cfg, LayeredWallConfig)

    entries: list[SurfaceEntry] = []
    for i, (mesh, material) in enumerate(zip(emit_meshes, chain)):
        if i == 0:
            name = "lumen"
            obj_filename = "lumen.obj"
        elif not is_layered and i == n_emit - 1:
            name = "outer"
            obj_filename = "outer.obj"
        else:
            name = f"interface_{i:02d}"
            obj_filename = f"surfaces/{name}.obj"
        entries.append(SurfaceEntry(
            name=name,
            material_name=material,
            mesh=mesh,
            obj_filename=obj_filename,
        ))
    return entries
