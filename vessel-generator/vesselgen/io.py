"""Vessel IO: write meshes (OBJ) and per-vessel JSON manifests.

Layout of one vessel folder::

    out/<vessel_name>/
      lumen.obj          # closed inward-facing lumen surface
      outer.obj          # closed inward-facing outer surface
      vessel.json        # generation parameters + branch summary (manifest)
      branches/
        branch_<id>.json # per-branch centerline + lumen/wall fields

The OBJ files use trimesh's writer, which respects the inward-facing
winding produced by :mod:`vesselgen.sweep`. Loading them back into the
simulator with the materials::

    rs.Mesh("lumen.obj", materials.get_index("vessel_wall"))
    rs.Mesh("outer.obj", materials.get_index("extravascular"))

reproduces the configuration used by ``ivus_example.py``'s thick-cylinder
phantom, but with realistic vessel geometry instead of a circular tube.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    CrossSectionConfig,
    SideBranchConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.vessel import Vessel


def _save_inward_obj(mesh, out_path: Path) -> None:
    """Write ``mesh`` to OBJ with face winding flipped (so normals point inward)."""
    import trimesh as _trimesh

    flipped = mesh.copy()
    flipped.invert()
    flipped.export(out_path)


def save_vessel(vessel: Vessel, out_dir: str | Path) -> Path:
    """Write a vessel folder and return the directory path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    branches_dir = out_dir / "branches"
    branches_dir.mkdir(exist_ok=True)

    # In-memory meshes use outward normals so trimesh.contains and the
    # boolean engine work correctly. The simulator expects inward normals
    # on its phantom surfaces (see phantom_maker.generate_cylinder_thick_mesh
    # in the simulator repository), so we flip the winding at export time.
    _save_inward_obj(vessel.lumen_mesh, out_dir / "lumen.obj")
    _save_inward_obj(vessel.outer_mesh, out_dir / "outer.obj")

    manifest = vessel.manifest_dict()
    with (out_dir / "vessel.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    for b in vessel.branches:
        with (branches_dir / f"branch_{b.branch_id:02d}.json").open("w") as f:
            json.dump(
                {
                    "name": b.name,
                    "branch_id": b.branch_id,
                    "is_parent": b.is_parent,
                    "parent_attachment_arclength_mm": b.parent_attachment_arclength_mm,
                    "centerline": b.centerline.to_dict(),
                    "thetas_rad": b.lumen_field.thetas.tolist(),
                    "stations_mm": np.linspace(
                        0.0, b.centerline.length_mm, b.lumen_field.n_stations
                    ).tolist(),
                    "lumen_radii_mm": b.lumen_field.radii.tolist(),
                    "wall_thicknesses_mm": b.wall_field.thicknesses.tolist(),
                    "lumen_mean_radius_mm": b.lumen_field.mean_radius.tolist(),
                },
                f,
                indent=2,
            )
    return out_dir


# ---------------------------------------------------------------------------
# Loader. Note: the loaded vessel is reconstructed from the meshes on disk
# and the per-branch JSONs (which reproduce the lumen/wall fields exactly).
# We do not re-run the sweep + boolean pipeline because the meshes on disk
# are the canonical geometry; the per-branch fields are kept so the
# sampling code can interpolate lumen contours at arbitrary arclength.
# ---------------------------------------------------------------------------


def load_vessel(in_dir: str | Path) -> Vessel:
    import trimesh

    from vesselgen.centerline import Centerline
    from vesselgen.cross_section import CrossSectionField
    from vesselgen.vessel import BranchHandle
    from vesselgen.wall import WallField

    in_dir = Path(in_dir)
    with (in_dir / "vessel.json").open() as f:
        manifest = json.load(f)

    cfg = _vessel_config_from_manifest(manifest)

    branches: list[BranchHandle] = []
    branches_dir = in_dir / "branches"
    for branch_file in sorted(branches_dir.glob("branch_*.json")):
        with branch_file.open() as f:
            b = json.load(f)
        cl_cfg = CenterlineConfig(
            length_mm=b["centerline"]["length_mm"],
            origin=tuple(b["centerline"]["origin"]),
            direction=tuple(b["centerline"]["direction"]),
            n_stations=b["centerline"]["n_stations"],
        )
        cl = Centerline(cl_cfg)
        radii = np.asarray(b["lumen_radii_mm"], dtype=float)
        thicknesses = np.asarray(b["wall_thicknesses_mm"], dtype=float)
        thetas = np.asarray(b["thetas_rad"], dtype=float)
        mean_radius = np.asarray(b["lumen_mean_radius_mm"], dtype=float)
        lumen = CrossSectionField(thetas=thetas, radii=radii, mean_radius=mean_radius)
        wall = WallField(
            thicknesses=thicknesses,
            mean_thickness=float(np.mean(thicknesses)),
        )
        branches.append(
            BranchHandle(
                name=b["name"],
                branch_id=int(b["branch_id"]),
                centerline=cl,
                lumen_field=lumen,
                wall_field=wall,
                is_parent=bool(b["is_parent"]),
                parent_attachment_arclength_mm=b.get("parent_attachment_arclength_mm"),
            )
        )

    lumen_mesh = trimesh.load(in_dir / "lumen.obj", force="mesh", process=True)
    outer_mesh = trimesh.load(in_dir / "outer.obj", force="mesh", process=True)
    # On-disk OBJs use inward normals (simulator convention). Flip back to
    # outward so the loaded Vessel has the same in-memory convention as a
    # freshly built one (needed for trimesh.contains and any future boolean
    # ops on the loaded vessel).
    lumen_mesh.invert()
    outer_mesh.invert()

    return Vessel(
        config=cfg,
        branches=branches,
        lumen_mesh=lumen_mesh,
        outer_mesh=outer_mesh,
        parent_branch_id=0,
        daughter_artifacts=[],
    )


def _branch_config_from_manifest(b: dict) -> BranchConfig:
    cl = b["centerline"]
    cs = b["cross_section"]
    w = b["wall"]
    return BranchConfig(
        centerline=CenterlineConfig(
            length_mm=cl["length_mm"],
            origin=tuple(cl["origin"]),
            direction=tuple(cl["direction"]),
            n_stations=cl["n_stations"],
        ),
        cross_section=CrossSectionConfig(
            mean_radius_mm=cs["mean_radius_mm"],
            distal_radius_mm=cs["distal_radius_mm"],
            taper_profile=cs["taper_profile"],
            perturbation_modes=tuple(cs["perturbation_modes"]),
            max_perturbation_frac=cs["max_perturbation_frac"],
            perturbation_decay=cs["perturbation_decay"],
            phase_drift_per_mm=cs["phase_drift_per_mm"],
            n_angles=cs["n_angles"],
        ),
        wall=WallConfig(
            mean_thickness_mm=w["mean_thickness_mm"],
            perturbation_modes=tuple(w["perturbation_modes"]),
            max_perturbation_frac=w["max_perturbation_frac"],
            perturbation_decay=w["perturbation_decay"],
            phase_drift_per_mm=w["phase_drift_per_mm"],
            min_thickness_mm=w["min_thickness_mm"],
        ),
        name=b["name"],
        seed=b.get("seed"),
    )


def _vessel_config_from_manifest(m: dict) -> VesselConfig:
    side_branches = [
        SideBranchConfig(
            parent_arclength_frac=sb["parent_arclength_frac"],
            azimuth_deg=sb["azimuth_deg"],
            polar_deg=sb["polar_deg"],
            branch=_branch_config_from_manifest(sb["branch"]),
        )
        for sb in m.get("side_branches", [])
    ]
    return VesselConfig(
        parent=_branch_config_from_manifest(m["parent"]),
        side_branches=side_branches,
        seed=int(m["seed"]),
        name=m["name"],
    )
