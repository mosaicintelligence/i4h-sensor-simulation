"""Vessel IO: write meshes (OBJ) and per-vessel JSON manifests.

Layout of one vessel folder (3-layer trilaminar wall)::

    out/<vessel_name>/
      lumen.obj                 # closed lumen surface (material: intima)
      surfaces/
        interface_01.obj        # intima/media interface (material: media)
        interface_02.obj        # media/adventitia interface (material: adventitia)
      vessel.json               # generation parameters + manifest
      branches/
        branch_<id>.json        # per-branch centerline + lumen/wall fields

The 3 emitted surfaces let raysim reproduce the canonical
bright-dark-bright IVUS wall appearance. Beyond ``interface_02`` rays
stay in the ``adventitia`` material until the FOV (the adventitia's
back boundary is acoustically invisible in clinical IVUS).

Legacy single-slab vessels still emit ``[lumen.obj, outer.obj]`` so
the historical CT-pullback calibration (``vessel_wall`` /
``extravascular`` materials) loads unchanged.

OBJ files are written with face windings flipped so the simulator,
which expects inward-facing normals on phantom surfaces, sees the
right orientation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vesselgen.config import (
    BranchConfig,
    CalcificationLesionConfig,
    CenterlineConfig,
    CrossSectionConfig,
    GuidewireConfig,
    LayerSpec,
    LayeredWallConfig,
    SideBranchConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.vessel import Vessel


def _save_inward_obj(mesh, out_path: Path) -> None:
    """Write ``mesh`` to OBJ with face winding flipped (so normals point inward).

    The raysim simulator requires per-vertex normals (``v//vn`` faces); trimesh's
    default OBJ writer omits them, so we emit a minimal Wavefront file ourselves.
    """
    flipped = mesh.copy()
    flipped.invert()
    verts = flipped.vertices
    normals = flipped.vertex_normals
    faces = flipped.faces

    with out_path.open("w") as f:
        f.write("# vesselgen inward-facing mesh for raysim\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("\n")
        for i, j, k in faces:
            a, b, c = i + 1, j + 1, k + 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def save_vessel(vessel: Vessel, out_dir: str | Path) -> Path:
    """Write a vessel folder and return the directory path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    branches_dir = out_dir / "branches"
    branches_dir.mkdir(exist_ok=True)
    surfaces_dir = out_dir / "surfaces"
    lesions_dir = out_dir / "lesions"

    # In-memory meshes use outward normals so trimesh.contains and the
    # boolean engine work correctly. The simulator expects inward normals
    # on its phantom surfaces (see phantom_maker.generate_cylinder_thick_mesh
    # in the simulator repository), so we flip the winding at export time.
    needs_surfaces_dir = any(s.obj_filename.startswith("surfaces/") for s in vessel.surfaces)
    if needs_surfaces_dir:
        surfaces_dir.mkdir(exist_ok=True)
    for s in vessel.surfaces:
        _save_inward_obj(s.mesh, out_dir / s.obj_filename)

    if vessel.lesions:
        lesions_dir.mkdir(exist_ok=True)
        for lesion in vessel.lesions:
            _save_inward_obj(lesion.mesh, lesions_dir / f"{lesion.name}.obj")

    if vessel.guidewire is not None:
        _save_inward_obj(vessel.guidewire.mesh, out_dir / "guidewire.obj")

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
    from vesselgen.guidewire import GuidewireArtifacts, _local_offset
    from vesselgen.inclusions import LesionMesh
    from vesselgen.vessel import BranchHandle, SurfaceEntry
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

    surfaces: list[SurfaceEntry] = []
    for entry in manifest.get("surfaces", []):
        obj_path = in_dir / entry["obj"]
        if not obj_path.exists():
            continue
        mesh = trimesh.load(obj_path, force="mesh", process=True)
        # On-disk OBJs use inward normals (simulator convention). Flip back to
        # outward so the loaded Vessel has the same in-memory convention as a
        # freshly built one (needed for trimesh.contains and any future
        # boolean ops on the loaded vessel).
        mesh.invert()
        surfaces.append(
            SurfaceEntry(
                name=entry["name"],
                material_name=entry["material"],
                mesh=mesh,
                obj_filename=entry["obj"],
            )
        )

    if not surfaces:
        # Pre-manifest legacy layout: load lumen.obj + outer.obj directly.
        lumen_obj = in_dir / "lumen.obj"
        outer_obj = in_dir / "outer.obj"
        if lumen_obj.exists():
            mesh = trimesh.load(lumen_obj, force="mesh", process=True)
            mesh.invert()
            surfaces.append(
                SurfaceEntry(
                    name="lumen",
                    material_name="vessel_wall",
                    mesh=mesh,
                    obj_filename="lumen.obj",
                )
            )
        if outer_obj.exists():
            mesh = trimesh.load(outer_obj, force="mesh", process=True)
            mesh.invert()
            surfaces.append(
                SurfaceEntry(
                    name="outer",
                    material_name="extravascular",
                    mesh=mesh,
                    obj_filename="outer.obj",
                )
            )

    lesions: list[LesionMesh] = []
    for entry in manifest.get("lesions", []):
        obj_path = in_dir / entry["obj"]
        if not obj_path.exists():
            continue
        mesh = trimesh.load(obj_path, force="mesh", process=True)
        mesh.invert()
        cfg_dict = entry.get("config", {})
        lesion_cfg = CalcificationLesionConfig(
            arclength_frac=cfg_dict.get("arclength_frac", 0.5),
            azimuth_deg=cfg_dict.get("azimuth_deg", 0.0),
            kind=entry.get("kind", "hard"),
            arc_extent_deg=cfg_dict.get("arc_extent_deg", 50.0),
            axial_extent_mm=cfg_dict.get("axial_extent_mm", 6.0),
            inner_offset_frac=cfg_dict.get("inner_offset_frac", 0.0),
            outer_offset_frac=cfg_dict.get("outer_offset_frac", 0.75),
            seed=cfg_dict.get("seed"),
        )
        lesions.append(
            LesionMesh(
                name=entry["name"],
                material_name=entry["material"],
                mesh=mesh,
                config=lesion_cfg,
            )
        )

    guidewire = None
    gw_entry = manifest.get("guidewire")
    if gw_entry is not None:
        obj_path = in_dir / gw_entry["obj"]
        if obj_path.exists():
            mesh = trimesh.load(obj_path, force="mesh", process=True)
            mesh.invert()
            gw_cfg_dict = gw_entry.get("config", {})
            gw_cfg = GuidewireConfig(
                diameter_mm=gw_cfg_dict.get("diameter_mm", 0.36),
                lateral_offset_mm=gw_cfg_dict.get("lateral_offset_mm", 0.0),
                offset_azimuth_deg=gw_cfg_dict.get("offset_azimuth_deg", 0.0),
                material_name=gw_cfg_dict.get("material_name", "tungsten"),
            )
            parent_branch = next((b for b in branches if b.is_parent), branches[0])
            n_stations = parent_branch.centerline.stations.shape[0]
            axis_positions = np.empty((n_stations, 3), dtype=float)
            offset = _local_offset(gw_cfg)
            for i, s in enumerate(parent_branch.centerline.stations):
                f = parent_branch.centerline.frame(float(s))
                axis_positions[i] = f.position + offset[0] * f.normal + offset[1] * f.binormal
            guidewire = GuidewireArtifacts(
                config=gw_cfg,
                mesh=mesh,
                material_name=gw_entry["material"],
                in_plane_position_mm=offset,
                axis_positions_world=axis_positions,
            )

    world_bg = manifest.get("world", {}).get("background_material", "lumen")

    if not surfaces:
        raise RuntimeError(
            f"vessel directory {in_dir} has no surface meshes (no manifest "
            "surfaces and no top-level lumen.obj)"
        )

    return Vessel(
        config=cfg,
        branches=branches,
        lumen_mesh=surfaces[0].mesh,
        parent_branch_id=0,
        daughter_artifacts=[],
        surfaces=surfaces,
        lesions=lesions,
        guidewire=guidewire,
        world_background_material=world_bg,
    )


def _wall_config_from_manifest(w: dict):
    kind = w.get("kind", "single")
    if kind == "layered":
        return LayeredWallConfig(
            layers=[
                LayerSpec(
                    material_name=ly["material_name"],
                    thickness_frac=ly["thickness_frac"],
                    max_perturbation_frac=ly.get("max_perturbation_frac", 0.4),
                )
                for ly in w["layers"]
            ],
            total_thickness_mm=w["total_thickness_mm"],
            min_thickness_mm=w.get("min_thickness_mm", 0.15),
        )
    return WallConfig(
        mean_thickness_mm=w["mean_thickness_mm"],
        perturbation_modes=tuple(w["perturbation_modes"]),
        max_perturbation_frac=w["max_perturbation_frac"],
        perturbation_decay=w["perturbation_decay"],
        phase_drift_per_mm=w["phase_drift_per_mm"],
        min_thickness_mm=w["min_thickness_mm"],
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
        wall=_wall_config_from_manifest(w),
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

    lesions = [
        CalcificationLesionConfig(
            arclength_frac=entry["config"]["arclength_frac"],
            azimuth_deg=entry["config"]["azimuth_deg"],
            kind=entry.get("kind", "hard"),
            arc_extent_deg=entry["config"].get("arc_extent_deg", 50.0),
            axial_extent_mm=entry["config"].get("axial_extent_mm", 6.0),
            inner_offset_frac=entry["config"].get("inner_offset_frac", 0.0),
            outer_offset_frac=entry["config"].get("outer_offset_frac", 0.75),
            seed=entry["config"].get("seed"),
        )
        for entry in m.get("lesions", [])
    ]

    guidewire = None
    gw_entry = m.get("guidewire")
    if gw_entry is not None:
        gw_cfg = gw_entry.get("config", {})
        guidewire = GuidewireConfig(
            diameter_mm=gw_cfg.get("diameter_mm", 0.36),
            lateral_offset_mm=gw_cfg.get("lateral_offset_mm", 0.0),
            offset_azimuth_deg=gw_cfg.get("offset_azimuth_deg", 0.0),
            material_name=gw_cfg.get("material_name", "tungsten"),
        )

    return VesselConfig(
        parent=_branch_config_from_manifest(m["parent"]),
        side_branches=side_branches,
        seed=int(m["seed"]),
        name=m["name"],
        lesions=lesions,
        guidewire=guidewire,
    )
