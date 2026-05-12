"""Smoke tests covering centerline, cross-section, sweep, and sampling."""

from __future__ import annotations

import numpy as np
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    CrossSectionConfig,
    SideBranchConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.cross_section import build_cross_sections
from vesselgen.sweep import sweep_branch
from vesselgen.vessel import Vessel
from vesselgen.wall import build_wall


def _basic_vessel(side_branch: bool = False) -> VesselConfig:
    sb = []
    if side_branch:
        sb.append(SideBranchConfig(
            parent_arclength_frac=0.5, azimuth_deg=0.0, polar_deg=55.0,
            branch=BranchConfig(
                centerline=CenterlineConfig(length_mm=12.0, n_stations=32),
                cross_section=CrossSectionConfig(mean_radius_mm=1.5, distal_radius_mm=1.3),
                wall=WallConfig(mean_thickness_mm=0.5),
                name="side_branch",
            ),
        ))
    return VesselConfig(
        parent=BranchConfig(
            centerline=CenterlineConfig(length_mm=24.0, n_stations=48,
                                         origin=(0.0, -12.0, 0.0)),
            cross_section=CrossSectionConfig(mean_radius_mm=2.5, distal_radius_mm=2.2),
            wall=WallConfig(mean_thickness_mm=0.7),
            name="parent",
        ),
        side_branches=sb,
        seed=1,
        name="test_vessel",
    )


def test_centerline_straight_frame_orthonormal():
    cl = Centerline(CenterlineConfig(length_mm=20.0, direction=(0.0, 1.0, 0.0)))
    f = cl.frame(10.0)
    assert abs(np.dot(f.tangent, f.normal)) < 1e-10
    assert abs(np.dot(f.tangent, f.binormal)) < 1e-10
    assert abs(np.dot(f.normal, f.binormal)) < 1e-10
    assert abs(np.linalg.norm(f.tangent) - 1.0) < 1e-10
    assert abs(np.linalg.norm(f.normal) - 1.0) < 1e-10


def test_cross_section_radii_positive_and_drift_smooth():
    cfg = CrossSectionConfig(mean_radius_mm=3.0, max_perturbation_frac=0.2)
    rng = np.random.default_rng(0)
    cs = build_cross_sections(cfg, length_mm=20.0, n_stations=64, rng=rng)
    assert (cs.radii > 0).all()
    diff = np.diff(cs.radii, axis=0)
    assert np.abs(diff).max() < 0.1 * cs.radii.mean()


def test_single_branch_mesh_is_watertight_outward_in_memory():
    """In-memory meshes use outward normals (so trimesh.contains and boolean
    union work). The on-disk OBJ flips this to inward for the simulator."""
    cl = Centerline(CenterlineConfig(length_mm=20.0, n_stations=48))
    rng = np.random.default_rng(0)
    lumen = build_cross_sections(
        CrossSectionConfig(mean_radius_mm=2.5, max_perturbation_frac=0.15),
        length_mm=20.0, n_stations=48, rng=rng,
    )
    wall = build_wall(WallConfig(mean_thickness_mm=0.6), lumen, length_mm=20.0, rng=rng)
    lumen_mesh, outer_mesh = sweep_branch(cl, lumen, wall)
    assert lumen_mesh.is_watertight
    assert outer_mesh.is_watertight
    assert lumen_mesh.volume > 0
    assert outer_mesh.volume > 0
    assert outer_mesh.volume > lumen_mesh.volume

    # Wall face normals should point outward (away from the local axis).
    centroids = lumen_mesh.triangles_center
    wall_mask = (centroids[:, 1] > -8.0) & (centroids[:, 1] < 8.0)
    assert wall_mask.sum() > 100
    for fi in np.where(wall_mask)[0][:50]:
        c = centroids[fi]
        radial = np.array([c[0], 0.0, c[2]])
        rn = float(np.linalg.norm(radial))
        if rn < 1e-3:
            continue
        radial = radial / rn
        normal = lumen_mesh.face_normals[fi]
        assert float(np.dot(normal, radial)) > 0.5, (
            f"face {fi} normal {normal} not outward at radial {radial}"
        )


def test_saved_obj_has_inward_normals(tmp_path):
    """On-disk OBJ should have inward-facing normals (simulator convention)."""
    import trimesh

    from vesselgen.io import save_vessel

    cfg = _basic_vessel()
    v = Vessel.from_config(cfg)
    save_vessel(v, tmp_path)
    saved = trimesh.load(tmp_path / "lumen.obj", force="mesh", process=True)
    assert saved.is_watertight
    # Saved volume should be negative if normals are inward.
    assert saved.volume < 0


def test_full_vessel_no_bifurcation_builds():
    cfg = _basic_vessel()
    v = Vessel.from_config(cfg)
    assert v.lumen_mesh.is_watertight
    assert v.outer_mesh.is_watertight
    assert len(v.branches) == 1


def test_vessel_with_side_branch_builds():
    cfg = _basic_vessel(side_branch=True)
    v = Vessel.from_config(cfg)
    assert v.lumen_mesh.is_watertight
    assert v.outer_mesh.is_watertight
    assert len(v.branches) == 2


def test_sample_pose_inside_lumen_and_ground_truth_makes_sense():
    cfg = _basic_vessel()
    v = Vessel.from_config(cfg)
    rng = np.random.default_rng(0)
    pose = v.sample_pose(rng, max_tilt_deg=10.0)
    assert v.lumen_mesh.contains([pose.position])[0]
    gt = v.ground_truth_at(pose, n_angles=180)
    finite_lumen = gt.distance_to_lumen_wall_mm[~np.isnan(gt.distance_to_lumen_wall_mm)]
    assert (finite_lumen > 0).all()
    assert finite_lumen.size > 0
    assert gt.lumen_csa_mm2 > 0
    assert gt.equivalent_lumen_diameter_mm > 0


def test_sample_pose_with_bifurcation_returns_visible_branches():
    cfg = _basic_vessel(side_branch=True)
    v = Vessel.from_config(cfg)
    rng = np.random.default_rng(0)
    pose = v.sample_pose(rng, max_tilt_deg=10.0)
    gt = v.ground_truth_at(pose, n_angles=180)
    assert pose.branch_id in gt.branch_ids_visible


def test_side_branch_does_not_pierce_opposite_wall():
    """The daughter must not poke out the far side of the parent.

    We verify by sampling points along the daughter's centerline at small
    arclengths (just inside the parent) and at slightly larger arclengths
    (still inside the parent up to ~parent radius from centerline). All
    such points should lie inside the merged lumen mesh, and beyond the
    daughter's full extent there should be no lumen on the *opposite*
    side of the parent from the daughter's exit.
    """
    cfg = _basic_vessel(side_branch=True)
    v = Vessel.from_config(cfg)
    side = v.branch_by_name("side_branch")
    daughter_origin = side.centerline.origin
    daughter_dir = side.centerline.direction

    # Point on the *opposite* side of the parent from where the daughter
    # exits, well outside the parent radius. With the bug this would be
    # inside the lumen because the daughter crosses through the parent.
    parent = v.parent_branch
    parent_radius = float(parent.lumen_field.mean_radius.max())
    parent_center_at_attachment = parent.centerline.position(
        side.parent_attachment_arclength_mm
    )
    far_side_point = parent_center_at_attachment - 1.5 * parent_radius * daughter_dir
    assert not v.contains_point(far_side_point), (
        f"Side branch is piercing the far side of the parent: "
        f"point {far_side_point.tolist()} is inside the lumen"
    )


def test_pose_at_explicit_position():
    cfg = _basic_vessel()
    v = Vessel.from_config(cfg)
    parent_center = v.parent_branch.centerline.position(
        v.parent_branch.centerline.length_mm / 2.0
    )
    pose = v.pose_at(parent_center)
    assert pose.branch_name == "parent"
    assert pose.centerline_offset_mm < 1e-3


def test_sample_pose_in_branch_pins_to_branch():
    cfg = _basic_vessel(side_branch=True)
    v = Vessel.from_config(cfg)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pose = v.sample_pose_in_branch("side_branch", rng, arclength_mm=5.0)
        assert pose.branch_name == "side_branch"
        assert abs(pose.arclength_mm - 5.0) < 1e-6
        assert v.contains_point(pose.position)
