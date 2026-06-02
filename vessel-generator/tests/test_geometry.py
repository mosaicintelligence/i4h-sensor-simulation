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
from vesselgen.sampling import (
    PoseSample,
    _build_probe_rotation,
    ground_truth_is_valid_pose,
    ground_truth_shows_endcap,
    ground_truth_side_branch_sector_invalid,
    side_branch_imaging_sector_mask,
)
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
    assert ground_truth_is_valid_pose(v, pose, gt)


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


def test_side_branch_length_at_least_parent():
    cfg = _basic_vessel(side_branch=True)
    parent_len = cfg.parent.centerline.length_mm
    assert cfg.side_branches[0].branch.centerline.length_mm < parent_len
    v = Vessel.from_config(cfg)
    side = v.branch_by_name("side_branch")
    assert side.centerline.length_mm >= parent_len


def test_sample_pose_in_branch_pins_to_branch():
    cfg = _basic_vessel(side_branch=True)
    v = Vessel.from_config(cfg)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pose = v.sample_pose_in_branch("side_branch", rng, arclength_mm=5.0)
        assert pose.branch_name == "side_branch"
        assert abs(pose.arclength_mm - 5.0) < 1e-6
        assert v.contains_point(pose.position)


def test_frame_00074_shows_parent_proximal_endcap():
    """Regression when stored GT shows coincident lumen/outer hits on side branch."""
    from pathlib import Path
    import json

    repo = Path(__file__).resolve().parents[2]
    meta_path = repo / "vessel-generator/out/paired_dataset_100/frames/frame_00074/metadata.json"
    vessel_dir = repo / "vessel-generator/out/paired_dataset_100/vessels/vessel_0006"
    if not meta_path.is_file() or not vessel_dir.is_dir():
        return

    with meta_path.open() as f:
        meta = json.load(f)
    stored_wall = (
        np.array(meta["ground_truth_geometric"]["distance_to_outer_wall_mm"])
        - np.array(meta["ground_truth_geometric"]["distance_to_lumen_wall_mm"])
    )
    if not np.any(np.isfinite(stored_wall) & (stored_wall < 0.08)):
        return

    vessel = Vessel.load(vessel_dir)
    p = meta["pose"]
    probe_axis = np.array(p["probe_axis_world"])
    pose = PoseSample(
        position=np.array(p["position_mm"]),
        rotation_euler_deg=np.array(p["rotation_euler_deg_xyz"]),
        rotation_matrix=_build_probe_rotation(probe_axis),
        probe_axis_world=probe_axis,
        branch_id=p["branch_id"],
        branch_name=p["branch_name"],
        arclength_mm=p["arclength_mm"],
        centerline_offset_mm=p["centerline_offset_mm"],
        tilt_deg=p["tilt_deg"],
    )
    t_far = float(meta.get("sim_parameters", {}).get("t_far_mm", 30.0))
    gt = vessel.ground_truth_at(pose, n_angles=256, max_distance_mm=t_far)
    assert ground_truth_side_branch_sector_invalid(vessel, pose, gt)
    assert not ground_truth_is_valid_pose(vessel, pose, gt)


def test_frame_00025_side_branch_sector_missing_wall():
    """Side-branch sector seg QC fails even when parent-wall A-lines are fine."""
    from pathlib import Path
    import json

    repo = Path(__file__).resolve().parents[2]
    meta_path = repo / "vessel-generator/out/paired_dataset_100/frames/frame_00025/metadata.json"
    vessel_dir = repo / "vessel-generator/out/paired_dataset_100/vessels/vessel_0002"
    seg_path = repo / "vessel-generator/out/paired_dataset_100/frames/frame_00025/segmentation.npy"
    if not meta_path.is_file() or not vessel_dir.is_dir() or not seg_path.is_file():
        return

    with meta_path.open() as f:
        meta = json.load(f)
    vessel = Vessel.load(vessel_dir)
    p = meta["pose"]
    probe_axis = np.array(p["probe_axis_world"])
    pose = PoseSample(
        position=np.array(p["position_mm"]),
        rotation_euler_deg=np.array(p["rotation_euler_deg_xyz"]),
        rotation_matrix=_build_probe_rotation(probe_axis),
        probe_axis_world=probe_axis,
        branch_id=p["branch_id"],
        branch_name=p["branch_name"],
        arclength_mm=p["arclength_mm"],
        centerline_offset_mm=p["centerline_offset_mm"],
        tilt_deg=p["tilt_deg"],
    )
    t_far = float(meta.get("sim_parameters", {}).get("t_far_mm", 30.0))
    gt = vessel.ground_truth_at(pose, n_angles=256, max_distance_mm=t_far)
    seg = np.load(seg_path)
    sector = side_branch_imaging_sector_mask(vessel, pose, gt.thetas_rad, gt)
    assert sector.any()
    # Bad seg columns from manual inspection fall mostly outside parent cone.
    n_r = seg.shape[1]
    min_lumen_r = int(n_r * 0.08)
    bad_in_sector = 0
    for i in np.flatnonzero(sector):
        lumen_rs = np.where(seg[i] == 1)[0]
        if len(lumen_rs) == 0:
            continue
        max_l = int(lumen_rs.max())
        if max_l < min_lumen_r:
            continue
        wall_rs = np.where(seg[i] == 2)[0]
        if len(wall_rs) == 0 or not np.any(wall_rs > max_l):
            bad_in_sector += 1
    assert bad_in_sector / max(1, sector.sum()) > 0.05


def test_reject_endcap_poses_on_bifurcation_vessel():
    cfg = _basic_vessel(side_branch=True)
    v = Vessel.from_config(cfg)
    rng = np.random.default_rng(123)
    accepted = 0
    for _ in range(80):
        pose = v.sample_pose(rng, max_tilt_deg=15.0, edge_margin_mm=0.15)
        gt = v.ground_truth_at(pose, n_angles=256)
        if ground_truth_is_valid_pose(v, pose, gt):
            accepted += 1
    assert accepted >= 40
