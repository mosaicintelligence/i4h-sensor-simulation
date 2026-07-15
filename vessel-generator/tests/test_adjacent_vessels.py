"""Tests for adjacent parallel vessels.

Covers the config sampling model (adjacency rate, dedicated adjacent primary
dimensions, mutual exclusivity with side branches), geometry validity + non-overlap
under perturbations (measured outer radius + mesh-level disjointness),
ground-truth polygon count, tight neighbor clustering, and the
ring-down/FOV septum edge-case metric where the ring-down artifact obscures
the boundary between adjacent vessels.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesselgen.adjacent import adjacent_min_separation_deg, measured_outer_radius_mm
from vesselgen.config import (
    AdjacentVesselConfig,
    BranchConfig,
    CenterlineConfig,
    CrossSectionConfig,
    GenerationConfig,
    LayeredWallConfig,
    SideBranchConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.sampling import (
    _imaging_ray_directions_world,
    _min_distance_to_polygon_edge,
    _point_in_polygon,
    _project_to_imaging_plane,
)
from vesselgen.vessel import Vessel


def _adjacent_config(
    seed: int,
    *,
    n_neighbors: int = 2,
    gap_range: tuple[float, float] = (0.5, 1.5),
) -> VesselConfig:
    """Draw a VesselConfig that is guaranteed to be an adjacency case.

    Lesions and guidewires are disabled: they are orthogonal to the
    adjacency geometry under test and dominate the build time.
    """
    cfg = GenerationConfig(
        adjacent_vessel_probability=1.0,
        aortic_scale_probability=0.0,
        adjacent_vessel_count_range=(n_neighbors, n_neighbors),
        adjacent_vessel_gap_mm_range=gap_range,
        calcification_probability=0.0,
        guidewire_probability=0.0,
    )
    rng = np.random.default_rng(seed)
    return cfg.sample(rng, seed=seed, name=f"adj_{seed}")


# ---------------------------------------------------------------------------
# Config sampling model
# ---------------------------------------------------------------------------


def test_adjacency_rate_matches_probability():
    cfg = GenerationConfig()  # defaults: adjacent 0.10, aortic 0.18
    rng = np.random.default_rng(0)
    n = 600
    n_adjacent = 0
    n_aortic = 0
    for i in range(n):
        v = cfg.sample(rng, seed=i)
        if v.adjacent_vessels:
            n_adjacent += 1
        if v.parent.cross_section.mean_radius_mm >= 8.0:
            n_aortic += 1
    adj_rate = n_adjacent / n
    aortic_rate = n_aortic / n
    # Adjacency is an exact whole-dataset marginal (its own scale bucket).
    assert abs(adj_rate - cfg.adjacent_vessel_probability) < 0.06
    # Adjacency is carved from the typical share, so aortic stays put.
    assert abs(aortic_rate - cfg.aortic_scale_probability) < 0.06


def test_adjacent_case_uses_dedicated_primary_range():
    cfg = GenerationConfig()
    rng = np.random.default_rng(1)
    lo, hi = cfg.adjacent_parent_radius_mm_range
    seen = 0
    for i in range(400):
        v = cfg.sample(rng, seed=i)
        if not v.adjacent_vessels:
            continue
        seen += 1
        assert lo <= v.parent.cross_section.mean_radius_mm <= hi
        assert len(v.side_branches) == 0
        assert 1 <= len(v.adjacent_vessels) <= 2
    assert seen > 20, "expected a healthy number of adjacency draws"


def test_scale_partition_probability_validation():
    with pytest.raises(ValueError):
        GenerationConfig(adjacent_vessel_probability=0.9, aortic_scale_probability=0.2)
    with pytest.raises(ValueError):
        GenerationConfig(adjacent_vessel_probability=1.5)


def test_side_branches_and_adjacent_vessels_are_mutually_exclusive():
    parent = BranchConfig(
        centerline=CenterlineConfig(length_mm=30.0, n_stations=32, origin=(0.0, -15.0, 0.0)),
        cross_section=CrossSectionConfig(mean_radius_mm=2.5),
        wall=WallConfig(mean_thickness_mm=0.6),
        name="parent",
    )
    neighbor = BranchConfig(
        centerline=CenterlineConfig(length_mm=30.0, n_stations=32, origin=(0.0, -15.0, 0.0)),
        cross_section=CrossSectionConfig(mean_radius_mm=2.5),
        wall=WallConfig(mean_thickness_mm=0.6),
        name="adjacent_0",
    )
    side = SideBranchConfig(branch=BranchConfig(name="side"))
    with pytest.raises(ValueError):
        VesselConfig(
            parent=parent,
            side_branches=[side],
            adjacent_vessels=[AdjacentVesselConfig(branch=neighbor)],
            seed=0,
        )


def test_force_side_branch_never_produces_adjacency():
    cfg = GenerationConfig()
    rng = np.random.default_rng(2)
    for i in range(200):
        v = cfg.sample(rng, seed=i, force_side_branch=True)
        assert not v.adjacent_vessels
        assert len(v.side_branches) >= 1


# ---------------------------------------------------------------------------
# Geometry validity + non-overlap (the collision-check test)
# ---------------------------------------------------------------------------


def _analytic_min_clearance_mm(cfg: VesselConfig) -> float:
    """Smallest wall-to-wall clearance over parent<->neighbor and
    neighbor<->neighbor pairs using the measured maximum outer radius."""
    parent_bound = measured_outer_radius_mm(cfg.parent)
    centers = []
    bounds = []
    for adj in cfg.adjacent_vessels:
        az = np.radians(adj.azimuth_deg)
        centers.append(
            np.array([adj.center_offset_mm * np.cos(az), adj.center_offset_mm * np.sin(az)])
        )
        bounds.append(measured_outer_radius_mm(adj.branch))

    worst = np.inf
    for c, b in zip(centers, bounds):
        # parent centered at origin
        worst = min(worst, float(np.linalg.norm(c)) - parent_bound - b)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            worst = min(worst, d - bounds[i] - bounds[j])
    return worst


@pytest.mark.parametrize("n_neighbors", [1, 2])
def test_neighbors_do_not_overlap_seed_sweep(n_neighbors: int):
    for seed in range(25):
        cfg = _adjacent_config(seed, n_neighbors=n_neighbors, gap_range=(0.3, 2.0))
        assert len(cfg.adjacent_vessels) == n_neighbors

        # 1) Analytic: every pair is clear by at least the configured gap floor.
        assert _analytic_min_clearance_mm(cfg) > 0.0

        # 2) Mesh-level: parent + neighbor shells are genuinely disjoint, which
        #    validates the analytic bound against the actual perturbed meshes.
        vessel = Vessel.from_config(cfg)
        assert vessel.lumen_mesh.is_watertight
        assert vessel.outer_mesh.is_watertight
        n_lumen_bodies = len(vessel.lumen_mesh.split(only_watertight=False))
        n_outer_bodies = len(vessel.outer_mesh.split(only_watertight=False))
        assert n_lumen_bodies == 1 + n_neighbors
        assert n_outer_bodies == 1 + n_neighbors
        assert len(vessel.adjacent_vessels) == n_neighbors


@pytest.mark.parametrize(
    "wall",
    [
        WallConfig(mean_thickness_mm=0.5, max_perturbation_frac=0.4),
        LayeredWallConfig.trilaminar(total_thickness_mm=0.6),
    ],
)
def test_measured_outer_radius_matches_swept_mesh(wall):
    """``measured_outer_radius_mm`` (used to seat neighbors) must equal the
    outer radius of the surface actually swept from the same fields.

    Placement measurement and mesh building both flow through
    ``vesselgen.wall.build_branch_fields``; this test asserts the measured
    outer radius matches the outermost swept shell for both a single-slab
    and a trilaminar wall across seeds.
    """
    from vesselgen.centerline import Centerline
    from vesselgen.sweep import sweep_layered_branch
    from vesselgen.wall import build_branch_fields

    for seed in range(6):
        branch = BranchConfig(
            centerline=CenterlineConfig(length_mm=30.0, n_stations=48, origin=(0.0, -15.0, 0.0)),
            cross_section=CrossSectionConfig(mean_radius_mm=2.2, max_perturbation_frac=0.12),
            wall=wall,
            name="branch",
            seed=seed,
        )
        measured = measured_outer_radius_mm(branch)

        lumen_field, layered_wall_field = build_branch_fields(branch, np.random.default_rng(seed))
        meshes = sweep_layered_branch(
            Centerline(branch.centerline), lumen_field, layered_wall_field
        )
        # Outermost swept shell = adventitia; the branch axis runs along +Y.
        outer_shell = meshes[-1]
        built = float(np.sqrt((outer_shell.vertices[:, [0, 2]] ** 2).sum(axis=1)).max())

        assert built == pytest.approx(measured, abs=1e-6)


def test_two_neighbors_pack_into_a_tight_cluster():
    separations = []
    for seed in range(120):
        cfg = _adjacent_config(seed, n_neighbors=2, gap_range=(0.5, 2.0))
        a, b = cfg.adjacent_vessels
        # wrapped angular separation in [0, 180]
        delta = abs(((a.azimuth_deg - b.azimuth_deg) + 180.0) % 360.0 - 180.0)
        # The second neighbor is packed right next to the first at exactly
        # the minimum separation that clears it (edge-to-edge = their gap).
        min_center = (
            measured_outer_radius_mm(a.branch)
            + measured_outer_radius_mm(b.branch)
            + 0.0  # gap already folded into center_offset; use >=0 lower bound here
        )
        delta_min = adjacent_min_separation_deg(a.center_offset_mm, b.center_offset_mm, min_center)
        assert delta >= delta_min - 1e-6
        separations.append(delta)
    separations = np.asarray(separations)
    # Clustered, not diametrically opposed: two similarly-sized neighbors
    # touching-but-clear sit well under 180 deg apart on the same side of
    # the parent, so the narrow tissue gaps stay in one small sector.
    assert separations.max() < 130.0


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_neighbors", [1, 2])
def test_ground_truth_polygon_count_is_one_plus_neighbors(n_neighbors: int):
    cfg = _adjacent_config(3, n_neighbors=n_neighbors)
    vessel = Vessel.from_config(cfg)
    # Place the probe at the parent center at mid-length with the probe axis
    # along the vessel (Y), so the imaging plane crosses all parallel tubes.
    parent = vessel.parent_branch
    center = parent.centerline.position(parent.centerline.length_mm / 2.0)
    pose = vessel.pose_at(center, probe_axis_world=np.array([0.0, 1.0, 0.0]))
    gt = vessel.ground_truth_at(pose, n_angles=360, max_distance_mm=40.0)
    assert len(gt.lumen_contour_polygons) == 1 + n_neighbors
    # Exactly one polygon (the parent) contains the probe origin.
    contains_origin = [_point_in_polygon(np.zeros(2), poly) for poly in gt.lumen_contour_polygons]
    assert sum(contains_origin) == 1


# ---------------------------------------------------------------------------
# Ring-down / FOV septum edge-case metric
# ---------------------------------------------------------------------------


def _septum_start_distance_mm(vessel: Vessel, pose, gt) -> float:
    """Distance from the probe to the parent outer wall along the ray aimed
    at the (first) neighbor -- i.e. where the vessel-vessel septum begins.

    When this falls inside the ring-down band the boundary between the two
    vessels is obscured on the B-mode -- the target training signal.
    Returns NaN when the neighbor direction has no in-plane component or the
    ray misses.
    """
    art = vessel.adjacent_vessels[0]
    s_proj, _ = art.centerline.project(pose.position)
    direction = art.centerline.position(s_proj) - pose.position
    in_plane = _project_to_imaging_plane(direction, pose.probe_axis_world)
    norm = float(np.linalg.norm(in_plane))
    if norm < 1e-9:
        return float("nan")
    ray_dirs = _imaging_ray_directions_world(pose, gt.thetas_rad)
    theta_idx = int(np.argmax(ray_dirs @ (in_plane / norm)))
    return float(gt.distance_to_outer_wall_mm[theta_idx])


def _nearest_neighbor_wall_mm(gt) -> float:
    """Distance from the probe origin to the nearest neighbor outer-wall edge
    in the imaging plane (inf when no neighbor polygon is in the plane)."""
    origin = np.zeros(2)
    neighbor_polys = [p for p in gt.outer_contour_polygons if not _point_in_polygon(origin, p)]
    if not neighbor_polys:
        return float("inf")
    return min(_min_distance_to_polygon_edge(origin, p) for p in neighbor_polys)


def test_ring_down_covers_septum_for_adjacent_case():
    """Across adjacent-vessels poses, a healthy fraction put the start of the
    vessel-vessel septum inside the ring-down band (obscured boundary), while
    the neighbor itself stays inside the FOV (so it is imaged, not cropped)."""
    ring_down_outer_mm = 2.8
    fov_mm = 20.0

    n_total = 0
    n_neighbor_in_plane = 0
    n_septum_within_ring_down = 0
    n_septum_visible_within_fov = 0
    n_neighbor_beyond_fov = 0

    for seed in range(10):
        # Tight gaps push the septum toward the ring-down zone.
        cfg = _adjacent_config(seed, n_neighbors=1, gap_range=(0.2, 1.0))
        vessel = Vessel.from_config(cfg)
        rng = np.random.default_rng(1000 + seed)
        for _ in range(30):
            try:
                pose = vessel.sample_pose(rng, max_tilt_deg=6.0, edge_margin_mm=0.05)
            except RuntimeError:
                continue
            gt = vessel.ground_truth_at(pose, n_angles=64, max_distance_mm=fov_mm)
            n_total += 1
            if np.isfinite(_nearest_neighbor_wall_mm(gt)):
                n_neighbor_in_plane += 1
            else:
                n_neighbor_beyond_fov += 1
                continue
            septum = _septum_start_distance_mm(vessel, pose, gt)
            if not np.isfinite(septum):
                continue
            if septum <= ring_down_outer_mm:
                n_septum_within_ring_down += 1
            else:
                n_septum_visible_within_fov += 1

    assert n_total > 0
    # The neighbor should be imaged (inside FOV) in nearly every pose.
    assert n_neighbor_in_plane / n_total > 0.9
    # The target edge case: for a healthy fraction of poses the inter-vessel
    # boundary starts inside the ring-down band, so it is obscured on B-mode.
    assert n_septum_within_ring_down / n_neighbor_in_plane > 0.25
    # And a complementary share keeps the boundary visible beyond ring-down,
    # so the dataset teaches both "obscured" and "resolved" appearances.
    assert n_septum_visible_within_fov / n_neighbor_in_plane > 0.15


def test_neighbor_pose_bias_raises_ring_down_coverage():
    """Biasing poses toward the neighbor should put the neighbor wall inside
    the ring-down band far more often than unbiased sampling, while unbiased
    sampling stays a no-op for its own distribution."""
    ring_down_outer_mm = 2.8

    def merge_fraction(bias_prob: float) -> float:
        n = 0
        n_merge = 0
        for seed in range(8):
            cfg = _adjacent_config(seed, n_neighbors=1, gap_range=(0.02, 0.15))
            vessel = Vessel.from_config(cfg)
            rng = np.random.default_rng(4000 + seed)
            for _ in range(40):
                try:
                    pose = vessel.sample_pose(
                        rng,
                        max_tilt_deg=6.0,
                        edge_margin_mm=0.05,
                        neighbor_bias_prob=bias_prob,
                    )
                except RuntimeError:
                    continue
                gt = vessel.ground_truth_at(pose, n_angles=90, max_distance_mm=17.5)
                nearest = _nearest_neighbor_wall_mm(gt)
                if not np.isfinite(nearest):
                    continue
                n += 1
                if nearest <= ring_down_outer_mm:
                    n_merge += 1
        assert n > 0
        return n_merge / n

    unbiased = merge_fraction(0.0)
    biased = merge_fraction(1.0)
    # Directed eccentricity is the dominant lever: it should clear a large
    # majority, well above the ~30-40% geometry-only ceiling of unbiased poses.
    assert biased > 0.8
    assert biased > unbiased + 0.3


def test_neighbor_bias_picks_either_neighbor_when_two_present():
    """With two neighbors, a biased pose should target either azimuth, not
    always the first entry in ``adjacent_vessels``."""
    counts = {0: 0, 1: 0}
    for seed in range(40):
        cfg = _adjacent_config(seed, n_neighbors=2, gap_range=(0.02, 0.15))
        if len(cfg.adjacent_vessels) != 2:
            continue
        a0, a1 = cfg.adjacent_vessels
        azimuths = [a0.azimuth_deg % 360.0, a1.azimuth_deg % 360.0]
        sep = abs(((azimuths[0] - azimuths[1]) + 180.0) % 360.0 - 180.0)
        if sep < 15.0:
            continue

        vessel = Vessel.from_config(cfg)
        parent = vessel.parent_branch
        rng = np.random.default_rng(5000 + seed)
        for _ in range(30):
            try:
                pose = vessel.sample_pose(
                    rng,
                    max_tilt_deg=0.0,
                    edge_margin_mm=0.05,
                    neighbor_bias_prob=1.0,
                )
            except RuntimeError:
                continue
            frame = parent.centerline.frame(pose.arclength_mm)
            offset = pose.position - frame.position
            local = np.array([float(offset @ frame.normal), float(offset @ frame.binormal)])
            if float(np.linalg.norm(local)) < 0.05:
                continue
            angle = float(np.degrees(np.arctan2(local[1], local[0])) % 360.0)

            def ang_dist(a: float, b: float) -> float:
                return abs(((a - b) + 180.0) % 360.0 - 180.0)

            nearest = int(np.argmin([ang_dist(angle, az) for az in azimuths]))
            counts[nearest] += 1

    assert counts[0] > 0
    assert counts[1] > 0


# ---------------------------------------------------------------------------
# Per-object mesh decomposition for the renderer (merged surfaces are kept for
# geometry/GT; each vessel is *also* written separately so raysim can load it
# as a distinct nested object).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_neighbors", [1, 2])
def test_per_object_meshes_and_manifest(tmp_path, n_neighbors):
    import json

    from vesselgen.io import save_vessel

    cfg = _adjacent_config(11, n_neighbors=n_neighbors)
    vessel = Vessel.from_config(cfg)
    vdir = save_vessel(vessel, tmp_path / "adj")

    with (vdir / "vessel.json").open() as f:
        manifest = json.load(f)

    assert manifest.get("surfaces_are_merged") is True
    objects = manifest["objects"]
    assert len(objects) == 1 + n_neighbors

    parent_obj = objects[0]
    assert parent_obj["role"] == "parent"
    assert parent_obj["probe_inside"] is True
    parent_materials = [s["material"] for s in parent_obj["surfaces"]]
    assert len(parent_obj["surfaces"]) >= 1
    assert parent_obj["interior_material"] == vessel.world_background_material
    assert parent_obj["surrounding_material"] == parent_materials[-1]

    for obj in objects[1:]:
        assert obj["role"] == "adjacent"
        assert obj["probe_inside"] is False
        # Neighbors share the parent's material chain by construction.
        assert [s["material"] for s in obj["surfaces"]] == parent_materials
        assert obj["interior_material"] == vessel.world_background_material
        assert obj["surrounding_material"] == parent_materials[-1]
        # Placement metadata the renderer needs to seat the neighbor.
        assert "center_offset_mm" in obj and "azimuth_deg" in obj
        assert "centerline" in obj and "origin" in obj["centerline"]

    # Every referenced per-object mesh file exists on disk, and they live
    # under objects/ (separate from the merged top-level surfaces).
    for obj in objects:
        for s in obj["surfaces"]:
            assert s["obj"].startswith("objects/")
            assert (vdir / s["obj"]).is_file()

    # The merged top-level surfaces are still written unchanged.
    for s in manifest["surfaces"]:
        assert not s["obj"].startswith("objects/")
        assert (vdir / s["obj"]).is_file()


@pytest.mark.parametrize("n_neighbors", [1, 2])
def test_merged_surfaces_equal_concat_of_object_meshes(n_neighbors):
    """The merged top-level surfaces must be exactly the concatenation of the
    per-object meshes, so the two on-disk representations can never silently
    diverge (no geometry is lost or altered by the split)."""
    cfg = _adjacent_config(7, n_neighbors=n_neighbors)
    vessel = Vessel.from_config(cfg)

    per_object = [vessel.parent_object_surfaces] + [art.surfaces for art in vessel.adjacent_vessels]
    assert len(vessel.surfaces) == len(vessel.parent_object_surfaces)

    for layer_idx, merged in enumerate(vessel.surfaces):
        exp_v = sum(len(obj[layer_idx].mesh.vertices) for obj in per_object)
        exp_f = sum(len(obj[layer_idx].mesh.faces) for obj in per_object)
        assert len(merged.mesh.vertices) == exp_v
        assert len(merged.mesh.faces) == exp_f
        assert merged.material_name == vessel.parent_object_surfaces[layer_idx].material_name


def test_no_objects_section_without_neighbors(tmp_path):
    """A vessel with no neighbors is unchanged: no objects/ tree, no objects
    section (the top-level surfaces already are the parent alone)."""
    import json

    from vesselgen.io import save_vessel

    cfg = GenerationConfig(
        adjacent_vessel_probability=0.0,
        calcification_probability=0.0,
        guidewire_probability=0.0,
    )
    rng = np.random.default_rng(0)
    vessel = Vessel.from_config(cfg.sample(rng, seed=0))
    assert not vessel.adjacent_vessels
    assert vessel.parent_object_surfaces == []

    vdir = save_vessel(vessel, tmp_path / "plain")
    with (vdir / "vessel.json").open() as f:
        manifest = json.load(f)
    assert "objects" not in manifest
    assert "surfaces_are_merged" not in manifest
    assert not (vdir / "objects").exists()


def test_vessel_with_neighbors_roundtrips_through_load(tmp_path):
    """Saving then loading a neighbor vessel must not error, and the config's
    adjacent-vessel entries survive the round trip."""
    from vesselgen.io import save_vessel

    cfg = _adjacent_config(5, n_neighbors=2)
    vessel = Vessel.from_config(cfg)
    vdir = save_vessel(vessel, tmp_path / "adj")

    loaded = Vessel.load(vdir)
    assert loaded.outer_mesh.is_watertight
    assert len(loaded.config.adjacent_vessels) == 2


def test_neighbor_bias_is_noop_without_neighbors():
    """A vessel with no parallel neighbors must ignore ``neighbor_bias_prob``
    and still return valid poses (the knob is adjacency-only)."""
    cfg = GenerationConfig(
        adjacent_vessel_probability=0.0,
        calcification_probability=0.0,
        guidewire_probability=0.0,
    )
    rng = np.random.default_rng(0)
    vessel = Vessel.from_config(cfg.sample(rng, seed=0))
    assert not vessel.adjacent_vessels
    prng = np.random.default_rng(7)
    for _ in range(10):
        pose = vessel.sample_pose(prng, max_tilt_deg=6.0, neighbor_bias_prob=1.0)
        assert pose.branch_name
