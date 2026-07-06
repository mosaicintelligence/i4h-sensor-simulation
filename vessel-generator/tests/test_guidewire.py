"""Tests for the guidewire mesh + clearance helpers."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from vesselgen.centerline import Centerline
from vesselgen.config import (
    CenterlineConfig,
    CrossSectionConfig,
    GuidewireConfig,
    VesselConfig,
    BranchConfig,
    WallConfig,
)
from vesselgen.guidewire import (
    build_guidewire,
    guidewire_clearance_mm,
    guidewire_in_plane_position,
    pose_avoids_guidewire,
)
from vesselgen.vessel import Vessel


def _make_centerline(length_mm: float = 50.0, n_stations: int = 32) -> Centerline:
    return Centerline(CenterlineConfig(length_mm=length_mm, n_stations=n_stations))


def test_guidewire_mesh_is_watertight_and_thin():
    centerline = _make_centerline()
    cfg = GuidewireConfig(diameter_mm=0.36, lateral_offset_mm=1.2, offset_azimuth_deg=30.0)
    art = build_guidewire(cfg, centerline)
    assert art.mesh.is_watertight, "guidewire mesh should be watertight"
    assert art.mesh.volume > 0.0
    # length >= centerline length, radius ~ 0.18 mm
    extents = art.mesh.extents
    assert extents.max() >= centerline.length_mm
    radial_axes = np.sort(extents)[:2]
    # The two short axes should be ~ diameter (within a few percent)
    assert np.all(radial_axes < cfg.diameter_mm * 1.2)
    assert np.all(radial_axes > cfg.diameter_mm * 0.8)


@pytest.mark.parametrize("diameter,offset",
                         [(0.36, 0.0), (0.36, 1.5), (0.46, 2.5), (0.89, 3.0)])
def test_guidewire_mesh_extends_past_centerline(diameter, offset):
    centerline = _make_centerline()
    cfg = GuidewireConfig(
        diameter_mm=diameter, lateral_offset_mm=offset, offset_azimuth_deg=0.0
    )
    art = build_guidewire(cfg, centerline)
    proj = (art.mesh.vertices - centerline.origin) @ centerline.direction
    assert proj.min() < 0.0
    assert proj.max() > centerline.length_mm


def test_guidewire_clearance_signs():
    cfg = GuidewireConfig(
        diameter_mm=0.4, lateral_offset_mm=2.0, offset_azimuth_deg=0.0
    )
    inside = guidewire_in_plane_position(cfg)
    assert guidewire_clearance_mm(inside, cfg) < 0.0
    far = inside + np.array([5.0, 0.0])
    assert guidewire_clearance_mm(far, cfg) > 4.0


def test_pose_avoids_guidewire_rejects_inside_and_accepts_outside():
    cfg = GuidewireConfig(
        diameter_mm=0.4, lateral_offset_mm=2.0, offset_azimuth_deg=90.0
    )
    wire_xy = guidewire_in_plane_position(cfg)
    assert not pose_avoids_guidewire(wire_xy, cfg, clearance_mm=0.1)
    assert pose_avoids_guidewire(wire_xy + np.array([1.0, 0.0]), cfg, clearance_mm=0.1)


def _build_vessel_with_guidewire(diameter=0.4, offset_frac=0.4, seed=0) -> Vessel:
    parent = BranchConfig(
        centerline=CenterlineConfig(length_mm=40.0, n_stations=24),
        cross_section=CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.2),
        wall=WallConfig(mean_thickness_mm=0.8, max_perturbation_frac=0.4),
        name="parent",
        seed=seed,
    )
    guidewire = GuidewireConfig(
        diameter_mm=diameter,
        lateral_offset_mm=offset_frac * 4.5,
        offset_azimuth_deg=45.0,
    )
    cfg = VesselConfig(parent=parent, side_branches=[], seed=seed, guidewire=guidewire)
    return Vessel.from_config(cfg)


def test_sample_pose_avoids_guidewire():
    """sample_pose should never produce a position inside the guidewire."""
    from vesselgen.sampling import sample_pose

    vessel = _build_vessel_with_guidewire()
    cfg = vessel.config.guidewire
    assert cfg is not None
    rng = np.random.default_rng(7)
    for _ in range(40):
        pose = sample_pose(vessel, rng, edge_margin_mm=0.1, guidewire_clearance_mm=0.2)
        parent = vessel.parent_branch
        s = pose.arclength_mm
        frame = parent.centerline.frame(s)
        rel = pose.position - frame.position
        local_xy = np.array([float(rel @ frame.normal), float(rel @ frame.binormal)])
        assert guidewire_clearance_mm(local_xy, cfg) >= 0.1, (
            f"sampled pose at local_xy={local_xy} is inside guidewire"
        )


def test_wall_contact_sampling_returns_points_near_lumen():
    """wall_contact_probability=1.0 places the probe against the wall."""
    from vesselgen.sampling import sample_pose

    vessel = _build_vessel_with_guidewire(diameter=0.3, offset_frac=0.0)
    rng = np.random.default_rng(3)
    near_wall_hits = 0
    for _ in range(30):
        pose = sample_pose(
            vessel, rng,
            edge_margin_mm=0.0,
            wall_contact_probability=1.0,
            wall_contact_margin_mm=0.02,
        )
        # eccentricity should be close to lumen radius
        parent_radius = float(np.mean(vessel.parent_branch.lumen_field.mean_radius))
        if pose.centerline_offset_mm > 0.7 * parent_radius:
            near_wall_hits += 1
    assert near_wall_hits >= 24, (
        f"only {near_wall_hits}/30 wall-contact poses were near the wall"
    )
