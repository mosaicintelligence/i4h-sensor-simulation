"""Catheter sampling avoids the guidewire.

A physical IVUS catheter and guidewire share the same lumen but cannot
physically overlap. :func:`vesselgen.sampling.sample_pose` therefore
applies an in-plane clearance check whenever the vessel manifest
declares a guidewire. This test exercises that interaction end-to-end:
build a vessel with a wire offset to one side, sample many poses, and
confirm every returned probe-centre clears the wire by the requested
margin.
"""

from __future__ import annotations

import numpy as np

from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    CrossSectionConfig,
    GuidewireConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.guidewire import guidewire_clearance_mm
from vesselgen.sampling import sample_pose
from vesselgen.vessel import Vessel


def _vessel_with_offset_guidewire() -> Vessel:
    cfg = VesselConfig(
        parent=BranchConfig(
            centerline=CenterlineConfig(length_mm=40.0, n_stations=24),
            cross_section=CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.3),
            wall=WallConfig(mean_thickness_mm=0.8, max_perturbation_frac=0.4),
            name="parent",
            seed=0,
        ),
        side_branches=[],
        seed=0,
        guidewire=GuidewireConfig(
            diameter_mm=0.46,
            lateral_offset_mm=2.5,
            offset_azimuth_deg=70.0,
        ),
    )
    return Vessel.from_config(cfg)


def test_sample_pose_never_lands_inside_guidewire():
    vessel = _vessel_with_offset_guidewire()
    cfg = vessel.config.guidewire
    rng = np.random.default_rng(11)
    n = 60
    for _ in range(n):
        pose = sample_pose(vessel, rng, edge_margin_mm=0.05, guidewire_clearance_mm=0.15)
        parent = vessel.parent_branch
        frame = parent.centerline.frame(pose.arclength_mm)
        rel = pose.position - frame.position
        local_xy = np.array([float(rel @ frame.normal), float(rel @ frame.binormal)])
        assert guidewire_clearance_mm(local_xy, cfg) >= 0.1, (
            f"sampled pose at {local_xy} is too close to guidewire centre "
            f"{cfg.lateral_offset_mm} mm at azimuth {cfg.offset_azimuth_deg}"
        )
