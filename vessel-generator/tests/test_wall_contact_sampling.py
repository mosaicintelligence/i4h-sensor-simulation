"""``wall_contact_probability`` puts the catheter against the lumen wall.

Real IVUS pullbacks frequently show frames where the probe rests against
one side of the vessel, producing a bright contact rim and a tangential
shadow. This test forces ``wall_contact_probability=1.0`` in
:func:`vesselgen.sampling.sample_pose` and checks that the returned
positions cluster near the lumen wall (eccentricity ~= lumen radius)
rather than being uniformly distributed inside the lumen.
"""

from __future__ import annotations

import numpy as np

from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    CrossSectionConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.sampling import sample_pose
from vesselgen.vessel import Vessel


def _vessel(seed: int = 0) -> Vessel:
    cfg = VesselConfig(
        parent=BranchConfig(
            centerline=CenterlineConfig(length_mm=40.0, n_stations=24),
            cross_section=CrossSectionConfig(mean_radius_mm=4.5, distal_radius_mm=4.2),
            wall=WallConfig(mean_thickness_mm=0.8, max_perturbation_frac=0.4),
            name="parent",
            seed=seed,
        ),
        side_branches=[],
        seed=seed,
    )
    return Vessel.from_config(cfg)


def test_wall_contact_eccentricity_is_high():
    vessel = _vessel(seed=0)
    rng = np.random.default_rng(0)
    parent_radius = float(np.mean(vessel.parent_branch.lumen_field.mean_radius))

    contact = []
    central = []
    for _ in range(50):
        pose = sample_pose(
            vessel,
            rng,
            edge_margin_mm=0.0,
            wall_contact_probability=1.0,
            wall_contact_margin_mm=0.02,
        )
        contact.append(pose.centerline_offset_mm)
    rng2 = np.random.default_rng(0)
    for _ in range(50):
        pose = sample_pose(vessel, rng2, edge_margin_mm=0.3, wall_contact_probability=0.0)
        central.append(pose.centerline_offset_mm)

    contact = np.asarray(contact)
    central = np.asarray(central)
    assert contact.mean() > central.mean() * 1.5, (
        f"wall-contact poses should be much more eccentric than central "
        f"poses (contact_mean={contact.mean():.2f}, central_mean={central.mean():.2f})"
    )
    near_wall_frac = (contact > 0.7 * parent_radius).mean()
    assert near_wall_frac >= 0.8, (
        f"only {near_wall_frac:.2f} of forced wall-contact poses landed "
        f"near the wall (parent_radius={parent_radius:.2f} mm)"
    )


def test_wall_contact_probability_zero_keeps_center_bias():
    vessel = _vessel(seed=1)
    rng = np.random.default_rng(1)
    parent_radius = float(np.mean(vessel.parent_branch.lumen_field.mean_radius))
    offsets = []
    for _ in range(30):
        pose = sample_pose(vessel, rng, edge_margin_mm=0.3, wall_contact_probability=0.0)
        offsets.append(pose.centerline_offset_mm)
    # Uniform-on-disk sampling has mean radius 2*R/3 ~ 0.67 R; the
    # rejection sample we use is similar in shape, so without the
    # wall-contact bias the eccentricity should sit well below the
    # wall (and well below the 80% wall threshold used above).
    assert np.mean(offsets) < 0.8 * parent_radius
    near_wall_frac = (np.asarray(offsets) > 0.7 * parent_radius).mean()
    assert near_wall_frac < 0.6
