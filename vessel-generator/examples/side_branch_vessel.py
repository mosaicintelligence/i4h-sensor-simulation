"""Generate a vessel with one side-branch ostium."""

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
from vesselgen.io import save_vessel
from vesselgen.vessel import Vessel
from vesselgen.visualize import preview_ground_truth, preview_vessel

OUT = Path("out/example_side_branch")


def main() -> None:
    cfg = VesselConfig(
        parent=BranchConfig(
            centerline=CenterlineConfig(
                length_mm=32.0,
                origin=(0.0, -16.0, 0.0),
                direction=(0.0, 1.0, 0.0),
                n_stations=80,
            ),
            cross_section=CrossSectionConfig(
                mean_radius_mm=3.2, distal_radius_mm=2.8, max_perturbation_frac=0.18
            ),
            wall=WallConfig(mean_thickness_mm=0.75, max_perturbation_frac=0.55),
            name="parent",
        ),
        side_branches=[
            SideBranchConfig(
                parent_arclength_frac=0.5,
                azimuth_deg=0.0,
                polar_deg=55.0,
                branch=BranchConfig(
                    centerline=CenterlineConfig(length_mm=15.0, n_stations=40),
                    cross_section=CrossSectionConfig(mean_radius_mm=1.5, distal_radius_mm=1.3),
                    wall=WallConfig(mean_thickness_mm=0.6),
                    name="side_branch",
                ),
            )
        ],
        seed=7,
        name="example_side_branch",
    )
    vessel = Vessel.from_config(cfg)
    save_vessel(vessel, OUT)

    rng = np.random.default_rng(0)
    poses = []
    side = vessel.branch_by_name("side_branch")
    for _ in range(12):
        poses.append(vessel.sample_pose(rng, max_tilt_deg=12.0))
    preview_vessel(vessel, OUT / "preview.png", poses=poses)

    for i, pose in enumerate(poses[:3]):
        gt = vessel.ground_truth_at(pose, n_angles=360)
        preview_ground_truth(pose, gt, OUT / f"sample_{i:02d}.png")

    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
