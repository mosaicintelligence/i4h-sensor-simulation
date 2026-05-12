"""Generate one straight peripheral-vessel segment with no bifurcation."""

from pathlib import Path

import numpy as np

from vesselgen.config import (
    BranchConfig,
    CenterlineConfig,
    CrossSectionConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.io import save_vessel
from vesselgen.vessel import Vessel
from vesselgen.visualize import preview_ground_truth, preview_vessel

OUT = Path("out/example_straight")


def main() -> None:
    cfg = VesselConfig(
        parent=BranchConfig(
            centerline=CenterlineConfig(
                length_mm=30.0, origin=(0.0, -15.0, 0.0), direction=(0.0, 1.0, 0.0),
                n_stations=64,
            ),
            cross_section=CrossSectionConfig(
                mean_radius_mm=3.0, distal_radius_mm=2.5, max_perturbation_frac=0.18
            ),
            wall=WallConfig(mean_thickness_mm=0.7, max_perturbation_frac=0.55),
            name="parent",
        ),
        seed=1,
        name="example_straight",
    )
    vessel = Vessel.from_config(cfg)
    save_vessel(vessel, OUT)

    rng = np.random.default_rng(0)
    poses = [vessel.sample_pose(rng, max_tilt_deg=12.0) for _ in range(8)]
    preview_vessel(vessel, OUT / "preview.png", poses=poses)
    for i, pose in enumerate(poses[:3]):
        gt = vessel.ground_truth_at(pose, n_angles=360)
        preview_ground_truth(pose, gt, OUT / f"sample_{i:02d}.png")
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
