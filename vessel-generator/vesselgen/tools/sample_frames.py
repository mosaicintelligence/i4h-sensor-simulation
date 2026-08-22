"""CLI: sample N (pose, ground truth) tuples from a saved vessel.

Each sample is written as a small JSON record so the simulator-driver code
can read them and feed each pose into ``rs.IVUSProbe``. Optionally writes
a per-sample preview PNG (the imaging-plane contour overlay).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vesselgen.vessel import Vessel
from vesselgen.visualize import preview_ground_truth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vessel", type=Path, required=True, help="Path to a saved vessel folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tilt-deg", type=float, default=15.0)
    p.add_argument("--edge-margin-mm", type=float, default=0.2)
    p.add_argument("--gt-angles", type=int, default=360)
    p.add_argument("--max-distance-mm", type=float, default=30.0)
    p.add_argument(
        "--write-previews", action="store_true", help="Write a per-sample PNG (slow if N is large)"
    )
    p.add_argument(
        "--fov-mm",
        type=float,
        default=None,
        help="Draw the imaging FOV as a dashed circle in previews (mm).",
    )
    p.add_argument(
        "--ring-down-mm",
        type=float,
        default=None,
        help=(
            "Outer radius (mm) of the catheter ring-down annulus to overlay on "
            "previews for small-vessel QA (e.g. 2.8). Omit to disable the overlay."
        ),
    )
    p.add_argument(
        "--ring-down-inner-mm",
        type=float,
        default=1.0,
        help="Inner radius (mm) of the ring-down dead zone (default 1.0). Used with --ring-down-mm.",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    vessel = Vessel.load(args.vessel)
    rng = np.random.default_rng(args.seed)

    summary = []
    for i in range(args.n):
        pose = vessel.sample_pose(
            rng,
            max_tilt_deg=args.max_tilt_deg,
            edge_margin_mm=args.edge_margin_mm,
        )
        gt = vessel.ground_truth_at(
            pose, n_angles=args.gt_angles, max_distance_mm=args.max_distance_mm
        )

        record = {
            "index": i,
            "pose": {
                "position_mm": pose.position.tolist(),
                "rotation_euler_deg_xyz": pose.rotation_euler_deg.tolist(),
                "probe_axis_world": pose.probe_axis_world.tolist(),
                "branch_id": pose.branch_id,
                "branch_name": pose.branch_name,
                "arclength_mm": pose.arclength_mm,
                "centerline_offset_mm": pose.centerline_offset_mm,
                "tilt_deg": pose.tilt_deg,
            },
            "ground_truth": {
                "n_angles": gt.n_angles,
                "thetas_rad": gt.thetas_rad.tolist(),
                "distance_to_lumen_wall_mm": gt.distance_to_lumen_wall_mm.tolist(),
                "distance_to_outer_wall_mm": gt.distance_to_outer_wall_mm.tolist(),
                "wall_thickness_mm": gt.wall_thickness_mm.tolist(),
                "lumen_csa_mm2": gt.lumen_csa_mm2,
                "outer_csa_mm2": gt.outer_csa_mm2,
                "equivalent_lumen_diameter_mm": gt.equivalent_lumen_diameter_mm,
                "branch_ids_visible": gt.branch_ids_visible,
                "n_lumen_polygons": len(gt.lumen_contour_polygons),
                "n_outer_polygons": len(gt.outer_contour_polygons),
            },
        }
        with (args.out / f"frame_{i:05d}.json").open("w") as f:
            json.dump(record, f, indent=2)
        summary.append(record)

        if args.write_previews:
            preview_ground_truth(
                pose,
                gt,
                args.out / f"frame_{i:05d}.png",
                fov_radius_mm=args.fov_mm,
                ring_down_outer_mm=args.ring_down_mm,
                ring_down_inner_mm=args.ring_down_inner_mm,
            )

    with (args.out / "summary.json").open("w") as f:
        json.dump({"n_samples": len(summary), "vessel": str(args.vessel)}, f, indent=2)
    print(f"Wrote {len(summary)} samples to {args.out}")


if __name__ == "__main__":
    main()
