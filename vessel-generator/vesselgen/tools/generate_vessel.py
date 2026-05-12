"""CLI: generate a single vessel and write its folder.

Examples
--------

A straight peripheral vessel with no bifurcation::

    python -m vesselgen.tools.generate_vessel --out out/vessel_straight --seed 1

A peripheral vessel with one side branch::

    python -m vesselgen.tools.generate_vessel --out out/vessel_side --seed 7 --side-branch
"""

from __future__ import annotations

import argparse
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
from vesselgen.visualize import preview_vessel


def _make_config(args: argparse.Namespace) -> VesselConfig:
    parent = BranchConfig(
        centerline=CenterlineConfig(
            length_mm=args.length_mm,
            origin=(0.0, -args.length_mm / 2.0, 0.0),
            direction=(0.0, 1.0, 0.0),
            n_stations=args.n_stations,
        ),
        cross_section=CrossSectionConfig(
            mean_radius_mm=args.radius_mm,
            distal_radius_mm=args.radius_mm * args.taper,
            max_perturbation_frac=args.lumen_perturbation,
        ),
        wall=WallConfig(
            mean_thickness_mm=args.wall_mm,
            max_perturbation_frac=args.wall_perturbation,
        ),
        name="parent",
        seed=args.seed,
    )

    side_branches = []
    if args.side_branch:
        side_branches.append(
            SideBranchConfig(
                parent_arclength_frac=args.side_arclength_frac,
                azimuth_deg=args.side_azimuth_deg,
                polar_deg=args.side_polar_deg,
                branch=BranchConfig(
                    centerline=CenterlineConfig(
                        length_mm=args.side_length_mm,
                        n_stations=max(24, args.n_stations // 2),
                    ),
                    cross_section=CrossSectionConfig(
                        mean_radius_mm=args.side_radius_mm,
                        distal_radius_mm=args.side_radius_mm * 0.85,
                    ),
                    wall=WallConfig(
                        mean_thickness_mm=args.wall_mm * 0.85,
                    ),
                    name="side_branch",
                    seed=args.seed + 100,
                ),
            )
        )

    return VesselConfig(
        parent=parent,
        side_branches=side_branches,
        seed=args.seed,
        name=args.name,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="vessel", help="Name written into the manifest.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--length-mm", type=float, default=30.0)
    p.add_argument("--radius-mm", type=float, default=3.0)
    p.add_argument("--taper", type=float, default=0.95, help="Distal/proximal radius ratio.")
    p.add_argument("--wall-mm", type=float, default=0.7)
    p.add_argument("--n-stations", type=int, default=64)
    p.add_argument("--lumen-perturbation", type=float, default=0.18)
    p.add_argument("--wall-perturbation", type=float, default=0.5)

    p.add_argument("--side-branch", action="store_true")
    p.add_argument("--side-arclength-frac", type=float, default=0.5)
    p.add_argument("--side-azimuth-deg", type=float, default=0.0)
    p.add_argument("--side-polar-deg", type=float, default=60.0)
    p.add_argument("--side-length-mm", type=float, default=15.0)
    p.add_argument("--side-radius-mm", type=float, default=1.5)

    p.add_argument("--no-preview", action="store_true")

    args = p.parse_args()
    cfg = _make_config(args)
    vessel = Vessel.from_config(cfg)
    out = save_vessel(vessel, args.out)
    if not args.no_preview:
        preview_vessel(vessel, args.out / "preview.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
