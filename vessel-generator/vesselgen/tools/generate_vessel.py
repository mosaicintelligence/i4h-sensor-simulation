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
    CalcificationLesionConfig,
    CenterlineConfig,
    CrossSectionConfig,
    DiseasedSectorConfig,
    GuidewireConfig,
    LayeredWallConfig,
    SideBranchConfig,
    VesselConfig,
    WallConfig,
)
from vesselgen.io import save_vessel
from vesselgen.vessel import Vessel
from vesselgen.visualize import preview_vessel


def _build_parent_wall(args: argparse.Namespace):
    if args.layers == 1:
        return WallConfig(
            mean_thickness_mm=args.wall_mm,
            max_perturbation_frac=args.wall_perturbation,
        )
    if args.layers == 2:
        return LayeredWallConfig.media_adventitia(total_thickness_mm=args.wall_mm)
    return LayeredWallConfig.trilaminar(total_thickness_mm=args.wall_mm)


def _build_lesions(args: argparse.Namespace) -> list[CalcificationLesionConfig]:
    if args.n_lesions <= 0:
        return []
    rng = np.random.default_rng(args.seed + 5000)
    base_az = float(args.lesion_azimuth_deg)
    kinds = args.lesion_kinds.split(",") if args.lesion_kinds else ["hard"]
    lesions = []
    for i in range(args.n_lesions):
        kind = kinds[i % len(kinds)]
        lesions.append(CalcificationLesionConfig(
            arclength_frac=float(rng.uniform(0.25, 0.75)),
            azimuth_deg=base_az + float(rng.uniform(-30.0, 30.0)),
            kind=kind,
            arc_extent_deg=float(rng.uniform(40.0, 80.0)),
            axial_extent_mm=float(rng.uniform(4.0, 10.0)),
            seed=args.seed + 10_000 + i,
        ))
    return lesions


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
        wall=_build_parent_wall(args),
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

    guidewire = None
    if args.guidewire:
        guidewire = GuidewireConfig(
            diameter_mm=args.guidewire_diameter_mm,
            lateral_offset_mm=args.guidewire_offset_mm,
            offset_azimuth_deg=args.guidewire_azimuth_deg,
        )

    lesions = _build_lesions(args) if not side_branches else []
    diseased_sector = None
    if lesions:
        diseased_sector = DiseasedSectorConfig(
            dominant_azimuth_deg=float(args.lesion_azimuth_deg),
            sector_extent_deg=120.0,
        )

    return VesselConfig(
        parent=parent,
        side_branches=side_branches,
        seed=args.seed,
        name=args.name,
        lesions=lesions,
        diseased_sector=diseased_sector,
        guidewire=guidewire,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="vessel", help="Name written into the manifest.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--length-mm", type=float, default=55.0)
    p.add_argument("--radius-mm", type=float, default=5.0,
                   help="Mean lumen radius (mm); 5 mm ≈ 10 mm femoral-scale diameter.")
    p.add_argument("--taper", type=float, default=0.95, help="Distal/proximal radius ratio.")
    p.add_argument("--wall-mm", type=float, default=0.85)
    p.add_argument("--n-stations", type=int, default=64)
    p.add_argument("--lumen-perturbation", type=float, default=0.15)
    p.add_argument("--wall-perturbation", type=float, default=0.45)

    p.add_argument("--side-branch", action="store_true")
    p.add_argument("--side-arclength-frac", type=float, default=0.5)
    p.add_argument("--side-azimuth-deg", type=float, default=0.0)
    p.add_argument("--side-polar-deg", type=float, default=60.0)
    p.add_argument("--side-length-mm", type=float, default=45.0)
    p.add_argument("--side-radius-mm", type=float, default=3.5)

    p.add_argument("--layers", type=int, choices=(1, 2, 3), default=1,
                   help="Number of concentric wall layers. 1=single layer "
                        "(legacy), 2=media+adventitia, 3=intima+media+adventitia.")
    p.add_argument("--n-lesions", type=int, default=0,
                   help="Number of in-wall lesions to embed in the parent. "
                        "Side branches must be off to use this.")
    p.add_argument("--lesion-kinds", type=str, default="hard",
                   help="Comma-separated lesion kinds cycled through "
                        "(hard, soft_lipid, fibrous, thrombus).")
    p.add_argument("--lesion-azimuth-deg", type=float, default=0.0,
                   help="Central angle of the diseased sector (deg).")

    p.add_argument("--guidewire", action="store_true",
                   help="Add a tungsten guidewire running through the lumen.")
    p.add_argument("--guidewire-diameter-mm", type=float, default=0.36,
                   help="Common interventional sizes: 0.36 (0.014\"), 0.46 (0.018\"), 0.89 (0.035\").")
    p.add_argument("--guidewire-offset-mm", type=float, default=0.0,
                   help="Lateral offset from the centerline.")
    p.add_argument("--guidewire-azimuth-deg", type=float, default=0.0)

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
