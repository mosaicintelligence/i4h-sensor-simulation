"""CLI: generate a batch of vessels."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesselgen.config import GenerationConfig
from vesselgen.library import generate_dataset


def main() -> None:
    default_cfg = GenerationConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--name-prefix", default="vessel")
    p.add_argument("--no-previews", action="store_true")

    p.add_argument(
        "--side-branch-probability",
        type=float,
        default=default_cfg.side_branch_probability,
    )
    p.add_argument(
        "--small-vessel-probability",
        type=float,
        default=default_cfg.small_vessel_probability,
    )
    p.add_argument(
        "--small-vessel-radius-mm-range",
        nargs=2,
        type=float,
        metavar=("LOW_MM", "HIGH_MM"),
        default=default_cfg.small_vessel_radius_mm_range,
    )
    p.add_argument(
        "--small-vessel-wall-thickness-mm-range",
        nargs=2,
        type=float,
        metavar=("LOW_MM", "HIGH_MM"),
        default=default_cfg.small_vessel_wall_thickness_mm_range,
    )
    p.add_argument(
        "--large-vessel-beyond-fov-probability",
        type=float,
        default=default_cfg.large_vessel_beyond_fov_probability,
    )
    p.add_argument(
        "--large-vessel-radius-mm-range",
        nargs=2,
        type=float,
        metavar=("LOW_MM", "HIGH_MM"),
        default=default_cfg.large_vessel_radius_mm_range,
    )
    p.add_argument(
        "--large-vessel-wall-thickness-mm-range",
        nargs=2,
        type=float,
        metavar=("LOW_MM", "HIGH_MM"),
        default=default_cfg.large_vessel_wall_thickness_mm_range,
    )

    args = p.parse_args()
    cfg = GenerationConfig(
        side_branch_probability=args.side_branch_probability,
        small_vessel_probability=args.small_vessel_probability,
        small_vessel_radius_mm_range=tuple(args.small_vessel_radius_mm_range),
        small_vessel_wall_thickness_mm_range=tuple(args.small_vessel_wall_thickness_mm_range),
        large_vessel_beyond_fov_probability=args.large_vessel_beyond_fov_probability,
        large_vessel_radius_mm_range=tuple(args.large_vessel_radius_mm_range),
        large_vessel_wall_thickness_mm_range=tuple(args.large_vessel_wall_thickness_mm_range),
    )
    written = generate_dataset(
        n=args.n,
        out_dir=args.out,
        config=cfg,
        base_seed=args.base_seed,
        name_prefix=args.name_prefix,
        write_previews=not args.no_previews,
    )
    print(f"Wrote {len(written)} vessels to {args.out}")


if __name__ == "__main__":
    main()
