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
        help="Fraction of non-adjacent-type vessels with a side branch (conditional rate).",
    )
    # Scale buckets are mutually exclusive draws from one partition (see
    # GenerationConfig); they must sum to <= 1. Pass 1.0 to exactly one flag
    # to generate only that case.
    p.add_argument(
        "--small-vessel-probability",
        type=float,
        default=default_cfg.small_vessel_probability,
        help="Fraction of vessels drawn from the small-vessel scale (wall at/inside ring-down).",
    )
    p.add_argument(
        "--large-vessel-beyond-fov-probability",
        type=float,
        default=default_cfg.large_vessel_beyond_fov_probability,
        help="Fraction of vessels drawn from the large beyond-FOV scale (NaN wall A-lines).",
    )
    p.add_argument(
        "--adjacent-vessel-probability",
        type=float,
        default=default_cfg.adjacent_vessel_probability,
        help="Fraction of vessels drawn as the adjacent-vessels type "
        "(parallel neighbors; see docs/configuration.md#adjacent-parallel-vessels).",
    )
    p.add_argument(
        "--aortic-scale-probability",
        type=float,
        default=default_cfg.aortic_scale_probability,
        help="Fraction of vessels drawn at aortic scale.",
    )

    args = p.parse_args()
    cfg = GenerationConfig(
        side_branch_probability=args.side_branch_probability,
        small_vessel_probability=args.small_vessel_probability,
        large_vessel_beyond_fov_probability=args.large_vessel_beyond_fov_probability,
        adjacent_vessel_probability=args.adjacent_vessel_probability,
        aortic_scale_probability=args.aortic_scale_probability,
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
