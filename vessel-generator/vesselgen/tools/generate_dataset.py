"""CLI: generate a batch of vessels."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesselgen.config import GenerationConfig, clamp_default_aortic_scale_probability
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
        help=(
            "Fraction of vessels drawn from the small-vessel scale (wall at or "
            "inside the ring-down disc). Radius/wall ranges keep their "
            "GenerationConfig defaults; set them via the config class for finer control."
        ),
    )
    p.add_argument(
        "--aortic-scale-probability",
        type=float,
        default=None,
        help=(
            "Optional explicit aortic-scale fraction. If omitted, the default "
            "aortic fraction is used and clamped only when needed so "
            "small-vessel + aortic-scale does not exceed 1.0."
        ),
    )

    args = p.parse_args()
    aortic_scale_probability = (
        args.aortic_scale_probability
        if args.aortic_scale_probability is not None
        else clamp_default_aortic_scale_probability(
            small_vessel_probability=args.small_vessel_probability,
            default_aortic_scale_probability=default_cfg.aortic_scale_probability,
        )
    )
    cfg = GenerationConfig(
        side_branch_probability=args.side_branch_probability,
        small_vessel_probability=args.small_vessel_probability,
        aortic_scale_probability=aortic_scale_probability,
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
