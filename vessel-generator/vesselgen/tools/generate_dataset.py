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

    p.add_argument("--side-branch-probability", type=float, default=0.45)
    p.add_argument(
        "--adjacent-vessel-probability",
        type=float,
        default=GenerationConfig.adjacent_vessel_probability,
        help="Fraction of vessels drawn as the adjacent-vessels type "
        "(parallel neighbors; see docs/configuration.md#adjacent-parallel-vessels).",
    )

    args = p.parse_args()
    # Adjacent and aortic scale draws are mutually exclusive buckets; cap the
    # default aortic share so the config partition remains valid.
    max_aortic_share = max(0.0, 1.0 - args.adjacent_vessel_probability)
    aortic_scale_probability = min(default_cfg.aortic_scale_probability, max_aortic_share)
    cfg = GenerationConfig(
        side_branch_probability=args.side_branch_probability,
        adjacent_vessel_probability=args.adjacent_vessel_probability,
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
