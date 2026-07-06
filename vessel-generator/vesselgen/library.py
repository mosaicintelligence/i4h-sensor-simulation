"""Batch generation of vessels from a :class:`GenerationConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from vesselgen.bifurcation import BifurcationError
from vesselgen.config import GenerationConfig, VesselConfig
from vesselgen.io import save_vessel
from vesselgen.vessel import Vessel
from vesselgen.visualize import preview_vessel


def generate_dataset(
    n: int,
    out_dir: str | Path,
    config: GenerationConfig | None = None,
    base_seed: int = 0,
    name_prefix: str = "vessel",
    write_previews: bool = True,
    max_resamples_per_vessel: int = 5,
) -> list[Path]:
    """Generate ``n`` vessels and write each to ``out_dir/<name_prefix>_NNNN``.

    On any per-vessel failure (e.g. boolean union failure), the function
    re-samples up to ``max_resamples_per_vessel`` times before giving up
    on that index. Returns the list of output directories that were
    actually written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or GenerationConfig()

    written: list[Path] = []
    for i in range(n):
        success = False
        for retry in range(max_resamples_per_vessel):
            seed = base_seed + i * 1000 + retry
            rng = np.random.default_rng(seed)
            cfg = config.sample(rng, seed=seed, name=f"{name_prefix}_{i:04d}")
            try:
                vessel = Vessel.from_config(cfg)
            except BifurcationError as e:
                print(f"[{cfg.name}] retry {retry+1}: {e}")
                continue
            vessel_dir = out_dir / cfg.name
            save_vessel(vessel, vessel_dir)
            if write_previews:
                preview_vessel(vessel, vessel_dir / "preview.png")
            written.append(vessel_dir)
            success = True
            break
        if not success:
            print(f"[{name_prefix}_{i:04d}] gave up after {max_resamples_per_vessel} retries")
    return written


def iter_dataset(
    n: int,
    config: GenerationConfig | None = None,
    base_seed: int = 0,
    name_prefix: str = "vessel",
    max_resamples_per_vessel: int = 5,
    require_side_branch: bool = False,
    start_index: int = 0,
) -> Iterator[tuple[VesselConfig, Vessel]]:
    """Iterator variant of :func:`generate_dataset` for in-memory consumption.

    ``start_index`` skips the first N vessel slots without sampling
    them. Used by ``render_paired_dataset.py --resume`` to continue a
    partially-completed run without redoing the vessels that already
    landed on disk: the seed schedule (``base_seed + i * 1000 + retry``)
    is index-keyed, so the resumed iterator picks up the exact vessel
    configs the original run would have produced.
    """
    config = config or GenerationConfig()
    for i in range(start_index, n):
        for retry in range(max_resamples_per_vessel):
            seed = base_seed + i * 1000 + retry
            rng = np.random.default_rng(seed)
            cfg = config.sample(
                rng,
                seed=seed,
                name=f"{name_prefix}_{i:04d}",
                force_side_branch=require_side_branch,
            )
            try:
                vessel = Vessel.from_config(cfg)
            except BifurcationError:
                continue
            yield cfg, vessel
            break
