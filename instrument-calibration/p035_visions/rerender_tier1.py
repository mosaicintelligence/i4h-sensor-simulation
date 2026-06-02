"""Re-render `tier1_results/tier1_results.md` from a saved summary.

Use this when you only want to regenerate the writeup (e.g. you've
added new bench-evidence figures or rewritten narrative copy in
`tier1_evaluation.render_markdown`) and don't need to re-execute the
CUDA simulation passes.

The script reads:

  * `tier1_results/tier1_summary.json`  — the per-test result blobs
    saved by the last full run of `tier1_evaluation.py`.
  * Whatever PSF anchor that run was scored against (`psf_anchor`
    key in the summary).

…and emits a fresh `tier1_results.md` using the current
`render_markdown` implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tier1_evaluation as t1  # noqa: E402


@dataclass
class _RestoredResult:
    name: str
    status: str
    summary: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "detail": self.detail,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", type=Path,
                   default=HERE / "tier1_results" / "tier1_summary.json",
                   help="Path to tier1_summary.json.")
    p.add_argument("--out", type=Path,
                   default=HERE / "tier1_results" / "tier1_results.md",
                   help="Where to write the rendered markdown.")
    p.add_argument("--n-frames-wire", type=int, default=8,
                   help="Sim wire-frame count to quote in the writeup "
                        "(visual-only; no compute is done here).")
    p.add_argument("--n-frames-anechoic", type=int, default=8,
                   help="Sim anechoic-frame count to quote in the writeup.")
    args = p.parse_args(argv)

    payload = json.loads(args.summary.read_text())
    # Older summaries stored a bare list of result dicts (the legacy
    # format); the newer one stores
    # ``{psf_anchor: ..., results: [...]}``.  Handle both.
    if isinstance(payload, list):
        result_dicts = payload
        anchor_label = "wave0"
    else:
        result_dicts = payload["results"]
        anchor_label = payload.get("psf_anchor", "wave0")
    results = [_RestoredResult(**r) for r in result_dicts]
    cfg = None  # render_markdown only reads fields we don't depend on
    anchor = t1.resolve_psf_anchor(anchor_label)

    t1.render_markdown(
        results,  # type: ignore[arg-type]
        cfg,
        args.n_frames_wire,
        args.n_frames_anechoic,
        args.out,
        psf_anchor=anchor,
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
