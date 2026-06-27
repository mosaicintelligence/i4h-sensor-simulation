# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay a recorded IVUS sensor trajectory as an animation.

Loads the CTA mesh (wireframe) and animates the recorded sensor frame — a moving
point, its three axis arrows, and a growing trail — through the saved poses.
See ``recordings/README.md`` for the recording formats.

    uv run mosaic_sim/replay.py mosaic_sim/recordings/ivus_pig_cta_000.npz
    uv run mosaic_sim/replay.py mosaic_sim/recordings/ivus_pig_cta_000.csv --viewer gl
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import newton.examples  # noqa: E402

import mosaic_sim  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("usage: uv run mosaic_sim/replay.py <recording.npz|.csv> [viewer flags]")
        sys.exit(1)
    recording = sys.argv.pop(1)
    sys.argv += ["--replay", recording]

    parser = mosaic_sim._build_parser()
    viewer, args = newton.examples.init(parser)
    example = mosaic_sim.Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
