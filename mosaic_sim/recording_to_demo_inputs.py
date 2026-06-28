# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Turn a mosaic_sim recording into the inputs the IVUS pullback pipeline needs.

The `simulated_pullback_from_CT` pipeline (and the IVUS_demo) consume a vessel as
``lumen STL + centerline.txt + snakes.csv``. This builds the centerline + snakes
from a catheter recording so the IVUS sim runs on the *physics* probe path
through your own mesh, instead of a hand-segmented centerline:

  - centerline.txt : the recorded sensor positions (CTA mm), one "x y z" per line
  - snakes.csv     : per-point normal (parallel-transport) + lumen radius
                     (Min/Max/Median), measured as distance to the mesh wall

Only needs numpy + scipy + trimesh (no newton/warp).

    python mosaic_sim/recording_to_demo_inputs.py recordings/ivus_pig_cta_000.npz \
        --mesh mosaic_sim/stls/pig_cta.stl --out pig_inputs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from to_pullback_poses import load_recording, parallel_transport_frames  # noqa: E402


def build_demo_inputs(recording: str, mesh_path: str, out_dir: str, reverse: bool = False):
    pos, mesh_name = load_recording(recording)  # [N, 3] in the CTA mesh frame
    if reverse:
        pos = pos[::-1].copy()
    _tang, normals, _binormals = parallel_transport_frames(pos)

    # Lumen radius ≈ distance from each centerline point to the nearest wall.
    mesh = trimesh.load(mesh_path, force="mesh")
    _closest, dist, _tri = trimesh.proximity.closest_point(mesh, pos)
    radius = np.clip(np.asarray(dist, dtype=float), 0.3, None)

    os.makedirs(out_dir, exist_ok=True)
    cl_path = os.path.join(out_dir, "centerline.txt")
    sn_path = os.path.join(out_dir, "snakes.csv")

    with open(cl_path, "w") as f:
        for p in pos:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    # Semicolon-delimited, matching geometry.load_snakes_csv's expected columns.
    with open(sn_path, "w") as f:
        f.write("point-x;point-y;point-z;normal-x;normal-y;normal-z;Min;Max;Median\n")
        for i, p in enumerate(pos):
            n = normals[i]
            r = float(radius[i])
            f.write(
                f"{p[0]:.6f};{p[1]:.6f};{p[2]:.6f};"
                f"{n[0]:.6f};{n[1]:.6f};{n[2]:.6f};"
                f"{0.85 * r:.6f};{1.15 * r:.6f};{r:.6f}\n"
            )

    print(
        f"{len(pos)} centerline points (mesh '{mesh_name}', radius "
        f"{radius.min():.2f}-{radius.max():.2f} mm)\n  -> {cl_path}\n  -> {sn_path}"
    )
    return cl_path, sn_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", help="mosaic_sim recording (.npz or .csv)")
    ap.add_argument("--mesh", required=True, help="the lumen STL the recording was made against")
    ap.add_argument("--out", default="pig_inputs", help="output directory")
    ap.add_argument("--reverse", action="store_true", help="reverse to pullback order (distal->proximal)")
    args = ap.parse_args()
    build_demo_inputs(args.recording, args.mesh, args.out, reverse=args.reverse)


if __name__ == "__main__":
    main()
