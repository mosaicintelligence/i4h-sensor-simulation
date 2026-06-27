# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert a mosaic_sim IVUS recording into a pullback-pose payload.

Bridges the catheter physics (``mosaic_sim.py`` -> ``recordings/ivus_*.{npz,csv}``)
to the ``simulated_pullback_from_CT`` IVUS pipeline. The recording is already in
the CTA mesh frame (mm); this builds the per-frame probe orientation in that
pipeline's convention (probe axes: x->normal, y->tangent/pullback-axis,
z->binormal) using a parallel-transport frame, and writes the same
``*_poses.csv`` / ``*_poses.json`` shape that pipeline's ``trajectory.py`` emits.

Only depends on numpy + scipy (no newton/warp), matching the pullback project.

    python mosaic_sim/to_pullback_poses.py recordings/ivus_pig_cta_000.npz
    python mosaic_sim/to_pullback_poses.py rec.csv --out-prefix out/pig --reverse

The resulting payload feeds ``run_pullback_simulation(..., poses_payload=...)``
together with the lumen mesh (the same STL) and an outer wall.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation


def load_recording(path: str):
    """Return (position[N,3] mm, mesh_name) from a .npz or .csv recording."""
    if path.endswith(".npz"):
        d = np.load(path)
        return np.asarray(d["position"], dtype=np.float64), str(d["mesh"])
    rows = [ln for ln in open(path) if ln.strip() and not ln.startswith("#") and not ln.startswith("time")]
    arr = np.array([[float(x) for x in r.split(",")] for r in rows])
    mesh = "unknown"
    for ln in open(path):
        if ln.startswith("# mesh="):
            mesh = ln.split("mesh=", 1)[1].split()[0]
            break
    return arr[:, 2:5], mesh


def parallel_transport_frames(points: np.ndarray):
    """Smooth, minimally-rotating (tangent, normal, binormal) frames along a path.

    Mirrors simulated_pullback_from_CT/src/trajectory.py so the orientation
    convention matches the IVUS pipeline exactly.
    """
    n = len(points)
    d = np.diff(points, axis=0)
    seg = np.linalg.norm(d, axis=1, keepdims=True)
    dirs = d / np.maximum(seg, 1e-9)
    tang = np.zeros((n, 3))
    tang[:-1] = dirs
    tang[-1] = dirs[-1]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)

    # Initial normal: any unit vector perpendicular to the first tangent.
    seed = np.array([0.0, 0.0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = seed - np.dot(seed, tang[0]) * tang[0]
    n0 /= np.linalg.norm(n0) + 1e-9

    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))
    normals[0] = n0
    binormals[0] = np.cross(n0, tang[0])
    for i in range(1, n):
        v_prev, v_cur = tang[i - 1], tang[i]
        axis = np.cross(v_prev, v_cur)
        an = np.linalg.norm(axis)
        if an < 1e-10:
            ni = normals[i - 1]
        else:
            ang = float(np.arccos(np.clip(np.dot(v_prev, v_cur), -1.0, 1.0)))
            ni = Rotation.from_rotvec(axis / an * ang).apply(normals[i - 1])
        ni = ni - np.dot(ni, v_cur) * v_cur
        ni /= np.linalg.norm(ni) + 1e-9
        bi = np.cross(ni, v_cur)
        bi /= np.linalg.norm(bi) + 1e-9
        normals[i] = ni
        binormals[i] = bi
    return tang, normals, binormals


def poses_from_recording(path: str, reverse: bool = False) -> dict:
    """Build a pullback-pose payload (pipeline format) from a recording."""
    position, mesh = load_recording(path)
    if reverse:  # insertion order -> pullback order (distal first)
        position = position[::-1].copy()
    tang, normals, binormals = parallel_transport_frames(position)
    seg = np.linalg.norm(np.diff(position, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    records = []
    for i in range(len(position)):
        rot_m = np.column_stack((normals[i], tang[i], binormals[i]))  # x,y,z = normal,tangent,binormal
        eul = Rotation.from_matrix(rot_m).as_euler("xyz", degrees=False)
        records.append(
            {
                "frame_index": i,
                "distance_mm": float(s[i]),
                "position_mm": [float(v) for v in position[i]],
                "rotation_rad_xyz": [float(v) for v in eul],
                "tangent": [float(v) for v in tang[i]],
                "normal": [float(v) for v in normals[i]],
                "binormal": [float(v) for v in binormals[i]],
            }
        )
    return {
        "n_frames": len(position),
        "distance_total_mm": float(s[-1]),
        "step_mm_effective": float(np.mean(seg)) if len(seg) else 0.0,
        "source": f"mosaic_sim recording {os.path.basename(path)}",
        "mesh": mesh,
        "poses": records,
    }


_CSV_COLUMNS = [
    "frame_index", "distance_mm",
    "pos_x", "pos_y", "pos_z",
    "rot_x", "rot_y", "rot_z",
    "tan_x", "tan_y", "tan_z",
    "nor_x", "nor_y", "nor_z",
    "bin_x", "bin_y", "bin_z",
]


def write_pose_csv(payload: dict, out_csv: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in payload["poses"]:
            p, rot, t, n, b = (r["position_mm"], r["rotation_rad_xyz"], r["tangent"], r["normal"], r["binormal"])
            w.writerow(
                {
                    "frame_index": r["frame_index"], "distance_mm": f"{r['distance_mm']:.6f}",
                    "pos_x": f"{p[0]:.6f}", "pos_y": f"{p[1]:.6f}", "pos_z": f"{p[2]:.6f}",
                    "rot_x": f"{rot[0]:.8f}", "rot_y": f"{rot[1]:.8f}", "rot_z": f"{rot[2]:.8f}",
                    "tan_x": f"{t[0]:.8f}", "tan_y": f"{t[1]:.8f}", "tan_z": f"{t[2]:.8f}",
                    "nor_x": f"{n[0]:.8f}", "nor_y": f"{n[1]:.8f}", "nor_z": f"{n[2]:.8f}",
                    "bin_x": f"{b[0]:.8f}", "bin_y": f"{b[1]:.8f}", "bin_z": f"{b[2]:.8f}",
                }
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", help="mosaic_sim recording (.npz or .csv)")
    ap.add_argument("--out-prefix", default=None, help="output prefix (default: alongside the recording)")
    ap.add_argument("--reverse", action="store_true", help="reverse to pullback order (distal -> proximal)")
    args = ap.parse_args()

    payload = poses_from_recording(args.recording, reverse=args.reverse)
    prefix = args.out_prefix or os.path.splitext(args.recording)[0]
    write_pose_csv(payload, prefix + "_poses.csv")
    with open(prefix + "_poses.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"{payload['n_frames']} poses, {payload['distance_total_mm']:.1f} mm total "
        f"-> {prefix}_poses.csv / .json"
    )


if __name__ == "__main__":
    main()
