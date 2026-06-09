from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import load_centerline_txt, load_snakes_csv


def _resample_polyline(points: np.ndarray, step_mm: float, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("Centerline length is zero.")
    n = int(math.floor(total / step_mm)) + 1
    if max_frames is not None and n > max_frames:
        n = max_frames
    s_new = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=np.float64)
    for dim in range(3):
        out[:, dim] = np.interp(s_new, s, points[:, dim])
    return out, s_new


def _parallel_transport_frames(points: np.ndarray, normal0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = points.shape[0]
    t = np.zeros((n, 3), dtype=np.float64)
    t[1:-1] = points[2:] - points[:-2]
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)

    if normal0 is None:
        guess = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(guess, t[0]))) > 0.9:
            guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n0 = guess - np.dot(guess, t[0]) * t[0]
    else:
        n0 = normal0 - np.dot(normal0, t[0]) * t[0]
    n0 /= np.maximum(np.linalg.norm(n0), 1e-9)

    nvec = np.zeros((n, 3), dtype=np.float64)
    bvec = np.zeros((n, 3), dtype=np.float64)
    nvec[0] = n0
    bvec[0] = np.cross(nvec[0], t[0])
    bvec[0] /= np.maximum(np.linalg.norm(bvec[0]), 1e-9)

    for i in range(1, n):
        v_prev = t[i - 1]
        v_cur = t[i]
        axis = np.cross(v_prev, v_cur)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-10:
            ni = nvec[i - 1]
        else:
            axis_u = axis / axis_norm
            angle = float(np.arccos(np.clip(np.dot(v_prev, v_cur), -1.0, 1.0)))
            ni = Rotation.from_rotvec(axis_u * angle).apply(nvec[i - 1])
        ni = ni - np.dot(ni, v_cur) * v_cur
        ni /= np.maximum(np.linalg.norm(ni), 1e-9)
        bi = np.cross(ni, v_cur)
        bi /= np.maximum(np.linalg.norm(bi), 1e-9)
        nvec[i] = ni
        bvec[i] = bi
    return t, nvec, bvec


def _euler_xyz_from_basis(tangent: np.ndarray, normal: np.ndarray, binormal: np.ndarray) -> np.ndarray:
    # Local probe axes in world frame:
    # x -> normal, y -> tangent (pullback axis), z -> binormal.
    rot_m = np.column_stack((normal, tangent, binormal))
    rot = Rotation.from_matrix(rot_m)
    return rot.as_euler("xyz", degrees=False)


def build_pullback_poses(
    centerline_txt_path: str | Path,
    snakes_csv_path: str | Path,
    step_mm: float,
    max_frames: int | None,
) -> dict[str, Any]:
    centerline = load_centerline_txt(centerline_txt_path)
    snakes = load_snakes_csv(snakes_csv_path)

    # Use the first snake normal as a stable initial orientation anchor.
    normal0 = np.asarray(snakes["normals"][0], dtype=np.float64)
    points, s = _resample_polyline(centerline, step_mm=step_mm, max_frames=max_frames)
    tangents, normals, binormals = _parallel_transport_frames(points, normal0=normal0)
    eulers = np.vstack(
        [_euler_xyz_from_basis(tangents[i], normals[i], binormals[i]) for i in range(points.shape[0])]
    )
    records: list[dict[str, Any]] = []
    for i in range(points.shape[0]):
        records.append(
            {
                "frame_index": i,
                "distance_mm": float(s[i]),
                "position_mm": [float(v) for v in points[i]],
                "rotation_rad_xyz": [float(v) for v in eulers[i]],
                "tangent": [float(v) for v in tangents[i]],
                "normal": [float(v) for v in normals[i]],
                "binormal": [float(v) for v in binormals[i]],
            }
        )
    return {
        "n_frames": int(points.shape[0]),
        "distance_total_mm": float(s[-1]),
        "step_mm_effective": float(s[1] - s[0]) if points.shape[0] > 1 else 0.0,
        "poses": records,
    }


def write_pose_csv(poses_payload: dict[str, Any], out_csv_path: str | Path) -> None:
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "frame_index",
        "distance_mm",
        "pos_x",
        "pos_y",
        "pos_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "tan_x",
        "tan_y",
        "tan_z",
        "nor_x",
        "nor_y",
        "nor_z",
        "bin_x",
        "bin_y",
        "bin_z",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for rec in poses_payload["poses"]:
            p = rec["position_mm"]
            r = rec["rotation_rad_xyz"]
            t = rec["tangent"]
            n = rec["normal"]
            b = rec["binormal"]
            w.writerow(
                {
                    "frame_index": rec["frame_index"],
                    "distance_mm": f"{rec['distance_mm']:.6f}",
                    "pos_x": f"{p[0]:.6f}",
                    "pos_y": f"{p[1]:.6f}",
                    "pos_z": f"{p[2]:.6f}",
                    "rot_x": f"{r[0]:.8f}",
                    "rot_y": f"{r[1]:.8f}",
                    "rot_z": f"{r[2]:.8f}",
                    "tan_x": f"{t[0]:.8f}",
                    "tan_y": f"{t[1]:.8f}",
                    "tan_z": f"{t[2]:.8f}",
                    "nor_x": f"{n[0]:.8f}",
                    "nor_y": f"{n[1]:.8f}",
                    "nor_z": f"{n[2]:.8f}",
                    "bin_x": f"{b[0]:.8f}",
                    "bin_y": f"{b[1]:.8f}",
                    "bin_z": f"{b[2]:.8f}",
                }
            )


def write_pose_json(poses_payload: dict[str, Any], out_json_path: str | Path) -> None:
    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(poses_payload, f, indent=2)
