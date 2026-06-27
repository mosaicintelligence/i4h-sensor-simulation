# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Turn a solid-lumen vessel STL into an open-ended, thick-walled shell.

The input STL (e.g. a Syndaver phantom) is a single watertight surface whose
interior is the lumen and whose tube ends are capped. This tool:

  1. removes the end caps (so every tube is open), then
  2. offsets the surface outward by a uniform wall thickness to create a second
     (outer) wall, and bridges the two walls at each opening,

producing a watertight tube **with wall thickness** whose **lumen is open at both
ends**. The result is written as a new STL you can load like any other mesh.

Usage:
    uv run mosaic_sim/make_shell.py mosaic_sim/stls/Syndaver.stl
    uv run mosaic_sim/make_shell.py in.stl --thickness 2.0 --out out_shell.stl
"""

from __future__ import annotations

import argparse
import collections
import math
import os

import numpy as np
import trimesh


def remove_end_caps(
    verts: np.ndarray,
    faces: np.ndarray,
    smooth_deg: float = 25.0,
    disk_min: float = 0.35,
    min_faces: int = 4,
) -> np.ndarray:
    """Return faces with the end-cap disks removed (tubes left open).

    Segments the surface into smooth regions separated by sharp creases
    (neighbour-to-neighbour normal test). The walls form one big region; each
    flat-ish end cap is cut off at its rim crease into its own compact,
    disk-like region. Those disk regions are the caps.
    """
    nf = len(faces)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12

    adj: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adj.setdefault((min(a, b), max(a, b)), []).append(fi)

    cos_t = math.cos(math.radians(smooth_deg))
    comp = -np.ones(nf, dtype=np.int64)
    cid = 0
    for seed in range(nf):
        if comp[seed] >= 0:
            continue
        comp[seed] = cid
        dq = collections.deque([seed])
        while dq:
            f = dq.popleft()
            tri = faces[f]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                for g in adj[(min(a, b), max(a, b))]:
                    if comp[g] < 0 and abs(float(fn[f] @ fn[g])) > cos_t:
                        comp[g] = cid
                        dq.append(g)
        cid += 1

    wall = int(np.argmax(np.bincount(comp)))
    remove = np.zeros(nf, dtype=bool)
    ncaps = 0
    for r in range(cid):
        if r == wall:
            continue
        idx = np.where(comp == r)[0]
        if len(idx) < min_faces:
            continue
        mn = fn[idx].mean(axis=0)
        mn /= np.linalg.norm(mn) + 1e-12
        vv = np.unique(faces[idx])
        p = verts[vv] - verts[vv].mean(axis=0)
        p = p - np.outer(p @ mn, mn)
        ev = np.linalg.eigvalsh(p.T @ p)
        if ev[1] / (ev[2] + 1e-12) > disk_min:  # disk-like => cap
            remove[idx] = True
            ncaps += 1
    print(f"  removed {ncaps} end caps ({int(remove.sum())} faces)")
    return faces[~remove]


def solidify(verts: np.ndarray, faces: np.ndarray, thickness: float):
    """Offset the open surface outward to make a second wall and bridge the rims.

    Returns ``(verts, faces)`` of a watertight shell with wall thickness whose
    tube openings remain open (annular rims).
    """
    surf = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vn = np.asarray(surf.vertex_normals)  # outward, winding is consistent

    nv = len(verts)
    inner = verts
    outer = verts + thickness * vn
    out_verts = np.vstack([inner, outer])

    inner_faces = faces[:, ::-1]  # flip so they face the lumen
    outer_faces = faces + nv  # original winding -> face outward

    # Bridge every boundary half-edge (a->b appears in exactly one face).
    edge_count: dict[tuple[int, int], int] = collections.defaultdict(int)
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_count[(min(a, b), max(a, b))] += 1
    rim = []
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if edge_count[(min(a, b), max(a, b))] == 1:  # boundary half-edge a->b
                rim.append([a, b, b + nv])
                rim.append([a, b + nv, a + nv])

    out_faces = np.vstack([inner_faces, outer_faces, np.asarray(rim, dtype=np.int64)])
    shell = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=True)
    shell.fix_normals()
    return shell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl", help="input vessel STL (solid-lumen, capped)")
    ap.add_argument("--thickness", type=float, default=1.5, help="wall thickness in model units (mm)")
    ap.add_argument("--out", default=None, help="output STL path (default: <input>_shell.stl)")
    ap.add_argument("--smooth-deg", type=float, default=25.0, help="cap-detection crease angle")
    args = ap.parse_args()

    out_path = args.out or os.path.splitext(args.stl)[0] + "_shell.stl"
    m = trimesh.load(args.stl, force="mesh")
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    print(f"loaded {args.stl}: {len(verts)} verts, {len(faces)} faces, watertight={m.is_watertight}")

    open_faces = remove_end_caps(verts, faces, smooth_deg=args.smooth_deg)
    shell = solidify(verts, open_faces, args.thickness)

    # Report: open lumen ends show up as boundary loops on the inner surface only
    # after solidify the shell is watertight, so quantify thickness via volume gain.
    print(
        f"  shell: {len(shell.vertices)} verts, {len(shell.faces)} faces, "
        f"watertight={shell.is_watertight}, thickness={args.thickness} mm"
    )
    shell.export(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
