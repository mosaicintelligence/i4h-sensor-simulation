# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert cylinder OBJ files from "axis along Y" to IVUS orientation:
#   - IVUS imaging plane = y-z, catheter axis = x.
#   - OBJ cylinders have axis along Y (y in [-2, 2], circle in x-z).
#   - Transform: (x, y, z) -> (y, x, z + z_center) so cylinder axis becomes x.
#   - Normals: (nx, ny, nz) -> (ny, nx, nz).
#
# Usage (from repo root):
#   python utils/cylinder_obj_to_ivus.py
# Writes mesh/Cylinder_inner_ivus.obj and mesh/Cylinder_outer_ivus.obj.

import os
import sys

# Default: run from ultrasound-raytracing (repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(REPO_ROOT, "mesh")
Z_CENTER = 10.0  # mm (vessel center depth in IVUS)


def transform_vertex(x: float, y: float, z: float) -> tuple:
    """OBJ axis Y -> IVUS axis X; place at z_center."""
    return (y, x, z + Z_CENTER)


def transform_normal(nx: float, ny: float, nz: float) -> tuple:
    """Rotate normal the same way as geometry: (nx,ny,nz) -> (ny,nx,nz)."""
    return (ny, nx, nz)


def convert_obj(in_path: str, out_path: str) -> None:
    vertices = []
    normals = []
    faces = []

    with open(in_path) as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                v = transform_vertex(x, y, z)
                vertices.append("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
            elif parts[0] == "vn" and len(parts) >= 4:
                nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                n = transform_normal(nx, ny, nz)
                normals.append("vn {:.6f} {:.6f} {:.6f}\n".format(n[0], n[1], n[2]))
            elif parts[0] == "f":
                faces.append(line + "\n")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# IVUS-oriented cylinder (axis = x, center z = {} mm)\n".format(Z_CENTER))
        f.write("# From {}\n".format(os.path.basename(in_path)))
        for v in vertices:
            f.write(v)
        for n in normals:
            f.write(n)
        for face in faces:
            f.write(face)
    print("Wrote {}".format(out_path))


def main():
    os.chdir(REPO_ROOT)
    inner_src = os.path.join(MESH_DIR, "Cylinder_inner.obj")
    outer_src = os.path.join(MESH_DIR, "Cylinder_outer.obj")
    inner_dst = os.path.join(MESH_DIR, "Cylinder_inner_ivus.obj")
    outer_dst = os.path.join(MESH_DIR, "Cylinder_outer_ivus.obj")

    if not os.path.isfile(inner_src):
        print("Not found: {}".format(inner_src), file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(outer_src):
        print("Not found: {}".format(outer_src), file=sys.stderr)
        sys.exit(1)

    convert_obj(inner_src, inner_dst)
    convert_obj(outer_src, outer_dst)


if __name__ == "__main__":
    main()
