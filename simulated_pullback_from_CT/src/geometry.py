from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from common import write_json


@dataclass
class WallModelConfig:
    thickness_scale_of_radius: float
    min_thickness_mm: float
    max_thickness_mm: float
    smooth_iterations: int


def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.remove_unreferenced_vertices()
    if hasattr(m, "nondegenerate_faces"):
        m.update_faces(m.nondegenerate_faces())
    if hasattr(m, "unique_faces"):
        m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    m.remove_infinite_values()
    _ = m.fill_holes()
    return m


def load_centerline_txt(path: str | Path) -> np.ndarray:
    points: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            x, y, z = s.split()
            points.append([float(x), float(y), float(z)])
    out = np.asarray(points, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3 or out.shape[0] < 2:
        raise ValueError(f"Invalid centerline file: {path}")
    return out


def load_snakes_csv(path: str | Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            if not clean.get("point-x"):
                continue
            rows.append(
                {
                    "point_x": float(clean["point-x"]),
                    "point_y": float(clean["point-y"]),
                    "point_z": float(clean["point-z"]),
                    "normal_x": float(clean["normal-x"]),
                    "normal_y": float(clean["normal-y"]),
                    "normal_z": float(clean["normal-z"]),
                    "min_r": float(clean["Min"]),
                    "max_r": float(clean["Max"]),
                    "median_r": float(clean["Median"]),
                }
            )
    if not rows:
        raise ValueError(f"No valid rows in snakes CSV: {path}")
    arr = lambda *k: np.asarray([[r[x] for x in k] for r in rows], dtype=np.float64)
    return {
        "points": arr("point_x", "point_y", "point_z"),
        "normals": arr("normal_x", "normal_y", "normal_z"),
        "r_min": np.asarray([r["min_r"] for r in rows], dtype=np.float64),
        "r_max": np.asarray([r["max_r"] for r in rows], dtype=np.float64),
        "r_median": np.asarray([r["median_r"] for r in rows], dtype=np.float64),
    }


def _vertex_thicknesses(
    vertices: np.ndarray,
    centerline_points: np.ndarray,
    centerline_radii_mm: np.ndarray,
    cfg: WallModelConfig,
) -> np.ndarray:
    tree = cKDTree(centerline_points)
    _, idx = tree.query(vertices, k=1)
    local_radius = centerline_radii_mm[idx]
    thickness = local_radius * float(cfg.thickness_scale_of_radius)
    thickness = np.clip(thickness, float(cfg.min_thickness_mm), float(cfg.max_thickness_mm))
    return thickness.astype(np.float64)


def _laplacian_smooth_scalar_on_mesh(values: np.ndarray, faces: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return values
    n = values.shape[0]
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    out = values.copy()
    for _ in range(iterations):
        nxt = out.copy()
        for i in range(n):
            if not neighbors[i]:
                continue
            ids = list(neighbors[i])
            nxt[i] = 0.5 * out[i] + 0.5 * float(np.mean(out[ids]))
        out = nxt
    return out


def _ensure_normals_outward(vertices: np.ndarray, normals: np.ndarray, centerline_points: np.ndarray) -> np.ndarray:
    tree = cKDTree(centerline_points)
    _, idx = tree.query(vertices, k=1)
    radial = vertices - centerline_points[idx]
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_unit = radial / np.maximum(radial_norm, 1e-9)
    score = float(np.mean(np.sum(radial_unit * normals, axis=1)))
    if score < 0.0:
        return -normals
    return normals


def _write_inward_obj(mesh: trimesh.Trimesh, out_path: str | Path) -> None:
    m = mesh.copy()
    m.invert()
    verts = m.vertices
    normals = m.vertex_normals
    faces = m.faces
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write("# inward-facing mesh for raysim\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("\n")
        for i, j, k in faces:
            a, b, c = i + 1, j + 1, k + 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def generate_wall_geometry(
    lumen_stl_path: str | Path,
    centerline_txt_path: str | Path,
    snakes_csv_path: str | Path,
    out_dir: str | Path,
    wall_cfg: WallModelConfig,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    lumen_mesh = trimesh.load_mesh(str(lumen_stl_path), process=True)
    if not isinstance(lumen_mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh from {lumen_stl_path}")
    lumen_mesh = _clean_mesh(lumen_mesh)

    centerline = load_centerline_txt(centerline_txt_path)
    snakes = load_snakes_csv(snakes_csv_path)

    radii = snakes["r_median"]
    if radii.shape[0] != snakes["points"].shape[0]:
        raise RuntimeError("Snakes radius array mismatch.")

    # Map mesh vertices to nearest snake points for local thickness.
    thickness_mm = _vertex_thicknesses(
        vertices=lumen_mesh.vertices,
        centerline_points=snakes["points"],
        centerline_radii_mm=radii,
        cfg=wall_cfg,
    )
    thickness_mm = _laplacian_smooth_scalar_on_mesh(
        thickness_mm, lumen_mesh.faces, wall_cfg.smooth_iterations
    )

    normals = np.asarray(lumen_mesh.vertex_normals, dtype=np.float64)
    normals = _ensure_normals_outward(
        vertices=np.asarray(lumen_mesh.vertices, dtype=np.float64),
        normals=normals,
        centerline_points=snakes["points"],
    )
    outer_vertices = np.asarray(lumen_mesh.vertices, dtype=np.float64) + normals * thickness_mm[:, None]
    outer_mesh = trimesh.Trimesh(vertices=outer_vertices, faces=lumen_mesh.faces.copy(), process=True)
    outer_mesh = _clean_mesh(outer_mesh)

    lumen_obj = out_root / "lumen.obj"
    outer_obj = out_root / "outer.obj"
    lumen_stl = out_root / "lumen.stl"
    outer_stl = out_root / "outer.stl"

    _write_inward_obj(lumen_mesh, lumen_obj)
    _write_inward_obj(outer_mesh, outer_obj)
    lumen_mesh.export(lumen_stl)
    outer_mesh.export(outer_stl)

    manifest = {
        "inputs": {
            "lumen_stl": str(Path(lumen_stl_path).resolve()),
            "centerline_txt": str(Path(centerline_txt_path).resolve()),
            "snakes_csv": str(Path(snakes_csv_path).resolve()),
        },
        "wall_model": {
            "thickness_scale_of_radius": wall_cfg.thickness_scale_of_radius,
            "min_thickness_mm": wall_cfg.min_thickness_mm,
            "max_thickness_mm": wall_cfg.max_thickness_mm,
            "smooth_iterations": wall_cfg.smooth_iterations,
        },
        "stats": {
            "lumen_vertices": int(lumen_mesh.vertices.shape[0]),
            "lumen_faces": int(lumen_mesh.faces.shape[0]),
            "outer_vertices": int(outer_mesh.vertices.shape[0]),
            "outer_faces": int(outer_mesh.faces.shape[0]),
            "lumen_watertight": bool(lumen_mesh.is_watertight),
            "outer_watertight": bool(outer_mesh.is_watertight),
            "thickness_min_mm": float(np.min(thickness_mm)),
            "thickness_mean_mm": float(np.mean(thickness_mm)),
            "thickness_max_mm": float(np.max(thickness_mm)),
        },
        "outputs": {
            "lumen_obj": str(lumen_obj),
            "outer_obj": str(outer_obj),
            "lumen_stl": str(lumen_stl),
            "outer_stl": str(outer_stl),
        },
    }
    write_json(out_root / "geometry_manifest.json", manifest)
    return manifest
