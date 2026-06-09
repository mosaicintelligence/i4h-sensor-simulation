from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import ensure_dir, load_yaml, write_json
from export_dicom import write_multiframe_dicom
from export_mp4 import write_polar_mp4
from export_story_views import write_story_views
from geometry import WallModelConfig, generate_wall_geometry
from sim_runner import run_pullback_simulation
from trajectory import build_pullback_poses, write_pose_csv, write_pose_json


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _validate_geometry(geom_manifest: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    stats = geom_manifest["stats"]
    checks.append(f"lumen_watertight={stats['lumen_watertight']}")
    checks.append(f"outer_watertight={stats['outer_watertight']}")
    checks.append(
        f"thickness_mm[min/mean/max]={stats['thickness_min_mm']:.3f}/"
        f"{stats['thickness_mean_mm']:.3f}/{stats['thickness_max_mm']:.3f}"
    )
    return checks


def _validate_trajectory(poses_payload: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    poses = poses_payload["poses"]
    checks.append(f"n_frames={poses_payload['n_frames']}")
    checks.append(f"distance_total_mm={poses_payload['distance_total_mm']:.3f}")
    if len(poses) >= 2:
        t0 = np.asarray(poses[0]["tangent"], dtype=np.float64)
        t1 = np.asarray(poses[-1]["tangent"], dtype=np.float64)
        checks.append(f"endpoint_tangent_dot={float(np.dot(t0, t1)):.6f}")
    return checks


def _validate_imaging(stack: np.ndarray, dicom_meta: dict[str, Any], fps: int) -> list[str]:
    checks: list[str] = []
    checks.append(f"polar_stack_shape={tuple(int(v) for v in stack.shape)}")
    checks.append(f"stack_value_range={float(np.min(stack)):.3f}..{float(np.max(stack)):.3f}")
    checks.append(f"dicom_frames={dicom_meta['n_frames']}")
    checks.append(f"fps={int(fps)}")
    return checks


def _write_report(
    out_path: Path,
    cfg: dict[str, Any],
    geom_manifest: dict[str, Any],
    poses_payload: dict[str, Any],
    sim_meta: dict[str, Any],
    dicom_meta: dict[str, Any],
    files: dict[str, str],
) -> None:
    geom_checks = _validate_geometry(geom_manifest)
    traj_checks = _validate_trajectory(poses_payload)
    img_checks = _validate_imaging(np.load(files["polar_stack_npy"]), dicom_meta, cfg["export"]["fps"])

    lines = [
        "# PAT23 Simulated IVUS Pullback Report",
        "",
        "## Inputs",
        f"- lumen STL: `{cfg['paths']['lumen_stl']}`",
        f"- centerline: `{cfg['paths']['centerline_txt']}`",
        f"- snakes CSV: `{cfg['paths']['snakes_csv']}`",
        f"- raysim YAML: `{cfg['paths']['raysim_yaml']}`",
        "",
        "## Geometry QA",
    ]
    lines.extend([f"- {x}" for x in geom_checks])
    lines.extend(
        [
            "",
            "## Trajectory QA",
            *[f"- {x}" for x in traj_checks],
            "",
            "## Imaging QA",
            *[f"- {x}" for x in img_checks],
            "",
            "## Outputs",
            f"- geometry manifest: `{files['geometry_manifest']}`",
            f"- poses CSV: `{files['poses_csv']}`",
            f"- poses JSON: `{files['poses_json']}`",
            f"- polar stack: `{files['polar_stack_npy']}`",
            f"- simulation manifest: `{files['simulation_manifest']}`",
            f"- DICOM: `{files['dicom']}`",
            f"- IVUS MP4: `{files['mp4']}`",
            f"- Camera-eye MP4: `{files['camera_eye_mp4']}`",
            f"- God's-eye MP4: `{files['gods_eye_mp4']}`",
            "",
            "## Provenance",
            f"- config sha256: `{_sha256(files['config'])}`",
            f"- geometry manifest sha256: `{_sha256(files['geometry_manifest'])}`",
            f"- simulation manifest sha256: `{_sha256(files['simulation_manifest'])}`",
            "",
            "## Notes",
            "- DICOM is a derived synthetic US multi-frame object with IVUS-oriented metadata.",
            "- Polar MP4 is rendered in cartesian display orientation (0 deg at top, clockwise).",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    out_root = ensure_dir(cfg["paths"]["output_root"])
    geometry_dir = ensure_dir(out_root / "geometry")
    trajectory_dir = ensure_dir(out_root / "trajectory")
    render_dir = ensure_dir(out_root / "render")
    export_dir = ensure_dir(out_root / "exports")

    wall_cfg = WallModelConfig(**cfg["wall_model"])
    geom_manifest = generate_wall_geometry(
        lumen_stl_path=cfg["paths"]["lumen_stl"],
        centerline_txt_path=cfg["paths"]["centerline_txt"],
        snakes_csv_path=cfg["paths"]["snakes_csv"],
        out_dir=geometry_dir,
        wall_cfg=wall_cfg,
    )

    poses_payload = build_pullback_poses(
        centerline_txt_path=cfg["paths"]["centerline_txt"],
        snakes_csv_path=cfg["paths"]["snakes_csv"],
        step_mm=float(cfg["trajectory"]["step_mm"]),
        max_frames=int(cfg["trajectory"]["max_frames"]),
    )

    poses_csv = trajectory_dir / cfg["export"]["pose_csv_filename"]
    poses_json = trajectory_dir / cfg["export"]["pose_json_filename"]
    write_pose_csv(poses_payload, poses_csv)
    write_pose_json(poses_payload, poses_json)

    sim_result = run_pullback_simulation(
        raysim_yaml_path=cfg["paths"]["raysim_yaml"],
        lumen_mesh_obj=geom_manifest["outputs"]["lumen_obj"],
        outer_mesh_obj=geom_manifest["outputs"]["outer_obj"],
        poses_payload=poses_payload,
        out_dir=render_dir,
        world_background_material=cfg["simulation"]["world_background_material"],
        lumen_mesh_material=cfg["simulation"]["lumen_mesh_material"],
        outer_mesh_material=cfg["simulation"]["outer_mesh_material"],
        material_overrides=cfg["simulation"].get("material_overrides", {}),
        show_progress=bool(cfg["simulation"]["show_progress"]),
    )
    stack = sim_result["stack"]
    sim_meta = sim_result["sim_meta"]

    dicom_path = export_dir / cfg["export"]["dicom_filename"]
    dicom_meta = write_multiframe_dicom(
        polar_stack=stack,
        t_far_mm=float(sim_meta["t_far_mm"]),
        fps=int(cfg["export"]["fps"]),
        out_path=dicom_path,
        dicom_cfg=cfg["dicom"],
    )
    mp4_path = export_dir / cfg["export"]["mp4_filename"]
    write_polar_mp4(
        polar_stack=stack,
        t_far_mm=float(sim_meta["t_far_mm"]),
        fps=int(cfg["export"]["fps"]),
        out_path=mp4_path,
        cart_size=512,
    )
    camera_eye_mp4_path = export_dir / cfg["export"]["camera_eye_mp4_filename"]
    gods_eye_mp4_path = export_dir / cfg["export"]["gods_eye_mp4_filename"]
    write_story_views(
        mesh_path=cfg["paths"]["lumen_stl"],
        poses_payload=poses_payload,
        fps=int(cfg["export"]["fps"]),
        camera_out_path=camera_eye_mp4_path,
        god_out_path=gods_eye_mp4_path,
    )

    pipeline_manifest = {
        "config_path": str(Path(config_path).resolve()),
        "outputs": {
            "geometry_manifest": str(Path(geometry_dir / "geometry_manifest.json")),
            "poses_csv": str(poses_csv),
            "poses_json": str(poses_json),
            "polar_stack_npy": str(render_dir / "polar_stack.npy"),
            "simulation_manifest": str(render_dir / "simulation_manifest.json"),
            "dicom": str(dicom_path),
            "mp4": str(mp4_path),
            "camera_eye_mp4": str(camera_eye_mp4_path),
            "gods_eye_mp4": str(gods_eye_mp4_path),
        },
        "sim": sim_meta,
        "dicom": dicom_meta,
    }
    write_json(out_root / "pipeline_manifest.json", pipeline_manifest)
    _write_report(
        out_path=out_root / cfg["export"]["run_report_filename"],
        cfg=cfg,
        geom_manifest=geom_manifest,
        poses_payload=poses_payload,
        sim_meta=sim_meta,
        dicom_meta=dicom_meta,
        files={
            "config": str(Path(config_path)),
            "geometry_manifest": str(geometry_dir / "geometry_manifest.json"),
            "poses_csv": str(poses_csv),
            "poses_json": str(poses_json),
            "polar_stack_npy": str(render_dir / "polar_stack.npy"),
            "simulation_manifest": str(render_dir / "simulation_manifest.json"),
            "dicom": str(dicom_path),
            "mp4": str(mp4_path),
            "camera_eye_mp4": str(camera_eye_mp4_path),
            "gods_eye_mp4": str(gods_eye_mp4_path),
        },
    )
    return pipeline_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAT23 CTA-to-simulated-IVUS pullback pipeline.")
    p.add_argument(
        "--config",
        required=True,
        help="Path to pipeline YAML config.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_pipeline(args.config)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
