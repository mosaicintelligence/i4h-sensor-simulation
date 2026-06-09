from __future__ import annotations

import io
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import raysim.cuda as rs
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image
from raysim import IvusSimConfig

# Import helpers from the existing pullback project to keep geometry generation
# consistent with the current simulation pipeline.
import sys

sys.path.append("/opt/ivus_demo/pullback_src")
from geometry import WallModelConfig, generate_wall_geometry  # noqa: E402
from trajectory import build_pullback_poses  # noqa: E402


@dataclass
class DemoConfig:
    lumen_stl: str
    centerline_txt: str
    snakes_csv: str
    raysim_yaml: str
    geometry_out_dir: str
    world_background_material: str = "lumen"
    lumen_mesh_material: str = "vessel_wall"
    outer_mesh_material: str = "extravascular"
    thickness_scale_of_radius: float = 0.08
    min_thickness_mm: float = 1.0
    max_thickness_mm: float = 3.0
    smooth_iterations: int = 2
    trajectory_step_mm: float = 0.4
    trajectory_max_frames: int = 2


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def load_config() -> DemoConfig:
    return DemoConfig(
        lumen_stl=_env("IVUS_LUMEN_STL", "/opt/ivus_demo/patient_23/mesh/aorta.stl"),
        centerline_txt=_env(
            "IVUS_CENTERLINE_TXT",
            "/opt/ivus_demo/patient_23/centerlines/Aorta_+_Left_Iliac.txt",
        ),
        snakes_csv=_env(
            "IVUS_SNAKES_CSV",
            "/opt/ivus_demo/patient_23/snakes/Aorta_+_Left_Iliac-snakes.csv",
        ),
        raysim_yaml=_env(
            "IVUS_RAYSIM_YAML",
            "/opt/ivus_demo/instrument-calibration/p035_visions/volcano_s5i.yaml",
        ),
        geometry_out_dir=_env("IVUS_GEOMETRY_DIR", "/tmp/ivus_demo/geometry"),
        world_background_material=_env("IVUS_WORLD_BACKGROUND_MATERIAL", "lumen"),
        lumen_mesh_material=_env("IVUS_LUMEN_MESH_MATERIAL", "vessel_wall"),
        outer_mesh_material=_env("IVUS_OUTER_MESH_MATERIAL", "extravascular"),
    )


def _build_probe(cfg: IvusSimConfig, position_mm: list[float], rotation_rad_xyz: list[float]) -> Any:
    pose = rs.Pose(
        np.asarray(position_mm, dtype=np.float32),
        np.asarray(rotation_rad_xyz, dtype=np.float32),
    )
    return rs.IVUSProbe(
        pose,
        int(cfg.probe.num_scanlines),
        float(cfg.probe.frequency_mhz),
        float(cfg.probe.elevational_height_mm),
        int(cfg.probe.num_elevational_samples),
        float(cfg.probe.f_num),
        float(cfg.probe.speed_of_sound_mm_per_us),
        float(cfg.probe.pulse_duration_cycles),
        float(cfg.probe.element_radius_mm),
        float(cfg.probe.focal_length_mm),
    )


def _as_theta_r(frame: np.ndarray, cfg: IvusSimConfig) -> np.ndarray:
    n_theta, n_r = int(cfg.sim.b_mode_size[0]), int(cfg.sim.b_mode_size[1])
    if frame.shape == (n_theta, n_r):
        return frame.astype(np.float32, copy=False)
    if frame.shape == (n_r, n_theta):
        return frame.T.astype(np.float32, copy=False)
    raise RuntimeError(
        f"Unexpected frame shape {frame.shape}; expected {(n_theta, n_r)} or {(n_r, n_theta)}"
    )


class DemoEngine:
    def __init__(self, cfg: DemoConfig) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.engine_meta: dict[str, Any] = {}

        wall_cfg = WallModelConfig(
            thickness_scale_of_radius=cfg.thickness_scale_of_radius,
            min_thickness_mm=cfg.min_thickness_mm,
            max_thickness_mm=cfg.max_thickness_mm,
            smooth_iterations=cfg.smooth_iterations,
        )
        geometry_manifest = generate_wall_geometry(
            lumen_stl_path=cfg.lumen_stl,
            centerline_txt_path=cfg.centerline_txt,
            snakes_csv_path=cfg.snakes_csv,
            out_dir=cfg.geometry_out_dir,
            wall_cfg=wall_cfg,
        )

        pose_payload = build_pullback_poses(
            centerline_txt_path=cfg.centerline_txt,
            snakes_csv_path=cfg.snakes_csv,
            step_mm=cfg.trajectory_step_mm,
            max_frames=cfg.trajectory_max_frames,
        )
        self.default_pose = pose_payload["poses"][0]

        self.raysim_cfg = IvusSimConfig.from_yaml(cfg.raysim_yaml)
        self.sim_params = self.raysim_cfg.to_sim_params()
        self.materials = rs.Materials()
        self.world = rs.World(cfg.world_background_material)
        self.world.add(
            rs.Mesh(
                str(geometry_manifest["outputs"]["lumen_obj"]),
                self.materials.get_index(cfg.lumen_mesh_material),
            )
        )
        self.world.add(
            rs.Mesh(
                str(geometry_manifest["outputs"]["outer_obj"]),
                self.materials.get_index(cfg.outer_mesh_material),
            )
        )
        self.sim = rs.RaytracingUltrasoundSimulator(self.world, self.materials)
        self.engine_meta = {
            "config": asdict(cfg),
            "geometry_manifest": geometry_manifest,
            "default_pose": self.default_pose,
            "b_mode_size": [int(v) for v in self.raysim_cfg.sim.b_mode_size],
            "t_far_mm": float(self.raysim_cfg.sim.t_far_mm),
        }

        # Warm-up once so the first external request does not pay startup cost.
        self.simulate(
            self.default_pose["position_mm"],
            self.default_pose["rotation_rad_xyz"],
        )

    def simulate(self, position_mm: list[float], rotation_rad_xyz: list[float]) -> np.ndarray:
        probe = _build_probe(self.raysim_cfg, position_mm, rotation_rad_xyz)
        with self.lock:
            frame = np.asarray(self.sim.simulate(probe, self.sim_params), dtype=np.float32)
        return _as_theta_r(frame, self.raysim_cfg)


def _to_uint8_percentile(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanpercentile(arr, 1.0))
    hi = float(np.nanpercentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _polar_to_cart_display(arr_theta_r: np.ndarray, t_far_mm: float, cart_size: int = 512) -> np.ndarray:
    n_theta, n_r = arr_theta_r.shape
    half = float(t_far_mm)
    xs = np.linspace(-half, half, cart_size)
    ys = np.linspace(-half, half, cart_size)
    x, y = np.meshgrid(xs, ys)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(x, y) % (2.0 * np.pi)  # 0 at north, clockwise
    r_idx = (r / half) * (n_r - 1)
    t_idx = (theta / (2.0 * np.pi)) * n_theta
    r_idx = np.clip(r_idx, 0.0, n_r - 1.001)
    t_idx = t_idx % n_theta

    r0 = np.floor(r_idx).astype(np.int64)
    r1 = r0 + 1
    t0 = np.floor(t_idx).astype(np.int64) % n_theta
    t1 = (t0 + 1) % n_theta

    wr = r_idx - r0
    wt = t_idx - np.floor(t_idx)
    out = (
        arr_theta_r[t0, r0] * (1.0 - wt) * (1.0 - wr)
        + arr_theta_r[t1, r0] * wt * (1.0 - wr)
        + arr_theta_r[t0, r1] * (1.0 - wt) * wr
        + arr_theta_r[t1, r1] * wt * wr
    )
    out[r > half] = 0.0
    return out.astype(np.float32)


def _png_from_array_uint8(gray: np.ndarray) -> bytes:
    image = Image.fromarray(gray)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


app = Flask(__name__)
cfg = load_config()
engine = DemoEngine(cfg)


@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "ok"})


@app.route("/metadata", methods=["GET"])
def metadata() -> Response:
    return jsonify(engine.engine_meta)


@app.route("/simulate", methods=["POST"])
def simulate() -> Response:
    payload = request.get_json(force=True, silent=False) or {}
    position_mm = payload.get("position_mm", engine.default_pose["position_mm"])
    rotation_rad_xyz = payload.get("rotation_rad_xyz", engine.default_pose["rotation_rad_xyz"])
    output = payload.get("output", "png")

    frame = engine.simulate(position_mm, rotation_rad_xyz)

    if output == "npy":
        out = io.BytesIO()
        np.save(out, frame)
        out.seek(0)
        return send_file(out, mimetype="application/octet-stream", download_name="frame.npy")
    if output == "json":
        return jsonify({"frame": frame.tolist(), "shape": list(frame.shape)})
    if output == "png":
        # Default display mode is circular/cartesian IVUS view.
        cart = _polar_to_cart_display(frame, t_far_mm=float(engine.engine_meta["t_far_mm"]), cart_size=512)
        png_bytes = _png_from_array_uint8(_to_uint8_percentile(cart))
        return Response(png_bytes, mimetype="image/png")
    if output == "png_polar":
        png_bytes = _png_from_array_uint8(_to_uint8_percentile(frame))
        return Response(png_bytes, mimetype="image/png")

    return jsonify({"error": "Invalid output. Use one of: png, png_polar, npy, json"}), 400


@app.route("/startup_state", methods=["GET"])
def startup_state() -> Response:
    return Response(json.dumps(engine.engine_meta, indent=2), mimetype="application/json")


if __name__ == "__main__":
    host = os.getenv("IVUS_DEMO_HOST", "0.0.0.0")
    port = int(os.getenv("IVUS_DEMO_PORT", "8000"))
    app.run(host=host, port=port, debug=False)
