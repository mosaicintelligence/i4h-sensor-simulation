from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def _import_raysim():
    try:
        import raysim.cuda as rs
    except ImportError as exc:
        raise ImportError(
            "raysim is required. Install with: "
            "pip install -e ./i4h-sensor-simulation/ultrasound-raytracing"
        ) from exc
    return rs


def _import_config():
    try:
        from raysim import IvusSimConfig
    except ImportError as exc:
        raise ImportError("IvusSimConfig unavailable from raysim installation.") from exc
    return IvusSimConfig


def _build_probe(rs, cfg, position_mm: list[float], rotation_rad_xyz: list[float]):
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


def _as_theta_r(frame: np.ndarray, cfg) -> np.ndarray:
    n_theta, n_r = int(cfg.sim.b_mode_size[0]), int(cfg.sim.b_mode_size[1])
    if frame.shape == (n_theta, n_r):
        return frame.astype(np.float32, copy=False)
    if frame.shape == (n_r, n_theta):
        return frame.T.astype(np.float32, copy=False)
    raise RuntimeError(
        f"Unexpected frame shape {frame.shape}; expected {(n_theta, n_r)} or {(n_r, n_theta)}"
    )


def run_pullback_simulation(
    raysim_yaml_path: str | Path,
    lumen_mesh_obj: str | Path,
    outer_mesh_obj: str | Path,
    poses_payload: dict[str, Any],
    out_dir: str | Path,
    world_background_material: str,
    lumen_mesh_material: str,
    outer_mesh_material: str,
    material_overrides: dict[str, dict[str, float]] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    rs = _import_raysim()
    IvusSimConfig = _import_config()

    cfg = IvusSimConfig.from_yaml(str(raysim_yaml_path))
    sim_params = cfg.to_sim_params()
    materials = rs.Materials()
    if material_overrides:
        for mat_name, overrides in material_overrides.items():
            if not overrides:
                continue
            kwargs = {
                k: float(v)
                for k, v in overrides.items()
                if k
                in {
                    "impedance",
                    "attenuation",
                    "speed_of_sound",
                    "mu0",
                    "mu1",
                    "sigma",
                    "specularity",
                }
                and v is not None
            }
            if kwargs:
                materials.update_material(str(mat_name), **kwargs)
    world = rs.World(str(world_background_material))
    world.add(rs.Mesh(str(lumen_mesh_obj), materials.get_index(str(lumen_mesh_material))))
    world.add(rs.Mesh(str(outer_mesh_obj), materials.get_index(str(outer_mesh_material))))
    sim = rs.RaytracingUltrasoundSimulator(world, materials)

    frames: list[np.ndarray] = []
    iterator = poses_payload["poses"]
    if show_progress:
        iterator = tqdm(iterator, total=len(poses_payload["poses"]), desc="Simulating pullback")
    for rec in iterator:
        probe = _build_probe(rs, cfg, rec["position_mm"], rec["rotation_rad_xyz"])
        b_mode = np.asarray(sim.simulate(probe, sim_params), dtype=np.float32)
        frames.append(_as_theta_r(b_mode, cfg))

    stack = np.stack(frames, axis=0).astype(np.float32)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "polar_stack.npy", stack)

    sim_meta = {
        "raysim_yaml": str(Path(raysim_yaml_path).resolve()),
        "lumen_mesh_obj": str(Path(lumen_mesh_obj).resolve()),
        "outer_mesh_obj": str(Path(outer_mesh_obj).resolve()),
        "n_frames": int(stack.shape[0]),
        "n_theta": int(stack.shape[1]),
        "n_r": int(stack.shape[2]),
        "t_far_mm": float(cfg.sim.t_far_mm),
        "b_mode_size": [int(v) for v in cfg.sim.b_mode_size],
        "material_overrides": material_overrides or {},
    }
    with (out_root / "simulation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(sim_meta, f, indent=2)
    return {"stack": stack, "sim_meta": sim_meta}
