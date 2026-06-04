# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Scene runner used as the default entrypoint of the raysim Docker image.

Reads a scene JSON description, builds the corresponding native `raysim`
objects (World, Materials, probe, SimParams), runs the simulation, and writes
the resulting B-mode array to a `.npy` file. Optionally renders pullback /
batch sweeps when ``frames`` is provided.

Scene JSON schema (all fields optional unless marked required)::

    {
      "world": {
        "background_material": "water",         # default: "water"
        "objects": [                            # default: []
          {"type": "sphere",
           "center_mm": [x, y, z],
           "radius_mm": r,
           "material": "vessel_wall"},
          {"type": "mesh",
           "path": "mesh/Cylinder.obj",         # resolved via mesh_search_paths
           "material": "vessel_wall"}
        ]
      },
      "materials": {                            # optional per-material overrides
        "milk": {"impedance_mrayl": 1.5,
                 "speed_of_sound_m_per_s": 1505,
                 "attenuation_db_per_cm_mhz": 0.5,
                 "mu0": 0.6, "mu1": 0.0,
                 "sigma": 0.0, "specularity": 0.0}
      },
      "probe": {                                # REQUIRED
        "type": "ivus",                         # ivus|curvilinear|linear_array|phased_array
        "frequency_mhz": 40.0,
        "pulse_duration_cycles": 2.0,
        "num_scanlines": 256,
        "f_num": 1.0,
        "elevational_height_mm": 0.0,
        "num_elevational_samples": 1,
        "speed_of_sound_mm_per_us": 1.54,
        "element_radius_mm": 0.6,
        "focal_length_mm": 4.0,
        "width_mm": 0.0,
        "pose": {"position_mm": [0,0,0],
                 "rotation_rad": [0,0,0]}
      },
      "sim_params": {                           # OR "config_yaml" below
        "t_far_mm": 10.0,
        "buffer_size": 4096,
        "b_mode_size": [512, 512],
        ...
      },
      "config_yaml": "configs/volcano_s5i.yaml", # optional: load IvusSimConfig
      "mesh_search_paths": ["mesh", "/work/mesh"], # default: ["mesh"]
      "frames": [                                # optional, render one image per entry
        {"probe_pose": {"position_mm": [0,0,-1.5],
                        "rotation_rad": [0,0,0]}},
        ...
      ],
      "output": {
        "format": "npy",                        # npy|npz; default npy
        "frames_layout": "stack"                # stack|files; default stack
      }
    }

When ``frames`` is omitted a single render is produced from ``probe.pose``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Try YAML lazily; only required when --scene is YAML or config_yaml is set.
try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - yaml is installed via [all] extras
    yaml = None  # type: ignore[assignment]

import raysim.cuda as rs
from raysim.config import IvusSimConfig

LOG = logging.getLogger("raysim.scene_runner")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _load_scene(scene_path: Path) -> dict[str, Any]:
    text = scene_path.read_text()
    suffix = scene_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML scenes.")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _resolve_mesh(path_str: str, search_paths: Iterable[Path]) -> Path:
    """Resolve a possibly-relative mesh path against the configured search paths."""
    p = Path(path_str)
    if p.is_absolute() and p.is_file():
        return p
    for root in search_paths:
        candidate = (root / p).resolve()
        if candidate.is_file():
            return candidate
    bundled = os.environ.get("RAYSIM_BUNDLED_MESHES")
    if bundled:
        candidate = (Path(bundled) / p).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Mesh not found: {path_str!r}. Searched: "
        + ", ".join(str(s) for s in search_paths)
        + (f", {bundled}" if bundled else "")
    )


def _apply_material_overrides(materials: rs.Materials, overrides: dict[str, dict[str, Any]]) -> None:
    """Apply per-material kwargs to the simulator's Materials table.

    Each entry is ``name -> {impedance_mrayl, speed_of_sound_m_per_s,
    attenuation_db_per_cm_mhz, mu0, mu1, sigma, specularity}``. Missing keys
    leave the corresponding field unchanged.
    """
    field_map = {
        "impedance_mrayl": "impedance",
        "impedance": "impedance",
        "speed_of_sound_m_per_s": "speed_of_sound",
        "speed_of_sound": "speed_of_sound",
        "attenuation_db_per_cm_mhz": "attenuation",
        "attenuation": "attenuation",
        "mu0": "mu0",
        "mu1": "mu1",
        "sigma": "sigma",
        "specularity": "specularity",
    }
    for name, kwargs in overrides.items():
        cpp_kwargs: dict[str, float] = {}
        for k, v in kwargs.items():
            mapped = field_map.get(k)
            if mapped is None:
                raise ValueError(
                    f"Unknown material field {k!r} for material {name!r}; "
                    f"expected one of {sorted(set(field_map))}."
                )
            cpp_kwargs[mapped] = float(v)
        LOG.info("Overriding material %s with %s", name, cpp_kwargs)
        materials.update_material(name, **cpp_kwargs)


def _build_world(
    spec: dict[str, Any],
    materials: rs.Materials,
    mesh_search_paths: list[Path],
) -> rs.World:
    background = spec.get("background_material", "water")
    world = rs.World(background)
    for obj in spec.get("objects", []):
        mat_name = obj.get("material")
        if mat_name is None:
            raise ValueError(f"Object missing 'material': {obj!r}")
        mat_id = materials.get_index(mat_name)
        kind = obj.get("type", "").lower()
        if kind == "sphere":
            center = np.asarray(obj["center_mm"], dtype=np.float32)
            if center.shape != (3,):
                raise ValueError(f"sphere.center_mm must be length-3; got shape {center.shape}")
            radius = float(obj["radius_mm"])
            world.add(rs.Sphere(center, radius, mat_id))
        elif kind == "mesh":
            mesh_path = _resolve_mesh(obj["path"], mesh_search_paths)
            world.add(rs.Mesh(str(mesh_path), mat_id))
        else:
            raise ValueError(f"Unknown object type {kind!r}; expected 'sphere' or 'mesh'.")
    return world


def _make_probe(spec: dict[str, Any]) -> rs.BaseProbe:
    kind = spec.get("type", "ivus").lower()
    pose_spec = spec.get("pose", {})
    pose = rs.Pose(
        np.asarray(pose_spec.get("position_mm", [0.0, 0.0, 0.0]), dtype=np.float32),
        np.asarray(pose_spec.get("rotation_rad", [0.0, 0.0, 0.0]), dtype=np.float32),
    )
    common = dict(
        num_scanlines=int(spec.get("num_scanlines", 256)),
        frequency=float(spec.get("frequency_mhz", 40.0)),
        elevational_height=float(spec.get("elevational_height_mm", 0.0)),
        num_el_samples=int(spec.get("num_elevational_samples", 1)),
        f_num=float(spec.get("f_num", 1.0)),
        speed_of_sound=float(spec.get("speed_of_sound_mm_per_us", 1.54)),
        pulse_duration=float(spec.get("pulse_duration_cycles", 2.0)),
    )
    if kind == "ivus":
        return rs.IVUSProbe(
            pose,
            num_angular_rays=common["num_scanlines"],
            frequency=common["frequency"],
            elevational_height=common["elevational_height"],
            num_el_samples=common["num_el_samples"],
            f_num=common["f_num"],
            speed_of_sound=common["speed_of_sound"],
            pulse_duration=common["pulse_duration"],
            element_radius_mm=float(spec.get("element_radius_mm", 0.6)),
            focal_length_mm=float(spec.get("focal_length_mm", 4.0)),
        )
    if kind == "curvilinear":
        return rs.CurvilinearProbe(
            pose,
            num_elements_x=common["num_scanlines"],
            sector_angle=float(spec.get("sector_angle_deg", 73.0)),
            radius=float(spec.get("element_radius_mm", 45.0)),
            frequency=common["frequency"],
            elevational_height=common["elevational_height"],
            num_el_samples=common["num_el_samples"],
            f_num=common["f_num"],
            speed_of_sound=common["speed_of_sound"],
            pulse_duration=common["pulse_duration"],
        )
    if kind == "linear_array":
        return rs.LinearArrayProbe(
            pose,
            num_elements_x=common["num_scanlines"],
            width=float(spec.get("width_mm", 60.0)),
            frequency=common["frequency"],
            elevational_height=common["elevational_height"],
            num_el_samples=common["num_el_samples"],
            f_num=common["f_num"],
            speed_of_sound=common["speed_of_sound"],
            pulse_duration=common["pulse_duration"],
        )
    if kind == "phased_array":
        return rs.PhasedArrayProbe(
            pose,
            num_elements_x=common["num_scanlines"],
            width=float(spec.get("width_mm", 20.0)),
            sector_angle=float(spec.get("sector_angle_deg", 90.0)),
            frequency=common["frequency"],
            elevational_height=common["elevational_height"],
            num_el_samples=common["num_el_samples"],
            f_num=common["f_num"],
            speed_of_sound=common["speed_of_sound"],
            pulse_duration=common["pulse_duration"],
        )
    raise ValueError(f"Unknown probe.type {kind!r}.")


def _apply_sim_params(params: rs.SimParams, overrides: dict[str, Any]) -> None:
    """Apply a flat dict of overrides onto an existing SimParams.

    Only fields with a public setter on the pybind11 SimParams class are
    accepted. Unknown keys raise to surface typos quickly. Common-case knobs
    like ``b_mode_size`` are coerced to the tuple type the binding expects.
    """
    aliases = {
        "t_far_mm": "t_far",
        "max_reflection_depth": "max_depth",
        "contact_epsilon_mm": "contact_epsilon",
        "scattering_resolution_mm": "scattering_resolution_mm",
        "catheter_dead_zone_mm": "catheter_dead_zone_mm",
        "noise_sigma": "noise_sigma",
    }
    for key, value in overrides.items():
        attr = aliases.get(key, key)
        if not hasattr(params, attr):
            raise AttributeError(f"SimParams has no field {key!r} (mapped to {attr!r}).")
        if attr == "b_mode_size":
            value = (int(value[0]), int(value[1]))
        setattr(params, attr, value)


def _build_sim_params(scene: dict[str, Any]) -> tuple[rs.SimParams, rs.BaseProbe | None]:
    """Materialise SimParams from the scene description.

    Resolution order:
        1. If ``config_yaml`` is set, load it through IvusSimConfig and start
           from its SimParams (probe also returned so the caller can opt to
           use it).
        2. Otherwise start from defaults.
        3. Overlay any inline ``sim_params`` dict.
    """
    base_params: rs.SimParams
    probe: rs.BaseProbe | None = None
    config_yaml = scene.get("config_yaml")
    if config_yaml:
        cfg = IvusSimConfig.from_yaml(config_yaml)
        base_params = cfg.to_sim_params()
        probe = cfg.to_probe()
        LOG.info("Loaded SimParams + probe defaults from %s", config_yaml)
    else:
        base_params = rs.SimParams()
        base_params.conv_psf = True
        base_params.buffer_size = 4096
        base_params.t_far = 10.0
        base_params.b_mode_size = (512, 512)

    inline = scene.get("sim_params") or {}
    if inline:
        _apply_sim_params(base_params, inline)
    return base_params, probe


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_one(
    simulator: rs.RaytracingUltrasoundSimulator,
    probe: rs.BaseProbe,
    sim_params: rs.SimParams,
    pose_override: dict[str, Any] | None,
) -> np.ndarray:
    if pose_override:
        new_pose = rs.Pose(
            np.asarray(pose_override.get("position_mm", [0.0, 0.0, 0.0]), dtype=np.float32),
            np.asarray(pose_override.get("rotation_rad", [0.0, 0.0, 0.0]), dtype=np.float32),
        )
        probe.set_pose(new_pose)
    return simulator.simulate(probe, sim_params)


def _write_outputs(
    images: list[np.ndarray],
    simulator: rs.RaytracingUltrasoundSimulator,
    output_path: Path,
    output_format: str,
    layout: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "min_x": float(simulator.get_min_x()),
        "max_x": float(simulator.get_max_x()),
        "min_z": float(simulator.get_min_z()),
        "max_z": float(simulator.get_max_z()),
        "num_frames": len(images),
    }

    if output_format == "npy":
        if layout == "files" or len(images) == 1:
            for idx, img in enumerate(images):
                target = output_path if len(images) == 1 else (
                    output_path.with_name(f"{output_path.stem}_{idx:04d}{output_path.suffix}")
                )
                np.save(target, img)
        else:  # stack
            np.save(output_path, np.stack(images, axis=0))
    elif output_format == "npz":
        np.savez_compressed(
            output_path,
            images=np.stack(images, axis=0),
            **{k: np.asarray(v) for k, v in metadata.items()},
        )
    else:
        raise ValueError(f"Unsupported output.format {output_format!r}.")

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    LOG.info("Wrote %d frame(s) to %s (metadata: %s)", len(images), output_path, meta_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raysim-run-scene",
        description="Run a raysim scene description inside the container.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Path to scene JSON or YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output array (.npy or .npz).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("RAYSIM_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )

    scene = _load_scene(args.scene)
    scene_dir = args.scene.resolve().parent
    mesh_search_paths = [Path(p) for p in scene.get("mesh_search_paths", ["mesh"])]
    mesh_search_paths = [(scene_dir / p).resolve() if not p.is_absolute() else p
                         for p in mesh_search_paths]

    materials = rs.Materials()
    material_overrides = scene.get("materials") or {}
    if material_overrides:
        _apply_material_overrides(materials, material_overrides)

    world = _build_world(scene.get("world", {}), materials, mesh_search_paths)

    sim_params, probe_from_yaml = _build_sim_params(scene)

    probe_spec = scene.get("probe")
    if probe_spec is not None:
        probe = _make_probe(probe_spec)
    elif probe_from_yaml is not None:
        probe = probe_from_yaml
    else:
        raise ValueError(
            "Scene must provide either a 'probe' block or a 'config_yaml' with one."
        )

    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    frames = scene.get("frames")
    output_block = scene.get("output") or {}
    output_format = output_block.get("format", args.output.suffix.lstrip(".") or "npy").lower()
    layout = output_block.get("frames_layout", "stack").lower()

    images: list[np.ndarray] = []
    t0 = time.perf_counter()
    if frames:
        for idx, frame in enumerate(frames):
            img = _render_one(simulator, probe, sim_params, frame.get("probe_pose"))
            images.append(np.asarray(img, dtype=np.float32))
            LOG.info("Rendered frame %d/%d", idx + 1, len(frames))
    else:
        img = _render_one(simulator, probe, sim_params, None)
        images.append(np.asarray(img, dtype=np.float32))
    dt = time.perf_counter() - t0
    LOG.info("Rendered %d frame(s) in %.2f s (%.2f fps).", len(images), dt, len(images) / max(dt, 1e-6))

    _write_outputs(images, simulator, args.output, output_format, layout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
