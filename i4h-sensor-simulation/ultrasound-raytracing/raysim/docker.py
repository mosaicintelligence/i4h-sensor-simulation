# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Host-side helpers for running the simulator inside the raysim Docker image.

The aim is to give users a Python API that mirrors how they configure
simulations today (objects, materials, probe, sim params) but dispatches the
actual rendering to a container. Nothing in this module imports the native
``raysim.cuda`` extension, so the host machine does not need a CUDA/OptiX
build — only Docker plus the NVIDIA Container Toolkit at run time.

Typical use::

    from raysim.docker import (
        DockerSimulator, MaterialOverride, MeshObject, Pose,
        ProbeSpec, Scene, SimParamsSpec, SphereObject,
    )

    scene = Scene(background_material="water")
    scene.add(MeshObject(path="mesh/Cylinder_inner.obj", material="vessel_wall"))
    scene.add(MeshObject(path="mesh/Cylinder_outer.obj", material="extravascular"))

    probe = ProbeSpec.ivus(
        pose=Pose(position_mm=(0, 0, 0)),
        frequency_mhz=40.0,
    )
    sim_params = SimParamsSpec(t_far_mm=10.0, b_mode_size=(512, 512))

    sim = DockerSimulator(image="raysim:latest", mounts={"./mesh": "/work/mesh"})
    image = sim.simulate(scene, probe, sim_params, output_dir="./out")

``image`` is a numpy array of shape ``(b_mode_h, b_mode_w)`` (single render)
or ``(num_frames, b_mode_h, b_mode_w)`` (pullback).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy is a hard dep of raysim
    raise ImportError(
        "numpy is required to use raysim.docker. "
        "Install with `pip install numpy` or `pip install raysim[all]`."
    ) from exc

LOG = logging.getLogger("raysim.docker")

__all__ = [
    "DockerSimulator",
    "MaterialOverride",
    "MeshObject",
    "Pose",
    "ProbeSpec",
    "Scene",
    "SimParamsSpec",
    "SphereObject",
    "ensure_image",
]


# ---------------------------------------------------------------------------
# Scene description dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """Probe pose. Positions in mm, rotations in radians (roll, pitch, yaw)."""

    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position_mm": [float(x) for x in self.position_mm],
            "rotation_rad": [float(x) for x in self.rotation_rad],
        }


@dataclass
class SphereObject:
    """Sphere primitive added to the world."""

    center_mm: tuple[float, float, float]
    radius_mm: float
    material: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sphere",
            "center_mm": [float(x) for x in self.center_mm],
            "radius_mm": float(self.radius_mm),
            "material": self.material,
        }


@dataclass
class MeshObject:
    """Mesh primitive added to the world. ``path`` is resolved inside the
    container via the scene's ``mesh_search_paths``."""

    path: str
    material: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "mesh", "path": self.path, "material": self.material}


SceneObject = SphereObject | MeshObject


@dataclass
class MaterialOverride:
    """Override one or more fields of a named material at runtime.

    Mirrors ``raysim.cuda.Materials.update_material`` semantics: only the
    fields you set are forwarded to the C++ side; others stay at the built-in
    default. Used by calibration sweeps that need to tweak speckle strength
    without rebuilding.
    """

    name: str
    impedance_mrayl: float | None = None
    speed_of_sound_m_per_s: float | None = None
    attenuation_db_per_cm_mhz: float | None = None
    mu0: float | None = None
    mu1: float | None = None
    sigma: float | None = None
    specularity: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            k: float(v)
            for k, v in {
                "impedance_mrayl": self.impedance_mrayl,
                "speed_of_sound_m_per_s": self.speed_of_sound_m_per_s,
                "attenuation_db_per_cm_mhz": self.attenuation_db_per_cm_mhz,
                "mu0": self.mu0,
                "mu1": self.mu1,
                "sigma": self.sigma,
                "specularity": self.specularity,
            }.items()
            if v is not None
        }


@dataclass
class Scene:
    """Collection of objects and material overrides that define the world.

    Use the helper ``add()`` to append objects (sphere or mesh). Material
    overrides are appended via ``override_material()``.
    """

    background_material: str = "water"
    objects: list[SceneObject] = field(default_factory=list)
    material_overrides: list[MaterialOverride] = field(default_factory=list)
    mesh_search_paths: list[str] = field(default_factory=lambda: ["mesh"])

    def add(self, obj: SceneObject) -> "Scene":
        self.objects.append(obj)
        return self

    def override_material(self, override: MaterialOverride) -> "Scene":
        self.material_overrides.append(override)
        return self

    def to_dict(self) -> dict[str, Any]:
        overrides_dict: dict[str, dict[str, float]] = {}
        for ov in self.material_overrides:
            overrides_dict[ov.name] = ov.to_dict()
        return {
            "background_material": self.background_material,
            "objects": [o.to_dict() for o in self.objects],
            "materials": overrides_dict,
            "mesh_search_paths": list(self.mesh_search_paths),
        }


@dataclass
class ProbeSpec:
    """Lightweight probe description; serialised verbatim to the scene runner.

    Use the ``ivus()``/``curvilinear()``/``linear_array()``/``phased_array()``
    factory classmethods to construct a sensible default for each probe type,
    then override the fields you care about.
    """

    type: str
    pose: Pose = field(default_factory=Pose)
    frequency_mhz: float = 40.0
    pulse_duration_cycles: float = 2.0
    num_scanlines: int = 256
    elevational_height_mm: float = 0.0
    num_elevational_samples: int = 1
    f_num: float = 1.0
    speed_of_sound_mm_per_us: float = 1.54
    element_radius_mm: float = 0.6
    focal_length_mm: float = 4.0
    width_mm: float = 0.0
    sector_angle_deg: float | None = None

    @classmethod
    def ivus(cls, **overrides: Any) -> "ProbeSpec":
        spec = cls(type="ivus", frequency_mhz=40.0, num_scanlines=256, f_num=1.0,
                   pulse_duration_cycles=2.0, elevational_height_mm=0.0,
                   element_radius_mm=0.6, focal_length_mm=4.0)
        return _apply_overrides(spec, overrides)

    @classmethod
    def curvilinear(cls, **overrides: Any) -> "ProbeSpec":
        spec = cls(type="curvilinear", frequency_mhz=2.5, num_scanlines=256,
                   f_num=0.7, pulse_duration_cycles=2.0,
                   elevational_height_mm=7.0, element_radius_mm=45.0,
                   sector_angle_deg=73.0)
        return _apply_overrides(spec, overrides)

    @classmethod
    def linear_array(cls, **overrides: Any) -> "ProbeSpec":
        spec = cls(type="linear_array", frequency_mhz=5.0, num_scanlines=256,
                   f_num=1.0, pulse_duration_cycles=2.0,
                   elevational_height_mm=5.0, width_mm=60.0)
        return _apply_overrides(spec, overrides)

    @classmethod
    def phased_array(cls, **overrides: Any) -> "ProbeSpec":
        spec = cls(type="phased_array", frequency_mhz=3.5, num_scanlines=128,
                   f_num=1.0, pulse_duration_cycles=2.0,
                   elevational_height_mm=5.0, width_mm=20.0,
                   sector_angle_deg=90.0)
        return _apply_overrides(spec, overrides)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["pose"] = self.pose.to_dict()
        if self.sector_angle_deg is None:
            d.pop("sector_angle_deg", None)
        return d


def _apply_overrides(spec: ProbeSpec, overrides: Mapping[str, Any]) -> ProbeSpec:
    """Field-validated kwargs override that gives nice errors on typos."""
    valid_fields = {f.name for f in dataclasses.fields(spec)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise AttributeError(
                f"Unknown ProbeSpec field {key!r}. Valid fields: {sorted(valid_fields)}"
            )
        setattr(spec, key, value)
    return spec


@dataclass
class SimParamsSpec:
    """Per-render simulator parameters. Maps 1:1 onto raysim.cuda.SimParams.

    Use ``config_yaml`` to start from a calibrated YAML (e.g. the PV .035 /
    Volcano s5i one) and then override individual fields.
    """

    config_yaml: str | None = None
    t_far_mm: float | None = None
    buffer_size: int | None = None
    b_mode_size: tuple[int, int] | None = None
    max_reflection_depth: int | None = None
    min_intensity: float | None = None
    contact_epsilon_mm: float | None = None
    conv_psf: bool | None = None
    median_clip_filter: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.t_far_mm is not None:
            d["t_far_mm"] = float(self.t_far_mm)
        if self.buffer_size is not None:
            d["buffer_size"] = int(self.buffer_size)
        if self.b_mode_size is not None:
            d["b_mode_size"] = [int(self.b_mode_size[0]), int(self.b_mode_size[1])]
        if self.max_reflection_depth is not None:
            d["max_reflection_depth"] = int(self.max_reflection_depth)
        if self.min_intensity is not None:
            d["min_intensity"] = float(self.min_intensity)
        if self.contact_epsilon_mm is not None:
            d["contact_epsilon_mm"] = float(self.contact_epsilon_mm)
        if self.conv_psf is not None:
            d["conv_psf"] = bool(self.conv_psf)
        if self.median_clip_filter is not None:
            d["median_clip_filter"] = bool(self.median_clip_filter)
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Docker orchestration
# ---------------------------------------------------------------------------


class DockerCommandError(RuntimeError):
    """Raised when a docker invocation exits non-zero."""

    def __init__(self, returncode: int, command: list[str], stderr: str = ""):
        self.returncode = returncode
        self.command = command
        self.stderr = stderr
        super().__init__(
            f"docker command exited {returncode}: {' '.join(command)}\n{stderr}".rstrip()
        )


def _docker_executable() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise FileNotFoundError(
            "`docker` not found on PATH. Install Docker Engine and the NVIDIA "
            "Container Toolkit, then retry."
        )
    return docker


def _resolve_mounts(mounts: Mapping[str, str] | None) -> list[tuple[Path, str]]:
    """Validate and normalise host->container mount mappings."""
    resolved: list[tuple[Path, str]] = []
    if not mounts:
        return resolved
    for host, container in mounts.items():
        host_path = Path(host).expanduser().resolve()
        if not host_path.exists():
            raise FileNotFoundError(f"Mount source does not exist: {host_path}")
        if not str(container).startswith("/"):
            raise ValueError(
                f"Container mount target must be an absolute path; got {container!r}"
            )
        resolved.append((host_path, str(container)))
    return resolved


def ensure_image(
    image: str,
    dockerfile_dir: Path | str | None = None,
    dockerfile: Path | str = "docker/Dockerfile",
    build_args: Mapping[str, str] | None = None,
    use_sudo: bool = False,
    pull: bool = False,
) -> None:
    """Ensure the requested image exists locally, building it if necessary.

    Parameters
    ----------
    image:
        Image tag (``repo:tag``).
    dockerfile_dir:
        Build context. Defaults to the ``ultrasound-raytracing`` directory that
        ships with this package.
    dockerfile:
        Path to the Dockerfile, relative to ``dockerfile_dir`` (or absolute).
    build_args:
        Optional ``--build-arg`` mapping (e.g. ``{"CUDA_ARCH": "89"}``).
    use_sudo:
        Prepend ``sudo`` to all docker commands. Required on hosts where the
        invoking user is not in the ``docker`` group.
    pull:
        If True, ``docker pull`` the image instead of building locally.
    """
    docker = _docker_executable()
    prefix = ["sudo", "-n"] if use_sudo else []

    inspect = subprocess.run(
        [*prefix, docker, "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        LOG.info("Image %s already present locally.", image)
        return

    if pull:
        cmd = [*prefix, docker, "pull", image]
        LOG.info("Pulling image: %s", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise DockerCommandError(result.returncode, cmd)
        return

    if dockerfile_dir is None:
        dockerfile_dir = Path(__file__).resolve().parents[1]
    context = Path(dockerfile_dir).resolve()
    df = Path(dockerfile)
    if not df.is_absolute():
        df = context / df
    if not df.is_file():
        raise FileNotFoundError(f"Dockerfile not found: {df}")

    cmd = [*prefix, docker, "build", "-f", str(df), "-t", image]
    for k, v in (build_args or {}).items():
        cmd.extend(["--build-arg", f"{k}={v}"])
    cmd.append(str(context))
    LOG.info("Building image: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise DockerCommandError(result.returncode, cmd)


@dataclass
class DockerSimulator:
    """Run the raysim simulator inside a Docker container.

    Parameters
    ----------
    image:
        Image tag to invoke.
    mounts:
        Optional ``{host_path: container_path}`` map. The scene's work
        directory (where the scene JSON and outputs live) is *always* mounted
        as ``/work``; everything else is auxiliary (mesh assets, calibration
        artefacts, etc.).
    gpus:
        Value passed to ``--gpus``. Defaults to ``"all"``. Set to ``None`` to
        skip the flag (mostly useful in tests / dry-run inspection).
    use_sudo:
        Prepend ``sudo`` to all docker commands (for hosts where the user
        isn't in the ``docker`` group).
    extra_run_args:
        Additional CLI flags appended to ``docker run`` before the image
        name. Use for advanced cases (custom networks, --shm-size, etc.).
    """

    image: str = "raysim:latest"
    mounts: Mapping[str, str] = field(default_factory=dict)
    gpus: str | None = "all"
    use_sudo: bool = False
    extra_run_args: Sequence[str] = field(default_factory=tuple)
    env: Mapping[str, str] = field(default_factory=dict)
    # When True (default), `docker run` is invoked with --user $(uid):$(gid) so
    # outputs written to mounted directories are owned by the host caller. Set
    # to False to keep the container's default (root) user, e.g. if the image
    # requires root for sub-processes.
    run_as_host_user: bool = True

    # ---- public API -------------------------------------------------------

    def simulate(
        self,
        scene: Scene,
        probe: ProbeSpec,
        sim_params: SimParamsSpec,
        *,
        frames: Iterable[Pose] | None = None,
        output_dir: Path | str | None = None,
        output_name: str = "b_mode.npy",
        scene_dir: Path | str | None = None,
        cleanup: bool = True,
    ) -> np.ndarray:
        """Render ``scene`` with ``probe`` / ``sim_params`` in the container.

        Parameters
        ----------
        scene, probe, sim_params:
            Scene description (see module docstring).
        frames:
            Optional iterable of poses for a pullback / sweep. When provided,
            one image is rendered per pose and the returned array has shape
            ``(num_frames, h, w)``.
        output_dir:
            Host directory where the container writes results. Defaults to a
            new temp dir that is cleaned up after the call (set ``cleanup=False``
            to keep it).
        output_name:
            File name for the result inside ``output_dir`` (``.npy``).
        scene_dir:
            Host directory used as the container's ``/work`` mount.
            Defaults to a temp dir; pass an existing path when scene assets
            (meshes, YAML configs) need to be in the same mount.
        cleanup:
            When True (default), temporary scene/output dirs are removed
            after the call.

        Returns
        -------
        numpy.ndarray
            Rendered B-mode image (or stack of frames).
        """
        scene_dict = self._build_scene_dict(scene, probe, sim_params, frames)

        managed_scene_dir = scene_dir is None
        managed_output_dir = output_dir is None
        scene_path_root = Path(scene_dir).resolve() if scene_dir else Path(
            tempfile.mkdtemp(prefix="raysim-scene-")
        )
        output_path_root = (
            Path(output_dir).resolve()
            if output_dir
            else Path(tempfile.mkdtemp(prefix="raysim-out-"))
        )
        output_path_root.mkdir(parents=True, exist_ok=True)
        scene_path_root.mkdir(parents=True, exist_ok=True)

        scene_file = scene_path_root / "scene.json"
        scene_file.write_text(json.dumps(scene_dict, indent=2))
        LOG.info("Wrote scene description to %s", scene_file)

        container_scene = "/work/scene.json"
        container_output = f"/out/{output_name}"

        try:
            self._run_container(scene_path_root, output_path_root, container_scene, container_output)
            output_file = output_path_root / output_name
            if not output_file.is_file():
                raise FileNotFoundError(
                    f"Simulator did not produce {output_file}. Check container logs."
                )
            data = np.load(output_file)
            return data
        finally:
            if cleanup and managed_scene_dir:
                shutil.rmtree(scene_path_root, ignore_errors=True)
            if cleanup and managed_output_dir:
                shutil.rmtree(output_path_root, ignore_errors=True)

    def docker_run_argv(
        self,
        scene_dir: Path | str,
        output_dir: Path | str,
        container_scene: str = "/work/scene.json",
        container_output: str = "/out/b_mode.npy",
    ) -> list[str]:
        """Return the docker CLI invocation without actually running it.

        Useful for dry-runs, debugging, or piping the command into shell
        scripts. The returned list does not include ``sudo`` unless
        ``self.use_sudo`` is set.
        """
        return self._build_run_argv(
            Path(scene_dir).resolve(),
            Path(output_dir).resolve(),
            container_scene,
            container_output,
        )

    # ---- internals --------------------------------------------------------

    def _build_scene_dict(
        self,
        scene: Scene,
        probe: ProbeSpec,
        sim_params: SimParamsSpec,
        frames: Iterable[Pose] | None,
    ) -> dict[str, Any]:
        scene_data = scene.to_dict()
        d: dict[str, Any] = {
            "world": {
                "background_material": scene_data["background_material"],
                "objects": scene_data["objects"],
            },
            "materials": scene_data["materials"],
            "mesh_search_paths": scene_data["mesh_search_paths"],
            "probe": probe.to_dict(),
            "sim_params": sim_params.to_dict(),
        }
        if sim_params.config_yaml:
            d["config_yaml"] = sim_params.config_yaml
        if frames is not None:
            d["frames"] = [
                {"probe_pose": (f.to_dict() if isinstance(f, Pose) else f)}
                for f in frames
            ]
        return d

    def _build_run_argv(
        self,
        scene_dir: Path,
        output_dir: Path,
        container_scene: str,
        container_output: str,
    ) -> list[str]:
        docker = _docker_executable()
        prefix = ["sudo", "-n"] if self.use_sudo else []
        argv = [*prefix, docker, "run", "--rm"]
        if self.gpus:
            argv.extend(["--gpus", self.gpus])
        if self.run_as_host_user:
            argv.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        for key, value in self.env.items():
            argv.extend(["-e", f"{key}={value}"])

        argv.extend(["-v", f"{scene_dir}:/work"])
        argv.extend(["-v", f"{output_dir}:/out"])
        for host, container in _resolve_mounts(self.mounts):
            argv.extend(["-v", f"{host}:{container}"])

        argv.extend(self.extra_run_args)
        argv.append(self.image)
        argv.extend(["--scene", container_scene, "--output", container_output])
        return argv

    def _run_container(
        self,
        scene_dir: Path,
        output_dir: Path,
        container_scene: str,
        container_output: str,
    ) -> None:
        argv = self._build_run_argv(scene_dir, output_dir, container_scene, container_output)
        LOG.info("Running container: %s", " ".join(argv))
        result = subprocess.run(argv)
        if result.returncode != 0:
            raise DockerCommandError(result.returncode, argv)
