# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mosaic catheter simulation — catheter insertion through a vessel mesh.

XPBD Cosserat elastic-rod solver (from xcath) with mesh-query vessel containment,
track-guided insertion, a bendable steerable tip and root rotation. Loads any
STL / OBJ / PLY / glTF (or USD) mesh as the vessel; auto-fits its scale,
auto-detects the containment winding, and lines an open end up to the rod tip so
the catheter starts threaded inside.

    uv run mosaic_sim/mosaic_sim.py                                      # default: stls/pig_cta.stl
    uv run mosaic_sim/mosaic_sim.py --mesh mosaic_sim/stls/leg_to_leg__simple.stl
    uv run mosaic_sim/mosaic_sim.py --mesh vessel.stl --mesh-scale 0.001 --stiffness 5e6

Keys: I/K = insert/retract, J/L = rotate root, +/- = bend tip, G = gravity,
      W = vessel wireframe, E = cycle entry opening, V = vessel-align mode.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.usd
from newton.solvers import xpbd_rod

# Rod solver + mesh-query vessel-containment kernels (vendored alongside this
# file so the folder is portable; only needs the `newton` package installed).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xcath_solver import (  # noqa: E402
    XCathRodSolver,
    compute_signed_distances,
    compute_smooth_vertex_normals,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_POINTS = 128
SEGMENT_LENGTH = 0.05
ROD_LENGTH = (NUM_POINTS - 1) * SEGMENT_LENGTH  # catheter length [m]
PARTICLE_MASS = 1.0
PARTICLE_RADIUS = 0.02
SUBSTEPS = 8
COLLISION_ITERATIONS = 1
TIP_NUM_EDGES = 16
LINEAR_DAMPING = 0.01
ANGULAR_DAMPING = 0.01
DAMPING_SLIDER_MAX = 0.2

# Custom meshes with no --mesh-scale are auto-fit so their bounding-box diagonal
# is this fraction of the catheter length (units-agnostic STL loading).
AUTOFIT_VESSEL_FRACTION = 0.9

# When aligning an opening to the rod tip, start the tip this far *inside* the
# lumen (sim units; ~1 cm given the auto-fit scale) so it begins threaded in
# rather than catching on the opening rim.
START_INSIDE_DEPTH = 0.2

# Virtual IVUS sensor location: this far proximal to (behind) the wire tip,
# measured in the CTA mesh's own units (mm for a typical STL).
SENSOR_BEHIND_TIP_MM = 5.0

# Default vessel mesh used when no --mesh is given.
DEFAULT_MESH_PATH = os.path.join(os.path.dirname(__file__), "stls", "pig_cta.stl")

# Per-mesh starting poses (filename -> (offset, rotation)) with the femoral entry
# lined up to the catheter track. Override per run with --mesh-offset /
# --mesh-rotation, or re-tune live in vessel-align mode (press V) and paste the
# printed values here. Falls back to DEFAULT_VESSEL_* for an unlisted mesh.
VESSEL_POSES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "Syndaver_shell.stl": (
        np.array([13.887, 0.018, 0.156], dtype=np.float32),
        np.array([-0.023, 0.159, -1.480], dtype=np.float32),
    ),
    "leg_to_leg__simple.stl": (
        np.array([-4.502, -4.263, -0.335], dtype=np.float32),
        np.array([-0.588, 1.136, 0.886], dtype=np.float32),
    ),
    # pig_cta.stl is not listed: it has many openings, so it auto-aligns opening
    # 0 to the rod tip on launch (press E to cycle to the femoral).
}
DEFAULT_VESSEL_OFFSET = np.array([13.887, 0.018, 0.156], dtype=np.float32)
DEFAULT_VESSEL_ROTATION = np.array([-0.023, 0.159, -1.480], dtype=np.float32)

_RECORDINGS_README = """\
# IVUS sensor-pose recordings

Each recording is the 6-DOF pose of a virtual IVUS sensor placed a fixed distance
**behind (proximal to) the catheter tip**, sampled once per render frame while
inserting, between when you pressed **R** (start) and **R** again (stop) in
`mosaic_sim.py`.

## Coordinate frame

All poses are in the **original CTA mesh frame** — the un-scaled, un-transformed
STL coordinates (millimetres for a typical STL). The simulation auto-fits and
re-poses the vessel internally, but every exported value is mapped back to the
raw STL frame, so the recording and the CTA `.stl` share one coordinate system.
Drop the matching `stls/<mesh>.stl` and a recording into your IVUS sim as-is.

Conventions:
- **position** `(x, y, z)` — sensor location, CTA units.
- **quaternion** `(qx, qy, qz, qw)` — rod material (Cosserat) frame orientation.
- **tangent** `(tx, ty, tz)` — unit axial direction pointing toward the tip;
  this is the **IVUS imaging-plane normal**. It is computed from wire geometry
  and is the most reliable rotational reference; use the quaternion for in-plane
  (roll) orientation.

## Files (per recording, sharing a `ivus_<mesh>_NNN` stem)

| File | Contents |
|------|----------|
| `*.csv` | Human-readable. Header comments + columns: `time_s, insertion, x, y, z, qx, qy, qz, qw, tx, ty, tz`. |
| `*.npz` | NumPy archive: arrays `time, insertion, position[N,3], quaternion[N,4], tangent[N,3], transforms[N,4,4]`, plus scalars `mesh, mesh_scale, sensor_mm`. |
| `*_transforms.npy` | Just the `[N,4,4]` homogeneous sensor-to-CTA transforms (rotation from the quaternion + translation). Easiest to feed a frame placer. |

`insertion` is the catheter insertion depth (sim units) and `time_s` the sim
time — use either to parameterize or resample the trajectory.

## Replaying as an animation

```bash
uv run mosaic_sim/replay.py mosaic_sim/recordings/ivus_<mesh>_000.npz
```

This loads the CTA mesh (wireframe) and animates the sensor frame — a moving
point, its three axis arrows, and a growing trail — through the recorded poses,
in the CTA frame. Accepts a `.npz` or `.csv`.
"""

# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------


def _first_mesh_prim(stage):
    """Return the first ``Mesh``-typed prim in a USD stage."""
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            return prim
    raise ValueError("No Mesh prim found in USD stage")


def load_vessel_mesh(path: str, prim_path: str | None = None):
    """Load a vessel mesh. USD (.usd/.usdc/.usda) via Newton's loader; everything
    else (STL/OBJ/PLY/glTF) via trimesh.

    Returns ``(vertices[N, 3] float32, normals[N, 3] float32 | None,
    indices[M] int32)``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".usd", ".usdc", ".usda"):
        from pxr import Usd  # noqa: PLC0415

        stage = Usd.Stage.Open(path)
        prim = stage.GetPrimAtPath(prim_path) if prim_path else _first_mesh_prim(stage)
        nm = newton.usd.get_mesh(prim, load_normals=True)
        verts = np.asarray(nm.vertices, dtype=np.float32)
        normals = np.asarray(nm.normals, dtype=np.float32) if nm.normals is not None else None
        indices = np.asarray(nm.indices, dtype=np.int32).reshape(-1)
        return verts, normals, indices

    import trimesh  # noqa: PLC0415

    mesh = trimesh.load(path, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    indices = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
    # Let the solver compute smooth vertex normals from the triangles.
    return verts, None, indices


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _set_root_position_kernel(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    new_pos: wp.vec3,
):
    """Set root particle position on both position arrays."""
    tid = wp.tid()
    if tid != 0:
        return
    positions[0] = new_pos
    predicted[0] = new_pos


@wp.kernel
def _set_particle_radius_kernel(
    particle_radius: wp.array(dtype=wp.float32),
    start: int,
    count: int,
    radius: float,
):
    i = wp.tid()
    if i >= count:
        return
    particle_radius[start + i] = radius


@wp.kernel
def _update_tip_rest_darboux_kernel(
    rest_darboux: wp.array(dtype=wp.vec3),
    num_edges: int,
    tip_num_edges: int,
    bend_angle: float,
):
    """Set rest Darboux vector: tip edges get curvature, others stay straight."""
    e = wp.tid()
    if e >= num_edges:
        return
    tip_start = num_edges - tip_num_edges
    if e >= tip_start:
        per_edge = bend_angle / float(tip_num_edges)
        rest_darboux[e] = wp.vec3(per_edge, 0.0, 0.0)
    else:
        rest_darboux[e] = wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w]."""
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        dtype=np.float32,
    )


def _qconj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit) of quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [x, y, z, w] (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = (float(c) for c in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (XYZ order, radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ],
        dtype=np.float32,
    )


def _transform_vertices(
    verts: np.ndarray,
    normals: np.ndarray | None,
    offset: np.ndarray,
    rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply rotation then translation to vertices (and rotation to normals)."""
    transformed = (verts @ rot.T) + offset
    transformed_normals = None
    if normals is not None:
        transformed_normals = (normals @ rot.T).astype(np.float32)
    return transformed.astype(np.float32), transformed_normals


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.device = wp.get_device()

        # --- Resolve CLI options ---
        self.replay_mode = False
        replay_path = getattr(args, "replay", None)
        mesh_path = getattr(args, "mesh", None)
        if replay_path and mesh_path is None:
            # Use the same vessel the recording was made against.
            rec_mesh = _peek_replay_mesh(replay_path)
            if rec_mesh:
                mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stls", rec_mesh)
        mesh_path = mesh_path or DEFAULT_MESH_PATH
        prim_path = getattr(args, "mesh_prim_path", None)
        scale_arg = getattr(args, "mesh_scale", None)

        # Tuneable parameters
        self.bend_stiffness = 0.1
        self.twist_stiffness = 1.0
        self.young_modulus = float(getattr(args, "stiffness", None) or 1_000_000.0)
        self.torsion_modulus = 1_000_000_000.0
        self.linear_damping = LINEAR_DAMPING
        self.angular_damping = ANGULAR_DAMPING
        self.gravity_enabled = False
        self.track_enabled = True
        self.track_stiffness = 1.0
        self.collision_enabled = True
        self.collision_radius = PARTICLE_RADIUS
        self.collision_pre_constraints_enabled = True
        self.collision_post_constraints_enabled = False
        self.collision_iterations = COLLISION_ITERATIONS
        self.mesh_edge_collision_enabled = False
        self.smooth_collision_normals_enabled = False
        # Containment sign convention (depends on mesh winding). Auto-detected per
        # mesh below so the lumen interior reads as "inside" instead of repelling
        # the catheter.
        self.sign_scale = -1.0
        self._aligned_to_opening = False
        self._interior_world_pt = None

        # Controls
        self.insertion = 0.0
        self.root_rotation = 0.0
        self.tip_bend_angle = 0.0
        self.tip_num_edges = TIP_NUM_EDGES
        self._g_pressed = False
        # Vessel-align mode: when on, the drive keys move the vessel instead of
        # the catheter, so the femoral entry can be lined up to the rod by eye
        # (no GUI clicking needed). Toggled with V.
        self.align_vessel_mode = False
        self._v_pressed = False
        # Vessel rendered as a light see-through wireframe by default (so the
        # catheter inside is visible); press W to toggle solid.
        self.vessel_wireframe = True
        self._w_pressed = False
        # Entry-opening selection: align an open tube end to the rod tip so
        # insertion feeds the catheter in. E cycles through the openings.
        self._openings: list = []
        self._entry_idx = 0
        self._e_pressed = False
        # IVUS sensor-pose recording (6DOF of a point behind the tip, in CTA frame).
        self._recording = False
        self._record_rows: list = []
        self._r_pressed = False

        # --- Load vessel mesh ---
        verts, normals, indices = load_vessel_mesh(mesh_path, prim_path)

        # Resolve scale: explicit --mesh-scale wins; otherwise auto-fit so the
        # mesh's units don't matter.
        if scale_arg is not None:
            scale = float(scale_arg)
        else:
            diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
            scale = (AUTOFIT_VESSEL_FRACTION * ROD_LENGTH / diag) if diag > 1e-9 else 1.0

        self._raw_verts = (verts * scale).astype(np.float32)
        self._raw_normals = normals
        self._raw_indices = indices.astype(np.int32)
        self.mesh_name = os.path.basename(mesh_path)
        self.mesh_scale = float(scale)  # sim units per CTA mesh unit (for exporting)

        # Mesh transform (Euler XYZ radians + translation): use a hand-tuned pose
        # if listed, else a default that the opening-alignment below usually
        # overrides for open meshes.
        pose = VESSEL_POSES.get(self.mesh_name)
        if pose is not None:
            self.mesh_offset = pose[0].copy()
            self.mesh_rotation = pose[1].copy()
        else:
            self.mesh_offset = DEFAULT_VESSEL_OFFSET.copy()
            self.mesh_rotation = DEFAULT_VESSEL_ROTATION.copy()

        # Detect open tube ends so an opening can be lined up to the rod tip.
        self._openings = self._detect_openings()

        # --entry N aligns opening N to the rod tip; otherwise, if the mesh has
        # open ends and no hand-tuned pose, auto-align opening 0 to the tip.
        entry_arg = getattr(args, "entry", None)
        if entry_arg is not None and self._openings:
            self._align_opening_to_tip(entry_arg)
        elif self._openings and self.mesh_name not in VESSEL_POSES:
            self._align_opening_to_tip(0)

        # CLI overrides let you align an imported vessel without the GUI sliders.
        offset_arg = getattr(args, "mesh_offset", None)
        rotation_arg = getattr(args, "mesh_rotation", None)
        if offset_arg is not None or rotation_arg is not None:
            if offset_arg is not None:
                self.mesh_offset = np.array(offset_arg, dtype=np.float32)
            if rotation_arg is not None:
                self.mesh_rotation = np.array(rotation_arg, dtype=np.float32)
            self._aligned_to_opening = False  # manual pose: don't trust the tip as interior

        print(
            f"Vessel '{self.mesh_name}': {len(self._raw_verts)} verts, "
            f"{self._raw_indices.shape[0] // 3} tris, scale {scale:g}."
        )

        # Build initial transformed mesh arrays + collision BVH
        self.vessel_verts_wp = None
        self.vessel_normals_wp = None
        self.collision_normals_wp = None
        self.vessel_indices_wp = wp.array(self._raw_indices, dtype=wp.int32, device=self.device)
        self.collision_mesh = None
        self._rebuild_vessel_mesh()

        # Auto-detect containment sign convention from the mesh winding.
        sign_arg = getattr(args, "sign_scale", None)
        if sign_arg is not None:
            self.sign_scale = float(sign_arg)
        elif self._aligned_to_opening and self._interior_world_pt is not None:
            self.sign_scale = self._detect_sign_scale(self._interior_world_pt)
            print(f"Auto sign_scale = {self.sign_scale:+.0f} (lumen interior reads as inside).")

        # --- Build rod ---
        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        positions = np.zeros((NUM_POINTS, 3), dtype=np.float32)
        for i in range(NUM_POINTS):
            positions[i, 0] = i * SEGMENT_LENGTH
            positions[i, 2] = 1.0

        self.track_start = positions[0].copy()
        self.track_end = positions[-1].copy()
        track_vec = self.track_end - self.track_start
        self.track_length = float(np.linalg.norm(track_vec))
        self.track_dir = track_vec / self.track_length

        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=self.collision_radius,
            particle_mass=PARTICLE_MASS,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            young_modulus=self.young_modulus,
            torsion_modulus=self.torsion_modulus,
            lock_root=True,
            lock_root_rotation=True,
        )

        self.model = builder.finalize()
        # Override gravity (starts disabled)
        self.model.gravity = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=self.device)

        self.solver = XCathRodSolver(
            model=self.model,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
            solver_backend="block_thomas",
            floor_z=None,
            collision_mesh=self.collision_mesh,
            collision_mesh_normals=self.collision_normals_wp,
            track_start=self.track_start,
            track_dir=self.track_dir,
            track_length=self.track_length,
            tip_num_edges=self.tip_num_edges,
            particle_radius=self.collision_radius,
            segment_length=SEGMENT_LENGTH,
            track_stiffness=self.track_stiffness,
            track_enabled=self.track_enabled,
            collision_enabled=self.collision_enabled,
            collision_iterations=self.collision_iterations,
            collision_pre_constraints_enabled=self.collision_pre_constraints_enabled,
            collision_post_constraints_enabled=self.collision_post_constraints_enabled,
            mesh_edge_collision_enabled=self.mesh_edge_collision_enabled,
            smooth_collision_normals_enabled=self.smooth_collision_normals_enabled,
            sign_scale=self.sign_scale,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Shorthand references
        self.ws = self.solver._rods[0]
        self.rod_particle_start = self.solver._rod_particle_starts[0]

        # Cache base root orientation
        self.base_orientation = self.ws.orientations_wp.numpy()[0].copy()

        # Rod tube mesher
        self.mesher = xpbd_rod.RodMesher(
            num_points=NUM_POINTS,
            radius=self.collision_radius,
            resolution=8,
            smoothing=3,
            device=self.device,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False

        self._report_tip_containment()

        if replay_path:
            self._load_replay(replay_path)

    def _detect_sign_scale(self, world_pt) -> float:
        """Pick sign_scale so a known-interior point reads as inside (negative).
        Mesh winding varies, so this avoids the catheter being repelled."""
        if self.collision_mesh is None:
            return self.sign_scale
        max_dist = 2.0 * float(self.collision_radius) + SEGMENT_LENGTH
        q = wp.array([[float(world_pt[0]), float(world_pt[1]), float(world_pt[2])]], dtype=wp.vec3, device=self.device)
        sd_wp, hit_wp = compute_signed_distances(
            q, self.collision_mesh.id, sign_scale=1.0, max_dist=max_dist, device=self.device
        )
        sd = float(sd_wp.numpy()[0])
        hit = bool(hit_wp.numpy()[0])
        if not hit:
            return self.sign_scale  # couldn't sample; keep current
        # With sign_scale=+1 the interior reads `sd`; we want it negative.
        return 1.0 if sd < 0.0 else -1.0

    def _report_tip_containment(self):
        """Print whether the rod tip starts inside the lumen, using the sim's own
        signed-distance query (negative = inside, given sign_scale)."""
        if self.collision_mesh is None:
            return
        ps = self.rod_particle_start
        sd_wp, hit_wp = compute_signed_distances(
            self.state_0.particle_q[ps : ps + NUM_POINTS],
            self.collision_mesh.id,
            sign_scale=self.solver.sign_scale,
            max_dist=self.solver.max_dist,
            device=self.device,
        )
        sd = sd_wp.numpy()
        hit = hit_wp.numpy().astype(bool)
        inside = (sd < 0.0) & hit
        print(
            f"Tip containment: tip signed-dist={sd[-1]:.3f} (neg=inside, hit={bool(hit[-1])}); "
            f"last 8 rod pts inside={inside[-8:].tolist()}; total inside {int(inside.sum())}/{NUM_POINTS}"
        )

    # ----- IVUS sensor-pose recording -----

    def _sensor_world_pose(self):
        """(position, tangent, quat) of the sensor SENSOR_BEHIND_TIP_MM behind the
        tip, in sim-world coords. tangent points toward the tip; quat [x,y,z,w] is
        the rod material frame."""
        ps = self.rod_particle_start
        P = self.state_0.particle_q.numpy()[ps : ps + NUM_POINTS].astype(np.float64)
        Q = self.ws.orientations_wp.numpy()  # [N, 4] (x,y,z,w), world frame
        target = SENSOR_BEHIND_TIP_MM * self.mesh_scale  # sim units behind the tip
        acc = 0.0
        i = NUM_POINTS - 1
        while i > 0:
            seg = float(np.linalg.norm(P[i] - P[i - 1]))
            if acc + seg >= target and seg > 1e-9:
                t = (target - acc) / seg
                pos = P[i] * (1.0 - t) + P[i - 1] * t
                tang = P[i] - P[i - 1]
                tang /= np.linalg.norm(tang) + 1e-12
                quat = Q[i - 1] if t > 0.5 else Q[i]
                return pos, tang, np.asarray(quat, dtype=np.float64)
            acc += seg
            i -= 1
        tang = P[1] - P[0]
        tang /= np.linalg.norm(tang) + 1e-12
        return P[0], tang, np.asarray(Q[0], dtype=np.float64)

    def _sensor_pose_row(self):
        """Sensor 6DOF in the ORIGINAL CTA frame:
        [t, insertion, x,y,z, qx,qy,qz,qw, tx,ty,tz]."""
        pos_w, tang_w, quat_w = self._sensor_world_pose()
        # world = scale * R @ mesh + offset  =>  mesh = R^T @ (world - offset) / scale
        R = _rotation_matrix(*self.mesh_rotation).astype(np.float64)
        off = self.mesh_offset.astype(np.float64)
        pos_cta = ((pos_w - off) @ R) / self.mesh_scale
        tang_cta = tang_w @ R
        tang_cta /= np.linalg.norm(tang_cta) + 1e-12
        qR = _quat_from_matrix(R).astype(np.float64)
        q_cta = _qmul(_qconj(qR), quat_w)
        q_cta = q_cta / (np.linalg.norm(q_cta) + 1e-12)
        return [
            round(self.sim_time, 6),
            round(float(self.insertion), 6),
            *[round(float(x), 6) for x in pos_cta],
            *[round(float(x), 6) for x in q_cta],
            *[round(float(x), 6) for x in tang_cta],
        ]

    def _toggle_recording(self):
        if self.replay_mode:
            return  # nothing to record while replaying
        if not self._recording:
            self._record_rows = []
            self._recording = True
            print(f"REC start — sensor {SENSOR_BEHIND_TIP_MM:.1f}mm behind tip, CTA frame. Press R to stop.")
        else:
            self._recording = False
            self._save_recording()

    def _save_recording(self):
        if not self._record_rows:
            print("REC stop — no samples captured.")
            return
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(self.mesh_name)[0]
        n = 0
        while os.path.exists(os.path.join(out_dir, f"ivus_{stem}_{n:03d}.csv")):
            n += 1
        base = os.path.join(out_dir, f"ivus_{stem}_{n:03d}")

        data = np.array(self._record_rows, dtype=np.float64)  # [N, 12]
        time, insertion = data[:, 0], data[:, 1]
        position, quaternion, tangent = data[:, 2:5], data[:, 5:9], data[:, 9:12]
        num = len(data)
        # 4x4 homogeneous CTA-frame transforms (rotation from quat + translation).
        transforms = np.tile(np.eye(4), (num, 1, 1))
        for i in range(num):
            transforms[i, :3, :3] = _matrix_from_quat(quaternion[i])
            transforms[i, :3, 3] = position[i]

        # CSV (human-readable).
        header = (
            f"# IVUS sensor pose, {SENSOR_BEHIND_TIP_MM:.1f} mm behind the wire tip\n"
            f"# mesh={self.mesh_name}  mesh_scale(sim_units_per_CTA_unit)={self.mesh_scale:.8f}\n"
            f"# coordinates are in the ORIGINAL CTA mesh frame (un-scaled STL units)\n"
            f"# quaternion [qx,qy,qz,qw]=rod material frame; tangent [tx,ty,tz]=axial dir toward tip\n"
            "time_s,insertion,x,y,z,qx,qy,qz,qw,tx,ty,tz\n"
        )
        with open(base + ".csv", "w") as fh:
            fh.write(header)
            for row in self._record_rows:
                fh.write(",".join(str(x) for x in row) + "\n")
        # NPZ (named arrays + metadata).
        np.savez(
            base + ".npz",
            time=time,
            insertion=insertion,
            position=position,
            quaternion=quaternion,
            tangent=tangent,
            transforms=transforms,
            mesh=self.mesh_name,
            mesh_scale=self.mesh_scale,
            sensor_mm=SENSOR_BEHIND_TIP_MM,
        )
        # Bare 4x4 transforms (easiest to feed an IVUS placer).
        np.save(base + "_transforms.npy", transforms.astype(np.float32))

        self._write_recordings_readme(out_dir)
        print(f"REC saved {num} samples -> {base}.{{csv, npz, _transforms.npy}}")

    # ----- Replay -----

    def _load_replay(self, path: str):
        """Load a recorded trajectory (CTA frame) and pre-transform it into the
        rendered sim-world frame so the marker overlays on the displayed vessel."""
        if path.endswith(".npz"):
            d = np.load(path)
            pos, quat = np.asarray(d["position"]), np.asarray(d["quaternion"])
        else:
            rows = [
                ln for ln in open(path) if ln.strip() and not ln.startswith("#") and not ln.startswith("time")
            ]
            arr = np.array([[float(x) for x in r.split(",")] for r in rows])
            pos, quat = arr[:, 2:5], arr[:, 5:9]
        R = _rotation_matrix(*self.mesh_rotation).astype(np.float64)
        off = self.mesh_offset.astype(np.float64)
        world_pos = (pos.astype(np.float64) * self.mesh_scale) @ R.T + off  # CTA -> world
        n = len(pos)
        world_axes = np.zeros((n, 3, 3))
        for i in range(n):
            world_axes[i] = R @ _matrix_from_quat(quat[i])  # frame axes (cols) in world
        self._replay_pos = world_pos.astype(np.float32)
        self._replay_axes = world_axes.astype(np.float32)
        self._replay_n = n
        self._replay_idx = 0.0
        self._replay_speed = 1.0
        self.replay_mode = True
        print(f"Replay: {n} samples from {os.path.basename(path)} (Space=pause; loops).")

    def _render_replay_marker(self):
        i = int(self._replay_idx) % self._replay_n
        p = self._replay_pos[i]
        axes = self._replay_axes[i]
        ln = max(4.0 * float(self.collision_radius), 0.25)
        dev = self.device
        self.viewer.log_points(
            "/sensor", wp.array([p], dtype=wp.vec3, device=dev), 1.5 * float(self.collision_radius), (1.0, 0.9, 0.2)
        )
        starts = wp.array([p, p, p], dtype=wp.vec3, device=dev)
        ends = wp.array(
            [p + ln * axes[:, 0], p + ln * axes[:, 1], p + ln * axes[:, 2]], dtype=wp.vec3, device=dev
        )
        cols = wp.array([(1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.35, 0.55, 1.0)], dtype=wp.vec3, device=dev)
        self.viewer.log_arrows("/sensor_axes", starts, ends, cols)
        if i >= 1:
            s = wp.array(self._replay_pos[:i], dtype=wp.vec3, device=dev)
            e = wp.array(self._replay_pos[1 : i + 1], dtype=wp.vec3, device=dev)
            self.viewer.log_lines("/trail", s, e, (1.0, 0.9, 0.2))

    @staticmethod
    def _write_recordings_readme(out_dir: str):
        readme = os.path.join(out_dir, "README.md")
        if os.path.exists(readme):
            return
        with open(readme, "w") as fh:
            fh.write(_RECORDINGS_README)

    def _apply_collision_radius(self):
        """Apply collision radius to solver and visual radius buffers."""
        radius = max(0.001, float(self.collision_radius))
        self.collision_radius = radius
        self.solver.target_phi = -radius
        self.solver.max_dist = 2.0 * radius + SEGMENT_LENGTH
        self.solver.mesh_edge_collision_radius = radius
        self.solver.mesh_edge_collision_query_radius = radius
        self.solver.mesh_edge_collision_max_triangles = 128
        self.mesher.radius = radius
        if self.model.particle_radius is not None:
            wp.launch(
                _set_particle_radius_kernel,
                dim=NUM_POINTS,
                inputs=[self.model.particle_radius, self.rod_particle_start, NUM_POINTS, radius],
                device=self.device,
            )

    # ----- Entry-opening alignment -----

    def _detect_openings(self):
        """Find boundary loops (open tube ends) in the raw (scaled, untransformed)
        frame. Returns [(ring_size, center, inward_axis), ...] largest-first;
        empty for a watertight mesh."""
        from collections import defaultdict  # noqa: PLC0415

        faces = self._raw_indices.reshape(-1, 3)
        ec: dict[tuple[int, int], int] = defaultdict(int)
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for e in ((a, b), (b, c), (c, a)):
                ec[(min(e), max(e))] += 1
        bedges = [e for e, n in ec.items() if n == 1]
        if not bedges:
            return []
        adj: dict[int, list[int]] = defaultdict(list)
        for a, b in bedges:
            adj[a].append(b)
            adj[b].append(a)
        seen: set[int] = set()
        loops = []
        for a, _b in bedges:
            if a in seen:
                continue
            comp: set[int] = set()
            stack = [a]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                for y in adj[x]:
                    if y not in comp:
                        stack.append(y)
            seen |= comp
            loops.append(sorted(comp))
        verts = self._raw_verts.astype(np.float64)
        center_all = verts.mean(axis=0)
        out = []
        for loop in loops:
            P = verts[loop]
            c = P.mean(axis=0)
            radius = float(np.linalg.norm(P - c, axis=1).mean())
            _u, _s, vt = np.linalg.svd(P - c)
            n = vt[2]
            n = n / (np.linalg.norm(n) + 1e-12)
            if np.dot(center_all - c, n) < 0:  # point inward (toward mesh body)
                n = -n
            out.append((len(loop), c.astype(np.float32), n.astype(np.float32), radius))
        out.sort(key=lambda t: -t[0])
        return out

    def _march_into_lumen(self, center, axis, radius, depth):
        """From an opening, march a short recentered centerline into the lumen.
        Returns (interior_point, tangent) — the point is pulled toward the local
        lumen center each step, so it lands inside the tube, not on a wall."""
        verts = self._raw_verts.astype(np.float64)
        p = np.asarray(center, dtype=np.float64).copy()
        d = np.asarray(axis, dtype=np.float64)
        d /= np.linalg.norm(d) + 1e-12
        tang = d.copy()
        step = max(float(radius) * 0.5, SEGMENT_LENGTH)
        total = 0.0
        while total < depth:
            cand = p + d * step
            near = verts[np.linalg.norm(verts - cand, axis=1) < radius * 1.8]
            if len(near) < 4:
                break  # marched out of the tube; keep last interior point
            cand = 0.4 * cand + 0.6 * near.mean(axis=0)  # recenter to lumen
            seg = cand - p
            nl = float(np.linalg.norm(seg))
            if nl < 1e-6:
                break
            d = seg / nl
            tang = d.copy()
            p = cand
            total += nl
        return p, tang

    def _align_opening_to_tip(self, idx: int):
        """Pose the vessel so opening ``idx`` sits at the rod tip with the lumen
        running forward along the track, so insertion feeds the catheter in."""
        if not self._openings:
            print("No open ends detected (watertight mesh) — can't align to an opening.")
            return
        idx %= len(self._openings)
        self._entry_idx = idx
        size, c, n, radius = self._openings[idx]
        # March a recentered centerline START_INSIDE_DEPTH into the lumen, then
        # put the rod TIP at that interior point with the rod aligned to the local
        # lumen tangent — so the tip starts genuinely inside the tube, centered.
        p_in, tangent = self._march_into_lumen(c, n, radius, START_INSIDE_DEPTH)
        track_dir = np.array([1.0, 0.0, 0.0])
        track_tip = np.array([ROD_LENGTH, 0.0, 1.0])  # front of the rod
        a = tangent.astype(np.float64)
        a /= np.linalg.norm(a) + 1e-12
        vv = np.cross(a, track_dir)
        s = np.linalg.norm(vv)
        cd = float(np.dot(a, track_dir))
        if s < 1e-9:
            R = np.eye(3) if cd > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            K = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
            R = np.eye(3) + K + K @ K * ((1 - cd) / (s * s))
        offset = track_tip - R @ p_in.astype(np.float64)
        ry = math.asin(max(-1.0, min(1.0, R[0, 2])))
        rx = math.atan2(-R[1, 2], R[2, 2])
        rz = math.atan2(-R[0, 1], R[0, 0])
        self.mesh_offset = offset.astype(np.float32)
        self.mesh_rotation = np.array([rx, ry, rz], dtype=np.float32)
        # The rod tip ends up exactly at the marched interior point — record it as
        # a known-inside sample for sign_scale auto-detection.
        self._aligned_to_opening = True
        self._interior_world_pt = track_tip.copy()
        print(
            f"Entry opening {idx}/{len(self._openings) - 1} (ring {size}) -> rod tip "
            f"({START_INSIDE_DEPTH:.2f} inside).  "
            f"--mesh-offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f} "
            f"--mesh-rotation {rx:.3f} {ry:.3f} {rz:.3f}"
        )
        if hasattr(self, "solver"):
            self._rebuild_vessel_mesh()

    # ----- Mesh transform -----

    def _rebuild_vessel_mesh(self):
        """Recompute transformed vessel vertices and rebuild collision BVH."""
        rot = _rotation_matrix(*self.mesh_rotation)
        verts, normals = _transform_vertices(self._raw_verts, self._raw_normals, self.mesh_offset, rot)
        self.vessel_verts_wp = wp.array(verts, dtype=wp.vec3, device=self.device)
        self.vessel_normals_wp = wp.array(normals, dtype=wp.vec3, device=self.device) if normals is not None else None
        collision_normals = normals
        if collision_normals is None or collision_normals.shape[0] != verts.shape[0]:
            collision_normals = compute_smooth_vertex_normals(verts, self._raw_indices)
        self.collision_normals_wp = wp.array(collision_normals, dtype=wp.vec3, device=self.device)
        self.collision_mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(self._raw_indices, dtype=wp.int32, device=self.device),
        )
        if hasattr(self, "solver"):
            self.solver.set_collision_mesh(self.collision_mesh, self.collision_normals_wp)

    # ----- Input handling -----

    def _handle_input(self):
        v = self.viewer
        if not hasattr(v, "is_key_down"):
            return

        try:
            import pyglet  # noqa: PLC0415

            KEY = pyglet.window.key
        except Exception:
            return

        insert_speed = 0.5
        rotate_speed = 1.5
        bend_speed = 1.0
        move_speed = 0.5  # vessel translate [m/s]
        vrot_speed = 0.8  # vessel rotate [rad/s]
        dt = self.frame_dt

        # Toggle vessel-align mode (edge-triggered on V)
        if v.is_key_down("v"):
            if not self._v_pressed:
                self.align_vessel_mode = not self.align_vessel_mode
                self._v_pressed = True
                if self.align_vessel_mode:
                    print(
                        "ALIGN VESSEL MODE ON — move/rotate the vessel:\n"
                        "  I/K=+X/-X  J/L=+Y/-Y  U/O=+Z/-Z\n"
                        "  ,/.=yaw(Z)  [/]=pitch(X)  ;/'=roll(Y)   V=back to driving"
                    )
                else:
                    print("ALIGN VESSEL MODE OFF — driving the catheter again.")
        else:
            self._v_pressed = False

        # Gravity toggle (edge-triggered) — available in either mode
        if v.is_key_down("g"):
            if not self._g_pressed:
                self.gravity_enabled = not self.gravity_enabled
                g = wp.vec3(0.0, 0.0, -9.81) if self.gravity_enabled else wp.vec3(0.0, 0.0, 0.0)
                self.ws.gravity = g
                self._g_pressed = True
        else:
            self._g_pressed = False

        # Wireframe toggle (edge-triggered) — available in either mode
        if v.is_key_down("w"):
            if not self._w_pressed:
                self.vessel_wireframe = not self.vessel_wireframe
                self._w_pressed = True
        else:
            self._w_pressed = False

        # Cycle entry opening -> rod tip (edge-triggered) to find the femoral
        if v.is_key_down("e"):
            if not self._e_pressed:
                if self._openings:
                    self._align_opening_to_tip(self._entry_idx + 1)
                else:
                    print("No open ends to cycle (watertight mesh).")
                self._e_pressed = True
        else:
            self._e_pressed = False

        # Record sensor pose (edge-triggered): R = start, R again = stop + save
        if v.is_key_down("r"):
            if not self._r_pressed:
                self._toggle_recording()
                self._r_pressed = True
        else:
            self._r_pressed = False

        if self.align_vessel_mode:
            self._handle_vessel_align(v, KEY, move_speed * dt, vrot_speed * dt)
            return

        # ----- Catheter driving -----

        # Insertion
        if v.is_key_down(KEY.PAGEUP) or v.is_key_down("2") or v.is_key_down("i"):
            self.insertion += insert_speed * dt
        if v.is_key_down(KEY.PAGEDOWN) or v.is_key_down("1") or v.is_key_down("k"):
            self.insertion -= insert_speed * dt
        self.insertion = max(0.0, self.insertion)

        # Root rotation around local Z
        if v.is_key_down(KEY.COMMA) or v.is_key_down("j"):
            self.root_rotation -= rotate_speed * dt
        if v.is_key_down(KEY.PERIOD) or v.is_key_down("l"):
            self.root_rotation += rotate_speed * dt

        # Tip bending
        if v.is_key_down(KEY.EQUAL) or v.is_key_down(KEY.NUM_ADD):
            self.tip_bend_angle += bend_speed * dt
        if v.is_key_down(KEY.MINUS) or v.is_key_down(KEY.NUM_SUBTRACT):
            self.tip_bend_angle -= bend_speed * dt
        self.tip_bend_angle = max(-1.5, min(1.5, self.tip_bend_angle))

    def _handle_vessel_align(self, v, KEY, dmove: float, drot: float):
        """Move/rotate the vessel with the keyboard so the femoral entry can be
        lined up to the catheter by eye. Rebuilds the BVH and prints the transform
        (copy it into --mesh-offset / --mesh-rotation to make it permanent)."""
        changed = False
        # Translation
        if v.is_key_down("i"):
            self.mesh_offset[0] += dmove; changed = True
        if v.is_key_down("k"):
            self.mesh_offset[0] -= dmove; changed = True
        if v.is_key_down("j"):
            self.mesh_offset[1] += dmove; changed = True
        if v.is_key_down("l"):
            self.mesh_offset[1] -= dmove; changed = True
        if v.is_key_down("u"):
            self.mesh_offset[2] += dmove; changed = True
        if v.is_key_down("o"):
            self.mesh_offset[2] -= dmove; changed = True
        # Rotation
        if v.is_key_down(KEY.COMMA):
            self.mesh_rotation[2] -= drot; changed = True
        if v.is_key_down(KEY.PERIOD):
            self.mesh_rotation[2] += drot; changed = True
        if v.is_key_down(KEY.BRACKETLEFT):
            self.mesh_rotation[0] -= drot; changed = True
        if v.is_key_down(KEY.BRACKETRIGHT):
            self.mesh_rotation[0] += drot; changed = True
        if v.is_key_down(KEY.SEMICOLON):
            self.mesh_rotation[1] -= drot; changed = True
        if v.is_key_down(KEY.APOSTROPHE):
            self.mesh_rotation[1] += drot; changed = True
        if changed:
            self._rebuild_vessel_mesh()
            o, r = self.mesh_offset, self.mesh_rotation
            print(
                f"  --mesh-offset {o[0]:.3f} {o[1]:.3f} {o[2]:.3f} "
                f"--mesh-rotation {r[0]:.3f} {r[1]:.3f} {r[2]:.3f}"
            )

    # ----- Root & tip control -----

    def _apply_root_control(self):
        ws = self.ws

        # Root position along track
        root_pos = self.track_start + self.track_dir * self.insertion
        wp.launch(
            _set_root_position_kernel,
            dim=1,
            inputs=[ws.positions_wp, ws.predicted_positions_wp, wp.vec3(*root_pos.tolist())],
            device=self.device,
        )

        # Root rotation
        half = self.root_rotation * 0.5
        q_twist = np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32)
        q_new = _qmul(self.base_orientation, q_twist)
        q_new /= np.linalg.norm(q_new)
        self.solver.set_root_orientation(
            0, wp.quat(float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3]))
        )

        # Tip bend rest Darboux
        wp.launch(
            _update_tip_rest_darboux_kernel,
            dim=ws.num_edges,
            inputs=[ws.rest_darboux_wp, ws.num_edges, self.tip_num_edges, self.tip_bend_angle],
            device=self.device,
        )

    # ----- Simulation step -----

    def step(self):
        if self.replay_mode:
            self._handle_input()  # camera/wireframe keys still work; no physics
            self._replay_idx += self._replay_speed
            if self._replay_idx >= self._replay_n:
                self._replay_idx = 0.0  # loop
            self.sim_time += self.frame_dt
            return

        self._handle_input()
        self.solver.track_enabled = self.track_enabled
        self.solver.track_stiffness = self.track_stiffness
        self.solver.linear_damping = max(0.0, min(DAMPING_SLIDER_MAX, float(self.linear_damping)))
        self.solver.angular_damping = max(0.0, min(DAMPING_SLIDER_MAX, float(self.angular_damping)))
        self.tip_num_edges = max(1, min(self.ws.num_edges, int(self.tip_num_edges)))
        self.solver.tip_num_edges = self.tip_num_edges
        self._apply_collision_radius()
        self.solver.collision_enabled = self.collision_enabled
        self.solver.collision_pre_constraints_enabled = self.collision_pre_constraints_enabled
        self.solver.collision_post_constraints_enabled = self.collision_post_constraints_enabled
        self.solver.mesh_edge_collision_enabled = self.mesh_edge_collision_enabled
        self.solver.smooth_collision_normals_enabled = self.smooth_collision_normals_enabled
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver.collision_iterations = max(1, int(self.collision_iterations))

        for _ in range(self.sim_substeps):
            self._apply_root_control()

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

        if self._recording:
            self._record_rows.append(self._sensor_pose_row())

    # ----- Rendering -----

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Vessel mesh (gray)
        self.viewer.log_mesh(
            "/vessel",
            self.vessel_verts_wp,
            self.vessel_indices_wp,
            self.vessel_normals_wp,
        )
        self._style_vessel_mesh(self.viewer, "/vessel")

        if self.replay_mode:
            # Animate the recorded sensor frame instead of the live catheter.
            self._render_replay_marker()
            self.viewer.end_frame()
            return

        # Rod tube mesh (blue)
        ps = self.rod_particle_start
        self.mesher.update(self.state_0.particle_q[ps : ps + NUM_POINTS])
        self.viewer.log_mesh(
            "/catheter",
            self.mesher.vertices,
            self.mesher.indices,
            self.mesher.normals,
            self.mesher.uvs,
        )
        self._set_mesh_color(self.viewer, "/catheter", 0.3, 0.45, 0.85)

        self.viewer.end_frame()

    # ----- Mesh color -----

    @staticmethod
    def _set_mesh_color(viewer, name: str, r: float, g: float, b: float):
        """Patch a logged GL mesh so it sets its albedo color before each draw."""
        if not hasattr(viewer, "objects"):
            return
        mesh_obj = viewer.objects.get(name)
        if mesh_obj is None or hasattr(mesh_obj, "_color_patched"):
            return
        try:
            from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

            gl_mod = RendererGL.gl
        except Exception:
            return

        original_render = mesh_obj.render

        def _colored_render(_gl=gl_mod, _r=r, _g=g, _b=b, _orig=original_render, _vao=mesh_obj.vao):
            _gl.glBindVertexArray(_vao)
            _gl.glVertexAttrib3f(7, _r, _g, _b)
            _gl.glBindVertexArray(0)
            _orig()

        mesh_obj.render = _colored_render
        mesh_obj._color_patched = True

    def _style_vessel_mesh(self, viewer, name: str):
        """Patch the vessel GL mesh to draw as a light see-through wireframe
        (default) or a soft solid, reading ``self.vessel_wireframe`` live so the
        W key toggles it. Wireframe makes the catheter inside the lumen visible."""
        if not hasattr(viewer, "objects"):
            return
        mesh_obj = viewer.objects.get(name)
        if mesh_obj is None or hasattr(mesh_obj, "_vessel_styled"):
            return
        try:
            from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

            gl = RendererGL.gl
        except Exception:
            return

        orig = mesh_obj.render

        def _render(_gl=gl, _orig=orig, _vao=mesh_obj.vao, _ex=self):
            wire = _ex.vessel_wireframe
            r, g, b = (0.72, 0.82, 0.95) if wire else (0.55, 0.57, 0.60)
            _gl.glBindVertexArray(_vao)
            _gl.glVertexAttrib3f(7, r, g, b)
            _gl.glBindVertexArray(0)
            if wire:
                # GL_LINE polygon mode = wireframe (core-profile safe; avoid
                # GL_LINE_SMOOTH / glLineWidth which error on macOS core profile).
                _gl.glPolygonMode(_gl.GL_FRONT_AND_BACK, _gl.GL_LINE)
                _orig()
                _gl.glPolygonMode(_gl.GL_FRONT_AND_BACK, _gl.GL_FILL)
            else:
                _orig()

        mesh_obj.render = _render
        mesh_obj._vessel_styled = True

    # ----- Stiffness update -----

    def _update_stiffness(self):
        ws = self.ws
        ws.young_modulus = self.young_modulus
        ws.torsion_modulus = self.torsion_modulus
        bs = np.full(
            (ws.num_edges, 3),
            [self.bend_stiffness, self.bend_stiffness, self.twist_stiffness],
            dtype=np.float32,
        )
        ws.bend_stiffness_wp.assign(wp.array(bs, dtype=wp.vec3, device=ws.device))

    # ----- ImGui -----

    def gui(self, imgui):
        imgui.text("Mosaic Catheter Simulation")
        imgui.text(f"Vessel: {self.mesh_name}")
        imgui.separator()

        _, self.insertion = imgui.slider_float("Insertion", self.insertion, 0.0, 10.0)
        _, self.tip_bend_angle = imgui.slider_float("Tip Bend", self.tip_bend_angle, -1.8, 1.8)
        _, self.tip_num_edges = imgui.slider_int("Tip Bend Segments", self.tip_num_edges, 1, 30)
        _, self.root_rotation = imgui.slider_float("Root Rotation ", self.root_rotation, -math.pi, math.pi)

        imgui.separator()

        changed_g, self.gravity_enabled = imgui.checkbox("Gravity", self.gravity_enabled)
        if changed_g:
            g = wp.vec3(0.0, 0.0, -9.81) if self.gravity_enabled else wp.vec3(0.0, 0.0, 0.0)
            self.ws.gravity = g

        _, self.vessel_wireframe = imgui.checkbox("Vessel Wireframe (W)", self.vessel_wireframe)
        _, self.track_enabled = imgui.checkbox("Track Sliding", self.track_enabled)
        _, self.collision_enabled = imgui.checkbox("Mesh Collision", self.collision_enabled)
        _, self.mesh_edge_collision_enabled = imgui.checkbox(
            "Use Mesh Edge Collision Path", self.mesh_edge_collision_enabled
        )
        _, self.smooth_collision_normals_enabled = imgui.checkbox(
            "Use Smooth Collision Normals", self.smooth_collision_normals_enabled
        )
        _, self.collision_pre_constraints_enabled = imgui.checkbox(
            "Collision Before Rod Constraints", self.collision_pre_constraints_enabled
        )
        _, self.collision_post_constraints_enabled = imgui.checkbox(
            "Collision After Rod Constraints", self.collision_post_constraints_enabled
        )

        imgui.separator()

        changed_b, self.bend_stiffness = imgui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_t, self.twist_stiffness = imgui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        if changed_b or changed_t:
            self._update_stiffness()

        changed_E, self.young_modulus = imgui.input_float("Young Modulus [Pa]", self.young_modulus, format="%.1f")
        changed_G, self.torsion_modulus = imgui.input_float("Torsion Modulus [Pa]", self.torsion_modulus, format="%.1f")
        if changed_E or changed_G:
            self._update_stiffness()

        _, self.track_stiffness = imgui.slider_float("Track Stiffness", self.track_stiffness, 0.0, 1.0)
        _, self.linear_damping = imgui.slider_float("Pos Damping", self.linear_damping, 0.0, DAMPING_SLIDER_MAX)
        _, self.angular_damping = imgui.slider_float("Rot Damping", self.angular_damping, 0.0, DAMPING_SLIDER_MAX)

        imgui.separator()

        _, self.sim_substeps = imgui.slider_int("Substeps", self.sim_substeps, 1, 32)
        _, self.collision_iterations = imgui.slider_int("Collision Iterations", self.collision_iterations, 1, 16)
        _, self.collision_radius = imgui.slider_float("Collision Radius", self.collision_radius, 0.001, 0.05)

        imgui.separator()
        imgui.text("Mesh Transform")
        mesh_changed = False
        c, self.mesh_offset[0] = imgui.slider_float("Mesh X", float(self.mesh_offset[0]), -20.0, 20.0)
        mesh_changed = mesh_changed or c
        c, self.mesh_offset[1] = imgui.slider_float("Mesh Y", float(self.mesh_offset[1]), -10.0, 10.0)
        mesh_changed = mesh_changed or c
        c, self.mesh_offset[2] = imgui.slider_float("Mesh Z", float(self.mesh_offset[2]), -10.0, 10.0)
        mesh_changed = mesh_changed or c
        c, self.mesh_rotation[0] = imgui.slider_float("Mesh Rot X", float(self.mesh_rotation[0]), -math.pi, math.pi)
        mesh_changed = mesh_changed or c
        c, self.mesh_rotation[1] = imgui.slider_float("Mesh Rot Y", float(self.mesh_rotation[1]), -math.pi, math.pi)
        mesh_changed = mesh_changed or c
        c, self.mesh_rotation[2] = imgui.slider_float("Mesh Rot Z", float(self.mesh_rotation[2]), -math.pi, math.pi)
        mesh_changed = mesh_changed or c
        if mesh_changed:
            print(
                f"Mesh transform: offset=({self.mesh_offset[0]:.3f}, {self.mesh_offset[1]:.3f}, {self.mesh_offset[2]:.3f})"
                f"  rot=({self.mesh_rotation[0]:.3f}, {self.mesh_rotation[1]:.3f}, {self.mesh_rotation[2]:.3f})"
            )
            self._rebuild_vessel_mesh()

        imgui.separator()
        imgui.text(f"Sim time: {self.sim_time:.2f}s")
        mode = "ALIGN VESSEL" if self.align_vessel_mode else "DRIVE CATHETER"
        imgui.text(f"Mode: {mode}   (press V to switch)")
        if self.align_vessel_mode:
            imgui.text("Vessel: I/K=X J/L=Y U/O=Z  ,/.=yaw [/]=pitch ;/'=roll")
        else:
            imgui.text("Catheter: I/K=insert  J/L=rotate  +/-=bend tip  G=gravity")
            imgui.text(f"          W=wireframe  E=cycle entry ({self._entry_idx})  R=record IVUS pose")
        rec = f"  ● REC {len(self._record_rows)}" if self._recording else ""
        imgui.text(f"Recording: {'ON' + rec if self._recording else 'off'} (R to toggle)")

    # ----- Test -----

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Particle positions must stay finite"
        if self.collision_enabled and self.collision_mesh is not None:
            ps = self.rod_particle_start
            signed_distance_wp, hit_wp = compute_signed_distances(
                self.state_0.particle_q[ps : ps + NUM_POINTS],
                self.collision_mesh.id,
                sign_scale=self.solver.sign_scale,
                max_dist=self.solver.max_dist,
                device=self.device,
            )
            signed_distance = signed_distance_wp.numpy()
            hit = hit_wp.numpy().astype(bool)
            inv_mass = self.ws.inv_masses_wp.numpy()
            active_hits = hit & (inv_mass > 0.0)
            assert np.all(signed_distance[active_hits] <= 1.0e-3), (
                "Contained catheter particles must not end outside the vessel"
            )


def _build_parser():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Vessel mesh file (STL/OBJ/PLY/glTF/USD). Defaults to stls/pig_cta.stl.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=None,
        help="Uniform scale for the vessel mesh (default: auto-fit to the catheter size).",
    )
    parser.add_argument(
        "--mesh-prim-path",
        type=str,
        default=None,
        help="For USD vessels, the Mesh prim path to load (default: first Mesh prim).",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=None,
        help="Catheter wire stiffness as Young's modulus [Pa] (default: 1e6).",
    )
    parser.add_argument(
        "--mesh-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="World-space translation of the vessel mesh [m]. Aligns it to the catheter track.",
    )
    parser.add_argument(
        "--mesh-rotation",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        default=None,
        help="Euler XYZ rotation of the vessel mesh [rad].",
    )
    parser.add_argument(
        "--entry",
        type=int,
        default=None,
        metavar="N",
        help="Align open-end index N (0=largest ring) to the rod tip. Press E in-app to cycle.",
    )
    parser.add_argument(
        "--sign-scale",
        type=float,
        default=None,
        choices=(-1.0, 1.0),
        help="Force containment sign (mesh winding). Default: auto-detected from the lumen.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        metavar="REC",
        help="Replay a recorded sensor trajectory (.npz/.csv) as an animation instead of simulating.",
    )
    return parser


def _peek_replay_mesh(path: str):
    """Read the mesh filename a recording was made against (.npz or .csv)."""
    if path.endswith(".npz"):
        return str(np.load(path)["mesh"])
    with open(path) as fh:
        for line in fh:
            if line.startswith("# mesh="):
                return line.split("mesh=", 1)[1].split()[0]
    return None


if __name__ == "__main__":
    parser = _build_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
