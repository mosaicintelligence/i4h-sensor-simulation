"""Vessel-phantom evaluation for the calibrated Volcano s5i IVUS simulator.

Runs short, visually-oriented scenarios on the *calibrated* simulator
configured by `volcano_s5i.yaml`:

  1. Centered probe, single-wall cylinder (existing 4 mm radius mesh) --
     clean specular boundary test on the bench phantom geometry.
  2. Centered probe, thick-wall clinical coronary (newly generated 2.5 /
     3.0 mm two-mesh phantom; lumen -> vessel_wall -> extravascular).
  3. Eccentric probe (1.0 mm offset) in the coronary phantom --
     depth-of-boundary varies with angle.
  4. Pullback over 5 frames along the catheter long axis in the
     coronary phantom -- reproducibility check.
  5. Calcified plaque (sphere-only geometry: vessel wall ring + two
     bright eccentric calcific inclusions modelled as `bone`).

For each scenario we save:
  - A single-frame palette image (Cartesian + unwrapped), showing the
    raw speckle texture and ring-down.
  - An n_frames-averaged palette image (Cartesian), with the per-frame
    catheter yaw jittered to suppress random speckle and expose the
    deterministic geometry / boundaries.

All renders use the calibrated:
  - probe params (10 MHz, 256 scanlines, calibrated focal_length /
    element_radius);
  - processing chain (TGC, log-multiplier, soft-reject, saturation);
  - envelope-domain noise (`processing.envelope_noise.sigma`);
  - ring-down injection (`processing.ring_down.{amplitude, extent_mm,
    waveform_path}`).

Output goes to `vessel_evaluation_output/` (or `--output-dir`).  A
short Markdown report (`VESSEL_EVALUATION_REPORT.md`) is written
alongside the images.

Run on a machine with the built raysim CUDA extension:

    cd <workspace>
    python instrument-calibration/p035_visions/vessel_evaluation.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Non-interactive backend for headless runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the calibrated config-loader, world builders, frame renderer,
# and Cartesian-resampler from the Tier 1 harness.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from tier1_evaluation import (  # noqa: E402
    load_calibrated_config,
    make_probe_at,
    render_frames,
    polar_to_cart_display,
)

# Mesh helpers from the legacy IVUS example (Cylinder.obj single-wall
# at r = 4 mm + Cylinder_inner.obj / Cylinder_outer.obj thick wall).
_EXAMPLES_DIR = (_HERE / ".." / ".." / "i4h-sensor-simulation"
                 / "ultrasound-raytracing" / "examples").resolve()
sys.path.insert(0, str(_EXAMPLES_DIR))
from ivus_example import _mesh_path  # noqa: E402

MESH_DIR = (_HERE / ".." / ".." / "i4h-sensor-simulation"
            / "ultrasound-raytracing" / "mesh").resolve()


# ---------------------------------------------------------------------------
# Mesh generation: inline cylinder OBJ writer (no caps, inward-normals).
# ---------------------------------------------------------------------------

def _write_open_cylinder_obj(path: Path, *, radius_mm: float,
                             length_mm: float = 4.0,
                             num_segments: int = 129,
                             radial_jitter_mm: float = 0.020,
                             normal_jitter_rad: float = 0.10,
                             seed: int = 1) -> None:
    """Write an open (no caps) cylinder mesh as Wavefront OBJ.

    Geometry layout (matches `utils/phantom_maker.py::generate_cylinder_mesh`):
      - Two vertex rings at y = -length/2 (ring 0) and y = +length/2 (ring 1).
      - Per-vertex normals point inward (toward axis) so rays from the probe
        at r = 0 hit the front face of the cylinder.
      - Face format v//vn with the inward-winding triangle convention.
      - Segment count 129 avoids alignment with the simulator's 256 rays.

    Surface roughness:
      - `radial_jitter_mm`: per-vertex random radial perturbation (uniform in
        +/- jitter).  Default 20 µm models sub-wavelength wall roughness at
        10 MHz (lambda ~ 150 µm).
      - `normal_jitter_rad`: per-vertex random tilt added to the inward normal
        (uniform in +/- jitter, applied in the local tangent + axial plane).
        Default 0.10 rad ~ 5.7° spreads the specular peak so the centered-
        probe-on-cylinder geometry no longer produces a uniformly saturated
        rim (rays hit at varying small off-normal angles).

    The geometric jitter is small enough not to perturb the apparent wall
    radius (axial PSF FWHM ~300 µm at 10 MHz dominates), but large enough to
    de-saturate the model's `cos^n` specular peak which is otherwise stuck
    at its maximum value for any concentric cylinder + centered probe.

    Setting both jitter knobs to 0 reproduces the original perfect cylinder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    half = length_mm / 2.0
    rng = np.random.default_rng(seed)
    verts: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    for ring in (0, 1):
        y = -half if ring == 0 else half
        for i in range(num_segments):
            theta = 2.0 * math.pi * i / num_segments
            r = radius_mm + float(rng.uniform(-radial_jitter_mm, radial_jitter_mm))
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            verts.append((x, y, z))
            # Inward normal (toward axis), with tangent-plane jitter.
            # Base inward unit vector:
            nx0, nz0 = -math.cos(theta), -math.sin(theta)
            # Tangent unit vector (perpendicular in xz plane):
            tx, tz = -math.sin(theta), math.cos(theta)
            # Random tilt in the tangent + axial direction:
            dtheta_tan = float(rng.uniform(-normal_jitter_rad, normal_jitter_rad))
            dtheta_ax = float(rng.uniform(-normal_jitter_rad, normal_jitter_rad))
            nx = nx0 + dtheta_tan * tx
            ny = dtheta_ax
            nz = nz0 + dtheta_tan * tz
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            normals.append((nx / norm, ny / norm, nz / norm))

    with path.open("w") as f:
        f.write(f"# Open cylinder, r = {radius_mm:.3f} mm, "
                f"length = {length_mm:.3f} mm, segments = {num_segments}\n")
        f.write(f"# Surface roughness: radial_jitter = {radial_jitter_mm:.4f} mm, "
                f"normal_jitter = {normal_jitter_rad:.4f} rad, seed = {seed}\n")
        f.write("# Axis along Y; normals inward (with jitter) for imaging from lumen\n\n")
        f.write("o Cylinder\n\n")
        for (x, y, z) in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for (nx, ny, nz) in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("\n")
        for i in range(num_segments):
            i_next = (i + 1) % num_segments
            v0 = i + 1
            v1 = i_next + 1
            v2 = i_next + num_segments + 1
            v3 = i + num_segments + 1
            f.write(f"f {v0}//{v0} {v2}//{v2} {v1}//{v1}\n")
            f.write(f"f {v0}//{v0} {v3}//{v3} {v2}//{v2}\n")


def _ensure_coronary_meshes(inner_r_mm: float = 2.5,
                             outer_r_mm: float = 3.0) -> tuple[Path, Path]:
    """Generate (and cache) a clinical-scale thick-wall coronary mesh.

    Filenames are tagged with the (inner, outer) radii so we don't
    overwrite the bench-scale `Cylinder_inner.obj` / `Cylinder_outer.obj`.
    Returns (inner_path, outer_path).
    """
    tag = f"r{inner_r_mm:.1f}_{outer_r_mm:.1f}".replace(".", "p")
    inner_path = MESH_DIR / f"Cylinder_coronary_{tag}_inner.obj"
    outer_path = MESH_DIR / f"Cylinder_coronary_{tag}_outer.obj"
    if not inner_path.is_file():
        _write_open_cylinder_obj(inner_path, radius_mm=inner_r_mm)
    if not outer_path.is_file():
        _write_open_cylinder_obj(outer_path, radius_mm=outer_r_mm)
    return inner_path, outer_path


# ---------------------------------------------------------------------------
# World builders (vessel-flavoured).
# ---------------------------------------------------------------------------

def build_single_wall_world(materials):
    """Bench-scale single-wall phantom (existing 4 mm radius Cylinder.obj).

    World background = lumen (catheter sits in blood-like medium); the
    cylinder surface uses vessel_wall material so the boundary echo
    looks like the lumen-to-wall interface of a real vessel.
    Everything outside the cylinder is still lumen (no extravascular
    layer in this scenario -- this is a geometric-accuracy reference).
    """
    import raysim as rs
    from raysim.ray_sim_python import Mesh

    cyl_path = _mesh_path("Cylinder.obj")
    if not os.path.isfile(cyl_path):
        raise FileNotFoundError(
            f"Cylinder.obj not found at {cyl_path}.  Generate with: "
            "python utils/phantom_maker.py cylinder --output mesh")
    world = rs.World("lumen")
    world.add(Mesh(cyl_path, materials.get_index("vessel_wall")))
    return world


def build_thick_wall_world(materials, *, inner_r_mm: float, outer_r_mm: float):
    """Clinical-scale thick-wall coronary phantom.

    Two concentric meshes at the requested radii; ray traversal
    encounters: lumen -> vessel_wall -> extravascular.
    """
    import raysim as rs
    from raysim.ray_sim_python import Mesh

    inner_path, outer_path = _ensure_coronary_meshes(inner_r_mm, outer_r_mm)
    world = rs.World("lumen")
    world.add(Mesh(str(inner_path), materials.get_index("vessel_wall")))
    world.add(Mesh(str(outer_path), materials.get_index("extravascular")))
    return world


def build_plaque_world(materials, *, wall_r_mm: float = 3.0,
                       n_wall_segments: int = 72,
                       wall_segment_radius_mm: float = 0.18):
    """Vessel wall approximated by a dense ring of spheres + two
    eccentric calcified inclusions modelled as `bone` (high impedance).

    Sphere geometry instead of mesh because OptiX cannot mix mesh and
    sphere primitives in a single GAS (acceleration structure).

    Calcific inclusions sit just inside the wall (between r=2.4 and
    r=2.7 mm) at azimuths +60 deg and -45 deg from north -- chosen to
    show eccentric brightening and an acoustic shadow.
    """
    import raysim as rs
    from raysim.ray_sim_python import Sphere

    world = rs.World("lumen")
    wall_mat = materials.get_index("vessel_wall")
    calc_mat = materials.get_index("bone")

    for i in range(n_wall_segments):
        theta = 2.0 * math.pi * i / n_wall_segments
        x = wall_r_mm * math.cos(theta)
        z = wall_r_mm * math.sin(theta)
        world.add(
            Sphere(
                np.array([x, 0.0, z], dtype=np.float32),
                wall_segment_radius_mm,
                wall_mat,
            )
        )

    plaque_specs = [
        # (r_mm, theta_deg_polar, radius_mm)
        (2.55, +60.0, 0.30),   # large calcific nodule, eccentric
        (2.65, -45.0, 0.22),   # smaller bright spot
    ]
    for r_mm, theta_deg, radius_mm in plaque_specs:
        theta = math.radians(theta_deg)
        x = r_mm * math.sin(theta)
        z = r_mm * math.cos(theta)
        world.add(
            Sphere(
                np.array([x, 0.0, z], dtype=np.float32),
                radius_mm,
                calc_mat,
            )
        )
    return world


# ---------------------------------------------------------------------------
# Rendering + plotting helpers.
# ---------------------------------------------------------------------------

def _render_single_and_mean(cfg, materials, sim_params, world_builder,
                            *, n_frames: int = 8,
                            position_mm=(0.0, 0.0, 0.0),
                            ring_down_enabled: bool = True):
    """Render n_frames at the requested probe pose and return both the
    first single frame and the per-pixel mean across n_frames.

    ``world_builder`` is a zero-argument callable that returns a fresh
    ``World`` object.  A new world + simulator is constructed for every
    frame because the C++ World object holds GPU state that the
    simulator's destructor mutates — reusing the same world across
    multiple RaytracingUltrasoundSimulator constructions triggers
    OPTIX_ERROR_INVALID_VALUE (same issue documented in
    tier1_evaluation.py::test_ringdown).

    Per-frame catheter-yaw jitter randomises the speckle realisation,
    so the mean preserves geometric structure (specular wall echoes,
    plaque, ring-down) and suppresses random speckle by ~sqrt(N).
    """
    import raysim as rs

    n_theta = int(cfg.sim.b_mode_size[0])
    n_r = int(cfg.sim.b_mode_size[1])
    rng = np.random.default_rng(2024)
    frames = []
    orig_rd = sim_params.ring_down.enabled
    sim_params.ring_down.enabled = ring_down_enabled
    for k in range(n_frames):
        sim_params.frame_seed = int(k + 1)
        sim_params.noise_seed = int(k + 1)
        yaw = 0.0 if n_frames == 1 else float(rng.uniform(-0.005, 0.005))
        probe = make_probe_at(cfg, position_mm=position_mm,
                              rotation_rad=(0.0, yaw, 0.0))
        # Fresh world + simulator per frame avoids OptiX GPU-state corruption
        # when the previous simulator is destroyed between frames.
        world = world_builder()
        sim = rs.RaytracingUltrasoundSimulator(world, materials)
        b_mode = np.asarray(sim.simulate(probe, sim_params))
        if b_mode.shape == (n_r, n_theta):
            b_mode = b_mode.T
        elif b_mode.shape != (n_theta, n_r):
            raise RuntimeError(
                f"Unexpected b_mode shape {b_mode.shape}; "
                f"expected ({n_theta}, {n_r}) or ({n_r}, {n_theta})."
            )
        frames.append(b_mode)
    sim_params.ring_down.enabled = orig_rd  # restore
    stack = np.stack(frames, axis=0)
    return stack[0], stack.mean(axis=0)


def acoustic_boundary_offset_mm(cfg) -> float:
    """Distal shift (mm) from a geometric interface to the visible texture edge.

    Bulk backscatter from the distal medium plus axial PSF smearing make the
    apparent speckle transition sit slightly farther from the probe than the
    mesh surface.  We approximate that shift as half the Tier-1 calibrated
    axial PSF FWHM, scaled with pulse length and frequency:

        FWHM ~ 0.318 mm at 10 MHz / 15 cycles (Test C anchor).
    """
    f_mhz = max(float(cfg.probe.frequency_mhz), 1e-6)
    n_cycles = float(cfg.probe.pulse_duration_cycles)
    ref_fwhm_mm = 0.318
    axial_fwhm_mm = ref_fwhm_mm * (n_cycles / 15.0) * (10.0 / f_mhz)
    return 0.5 * axial_fwhm_mm


def _draw_ring_overlays(ax, ring_overlays: list[tuple[float, str, str]], *,
                        acoustic_offset_mm: float | None,
                        polar: bool = False) -> None:
    """Draw geometric boundary rings and optional acoustic-boundary rings."""
    theta = np.linspace(0, 2 * np.pi, 512)
    for (r_mm, color, label) in ring_overlays:
        if polar:
            ax.axhline(r_mm, color=color, lw=0.9, ls="--", alpha=0.85,
                       label=f"{label} geom (r = {r_mm:.2f} mm)")
        else:
            ax.plot(r_mm * np.cos(theta), r_mm * np.sin(theta),
                    color=color, lw=1.0, ls="--", alpha=0.85,
                    label=f"{label} geom (r = {r_mm:.2f} mm)")
        if acoustic_offset_mm is not None:
            r_ac = r_mm + acoustic_offset_mm
            if polar:
                ax.axhline(r_ac, color=color, lw=0.9, ls=":", alpha=0.75,
                           label=(f"{label} acoustic "
                                  f"(r = {r_ac:.2f} mm)"))
            else:
                ax.plot(r_ac * np.cos(theta), r_ac * np.sin(theta),
                        color=color, lw=1.0, ls=":", alpha=0.75,
                        label=(f"{label} acoustic "
                               f"(r = {r_ac:.2f} mm)"))


def _save_scenario_figure(single_frame, mean_frame, *, t_far_mm: float,
                          title: str, subtitle: str, out_path: Path,
                          ring_overlays: list[tuple[float, str, str]] | None = None,
                          display_r_mm: float | None = None,
                          cfg=None,
                          show_acoustic_boundaries: bool = True):
    """Save a 2x2 figure for one scenario.

    Layout:
        (0, 0)  Cartesian polar B-mode, single frame  -- raw speckle
        (0, 1)  Cartesian polar B-mode, mean of N     -- geometry crisp
        (1, 0)  Unwrapped polar (theta vs r), single frame
        (1, 1)  Unwrapped polar (theta vs r), mean of N

    ``display_r_mm``: clip the Cartesian and unwrapped axes to this
    radius (mm) so the vessel fills the panel.  Defaults to 2× the
    outermost ring_overlay radius, or 10 mm if no overlays given.

    Boundary overlays (when ``ring_overlays`` is set):
        ``--`` dashed: geometric mesh surface (ground truth from OBJ).
        ``:`` dotted: acoustic boundary (geom + half axial PSF FWHM) --
        where bulk-speckle texture change is expected to appear.
    """
    acoustic_offset_mm = None
    if show_acoustic_boundaries and ring_overlays and cfg is not None:
        acoustic_offset_mm = acoustic_boundary_offset_mm(cfg)

    # Determine how far out to zoom.
    if display_r_mm is None:
        if ring_overlays:
            max_r = max(r for (r, _, _) in ring_overlays)
            display_r_mm = max_r * 3.0
        else:
            display_r_mm = 10.0
    display_r_mm = min(display_r_mm, t_far_mm)

    cart_single = polar_to_cart_display(single_frame, t_far_mm=t_far_mm,
                                        cart_size=512)
    cart_mean = polar_to_cart_display(mean_frame, t_far_mm=t_far_mm,
                                      cart_size=512)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    extent_cart = (-t_far_mm, t_far_mm, t_far_mm, -t_far_mm)

    for ax, arr, panel_title in [
        (axes[0, 0], cart_single, "Cartesian, single frame"),
        (axes[0, 1], cart_mean, "Cartesian, mean of N"),
    ]:
        ax.imshow(np.ma.masked_invalid(arr), cmap="gray",
                  vmin=0.0, vmax=255.0, extent=extent_cart)
        ax.set_xlim(-display_r_mm, display_r_mm)
        ax.set_ylim(display_r_mm, -display_r_mm)   # y increases downward
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(panel_title, fontsize=10)
        if ring_overlays:
            _draw_ring_overlays(ax, ring_overlays,
                                acoustic_offset_mm=acoustic_offset_mm,
                                polar=False)
            ax.legend(loc="lower right", fontsize=6, framealpha=0.7)
        # Radial grid rings at sensible intervals within the display window.
        grid_step = 2.0 if display_r_mm <= 12 else 5.0
        r_grid = grid_step
        while r_grid <= display_r_mm:
            theta = np.linspace(0, 2 * np.pi, 256)
            ax.plot(r_grid * np.cos(theta), r_grid * np.sin(theta),
                    color="0.5", lw=0.3, ls=":", alpha=0.4)
            r_grid += grid_step

    unwrap_extent = (0.0, 360.0, t_far_mm, 0.0)
    for ax, arr, panel_title in [
        (axes[1, 0], single_frame, "Unwrapped, single frame"),
        (axes[1, 1], mean_frame, "Unwrapped, mean of N"),
    ]:
        ax.imshow(arr.T, cmap="gray", vmin=0.0, vmax=255.0,
                  extent=unwrap_extent, aspect="auto")
        ax.set_ylim(display_r_mm, 0.0)
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("radial depth (mm)")
        ax.set_title(panel_title, fontsize=10)
        if ring_overlays:
            _draw_ring_overlays(ax, ring_overlays,
                                acoustic_offset_mm=acoustic_offset_mm,
                                polar=True)
            ax.legend(loc="upper right", fontsize=6, framealpha=0.7)

    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------

SCENARIO_METADATA: dict[str, dict] = {}


def scenario_01_single_wall(cfg, materials, sim_params, out_dir: Path,
                             n_frames: int):
    single, mean = _render_single_and_mean(
        cfg, materials, sim_params,
        lambda: build_single_wall_world(materials),
        n_frames=n_frames,
    )
    out_path = out_dir / "01_single_wall.png"
    _save_scenario_figure(
        single, mean,
        t_far_mm=float(cfg.sim.t_far_mm),
        title="Scenario 1 -- centered probe, single-wall (4 mm cylinder)",
        subtitle=("World: lumen background, vessel_wall mesh at r = 4 mm. "
                  "No extravascular layer; outside the wall is also lumen. "
                  f"N = {n_frames} frames averaged."),
        out_path=out_path,
        ring_overlays=[(4.0, "tab:red", "vessel wall")],
        display_r_mm=10.0,
        cfg=cfg,
    )
    return out_path


SCENARIO_METADATA["01_single_wall"] = {
    "name": "Single-wall (4 mm)",
    "purpose": (
        "Geometric accuracy reference at the bench-phantom scale.  A "
        "single specular interface at r = 4 mm tests that the calibrated "
        "PSF + ring-down + TGC pipeline renders a clean closed ring at "
        "the right depth, with no angular drop-outs or systematic depth "
        "errors."
    ),
    "expected": (
        "Closed bright ring at r = 4 mm in both the single-frame and "
        "averaged renders.  Ring-down disc at r ~ 1.8 mm in every panel. "
        "Inside the wall (r < 4 mm) is the lumen speckle background; "
        "outside is the same speckle (no impedance change)."
    ),
}


def scenario_02_thick_wall_coronary(cfg, materials, sim_params, out_dir: Path,
                                     n_frames: int,
                                     inner_r_mm: float = 2.5,
                                     outer_r_mm: float = 3.0):
    single, mean = _render_single_and_mean(
        cfg, materials, sim_params,
        lambda: build_thick_wall_world(materials,
                                       inner_r_mm=inner_r_mm,
                                       outer_r_mm=outer_r_mm),
        n_frames=n_frames,
    )
    out_path = out_dir / "02_thick_wall_coronary.png"
    _save_scenario_figure(
        single, mean,
        t_far_mm=float(cfg.sim.t_far_mm),
        title=("Scenario 2 -- centered probe, thick-wall coronary "
               f"(inner r = {inner_r_mm:.1f} mm, outer r = {outer_r_mm:.1f} mm)"),
        subtitle=("World: lumen -> vessel_wall (0.5 mm thick) -> "
                  "extravascular.  Clinical-scale phantom; ring-down at "
                  "r ~ 1.8 mm sits just inside the inner wall.  "
                  f"N = {n_frames} frames averaged."),
        out_path=out_path,
        ring_overlays=[
            (inner_r_mm, "tab:red", "inner wall"),
            (outer_r_mm, "tab:orange", "outer wall"),
        ],
        display_r_mm=8.0,
        cfg=cfg,
    )
    return out_path


SCENARIO_METADATA["02_thick_wall_coronary"] = {
    "name": "Thick-wall coronary (2.5 / 3.0 mm)",
    "purpose": (
        "Clinical-scale vessel rendering.  Two concentric specular "
        "interfaces (lumen->wall at r=2.5 mm, wall->extravascular at "
        "r=3.0 mm) test depth-dependent attenuation and contrast.  The "
        "extravascular speckle should attenuate with depth via the "
        "calibrated `attenuation_db_per_cm_mhz` + TGC schedule."
    ),
    "expected": (
        "Two concentric rings ~0.5 mm apart, inner at r=2.5 mm.  The "
        "inner ring sits just outside the ring-down disc (r ~ 1.8 mm).  "
        "Extravascular speckle fills the FOV beyond r=3 mm and "
        "attenuates smoothly to background by r ~ 20-25 mm."
    ),
}


def scenario_03_eccentric(cfg, materials, sim_params, out_dir: Path,
                           n_frames: int,
                           inner_r_mm: float = 2.5,
                           outer_r_mm: float = 3.0,
                           offset_mm: float = 1.0):
    world = build_thick_wall_world(materials,
                                   inner_r_mm=inner_r_mm,
                                   outer_r_mm=outer_r_mm)
    # Move the probe off-axis along +x.  The cylinder is centred at the
    # origin, so the closest inner-wall point is at (inner_r - offset).
    single, mean = _render_single_and_mean(
        cfg, materials, sim_params,
        lambda: build_thick_wall_world(materials,
                                       inner_r_mm=inner_r_mm,
                                       outer_r_mm=outer_r_mm),
        n_frames=n_frames,
        position_mm=(float(offset_mm), 0.0, 0.0),
    )
    out_path = out_dir / "03_eccentric.png"
    _save_scenario_figure(
        single, mean,
        t_far_mm=float(cfg.sim.t_far_mm),
        title=(f"Scenario 3 -- eccentric probe ({offset_mm:.1f} mm "
               "off-axis along +x) in coronary phantom"),
        subtitle=(f"World: same thick-wall coronary as scenario 2 "
                  f"(inner r = {inner_r_mm:.1f} mm, outer r = "
                  f"{outer_r_mm:.1f} mm).  Catheter shifted +{offset_mm:.1f} "
                  "mm.  Boundary depth varies sinusoidally with angle.  "
                  f"N = {n_frames} frames averaged."),
        out_path=out_path,
        ring_overlays=[
            (inner_r_mm, "tab:red", "inner wall"),
            (outer_r_mm, "tab:orange", "outer wall"),
        ],
        display_r_mm=8.0,
        cfg=cfg,
    )
    return out_path


SCENARIO_METADATA["03_eccentric"] = {
    "name": "Eccentric probe (1.0 mm offset)",
    "purpose": (
        "Geometric accuracy under off-axis catheter pose.  The inner-"
        "wall depth must vary sinusoidally with angle, from "
        "inner_r - offset on the +x side (closest) to inner_r + offset "
        "on the -x side (farthest)."
    ),
    "expected": (
        "Both rings remain closed but oscillate in depth across angle.  "
        "For offset=1.0 mm and inner_r=2.5 mm the inner wall ranges "
        "from r=1.5 mm (against the catheter, possibly buried in the "
        "ring-down) to r=3.5 mm.  No angular drop-outs."
    ),
}


def scenario_04_pullback(cfg, materials, sim_params, out_dir: Path,
                          n_frames: int,
                          n_pullback: int = 5,
                          z_range_mm: tuple[float, float] = (-1.0, 1.0),
                          inner_r_mm: float = 2.5,
                          outer_r_mm: float = 3.0):
    zs = np.linspace(z_range_mm[0], z_range_mm[1], n_pullback)
    cart_size = 350
    fig, axes = plt.subplots(1, n_pullback,
                             figsize=(3.0 * n_pullback, 3.4),
                             squeeze=False)
    extent_cart = (-cfg.sim.t_far_mm, cfg.sim.t_far_mm,
                    cfg.sim.t_far_mm, -cfg.sim.t_far_mm)
    for i, z in enumerate(zs):
        _, mean_frame = _render_single_and_mean(
            cfg, materials, sim_params,
            lambda: build_thick_wall_world(materials,
                                           inner_r_mm=inner_r_mm,
                                           outer_r_mm=outer_r_mm),
            n_frames=n_frames,
            position_mm=(0.0, 0.0, float(z)),
        )
        cart = polar_to_cart_display(mean_frame,
                                     t_far_mm=float(cfg.sim.t_far_mm),
                                     cart_size=cart_size)
        ax = axes[0, i]
        ax.imshow(np.ma.masked_invalid(cart), cmap="gray",
                  vmin=0.0, vmax=255.0, extent=extent_cart)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        if i == 0:
            ax.set_ylabel("y (mm)")
        ax.set_title(f"z = {z:+.2f} mm", fontsize=10)
        # Ring overlays (ground truth).
        theta = np.linspace(0, 2 * np.pi, 256)
        ax.plot(inner_r_mm * np.cos(theta), inner_r_mm * np.sin(theta),
                color="tab:red", lw=0.6, alpha=0.7)
        ax.plot(outer_r_mm * np.cos(theta), outer_r_mm * np.sin(theta),
                color="tab:orange", lw=0.6, alpha=0.7)
        # Constrain FOV for readability.
        zoom = 8.0
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(zoom, -zoom)

    fig.suptitle(
        f"Scenario 4 -- pullback over {n_pullback} z positions in "
        f"coronary phantom  (inner r = {inner_r_mm:.1f} mm, outer r = "
        f"{outer_r_mm:.1f} mm)\n"
        "Same geometry at each z; consistency check.  Each panel is "
        f"the mean of N = {n_frames} frames.  Cropped to +/-8 mm.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = out_dir / "04_pullback.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


SCENARIO_METADATA["04_pullback"] = {
    "name": "Pullback (5 z positions)",
    "purpose": (
        "Reproducibility / consistency along the catheter long axis.  "
        "The cylinder is translationally symmetric in y (the catheter "
        "axis), so each z position should render an identical cross-"
        "section.  Detects z-dependent artefacts (e.g. elevational PSF "
        "boundary effects, mesh-length truncation, slice-thickness "
        "leakage)."
    ),
    "expected": (
        "Identical concentric rings in every z panel; no systematic "
        "drift in ring depth or contrast.  End-of-mesh truncation may "
        "be visible at the most extreme z positions if the cylinder "
        "length (4 mm) is comparable to the elevational PSF."
    ),
}


def scenario_05_plaque(cfg, materials, sim_params, out_dir: Path,
                        n_frames: int):
    single, mean = _render_single_and_mean(
        cfg, materials, sim_params,
        lambda: build_plaque_world(materials),
        n_frames=n_frames,
    )
    out_path = out_dir / "05_calcified_plaque.png"
    _save_scenario_figure(
        single, mean,
        t_far_mm=float(cfg.sim.t_far_mm),
        title="Scenario 5 -- calcified plaque (sphere-only vessel wall)",
        subtitle=("World: lumen background, vessel_wall ring of "
                  "72 spheres at r = 3.0 mm + two `bone` (calcific) "
                  "inclusions at (r=2.55, theta=+60 deg) and "
                  "(r=2.65, theta=-45 deg).  `bone` impedance = "
                  "7.80 MRayl gives a strong specular echo and a "
                  f"plausible acoustic shadow behind the nodule.  "
                  f"N = {n_frames} frames averaged."),
        out_path=out_path,
        ring_overlays=[(3.0, "tab:red", "vessel wall")],
        display_r_mm=8.0,
        cfg=cfg,
    )
    return out_path


SCENARIO_METADATA["05_calcified_plaque"] = {
    "name": "Calcified plaque (eccentric calcific inclusions)",
    "purpose": (
        "Contrast + shadow demonstration.  Two `bone`-material spheres "
        "embedded at the lumen-wall interface produce specular echoes "
        "much brighter than the surrounding wall + a radial shadow "
        "behind each (sound that reflects off the inclusion does not "
        "penetrate, so the wall behind the calcific spot reads at the "
        "noise floor)."
    ),
    "expected": (
        "The vessel-wall ring is visible at r=3.0 mm but interrupted "
        "by two bright specular spots near r=2.55 / 2.65 mm.  Each "
        "bright spot is followed by a wedge-shaped low-intensity "
        "region (acoustic shadow) extending outward to the edge of "
        "the FOV.  The shadow widens with depth because the "
        "calcific blocker subtends a larger angle from the probe at "
        "closer radii."
    ),
}


# ---------------------------------------------------------------------------
# Scenario 06: femoral artery (large peripheral vessel)
# ---------------------------------------------------------------------------

def scenario_06_femoral(cfg, materials, sim_params, out_dir: Path,
                         n_frames: int,
                         inner_r_mm: float = 4.5,
                         outer_r_mm: float = 5.3):
    """Femoral-scale vessel: ~9 mm lumen diameter, 0.8 mm wall.

    A femoral artery (superficial femoral or common femoral) has a typical
    lumen diameter of 6–10 mm.  At inner_r = 4.5 mm the lumen spans from
    the ring-down edge (~1.8 mm) out to 4.5 mm — a 2.7 mm gap of visible
    lumen speckle — making the vessel anatomy obvious even without zooming.
    """
    single, mean = _render_single_and_mean(
        cfg, materials, sim_params,
        lambda: build_thick_wall_world(materials,
                                       inner_r_mm=inner_r_mm,
                                       outer_r_mm=outer_r_mm),
        n_frames=n_frames,
    )
    out_path = out_dir / "06_femoral.png"
    _save_scenario_figure(
        single, mean,
        t_far_mm=float(cfg.sim.t_far_mm),
        title=(f"Scenario 6 -- femoral-scale vessel "
               f"(inner r = {inner_r_mm:.1f} mm, outer r = {outer_r_mm:.1f} mm)"),
        subtitle=(f"Lumen diameter = {2*inner_r_mm:.0f} mm; wall thickness = "
                  f"{outer_r_mm - inner_r_mm:.1f} mm.  "
                  "World: lumen -> vessel_wall -> extravascular.  "
                  f"N = {n_frames} frames averaged.  "
                  "Ring-down end (~1.8 mm) well inside the lumen."),
        out_path=out_path,
        ring_overlays=[
            (inner_r_mm, "tab:red", "inner wall"),
            (outer_r_mm, "tab:orange", "outer wall"),
        ],
        display_r_mm=12.0,
        cfg=cfg,
    )
    return out_path


SCENARIO_METADATA["06_femoral"] = {
    "name": "Femoral-scale vessel (9 mm diameter)",
    "purpose": (
        "Large-vessel rendering at femoral-artery scale.  Inner r = 4.5 mm "
        "(9 mm lumen diameter) leaves a 2.7 mm gap of visible lumen speckle "
        "between the ring-down edge (~1.8 mm) and the inner wall, making the "
        "vessel anatomy obvious without zooming.  Tests that the calibrated "
        "TGC + attenuation schedule handles the larger depth range correctly."
    ),
    "expected": (
        "Two concentric rings 0.8 mm apart: inner at r = 4.5 mm, outer at "
        "r = 5.3 mm.  The lumen fills the display with visible speckle from "
        "~2 mm to 4.5 mm.  Extravascular speckle beyond r = 5.3 mm "
        "attenuates smoothly.  Both rings should be equally bright (no "
        "depth-dependent ring-brightness bias from miscalibrated TGC)."
    ),
}


# ---------------------------------------------------------------------------
# Scenario 07: aortic / large-vessel scale
# ---------------------------------------------------------------------------

def scenario_07_aorta(cfg, materials, sim_params, out_dir: Path,
                       n_frames: int,
                       inner_r_mm: float = 12.0,
                       outer_r_mm: float = 13.2):
    """Aorta-scale vessel: ~24 mm lumen diameter, 1.2 mm wall.

    At 10 MHz the ring-down extends to r~3 mm.  With inner_r=12 mm there
    is a 9 mm gap of dark lumen clearly separating the ring-down disc from
    the vessel wall — the anatomy is unambiguous regardless of the ring-down
    state.  Also tests the TGC and attenuation model at 12-13 mm depth.
    """
    single, mean = _render_single_and_mean(
        cfg, materials, sim_params,
        lambda: build_thick_wall_world(materials,
                                       inner_r_mm=inner_r_mm,
                                       outer_r_mm=outer_r_mm),
        n_frames=n_frames,
    )
    out_path = out_dir / "07_aorta.png"
    _save_scenario_figure(
        single, mean,
        t_far_mm=float(cfg.sim.t_far_mm),
        title=(f"Scenario 7 -- aorta-scale vessel "
               f"(inner r = {inner_r_mm:.0f} mm, outer r = {outer_r_mm:.1f} mm)"),
        subtitle=(f"Lumen diameter = {2*inner_r_mm:.0f} mm; wall thickness = "
                  f"{outer_r_mm - inner_r_mm:.1f} mm.  "
                  "Ring-down ends at r~3 mm, leaving a 9 mm clear lumen gap.  "
                  f"N = {n_frames} frames averaged."),
        out_path=out_path,
        ring_overlays=[
            (inner_r_mm, "tab:red", "inner wall"),
            (outer_r_mm, "tab:orange", "outer wall"),
        ],
        display_r_mm=20.0,
    )
    return out_path


SCENARIO_METADATA["07_aorta"] = {
    "name": "Aorta-scale vessel (24 mm diameter)",
    "purpose": (
        "Large-vessel demo at aortic scale where the ring-down zone "
        "(r~3 mm) is far inside the lumen, leaving a clear anatomical "
        "separation between artifact and vessel wall.  Tests the sim at "
        "12-13 mm target depth where TGC and attenuation gradients are "
        "most apparent."
    ),
    "expected": (
        "Two concentric rings 1.2 mm apart: inner at r=12 mm, outer at "
        "r=13.2 mm.  Clear dark lumen from r~3 mm to 12 mm.  Both rings "
        "visible as bright lines (not a saturated disc).  Extravascular "
        "speckle beyond r=13.2 mm decreasing in brightness with depth."
    ),
}


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------

def _simulation_model_markdown(cfg) -> list[str]:
    """Markdown blocks describing the simulator and its processing pipeline."""
    f_mhz = float(cfg.probe.frequency_mhz)
    n_cycles = float(cfg.probe.pulse_duration_cycles)
    n_theta, n_r = int(cfg.sim.b_mode_size[0]), int(cfg.sim.b_mode_size[1])
    t_far = float(cfg.sim.t_far_mm)
    acoustic_delta = acoustic_boundary_offset_mm(cfg)
    lines: list[str] = []
    lines.append("## Simulation model\n\n")
    lines.append(
        "These renders use **raysim**, a GPU Monte-Carlo ultrasound simulator "
        "(NVIDIA OptiX ray tracing + CUDA post-processing).  All calibrated "
        "instrument and processing parameters come from "
        "`instrument-calibration/p035_visions/volcano_s5i.yaml`, fit against "
        "the Volcano s5i / `ivus_test_0515` bench corpus (wire PSF, milk-bath "
        "depth uniformity, cyst noise floor, ring-down template).\n\n"
    )

    lines.append("### Geometry and materials\n\n")
    lines.append(
        "Each scenario builds an IVUS **World**: a 3-D scene centred on the "
        "catheter origin with a background medium and optional mesh / sphere "
        "primitives.\n\n"
    )
    lines.append("| Layer | Material | Role |\n")
    lines.append("|---|---|---|\n")
    lines.append(
        "| Catheter lumen | `lumen` (blood-like, Z ≈ 1.68 MRayl) | "
        "Background inside the vessel; low scatter (μ₀ ≈ 0.1) |\n"
    )
    lines.append(
        "| Vessel wall | `vessel_wall` (Z ≈ 1.82 MRayl) | "
        "Thin cylindrical mesh; moderate bulk scatter |\n"
    )
    lines.append(
        "| Perivascular tissue | `extravascular` (Z ≈ 1.62 MRayl) | "
        "Outside the outer wall; speckle-filled background |\n"
    )
    lines.append(
        "| Wire targets (bench) | `tungsten` (Z ≈ 101 MRayl) | "
        "PSF / gain calibration phantoms only |\n"
    )
    lines.append(
        "| Calcific inclusions (scenario 05) | `bone` (Z ≈ 7.8 MRayl) | "
        "High-impedance stand-in for calcified plaque |\n"
    )
    lines.append("\n")
    lines.append(
        "Cylinder meshes are generated with ~20 µm radial jitter and "
        "~0.1 rad normal jitter so concentric walls do not produce an "
        "artificial uniformly-bright rim at normal incidence.\n\n"
    )

    lines.append("### Acoustic physics (OptiX trace)\n\n")
    lines.append(
        "For each of the **256 angular rays** the simulator launches GPU "
        "paths from the IVUS probe at the origin.  Along each path:\n\n"
    )
    lines.append(
        "1. **Bulk backscatter** — Bernoulli–Gaussian scatterers (μ₀, σ per "
        "material) integrated along the ray with **Beer–Lambert** depth "
        "attenuation (dB/cm/MHz × frequency).\n"
    )
    lines.append(
        "2. **Specular interfaces** — at every material boundary the scanline "
        "receives (a) the **Fresnel intensity reflection** R from the "
        "impedance contrast, plus (b) an empirical **directivity term** "
        "cosⁿ(θ) scaled by R (Mattausch 2016).  Soft-tissue interfaces "
        "(lumen→wall, R ≈ 0.2 %) therefore appear as **texture transitions**, "
        "not bright rims; high-impedance targets (tungsten, calcification) "
        "remain bright.\n"
    )
    lines.append(
        "3. **Refraction** — transmitted energy continues into the distal "
        "medium (e.g. lumen → wall → extravascular), producing bulk speckle "
        "that begins slightly **distal** to the geometric surface.\n\n"
    )
    lines.append(
        f"Probe: **{f_mhz:.0f} MHz**, pulse **{n_cycles:.0f} cycles** "
        f"(axial PSF FWHM ≈ 0.318 mm at anchor settings).  "
        f"Radial FOV **t_far = {t_far:.0f} mm** sampled at "
        f"**{n_r}** depth bins × **{n_theta}** angles.\n\n"
    )

    lines.append("### Signal-processing pipeline (execution order)\n\n")
    lines.append(
        "After ray tracing, each scanline passes through the calibrated "
        "receive chain below.  Stages run **top to bottom** on the GPU "
        "(see diagram).\n\n"
    )
    lines.append("```mermaid\n")
    lines.append("flowchart TB\n")
    lines.append('  subgraph physics ["1 — Acoustic forward model (OptiX)"]\n')
    lines.append("    W[World geometry + materials]\n")
    lines.append("    P[IVUS probe: 256 rays × t_far depth samples]\n")
    lines.append("    RT[GPU ray trace]\n")
    lines.append("    SC[Bulk scatter + Beer-Lambert attenuation]\n")
    lines.append("    IF[Fresnel R + R-scaled specular at interfaces]\n")
    lines.append("    RF[(Raw scanline buffer)]\n")
    lines.append("    W --> P --> RT\n")
    lines.append("    RT --> SC --> RF\n")
    lines.append("    RT --> IF --> RF\n")
    lines.append("  end\n")
    lines.append("\n")
    lines.append('  subgraph rfchain ["2 — RF-domain processing"]\n')
    lines.append("    N0[Pre-PSF additive RF noise]\n")
    lines.append("    AX[Axial PSF convolution]\n")
    lines.append("    LAT[Lateral PSF — depth-dependent]\n")
    lines.append("    TGC[Time-gain compensation curve]\n")
    lines.append("    GAIN[Reference gain gain_db]\n")
    lines.append("    RF --> N0 --> AX --> LAT --> TGC --> GAIN\n")
    lines.append("  end\n")
    lines.append("\n")
    lines.append('  subgraph envelope ["3 — Envelope detection"]\n')
    lines.append("    H[Hilbert transform → envelope magnitude]\n")
    lines.append("    N1[Post-envelope Gaussian noise]\n")
    lines.append("    LPF[Post-Hilbert radial low-pass]\n")
    lines.append("    RD[Ring-down template add — optional]\n")
    lines.append("    GAIN --> H --> N1 --> LPF --> RD\n")
    lines.append("  end\n")
    lines.append("\n")
    lines.append('  subgraph display ["4 — Display / B-mode"]\n')
    lines.append("    LOG[Log compression log_multiplier]\n")
    lines.append("    DW[Display window reject / saturation palette]\n")
    lines.append("    DZ[Catheter dead-zone mask]\n")
    lines.append("    SCNV[IVUS scan conversion → palette image]\n")
    lines.append("    OUT[(B-mode 0–255)]\n")
    lines.append("    RD --> LOG --> DW --> DZ --> SCNV --> OUT\n")
    lines.append("  end\n")
    lines.append("```\n\n")

    lines.append("| Stage | Calibrated parameter(s) | Notes |\n")
    lines.append("|---|---|---|\n")
    lines.append(
        "| Axial PSF | `probe.pulse_duration_cycles` | Tier 1 Test C vs "
        "tungsten wire FWHM |\n"
    )
    lines.append(
        "| Lateral PSF | `processing.lateral_psf_kernel.sigma_theta_rad` | "
        "Depth-dependent arc width; Test D |\n"
    )
    lines.append(
        "| TGC | `processing.tgc_control_points` | Depth compensation; "
        "Test H / I |\n"
    )
    lines.append(
        "| Reference gain | `processing.gain_db` | Wire-peak brightness; "
        "Test E |\n"
    )
    lines.append(
        "| Pre-PSF noise | `processing.noise.sigma` | Coherent + random "
        "speckle floor |\n"
    )
    lines.append(
        "| Post-envelope noise | `processing.envelope_noise.*` | "
        "Anechoic / deep noise floor; Test F |\n"
    )
    lines.append(
        "| Ring-down | `processing.ring_down.*` | Catheter near-field "
        "artifact; **disabled by default** in this vessel eval |\n"
    )
    lines.append(
        "| Log compression | `processing.log_multiplier`, `log_floor` | "
        "Palette mapping; E7 wire anchor |\n"
    )
    lines.append(
        "| Display window | `reject_palette`, `saturation_palette` | "
        "Device floor / ceiling |\n"
    )
    lines.append("\n")

    lines.append("### Figure overlays\n\n")
    lines.append(
        "Boundary circles / lines on each panel mark two radii per wall:\n\n"
    )
    lines.append(
        "- **Dashed (`--`)** — **geometric** mesh surface (OBJ ground truth).\n"
    )
    lines.append(
        f"- **Dotted (`:`)** — **acoustic** boundary = geometric + "
        f"Δr ≈ **{acoustic_delta:.2f} mm** (half the calibrated axial PSF "
        f"FWHM).  Bulk speckle from the distal medium and PSF smearing make "
        f"the visible texture change appear slightly distal to the mesh.\n\n"
    )
    lines.append(
        "Each scenario renders **N = 8** frames with small catheter-yaw "
        "jitter; the **mean-of-N** panels suppress random speckle while "
        "preserving deterministic structure (walls, ring-down, plaque).\n\n"
    )
    lines.append("---\n\n")
    return lines


def write_report(out_dir: Path, scenario_paths: dict[str, Path],
                 cfg) -> Path:
    report_path = out_dir / "VESSEL_EVALUATION_REPORT.md"
    lines: list[str] = []
    lines.append("# Vessel-phantom evaluation -- calibrated Volcano s5i sim\n")
    lines.append("\n")
    lines.append(
        "All renders use the calibrated `volcano_s5i.yaml` configuration "
        f"({cfg.probe.frequency_mhz:.0f} MHz, {cfg.probe.num_scanlines} "
        f"scanlines, t_far = {cfg.sim.t_far_mm:.0f} mm).  Display range is "
        "palette 0–255 (raw sim output).  See **Simulation model** below for "
        "the full physics and processing chain.\n\n"
    )
    lines.extend(_simulation_model_markdown(cfg))
    lines.append("## Scenario summary\n\n")
    lines.append("| # | Name | Image |\n")
    lines.append("|---|---|---|\n")
    for key, meta in SCENARIO_METADATA.items():
        img = scenario_paths.get(key)
        img_name = img.name if img else "*(not run)*"
        lines.append(f"| {key.split('_')[0]} | {meta['name']} | "
                     f"`{img_name}` |\n")
    lines.append("\n")

    for key, meta in SCENARIO_METADATA.items():
        lines.append(f"## {meta['name']}\n\n")
        lines.append("**Purpose.**  " + meta["purpose"] + "\n\n")
        lines.append("**Expected if the simulation is accurate.**  "
                     + meta["expected"] + "\n\n")
        img = scenario_paths.get(key)
        if img is not None:
            lines.append(f"![{meta['name']}]({img.name})\n\n")
        lines.append("---\n\n")

    report_path.write_text("".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

SCENARIO_RUNNERS = {
    "01_single_wall": scenario_01_single_wall,
    "02_thick_wall_coronary": scenario_02_thick_wall_coronary,
    "03_eccentric": scenario_03_eccentric,
    "04_pullback": scenario_04_pullback,
    "05_calcified_plaque": scenario_05_plaque,
    "06_femoral": scenario_06_femoral,
    "07_aorta": scenario_07_aorta,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Render vessel-phantom scenarios on the calibrated "
                      "Volcano s5i IVUS simulator."),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_HERE / "vessel_evaluation_output"),
        help="Output directory for images + report.",
    )
    parser.add_argument(
        "--scenarios", type=str,
        default=",".join(SCENARIO_RUNNERS.keys()),
        help="Comma-separated scenario keys to run.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=8,
        help=("Number of frames per scenario; the displayed render is "
              "the per-pixel mean (speckle suppression by ~sqrt(N)).  "
              "Single-frame panel is always saved alongside."),
    )
    parser.add_argument(
        "--ring-down", action="store_true", default=False,
        help=("Enable the calibrated ring-down artifact (off by default for "
              "vessel rendering).  Ring-down extends to r~3 mm, which buries "
              "coronary-scale vessels.  Default is OFF to reveal anatomy."),
    )
    parser.add_argument(
        "--gain-offset-db", type=float, default=0.0,
        help=("Optional adjustment to gain_db relative to the calibrated "
              "value (default 0 dB).  Use a negative offset if vessel-wall "
              "echoes saturate."),
    )
    parser.add_argument(
        "--flatten-tgc", action="store_true", default=False,
        help=("Replace the calibrated TGC schedule with a flat 0 dB profile "
              "(off by default — milk and lumen have similar acoustic "
              "properties so the milk-tuned TGC is approximately right for "
              "lumen propagation)."),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, materials, sim_params = load_calibrated_config()
    print(f"[vessel-eval] Loaded calibrated config from "
          f"{_HERE / 'volcano_s5i.yaml'}")
    print(f"[vessel-eval] Probe: {cfg.probe.frequency_mhz:.1f} MHz, "
          f"{cfg.probe.num_scanlines} scanlines, t_far = "
          f"{cfg.sim.t_far_mm:.1f} mm, "
          f"b_mode_size = {tuple(cfg.sim.b_mode_size)}")
    print(f"[vessel-eval] Ring-down: enabled = "
          f"{bool(cfg.processing.ring_down.enabled)}, amplitude = "
          f"{cfg.processing.ring_down.amplitude:.4f}, extent_mm = "
          f"{cfg.processing.ring_down.extent_mm:.2f}")

    # --- Apply vessel-rendering processing knobs ----------------------------
    # The calibrated processing block is fit to milk-bath bench tests, not
    # vessel imaging.  Three knobs need adjustment for sensible vessel renders:
    #   (a) gain_db is tuned to put milk speckle at mid-palette; clean
    #       vessel-wall specular echoes are 10-100x brighter than milk
    #       speckle and saturate.  Default offset = -15 dB.
    #   (b) TGC is tuned for milk attenuation (~0.5 dB/cm/MHz).  Lumen is
    #       0.2 dB/cm/MHz so the milk TGC over-amplifies signals coming
    #       back through the lumen.  Default: flatten TGC to 0 dB.
    #   (c) Ring-down extends to r~3 mm — buries coronary-scale vessels.
    #       Default: off for vessel rendering.
    orig_gain_db = sim_params.gain_db
    sim_params.gain_db = float(orig_gain_db + args.gain_offset_db)
    print(f"[vessel-eval] gain_db: {orig_gain_db:.2f} -> "
          f"{sim_params.gain_db:.2f} dB ({args.gain_offset_db:+.1f} dB offset)")

    if args.flatten_tgc:
        # Replace TGC control points with a flat 0 dB schedule across the FOV.
        import raysim as rs
        flat_cp = [
            rs.TgcControlPoint(depth_cm=0.0, gain_db=0.0),
            rs.TgcControlPoint(depth_cm=float(cfg.sim.t_far_mm) / 10.0,
                               gain_db=0.0),
        ]
        sim_params.tgc_control_points = flat_cp
        print("[vessel-eval] TGC: flattened to 0 dB (milk-calibrated schedule "
              "would over-amplify lumen propagation).")
    else:
        print("[vessel-eval] TGC: using calibrated (milk-tuned) schedule.")

    if args.ring_down:
        sim_params.ring_down.enabled = True
        print("[vessel-eval] Ring-down ENABLED.  Note: artifact extends to "
              "r~3 mm; coronary walls at r<3 mm are buried.")
    else:
        sim_params.ring_down.enabled = False
        print("[vessel-eval] Ring-down DISABLED (default).  Pass --ring-down "
              "to enable the calibrated near-field artifact.")

    keys = [k.strip() for k in args.scenarios.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SCENARIO_RUNNERS]
    if unknown:
        print(f"[vessel-eval] Unknown scenarios: {unknown}")
        print(f"[vessel-eval] Available: "
              f"{list(SCENARIO_RUNNERS.keys())}")
        sys.exit(1)

    scenario_paths: dict[str, Path] = {}
    for key in keys:
        print(f"[vessel-eval] Running scenario {key} ...")
        runner = SCENARIO_RUNNERS[key]
        path = runner(cfg, materials, sim_params, out_dir, args.n_frames)
        scenario_paths[key] = path
        print(f"[vessel-eval]   -> {path}")

    report_path = write_report(out_dir, scenario_paths, cfg)
    print(f"[vessel-eval] Report: {report_path}")
    print(f"[vessel-eval] All outputs in {out_dir}")


if __name__ == "__main__":
    main()
