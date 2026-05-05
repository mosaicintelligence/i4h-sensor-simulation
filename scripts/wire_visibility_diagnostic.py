#!/usr/bin/env python3
"""Per-wire visibility diagnostic for Pass 6 v2.

Question: why are wires at r ∈ {15, 20} mm invisible while r ∈ {5, 10, 25} mm
are visible? Hypotheses:
  (a) stochastic ray miss (geometry) — outer wires not hit often enough.
  (b) buried-by-noise (focal-zone hump) — focal-zone (15-20mm) bg noise std
      is highest, so wires with even a modest peak get drowned.
  (c) per-bin amplitude collapse — focal PSF concentrates wire energy
      strongly per-bin, but background is *also* concentrated, so contrast
      is constant. Visibility is purely about the variance of the local bg.

Method: render the Pass 6 v2 calibrated wire phantom AND a matched anechoic
world, then for each wire radius compute:
  * peak palette in the wire patch (per frame; median + max across frames)
  * hit-rate per frame (fraction of frames with peak > local bg+10 palette)
  * local bg mean / std at the same radial annulus (from anechoic frames)

Run from workspace root:
  python scripts/wire_visibility_diagnostic.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
SIM_PKG_ROOT = WORKSPACE / "i4h-sensor-simulation" / "ultrasound-raytracing"
CALIB_PKG_ROOT = WORKSPACE / "instrument-calibration" / "p035_visions"
YAML_PATH = CALIB_PKG_ROOT / "volcano_s5i.yaml"

sys.path.insert(0, str(SIM_PKG_ROOT))
sys.path.insert(0, str(CALIB_PKG_ROOT))

import raysim as rs  # noqa: E402
from raysim.config import IvusSimConfig  # noqa: E402
from raysim.ray_sim_python import Sphere  # noqa: E402

WIRE_RADII_MM = (5.0, 10.0, 15.0, 20.0, 25.0)
WIRE_DIAMETER_MM = 0.127
WIRE_SPHERE_RADIUS_MM = WIRE_DIAMETER_MM / 2.0
N_FRAMES = 30


def build_wire_world(materials):
    world = rs.World("lumen")
    wire_mat = materials.get_index("bone")
    positions = []
    for i, r in enumerate(WIRE_RADII_MM):
        theta = i * 2 * math.pi / len(WIRE_RADII_MM)
        x = r * math.sin(theta)
        z = r * math.cos(theta)
        center = np.array([x, 0.0, z], dtype=np.float32)
        world.add(Sphere(center, WIRE_SPHERE_RADIUS_MM, wire_mat))
        positions.append((r, math.degrees(theta) % 360.0, x, z))
    return world, positions


def build_anechoic_world(materials):
    world = rs.World("lumen")
    far_center = np.array([0.0, 0.0, 1000.0], dtype=np.float32)
    world.add(Sphere(far_center, 0.001, materials.get_index("lumen")))
    return world


def make_probe_at(cfg, rotation_rad=(0.0, 0.0, 0.0)):
    pose = rs.Pose(
        np.array(cfg.probe.pose.position_mm, dtype=np.float32),
        np.array(rotation_rad, dtype=np.float32),
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


def render_frames(cfg, world, materials, n_frames):
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    sp = cfg.to_sim_params()
    frames = []
    rng = np.random.default_rng(2024)
    for k in range(n_frames):
        yaw = float(rng.uniform(-0.005, 0.005)) if n_frames > 1 else 0.0
        probe_obj = make_probe_at(cfg, rotation_rad=(0.0, yaw, 0.0))
        sp.frame_seed = int(k + 1)
        b = sim.simulate(probe_obj, sp)
        frames.append(np.asarray(b))
    return np.stack(frames, axis=0)


def main():
    cfg = IvusSimConfig.from_yaml(YAML_PATH)
    materials = rs.Materials()

    # Polar axes
    n_theta, n_r = int(cfg.sim.b_mode_size[0]), int(cfg.sim.b_mode_size[1])
    t_far = float(cfg.sim.t_far_mm)
    dtheta_deg = 360.0 / n_theta
    dr_mm = t_far / n_r
    theta_deg = (np.arange(n_theta) + 0.5) * dtheta_deg
    r_mm = (np.arange(n_r) + 0.5) * dr_mm

    print(f"Polar grid: {n_theta} theta x {n_r} r, dtheta={dtheta_deg:.3f}°, "
          f"dr={dr_mm:.4f} mm, t_far={t_far} mm")

    # ----------------------------------------------------------- wire phantom
    print(f"\n=== Rendering wire phantom (Pass 6 v2 calibrated YAML) — {N_FRAMES} frames ===")
    wire_world, positions = build_wire_world(materials)
    print("Wire positions (r mm, θ deg):")
    for r, th, _, _ in positions:
        print(f"  r={r:5.2f}  θ={th:6.2f}°")
    wire_frames = render_frames(cfg, wire_world, materials, N_FRAMES)
    if wire_frames.shape[1:] == (n_r, n_theta):
        wire_frames = wire_frames.transpose(0, 2, 1)
    print(f"  frames shape={wire_frames.shape} (frame, theta, r), "
          f"min={wire_frames.min():.1f}, max={wire_frames.max():.1f}, "
          f"mean={wire_frames.mean():.1f}")

    # --------------------------------------------------------- anechoic baseline
    print(f"\n=== Rendering anechoic (no wires) — {N_FRAMES} frames ===")
    aneco_world = build_anechoic_world(materials)
    aneco_frames = render_frames(cfg, aneco_world, materials, N_FRAMES)
    if aneco_frames.shape[1:] == (n_r, n_theta):
        aneco_frames = aneco_frames.transpose(0, 2, 1)
    print(f"  mean palette={aneco_frames.mean():.1f}, std={aneco_frames.std():.1f}")

    # Pre-compute per-radius bg statistics across all theta (anechoic)
    bg_per_r_mean = aneco_frames.mean(axis=(0, 1))
    bg_per_r_std = aneco_frames.std(axis=(0, 1))

    # --------------------------------------------------------- per-wire metrics
    print(f"\n{'r':>5} {'θ°':>6} | {'wire peak':>9} {'wire med':>8} {'hit/frm':>7} | "
          f"{'bg mean':>7} {'bg std':>6} {'bg p99':>6} | {'SNR':>5} {'visible?':>8}")
    print("-" * 90)
    for (r, th, _, _) in positions:
        # Theta bin around the wire center; ±2 bins for hit search
        th_bin_center = int(round(th / dtheta_deg)) % n_theta
        th_window = 5  # ±2 bins
        th_lo = (th_bin_center - th_window // 2) % n_theta
        # Radial: ±1 mm around r
        r_lo_idx = max(0, int(round((r - 1.0) / dr_mm)))
        r_hi_idx = min(n_r, int(round((r + 1.0) / dr_mm)))

        # Build (frame, ±θ, r-window) patch
        if th_lo + th_window <= n_theta:
            patch = wire_frames[:, th_lo:th_lo + th_window, r_lo_idx:r_hi_idx]
        else:
            # wraparound
            wrap = (th_lo + th_window) - n_theta
            p1 = wire_frames[:, th_lo:, r_lo_idx:r_hi_idx]
            p2 = wire_frames[:, :wrap, r_lo_idx:r_hi_idx]
            patch = np.concatenate([p1, p2], axis=1)

        # Per-frame peak in patch
        peaks = patch.reshape(N_FRAMES, -1).max(axis=1)

        # Local bg from anechoic at same radial annulus, full theta
        aneco_band = aneco_frames[:, :, r_lo_idx:r_hi_idx]
        bg_mean = float(aneco_band.mean())
        bg_std = float(aneco_band.std())
        bg_p99 = float(np.percentile(aneco_band, 99.0))

        # Hit rate: peak > bg_mean + 3*bg_std (visible bump)
        hit_threshold = bg_mean + 3 * bg_std
        hit_rate = float((peaks > hit_threshold).mean())

        peak_med = float(np.median(peaks))
        peak_max = float(peaks.max())
        snr = (peak_med - bg_mean) / max(bg_std, 1e-3)

        visible = "YES" if (snr >= 3.0 and hit_rate >= 0.5) else \
                  ("MAYBE" if snr >= 1.5 else "no")
        print(f"{r:5.1f} {th:6.1f} | {peak_max:9.1f} {peak_med:8.1f} {hit_rate:7.2f} | "
              f"{bg_mean:7.1f} {bg_std:6.1f} {bg_p99:6.1f} | {snr:5.1f} {visible:>8}")

    # --------------------------------------------------------- ringdown clean
    print("\nNote: 'wire peak' is max in ±2-bin × ±1mm patch around expected wire location.")
    print("'bg' is from the matched anechoic world, full-theta annulus at same r±1mm.")
    print("'SNR' = (wire peak median - bg mean) / bg std. SNR>=3 typically visible.")


if __name__ == "__main__":
    main()
