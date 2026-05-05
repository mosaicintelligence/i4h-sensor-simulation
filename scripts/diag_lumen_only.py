"""Render Pass 6 v2 wire phantom with noise OFF, measure bg vs depth.

Tells us whether the focal-zone bg hump is from (a) the pre-PSF noise being
narrowed by PSF concentration, or (b) the lumen-scatter texture itself.
If lumen-only bg is FLAT vs depth, user's choice C (more lumen scatter)
will fix the hump. If lumen-only bg ALSO humps, the hump is intrinsic to the
depth-dependent PSF and we need a different fix.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "i4h-sensor-simulation" / "ultrasound-raytracing"))
sys.path.insert(0, str(WORKSPACE / "instrument-calibration" / "p035_visions"))

import raysim as rs
from raysim.config import IvusSimConfig
from raysim.ray_sim_python import Sphere

YAML = WORKSPACE / "instrument-calibration" / "p035_visions" / "volcano_s5i.yaml"
N_FRAMES = 12

def make_probe(cfg, yaw=0.0):
    pose = rs.Pose(np.array(cfg.probe.pose.position_mm, dtype=np.float32),
                   np.array((0.0, yaw, 0.0), dtype=np.float32))
    return rs.IVUSProbe(pose, int(cfg.probe.num_scanlines),
                        float(cfg.probe.frequency_mhz),
                        float(cfg.probe.elevational_height_mm),
                        int(cfg.probe.num_elevational_samples),
                        float(cfg.probe.f_num),
                        float(cfg.probe.speed_of_sound_mm_per_us),
                        float(cfg.probe.pulse_duration_cycles),
                        float(cfg.probe.element_radius_mm),
                        float(cfg.probe.focal_length_mm))

def build_aneco(materials):
    w = rs.World("lumen")
    w.add(Sphere(np.array([0.0, 0.0, 1000.0], dtype=np.float32), 0.001,
                 materials.get_index("lumen")))
    return w

def render(cfg, world, materials, n, override_sigma=None):
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    sp = cfg.to_sim_params()
    if override_sigma is not None:
        sp.noise_sigma = override_sigma
    rng = np.random.default_rng(2024)
    out = []
    for k in range(n):
        yaw = float(rng.uniform(-0.005, 0.005)) if n > 1 else 0.0
        sp.frame_seed = k + 1
        out.append(np.asarray(sim.simulate(make_probe(cfg, yaw), sp)))
    f = np.stack(out, axis=0)
    n_theta, n_r = int(cfg.sim.b_mode_size[0]), int(cfg.sim.b_mode_size[1])
    if f.shape[1:] == (n_r, n_theta):
        f = f.transpose(0, 2, 1)
    return f  # (frame, theta, r)

def main():
    cfg = IvusSimConfig.from_yaml(YAML)
    materials = rs.Materials()
    n_r = int(cfg.sim.b_mode_size[1])
    dr = float(cfg.sim.t_far_mm) / n_r
    r_mm = (np.arange(n_r) + 0.5) * dr
    print("\n[A] noise OFF, lumen-scatter ONLY (current YAML lumen mu=0.1, sigma=0.1):")
    fA = render(cfg, build_aneco(materials), materials, N_FRAMES, override_sigma=0.0)
    print(f"  global mean={fA.mean():.1f}, std={fA.std():.1f}")

    print(f"\n[B] noise ON ({cfg.processing.noise.sigma}), lumen-scatter:")
    fB = render(cfg, build_aneco(materials), materials, N_FRAMES, override_sigma=None)
    print(f"  global mean={fB.mean():.1f}, std={fB.std():.1f}")

    print("\n  r(mm) | A: lumen-only mean / std | B: lumen+noise mean / std")
    print("  ------+---------------------------+---------------------------")
    for r_target in [3, 5, 8, 10, 12, 15, 17, 19, 20, 22, 25, 28]:
        ri = int(round(r_target / dr))
        if ri >= n_r: continue
        bandA = fA[:, :, max(0, ri-5):min(n_r, ri+5)]
        bandB = fB[:, :, max(0, ri-5):min(n_r, ri+5)]
        print(f"  {r_target:5.1f} | {bandA.mean():8.1f} / {bandA.std():6.1f}        | "
              f"{bandB.mean():8.1f} / {bandB.std():6.1f}")

if __name__ == "__main__":
    main()
