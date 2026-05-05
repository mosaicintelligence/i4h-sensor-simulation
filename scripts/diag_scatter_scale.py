"""Render anechoic with noise OFF at several scatter_integral_scale values.

Goal: find a scatter strength where bg mean ~46 (bench target) and check
whether lumen scatter has a focal-zone hump like noise does.
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
N_FRAMES = 8

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

def render_aneco(cfg, materials, scatter_scale, noise_sigma):
    w = rs.World("lumen")
    w.add(Sphere(np.array([0.0, 0.0, 1000.0], dtype=np.float32), 0.001,
                 materials.get_index("lumen")))
    sim = rs.RaytracingUltrasoundSimulator(w, materials)
    sp = cfg.to_sim_params()
    sp.scatter_integral_scale = float(scatter_scale)
    sp.noise_sigma = float(noise_sigma)
    rng = np.random.default_rng(2024)
    out = []
    for k in range(N_FRAMES):
        yaw = float(rng.uniform(-0.005, 0.005)) if N_FRAMES > 1 else 0.0
        sp.frame_seed = k + 1
        out.append(np.asarray(sim.simulate(make_probe(cfg, yaw), sp)))
    f = np.stack(out, axis=0)
    n_theta, n_r = int(cfg.sim.b_mode_size[0]), int(cfg.sim.b_mode_size[1])
    if f.shape[1:] == (n_r, n_theta):
        f = f.transpose(0, 2, 1)
    return f

def profile(f, dr, label):
    n_r = f.shape[2]
    rs_mm = [3, 5, 8, 10, 12, 15, 17, 19, 22, 25, 28]
    print(f"\n  {label}")
    print("  r(mm)  mean   std")
    for r_t in rs_mm:
        ri = int(round(r_t / dr))
        if ri >= n_r: continue
        b = f[:, :, max(0, ri-5):min(n_r, ri+5)]
        print(f"  {r_t:5.1f} {b.mean():6.1f} {b.std():5.1f}")

def main():
    cfg = IvusSimConfig.from_yaml(YAML)
    mats = rs.Materials()
    n_r = int(cfg.sim.b_mode_size[1])
    dr = float(cfg.sim.t_far_mm) / n_r

    for scale in [40, 200, 1000, 5000, 25000]:
        f = render_aneco(cfg, mats, scatter_scale=scale, noise_sigma=0.0)
        print(f"\n[scale={scale}, noise=0] global mean={f.mean():.1f} std={f.std():.1f} max={f.max():.1f}")
        profile(f, dr, f"depth profile @ scale={scale}:")

if __name__ == "__main__":
    main()
