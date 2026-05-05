"""Bench anechoic depth profile vs sim Pass 6 v2.

Compare the bench's actual mean(palette) vs depth across the 5 reference frames
(FILE0000-FILE0004) — wire-masked — against Pass 6 v2's anechoic render.
Tells us what bench shape we should be matching.
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
BENCH_POLAR = WORKSPACE / "P_035_PointScatter" / "derived" / "polar"
BENCH_FRAMES = ["FILE0000", "FILE0001", "FILE0002", "FILE0003", "FILE0004"]

# Bench geometry: D=60mm display, 500 px diameter -> 0.12 mm/px radial pitch.
BENCH_DR_MM = 0.12

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

def render_aneco(cfg, materials, n=8):
    w = rs.World("lumen")
    w.add(Sphere(np.array([0.0, 0.0, 1000.0], dtype=np.float32), 0.001,
                 materials.get_index("lumen")))
    sim = rs.RaytracingUltrasoundSimulator(w, materials)
    sp = cfg.to_sim_params()
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
    return f

def load_bench():
    frames = []
    for name in BENCH_FRAMES:
        a = np.load(BENCH_POLAR / f"{name}.npy").astype(np.float32)
        frames.append(a)
    return np.stack(frames, axis=0)  # (frame, theta, r) presumably

def wire_mask_bench(b):
    """Mask out wires per-frame using p70 threshold per radial bin (matches tier1)."""
    n_r = b.shape[2]
    masked = b.copy()
    # For each (frame, r), threshold across theta
    for k in range(b.shape[0]):
        for ri in range(n_r):
            col = b[k, :, ri]
            t = np.percentile(col, 70.0)
            masked[k, col > t, ri] = np.nan
    return masked

def bench_depth_profile(b_masked, dr_mm):
    n_r = b_masked.shape[2]
    r = (np.arange(n_r) + 0.5) * dr_mm
    means = np.nanmean(b_masked, axis=(0, 1))
    return r, means

def sim_depth_profile(s, dr_mm):
    n_r = s.shape[2]
    r = (np.arange(n_r) + 0.5) * dr_mm
    means = s.mean(axis=(0, 1))
    return r, means

def main():
    print("Loading bench...")
    b = load_bench()
    print(f"  bench shape={b.shape}, range [{b.min()}, {b.max()}]")
    # Determine if (frame, theta, r) or (frame, r, theta)
    # bench is typically (theta, r) per frame, theta=N_a, r=N_d
    # Check by aspect
    bm = wire_mask_bench(b)
    r_bench, prof_bench = bench_depth_profile(bm, BENCH_DR_MM)

    print("\nRendering sim (Pass 6 v2)...")
    cfg = IvusSimConfig.from_yaml(YAML)
    mats = rs.Materials()
    s = render_aneco(cfg, mats, n=8)
    sim_dr = float(cfg.sim.t_far_mm) / s.shape[2]
    r_sim, prof_sim = sim_depth_profile(s, sim_dr)

    print(f"\n  Bench:  range r=[{r_bench[0]:.2f}, {r_bench[-1]:.2f}] mm, n={len(prof_bench)} bins, dr={BENCH_DR_MM}")
    print(f"  Sim:    range r=[{r_sim[0]:.2f}, {r_sim[-1]:.2f}] mm, n={len(prof_sim)} bins, dr={sim_dr:.4f}")

    # Print interleaved at common depths
    print(f"\n  r(mm) | bench mean | sim mean | delta")
    print("  ------+------------+----------+------")
    for r_t in [3, 5, 7, 10, 12, 15, 17, 19, 22, 25, 28]:
        bi = int(round(r_t / BENCH_DR_MM))
        si = int(round(r_t / sim_dr))
        if bi < len(prof_bench) and si < len(prof_sim):
            print(f"  {r_t:5.1f} | {prof_bench[bi]:10.1f} | {prof_sim[si]:8.1f} | {prof_sim[si]-prof_bench[bi]:+5.1f}")

    print(f"\n  Bench overall mean (4-29mm)  = {np.nanmean(prof_bench[(r_bench>4)&(r_bench<29)]):.1f}")
    print(f"  Sim   overall mean (4-29mm)  = {np.mean(prof_sim[(r_sim>4)&(r_sim<29)]):.1f}")
    print(f"  Bench peak-to-trough (4-29mm)= {np.nanmax(prof_bench[(r_bench>4)&(r_bench<29)]) - np.nanmin(prof_bench[(r_bench>4)&(r_bench<29)]):.1f}")
    print(f"  Sim   peak-to-trough (4-29mm)= {np.max(prof_sim[(r_sim>4)&(r_sim<29)]) - np.min(prof_sim[(r_sim>4)&(r_sim<29)]):.1f}")

if __name__ == "__main__":
    main()
