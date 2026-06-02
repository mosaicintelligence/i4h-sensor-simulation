#!/usr/bin/env python3
"""Quick visualization of the B2 diagnostic frame to investigate why some
wires render 47x brighter than others (peak palette 2954 vs 2720 = 230
palette delta = 47x in amplitude)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
SIM_ROOT = WORKSPACE_ROOT / "i4h-sensor-simulation" / "ultrasound-raytracing"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import raysim as rs  # noqa: E402
from raysim import IvusSimConfig  # noqa: E402

from tier1_evaluation import (  # noqa: E402
    B2_WIRE_POSITIONS,
    b_mode_to_theta_r,
    build_b2_wire_world,
    polar_axes,
    render_frames,
)

YAML_PATH = HERE / "volcano_s5i.yaml"
OUT_DIR = HERE / "tier1_results" / "figures"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = IvusSimConfig.from_yaml(YAML_PATH)
    sim_params_diag = cfg.to_sim_params()
    sim_params_diag.log_floor = 1e-19
    sim_params_diag.ring_down.enabled = False
    sim_params_diag.reject_palette = 0.0
    sim_params_diag.saturation_palette = 0.0
    sim_params_diag.median_clip_filter = False

    materials = rs.Materials()
    theta_deg, r_mm, dtheta_deg, dr_mm = polar_axes(cfg)
    world, positions = build_b2_wire_world(materials, theta_rotate_deg=0.0)
    frames = render_frames(cfg, world, materials, 1, sim_params_diag)
    frame = b_mode_to_theta_r(frames[0], cfg)  # (n_theta, n_r)
    print(f"frame shape={frame.shape} min={frame.min():.0f} max={frame.max():.0f}")

    # Print peak per wire
    print("Wire    r_mm  theta  peak_palette  peak_pos(theta_idx,r_idx)")
    for pos in positions:
        # Find peak in a (search_angle, search_r) window
        n_t = frame.shape[0]
        n_r = frame.shape[1]
        pred_t = int(round(pos["theta_sim_deg"] / dtheta_deg)) % n_t
        pred_r = int(round(pos["r_mm"] / dr_mm))
        half_t = 8
        half_r = max(2, int(round(2.0 / dr_mm)))
        t_idx = [(pred_t + d) % n_t for d in range(-half_t, half_t + 1)]
        r0 = max(0, pred_r - half_r)
        r1 = min(n_r, pred_r + half_r + 1)
        sub = frame[t_idx, :][:, r0:r1]
        local = np.unravel_index(int(np.argmax(sub)), sub.shape)
        pk_t = t_idx[local[0]]
        pk_r = r0 + local[1]
        print(f"  w{pos['wire_idx']:>2}  {pos['r_mm']:5.1f}  {pos['theta_sim_deg']:5.1f}  "
              f"{frame[pk_t, pk_r]:6.0f}  ({pk_t:3d}, {pk_r:3d})  pos_r={r_mm[pk_r]:.2f} mm")

    # Render the frame as a polar B-mode + unwrapped
    fig = plt.figure(figsize=(15, 7))
    t_far = float(cfg.sim.t_far_mm)

    # Unwrapped
    ax2 = fig.add_subplot(1, 2, 1)
    im = ax2.imshow(frame.T, extent=(0, 360, t_far, 0), aspect="auto",
                     cmap="viridis", vmin=2400, vmax=3100)
    ax2.set_xlabel("theta (deg)")
    ax2.set_ylabel("r (mm)")
    ax2.set_title("Diagnostic frame (unwrapped) -- viridis, vmin=2400, vmax=3100")
    plt.colorbar(im, ax=ax2)
    for pos in positions:
        ax2.plot([pos["theta_sim_deg"]], [pos["r_mm"]], marker="o", mfc="none",
                  mec="white", mew=1.2, markersize=12)
        ax2.annotate(f"w{pos['wire_idx']}", (pos["theta_sim_deg"], pos["r_mm"]),
                       color="white", fontsize=8,
                       textcoords="offset points", xytext=(8, -3))

    # Polar
    ax3 = fig.add_subplot(1, 2, 2, projection="polar")
    n_theta = frame.shape[0]
    theta_edges = np.deg2rad(np.linspace(0, 360, n_theta + 1))
    r_edges = np.linspace(0, t_far, frame.shape[1] + 1)
    Theta, R = np.meshgrid(theta_edges, r_edges)
    ax3.pcolormesh(Theta, R, frame.T, cmap="viridis", shading="flat", vmin=2400, vmax=3100)
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)
    ax3.set_title("Diagnostic frame (polar)")
    for pos in positions:
        ax3.plot([math.radians(pos["theta_sim_deg"])], [pos["r_mm"]],
                  marker="o", mfc="none", mec="white", mew=1.2, markersize=12)

    plt.tight_layout()
    out = OUT_DIR / "cd_diagnostic_frame.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
