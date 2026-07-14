# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Throughput benchmark for the IVUS synthetic-aperture channel-capture path.

Runs ``--frames`` consecutive channel captures (a pullback: the probe advances
along the vessel axis between frames, like a real IVUS acquisition) and reports
per-frame wall time and aggregate throughput. Scene and acceleration structure
are built once; each frame re-uses them, so this measures steady-state
simulation throughput, not setup cost.

Usage (from ``ultrasound-raytracing/``)::

    python examples/ivus_channel_capture_benchmark.py --frames 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raysim.cuda as rs  # noqa: E402

from ivus_channel_capture_demo import (  # noqa: E402
    BUFFER_SIZE,
    NUM_ANGULAR,
    NUM_TX_RAYS,
    RING_RADIUS_MM,
    T_FAR_MM,
    TX_FAN_HALF_DEG,
    build_probe,
    build_vessel_world,
)

OUTPUT_DIR = "ivus_channel_capture_output"


def main() -> None:
    parser = argparse.ArgumentParser(description="IVUS channel-capture throughput benchmark.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames (default 100).")
    parser.add_argument("--pullback-mm", type=float, default=2.0,
                        help="Total pullback distance along the vessel axis (default 2 mm).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warm-up frames excluded from statistics (default 3).")
    args = parser.parse_args()

    materials = rs.Materials()
    world = build_vessel_world(materials)
    probe = build_probe()
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    cc = rs.ChannelCaptureParams()
    cc.t_far = T_FAR_MM
    cc.buffer_size = BUFFER_SIZE
    cc.num_tx_rays = NUM_TX_RAYS
    cc.num_tx = NUM_ANGULAR
    cc.max_depth = 3
    cc.ivus_ring_radius_mm = RING_RADIUS_MM
    cc.ivus_tx_fan_half_deg = TX_FAN_HALF_DEG
    cc.enable_cuda_timing = False

    # Pullback along the vessel (mesh cylinder) axis = y.
    ys = np.linspace(-args.pullback_mm / 2.0, args.pullback_mm / 2.0, args.frames,
                     dtype=np.float32)

    cube_bytes = NUM_ANGULAR * NUM_ANGULAR * BUFFER_SIZE * 4
    rays_per_frame = NUM_ANGULAR * NUM_TX_RAYS
    print(f"[bench] {args.frames} frames | cube {NUM_ANGULAR}x{NUM_ANGULAR}x{BUFFER_SIZE} "
          f"({cube_bytes / 2**20:.0f} MiB) | {NUM_ANGULAR} TX launches x {NUM_TX_RAYS} rays "
          f"= {rays_per_frame} primary rays/frame")

    times = []
    checksum = 0.0
    for i in range(args.frames):
        probe.set_pose(rs.Pose(
            position=np.array([0.0, ys[i], 0.0], dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        ))
        t0 = time.perf_counter()
        cap = simulator.simulate_channel_capture(probe, cc)
        dt = time.perf_counter() - t0
        times.append(dt)
        # Touch the result so download cost is honestly included and nothing is
        # optimised away.
        checksum += float(cap["rf"][0, 0, :8].sum())
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[bench] frame {i + 1:3d}/{args.frames}: {dt * 1e3:8.1f} ms")

    t = np.asarray(times[args.warmup:])
    stats = {
        "frames_total": args.frames,
        "frames_measured": int(t.size),
        "mean_ms": float(t.mean() * 1e3),
        "std_ms": float(t.std() * 1e3),
        "min_ms": float(t.min() * 1e3),
        "max_ms": float(t.max() * 1e3),
        "p50_ms": float(np.percentile(t, 50) * 1e3),
        "p95_ms": float(np.percentile(t, 95) * 1e3),
        "fps": float(1.0 / t.mean()),
        "tx_events_per_s": float(NUM_ANGULAR / t.mean()),
        "primary_rays_per_s": float(rays_per_frame / t.mean()),
        "rf_samples_per_s": float(NUM_ANGULAR * NUM_ANGULAR * BUFFER_SIZE / t.mean()),
        "config": {
            "num_angular": NUM_ANGULAR, "num_tx_rays": NUM_TX_RAYS,
            "buffer_size": BUFFER_SIZE, "t_far_mm": T_FAR_MM,
            "ring_radius_mm": RING_RADIUS_MM, "tx_fan_half_deg": TX_FAN_HALF_DEG,
            "max_depth": 3,
        },
    }

    print("\n[bench] steady-state results "
          f"(excluding {args.warmup} warm-up frames):")
    print(f"  per-frame: mean {stats['mean_ms']:.1f} ms  (std {stats['std_ms']:.1f}, "
          f"p50 {stats['p50_ms']:.1f}, p95 {stats['p95_ms']:.1f}, "
          f"min {stats['min_ms']:.1f}, max {stats['max_ms']:.1f})")
    print(f"  throughput: {stats['fps']:.2f} frames/s | "
          f"{stats['tx_events_per_s']:.0f} TX events/s | "
          f"{stats['primary_rays_per_s'] / 1e3:.0f} k primary rays/s | "
          f"{stats['rf_samples_per_s'] / 1e6:.0f} M RF samples/s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, "benchmark_throughput.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[bench] wrote {out_json}  (checksum {checksum:.3e})")


if __name__ == "__main__":
    main()
