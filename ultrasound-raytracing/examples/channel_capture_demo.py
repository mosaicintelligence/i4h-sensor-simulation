# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Channel-capture demo: phased-array full-matrix capture (FMC) with the OptiX raytracer.

This is the development-time companion to ``CHANNEL_CAPTURE.md`` at the repo root.
It exercises ``RaytracingUltrasoundSimulator.simulate_channel_capture`` on a
phased-array probe imaging a few point reflectors in a uniform medium, then:

  1. Saves the per-element RF cube ``rf`` of shape ``(N_TX, N_RX, N_samples)``.
  2. Plots one TX slice ``rf[tx, :, :]`` and overlays the analytic hyperbolic
     moveout predicted by ``t = (|p - r_tx| + |p - r_e|) / c`` for each scatterer.
  3. Runs a textbook delay-and-sum (DAS) beamformer in NumPy to reconstruct a
     B-mode image from the channel cube.
  4. Runs the legacy ``simulator.simulate(...)`` on the same scene for a
     side-by-side B-mode comparison.

Run from the ``ultrasound-raytracing`` directory after building the Python
extension::

    python examples/channel_capture_demo.py

Outputs are written to ``channel_capture_output/``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import raysim.cuda as rs  # noqa: E402


# ---------------------------------------------------------------------------
# Phantom + probe configuration
# ---------------------------------------------------------------------------

# A phased array with these parameters is small enough that a 64x64 channel
# cube fits in a few tens of MB while still being visually instructive.
PROBE_NUM_ELEMENTS = 64
PROBE_WIDTH_MM = 20.0
PROBE_FREQUENCY_MHZ = 3.5
PROBE_SECTOR_ANGLE_DEG = 60.0

# Channel-capture: enough rays to densely sample the sector for each TX event.
NUM_TX_RAYS = 512
BUFFER_SIZE = 2048
T_FAR_MM = 80.0  # Max one-way path (TX leg + RX leg) stored at the last bin.

# Point reflectors (sphere centres in world coords). Probe sits at the origin
# looking down +z, so these depths are how far the reflectors are below the array.
# We use a strong-impedance material (bone) inside a water background to get
# clear specular hits without burying them in scatter.
REFLECTORS: List[Tuple[float, float, float]] = [
    (0.0, 0.0, 30.0),
    (-6.0, 0.0, 40.0),
    (5.0, 0.0, 55.0),
]
REFLECTOR_RADIUS_MM = 0.4

OUTPUT_DIR = "channel_capture_output"


def build_world() -> Tuple[rs.World, rs.Materials]:
    """Build a uniform water world with a few small high-impedance reflectors."""
    materials = rs.Materials()
    world = rs.World("water")
    bone_id = materials.get_index("bone")
    for x, y, z in REFLECTORS:
        sphere = rs.Sphere(np.array([x, y, z], dtype=np.float32), REFLECTOR_RADIUS_MM, bone_id)
        world.add(sphere)
    return world, materials


def build_probe() -> rs.PhasedArrayProbe:
    """Build the phased-array probe at the origin, axial = +z."""
    pose = rs.Pose(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    return rs.PhasedArrayProbe(
        pose,
        num_elements_x=PROBE_NUM_ELEMENTS,
        width=PROBE_WIDTH_MM,
        sector_angle=PROBE_SECTOR_ANGLE_DEG,
        frequency=PROBE_FREQUENCY_MHZ,
        elevational_height=5.0,
        num_el_samples=1,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_one_tx_slice(rf: np.ndarray,
                      tx_index: int,
                      tx_positions: np.ndarray,
                      rx_positions: np.ndarray,
                      reflectors: List[Tuple[float, float, float]],
                      t_far: float,
                      out_path: str) -> None:
    """Show ``rf[tx_index, :, :]`` and overlay analytic hyperbolae."""
    n_rx, n_samples = rf.shape[1], rf.shape[2]
    slice_2d = rf[tx_index]  # (N_RX, N_samples)
    bin_to_path = t_far / max(n_samples - 1, 1)
    extent = [0.0, t_far, n_rx - 1, 0.0]  # (x_min, x_max, y_max, y_min)

    fig, ax = plt.subplots(figsize=(10, 5))
    norm = np.percentile(np.abs(slice_2d), 99.5) or 1.0
    im = ax.imshow(slice_2d / norm, aspect="auto", cmap="seismic", vmin=-1, vmax=1, extent=extent)
    ax.set_xlabel("Total path length (TX + RX) [mm]")
    ax.set_ylabel("RX element index")
    ax.set_title(
        f"Channel-capture slice rf[tx={tx_index}, :, :] -- predicted moveout overlaid")

    r_tx = tx_positions[tx_index]
    for (x, y, z) in reflectors:
        p = np.array([x, y, z], dtype=np.float32)
        t_tx_path = float(np.linalg.norm(p - r_tx))
        # For each RX element, predicted total path:
        t_total = t_tx_path + np.linalg.norm(rx_positions - p, axis=1)
        # Plot vs RX index. Skip rays that fall outside the recorded window.
        in_range = t_total < t_far
        ax.plot(t_total[in_range], np.where(in_range)[0], color="black", lw=1.0,
                ls="--", alpha=0.7)

    fig.colorbar(im, ax=ax, label="Normalized RF (signed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_self_channel_vs_legacy(rf: np.ndarray,
                                legacy_rf_diag: np.ndarray | None,
                                t_far: float,
                                out_path: str) -> None:
    """Compare the diagonal rf[i,i,:] (TX = RX) against the legacy scanline output."""
    n_tx = rf.shape[0]
    diag = np.stack([rf[i, i, :] for i in range(n_tx)], axis=0)  # (N_tx, N_samples)
    fig, axes = plt.subplots(1, 2 if legacy_rf_diag is not None else 1,
                             figsize=(12, 5), squeeze=False)
    norm = np.percentile(np.abs(diag), 99.5) or 1.0
    extent = [0.0, t_far, n_tx - 1, 0.0]
    axes[0, 0].imshow(diag / norm, aspect="auto", cmap="gray",
                      vmin=0, vmax=1, extent=extent)
    axes[0, 0].set_title("Channel-capture self-channel rf[i, i, :]\n(unconvolved RF)")
    axes[0, 0].set_xlabel("Total path length [mm]")
    axes[0, 0].set_ylabel("Element index (TX = RX)")
    if legacy_rf_diag is not None:
        norm2 = np.percentile(np.abs(legacy_rf_diag), 99.5) or 1.0
        axes[0, 1].imshow(np.abs(legacy_rf_diag) / norm2, aspect="auto", cmap="gray",
                          vmin=0, vmax=1, extent=extent)
        axes[0, 1].set_title("Legacy scanline output (|rf|)\nfor reference")
        axes[0, 1].set_xlabel("Path length [mm]")
        axes[0, 1].set_ylabel("Beam index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Beamforming (delay-and-sum)
# ---------------------------------------------------------------------------


def das_bmode(rf: np.ndarray,
              tx_positions: np.ndarray,
              rx_positions: np.ndarray,
              t_far: float,
              x_range_mm: Tuple[float, float],
              z_range_mm: Tuple[float, float],
              n_x: int = 256,
              n_z: int = 256) -> np.ndarray:
    """Textbook FMC delay-and-sum.

    For each pixel ``(x, z)``, the DAS image is

        I(x, z) = sum_tx sum_rx rf[tx, rx, bin(t_tx + t_rx)]

    where ``t_tx = ||p - r_tx||`` and ``t_rx = ||p - r_rx||``. We use
    nearest-neighbour bin lookup and absolute-value envelope detection at the
    end (no Hilbert transform; this is a development-level demo).

    The arithmetic is intentionally straightforward NumPy so the link between
    the channel cube and the image is easy to follow line-by-line.
    """
    n_tx, n_rx, n_samples = rf.shape
    bin_to_path = t_far / max(n_samples - 1, 1)
    inv_bin = 1.0 / bin_to_path

    xs = np.linspace(x_range_mm[0], x_range_mm[1], n_x, dtype=np.float32)
    zs = np.linspace(z_range_mm[0], z_range_mm[1], n_z, dtype=np.float32)
    Z, X = np.meshgrid(zs, xs, indexing="ij")  # (n_z, n_x)
    pixels = np.stack([X, np.zeros_like(X), Z], axis=-1).reshape(-1, 3)  # (P, 3)

    # Distances pixel <-> element. Shape (P, N_elem).
    d_tx = np.linalg.norm(pixels[:, None, :] - tx_positions[None, :, :], axis=2)
    d_rx = np.linalg.norm(pixels[:, None, :] - rx_positions[None, :, :], axis=2)

    img = np.zeros(pixels.shape[0], dtype=np.float32)
    for tx in range(n_tx):
        # bins[(P, N_RX)] for this TX
        total = (d_tx[:, tx, None] + d_rx)  # (P, N_RX)
        bins = np.clip((total * inv_bin + 0.5).astype(np.int64), 0, n_samples - 1)
        # Gather RF: shape (P, N_RX) — fancy index per RX channel.
        # rf[tx] has shape (N_RX, N_samples); take rf[tx, e, bins[:, e]] for each e.
        # Vectorise via take_along_axis.
        gathered = np.take_along_axis(rf[tx], bins.T, axis=1)  # (N_RX, P)
        img += gathered.sum(axis=0)

    img = img.reshape(n_z, n_x)
    return np.abs(img)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[setup] Building world and probe...")
    world, materials = build_world()
    probe = build_probe()
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)

    # ---- Channel capture --------------------------------------------------
    cc_params = rs.ChannelCaptureParams()
    cc_params.t_far = T_FAR_MM
    cc_params.buffer_size = BUFFER_SIZE
    cc_params.num_tx_rays = NUM_TX_RAYS
    cc_params.num_tx = PROBE_NUM_ELEMENTS  # full FMC
    cc_params.max_depth = 4
    cc_params.enable_cuda_timing = True

    print(f"[capture] Running FMC: {PROBE_NUM_ELEMENTS} TX events x "
          f"{PROBE_NUM_ELEMENTS} RX channels x {BUFFER_SIZE} samples ...")
    t0 = time.time()
    cap = simulator.simulate_channel_capture(probe, cc_params)
    elapsed = time.time() - t0
    rf = cap["rf"]  # (N_TX, N_RX, N_samples) float32
    tx_positions = cap["tx_positions"]
    rx_positions = cap["rx_positions"]
    c = float(cap["speed_of_sound"])
    t_far = float(cap["t_far"])
    print(f"[capture] done in {elapsed:.2f} s. rf.shape = {rf.shape}, "
          f"speed_of_sound = {c} mm/us, t_far = {t_far} mm")
    print(f"[capture] rf min/max = {rf.min():.3e} / {rf.max():.3e}; "
          f"nonzero fraction = {np.count_nonzero(rf) / rf.size:.4f}")

    # Save the raw cube so downstream notebooks can play with it.
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "channel_rf.npz"),
        rf=rf,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        speed_of_sound=c,
        t_far=t_far,
        reflectors=np.asarray(REFLECTORS, dtype=np.float32),
        sector_angle_deg=PROBE_SECTOR_ANGLE_DEG,
    )
    print(f"[save] Wrote {OUTPUT_DIR}/channel_rf.npz")

    # ---- Validation: hyperbolic moveout overlay ---------------------------
    mid_tx = PROBE_NUM_ELEMENTS // 2
    plot_one_tx_slice(
        rf=rf,
        tx_index=mid_tx,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        reflectors=REFLECTORS,
        t_far=t_far,
        out_path=os.path.join(OUTPUT_DIR, "01_rf_slice_with_moveout.png"),
    )
    print("[plot] 01_rf_slice_with_moveout.png")

    # Quantitative moveout error for the central reflector.
    p = np.array(REFLECTORS[0], dtype=np.float32)
    r_tx_pos = tx_positions[mid_tx]
    t_tx_path = float(np.linalg.norm(p - r_tx_pos))
    t_total_pred = t_tx_path + np.linalg.norm(rx_positions - p, axis=1)
    bin_pred = np.clip(np.round(t_total_pred / t_far * (BUFFER_SIZE - 1)).astype(int),
                       0, BUFFER_SIZE - 1)
    # Find each RX channel's argmax in a small window around the predicted bin.
    win = 6
    bin_obs = np.zeros(rx_positions.shape[0], dtype=int)
    for e in range(rx_positions.shape[0]):
        lo = max(bin_pred[e] - win, 0)
        hi = min(bin_pred[e] + win + 1, BUFFER_SIZE)
        bin_obs[e] = lo + int(np.argmax(np.abs(rf[mid_tx, e, lo:hi])))
    err_samples = bin_obs - bin_pred
    print(f"[validate] hyperbolic moveout error for reflector @ {REFLECTORS[0]}: "
          f"mean = {err_samples.mean():+.2f} samples, "
          f"std = {err_samples.std():.2f}, max abs = {np.abs(err_samples).max()}")

    # ---- DAS B-mode reconstruction ---------------------------------------
    print("[beamform] Running DAS B-mode (NumPy)...")
    t0 = time.time()
    das_img = das_bmode(
        rf=rf,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        t_far=t_far,
        x_range_mm=(-PROBE_WIDTH_MM, PROBE_WIDTH_MM),
        z_range_mm=(2.0, 70.0),
        n_x=256,
        n_z=256,
    )
    das_elapsed = time.time() - t0
    print(f"[beamform] done in {das_elapsed:.2f} s. image shape = {das_img.shape}")

    # ---- Legacy scanline + post-processing for comparison ----------------
    print("[legacy] Running simulator.simulate() on the same scene for B-mode comparison...")
    legacy_params = rs.SimParams()
    legacy_params.t_far = T_FAR_MM
    legacy_params.buffer_size = BUFFER_SIZE
    legacy_params.conv_psf = True
    legacy_params.b_mode_size = (512, 512)
    legacy_image = simulator.simulate(probe, legacy_params)
    print(f"[legacy] done. legacy image shape = {legacy_image.shape}")

    # ---- Plot side-by-side -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    das_db = 20.0 * np.log10(das_img / (das_img.max() + 1e-9) + 1e-6)
    axes[0].imshow(
        das_db,
        cmap="gray",
        extent=[-PROBE_WIDTH_MM, PROBE_WIDTH_MM, 70.0, 2.0],
        vmin=-50.0, vmax=0.0,
    )
    axes[0].set_title("DAS B-mode from channel-capture cube")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("depth z [mm]")
    for (x, y, z) in REFLECTORS:
        axes[0].plot(x, z, "rx", ms=8, mew=1.5)

    axes[1].imshow(
        legacy_image,
        cmap="gray",
        extent=[simulator.get_min_x(), simulator.get_max_x(),
                simulator.get_max_z(), simulator.get_min_z()],
        vmin=-60.0, vmax=0.0,
    )
    axes[1].set_title("Legacy simulate() B-mode (PSF + TGC + log)")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("depth z [mm]")
    for (x, y, z) in REFLECTORS:
        axes[1].plot(x, z, "rx", ms=8, mew=1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_das_vs_legacy_bmode.png"), dpi=150)
    plt.close(fig)
    print("[plot] 02_das_vs_legacy_bmode.png")

    # Self-channel diagonal vs. nothing (legacy slice is in steered-angle space,
    # not element-vs-time, so we just show the channel-capture diagonal for now).
    plot_self_channel_vs_legacy(
        rf=rf,
        legacy_rf_diag=None,
        t_far=t_far,
        out_path=os.path.join(OUTPUT_DIR, "03_self_channel_diag.png"),
    )
    print("[plot] 03_self_channel_diag.png")
    print(f"\nAll outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
