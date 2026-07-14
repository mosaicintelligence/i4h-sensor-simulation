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
IVUS synthetic-aperture channel-capture demo.

This is the IVUS counterpart to ``channel_capture_demo.py`` (which does
phased-array FMC). It exercises ``simulate_channel_capture`` on an
``IVUSProbe`` imaging a vessel-wall phantom with volumetric scatter, then
reconstructs a cross-sectional B-mode image *from the raw channel cube* using a
delay-and-sum (DAS) synthetic-aperture beamformer.

Model
-----
A physical rotating IVUS transducer is a single element mounted a small radius
off the catheter axis. We model one frame as a synthetic aperture: ``N`` angular
element positions on a ring of radius ``ivus_ring_radius_mm``, each facing
radially outward. Element ``k`` transmits a fan of rays about its radial
direction; *every* element receives, giving a channel cube

    rf[tx, rx, sample]    shape (N, N, buffer_size)

where the arrival time of an echo from a world point ``p`` at receive element
``e`` for transmit element ``t`` is ``(|p - r_t| + |p - r_e|) / c``.

The raw ray-traced deposits are non-negative energy spikes. To turn them into
realistic RF (and to make coherent beamforming form speckle rather than a smooth
energy map) each channel is convolved with a bipolar pulse at the probe centre
frequency, time-gain compensated, and converted to its analytic (IQ) signal.
Beamforming is then a coherent, phase-sensitive delay-and-sum on the complex
cube, with envelope detection after the sum -- exactly how a real IQ beamformer
processes channel data. All of this happens offline here, in NumPy.

Outputs (written to ``ivus_channel_capture_output/``):
  1. ``channel_rf.npz``            - raw cube + geometry for offline work.
  2. ``01_rf_slice.png``           - one TX slice (pulsed RF) with the predicted
                                     wall moveout overlaid (geometry sanity check).
  3. ``02_das_reconstruction.png`` - DAS cross-section (Cartesian) with the
                                     true inner/outer wall radii overlaid.
  4. ``03_das_vs_legacy.png``      - DAS-from-channels (unwrapped angle x depth)
                                     vs the legacy scanline B-mode, same phantom.

Run from the ``ultrasound-raytracing`` directory after building the extension::

    python examples/ivus_channel_capture_demo.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np
from scipy.signal import fftconvolve, hilbert

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raysim.cuda as rs  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_ANGULAR = 128          # synthetic-aperture element positions around the ring
FREQUENCY_MHZ = 40.0
T_FAR_MM = 14.0            # max TOTAL (TX + RX) path stored at the last bin
BUFFER_SIZE = 4096         # fine time sampling so the RF pulse is well resolved
NUM_TX_RAYS = 256          # rays per TX firing (samples the transmit wedge)
RING_RADIUS_MM = 0.5       # off-axis radius of the rotating element
TX_FAN_HALF_DEG = 35.0     # half-width of each element's transmit wedge

# Pulse / reconstruction parameters.
PULSE_CYCLES = 1.5         # -6 dB pulse length in cycles (broadband IVUS pulse)
TGC_DB_PER_MM = 6.0        # time-gain compensation slope (total-path mm)
TGC_MAX_DB = 60.0          # cap on TGC gain to avoid amplifying noise
DYNAMIC_RANGE_DB = 55.0    # display window below the reference level
DISPLAY_PERCENTILE = 98.0  # normalize to this percentile (not the bright specular peak)

# Vessel phantom (thick cylinder, axis along Y, imaging plane xz).
INNER_WALL_MM = 3.5
OUTER_WALL_MM = 4.0

OUTPUT_DIR = "ivus_channel_capture_output"


def _mesh_path(name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mesh", name))


def build_vessel_world(materials: rs.Materials) -> rs.World:
    """Thick-walled vessel: lumen (background) + wall + extravascular tissue."""
    world = rs.World("lumen")  # blood-filled lumen as the background medium
    wall_material = materials.get_index("vessel_wall")
    extra_material = materials.get_index("extravascular")

    inner_path = _mesh_path("Cylinder_inner.obj")
    outer_path = _mesh_path("Cylinder_outer.obj")
    for p in (inner_path, outer_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Vessel mesh not found: {p}\n"
                "Generate with: python utils/phantom_maker.py cylinder --output mesh --cylinder-thick"
            )
    world.add(rs.Mesh(inner_path, wall_material))
    world.add(rs.Mesh(outer_path, extra_material))
    return world


def build_probe() -> rs.IVUSProbe:
    """IVUS probe at the vessel centre, imaging plane = xz."""
    pose = rs.Pose(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    return rs.IVUSProbe(
        pose,
        num_angular_rays=NUM_ANGULAR,
        frequency=FREQUENCY_MHZ,
        elevational_height=0.0,
        num_el_samples=1,
        speed_of_sound=1.54,
        pulse_duration=2.0,
    )


# ---------------------------------------------------------------------------
# RF conditioning: pulse modulation, TGC, analytic (IQ) signal
# ---------------------------------------------------------------------------


def make_pulse(freq_mhz: float, c: float, dpath_mm: float, cycles: float) -> np.ndarray:
    """Bipolar transmit/receive pulse sampled on the total-path (mm) axis.

    An echo at total path ``L`` oscillates as ``cos(2*pi*fc*L/c)``, i.e. with a
    spatial period ``lambda = c / fc`` in path-length units. We window that
    carrier with a Gaussian whose -6 dB length is ``cycles`` wavelengths. The
    kernel is zero-mean so it is genuinely bipolar (no DC pedestal), which is
    what lets coherent DAS form speckle instead of a smooth energy map.
    """
    lam = c / freq_mhz                       # mm of path per RF cycle
    sigma = cycles * lam / 2.355             # Gaussian sigma from -6 dB length
    half = int(np.ceil(3.0 * sigma / dpath_mm))
    x = np.arange(-half, half + 1) * dpath_mm
    env = np.exp(-0.5 * (x / sigma) ** 2)
    pulse = env * np.cos(2.0 * np.pi * x / lam)
    pulse -= pulse.mean()                    # zero DC -> bipolar
    return pulse.astype(np.float32)


def condition_rf(rf: np.ndarray, freq_mhz: float, c: float, t_far: float,
                 cycles: float) -> np.ndarray:
    """Convolve every channel with the pulse and return the analytic (IQ) cube."""
    n_samples = rf.shape[-1]
    dpath = t_far / (n_samples - 1)
    pulse = make_pulse(freq_mhz, c, dpath, cycles)
    rf_p = fftconvolve(rf, pulse[None, None, :], mode="same", axes=-1)
    # Analytic signal along time so DAS can be done on complex data (envelope
    # detection after beamforming, exactly as a real IQ beamformer).
    return hilbert(rf_p, axis=-1).astype(np.complex64)


def tgc_gain(n_samples: int, t_far: float, db_per_mm: float, max_db: float) -> np.ndarray:
    """Time-gain-compensation multiplier vs. sample (total-path) index."""
    path = np.linspace(0.0, t_far, n_samples, dtype=np.float32)
    gain_db = np.minimum(db_per_mm * path, max_db)
    return (10.0 ** (gain_db / 20.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic-aperture IQ delay-and-sum reconstruction from the channel cube
# ---------------------------------------------------------------------------


def das_beamform(
    iq: np.ndarray,
    tx_positions: np.ndarray,
    tx_dirs: np.ndarray,
    rx_positions: np.ndarray,
    rx_dirs: np.ndarray,
    t_far: float,
    pixels: np.ndarray,
    tx_fan_half_deg: float = TX_FAN_HALF_DEG,
) -> np.ndarray:
    """Coherent synthetic-aperture DAS on the analytic channel cube.

    For each pixel ``p`` (rows of ``pixels``, world xz coords):

        S(p) = sum_tx sum_rx  w_tx(p) * w_rx(p) * iq[tx, rx, bin(|p-r_tx| + |p-r_rx|)]

    and the returned envelope is ``|S(p)|``. Because ``iq`` is the analytic
    (complex, bipolar) signal, contributions add with phase: scatterers
    interfere to produce speckle and the walls focus to their true radii.
    ``w_tx`` gates the transmit wedge; ``w_rx`` gates receive directivity.
    """
    n_tx, n_rx, n_samples = iq.shape
    inv_bin = (n_samples - 1) / t_far

    d_tx = np.linalg.norm(pixels[:, None, :] - tx_positions[None, :, :], axis=2)
    d_rx = np.linalg.norm(pixels[:, None, :] - rx_positions[None, :, :], axis=2)

    # Receive directivity (independent of tx): element faces the pixel.
    vec_rx = pixels[None, :, :] - rx_positions[:, None, :]      # (N_RX, P, 3)
    d_rx_pt = np.maximum(d_rx.T, 1e-6)                          # (N_RX, P)
    cos_rx = np.einsum("epc,ec->ep", vec_rx, rx_dirs) / d_rx_pt
    rx_face = (cos_rx > 0.0)                                    # (N_RX, P)

    cos_fan = np.cos(np.radians(tx_fan_half_deg))
    img = np.zeros(pixels.shape[0], dtype=np.complex64)
    for tx in range(n_tx):
        vec_tx = pixels - tx_positions[tx][None, :]            # (P, 3)
        dist_tx = d_tx[:, tx]
        cos_tx = (vec_tx @ tx_dirs[tx]) / np.maximum(dist_tx, 1e-6)
        tx_mask = cos_tx >= cos_fan
        if not np.any(tx_mask):
            continue

        total = dist_tx[:, None] + d_rx                        # (P, N_RX)
        bins = np.clip((total * inv_bin + 0.5).astype(np.int64), 0, n_samples - 1)
        gathered = np.take_along_axis(iq[tx], bins.T, axis=1)  # (N_RX, P) complex
        gathered = np.where(rx_face, gathered, 0.0)
        contrib = gathered.sum(axis=0)                         # (P,)
        img += np.where(tx_mask, contrib, 0.0)

    return np.abs(img)


def cartesian_pixels(half_extent_mm: float, n_pix: int):
    xs = np.linspace(-half_extent_mm, half_extent_mm, n_pix, dtype=np.float32)
    zs = np.linspace(-half_extent_mm, half_extent_mm, n_pix, dtype=np.float32)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    pixels = np.stack([X.ravel(), np.zeros_like(X.ravel()), Z.ravel()], axis=-1)
    return pixels, xs, zs


def polar_pixels(max_depth_mm: float, n_depth: int, n_angle: int):
    """Unwrapped (depth x angle) sampling grid matching the legacy IVUS view.

    Angle convention matches ``IVUSProbe``: direction ``(sin th, 0, cos th)``.
    """
    depths = np.linspace(0.0, max_depth_mm, n_depth, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False, dtype=np.float32)
    R, TH = np.meshgrid(depths, angles, indexing="ij")     # (n_depth, n_angle)
    X = (R * np.sin(TH)).ravel()
    Z = (R * np.cos(TH)).ravel()
    pixels = np.stack([X, np.zeros_like(X), Z], axis=-1)
    return pixels, depths, np.degrees(angles)


def to_db(img: np.ndarray, floor_db: float = -DYNAMIC_RANGE_DB,
          ref_percentile: float = DISPLAY_PERCENTILE) -> np.ndarray:
    """Log-compress an envelope image.

    The reference level is a high percentile rather than the global maximum, so
    the handful of very bright specular wall pixels clip at 0 dB instead of
    compressing all the diffuse tissue speckle into the noise floor (this is the
    practical equivalent of the gain/reject the legacy display applies).
    """
    ref = np.percentile(img, ref_percentile)
    if ref <= 0:
        ref = img.max()
    if ref <= 0:
        return np.full_like(img, floor_db)
    db = 20.0 * np.log10(img / ref + 1e-9)
    return np.clip(db, floor_db, 0.0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_rf_slice(rf, tx_index, tx_positions, rx_positions, t_far, out_path):
    n_rx, n_samples = rf.shape[1], rf.shape[2]
    slice_2d = rf[tx_index]
    extent = [0.0, t_far, n_rx - 1, 0.0]
    fig, ax = plt.subplots(figsize=(9, 5))
    norm = np.percentile(np.abs(slice_2d), 99.5) or 1.0
    im = ax.imshow(slice_2d / norm, aspect="auto", cmap="seismic",
                   vmin=-1, vmax=1, extent=extent)
    # Predicted moveout for the inner and outer wall along the TX radial line.
    r_tx = tx_positions[tx_index]
    radial = r_tx / (np.linalg.norm(r_tx) + 1e-9)
    for wall_r, color in ((INNER_WALL_MM, "cyan"), (OUTER_WALL_MM, "lime")):
        p = radial * wall_r
        t_total = np.linalg.norm(p - r_tx) + np.linalg.norm(rx_positions - p, axis=1)
        m = t_total < t_far
        ax.plot(t_total[m], np.where(m)[0], color=color, lw=1.0, ls="--", alpha=0.8,
                label=f"wall r={wall_r} mm")
    ax.set_xlabel("Total path length (TX + RX) [mm]")
    ax.set_ylabel("RX element index")
    ax.set_title(f"IVUS channel slice rf[tx={tx_index}, :, :] with predicted wall moveout")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="RF (normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cartesian(img_db, xs, zs, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [xs[0], xs[-1], zs[-1], zs[0]]
    im = ax.imshow(img_db, cmap="gray", extent=extent, vmin=-DYNAMIC_RANGE_DB, vmax=0)
    ax.set_title("DAS reconstruction from channel cube (Cartesian)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    for wall_r, color in ((INNER_WALL_MM, "cyan"), (OUTER_WALL_MM, "lime")):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(wall_r * np.cos(th), wall_r * np.sin(th), color=color, lw=1.0,
                ls="--", alpha=0.7, label=f"true wall r={wall_r} mm")
    ax.plot(0, 0, "r+", ms=10, mew=2, label="catheter")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_unwrapped_vs_legacy(unwrap_db, depths, angles, legacy_img, sim, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ext = [angles[0], angles[-1], depths[-1], depths[0]]
    im0 = axes[0].imshow(unwrap_db, cmap="gray", aspect="auto", extent=ext,
                         vmin=-DYNAMIC_RANGE_DB, vmax=0)
    axes[0].set_title("DAS from channel cube (unwrapped: angle x depth)")
    axes[0].set_xlabel("Angle [deg]")
    axes[0].set_ylabel("Depth [mm]")
    fig.colorbar(im0, ax=axes[0], label="dB")

    im1 = axes[1].imshow(
        legacy_img, cmap="gray", aspect="auto",
        extent=[sim.get_min_x(), sim.get_max_x(), sim.get_max_z(), sim.get_min_z()],
        vmin=-60, vmax=0,
    )
    axes[1].set_title("Legacy simulate() B-mode (unwrapped: angle x depth)")
    axes[1].set_xlabel("Angle [deg]")
    axes[1].set_ylabel("Depth [mm]")
    fig.colorbar(im1, ax=axes[1], label="dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[setup] Building vessel world and IVUS probe...")
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
    cc.enable_cuda_timing = True

    print(f"[capture] IVUS synthetic aperture: {NUM_ANGULAR} TX x {NUM_ANGULAR} RX "
          f"x {BUFFER_SIZE} samples ...")
    t0 = time.time()
    cap = simulator.simulate_channel_capture(probe, cc)
    dt = time.time() - t0
    rf = cap["rf"]
    tx_positions = cap["tx_positions"]
    rx_positions = cap["rx_positions"]
    c = float(cap["speed_of_sound"])
    t_far = float(cap["t_far"])
    print(f"[capture] done in {dt:.2f} s. rf.shape={rf.shape}, c={c} mm/us, t_far={t_far} mm")
    print(f"[capture] rf min/max={rf.min():.3e}/{rf.max():.3e}, "
          f"nonzero fraction={np.count_nonzero(rf)/rf.size:.4f}")

    # Radial firing / facing directions for the ring (unit, xz plane).
    tx_dirs = tx_positions / (np.linalg.norm(tx_positions, axis=1, keepdims=True) + 1e-9)
    rx_dirs = rx_positions / (np.linalg.norm(rx_positions, axis=1, keepdims=True) + 1e-9)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "channel_rf.npz"),
        rf=rf, tx_positions=tx_positions, rx_positions=rx_positions,
        tx_dirs=tx_dirs, rx_dirs=rx_dirs, speed_of_sound=c, t_far=t_far,
        inner_wall_mm=INNER_WALL_MM, outer_wall_mm=OUTER_WALL_MM,
    )
    print(f"[save] {OUTPUT_DIR}/channel_rf.npz")

    # ---- Condition the raw cube: pulse modulation + TGC + analytic signal ----
    print("[dsp] Applying transmit/receive pulse, TGC, and Hilbert (IQ)...")
    iq = condition_rf(rf, FREQUENCY_MHZ, c, t_far, PULSE_CYCLES)
    gain = tgc_gain(iq.shape[-1], t_far, TGC_DB_PER_MM, TGC_MAX_DB)
    iq *= gain[None, None, :]

    # Show the pulsed (bipolar) RF for the moveout check.
    rf_pulsed = np.real(iq)
    plot_rf_slice(rf_pulsed, NUM_ANGULAR // 2, tx_positions, rx_positions, t_far,
                  os.path.join(OUTPUT_DIR, "01_rf_slice.png"))
    print("[plot] 01_rf_slice.png")

    max_depth_mm = OUTER_WALL_MM + 2.0

    # ---- Cartesian reconstruction (native IVUS cross-section view) --------
    print("[beamform] DAS synthetic-aperture IQ reconstruction (Cartesian)...")
    t0 = time.time()
    pix_c, xs, zs = cartesian_pixels(half_extent_mm=OUTER_WALL_MM + 1.5, n_pix=400)
    img_c = das_beamform(iq, tx_positions, tx_dirs, rx_positions, rx_dirs, t_far, pix_c)
    img_c = img_c.reshape(400, 400)
    print(f"[beamform] Cartesian done in {time.time()-t0:.2f} s.")
    plot_cartesian(to_db(img_c), xs, zs,
                   os.path.join(OUTPUT_DIR, "02_das_reconstruction.png"))
    print("[plot] 02_das_reconstruction.png")

    # ---- Unwrapped reconstruction (depth x angle) for legacy comparison --
    print("[beamform] DAS synthetic-aperture IQ reconstruction (unwrapped)...")
    t0 = time.time()
    n_depth, n_angle = 512, 512
    pix_u, depths, angles = polar_pixels(max_depth_mm, n_depth, n_angle)
    img_u = das_beamform(iq, tx_positions, tx_dirs, rx_positions, rx_dirs, t_far, pix_u)
    img_u = img_u.reshape(n_depth, n_angle)
    print(f"[beamform] unwrapped done in {time.time()-t0:.2f} s.")

    # Legacy scanline B-mode for the same phantom (unwrapped angle x depth).
    print("[legacy] simulate() for comparison...")
    sp = rs.SimParams()
    sp.conv_psf = True
    sp.buffer_size = 4096
    sp.t_far = max_depth_mm
    sp.b_mode_size = (512, 512)
    legacy_img = simulator.simulate(probe, sp)
    plot_unwrapped_vs_legacy(to_db(img_u), depths, angles, legacy_img, simulator,
                             os.path.join(OUTPUT_DIR, "03_das_vs_legacy.png"))
    print("[plot] 03_das_vs_legacy.png")
    print(f"\nAll outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
