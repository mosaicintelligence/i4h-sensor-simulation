"""Shared helpers for the P_035_PointScatter polar-domain extractors.

Used by p035_extract_noise.py (E5) and p035_extract_tgc.py (E4).
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running from any directory.
sys.path.insert(0, str(Path(__file__).parent))
from extract_calibration_inputs import parse_scattering_box_dxf, Wire  # noqa: E402


@dataclass(frozen=True)
class PolarFrame:
    file: str
    arr: np.ndarray              # (num_theta, num_r) palette values, float32
    dr_mm: float
    num_theta: int
    num_r: int
    gain_slider: float
    diameter_mm: float
    depth_mm: float
    pixel_spacing_mm: float
    apparatus_cx_px: float
    apparatus_cy_px: float
    theta0_rad: float
    chirality: int
    radial_scale: float
    rows: int
    cols: int


def load_polar_dataset(
    polar_dir: Path,
    align_csv: Path,
    meta_csv: Path,
) -> list[PolarFrame]:
    """Load every polar .npy alongside its alignment + acquisition metadata."""
    fits: dict[str, dict[str, str]] = {}
    with align_csv.open() as f:
        for row in csv.DictReader(f):
            fits[row["file"]] = row

    metas: dict[str, dict[str, str]] = {}
    with meta_csv.open() as f:
        for row in csv.DictReader(f):
            metas[row["file"]] = row

    manifest: dict[str, dict[str, str]] = {}
    with (polar_dir / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            manifest[row["file"]] = row

    out: list[PolarFrame] = []
    for name, m in manifest.items():
        if name not in fits or name not in metas:
            continue
        arr = np.load(polar_dir / f"{name}.npy").astype(np.float32)
        fit = fits[name]
        meta = metas[name]
        out.append(PolarFrame(
            file=name,
            arr=arr,
            dr_mm=float(m["dr_mm"]),
            num_theta=int(m["num_theta"]),
            num_r=int(m["num_r"]),
            gain_slider=float(m["gain_slider"]),
            diameter_mm=float(m["diameter_mm"]),
            depth_mm=float(m["depth_mm"]),
            pixel_spacing_mm=float(meta["pixel_spacing_mm"]),
            apparatus_cx_px=float(fit["apparatus_cx_px"]),
            apparatus_cy_px=float(fit["apparatus_cy_px"]),
            theta0_rad=math.radians(float(fit["theta0_deg"])),
            chirality=int(fit["chirality"]),
            radial_scale=float(fit["radial_scale"]),
            rows=int(meta["rows"]),
            cols=int(meta["cols"]),
        ))
    return out


def predict_wire_polar_positions(
    frame: PolarFrame, wires: list[Wire]
) -> list[tuple[int, float, float]]:
    """Polar (theta_design_rad, r_mm) of each wire AS IT APPEARS in `frame.arr`.

    Accounts for the per-frame apparatus-vs-device wiggle and radial_scale.
    Polar origin = device center = image center.
    """
    px = frame.pixel_spacing_mm
    cx_dev = frame.cols / 2.0
    cy_dev = frame.rows / 2.0
    out = []
    for w in wires:
        a = frame.theta0_rad + frame.chirality * math.radians(w.theta_deg)
        x_img = frame.apparatus_cx_px + w.r_mm * frame.radial_scale * math.cos(a) / px
        y_img = frame.apparatus_cy_px - w.r_mm * frame.radial_scale * math.sin(a) / px
        dx_mm = (x_img - cx_dev) * px
        dy_mm = -(y_img - cy_dev) * px
        r_polar = math.hypot(dx_mm, dy_mm)
        theta_polar_image = math.atan2(dy_mm, dx_mm)
        theta_polar_design = (frame.chirality * (theta_polar_image - frame.theta0_rad)) \
            % (2.0 * math.pi)
        out.append((w.index, theta_polar_design, r_polar))
    return out


def build_anechoic_mask(
    frame: PolarFrame,
    wires: list[Wire],
    exclude_angle_deg: float = 15.0,
    inner_radial_mm: float = 5.0,
    outer_radial_margin_mm: float = 1.0,
    reject_palette: float = 11.0,
) -> np.ndarray:
    """Boolean mask (num_theta, num_r) selecting the anechoic background.

    Excludes:
      * inner ringdown / reject zone (r < inner_radial_mm)
      * outer FOV edge (r > depth_mm - outer_radial_margin_mm)
      * angular sectors within +/- exclude_angle_deg of any wire (all depths,
        since each wire's PSF tail and any reverberation streak it creates
        spans most of the radial axis)
      * pixels at the device's reject floor (palette <= reject_palette)
    """
    num_theta = frame.num_theta
    num_r = frame.num_r
    dr = frame.dr_mm

    mask = np.ones((num_theta, num_r), dtype=bool)

    r_grid = (np.arange(num_r) + 0.5) * dr
    inner = r_grid < inner_radial_mm
    outer = r_grid > (frame.depth_mm - outer_radial_margin_mm)
    mask[:, inner | outer] = False

    preds = predict_wire_polar_positions(frame, wires)
    half_width_bins = int(round(
        math.radians(exclude_angle_deg) / (2.0 * math.pi / num_theta)
    ))
    for _, theta_d, r_w in preds:
        if r_w > frame.depth_mm:
            # Wire would be out of this frame's FOV: still might reverberate
            # back, so we exclude its angular sector to be safe if its
            # nominal angle lands in [0, depth].
            pass
        i_theta = int(round(theta_d / (2.0 * math.pi) * num_theta))
        for d in range(-half_width_bins, half_width_bins + 1):
            i = (i_theta + d) % num_theta
            mask[i, :] = False

    mask &= frame.arr > reject_palette
    return mask
