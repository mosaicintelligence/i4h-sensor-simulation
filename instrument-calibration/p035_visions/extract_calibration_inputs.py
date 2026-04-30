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

"""Pre-processing utilities for the P_035_PointScatter capture.

This script does the human-checkable preliminaries needed before the
quantitative calibration analyses on the Volcano s5 / Visions PV .035
"point scatterer" capture:

  1. Parse the AutoCAD sketch of the scattering box and emit a clean
     wire-positions CSV (`r_mm`, `theta_deg`, `x_mm`, `y_mm`) plus a
     companion JSON with box / catheter geometry.
  2. Convert each DICOM frame in the capture folder to a PNG so the
     operator can eyeball brightness / saturation, with an optional
     overlay of the expected wire positions transformed into the
     image's pixel grid (using the DICOM `PhysicalDeltaX`).
  3. Auto-estimate a per-frame rotation theta0 (the rigid-body angle by
     which the catheter element ring sits relative to the DXF +x axis)
     by template-matching the design wire ring against bright peaks.
     Re-render overlays with the rotation applied.

Usage
-----
    python3 instrument-calibration/p035_visions/extract_calibration_inputs.py \
        --capture-dir P_035_PointScatter \
        --out-dir     P_035_PointScatter/derived \
        --overlay

Run from the workspace root so relative paths resolve.

The output layout is::

    derived/
      wire_positions.csv          # per-wire (r, theta, x, y) in mm
      box_geometry.json           # outer disc, catheter disc
      pngs/FILE0000.png ...       # raw grayscale, 1:1 with DICOM pixels
      pngs_overlay/FILE0000.png   # same with wire ring + design positions
      brightness_summary.csv      # min / mean / p99 / max grayscale per file

Designed to be reproducible: every output is regenerated from the inputs
on every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import ezdxf
import numpy as np
import pydicom
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Wire:
    index: int
    x_mm: float
    y_mm: float

    @property
    def r_mm(self) -> float:
        return math.hypot(self.x_mm, self.y_mm)

    @property
    def theta_deg(self) -> float:
        # CCW from +x, in [0, 360).
        a = math.degrees(math.atan2(self.y_mm, self.x_mm))
        return a + 360.0 if a < 0 else a


@dataclass(frozen=True)
class BoxGeometry:
    outer_radius_mm: float
    catheter_radius_mm: float
    wire_design_radius_mm: float


# --------------------------------------------------------------------- DXF ---


def parse_scattering_box_dxf(dxf_path: Path) -> tuple[list[Wire], BoxGeometry]:
    """Extract wires + box geometry from the IVUS scattering-box AutoCAD file.

    The sketch contains exactly 11 CIRCLE entities and nothing else:
      - one with r ~= 50 mm at origin (outer enclosure)
      - one with r ~= 1.9 mm at origin (catheter cross-section)
      - nine with r = 0.5 mm at the wire (x, y) locations

    The classification is done by radius binning so it remains robust if the
    drawing units / scale change.
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    circles = [e for e in msp if e.dxftype() == "CIRCLE"]
    if not circles:
        raise RuntimeError(f"No CIRCLE entities found in {dxf_path}")

    # Sort by radius so big -> outer, medium -> catheter, small (~0.5 mm) -> wires.
    by_radius = sorted(circles, key=lambda c: c.dxf.radius, reverse=True)
    outer = by_radius[0]
    catheter = by_radius[1]
    wire_circles = [c for c in by_radius[2:] if c.dxf.radius < 1.0]

    if abs(outer.dxf.center.x) > 0.5 or abs(outer.dxf.center.y) > 0.5:
        raise RuntimeError(
            f"Outer circle is not at the origin: "
            f"({outer.dxf.center.x:.3f}, {outer.dxf.center.y:.3f}) mm"
        )
    if abs(catheter.dxf.center.x) > 0.5 or abs(catheter.dxf.center.y) > 0.5:
        raise RuntimeError(
            f"Catheter circle is not at the origin: "
            f"({catheter.dxf.center.x:.3f}, {catheter.dxf.center.y:.3f}) mm"
        )

    wires_unsorted = [
        Wire(index=-1, x_mm=float(c.dxf.center.x), y_mm=float(c.dxf.center.y))
        for c in wire_circles
    ]
    # Index in order of increasing radial distance from the catheter.
    wires_by_r = sorted(wires_unsorted, key=lambda w: w.r_mm)
    wires = [Wire(index=i + 1, x_mm=w.x_mm, y_mm=w.y_mm) for i, w in enumerate(wires_by_r)]

    geom = BoxGeometry(
        outer_radius_mm=float(outer.dxf.radius),
        catheter_radius_mm=float(catheter.dxf.radius),
        wire_design_radius_mm=float(wire_circles[0].dxf.radius),
    )
    return wires, geom


def write_wire_positions_csv(wires: Iterable[Wire], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wire_index", "r_mm", "theta_deg", "x_mm", "y_mm"])
        for wire in wires:
            w.writerow(
                [
                    wire.index,
                    f"{wire.r_mm:.4f}",
                    f"{wire.theta_deg:.4f}",
                    f"{wire.x_mm:.4f}",
                    f"{wire.y_mm:.4f}",
                ]
            )


def write_box_geometry_json(geom: BoxGeometry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(geom), f, indent=2)
        f.write("\n")


# ------------------------------------------------------------------ DICOM ---


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    instance_number: int
    rows: int
    cols: int
    pixel_spacing_mm: float
    array: np.ndarray  # uint8, shape (rows, cols)


def load_frame(path: Path) -> FrameInfo:
    """Read a single P_035 DICOM and return its grayscale array + metadata."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=False)
    arr = ds.pixel_array
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # PALETTE COLOR is decoded as RGB by pydicom; collapse to luma.
        arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    elif arr.ndim != 2:
        raise RuntimeError(f"Unexpected pixel_array shape {arr.shape} in {path}")

    # PhysicalDeltaX is in cm in the IVUS region descriptor; PixelSpacing is in mm.
    pixel_spacing_mm: float | None = None
    if "SequenceOfUltrasoundRegions" in ds and len(ds.SequenceOfUltrasoundRegions) > 0:
        region = ds.SequenceOfUltrasoundRegions[0]
        if hasattr(region, "PhysicalDeltaX"):
            pixel_spacing_mm = float(region.PhysicalDeltaX) * 10.0
    if pixel_spacing_mm is None and hasattr(ds, "PixelSpacing"):
        pixel_spacing_mm = float(ds.PixelSpacing[0])
    if pixel_spacing_mm is None:
        raise RuntimeError(f"No pixel spacing in {path}")

    return FrameInfo(
        path=path,
        instance_number=int(getattr(ds, "InstanceNumber", -1)),
        rows=int(ds.Rows),
        cols=int(ds.Columns),
        pixel_spacing_mm=pixel_spacing_mm,
        array=arr.astype(np.uint8),
    )


def save_grayscale_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def save_overlay_png(
    frame: FrameInfo,
    wires: list[Wire],
    geom: BoxGeometry,
    path: Path,
    theta0_deg: float = 0.0,
    chirality: int = +1,
    detected_peaks: list[tuple[int, float, float, float]] | None = None,
) -> None:
    """Draw expected wire centers + outer ring + catheter on the frame.

    Parameters
    ----------
    theta0_deg
        Catheter clocking angle to apply to the design wire azimuths
        before drawing. The corrected position is at world coordinates
        (r * cos(chirality*theta + theta0), r * sin(chirality*theta + theta0)).
    chirality
        +1 if the device displays the polar disc with the same handedness
        as the DXF (CCW positive), -1 if it mirror-flips the azimuth.
    detected_peaks
        Optional list of (wire_index, r_meas_mm, theta_meas_deg, snr) tuples,
        drawn as small yellow circles to compare against the magenta crosses.
    """
    img = Image.fromarray(frame.array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    cx = frame.cols / 2.0
    cy = frame.rows / 2.0
    px_per_mm = 1.0 / frame.pixel_spacing_mm

    def world_to_px(x_mm: float, y_mm: float) -> tuple[float, float]:
        # DICOM image y points DOWN, world y points UP -> negate y.
        return (cx + x_mm * px_per_mm, cy - y_mm * px_per_mm)

    def polar_to_px(r_mm: float, theta_deg: float) -> tuple[float, float]:
        a = math.radians(theta_deg)
        return world_to_px(r_mm * math.cos(a), r_mm * math.sin(a))

    # Catheter outline (cyan).
    r_px = geom.catheter_radius_mm * px_per_mm
    draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=(0, 255, 255), width=1)

    # Display field of view (green) — the smaller of disc inscribed in image
    # vs the box outer radius.
    fov_radius_mm = min(geom.outer_radius_mm, frame.cols * frame.pixel_spacing_mm / 2.0)
    r_px = fov_radius_mm * px_per_mm
    draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=(0, 200, 0), width=1)

    # Each wire: a small magenta cross + radius/index label, only if it
    # falls inside this frame's field of view.
    half_diag_mm = (frame.pixel_spacing_mm * min(frame.rows, frame.cols)) / 2.0
    for w in wires:
        if w.r_mm > half_diag_mm:
            continue
        px, py = polar_to_px(w.r_mm, chirality * w.theta_deg + theta0_deg)
        s = 4
        draw.line([px - s, py, px + s, py], fill=(255, 0, 255), width=1)
        draw.line([px, py - s, px, py + s], fill=(255, 0, 255), width=1)
        draw.text((px + s + 2, py - 6), f"{w.index}:{w.r_mm:.0f}mm", fill=(255, 0, 255))

    # Detected peaks (yellow circles).
    if detected_peaks:
        for idx, r_meas, theta_meas, _snr in detected_peaks:
            px, py = polar_to_px(r_meas, theta_meas)
            s = 5
            draw.ellipse(
                [px - s, py - s, px + s, py + s],
                outline=(255, 255, 0),
                width=1,
            )

    # Frame metadata in the corner.
    draw.text(
        (4, 4),
        (
            f"{frame.path.name}  inst={frame.instance_number}  "
            f"px={frame.pixel_spacing_mm:.3f} mm  "
            f"theta0={theta0_deg:+.2f} deg  chirality={chirality:+d}"
        ),
        fill=(255, 255, 0),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ----------------------------------------------------------- rotation est ---


def estimate_catheter_center_px(arr: np.ndarray) -> tuple[float, float]:
    """Locate the catheter dome center in the image.

    The s5 burns a bright/saturated dome (the AR-subtracted ring-down
    column near r=0) at the polar origin. We pick the centroid of the
    saturated cluster within a 30 px window of the image center.
    """
    rows, cols = arr.shape
    cx0, cy0 = cols / 2.0, rows / 2.0
    half = 30
    x0, x1 = max(0, int(cx0 - half)), min(cols, int(cx0 + half))
    y0, y1 = max(0, int(cy0 - half)), min(rows, int(cy0 + half))
    sub = arr[y0:y1, x0:x1].astype(np.float32)
    mask = sub >= max(200, sub.max() - 5)
    if mask.sum() < 4:
        return cx0, cy0
    ys, xs = np.nonzero(mask)
    return float(xs.mean() + x0), float(ys.mean() + y0)


def polar_resample(
    arr: np.ndarray,
    cx: float,
    cy: float,
    px_per_mm: float,
    r_max_mm: float,
    n_radial: int,
    n_az: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the Cartesian frame into a (n_radial, n_az) polar grid.

    Returns (polar, rs_mm, azs_rad). Pixels outside the image are zeroed.
    """
    rows, cols = arr.shape
    rs_mm = np.linspace(0.0, r_max_mm, n_radial)
    azs_rad = np.linspace(0.0, 2.0 * math.pi, n_az, endpoint=False)
    rg, ag = np.meshgrid(rs_mm, azs_rad, indexing="ij")
    xs = cx + rg * px_per_mm * np.cos(ag)
    ys = cy - rg * px_per_mm * np.sin(ag)  # image y points down
    xi = np.round(xs).astype(np.int32)
    yi = np.round(ys).astype(np.int32)
    in_bounds = (xi >= 0) & (xi < cols) & (yi >= 0) & (yi < rows)
    polar = np.zeros_like(rg, dtype=np.float32)
    polar[in_bounds] = arr[yi[in_bounds], xi[in_bounds]]
    return polar, rs_mm, azs_rad


def estimate_theta0_for_frame(
    frame: FrameInfo,
    wires: list[Wire],
    radial_tol_mm: float = 2.4,
    n_az: int = 720,
    n_radial_per_mm: int = 4,
) -> tuple[float | None, list[tuple[int, float, float, float]]]:
    """Estimate the catheter clocking angle theta0 for one frame.

    Strategy: GLOBAL rotation search.
      1. Polar-resample the frame.
      2. Per design wire (that fits in the FOV), sum brightness over a
         thin annulus around its design radius -> azimuth profile I_n(theta).
      3. For every candidate theta0 in [0, 360 deg) -- and separately for
         the chirality-flipped case -- compute
              S(theta0) = sum_n I_n(theta_n + theta0)
         and pick the (chirality, theta0) with the largest S.
      4. After fixing theta0, locate each wire's bright peak in its
         annulus around (theta_design + theta0) and return per-wire
         (r_meas, theta_meas, snr) for diagnostics.

    `radial_tol_mm` is wider than 1.5 mm to absorb modest catheter
    decentering inside the box (the wires are 5 mm apart so 2.4 mm is
    still unambiguous).
    """
    rows, cols = frame.array.shape
    fov_mm = (min(rows, cols) * frame.pixel_spacing_mm) / 2.0 - 0.5
    cx, cy = estimate_catheter_center_px(frame.array)
    px_per_mm = 1.0 / frame.pixel_spacing_mm

    visible_wires = [w for w in wires if 6.0 <= w.r_mm <= fov_mm]
    if len(visible_wires) < 2:
        return None, []

    # Polar resample once to a fine grid.
    r_max_mm = min(fov_mm, max(w.r_mm for w in visible_wires) + radial_tol_mm)
    n_radial = max(64, int(round(r_max_mm * n_radial_per_mm)))
    polar, rs_mm, azs_rad = polar_resample(
        frame.array,
        cx,
        cy,
        px_per_mm,
        r_max_mm=r_max_mm,
        n_radial=n_radial,
        n_az=n_az,
    )

    # Per-wire azimuth profile: max brightness across the radial annulus,
    # minus the per-annulus median (suppresses speckle background).
    az_profiles: dict[int, np.ndarray] = {}
    for w in visible_wires:
        mask = np.abs(rs_mm - w.r_mm) <= radial_tol_mm
        annulus = polar[mask, :]  # (n_r, n_az)
        prof = annulus.max(axis=0)
        prof = prof - np.median(prof)
        # Robust normalize so high-gain frames don't dominate.
        scale = np.median(np.abs(prof)) * 1.4826
        if scale <= 0:
            scale = prof.std() or 1.0
        az_profiles[w.index] = prof / scale

    # Build the design-wire delta train, evaluated on the same n_az grid.
    # For each candidate theta0 and chirality s in {+1, -1}, the score is
    #   S(theta0, s) = sum_n az_profiles[n][round( (s*theta_n + theta0) * n_az / 360 )].
    # Implement via per-wire circular shift + sum.
    az_step_deg = 360.0 / n_az

    best = (-math.inf, 0.0, +1)
    for chirality in (+1, -1):
        accum = np.zeros(n_az, dtype=np.float32)
        for w in visible_wires:
            shift = int(round((chirality * w.theta_deg) / az_step_deg))
            # az_profiles[w.index] is indexed by theta_meas; we want, at
            # theta0 index k, the value at theta_meas = k + chirality*theta_n
            # so we shift the profile LEFT by `shift` indices.
            accum += np.roll(az_profiles[w.index], -shift)
        peak_idx = int(np.argmax(accum))
        peak_val = float(accum[peak_idx])
        if peak_val > best[0]:
            best = (peak_val, peak_idx * az_step_deg, chirality)
    _, theta0_deg, chirality = best

    # Per-wire diagnostics: peak in the annulus around expected theta.
    detections: list[tuple[int, float, float, float]] = []
    half_az_window_deg = 360.0 / len(visible_wires) / 2.0  # +/- half-spacing
    for w in visible_wires:
        expected_theta = (chirality * w.theta_deg + theta0_deg) % 360.0
        # window around expected_theta
        d_az = ((np.degrees(azs_rad) - expected_theta + 540.0) % 360.0) - 180.0
        win = np.abs(d_az) <= half_az_window_deg
        prof = az_profiles[w.index]
        if not win.any():
            continue
        sub = prof.copy()
        sub[~win] = -np.inf
        peak_az_idx = int(np.argmax(sub))
        peak_val = float(sub[peak_az_idx])
        # SNR: peak value (already normalized by robust std) is in "sigmas"
        snr = peak_val
        if snr < 3.0:
            continue
        # Refine radial peak inside the annulus.
        mask = np.abs(rs_mm - w.r_mm) <= radial_tol_mm
        annulus = polar[mask, :]
        radial_profile = annulus[:, peak_az_idx]
        r_peak_idx = int(np.argmax(radial_profile))
        r_meas_mm = float(rs_mm[mask][r_peak_idx])
        theta_meas_deg = float(np.degrees(azs_rad[peak_az_idx])) % 360.0
        detections.append((w.index, r_meas_mm, theta_meas_deg, snr))

    # Encode chirality in theta0: the convention we'll use downstream is
    # that the device azimuth is (chirality * theta_design + theta0) mod 360.
    # We pack chirality into the sign of an extra return field by adding
    # 1000 to theta0 when chirality is -1; downstream readers should use
    # ((theta0 + 540) mod 360) - 180 for plotting and read the chirality
    # from the alignment.csv chirality column. Here we simply return the
    # raw degrees [0, 360) and let the caller record chirality separately.
    # NOTE: we currently encode chirality by storing theta0 in [0, 360) and
    # using a parallel per-frame chirality (handled at the call site).
    return theta0_deg, detections, chirality  # type: ignore[return-value]


def write_alignment_csv(
    rows: list[
        tuple[
            FrameInfo,
            float | None,
            int,
            list[tuple[int, float, float, float]],
        ]
    ],
    wires: list[Wire],
    path: Path,
) -> None:
    """Write per-frame theta0 estimate + per-wire residual diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_index = {w.index: w for w in wires}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "instance_number",
                "pixel_spacing_mm",
                "field_of_view_mm",
                "chirality",
                "n_wires_detected",
                "theta0_deg",
                "wire_indices_detected",
                "per_wire_radial_error_mm",
                "per_wire_azimuth_residual_deg",
                "per_wire_snr",
            ]
        )
        for fr, theta0, chirality, dets in rows:
            fov = fr.cols * fr.pixel_spacing_mm
            indices = ";".join(str(d[0]) for d in dets)
            if theta0 is None:
                w.writerow(
                    [
                        fr.path.name,
                        fr.instance_number,
                        f"{fr.pixel_spacing_mm:.4f}",
                        f"{fov:.2f}",
                        "",
                        0,
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue
            radial_err = []
            azim_resid = []
            snrs = []
            for idx, r_meas, theta_meas, snr in dets:
                wd = by_index[idx]
                radial_err.append(f"{r_meas - wd.r_mm:+.3f}")
                expected = (chirality * wd.theta_deg + theta0) % 360.0
                resid = (theta_meas - expected + 540.0) % 360.0 - 180.0
                azim_resid.append(f"{resid:+.3f}")
                snrs.append(f"{snr:.2f}")
            w.writerow(
                [
                    fr.path.name,
                    fr.instance_number,
                    f"{fr.pixel_spacing_mm:.4f}",
                    f"{fov:.2f}",
                    f"{chirality:+d}",
                    len(dets),
                    f"{theta0:+.3f}",
                    indices,
                    ";".join(radial_err),
                    ";".join(azim_resid),
                    ";".join(snrs),
                ]
            )


def write_brightness_summary(frames: list[FrameInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "instance_number",
                "rows",
                "cols",
                "pixel_spacing_mm",
                "field_of_view_mm",
                "min_gray",
                "mean_gray",
                "p50_gray",
                "p99_gray",
                "max_gray",
                "saturated_fraction",
            ]
        )
        for fr in frames:
            arr = fr.array
            field_of_view = fr.cols * fr.pixel_spacing_mm
            saturated = float(np.mean(arr >= 239))
            w.writerow(
                [
                    fr.path.name,
                    fr.instance_number,
                    fr.rows,
                    fr.cols,
                    f"{fr.pixel_spacing_mm:.4f}",
                    f"{field_of_view:.2f}",
                    int(arr.min()),
                    f"{arr.mean():.2f}",
                    int(np.percentile(arr, 50)),
                    int(np.percentile(arr, 99)),
                    int(arr.max()),
                    f"{saturated:.5f}",
                ]
            )


# -------------------------------------------------------------------- main ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_capture = repo_root / "P_035_PointScatter"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--capture-dir",
        type=Path,
        default=default_capture,
        help="Folder containing the FILEnnnn DICOMs and the DXF sketch.",
    )
    p.add_argument(
        "--dxf-name",
        default="IVUS Scattering Box - Sketch 1.dxf",
        help="DXF file name inside --capture-dir.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write derived outputs (defaults to <capture-dir>/derived).",
    )
    p.add_argument(
        "--overlay",
        action="store_true",
        help="Also write per-frame PNGs with wire + box overlay.",
    )
    p.add_argument(
        "--align",
        action="store_true",
        help=(
            "Auto-estimate theta0 (catheter clocking angle) per frame "
            "by template-matching the design wire ring against bright "
            "peaks. Writes alignment.csv. The rig-wide theta0 is the "
            "value from the highest-confidence frame and gets applied "
            "to ALL overlays (the catheter doesn't rotate between "
            "captures within one experiment)."
        ),
    )
    p.add_argument(
        "--theta0",
        type=float,
        default=None,
        help=(
            "Override the auto-estimated rig-wide theta0 (deg). "
            "Use this if the auto-estimate is visibly wrong; tune "
            "by 5-deg steps until the magenta crosses sit on the "
            "bright wire echoes in the cleanest D60/gain-54 frame."
        ),
    )
    p.add_argument(
        "--chirality",
        type=int,
        choices=(-1, +1),
        default=None,
        help=(
            "Override the auto-estimated chirality (-1 = device "
            "mirror-flips the polar disc azimuth vs. the DXF, "
            "+1 = same handedness)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    capture_dir: Path = args.capture_dir.resolve()
    out_dir: Path = (args.out_dir or capture_dir / "derived").resolve()

    if not capture_dir.is_dir():
        print(f"capture dir not found: {capture_dir}", file=sys.stderr)
        return 2
    dxf_path = capture_dir / args.dxf_name
    if not dxf_path.is_file():
        print(f"DXF not found: {dxf_path}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    pngs_dir = out_dir / "pngs"
    overlay_dir = out_dir / "pngs_overlay"

    # 1) DXF -> wire CSV + box geometry.
    wires, geom = parse_scattering_box_dxf(dxf_path)
    write_wire_positions_csv(wires, out_dir / "wire_positions.csv")
    write_box_geometry_json(geom, out_dir / "box_geometry.json")
    print(f"DXF parsed: {len(wires)} wires inside outer disc r={geom.outer_radius_mm:.2f} mm,")
    print(f"  catheter r={geom.catheter_radius_mm:.2f} mm")
    for w in wires:
        print(f"    wire {w.index:2d}: r={w.r_mm:6.3f} mm  theta={w.theta_deg:7.2f} deg")

    # 2) DICOMs -> PNGs + brightness summary.
    dicom_paths = sorted(p for p in capture_dir.iterdir() if p.name.startswith("FILE"))
    if not dicom_paths:
        print(f"No FILEnnnn DICOMs found in {capture_dir}", file=sys.stderr)
        return 2

    frames: list[FrameInfo] = []
    for dp in dicom_paths:
        frames.append(load_frame(dp))

    # Optional rotation estimation (per-frame theta0 + chirality + peaks).
    # Then collapse to a single rig-wide (chirality, theta0).
    per_frame: dict[
        str, tuple[float | None, int, list[tuple[int, float, float, float]]]
    ] = {}
    rig_theta0: float = 0.0
    rig_chirality: int = +1

    if args.align:
        rows = []
        for fr in frames:
            theta0, dets, chirality = estimate_theta0_for_frame(fr, wires)  # type: ignore[misc]
            per_frame[fr.path.name] = (theta0, chirality, dets)
            rows.append((fr, theta0, chirality, dets))
        write_alignment_csv(rows, wires, out_dir / "alignment.csv")
        print(f"Alignment CSV -> {out_dir / 'alignment.csv'}")

        # Rig-wide consensus: the catheter doesn't rotate between frames,
        # so the true theta0 is the MODE of the per-frame estimates.
        # Spurious local maxima scatter; the right answer clusters.
        candidates = [
            (theta0 % 360.0, chir, len(dets), sum(d[3] for d in dets), name)
            for name, (theta0, chir, dets) in per_frame.items()
            if theta0 is not None and len(dets) >= 3
        ]
        if candidates:
            # Vote in 20-degree bins for each chirality.
            bin_width = 20.0
            n_bins = int(round(360.0 / bin_width))
            best_bin = (-math.inf, 0, +1, 0.0, [])
            for chir in (+1, -1):
                cands_c = [c for c in candidates if c[1] == chir]
                if not cands_c:
                    continue
                votes = [[] for _ in range(n_bins)]
                for theta0, _c, n, sumsnr, name in cands_c:
                    b = int(theta0 // bin_width) % n_bins
                    votes[b].append((theta0, n, sumsnr, name))
                # Score each bin window of width 3*bin_width (60 deg) centered on it
                for b in range(n_bins):
                    members = []
                    for db in (-1, 0, 1):
                        members += votes[(b + db) % n_bins]
                    if not members:
                        continue
                    score = (
                        len(members) * 1000.0
                        + sum(m[2] for m in members)
                    )
                    if score > best_bin[0]:
                        best_bin = (score, len(members), chir, b * bin_width + bin_width / 2.0, members)
            _, n_in_cluster, rig_chirality, _bin_center, members = best_bin
            # Final theta0 is the SNR-weighted circular mean of the cluster.
            rad = np.radians([m[0] for m in members])
            wts = np.array([m[2] for m in members], dtype=np.float64)
            rig_theta0 = math.degrees(
                math.atan2(np.sum(wts * np.sin(rad)), np.sum(wts * np.cos(rad)))
            ) % 360.0
            print(
                f"  rig-wide theta0 (mode of {n_in_cluster}/{len(candidates)} per-frame estimates): "
                f"{rig_theta0:+.2f} deg  chirality={rig_chirality:+d}"
            )
            print("  cluster members:")
            for theta0, n, sumsnr, name in sorted(members, key=lambda m: -m[2]):
                print(f"    {name}  theta0={theta0:+7.2f}  n={n}  sum_SNR={sumsnr:.1f}")
            print("  outliers (per-frame estimates not in the consensus cluster):")
            in_cluster_names = {m[3] for m in members}
            for theta0, chir, n, sumsnr, name in candidates:
                if name in in_cluster_names and chir == rig_chirality:
                    continue
                print(
                    f"    {name}  theta0={theta0:+7.2f}  chir={chir:+d}  "
                    f"n={n}  sum_SNR={sumsnr:.1f}"
                )

    if args.theta0 is not None:
        rig_theta0 = args.theta0
        print(f"  CLI override: rig-wide theta0 = {rig_theta0:+.2f} deg")
    if args.chirality is not None:
        rig_chirality = args.chirality
        print(f"  CLI override: rig-wide chirality = {rig_chirality:+d}")

    # Save plain + overlay PNGs (overlay uses the rig-wide theta0).
    for fr in frames:
        save_grayscale_png(fr.array, pngs_dir / f"{fr.path.name}.png")
        if args.overlay:
            _t, _c, dets = per_frame.get(fr.path.name, (0.0, +1, []))
            save_overlay_png(
                fr,
                wires,
                geom,
                overlay_dir / f"{fr.path.name}.png",
                theta0_deg=rig_theta0,
                chirality=rig_chirality,
                detected_peaks=dets,
            )

    write_brightness_summary(frames, out_dir / "brightness_summary.csv")
    print(f"Converted {len(frames)} DICOMs -> {pngs_dir}")
    if args.overlay:
        print(f"Overlay PNGs -> {overlay_dir}")
    print(f"Brightness summary -> {out_dir / 'brightness_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
