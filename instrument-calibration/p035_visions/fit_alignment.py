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

"""Fit per-frame alignment from human wire annotations.

Reads ``derived/wire_annotations.json`` (produced by
``p035_annotate_wires.py``) and solves, independently for every frame,
the four-parameter rigid alignment model

  x_pred(k) = cx + (r_k / px) * scale * cos(chir * theta_k + theta0)
  y_pred(k) = cy - (r_k / px) * scale * sin(chir * theta_k + theta0)

where (cx, cy) is the catheter center in image pixels, (r_k, theta_k)
are the design (radius, azimuth) of the wire, ``px`` is the DICOM
pixel spacing in mm, and (chir, theta0, scale) are the catheter
chirality (+/-1), clocking angle (deg), and effective radial scale
(== c_actual / 1540 m/s, dimensionless).

Two ambiguities are searched exhaustively per frame:

  * **Wire-index offset.** The annotator wasn't sure whether the first
    visible wire was design wire 2 or design wire 3. We fit both
    hypotheses and pick the one with the smaller residual.
  * **Chirality.** The s5 may flip the polar disc azimuth relative to
    the DXF. Both signs are tried.

The model is linear in (cx, cy, a, b) where a = scale*cos(theta0)/px
and b = scale*sin(theta0)/px, so each candidate is a single
least-squares solve. No nonlinear optimization needed.

Outputs
-------
  derived/alignment_fit.csv     # one row per frame
  derived/alignment_fit.json    # same data, structured
  derived/pngs_fit/<file>.png   # overlay using the fitted alignment

Usage
-----
    python3 tools/p035_fit_alignment.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from extract_calibration_inputs import (  # noqa: E402
    BoxGeometry,
    FrameInfo,
    Wire,
    estimate_catheter_center_px,
    load_frame,
    parse_scattering_box_dxf,
)


# ----------------------------------------------------------------- types ---


@dataclass(frozen=True)
class FrameFit:
    file: str
    pixel_spacing_mm: float
    # Apparatus center: origin of the wire spiral (fitted from clicks).
    apparatus_cx_px: float
    apparatus_cy_px: float
    apparatus_cx_mm_off_image_center: float
    apparatus_cy_mm_off_image_center: float
    # Device center: origin of the catheter beams (centroid of the ring-down
    # dome in image space). Used for PSF / ring-down / FOV analyses.
    device_cx_px: float
    device_cy_px: float
    # Offset from device to apparatus (the catheter "wiggle" within the jig).
    apparatus_minus_device_x_mm: float
    apparatus_minus_device_y_mm: float
    apparatus_minus_device_mag_mm: float
    # Wire-spiral geometry parameters.
    theta0_deg: float
    chirality: int
    radial_scale: float            # ratio c_actual / 1540 m/s
    wire_offset: int               # -1 = first click is design wire 1; 0 = wire 2; etc.
    n_clicks_used: int
    excluded_user_wire_indices: list[int]
    rms_residual_mm: float
    max_residual_mm: float
    used_design_indices: list[int]
    per_click_residual_mm: list[float]


# ------------------------------------------------------------------ I/O ---


def load_annotations(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_fit_csv(fits: list[FrameFit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "pixel_spacing_mm",
                "apparatus_cx_px",
                "apparatus_cy_px",
                "device_cx_px",
                "device_cy_px",
                "apparatus_off_image_center_x_mm",
                "apparatus_off_image_center_y_mm",
                "apparatus_minus_device_x_mm",
                "apparatus_minus_device_y_mm",
                "wiggle_mm",
                "theta0_deg",
                "chirality",
                "radial_scale",
                "implied_sound_speed_m_per_s",
                "wire_offset",
                "first_visible_wire",
                "n_clicks_used",
                "excluded_user_wire_indices",
                "rms_residual_mm",
                "max_residual_mm",
                "used_design_indices",
            ]
        )
        for f_ in fits:
            w.writerow(
                [
                    f_.file,
                    f_.pixel_spacing_mm,
                    f"{f_.apparatus_cx_px:.3f}",
                    f"{f_.apparatus_cy_px:.3f}",
                    f"{f_.device_cx_px:.3f}",
                    f"{f_.device_cy_px:.3f}",
                    f"{f_.apparatus_cx_mm_off_image_center:+.3f}",
                    f"{f_.apparatus_cy_mm_off_image_center:+.3f}",
                    f"{f_.apparatus_minus_device_x_mm:+.3f}",
                    f"{f_.apparatus_minus_device_y_mm:+.3f}",
                    f"{f_.apparatus_minus_device_mag_mm:.3f}",
                    f"{f_.theta0_deg:+.3f}",
                    f"{f_.chirality:+d}",
                    f"{f_.radial_scale:.5f}",
                    f"{f_.radial_scale * 1540.0:.1f}",
                    f_.wire_offset,
                    2 + f_.wire_offset,
                    f_.n_clicks_used,
                    ";".join(str(i) for i in f_.excluded_user_wire_indices),
                    f"{f_.rms_residual_mm:.4f}",
                    f"{f_.max_residual_mm:.4f}",
                    ";".join(str(i) for i in f_.used_design_indices),
                ]
            )


def write_fit_json(fits: list[FrameFit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "schema_version": 2,
                "frames": [
                    {
                        "file": f_.file,
                        "pixel_spacing_mm": f_.pixel_spacing_mm,
                        # Apparatus center: where the wire spiral is centered.
                        # Use this when transforming design (r, theta) to image px.
                        "apparatus_center_px": [f_.apparatus_cx_px, f_.apparatus_cy_px],
                        "apparatus_offset_from_image_center_mm": [
                            f_.apparatus_cx_mm_off_image_center,
                            f_.apparatus_cy_mm_off_image_center,
                        ],
                        # Device center: the s5's polar-origin pixel
                        # (geometric center of the ultrasound region).
                        # Use this for PSF, ring-down, and FOV analyses.
                        "device_center_px": [f_.device_cx_px, f_.device_cy_px],
                        # Vector from device to apparatus, in mm. Magnitude is
                        # the per-frame catheter "wiggle" inside the wire jig.
                        "apparatus_minus_device_mm": [
                            f_.apparatus_minus_device_x_mm,
                            f_.apparatus_minus_device_y_mm,
                        ],
                        "apparatus_wiggle_mm": f_.apparatus_minus_device_mag_mm,
                        "theta0_deg": f_.theta0_deg,
                        "chirality": f_.chirality,
                        "radial_scale": f_.radial_scale,
                        "implied_sound_speed_m_per_s": f_.radial_scale * 1540.0,
                        "wire_offset": f_.wire_offset,
                        "first_visible_wire": 2 + f_.wire_offset,
                        "n_clicks_used": f_.n_clicks_used,
                        "excluded_user_wire_indices": f_.excluded_user_wire_indices,
                        "rms_residual_mm": f_.rms_residual_mm,
                        "max_residual_mm": f_.max_residual_mm,
                        "used_design_indices": f_.used_design_indices,
                        "per_click_residual_mm": f_.per_click_residual_mm,
                    }
                    for f_ in fits
                ],
            },
            f,
            indent=2,
        )
        f.write("\n")


# ----------------------------------------------------------------- fit ---


def device_center_px(annotation: dict) -> tuple[float, float]:
    """Return the catheter scan-conversion polar origin for this frame.

    The IVUS device center is the polar origin of the s5's scan
    converter -- the pixel that corresponds to radial range r=0 along
    every A-line. By construction it is a FIXED pixel inside the
    DICOM, not something we should detect from image content.

    DICOM `SequenceOfUltrasoundRegions` may declare it explicitly via
    `ReferencePixelX0`/`ReferencePixelY0`. The Volcano s5 does NOT
    write those tags, so we fall back to the standard convention:
    the geometric center of the declared region (which on the s5 is
    the entire 500x500 image, hence (250, 250)).

    The bright ring-down dome is approximately co-located with this
    pixel but its visual centroid is not necessarily exactly there
    because the dome is hollow and can be asymmetric -- which is why
    we trust the scan converter, not pixel detection.
    """
    rows = int(annotation["rows"])
    cols = int(annotation["cols"])
    return (cols / 2.0, rows / 2.0)


def _solve_lls(
    xy_obs: np.ndarray,         # shape (N, 2)
    r_design_mm: np.ndarray,    # shape (N,)
    theta_design_deg: np.ndarray,  # shape (N,)
    px_spacing_mm: float,
    chirality: int,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """Solve the linear LS problem for one (chirality, offset) hypothesis.

    Returns (cx, cy, theta0_deg, scale, rms_residual_mm, residuals_mm).
    """
    n = xy_obs.shape[0]
    phi = np.radians(chirality * theta_design_deg)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    r_px = r_design_mm / px_spacing_mm  # design radius in image pixels

    # Build the linear system: x_obs = cx + r_px * cphi * a - r_px * sphi * b
    #                          y_obs = cy - r_px * sphi * a - r_px * cphi * b
    # where a = scale*cos(theta0)/1, b = scale*sin(theta0)/1.
    A = np.zeros((2 * n, 4))
    rhs = np.zeros(2 * n)
    A[:n, 0] = 1.0                # cx coef in x rows
    A[n:, 1] = 1.0                # cy coef in y rows
    A[:n, 2] = r_px * cphi        # a coef in x rows
    A[:n, 3] = -r_px * sphi       # b coef in x rows
    A[n:, 2] = -r_px * sphi       # a coef in y rows
    A[n:, 3] = -r_px * cphi       # b coef in y rows
    rhs[:n] = xy_obs[:, 0]
    rhs[n:] = xy_obs[:, 1]

    sol, _resid_lstsq, _rank, _sv = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy, a, b = sol
    pred = A @ sol
    resid_px = pred - rhs
    # Per-click residual magnitude in pixels -> mm.
    resid_x = resid_px[:n]
    resid_y = resid_px[n:]
    per_click_resid_mm = np.hypot(resid_x, resid_y) * px_spacing_mm
    rms_mm = float(np.sqrt(np.mean(per_click_resid_mm ** 2)))

    scale = math.hypot(a, b)
    theta0_deg = math.degrees(math.atan2(b, a)) % 360.0
    return float(cx), float(cy), theta0_deg, scale, rms_mm, per_click_resid_mm


def fit_frame(
    annotation: dict,
    wires: list[Wire],
    frame: FrameInfo,
    offset_range: tuple[int, int] = (-1, 1),
) -> FrameFit | None:
    """Fit one frame's annotation by trying both chirality and wire offsets.

    The device center is taken as the s5's polar-origin pixel (image
    center on this scanner) so the apparatus-to-device offset can be
    reported as the per-frame catheter wiggle inside the wire jig.
    """
    if annotation.get("unreadable"):
        return None
    px = float(annotation["pixel_spacing_mm"])
    rows = int(annotation["rows"])
    cols = int(annotation["cols"])

    # User-specified per-frame click exclusions (e.g. obvious misclicks
    # on speckle).  Stored under "excluded_user_wire_indices" in the
    # annotation JSON.
    excluded = set(int(i) for i in annotation.get("excluded_user_wire_indices", []))

    # Filter clicks to keep only the valid (non-skipped, non-excluded) ones.
    clicks: list[tuple[int, float, float]] = []
    for c in annotation.get("clicks", []):
        if c.get("wire_index") is None:
            continue
        if int(c["wire_index"]) in excluded:
            continue
        clicks.append((int(c["wire_index"]), float(c["x_px"]), float(c["y_px"])))
    if len(clicks) < 3:  # 4 unknowns; need >=3 clicks to be even nearly identifiable
        return None

    wires_by_index = {w.index: w for w in wires}
    best: tuple[float, tuple[float, float, float, float, int, int, list[int],
                              float, float, np.ndarray]] | None = None

    for chirality in (+1, -1):
        for offset in range(offset_range[0], offset_range[1] + 1):
            # Map user-labeled wire index -> design wire index by adding offset.
            design_idx = []
            xy = []
            for user_wi, x, y in clicks:
                di = user_wi + offset
                if di not in wires_by_index:
                    design_idx = []
                    break
                design_idx.append(di)
                xy.append((x, y))
            if not design_idx:
                continue

            r_design = np.array([wires_by_index[i].r_mm for i in design_idx])
            t_design = np.array([wires_by_index[i].theta_deg for i in design_idx])
            xy_arr = np.array(xy)
            cx, cy, theta0, scale, rms, per_resid = _solve_lls(
                xy_arr, r_design, t_design, px, chirality
            )
            if best is None or rms < best[0]:
                best = (rms, (cx, cy, theta0, scale, chirality, offset,
                              design_idx, rms, float(per_resid.max()), per_resid))

    if best is None:
        return None
    (_rms, (apx, apy, theta0, scale, chirality, offset, design_idx,
            rms, max_resid_mm, per_resid)) = best

    # The device center is the polar origin of the s5's scan
    # converter -- a fixed pixel inside the DICOM, NOT something to
    # detect from image content. See `device_center_px` for details.
    dx, dy = device_center_px(annotation)

    apparatus_minus_device_x_mm = (apx - dx) * px
    apparatus_minus_device_y_mm = -(apy - dy) * px  # world y up
    wiggle_mm = math.hypot(apparatus_minus_device_x_mm, apparatus_minus_device_y_mm)

    return FrameFit(
        file=annotation["file"],
        pixel_spacing_mm=px,
        apparatus_cx_px=apx,
        apparatus_cy_px=apy,
        apparatus_cx_mm_off_image_center=(apx - cols / 2.0) * px,
        apparatus_cy_mm_off_image_center=-(apy - rows / 2.0) * px,
        device_cx_px=dx,
        device_cy_px=dy,
        apparatus_minus_device_x_mm=apparatus_minus_device_x_mm,
        apparatus_minus_device_y_mm=apparatus_minus_device_y_mm,
        apparatus_minus_device_mag_mm=wiggle_mm,
        theta0_deg=theta0,
        chirality=chirality,
        radial_scale=scale,
        wire_offset=offset,
        n_clicks_used=len(design_idx),
        excluded_user_wire_indices=sorted(excluded),
        rms_residual_mm=rms,
        max_residual_mm=max_resid_mm,
        used_design_indices=design_idx,
        per_click_residual_mm=[float(x) for x in per_resid],
    )


# ----------------------------------------------------------- overlay ---


def render_fit_overlay(
    frame: FrameInfo,
    fit: FrameFit | None,
    wires: list[Wire],
    geom: BoxGeometry,
    annotation: dict,
    out_path: Path,
) -> None:
    img = Image.fromarray(frame.array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    px_per_mm = 1.0 / frame.pixel_spacing_mm

    if fit is None:
        apx, apy = annotation.get("catheter_center_px", [frame.cols / 2, frame.rows / 2])
        dpx, dpy = apx, apy
        chirality = +1
        theta0 = 0.0
        scale = 1.0
    else:
        apx, apy = fit.apparatus_cx_px, fit.apparatus_cy_px
        dpx, dpy = fit.device_cx_px, fit.device_cy_px
        chirality = fit.chirality
        theta0 = fit.theta0_deg
        scale = fit.radial_scale

    def polar_to_px_apparatus(r_mm: float, theta_deg: float) -> tuple[float, float]:
        a = math.radians(theta_deg)
        return (apx + r_mm * scale * px_per_mm * math.cos(a),
                apy - r_mm * scale * px_per_mm * math.sin(a))

    # Excluded clicks for visual feedback.
    excluded = set(int(i) for i in annotation.get("excluded_user_wire_indices", []))

    # ----- Device center (cyan) -----
    # Catheter cross-section outline + center crosshair.
    r_px = geom.catheter_radius_mm * px_per_mm
    draw.ellipse([dpx - r_px, dpy - r_px, dpx + r_px, dpy + r_px],
                 outline=(0, 255, 255), width=1)
    draw.line([dpx - 7, dpy, dpx + 7, dpy], fill=(0, 255, 255), width=1)
    draw.line([dpx, dpy - 7, dpx, dpy + 7], fill=(0, 255, 255), width=1)
    draw.text((dpx + 8, dpy + 4), "device", fill=(0, 255, 255))

    # ----- Apparatus center (orange) + wiggle vector -----
    s = 5
    draw.line([apx - s, apy - s, apx + s, apy + s], fill=(255, 165, 0), width=1)
    draw.line([apx - s, apy + s, apx + s, apy - s], fill=(255, 165, 0), width=1)
    draw.text((apx + 8, apy - 12), "apparatus", fill=(255, 165, 0))
    if fit is not None and fit.apparatus_minus_device_mag_mm > 0.05:
        draw.line([dpx, dpy, apx, apy], fill=(255, 165, 0), width=1)

    # Field of view (green) — anchored on the device, not the apparatus.
    fov_radius_mm = min(geom.outer_radius_mm,
                        frame.cols * frame.pixel_spacing_mm / 2.0)
    r_px = fov_radius_mm * px_per_mm
    draw.ellipse([dpx - r_px, dpy - r_px, dpx + r_px, dpy + r_px],
                 outline=(0, 200, 0), width=1)

    # Predicted wire positions (magenta) using fitted alignment, anchored
    # on the apparatus center.
    for w in wires:
        px_, py_ = polar_to_px_apparatus(w.r_mm, chirality * w.theta_deg + theta0)
        if not (0 <= px_ < frame.cols and 0 <= py_ < frame.rows):
            continue
        s = 5
        draw.line([px_ - s, py_, px_ + s, py_], fill=(255, 0, 255), width=1)
        draw.line([px_, py_ - s, px_, py_ + s], fill=(255, 0, 255), width=1)
        draw.text((px_ + s + 2, py_ - 6), f"w{w.index}", fill=(255, 0, 255))

    # User clicks (yellow if used, dim red if excluded).
    for c in annotation.get("clicks", []):
        wi = c.get("wire_index")
        if wi is None:
            continue
        x, y = c["x_px"], c["y_px"]
        s = 6
        if int(wi) in excluded:
            color = (200, 60, 60)
            label = f"u{wi} (excl)"
        else:
            color = (255, 255, 0)
            label = f"u{wi}"
        draw.ellipse([x - s, y - s, x + s, y + s], outline=color, width=1)
        draw.text((x + s + 2, y + 2), label, fill=color)

    # Header text.
    if fit is not None:
        header = (
            f"{frame.path.name}  px={frame.pixel_spacing_mm:.3f} mm  "
            f"theta0={fit.theta0_deg:+.2f}  chir={fit.chirality:+d}  "
            f"scale={fit.radial_scale:.4f} (c={fit.radial_scale*1540:.0f} m/s)  "
            f"first_visible_wire={2 + fit.wire_offset}  "
            f"rms={fit.rms_residual_mm:.3f} mm  "
            f"wiggle={fit.apparatus_minus_device_mag_mm:.2f} mm"
        )
    else:
        header = f"{frame.path.name}  (NO FIT — too few clicks)"
    draw.text((4, 4), header, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# ---------------------------------------------------------------- main ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_capture = repo_root / "P_035_PointScatter"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", type=Path, default=default_capture)
    p.add_argument("--dxf-name", default="IVUS Scattering Box - Sketch 1.dxf")
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Annotations JSON (default: <capture-dir>/derived/wire_annotations.json).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <capture-dir>/derived).",
    )
    p.add_argument(
        "--offset-range",
        type=int,
        nargs=2,
        default=[-1, 1],
        metavar=("MIN", "MAX"),
        help=(
            "Range of wire-index offsets to try (default: -1 to +1). "
            "offset = -1 means the user's 'wire 2' click is actually "
            "design wire 1 (i.e. the inner wire was visible after all). "
            "offset = +1 means it's actually design wire 3 (inner two "
            "wires were missed)."
        ),
    )
    p.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip writing per-frame overlay PNGs (faster).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture_dir = args.capture_dir.resolve()
    out_dir = (args.out_dir or capture_dir / "derived").resolve()
    annotations_path = args.annotations or out_dir / "wire_annotations.json"

    if not annotations_path.is_file():
        print(f"annotations not found: {annotations_path}", file=sys.stderr)
        return 2

    wires, geom = parse_scattering_box_dxf(capture_dir / args.dxf_name)
    annotations = load_annotations(annotations_path)
    frames_by_name = {a["file"]: a for a in annotations["frames"]}

    # Pre-load the DICOMs once -- needed by the device-center detection.
    frame_cache: dict[str, FrameInfo] = {}
    for name in sorted(frames_by_name):
        frame_cache[name] = load_frame(capture_dir / name)

    fits: list[FrameFit] = []
    print(f"{'file':<10}  {'chir':>4} {'off':>3} {'th0':>8} {'scale':>7} "
          f"{'rms_mm':>7} {'max_mm':>7} {'wiggle':>7}  used_indices")
    for name in sorted(frames_by_name):
        ann = frames_by_name[name]
        fr = frame_cache[name]
        fit = fit_frame(
            ann,
            wires,
            fr,
            offset_range=tuple(args.offset_range),  # type: ignore[arg-type]
        )
        if fit is None:
            print(f"{name:<10}  (skipped — unreadable or <3 clicks)")
            continue
        fits.append(fit)
        print(
            f"{name:<10}  {fit.chirality:>+4d} {fit.wire_offset:>3d} "
            f"{fit.theta0_deg:>+8.2f} {fit.radial_scale:>7.4f} "
            f"{fit.rms_residual_mm:>7.3f} {fit.max_residual_mm:>7.3f} "
            f"{fit.apparatus_minus_device_mag_mm:>7.3f}  "
            f"{fit.used_design_indices}"
        )

    write_fit_csv(fits, out_dir / "alignment_fit.csv")
    write_fit_json(fits, out_dir / "alignment_fit.json")
    print(f"\nWrote: {out_dir / 'alignment_fit.csv'}")
    print(f"Wrote: {out_dir / 'alignment_fit.json'}")

    # Aggregate stats.
    if fits:
        scales = np.array([f.radial_scale for f in fits])
        offsets = np.array([f.wire_offset for f in fits])
        chirs = np.array([f.chirality for f in fits])
        wiggle = np.array([f.apparatus_minus_device_mag_mm for f in fits])
        wig_x = np.array([f.apparatus_minus_device_x_mm for f in fits])
        wig_y = np.array([f.apparatus_minus_device_y_mm for f in fits])
        # Device center is fixed at the polar origin (image center for s5).
        # The "wiggle" therefore is the apparatus-vs-device offset, i.e.
        # how far the wire jig sat off the catheter axis in each capture.
        print("\nSummary across all fitted frames:")
        print(f"  chirality: {dict(zip(*np.unique(chirs, return_counts=True)))}")
        print(f"  wire_offset: {dict(zip(*np.unique(offsets, return_counts=True)))}")
        print(f"  device center: fixed at image center (s5 polar origin)")
        print(f"  apparatus offset from device center (mm): "
              f"x median={np.median(wig_x):+.3f}  "
              f"y median={np.median(wig_y):+.3f}  "
              f"|.| median={np.median(wiggle):.3f}  max={wiggle.max():.3f}")
        print(f"  radial_scale median={np.median(scales):.4f}  "
              f"min={scales.min():.4f}  max={scales.max():.4f}  "
              f"std={scales.std():.4f}")
        # The s5 scan-converts assuming c_assumed = 1540 m/s (Service
        # Manual). For a wire at its design radius:
        #   r_image = r_design * (c_assumed / c_actual)
        # so radial_scale * c_actual = c_assumed. Solving for c_actual
        # gives the in-water sound speed implied by the fit (modulo any
        # phantom build tolerance, which empirically is ~2%).
        sos_implied = 1540.0 / np.median(scales)
        print(f"  implied in-water SoS median={sos_implied:.1f} m/s "
              f"(water at 20-25 C is ~1480-1497 m/s; "
              f"residual ~2% is jig tolerance)")

    # Per-frame overlays using the fitted alignment.
    if not args.no_overlays:
        out_pngs = out_dir / "pngs_fit"
        for fit in fits:
            frame = frame_cache[fit.file]
            ann = frames_by_name[fit.file]
            render_fit_overlay(frame, fit, wires, geom, ann,
                               out_pngs / f"{fit.file}.png")
        print(f"Wrote: {len(fits)} overlay PNGs -> {out_pngs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
