#!/usr/bin/env python3
"""Un-scan-convert the P_035_PointScatter DICOMs to polar (theta, r).

This is the foundational artifact for every downstream parameter fit
(PSF / TGC / ring-down / gain LUT). Working in polar coordinates around
the device origin makes each calibration much cleaner:

  * a wire becomes a clean axial x azimuthal Gaussian patch instead of
    a smeared parallelogram;
  * the ring-down is a single 1D depth profile (mean over theta);
  * TGC is a single 1D depth profile of the wire-free angular sectors;
  * the multi-gain LUT is well-defined per-wire amplitude vs slider.

Coordinate convention
---------------------
Polar origin = device center = image center (s5 polar origin pixel).
Polar angle  = "design frame" -- after per-frame un-rotation by theta0
               and un-flipping by chirality, design wire k at angle
               theta_design[k] sits at the same polar bin in EVERY frame.
Polar radius = mm of displayed range (not corrected for radial_scale --
               the polar image preserves what the device displayed; the
               apparatus offset / radial_scale just shift where the wires
               land within it).

Because the apparatus center is offset from the device center by the
"wiggle", design wire k does NOT land at exactly (theta_d, r_d) in the
polar image -- it lands at the polar coords of (apparatus center +
wire offset) seen from the device. The script computes those exact
predicted polar positions and overlays them on each PNG so the
geometry can be sanity-checked.

Outputs
-------
  derived/polar/{FILE}.npy  : float32 (num_theta, num_r), pixel values in [0, 255]
  derived/polar/{FILE}.png  : grayscale rendering with predicted wires + axes
  derived/polar/manifest.csv: per-frame {file, num_theta, num_r, dr_mm,
                              dtheta_rad, depth_mm, gain, diameter_mm,
                              theta0_deg, chirality, radial_scale}
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
import pydicom
from PIL import Image, ImageDraw

# Allow running both as a script and an importable module.
sys.path.insert(0, str(Path(__file__).parent))
from extract_calibration_inputs import (  # noqa: E402
    parse_scattering_box_dxf,
    BoxGeometry,
    Wire,
)


@dataclass(frozen=True)
class FrameAlign:
    """Subset of alignment_fit.csv columns we need to do an unwrap."""
    file: str
    pixel_spacing_mm: float
    rows: int
    cols: int
    apparatus_cx_px: float
    apparatus_cy_px: float
    theta0_rad: float
    chirality: int
    radial_scale: float
    wire_offset: int
    depth_mm: float
    gain_slider: float
    diameter_mm: float


def load_alignment(align_csv: Path, meta_csv: Path) -> list[FrameAlign]:
    fits: dict[str, dict[str, str]] = {}
    with align_csv.open() as f:
        for row in csv.DictReader(f):
            fits[row["file"]] = row

    out: list[FrameAlign] = []
    with meta_csv.open() as f:
        for row in csv.DictReader(f):
            name = row["file"]
            if name not in fits:
                continue
            fit = fits[name]
            out.append(
                FrameAlign(
                    file=name,
                    pixel_spacing_mm=float(row["pixel_spacing_mm"]),
                    rows=int(row["rows"]),
                    cols=int(row["cols"]),
                    apparatus_cx_px=float(fit["apparatus_cx_px"]),
                    apparatus_cy_px=float(fit["apparatus_cy_px"]),
                    theta0_rad=math.radians(float(fit["theta0_deg"])),
                    chirality=int(fit["chirality"]),
                    radial_scale=float(fit["radial_scale"]),
                    wire_offset=int(fit["wire_offset"]),
                    depth_mm=float(row["depth_mm"]),
                    gain_slider=float(row["gain_slider"]),
                    diameter_mm=float(row["diameter_mm"]),
                )
            )
    return out


def read_dicom_array(path: Path) -> np.ndarray:
    """Read an 8-bit grayscale image out of the s5 PALETTE COLOR DICOM.

    The s5 stores the B-mode image as a single-channel index into a
    256-entry grayscale palette where palette[k] = (k, k, k) on this
    dataset, so the index IS the brightness. We return that index plane
    as uint8.

    Handles single-frame stills (P_035: shape (H,W) or (H,W,C)) and the
    multi-frame video clips in ivus_test_0508 (shape (N,H,W) or
    (N,H,W,C)). Multi-frame clips are collapsed to the median across N
    to denoise speckle and average the SA-orientation brightness
    variability that the catheter's continuous rotation produces.
    """
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]  # palette channels are identical on this scanner
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]
    elif arr.ndim == 3:
        arr = np.median(arr, axis=0)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected pixel_array shape {arr.shape} in {path}")
    return arr.astype(np.uint8)


def polar_resample(
    arr: np.ndarray,
    cx_px: float,
    cy_px: float,
    pixel_spacing_mm: float,
    theta0_rad: float,
    chirality: int,
    num_theta: int,
    num_r: int,
    dr_mm: float,
) -> np.ndarray:
    """Bilinear polar resample around (cx_px, cy_px).

    The output theta axis is in DESIGN FRAME: theta=0 corresponds to
    design angle 0 (i.e. the same physical wire direction across every
    frame). The image-frame angle is recovered as
      image_theta = theta0_rad + chirality * theta_design
    """
    rows, cols = arr.shape
    theta_design = (np.arange(num_theta, dtype=np.float64) + 0.5) * (2.0 * math.pi / num_theta)
    r_mm = (np.arange(num_r, dtype=np.float64) + 0.5) * dr_mm

    image_theta = theta0_rad + chirality * theta_design
    cos_t = np.cos(image_theta)[:, None]    # (num_theta, 1)
    sin_t = np.sin(image_theta)[:, None]
    r_px = (r_mm / pixel_spacing_mm)[None, :]  # (1, num_r)

    xs = cx_px + r_px * cos_t                  # (num_theta, num_r), image x
    ys = cy_px - r_px * sin_t                  # image y (down)

    # Bilinear sample
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)

    valid = (x0 >= 0) & (x1 < cols) & (y0 >= 0) & (y1 < rows)
    x0c = np.clip(x0, 0, cols - 1)
    x1c = np.clip(x1, 0, cols - 1)
    y0c = np.clip(y0, 0, rows - 1)
    y1c = np.clip(y1, 0, rows - 1)

    a = arr.astype(np.float32)
    v = (
        a[y0c, x0c] * (1 - wx) * (1 - wy)
        + a[y0c, x1c] * wx * (1 - wy)
        + a[y1c, x0c] * (1 - wx) * wy
        + a[y1c, x1c] * wx * wy
    )
    v[~valid] = 0.0
    return v


def predict_wire_polar(
    fit: FrameAlign, wires: list[Wire], device_cx_px: float, device_cy_px: float
) -> list[tuple[int, float, float]]:
    """Return [(design_index, theta_design_rad, r_mm_from_device)] for each wire.

    The returned theta/r are the polar coordinates AS THEY APPEAR in the
    polar image (i.e. accounting for the apparatus/device wiggle and
    the radial_scale).
    """
    px = fit.pixel_spacing_mm
    out = []
    for w in wires:
        # Image-frame pixel position of the wire (apparatus + design):
        a = fit.theta0_rad + fit.chirality * math.radians(w.theta_deg)
        x_img = fit.apparatus_cx_px + w.r_mm * fit.radial_scale * math.cos(a) / px
        y_img = fit.apparatus_cy_px - w.r_mm * fit.radial_scale * math.sin(a) / px
        # Vector from device center to wire, in mm world coords (y up):
        dx_mm = (x_img - device_cx_px) * px
        dy_mm = -(y_img - device_cy_px) * px
        r_polar = math.hypot(dx_mm, dy_mm)
        # Image-frame polar angle (math CCW from +x), then un-rotate to design frame.
        theta_polar_image = math.atan2(dy_mm, dx_mm)
        theta_polar_design = fit.chirality * (theta_polar_image - fit.theta0_rad)
        # Wrap to [0, 2*pi)
        theta_polar_design = theta_polar_design % (2.0 * math.pi)
        out.append((w.index, theta_polar_design, r_polar))
    return out


def render_polar_png(
    polar: np.ndarray,
    fit: FrameAlign,
    wires: list[Wire],
    geom: BoxGeometry,
    device_cx_px: float,
    device_cy_px: float,
    out_path: Path,
    dr_mm: float,
) -> None:
    """Render the polar image as a (theta x r) grayscale PNG with overlays.

    Layout: rows = theta axis (top = 0 deg, increasing downward), cols = r axis
    (left = device center, right = max range). This is the convention used
    by most ultrasound papers when displaying unwrapped IVUS.
    """
    num_theta, num_r = polar.shape

    # Stretch r axis to a square-ish display so wire spots aren't pancaked.
    target_h = num_theta
    target_w = max(num_r, num_theta // 2)
    img = Image.fromarray(polar.astype(np.uint8), mode="L").resize(
        (target_w, target_h), resample=Image.BILINEAR
    ).convert("RGB")
    draw = ImageDraw.Draw(img)
    sx = target_w / num_r       # px per radial bin
    sy = target_h / num_theta   # px per theta bin

    # ---- predicted wire crosshairs ----
    preds = predict_wire_polar(fit, wires, device_cx_px, device_cy_px)
    for idx, th_d, r_mm in preds:
        i_theta = (th_d / (2.0 * math.pi)) * num_theta
        j_r = r_mm / dr_mm
        x = j_r * sx
        y = i_theta * sy
        s = 6
        draw.line([x - s, y, x + s, y], fill=(255, 0, 255), width=1)
        draw.line([x, y - s, x, y + s], fill=(255, 0, 255), width=1)
        draw.text((x + 7, y - 8), f"{idx}", fill=(255, 0, 255))

    # ---- depth ticks at every 5 mm ----
    for r_tick in range(0, int(num_r * dr_mm) + 1, 5):
        x = (r_tick / dr_mm) * sx
        draw.line([x, 0, x, target_h], fill=(0, 200, 0), width=1)
        draw.text((x + 2, 2), f"{r_tick}mm", fill=(0, 200, 0))

    # ---- ring-down zone shading (inner geom.catheter_radius_mm) ----
    rd_x = (geom.catheter_radius_mm / dr_mm) * sx
    draw.line([rd_x, 0, rd_x, target_h], fill=(0, 255, 255), width=1)
    draw.text((rd_x + 2, target_h - 14), "catheter OD", fill=(0, 255, 255))

    # ---- annotation banner ----
    banner = (
        f"{fit.file}  gain={fit.gain_slider:.0f}  D={fit.diameter_mm:.0f}mm  "
        f"theta0={math.degrees(fit.theta0_rad):+.1f}deg  chir={fit.chirality:+d}  "
        f"scale={fit.radial_scale:.4f}"
    )
    draw.text((4, target_h - 28), banner, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("P_035_PointScatter"))
    p.add_argument("--align-csv", type=Path,
                   default=Path("P_035_PointScatter/derived/alignment_fit.csv"))
    p.add_argument("--meta-csv", type=Path,
                   default=Path("P_035_PointScatter/derived/frames_meta.csv"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("P_035_PointScatter/derived/polar"))
    p.add_argument("--dxf", type=Path,
                   default=Path("P_035_PointScatter/IVUS Scattering Box - Sketch 1.dxf"))
    p.add_argument("--num-theta", type=int, default=1024,
                   help="Polar angle samples per frame (default 1024 -- ~4x oversampling vs s5 SA)")
    p.add_argument("--no-pngs", action="store_true")
    args = p.parse_args(argv)

    if not args.align_csv.exists():
        print(f"Missing {args.align_csv}; run p035_fit_alignment.py first.", file=sys.stderr)
        return 1
    if not args.meta_csv.exists():
        print(f"Missing {args.meta_csv}; run extract_metadata.py first.", file=sys.stderr)
        return 1

    fits = load_alignment(args.align_csv, args.meta_csv)
    print(f"Loaded {len(fits)} aligned frames.")

    wires, geom = parse_scattering_box_dxf(args.dxf)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for fit in fits:
        path = args.dataset / fit.file
        arr = read_dicom_array(path)

        # Device center = image center (polar origin of the s5 scan converter).
        cx_dev = fit.cols / 2.0
        cy_dev = fit.rows / 2.0

        # Radial sampling: 1 polar bin = 1 displayed pixel. Goes out to
        # DepthOfScanField (= half of the displayed diameter).
        dr_mm = fit.pixel_spacing_mm
        num_r = int(round(fit.depth_mm / dr_mm))

        polar = polar_resample(
            arr, cx_dev, cy_dev, fit.pixel_spacing_mm,
            fit.theta0_rad, fit.chirality,
            args.num_theta, num_r, dr_mm,
        )
        np.save(args.out_dir / f"{fit.file}.npy", polar.astype(np.float32))

        if not args.no_pngs:
            render_polar_png(polar, fit, wires, geom, cx_dev, cy_dev,
                             args.out_dir / f"{fit.file}.png", dr_mm)

        manifest_rows.append({
            "file": fit.file,
            "num_theta": args.num_theta,
            "num_r": num_r,
            "dr_mm": f"{dr_mm:.6f}",
            "dtheta_rad": f"{2.0 * math.pi / args.num_theta:.8f}",
            "depth_mm": f"{fit.depth_mm:.3f}",
            "gain_slider": f"{fit.gain_slider:.1f}",
            "diameter_mm": f"{fit.diameter_mm:.1f}",
            "theta0_deg": f"{math.degrees(fit.theta0_rad):+.3f}",
            "chirality": fit.chirality,
            "radial_scale": f"{fit.radial_scale:.4f}",
        })
        print(f"  {fit.file}: ({args.num_theta} x {num_r}) dr={dr_mm:.4f}mm "
              f"depth={fit.depth_mm:.1f}mm gain={fit.gain_slider:.0f} "
              f"D={fit.diameter_mm:.0f}mm")

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nWrote {len(manifest_rows)} polar arrays to {args.out_dir}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
