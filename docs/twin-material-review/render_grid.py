#!/usr/bin/env python3
"""Render Old (stock #20) vs New (twin_demo.yaml) 4x4 boards.

Same ivus-probe binary. Sixteen centerline_L stations in the pat05 calc window.
Writes poses.json always; writes grid.png when OptiX + sidecar scene helpers
are available.

  python docs/twin-material-review/render_grid.py
  python docs/twin-material-review/render_grid.py --poses-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TWIN_DEMO = REPO / "instrument-calibration" / "p035_visions" / "twin_demo.yaml"
VOLCANO = REPO / "instrument-calibration" / "p035_visions" / "volcano_s5i.yaml"

DEFAULT_POSE_JSON = Path(
    "/home/liam/Code/mosaic-ui-ux/twin_web/apps/twin/public/look/"
    "pat05_ivus_poses/centerline_L_pullback_1mm.json"
)
STATIONS_MM = tuple(range(50, 126, 5))  # 16 stations, calc window ~50–131 mm
CELL = 256


def pick_poses(pose_json: Path) -> list[dict]:
    doc = json.loads(pose_json.read_text(encoding="utf-8"))
    poses = doc["poses"]
    picked = []
    for station in STATIONS_MM:
        best = min(poses, key=lambda p: abs(float(p["stationMm"]) - station))
        picked.append(
            {
                "pathId": doc.get("pathId", "centerline_L"),
                "targetStationMm": station,
                "i": int(best["i"]),
                "stationMm": float(best["stationMm"]),
                "pullbackMm": float(best["pullbackMm"]),
                "position_mm": [float(x) for x in best["position_mm"]],
                "rotation_rad_xyz": [float(x) for x in best["rotation_rad_xyz"]],
            }
        )
    return picked


def write_poses_json(picked: list[dict], dest: Path) -> None:
    dest.write_text(
        json.dumps(
            {
                "pathId": "centerline_L",
                "note": "16 stations in the pat05 calc window. Same poses for stock and overlay.",
                "stations_mm": list(STATIONS_MM),
                "poses": picked,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _tile(frames: list[np.ndarray], *, title: str) -> np.ndarray:
    assert len(frames) == 16
    pad = 6
    title_h = 28
    label_h = 16
    cell = int(frames[0].shape[0])
    board_w = 4 * cell + 5 * pad
    board_h = title_h + 4 * (cell + label_h) + 5 * pad
    canvas = np.full((board_h, board_w), 32, dtype=np.uint8)
    for i, gray in enumerate(frames):
        r, c = divmod(i, 4)
        y = title_h + pad + r * (cell + label_h + pad)
        x = pad + c * (cell + pad)
        canvas[y : y + cell, x : x + cell] = gray
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((pad, 6), title, fill=(255, 255, 255))
        for i, station in enumerate(STATIONS_MM):
            r, c = divmod(i, 4)
            y = title_h + pad + r * (cell + label_h + pad) + cell + 1
            x = pad + c * (cell + pad)
            draw.text((x + 4, y), f"{station} mm", fill=(200, 200, 200))
        return np.asarray(img)
    except ImportError:
        return np.stack([canvas, canvas, canvas], axis=-1)


def _stack(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    gap = 16
    h = old.shape[0] + gap + new.shape[0]
    w = max(old.shape[1], new.shape[1])
    out = np.full((h, w, 3), 16, dtype=np.uint8)
    out[: old.shape[0], : old.shape[1]] = old
    out[old.shape[0] + gap :, : new.shape[1]] = new
    return out


def _save_png(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    except ImportError:
        pass
    import imageio.v2 as imageio

    imageio.imwrite(path, rgb)


def render_boards(picked: list[dict], args: argparse.Namespace) -> Path:
    twin_http = Path(
        os.environ.get("MOSAIC_TWIN_HTTP", "/home/liam/Code/mosaic-twin-ivus-http")
    ).resolve()
    if str(twin_http) not in sys.path:
        sys.path.insert(0, str(twin_http))

    from sim.nav_sim.twin_http.protocol import (
        cartesian_png_from_bmode,
        decode_gray_png,
        pose_frame_from_simulate_body,
    )
    from sim.nav_sim.twin_http.ui_scene import UiMeshRenderer

    lumen = Path(args.ui_lumen)
    calcium = Path(args.ui_calcium)
    if not lumen.is_file() or not calcium.is_file():
        raise FileNotFoundError(f"need --ui-lumen and --ui-calcium; got {lumen} {calcium}")

    common = dict(
        ui_lumen=lumen,
        ui_calcium=calcium,
        raysim_yaml_path=str(args.raysim_yaml),
        pose_dir=Path(args.pose_dir) if args.pose_dir else None,
        t_far_mm=17.5,
        cart_size=CELL,
        n_r=512,
        ivus_rays_per_scanline=4,
        scatter_coherence="frozen",
    )

    with tempfile.TemporaryDirectory(prefix="twin-mat-stock-") as stock_dir, tempfile.TemporaryDirectory(
        prefix="twin-mat-demo-"
    ) as demo_dir:
        stock = UiMeshRenderer(
            **common,
            apply_twin_overlay=False,
            scene_out_dir=Path(stock_dir),
        )
        demo = UiMeshRenderer(
            **common,
            materials_yaml=args.materials_yaml,
            apply_twin_overlay=True,
            scene_out_dir=Path(demo_dir),
        )

        def frames_for(renderer: UiMeshRenderer) -> list[np.ndarray]:
            out = []
            for row in picked:
                pose = pose_frame_from_simulate_body(
                    {
                        "position_mm": row["position_mm"],
                        "rotation_rad_xyz": row["rotation_rad_xyz"],
                        "output": "png",
                    },
                    frame_index=int(row["i"]),
                )
                polar = renderer.render_pose(pose)
                png = cartesian_png_from_bmode(
                    polar,
                    b_mode_size=renderer.b_mode_size,
                    t_far_mm=renderer.t_far_mm,
                    cart_size=CELL,
                )
                out.append(decode_gray_png(png))
                print(f"  station {row['targetStationMm']} mm", flush=True)
            return out

        print("stock #20 (no overlay)", flush=True)
        old_frames = frames_for(stock)
        print("demo-app overlay", flush=True)
        new_frames = frames_for(demo)
    old_board = _tile(old_frames, title="Old / stock  (#20 C++ table, no overlay)")
    new_board = _tile(new_frames, title="New / demo-app  (twin_demo.yaml overlay)")
    grid = _stack(old_board, new_board)
    dest = HERE / "grid.png"
    _save_png(dest, grid)
    print(f"wrote {dest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    cache = Path.home() / ".cache" / "twin-cta-sources"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-json", type=Path, default=DEFAULT_POSE_JSON)
    parser.add_argument("--poses-only", action="store_true")
    parser.add_argument("--raysim-yaml", type=Path, default=VOLCANO)
    parser.add_argument("--materials-yaml", type=Path, default=TWIN_DEMO)
    parser.add_argument(
        "--ui-lumen",
        default=str(
            cache
            / "data_AGORALABS_MOSAIC_CTA_CASES_20260421_processed_PAT005_derived_PAT005_therenva_lumenwcalcium"
            / "mesh"
            / "aorta.stl"
        ),
    )
    parser.add_argument(
        "--ui-calcium",
        default=str(
            cache
            / "data_AGORALABS_MOSAIC_CTA_CASES_20260421_processed_PAT005_derived_PAT005_therenva_calcium"
            / "mesh"
            / "aorta.stl"
        ),
    )
    parser.add_argument(
        "--pose-dir",
        default="/home/liam/Code/mosaic-ui-ux/twin_web/apps/twin/public/look/pat05_ivus_poses",
    )
    args = parser.parse_args(argv)
    if not args.pose_json.is_file():
        raise SystemExit(f"pose json not found: {args.pose_json}")
    picked = pick_poses(args.pose_json)
    write_poses_json(picked, HERE / "poses.json")
    print(f"wrote {HERE / 'poses.json'} ({len(picked)} poses)", flush=True)
    if args.poses_only:
        return 0
    render_boards(picked, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
