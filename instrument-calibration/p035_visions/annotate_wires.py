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

"""Manual wire-spot annotation GUI for the P_035_PointScatter capture.

The auto-alignment in extract_calibration_inputs.py is unreliable
when:

  - the catheter was rotated between captures (per-frame theta0 differs);
  - many wires are off-screen (D35 / D40 frames);
  - the gain is high (gain=64) and speckle drowns the wire echoes.

This tool lets the operator click each visible wire bright spot in
RADIAL ORDER (innermost first). Click N goes to wire N, so as long as
you click strictly inside-out and skip wires you can't see, indices
are assigned correctly. The clicked positions are saved to a JSON file
that downstream calibration scripts can consume to fit per-frame
theta0, catheter-center offset, and an effective sound-speed scale.

Controls
--------
  left-click    add a point (wire index = current click count)
  u             undo last click
  s             skip wire (e.g. inner wire that's hidden by ring-down):
                inserts a None at the next index
  c             clear all clicks for this frame
  n             next frame (saves current frame's clicks)
  b            previous frame
  S             toggle "frame is unreadable" flag (no annotations)
  d             toggle dim/bright contrast view
  q             quit (auto-saves)
  h             print help

Files
-----
  in  : --capture-dir / FILEnnnn DICOMs and the DXF
  out : --capture-dir / derived / wire_annotations.json
        (resumed automatically if it already exists)

Usage
-----
    python3 instrument-calibration/p035_visions/annotate_wires.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Reuse the parsing utilities from the sibling script so the DXF / DICOM
# code lives in one place.
sys.path.insert(0, str(Path(__file__).parent))
from extract_calibration_inputs import (  # noqa: E402
    BoxGeometry,
    FrameInfo,
    Wire,
    estimate_catheter_center_px,
    load_frame,
    parse_scattering_box_dxf,
)


# Click record: per-frame list of (wire_index, x_px, y_px). A wire_index
# of None means "skipped" (the user pressed 's' at that slot).
@dataclass
class FrameAnnotation:
    file: str
    instance_number: int
    pixel_spacing_mm: float
    rows: int
    cols: int
    catheter_center_px: tuple[float, float]
    unreadable: bool = False
    # Each entry is {"wire_index": int|None, "x_px": float, "y_px": float}.
    clicks: list[dict] = None  # type: ignore[assignment]

    def to_json(self) -> dict:
        return {
            "file": self.file,
            "instance_number": self.instance_number,
            "pixel_spacing_mm": self.pixel_spacing_mm,
            "rows": self.rows,
            "cols": self.cols,
            "catheter_center_px": list(self.catheter_center_px),
            "unreadable": self.unreadable,
            "clicks": self.clicks or [],
        }

    @classmethod
    def from_json(cls, d: dict) -> "FrameAnnotation":
        return cls(
            file=d["file"],
            instance_number=d["instance_number"],
            pixel_spacing_mm=d["pixel_spacing_mm"],
            rows=d["rows"],
            cols=d["cols"],
            catheter_center_px=tuple(d["catheter_center_px"]),  # type: ignore[arg-type]
            unreadable=d.get("unreadable", False),
            clicks=list(d.get("clicks", [])),
        )


def initial_annotation(frame: FrameInfo) -> FrameAnnotation:
    cx, cy = estimate_catheter_center_px(frame.array)
    return FrameAnnotation(
        file=frame.path.name,
        instance_number=frame.instance_number,
        pixel_spacing_mm=frame.pixel_spacing_mm,
        rows=frame.rows,
        cols=frame.cols,
        catheter_center_px=(cx, cy),
        clicks=[],
    )


def load_annotations(path: Path) -> dict[str, FrameAnnotation]:
    if not path.exists():
        return {}
    with path.open() as f:
        d = json.load(f)
    return {a["file"]: FrameAnnotation.from_json(a) for a in d["frames"]}


def save_annotations(
    path: Path,
    annotations: dict[str, FrameAnnotation],
    wires: list[Wire],
    geom: BoxGeometry,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "wire_design": [
            {"wire_index": w.index, "r_mm": w.r_mm, "theta_deg": w.theta_deg}
            for w in wires
        ],
        "box_geometry": {
            "outer_radius_mm": geom.outer_radius_mm,
            "catheter_radius_mm": geom.catheter_radius_mm,
        },
        "frames": [a.to_json() for a in annotations.values()],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


# ----------------------------------------------------------------- GUI ---


class Annotator:
    """Matplotlib-backed click-to-annotate UI."""

    HELP = (
        "Controls:\n"
        "  left-click  add wire (index = next radial slot)\n"
        "  u          undo last click\n"
        "  s          skip the next wire slot (e.g. inner wire hidden)\n"
        "  c          clear this frame's clicks\n"
        "  S          toggle 'frame unreadable'\n"
        "  d          toggle bright contrast\n"
        "  n          next frame (auto-save)\n"
        "  b          previous frame (auto-save)\n"
        "  q          quit (auto-save)\n"
        "  h          help\n"
    )

    def __init__(
        self,
        frames: list[FrameInfo],
        wires: list[Wire],
        geom: BoxGeometry,
        annotations: dict[str, FrameAnnotation],
        save_path: Path,
    ) -> None:
        self.frames = frames
        self.wires = wires
        self.wires_by_index = {w.index: w for w in wires}
        self.geom = geom
        self.annotations = annotations
        self.save_path = save_path
        self.idx = self._first_unannotated_index()
        self.bright = False
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        print(self.HELP)

    def _first_unannotated_index(self) -> int:
        for i, fr in enumerate(self.frames):
            ann = self.annotations.get(fr.path.name)
            if ann is None or (not ann.unreadable and not ann.clicks):
                return i
        return 0

    def _current(self) -> tuple[FrameInfo, FrameAnnotation]:
        fr = self.frames[self.idx]
        ann = self.annotations.get(fr.path.name)
        if ann is None:
            ann = initial_annotation(fr)
            self.annotations[fr.path.name] = ann
        return fr, ann

    def _next_wire_index(self, ann: FrameAnnotation) -> int:
        """Index of the next unfilled wire slot (1-based)."""
        used = [c["wire_index"] for c in ann.clicks if c["wire_index"] is not None]
        skipped = [
            i + 1
            for i, c in enumerate(ann.clicks)
            if c["wire_index"] is None
        ]
        # Slot index = (number of clicks so far) + 1 if filled inside-out.
        # We assign indices contiguously: 1, 2, 3, ... unless skipped.
        filled = len(ann.clicks)
        return filled + 1

    # --------------- drawing ---

    def _redraw(self) -> None:
        fr, ann = self._current()
        self.ax.clear()

        arr = fr.array.astype(np.float32)
        if self.bright:
            # mild log stretch for low-gain frames
            arr = np.log1p(arr) / math.log1p(255) * 255
        self.ax.imshow(arr, cmap="gray", vmin=0, vmax=255, origin="upper")

        # Catheter outline.
        cx, cy = ann.catheter_center_px
        self._draw_circle(cx, cy, self.geom.catheter_radius_mm / fr.pixel_spacing_mm,
                          color="cyan", lw=1.0)

        # Existing clicks.
        for i, c in enumerate(ann.clicks):
            x, y = c["x_px"], c["y_px"]
            wi = c["wire_index"]
            if wi is None:
                # Skipped slot; show as a small grey marker at the catheter edge
                continue
            color = "yellow" if i == len(ann.clicks) - 1 else "lime"
            self.ax.plot(x, y, "o", mfc="none", mec=color, mew=1.5, ms=14)
            w = self.wires_by_index.get(wi)
            label = f"w{wi}"
            if w is not None:
                label += f" r={w.r_mm:.0f}"
            self.ax.text(x + 6, y - 6, label, color=color, fontsize=9)

        # Status bar (top-left).
        n_clicks = len(ann.clicks)
        next_slot = self._next_wire_index(ann)
        unread = " UNREADABLE" if ann.unreadable else ""
        title = (
            f"[{self.idx + 1}/{len(self.frames)}]  {fr.path.name}  "
            f"px={fr.pixel_spacing_mm:.3f} mm  fov={fr.cols * fr.pixel_spacing_mm:.0f} mm"
            f"  clicks={n_clicks}  next=wire {next_slot}{unread}"
        )
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.fig.canvas.draw_idle()

    def _draw_circle(self, cx, cy, r, color, lw):
        circ = plt.Circle((cx, cy), r, fill=False, edgecolor=color, lw=lw)
        self.ax.add_patch(circ)

    # --------------- callbacks ---

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        fr, ann = self._current()
        wi = self._next_wire_index(ann)
        if wi > len(self.wires):
            print(f"  Already have {len(self.wires)} clicks; press 'u' to undo.")
            return
        ann.clicks.append({"wire_index": wi, "x_px": float(event.xdata), "y_px": float(event.ydata)})
        ann.unreadable = False
        self._redraw()

    def _on_key(self, event) -> None:
        fr, ann = self._current()
        key = event.key
        if key == "u":
            if ann.clicks:
                removed = ann.clicks.pop()
                print(f"  undo: removed wire {removed['wire_index']}")
                self._redraw()
        elif key == "s":
            wi = self._next_wire_index(ann)
            if wi > len(self.wires):
                print("  no more wire slots to skip")
                return
            ann.clicks.append({"wire_index": None, "x_px": None, "y_px": None})
            print(f"  skipped wire slot {wi}")
            self._redraw()
        elif key == "c":
            ann.clicks = []
            print("  cleared all clicks for this frame")
            self._redraw()
        elif key == "S":
            ann.unreadable = not ann.unreadable
            if ann.unreadable:
                ann.clicks = []
            print(f"  unreadable = {ann.unreadable}")
            self._redraw()
        elif key == "d":
            self.bright = not self.bright
            self._redraw()
        elif key == "n":
            self._save()
            if self.idx + 1 < len(self.frames):
                self.idx += 1
                self._redraw()
            else:
                print("  already at last frame")
        elif key == "b":
            self._save()
            if self.idx > 0:
                self.idx -= 1
                self._redraw()
            else:
                print("  already at first frame")
        elif key == "q":
            self._save()
            plt.close(self.fig)
        elif key == "h":
            print(self.HELP)

    def _save(self) -> None:
        save_annotations(self.save_path, self.annotations, self.wires, self.geom)

    def run(self) -> None:
        self._redraw()
        plt.show()
        self._save()
        print(f"\nAnnotations saved to {self.save_path}")


# ---------------------------------------------------------------- main ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_capture = repo_root / "P_035_PointScatter"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", type=Path, default=default_capture)
    p.add_argument("--dxf-name", default="IVUS Scattering Box - Sketch 1.dxf")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: <capture-dir>/derived/wire_annotations.json).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture_dir: Path = args.capture_dir.resolve()
    if not capture_dir.is_dir():
        print(f"capture dir not found: {capture_dir}", file=sys.stderr)
        return 2
    dxf_path = capture_dir / args.dxf_name
    if not dxf_path.is_file():
        print(f"DXF not found: {dxf_path}", file=sys.stderr)
        return 2

    out_path = args.out or (capture_dir / "derived" / "wire_annotations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wires, geom = parse_scattering_box_dxf(dxf_path)
    dicom_paths = sorted(p for p in capture_dir.iterdir() if p.name.startswith("FILE"))
    if not dicom_paths:
        print(f"no FILEnnnn DICOMs in {capture_dir}", file=sys.stderr)
        return 2

    frames = [load_frame(p) for p in dicom_paths]
    existing = load_annotations(out_path)
    print(f"Loaded {len(frames)} frames; resuming with {len(existing)} prior annotations.")
    Annotator(frames, wires, geom, existing, out_path).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
