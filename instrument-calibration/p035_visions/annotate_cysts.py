#!/usr/bin/env python3
"""Manual cyst-spot annotation GUI for the T1-E5* milk-agar cyst phantom.

The bench operator rotated the phantom roughly to 0 / 90 / 180 / 270 deg
across takes, but mis-labeled positions in DICOM PatientID. This tool
lets the operator click each visible cyst CENTER in each frame; the
rigid-fit downstream (`fit_cyst_alignment.py`) then recovers the
per-frame phantom orientation so the speckle / cyst-contrast analysis
runs in the correct (design) frame.

Design radii (per `instrument-calibration/docs/interim_milk_phantom_sop.md`,
2% agar recipe = 8 mm rigid plastic-straw cysts):

    {(r_mm=12, theta=0°), (16, 90°), (16, 270°), (20, 180°)}

(canonical 4 mm PTFE radii are {10, 14, 14, 18} at the same azimuths;
toggle via --cyst-diameter-mm 4 if your build uses PTFE.)

Controls
--------
  left-click   add cyst center (next available design slot in radial order)
  scroll up    increase the displayed/saved cyst diameter
  scroll down  decrease the displayed/saved cyst diameter
  u            undo last cyst
  s            skip the next design slot (cyst not visible in this frame)
  c            clear this frame's cysts
  S            toggle "frame unreadable / no cysts" flag
  d            toggle bright contrast view
  n            next