# mosaic_sim — catheter physics → IVUS pullback poses

A GPU catheter-insertion simulation (XPBD Cosserat elastic rod through a vessel
mesh) that records the 6-DOF trajectory of a virtual IVUS sensor and exports it
**in the CTA mesh frame**, ready to drive the `simulated_pullback_from_CT` IVUS
pipeline with a physics-based probe path instead of a geometric centerline.

```
mosaic_sim.py   ──►  recordings/ivus_*.{csv,npz,_transforms.npy}   ──►  to_pullback_poses.py   ──►  *_poses.csv/json   ──►  run_pullback_simulation(...)
  catheter physics        sensor 6-DOF in CTA-mm frame                  normal/tangent/binormal frames        raysim IVUS frames
  (needs newton+warp)                                                   (numpy + scipy only)
```

The two stages are decoupled by the recording file, so only the recorder needs
the heavy `newton`/`warp` dependency — the adapter and the IVUS pipeline stay
pure numpy/scipy.

## 1. Record a catheter run (needs `newton` + `warp`)

```bash
uv run mosaic_sim/mosaic_sim.py --viewer gl            # default: stls/pig_cta.stl
```

- An open vessel end is auto-aligned to the rod tip (~1 cm inside); press `E` to
  cycle openings to the femoral.
- Drive: `I`/`K` insert/retract, `J`/`L` rotate, `+`/`-` bend tip, `W` wireframe.
- Press **`R`** to start recording, drive the insertion, press **`R`** to stop.
  It writes `recordings/ivus_<mesh>_NNN.{csv,npz,_transforms.npy}` (see
  `recordings/README.md` for the formats — all in the original CTA STL frame).

Replay a recording as an animation:

```bash
uv run mosaic_sim/replay.py mosaic_sim/recordings/ivus_pig_cta_000.npz --viewer gl
```

## 2. Convert to pullback poses (numpy + scipy only)

```bash
python mosaic_sim/to_pullback_poses.py mosaic_sim/recordings/ivus_pig_cta_000.npz
# -> ivus_pig_cta_000_poses.csv / .json   (add --reverse for distal->proximal pullback order)
```

This emits the exact pose schema `simulated_pullback_from_CT/src/trajectory.py`
produces — `position_mm` plus a parallel-transport orientation frame in the
pipeline convention (probe axes `x→normal, y→tangent/pullback-axis, z→binormal`).

## 3. Feed the IVUS pipeline

The payload from step 2 is a drop-in for the pullback simulator. Use the same
STL as the lumen mesh and an outer wall (e.g. `make_shell.py`, or the pipeline's
`geometry.generate_wall_geometry`):

```python
from to_pullback_poses import poses_from_recording
from sim_runner import run_pullback_simulation   # simulated_pullback_from_CT/src

poses = poses_from_recording("recordings/ivus_pig_cta_000.npz", reverse=True)
run_pullback_simulation(
    raysim_yaml_path=...,
    lumen_mesh_obj="stls/pig_cta.stl",   # the CTA the run was recorded against
    outer_mesh_obj=...,                  # generated wall
    poses_payload=poses,
    out_dir=...,
    ...,
)
```

## Files

| File | Purpose |
|------|---------|
| `mosaic_sim.py` | the catheter simulation + sensor recording |
| `replay.py` | animate a saved recording |
| `to_pullback_poses.py` | recording → pullback-pose payload (the IVUS bridge) |
| `xcath_solver.py` | XPBD rod solver + mesh containment (vendored; imports `newton`) |
| `make_shell.py` | turn a solid-lumen STL into an open thick-walled shell |
| `stls/` | vessel meshes (`pig_cta`, `leg_to_leg__simple`) |
