# Twin demo-app overlay vs stock `#20`

Stock values are the compiled rows in
`i4h-sensor-simulation/ultrasound-raytracing/csrc/core/material.cpp` (PR `#20`).
Demo-app values are
[`instrument-calibration/p035_visions/twin_demo.yaml`](../../instrument-calibration/p035_visions/twin_demo.yaml).
`n` is Mattausch specularity (C++ default 1 when the 6-arg constructor is used).

**Bold** cells differ from stock.

| name | look | Z (MRayl) | α (dB/cm/MHz) | c (m/s) | μ0 | μ1 | σ | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| intima | stock `#20` | 1.92 | 1.1 | 1600 | 0.85 | 0.7 | 0.75 | 1 |
| intima | demo-app | **1.87** | **0.93** | 1600 | **0.55** | **0.6** | **0.49** | 1 |
| media | stock `#20` | 1.66 | 0.9 | 1571 | 0.25 | 0.15 | 0.05 | 1 |
| media | demo-app | **1.83** | **0.80** | **1570** | **0.12** | 0.15 | **0.10** | 1 |
| adventitia | stock `#20` | 2.05 | 1.3 | 1620 | 1.0 | 0.8 | 0.95 | 1 |
| adventitia | demo-app | **1.86** | **1.10** | **1610** | **0.95** | **0.75** | **0.85** | 1 |
| calcified_plaque | stock `#20` | 2.23 | 20.0 | 2500 | 1.0 | 1.5 | 16.0 | 1 |
| calcified_plaque | demo-app | **2.12** | **12.0** | **2000** | **1.25** | **1.25** | **6.0** | 1 |
| lipid_pool | stock `#20` | 1.70 | 0.4 | 1480 | 0.12 | 0.08 | 0.05 | 1 |
| lipid_pool | demo-app | 1.70 | 0.4 | 1480 | 0.12 | 0.08 | 0.05 | 1 |
| fibrous_plaque | stock `#20` | 2.02 | 1.4 | 1620 | 1.5 | 1.0 | 0.8 | 1 |
| fibrous_plaque | demo-app | 2.02 | 1.4 | 1620 | 1.5 | 1.0 | 0.8 | 1 |
| thrombus | stock `#20` | 1.70 | 0.4 | 1560 | 0.55 | 0.40 | 0.30 | 1 |
| thrombus | demo-app | 1.70 | 0.4 | 1560 | 0.55 | 0.40 | 0.30 | 1 |

Tungsten (catheter) stays `#20` α 60 — not in this overlay.

## How the grids were made

Same `ivus-probe` binary (elevational `#21`). Scene is pat05 trilaminar lumen +
calc. Sixteen `centerline_L` stations in the calc window (50–125 mm). See
[`poses.json`](poses.json).

- **Old / stock** — no overlay
- **New / demo-app** — `twin_demo.yaml` via `apply_materials()`

```bash
# from this repo, with MOSAIC_TWIN_HTTP pointing at the sidecar tree
python docs/twin-material-review/render_grid.py
```
