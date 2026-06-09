# IVUS Demo Docker

This folder defines a GPU-enabled Docker container for realtime IVUS demo simulation.

On startup, the container:

1. Loads static patient segmentation inputs from patient 23.
2. Builds/loads lumen + outer geometry for the vessel wall model.
3. Loads `raysim` with Volcano P035-style defaults from `volcano_s5i.yaml`.
4. Creates the simulation world/materials once and keeps them resident while the container is running.

After startup, clients send probe pose requests and receive a simulated IVUS frame.

## What Is Hardcoded

- Patient data: `patient_23` (left iliac centerline/snakes + aorta lumen STL).
- Probe config: `instrument-calibration/p035_visions/volcano_s5i.yaml`.
- World materials:
  - background: `lumen`
  - lumen mesh: `vessel_wall`
  - outer mesh: `extravascular`

These can be overridden with environment variables if needed.

## Build

Run from repository root (`/home/jocelynbarker/i4h-sensor-simulation`):

```bash
docker build -f IVUS_demo/Dockerfile -t ivus-demo:latest .
```

## Run

NVIDIA GPU runtime is required.

```bash
docker run --rm --gpus all -p 8000:8000 ivus-demo:latest
```

Service endpoints:

- `GET /health`
- `GET /metadata`
- `GET /startup_state`
- `POST /simulate`

## Requesting a Simulation

`POST /simulate` accepts:

- `position_mm`: `[x, y, z]`
- `rotation_rad_xyz`: `[rx, ry, rz]`
- `output`: one of `png`, `png_polar`, `npy`, `json` (default: `png`)

`png` returns a circular/cartesian IVUS display image (what users usually expect).
Use `png_polar` for the unwrapped theta-by-radius image.

If pose fields are omitted, the startup default pose (first pullback pose) is used.

### Example: PNG frame

```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "position_mm": [25.0, -35.0, 90.0],
    "rotation_rad_xyz": [0.0, 1.5708, 0.0],
    "output": "png"
  }' \
  --output frame.png
```

### Example: NumPy frame

```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "position_mm": [25.0, -35.0, 90.0],
    "rotation_rad_xyz": [0.0, 1.5708, 0.0],
    "output": "npy"
  }' \
  --output frame.npy
```

### Example: JSON frame

```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "position_mm": [25.0, -35.0, 90.0],
    "rotation_rad_xyz": [0.0, 1.5708, 0.0],
    "output": "json"
  }'
```

## Optional Environment Overrides

Defaults are set for patient 23 + Volcano P035. You can override these paths/materials at runtime:

- `IVUS_LUMEN_STL`
- `IVUS_CENTERLINE_TXT`
- `IVUS_SNAKES_CSV`
- `IVUS_RAYSIM_YAML`
- `IVUS_GEOMETRY_DIR` (default `/tmp/ivus_demo/geometry`)
- `IVUS_WORLD_BACKGROUND_MATERIAL`
- `IVUS_LUMEN_MESH_MATERIAL`
- `IVUS_OUTER_MESH_MATERIAL`
- `IVUS_DEMO_HOST` (default `0.0.0.0`)
- `IVUS_DEMO_PORT` (default `8000`)
