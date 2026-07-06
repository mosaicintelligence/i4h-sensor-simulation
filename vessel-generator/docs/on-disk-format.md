# On-disk format

Each generated vessel is a self-contained folder the simulator loads via
`vessel.json`. Authoritative implementation: `vesselgen/io.py`.

## Trilaminar layout (recommended for IVUS)

```
out/<vessel_name>/
  lumen.obj                     # closed lumen surface (material: intima)
  surfaces/
    interface_01.obj            # intima/media interface (material: media)
    interface_02.obj            # media/adventitia interface (material: adventitia)
  lesions/                      # optional, one OBJ per inclusion
    lesion_00_hard.obj
  guidewire.obj                 # optional tungsten wire
  preview.png                   # QC cross-section gallery
  vessel.json                   # manifest + generation parameters
  branches/
    branch_00.json              # parent centerline + cross-section fields
    branch_01.json              # side branch (if present)
```

The three emitted surfaces let raysim reproduce the bright–dark–bright IVUS
wall appearance. Beyond `interface_02`, rays stay in the `adventitia`
material until the FOV — the adventitia back boundary is acoustically
invisible in clinical IVUS.

## Legacy single-slab layout

When `--layers 1`, the folder contains:

```
  lumen.obj                     # material: vessel_wall
  outer.obj                     # material: extravascular / peri_adventitia
```

Legacy two-mesh callers (e.g. CT-pullback calibration) load unchanged.

## Conventions

- **Vessel axis along Y**, cross-sections in the **xz** plane.
- **Units: mm** everywhere.
- **Inward-facing normals** on OBJ export (simulator expects front faces
  visible from inside the lumen).

## `vessel.json` manifest

The manifest lists every mesh the simulator should load with its material
assignment. Key fields:

| Field | Description |
|-------|-------------|
| `name` | Vessel identifier |
| `seed` | Generation seed |
| `wall.kind` | `"single"` or `"layered"` |
| `surfaces[]` | `{obj, material}` pairs |

| `lesions[]` | `{name, obj, material, kind}` |

| `guidewire` | `{obj, material, diameter_mm, ...}` or null |

| `branches[]` | Branch topology summary |
| `n_layers` | 1, 2, or 3 |

Example surface chain for trilaminar:

    {"obj": "lumen.obj", "material": "intima"},
    {"obj": "surfaces/interface_01.obj", "material": "media"},
    {"obj": "surfaces/interface_02.obj", "material": "adventitia"}
    {"obj_filename": "lumen.obj", "material": "intima"},
    {"obj_filename": "surfaces/interface_01.obj", "material": "media"},
    {"obj_filename": "surfaces/interface_02.obj", "material": "adventitia"}
  ]
}
```

## Branch JSON

Each `branches/branch_NN.json` stores:

- Centerline polyline and local frames
- Per-station lumen radii and wall thickness parameters
- Parent attachment arclength (for side branches)

These files enable pose replay and debugging but are not loaded by raysim
directly.

## Loading in Python

```python
from vesselgen import Vessel

v = Vessel.load("out/vessel_0001")
pose = v.sample_pose(rng)
gt = v.ground_truth_at(pose)
```

## Manifest-aware simulator loading

The paired renderer and `simulated_pullback_from_CT` load all manifest
meshes in nested material order. See
[Simulator integration](simulator-integration.md).
