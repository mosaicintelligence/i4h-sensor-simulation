# Segmentation labels

Canonical label IDs for paired-dataset segmentation masks. Implementation:
`vesselgen/labels.py`.

## Label table

| ID | Name | Material source |
|----|------|-----------------|
| 0 | background | Outside imaging FOV |
| 1 | lumen | `lumen` |
| 2 | intima | First wall layer |
| 3 | media | Second wall layer |
| 4 | adventitia | Third wall layer |
| 5 | peri_adventitia | Legacy extravascular (see below) |
| 6 | calcified_plaque | `calcified_plaque` |
| 7 | lipid_pool | `lipid_pool` |
| 8 | fibrous_plaque | `fibrous_plaque` |
| 9 | thrombus | `thrombus` |
| 10 | guidewire | `tungsten` |
| 11 | vessel_wall | Legacy single-slab wall |

IDs are dense and append-only — new classes extend the table without
renumbering. Saved dataset manifests record IDs by name for backward
compatibility.

## Material mapping

```python
from vesselgen.labels import material_to_label, LABEL_WALL_LIKE

label_id = material_to_label["media"]
```

## Legacy 4-class collapse

Pre-2026-06 datasets used a simpler scheme (background, lumen, wall,
extravascular). Collapse fine-grained labels for older training heads:

```python
import numpy as np
from vesselgen.labels import (
    LABEL_BACKGROUND, LABEL_LUMEN, LABEL_WALL_LIKE, LABEL_PERI_ADVENTITIA,
)

def collapse_legacy(seg):
    out = np.zeros_like(seg)
    out[seg == LABEL_LUMEN] = 1
    out[np.isin(seg, LABEL_WALL_LIKE)] = 2
    out[seg == LABEL_PERI_ADVENTITIA] = 3
    return out
```

`LABEL_WALL_LIKE` unions intima, media, adventitia, and legacy
`vessel_wall`.

## World-model change (June 2026)

The current pipeline drops the outer mesh and sets the raysim world
background to `lumen` (blood pool). Pixels beyond `interface_02` are
labelled `adventitia` (ID 4), not `peri_adventitia` (ID 5). ID 5 is
retained only for backward compatibility with older renders.

## Overlay colours

`LABEL_RGBA` in `labels.py` defines the colour palette used by
`render_paired_dataset.py` overlay PNGs. Each label maps to an RGBA tuple
for visual QC.

## Rasterization

The paired renderer projects geometric GT polygons onto the B-mode display
grid and fills enclosed regions with label IDs. Coordinate convention
matches raysim scanline geometry — see the module docstring in
`examples/render_paired_dataset.py`.

For acoustic-boundary offsets (echo vs geometric boundary), see
[Simulator integration](simulator-integration.md).
