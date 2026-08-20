# Twin app (HTTP sidecar + this OptiX engine)

Twin is HTTP-only. The UI lives on Mosaic `ui-ux-*` orphans; the sidecar is Mosaic
`liam/twin-ivus-http`. **This repo is the OptiX engine**, not the app.

CTA meshes, ACENet weights, and sidecar spin-up: Mosaic
[`sim/nav_sim/twin_http/SPIN_UP.md`](https://github.com/mosaicintelligence/mosaic/blob/liam/twin-ivus-http/sim/nav_sim/twin_http/SPIN_UP.md).

Poses for pat05: Twin packaged pullback
`twin_web/apps/twin/public/look/pat05_ivus_poses/`, path **`centerline_L`**
(the calc cube, ~station 50–131 mm).

## Stock look vs demo-app look

| Look | What loads | `/metadata.materials` |
|---|---|---|
| Stock | Compiled PR `#20` table in `csrc/core/material.cpp`. No overlay. | `ivus-probe-stock` |
| Demo-app | Same binary, then [`twin_demo.yaml`](../instrument-calibration/p035_visions/twin_demo.yaml) via `IvusSimConfig.apply_materials()` | `twin-demo` |

`volcano_s5i.yaml` stays probe / PSF / TGC. Its `materials:` block is decorative
and is **not** applied — Tier 1 milk / PSF keep the C++ stock table.

Overlay-only value changes do **not** need a rebuild. Rebuild is still required
if `material.cpp` **names** change (the seven Twin names were registered in `#20`).

```python
from raysim import IvusSimConfig
import raysim.cuda as rs

mats = rs.Materials()  # stock #20
cfg = IvusSimConfig.from_yaml("instrument-calibration/p035_visions/twin_demo.yaml")
cfg.apply_materials(mats)  # seven Twin rows only
```

Sidecar: keep `--raysim-yaml` pointed at `volcano_s5i.yaml`. The sidecar loads
sibling `twin_demo.yaml` (or `--materials-yaml`). Pass `--stock-materials` for
the compiled table.

Vessel-generator and Tier 1 / Tier 2 training stay on `#20`. Twin look is opt-in.

Review boards (same 16 `centerline_L` poses):
[`docs/twin-material-review/`](twin-material-review/parameters.md).
