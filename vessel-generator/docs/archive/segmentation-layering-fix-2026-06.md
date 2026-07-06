# Segmentation labelling + world-model fixes

> **Status (June 2026):** the rasterizer fix, the world-model
> simplification, and the always-trilaminar bifurcation rework are
> all implemented and passing the vesselgen test suite. The 10k
> dataset still has to be re-rendered to pick up the new geometry
> (segmentation-only regen is no longer enough because the wall
> meshes themselves changed).

This doc captures the planned changes to (a) fix the segmentation
off-by-one bug that wipes out the intima label, and (b) align the
simulator's world model with the anatomy we want IVUS frames to teach
(lumen-pool blood + intima/media/adventitia + acoustically-invisible
adventitia back-boundary, even at bifurcations).

The changes here are interleaved -- (a) is local to
`render_paired_dataset.py::_cartesian_label_grid`, (b) reshapes the
mesh chain in `vesselgen/vessel.py` and the world setup in raysim
config -- so we'll land them together to avoid temporarily-broken
seg/image alignment.


## 1. Bug: `_cartesian_label_grid` is off by one

`_interior_material_chain(wall_cfg)` for a trilam wall returns
`["intima", "media", "adventitia", world_background]`, paired with
meshes `[lumen, interface_01, interface_02, outer]`.

Convention (matches raysim): `material[mesh_i]` is the material a ray
**enters when crossing `mesh_i` outward**, i.e. the material on the
**outside** of the mesh in 3D-volume terms. Equivalently it is the
material of the shell *just outside* surface `i`.

`_cartesian_label_grid` iterates the surface list from outer to inner
and paints "everything inside this surface's polygon" with
`material[this surface]`. That's wrong, because the polygon's interior
is everything **just inside** plus deeper -- we should paint with the
material **just inside** the surface (which is `material[i-1]`).

Effect, traced for a trilam vessel:

| Step | Action | Region affected | Should be | Currently painted as |
|------|--------|-----------------|-----------|----------------------|
| init | extravascular for in-FOV | everything | extravascular | extravascular |
| 1 | skip outer | adventitia ring | ADVENTITIA | extravascular ❌ |
| 2 | paint inside interface_02 with `material[interface_02]` = adventitia | media + intima + lumen rings | MEDIA / INTIMA / LUMEN | adventitia ❌ |
| 3 | paint inside interface_01 with `material[interface_01]` = media | intima + lumen | INTIMA / LUMEN | media ❌ |
| 4 | special-case "lumen" → LABEL_LUMEN | lumen pool | LUMEN | lumen ✓ |

Net result: every wall ring is labelled with **the next layer outward**,
the actual adventitia ring is silently absorbed into the extravascular
fill, and **intima never appears at all**.

Confirmed empirically across the 10k dataset:

```
label 0 (background):    100.0% of frames
label 1 (lumen):         100.0% of frames
label 2 (intima):          0.0% of frames   <- never assigned
label 3 (media):          31.8% of frames   <- actually intima ring
label 4 (adventitia):     36.6% of frames   <- actually media ring
label 5 (extravascular): 100.0% of frames   <- swallows the actual adventitia ring
label 11 (vessel_wall):    0.0% of frames   <- never assigned in legacy single-slab walls
```

The legacy single-slab path (used at bifurcations) hit the same bug:
the lone "outer" surface was the only non-skipped surface, so its
material was never painted, and the entire wall ring fell through to
the initial extravascular fill.

### Fix

In `_cartesian_label_grid`, iterate outermost→innermost and paint
`surfaces[i]`'s polygon interior with the material of `surfaces[i-1]`
(the material just inside `surfaces[i]`). The innermost surface is the
lumen mesh, which we paint with `LABEL_LUMEN` as before.

Pseudo-fix:

```python
surfaces = list(gt.surface_polygons)  # innermost-to-outermost
n = len(surfaces)
for i in range(n - 1, 0, -1):                            # outer -> inner
    name, _, polys = surfaces[i]
    if not polys:
        continue
    inner_material = surfaces[i - 1][1]                  # shell just-inside surface i
    label = material_to_label.get(inner_material, LABEL_VESSEL_WALL)
    mask = _polygon_mask(polys, plane_pts, offset_mm=acoustic_offset_mm)
    labels[mask] = label

# innermost: paint the lumen polygon's interior as blood pool
lumen_polys = surfaces[0][2] if surfaces else gt.lumen_contour_polygons
mask = _polygon_mask(lumen_polys, plane_pts, offset_mm=acoustic_offset_mm)
labels[mask] = LABEL_LUMEN
```

This restores intima (label 2) and shifts media/adventitia to the
geometrically correct rings.

The same fix applies to `_sample_cart_mask_on_polar` consumers and to
the `--regenerate-segmentations-only` re-labelling path; the label
function is the single source of truth.


## 2. World-model change: lumen-default + drop the outer mesh

### Current state

```
chain = [intima, media, adventitia, world_background]
        ↑ on lumen mesh         ↑ on outer mesh

world_background = "extravascular"
4 closed meshes per branch: lumen, interface_01, interface_02, outer
```

In raysim every ray starts with `current_material_id =
background_material_id` and only updates on the first hit. The probe
sits inside the lumen mesh, so until the first ray-mesh hit the
**lumen pool is rendered with the world-background material** --
currently `"extravascular"`. That is anatomically wrong.

After the ray crosses the outer mesh on its way out, raysim adopts
`chain[-1] = "extravascular"`, which has very different impedance from
adventitia, producing a bright back-of-adventitia rim that does not
exist in real IVUS images.

### Target

3 closed meshes per branch (drop the outer mesh entirely):

```
meshes = [lumen, interface_01, interface_02]
chain  = [intima, media, adventitia]

world_background = "lumen"
```

| Region | Material (raysim) | Segmentation label |
|--------|-------------------|--------------------|
| Inside lumen mesh (blood pool) | `lumen` (= world background) | `LABEL_LUMEN` |
| Inside interface_01, outside lumen | `intima` | `LABEL_INTIMA` |
| Inside interface_02, outside interface_01 | `media` | `LABEL_MEDIA` |
| Outside interface_02 (everywhere else inside FOV) | `adventitia` | `LABEL_ADVENTITIA` |
| Outside FOV | -- | `LABEL_BACKGROUND` |

`LABEL_EXTRAVASCULAR` is renamed to `LABEL_PERI_ADVENTITIA` in the
schema. The id (5) and palette stay the same so legacy datasets keep
their colours; the new pipeline never emits this label (everything
inside the FOV beyond interface_02 collapses to `LABEL_ADVENTITIA`).

So:

- `World(background_material="lumen")`: the lumen pool now renders as
  blood, matching real IVUS.

- `chain` becomes `[intima, media, adventitia]` (trilam) or
  `["vessel_wall"]` (legacy single-slab). No artificial outer mesh.

- The adventitia is geometrically unbounded on its outside. Acoustically
  raysim just keeps propagating in adventitia after crossing
  interface_02 -- exactly what the user asked for ("the back boundary
  of the adventitia is not acoustically visible"). Anything past
  interface_02 in the segmentation is simply labelled `ADVENTITIA`.

- `LABEL_EXTRAVASCULAR` is renamed in place to `LABEL_PERI_ADVENTITIA`
  (same id 5, same palette colour). The new pipeline never emits this
  label; everything in-FOV beyond interface_02 is `LABEL_ADVENTITIA`.

### Concrete code touches

1. `vesselgen/wall.py::LayeredWallField`: drop the outermost interface
   from `interface_radii` (or stop generating it) so the layer count
   matches `len(layers)` exactly, not `len(layers) + 1`.

2. `vesselgen/sweep.py::sweep_layered_branch`: returns `n_layers` meshes
   (lumen + each interior interface), no outer cap.

3. `vesselgen/vessel.py`:
   - `_interior_material_chain`: return `[ly.material_name for ly in
     wall_cfg.layers]` -- no `world_background` appended.
   - `_build_layered_surface_list`: handle `n_layers` meshes; index 0 is
     `"lumen"`, all others are `"interface_kk"`. Drop the special
     `"outer"` name.
   - `Vessel.from_config`: stop tracking `outer_mesh` separately; the
     adventitia mesh (= outermost interior interface) becomes the
     "outer-most" closed surface for things like `outer_contour_polygons`
     in the legacy code paths.

4. `examples/render_paired_dataset.py` and any other simulator-config
   site that constructs a `raysim.World`: change the default world
   material from `extravascular` to `lumen`.

5. `_cartesian_label_grid`: initial in-FOV fill becomes
   `LABEL_ADVENTITIA` (replacing the old `LABEL_EXTRAVASCULAR` fill).
   With the off-by-one fix from #1 already in place, the painting walks
   inward and overwrites the inner shells with media → intima → lumen
   correctly. The "outer" name is gone, so the `if name == "outer":
   continue` branch is removed.

6. Material table sanity: confirm the simulator's `lumen` material
   has the right blood-like acoustic params (low impedance contrast vs
   intima, low scatter, low specularity). If not, tune in `material.cpp`.

### Side effects

- All 10k existing renders were produced with `world_background =
  extravascular`, so their lumen-pool brightness is wrong by today's
  standards. Once we change the world default we will need to re-render
  the whole dataset (relabel-only is **not** sufficient).

- Saved manifests and OBJ files: vessels written by the old
  `save_vessel` path emitted `outer.obj`. The new pipeline emits one
  fewer file. Loaders need a tiny migration guard for legacy manifests.


## 3. Always-trilaminar walls, including bifurcations

Today, when `cfg.side_branches` is non-empty, `Vessel.from_config`
falls back to `_build_legacy_two_surface_list`: a single-slab
`vessel_wall` mesh between lumen and outer. About half of our
production frames hit this path (we set
`side_branch_probability = 0.5`), so half the dataset never shows
trilaminar structure even though the wall config requested it.

### Reason for the fallback

`bifurcation.attach_side_branch` runs `trimesh.boolean.union` on the
parent and daughter meshes -- once for lumen, once for outer. With
trilaminar walls each layer is its own closed mesh, so we'd need
**three** boolean unions (lumen, interface_01, interface_02) each of
which can fail or self-intersect when the daughter and parent layer
thicknesses don't match exactly at the ostium. (After the
"drop the outer mesh" change in #2 we no longer have a fourth `outer`
mesh to union.)

### Options

| Option | Visual outcome | Per-vessel build cost | Failure rate |
|--------|----------------|----------------------|--------------|
| **A. Full layered union** at ostium | Trilam everywhere, including ostium frames. Most realistic. | ~1.5× current bifurcation cost (3 unions instead of 2, extra ~0.5-2 s per vessel; total ~+1-3 hours over 10k). | High on tight ostia: thin interface_01/02 meshes can self-intersect across the junction → manifold engine raises → sampler retries. |
| **B. Layered + smooth-near-ostium** | Same trilam appearance, but each layer thickness tapers gently to the parent's local thickness near the junction so the inner interfaces stay well-separated. Marginal realism loss at the immediate ostium (~0.3-0.5 mm). | ~1.3× current. | Low. Safe for production. |
| **C. Status quo single-slab fallback** | ~50% of frames have a uniform wall slab, no trilam at side-branches. | No change. | None. |

Recommended starting point: **Option B** -- get trilaminar coverage on
~100% of frames at modest cost, with a flag to escalate to A later if
we see anatomical unrealism at the ostium that hurts model performance.

### Concrete code touches

1. `vesselgen/bifurcation.py`: extend `attach_side_branch` to take
   layered-wall fields for both parent and daughter, build per-layer
   meshes (using `sweep_layered_branch` on each), and run a `_safe_union`
   per layer (3 unions: lumen + 2 interior interfaces). Daughter layer
   fractions must match parent's at the ostium (option B: lerp daughter
   layer fractions to parent's over the first ~ostium-radius mm of
   arclength).

2. `vesselgen/vessel.py::Vessel.from_config`: drop the
   `if cfg.side_branches: surfaces = _build_legacy_two_surface_list(...)`
   guard so the layered surface list is always used. Keep the legacy
   single-slab path available behind a config flag for ablation runs.

3. `examples/render_paired_dataset.py::randomize_layered_wall_config`:
   no change needed if it already produces a `LayeredWallConfig` for
   every vessel; remove any "downgrade-on-bifurcation" comments.


## 4. Implementation order

To avoid a stretch of broken alignment between image and
segmentation:

1. Land the rasterizer fix (#1) on its own. The image renderer
   doesn't change, but every saved seg flips to the right schema.
   Regenerate the existing 10k segs with
   `--regenerate-segmentations-only` to keep them usable.
2. Land the world-model change (#2). This invalidates the existing
   image renders, so re-render the dataset end-to-end.
3. Land the always-trilaminar bifurcation work (#3). This widens the
   trilaminar coverage to ~100% of frames; rerender once more.

Each step is independently testable -- the rasterizer fix can be
verified frame-by-frame against the polar arrays without touching
the simulator.
