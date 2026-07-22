# Project brief: CTA lumen → IVUS-ready vessel model

**Status:** Proposed (ready for execution handoff)  
**Owner (execution):** TBD  
**Primary consumers:** `IVUS_demo`, `simulated_pullback_from_CT`, downstream physics-based probe-localization work  
**Related packages:** `vessel-generator`, `simulated_pullback_from_CT`, `IVUS_demo`, `i4h-sensor-simulation/ultrasound-raytracing`

---

## 1. Problem

We can produce **lumen-only segmentations of full vascular systems from CTA** (the same class of asset used by `IVUS_demo` for patient-23). Today those assets are turned into simulator geometry with a **single-slab wall** — one outer surface offset from the lumen by a smoothed radius-proportional thickness (`simulated_pullback_from_CT/src/geometry.py::generate_wall_geometry`). That is enough for a coarse demo, but it is not clinically plausible IVUS anatomy:

- No **intima / media / adventitia** layering → missing the bright–dark–bright wall signature IVUS is trained to see.
- No **plaque / calcification** inclusions.
- Only a **single primary path** with a hand-supplied centerline — no branches, no topology.
- Output is a legacy two-mesh layout, not the layered `vessel.json` manifest the modern simulator path expects.

Meanwhile, `vessel-generator` already implements layered walls, eccentric wall thickness, and in-wall lesions — but only for **procedurally generated** straight vessels with a synthetic lumen and a known parametric centerline.

**Gap:** we need to lift a real, branched CTA lumen segmentation into a trilaminar IVUS phantom (with optional disease) that (a) reuses the CTA lumen as-is as the intima, (b) preserves full branch topology, and (c) supports **arbitrary probe poses anywhere inside the lumen** — not just along a centerline — because downstream physics-based simulation will use it for probe localization.

---

## 2. Goal

Build a pipeline that:

1. **Ingests** a lumen-only vascular segmentation mesh from CTA. The lumen mesh is the **only required input**. No centerline, no radii CSV, no branch labels are provided.
2. **Reconstructs the full vascular topology** (branch tree, junctions, per-branch centerlines and local radii) directly from the mesh.
3. **Uses the input lumen as the intima layer verbatim** — the tool **must not** synthesize a new intima surface. It only adds the media and adventitia around it.
4. **Adds media and adventitia** as concentric layers whose **total wall thickness varies along the vessel and correlates with local lumen diameter**, matching real anatomy.
5. **Optionally places** calcifications and plaques (reusing vessel-generator lesion kinds), toggleable on/off per run.
6. **Exports** a vessel folder that loads into raysim via the same manifest contract used by procedural vessels, and that supports **pose sampling anywhere inside the lumen** (not restricted to a centerline).

---

## 3. Current state (what to reuse)

| Component | What it provides | Limitation for this project |
|-----------|------------------|-----------------------------|
| `IVUS_demo/` | Live demo service loading a patient lumen STL and rendering with P035 config | Single-slab wall; assumes user-supplied centerline + snakes CSV; single path |
| `simulated_pullback_from_CT/src/geometry.py` | Lumen STL → outer mesh via normal offset with radius-proportional thickness | One homogeneous wall; requires external centerline + radii; no branches |
| `vessel-generator/vesselgen/wall.py` | Trilaminar `LayeredWallField` with Fourier-perturbed per-angle thickness | Assumes a parametric (station, angle) grid from a synthetic sweep |
| `vessel-generator/vesselgen/inclusions.py` | Calcified / lipid / fibrous / thrombus solids between lumen and outer wall | Same (station, angle) grid assumption; interacts poorly with boolean unions |
| `vessel-generator/vesselgen/bifurcation.py` | Side-branch attachment via per-layer boolean union | One side branch max; not a full topology reconstructor |
| `vessel-generator/docs/on-disk-format.md` | Trilaminar `vessel.json` schema (`lumen.obj`, `surfaces/interface_01.obj`, `surfaces/interface_02.obj`, optional `lesions/…`, `branches/…`) | Target output contract |
| `vessel-generator/docs/segmentation-labels.md` | Material / label IDs (`intima`, `media`, `adventitia`, `calcified_plaque`, …) | Downstream GT contract |
| `simulated_pullback_from_CT` `vessel_manifest_path` bridge | Loads a vessel-generator manifest into a CT pullback sim run | Needs a manifest producer for real anatomy |
| `IVUS_demo/README.md` env overrides | Shows the current demo IO contract | Will need to be updated once the demo consumes layered manifests |

---

## 4. Input contract (this project)

The only required input is a **lumen surface mesh** from CTA segmentation:

- Format: STL / OBJ / PLY (closed, mm units, patient-space coordinates).
- Content: the full segmented vascular system for the region of interest (may include arbitrary branching).
- May be a **single connected surface** covering the whole tree, or a set of per-region meshes that the loader concatenates.

**Explicitly NOT provided:**

- Centerlines (the tool must derive them).
- Per-station radii / snakes tables (the tool must estimate them).
- Branch labels or a topology graph (the tool must build one).

The tool may **accept optional hints** (e.g. a user-marked ostium point) but must not require them.

---

## 5. Objectives

### O1 — Topology and centerline reconstruction from the lumen mesh

Recover a **full vascular system map** from the lumen alone.

Requirements:

- Voxelize or otherwise sample the lumen interior and compute a **skeleton / medial axis** for the entire tree.
- Segment the skeleton into **branches** joined at **junction nodes**; assign stable branch IDs.
- For each branch produce a smooth centerline polyline with local tangent frames.
- For each centerline sample, estimate a **local lumen radius** (e.g. inscribed-sphere radius / EDT of the lumen interior at the medial point).
- Handle acyclic bifurcating topology at minimum; document behavior if loops are encountered.

Acceptance: for a representative CTA patient (e.g. patient-23 aorta + iliacs), the tool emits a labeled branch graph plus centerlines and per-station radii that visually agree with the mesh (QC overlay required).

> A single primary centerline is an unacceptable deliverable. The reconstructed graph must cover every branch present in the input segmentation.

### O2 — Trilaminar wall with the input lumen kept as the intima

Emit exactly three concentric surfaces:

| Surface | Material | Provenance |
|---------|----------|------------|
| `lumen.obj` | `intima` | **Input CTA lumen mesh, unchanged (topology preserved; only cleaned as needed for watertightness).** The tool **must not** synthesize or replace this surface. |
| `surfaces/interface_01.obj` | `media` | Generated: offset from the intima outward by the intima→media thickness fraction of the local total wall thickness. |
| `surfaces/interface_02.obj` | `adventitia` | Generated: offset from the intima outward by the full local total wall thickness. |

Design constraints:

- The generated interfaces must share vertex correspondence with (or be a direct offset of) the intima where feasible, so the intima–media junction is exactly the input mesh.
- Interfaces must not cross each other or the lumen, at any (position, direction).
- Meshes must be watertight enough for raysim and use the export normal convention already documented in `vessel-generator/docs/design.md` (inward-facing on OBJ export).
- Support the full branched topology — layer offsets must remain sane at junctions (no self-intersection near ostia).

### O3 — Anatomy-driven variable wall thickness

Total wall thickness `T(x)` must vary over the surface. It should follow **two** trends simultaneously — a **diameter-driven baseline** and a **biological-style variation on top of it**, so that even at fixed local diameter `T` is not constant.

Decompose the model as:

```
T(x) = T_base(D_local(x)) · (1 + δ_anatomy(x))
       └── diameter trend ──┘   └── natural variation ──┘
```

Requirements:

1. **Diameter-driven baseline `T_base`.**
   - `T_base` is a monotone (near-linear) function of local lumen diameter with configurable slope and min/max clamps (e.g. `T_base ≈ max(t_min, min(t_max, α · D_local))`).
   - Establishes the coarse trend: aorta > iliac > femoral in absolute wall thickness.

2. **Anatomical variation `δ_anatomy` — this must be present, not optional.**
   The thickness field must include biologically plausible non-uniformity even where the diameter is nearly constant, matching what real vessel walls look like on IVUS. Combine at least these components:
   - **Circumferential eccentricity** — one wall is thicker than the opposite wall at a given cross-section; the "thick side" rotates slowly along arclength. (Same math family as vessel-generator's `WallConfig` Fourier perturbation.)
   - **Longitudinal variation** — smooth thickening / thinning along the vessel length, on top of the diameter trend, so `T` on a straight constant-diameter segment is still not flat.
   - **Low-frequency "patchy" variation** — larger-scale bumps in `T` (e.g. a Gaussian-random-field or a few blob centers) so different segments of the same branch have subtly different wall thicknesses, as they do biologically. These are **wall-thickness variations, not lesions**.
   - Amplitudes are bounded (e.g. `|δ_anatomy| ≤ ~40–60%` of `T_base`) so `T` never crosses zero or the max clamp.
   - The variation must be **deterministic given a seed** (reproducible per patient) but decoupled from the disease system in O4 — turning disease off must not eliminate anatomical thickness variation.

3. **Junctions.** The combined field must vary smoothly across bifurcations (no sharp step), while still allowing each branch to sit near its own diameter-driven baseline.

4. **Layering.** The intima→media→adventitia split (per-layer fractions of `T`) is configurable and defaults to vessel-generator's trilaminar defaults. Layer boundaries inherit the same variation as `T`.

5. **Configurability.** Baseline slope/clamps, each perturbation component's amplitude and spatial scale, layer fractions, and the seed are all exposed through config.

Acceptance: for a test patient, published QC artifacts must show
  (a) a scatter of `T` vs `D_local` with a clear positive trend **and** meaningful spread at fixed `D_local` (i.e. variation is not just noise-floor),
  (b) a colormap of `T` on the outer surface showing eccentric and along-length variation on a single branch,
  (c) smooth field across bifurcations,
  (d) different branches sitting near their respective diameter-scaled baselines.

### O4 — Optional calcifications and plaques

Same lesion vocabulary as vessel-generator (`hard` / `soft_lipid` / `fibrous` / `thrombus`).

Requirements:

- Global on/off toggle plus per-kind knobs (count, arc extent, axial extent, diseased sector, seed).
- Lesions are placed as **closed solids embedded between the intima and adventitia** using the per-vertex thickness field.
- Placement must work with the branched topology: lesions can be requested on specific branches or drawn randomly from the tree, but must never cross into an adjacent branch or through the adventitia.
- When disabled, the manifest contains layer surfaces only.

Acceptance: A/B renders on the same anatomy with and without disease flag differ only in lesion echoes; no wall regeneration.

### O5 — Pose sampling anywhere inside the lumen

Downstream physics-based simulation will use this model for **probe localization**, so poses must be samplable at arbitrary points inside the lumen — not just on the centerline.

Requirements:

- A sampling API returning `PoseSample`-compatible outputs (see `vessel-generator/vesselgen/sampling.py`) that:
  - Draws a position uniformly (or by configurable density) from the **interior of the lumen volume**, across all branches.
  - Chooses a probe axis that is a small random tilt off the local vessel tangent at the nearest centerline point (branch-aware).
  - Respects a configurable wall clearance / edge margin.
  - Optionally biases toward wall-contact poses for artifact realism.
- The API must expose which branch a sampled pose belongs to, and the arclength within that branch, so downstream tools can label frames.
- Poses must be usable directly by `IVUS_demo` and by the pullback / paired-render code paths.

Acceptance: A test script draws N random poses inside a reconstructed patient tree, renders frames through raysim, and verifies (a) frames come from all branches, (b) probe never intersects the wall unless wall-contact mode is on, (c) per-frame branch labels match the geometry.

### O6 — Export compatible with the existing IVUS stack

Emit the vesselgen on-disk contract (`vessel-generator/docs/on-disk-format.md`), extended for real branched anatomy:

```
<out>/
  lumen.obj                    # = input CTA lumen (cleaned), material: intima
  surfaces/
    interface_01.obj           # media
    interface_02.obj           # adventitia
  lesions/                     # optional
    lesion_00_hard.obj
  branches/
    branch_00.json             # centerline + per-station radii + parent link
    branch_01.json
    ...
  vessel.json                  # manifest: surfaces, lesions, branch graph, provenance
  preview.png                  # QC (thickness map, branch coloring, cross-sections)
```

Requirements:

- `vessel.json` extends the current schema with an explicit **branch graph** (nodes, edges, parent branch IDs, junction arclengths).
- The exporter is consumed by:
  1. `simulated_pullback_from_CT` via `vessel_manifest_path` (preferred first integration), and
  2. `IVUS_demo` (replaces the current single-slab geometry stage; may be feature-flagged during rollout).

### O7 — Engineering quality bar

- Tests: topology extraction (branch count on synthetic trees), non-crossing layers, wall-thickness / diameter correlation, lesion containment across branches, pose sampler stays in-lumen, manifest round-trip, normal orientation.
- QC artifacts per run: branch-graph overlay, thickness colormap on outer surface, cross-section gallery at bifurcations, sample-pose scatter.
- Documented CLI + Python API. Core library must have no hard-coded patient paths.

---

## 6. Suggested technical approach

Guidance, not mandatory design.

### 6.1 Recommended stages

```text
CTA lumen mesh (only input)
        │
        ▼
  (1) Clean / verify watertight
        │
        ▼
  (2) Voxelize interior → skeleton (medial axis)
        │
        ▼
  (3) Skeleton → branch graph + smoothed centerlines
        │
        ▼
  (4) EDT / inscribed sphere → per-vertex local diameter
        │
        ▼
  (5) Thickness field T(x) = f(local_diameter) + smooth perturbation
        │
        ▼
  (6) Offset intima → media (frac_intima_media · T)
      Offset intima → adventitia (T)
      (Preserve intima vertex-correspondence, junction-safe)
        │
        ▼
  (7) Optional: sample lesions per branch using T(x) as the wall envelope
        │
        ▼
  (8) Export vessel.json + surfaces + branches + preview
        │
        ▼
  (9) Provide interior pose-sampling API bound to the branch graph
```

### 6.2 Hard problems to plan for

1. **Topology extraction from a raw CTA mesh.**  
   Skeletonization of branched vascular meshes is a solved-but-fiddly problem. Candidate libraries: `skimage.morphology.skeletonize_3d` on a voxelized interior, `vmtk` (vascular modeling toolkit) for centerline + Voronoi diagrams, or mesh contraction (Au et al.). Expect to spend real time on cleanup at ostia. Junctions need to be classified and their arclengths recorded per branch — this feeds both O2 (interface generation) and O5 (pose labeling).

2. **Offsetting a branched triangulated lumen without self-intersection at junctions.**  
   A pure per-vertex normal offset breaks near ostia (adjacent branches’ offsets collide). Options:
   - Locally reduce offset magnitude near junctions (blend from `T` down to a fraction of `T` inside a small junction disk).
   - Compute the offset in a distance-field/level-set formulation and re-mesh (marching cubes on `SDF(lumen) − T(x)`).
   - Use vessel-generator’s per-branch layered sweep followed by boolean union, using each reconstructed branch centerline.
   
   Recommend prototyping the level-set approach first: it naturally respects branched topology and gives a clean adventitia surface without per-junction hand-coding.

3. **Intima preservation.**  
   Whatever offset strategy is chosen, `lumen.obj` in the output **must be the input mesh** (post cleanup — hole-fill, degenerate triangle removal, inward-normal export). Any re-meshing step (e.g. from a level-set) applies to the media/adventitia only; the intima triangulation is copied through untouched to guarantee this.

4. **Anatomy-scaled thickness.**  
   A simple `T_base = max(t_min, min(t_max, α · D_local))` with `α ≈ 0.08` and reasonable clamps is a good starting prior for the diameter trend (it matches the current CT-demo scaling). **On top of it**, always add the anatomical variation stack from O3 — circumferential eccentricity + smooth longitudinal drift + low-frequency patchy field — so the wall is never perfectly uniform even on constant-diameter segments. Treat the anatomical variation as required, not cosmetic.

5. **Coordinate frame.**  
   Keep the patient/CTA world frame throughout. Do not remap into vessel-generator’s Y-axis convention; the pullback / demo consumers already work in patient space.

6. **Interior pose sampling with branch labels.**  
   Precompute a lumen interior sampler (e.g. uniform in voxelized interior, or per-branch stratified via arclength). Each sample carries the branch ID of its nearest centerline segment and the local arclength, which is what physics-based localization will consume as ground truth.

### 6.3 Config sketch (illustrative)

```yaml
input:
  lumen_mesh: path/to/lumen.stl   # ONLY required input

topology:
  voxel_size_mm: 0.3
  min_branch_length_mm: 3.0
  junction_smoothing_mm: 2.0

wall:
  # Trilaminar only: input mesh IS the intima; media + adventitia are generated.
  layer_fracs:
    intima_media: 0.15   # media boundary at 0.15 * T outward from intima
    media_adventitia: 1.0
  thickness:
    # Diameter-driven baseline
    scale_of_local_diameter: 0.08
    min_mm: 0.35
    max_mm: 2.0
    # Biological variation stack (always applied)
    variation:
      seed: 7
      circumferential:
        max_frac: 0.35             # eccentricity amplitude
        angular_modes: [1, 2, 3]
        phase_drift_per_mm: 0.05
      longitudinal:
        max_frac: 0.15
        period_mm: 20
      patchy:
        max_frac: 0.20
        correlation_length_mm: 15  # low-freq GRF / blob field

disease:
  enabled: false
  seed: 42
  n_lesions: 2
  kinds: [hard, soft_lipid]
  branches: any            # or a list of branch IDs

pose_sampling:
  edge_margin_mm: 0.2
  max_tilt_deg: 15.0
  wall_contact_probability: 0.1

output:
  dir: out/patient23_ivus_model
```

---

## 7. Scope

### In scope (v1)

- Ingest a CTA lumen mesh (only).
- Full branch-graph reconstruction with per-branch centerlines and radii.
- Trilaminar wall (input = intima; generated media + adventitia) with anatomy-scaled variable thickness.
- Optional plaques / calcifications with on/off control.
- Interior pose-sampling API with branch labeling.
- Export to vesselgen-compatible manifest, extended with the branch graph.
- Integration into pullback and demo consumers.

### Out of scope (v1) — defer unless needed

- Automatic disease inference from imaging (placement is synthetic / config-driven).
- Changing raysim material physics tables.
- Non-tree topologies (fistulas, loops) beyond a documented failure mode.
- Coronary-scale defaults (current calibration is PV .035 peripheral; keep unless product asks otherwise).
- Full ML dataset generation on top of this model.

---

## 8. Proposed milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| M0 | Design spike | Skeletonization approach chosen; offset strategy chosen (level-set vs sweep+union); patient baseline (current single-slab) captured for comparison |
| M1 | Topology + centerlines | Branch graph + centerlines + per-station radii extracted from a real CTA lumen; QC overlay published |
| M2 | Trilaminar wall MVP | Media + adventitia interfaces generated with anatomy-scaled thickness; intima preserved bit-for-bit; passes non-crossing tests across the full tree |
| M3 | Manifest export + consumer wiring | `vessel.json` (with branch graph) round-trips; pullback and/or `IVUS_demo` renders trilaminar frames from the reconstructed model |
| M4 | Interior pose sampler | Random-in-lumen sampling with branch labels; validated by a rendered scatter of poses across all branches |
| M5 | Optional disease | Plaques/calcifications toggleable, containment-verified across branches |
| M6 | Handoff polish | Docs, tests, example config, known-limitations page, second-patient smoke test |

Sequencing note: M1 → M2 → M3 unblocks visual verification; M4 unblocks downstream localization work; M5 is additive.

---

## 9. Success criteria (definition of done)

1. One command (or documented Python API) converts a **lumen-only CTA mesh** → trilaminar vessel folder with a full branch graph.
2. The output `lumen.obj` is **identical (post-cleanup only) to the input mesh**; media + adventitia are added around it.
3. Total wall thickness clearly **correlates with local diameter and also varies anatomically on top of that trend** — visible spread at fixed diameter, eccentricity around the circumference, and longitudinal / patchy variation along each branch (verifiable in QC colormap + `T` vs `D_local` scatter).
4. Rendered IVUS frames show trilaminar wall behavior with correct material nesting.
5. **All branches** present in the input segmentation are present in the branch graph and reachable by the pose sampler.
6. Disease off/on differs only in lesion meshes / manifest entries.
7. Interior pose sampling returns poses **anywhere inside the lumen** with correct branch labels, usable by the physics-based localization consumer.
8. Loads through the same manifest path used by modern vesselgen / CT-pullback integration.
9. Tests + operator README exist so a new engineer can run a fresh patient without tribal knowledge.

---

## 10. Open decisions for the executing engineer

Resolve early and record in the project README:

1. Skeletonization backend: voxel skeleton (skimage), `vmtk`, mesh contraction, or a hybrid?
2. Offset backend: level-set / SDF re-meshing vs per-branch parametric sweep + boolean union?
3. Diameter estimator: EDT-based inscribed-sphere at medial points vs cross-section fitting on planes perpendicular to the centerline?
4. Thickness prior: linear-in-diameter with clamps, or a piecewise curve tuned per anatomical class (aorta vs iliac vs femoral)?
5. Package home: extend `simulated_pullback_from_CT`, add a new module under `vessel-generator/vesselgen/`, or create a new top-level package? (Lean recommendation: library code under `vessel-generator`, thin CLI, patient IO helpers shared with `simulated_pullback_from_CT`.)
6. Rollout for `IVUS_demo`: hard swap vs feature-flagged parallel path until M3 is stable.

---

## 11. Key references (read first)

| Doc / code | Why |
|------------|-----|
| `IVUS_demo/README.md` + `IVUS_demo/app.py` | Current segmented-patient demo contract and materials |
| `simulated_pullback_from_CT/src/geometry.py` | Existing lumen → single wall offset with radius-proportional thickness |
| `vessel-generator/docs/on-disk-format.md` | Target export schema (to be extended for branch graph) |
| `vessel-generator/docs/configuration.md` | Layer / lesion / pose-sampling knobs |
| `vessel-generator/docs/design.md` | Normal convention, wall math, lesion constraints |
| `vessel-generator/vesselgen/wall.py` | Trilaminar `LayeredWallField` reference |
| `vessel-generator/vesselgen/inclusions.py` | Plaque / calcium mesh generation |
| `vessel-generator/vesselgen/sampling.py` | Existing pose sampler (to be extended for branch-aware interior sampling) |
| `vessel-generator/vesselgen/bifurcation.py` | Existing (limited) side-branch attachment for comparison |
| `vessel-generator/docs/simulator-integration.md` | Manifest loading + world background model |
| `instrument-calibration/p035_visions/volcano_s5i.yaml` | Canonical probe/sim operating point |

---

## 12. First concrete tasks (checklist for assignee)

- [ ] Reproduce the current single-slab render for one CTA patient as a baseline for A/B comparison.
- [ ] Spike: voxelize a CTA lumen mesh and extract a branch graph + centerlines; publish QC overlay.
- [ ] Add local-diameter estimator; plot the diameter-scaled baseline `T_base` on the mesh.
- [ ] Add the anatomical variation stack (circumferential eccentricity + longitudinal drift + patchy field); publish a `T` vs `D_local` scatter showing trend + spread, and a colormap on the outer surface.
- [ ] Prototype trilaminar offset (level-set or sweep+union) on a segment containing at least one bifurcation.
- [ ] Verify intima passthrough: `lumen.obj` in output equals input mesh after cleanup only.
- [ ] Implement `vessel.json` exporter with branch graph; validate manifest loads in the pullback path.
- [ ] Implement interior pose sampler; render a scatter of frames across branches.
- [ ] Port lesion sampling onto the reconstructed wall field; add on/off toggle.
- [ ] Integrate with `IVUS_demo` (feature-flagged initially).
- [ ] Write tests + operator README; ship before/after example renders in the PR description.

---

## 13. Product intent (one-liner)

**Given only a CTA lumen segmentation of a full vascular tree, produce an IVUS-faithful trilaminar phantom** — using the CTA lumen itself as the intima, adding anatomy-scaled media and adventitia, preserving the entire branch topology, and supporting arbitrary in-lumen probe poses for downstream physics-based localization.
