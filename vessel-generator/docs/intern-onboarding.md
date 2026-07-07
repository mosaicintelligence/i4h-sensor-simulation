# Intern onboarding: vessel-generator scenario expansion

Welcome! This document orients you to the `vessel-generator` package and
describes the features you will add. Read it start-to-finish before writing
code.

**Your goal:** extend the procedural IVUS dataset generator so it better
covers three anatomy situations that were under-represented in the prior
simulated dataset:

1. **Large vessels** — walls that extend beyond the imaging field of view (FOV)
2. **Small vessels** — lumina whose walls sit inside or near the catheter
   ring-down artifact
3. **Adjacent vessels** — 1–2 parallel neighboring vessels visible alongside
   the primary vessel (probe inside the primary lumen)

Each scenario must be **configurable at batch-generation time**, the same way
existing knobs like `side_branch_probability` and `aortic_scale_probability`
are today.

> **Work geometry-first.** Almost all of this project — config, mesh generation,
> pose sampling, A-line ground truth, pytest, and visual QC via `preview.png` /
> `vesselgen-frames` — runs with **plain Python only**. You do **not** need
> CUDA, conda, Docker, or a GPU to implement and land the three features.
>
> Setting up the IVUS simulator is the hardest part of the environment and
> should **not block you**. Defer rendering until your geometry and tests are
> done; your mentor or a cloud VM can help with end-to-end IVUS validation
> afterward.

---

## Table of contents

1. [What this package does](#what-this-package-does)
2. [Repository layout](#repository-layout)
3. [Development environment (geometry only — start here)](#development-environment-geometry-only--start-here)
4. [How generation works today](#how-generation-works-today)
5. [Your tasks](#your-tasks)
6. [Configuration requirements](#configuration-requirements)
7. [Code quality and style expectations](#code-quality-and-style-expectations)
8. [Git and pull request workflow](#git-and-pull-request-workflow)
9. [Using Cursor (and other AI tools) effectively](#using-cursor-and-other-ai-tools-effectively)
10. [How to verify your work](#how-to-verify-your-work)
11. [Suggested implementation order](#suggested-implementation-order)
12. [IVUS rendering setup (optional — do later)](#ivus-rendering-setup-optional--do-later)
13. [Further reading](#further-reading)

---

## What this package does

`vessel-generator` builds **procedural 3D vessel anatomy** and pairs it with
**geometric ground truth** (contours, per-angle wall distances, segmentation
labels) for training IVUS segmentation models.

There are two layers:

| Layer | What it does | GPU required? |
|-------|--------------|---------------|
| **Geometry** | Builds meshes, samples catheter poses, computes GT | No |
| **Rendering** | Ray-traces B-mode images through the calibrated IVUS simulator (raysim) | Yes |

Most of your work lives in the **geometry layer**. Treat IVUS rendering as a
**final validation step**, not a day-one prerequisite.

Calibration reference (for when you reach rendering): PV .035 / Volcano s5i at
`instrument-calibration/p035_visions/volcano_s5i.yaml`.

---

## Repository layout

Work from the **monorepo root**:

```
i4h-sensor-simulation/          ← repo root (run commands from here)
├── vessel-generator/           ← YOUR PRIMARY PACKAGE
│   ├── vesselgen/              ← importable Python package
│   │   ├── config.py           ← generation parameters (start here for new knobs)
│   │   ├── vessel.py           ← top-level Vessel class
│   │   ├── sampling.py         ← pose sampling + A-line ground truth
│   │   ├── bifurcation.py      ← side-branch attachment (reference for new geometry)
│   │   ├── io.py               ← save/load vessel.json + OBJ meshes
│   │   ├── labels.py           ← segmentation label IDs
│   │   └── tools/              ← CLI entry points
│   ├── examples/
│   │   └── render_paired_dataset.py   ← end-to-end B-mode + seg dataset builder
│   ├── docs/                   ← package documentation (including this file)
│   └── tests/                  ← pytest suite (add tests for your features)
├── instrument-calibration/     ← calibrated probe YAML + evaluation helpers
└── i4h-sensor-simulation/
    └── ultrasound-raytracing/  ← raysim simulator (CUDA + Docker)
```

### Read these docs first (in order)

Read **this document** fully before the others. The remaining docs describe
the existing package; use the "What to notice" column to connect them to your
three features (and avoid confusing similar-but-different concepts).

| # | Doc | What to notice for your work |
|---|-----|------------------------------|
| 1 | **This doc** (`intern-onboarding.md`) | Your task spec, workflow, and acceptance criteria |
| 2 | [quickstart.md](quickstart.md) | **Path 1 only** for now (geometry). Skip Path 2 (paired rendering) until [IVUS rendering setup](#ivus-rendering-setup-optional--do-later) |
| 3 | [pipeline-overview.md](pipeline-overview.md) | Geometry layer vs rendering layer — you live in the left half of the diagram for most of the project |
| 4 | [configuration.md](configuration.md) | Existing scale/sampling knobs. **`aortic_scale_probability`** is the closest prior art for large vessels but does not cover beyond-FOV A-lines. **`side_branch_probability`** is *not* adjacent vessels |
| 5 | [design.md](design.md) | Coordinate conventions, mesh normal rules, and [limitations](design.md#limitations) (side branches ≠ Y-junctions; lesions + bifurcation are mutually exclusive) |
| 6 | [segmentation-labels.md](segmentation-labels.md) | Skim for context; you mainly need this when validating rendered segmentations |
| 7 | [simulator-integration.md](simulator-integration.md) | Defer until rendering phase — describes how meshes connect to raysim |

---

## Development environment (geometry only — start here)

### Step 1 — Install and smoke test (no CUDA, no Docker, no GPU)

**Yes — Step 1 works on a normal laptop.** You only need Python 3.10+ and
`pip`. No conda environment, no NVIDIA driver, no Docker, and no raysim build.

This setup is sufficient for **all geometry implementation, pytest, and visual
QC** throughout the project.

```bash
cd /path/to/i4h-sensor-simulation
pip install -e "vessel-generator[dev]"

# Smoke test: generate one vessel
vesselgen-vessel \
  --out vessel-generator/out/intern_smoke \
  --seed 1 \
  --layers 3 \
  --guidewire

# Run the existing test suite
pytest vessel-generator/tests/ -v

# Format check (see Code quality section)
black vessel-generator/
```

Open `vessel-generator/out/intern_smoke/preview.png` to inspect the
cross-section gallery. Open `vessel.json` to see the manifest format.

### Step 2 — Geometry QC without the simulator

These commands also require **no CUDA, Docker, or GPU**:

```bash
# Batch geometry
vesselgen-dataset --n 20 --out vessel-generator/out/intern_batch --base-seed 0

# Sample poses + geometric ground truth + preview overlays
vesselgen-frames \
  --vessel vessel-generator/out/intern_smoke \
  --out vessel-generator/out/intern_smoke/frames \
  --n 20 \
  --write-previews
```

Inspect `frames/pose_XXXX_preview.png` — these overlays show lumen/wall
contours on a schematic imaging plane and are your primary visual QA tool
**before** any IVUS rendering.

### Step 3 — Understand the CLI tools

| Command | Purpose | Simulator needed? |
|---------|---------|-------------------|
| `vesselgen-vessel` | Generate one vessel with explicit flags | No |
| `vesselgen-dataset` | Generate N vessels (geometry only) | No |
| `vesselgen-frames` | Sample poses + geometric GT JSON + previews | No |
| `render_paired_dataset.py` | Full B-mode + segmentation dataset | Yes (see [later section](#ivus-rendering-setup-optional--do-later)) |

### What you do **not** need for the main project work

- NVIDIA GPU
- Built raysim CUDA extension
- Conda / `ultrasound` environment
- Docker

Do not spend your first week fighting simulator setup. Get the three geometry
features working and tested first.

---

## How generation works today

### Pipeline overview

```
GenerationConfig.sample()  →  VesselConfig  →  Vessel.from_config()
       ↓
save_vessel()  →  OBJ meshes + vessel.json manifest
       ↓
sample_pose()  →  PoseSample (catheter position + orientation)
       ↓
ground_truth_at()  →  GroundTruth (polygons + A-line distances)
       ↓
[rendering] raysim simulate  →  B-mode image.npy
       ↓
acoustic_segmentation_mask()  →  segmentation.npy
```

### Configuration pattern (follow this for your features)

All batch-sampling knobs live in `GenerationConfig` (`vesselgen/config.py`).
The pattern is:

1. Add probability/range fields to `GenerationConfig`
2. Draw concrete values inside `GenerationConfig.sample()`
3. Pass them into `VesselConfig` (add new sub-configs if needed)
4. Teach `Vessel.from_config()` to build the new geometry
5. Expose the knobs on `vesselgen-dataset` CLI flags (optional but preferred)
6. Wire through `render_paired_dataset.py::generate_paired_dataset()` if the
   paired renderer overrides `GenerationConfig` internally

Existing examples of this pattern:

| Knob | Default | What it controls |
|------|---------|------------------|
| `aortic_scale_probability` | 0.18 | Draws 16–23 mm lumina (large, but still mostly in FOV) |
| `side_branch_probability` | 0.45 | Attaches one bifurcating side branch |
| `guidewire_probability` | 0.70 | Adds tungsten guidewire along centerline |
| `wall_contact_probability` | 0.12 | Probe-against-wall poses |

### A-lines (per-angle wall distances)

Stored in frame `metadata.json` under `ground_truth_geometric`. Computed in
`sampling.py::ground_truth_at()`:

- `distance_to_lumen_wall_mm` — distance from probe to lumen wall per angle
- `distance_to_outer_wall_mm` — distance to outer (adventitia) wall per angle
- Rays that do not hit a surface within `max_distance_mm` (the FOV radius
  `t_far_mm`) return **`np.nan`**

This NaN convention is how "wall not visible on this A-line" is represented
today. Your large-vessel work must preserve and test this behavior.

### Ring-down

Ring-down is **not** a geometry feature. It is applied at render time by the
simulator (`vesselgen/sim_randomization.py`). At 10 MHz the catheter
ring-down occupies roughly **r < 2–3.6 mm** from the probe origin.

Today's sampler deliberately draws lumen radii **≥ 4 mm** so the vessel wall
sits outside the ring-down disc. Your small-vessel feature will intentionally
violate that constraint.

### Field of view (FOV)

Imaging reaches out to `t_far_mm` (typically 17.5, 20.0, or 30.0 mm per
frame, randomized in `SimRandomizationConfig`). Pixels beyond this radius are
label `0` (background). A-lines beyond `t_far_mm` should be `NaN`.

### Side branches ≠ adjacent vessels

The existing `side_branch` feature attaches a **bifurcating daughter branch**
that shares a wall with the parent (Y-ish junction via boolean union). This
is different from **adjacent parallel vessels** (e.g., artery + vein running
side by side with a gap between them). Do not try to repurpose side-branch
code for adjacent vessels — add a new geometry concept.

---

## Your tasks

### Task 1 — Large vessels (wall beyond FOV)

**Clinical motivation:** In lab data, some vessels were so large that portions
of the wall extended past the IVUS field of view. The segmentation model must
learn that some A-lines have **no visible wall** because the anatomy continues
past what the probe can see.

**What to implement:**

- A new vessel scale (or extension of the aortic draw) where the outer wall
  exceeds `t_far_mm` on some angular sectors for typical poses.
- Geometric ground truth must mark those A-lines as **`NaN`** in
  `distance_to_lumen_wall_mm` and `distance_to_outer_wall_mm` (already the
  convention when no intersection occurs within `max_distance_mm`).
- Segmentation labels must remain correct: in-FOV regions without a visible
  wall echo should not be falsely labeled as wall tissue.
- Configurable occurrence rate (e.g., `large_vessel_beyond_fov_probability`)
  plus radius/geometry ranges.

**Design hints:**

- The current `aortic_scale_probability` (18%) draws 16–23 mm lumina, but FOV
  goes out to 30 mm — so "aortic" alone may not produce enough beyond-FOV
  cases. You likely need larger radii and/or eccentric wall geometry so some
  angular sectors exceed `t_far_mm` even when the mean radius does not.
- Consider coupling large-vessel draws to FOV: a vessel drawn for a 17.5 mm
  FOV frame is more likely to exceed the boundary than one at 30 mm.
- Look at `sampling.py::_ray_polygon_intersection_distance` and
  `ground_truth_at(max_distance_mm=...)` — the NaN path already exists;
  your job is to **generate anatomy that triggers it reliably** and to add
  tests proving it.

**Acceptance criteria:**

- [ ] Batch generator produces large-vessel cases at the configured rate
- [ ] For generated poses, some A-lines have `NaN` wall distances where the
      true wall lies beyond `t_far_mm`
- [ ] `preview.png` and `vesselgen-frames` pose previews show the expected "open" sectors
- [ ] pytest covers the new config draw and A-line NaN behavior
- [ ] *(Optional, post-merge)* Rendered `overlay.png` confirms IVUS appearance when simulator is available

---

### Task 2 — Small vessels (wall inside ring-down zone)

**Clinical motivation:** Lab data included vessels so small that much of the
wall sat inside the catheter ring-down artifact, obscuring the wall echo.
Models trained without this confound merge structures when ring-down covers
neighboring walls.

**What to implement:**

- A new vessel scale with **small lumen radii** (wall at or inside the
  ~2–3.6 mm ring-down disc).
- Configurable occurrence rate (e.g., `small_vessel_probability`) plus a
  `small_vessel_radius_mm_range`.
- Poses should still place the probe **inside the lumen** (existing rejection
  sampler in `sampling.py`).
- Labels and A-lines must remain geometrically correct in the visible region
  beyond ring-down.

**Design hints:**

- Read the ring-down note in `config.py` (lines 15–16): today's defaults
  intentionally avoid this case. Your feature opts in explicitly.
- Ring-down appearance comes from the simulator; you do not need to model the
  artifact in geometry. Validate visually with rendered B-mode frames.
- Small vessels may interact with adjacent-vessel scenarios (Task 3) — design
  configs so both can be independently enabled.

**Acceptance criteria:**

- [ ] Batch generator produces small-vessel cases at the configured rate
- [ ] Mean lumen radius falls within your declared small-vessel range
- [ ] Geometric GT and `vesselgen-frames` previews are self-consistent
- [ ] pytest covers config sampling and basic geometry validity (watertight
      meshes, probe fits inside lumen)
- [ ] *(Optional, post-merge)* Rendered frames show wall echo obscured by ring-down

---

### Task 3 — Adjacent parallel vessels

**Clinical motivation:** Some pullback regions show two parallel vessels
(e.g., artery and vein). When ring-down obscures the wall between them,
segmentation models tend to **merge** the two structures. We need training
examples with a primary vessel (probe inside) and 1–2 nearby, non-overlapping
parallel vessels of similar size.

**What to implement:**

- New geometry: **1–2 neighbor vessels** running parallel to the parent,
  offset laterally so they do **not** share a wall (unlike side branches).
- The catheter/probe remains inside the **primary** (parent) lumen.
- Neighbors appear in the IVUS image as separate lumina/walls.
- Each neighbor needs its own meshes and manifest entries so raysim renders
  them and GT/segmentation distinguish them.
- Configurable occurrence rate, neighbor count (1–2), lateral offset range,
  and size ratio relative to parent.

**Design hints:**

- Study `bifurcation.py` for mesh attachment patterns, but adjacent vessels
  should **not** boolean-union with the parent — they are separate objects
  placed nearby.
- You will likely add something like `AdjacentVesselConfig` to `config.py` and
  teach `Vessel.from_config()` / `io.save_vessel()` to emit extra branch
  entries or a new manifest section. **Coordinate with your mentor** on
  whether neighbors get their own label IDs or share the parent's wall labels
  — document your choice in the PR.
- Pose sampling should remain in the parent lumen. Neighbors are visible
  because the IVUS beam reaches them laterally.
- Pay special attention to the **small vessel + adjacent vessel** combination
  (the clinically problematic merge case). Consider a dedicated probability
  or a pose-enrichment strategy that produces this combination at a configurable
  rate.

**Acceptance criteria:**

- [ ] Batch generator produces adjacent-vessel cases at the configured rate
- [ ] 1–2 neighbors appear in `preview.png` offset from the parent
- [ ] `vesselgen-frames` previews show distinct neighboring contours
- [ ] Neighbors do not overlap the parent mesh (validate with mesh clearance
      checks)
- [ ] pytest covers config sampling, mesh validity, and GT polygon count
- [ ] *(Optional, post-merge)* Rendered frames show distinct lumina; segmentation
      distinguishes primary vs. neighbor tissue

---

## Configuration requirements

Add new fields to `GenerationConfig` following existing naming conventions.
Suggested starting points (adjust ranges with your mentor):

```python
# Large vessels — wall extends beyond FOV on some A-lines
large_vessel_beyond_fov_probability: float = 0.10
large_vessel_radius_mm_range: tuple[float, float] = (10.0, 14.0)

# Small vessels — wall inside/near ring-down
small_vessel_probability: float = 0.10
small_vessel_radius_mm_range: tuple[float, float] = (1.8, 3.5)

# Adjacent parallel vessels
adjacent_vessel_probability: float = 0.15
adjacent_vessel_count_range: tuple[int, int] = (1, 2)
adjacent_vessel_offset_mm_range: tuple[float, float] = (3.0, 8.0)
adjacent_vessel_radius_frac_range: tuple[float, float] = (0.7, 1.1)

# Optional: elevate rate of small + adjacent together
small_adjacent_combo_probability: float = 0.05
```

**Rules:**

- Every `_probability` field must be in `[0.0, 1.0]`
- Probabilities are **per-vessel** draws inside `GenerationConfig.sample()`
- Document new fields in `docs/configuration.md`
- If you add CLI flags, update `docs/cli-reference.md`
- Ensure probabilities are **independent** unless you explicitly model a
  joint draw (like `small_adjacent_combo_probability`)

---

## Code quality and style expectations

This is production training-data code. Sloppy geometry bugs silently poison
model training. Hold yourself to these standards:

### General

- **Read before you write.** Open the nearest existing feature (e.g.,
  `side_branch_probability`) and mirror its structure.
- **Smallest correct diff.** Do not refactor unrelated code. Do not add
  abstractions for one-off helpers.
- **Deterministic tests.** Pass explicit `seed=` values; use
  `numpy.random.default_rng(seed)`.
- **No silent failures.** Raise clear exceptions (see `BifurcationError` in
  `bifurcation.py`) rather than returning broken meshes.

### Style conventions (match the existing package)

| Convention | Example in codebase |
|------------|---------------------|
| `from __future__ import annotations` | Top of every module |
| Dataclasses for config | `config.py` |
| `snake_case` functions, `PascalCase` classes | `sample_pose`, `VesselConfig` |
| Units in mm; angles in degrees unless named `_rad` | `mean_radius_mm`, `thetas_rad` |
| Vessel axis along **Y**, cross-section in **xz** | `design.md` |
| Docstrings on public classes/functions | `sampling.py::GroundTruth` |
| Type hints on function signatures | Throughout `vesselgen/` |

### Formatting with Black

The package ships a Black config in `pyproject.toml` (`line-length = 100`).
Run it on every file you touch before opening a PR:

```bash
pip install -e "vessel-generator[dev]"   # includes black
black vessel-generator/vesselgen vessel-generator/tests vessel-generator/examples
```

You do not need to reformat the entire existing codebase — only your changed
files (though running on the paths above before each PR is fine).

### Tests (required)

Add tests under `vessel-generator/tests/`. Follow existing patterns:

```bash
pytest vessel-generator/tests/test_your_feature.py -v
pytest vessel-generator/tests/ -v   # full suite must pass
```

Every new config knob needs at least:

1. A test that the probability produces the feature at the expected rate
   (use a fixed seed and a batch of ~100–200 draws)
2. A test that generated geometry is valid (watertight mesh, probe fits,
   neighbors don't overlap, etc.)
3. A test that ground truth behaves correctly (A-line NaNs for Task 1,
   polygon counts for Task 3)

### Documentation (required)

Update `docs/configuration.md` with your new knobs. If you add CLI flags,
update `docs/cli-reference.md`.

---

## Git and pull request workflow

You will use **three levels of branches**. Feature work never goes directly
into `ivus-probe` or `main` until the very end.

```
ivus-probe                    ← team integration branch (merge here last)
    └── vessel-gen-features-lab-1   ← YOUR main working branch
            ├── feature/small-vessels
            ├── feature/large-vessels-beyond-fov
            └── feature/adjacent-vessels
```

| Branch | Who uses it | Purpose |
|--------|-------------|---------|
| `main` | Everyone | Production — **do not push or PR here** |
| `ivus-probe` | Team | Integration branch for IVUS work — **final merge target only** |
| `vessel-gen-features-lab-1` | You | Your integration branch; all three features land here first |
| `feature/...` | You | One branch per task; short-lived |

**Never open PRs into `main`.** During the project, open feature PRs into
`vessel-gen-features-lab-1` only. When all three features are done, you merge
`vessel-gen-features-lab-1` into `ivus-probe` once (see
[Final handoff](#final-handoff-all-features-complete)).

### Git vocabulary (30-second version)

| Term | Meaning |
|------|---------|
| **Branch** | A parallel line of work. Changes on one branch do not affect others until you merge. |
| **Commit** | A saved snapshot of your changes on your machine, with a message describing what you did. |
| **Push** | Upload your commits from your laptop to GitHub (`origin`). |
| **Pull** | Download the latest commits from GitHub to your laptop. |
| **PR (pull request)** | A request on GitHub to merge your branch into another branch. Your mentor reviews it before it merges. |

### One-time setup

Run once when you start the project (from the repo root):

```bash
cd /path/to/i4h-sensor-simulation

# Download the latest branch list from GitHub
git fetch origin

# Switch to your main working branch
git checkout vessel-gen-features-lab-1

# Make sure your local copy matches GitHub
git pull origin vessel-gen-features-lab-1
```

Confirm you are on the right branch:

```bash
git branch --show-current
# should print: vessel-gen-features-lab-1
```

Install the package and run the smoke test (see [Step 1](#step-1--install-and-smoke-test-no-cuda-no-docker-no-gpu)).

### Per-feature workflow

Repeat these steps for **each task** (small vessels, large vessels, adjacent
vessels). One feature branch and one PR at a time.

#### 1. Start from a clean lab branch

Always branch off the latest `vessel-gen-features-lab-1`:

```bash
git checkout vessel-gen-features-lab-1
git pull origin vessel-gen-features-lab-1
```

#### 2. Create a feature branch

Use a short, descriptive name:

```bash
# Task 2 example
git checkout -b feature/small-vessels

# Task 1 example (when you start that task)
# git checkout -b feature/large-vessels-beyond-fov

# Task 3 example
# git checkout -b feature/adjacent-vessels
```

#### 3. Do your work, then commit

Check what changed:

```bash
git status          # list modified files
git diff            # show line-by-line changes (optional)
```

Run tests before every commit:

```bash
pytest vessel-generator/tests/ -v
black vessel-generator/vesselgen vessel-generator/tests
```

Stage and commit (repeat as you make progress — small commits are fine):

```bash
git add vessel-generator/vesselgen/config.py
git add vessel-generator/tests/test_small_vessels.py
# ... add each file you changed, or:
git add vessel-generator/

git commit -m "Add small-vessel probability and radius range to GenerationConfig"
```

Write commit messages in plain English: what changed and why, not just "fix" or
"update".

#### 4. Push the feature branch to GitHub

First push for a new branch (sets upstream tracking):

```bash
git push -u origin feature/small-vessels
```

Later pushes on the same branch:

```bash
git push
```

#### 5. Open a pull request

On GitHub (or ask your mentor to show you `gh pr create`):

- **Base branch:** `vessel-gen-features-lab-1`
- **Compare branch:** `feature/small-vessels` (your feature branch)
- **Not** `ivus-probe` or `main`

Using the GitHub CLI (if installed):

```bash
gh pr create \
  --base vessel-gen-features-lab-1 \
  --head feature/small-vessels \
  --title "Add small-vessel scenario sampling" \
  --body "Implements Task 2 from intern onboarding. See preview.png in PR comments."
```

#### 6. After the PR merges — sync your lab branch

Once your mentor merges the PR on GitHub, update your local lab branch before
starting the next feature:

```bash
git checkout vessel-gen-features-lab-1
git pull origin vessel-gen-features-lab-1
```

You can delete the old feature branch locally (optional):

```bash
git branch -d feature/small-vessels
```

### One feature per PR

| PR | Base branch | Compare branch | Scope |
|----|-------------|----------------|-------|
| PR 1 | `vessel-gen-features-lab-1` | `feature/small-vessels` | Small vessels (Task 2) |
| PR 2 | `vessel-gen-features-lab-1` | `feature/large-vessels-beyond-fov` | Large vessels (Task 1) |
| PR 3 | `vessel-gen-features-lab-1` | `feature/adjacent-vessels` | Adjacent vessels (Task 3) |

Do not bundle all three features into one PR.

### Before requesting review

```bash
pytest vessel-generator/tests/ -v
black vessel-generator/vesselgen vessel-generator/tests
git push
```

Include in the PR description:

- Which task the PR implements
- Example `vesselgen-vessel` / `vesselgen-frames` commands to reproduce QC output
- Screenshot of at least one `preview.png` or `pose_XXXX_preview.png`
- Note whether IVUS rendering was validated (optional at this stage)

### Final handoff (all features complete)

When all three feature PRs are merged into `vessel-gen-features-lab-1` and
tests pass on that branch:

#### 1. Remove this onboarding document

This file is only for your onboarding — it should not land on `ivus-probe`.

```bash
git checkout vessel-gen-features-lab-1
git pull origin vessel-gen-features-lab-1

git rm vessel-generator/docs/intern-onboarding.md
git commit -m "Remove intern onboarding doc after project completion"
git push origin vessel-gen-features-lab-1
```

#### 2. Merge the lab branch into `ivus-probe`

Open **one final PR**:

- **Base branch:** `ivus-probe`
- **Compare branch:** `vessel-gen-features-lab-1`

```bash
gh pr create \
  --base ivus-probe \
  --head vessel-gen-features-lab-1 \
  --title "Add vessel scenario features (small, large beyond-FOV, adjacent)" \
  --body "Integrates all three vessel-gen lab features. Intern onboarding doc removed."
```

Your mentor will review and merge this PR. **Do not merge into `main`.**

### If you get stuck on git

| Problem | What to run |
|---------|-------------|
| "Your branch is behind" | `git pull origin vessel-gen-features-lab-1` (on the branch it names) |
| Committed on wrong branch | Ask your mentor before `git reset` — do not force-push |
| Merge conflicts after pull | Ask your mentor; do not guess at conflict resolution |
| Accidentally edited `main` | `git stash`, `git checkout vessel-gen-features-lab-1`, `git stash pop` |

When asking for help, include the output of `git status` and the exact command
that failed.

---

## Using Cursor (and other AI tools) effectively

You are encouraged to use Cursor. These practices keep AI assistance from
introducing subtle bugs:

### Workflow: Plan mode first, then build

For **each feature**, start in Cursor **Plan mode** (not Agent/build mode):

1. @-mention **`docs/intern-onboarding.md`** (this document) and the relevant
   reference code (e.g. `config.py::GenerationConfig.sample`).
2. Ask the agent to produce a **step-by-step implementation plan** for that
   feature only.
3. **Read the full plan yourself.** You should understand every step before
   any code is written. Push back with detailed feedback — correct coordinate
   conventions, config field names, test cases, files to touch, what *not* to
   change.
4. Only switch to **Agent/build mode** once you agree with the plan.
5. After implementation, run pytest and black yourself; do not trust the
   agent's "done" message.

If the plan references IVUS rendering or raysim setup, redirect it: geometry
and pytest first, rendering later.

### Do

1. **Scope each prompt narrowly.** "Add `small_vessel_probability` to
   `GenerationConfig` following the `aortic_scale_probability` pattern" is
   better than "implement all three tasks."
2. **Point the agent at reference code.** Paste or @-mention
   `config.py::GenerationConfig.sample` and say "follow this pattern."
3. **Always @-mention this onboarding doc** so the agent knows the task
   boundaries and acceptance criteria.
4. **Ask for tests in the same prompt** as the implementation.
5. **Review every diff yourself.** Read the changed lines; do not merge
   blindly. AI often misses coordinate conventions (Y-axis vessel) and
   manifest material names.
6. **Run pytest and black after every change.** If tests fail, fix before moving on.
7. **Use `@docs/design.md` and `@docs/configuration.md`** in Cursor so the
   agent respects package conventions.

### Do not

- Let the agent rewrite large files wholesale.
- Accept code that imports `raysim.cuda` into geometry modules (keep rendering
  concerns in `examples/render_paired_dataset.py` and `sim_randomization.py`).
- Skip visual inspection because tests pass — mesh bugs can pass watertight
  checks and still look wrong in IVUS.
- Invent new label IDs without discussing with your mentor (downstream
  training code depends on the table in `labels.py`).

### Suggested Cursor verification prompts

After implementing each task, ask Cursor:

> "Write a pytest that draws 200 vessels with
> `large_vessel_beyond_fov_probability=1.0` and asserts at least 80% have
> some NaN A-line wall distances at `t_far_mm=17.5`."

> "Review my diff against `bifurcation.py` — does the adjacent vessel
> attachment respect outward-normals-in-memory / inward-normals-on-disk?"

---

## How to verify your work

Use this checklist per task. **Geometry checks are required; IVUS rendering
checks are optional** until you have simulator access.

### Automated (run on every commit)

```bash
pytest vessel-generator/tests/ -v
black vessel-generator/vesselgen vessel-generator/tests   # on files you changed
```

### Geometry QC (no GPU — primary verification)

```bash
# Force your feature on for inspection
vesselgen-vessel --out vessel-generator/out/qc_large --seed 42 --layers 3
# (once you add CLI flags, pass --large-vessel etc.)

# Sample poses + GT
vesselgen-frames \
  --vessel vessel-generator/out/qc_large \
  --out vessel-generator/out/qc_large/frames \
  --n 20 \
  --write-previews
```

Inspect:

- `preview.png` — cross-sections look anatomically plausible
- `frames/pose_XXXX_preview.png` — GT contours align with mesh geometry
- `frames/pose_XXXX.json` — A-line arrays for Task 1 (look for `null`/NaN
  values in `distance_to_*_wall_mm`)

### Batch statistics (no GPU)

Write a short script (or test) that generates N=500 vessels with your new
probabilities set to known values and prints the empirical occurrence rate.
It should match the configured probability within sampling noise (~±5% for
p=0.15, N=500).

### IVUS rendering QC (optional — GPU, VM, or Docker)

Skip this subsection until you have completed geometry + pytest for a task
and have simulator access. It is **not** required to merge geometry PRs.

```bash
# Native raysim on a GPU machine (see IVUS rendering setup section)
python vessel-generator/examples/render_paired_dataset.py \
  --out vessel-generator/out/qc_paired \
  --n 20 \
  --frames-per-vessel 3 \
  --seed 0
```

Or render individual vessels through `raysim.docker` (see
[IVUS rendering setup](#ivus-rendering-setup-optional--do-later)).

For each scenario, open at least 5 frames and check:

| Task | What to look for in `image.png` / `overlay.png` |
|------|--------------------------------------------------|
| Large vessel | Sectors where the wall echo disappears before the FOV edge; overlay shows lumen extending to FOV boundary without a wall ring on those angles |
| Small vessel | Bright central ring-down covers a large fraction of the visible wall |
| Adjacent vessel | Two (or three) distinct lumina; walls do not merge when ring-down is on |

### Regression

Confirm you did not break existing scenarios:

```bash
vesselgen-dataset --n 20 --out vessel-generator/out/regression --base-seed 99
pytest vessel-generator/tests/ -v
```

---

## Suggested implementation order

1. **Read this doc + run Step 1 smoke tests** (half day) — no simulator setup
2. **Task 2 — Small vessels** — smallest geometry change; good warmup → **PR 1**
3. **Task 1 — Large vessels / beyond-FOV A-lines** → **PR 2**
4. **Task 3 — Adjacent vessels** — largest new geometry concept → **PR 3**
5. **Config + CLI + docs pass** — folded into each feature PR, not a separate mega-PR
6. **Final handoff** — remove this doc; merge `vessel-gen-features-lab-1` → `ivus-probe`
7. **IVUS rendering validation** (optional) — after geometry PRs merge, using
   Docker, a cloud VM, or a team GPU machine

---

## IVUS rendering setup (optional — do later)

You can complete and merge all three geometry features without ever running
the simulator. When you (or your mentor) are ready for end-to-end IVUS
validation, use one of the paths below.

**Expect this to be the hardest environment setup step.** Budget time for it
separately from feature work; do not block geometry PRs on it.

### Current state

**The paired-dataset renderer does not yet have a `--docker` flag.** Today,
`render_paired_dataset.py` imports the native `raysim` CUDA extension directly
via `VesselRenderer` in `examples/render_paired_dataset.py`.

| Component | Runs without simulator? |
|-----------|-------------------------|
| `vesselgen` geometry + GT + previews | **Yes** |
| `render_paired_dataset.py` | No — requires native raysim or manual Docker wiring |
| `raysim.docker.DockerSimulator` | No on host CUDA build, but host only needs Docker + GPU |

### Option A — Cloud VM with native raysim (recommended for full pipeline)

Follow the step-by-step VM guide (Google Cloud GPU instance, NVIDIA driver,
OptiX runtime, conda build):

```
i4h-sensor-simulation/ultrasound-raytracing/docs/vm_setup.md
```

Then build raysim per
[quick_start.md](../../i4h-sensor-simulation/ultrasound-raytracing/docs/quick_start.md)
and run:

```bash
conda activate ultrasound
pip install -e vessel-generator

python vessel-generator/examples/render_paired_dataset.py \
  --out vessel-generator/out/intern_paired_smoke \
  --n 4 \
  --frames-per-vessel 2 \
  --seed 42
```

Inspect `frames/frame_00000/image.png`, `segmentation.png`, and `overlay.png`.

### Option B — Docker (no local CUDA/OptiX build)

The raysim package ships a Docker workflow. The host needs Docker, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
and a GPU — but **not** a local raysim compile.

Verify GPU access inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Build the image (one time, ~8–15 min):

```bash
cd i4h-sensor-simulation/ultrasound-raytracing
docker build -f docker/Dockerfile -t raysim:latest .
```

Full details: `i4h-sensor-simulation/ultrasound-raytracing/docs/docker_build.md`.

Smoke test:

```bash
cd i4h-sensor-simulation/ultrasound-raytracing
python examples/docker_quickstart.py --output-dir /tmp/raysim_smoke
```

To render a vessel you generated with `vesselgen-vessel`, use the
`raysim.docker` API (see `examples/docker_ivus_example.py`). Use
`render_paired_dataset.py::build_vessel_world` as the reference for mesh
paths and material names from `vessel.json`.

Adding optional Docker rendering to `VesselRenderer` (so
`render_paired_dataset.py --docker` works end-to-end) is a valuable stretch
goal. Coordinate with your mentor — not required to land the geometry features.

---

## Further reading

| Resource | Path |
|----------|------|
| Package README | `vessel-generator/README.md` |
| Config dataclasses | `vesselgen/config.py` |
| A-line computation | `vesselgen/sampling.py` |
| Segmentation rasterization | `examples/render_paired_dataset.py::_cartesian_label_grid` |
| Raysim Docker docs | `i4h-sensor-simulation/ultrasound-raytracing/docs/docker_build.md` |
| Cloud VM setup (native raysim) | `i4h-sensor-simulation/ultrasound-raytracing/docs/vm_setup.md` |
| Calibration scenarios that motivated current scales | `instrument-calibration/p035_visions/vessel_evaluation.py` |
| Example paired dataset (on disk) | `ivus_paired_dataset_10k/frames/frame_00000/` |

When you are stuck for more than 30 minutes, ask your mentor. Include: what you
tried, the command you ran, and the error output or screenshot.
