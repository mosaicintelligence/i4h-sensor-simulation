# Simulator Calibration Wiring — Pass 2 Handoff

This document is a handoff brief for an agent picking up the simulator-calibration
wiring on a machine that **can build and run the CUDA/OptiX simulator**. The
prior agent did the design work and Pass 1 plumbing on a machine without a GPU,
so the calibrated YAML is now reaching the C++ pipeline on paper but has not yet
been validated against a real run, and the new physics work (ring-down) is not
yet implemented.

Read this top to bottom before touching code; everything you need is linked.

---

## 1. Repository layout (relevant pieces only)

The workspace root contains two co-located projects plus the captured DICOM
dataset:

```
<workspace-root>/
├── i4h-sensor-simulation/            # The simulator (this is the git repo)
│   └── ultrasound-raytracing/
│       ├── csrc/                     # C++ / CUDA / OptiX sources
│       ├── include/                  # Public headers
│       ├── raysim/                   # Python package (bindings + config)
│       ├── examples/                 # Runnable scripts (ivus_example.py, etc.)
│       └── docs/                     # ivus_implementation_writeup.md, this file
├── instrument-calibration/           # Calibrated YAML + per-instrument fitters
│   └── p035_visions/
│       ├── volcano_s5i.yaml          # Canonical YAML produced by the calibration pipeline
│       ├── parameter_sheet.csv       # Provenance / running spreadsheet
│       ├── calibration_delta.md      # What's measured vs. what's still pending
│       ├── extract_*.py / fit_*.py   # The fitters (do not need to run for Pass 2)
│       └── README.md
└── P_035_PointScatter/               # Raw + derived data for the PV .035 instrument
    └── derived/ringdown/             # Includes the ringdown template Pass 2 needs
```

`i4h-sensor-simulation` is the only git repo; the other two folders are
working data and per-instrument calibration outputs that the simulator consumes
at runtime via paths in the YAML.

---

## 2. What Pass 1 already did (summary, with primary references)

The full writeup is in
[`ultrasound-raytracing/docs/ivus_implementation_writeup.md`](./ivus_implementation_writeup.md)
§11 (`Configuration-driven processing pipeline (Pass 1)`). One-paragraph version:

> Every parameter in `simulate()` that was previously a hard-coded literal —
> the TGC control points, the `log_compression` multiplier and floor, the
> `median_clip_filter` kernel size and dB clamps, and the scatter pipeline
> knobs — is now read from `SimParams` instead. Defaults exactly reproduce the
> prior literals, so any existing example/script that constructs `SimParams()`
> and only sets the previously exposed fields produces a bit-identical
> pipeline. New `SimParams::tgc_control_points` (a `std::vector<TgcControlPoint>`)
> with the empty-list-means-default semantics carries the calibrated TGC
> schedule from the YAML. `IvusSimConfig.to_sim_params()` was extended to set
> all the new fields, and the `_PARTIALLY_WIRED_PATHS` registry is now empty.

Files changed (all on whatever feature branch you're picking up — there are
no committed changes yet, see §6):

| File | What changed |
|---|---|
| `include/raysim/cuda/optix_trace.hpp` | Added the three `Params` fields the .cu/.cpp were already using (latent bug fix, see §11.1 of the writeup). |
| `include/raysim/core/raytracing_ultrasound_simulator.hpp` | New public `struct TgcControlPoint`; extended `SimParams` with 9 new fields. |
| `csrc/core/raytracing_ultrasound_simulator.cpp` | `simulate()` now sources scatter params, TGC schedule, log compression, and median clip from `SimParams`. User-supplied TGC bypasses the probe-type cache and rebuilds per frame. |
| `csrc/python/raysim_bindings.cpp` | Bound `TgcControlPoint`; added `def_readwrite` for every new `SimParams` field; updated docstring. |
| `raysim/__init__.py` | Re-exports `TgcControlPoint`. |
| `raysim/config.py` | `IvusSimConfig.to_sim_params()` sets every Pass 1 field; `_PARTIALLY_WIRED_PATHS = ()`. |

There were **no** changes to:

- The CUDA kernels themselves (the existing `log_compression`, `mul_row`,
  `median_clip_filter` already accepted the right types).
- The OptiX raytracing kernel.
- Any probe class, scan-conversion path, or material.

So if a Pass 1 run produces a different output than `main`, you almost
certainly have a misrouted parameter, not a regression in physics.

### 2.1 Open design decisions already made by the user

The user explicitly chose the following options before Pass 1 landed; treat
these as fixed unless they revisit them:

1. **Scope of Pass 2 (option 1b):** ring-down + log-compression knobs
   (`dynamic_range_db`, `reject_db`). Skip noise and per-slider `gain_db` for
   now.
2. **`ring_down.enabled = false` semantics (option 2a):** truly no ring-down
   signal. `enabled = true` adds the calibrated residual that survives the
   device's Acoustic Reference subtraction.
3. **Phasing (option 3b):** Pass 1 (plumbing) first, validate, then Pass 2
   (ring-down + new physics). This is the validation gate you're standing at.
4. **Build environment (option 4b):** the prior agent had no GPU/CUDA; you do.
   They were extra defensive with defaults to make the validation step
   trustworthy.

---

## 3. First job: validate Pass 1 actually works on the GPU build

The point of Pass 1 was to be transparent. Before any new code, prove that:

### 3.1 The simulator still builds

```bash
cd i4h-sensor-simulation/ultrasound-raytracing
# however you build this (cmake/setup.py/etc.) — see ultrasound-raytracing/README.md
```

Things to watch for that the prior agent could not check:

- The `Params` struct in `optix_trace.hpp` gained three new fields. Anything
  that does a `static_assert(sizeof(Params) == ...)` or relies on a fixed
  layout (host/device side) will break. Grep for `sizeof(Params)`.
- `SimParams` gained `std::vector<TgcControlPoint>`. If any consumer
  brace-initialises `SimParams{}` with positional arguments (rather than
  designated initialisers / field-by-field assignment) the order shift will
  silently misassign. Grep for `SimParams{` (with brace) — examples and tests
  use field-by-field assignment, but check anyway.
- pybind11/STL: the bindings rely on `<pybind11/stl.h>` (already included) for
  `std::vector<TgcControlPoint>`. No new pybind opt-in needed.

### 3.2 Default behaviour is byte-identical

This is the most important check. Run an unchanged example that only sets
fields exposed before Pass 1, and diff the output against a known-good
reference from `main`:

```bash
# Pre-change reference (one-time): check out main, run, save the b_mode npy.
git stash                          # or work in a separate worktree
git checkout main
python3 examples/ivus_example.py   # save the resulting B-mode somewhere
git checkout -                     # back to the Pass 1 branch
git stash pop

# Now run the same example on Pass 1 and compare.
python3 examples/ivus_example.py
python3 -c "import numpy as np; a=np.load('reference.npy'); b=np.load('passed1.npy'); print('max abs diff:', np.max(np.abs(a-b)))"
```

Expected: **identical** (or within float-rounding noise from non-deterministic
CUDA reductions, which we don't believe are on this path). Any visible diff is
a bug.

A second, weaker check — load the calibrated YAML and confirm
`partially_wired_fields()` is empty and `pending_fields()` lists exactly the
Pass 2 items:

```python
import sys
sys.path.insert(0, "i4h-sensor-simulation/ultrasound-raytracing")
from raysim.config import IvusSimConfig
cfg = IvusSimConfig.from_yaml("instrument-calibration/p035_visions/volcano_s5i.yaml")
assert cfg.partially_wired_fields() == []
print(cfg.pending_fields())
# Should print exactly:
#   processing.dynamic_range_db = 40.6
#   processing.reject_db = -40.6
#   processing.noise.sigma = 2.6347
#   processing.ring_down.amplitude = 46.37
#   processing.ring_down.extent_mm = 3.0
#   processing.ring_down.decay = "measured"
#   processing.ring_down.waveform_path = "P_035_PointScatter/derived/ringdown/ringdown_template_g54_d60.npy"
```

### 3.3 The YAML drives the pipeline

End-to-end smoke test for Pass 1:

```python
from raysim import IvusSimConfig, RaytracingUltrasoundSimulator, World, Materials

cfg = IvusSimConfig.from_yaml("instrument-calibration/p035_visions/volcano_s5i.yaml")
probe = cfg.to_probe()
sim_params = cfg.to_sim_params()
# build a phantom World/Materials however the existing examples do
sim = RaytracingUltrasoundSimulator(world, materials)
b_mode = sim.simulate(probe, sim_params)
```

Things to verify visually on this run **before** moving to Pass 2:

- The TGC schedule from the YAML actually applied (you can verify by
  temporarily enabling `sim_params.write_debug_images = True` and inspecting
  `debug_images/2_tgc.png`; depth-dependent gain should match the 5-point
  schedule, not the 2-point IVUS default).
- The log compression looks right with `log_multiplier = 112.3` and
  `log_floor = 1.0`. The output range will be much larger than with the old
  defaults (20.0 / 1e-19), so don't be surprised — that's the calibrated
  device-display scaling.
- The median clip thresholds (`-60`, `0`) and kernel size (5) are unchanged,
  so the median-clip-filter step should look the same as before.

If any of the above is wrong, fix Pass 1 plumbing before starting Pass 2.

---

## 4. Pass 2: ring-down injection + log-compression dynamic-range/reject

### 4.1 What the user wants

A ring-down stage that:

1. Reads the **measured waveform template** at
   `processing.ring_down.waveform_path` (a `.npy` of palette values along the
   A-line for the gain-54 reference).
2. Adds it to the simulated A-line **before envelope detection** (or its dB
   equivalent before log compression — see §4.4 for the choice).
3. Has an explicit **on/off** toggle. Off ⇒ no signal at all; on ⇒ the
   calibrated residual that survives the device's AR subtraction.
4. Honours the existing dynamic-range/reject knobs in the YAML
   (`dynamic_range_db = 40.6`, `reject_db = -40.6`) so the on-screen palette
   floor/ceiling matches the device.

Noise (`processing.noise.{type, sigma}`) is intentionally deferred until after
Pass 2 — see the writeup §12.4 for why (you measure noise on a
ring-down-subtracted simulator output, not before).

### 4.2 Schema additions you'll need

The current `RingDownConfig` (in `raysim/config.py`) has no `enabled` field —
the prior plan was to use one but it never landed because the C++ side wasn't
ready. Add it now so the schema and the C++ semantics agree:

```python
@dataclass
class RingDownConfig:
    enabled: bool = False                # NEW — Pass 2; false = truly no signal
    amplitude: float = 0.0
    extent_mm: float = 0.5
    decay: str = "exponential"           # exponential | hanning | measured
    waveform_path: Optional[str] = None
    subtract_reference: bool = True
```

Then add `ring_down.enabled = true` to
`instrument-calibration/p035_visions/volcano_s5i.yaml` so the calibrated
config keeps its current behaviour. Default `False` in the schema means
existing configs that omit the field stay silent (matches user choice 2a).

You'll also need to remove `processing.ring_down.*` and the two log-compression
display knobs (`processing.dynamic_range_db`, `processing.reject_db`) from
`_FUTURE_PATHS` in `raysim/config.py` once they're wired.

### 4.3 C++ surface area

**`include/raysim/core/raytracing_ultrasound_simulator.hpp` — extend `SimParams`:**

```cpp
// Pass 2 — ring-down injection. enabled=false means no ring-down signal at all.
struct RingDownParams {
  bool enabled = false;
  float amplitude = 0.f;          // envelope amplitude at simulator reference gain
  float extent_mm = 0.5f;         // hard cutoff in radial mm (waveform truncated past this)
  std::string decay = "exponential";  // "exponential" | "hanning" | "measured"
  std::vector<float> waveform;    // populated by host code from .npy when decay == "measured"
  // Note: subtract_reference is informational — the device already does it; we model the residual.
};

// add to SimParams:
RingDownParams ring_down;

// Pass 2 — log compression display window (dynamic range / reject palette).
// 0.f means "disabled" so existing default callers keep the current 0..log_multiplier*log10 mapping.
float dynamic_range_db = 0.f;
float reject_db = 0.f;
```

`waveform` lives on the **host** side as `std::vector<float>` so the bindings
can hand a numpy array straight into it; `simulate()` is responsible for
uploading it to a `CudaMemory` buffer the first time it runs (or whenever the
array changes — keep a hash or pointer-comparison cache like the TGC path
does).

**`csrc/core/raytracing_ultrasound_simulator.cpp` — add a stage between TGC and
envelope detection:**

```cpp
// 1.6 Ring-down injection (Pass 2)
if (sim_params.ring_down.enabled) {
  CudaTiming cuda_timing(sim_params.enable_cuda_timing, "Ring-down", sim_params.stream);
  // Build / re-upload the waveform on demand (cache by pointer/size).
  // Add it row-wise to d_scanlines: scanline[ray, depth] += amplitude * waveform[depth].
  // Truncate / zero past extent_mm (convert mm -> sample index using SAMPLING_FREQ and SoS).
}
```

The simplest CUDA kernel to add is a `add_row` analogue of `mul_row` (a
vector added element-wise to every row). Both `mul_row` and the existing
launchers in `cuda_algorithms.{cu,hpp}` are good templates; reuse the
infrastructure.

**Log-compression display window:** apply the dynamic-range / reject after the
existing `log_compression` call:

```cpp
if (sim_params.dynamic_range_db > 0.f) {
  // Map [reject_db, reject_db + dynamic_range_db] -> [0, log_multiplier * dynamic_range_db / 20].
  // I.e. clamp + offset on the post-log scanlines so the displayed palette matches the device.
}
```

This may simplify to a small fused kernel in `cuda_algorithms.cu` (clamp +
linear remap). Keep the default-off semantics: when both are zero in
`SimParams`, do nothing (the existing pipeline behaviour is unchanged).

### 4.4 Where in the pipeline does ring-down go?

The YAML's `amplitude = 46.37` is in **envelope amplitude** units at the
simulator's reference gain. Two reasonable injection points:

- **Pre-Hilbert (RF):** add the waveform to `d_scanlines` directly between
  TGC (1.5) and envelope detection (2). Most physical; the Hilbert + log
  compression naturally handle the resulting peak shape.
- **Post-envelope (pre-log):** add the waveform to the magnitude buffer
  between the Hilbert step and the log compression. Easier to reason about
  numerically (waveform is already an envelope), but skips the actual
  Hilbert response.

The user's calibrated template (`ringdown_template_g54_d60.npy`) is in
**palette units** along the A-line (see the comment block in
`volcano_s5i.yaml` around line 200 for the conversion: `amp = 10^(palette /
log_multiplier)`). The simulator should do the conversion when it loads the
template, so what flows through `simulate()` is already an envelope amplitude
ready to be added pre-log.

Recommendation: **start with the post-envelope path** (simpler, deterministic),
then switch to pre-Hilbert if the comparison frame doesn't match. The decision
isn't load-bearing for the API — it's an internal detail of the ring-down
stage.

### 4.5 Bindings

For each new field on `SimParams`:

- Add a `def_readwrite` entry in `csrc/python/raysim_bindings.cpp`.
- For `ring_down` (the nested struct), bind `RingDownParams` as its own pybind
  class (similar to how `TgcControlPoint` was bound in Pass 1) and expose
  `params.ring_down` as a `def_readwrite` of that struct.
- The `waveform` field should accept a `py::array_t<float>` that copies into
  the `std::vector<float>`; numpy lifetime is independent.

Re-export `RingDownParams` from `raysim/__init__.py` for parity with
`TgcControlPoint`.

### 4.6 `IvusSimConfig.to_sim_params()` updates

In `raysim/config.py`:

```python
# Ring-down: load waveform from .npy if decay == "measured" and the path is set.
rd = self.processing.ring_down
params.ring_down.enabled = bool(rd.enabled)
params.ring_down.amplitude = float(rd.amplitude)
params.ring_down.extent_mm = float(rd.extent_mm)
params.ring_down.decay = str(rd.decay)
if rd.decay == "measured" and rd.waveform_path:
    import numpy as np
    waveform_palette = np.load(_resolve_relative(rd.waveform_path))
    # convert palette -> envelope amplitude using the calibrated log_multiplier:
    waveform_amp = 10.0 ** (waveform_palette.astype(np.float32) / float(self.processing.log_multiplier))
    params.ring_down.waveform = waveform_amp  # numpy -> std::vector via pybind

params.dynamic_range_db = float(self.processing.dynamic_range_db)
params.reject_db = float(self.processing.reject_db)
```

`_resolve_relative()` is a helper you'll need: `RingDownConfig.waveform_path`
in the YAML is workspace-relative (e.g.
`P_035_PointScatter/derived/ringdown/...`). `IvusSimConfig.from_yaml(path)`
already knows the YAML's directory, so resolve relative to the workspace root
(parent of `instrument-calibration/`) for that one path. Search the existing
`config.py` for any existing path-resolution helper before adding a new one.

---

## 5. Acceptance test for Pass 2

After Pass 2 lands, run the calibrated YAML and visually compare the
simulator output to a real PV .035 frame from `P_035_PointScatter/`:

1. **`enabled=false` smoke test:** the inner ~3 mm should look like a quiet
   lumen with only TGC-shaped speckle. No catheter signature.
2. **`enabled=true` baseline:** the inner ~3 mm should show a peak palette
   value comparable to the real frames (peak at ~239 saturation depending on
   gain), decaying to the speckle floor by ~3 mm (matches `extent_mm` and
   `decay = measured`).
3. **`dynamic_range_db = 40.6`, `reject_db = -40.6`:** the reject palette
   should sit at 11 (matches the device-default reject). Anything below
   reject is clamped to 11; anything above the saturation point is clamped to
   239.

The `instrument-calibration/p035_visions/calibration_delta.md` file has the
real-frame statistics you can compare against, plus links to overview PNGs in
`P_035_PointScatter/derived/`.

---

## 6. Git state and conventions

When the prior agent ran, `Is directory a git repo: No` was reported by the
shell — the work happened in a working tree without an active git repo at the
workspace root. The actual repo is `i4h-sensor-simulation/`; check its git
status before doing anything:

```bash
cd i4h-sensor-simulation
git status
git diff main --stat                # everything Pass 1 touched should appear here
```

The prior agent did **not** commit anything. Suggested branch hygiene:

- `pass1-calibration-plumbing` — exactly the Pass 1 changes (six files listed
  in §2). Open as a PR and merge once the §3 validation passes.
- `pass2-ringdown` — Pass 2 work on top of Pass 1.

Conventions in this repo (observed from existing code, follow them):

- Header comments cite literature with PMC IDs / journal references where
  physics is involved.
- Headers use `#ifndef`-style include guards already; copy that style for any
  new header.
- The codebase uses `spdlog::info`/`spdlog::error`; use it for any new log
  lines in bindings or simulator init.

---

## 7. Documents you should read before starting

In rough priority order:

1. **`ultrasound-raytracing/docs/ivus_implementation_writeup.md`, §11** — the
   full Pass 1 writeup with diffs and field-by-field tables.
2. **`instrument-calibration/p035_visions/volcano_s5i.yaml`** — the YAML
   itself, especially the comment blocks around `log_multiplier` and
   `ring_down`. The block around line 95–145 has the calibration provenance
   you need to understand the magnitudes.
3. **`instrument-calibration/p035_visions/calibration_delta.md`** — what's
   measured vs. what's still pending; explains why `gain_db` and `noise.sigma`
   are deferred.
4. **`docs/ivus_calibration_protocol.md`** — the instrument-agnostic
   protocol; useful as background but not required for Pass 2.
5. **`P_035_PointScatter/derived/ringdown/ringdown_fit.json`** — the fitted
   ring-down parameters per (gain, diameter) cell. Useful for cross-checking
   `volcano_s5i.yaml` numbers and for understanding decay-shape choices.

Skip the per-instrument fitter scripts in `instrument-calibration/p035_visions/extract_*.py`
unless you need to regenerate the calibration; for Pass 2 you only need to
*consume* the YAML they already produced.

---

## 8. Quick checklist

- [ ] §3.1 — Pass 1 builds with CUDA/OptiX.
- [ ] §3.2 — Default-config example output is bit-identical to `main`.
- [ ] §3.3 — Calibrated YAML drives a successful run; TGC, log-compression,
      median-clip stages match the YAML.
- [ ] (if §3 passes) Commit Pass 1 on its own branch and open a PR.
- [ ] §4.2 — Add `enabled` to `RingDownConfig`; add `ring_down.enabled: true`
      to the calibrated YAML.
- [ ] §4.3 — Extend `SimParams` (`RingDownParams` nested struct,
      `dynamic_range_db`, `reject_db`); add an `add_row` CUDA helper and a
      ring-down injection stage between TGC and envelope detection (or
      post-envelope, see §4.4).
- [ ] §4.5 — Add bindings for the new types/fields and re-export.
- [ ] §4.6 — Update `IvusSimConfig.to_sim_params()` to load the waveform
      template, convert palette → envelope amplitude, and forward the new
      fields. Drop the relevant entries from `_FUTURE_PATHS`.
- [ ] §5 — Acceptance test: `enabled=false`/`enabled=true` against real
      frames; `dynamic_range_db` / `reject_db` reproduce reject palette = 11
      and saturation = 239.
- [ ] Commit Pass 2 and update `ivus_implementation_writeup.md` with a §11.7
      (or new section) covering the ring-down stage.

Pass 1 was deliberately invisible. Pass 2 is when the simulator actually
starts to look like a PV .035 — keep the comparisons frame-by-frame against
`P_035_PointScatter/` rather than against intuition.
