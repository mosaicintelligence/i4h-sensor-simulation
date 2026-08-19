# Tier 1 waiver — elevational 1.5 mm / 8-plane default

**Product choice:** `volcano_s5i.yaml` ships `elevational_height_mm: 1.5` and
`num_elevational_samples: 8` as an explicit 2.5D operating point, not as an
E3-calibrated slice thickness. We are **not** reverting to 0.0 / 1 until E3.

**Tier 1 has not been re-run** against this default in this change. The
checked-in `tier1_results/` still describe the previous 2D (`N = 1`) catalog.
Until a full `tier1_evaluation.py` pass is recorded, treat the following
parameter-bank rows as **expected to move** (waiver, not an automatic FAIL of
the new catalog):

| Test | Why it may move | Gate until re-run |
|---|---|---|
| **M. Speckle** | Each plane is an independent speckle draw (`scatter_angular_decorrelate`); `mean_planes` plus the `sqrt(N)` shim keep RMS near the 2D calibration but the *realization* and radial autocorrelation can still shift. | Waiver — re-derive `radial_corr_mm` / CoV thresholds if they miss after a 2.5D re-run. |
| **I. Depth uniformity** | Same speckle-field change; milk bath statistics are realization-dependent. | Waiver — shape (span ratio) is the gate; magnitude may drift. |
| **C / D. Axial / lateral PSF** | Coherent wire echoes are y-invariant in the B2 phantom, so FWHM should stay close. Off-plane energy in a finite elevational aperture can slightly broaden the apparent PSF. | Waiver only if FWHM moves outside existing tol; expected to still PASS. |
| **E. Ring-down** | Near-field catheter artifact is not an elevational scatter average; small changes possible from the extra ray planes. | Waiver only if peak depth / extent miss; expected to still PASS. |
| **F. Noise floor** | YAML `noise.sigma` is 0; pre-PSF RF noise is a no-op. Envelope-noise path unchanged. | Not expected to move. |
| **A / B. Config round-trip / self-consistency** | YAML field values changed; round-trip must accept 1.5 / 8. | Must PASS (schema, not bench). |
| **G / H. Log compression / TGC** | Processing chain is 2D after collapse; independent of N except via RF statistics. | Not expected to move. |

**Not waived:** CUDA illegal-address on `N > 1` (regression covered by
`ultrasound-raytracing/tests/test_elevational_collapse.py`).

**Closes the waiver:** land a 2.5D `tier1_results/` refresh, or replace 1.5 mm
with an E3-measured FWHM and re-gate M / I.
