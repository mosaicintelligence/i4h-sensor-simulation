# IVUS simulation monorepo

Mosaic’s IVUS simulation stack: a GPU OptiX raysim, PV .035 / Volcano s5i
calibration, procedural vessel generation, and demo / CTA-pullback helpers.
Active IVUS work lives on the `ivus-probe` branch; GitHub `main` may still
look like upstream NVIDIA i4h until that branch is merged.

The GitHub clone directory is typically `i4h-sensor-simulation/`; a local
checkout may be renamed (e.g. `ivus-sim/`). Paths below are relative to the
monorepo root either way.

Raw bench DICOMs under `ivus_test_*` are intentionally **not** in git (out of
band). Small manifests, `derived/` summaries, YAML, and templates may still
be tracked.

## Layout

```
.
├── i4h-sensor-simulation/ultrasound-raytracing/  # OptiX IVUS simulator (raysim)
├── i4h-sensor-simulation/docs/                   # Physics primer, technical guide, tutorial
├── instrument-calibration/                       # Probe YAML fitting + Tier 1 gates
├── vessel-generator/                             # Procedural anatomy + paired B-mode datasets
├── IVUS_demo/                                    # Live pose → frame Docker demo
└── simulated_pullback_from_CT/                   # CTA lumen → IVUS pullback pipeline
```

## Start here by use case

| Want to… | Go to |
|---|---|
| Render a first IVUS frame / install the CUDA sim | [ultrasound-raytracing quick start](i4h-sensor-simulation/ultrasound-raytracing/docs/quick_start.md) |
| Understand IVUS raysim physics, pipeline, and limits | [IVUS implementation writeup](i4h-sensor-simulation/ultrasound-raytracing/docs/ivus_implementation_writeup.md) · [technical guide](i4h-sensor-simulation/docs/ultrasound_simulator_technical_guide.md) |
| Calibrate / onboard a new catheter + console | [Probe onboarding protocol](instrument-calibration/docs/probe_onboarding_protocol.md) |
| PV .035 shipping config and calibration status | [volcano_s5i.yaml](instrument-calibration/p035_visions/volcano_s5i.yaml) · [elevational waiver](instrument-calibration/p035_visions/tier1_elevational_waiver.md) · [Tier 1 results](instrument-calibration/p035_visions/tier1_results/tier1_results.md) |
| Generate training vessels / paired B-mode | [vessel-generator quick start](vessel-generator/docs/quickstart.md) |
| Live UI / demo service | [IVUS_demo README](IVUS_demo/README.md) |
| CTA pullback | [simulated_pullback_from_CT README](simulated_pullback_from_CT/README.md) |

Package-local READMEs under each folder have install details and deeper links.
