# Simulator configs

This folder is for **simulator-side** runtime configurations (sample
configs, templates, defaults).

The per-instrument calibrated YAML files are **outputs** of the
calibration pipelines that live alongside this repo, not in this folder.

## Where the per-instrument configs live

| Instrument | Canonical YAML |
|------------|----------------|
| Visions PV .035 / Volcano s5i (10 MHz) | `../../../instrument-calibration/p035_visions/volcano_s5i.yaml` |

Point the simulator at one of those files when you run it, e.g.

```bash
cd /path/to/ivus-sim
python3 -m raysim ... --config instrument-calibration/p035_visions/volcano_s5i.yaml
```

The reason these YAMLs are not committed inside the simulator repo is
that they are *generated* by an instrument-specific calibration pipeline
and rewritten every time fresh bench data arrives. Keeping the
generator and its output together (in `instrument-calibration/<id>/`)
keeps the source-of-truth obvious.
