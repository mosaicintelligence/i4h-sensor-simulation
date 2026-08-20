# Ultrasound raytracing (package docs)

This directory is the **OptiX ultrasound / IVUS simulator** package inside
the Mosaic IVUS simulation monorepo. For monorepo orientation (calibration,
vessel-generator, demos), start at the [root README](../README.md).

A high-performance GPU-accelerated ultrasound simulator using NVIDIA OptiX
raytracing with Python bindings. Supports curvilinear, linear-array,
phased-array, and IVUS probes; ships with the PV .035 / Volcano s5i YAML
(bench-derived processing params; elevational aperture is still an
uncalibrated 2.5D product default — see
[`tier1_elevational_waiver.md`](../instrument-calibration/p035_visions/tier1_elevational_waiver.md)).

## Components in this tree

### Ultrasound Raytracing Simulator

![image](./docs/ultrasound-raytracing.png)

Key features:
- GPU acceleration with CUDA and NVIDIA OptiX
- Python interface for ease of use
- Real-time simulation capabilities

[Simulator README / install](./ultrasound-raytracing/README.md)

Sibling packages (same monorepo root):
- [Vessel Generator](../vessel-generator/README.md) — procedural anatomy + paired IVUS datasets
- [Instrument calibration](../instrument-calibration/README.md) — probe YAML fitting

## Repository Structure

```
i4h-sensor-simulation/
├── docs/                      # Cross-component documentation
│   ├── ultrasound_simulator_foundations.md    # Physics primer
│   ├── ultrasound_simulator_technical_guide.md# Pipeline + physics deep dive
│   ├── ultrasound_simulator_tutorial.md       # Hands-on tutorial (7+ examples)
│   └── simulation_acceptance_criteria.md      # Porcine-lab acceptance gates
├── ultrasound-raytracing/     # Ultrasound raytracing simulator
    ├── .devcontainer/         # Development container configuration
    ├── cmake/                 # CMake build configuration
    ├── configs/               # Reference simulator configurations
    ├── csrc/                  # C++/CUDA source code
    ├── docs/                  # Simulator-local docs (quick_start, vm_setup, docker_build, ivus writeup)
    ├── examples/              # Usage examples and demos
    ├── include/               # C++ header files
    ├── mesh/                  # Bundled phantom meshes (cylinder, etc.)
    ├── raysim/                # Python package and bindings
    └── utils/                 # Utility scripts (phantom maker, etc.)
```

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Driver 555+
- CUDA 12.6+
- CMake 3.24+
- Python 3.10+

See individual component READMEs for specific requirements.

## Getting Started

Work from the **monorepo root** (the parent of this `i4h-sensor-simulation/`
folder). Clone the Mosaic repo / check out `ivus-probe`, then:

1. **Just want a rendered frame?** → [Quick Start](./ultrasound-raytracing/docs/quick_start.md)
2. **Generate realistic IVUS training data?** → [Vessel Generator Quick Start](../vessel-generator/docs/quickstart.md)
3. **Setting up a fresh cloud GPU?** → [VM Setup (GCP)](./ultrasound-raytracing/docs/vm_setup.md)
4. **Installing on a machine you already own?** → [Simulator README](./ultrasound-raytracing/README.md)
5. **Want to learn the simulator step by step?** → [Tutorial](./docs/ultrasound_simulator_tutorial.md)
6. **Curious about the physics?** → [Technical Guide](./docs/ultrasound_simulator_technical_guide.md)
7. **Calibrate a probe?** → [Probe onboarding](../instrument-calibration/docs/probe_onboarding_protocol.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Security

Please see our [Security Policy](./SECURITY.md) for information on reporting security vulnerabilities.

## Support

For questions and support, please open an issue in the GitHub repository.
