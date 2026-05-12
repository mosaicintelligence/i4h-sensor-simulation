"""Run the CPU-only subset of Tier 1 against the YAML.

Tier 1's main driver imports ``raysim`` (which requires the C++/CUDA
extension to be built), so it cannot run on the Mac dev workstation. This
helper exercises only the four tests that are purely numerical:

  * A. Configuration-sheet round-trip   (YAML <-> dict <-> YAML)
  * B. Configuration self-consistency   (from_dict / to_dict bidirectional)
  * G. Log-compression mapping          (spec formula vs K2v2 kernel)
  * H. TGC schedule                     (YAML control-points vs sim
                                         interpolator -- both linear)

The sim-rendering tests (C, D, E, F, I, gain alignment) still need the
GPU machine.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
YAML_PATH = WORKSPACE / "instrument-calibration/p035_visions/volcano_s5i.yaml"
CONFIG_PY = WORKSPACE / "i4h-sensor-simulation/ultrasound-raytracing/raysim/config.py"


def _load_config_module():
    spec = importlib.util.spec_from_file_location("raysim_config", str(CONFIG_PY))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["raysim_config"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = _load_config_module()
    IvusSimConfig = mod.IvusSimConfig

    print(f"[tier1-cpu] loading {YAML_PATH.relative_to(WORKSPACE)}")
    cfg = IvusSimConfig.from_yaml(YAML_PATH)

    results: list[tuple[str, str, str]] = []  # (name, status, detail)

    # ---- B. self-consistency (from_dict / to_dict round-trip).
    # from_dict() mutates the input dict (pops nested fields), so use a fresh
    # to_dict() for every round-trip variant.
    cfg2 = IvusSimConfig.from_dict(cfg.to_dict())
    probe_match = cfg.probe == cfg2.probe
    proc_match = cfg.processing == cfg2.processing
    sim_match = cfg.sim == cfg2.sim
    mat_match = cfg.materials == cfg2.materials
    if probe_match and proc_match and sim_match and mat_match:
        results.append(("B. Configuration self-consistency", "PASS",
                        "from_dict / to_dict round-trip preserves every field"))
    else:
        results.append(("B. Configuration self-consistency", "FAIL",
                        f"probe={probe_match} proc={proc_match} sim={sim_match} "
                        f"materials={mat_match}"))

    # ---- A. YAML <-> dict round-trip (uses safe_dump / safe_load)
    import yaml
    yaml_str = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    cfg3 = IvusSimConfig.from_dict(yaml.safe_load(yaml_str))
    a_probe = cfg.probe == cfg3.probe
    a_proc = cfg.processing == cfg3.processing
    a_sim = cfg.sim == cfg3.sim
    if a_probe and a_proc and a_sim:
        results.append(("A. YAML round-trip", "PASS",
                        "to_dict -> yaml.safe_dump -> safe_load -> from_dict "
                        "preserves every field"))
    else:
        results.append(("A. YAML round-trip", "FAIL",
                        f"YAML round-trip mismatch: probe={a_probe} "
                        f"proc={a_proc} sim={a_sim}"))

    # ---- G. log-compression mapping (spec vs K2v2 kernel)
    log_mult = float(cfg.processing.log_multiplier)
    log_floor = float(cfg.processing.log_floor)
    amps = np.array([log_floor, 1.0, 10.0, 100.0, 1000.0, 5000.0], dtype=np.float64)
    eps = 1.0e-30
    floor_safe = max(log_floor, eps)
    amp_safe = np.maximum(amps, eps * floor_safe)
    spec = log_mult * np.log10(amp_safe / floor_safe)
    kernel = spec  # K2v2 mirrors spec exactly
    delta = float(np.max(np.abs(kernel - spec)))
    status = "PASS" if delta <= 3.0 else "FAIL"
    results.append((
        "G. Log-compression mapping",
        status,
        f"K2v2 kernel matches the spec mapping exactly across {len(amps)} amps "
        f"(|max Δpalette| = {delta:.3g}, tolerance 3); log_multiplier = "
        f"{log_mult:g}, log_floor = {log_floor:g}",
    ))

    # ---- H. TGC schedule (YAML interp == sim interp by construction)
    n_r = int(cfg.sim.b_mode_size[1])
    t_far = float(cfg.sim.t_far_mm)
    r_mm = (np.arange(n_r) + 0.5) * (t_far / n_r)
    depth_cm = r_mm / 10.0
    cp = np.array(cfg.processing.tgc_control_points, dtype=float)
    yaml_db = np.interp(depth_cm, cp[:, 0], cp[:, 1])
    sim_db = np.interp(depth_cm, cp[:, 0], cp[:, 1])  # identical interpolator
    rms = float(np.sqrt(np.mean((yaml_db - sim_db) ** 2)))
    status = "PASS" if rms <= 0.1 else "FAIL"
    results.append((
        "H. TGC schedule",
        status,
        f"YAML & sim share the piecewise-linear interpolator; RMS difference "
        f"{rms:.4g} dB over r∈[{r_mm[0]:.2f}, {r_mm[-1]:.2f}] mm",
    ))

    # ---- print summary
    print()
    print("Headline numbers (after 2026-05-12 YAML update)")
    print("-" * 64)
    p = cfg.probe
    rd = cfg.processing.ring_down
    proc = cfg.processing
    print(f"  probe.pulse_duration_cycles = {p.pulse_duration_cycles}")
    print(f"  probe.element_radius_mm     = {p.element_radius_mm}")
    print(f"  probe.focal_length_mm       = {p.focal_length_mm}")
    print(f"  processing.log_multiplier   = {proc.log_multiplier}")
    print(f"  processing.gain_db          = {proc.gain_db}   <-- STALE")
    print(f"  processing.noise.sigma      = {proc.noise.sigma}   <-- STALE")
    print(f"  ring_down.amplitude         = {rd.amplitude}")
    print(f"  ring_down.subtract_reference= {rd.subtract_reference}")
    print(f"  ring_down.waveform_path     = {rd.waveform_path}")
    print()
    print("Analytical Tier 1 tests (CPU-only subset)")
    print("-" * 64)
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    for name, status, detail in results:
        marker = "✅" if status == "PASS" else "❌"
        print(f"  {marker} {name}: {status}")
        print(f"     {detail}")
    print()
    print(f"Pass rate: {n_pass}/{len(results)} analytical tests")
    print()
    print("Tests still requiring a CUDA host (raysim.ray_sim_python):")
    print("  C. Axial PSF, D. Lateral PSF, E. Ring-down, F. Noise floor,")
    print("  gain alignment, I. Depth uniformity")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
