# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ultrasound simulation evaluation script (validation-oriented).

Runs test scenarios aligned with how ultrasound simulation models are validated
in the literature: geometric accuracy, depth-dependent attenuation, resolution
(point/line spread), contrast, and scan uniformity. The report cites common
validation practices (phantom comparison, reference solutions, image-quality
metrics) and maps each scenario to what to verify.

References: phantom-based geometric accuracy and resolution (e.g. tissue-mimicking
phantoms, IEC-style tests); comparison to analytical/reference solutions (e.g.
Field II / FOCUS vs. Rayleigh–Sommerfeld); speckle and PSF-based validation;
benchmark problems for intercomparison. See EVALUATION_REPORT.md for methodology
and references.

Requires the ultrasound-raytracing package built with IVUS probe support
(IVUSProbe and materials: lumen, vessel_wall, extravascular).

Usage:
    python examples/ivus_evaluation.py [--output-dir DIR] [--scenarios A,B,...]

Output:
    - Images in <output-dir>/ (default: ivus_evaluation_output/)
    - EVALUATION_REPORT.md with validation methodology, scenario mapping, and references.
"""

import argparse
import os
import sys

# Ensure both package root and examples dir are on path (run from repo root or examples/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, _script_dir)

# Non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import raysim.cuda as rs
from tqdm import tqdm

# Reuse IVUS example helpers
from ivus_example import (
    build_vessel_world,
    get_sim_params,
    save_unwrapped_frame,
    _mesh_path,
    MIN_VAL,
    MAX_VAL,
)

OUTPUT_DIR_DEFAULT = "ivus_evaluation_output"

# ---------------------------------------------------------------------------
# Probe helper (allows overriding frequency and position for tests)
# ---------------------------------------------------------------------------


def make_probe(position_mm, rotation_rad=None, frequency=40.0, **kwargs):
    """IVUS probe at the given position with optional parameter overrides."""
    if rotation_rad is None:
        rotation_rad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    defaults = dict(
        num_angular_rays=256,
        elevational_height=0.0,
        num_el_samples=1,
        f_num=1.0,
        speed_of_sound=1.54,
        pulse_duration=2.0,
    )
    defaults.update(kwargs)
    return rs.IVUSProbe(
        rs.Pose(position=position_mm, rotation=rotation_rad),
        num_angular_rays=defaults["num_angular_rays"],
        frequency=float(frequency),
        elevational_height=defaults["elevational_height"],
        num_el_samples=defaults["num_el_samples"],
        f_num=defaults["f_num"],
        speed_of_sound=defaults["speed_of_sound"],
        pulse_duration=defaults["pulse_duration"],
    )


# ---------------------------------------------------------------------------
# Scenario definitions: (name, description, run_func, expected_observations)
# ---------------------------------------------------------------------------

SCENARIO_METADATA = {}


def run_scenario_1_centered_single_wall(output_dir):
    """1. Centered probe, single-layer vessel wall (thin cylinder)."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=False)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    probe = make_probe(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b_mode = sim.simulate(probe, params)
    path = os.path.join(output_dir, "01_centered_single_wall.png")
    save_unwrapped_frame(
        b_mode, sim, path,
        title="Scenario 1: Geometric accuracy – single boundary (centered)",
    )
    return path


SCENARIO_METADATA["01_centered_single_wall"] = {
    "name": "Geometric accuracy: single boundary (centered)",
    "validation_type": "Geometric accuracy",
    "purpose": "Known geometry: circular boundary at fixed radius. Validates that reflector "
    "depth and angular position are correctly rendered (cf. phantom-based geometric "
    "accuracy in IEC/QA practice and simulation–phantom comparison).",
    "expected_if_accurate": (
        "Single bright interface at constant depth (~4 mm) over 360°. Weakly scattering "
        "interior (dark). Depth and angular symmetry match the phantom geometry; no "
        "systematic depth error or angular dropout."
    ),
    "what_to_verify": (
        "Ring at ~4 mm depth; symmetric in angle; dark interior; no axial streak artifacts "
        "or wrong depth scale."
    ),
}


def run_scenario_2_centered_thick_wall(output_dir):
    """2. Centered probe, thick vessel wall (two-layer cylinder)."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=True)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    probe = make_probe(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b_mode = sim.simulate(probe, params)
    path = os.path.join(output_dir, "02_centered_thick_wall.png")
    save_unwrapped_frame(
        b_mode, sim, path,
        title="Scenario 2: Attenuation with depth – two-layer boundary",
    )
    return path


SCENARIO_METADATA["02_centered_thick_wall"] = {
    "name": "Attenuation with depth & contrast (two-layer)",
    "validation_type": "Attenuation; contrast",
    "purpose": "Two interfaces at known depths test depth-dependent attenuation (deeper "
    "echoes should not be brighter than shallower ones) and contrast between layers "
    "(cf. contrast resolution and TMM phantoms in validation literature).",
    "expected_if_accurate": (
        "Two concentric bright interfaces at ~3.5 mm and ~4 mm. The deeper interface "
        "should be no brighter than the shallower one (attenuation with depth). "
        "Interior remains dark."
    ),
    "what_to_verify": (
        "Two distinct rings at correct depths; outer ring not brighter than inner; "
        "continuous in angle; plausible intensity ordering with depth."
    ),
}


def run_scenario_3_eccentric_probe(output_dir):
    """3. Eccentric probe: offset from vessel center."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=False)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    # Probe 1.2 mm off center along +x
    position = np.array([1.2, 0.0, 0.0], dtype=np.float32)
    probe = make_probe(position)
    b_mode = sim.simulate(probe, params)
    path = os.path.join(output_dir, "03_eccentric_probe.png")
    save_unwrapped_frame(
        b_mode, sim, path,
        title="Scenario 3: Geometric accuracy – eccentric transducer (offset +x 1.2 mm)",
    )
    return path


SCENARIO_METADATA["03_eccentric_probe"] = {
    "name": "Geometric accuracy: known offset (eccentric)",
    "validation_type": "Geometric accuracy",
    "purpose": "Transducer offset from symmetry axis: depth-of-interface must vary with "
    "angle in a predictable way (geometry). Validates that ray/geometry and "
    "coordinate mapping are correct (cf. geometric accuracy in phantom validation).",
    "expected_if_accurate": (
        "Interface depth varies with angle: minimum depth where probe is nearest the "
        "boundary (~2.8 mm), maximum on the opposite side (~5.2 mm). Closed ring; "
        "no artificial gaps or dropouts."
    ),
    "what_to_verify": (
        "Depth varies sinusoidally with angle; min ~2.8 mm, max ~5.2 mm; continuous "
        "interface; no spurious dropout or wrong depth scale."
    ),
}


def run_scenario_4_point_reflectors(output_dir):
    """4. Point reflectors: sphere phantom (ring + interior spheres)."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=False, use_thick_cylinder=False)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    probe = make_probe(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b_mode = sim.simulate(probe, params)
    path = os.path.join(output_dir, "04_point_reflectors.png")
    save_unwrapped_frame(
        b_mode, sim, path,
        title="Scenario 4: Resolution & geometry – point reflectors",
    )
    return path


SCENARIO_METADATA["04_point_reflectors"] = {
    "name": "Resolution & geometry: point reflectors",
    "validation_type": "Resolution (PSF); geometric accuracy",
    "purpose": "Discrete point-like reflectors at known positions: validate geometric "
    "placement (angle, depth) and effective resolution (spread of the echo). Standard "
    "in phantom validation (wire targets, point spread, IEC-style resolution tests).",
    "expected_if_accurate": (
        "Bright spots at known (angle, depth) corresponding to reflector positions "
        "(e.g. ~(1.5, 2) and ~(-2, 3) mm in imaging plane). Ring from boundary at ~4 mm. "
        "Limited spread of each spot indicates reasonable axial/lateral resolution."
    ),
    "what_to_verify": (
        "Reflector positions match known geometry; no ghost echoes or wrong depths; "
        "point-like appearance (not excessively blurred) indicates adequate resolution."
    ),
}


def run_scenario_5_pullback(output_dir, n_frames=5):
    """5. Pullback: multiple frames along vessel axis."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=False)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    z_positions = np.linspace(-1.0, 1.0, n_frames)
    paths = []
    for i, z in enumerate(tqdm(z_positions, desc="Pullback")):
        probe = make_probe(np.array([0.0, 0.0, z], dtype=np.float32))
        b_mode = sim.simulate(probe, params)
        path = os.path.join(output_dir, f"05_pullback_frame_{i:02d}_z{z:.2f}.png")
        save_unwrapped_frame(
            b_mode, sim, path,
            title=f"Scenario 5: Pullback z = {z:.2f} mm",
        )
        paths.append(path)
    return paths


SCENARIO_METADATA["05_pullback"] = {
    "name": "Temporal/positional consistency (pullback)",
    "validation_type": "Reproducibility; geometric consistency",
    "purpose": "Same phantom at different probe positions: cross-section geometry should "
    "remain consistent (reproducibility and absence of position-dependent artifacts). "
    "Common in multi-frame and 3D validation.",
    "expected_if_accurate": (
        "Each frame shows the same interface depth (~4 mm) and shape. No systematic "
        "drift or new artifacts at specific z; only end effects if phantom is finite."
    ),
    "what_to_verify": (
        "Interface depth and shape consistent across frames; no z-dependent artifacts "
        "or spurious bands."
    ),
}


def run_scenario_6_lower_frequency(output_dir):
    """6. Lower center frequency (20 MHz vs 40 MHz)."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=False)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    probe = make_probe(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        frequency=20.0,
    )
    b_mode = sim.simulate(probe, params)
    path = os.path.join(output_dir, "06_lower_frequency_20MHz.png")
    save_unwrapped_frame(
        b_mode, sim, path,
        title="Scenario 6: Frequency dependence – 20 MHz vs 40 MHz",
    )
    return path


SCENARIO_METADATA["06_lower_frequency"] = {
    "name": "Frequency dependence (20 vs 40 MHz)",
    "validation_type": "Physical consistency; resolution vs frequency",
    "purpose": "Same geometry at different center frequency: depth should be unchanged "
    "(geometry); texture/resolution may coarsen at lower frequency. Validates that "
    "frequency is applied consistently (attenuation, PSF) without breaking geometry.",
    "expected_if_accurate": (
        "Interface at same depth (~4 mm) as baseline. Coarser speckle at 20 MHz is "
        "plausible; no spurious depth shift or loss of interface."
    ),
    "what_to_verify": (
        "Depth unchanged vs scenario 1; optionally coarser texture at 20 MHz; no "
        "unexpected artifacts or depth error."
    ),
}


def run_scenario_7_save(output_dir):
    """7. Angular uniformity – save under distinct name for report."""
    materials = rs.Materials()
    world = build_vessel_world(materials, use_cylinder=True, use_thick_cylinder=False)
    sim = rs.RaytracingUltrasoundSimulator(world, materials)
    params = get_sim_params()
    probe = make_probe(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b_mode = sim.simulate(probe, params)
    path = os.path.join(output_dir, "07_angular_uniformity.png")
    save_unwrapped_frame(
        b_mode, sim, path,
        title="Scenario 7: Scan uniformity – angular (360°)",
    )
    return path


SCENARIO_METADATA["07_angular_uniformity"] = {
    "name": "Scan uniformity (angular)",
    "validation_type": "Image uniformity; artifact check",
    "purpose": "Uniform phantom: sensitivity should not show systematic bands aligned "
    "with the scan pattern (ray alignment, mesh aliasing). Standard uniformity check "
    "in QA and simulation validation.",
    "expected_if_accurate": (
        "Uniform response around 360° aside from real geometry. No fixed periodic "
        "bright/dark bands at scan angles (e.g. 0°, 90°) from sampling or alignment."
    ),
    "what_to_verify": (
        "No periodic bands aligned with angular sampling; ring intensity smooth "
        "around circumference aside from speckle."
    ),
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(output_dir, scenario_results):
    """Write EVALUATION_REPORT.md with scenario descriptions and image references."""
    path = os.path.join(output_dir, "EVALUATION_REPORT.md")
    # Build quick-reference table
    table_lines = [
        "| Scenario | Key image(s) |",
        "|----------|--------------|",
    ]
    for key in SCENARIO_METADATA:
        meta = SCENARIO_METADATA[key]
        if key in scenario_results and scenario_results[key] is not None:
            r = scenario_results[key]
            names = [os.path.basename(p) for p in (r if isinstance(r, list) else [r])]
            table_lines.append("| " + meta["name"][:40] + " | " + ", ".join(names[:5]) + (" ..." if len(names) > 5 else "") + " |")
        else:
            table_lines.append("| " + meta["name"][:40] + " | *(not run)* |")
    table_lines.append("")

    lines = [
        "# Ultrasound Simulation Evaluation Report",
        "",
        "This report describes test scenarios used to validate the ultrasound "
        "simulation. The scenarios are aligned with **validation practices reported "
        "in the literature**: geometric accuracy, attenuation with depth, resolution "
        "(point spread), contrast, scan uniformity, and reproducibility. For each "
        "scenario we state the validation aim, what to expect if the model is "
        "accurate, and what to verify in the generated images.",
        "",
        "## Quick reference",
        "",
    ] + table_lines + [
        "## Validation methodology (literature)",
        "",
        "Ultrasound simulation models are typically validated by:",
        "",
        "1. **Geometric accuracy** – Compare reflector positions (depth, lateral/angular "
        "position) in the image to known phantom geometry or to reference solutions. "
        "Tissue-mimicking phantoms with targets of known size/position are standard "
        "(e.g. gel wax, IEC-style phantoms; distance/diameter measurements).",
        "",
        "2. **Attenuation with depth** – Deeper echoes should not be brighter than "
        "shallower ones for the same reflectivity; intensity decay with depth is "
        "routinely checked in phantom and simulation studies.",
        "",
        "3. **Resolution (PSF)** – Point or line targets yield a point/line spread "
        "function; axial and lateral resolution (e.g. FWHM) are compared to "
        "measurements or to reference simulators (Field II, FOCUS, Rayleigh–Sommerfeld).",
        "",
        "4. **Contrast** – Regions with different acoustic properties should show "
        "plausible contrast in the image (e.g. hypoechoic spheres, layer boundaries); "
        "contrast-to-noise and similar metrics are used in IEC and QA standards.",
        "",
        "5. **Scan uniformity** – Sensitivity should not exhibit systematic bands "
        "aligned with the scan pattern; uniformity over the field is a standard QA check.",
        "",
        "6. **Reproducibility / consistency** – Same phantom at different positions "
        "or frames should give consistent geometry and no position-dependent artifacts.",
        "",
        "7. **Reference comparisons** – Where available, comparison to analytical "
        "solutions (e.g. Rayleigh–Sommerfeld), or to established simulators (Field II, "
        "FOCUS), or to experimental phantom data provides a strong validation.",
        "",
        "---",
        "",
    ]

    for key, meta in SCENARIO_METADATA.items():
        lines.append(f"## Scenario: {meta['name']}")
        lines.append("")
        if meta.get("validation_type"):
            lines.append("**Validation type:** " + meta["validation_type"])
            lines.append("")
        lines.append("**Purpose:** " + meta["purpose"])
        lines.append("")
        lines.append("**Expected if the simulation is accurate:**")
        lines.append("")
        lines.append(meta["expected_if_accurate"])
        lines.append("")
        lines.append("**What to verify in the output:**")
        lines.append("")
        lines.append(meta["what_to_verify"])
        lines.append("")
        if key in scenario_results:
            result = scenario_results[key]
            if isinstance(result, list):
                lines.append("**Generated images:**")
                for p in result:
                    name = os.path.basename(p)
                    lines.append(f"- `{name}`")
            else:
                name = os.path.basename(result)
                lines.append(f"**Generated image:** `{name}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "Use this report with the generated images to check geometric accuracy, "
        "depth-dependent attenuation, resolution/contrast, and scan uniformity. "
        "For quantitative validation, compare to phantom data, known scatterer "
        "positions (e.g. scenario 4), or to reference simulators where available."
    )
    lines.append("")
    lines.append("## References (validation methodology)")
    lines.append("")
    lines.append(
        "- Phantom-based geometric accuracy and resolution: tissue-mimicking phantoms, "
        "target dimensions vs. imaging (e.g. CT vs. US); IEC TS 62791, 62736 (sphere "
        "phantoms, contrast, resolution)."
    )
    lines.append(
        "- Reference solutions: Field II, FOCUS; comparison to Rayleigh–Sommerfeld "
        "integral for pressure/phase; FNM as reference (e.g. FOCUS validation pages)."
    )
    lines.append(
        "- Speckle and PSF: first/second-order statistics; PSF convolution for "
        "image formation; FWHM for resolution (e.g. computer speckle models, "
        "image-based PSF estimation)."
    )
    lines.append(
        "- Benchmark problems: standardized benchmarks for intercomparison (e.g. "
        "transcranial ultrasound benchmark: focus position/size, field metrics)."
    )
    lines.append(
        "- Ray-based validation: Simsonic and similar ray-based tools validated against "
        "experimental A-scans/B-scans; UltraRay/UltraScatter for full-path and "
        "scattering validation."
    )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main: run selected scenarios and write report
# ---------------------------------------------------------------------------

SCENARIO_RUNNERS = {
    "01_centered_single_wall": lambda d: run_scenario_1_centered_single_wall(d),
    "02_centered_thick_wall": lambda d: run_scenario_2_centered_thick_wall(d),
    "03_eccentric_probe": lambda d: run_scenario_3_eccentric_probe(d),
    "04_point_reflectors": lambda d: run_scenario_4_point_reflectors(d),
    "05_pullback": lambda d: run_scenario_5_pullback(d),
    "06_lower_frequency": lambda d: run_scenario_6_lower_frequency(d),
    "07_angular_uniformity": lambda d: run_scenario_7_save(d),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run IVUS simulation evaluation scenarios and generate a report.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory for images and report (default: {OUTPUT_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(SCENARIO_RUNNERS.keys()),
        help="Comma-separated scenario keys to run (default: all).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    keys = [k.strip() for k in args.scenarios.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SCENARIO_RUNNERS]
    if unknown:
        print("Unknown scenarios:", unknown)
        print("Available:", list(SCENARIO_RUNNERS.keys()))
        sys.exit(1)

    scenario_results = {}
    for key in keys:
        print(f"Running scenario: {key}")
        try:
            result = SCENARIO_RUNNERS[key](output_dir)
            scenario_results[key] = result
        except FileNotFoundError as e:
            print(f"  Skip {key}: {e}")
            scenario_results[key] = None
        except Exception as e:
            print(f"  Error in {key}: {e}")
            raise

    report_path = write_report(output_dir, scenario_results)
    print(f"Report written to {report_path}")
    print(f"Images and report are in {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
