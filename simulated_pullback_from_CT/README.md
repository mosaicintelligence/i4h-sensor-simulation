# simulated_pullback_from_CT

Standalone PAT23 CTA-to-IVUS pullback pipeline.

This project creates a simulated IVUS pullback from:

- lumen mesh: `patient_23/mesh/aorta.stl`
- centerline: `patient_23/centerlines/Aorta_Extended.txt`
- radius metadata: `patient_23/snakes/Aorta_Extended-snakes.csv`

It treats `ultrasound-raytracing` (`raysim`) as a dependency and does not modify it.

## Outputs

Default output root: `simulated_pullback_from_CT/outputs/patient_23`

- generated meshes: lumen + outer wall OBJ/STL + geometry manifest
- pullback poses CSV/JSON
- simulated frame stack (`polar_stack.npy`)
- multi-frame DICOM (`simulated_ivus_pullback.dcm`)
- polar-orientation IVUS MP4 (`simulated_ivus_pullback_polar.mp4`)
- camera-eye MP4 (`simulated_ivus_pullback_camera_eye.mp4`)
- god's-eye probe-trace MP4 (`simulated_ivus_pullback_gods_eye.mp4`)
- run report (`run_report.md`)

## Environment

Recommended Python: 3.10+

Install dependencies from workspace root:

```bash
pip install -e ./i4h-sensor-simulation/ultrasound-raytracing
pip install -r ./simulated_pullback_from_CT/requirements.txt
```

## Run

From workspace root:

```bash
python simulated_pullback_from_CT/src/pipeline.py \
  --config simulated_pullback_from_CT/configs/patient23_pullback.yaml
```

## Notes

- Geometry offsets assume millimeter units (consistent with patient assets and raysim).
- Wall generation uses a geometric normal-offset model (configurable thickness scaling and clamps).
- DICOM export writes a derived multi-frame US object with key tags and provenance sidecar JSON.
