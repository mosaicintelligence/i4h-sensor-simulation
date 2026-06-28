#!/usr/bin/env bash
# Run an IVUS pullback on the GPU VM from mosaic_sim centerline/snakes inputs.
#
# Prereqs (on the VM): the ivus-demo image built, gcloud authed as you, and the
# centerline.txt + snakes.csv already uploaded to $BUCKET/pig_inputs/ (generated
# on your Mac with recording_to_demo_inputs.py).
#
#   bash mosaic_sim/run_pig_pullback.sh
#
# Override defaults via env, e.g.:  OUT_NAME=pig_cta_rf bash .../run_pig_pullback.sh
set -euo pipefail

BUCKET="${BUCKET:-gs://mosaic-research-ali-ivus}"
REPO="${REPO:-$HOME/i4h-sensor-simulation}"
LUMEN="${LUMEN:-$REPO/mosaic_sim/stls/pig_cta.stl}"
OUT_NAME="${OUT_NAME:-pig_cta_rf}"
RAYSIM_YAML="/opt/ivus_demo/instrument-calibration/p035_visions/volcano_s5i.yaml"

echo ">> fetching centerline/snakes from $BUCKET/pig_inputs/"
mkdir -p "$HOME/pig_inputs"
gcloud storage cp "$BUCKET/pig_inputs/*" "$HOME/pig_inputs/"

echo ">> ensuring the ivus container is running"
if ! sudo docker ps --format '{{.Names}}' | grep -qx ivus; then
  sudo docker rm -f ivus 2>/dev/null || true
  sudo docker run --rm -d --gpus all -p 8000:8000 --name ivus ivus-demo:latest
  sleep 8
fi

echo ">> staging lumen + inputs + config into the container"
sudo docker cp "$LUMEN" ivus:/tmp/pig_cta.stl
sudo docker cp "$HOME/pig_inputs/centerline.txt" ivus:/tmp/centerline.txt
sudo docker cp "$HOME/pig_inputs/snakes.csv" ivus:/tmp/snakes.csv

cat > /tmp/pig_pullback.yaml <<YAML
paths:
  patient_root: /tmp
  lumen_stl: /tmp/pig_cta.stl
  centerline_txt: /tmp/centerline.txt
  snakes_csv: /tmp/snakes.csv
  raysim_yaml: ${RAYSIM_YAML}
  output_root: /opt/ivus_demo/outputs/${OUT_NAME}
wall_model: {thickness_scale_of_radius: 0.08, min_thickness_mm: 1.0, max_thickness_mm: 3.0, smooth_iterations: 2}
trajectory: {step_mm: 0.8, max_frames: 500}
simulation:
  world_background_material: lumen
  lumen_mesh_material: vessel_wall
  outer_mesh_material: extravascular
  material_overrides:
    lumen: {mu0: 0.22, mu1: 0.16, sigma: 0.24}
    blood: {mu0: 0.22, mu1: 0.16, sigma: 0.24}
  show_progress: true
export:
  fps: 30
  dicom_filename: simulated_ivus_pullback.dcm
  mp4_filename: simulated_ivus_pullback_polar.mp4
  camera_eye_mp4_filename: simulated_ivus_pullback_camera_eye.mp4
  gods_eye_mp4_filename: simulated_ivus_pullback_gods_eye.mp4
  npy_filename: polar_stack.npy
  pose_csv_filename: pullback_poses.csv
  pose_json_filename: pullback_poses.json
  geometry_manifest_filename: geometry_manifest.json
  run_report_filename: run_report.md
dicom: {patient_name: PIG^CTA, patient_id: PIG_CTA_SIM, study_description: Mosaic catheter IVUS, series_description: ${OUT_NAME}, manufacturer: I4H, model_name: MOSAIC_CATHETER, gain_slider_ref: 54}
YAML
sudo docker cp /tmp/pig_pullback.yaml ivus:/tmp/pig_pullback.yaml

echo ">> running the pullback (raytraces all frames + exports MP4/DICOM)"
sudo docker exec ivus python /opt/ivus_demo/pullback_src/pipeline.py --config /tmp/pig_pullback.yaml

echo ">> copying results out and uploading MP4s"
rm -rf "$HOME/$OUT_NAME"
sudo docker cp "ivus:/opt/ivus_demo/outputs/${OUT_NAME}" "$HOME/$OUT_NAME"
gcloud storage cp "$HOME/$OUT_NAME/exports/"*.mp4 "$BUCKET/$OUT_NAME/"

echo ">> done -> $BUCKET/$OUT_NAME/  (open simulated_ivus_pullback_polar.mp4 in the Cloud Console)"
