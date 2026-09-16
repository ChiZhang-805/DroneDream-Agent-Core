#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: verify_gazebo_semantic_camera.sh PX4_ROOT EVIDENCE_ROOT" >&2
  exit 64
fi

px4_root=$(realpath "$1")
evidence_root=$(realpath "$2")
world_sdf="$evidence_root/school-map/world.vision-training.sdf"
vehicle_sdf="$evidence_root/my-drone/model.vision-training.sdf"
for required in "$world_sdf" "$vehicle_sdf"; do
  if [[ ! -f "$required" ]]; then
    echo "required semantic-camera artifact is missing: $required" >&2
    exit 66
  fi
done

partition="dronedream_vision_semantic_$$"
export GZ_PARTITION="$partition"
export GZ_IP=127.0.0.1
export HEADLESS=1
export GZ_SIM_RESOURCE_PATH="$evidence_root/school-map:$evidence_root:$px4_root/Tools/simulation/gz/models:$px4_root/Tools/simulation/gz/worlds"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$px4_root/build/px4_sitl_default/src/modules/simulation/gz_plugins"
export GZ_SIM_SERVER_CONFIG_PATH="$px4_root/src/modules/simulation/gz_bridge/server.config"

gazebo_log="$evidence_root/gazebo-vision.log"
setsid gz sim -r -s "$world_sdf" >"$gazebo_log" 2>&1 &
gazebo_pid=$!
cleanup() {
  kill -TERM -- "-$gazebo_pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 -- "-$gazebo_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  kill -KILL -- "-$gazebo_pid" 2>/dev/null || true
  wait "$gazebo_pid" 2>/dev/null || true
}
trap cleanup EXIT

world_ready=false
for _ in $(seq 1 60); do
  if gz service -i --service /world/school_map_world/scene/info >/dev/null 2>&1; then
    world_ready=true
    break
  fi
  sleep 0.5
done
if [[ "$world_ready" != true ]]; then
  echo "semantic-camera Gazebo world did not become ready" >&2
  tail -100 "$gazebo_log" >&2
  exit 70
fi

request="sdf_filename: \"$vehicle_sdf\" name: \"my_drone\" pose { position { x: -42.25 y: 15.3 z: 7.487 } } allow_renaming: false"
create_output=$(gz service \
  -s /world/school_map_world/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 10000 \
  --req "$request")
printf '%s\n' "$create_output" >"$evidence_root/vehicle-create.txt"
if [[ "$create_output" != *"data: true"* ]]; then
  echo "Gazebo rejected the semantic-camera vehicle" >&2
  tail -120 "$gazebo_log" >&2
  exit 71
fi

label_topic=/dronedream/vision/semantic/labels_map
topic_ready=false
for _ in $(seq 1 90); do
  if gz topic -l | grep -Fx "$label_topic" >/dev/null; then
    topic_ready=true
    break
  fi
  sleep 0.5
done
gz topic -l | grep -E 'semantic|IMX214|depth_camera' \
  >"$evidence_root/sensor-topics.txt" || true
if [[ "$topic_ready" != true ]]; then
  echo "semantic label topic was not advertised" >&2
  tail -160 "$gazebo_log" >&2
  exit 72
fi

timeout 20 gz topic -e -t "$label_topic" -n 1 \
  >"$evidence_root/semantic-label-sample.txt"
if [[ ! -s "$evidence_root/semantic-label-sample.txt" ]]; then
  echo "semantic label topic did not publish a sample" >&2
  exit 73
fi

sample_bytes=$(wc -c <"$evidence_root/semantic-label-sample.txt")
printf 'semantic_topic=%s\nsemantic_sample_bytes=%s\n' \
  "$label_topic" "$sample_bytes" \
  >"$evidence_root/semantic-camera-verification.txt"
cat "$evidence_root/semantic-camera-verification.txt"
