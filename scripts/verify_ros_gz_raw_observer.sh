#!/usr/bin/env bash

set -eo pipefail
source /opt/ros/jazzy/setup.bash
ros_workspace_setup="${DRONEDREAM_ROS_WS_SETUP:-/home/dronedream/.local/share/dronedream-autonomy/v0.1.0/ros_ws-merged/install/setup.bash}"
if [[ ! -f "$ros_workspace_setup" ]]; then
  printf 'DroneDream ROS workspace setup is missing: %s\n' "$ros_workspace_setup" >&2
  exit 39
fi
source "$ros_workspace_setup"
set -u

world_path="${1:?usage: verify_ros_gz_raw_observer.sh WORLD_SDF VEHICLE_SDF [OUTPUT_DIR]}"
vehicle_path="${2:?usage: verify_ros_gz_raw_observer.sh WORLD_SDF VEHICLE_SDF [OUTPUT_DIR]}"
output_dir="${3:-/tmp/dronedream-raw-observer}"
if [[ ! -f "$world_path" || ! -f "$vehicle_path" ]]; then
  printf 'Qualified world or vehicle file is missing\n' >&2
  exit 40
fi
mkdir -p "$output_dir"

export GZ_CONFIG_PATH="/usr/share/gz:${GZ_CONFIG_PATH:-}"
export GZ_SIM_RESOURCE_PATH="/opt/PX4-Autopilot/Tools/simulation/gz/models:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_PARTITION="dronedream_raw_observer_${$}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export ROS2_DISABLE_DAEMON=1
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface address="127.0.0.1"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'

terminate_group() {
  local process_group="$1"
  kill -TERM -- "-$process_group" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! pgrep -g "$process_group" >/dev/null 2>&1; then return; fi
    sleep 0.1
  done
  kill -KILL -- "-$process_group" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! pgrep -g "$process_group" >/dev/null 2>&1; then return; fi
    sleep 0.1
  done
  printf 'Process group %s survived cleanup\n' "$process_group" >&2
}

cleanup() {
  for process_id in "${observer_pid:-}" "${bridge_pid:-}" "${gz_pid:-}"; do
    if [[ -n "$process_id" ]]; then terminate_group "$process_id"; fi
  done
}
trap cleanup EXIT

setsid /usr/bin/gz sim -r -s -v 2 "$world_path" >"$output_dir/gazebo.log" 2>&1 &
gz_pid=$!
for _ in $(seq 1 60); do
  if /usr/bin/gz topic -l 2>/dev/null | grep -qx '/clock'; then break; fi
  if ! kill -0 "$gz_pid" 2>/dev/null; then exit 41; fi
  sleep 1
done

spawn_request="sdf_filename: \"$vehicle_path\" name: \"my_drone\" pose { position { x: -42.25 y: 15.3 z: 7.487 } } allow_renaming: false"
if ! /usr/bin/gz service -s /world/school_map_world/create \
  --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
  --timeout 5000 --req "$spawn_request" \
  >"$output_dir/vehicle-spawn.txt" 2>"$output_dir/vehicle-spawn.log"; then
  cat "$output_dir/vehicle-spawn.log" >&2
  exit 44
fi
grep -q 'data: true' "$output_dir/vehicle-spawn.txt"
if ! timeout 20 /usr/bin/gz topic -e -n 1 \
  -t /world/school_map_world/dynamic_pose/info \
  >"$output_dir/gazebo-dynamic-pose.txt"; then
  printf 'Gazebo did not publish a dynamic vehicle pose\n' >&2
  exit 45
fi
grep -q 'name: "my_drone"' "$output_dir/gazebo-dynamic-pose.txt"

setsid ros2 run ros_gz_bridge parameter_bridge \
  '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' \
  >"$output_dir/clock-bridge.log" 2>&1 &
bridge_pid=$!

setsid ros2 run dronedream_agent_ros gazebo_pose_observer --ros-args \
  -p use_sim_time:=true \
  -p entity_name:=my_drone \
  -p gazebo_pose_topic:=/world/school_map_world/dynamic_pose/info \
  -p contract_id:=acceptance-contract \
  -p segment_id:=acceptance-segment \
  >"$output_dir/observer.log" 2>&1 &
observer_pid=$!
sleep 3

if ! kill -0 "$observer_pid" 2>/dev/null; then
  tail -100 "$output_dir/observer.log" >&2
  exit 42
fi

if ! timeout 30 ros2 topic echo --no-daemon --spin-time 2 --once \
  /dronedream/mission_observation dronedream_agent_msgs/msg/MissionObservation \
  >"$output_dir/mission-observation.yaml"; then
  tail -100 "$output_dir/observer.log" >&2
  exit 43
fi

grep -q 'source_topic: /world/school_map_world/dynamic_pose/info' \
  "$output_dir/mission-observation.yaml"
grep -q 'frame_id: map_enu' "$output_dir/mission-observation.yaml"
printf 'ROS_GZ_RAW_OBSERVER_ACCEPTED output=%s\n' "$output_dir/mission-observation.yaml"
