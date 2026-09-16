#!/usr/bin/env bash

set -euo pipefail

world_path="${1:?usage: verify_gazebo_payload_mount.sh WORLD_SDF VEHICLE_SDF PAYLOAD_SDF EXECUTOR OUTPUT_DIR}"
vehicle_path="${2:?usage: verify_gazebo_payload_mount.sh WORLD_SDF VEHICLE_SDF PAYLOAD_SDF EXECUTOR OUTPUT_DIR}"
payload_path="${3:?usage: verify_gazebo_payload_mount.sh WORLD_SDF VEHICLE_SDF PAYLOAD_SDF EXECUTOR OUTPUT_DIR}"
executor_path="${4:?usage: verify_gazebo_payload_mount.sh WORLD_SDF VEHICLE_SDF PAYLOAD_SDF EXECUTOR OUTPUT_DIR}"
output_dir="${5:?usage: verify_gazebo_payload_mount.sh WORLD_SDF VEHICLE_SDF PAYLOAD_SDF EXECUTOR OUTPUT_DIR}"
python_binary="${DRONEDREAM_RUNTIME_PYTHON:-/home/dronedream/.local/share/dronedream-autonomy/v0.1.0/venv/bin/python}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for required_path in "$world_path" "$vehicle_path" "$payload_path" "$executor_path" "$python_binary"; do
  if [[ ! -f "$required_path" ]]; then
    printf 'Required payload-mount acceptance input is missing: %s\n' "$required_path" >&2
    exit 40
  fi
done
mkdir -p "$output_dir"

export GZ_CONFIG_PATH="/usr/share/gz:${GZ_CONFIG_PATH:-}"
export GZ_SIM_RESOURCE_PATH="/opt/PX4-Autopilot/Tools/simulation/gz/models:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_PARTITION="dronedream_payload_mount_${$}"

terminate_group() {
  local process_group="$1"
  kill -TERM -- "-$process_group" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -g "$process_group" >/dev/null 2>&1; then return; fi
    sleep 0.1
  done
  kill -KILL -- "-$process_group" 2>/dev/null || true
}

cleanup() {
  if [[ -n "${gz_pid:-}" ]]; then terminate_group "$gz_pid"; fi
}
trap cleanup EXIT

setsid /usr/bin/gz sim -r -s -v 2 "$world_path" >"$output_dir/gazebo.log" 2>&1 &
gz_pid=$!
for _ in $(seq 1 60); do
  if /usr/bin/gz topic -l 2>/dev/null | grep -qx '/clock'; then break; fi
  if ! kill -0 "$gz_pid" 2>/dev/null; then exit 41; fi
  sleep 1
done

spawn_entity() {
  local sdf_path="$1"
  local model_name="$2"
  local z_m="$3"
  local request
  request="sdf_filename: \"$sdf_path\" name: \"$model_name\" pose { position { x: 48.5 y: 1.25 z: $z_m } } allow_renaming: false"
  /usr/bin/gz service -s /world/school_map_world/create \
    --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
    --timeout 5000 --req "$request" >"$output_dir/${model_name}-spawn.txt"
  grep -q 'data: true' "$output_dir/${model_name}-spawn.txt"
}

spawn_entity "$payload_path" takeout_payload 1.242
spawn_entity "$vehicle_path" my_drone 1.1
sleep 1

timeout 10 /usr/bin/gz topic -e -n 1 \
  -t /model/my_drone/takeout_payload/state \
  >"$output_dir/preflight-detach-state.txt" &
detach_listener_pid=$!
sleep 0.2
/usr/bin/gz topic -t /model/my_drone/takeout_payload/detach \
  -m gz.msgs.Empty -p ''
wait "$detach_listener_pid"
grep -Eq 'data: (true|"detached")' "$output_dir/preflight-detach-state.txt"
sleep 1

PX4_GAZEBO_WORLD_NAME=school_map_world "$python_binary" \
  "$script_root/verify_gazebo_payload_mount.py" \
  "$executor_path" >"$output_dir/acceptance.json"
grep -q '"accepted": true' "$output_dir/acceptance.json"
printf 'GAZEBO_PAYLOAD_MOUNT_ACCEPTED output=%s\n' "$output_dir/acceptance.json"
