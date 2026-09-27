#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WORK_ROOT RUN_DIR" >&2
  exit 64
fi

runtime_root="$HOME/.local/share/dronedream-autonomy/v0.1.0"
set +u
source /opt/ros/jazzy/setup.bash
source "$runtime_root/ros_ws-merged/install/setup.bash"
set -u
export ROS_DOMAIN_ID=74
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS2_DISABLE_DAEMON=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/q/DroneDream-Workspace/Agent-Core/DroneDream-Agent-Core/src:${PYTHONPATH:-}

exec "$runtime_root/venv/bin/python" \
  -m dronedream_agent_core.asset_qualification_cli \
  --work-root "$1" \
  --run-dir "$2" \
  --px4-root /opt/PX4-Autopilot \
  --executor /mnt/q/DroneDream-Workspace/Agent-Core/DroneDream-Agent-Core/scripts/px4_checkpoint_executor.py \
  --base-executor /mnt/q/DroneDream-Workspace/Agent-Core/DroneDream-Agent-Core/runtime/px4_offboard_track_executor.py \
  --ros-workspace "$runtime_root/ros_ws-merged" \
  --simulation-ground-truth-control
