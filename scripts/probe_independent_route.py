"""Isolated native route-control probe; never edits a user's mission or reports a full pass."""
import argparse
import json
import shutil
import threading
import time
from pathlib import Path

from dronedream_agent_core.context import ContextStore
from dronedream_agent_core.execution import execute_prepared_mission
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.runtime_scheduling import configure_sensor_thread_handoff


# 功能：复用保留计划，在隔离目录检查独立控制链；限时通过正式中止通道安全落地。
# 输入：既有探测目录、安装资源、源码与新输出目录；输出：原始飞行证据，不覆盖用户数据。
def main():
    parser = argparse.ArgumentParser()
    for name in ("previous", "core", "runtime", "output", "camera-clock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tracking-seconds", type=float, default=90.)
    args = parser.parse_args()
    if not 10 <= args.tracking_seconds <= 300:
        parser.error("tracking duration must be 10..300 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    fusion = json.loads((args.previous / "fusion-inputs.json").read_text())
    spawn = json.loads((args.previous / "simulation/vehicle_spawn.json").read_text())
    world = Path(fusion["world_sdf"]).parent
    vehicle = Path(spawn["sdf_path"]).parent.parent
    plan = args.output / "prepared"
    shutil.copytree(args.previous / "prepared", plan)
    resources = args.output / "resources"
    resources.mkdir()
    for name in ("px4_offboard_track_executor.py", "px4_map_fusion_experiment_executor.py"):
        shutil.copy2(args.core / "runtime" / name, resources / name)
    for name in ("px4_checkpoint_executor.py", "runtime_depth_safety_worker.py"):
        shutil.copy2(args.core / "scripts" / name, resources / name)
    for name in ("payload-placement", "native-sensors"):
        shutil.copytree(args.runtime / name, resources / name)
    from dronedream_agent_core.simulation_camera_clock import validate_native_camera_clock, CAMERA_CLOCK_FILES
    validate_native_camera_clock(args.camera_clock, source_root=args.core / "native/camera_clock")
    (resources / "camera-clock").mkdir()
    for name in CAMERA_CLOCK_FILES:
        shutil.copy2(args.camera_clock / name, resources / "camera-clock" / name)
    configure_sensor_thread_handoff()
    stop = threading.Event()
    simulation = args.output / "simulation"

    # 功能：限制探测占用时间，不强杀飞控、不篡改完成状态。
    # 输入：实时阶段、单调时钟；输出：只创建本次运行的安全中止请求。
    def bound_probe():
        deadline, airborne = time.monotonic() + 600, None
        while not stop.wait(.5):
            try:
                state = json.loads((simulation / "runtime-phase.json").read_text())
                phase = state.get("enclosing_executor_state", state).get("phase")
            except (OSError, ValueError):
                phase = None
            if phase in ("TRACK", "TRACKING", "NAVIGATING", "CRUISE") and airborne is None:
                airborne = time.monotonic()
            if time.monotonic() > deadline or (airborne is not None and time.monotonic() - airborne > args.tracking_seconds):
                if simulation.exists():
                    abort = simulation / "live_abort.request.json"
                    if not abort.exists():
                        publish_runtime_json(abort, dict(reason="BOUNDED_ROUTE_PROBE_COMPLETE", world_paused=False), replace_existing=False)
                return
    monitor = threading.Thread(target=bound_probe, daemon=True)
    monitor.start()
    context = ContextStore(args.output / "context.sqlite3")
    try:
        prepared = json.loads((plan / "prepared-mission.json").read_text())
        result = execute_prepared_mission(
            prepared_path=plan / "prepared-mission.json",
            execution_authority_path=plan / "model-harness-execution-authority.json",
            confirm_contract_id=prepared["contract"]["contract_id"],
            run_dir=simulation, world_sdf=world / "world.sdf", semantic_path=world / "semantic.json",
            vehicle_sdf=vehicle / "gazebo/model.sdf", controller_params_path=vehicle / "controller_params.json",
            executor_path=resources / "px4_offboard_track_executor.py", px4_root=Path("/opt/PX4-Autopilot"),
            ros_workspace=Path("/home/dronedream/.local/share/dronedream-autonomy/v0.1.0/ros_ws-merged"),
            completion_provider="kimi", checkpoint_provider="kimi", runtime_interrupt_provider="kimi",
            context_store=context, checkpoint_executor_path=resources / "px4_checkpoint_executor.py",
            independent_route_control=True, simulation_map_fusion=True,
            heading_policy="route-tangent-relative", maximum_yaw_rate_deg_s=20.,
            map_graph_path=world.parent / "navigation-graph.json", vehicle_metadata_path=vehicle / "vehicle.json")
        print(json.dumps(dict(status=result.status, full_mission_accepted=False,
                              formal_training_additions=0)), flush=True)
    finally:
        stop.set()
        monitor.join(2)
        context.close()


if __name__ == "__main__":
    main()
