"""Capture moving Gazebo camera geometry; no PX4, actuators, or policy authority.

Uses an isolated transport partition and only terminates its own simulator.
All raw pixels, source timestamps, commands and independent measured poses stay
in a new evidence directory. A service acknowledgement alone cannot pass it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dronedream_agent_core.geometry_fixture_capture import FixtureCapture
from dronedream_agent_core.geometry_fixture_transport import FixturePoseRequests
from dronedream_agent_core.geometry_motion_fixture import (
    DEPTH_TOPIC,
    POSE_TOPIC,
    RIG_NAME,
    build_fixture_world,
    bytes_digest,
    camera_calibration,
    fixture_pose,
    fixture_sensor_mount,
)
from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_agent_core.simulation_graphics_lifetime import prepare_render_process_environment
from dronedream_agent_core.simulation_sensor_frames import simulation_pose_time_ns
from dronedream_agent_core.static_render_batching import prepare_static_render_world
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   将内部报告转换并严格验证为有限 JSON 后独占写入，拒绝覆盖旧证据或留下序列化半文件。
# 输入：
#   path：新证据文件路径。
#   value：内部报告对象，元组等标准 JSON 可表示容器按序列化后的内容验证。
# 输出：
#   None：不返回业务数据。
def write_json(path, value):
    rendered = json.dumps(value, indent=2, allow_nan=False)
    decode_json(rendered, limit=32*1024*1024-1, node_limit=1_000_000)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.write("\n")


# 功能：
#   有界读取普通源文件，拒绝空文件、链接及读取时身份或大小变化。
# 输入：
#   path：源文件路径。
#   maximum：实际允许读取的字节上限。
# 输出：
#   value：被后续解析与摘要共同使用的源字节。
def load_source(path, maximum):
    value = read_plugin_file(path, limit=maximum)
    if not value:
        raise ValueError("FIXTURE_SOURCE_SIZE_INVALID")
    return value


# 功能：
#   1. 一次读取四项来源并绑定实际字节，严格解析路线的起始位置和初始方向。
#   2. 仅用路线前两点确定相机局部扫描，不将路线视为已执行的飞行任务。
# 输入：
#   args：包含 world、semantic、camera、route 路径的参数对象。
# 输出：
#   snapshot：来源摘要、原始字节、扫描原点与方向及相机校准。
def source_snapshot(args):
    sources, contents = {}, {}
    for name, maximum in (("world", 32*1024*1024), ("semantic", 16*1024*1024),
                          ("camera", 1024*1024), ("route", 1024*1024)):
        path = getattr(args, name)
        raw = load_source(path, maximum)
        contents[name] = raw
        sources[name] = {"path": str(path.resolve()), "sha256": bytes_digest(raw),
                         "maximum_bytes": maximum}
    route = decode_json(contents["route"], limit=1024*1024)
    positions = route.get("positions_m") if isinstance(route, dict) else None
    if not isinstance(positions, list) or len(positions) < 2:
        raise ValueError("FIXTURE_ROUTE_ORIGIN_MISSING")
    for point in positions[:2]:
        if (not isinstance(point, dict) or any(type(point.get(axis)) not in (int, float)
                or not math.isfinite(point[axis]) or abs(point[axis]) > 1e5 for axis in "xyz")):
            raise ValueError("FIXTURE_ROUTE_ORIGIN_INVALID")
    origin = [positions[0][axis] for axis in "xyz"]
    direction = [positions[1][axis] - positions[0][axis] for axis in "xyz"]
    fixture_pose(origin, direction, 0)
    calibration = camera_calibration(contents["camera"])
    fixture_sensor_mount(calibration)
    snapshot = {"sources": sources, "contents": contents, "origin": origin,
                "direction": direction, "calibration": calibration}
    return snapshot


# 功能：
#   结束时逐个核对来源摘要，源文件缺失或变成非法文件也记录为不一致。
# 输入：
#   sources：启动时绑定的来源路径、摘要和读取预算。
# 输出：
#   unchanged：所有当前来源都与启动快照一致时为 True。
def sources_match(sources):
    unchanged = True
    for source in sources.values():
        try:
            raw = load_source(Path(source["path"]), source["maximum_bytes"])
            unchanged = unchanged and bytes_digest(raw) == source["sha256"]
        except (OSError, ValueError):
            unchanged = False
    return unchanged


# 功能：
#   执行一项独立清理或诊断，异常加入有界错误列表，不阻止之后的进程回收。
# 输入：
#   label：清理步骤名。
#   action：无参数清理或诊断函数。
#   errors：接收清理错误的共享列表。
# 输出：
#   result：步骤结果；出现异常时为 None。
def cleanup_step(label, action, errors):
    result = None
    try:
        result = action()
    except Exception as error:
        if len(errors) < 64:
            errors.append(f"{label}:{type(error).__name__}:{error}"[:512])
    return result


# 功能：
#   有界读取当前自有仿真进程的 proc 文本；不以通常为零的虚拟文件 stat 大小作为内容长度。
# 输入：
#   path：只读诊断路径。
#   limit：允许的文本字节数。
# 输出：
#   text：严格解码的诊断文本。
def proc_text(path, limit=32*1024):
    if type(limit) is not int or not 1 <= limit <= 16*1024*1024:
        raise ValueError("FIXTURE_PROC_DIAGNOSTIC_BUDGET_INVALID")
    with path.open("rb") as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError("FIXTURE_PROC_DIAGNOSTIC_TOO_LARGE")
    text = raw.decode("utf-8")
    return text


# 功能：
#   在停止前保存自有仿真进程的内存映射；调试器场景只读取其唯一直接子进程。
# 输入：
#   process：本次以独立会话启动的仿真进程或 None。
#   debugger：是否使用调试器包装。
#   output：本次新建的证据目录。
# 输出：
#   None：不返回业务数据。
def capture_process_maps(process, debugger, output):
    if process is None or process.poll() is not None:
        return
    target_pid = process.pid
    if debugger:
        children = proc_text(Path(f"/proc/{target_pid}/task/{target_pid}/children")).split()
        if len(children) == 1:
            target_pid = int(children[0])
            if not 0 < target_pid < (1 << 31):
                raise ValueError("FIXTURE_DIAGNOSTIC_CHILD_PID_INVALID")
    maps = proc_text(Path(f"/proc/{target_pid}/maps"), 16*1024*1024)
    with (output / "native-maps-before-stop.txt").open("x", encoding="utf-8") as stream:
        stream.write(maps)


# 功能：
#   读取仍未退出的自有进程状态及线程等待点，诊断缺失时由清理调用者记录错误。
# 输入：
#   process：当前等待退出的仿真进程。
# 输出：
#   diagnostic：进程标识、状态与线程等待位置。
def stop_diagnostic(process):
    proc_dir = Path(f"/proc/{process.pid}")
    channels = {}
    for path in proc_dir.glob("task/*/wchan"):
        if len(channels) >= 4096:
            raise ValueError("FIXTURE_DIAGNOSTIC_THREAD_LIMIT")
        channels[path.parent.name] = proc_text(path, 1024).strip()
    diagnostic = {"pid": process.pid, "after_wait_seconds": 8,
                  "process_status": proc_text(proc_dir / "status"),
                  "thread_wait_channels": channels}
    return diagnostic


# 功能：
#   向当前实例仍存活的独立进程组发信号；正常退出造成的竞争不误报为发送失败。
# 输入：
#   process：由当前采集器以 start_new_session 启动的进程。
#   signum：发送给该进程组的信号。
# 输出：
#   None：不返回业务数据。
def signal_owned_process(process, signum):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        if process.poll() is None:
            raise


# 功能：
#   1. 给已确认停止的服务器退出时间；未收到确认时向自有进程组发送中断信号。
#   2. 分阶段等待并采集诊断，诊断失败仍执行强制回收，强制退出明确记录为失败。
# 输入：
#   process：本次自有仿真进程或 None。
#   server_stop_acknowledged：是否已收到当前服务器停止确认。
#   errors：接收清理错误的列表。
# 输出：
#   diagnostic：宽限期后取得的停止诊断；无需诊断或诊断失败时为 None。
def stop_owned_simulator(process, server_stop_acknowledged, errors):
    diagnostic = None
    if process is None or process.poll() is not None:
        return diagnostic
    if not server_stop_acknowledged:
        cleanup_step("SIMULATOR_INTERRUPT", lambda: signal_owned_process(process, signal.SIGINT),
                     errors)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        diagnostic = cleanup_step("STOP_DIAGNOSTIC", lambda: stop_diagnostic(process), errors)
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            errors.append("SIMULATOR_REQUIRED_FORCED_TERMINATION")
            signal_owned_process(process, signal.SIGKILL)
            process.wait(timeout=5)
    return diagnostic


# 功能：
#   运行无飞控、无电机的隔离相机校准，保存真实像素与源时间，并要求实测运动及完整清理。
# 输入：
#   args：来源文件、输出目录、扫描时长及明确的调试选项。
# 输出：
#   exit_code：全部采集条件成立时为零，否则为一；不会授予飞行或模型控制资格。
def run(args):
    if sys.platform != "linux" or sys.byteorder != "little":
        raise ValueError("FIXTURE_REQUIRES_LITTLE_ENDIAN_RUNTIME")
    if not 8 <= args.duration <= 30 or not 30 <= args.wall_timeout <= 300:
        raise ValueError("FIXTURE_DURATION_OUTSIDE_BOUNDS")
    snapshot = source_snapshot(args)
    sources, camera = snapshot["sources"], snapshot["contents"]["camera"]
    origin, direction = snapshot["origin"], snapshot["direction"]
    calibration = snapshot["calibration"]
    args.output.mkdir(parents=True, exist_ok=False)
    render_world, render_receipt = prepare_static_render_world(args.world, args.output / "render",
        expected_source_sha256=sources["world"]["sha256"])
    rendered = load_source(render_world, 32*1024*1024)
    if bytes_digest(rendered) != render_receipt["render_world_sha256"]:
        raise ValueError("FIXTURE_RENDER_WORLD_CHANGED")
    fixture = build_fixture_world(rendered, camera, origin)
    fixture_path = render_world.with_name("camera-fixture.sdf")
    with fixture_path.open("xb") as stream:
        stream.write(fixture)
    world_name = ET.fromstring(fixture).find("world").get("name")
    raw_dir = args.output / "raw"
    raw_dir.mkdir()
    partition = f"dronedream-geometry-fixture-{uuid.uuid4().hex}"
    os.environ["GZ_PARTITION"] = partition
    # Runtime-owned dependencies are read, never installed or modified here.
    sys.path.append("/usr/lib/python3/dist-packages")
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_pb2 import Pose
    from gz.transport13 import Node

    node = Node()
    capture = FixtureCapture(width=calibration.width, height=calibration.height)
    image_enum = Image.DESCRIPTOR.fields_by_name["pixel_format_type"].enum_type

    # 功能：
    #   同时记录仿真源时间、单调接收时间和墙钟时间，不混用三种时钟。
    # 输入：
    #   message：带来源时间戳的原生消息。
    # 输出：
    #   record：三类时间组成的记录。
    def receipt(message):
        record = {"simulation_time_ns": simulation_pose_time_ns(message),
                "received_monotonic_seconds": time.monotonic(),
                "received_unix_ms": time.time_ns() / 1e6}
        return record

    # 功能：
    #   只接受固定夹具的实测姿态，将位置、四元数和报头送入有界采集器。
    # 输入：
    #   message：原生 Pose 消息。
    # 输出：
    #   None：不返回业务数据。
    def on_pose(message):
        try:
            if message.name != RIG_NAME:
                raise ValueError("FIXTURE_POSE_IDENTITY_MISMATCH")
            record = {**receipt(message), "name": message.name,
                      "position_m": [message.position.x, message.position.y, message.position.z],
                      "orientation_wxyz": [message.orientation.w, message.orientation.x,
                                           message.orientation.y, message.orientation.z],
                      "header_data": {d.key: list(d.value) for d in message.header.data}}
            capture.receive_pose(record)
        except Exception as exc:
            capture.fail(f"POSE_CALLBACK:{type(exc).__name__}:{exc}")

    # 功能：
    #   将原生深度消息的布局、来源时间及像素交给采集器，错误转成采集诊断。
    # 输入：
    #   message：原生 Image 消息。
    # 输出：
    #   None：不返回业务数据。
    def on_image(message):
        try:
            record = {**receipt(message), "width": message.width, "height": message.height,
                      "step": message.step,
                      "pixel_format": image_enum.values_by_number[message.pixel_format_type].name,
                      "header_data": {d.key: list(d.value) for d in message.header.data}}
            capture.receive_image(record, bytes(message.data))
        except Exception as exc:
            capture.fail(f"IMAGE_CALLBACK:{type(exc).__name__}:{exc}")

    commands, frames, subscriptions = [], [], []
    discovered_services = []
    pose_service = f"/world/{world_name}/set_pose"
    command_transport = None
    simulator_env = os.environ.copy()
    retained_wsl_library = None
    if args.retain_wsl_d3d12:
        simulator_env, retained_wsl_library = prepare_render_process_environment(simulator_env)
        if not retained_wsl_library["applied"]:
            raise ValueError("FIXTURE_WSL_RETENTION_REQUIRES_EXPLICIT_WSL_D3D12_ENV")
    started, epoch, process, issue, completed_sweep = time.monotonic(), None, None, None, False
    close_errors = []
    server_stop_acknowledged = False
    shutdown_diagnostic = None
    log = (args.output / "gazebo.log").open("xb")
    next_command, next_progress, last_command_stamp = 0., 0., -1

    # 功能：
    #   为取出的每张原始深度图独占创建文件，并将同一字节的摘要加入帧清单。
    # 输入：
    #   queued：已从采集队列移交的元数据与不可变像素字节。
    # 输出：
    #   None：不返回业务数据。
    def save_images(queued):
        for record, data in queued:
            name = f"{len(frames):04d}.depth.f32"
            with (raw_dir / name).open("xb") as stream:
                stream.write(data)
            frames.append({**record, "path": f"raw/{name}", "sha256": bytes_digest(data),
                           "bytes": len(data)})

    try:
        command_transport = FixturePoseRequests(partition, world_name)
        for msg_type, topic, callback in ((Pose, POSE_TOPIC, on_pose),
                                          (Image, DEPTH_TOPIC, on_image)):
            if not node.subscribe(msg_type, topic, callback):
                raise RuntimeError(f"FIXTURE_SUBSCRIBE_FAILED:{topic}")
            subscriptions.append(topic)
        simulator_command = ["gz", "sim", "-s", "-r", "-v", "3", "--headless-rendering",
                             str(fixture_path.resolve())]
        if args.debugger:
            simulator_command = ["gdb", "--batch", "-ex", "set pagination off",
                "-ex", "handle SIGPIPE nostop noprint pass", "-ex", "run", "-ex", "bt",
                "-ex", "info proc mappings",
                "--args", "/usr/bin/ruby", "/usr/bin/gz", *simulator_command[1:]]
        process = subprocess.Popen(simulator_command, stdout=log, stderr=log,
                                   env=simulator_env, start_new_session=True)
        while time.monotonic() - started <= args.wall_timeout:
            queued, stamp, errors = capture.snapshot()
            save_images(queued)
            if errors:
                raise RuntimeError(";".join(errors))
            if process.poll() is not None:
                raise RuntimeError(f"FIXTURE_SIMULATOR_EXITED:{process.returncode}")
            now = time.monotonic()
            if epoch is None:
                discovered_services = node.service_list()
            if (epoch is None and len(frames) >= 3 and stamp >= 0
                    and pose_service in discovered_services):
                epoch = stamp
            if epoch is not None:
                elapsed = (stamp - epoch) / 1e9
                if elapsed >= args.duration + .8:
                    completed_sweep = True
                    break
                if now >= next_command and stamp > last_command_stamp:
                    wanted = fixture_pose(origin, direction, elapsed / args.duration)
                    requested = time.monotonic()
                    response = command_transport.request(wanted)
                    commands.append({"source_pose_time_ns_at_request": stamp, "wanted": wanted,
                        "requested_monotonic_seconds": requested,
                        "request_latency_ms": (time.monotonic()-requested)*1000,
                        **response})
                    if response["acknowledged"] is not True:
                        raise RuntimeError("FIXTURE_POSE_COMMAND_NOT_ACKNOWLEDGED")
                    next_command, last_command_stamp = now + .05, stamp
            if now >= next_progress:
                print(json.dumps({"elapsed_wall_seconds": round(now-started, 1),
                    "simulation_motion_seconds": None if epoch is None else (stamp-epoch)/1e9,
                    "saved_depth_frames": len(frames), "measured_poses": len(capture.poses),
                    "commands": len(commands)}), flush=True)
                next_progress = now + 5
            time.sleep(.01)
        if not completed_sweep:
            raise RuntimeError("FIXTURE_CAPTURE_WALL_TIMEOUT")
    except Exception as exc:
        issue = f"{type(exc).__name__}:{exc}"
    finally:
        capture.close()
        cleanup_step("PRE_STOP_MAPS", lambda: capture_process_maps(process, args.debugger,
                     args.output), close_errors)
        for topic in subscriptions:
            if cleanup_step("UNSUBSCRIBE", lambda item=topic: node.unsubscribe(item),
                            close_errors) is not True:
                close_errors.append(f"UNSUBSCRIBE_FAILED:{topic}")
        if command_transport is not None:
            server_stop_acknowledged = cleanup_step("STOP_SERVER", command_transport.stop_server,
                                                   close_errors) is True
            if cleanup_step("CLOSE_TRANSPORT", command_transport.close, close_errors) is not True:
                close_errors.append("COMMAND_TRANSPORT_DID_NOT_CLOSE_CLEANLY")
        queued, _, errors = capture.snapshot()
        cleanup_step("FINAL_IMAGES", lambda: save_images(queued), close_errors)
        shutdown_diagnostic = cleanup_step("STOP_SIMULATOR", lambda: stop_owned_simulator(
            process, server_stop_acknowledged, close_errors), close_errors)
        cleanup_step("LOG_CLOSE", log.close, close_errors)
    pose_data = capture.poses
    active_poses = [p for p in pose_data if epoch is not None
                    and epoch <= p["simulation_time_ns"] <= epoch + int(args.duration*1e9)]
    span = np.ptp([p["position_m"] for p in active_poses], axis=0).tolist() if active_poses else []
    angles = [2*np.arccos(np.clip(abs(p["orientation_wxyz"][0]), 0, 1))
              for p in active_poses]
    measured_motion = bool(span and np.linalg.norm(span) > .65 and max(angles) > .3)
    sources_unchanged = sources_match(sources)
    write_json(args.output / "poses.json", pose_data)
    write_json(args.output / "frames.json", frames)
    write_json(args.output / "commands.json", commands)
    repo_root = Path(__file__).resolve().parents[1]
    report = {"schema": "dronedream.camera-motion-calibration", "complete": bool(
            completed_sweep and measured_motion and len(frames) >= 35
            and len({f["sha256"] for f in frames}) >= 20 and not issue
            and not errors and not close_errors and sources_unchanged
            and process is not None and process.returncode == 0 and not args.debugger),
        "issue": issue, "capture_errors": list(errors), "close_errors": close_errors,
        "sources": sources, "sources_unchanged": sources_unchanged,
        "render_receipt": render_receipt, "fixture_sha256": bytes_digest(fixture),
        "calibration": asdict(calibration), "calibration_sha256": calibration.sha256,
        "partition": partition, "world_name": world_name, "rig_name": RIG_NAME,
        "origin_m": origin, "direction": direction, "motion_epoch_simulation_ns": epoch,
        "requested_simulation_duration_seconds": args.duration,
        "elapsed_wall_seconds": time.monotonic()-started,
        "measured_position_span_m": span, "maximum_measured_rotation_rad": max(angles, default=0),
        "measured_motion": measured_motion, "frame_count": len(frames),
        "pose_count": len(pose_data), "command_count": len(commands),
        "server_stop_acknowledged": server_stop_acknowledged,
        "simulator_exit_code": None if process is None else process.returncode,
        "shutdown_diagnostic": shutdown_diagnostic,
        "debugger_diagnostic_only": args.debugger,
        "images_skipped_by_sampling": capture.skipped_images,
        "depth_topic": DEPTH_TOPIC, "pose_topic": POSE_TOPIC,
        "discovered_services": discovered_services,
        "command_transport": "isolated spawn process without Python subscriptions",
        "render_environment": {k: simulator_env.get(k) for k in
            ("GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME", "LIBGL_ALWAYS_SOFTWARE",
             "EGL_PLATFORM", "LD_PRELOAD")},
        "retained_wsl_library": retained_wsl_library,
        "pose_publication": "individual Pose header source time; not Pose_V top-level header",
        "files": {name: bytes_digest((args.output / name).read_bytes())
                  for name in ("poses.json", "frames.json", "commands.json")},
        "implementation": {str(p.relative_to(repo_root)):
                           bytes_digest(p.read_bytes()) for p in (
            Path(__file__).resolve(),
            repo_root / "src/dronedream_agent_core/geometry_motion_fixture.py",
            repo_root / "src/dronedream_agent_core/geometry_fixture_capture.py",
            repo_root / "src/dronedream_agent_core/geometry_fixture_transport.py",
            repo_root / "src/dronedream_agent_core/simulation_graphics_lifetime.py")},
        "native_estimator_qualification": False, "covariance_qualified": False,
        "model_control_qualification": False, "actuator_commands_sent": False,
        "limitation": "Camera-only kinematic jig; no vehicle dynamics or airframe occlusion. "
                      "Command acknowledgements do not constitute observed motion."}
    write_json(args.output / "capture.json", report)
    print(json.dumps({k: report[k] for k in ("complete", "issue", "frame_count", "pose_count",
        "command_count", "measured_position_span_m", "maximum_measured_rotation_rad",
        "elapsed_wall_seconds")}), flush=True)
    exit_code = 0 if report["complete"] else 1
    return exit_code


# 功能：
#   解析明确来源路径和有界校准时长，执行隔离采集并以结果退出命令行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("world", "semantic", "camera", "route", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--duration", type=float, default=16.)
    parser.add_argument("--wall-timeout", type=float, default=180.)
    parser.add_argument("--debugger", action="store_true",
                        help="Capture a native crash backtrace; never mark debugger run qualified")
    parser.add_argument("--retain-wsl-d3d12", action="store_true",
                        help="Keep the installed WSL D3D12 library mapped in this Gazebo process")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
