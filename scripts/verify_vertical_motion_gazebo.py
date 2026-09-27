"""隔离 Gazebo 速度接口验收；真实读取位姿，不调用模型，不提供 PX4/负载训练资格。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from dronedream_agent_core.collision import vehicle_clearance
from dronedream_agent_core.contracts import LocalPlannerRequest, Vector3, VehicleAsset
from dronedream_agent_core.dynamic_safety import predictive_safety_decision
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.vertical_navigation import CeilingTransition, VerticalMotionGuard


# 功能：
#   从本次实际加载的 SDF 提取静态盒，防止探针使用与世界不相符的碰撞几何。
# 输入：
#   path：测试 SDF 文件路径。
# 输出：
#   primitives：明确限定为本夹具静态无旋转盒的几何列表。
def geometry(path):
    primitives = []
    for model in ET.parse(path).findall(".//world/model"):
        if model.findtext("static") != "true":
            continue
        pose = [float(n) for n in model.findtext("pose").split()]
        size = [float(n) for n in model.findtext("link/collision/geometry/box/size").split()]
        if pose[3:] != [0., 0., 0.]:
            raise ValueError("PROBE_FIXTURE_ROTATION_UNSUPPORTED")
        keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
        primitives.append(dict(zip(keys, (*pose[:3], *size), strict=True)))
    return primitives


# 功能：
#   在独立传输分区启动有界 Gazebo 进程，执行屋檐退出后爬升并保存原生观测和动作。
# 输入：
#   output：新的证据目录；world_path：受控测试世界。
# 输出：
#   result：本组件检查的结果，不表示产品模型成功或训练数据增加。
def verify(output: Path, world_path: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    os.environ["GZ_PARTITION"] = "dronedream-vertical-probe-" + uuid.uuid4().hex
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.twist_pb2 import Twist
    from gz.transport13 import Node

    latest, condition = [], threading.Condition()

    # 功能：
    #   只接收模型本身的原生位姿，保留 Gazebo 源时间，不使用发出的目标伪装观测。
    # 输入：
    #   message：Gazebo 位姿向量消息。
    # 输出：
    #   None：替换最新观测并唤醒检查循环。
    def receive(message):
        for pose in message.pose:
            if pose.name == "probe":
                # PosePublisher 将源时钟写在每个 Pose 内，不在外层 Pose_V 中。
                stamp = pose.header.stamp.sec + pose.header.stamp.nsec / 1e9
                with condition:
                    latest[:] = [(stamp, (pose.position.x, pose.position.y, pose.position.z))]
                    condition.notify_all()

    node = Node()
    if not node.subscribe(Pose_V, "/model/probe/pose", receive):
        raise RuntimeError("GAZEBO_PROBE_SUBSCRIPTION_FAILED")
    publisher = node.advertise("/vertical_probe/cmd_vel", Twist)
    primitives = geometry(world_path)
    vehicle = VehicleAsset(asset_id="isolated-probe", name="isolated-probe", dry_mass_kg=2.,
        max_takeoff_mass_kg=3., body_radius_m=.25, body_height_m=.4, max_speed_mps=1.,
        max_acceleration_mps2=1., qualified_range_m=100., reserve_battery_percent=30.,
        max_pickup_payload_kg=.5, sensors=["probe-native-pose"])
    world = MetricVoxelMap(resolution_m=.25, minimum_bound_m=Vector3(x=-2., y=-2., z=0.),
                           maximum_bound_m=Vector3(x=6., y=2., z=6.))
    classifications = []
    for x in range(32):
        for y in range(16):
            for z in range(24):
                center = world.center_for((x, y, z))
                occupied = any(vehicle_clearance((center.x, center.y, center.z), p,
                    radius_m=.25, half_height_m=.2) < 0. for p in primitives)
                classifications.append((center, occupied))
    world.seed_known_static_region(classifications, source_sha256=sha256_json(primitives))
    route = [Vector3(x=0., y=0., z=2.), Vector3(x=3., y=0., z=3.)]
    world.bind_qualified_route(route, route_sha256=sha256_json(route), minimum_clearance_m=.15)

    # 功能：
    #   检查动作完整路径处于同一测试世界已知静态空间，精确碰撞仍由预测器独立判断。
    # 输入：
    #   points、margin：扫掠路径及预测器已采用的余量。
    # 输出：
    #   covered：静态覆盖判定。
    def coverage(points, margin):
        path = [Vector3(x=x, y=y, z=z) for x, y, z in points]
        covered = world.qualified_static_path_covered(path)
        return covered

    rows, previous, transition = [], None, CeilingTransition()
    reached, failure = False, None
    with (output / "gazebo.log").open("wb") as log:
        command = ["gz", "sim", "-s", "-r", "--iterations", "16000", str(world_path)]
        child = subprocess.Popen(command,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 60.
            while time.monotonic() < deadline:
                with condition:
                    condition.wait(.05)
                    sample = latest[0] if latest else None
                if sample is None or previous is not None and sample[0] <= previous[0]:
                    if child.poll() is not None:
                        raise RuntimeError("GAZEBO_PROBE_EXITED_BEFORE_COMPLETION")
                    continue
                stamp, position = sample
                velocity = ((0., 0., 0.) if previous is None else
                    tuple((position[i] - previous[1][i]) / (stamp - previous[0]) for i in range(3)))
                previous = sample
                guard = VerticalMotionGuard(vehicle=vehicle, payload={"state": "detached"},
                    primitives=primitives, position=position, source_seconds=stamp,
                    transition=transition, clearance=.15, coverage_check=coverage,
                    coverage_identity=world.motion_coverage_identity())
                request = LocalPlannerRequest(
                    current_position_m=Vector3(**dict(zip("xyz", position, strict=True))),
                    current_velocity_mps=Vector3(**dict(zip("xyz", velocity, strict=True))),
                    target_position_m=Vector3(x=4., y=0., z=3.2),
                    requested_velocity_mps=Vector3(x=.6, y=0., z=.35),
                    vehicle_radius_m=.25, vehicle_height_m=.4, max_speed_mps=1.,
                    max_acceleration_mps2=1., required_clearance_m=.15,
                    prediction_horizon_seconds=3., prediction_step_seconds=.1)
                decision = predictive_safety_decision(request, primitives, motion_check=guard.check)
                command = Twist()
                if decision.action in {"continue", "slow"}:
                    command.linear.x = decision.selected_velocity_mps.x
                    command.linear.y = decision.selected_velocity_mps.y
                    command.linear.z = decision.selected_velocity_mps.z
                publisher.publish(command)
                measured_clearance = min(vehicle_clearance(position, p,
                    radius_m=.25, half_height_m=.2) for p in primitives)
                rows.append({"simulation_seconds": stamp, "position_m": position,
                             "velocity_mps": velocity, "action": decision.model_dump(mode="json"),
                             "motion_context": guard.context, "clearance_m": measured_clearance})
                if measured_clearance < .14:
                    raise RuntimeError("GAZEBO_PROBE_CLEARANCE_VIOLATION")
                if position[0] >= 2.2 and position[2] >= 3.:
                    reached = True
                    break
            if not reached:
                raise RuntimeError("GAZEBO_PROBE_GOAL_NOT_REACHED")
        except Exception as error:
            failure = str(error)
        finally:
            publisher.publish(Twist())
            node.unsubscribe("/model/probe/pose")
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
    result = {"schema": "dronedream.vertical-gazebo-probe.v1", "passed": reached and not failure,
              "failure": failure, "samples": len(rows), "observations": rows,
              "native_pose_used": True, "cloud_model_called": False,
              "local_neural_model_called": False, "px4_qualification": False,
              "payload_dynamics_qualification": False, "training_images_added": 0,
              "actuation": "Gazebo VelocityControl; zero-gravity interface probe only"}
    publish_runtime_json(output / "result.json", result, replace_existing=False)
    print(json.dumps({key: value for key, value in result.items() if key != "observations"}))
    return result


# 功能：
#   将组件验收结果转换为可用于脚本检查的退出码。
# 输入：
#   命令行参数：新的证据目录与测试世界路径。
# 输出：
#   exit_code：仅组件探针成功时为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output, Path(__file__).parents[1] / "tests/fixtures/vertical_exit.sdf")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
