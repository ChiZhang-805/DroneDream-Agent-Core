"""Preserve Gazebo entity identity while publishing typed ROS observations."""

from __future__ import annotations

import math
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from dronedream_agent_msgs.msg import MissionObservation
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .runtime_io import read_object


@dataclass(frozen=True)
class EntityPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    simulator_sec: int
    simulator_nsec: int
    received_monotonic: float


class GazeboPoseObserver(Node):
    """Read raw Pose_V names, then publish schema-stable MissionObservation."""

    # 功能：
    #   用显式地图话题和实体身份创建观测桥，不默认回退到旧学校地图或旧无人机。
    # 输入：
    #   self：待初始化的 ROS 节点。
    # 输出：
    #   None：完成订阅及定时器创建，不返回业务数据。
    def __init__(self) -> None:
        super().__init__("dronedream_gazebo_pose_observer")
        self.declare_parameter("gazebo_pose_topic", "")
        self.declare_parameter("entity_name", "")
        self.declare_parameter("contract_id", "")
        self.declare_parameter("segment_id", "")
        self.declare_parameter("runtime_phase_path", "")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("maximum_pose_age_seconds", 0.25)
        self._gazebo_topic = self.get_parameter("gazebo_pose_topic").value
        self._entity_name = self.get_parameter("entity_name").value
        self._contract_id = self.get_parameter("contract_id").value
        self._segment_id = self.get_parameter("segment_id").value
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self._gazebo_topic,
                self._entity_name,
                self._contract_id,
            )
        ):
            raise ValueError("gazebo_pose_topic, entity_name and contract_id are required")
        runtime_phase_path = str(self.get_parameter("runtime_phase_path").value)
        self._runtime_phase_path = Path(runtime_phase_path) if runtime_phase_path else None
        self._last_valid_phase = None
        self._last_valid_phase_at = None
        self._phase_read_issue = None
        publish_hz = float(self.get_parameter("publish_hz").value)
        if not 1.0 <= publish_hz <= 100.0:
            raise ValueError("publish_hz must be between 1 and 100")
        self._maximum_pose_age_seconds = float(self.get_parameter("maximum_pose_age_seconds").value)
        if not 0 < self._maximum_pose_age_seconds <= 2.0:
            raise ValueError("maximum_pose_age_seconds must be in (0, 2]")

        self._lock = threading.Lock()
        self._latest: EntityPose | None = None
        self._sequence = 0
        self._published_sequence = 0
        self._last_source_stamp = None
        self._publisher = self.create_publisher(
            MissionObservation, "/dronedream/mission_observation", qos_profile_sensor_data
        )
        self._gazebo = GazeboNode()
        if not self._gazebo.subscribe(Pose_V, self._gazebo_topic, self._on_gazebo_pose):
            raise RuntimeError("Gazebo pose subscription failed")
        self._timer = self.create_timer(1.0 / publish_hz, self._publish_latest)
        self.get_logger().info(
            f"observing Gazebo entity {self._entity_name!r} on {self._gazebo_topic}"
        )

    # 功能：
    #   接收唯一匹配实体的有限姿态，归一化四元数，保存原仿真时间并拒绝重复或倒退时间。
    # 输入：
    #   self：当前桥接节点。
    #   message：Gazebo 的原始带实体名称姿态消息。
    # 输出：
    #   None：仅对有效的新传感器样本更新缓存及序号。
    def _on_gazebo_pose(self, message: Pose_V) -> None:
        poses = [pose for pose in message.pose if pose.name == self._entity_name]
        if len(poses) != 1:
            return
        pose = poses[0]
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in values):
            return
        norm = math.hypot(*values[3:])
        if norm < 1e-9 or not math.isfinite(norm):
            return
        stamp = message.header.stamp
        if stamp.sec < 0 or not 0 <= stamp.nsec < 1_000_000_000:
            return
        source_stamp = (stamp.sec, stamp.nsec)
        value = EntityPose(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z,
            qx=pose.orientation.x / norm,
            qy=pose.orientation.y / norm,
            qz=pose.orientation.z / norm,
            qw=pose.orientation.w / norm,
            simulator_sec=stamp.sec,
            simulator_nsec=stamp.nsec,
            received_monotonic=time.monotonic(),
        )
        with self._lock:
            # 仿真重启必须重建本次执行节点，不能把另一时间线伪装成当前任务的新观测。
            if self._last_source_stamp is not None and source_stamp <= self._last_source_stamp:
                return
            self._last_source_stamp = source_stamp
            self._latest = value
            self._sequence += 1

    # 功能：
    #   每个有效样本最多发布一次；超过本机年龄预算则丢弃，不用新时间戳包装旧姿态。
    # 输入：
    #   self：持有原始观测缓存的桥接节点。
    # 输出：
    #   None：发布带真实仿真采样时间的观测，缺失信息保持明确不可用。
    def _publish_latest(self) -> None:
        with self._lock:
            pose = self._latest
            sequence = self._sequence
        if pose is None or sequence <= self._published_sequence:
            return
        if not 0 <= time.monotonic() - pose.received_monotonic <= self._maximum_pose_age_seconds:
            return
        now = self.get_clock().now().to_msg()
        message = MissionObservation()
        message.header.stamp = now
        message.header.frame_id = "map_enu"
        message.contract_id = self._contract_id
        message.segment_id = self._segment_id
        message.runtime_phase = self._runtime_phase()
        message.sequence = sequence
        message.simulator_time.sec = pose.simulator_sec
        message.simulator_time.nanosec = pose.simulator_nsec
        message.pose_enu.position.x = pose.x
        message.pose_enu.position.y = pose.y
        message.pose_enu.position.z = pose.z
        message.pose_enu.orientation.x = pose.qx
        message.pose_enu.orientation.y = pose.qy
        message.pose_enu.orientation.z = pose.qz
        message.pose_enu.orientation.w = pose.qw
        message.battery_available = False
        message.clearance_available = False
        message.collision_monitor_available = False
        message.localization_ok = True
        message.link_ok = True
        message.geofence_ok = False
        message.target_reached = False
        message.deviation_code = "UNASSESSED"
        message.source_topic = self._gazebo_topic
        self._publisher.publish(message)
        self._published_sequence = sequence

    # 功能：
    #   阶段是执行元数据，不是姿态或控制指令。临时 I/O/截断读取最多沿用
    #   100ms 内已验证阶段；不能刷新期限、重放传感器或把未知阶段当作起飞前。
    #   无缓存、持续故障、时钟倒退或显式非法结构仍拒绝，日志保留具体故障类别。
    # 输入：
    #   self：包含当前运行阶段路径的节点。
    # 输出：
    #   phase：已识别阶段，或使原生安全组件拒绝的 UNRECOGNIZED 标识。
    def _runtime_phase(self) -> str:
        if self._runtime_phase_path is None:
            return "PREFLIGHT"
        now = time.monotonic()
        try:
            payload = read_object(self._runtime_phase_path)
        except (OSError, json.JSONDecodeError) as error:
            issue = type(error).__name__
            if getattr(self, "_phase_read_issue", None) != issue:
                self.get_logger().warning(f"runtime phase read interrupted: {issue}; bounded recovery only")
            self._phase_read_issue = issue
            previous_at = getattr(self, "_last_valid_phase_at", None)
            if previous_at is not None and 0 <= now - previous_at <= .1:
                return self._last_valid_phase
            self._last_valid_phase_at = None
            return "UNRECOGNIZED:INVALID_PHASE_RECORD"
        except ValueError:
            self._last_valid_phase_at = None
            return "UNRECOGNIZED:INVALID_PHASE_RECORD"
        phase = payload.get("phase")
        if not isinstance(phase, str) or not 0 < len(phase) <= 128:
            self._last_valid_phase_at = None
            return "UNRECOGNIZED:INVALID_PHASE_LABEL"
        allowed = {
            "PREFLIGHT",
            "TAKEOFF",
            "ACTION",
            "TRACK",
            "HOLDING",
            "LOCAL_SLOW",
            "LOCAL_REPLAN",
            "MODEL_AUTHORITY_HOLD",
            "PERCEPTION_REFRESH_HOLD",
            "PERCEPTION_STARTUP_HOLD",
            "TRACKING_RECOVERY",
            "WAYPOINT_SETTLE",
            "CHECKPOINT",
            "PAUSED",
            "LANDING",
            "LANDED",
            "COMPLETE",
            "FAILED",
        }
        # An unknown live phase is a schema drift, not preflight.  Surface a
        # value that the native capability rejects so the host fails closed
        # instead of silently widening its deadline during flight.
        if phase not in allowed:
            self._last_valid_phase_at = None
            return f"UNRECOGNIZED:{phase}"
        if getattr(self, "_phase_read_issue", None) is not None:
            self.get_logger().info("runtime phase read recovered from transient transport failure")
        self._phase_read_issue = None
        self._last_valid_phase, self._last_valid_phase_at = phase, now
        return phase


# 功能：
#   启动真实观测桥，退出时回收 ROS；只抑制上下文已经关闭时的清理竞争。
# 输入：
#   args：可选 ROS 命令行参数。
# 输出：
#   None：不返回业务数据。
def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = GazeboPoseObserver()
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except Exception:
        # Process-group shutdown can invalidate the ROS context while the
        # executor is rebuilding its wait set.  Suppress only that cleanup
        # race; a live-context runtime failure must still surface.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
