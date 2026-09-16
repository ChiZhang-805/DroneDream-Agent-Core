"""Translate an authorized typed ROS safety event into the executor abort gate."""

from __future__ import annotations

from pathlib import Path

import rclpy
from dronedream_agent_msgs.msg import SafetyEvent
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from .runtime_io import publish_abort


class SafetyEventGuard(Node):
    """Fail closed without giving a native plugin direct actuator authority."""

    _ALLOWED_ACTIONS = {"safe_hold_then_land", "emergency_land"}

    # 功能：
    #   绑定本次任务和中止路径，创建可靠安全事件订阅；节点本身不直接操作电机。
    # 输入：
    #   self：待初始化的 ROS 安全节点。
    # 输出：
    #   None：初始化订阅和发布状态，不返回业务数据。
    def __init__(self) -> None:
        super().__init__("dronedream_safety_event_guard")
        self.declare_parameter("contract_id", "")
        self.declare_parameter("abort_file", "")
        self.declare_parameter("safety_event_topic", "/dronedream/safety_event")
        self._contract_id = str(self.get_parameter("contract_id").value)
        abort_file = str(self.get_parameter("abort_file").value)
        if not self._contract_id or not abort_file:
            raise ValueError("contract_id and abort_file are required")
        self._abort_file = Path(abort_file)
        self._handled = False
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        self._subscription = self.create_subscription(
            SafetyEvent,
            str(self.get_parameter("safety_event_topic").value),
            self._on_safety_event,
            qos,
        )

    # 功能：
    #   仅将本次任务认可的严重安全事件送入执行器中止门控，保留已经存在的中止原因。
    # 输入：
    #   self：当前安全节点。
    #   message：类型化安全事件。
    # 输出：
    #   None：独占发布成功或已有中止时标记已处理；写入异常保留重试机会。
    def _on_safety_event(self, message: SafetyEvent) -> None:
        if self._handled:
            return
        if message.contract_id != self._contract_id:
            self.get_logger().error("rejected safety event for a different mission contract")
            return
        if int(message.severity) < 3 or message.action not in self._ALLOWED_ACTIONS:
            self.get_logger().error("rejected unrecognized or insufficient-severity safety event")
            return
        payload = {
            "reason": "NATIVE_RUNTIME_SAFETY_EVENT",
            "world_paused": False,
            "contract_id": self._contract_id,
            "action": message.action,
            "severity": int(message.severity),
            "observation_sequence": int(message.observation_sequence),
            "observation_age_ms": int(message.observation_age_ms),
            "deadline_ms": int(message.deadline_ms),
            "issue_codes": list(message.issue_codes),
            "source": "dronedream_agent_ros.safety_event_guard",
        }
        publish_abort(self._abort_file, payload)
        self._handled = True
        self.get_logger().fatal("authorized native safety event entered the executor abort gate")


# 功能：
#   运行安全事件转换节点，异常、中断及初始化失败都结束本次 ROS 上下文。
# 输入：
#   args：可选 ROS 命令行参数。
# 输出：
#   None：不返回业务数据。
def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SafetyEventGuard()
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
