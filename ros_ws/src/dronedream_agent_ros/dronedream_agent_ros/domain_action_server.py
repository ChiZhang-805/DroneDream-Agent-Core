"""Gazebo-only ROS 2 service host for physically bound domain actions."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import rclpy
from dronedream_agent_msgs.msg import MissionObservation
from dronedream_agent_msgs.srv import ExecuteDomainAction
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .domain_action_logic import (
    DECLARED_ACTIONS,
    evaluate_simulation_action,
    finite_number,
    load_target_points,
)


# 功能：
#   将已声明动作名称转换为 ROS 服务路径，不以此转换授予执行能力。
# 输入：
#   action：代码内声明的动作名称。
# 输出：
#   service_name：动作对应的绝对服务路径。
def _service_name(action: str) -> str:
    suffix = action.removeprefix("native.").replace(".", "/").replace("-", "_")
    service_name = f"/dronedream/domain_actions/{suffix}"
    return service_name


class SimulationDomainActionServer(Node):
    """Report arrival preconditions without fabricating scan or identity verification."""

    # 功能：
    #   绑定当前任务、目标及观测，注册可明确返回缺失识别后端的服务；不自动宣称识别成功。
    # 输入：
    #   self：待初始化的 ROS 节点。
    # 输出：
    #   None：创建服务和订阅，不返回业务数据。
    def __init__(self) -> None:
        super().__init__("dronedream_simulation_domain_action_server")
        self.declare_parameter("contract_id", "")
        self.declare_parameter("track_path", "")
        self.declare_parameter("checkpoint_contract_path", "")
        self.declare_parameter("maximum_target_distance_m", 1.0)
        self.declare_parameter("maximum_observation_age_seconds", 2.0)
        self._contract_id = str(self.get_parameter("contract_id").value)
        if not self._contract_id:
            raise ValueError("contract_id is required")
        self._maximum_target_distance_m = self.get_parameter("maximum_target_distance_m").value
        self._maximum_observation_age_seconds = self.get_parameter(
            "maximum_observation_age_seconds"
        ).value
        if any(
            not finite_number(value) or value <= 0
            for value in (
                self._maximum_target_distance_m,
                self._maximum_observation_age_seconds,
            )
        ):
            raise ValueError("domain action distance and age limits must be finite and positive")
        self._target_points = load_target_points(
            Path(str(self.get_parameter("track_path").value)),
            Path(str(self.get_parameter("checkpoint_contract_path").value)),
            expected_contract_id=self._contract_id,
        )
        self._latest_observation: MissionObservation | None = None
        self._latest_observation_monotonic: float | None = None
        self._subscription = self.create_subscription(
            MissionObservation,
            "/dronedream/mission_observation",
            self._on_observation,
            qos_profile_sensor_data,
        )
        self._services = [
            self.create_service(
                ExecuteDomainAction,
                _service_name(action),
                partial(self._execute, expected_action=action),
            )
            for action in DECLARED_ACTIONS
        ]
        self.get_logger().info(
            f"serving {len(self._services)} simulation domain actions for "
            f"contract {self._contract_id}"
        )

    # 功能：
    #   接收当前任务中序号严格递增的真实位置观测，重复或乱序消息不能刷新时效。
    # 输入：
    #   self：当前服务节点。
    #   observation：来自传感器桥接的类型化观测。
    # 输出：
    #   None：仅更新通过入口校验的最新观测与单调接收时间。
    def _on_observation(self, observation: MissionObservation) -> None:
        if (
            observation.contract_id != self._contract_id
            or not observation.localization_ok
            or not observation.link_ok
            or observation.sequence <= 0
        ):
            return
        if (
            self._latest_observation is not None
            and observation.sequence <= self._latest_observation.sequence
        ):
            return
        self._latest_observation = observation
        self._latest_observation_monotonic = time.monotonic()

    # 功能：
    #   使用最新已接收观测检查动作前置条件，返回真实错误或证据；缺观测不能执行。
    # 输入：
    #   self：当前服务节点。
    #   request：包含任务身份、动作和目标的 ROS 请求。
    #   response：ROS 提供的待填充响应。
    #   expected_action：当前服务入口声明的动作。
    # 输出：
    #   response：带成功标识、证据及失败原因的服务响应。
    def _execute(
        self,
        request: ExecuteDomainAction.Request,
        response: ExecuteDomainAction.Response,
        *,
        expected_action: str,
    ) -> ExecuteDomainAction.Response:
        observation = self._latest_observation
        observed_at = self._latest_observation_monotonic
        if observation is None or observed_at is None:
            response.success = False
            response.evidence = []
            response.details_json = "{}"
            response.issue_code = "DOMAIN_ACTION_OBSERVATION_MISSING"
            return response
        result = evaluate_simulation_action(
            expected_contract_id=self._contract_id,
            expected_action=expected_action,
            request_contract_id=request.contract_id,
            task_id=request.task_id,
            request_action=request.action,
            target_node=request.target_node,
            arguments_json=request.arguments_json,
            target_points=self._target_points,
            observation_contract_id=observation.contract_id,
            observation_sequence=int(observation.sequence),
            observation_position_enu_m=(
                float(observation.pose_enu.position.x),
                float(observation.pose_enu.position.y),
                float(observation.pose_enu.position.z),
            ),
            observation_age_seconds=time.monotonic() - observed_at,
            maximum_observation_age_seconds=self._maximum_observation_age_seconds,
            maximum_target_distance_m=self._maximum_target_distance_m,
        )
        response.success = bool(result["success"])
        response.evidence = list(result["evidence"])
        response.details_json = str(result["details_json"])
        response.issue_code = str(result["issue_code"])
        return response


# 功能：
#   运行仿真动作服务，正常结束、中断或初始化失败后都关闭本次 ROS 上下文。
# 输入：
#   args：可选 ROS 命令行参数。
# 输出：
#   None：不返回业务数据。
def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SimulationDomainActionServer()
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
