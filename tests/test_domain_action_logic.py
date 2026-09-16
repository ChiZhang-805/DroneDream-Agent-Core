from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch


# 功能：
#   导入实际 ROS 包的纯逻辑模块，不模拟产品的动作判断。
# 输入：
#   无。
# 输出：
#   module：无需 ROS 安装即可测试的动作逻辑模块。
def _logic_module():
    root = Path(__file__).parents[1] / "ros_ws" / "src" / "dronedream_agent_ros"
    with patch.object(sys, "path", [str(root), *sys.path]):
        module = importlib.import_module("dronedream_agent_ros.domain_action_logic")
    return module


# 功能：
#   写入隔离的真实轨迹与检查点文件，使用产品读取函数建立目标映射。
# 输入：
#   tmp_path：本项测试目录。
# 输出：
#   fixture：动作逻辑模块与目标映射组成的二元组。
def _targets(tmp_path: Path):
    logic = _logic_module()
    track = tmp_path / "track.json"
    checkpoints = tmp_path / "checkpoints.json"
    track.write_text(
        json.dumps(
            {
                "source_world_points": [
                    {"east_m": 0.0, "north_m": 0.0, "up_m": 1.0},
                    {"east_m": 4.0, "north_m": 5.0, "up_m": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    checkpoints.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "checkpoint_id": "checkpoint-001",
                        "track_point_index": 1,
                        "target_node": "verified-001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fixture = logic, logic.load_target_points(track, checkpoints)
    return fixture


# 功能：
#   到达目标附近只满足位置前置条件，没有解码及认证后端时不得返回扫码成功。
# 输入：
#   tmp_path：本项测试目录。
# 输出：
#   None：断言失败回执与空证据，不返回业务数据。
def test_scan_code_requires_sensor_evidence_beyond_target_proximity(tmp_path: Path) -> None:
    logic, targets = _targets(tmp_path)
    result = logic.evaluate_simulation_action(
        expected_contract_id="mission-1",
        expected_action="native.payload.scan-code",
        request_contract_id="mission-1",
        task_id="scan",
        request_action="native.payload.scan-code",
        target_node="verified-001",
        arguments_json="{}",
        target_points=targets,
        observation_contract_id="mission-1",
        observation_sequence=12,
        observation_position_enu_m=(4.1, 5.0, 1.8),
        observation_age_seconds=0.1,
        maximum_observation_age_seconds=2.0,
        maximum_target_distance_m=0.75,
    )

    assert result["success"] is False
    assert result["evidence"] == []
    assert result["issue_code"] == "DOMAIN_ACTION_SENSOR_VERIFIER_UNAVAILABLE"
    details = json.loads(result["details_json"])
    assert details["backend"] == "semantic-target-proximity"
    assert details["proximity_verified"] is True
    assert details["sensor_verifier_available"] is False
    assert details["target_distance_m"] < 0.25


# 功能：
#   尚未到达目标时返回距离前置条件失败，不产生识别成功证据。
# 输入：
#   tmp_path：本项测试目录。
# 输出：
#   None：不返回业务数据。
def test_scan_code_rejects_when_drone_is_not_at_bound_target(tmp_path: Path) -> None:
    logic, targets = _targets(tmp_path)
    result = logic.evaluate_simulation_action(
        expected_contract_id="mission-1",
        expected_action="native.payload.scan-code",
        request_contract_id="mission-1",
        task_id="scan",
        request_action="native.payload.scan-code",
        target_node="verified-001",
        arguments_json="{}",
        target_points=targets,
        observation_contract_id="mission-1",
        observation_sequence=12,
        observation_position_enu_m=(0.0, 0.0, 1.0),
        observation_age_seconds=0.1,
        maximum_observation_age_seconds=2.0,
        maximum_target_distance_m=0.75,
    )

    assert result["success"] is False
    assert result["issue_code"] == "DOMAIN_ACTION_TARGET_NOT_REACHED"
    assert result["evidence"] == []
