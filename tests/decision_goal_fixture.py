"""Synthetic goal binding only; no simulation imports or formal training data."""

import copy

from test_decision_label_evidence import native_fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：构造语义终点不同于局部控制点的合成测试证据，避免云端测试依赖仿真启动器。
# 输入：无；输出：原状态、命令、输入快照与ENU米制终点，不属于正式数据。
def local_target_fixture():
    row, outcome = native_fixture()
    state, goal = row["state"], outcome["goal_world_enu_m"]
    snapshot = {
        "schema_version": "dronedream.text-navigation-snapshot.v1",
        "source_of_truth": "metric-range-and-localization-evidence-not-rendered-image",
        "strategic_context": {"task": {"navigation_goal_id": state["goal_id"]}},
        "known_static_map": {
            "source_sha256": state["map_sha256"],
            "qualified_route_sha256": state["route_sha256"],
        },
        "goal_position_m": copy.deepcopy(goal),
    }
    snapshot["snapshot_sha256"] = decision_digest(snapshot)
    command = copy.deepcopy(outcome["commands"][0])
    command["evaluated_target_position_m"]["x"] = 0.2
    command["model_navigation_snapshot_sha256"] = snapshot["snapshot_sha256"]
    command["requested_control_intent"]["navigation_snapshot_sha256"] = snapshot["snapshot_sha256"]
    return state, RuntimeLocalSafetyCommand.model_validate(command), snapshot, goal
