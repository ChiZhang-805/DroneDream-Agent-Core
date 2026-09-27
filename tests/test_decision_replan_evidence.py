"""Synthetic replanning proof checks; planner output alone never counts as flight data."""

import copy

import pytest
from test_decision_stage_wait import wait_fixture

from dronedream_agent_core.contracts import GraphRoute, RuntimeLocalSafetyCommand
from dronedream_agent_core.decision_replan_evidence import verify_replan_continuation
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：把合成恢复窗口绑定到新的路由与原始导航快照；输出：测试夹具，不是实际仿真。
def fixture():
    row, doc = wait_fixture()
    continuation = doc["continuation"]
    resumed, outcome = continuation["row"], continuation["outcome"]
    points = [{"x": 0., "y": 0., "z": 1.5}, {"x": 2., "y": 0., "z": 1.5}]
    route = GraphRoute(start_node="start", goal_node="goal", node_ids=["start", "goal"],
                       edge_ids=["new-edge"], positions_m=points, route_length_m=2.,
                       all_edges_flight_verified=False).model_dump(mode="json")
    digest = decision_digest(route)
    resumed["state"]["route_sha256"] = digest
    outcome["state_sha256"] = decision_digest(resumed["state"])
    outcome["behavior_application"]["state_sha256"] = outcome["state_sha256"]
    snapshot = {
        "schema_version": "dronedream.text-navigation-snapshot.v1",
        "source_of_truth": "metric-range-and-localization-evidence-not-rendered-image",
        "strategic_context": {"task": {"navigation_goal_id": resumed["state"]["goal_id"]}},
        "known_static_map": {"source_sha256": resumed["state"]["map_sha256"],
                             "qualified_route_sha256": digest},
        "goal_position_m": outcome["goal_world_enu_m"],
    }
    snapshot["snapshot_sha256"] = decision_digest(snapshot)
    outcome["navigation_snapshots"] = {snapshot["snapshot_sha256"]: snapshot}
    for command, actual in zip(outcome["commands"], outcome["control_applications"], strict=True):
        command["model_navigation_snapshot_sha256"] = snapshot["snapshot_sha256"]
        command["requested_control_intent"]["navigation_snapshot_sha256"] = snapshot["snapshot_sha256"]
        validated = RuntimeLocalSafetyCommand.model_validate(command)
        actual["command_sha256"] = decision_digest(validated.model_dump(mode="json"))
        actual["intent"] = copy.deepcopy(command["requested_control_intent"])
    outcome["behavior_application"]["control_application_hashes"] = [
        decision_digest(a) for a in outcome["control_applications"]]
    doc.update(replan_event={"new_route_sha256": digest}, replacement_route=route,
               replanned_path_world_enu_m=points)
    return row, doc


# 功能：只有新路线身份与实际推进相符才接受；输入：完整合成结果；输出：通过且不改证据。
def test_requires_bound_executed_replanned_route():
    row, doc = fixture()
    original = copy.deepcopy(doc)
    verify_replan_continuation(row, doc, end_ms=12000, last_position=(0., 0., 1.5))
    assert doc == original


# 功能：拒绝仅规划、借用别的任务、未采用新路线、跳位置或无进展的假成功。
@pytest.mark.parametrize("fault", ["no-continuation", "old-route", "map", "episode", "split",
                                 "late", "no-snapshot", "changed-path", "jump", "no-progress"])
def test_replan_rejects_unexecuted_or_unrelated_continuation(fault):
    row, doc = fixture()
    continuation = doc["continuation"]
    if fault == "no-continuation":
        del doc["continuation"]
    elif fault == "old-route":
        continuation["row"]["state"]["route_sha256"] = row["state"]["route_sha256"]
    elif fault in {"episode", "split"}:
        continuation["row"]["episode_id" if fault == "episode" else "split"] = "other"
    elif fault == "map":
        continuation["row"]["state"]["map_sha256"] = "f" * 64
    elif fault == "late":
        continuation["row"]["state"]["frame"]["observed_at_ms"] = 90000
    elif fault == "no-snapshot":
        del continuation["outcome"]["navigation_snapshots"]
    elif fault == "changed-path":
        doc["replanned_path_world_enu_m"][1]["x"] = 3.
    elif fault == "jump":
        continuation["outcome"]["observations"][0]["current_position_m"]["x"] = 1.
    else:
        for observed in continuation["outcome"]["observations"]:
            observed["current_position_m"]["x"] = 0.
    with pytest.raises(ValueError):
        verify_replan_continuation(row, doc, end_ms=12000, last_position=(0., 0., 1.5))
