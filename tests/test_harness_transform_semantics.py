"""Preparation transforms must preserve user constraints and required success conditions."""

import pytest

from dronedream_agent_core.contracts import IntentArtifact, TaskGraph, TaskNode
from dronedream_agent_plugins.harness_transform_plugins import (
    _canonicalize_intent,
    _enrich_task_evidence,
)


# 功能：
#   验证约束规范化保留大小写敏感的位置身份，只去除空白规范化后完全相同的文本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_constraint_normalization_preserves_case_sensitive_entity_names():
    intent = IntentArtifact(
        goal="Pick up and return",
        start_entity="start",
        target_entity="target",
        return_entity="start",
        payload_action="pickup",
        constraints=["Avoid zone A", "Avoid zone a", "  Avoid   zone A  "],
    )
    original = list(intent.constraints)
    normalized = _canonicalize_intent(value=intent)
    assert normalized.constraints == ["Avoid zone A", "Avoid zone a"]
    assert intent.constraints == original


# 功能：
#   验证证据列表满额且缺少本动作必需条件时拒绝增强，不悄悄省略必需条件。
# 输入：
#   already_present：必需的载荷交接条件是否已占一个证据位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("already_present", [False, True])
def test_full_evidence_list_cannot_silently_omit_required_condition(already_present):
    required = "payload attachment and custody state confirmed"
    evidence = [f"condition {index}" for index in range(16)]
    if already_present:
        evidence[-1] = required
    graph = TaskGraph(
        nodes=[
            TaskNode(
                task_id="pickup",
                action="pickup",
                target_node="target",
                success_evidence=evidence,
                fallback="hold",
            )
        ]
    )
    original = graph.model_dump(mode="python")
    if already_present:
        enriched = _enrich_task_evidence(value=graph)
        assert enriched.nodes[0].success_evidence == evidence
    else:
        with pytest.raises(ValueError, match="TASK_EVIDENCE_CAPACITY_EXCEEDED"):
            _enrich_task_evidence(value=graph)
    assert graph.model_dump(mode="python") == original


# 功能：
#   验证原证据去重后可用的空位用于新增必需条件，保留原证据且不伪造任务成功。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_evidence_releases_capacity_without_erasing_distinct_conditions():
    graph = TaskGraph(
        nodes=[
            TaskNode(
                task_id="pickup",
                action="pickup",
                target_node="target",
                success_evidence=["identity   matches"] * 16,
                fallback="hold",
            )
        ]
    )
    enriched = _enrich_task_evidence(value=graph)
    assert enriched.nodes[0].success_evidence == [
        "identity matches",
        "payload attachment and custody state confirmed",
    ]
    assert len(graph.nodes[0].success_evidence) == 16
