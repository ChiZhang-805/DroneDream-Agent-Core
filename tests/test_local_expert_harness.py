from __future__ import annotations

import pytest

from dronedream_agent_core.local_expert_harness import (
    requested_navigation_expert,
    route_local_experts,
)


# 功能：
#   验证明确的进度停滞触发恢复专家，而非继续调用普通巡航策略。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_requests_recovery_for_stalled_progress() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "phase": "TRACK",
                "control_profile": "transit",
                "decision_trigger": "progress-stalled",
            }
        }
    }

    assert requested_navigation_expert(snapshot) == "recovery-policy"
    decision = route_local_experts(
        snapshot,
        available_roles={
            "local-navigation-policy",
            "recovery-policy",
            "risk-critic",
            "state-anomaly-detector",
        },
    )

    assert decision.selected_navigation_role == "recovery-policy"
    assert decision.fallback_used is False
    assert decision.advisory_roles == ["risk-critic", "state-anomaly-detector"]


# 功能：
#   验证跟踪器生命周期名称不等同于语义恢复事件，避免错误选择恢复模型。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_does_not_treat_tracker_lifecycle_as_recovery_incident() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "phase": "TRACKING_RECOVERY",
                "control_profile": "cruise",
                "decision_trigger": "periodic",
                "recovery_episode_id": None,
            }
        }
    }

    decision = route_local_experts(
        snapshot,
        available_roles={
            "local-navigation-policy",
            "recovery-policy",
            "risk-critic",
        },
    )

    assert decision.requested_navigation_role == "local-navigation-policy"
    assert decision.selected_navigation_role == "local-navigation-policy"
    assert decision.fallback_used is False


# 功能：
#   验证已挂载载荷且动力学就绪时，将已安装载荷适配器加入咨询列表。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_invokes_wired_payload_adapter() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "payload": {
                    "state": "attached",
                    "dynamics": {"available": True, "ready": True},
                }
            }
        },
        available_roles={
            "local-navigation-policy",
            "risk-critic",
            "payload-dynamics-adapter",
        },
    )

    assert decision.advisory_roles == ["risk-critic", "payload-dynamics-adapter"]
    assert "LOCAL_EXPERT_UNWIRED_ADVISOR_IGNORED" not in decision.reason_codes


# 功能：
#   验证动力学未就绪时不调用载荷适配器，并保留阻止运动的原因供策略端执行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_holds_payload_adapter_out_until_dynamics_are_fresh() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "payload": {
                    "state": "loaded-stable",
                    "dynamics": {"available": True, "ready": False},
                }
            }
        },
        available_roles={
            "local-navigation-policy",
            "risk-critic",
            "payload-dynamics-adapter",
        },
    )

    assert decision.advisory_roles == ["risk-critic"]
    assert "LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY" in decision.reason_codes


# 功能：
#   验证载荷已挂载但缺少合格专家时生成独立原因，不伪造专家调用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_marks_loaded_payload_without_qualified_adapter() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "payload": {
                    "state": "loaded-stable",
                    "dynamics": {"available": True, "ready": True},
                }
            }
        },
        available_roles={"local-navigation-policy", "risk-critic"},
    )

    assert "payload-dynamics-adapter" not in decision.advisory_roles
    assert "LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE" in decision.reason_codes


# 功能：
#   验证动力学可用不能替代载荷重量上限的独立验证。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_marks_explicitly_unverified_payload_limit() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "payload": {
                    "state": "loaded-stable",
                    "within_declared_payload_limit": False,
                    "dynamics": {"available": True, "ready": True},
                }
            }
        },
        available_roles={
            "local-navigation-policy",
            "risk-critic",
            "payload-dynamics-adapter",
        },
    )

    assert "LOCAL_EXPERT_PAYLOAD_LIMIT_NOT_VERIFIED" in decision.reason_codes


# 功能：
#   验证缺少专家与动力学过期分别报告，修复其中一项不能掩盖另一项。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_reports_missing_adapter_and_stale_payload_dynamics_independently() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "payload": {
                    "state": "loaded-stable",
                    "dynamics": {"available": True, "ready": False},
                }
            }
        },
        available_roles={"local-navigation-policy", "risk-critic"},
    )

    assert "LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE" in decision.reason_codes
    assert "LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY" in decision.reason_codes


# 功能：
#   验证悬停阶段本身不能证明已有载荷，不据此假造载荷专家输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_does_not_invent_payload_state_from_hover_phase() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "task": {"phase": "HOVER", "control_profile": "precision"},
                "payload": {"state": "no-runtime-payload-evidence"},
            }
        },
        available_roles={
            "local-navigation-policy",
            "payload-dynamics-adapter",
        },
    )

    assert "payload-dynamics-adapter" not in decision.advisory_roles


# 功能：
#   验证输入角色集合无序时，专家咨询及记录顺序仍固定。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_uses_stable_advisor_audit_order() -> None:
    decision = route_local_experts(
        {
            "strategic_context": {
                "task": {"phase": "HOVER", "control_profile": "precision"},
                "payload": {
                    "state": "loaded-stable",
                    "dynamics": {"available": True, "ready": True},
                },
            }
        },
        available_roles={
            "local-navigation-policy",
            "state-anomaly-detector",
            "payload-dynamics-adapter",
            "settle-stability-critic",
            "perception-health-critic",
            "risk-critic",
        },
    )

    assert decision.advisory_roles == [
        "risk-critic",
        "perception-health-critic",
        "settle-stability-critic",
        "payload-dynamics-adapter",
        "state-anomaly-detector",
    ]


# 功能：
#   验证缺少精细操作专家时保留回退标记，默认禁止由普通模型冒充该专家。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_falls_back_audibly_without_untrained_precision_expert() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "phase": "PICKUP",
                "control_profile": "precision",
                "decision_trigger": "periodic",
            }
        }
    }

    decision = route_local_experts(
        snapshot,
        available_roles={"local-navigation-policy", "risk-critic"},
    )

    assert decision.requested_navigation_role == "precision-maneuver-policy"
    assert decision.selected_navigation_role == "local-navigation-policy"
    assert decision.fallback_used is True
    assert not decision.motion_permitted
    assert "LOCAL_EXPERT_REQUIRED_SPECIALIST_UNAVAILABLE" in decision.reason_codes


# 功能：
#   验证停稳阶段优先选择精细控制专家，即使粗粒度配置标为巡航。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_uses_precision_for_waypoint_settling() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "phase": "WAYPOINT_SETTLE",
                "control_profile": "transit",
                "decision_trigger": "periodic",
            }
        }
    }

    decision = route_local_experts(
        snapshot,
        available_roles={
            "local-navigation-policy",
            "precision-maneuver-policy",
            "risk-critic",
            "cross-modal-consistency-critic",
        },
    )

    assert decision.selected_navigation_role == "precision-maneuver-policy"
    assert decision.advisory_roles == [
        "risk-critic",
        "cross-modal-consistency-critic",
    ]


# 功能：
#   验证正常局部重规划保留精细控制专家，不错误解释为恢复事件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_keeps_normal_local_replan_with_precision_expert() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "phase": "LOCAL_REPLAN",
                "control_profile": "precision",
                "decision_trigger": "periodic",
            }
        }
    }

    decision = route_local_experts(
        snapshot,
        available_roles={
            "local-navigation-policy",
            "precision-maneuver-policy",
            "recovery-policy",
            "risk-critic",
        },
    )

    assert decision.selected_navigation_role == "precision-maneuver-policy"


# 功能：
#   验证局部重规划停滞时调用恢复专家，同时保留精细工况需要的稳定性专家。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_uses_recovery_for_stalled_local_replan() -> None:
    snapshot = {
        "strategic_context": {
            "task": {
                "phase": "LOCAL_REPLAN",
                "control_profile": "precision",
                "decision_trigger": "progress-stalled",
            }
        }
    }

    decision = route_local_experts(
        snapshot,
        available_roles={
            "local-navigation-policy",
            "precision-maneuver-policy",
            "recovery-policy",
            "risk-critic",
            "settle-stability-critic",
        },
    )

    assert decision.selected_navigation_role == "recovery-policy"
    assert decision.advisory_roles == ["risk-critic", "settle-stability-critic"]


# 功能：
#   验证普通策略回退必须由真正布尔值明确配置，字符串不能开启权限。
# 输入：
#   value：非法回退配置值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", ["false", "true", 1, 0, None, [], {}])
def test_fallback_permission_is_not_coerced(value):
    with pytest.raises(ValueError, match="FALLBACK"):
        route_local_experts({"strategic_context": {"task": {"phase": "PICKUP"}}},
                            available_roles={"local-navigation-policy", "risk-critic"},
                            allow_general_fallback=value)


# 功能：
#   验证损坏的上下文不会静默降级为空任务再选择普通巡航模型。
# 输入：
#   snapshot：结构错误的导航快照。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("snapshot", [
    [], None, {"strategic_context": []}, {"strategic_context": {"task": "HOVER"}},
    {"strategic_context": {"payload": []}},
    {"strategic_context": {"payload": {"dynamics": "ready"}}},
])
def test_malformed_context_does_not_become_transit(snapshot):
    with pytest.raises(ValueError, match="CONTEXT"):
        route_local_experts(snapshot, available_roles={"local-navigation-policy", "risk-critic"})


# 功能：
#   验证角色集合类型错误时在路由边界明确拒绝，不触发散列异常或拆分字符串。
# 输入：
#   roles：非法角色集合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("roles", [None, "local-navigation-policy", [["local-navigation-policy"]],
                                  {"local-navigation-policy": True}, ["risk-critic", 1]])
def test_available_roles_require_a_collection_of_names(roles):
    with pytest.raises(ValueError, match="ROLES"):
        route_local_experts({}, available_roles=roles)


# 功能：
#   验证真实布尔配置开启兼容回退时留下明确标记，且不隐去风险专家要求。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_explicit_compatibility_fallback_preserves_attribution():
    decision = route_local_experts({"strategic_context": {"task": {"phase": "HOVER"}}},
                                  available_roles={"local-navigation-policy", "risk-critic"},
                                  allow_general_fallback=True)
    assert decision.motion_permitted
    assert decision.fallback_used
    assert "LOCAL_EXPERT_GENERAL_FALLBACK" in decision.reason_codes
