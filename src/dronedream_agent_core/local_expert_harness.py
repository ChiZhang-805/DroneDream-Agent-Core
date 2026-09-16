"""Deterministic routing for qualified local flight experts.

The router is deliberately not a learned model.  It binds the current mission
phase to one qualified navigation expert, records every fallback, and lists the
independent advisors that must be consulted. Continuous policies propose body
velocity and yaw-rate axes. Candidate selection remains an explicitly declared
compatibility mode; neither mode can bypass deterministic collision limits.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

from pydantic import Field

from .contracts import StrictModel

NavigationExpertRole = Literal[
    "local-navigation-policy",
    "precision-maneuver-policy",
    "recovery-policy",
]
AdvisoryExpertRole = Literal[
    "risk-critic",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
]

NAVIGATION_EXPERT_ROLES: tuple[NavigationExpertRole, ...] = (
    "local-navigation-policy",
    "precision-maneuver-policy",
    "recovery-policy",
)
ADVISORY_EXPERT_ROLES: tuple[AdvisoryExpertRole, ...] = (
    "risk-critic",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
)
# Only roles in this tuple are allowed to appear as invoked runtime advisors.
# The ordering is also the stable audit order used by model-call receipts.
RUNTIME_ADVISORY_EXPERT_ROLES: tuple[AdvisoryExpertRole, ...] = (
    "risk-critic",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
)

_PRECISION_PHASES = {
    "TAKEOFF",
    "CHECKPOINT",
    "ACTION",
    "PICKUP",
    "HOVER",
    "WAYPOINT_SETTLE",
    "LOCAL_SLOW",
    "LAND",
    "LANDING",
}
_RECOVERY_TRIGGERS = {"progress-stalled", "dynamic-obstacle"}


class LocalExpertRoutingDecision(StrictModel):
    """One auditable Harness decision about which local experts may run."""

    requested_navigation_role: NavigationExpertRole
    selected_navigation_role: NavigationExpertRole
    advisory_roles: list[AdvisoryExpertRole] = Field(default_factory=list, max_length=6)
    runtime_phase: str = Field(default="UNKNOWN", min_length=1, max_length=64)
    control_profile: Literal["transit", "precision", "unknown"] = "unknown"
    decision_trigger: str = Field(default="unknown", min_length=1, max_length=64)
    fallback_used: bool = False
    motion_permitted: bool = True
    reason_codes: list[str] = Field(min_length=1, max_length=8)


# 功能：
#   读取可选上下文对象；缺省或 null 表示未提供，但错误容器类型不能伪装成空任务。
# 输入：
#   value：待读取的可选上下文。
# 输出：
#   mapping：上下文的浅层副本。
def _mapping(value: object) -> dict[str, object]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("LOCAL_EXPERT_CONTEXT_INVALID")
    mapping = {} if value is None else dict(value)
    return mapping


# 功能：
#   读取有界上下文标签，不把数组、数值或对象的字符串表示当作任务语义。
# 输入：
#   value：可选标签值。
#   default：缺省或空白标签的替代值。
# 输出：
#   label：去除两端空白后的标签。
def _context_label(value: object, default: str) -> str:
    if value is None:
        return default
    if type(value) is not str or len(value) > 64:
        raise ValueError("LOCAL_EXPERT_CONTEXT_LABEL_INVALID")
    label = value.strip() or default
    return label


# 功能：
#   独立于导航专家识别精细飞行工况，使恢复动作仍保留停稳等约束。
# 输入：
#   task：当前任务的阶段、控制配置及检查点标志。
# 输出：
#   precision：当前是否属于精细操作工况。
def _precision_context(task: Mapping[str, object]) -> bool:
    precision = (
        _context_label(task.get("control_profile"), "unknown").lower() == "precision"
        or task.get("action_checkpoint_goal") is True
        or _context_label(task.get("phase"), "UNKNOWN").upper() in _PRECISION_PHASES
    )
    return precision


# 功能：
#   根据任务语义选择所需导航专家，不以已安装模型反推需求，也不授予执行权限。
# 输入：
#   snapshot：导航快照及其中的任务上下文。
# 输出：
#   requested：普通导航、精细操作或恢复专家角色。
def requested_navigation_expert(snapshot: Mapping[str, object]) -> NavigationExpertRole:
    if not isinstance(snapshot, Mapping):
        raise ValueError("LOCAL_EXPERT_CONTEXT_INVALID")
    strategic = _mapping(snapshot.get("strategic_context"))
    task = _mapping(strategic.get("task"))
    trigger = _context_label(task.get("decision_trigger"), "unknown").lower()

    # Executor phase names describe controller lifecycle, not proof that the
    # vehicle is in one semantic recovery incident.  In particular,
    # TRACKING_RECOVERY is also used while the ordinary tracker reacquires a
    # moving setpoint.  Route to the recovery specialist only for an explicit,
    # identity-bearing stall or dynamic-obstacle trigger; the runtime owns the
    # corresponding recovery_episode_id.
    if trigger in _RECOVERY_TRIGGERS:
        requested: NavigationExpertRole = "recovery-policy"
    elif _precision_context(task):
        requested = "precision-maneuver-policy"
    else:
        requested = "local-navigation-policy"
    return requested


# 功能：
#   1. 选定一个运动提议专家及适用的独立咨询专家，按固定顺序记录调用需求。
#   2. 缺少专用专家时只在显式兼容配置下回退，缺少风险专家时禁止运动。
#   3. 记录载荷验证缺口供策略端执行；motion_permitted 只表示路由层允许，
#      策略端仍须执行载荷、时序、物理安全门槛，不能据本结果直接驱动飞控。
# 输入：
#   snapshot：当前任务、飞行阶段及载荷上下文。
#   available_roles：合格模型包加载器提供的可用角色集合。
#   allow_general_fallback：是否明确允许兼容模式使用普通导航专家回退。
# 输出：
#   decision：所需与选中专家、咨询角色、回退信息及后续门槛原因。
def route_local_experts(
    snapshot: Mapping[str, object],
    *,
    available_roles: Collection[str],
    allow_general_fallback: bool = False,
) -> LocalExpertRoutingDecision:
    if not isinstance(snapshot, Mapping):
        raise ValueError("LOCAL_EXPERT_CONTEXT_INVALID")
    if type(allow_general_fallback) is not bool:
        raise ValueError("LOCAL_EXPERT_FALLBACK_FLAG_INVALID")
    if (not isinstance(available_roles, Collection)
            or isinstance(available_roles, (str, bytes, Mapping))
            or any(type(role) is not str or not role or len(role) > 80
                   for role in available_roles)):
        raise ValueError("LOCAL_EXPERT_AVAILABLE_ROLES_INVALID")
    available = set(available_roles)
    if "local-navigation-policy" not in available:
        raise ValueError("local expert routing requires a general navigation policy")

    strategic = _mapping(snapshot.get("strategic_context"))
    task = _mapping(strategic.get("task"))
    phase = _context_label(task.get("phase"), "UNKNOWN").upper()
    raw_profile = _context_label(task.get("control_profile"), "unknown").lower()
    profile: Literal["transit", "precision", "unknown"] = (
        raw_profile if raw_profile in {"transit", "precision"} else "unknown"
    )
    trigger = _context_label(task.get("decision_trigger"), "unknown").lower()
    requested = requested_navigation_expert(snapshot)
    fallback = requested not in available
    selected: NavigationExpertRole = (
        "local-navigation-policy" if fallback else requested
    )
    reasons = [f"LOCAL_EXPERT_REQUESTED_{requested.upper().replace('-', '_')}"]
    if fallback:
        reasons.append("LOCAL_EXPERT_GENERAL_FALLBACK" if allow_general_fallback
                       else "LOCAL_EXPERT_REQUIRED_SPECIALIST_UNAVAILABLE")
    else:
        reasons.append("LOCAL_EXPERT_REQUEST_SATISFIED")
    payload = _mapping(strategic.get("payload"))
    payload_state = _context_label(payload.get("state"), "").lower()
    payload_relevant = payload_state in {
        "attached",
        "custody-confirmed",
        "loaded-stable",
    }
    payload_dynamics = _mapping(payload.get("dynamics"))
    payload_dynamics_ready = bool(
        payload_dynamics.get("available") is True
        and payload_dynamics.get("ready") is True
    )
    payload_adapter_available = "payload-dynamics-adapter" in available
    if payload_relevant:
        # Absence, null, numeric truthiness and textual "true" are not evidence.
        if payload.get("within_declared_payload_limit") is not True:
            reasons.append("LOCAL_EXPERT_PAYLOAD_LIMIT_NOT_VERIFIED")
        if not payload_adapter_available:
            reasons.append("LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE")
        if not payload_dynamics_ready:
            reasons.append("LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY")
    # A recovery vote does not erase the flight regime that made precision
    # control necessary.  Keep the temporal settle critic active when a
    # stalled pickup, hover, landing, or tight-space maneuver is routed to the
    # recovery specialist.
    precision_relevant = _precision_context(task)
    advisors = [
        role
        for role in RUNTIME_ADVISORY_EXPERT_ROLES
        if role in available
        and (
            role != "payload-dynamics-adapter"
            or (payload_relevant and payload_dynamics_ready)
        )
        and (role != "settle-stability-critic" or precision_relevant)
    ]
    if "risk-critic" not in advisors:
        reasons.append("LOCAL_EXPERT_RISK_CRITIC_UNAVAILABLE")
    decision = LocalExpertRoutingDecision(
        requested_navigation_role=requested,
        selected_navigation_role=selected,
        advisory_roles=advisors,
        runtime_phase=phase,
        control_profile=profile,
        decision_trigger=trigger,
        fallback_used=fallback,
        motion_permitted=(not fallback or allow_general_fallback) and "risk-critic" in advisors,
        reason_codes=reasons,
    )
    return decision
