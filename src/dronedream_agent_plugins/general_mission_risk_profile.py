"""Qualitative contract-text hints, not calibrated probabilities or live obstacle detection."""

from __future__ import annotations

import re
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition, ToolEnvironment
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)
from dronedream_agent_core.tools import ToolPlugin

from ._helpers import constraint_text

INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["goal", "payload_action", "constraints"],
    "properties": {
        "goal": {"type": "string", "minLength": 1, "maxLength": 4000},
        "payload_action": {"type": "string", "minLength": 1, "maxLength": 120},
        "constraints": {
            "type": "array",
            "maxItems": 32,
            "items": {"type": "string", "maxLength": 4000},
        },
    },
}
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["risk_level", "risk_factors", "recommended_focus"],
    "properties": {
        "risk_level": {"type": "string", "enum": ["normal", "elevated", "high"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "recommended_focus": {"type": "array", "items": {"type": "string"}},
    },
}


# 功能：
#   根据目标、动作类别和约束提出审查关注点；定性等级不是风险概率，也不是实时环境识别。
# 输入：
#   value：包含 goal、payload_action 与 constraints 的合同文字片段。
# 输出：
#   profile：关注因素、定性等级与建议检查项。
def _profile(value: dict[str, object]) -> dict[str, Any]:
    constraints = constraint_text(value.get("constraints"))
    goal = value.get("goal")
    action = value.get("payload_action")
    if (
        not isinstance(goal, str)
        or not 1 <= len(goal.strip()) <= 4000
        or not isinstance(action, str)
        or not 1 <= len(action) <= 120
    ):
        raise ValueError("MISSION_ADVISOR_REQUEST_INVALID")
    rendered = f"{goal.lower()} {constraints}"
    factors: list[str] = []
    focus = ["telemetry continuity", "abort availability"]
    if action in {"pickup", "dropoff", "deliver", "release", "attach", "detach"} or (
        action.startswith("payload.")
    ):
        factors.append("payload interaction")
        focus.extend(["payload identity", "mass and retention confirmation"])
    elif action not in {
        "none",
        "navigate",
        "traverse",
        "return",
        "land",
        "takeoff",
        "hold",
        "wait",
    }:
        # 未识别领域动作需要单独核查，不能把导航动作或所有未知动作一概当作载荷交互。
        factors.append("unclassified domain action")
        focus.append("qualified action adapter and required evidence")
    if re.search(r"\b(?:indoors?|doors?)\b", rendered) or any(
        token in rendered for token in ("室内", "楼")
    ):
        factors.append("constrained indoor geometry")
        focus.append("continuous vehicle-envelope clearance")
    if re.search(r"\bfast\b", rendered) or any(token in rendered for token in ("快速", "尽快")):
        factors.append("time pressure")
        focus.append("speed must remain subordinate to safety margins")
    level = "high" if len(factors) >= 2 else "elevated" if factors else "normal"
    profile = {"risk_level": level, "risk_factors": factors, "recommended_focus": focus}
    return profile


# 功能：
#   暴露合同文字风险顾问，固定各字段来源，不访问仿真器或执行器。
# 输入：
#   _environment：通用工具工厂提供的宿主环境，本工具不使用。
# 输出：
#   tools：带有参数契约与路由提示的只读工具列表。
def _tools(_environment: ToolEnvironment) -> list[ToolPlugin]:
    tools = [
        ToolPlugin(
            tool_id="general.mission-risk-profile",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA,
            handler=_profile,
            routing_metadata={
                "recommended_when": {"always": True},
                "domains": ["safety", "payload", "planning"],
                "purpose": "Derive a structured advisory risk profile from the mission contract.",
                "required_argument_sources": {
                    "goal": "mission_contract.goal",
                    "payload_action": "mission_contract.payload_action",
                    "constraints": "mission_contract.constraints",
                },
            },
        )
    ]
    return tools


# 功能：
#   注册定性风险顾问；其建议不能覆盖核心安全门控或变成人工批准。
# 输入：
#   无。
# 输出：
#   definition：只读风险顾问的清单、布局及工具工厂。
def plugin_definition() -> PluginDefinition:
    definition = PluginDefinition(
        manifest=PluginManifest(
            plugin_id="general.mission-risk-profile",
            name="任务风险画像",
            version="1.0.0",
            description="从任务目标、载荷动作和约束中补充结构化风险关注点。",
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definition"
            ),
            capabilities=[
                PluginCapability(
                    capability_id="general.mission-risk-profile",
                    kind="evidence",
                    name="任务风险画像",
                    description="生成不具有控制权的任务风险建议。",
                    input_schema=INPUT_SCHEMA,
                    output_schema=OUTPUT_SCHEMA,
                    metadata={
                        "recommended_when": {"always": True},
                        "domains": ["safety", "payload", "planning"],
                        "purpose": (
                            "Derive a structured advisory risk profile from the mission contract."
                        ),
                        "required_argument_sources": {
                            "goal": "mission_contract.goal",
                            "payload_action": "mission_contract.payload_action",
                            "constraints": "mission_contract.constraints",
                        },
                    },
                )
            ],
            permissions=["mission.read"],
            default_enabled=True,
            removable=False,
            placement=PluginPlacement(
                category_id="general",
                category_label="通用增强",
                slot_id="general.mission-advisors",
                slot_label="任务顾问",
                activation_mode="multiple",
                scope="general",
                category_order=0,
                slot_order=10,
                plugin_order=10,
            ),
        ),
        tool_factory=_tools,
    )
    return definition
