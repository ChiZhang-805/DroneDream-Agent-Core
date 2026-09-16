"""Render detached timeline messages; publishing or operator confirmation happens elsewhere."""

from __future__ import annotations

import sys
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import bounded_number, hook_plugin


# 功能：
#   复制有界通知输入并检查必填文字，避免任意对象被渲染或可变元数据与调用方共享。
# 输入：
#   value：准备渲染的通知摘要。
#   text_fields：必须存在且非空的文本字段名。
# 输出：
#   detached：通过文本检查且解除共享引用的摘要。
def _summary(value: dict[str, object], text_fields: tuple[str, ...]) -> dict[str, object]:
    detached = copy_json(value)
    if not isinstance(detached, dict) or any(
        not isinstance(detached.get(key), str) or not detached[key].strip() for key in text_fields
    ):
        raise ValueError("NOTIFICATION_SUMMARY_INVALID")
    return detached


# 功能：
#   渲染绑定合同、计划修订及插件快照的就绪消息；通知本身不授权起飞。
# 输入：
#   summary：含计划目标、身份字段及可选语言设置的摘要。
#   _：渲染器不使用的扩展参数。
# 输出：
#   notification：任务时间线消息及对应身份元数据。
def _plan_ready(*, summary: dict[str, object], **_: Any) -> dict[str, object]:
    summary = _summary(summary, ("goal", "contract_id", "plan_revision_id", "plugin_snapshot_id"))
    english = summary.get("locale") == "en-US"
    notification = {
        "channel": "task-timeline",
        "kind": "plan",
        "content": (
            f"Plan ready: {summary['goal']}" if english else f"计划已生成：{summary['goal']}"
        ),
        "metadata": {
            "contract_id": summary["contract_id"],
            "plan_revision_id": summary["plan_revision_id"],
            "plugin_snapshot_id": summary["plugin_snapshot_id"],
        },
    }
    return notification


# 功能：
#   显示有限净空及非负调用计数，拒绝把字符串或布尔值强转成已测指标。
# 输入：
#   summary：含净空、模型调用数、规划次数和目录摘要的通知数据。
#   _：指标渲染器不使用的扩展参数。
# 输出：
#   notification：指标说明和规划来源元数据。
def _planning_metrics(*, summary: dict[str, object], **_: Any) -> dict[str, object]:
    summary = _summary(summary, ("plugin_catalog_sha256",))
    if not bounded_number(
        summary.get("minimum_clearance_m"), -sys.float_info.max, sys.float_info.max
    ) or any(
        type(summary.get(key)) is not int or summary[key] < 0
        for key in ("model_calls", "planning_attempts")
    ):
        raise ValueError("NOTIFICATION_METRICS_INVALID")
    english = summary.get("locale") == "en-US"
    notification = {
        "channel": "task-timeline",
        "kind": "status",
        "content": (
            f"Planning evidence: {summary['model_calls']} model calls, "
            f"minimum clearance {float(summary['minimum_clearance_m']):.2f} m"
            if english
            else f"规划证据：{summary['model_calls']} 次模型调用，"
            f"最小净空 {float(summary['minimum_clearance_m']):.2f} m"
        ),
        "metadata": {
            "planning_attempts": summary["planning_attempts"],
            "plugin_catalog_sha256": summary["plugin_catalog_sha256"],
        },
    }
    return notification


# 功能：
#   提醒用户核对任务与资产，不生成代替用户的确认记录或已完成检查标记。
# 输入：
#   summary：目标、返程地点、合同标识及可选语言设置。
#   _：提醒渲染器不使用的扩展参数。
# 输出：
#   notification：执行前提醒及相关任务元数据。
def _operator_checklist(*, summary: dict[str, object], **_: Any) -> dict[str, object]:
    summary = _summary(summary, ("target_entity", "return_entity", "contract_id"))
    english = summary.get("locale") == "en-US"
    notification = {
        "channel": "task-timeline",
        "kind": "status",
        "content": (
            "Before execution, confirm the target, return location, map, drone, "
            "and mission contract."
            if english
            else "执行前请确认目标、返程位置、地图、无人机和任务合同。"
        ),
        "metadata": {
            "target_entity": summary["target_entity"],
            "return_entity": summary["return_entity"],
            "contract_id": summary["contract_id"],
        },
    }
    return notification


# 功能：
#   注册只读时间线渲染器，通知失败独立隔离，不改变任务执行权限。
# 输入：
#   无。
# 输出：
#   definitions：计划、指标和人工确认提醒的插件定义列表。
def plugin_definitions() -> list[PluginDefinition]:
    values = [
        (
            "notification.plan-ready",
            "计划就绪通知",
            "在任务对话时间线中发布与合同和插件快照绑定的计划就绪消息。",
            _plan_ready,
            True,
        ),
        (
            "notification.planning-metrics",
            "规划指标通知",
            "在时间线中补充模型调用、规划轮次和最小净空摘要。",
            _planning_metrics,
            False,
        ),
        (
            "notification.operator-checklist",
            "执行前确认提醒",
            "在计划生成后加入执行前人工确认清单。",
            _operator_checklist,
            False,
        ),
    ]
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id=plugin_id,
            name=name,
            description=description,
            capability_id=f"{plugin_id}.render",
            capability_kind="notification",
            capability_name=name,
            capability_description=description,
            category_id="interaction",
            category_label="交互与通知",
            slot_id="notifications.plan-ready",
            slot_label="计划就绪通知",
            activation_mode="multiple",
            category_order=90,
            slot_order=20,
            plugin_order=index * 10,
            hooks={"render_plan_notification": handler},
            default_enabled=enabled,
            failure_mode="isolate",
            swap_policy="anytime",
            permissions=["mission.read"],
        )
        for index, (plugin_id, name, description, handler, enabled) in enumerate(values, start=1)
    ]
    return definitions
