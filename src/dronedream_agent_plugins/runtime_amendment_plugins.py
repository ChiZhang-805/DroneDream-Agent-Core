"""Refine runtime requests without inventing device endpoints or actuation authority."""

from __future__ import annotations

import re
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.runtime_language import affirmative_phrase_present, phrase_mentioned
from dronedream_agent_core.runtime_payload_binding import resolve_runtime_payload
from dronedream_plugin_sdk.protocol import copy_json

from ._helpers import bounded_number, hook_plugin

ACTION_PATTERNS = [
    ("safe_land", ("安全降落", "立即降落", "land now", "safe land")),
    (
        "operator_release",
        ("释放控制权", "退出接管", "交还控制", "release control", "give back control"),
    ),
    ("set_return_point", ("返航点", "返回点", "return point")),
    ("return_home", ("返航", "回来", "return home", "come back")),
    ("set_speed", ("快一点", "慢一点", "速度", "speed", "faster", "slower")),
    ("set_coverage", ("覆盖范围", "coverage")),
    ("camera_control", ("拍照", "录像", "相机", "camera", "record")),
    ("payload_control", ("释放载荷", "放下", "payload", "release")),
    ("set_avoidance", ("避障", "绕开", "avoid")),
    ("follow_target", ("跟随", "follow")),
    ("operator_takeover", ("接管", "人工控制", "takeover")),
    (
        "redirect",
        (
            "改去",
            "改到",
            "改道",
            "换到",
            "换成",
            "改变目的地",
            "不要继续去",
            "instead",
            "reroute",
            "redirect",
            "change destination",
        ),
    ),
    # 「先悬停，再改道」是带安全前提的改道，不是永久暂停，因此具体改令优先于暂停词。
    ("pause", ("暂停", "先停", "悬停", "pause", "hold")),
    ("resume", ("继续", "恢复", "resume", "continue")),
]


# 功能：
#   仅从明确肯定、非延后的动作短语补全一个参数；冲突时要求重新理解，不能按顺序猜测。
#   模型已填入的参数也须服从同样的否定、延后和冲突检查，不能绕过文字约束。
# 输入：
#   text：已经规范化的用户文字。
#   choices：参数取值与明确动作短语的映射。
#   selected：模型已填入的参数取值，None 表示尚未提供。
# 输出：
#   choice：唯一肯定的参数取值，没有可靠匹配时为 None。
def _explicit_parameter_choice(
    text: str, choices: dict[str, tuple[str, ...]], *, selected: object = None
) -> str | None:
    if selected is not None and not isinstance(selected, str):
        raise ValueError("AMENDMENT_PERIPHERAL_COMMAND_INVALID")
    matches = [
        value
        for value, phrases in choices.items()
        if any(affirmative_phrase_present(text, phrase, allow_deferred=False) for phrase in phrases)
    ]
    if len(matches) > 1:
        raise ValueError("AMENDMENT_PERIPHERAL_COMMAND_AMBIGUOUS")
    if (
        selected in choices
        and selected not in matches
        and any(phrase_mentioned(text, phrase) for phrase in choices[selected])
    ):
        raise ValueError("AMENDMENT_PERIPHERAL_COMMAND_NEGATED_OR_DEFERRED")
    if selected is not None and matches and selected != matches[0]:
        raise ValueError("AMENDMENT_PERIPHERAL_COMMAND_CONTRADICTS_REQUEST")
    choice = matches[0] if matches else None
    return choice


# 功能：
#   1. 对模型改令结果施加保守词法约束，否定或询问不能触发即时动作。
#   2. 仅修复已知类别歧义，补全明确速度及外围动作参数，不凭名词新增不可逆控制。
#   3. 返回结构化建议，实际悬停确认、任务身份、权限与设备读回仍由核心检查。
# 输入：
#   value：模型已生成的改令分类及参数。
#   message：包含原始 text 的运行期用户消息。
#   prepared：当前冻结任务，用于解析真实返航地点。
#   _：分类器不使用的扩展参数。
# 输出：
#   classification：独立的分类结果、参数及计划修订要求。
def _classify(
    *, value: dict[str, object], message: Any, prepared: Any, **_: Any
) -> dict[str, object]:
    normalized = message.text.casefold()
    value = copy_json(value)
    action = str(value.get("requested_action", "replan"))
    # 动作只以否定形式出现时撤回错误分类，由核心保持安全状态并继续理解。
    for candidate, patterns in ACTION_PATTERNS:
        if (
            candidate == action
            and any(phrase_mentioned(normalized, p) for p in patterns)
            and not any(
                affirmative_phrase_present(normalized, p, allow_deferred=candidate != "safe_land")
                for p in patterns
            )
        ):
            action = "replan"
            value["parameters"] = {}
            break
    for candidate, patterns in ACTION_PATTERNS:
        if any(
            affirmative_phrase_present(normalized, pattern, allow_deferred=candidate != "safe_land")
            for pattern in patterns
        ):
            # 仅修复明确的类别歧义；改道时提及速度不能吞掉改道目标。
            if (
                candidate != action
                and action not in {"replan", "adjust_motion", "pause", "resume"}
                and (candidate, action)
                not in {
                    ("set_return_point", "return_home"),
                    ("operator_release", "payload_control"),
                }
            ):
                continue
            # 模型已提供结构化语义，外围名词或未来降落步骤不得升级成即时不可逆操作。
            if (
                candidate
                in {
                    "safe_land",
                    "camera_control",
                    "payload_control",
                    "set_avoidance",
                }
                and candidate != action
            ):
                continue
            action = candidate
            break
    parameters = copy_json(value.get("parameters", {}))
    if not isinstance(parameters, dict):
        raise ValueError("AMENDMENT_PARAMETERS_NOT_OBJECT")
    speeds = re.findall(
        r"(?<![0-9A-Za-z_.+-])([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:m/s|米每秒)",
        normalized,
    )
    if action == "set_speed" and speeds:
        values = {float(speed) for speed in speeds}
        if len(values) != 1 or not bounded_number(next(iter(values)), 0.1, 3):
            raise ValueError("AMENDMENT_SPEED_AMBIGUOUS_OR_OUT_OF_RANGE")
        parameters["maximum_speed_mps"] = next(iter(values))
    if action == "camera_control":
        command = _explicit_parameter_choice(
            normalized,
            {
                "stop_video": ("停止录像", "停止录制", "stop video"),
                "start_video": ("开始录像", "开始录制", "start video", "record"),
                "take_photo": ("拍照", "take photo"),
            },
            selected=parameters.get("command"),
        )
        if "command" not in parameters and command is not None:
            parameters["command"] = command
    if action == "payload_control":
        operation = _explicit_parameter_choice(
            normalized,
            {
                "detach": ("释放", "放下", "卸载", "detach", "release"),
                "attach": ("抓取", "挂载", "拿起", "attach", "pickup"),
            },
            selected=parameters.get("operation"),
        )
        if "operation" not in parameters and operation is not None:
            parameters["operation"] = operation
    if action == "set_avoidance":
        selected_mode = None
        if "enabled" in parameters:
            if type(parameters["enabled"]) is not bool:
                raise ValueError("AMENDMENT_PERIPHERAL_COMMAND_INVALID")
            selected_mode = "enabled" if parameters["enabled"] else "disabled"
        mode = _explicit_parameter_choice(
            normalized,
            {
                "disabled": ("关闭避障", "禁用避障", "disable avoidance"),
                "enabled": ("开启避障", "打开避障", "启用避障", "enable avoidance"),
            },
            selected=selected_mode,
        )
        if "enabled" not in parameters and mode is not None:
            parameters["enabled"] = mode == "enabled"
    target_entity = value.get("target_entity")
    if action == "return_home":
        target_entity = prepared.contract.return_node
    classification = {
        **value,
        "requested_action": action,
        "target_entity": target_entity,
        "parameters": parameters,
        "requires_plan_revision": action
        not in {
            "resume",
            "pause",
            "safe_land",
            "operator_release",
        },
    }
    return classification


# 功能：
#   校验改令参数并从冻结任务解析载荷接口，生成无执行权限的指令，问题码必须由核心处理。
# 输入：
#   classification：经过分类的结构化改令。
#   acknowledgement：宿主统一传入的悬停回执，本钩子不据此授予权限。
#   prepared：冻结任务及其资产派生的运行动作绑定。
#   _：保守策略不使用的扩展参数。
# 输出：
#   directive：动作、独立参数、核心授权要求及参数问题码。
def _conservative_directive(
    *, classification: Any, acknowledgement: Any, prepared: Any, **_: Any
) -> dict[str, object]:
    action = classification.requested_action
    parameters = copy_json(classification.parameters)
    issues: list[str] = []
    if action == "set_speed":
        speed = parameters.get("maximum_speed_mps")
        if speed is None:
            issues.append("SPEED_VALUE_REQUIRED")
        elif not bounded_number(speed, 0.1, 3):
            issues.append("SPEED_OUTSIDE_QUALIFIED_ENVELOPE")
    if action in {"redirect", "set_return_point", "follow_target"} and not (
        classification.target_entity
    ):
        issues.append("TARGET_ENTITY_REQUIRED")
    if (
        action == "set_coverage"
        and not classification.target_entity
        and not parameters.get("polygon_enu_m")
    ):
        issues.append("COVERAGE_TARGET_OR_POLYGON_REQUIRED")
    if action == "camera_control" and parameters.get("command") not in {
        "take_photo",
        "start_video",
        "stop_video",
    }:
        issues.append("CAMERA_COMMAND_REQUIRED")
    if action == "camera_control":
        parameters.setdefault("component_id", 100)
    if action == "payload_control":
        try:
            parameters = resolve_runtime_payload(prepared, parameters)
        except ValueError as error:
            issues.append(str(error))
    if action == "set_avoidance" and not isinstance(parameters.get("enabled"), bool):
        issues.append("AVOIDANCE_BOOLEAN_REQUIRED")
    if action == "follow_target":
        topic = parameters.get("target_pose_topic")
        if not isinstance(topic, str) or not topic.startswith("/"):
            issues.append("FOLLOW_TARGET_POSE_TOPIC_REQUIRED")
        parameters.setdefault("follow_duration_seconds", 30.0)
        parameters.setdefault("standoff_m", 2.0)
        parameters.setdefault("altitude_offset_m", 1.0)
        parameters.setdefault("maximum_speed_mps", 1.0)
        parameters.setdefault("target_update_rate_hz", 2.0)
        numeric_limits = {
            "follow_duration_seconds": (1.0, 300.0),
            "standoff_m": (0.5, 20.0),
            "altitude_offset_m": (-5.0, 20.0),
            "maximum_speed_mps": (0.1, 3.0),
            "target_update_rate_hz": (0.5, 10.0),
        }
        for name, (minimum, maximum) in numeric_limits.items():
            value = parameters.get(name)
            if not bounded_number(value, minimum, maximum):
                issues.append(f"FOLLOW_{name.upper()}_INVALID")
    directive = {
        "action": action,
        "parameters": parameters,
        "requires_stable_hold": True,
        "requires_plan_revision": classification.requires_plan_revision,
        "requires_core_authorization": True,
        "issue_codes": issues,
    }
    return directive


# 功能：
#   注册核心授权前的分类收紧和参数检查钩子，切换仅允许发生于安全悬停阶段。
# 输入：
#   无。
# 输出：
#   definitions：运行改令分类与保守应用策略插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        hook_plugin(
            module_name=__name__,
            plugin_id="runtime.amendment-language-classifier",
            name="运行期改令分类",
            description="把改目的地、速度、暂停、返航、相机、载荷和接管等指令规范化。",
            capability_id="runtime.amendment-language-classifier.classify",
            capability_kind="runtime-amendment",
            capability_name="运行期改令分类",
            capability_description="模型分类后再由确定性语言规则收紧动作类型。",
            category_id="runtime",
            category_label="运行期与在线换路",
            slot_id="runtime.amendment-classifier",
            slot_label="改令分类",
            activation_mode="single",
            category_order=70,
            slot_order=10,
            plugin_order=10,
            hooks={"classify_amendment": _classify},
            default_enabled=True,
            failure_mode="fail-closed",
            swap_policy="safe-hold",
        ),
        hook_plugin(
            module_name=__name__,
            plugin_id="runtime.amendment-conservative",
            name="保守改令策略",
            description="所有改令先稳定悬停，再检查参数、计划修订和核心授权。",
            capability_id="runtime.amendment-conservative.apply",
            capability_kind="runtime-amendment",
            capability_name="保守改令策略",
            capability_description="生成无执行权限的结构化改令指令。",
            category_id="runtime",
            category_label="运行期与在线换路",
            slot_id="runtime.amendment-policy",
            slot_label="改令应用策略",
            activation_mode="single",
            category_order=70,
            slot_order=20,
            plugin_order=10,
            hooks={"apply_amendment": _conservative_directive},
            default_enabled=True,
            failure_mode="fail-closed",
            swap_policy="safe-hold",
        ),
    ]
    return definitions
