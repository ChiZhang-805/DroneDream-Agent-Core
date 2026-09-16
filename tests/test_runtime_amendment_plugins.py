from __future__ import annotations

from types import SimpleNamespace

import pytest

from dronedream_agent_core.runtime_interrupt import _emergency_override_requested
from dronedream_agent_core.runtime_payload_binding import resolve_runtime_payload
from dronedream_agent_plugins.runtime_amendment_plugins import _classify, _conservative_directive


# 功能：
#   验证改道前的暂时安全悬停不吞掉新目的地，也不误成永久暂停。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_redirect_outranks_hover_and_continue_safety_wording() -> None:
    value = {
        "message_kind": "mission_amendment",
        "requested_action": "pause",
        "target_entity": "校园门口的保安亭",
        "requires_plan_revision": False,
        "summary": "Change destination after stable hover.",
        "parameters": {},
    }
    message = SimpleNamespace(
        text=(
            "不要继续去原来的外卖点了。我的外卖现在在校园门口的保安亭，"
            "请先安全悬停，再根据当前位置改道去校园门口，确认安全后继续执行。"
        )
    )
    prepared = SimpleNamespace(contract=SimpleNamespace(return_node="office"))

    classified = _classify(
        value=value,
        message=message,
        prepared=prepared,
    )

    assert classified["requested_action"] == "redirect"
    assert classified["target_entity"] == "校园门口的保安亭"
    assert classified["requires_plan_revision"] is True


# 功能：
#   验证交还人工控制权与释放实物载荷分开，类别修正不能补出载荷操作参数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_release_control_is_not_misclassified_as_payload_release() -> None:
    value = {
        "message_kind": "informational",
        "requested_action": "payload_control",
        "target_entity": None,
        "requires_plan_revision": True,
        "summary": "Operator requests control release.",
        "parameters": {},
    }
    classified = _classify(
        value=value,
        message=SimpleNamespace(text="我退出接管，释放控制权并让任务继续"),
        prepared=SimpleNamespace(contract=SimpleNamespace(return_node="office")),
    )

    assert classified["requested_action"] == "operator_release"
    assert classified["requires_plan_revision"] is False
    assert "operation" not in classified["parameters"]


# 功能：
#   使用最小模型提案运行词法收紧，不加载真实设备或赋予执行权限。
# 输入：
#   text：用户改令原文。
#   action：模型先前选择的动作。
#   parameters：模型提议的可选参数。
# 输出：
#   classification：正式分类器生成的独立分类结果。
def _classified(text: str, action: str, parameters=None):
    classification = _classify(
        value={
            "requested_action": action,
            "parameters": parameters or {},
            "target_entity": "new-place",
        },
        message=SimpleNamespace(text=text),
        prepared=SimpleNamespace(contract=SimpleNamespace(return_node="home")),
    )
    return classification


# 功能：
#   验证否定、询问及到达后的降落表述都不会触发紧急立即降落覆盖。
# 输入：
#   text：不应视为立即降落授权的文字。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "text",
    [
        "到达目标后安全降落",
        "after reaching the office, then land now",
        "不要立即降落",
        "do not land now",
        "should we land now?",
        "land now?",
        "到达目标后，立即降落",
        "after arriving at the office, land now",
    ],
)
def test_negated_deferred_or_questioned_landing_is_not_immediate_override(text) -> None:
    assert not _emergency_override_requested(text)


# 功能：
#   验证明确立即降落要求保留核心紧急覆盖通道，保守过滤不吞掉这些有效表述。
# 输入：
#   text：明确的紧急降落文字。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("text", ["land now", "请立即降落", "请立即停止并降落"])
def test_explicit_emergency_retains_core_override(text) -> None:
    assert _emergency_override_requested(text)


# 功能：
#   验证返航点编辑不是立刻返航，改道中的速度与未来降落说明不取代改道动作。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_return_point_edit_is_not_immediate_return_and_speed_does_not_hijack_redirect() -> None:
    assert _classified("修改返航点为操场", "return_home")["requested_action"] == "set_return_point"
    assert _classified("改去操场，控制好速度", "redirect")["requested_action"] == "redirect"
    assert _classified("去操场后安全降落", "redirect")["requested_action"] == "redirect"


# 功能：
#   验证名词提及不升级成外围动作，明确否定的载荷释放会撤回并清空操作参数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_keyword_mentions_do_not_promote_irreversible_commands() -> None:
    assert _classified("查看 payload 的当前情况", "replan")["requested_action"] == "replan"
    result = _classified("do not release the payload", "payload_control", {"operation": "detach"})
    assert result["requested_action"] == "replan"
    assert result["parameters"] == {}
    assert _classified("show recording_status", "replan")["requested_action"] == "replan"


# 功能：
#   验证显式速度的负号不丢失，多个不同速度值必须报歧义而非擅选一个。
# 输入：
#   text：包含负速度或冲突速度值的改令。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("text", ["速度-2 m/s", "speed -2 m/s", "speed 1 m/s then 2 m/s"])
def test_speed_parser_never_drops_a_minus_sign_or_picks_one_of_conflicting_values(text) -> None:
    with pytest.raises(ValueError, match="SPEED_AMBIGUOUS_OR_OUT_OF_RANGE"):
        _classified(text, "set_speed")


# 功能：
#   验证合法米每秒解析、布尔速度拒绝和深层参数副本隔离都真实生效。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_speed_parser_and_directive_are_strict_without_aliasing_parameters() -> None:
    assert _classified("速度1.5米每秒", "set_speed")["parameters"]["maximum_speed_mps"] == 1.5
    parameters = {"maximum_speed_mps": True, "nested": [1]}
    classification = SimpleNamespace(
        requested_action="set_speed", parameters=parameters, requires_plan_revision=True
    )
    directive = _conservative_directive(
        classification=classification, acknowledgement=None, prepared=None
    )
    assert "SPEED_OUTSIDE_QUALIFIED_ENVELOPE" in directive["issue_codes"]
    directive["parameters"]["nested"].append(2)
    assert parameters["nested"] == [1]


# 功能：
#   构造含一项资产派生载荷绑定的冻结任务夹具，不连接仿真器。
# 输入：
#   parameters：测试运行动作的固定接口参数。
# 输出：
#   prepared：合同身份与运行动作一致的最小任务对象。
def _payload_mission(parameters: dict[str, object]):
    prepared = SimpleNamespace(
        contract=SimpleNamespace(contract_id="mission"),
        runtime_actions=SimpleNamespace(
            contract_id="mission",
            steps=[
                SimpleNamespace(
                    driver="gazebo-payload",
                    authority="actuate",
                    parameters=parameters,
                )
            ],
        ),
    )
    return prepared


# 功能：
#   验证载荷接口来自当前冻结任务且返回副本，缺失绑定或尝试换端点必须拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_payload_binding_uses_selected_mission_not_a_fixed_demo_vehicle() -> None:
    binding = {
        "operation": "attach",
        "protocol": "gazebo-transport",
        "topic": "/model/selected_vehicle/attach",
        "output_topic": "/selected/state",
    }
    prepared = _payload_mission(binding)
    result = resolve_runtime_payload(prepared, {"operation": "attach"})
    assert result == binding
    result["topic"] = "changed"
    assert binding["topic"] == "/model/selected_vehicle/attach"
    with pytest.raises(ValueError, match="FROZEN_PARAMETER_OVERRIDE"):
        resolve_runtime_payload(prepared, {"operation": "attach", "topic": "/another/vehicle"})
    with pytest.raises(ValueError, match="FROZEN_BINDING_REQUIRED"):
        resolve_runtime_payload(None, {"operation": "attach"})
    with pytest.raises(ValueError, match="FROZEN_BINDING_REQUIRED"):
        resolve_runtime_payload(prepared, {"operation": "detach"})


# 功能：
#   验证同一操作有两个不同冻结端点时不能让模型参数自行选择其中一个。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ambiguous_payload_bindings_cannot_be_selected_by_the_model() -> None:
    prepared = _payload_mission({"operation": "attach", "topic": "/one"})
    prepared.runtime_actions.steps.append(
        SimpleNamespace(
            driver="gazebo-payload",
            authority="actuate",
            parameters={"operation": "attach", "topic": "/two"},
        )
    )
    with pytest.raises(ValueError, match="FROZEN_BINDING_REQUIRED"):
        resolve_runtime_payload(prepared, {"operation": "attach", "topic": "/one"})


# 功能：
#   验证外围参数补全只采用明确肯定的即时动作，不借否定、未来步骤或标识符子串造命令。
# 输入：
#   text：包含相机、载荷或避障表述的用户文字。
#   action：模型已经提出的动作类别。
#   key：需要检查的具体控制参数。
#   expected：允许补全的值，None 表示必须保留缺失以交给后续门控拒绝。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "text,action,key,expected",
    [
        ("Use camera to take photo, do not stop video", "camera_control", "command", "take_photo"),
        (
            "Use camera to start video, do not take photo",
            "camera_control",
            "command",
            "start_video",
        ),
        ("Use camera to take photo after arriving", "camera_control", "command", None),
        ("camera recording_status", "camera_control", "command", None),
        ("检查载荷，不要释放，挂载载荷", "payload_control", "operation", "attach"),
        ("Do not disable avoidance", "set_avoidance", "enabled", None),
        ("Enable avoidance", "set_avoidance", "enabled", True),
        ("Disable avoidance", "set_avoidance", "enabled", False),
    ],
)
def test_peripheral_parameter_completion_respects_affirmation(text, action, key, expected):
    result = _classified(text, action)
    if expected is None:
        assert key not in result["parameters"]
    else:
        assert result["parameters"][key] == expected


# 功能：
#   验证一个即时参数槽出现多个肯定命令时不按关键词顺序擅自选取其中一个。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_conflicting_peripheral_commands_require_interpretation():
    with pytest.raises(ValueError, match="PERIPHERAL_COMMAND_AMBIGUOUS"):
        _classified("camera: take photo, start video", "camera_control")


# 功能：
#   验证模型已经填入的外围参数也不能保留与明确文字相反、被否定或延后的动作。
# 输入：
#   text：明确约束动作的用户文字。
#   action：模型提出的动作类别。
#   parameters：模型已填入但与文字矛盾的参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "text,action,parameters",
    [
        ("camera: take photo, do not stop video", "camera_control", {"command": "stop_video"}),
        ("camera: do not stop video", "camera_control", {"command": "stop_video"}),
        ("camera: take photo after arriving", "camera_control", {"command": "take_photo"}),
        ("payload: attach, do not release", "payload_control", {"operation": "detach"}),
        ("Do not disable avoidance", "set_avoidance", {"enabled": False}),
        ("Enable avoidance", "set_avoidance", {"enabled": False}),
    ],
)
def test_prepopulated_peripheral_parameters_cannot_bypass_language_veto(text, action, parameters):
    with pytest.raises(ValueError, match="PERIPHERAL_COMMAND"):
        _classified(text, action, parameters)


# 功能：
#   验证补全限制保留常用肯定即时操作，不把停止录像与开始录像混为一类。
# 输入：
#   text：明确肯定的即时操作。
#   action：模型已选择的操作类别。
#   expected：应补全的具体参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "text,action,expected",
    [
        ("camera: stop video", "camera_control", {"command": "stop_video"}),
        ("相机停止录像", "camera_control", {"command": "stop_video"}),
        ("请开始录像", "camera_control", {"command": "start_video"}),
        ("payload: attach", "payload_control", {"operation": "attach"}),
        ("payload: detach", "payload_control", {"operation": "detach"}),
    ],
)
def test_unambiguous_immediate_peripheral_requests_remain_supported(text, action, expected):
    assert _classified(text, action)["parameters"] == expected
    assert _classified(text, action, expected)["parameters"] == expected
