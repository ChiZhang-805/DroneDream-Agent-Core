"""Offline boundary regressions: transformations and receipts must not invent authority/evidence."""

import base64
import hashlib
from types import SimpleNamespace

import pytest
from test_planning_specialists import _fixtures

from dronedream_agent_core.contracts import AttachmentArtifact
from dronedream_agent_core.model_harness.model_port import StructuredModelPort
from dronedream_agent_plugins.harness_profiles import _profile
from dronedream_agent_plugins.harness_transform_plugins import (
    _apply_retry_budget,
    _fuse_tool_advice,
    _language_features,
    _phase_speed_envelope,
    _telemetry_waypoint_settle,
)
from dronedream_agent_plugins.mission_action_plugins import _pack
from dronedream_agent_plugins.model_runtime_plugins import (
    _image_preprocessor,
    _privacy_first_router,
    _resilient_router,
    _usage_meter,
)
from dronedream_agent_plugins.workflow_topologies import _core_nodes


# 功能：
#   验证重试次数不接受布尔、文本、小数或超界值，不能靠类型转换扩大交互预算。
# 输入：
#   value：待写入重试配置的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "2", 2.5, -1, 9, float("inf")])
def test_retry_transform_rejects_unvalidated_counts(value):
    with pytest.raises(ValueError):
        _apply_retry_budget(value=_fixtures()[3], configuration={"interaction_retries": value})


# 功能：
#   验证阶段速度上限必须有限、为正且使用已知阶段，不接受布尔或文本伪装的数字。
# 输入：
#   caps：本例的非法阶段限速配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "caps",
    [{"pickup": 0}, {"transit": float("nan")}, {"unknown": 1}, {"land": True}, {"stairs": "0.1"}],
)
def test_phase_speed_caps_are_positive_finite_and_known(caps):
    with pytest.raises(ValueError):
        _phase_speed_envelope(value=_fixtures()[-1], configuration={"phase_speed_caps_mps": caps})


# 功能：
#   验证限速变换只会收紧输入约束，且返回副本不会反向修改原轨迹。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_speed_transform_never_increases_incoming_limit_or_mutates_original():
    track = _fixtures()[-1]
    track.points[0].speed_limit_mps = 0.1
    changed = _phase_speed_envelope(value=track)
    assert changed.points[0].speed_limit_mps == 0.1
    assert changed.points[1].speed_limit_mps == 0.5
    assert track.points[1].speed_limit_mps == 1
    changed.points[0].x += 10
    assert track.points[0].x == 0


# 功能：
#   验证稳定观测窗口大于等待超时会被拒绝，避免配置出不可能满足的等待条件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_settle_window_must_fit_the_deadline():
    with pytest.raises(ValueError, match="WINDOW_EXCEEDS_TIMEOUT"):
        _telemetry_waypoint_settle(
            value=_fixtures()[-1],
            configuration={
                "stable_window_seconds": 2,
                "settle_timeout_seconds": 1,
            },
        )


# 功能：
#   验证工具结果融合只去除相同内容，保留不同位置和相反结论，并解除嵌套引用共享。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tool_fusion_preserves_distinct_calls_and_contradictory_evidence():
    first = {"tool_id": "clearance", "accepted": True, "point": {"x": 1}}
    dissent = {"tool_id": "clearance", "accepted": False, "point": {"x": 1}}
    different = {"tool_id": "clearance", "accepted": True, "point": {"x": 2}}
    output = _fuse_tool_advice(value=[first, dissent, different, first])
    assert len(output) == 3
    assert dissent in output and different in output
    output[0]["point"]["x"] = 99
    assert first["point"]["x"] == 1


# 功能：
#   验证无法识别文字体系时返回未知语言，不改原文或共享元数据引用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_language_hint_preserves_text_and_handles_no_recognized_script():
    original = {"message": "12345 🛩", "metadata": {"x": 1}}
    result = _language_features(value=original)
    assert result["message"] == original["message"]
    assert result["language_features"]["dominant_language"] == "unknown"
    result["metadata"]["x"] = 2
    assert original["metadata"]["x"] == 1


# 功能：
#   验证配置与动作声明在捕获和返回两个阶段都隔离可变数据，不能污染后续任务。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_profile_and_action_declarations_are_detached_at_capture_and_return():
    priorities = ["safety"]
    resolve = _profile(
        "test",
        priorities=priorities,
        required_evidence=["proof"],
        operating_style="bounded",
        recommended_plugins=["test.plugin"],
    )
    priorities.append("mutated")
    value = resolve()
    value["priorities"].append("mutated again")
    assert resolve()["priorities"] == ["safety"]
    actions = [{"action": "pickup", "defaults": {"hover": 10}}]
    declare = _pack("delivery", actions)
    actions[0]["defaults"]["hover"] = 99
    first = declare()
    first["actions"][0]["defaults"]["hover"] = 42
    assert declare()["actions"][0]["defaults"]["hover"] == 10


# 功能：
#   验证拓扑审查次数严格遵守类型与范围，不能通过隐式转换跳过审查或无限扩张。
# 输入：
#   count：非法审查次数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [True, 0, 4, 1.5, "1"])
def test_topology_review_count_cannot_bypass_declared_bounds(count):
    with pytest.raises(ValueError):
        _core_nodes(review_count=count, optional_advisors=True)


# 功能：
#   验证空目录不生成不存在的主端口，本地候选去重，正常回退保留声明顺序。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_router_does_not_invent_primary_or_duplicate_local_ports():
    assert (
        _privacy_first_router(requested_port="absent", available_ports=[], role="critic")[
            "candidates"
        ]
        == []
    )
    assert _privacy_first_router(
        requested_port="primary",
        available_ports=["primary", "local-vision", "local-vision"],
        role="critic",
    )["candidates"] == ["local-vision"]
    assert _resilient_router(
        requested_port="primary", available_ports=["primary", "critic"], role="critic"
    )["candidates"] == ["primary", "critic"]


# 功能：
#   验证用量未报告与真实零值保持区分，只有输入输出都已报告时才得出总量。
# 输入：
#   counts：供应商报告的输入、输出词元计数二元组。
#   total：本例应得到的总量或未知值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "counts,total", [((None, None), None), ((0, 0), 0), ((10, None), None), ((10, 4), 14)]
)
def test_usage_receipt_preserves_unknown_vs_zero(counts, total):
    receipt = _usage_meter(
        record=SimpleNamespace(
            input_tokens=counts[0], output_tokens=counts[1], provider="test", model="test"
        )
    )
    assert receipt["input_tokens"] == counts[0]
    assert receipt["output_tokens"] == counts[1]
    assert receipt["total_tokens"] == total


# 功能：
#   验证用量字段拒绝负数、非整数及隐式转换，避免伪造计数进入计量回执。
# 输入：
#   count：本例的非法输入词元计数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [True, -1, 1.5, "3", float("nan")])
def test_usage_receipt_rejects_coerced_counts(count):
    with pytest.raises(ValueError, match="TOKENS_INVALID"):
        _usage_meter(record=SimpleNamespace(input_tokens=count, output_tokens=0))


# 功能：
#   构造已解码的图像引用夹具，不读取实际图像、账户或凭证。
# 输入：
#   无。
# 输出：
#   attachment：含固定摘要与路径的类型化附件。
def _image_attachment():
    attachment = AttachmentArtifact(
        attachment_id="attachment-" + "a" * 32,
        display_name="frame.png",
        content_type="image/png",
        size_bytes=4,
        source_sha256="b" * 64,
        decoder_plugin_id="attachment.image",
        decoded_kind="image",
        model_input={"type": "input_image_reference", "source_path": "frame.png"},
    )
    return attachment


# 功能：
#   验证图像预处理保留来源摘要，但拒绝空路径或将文本附件伪装成图像。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_image_preprocessing_rejects_missing_path_and_nonimage_disguise():
    attachment = _image_attachment()
    assert _image_preprocessor(attachments=[attachment])["media"][0]["sha256"] == "b" * 64
    attachment.model_input["source_path"] = ""
    with pytest.raises(ValueError, match="REFERENCE_INVALID"):
        _image_preprocessor(attachments=[attachment])
    attachment = _image_attachment()
    attachment.decoded_kind = "text"
    with pytest.raises(ValueError, match="REFERENCE_INVALID"):
        _image_preprocessor(attachments=[attachment])


# 功能：
#   验证模型传输实际读取时重新核对图像文件摘要，拒绝预处理后被替换的字节。
# 输入：
#   tmp_path：仅存放本例虚拟图像字节的临时目录。
# 输出：
#   None：不返回业务数据。
def test_model_media_verifies_file_identity_at_the_actual_read(tmp_path):
    path = tmp_path / "frame.png"
    content = b"test image bytes"
    path.write_bytes(content)
    media = {
        "kind": "image-file",
        "path": str(path),
        "content_type": "image/png",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert StructuredModelPort._media_data_url(media).endswith(base64.b64encode(content).decode())
    path.write_bytes(b"different frame")
    with pytest.raises(ValueError, match="does not match"):
        StructuredModelPort._media_data_url(media)


# 功能：
#   验证内存帧必须不可变且摘要齐全一致，不借助文件路径绕过帧身份约束。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_inline_frame_is_immutable_hash_bound_and_needs_no_file():
    content = b"owned capture bytes"
    media = {
        "kind": "image-file",
        "content_bytes": content,
        "content_type": "image/png",
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }
    assert StructuredModelPort._media_data_url(media).endswith(base64.b64encode(content).decode())
    with pytest.raises(ValueError, match="inconsistent"):
        StructuredModelPort._media_data_url(media | {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="immutable"):
        StructuredModelPort._media_data_url(media | {"content_bytes": bytearray(content)})
    with pytest.raises(ValueError, match="missing"):
        StructuredModelPort._media_data_url({"kind": "image-file", "content_bytes": content})
