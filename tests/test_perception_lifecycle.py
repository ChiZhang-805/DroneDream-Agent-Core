import json

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_lifecycle import (
    capture_control_completion,
    verify_control_completion,
)


# 功能：
#   构造具有原生定位绑定、近期来源和可用实时特征的最小健康夹具。
# 输入：
#   无。
# 输出：
#   packet：独立的合法健康字典。
def health():
    packet = {"updated_at_unix_ms": 1000, "stream_age_seconds": .05,
            "localization_observed_at_unix_ms": 990,
            "stream_healthy": True, "identity_accepted": True,
            "truth_correction_applied": False, "realtime_features_ready": True,
            "pose_source": "native-estimator-fixed-deployment-binding"}
    return packet


# 功能：
#   将感知完成回执放入活动轨迹相符的执行器计时夹具。
# 输入：
#   record：待验证的感知回执。
# 输出：
#   result：带执行完成状态和轨迹身份的计时字典。
def timing(record):
    result = {"status": "complete", "active_track_sha256": "a" * 64,
            "perception_control_completion": record}
    return result


# 功能：
#   验证结束时冻结的合法证据不受后续停流影响，但执行失败或轨迹改变仍拒绝。
# 输入：
#   tmp_path：本次健康文件的独立目录。
# 输出：
#   None：不返回业务数据。
def test_shutdown_expiration_does_not_rewrite_completed_control_evidence(tmp_path):
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health()))
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    path.write_text(json.dumps({"stream_healthy": False, "updated_at_unix_ms": 1500}))
    assert verify_control_completion(timing(record)) is True
    assert record["health"]["stream_healthy"] is True
    assert verify_control_completion({**timing(record), "status": "failed"}) is False
    assert verify_control_completion({**timing(record), "active_track_sha256": "b" * 64}) is False


# 功能：
#   逐项注入时效、身份、特征和真值替代错误，确保实际控制结束失败不被停流解释掩盖。
# 输入：
#   tmp_path：本次健康文件的独立目录。
#   change：注入的单项非法健康状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", [
    {"stream_healthy": False}, {"identity_accepted": False}, {"truth_correction_applied": True},
    {"realtime_features_ready": False}, {"updated_at_unix_ms": 500},
    {"updated_at_unix_ms": 1100}, {"stream_age_seconds": .245},
    {"stream_age_seconds": float("nan")}, {"stream_age_seconds": -1.},
    {"pose_source": "gazebo-ground-truth"},
    {"localization_observed_at_unix_ms": 700},
    {"localization_observed_at_unix_ms": None},
    {"localization_observed_at_unix_ms": True},
    {"localization_observed_at_unix_ms": 1100},
])
def test_actual_control_completion_failure_is_not_excused_by_later_shutdown(tmp_path, change):
    path = tmp_path / "health.json"
    path.write_text(json.dumps({**health(), **change}))
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    assert verify_control_completion(timing(record)) is False


# 功能：
#   拒绝缺失、畸形及被篡改的回执，健康失败即使重算摘要也不能通过。
# 输入：
#   tmp_path：本次健康文件的独立目录。
# 输出：
#   None：不返回业务数据。
def test_missing_malformed_or_tampered_completion_evidence_rejected(tmp_path):
    path = tmp_path / "health.json"
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    assert verify_control_completion(timing(record)) is False
    path.write_text("[]")
    assert capture_control_completion(path, track_sha256="a" * 64,
                                      completed_at_unix_ms=1010)["accepted"] is False
    path.write_text(json.dumps(health()))
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    record["health"]["stream_healthy"] = False
    assert verify_control_completion(timing(record)) is False
    content = {k: v for k, v in record.items() if k != "receipt_sha256"}
    record["receipt_sha256"] = sha256_json(content)
    assert verify_control_completion(timing(record)) is False


# 功能：
#   验证极大整数年龄不会在清理期间触发浮点转换溢出，而是保留拒绝结果。
# 输入：
#   tmp_path：本次健康文件的独立目录。
# 输出：
#   None：不返回业务数据。
def test_extreme_age_is_rejected_without_interrupting_cleanup(tmp_path):
    path = tmp_path / "health.json"
    path.write_text(json.dumps({**health(), "stream_age_seconds": 10**400}))
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    assert record["accepted"] is False
    assert verify_control_completion(timing(record)) is False


# 功能：
#   确认过深 JSON 生成失败回执而不使清理流程因解析递归错误中断。
# 输入：
#   tmp_path：本次健康文件的独立目录。
# 输出：
#   None：不返回业务数据。
def test_deeply_nested_packet_becomes_failed_evidence(tmp_path):
    path = tmp_path / "health.json"
    path.write_text('{"nested":' + '[' * 2000 + '0' + ']' * 2000 + '}')
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    assert record["accepted"] is False
    assert record["read_issue"] is not None
    assert verify_control_completion(timing(record)) is False


# 功能：
#   验证重复健康字段及指数溢出不能经宽松 JSON 解析变成合格的完成证据。
# 输入：
#   tmp_path：本次测试的独立目录。
#   field：追加到正常健康对象的歧义或非有限字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ['"stream_healthy":true', '"nested":{"v":1,"v":2}',
                                  '"extra":1e400'])
def test_ambiguous_health_packet_is_rejected(tmp_path, field):
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health())[:-1] + "," + field + "}")
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    assert record["accepted"] is False and record["read_issue"] is not None
    assert verify_control_completion(timing(record)) is False


# 功能：
#   确认错误轨迹或完成时刻生成明确失败记录，不中断调用方后续的降落清理。
# 输入：
#   tmp_path：本次测试的独立目录。
#   track：候选轨迹摘要。
#   completed：候选完成 UNIX 毫秒。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("track", "completed"), [
    ("not-a-digest", 1010), (None, 1010), (True, 1010), (float("nan"), 1010),
    ("a" * 64, None), ("a" * 64, "1010"), ("a" * 64, float("nan")),
    ("a" * 64, True), ("a" * 64, -1), ("a" * 64, 10 ** 400),
])
def test_invalid_completion_identity_is_rejected_without_interrupting_cleanup(tmp_path, track,
                                                                            completed):
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health()))
    record = capture_control_completion(path, track_sha256=track, completed_at_unix_ms=completed)
    assert record["accepted"] is False and record["read_issue"] is not None
    assert verify_control_completion(timing(record)) is False


# 功能：
#   在伪造者重算摘要后仍独立拒绝不可能的负 UNIX 时刻，摘要不能代替时间契约。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rehashed_negative_completion_cannot_pass_verification():
    record = {"event": "controlled-motion-completed-before-landing", "track_sha256": "a" * 64,
              "completed_at_unix_ms": -1, "health": {**health(), "updated_at_unix_ms": -2},
              "accepted": True, "read_issue": None}
    record["receipt_sha256"] = sha256_json(record)
    assert verify_control_completion(timing(record)) is False


# 功能：
#   确认非对象计时记录被稳定拒绝，不在验收阶段引发属性错误。
# 输入：
#   value：非法计时记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, [], True, "complete"])
def test_verifier_rejects_non_mapping_timing(value):
    assert verify_control_completion(value) is False


# 功能：
#   确认内存中的非字符串键与过量回执不能利用摘要规范化绕过 JSON 边界。
# 输入：
#   tmp_path：独立健康文件目录。
#   extra：注入健康回执的非规范或超预算内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("extra", [{1: "not-a-string-key"}, {"data": "x" * 300_000}])
def test_verification_checks_json_budget_before_canonical_hashing(tmp_path, extra):
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health()))
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    record["health"].update(extra)
    record.pop("receipt_sha256")
    record["receipt_sha256"] = sha256_json(record)
    assert verify_control_completion(timing(record)) is False


# 功能：
#   验证读取失败记录只保留有界错误类别，不复制可能含用户路径或大段内容的异常消息。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：限定当前测试内的文件打开异常注入。
# 输出：
#   None：不返回业务数据。
def test_read_failure_diagnostic_does_not_copy_unbounded_exception(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "health.json"
    path.write_text(json.dumps(health()))

    # 功能：
    #   模拟底层文件打开失败并附带不应进入回执的大段内部消息。
    # 输入：
    #   self：被打开的路径。
    #   args：位置参数。
    #   kwargs：关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_open(self, *args, **kwargs):
        raise OSError("internal-path-marker:" + "x" * 300_000)

    monkeypatch.setattr(Path, "open", fail_open)
    record = capture_control_completion(path, track_sha256="a" * 64, completed_at_unix_ms=1010)
    assert record["accepted"] is False
    assert len(record["read_issue"]) <= 128
    assert "internal-path-marker" not in record["read_issue"]
