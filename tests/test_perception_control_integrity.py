"""感知/模型/控制交界反例；手工构造内部状态仅为单元测试，不是飞行证据。"""

import hashlib
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from test_perception_runtime import _frame, _world

from dronedream_agent_core.contracts import TextNavigationDecision, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    RuntimePerceptionFusion,
    _failure_diagnostic,
    _NavigationWorkerResult,
    request_text_navigation_decision,
    validate_text_navigation_decision,
)


# 功能：
#   构造最小有摘要的候选快照，隔离类型边界与实际几何寻路。
# 输入：
#   gate：待测试的米制门控值。
# 输出：
#   snapshot：离线候选导航字典。
def _snapshot(gate=True):
    snapshot = {
        "perception_health": {"stream_healthy": True},
        "authorized_candidate_paths": [
            {"candidate_id": "candidate-a", "deterministic_metric_path_validated": gate}
        ],
    }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    return snapshot


# 功能：
#   验证米制门控必须是布尔 True，字符串及数字不能成为路径授权。
# 输入：
#   gate：不合法或未通过的门控值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("gate", [False, "false", "true", 1, None])
def test_navigation_requires_explicit_metric_gate(gate):
    snapshot = _snapshot(gate)
    decision = TextNavigationDecision(
        snapshot_sha256=snapshot["snapshot_sha256"],
        action="select-candidate",
        selected_candidate_id="candidate-a",
        rationale_summary="fixture",
    )
    with pytest.raises(ValueError, match="METRIC_VALIDATED"):
        validate_text_navigation_decision(snapshot, decision)


# 功能：
#   验证模型端口不能改写输入后用自己重算的摘要使越权选择通过验证。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_port_cannot_rewrite_its_authorized_input():
    snapshot = _snapshot()

    # 功能：
    #   模拟不守约端口改写候选身份及摘要，检验宿主是否持有独立输入。
    # 输入：
    #   kwargs：宿主传给端口的调用参数。
    # 输出：
    #   result：指向篡改候选的测试决策及实际调用标识。
    def mutate(**kwargs):
        submitted = kwargs["input_artifact"]["text_navigation_snapshot"]
        submitted["authorized_candidate_paths"][0]["candidate_id"] = "foreign"
        submitted.pop("snapshot_sha256")
        submitted["snapshot_sha256"] = sha256_json(submitted)
        result = SimpleNamespace(
            artifact=TextNavigationDecision(
                snapshot_sha256=submitted["snapshot_sha256"],
                action="select-candidate",
                selected_candidate_id="foreign",
                rationale_summary="fixture",
            ),
            record=SimpleNamespace(call_id="real-test-call"),
        )
        return result

    with pytest.raises(ValueError, match="SNAPSHOT_MISMATCH") as caught:
        request_text_navigation_decision(port=SimpleNamespace(call=mutate), snapshot=snapshot)
    assert snapshot == _snapshot()
    assert caught.value.model_call_record.call_id == "real-test-call"


# 功能：
#   验证选择结果不再共享输入候选字典，调用方后续改写不会改变已校验的候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_selected_candidate_is_detached():
    snapshot = _snapshot()
    decision = TextNavigationDecision(
        snapshot_sha256=snapshot["snapshot_sha256"],
        action="select-candidate",
        selected_candidate_id="candidate-a",
        rationale_summary="fixture",
    )
    selected = validate_text_navigation_decision(snapshot, decision)
    selected["candidate_id"] = "changed-after-validation"
    assert snapshot == _snapshot()


# 功能：
#   验证图像证据摘要后使用同一份字节，即使原文件随后被替换。
# 输入：
#   tmp_path：pytest 的隔离临时目录。
# 输出：
#   None：不返回业务数据。
def test_file_image_is_frozen_before_model_dispatch(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"first-image-fixture")
    item = {"kind": "image-file", "path": str(path)}
    digest = EventDrivenIndoorNavigationCoordinator._validated_visual_sha256(item, path=path)
    path.write_bytes(b"replaced-image-fixture")
    assert item["content_bytes"] == b"first-image-fixture"
    assert item["content_sha256"] == hashlib.sha256(b"first-image-fixture").hexdigest() == digest


# 功能：
#   验证文件模式也不能忽略调用方提交的错误图像摘要。
# 输入：
#   tmp_path：pytest 的隔离临时目录。
# 输出：
#   None：不返回业务数据。
def test_file_image_rejects_conflicting_digest(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"image-fixture")
    with pytest.raises(ValueError, match="hash does not match"):
        EventDrivenIndoorNavigationCoordinator._validated_visual_sha256(
            {"content_sha256": "0" * 64}, path=path
        )


# 功能：
#   建立没有付费调用的测试协调器，并可按需放入明确标记的测试租约。
# 输入：
#   leased：是否手工设置单元测试租约。
# 输出：
#   coordinator：隔离地图和假端口上的协调器。
def _coordinator(leased=False):
    fusion = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    fusion.ingest(_frame(), now_unix_ms=1000)
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion, port=SimpleNamespace(), required_clearance_m=0
    )
    if leased:
        coordinator._active_path = (Vector3(x=0.25, y=0.25, z=0.25), Vector3(x=3, y=0.25, z=0.25))
        coordinator._active_issued_at_unix_ms = 1000
        coordinator._active_path_until_unix_ms = 1200
        coordinator._active_navigation_goal_id = "goal-a"
    return coordinator


# 功能：
#   验证关闭、时钟倒退或丢失当前目标身份都不能保留旧模型控制权限。
# 输入：
#   defect：本次要施加的控制生命周期错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("defect", ["closed", "clock", "goal"])
def test_control_authority_ends_at_its_boundary(defect):
    coordinator = _coordinator(leased=True)
    try:
        if defect == "closed":
            coordinator.close()
        directive = coordinator.controller_directive(
            current_position_m=Vector3(x=0.25, y=0.25, z=0.25),
            fallback_target_m=Vector3(x=3, y=0.25, z=0.25),
            now_unix_ms=999 if defect == "clock" else 1001,
            navigation_goal_id=None if defect == "goal" else "goal-a",
        )
        assert not directive.model_navigation_authorized
        assert directive.target_m == Vector3(x=0.25, y=0.25, z=0.25)
        if defect == "closed":
            with pytest.raises(ValueError, match="CLOSED"):
                coordinator.schedule(
                    goal_position_m=Vector3(x=3, y=0.25, z=0.25),
                    now_unix_ms=1001,
                    trigger="initial",
                )
    finally:
        coordinator.close()


# 功能：
#   验证候选模式超时也不创建并发线程，旧推理完成后只回收调用痕迹而不重新授予控制权。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_timed_out_candidate_work_is_quarantined_and_accounted():
    coordinator = _coordinator()
    future = Future()
    assert future.set_running_or_notify_cancel()
    try:
        coordinator._quarantine_timed_out_work(future, reset_transport=False)
        assert coordinator.request_pending
        assert coordinator._executor_generation == 0
        future.set_result(
            _NavigationWorkerResult(
                snapshot=None,
                failure_reason="expired",
                discarded_call_record=SimpleNamespace(call_id="late-test-call"),
            )
        )
        assert coordinator.poll(now_unix_ms=1100) is None
        assert not coordinator.request_pending
        assert coordinator.pop_model_call_record().call_id == "late-test-call"
        assert coordinator.pop_model_call_record() is None
        assert not coordinator._active_path
    finally:
        coordinator.close()


# 功能：
#   验证待保存回执达到容量时停止新推理，而非覆盖已消耗的模型调用记录。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_evidence_backpressure_keeps_call_records():
    coordinator = _coordinator()
    try:
        coordinator._model_call_records.extend(SimpleNamespace(call_id=str(i)) for i in range(64))
        receipt = coordinator.schedule(
            goal_position_m=Vector3(x=3, y=0.25, z=0.25), now_unix_ms=1001, trigger="initial"
        )
        assert receipt.hold_reason == "MODEL_EVIDENCE_BACKPRESSURE"
        assert coordinator.pop_model_call_record().call_id == "0"
        assert len(coordinator._model_call_records) == 63
    finally:
        coordinator.close()


# 功能：
#   验证异常自定义字符串转换或无效诊断指标不会再次破坏失败回执。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failure_diagnostics_do_not_call_unsafe_formatters():
    class BrokenError(ValueError):
        # 功能：
        #   模拟不可靠异常对象的文本转换，测试诊断层是否绕开该操作。
        # 输入：
        #   self：故意损坏的异常。
        # 输出：
        #   None：本测试分支只抛异常。
        def __str__(self):
            raise RuntimeError("formatter must not run")

    error = BrokenError()
    error.diagnostic_metrics = {"latency-ms": 2.5, "bad": float("nan"), "text": "secret"}
    diagnostic = _failure_diagnostic(error, stage="model-invocation")
    assert diagnostic["failure_diagnostic_metrics"] == {"latency-ms": 2.5}
    assert "<locals>" not in diagnostic["failure_exception_type"]
    assert len(diagnostic["failure_fingerprint"]) == 64
