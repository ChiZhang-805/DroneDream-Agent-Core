from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_execution_runtime_action_receipts import _prepared_with_one_runtime_action

from dronedream_agent_core import execution
from dronedream_agent_core.context import ContextStore
from dronedream_agent_core.contracts import (
    CompletionAssessment,
    ModelCallRecord,
    Px4GazeboGates,
    RuntimeActionExecutionReceipt,
    RuntimeAssessment,
    RuntimeCheckpoint,
    RuntimeCheckpointContract,
    RuntimeCheckpointDecision,
    RuntimeCheckpointRequest,
    Vector3,
)
from dronedream_agent_core.evidence import EvidenceChain
from dronedream_agent_core.execution import (
    PreparedMissionBindingError,
    _active_runtime_gates_passed,
    _adopted_runtime_replacements,
    _artifact_collection_summary,
    _canonical_json_bytes,
    _file_sha256,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.model_port import StructuredCallResult


# 功能：
#   验证缺失必备门控的内部构造对象不能利用空集合全真而通过验收。
# 输入：
#   无。
# 输出：
#   无：不完整证据被拒绝的断言结果。
def test_runtime_acceptance_rejects_missing_required_gates():
    assert not _active_runtime_gates_passed(SimpleNamespace(gates=Px4GazeboGates.model_construct()))


# 功能：
#   验证内部被改成字符串的假值不能被当成通过门控。
# 输入：
#   无。
# 输出：
#   无：严格布尔验收的断言结果。
def test_runtime_acceptance_rejects_mutated_truthy_gate():
    gates = Px4GazeboGates.model_construct(
        **{name: True for name, field in Px4GazeboGates.model_fields.items() if field.is_required()}
    )
    gates.__dict__["landing_confirmed"] = "false"
    assert not _active_runtime_gates_passed(SimpleNamespace(gates=gates))


# 功能：
#   验证既有采纳回执缺少替换制品时必须报错，不静默退回旧任务。
# 输入：
#   tmp_path：隔离的运行目录。
# 输出：
#   无：残缺替换链被拒绝的断言结果。
def test_adopted_replacement_missing_artifact_is_not_ignored(tmp_path):
    directory = tmp_path / "runtime-control" / "adoptions"
    directory.mkdir(parents=True)
    (directory / ("runtime-msg-" + "a" * 32 + ".json")).write_text("{}", encoding="utf-8")
    with pytest.raises((PreparedMissionBindingError, ValueError, OSError)):
        _adopted_runtime_replacements(tmp_path)


# 功能：
#   验证完成模型输入的规范 JSON 不允许 NaN 等非标准数值。
# 输入：
#   无。
# 输出：
#   无：非法证据无法进入模型输入摘要的断言结果。
def test_completion_canonical_bytes_reject_nonfinite_numbers():
    with pytest.raises(ValueError):
        _canonical_json_bytes({"distance": float("nan")})


# 功能：
#   验证制品大小不接受负数或布尔，避免验收汇总伪报总字节数。
# 输入：
#   invalid：损坏的大小值。
# 输出：
#   无：损坏清单被拒绝的断言结果。
@pytest.mark.parametrize("invalid", [-1, True, "12"])
def test_completion_inventory_requires_real_nonnegative_byte_counts(invalid):
    with pytest.raises(ValueError):
        _artifact_collection_summary([{"size_bytes": invalid, "sha256": "a" * 64}])


# 功能：
#   验证日志摘要采用分块读取，不将整个长飞行日志一次载入内存。
# 输入：
#   tmp_path：隔离日志路径；monkeypatch：阻止无界读取方法。
# 输出：
#   无：实际日志摘要仍计算成功的断言结果。
def test_runtime_file_hash_does_not_require_read_bytes(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    path = tmp_path / "flight.ulg"
    path.write_bytes(b"bounded-log")

    # 功能：
    #   让旧的一次性读取路径确定失败，不替换真实磁盘内容。
    # 输入：
    #   self：原始文件路径。
    # 输出：
    #   无：抛出断言错误。
    def forbidden(self):
        raise AssertionError("unbounded read_bytes called")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert _file_sha256(path) == hashlib.sha256(b"bounded-log").hexdigest()


# 功能：
#   创建明确标记为离线测试的调用记录，不产生真实模型调用或飛行证据。
# 输入：
#   artifact：测试输出；role：测试角色；input_artifact：输入绑定。
# 输出：
#   record：对应测试内容的模型记录。
def _record(artifact, role, input_artifact):
    record = ModelCallRecord(
        call_id="model-" + "a" * 24, role=role, attempt=1,
        input_sha256=sha256_json(input_artifact), output_sha256=sha256_json(artifact),
        output_schema=type(artifact).__name__, provider="offline-fixture", model="boundary-test",
        latency_ms=0, created_at=datetime.now(UTC),
    )
    return record


# 功能：
#   写入一个独立检查点的离线请求和决策，便于真实磁盘解析与绑定检查。
# 输入：
#   tmp_path：隔离证据目录；target：请求实际绑定的目标节点。
# 输出：
#   fixture：请求、决定和所需检查点合同组成的元组。
def _checkpoint_fixture(tmp_path, target="locker"):
    checkpoint = RuntimeCheckpoint(
        checkpoint_id="checkpoint-001", segment_id="segment-001", task_id="visit",
        track_point_index=1, target_node=target,
    )
    request = RuntimeCheckpointRequest(
        contract_id="offline-mission", checkpoint=checkpoint,
        observed_position_ned_m=Vector3(x=0, y=0, z=-1),
        observed_velocity_ned_mps=Vector3(x=0, y=0, z=0),
        commanded_position_ned_m=Vector3(x=0, y=0, z=-1),
        position_error_m=0, speed_mps=0, battery_percent=80,
        deterministic_gates={"observed": True},
    )
    assessment = RuntimeAssessment(action="accept")
    decision = RuntimeCheckpointDecision(
        request_sha256=sha256_json(request), assessment=assessment,
        model_call=_record(assessment, "execution_monitor", {"request": request.model_dump()}),
        continue_authorized=True,
    )
    directory = tmp_path / "checkpoints"
    directory.mkdir(exist_ok=True)
    (directory / "checkpoint-001.request.json").write_text(
        request.model_dump_json(), encoding="utf-8"
    )
    (directory / "checkpoint-001.decision.json").write_text(
        decision.model_dump_json(), encoding="utf-8"
    )
    contract = RuntimeCheckpointContract(
        contract_id="offline-mission", checkpoints=[checkpoint.model_copy(deep=True)]
    )
    fixture = request, decision, contract
    return fixture


# 功能：
#   验证检查点正常逐项绑定可通过，但重复响应、错目标、缺请求或磁盘决策被改动不能通过。
# 输入：
#   tmp_path：隔离目录；fault：所注入的证据错误。
# 输出：
#   无：联合门控对各类真实磁盘变动的断言结果。
@pytest.mark.parametrize("fault", [None, "duplicate", "target", "missing", "changed", "unbound"])
def test_checkpoint_acceptance_matches_each_required_definition(tmp_path, fault):
    request, decision, contract = _checkpoint_fixture(tmp_path)
    decisions = [decision]
    if fault == "duplicate":
        decisions.append(decision.model_copy(deep=True))
    elif fault == "target":
        contract.checkpoints[0].target_node = "different-locker"
    elif fault == "missing":
        (tmp_path / "checkpoints" / "checkpoint-001.request.json").unlink()
    elif fault == "changed":
        changed = decision.model_copy(deep=True)
        changed.continue_authorized = False
        (tmp_path / "checkpoints" / "checkpoint-001.decision.json").write_text(
            changed.model_dump_json(), encoding="utf-8"
        )
    elif fault == "unbound":
        decision.model_call.output_sha256 = "b" * 64
        (tmp_path / "checkpoints" / "checkpoint-001.decision.json").write_text(
            decision.model_dump_json(), encoding="utf-8"
        )
    assert execution._checkpoint_receipts_passed(contract, [contract], decisions, tmp_path) is (
        fault is None
    )


# 功能：
#   验证动作在修订中原样保留时，原合同下的实际成功回执仍有效；缺少必需证据则拒绝。
# 输入：
#   tmp_path：隔离证据目录；monkeypatch：只替代替换链加载；missing_evidence：是否删除观察记录。
# 输出：
#   无：跨合同动作身份及实际证据要求的断言结果。
@pytest.mark.parametrize("missing_evidence", [False, True])
def test_retained_action_receipt_keeps_original_contract(tmp_path, monkeypatch, missing_evidence):
    prepared = _prepared_with_one_runtime_action()
    original = prepared.runtime_actions
    revised = original.model_copy(deep=True)
    revised.task_graph_sha256 = "d" * 64
    replacement = SimpleNamespace(
        runtime_actions=revised, prior_track_sha256=sha256_json(prepared.px4_track),
        track=prepared.px4_track, superseded_runtime_action_step_ids=[],
    )
    monkeypatch.setattr(execution, "_adopted_runtime_replacements", lambda *_: [replacement])
    step = original.steps[0]
    receipt = RuntimeActionExecutionReceipt(
        execution_contract_sha256=sha256_json(original), step_sha256=sha256_json(step),
        step_id=step.step_id, task_id=step.task_id, action=step.action,
        adapter_id=step.adapter_id, runtime_executor=step.runtime_executor, status="accepted",
        attempts=1, started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        observed_success_evidence=[] if missing_evidence else step.required_success_evidence,
        deterministic_gates={"observed": True},
    )
    directory = tmp_path / "runtime-actions" / "receipts"
    directory.mkdir(parents=True)
    (directory / "action-001.receipt.json").write_text(receipt.model_dump_json(), encoding="utf-8")
    if missing_evidence:
        with pytest.raises(PreparedMissionBindingError, match="success_evidence"):
            execution._load_runtime_action_receipts(prepared, tmp_path)
    else:
        receipts, required = execution._load_runtime_action_receipts(prepared, tmp_path)
        assert receipts == [receipt]
        assert required == {step.step_id}


class _CompletionPort:
    # 功能：
    #   建立不访问网络的完成模型端口，用于注入调用或关闭异常。
    # 输入：
    #   fault：测试故障；events：外部可核对的调用轨迹。
    # 输出：
    #   无：保存测试状态。
    def __init__(self, fault, events):
        self.fault, self.events = fault, events

    # 功能：
    #   返回明确的离线测试响应，或在请求阶段抛出指定故障。
    # 输入：
    #   kwargs：生产核验器传入的冻结输入和角色。
    # 输出：
    #   result：离线响应及内容绑定记录。
    def call(self, **kwargs):
        self.events.append("call")
        if self.fault == "call-and-close":
            raise LookupError("primary-call-failure")
        artifact = CompletionAssessment(accepted=True)
        record = _record(artifact, "completion_verifier", kwargs["input_artifact"])
        if self.fault == "binding":
            record.output_sha256 = "b" * 64
        result = StructuredCallResult(artifact=artifact, record=record)
        return result

    # 功能：
    #   记录端口关闭并可注入独立清理失败。
    # 输入：
    #   self：离线端口。
    # 输出：
    #   无：写入轨迹或抛出清理错误。
    def close(self):
        self.events.append("close")
        if self.fault == "call-and-close":
            raise OSError("cleanup-error")


# 功能：
#   验证输出验收失败前已经保存调用，且每条退出路径关闭模型端口、不掩盖原始错误。
# 输入：
#   tmp_path：隔离账本；monkeypatch：安装离线端口；fault：故障类型。
# 输出：
#   无：调用痕迹与关闭行为的断言结果。
@pytest.mark.parametrize("fault", [None, "binding", "call-and-close"])
def test_completion_port_records_before_validation_and_always_closes(tmp_path, monkeypatch, fault):
    events = []
    port = _CompletionPort(fault, events)
    monkeypatch.setattr(execution, "StructuredModelPort", lambda *args, **kwargs: port)
    chain = EvidenceChain(tmp_path / "completion.jsonl")
    context = ContextStore(tmp_path / "context.sqlite")
    kwargs = dict(
        provider="kimi", timeout_seconds=1, instructions="Offline test only", artifact={"x": 1},
        context_id="offline::completion", conversation_id="offline", chain=chain,
        context_store=context,
    )
    try:
        if fault == "call-and-close":
            with pytest.raises(LookupError, match="primary-call-failure") as captured:
                execution._invoke_completion_verifier(**kwargs)
            assert "OSError" in " ".join(captured.value.__notes__)
        elif fault == "binding":
            with pytest.raises(PreparedMissionBindingError, match="BINDING"):
                execution._invoke_completion_verifier(**kwargs)
        else:
            result = execution._invoke_completion_verifier(**kwargs)
            assert result.artifact.accepted
        assert events == ["call", "close"]
        if fault != "call-and-close":
            records = chain.read()
            assert len(records) == (2 if fault == "binding" else 1)
            assert records[0].event_type == "model.completion_verifier.received"
    finally:
        context.close()


class _Worker:
    # 功能：
    #   持有离线协调器清理轨迹与可选故障。
    # 输入：
    #   label：工作者名称；events：清理轨迹；fails：是否清理失败。
    # 输出：
    #   无：保存测试配置。
    def __init__(self, label, events, fails=False):
        self.label, self.events, self.fails = label, events, fails

    # 功能：
    #   记录停止尝试并可模拟协调器退出失败，不启动真实线程或飞行。
    # 输入：
    #   timeout_seconds：停止预算。
    # 输出：
    #   无：记录轨迹或抛出清理异常。
    def stop(self, *, timeout_seconds):
        assert timeout_seconds > 0
        self.events.append(self.label)
        if self.fails:
            raise OSError("worker-stop-failed")


# 功能：
#   验证一个工作者停止失败时仍尝试全部工作者，并保留已有主异常。
# 输入：
#   无。
# 输出：
#   无：逆序清理、集合释放与异常归因的断言结果。
def test_worker_cleanup_preserves_primary_failure_and_stops_all_workers():
    events = []
    workers = [_Worker("first", events), _Worker("second", events, True)]
    with pytest.raises(LookupError, match="primary") as captured:
        try:
            raise LookupError("primary")
        finally:
            execution._stop_execution_workers(workers, 1)
    assert events == ["second", "first"]
    assert workers == []
    assert "OSError" in " ".join(captured.value.__notes__)


# 功能：
#   用真实准备包、生产插件和文件链检查失败飞行不能被模型接受改成成功，复验不改原记录。
# 输入：
#   tmp_path、prepared、context：生产准备测试生成的隔离制品和账本。
#   monkeypatch：只安装离线完成模型，不启动模拟器或真实飞行。
# 输出：
#   无：完成判定、原始证据保持与独立复验输出的断言结果。
def _exercise_failed_execution_and_reverification(tmp_path, prepared, context, monkeypatch):
    from dronedream_agent_core.contracts import (
        Px4GazeboArtifactBindings,
        Px4GazeboMeasurements,
        Px4GazeboRunEvidence,
    )
    from dronedream_agent_core.runtime_control_io import publish_runtime_json

    package = tmp_path / "prepared"
    # 执行入口按已落盘任务包计算摘要；准备器内存对象的未设置默认字段不是同一摘要域。
    prepared = execution._read_contract(package / "prepared-mission.json", type(prepared))
    route_path = package / "08-execution-route.json"
    clearance_path = package / "09-route-clearance.json"
    track_path = package / "10-px4-track.json"
    gates = {
        name: True for name, field in Px4GazeboGates.model_fields.items() if field.is_required()
    }
    gates["landing_confirmed"] = False
    gates["px4_ulog_present"] = False
    runtime = Px4GazeboRunEvidence(
        schema_version="dronedream.generic-px4-gazebo-run.v1", status="failed",
        world="offline-test-world", vehicle="offline-test-vehicle",
        gates=Px4GazeboGates(**gates),
        measurements=Px4GazeboMeasurements(pose_sample_count=0, ros_observation_rows=0),
        artifacts=Px4GazeboArtifactBindings(
            world_sha256="a" * 64, semantic_sha256=prepared.contract.map_semantic_sha256,
            vehicle_sha256=prepared.contract.vehicle_sha256,
            route_sha256=_file_sha256(route_path), track_sha256=_file_sha256(track_path),
            clearance_sha256=_file_sha256(clearance_path), controller_params_sha256="b" * 64,
            executor_sha256="c" * 64, px4_ulogs=[], ros_workspace="offline-unused",
        ),
    )
    run_dir = tmp_path / "offline-evidence"
    timing = {"status": "completed", "cleanup": "landed-and-disarmed", "land_confirmed_t": 9.0}
    publish_runtime_json(run_dir / "mission_evidence.json", runtime)
    publish_runtime_json(run_dir / "offboard_timing.json", timing)
    events = []
    monkeypatch.setattr(
        execution, "StructuredModelPort", lambda *args, **kwargs: _CompletionPort(None, events)
    )
    result = execution._complete(
        prepared=prepared, route_path=route_path, clearance_path=clearance_path,
        track_path=track_path, route=prepared.execution_route, clearance=prepared.route_clearance,
        track=prepared.px4_track, runtime=runtime, offboard_timing=timing, run_dir=run_dir,
        completion_provider="kimi", context_store=context, model_timeout_seconds=1,
        evidence_filename="workflow-evidence.jsonl", result_filename="workflow-result.json",
        checkpoint_decisions=[], runtime_action_receipts=[], required_runtime_action_step_ids=set(),
        runtime_interruption_decisions=[], expected_checkpoint_count=len(
            prepared.runtime_checkpoints.checkpoints
        ),
    )
    assert result.status == "failed" and result.completion_assessment.accepted
    originals = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    reviewed = execution.reverify_prepared_run(
        prepared_path=package / "prepared-mission.json",
        confirm_contract_id=prepared.contract.contract_id, run_dir=run_dir,
        semantic_path=tmp_path / "semantic.json", vehicle_sdf=tmp_path / "vehicle.sdf",
        completion_provider="kimi", context_store=context, model_timeout_seconds=1,
    )
    assert reviewed.status == "failed"
    assert reviewed.runtime_evidence.gates.landing_confirmed is False
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert len(list((run_dir / "reviews").glob("*/workflow-result.json"))) == 1
    assert events == ["call", "close", "call", "close"]
