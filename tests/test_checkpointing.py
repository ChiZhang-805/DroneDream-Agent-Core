import pytest

from dronedream_agent_core.checkpointing import checkpoint_continue_authorized
from dronedream_agent_core.contracts import (
    RuntimeAssessment,
    RuntimeCheckpoint,
    RuntimeCheckpointRequest,
    Vector3,
)


# 功能：
#   构造一个已停稳的类型化检查点请求，允许分别破坏确定性门控来验证授权条件。
# 输入：
#   gate_updates：覆盖默认通过门控的测试值。
# 输出：
#   request：包含任务身份、坐标、速度、电量及门控的检查点请求。
def _request(**gate_updates: bool) -> RuntimeCheckpointRequest:
    gates = {
        "position_within_tolerance": True,
        "speed_within_hold_limit": True,
        "battery_above_reserve": True,
        "no_collision": True,
    }
    gates.update(gate_updates)
    request = RuntimeCheckpointRequest(
        contract_id="mission-test",
        checkpoint=RuntimeCheckpoint(
            checkpoint_id="checkpoint-001",
            segment_id="segment-001",
            task_id="task-001",
            track_point_index=1,
            target_node="node-b",
        ),
        observed_position_ned_m=Vector3(x=1.0, y=2.0, z=-3.0),
        observed_velocity_ned_mps=Vector3(x=0.0, y=0.0, z=0.0),
        commanded_position_ned_m=Vector3(x=1.0, y=2.0, z=-3.0),
        position_error_m=0.0,
        speed_mps=0.0,
        battery_percent=70.0,
        deterministic_gates=gates,
    )
    return request


# 功能：
#   验证模型接受不能覆盖代码明确报告的碰撞失败，继续执行必须同时满足两类条件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_accept_requires_every_code_gate() -> None:
    assert not checkpoint_continue_authorized(
        request=_request(no_collision=False),
        assessment=RuntimeAssessment(action="accept"),
        binding_gates={"checkpoint_binding": True},
    )


# 功能：
#   验证代码门控通过时仍需模型明确接受，模型要求悬停不能被改成继续执行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_code_accepts_only_bound_safe_model_accept() -> None:
    request = _request()
    binding = {"checkpoint_binding": True, "track_point_binding": True}
    assert checkpoint_continue_authorized(
        request=request,
        assessment=RuntimeAssessment(action="accept"),
        binding_gates=binding,
    )
    assert not checkpoint_continue_authorized(
        request=request,
        assessment=RuntimeAssessment(action="hold"),
        binding_gates=binding,
    )


# 功能：
#   验证段内部检查点在全局轨迹索引、坐标与任务绑定都有效时可以通过授权条件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_internal_checkpoint_can_be_authorized_when_its_hash_bound_point_is_valid() -> None:
    request = _request()
    binding = {
        "global_track_point_index_in_range": True,
        "checkpoint_maps_to_hash_bound_plan_point": True,
        "checkpoint_target_matches_hash_bound_plan_point": True,
        "checkpoint_task_matches_segment": True,
        "px4_local_ned_command_matches_global_track_point": True,
    }

    assert checkpoint_continue_authorized(
        request=request,
        assessment=RuntimeAssessment(action="accept"),
        binding_gates=binding,
    )


# 功能：
#   验证空身份绑定或非布尔的通过值不能授予继续执行权限。
# 输入：
#   binding：缺失或错误类型的身份绑定证据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("binding", [{}, {"binding": 1}, {"binding": "true"}])
def test_absent_or_untyped_binding_evidence_does_not_grant_continuation(binding):
    assert not checkpoint_continue_authorized(
        request=_request(),
        assessment=RuntimeAssessment(action="accept"),
        binding_gates=binding,
    )


# 功能：
#   验证模型等待期间停止或换版后，迟到结果不能生成新检查点决定或触发继续执行。
# 输入：
#   tmp_path：本次检查点及中止文件的隔离目录。
#   monkeypatch：模拟模型调用期间活动计划发生改变。
#   stop_requested：本例选择停止协调器，或采用新计划版本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("stop_requested", [True, False], ids=["shutdown", "new-revision"])
def test_stale_model_call_discards_late_checkpoint_decision(tmp_path, monkeypatch, stop_requested):
    import threading
    from types import SimpleNamespace

    from test_mission_verification import _artifacts

    from dronedream_agent_core.checkpointing import CheckpointCoordinator

    values = _artifacts()
    coordinator = object.__new__(CheckpointCoordinator)
    coordinator.prepared = SimpleNamespace(
        contract=values[0], plan=values[4], task_graph=values[2], px4_track=values[7]
    )
    coordinator.contract, coordinator.run_dir = values[8], tmp_path
    coordinator.abort_file, coordinator.receipt_path = tmp_path / "abort.json", tmp_path / "hooks"
    coordinator._stop, coordinator.error, coordinator.decisions = threading.Event(), None, []
    coordinator.extensions = SimpleNamespace(
        invoke_multiple=lambda *a, **kw: ([], []),
        invoke_pipeline=lambda *args, **kw: (args[2], []),
    )
    if not stop_requested:
        # 模拟模型等待期间计划被采纳替换；磁盘读取器的真实摘要验证由独立测试覆盖。
        revisions = iter([None, object()])
        monkeypatch.setattr(
            "dronedream_agent_core.checkpointing.load_active_replacement", lambda _: next(revisions)
        )

    # 功能：
    #   在模拟调用返回前触发停止，并返回无字段对象，确认失效后不再读取模型结果。
    # 输入：
    #   kwargs：协调器传入的模型调用参数。
    # 输出：
    #   response：仅用于检测错误字段访问的哨兵对象。
    def late_response(**kwargs):
        if stop_requested:
            coordinator._stop.set()
        response = object()
        return response

    coordinator.port = SimpleNamespace(call=late_response, close=lambda: None)
    request = _request().model_copy(
        update={
            "contract_id": values[0].contract_id,
            "checkpoint": values[8].checkpoints[0],
            "commanded_position_ned_m": Vector3(x=0.0, y=2.0, z=-1.0),
        }
    )
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    (folder / "checkpoint-001.request.json").write_text(request.model_dump_json())
    coordinator._run()
    assert not coordinator.decisions
    if stop_requested:
        assert coordinator.error is None and not coordinator.abort_file.exists()
    else:
        assert isinstance(coordinator.error, RuntimeError)
        assert "revision changed" in str(coordinator.error)
        assert coordinator.abort_file.exists()
    assert not list(folder.glob("*.decision.json"))


# 功能：
#   构造单检查点真实文件输入及类型化模型响应，隔离模型供应商并保留协调器运行逻辑。
# 输入：
#   tmp_path：本次检查点运行目录。
# 输出：
#   coordinator：可直接运行一次检查点循环的测试协调器。
def _ready_coordinator(tmp_path):
    import threading
    from types import SimpleNamespace

    from test_mission_verification import _artifacts
    from test_runtime_plugins import _model_output

    from dronedream_agent_core.checkpointing import CheckpointCoordinator
    from dronedream_agent_core.hashing import sha256_json

    values = _artifacts()
    coordinator = object.__new__(CheckpointCoordinator)
    coordinator.prepared = SimpleNamespace(
        contract=values[0], plan=values[4], task_graph=values[2], px4_track=values[7]
    )
    coordinator.contract, coordinator.run_dir = values[8], tmp_path
    coordinator.abort_file, coordinator.receipt_path = tmp_path / "abort.json", tmp_path / "hooks"
    coordinator._stop, coordinator.error, coordinator.decisions = threading.Event(), None, []
    coordinator._publication_lock = threading.Lock()
    coordinator.extensions = SimpleNamespace(
        invoke_multiple=lambda *a, **kw: ([], []),
        invoke_pipeline=lambda *args, **kw: (args[2], []),
    )
    assessment = RuntimeAssessment(action="accept")
    _, record = _model_output()
    record = record.model_copy(update={
        "output_schema": "RuntimeAssessment", "output_sha256": sha256_json(assessment),
    })
    response = SimpleNamespace(artifact=assessment, record=record)
    coordinator.port = SimpleNamespace(call=lambda **kwargs: response, close=lambda: None)
    request = _request().model_copy(update={
        "contract_id": values[0].contract_id, "checkpoint": values[8].checkpoints[0],
        "commanded_position_ned_m": Vector3(x=0.0, y=2.0, z=-1.0),
    })
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    (folder / "checkpoint-001.request.json").write_text(request.model_dump_json())
    return coordinator


# 功能：
#   在输出校验或回执写入期间停止／换计划，验证模型调用结束后仍会丢弃迟到的决定。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：模拟生命周期改变和单次运行结束。
#   phase：在输出守卫还是其回执写入期间触发改变。
#   stop_requested：选择停止或活动计划替换。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase", ["guard", "receipt"])
@pytest.mark.parametrize("stop_requested", [True, False], ids=["shutdown", "new-revision"])
def test_checkpoint_rechecks_lifecycle_after_output_guard(
    tmp_path, monkeypatch, phase, stop_requested,
):
    from dronedream_agent_core import checkpointing as module

    coordinator = _ready_coordinator(tmp_path)
    active = [None]
    monkeypatch.setattr(module, "load_active_replacement", lambda _: active[0])

    # 功能：
    #   在指定阶段使当前模型响应失效，不发送真实控制或改写磁盘采纳记录。
    # 输入：
    #   无；使用外层测试选择。
    # 输出：
    #   None：不返回业务数据。
    def invalidate():
        if stop_requested:
            coordinator._stop.set()
        else:
            active[0] = object()

    guarded = []

    # 功能：
    #   标识模型已通过输出校验，并按本例设置模拟此时生命周期变化。
    # 输入：
    #   args：输出守卫的位置参数。
    #   kwargs：角色、制品和记录等守卫参数。
    # 输出：
    #   receipts：空测试回执列表。
    def guard(*args, **kwargs):
        guarded.append(True)
        if phase == "guard":
            invalidate()
        receipts = []
        return receipts

    # 功能：
    #   仅在输出守卫完成后的回执写入点模拟生命周期变化。
    # 输入：
    #   path：协调器的日志路径。
    #   receipts：本次回执列表。
    # 输出：
    #   None：不返回业务数据。
    def append(path, receipts):
        if guarded and phase == "receipt":
            invalidate()

    original_write = module._atomic_json

    # 功能：
    #   保留真实文件发布行为，若旧逻辑错误发布了决定，则终止循环以使失败可复现。
    # 输入：
    #   path：发布路径。
    #   payload：发布内容。
    #   kwargs：原子写入选项。
    # 输出：
    #   None：不返回业务数据。
    def bounded_write(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        if path.name.endswith(".decision.json"):
            coordinator._stop.set()

    monkeypatch.setattr(module, "validate_runtime_model_output", guard)
    monkeypatch.setattr(module, "append_hook_receipts", append)
    monkeypatch.setattr(module, "_atomic_json", bounded_write)
    coordinator._run()
    assert guarded
    assert not coordinator.decisions
    assert not list((tmp_path / "checkpoints").glob("*.decision.json"))
    if stop_requested:
        assert coordinator.error is None and not coordinator.abort_file.exists()
    else:
        assert "revision changed" in str(coordinator.error)
        assert coordinator.abort_file.exists()


# 功能：
#   验证发布失败的决定不能提前登记为已发布，错误仍触发中止证据。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：模拟决定文件写入失败。
# 输出：
#   None：不返回业务数据。
def test_failed_decision_publication_does_not_enter_success_history(tmp_path, monkeypatch):
    from dronedream_agent_core import checkpointing as module

    coordinator = _ready_coordinator(tmp_path)
    original_write = module._atomic_json

    # 功能：
    #   只拒绝决定文件，保留中止记录的真实发布能力。
    # 输入：
    #   path：目标路径。
    #   payload：待写内容。
    #   kwargs：原子写入参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_decision(path, payload, **kwargs):
        if path.name.endswith(".decision.json"):
            raise OSError("test decision publication failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(module, "_atomic_json", fail_decision)
    coordinator._run()
    assert isinstance(coordinator.error, OSError)
    assert not coordinator.decisions
    assert coordinator.abort_file.exists()


# 功能：
#   验证已有的同名暂存文件不是当前调用所有，独占创建失败时必须保留其内容。
# 输入：
#   tmp_path：暂存和目标目录。
#   monkeypatch：固定随机标识以复现命名冲突。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_atomic_write_preserves_unowned_collision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dronedream_agent_core import checkpointing as module
    from dronedream_agent_core import runtime_control_io

    monkeypatch.setattr(runtime_control_io, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    path = tmp_path / "decision.json"
    temporary = path.with_name(f".{path.name}.{'a' * 32}.tmp")
    temporary.write_text("other owner", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._atomic_json(path, {"accepted": True})
    assert temporary.read_text(encoding="utf-8") == "other owner"
    assert not path.exists()


# 功能：
#   验证类型化模型中的非有限值不能先被序列化成 null，再作为正常检查点证据写入。
# 输入：
#   tmp_path：应保持未创建的输出目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_atomic_write_rejects_nonfinite_model_before_io(tmp_path):
    from dronedream_agent_core.checkpointing import _atomic_json

    path = tmp_path / "new" / "decision.json"
    payload = Vector3(x=0, y=0, z=1).model_copy(update={"x": float("nan")})
    with pytest.raises(ValueError):
        _atomic_json(path, payload)
    assert not path.parent.exists()


# 功能：
#   验证缺失检查点合同不能使用旧路径自动推导，以免绕过已确认的准备阶段产物。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_checkpoint_contract_is_not_derived_from_old_plan():
    from types import SimpleNamespace

    from test_mission_verification import _artifacts

    from dronedream_agent_core.checkpointing import checkpoint_contract_for

    values = _artifacts()
    prepared = SimpleNamespace(runtime_checkpoints=None, contract=values[0], plan=values[4])
    with pytest.raises(ValueError, match="CHECKPOINT_CONTRACT_REQUIRED"):
        checkpoint_contract_for(prepared)


# 功能：
#   验证读取的检查点合同不共享可变列表，调用方修改不能反向改变冻结任务。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_contract_is_detached_from_prepared_mission():
    from types import SimpleNamespace

    from test_mission_verification import _artifacts

    from dronedream_agent_core.checkpointing import checkpoint_contract_for

    values = _artifacts()
    prepared = SimpleNamespace(runtime_checkpoints=values[8], contract=values[0])
    selected = checkpoint_contract_for(prepared)
    selected.checkpoints[0].target_node = "changed-by-caller"
    assert prepared.runtime_checkpoints.checkpoints[0].target_node == "target"


# 功能：
#   验证同一任务的检查点合同不能混入其他任务身份或重复检查点标识。
# 输入：
#   defect：选择任务身份不匹配或重复检查点。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("defect", ["contract", "duplicate"])
def test_checkpoint_contract_rejects_mixed_or_duplicate_identity(defect):
    from types import SimpleNamespace

    from test_mission_verification import _artifacts

    from dronedream_agent_core.checkpointing import checkpoint_contract_for

    values = _artifacts()
    contract = values[8]
    if defect == "contract":
        contract.contract_id = "different-contract"
    else:
        contract.checkpoints.append(contract.checkpoints[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="MISMATCH|DUPLICATED"):
        checkpoint_contract_for(SimpleNamespace(runtime_checkpoints=contract, contract=values[0]))


# 功能：
#   验证关闭未启动协调器可重复执行，关闭后不能再启动旁路线程。
# 输入：
#   tmp_path：构造协调器的测试目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_stop_before_start_is_idempotent_and_prevents_restart(tmp_path):
    import threading

    coordinator = _ready_coordinator(tmp_path)
    coordinator._thread = threading.Thread(target=lambda: None)
    coordinator.stop(timeout_seconds=0.1)
    coordinator.stop(timeout_seconds=0.1)
    assert coordinator._stop.is_set()
    with pytest.raises(RuntimeError, match="already stopped"):
        coordinator.start()
    assert coordinator._thread.ident is None


# 功能：
#   验证等待在途发布也计入关闭预算，超时会报错且停止信号仍保持已发送。
# 输入：
#   tmp_path：测试协调器目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_stop_is_bounded_when_publication_is_busy(tmp_path):
    import threading

    coordinator = _ready_coordinator(tmp_path)
    coordinator._thread = threading.Thread(target=lambda: None)
    with (
        coordinator._publication_lock,
        pytest.raises(TimeoutError, match="publication did not stop"),
    ):
        coordinator.stop(timeout_seconds=0.01)
    assert coordinator._stop.is_set()
    coordinator.stop(timeout_seconds=0.1)


# 功能：
#   验证正常接受决定经过真实文件发布后才进入历史，已存在的决定不能被后续写入覆盖。
# 输入：
#   tmp_path：当前运行目录。
#   monkeypatch：发布一次决定后停止循环。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_publishes_bound_decision_once(tmp_path, monkeypatch):
    from dronedream_agent_core import checkpointing as module
    from dronedream_agent_core.contracts import RuntimeCheckpointDecision

    coordinator = _ready_coordinator(tmp_path)
    original_write = module._atomic_json

    # 功能：
    #   记录真实发布时历史列表的顺序，发布完成后使单次测试循环结束。
    # 输入：
    #   path：决定文件路径。
    #   payload：待发布决定。
    #   kwargs：独占发布选项。
    # 输出：
    #   None：不返回业务数据。
    def write_once(path, payload, **kwargs):
        assert not coordinator.decisions
        original_write(path, payload, **kwargs)
        coordinator._stop.set()

    monkeypatch.setattr(module, "_atomic_json", write_once)
    coordinator._run()
    assert coordinator.error is None and len(coordinator.decisions) == 1
    path = tmp_path / "checkpoints" / "checkpoint-001.decision.json"
    content = path.read_bytes()
    decision = RuntimeCheckpointDecision.model_validate_json(content)
    assert decision.continue_authorized is True
    assert decision == coordinator.decisions[0]
    with pytest.raises(FileExistsError):
        original_write(path, {"different": "decision"}, replace_existing=False)
    assert path.read_bytes() == content
    assert not list(path.parent.glob("*.tmp"))


# 功能：
#   验证暂存被替换时不发布替身，发布后同名路径被重用时也不误删其他写入者的文件。
# 输入：
#   tmp_path：独立输出与被保留暂存的目录。
#   monkeypatch：在文件关闭或发布完成时模拟另一写入者。
#   phase：选择发布前替换或发布后重用同名路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase", ["before", "after"])
def test_checkpoint_atomic_write_checks_temporary_ownership(tmp_path, monkeypatch, phase):
    from contextlib import contextmanager
    from pathlib import Path

    from dronedream_agent_core import checkpointing as module

    path = tmp_path / "decision.json"
    captured = []
    original_open, original_replace = Path.open, Path.replace

    # 功能：
    #   保留真实流的身份，在独占暂存关闭后才模拟名称被另一个文件占用。
    # 输入：
    #   selected：待打开路径。
    #   args：文件打开模式等位置参数。
    #   kwargs：编码等文件打开选项。
    # 输出：
    #   stream：原始文件流，只向写入器交付一次。
    @contextmanager
    def replace_after_close(selected, *args, **kwargs):
        with original_open(selected, *args, **kwargs) as stream:
            yield stream
        if args and args[0] == "x" and selected.suffix == ".tmp":
            captured.append(selected)
            if phase == "before":
                selected.rename(tmp_path / "retained-original")
                with original_open(selected, "w", encoding="utf-8") as replacement:
                    replacement.write("other owner")

    # 功能：
    #   完成真实发布后，在空出的旧暂存路径上创建其他写入者的文件。
    # 输入：
    #   selected：原暂存路径。
    #   target：发布目标路径。
    # 输出：
    #   result：真实替换操作返回的目标路径。
    def reuse_after_publish(selected, target):
        result = original_replace(selected, target)
        with original_open(selected, "w", encoding="utf-8") as replacement:
            replacement.write("other owner")
        return result

    with monkeypatch.context() as patched:
        patched.setattr(Path, "open", replace_after_close)
        if phase == "after":
            patched.setattr(Path, "replace", reuse_after_publish)
            module._atomic_json(path, {"accepted": True})
        else:
            with pytest.raises(ValueError, match="TEMPORARY_REPLACED"):
                module._atomic_json(path, {"accepted": True})
    assert len(captured) == 1
    assert captured[0].read_text(encoding="utf-8") == "other owner"
    if phase == "before":
        assert not path.exists()
        assert (tmp_path / "retained-original").is_file()
    else:
        assert '"accepted": true' in path.read_text(encoding="utf-8")
