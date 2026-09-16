"""Exercise the real coordinator loop with isolated model and lifecycle ports."""

import threading
from types import SimpleNamespace

import pytest
from test_mission_verification import _artifacts
from test_runtime_interrupt import _ack, _message
from test_runtime_plugins import _model_output
from test_runtime_revision import _replacement

from dronedream_agent_core import context, runtime_interrupt
from dronedream_agent_core.contracts import RuntimeAmendmentDirective, RuntimeMessageClassification
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   建立真实改令文件输入及类型化模型响应，使用隔离生命周期端口记录状态，不访问用户数据库。
# 输入：
#   tmp_path：本轮运行目录。
#   monkeypatch：替换生命周期数据库构造器。
#   replan：是否构造需要重规划的速度修改，而非信息性继续请求。
# 输出：
#   fixture：协调器、消息与生命周期状态记录组成的三元组。
def _ready_interruption(tmp_path, monkeypatch, *, replan=False):
    values = _artifacts()
    message = _message("请把速度调低" if replan else "天气不错，继续任务")
    control_dir = tmp_path / "runtime-control"
    session = runtime_interrupt.create_runtime_control_session(
        control_dir=control_dir, conversation_id=message.conversation_id,
        mission_id=message.mission_id, plan_revision_id=message.plan_revision_id,
        contract_id=message.contract_id, execution_id=message.execution_id,
        prepared_mission_sha256="e" * 64,
    )
    runtime_interrupt._atomic_json(control_dir / "claimed" / f"{message.message_id}.json", message)
    runtime_interrupt._atomic_json(
        control_dir / "acks" / f"{message.message_id}.json", _ack(message),
    )
    coordinator = object.__new__(runtime_interrupt.RuntimeInterruptionCoordinator)
    coordinator.session, coordinator.control_dir = session, control_dir
    coordinator.abort_file, coordinator.lifecycle_db_path = tmp_path / "abort.json", tmp_path / "db"
    coordinator.prepared = SimpleNamespace(
        contract=values[0], plan=values[4], task_graph=values[2], px4_track=values[7],
        runtime_actions=values[9], plugin_snapshot=None,
    )
    coordinator.map_graph = coordinator.map_catalog = coordinator.vehicle = None
    coordinator.semantic_path = tmp_path / "semantic.json"
    coordinator.receipt_path = tmp_path / "hooks.jsonl"
    coordinator._stop, coordinator._publication_lock = threading.Event(), threading.Lock()
    coordinator.error, coordinator.decisions, coordinator.adoption_timeout_seconds = None, [], 0.01
    classification = RuntimeMessageClassification(
        message_kind="motion_adjustment" if replan else "informational",
        requested_action="set_speed" if replan else "resume",
        requires_plan_revision=replan, summary="Local lifecycle fixture",
        parameters={"maximum_speed_mps": 0.3} if replan else {},
    )
    directive = RuntimeAmendmentDirective(
        action=classification.requested_action, parameters=classification.parameters,
    )
    _, record = _model_output()
    record = record.model_copy(update={
        "role": "runtime_message_classifier", "output_schema": "RuntimeMessageClassification",
        "output_sha256": sha256_json(classification),
    })
    response = SimpleNamespace(artifact=classification, record=record)
    coordinator.port = SimpleNamespace(call=lambda **kwargs: response)
    coordinator.extensions = SimpleNamespace(
        invoke_pipeline=lambda *args, **kwargs: (args[2], []),
        invoke_single=lambda *args, **kwargs: (directive.model_dump(mode="json"), []),
    )
    states = []
    store = SimpleNamespace(
        lifecycle=SimpleNamespace(
            set_execution_state=lambda **kwargs: states.append(kwargs["state"]),
        ),
        close=lambda: states.append("store-closed"),
    )
    monkeypatch.setattr(context, "ContextStore", lambda path: store)
    fixture = coordinator, message, states
    return fixture


# 功能：
#   验证输出守卫耗时期间收到停止信号后，不能发布新改令决定或把任务标回执行中。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：在输出校验完成时触发停止。
# 输出：
#   None：不返回业务数据。
def test_interruption_stop_during_output_guard_discards_decision(tmp_path, monkeypatch):
    coordinator, message, states = _ready_interruption(tmp_path, monkeypatch)
    original_guard = runtime_interrupt.validate_runtime_model_output

    # 功能：
    #   执行真实输出守卫后模拟停止，在模型已经返回的阶段复现迟到决定问题。
    # 输入：
    #   args：守卫位置参数。
    #   kwargs：角色、结构化制品与调用记录。
    # 输出：
    #   receipts：真实守卫产生的回执列表。
    def stop_after_guard(*args, **kwargs):
        receipts = original_guard(*args, **kwargs)
        coordinator._stop.set()
        return receipts

    monkeypatch.setattr(runtime_interrupt, "validate_runtime_model_output", stop_after_guard)
    coordinator._run()
    assert coordinator.error is None
    assert coordinator.decisions == []
    assert not (coordinator.control_dir / "decisions" / f"{message.message_id}.json").exists()
    assert "executing" not in states and states[-1] == "store-closed"


# 功能：
#   验证重规划计算期间停止后不发布新轨迹，也不等待已经不能到来的执行器采纳。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：在替换轨迹计算完成时触发停止。
# 输出：
#   None：不返回业务数据。
def test_interruption_stop_during_replan_does_not_publish_replacement(tmp_path, monkeypatch):
    coordinator, message, states = _ready_interruption(tmp_path, monkeypatch, replan=True)
    calls = []

    # 功能：
    #   返回类型完整的替换制品并模拟规划期间收到停止信号，测试仅检查生命周期边界。
    # 输入：
    #   kwargs：核心传入的当前任务和轨迹绑定参数。
    # 输出：
    #   replacement：用于检查停止后发布行为的隔离替换制品。
    def late_builder(**kwargs):
        calls.append(kwargs)
        coordinator._stop.set()
        replacement = _replacement().model_copy(update={
            "message_id": message.message_id, "execution_id": message.execution_id,
        })
        return replacement

    monkeypatch.setattr(runtime_interrupt, "build_runtime_speed_replacement", late_builder)
    coordinator._run()
    assert calls and coordinator.error is None
    assert not (coordinator.control_dir / "replacements" / f"{message.message_id}.json").exists()
    assert "executing" not in states and states[-1] == "store-closed"


# 功能：
#   验证改令决定发布失败时没有成功历史，仍写入中止记录并关闭生命周期存储。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：只在决定发布时模拟 I/O 错误。
# 输出：
#   None：不返回业务数据。
def test_interruption_failed_decision_publish_does_not_enter_history(tmp_path, monkeypatch):
    coordinator, _, states = _ready_interruption(tmp_path, monkeypatch)
    original_write = runtime_interrupt._atomic_json

    # 功能：
    #   拒绝决定文件写入，其他证据仍通过真实发布器写入。
    # 输入：
    #   path：证据目标。
    #   payload：证据内容。
    #   kwargs：发布参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_decision(path, payload, **kwargs):
        if path.parent.name == "decisions":
            raise OSError("test publication failed")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(runtime_interrupt, "_atomic_json", fail_decision)
    coordinator._run()
    assert isinstance(coordinator.error, OSError)
    assert coordinator.decisions == []
    assert coordinator.abort_file.exists() and states[-1] == "store-closed"


# 功能：
#   验证生命周期存储初始化或关闭失败也进入协调器错误状态并产生中止记录，不能静默退出。
# 输入：
#   tmp_path：本次独立运行目录。
#   monkeypatch：替换生命周期存储故障点。
#   phase：初始化失败或关闭失败。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("phase", ["open", "close"])
def test_interruption_store_failure_is_reported(tmp_path, monkeypatch, phase):
    coordinator, _, _ = _ready_interruption(tmp_path, monkeypatch)
    coordinator._stop.set()

    # 功能：
    #   模拟存储构造或关闭异常。
    # 输入：
    #   args：调用方提供的存储路径等参数。
    # 输出：
    #   None：不返回业务数据。
    def fail(*args):
        raise OSError(f"test store {phase} failure")

    if phase == "open":
        monkeypatch.setattr(context, "ContextStore", fail)
    else:
        monkeypatch.setattr(context, "ContextStore", lambda path: SimpleNamespace(close=fail))
    coordinator._run()
    assert isinstance(coordinator.error, OSError)
    assert coordinator.abort_file.exists()


# 功能：
#   验证分类模型使用执行器当前采纳的任务图、轨迹摘要和目标，不再次引用最初任务的过期内容。
# 输入：
#   tmp_path：当前运行目录。
#   monkeypatch：提供独立的活动计划并截取模型输入。
# 输出：
#   None：不返回业务数据。
def test_interruption_model_input_uses_current_adopted_revision(tmp_path, monkeypatch):
    coordinator, _, _ = _ready_interruption(tmp_path, monkeypatch)
    active = _replacement()
    active.revised_task_graph.nodes[1].target_node = "new-destination"
    active.runtime_actions.task_graph_sha256 = sha256_json(active.revised_task_graph)
    active.target_node = "new-destination"
    active.replacement_sequence = 3
    monkeypatch.setattr(runtime_interrupt, "load_active_replacement", lambda *args, **kw: active)
    calls = []

    # 功能：
    #   保存模型输入后结束测试，不运行云端推理或发布控制结果。
    # 输入：
    #   kwargs：协调器传入的角色、任务上下文及输出类型。
    # 输出：
    #   result：停止后不应再访问的哨兵对象。
    def inspect_input(**kwargs):
        calls.append(kwargs["input_artifact"])
        coordinator._stop.set()
        result = object()
        return result

    coordinator.port = SimpleNamespace(call=inspect_input)
    coordinator._run()
    assert coordinator.error is None and len(calls) == 1
    assert calls[0]["current_task_graph"] == active.revised_task_graph.model_dump(mode="json")
    assert calls[0]["current_target_node"] == "new-destination"
    assert calls[0]["current_replacement_sequence"] == 3
    assert calls[0]["current_execution_track_sha256"] == sha256_json(active.track)
    assert coordinator.decisions == []


# 功能：
#   验证模型校验期间关闭会话或切换活动计划后，未关闭的 Python 线程也不能继续发布决定。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：在真实输出守卫完成后改变会话或当前计划。
#   change：会话关闭或计划替换。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", ["session_closed", "revision_changed"])
def test_interruption_publication_rechecks_session_and_revision(tmp_path, monkeypatch, change):
    coordinator, message, states = _ready_interruption(tmp_path, monkeypatch)
    active = [None]
    original_guard = runtime_interrupt.validate_runtime_model_output
    monkeypatch.setattr(runtime_interrupt, "load_active_replacement", lambda *args, **kw: active[0])

    # 功能：
    #   保留真实输出校验，并在发布之前模拟另一条控制链关闭会话或采纳新计划。
    # 输入：
    #   args：输出守卫位置参数。
    #   kwargs：模型制品、角色与记录。
    # 输出：
    #   receipts：真实守卫返回的回执。
    def change_after_guard(*args, **kwargs):
        receipts = original_guard(*args, **kwargs)
        if change == "session_closed":
            runtime_interrupt.close_runtime_control_session(coordinator.control_dir)
        else:
            active[0] = _replacement()
        return receipts

    monkeypatch.setattr(runtime_interrupt, "validate_runtime_model_output", change_after_guard)
    coordinator._run()
    assert not coordinator.decisions and "executing" not in states
    assert not (coordinator.control_dir / "decisions" / f"{message.message_id}.json").exists()
    if change == "session_closed":
        assert coordinator.error is None
    else:
        assert "ACTIVE_REVISION_CHANGED" in str(coordinator.error)
        assert coordinator.abort_file.exists()


# 功能：
#   验证改令协调器在未启动时可关闭，发布被占用时有界超时，关闭后不可启动。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：隔离生命周期存储。
# 输出：
#   None：不返回业务数据。
def test_interruption_stop_is_bounded_and_prevents_restart(tmp_path, monkeypatch):
    coordinator, _, _ = _ready_interruption(tmp_path, monkeypatch)
    coordinator._thread = threading.Thread(target=lambda: None)
    with (
        coordinator._publication_lock,
        pytest.raises(TimeoutError, match="publication did not stop"),
    ):
        coordinator.stop(0.01)
    assert coordinator._stop.is_set()
    coordinator.stop(0.1)
    coordinator.stop(0.1)
    with pytest.raises(RuntimeError, match="already stopped"):
        coordinator.start()
