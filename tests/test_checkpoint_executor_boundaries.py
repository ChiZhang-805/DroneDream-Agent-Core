"""Executor boundary failures using isolated files and devices, never aircraft or model APIs."""

import asyncio
import json
import math
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_commands import _ack, _decision, _load_executor, _message, _two_segment_track

from dronedream_agent_core import runtime_control_io
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.process_capture import ProcessCaptureCancelled, capture_process
from dronedream_agent_core.runtime_commands import build_runtime_command


# 功能：
#   验证执行器拒绝损坏模型中的非有限数值，而非将其写成 null。
# 输入：
#   tmp_path：隔离输出目录。
# 输出：
#   None：不返回业务数据。
def test_executor_rejects_nonfinite_model_before_writing(tmp_path):
    executor = _load_executor()
    path = tmp_path / "new" / "record.json"
    bad = Vector3(x=1, y=0, z=0).model_copy(update={"x": float("nan")})
    with pytest.raises(ValueError):
        executor._atomic_json(path, bad)
    assert not path.parent.exists()


# 功能：
#   验证目标替换失败后只回收自有暂存，同时保留上一份完整证据。
# 输入：
#   tmp_path：已有证据目录。
#   monkeypatch：模拟持续存在的目标读锁。
# 输出：
#   None：不返回业务数据。
def test_executor_failed_replace_cleans_only_owned_temporary(tmp_path, monkeypatch):
    executor = _load_executor()
    path = tmp_path / "record.json"
    path.write_bytes(b'{"before":true}')
    foreign = tmp_path / "record.json.tmp"
    foreign.write_bytes(b"another writer")

    # 功能：
    #   拒绝本次替换，复现发布路径最终失败而非序列化失败。
    # 输入：
    #   self：发布器暂存路径。
    #   target：目标证据路径。
    # 输出：
    #   None：不返回业务数据。
    def locked(self, target):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(PermissionError):
        executor._atomic_json(path, {"after": True}, replace_timeout_seconds=0)
    assert path.read_bytes() == b'{"before":true}'
    assert foreign.read_bytes() == b"another writer"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["record.json", "record.json.tmp"]


# 功能：
#   验证并发产生同名暂存时不覆盖、也不删除另一发布者的文件。
# 输入：
#   tmp_path：隔离输出目录。
#   monkeypatch：固定共享发布器的随机暂存标识。
# 输出：
#   None：不返回业务数据。
def test_executor_uses_shared_exclusive_publication(tmp_path, monkeypatch):
    executor = _load_executor()
    monkeypatch.setattr(runtime_control_io, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    path = tmp_path / "record.json"
    foreign = tmp_path / f".record.json.{'a' * 32}.tmp"
    foreign.write_bytes(b"another writer")
    with pytest.raises(FileExistsError):
        executor._atomic_json(path, {"ready": True})
    assert foreign.read_bytes() == b"another writer" and not path.exists()


# 功能：
#   验证悬停链路异常或任务取消后，设备子任务在返回控制权前已退出。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：模拟正在运行的设备动作和发生异常的悬停。
#   cancel：是否模拟取消而非普通设备链路故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("cancel", [False, True])
def test_runtime_command_joins_device_after_hold_failure(tmp_path, monkeypatch, cancel):
    executor = _load_executor()
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(
        message,
        acknowledgement,
        action="camera_control",
        parameters={"command": "take_photo", "component_id": 100},
    )
    command = build_runtime_command(
        message=message, acknowledgement=acknowledgement, decision=decision
    )
    interruption = executor.RuntimeInterruptDetected(
        message, tmp_path / "claimed", datetime.now(UTC)
    )

    # 功能：
    #   在同一事件循环内观察设备子任务，避免 asyncio.run 的全局清理掩盖泄漏。
    # 输入：
    #   无显式参数；使用外层的 executor、command 与故障配置。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        started, stopped = asyncio.Event(), asyncio.Event()
        tasks = []

        # 功能：
        #   模拟持续运行的相机操作，在收到取消后留下真实退出证据。
        # 输入：
        #   parameters：执行器传入的相机操作参数。
        # 输出：
        #   None：不返回业务数据。
        async def camera(parameters):
            tasks.append(asyncio.current_task())
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        # 功能：
        #   等设备真正启动后使悬停链路失败，触发执行器异常清理分支。
        # 输入：
        #   kwargs：悬停控制上下文。
        # 输出：
        #   None：不返回业务数据。
        async def failed_hold(**kwargs):
            await started.wait()
            raise asyncio.CancelledError() if cancel else RuntimeError("hold failed")

        monkeypatch.setattr(executor, "_runtime_hold_tick", failed_hold)
        try:
            with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
                await executor._execute_runtime_command(
                    base=SimpleNamespace(),
                    client=SimpleNamespace(execute_camera_command=camera),
                    hold_setpoint=SimpleNamespace(),
                    interruption=interruption,
                    command=command,
                    control_dir=tmp_path,
                    abort_file=tmp_path / "abort",
                    rate_hz=20,
                    timeout_seconds=1,
                )
            assert stopped.is_set() and tasks[0].done()
            assert not (tmp_path / "adoptions" / f"{message.message_id}.json").exists()
        finally:
            # 失败版本也只清理本测试创建的任务，不影响其他测试的事件循环。
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


# 功能：
#   验证已验证负载只能增加最多五厘米的稳定半径，不能放大原本更严格的小空间容差。
# 输入：
#   tolerance：原配置位置容差，单位米。
#   expected：允许的最终位置容差，单位米。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "tolerance, expected", [(0.02, 0.07), (0.1, 0.15), (0.2, 0.25), (0.3, 0.3)]
)
def test_payload_tolerance_addition_is_bounded(tolerance, expected):
    executor = _load_executor()
    contract = SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id="attach",
                driver="payload-transition",
                parameters={"operation": "postattach-stability"},
            )
        ]
    )
    assert executor._payload_aware_waypoint_position_tolerance_m(
        configured_tolerance_m=tolerance,
        runtime_action_contract=contract,
        completed_action_step_ids={"attach"},
    ) == pytest.approx(expected)


# 功能：
#   验证恢复期限包含订阅重启耗时，不能在超期重启后继续接纳观测。
# 输入：
#   tmp_path：恢复诊断目录。
#   monkeypatch：隔离诊断文件与遥测发布，不接触飞控。
# 输出：
#   None：不返回业务数据。
def test_telemetry_recovery_bounds_stuck_restart(tmp_path, monkeypatch):
    executor = _load_executor()
    events, published = [], []
    monkeypatch.setattr(
        executor, "_record_px4_telemetry_recovery", lambda *args, **kwargs: events.append(kwargs)
    )
    monkeypatch.setattr(
        executor, "_publish_px4_identity_telemetry", lambda **kwargs: published.append(kwargs)
    )

    # 功能：
    #   仅首次采样失败，后续故意可成功，用来检查过期重启后是否误续飞。
    # 输入：
    #   seconds：本次采样预算。
    # 输出：
    #   sample：隔离位置样本。
    async def sample(seconds):
        if not events:
            raise TimeoutError("stream stalled")
        sample = SimpleNamespace(north_m=0)
        return sample

    # 功能：
    #   模拟超过整体恢复预算的订阅重启。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def restart():
        await asyncio.sleep(0.08)

    client = SimpleNamespace(
        sample_position_velocity_ned=sample, restart_position_velocity_ned_stream=restart
    )
    with pytest.raises(TimeoutError, match="did not recover within"):
        asyncio.run(
            executor._refresh_px4_identity_telemetry(
                args=SimpleNamespace(
                    tracking_telemetry_timeout_seconds=0.005,
                    tracking_telemetry_recovery_timeout_seconds=0.02,
                    run_dir=tmp_path,
                ),
                client=client,
                coordinate_contract=None,
            )
        )
    assert not published and events[-1]["status"] == "failed"


# 功能：
#   验证协程取消会传到 ROS 工作线程，并且线程退出后才向调用方传播取消。
# 输入：
#   monkeypatch：替换 ROS 可执行文件发现与进程捕获器，不调用真实服务。
# 输出：
#   None：不返回业务数据。
def test_ros2_cancel_joins_owned_worker(monkeypatch):
    executor = _load_executor()
    started, stopped = threading.Event(), threading.Event()
    monkeypatch.setattr(executor.shutil, "which", lambda _: "ros2")

    # 功能：
    #   模拟等待取消的进程捕获线程，若取消信号未传入则有界失败。
    # 输入：
    #   command：待调用服务参数。
    #   kwargs：含 cancel_event 的进程资源配置。
    # 输出：
    #   None：不返回业务数据。
    def waiting_capture(command, **kwargs):
        started.set()
        try:
            assert kwargs["cancel_event"].wait(2), "worker did not receive cancellation"
            raise ProcessCaptureCancelled("cancelled fixture")
        finally:
            stopped.set()

    # 功能：
    #   启动实际协程包装器并在其工作线程启动后发出取消，随后检查线程已退出。
    # 输入：
    #   无显式参数；使用外层事件和 executor。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        task = asyncio.create_task(
            executor._execute_ros2_domain_action(
                {"service_name": "/test", "service_type": "test/srv/Action", "request": {}},
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stopped.is_set()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(executor, "capture_process", waiting_capture)
    asyncio.run(scenario())


# 功能：
#   验证取消捕获会终止本测试实际创建的 Python 子进程，且返回前进程已被回收。
# 输入：
#   monkeypatch：记录实际创建的进程句柄，不替换进程等待和清理实现。
# 输出：
#   None：不返回业务数据。
def test_cancelled_capture_reaps_real_owned_process(monkeypatch):
    from dronedream_agent_core import process_capture

    created, cancel = [], threading.Event()
    original = process_capture.subprocess.Popen

    # 功能：
    #   保存新进程句柄并立即请求取消，用真实进程验证清理而非只检查方法调用。
    # 输入：
    #   args：进程命令。
    #   kwargs：独立环境与管道配置。
    # 输出：
    #   process：实际创建的自有进程。
    def record_process(*args, **kwargs):
        process = original(*args, **kwargs)
        created.append(process)
        cancel.set()
        return process

    monkeypatch.setattr(process_capture.subprocess, "Popen", record_process)
    with pytest.raises(ProcessCaptureCancelled):
        capture_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=b"",
            maximum_bytes=4096,
            timeout=3,
            environment=dict(os.environ),
            resource_policy=PluginResourcePolicy(),
            cancel_event=cancel,
        )
    assert len(created) == 1 and created[0].poll() is not None


# 功能：
#   验证驱动确认必须是真布尔值，不能用字符串或非零数字冒充实际设备确认。
# 输入：
#   confirmed：故意提供的错误确认类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("confirmed", ["false", "true", 1, [], None])
def test_action_evidence_requires_boolean_confirmation(confirmed):
    executor = _load_executor()
    assert (
        executor._observed_runtime_action_evidence(
            SimpleNamespace(driver="mavsdk-camera"),
            {"confirmed": confirmed},
        )
        == []
    )


# 功能：
#   验证负载就绪标记与证据列表均按原类型检查，不能把字符串的真值或字符拆分当证据。
# 输入：
#   driver：待检查的驱动类型。
#   extra：其损坏的设备回读字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "driver, extra",
    [
        (
            "payload-transition",
            {
                "operation": "precontact",
                "stable_precontact_hover": "false",
                "no_payload_contact": True,
            },
        ),
        (
            "payload-transition",
            {
                "operation": "confirm-custody",
                "detached": False,
                "payload_physics_binding_confirmed": 1,
                "custody_state_accepted": True,
            },
        ),
        (
            "payload-transition",
            {
                "operation": "postattach-stability",
                "detached": False,
                "loaded_hover_stable": True,
                "return_authorized": "false",
            },
        ),
        ("ros2-service", {"evidence": "done"}),
        ("ros2-service", {"evidence": [True]}),
    ],
)
def test_action_evidence_rejects_coerced_fields(driver, extra):
    executor = _load_executor()
    assert (
        executor._observed_runtime_action_evidence(
            SimpleNamespace(driver=driver),
            {"confirmed": True, **extra},
        )
        == []
    )


# 功能：
#   验证段编号、策略类型和有限数值异常均被稳定拒绝，不隐式转换或抛出非预期溢出。
# 输入：
#   tmp_path：隔离的策略文件目录。
#   change：将注入首段的损坏字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        {"segment_index": False},
        {"segment_index": 0.0},
        {"control_profile": []},
        {"tracking_lag_limit_m": 10**400},
    ],
    ids=["boolean-index", "float-index", "list-profile", "huge-limit"],
)
def test_tracking_policy_rejects_wrong_types(tmp_path, change):
    executor = _load_executor()
    track = _two_segment_track()
    payload = {
        "schema_version": "dronedream.tracking-corridor-policy.v1",
        "track_sha256": sha256_json(track),
        "segment_policies": [
            {
                "segment_index": i,
                "tracking_lag_limit_m": 0.2,
                "tracking_rejoin_tolerance_m": 0.1,
                "control_profile": "precision",
            }
            for i in range(2)
        ],
    }
    payload["segment_policies"][0].update(change)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        executor._load_tracking_segment_policies(path, track=track)


# 功能：
#   验证重复 JSON 键不能用后出现的空策略数组覆盖前一字段并被当成合法配置。
# 输入：
#   tmp_path：隔离的策略文件目录。
# 输出：
#   None：不返回业务数据。
def test_tracking_policy_rejects_duplicate_keys(tmp_path):
    executor = _load_executor()
    track = _two_segment_track()
    segment = {
        "tracking_lag_limit_m": 0.2,
        "tracking_rejoin_tolerance_m": 0.1,
        "control_profile": "precision",
    }
    policies = [{**segment, "segment_index": i} for i in range(2)]
    text = (
        '{"schema_version":"dronedream.tracking-corridor-policy.v1",'
        f'"track_sha256":"{sha256_json(track)}",'
        '"segment_policies":null,"segment_policies":' + json.dumps(policies) + "}"
    )
    path = tmp_path / "policy.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        executor._load_tracking_segment_policies(path, track=track)


# 功能：
#   验证基础模块初始化失败后恢复原模块注册，不遗留半初始化对象供后续调用。
# 输入：
#   tmp_path：隔离模块文件目录。
#   monkeypatch：保存并恢复模块注册表的测试状态。
# 输出：
#   None：不返回业务数据。
def test_failed_base_import_restores_previous_module(tmp_path, monkeypatch):
    executor = _load_executor()
    name = "dronedream_proven_px4_base"
    previous = SimpleNamespace(ready=True)
    monkeypatch.setitem(sys.modules, name, previous)
    path = tmp_path / "broken_base.py"
    path.write_text("raise RuntimeError('fixture initialization failed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fixture initialization failed"):
        executor._load_base(path)
    assert sys.modules[name] is previous
    assert not (tmp_path / "__pycache__").exists()


# 功能：
#   重复加载基础执行器时不在只读资源树留下缓存，也不复用同长度旧源码的缓存。
# 输入：
#   tmp_path：隔离资源目录；monkeypatch：恢复模块注册表。
# 输出：
#   无。
def test_base_import_preserves_resource_inventory(tmp_path, monkeypatch):
    executor = _load_executor()
    monkeypatch.setitem(sys.modules, "dronedream_proven_px4_base", None)
    path = tmp_path / "base.py"
    path.write_bytes(b"answer = 1\n")
    assert executor._load_base(path).answer == 1
    path.write_bytes(b"answer = 2\n")
    assert executor._load_base(path).answer == 2
    assert list(tmp_path.iterdir()) == [path]


# 功能：
#   验证替换检查点稳定等待继续使用本地控制刷新，不能只回读遥测却下发旧计划目标。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：截获稳定等待入口，在设备或云端调用前停止测试。
# 输出：
#   None：不返回业务数据。
def test_replacement_checkpoint_preserves_control_refresh(tmp_path, monkeypatch):
    executor = _load_executor()
    refresh = object()

    # 功能：
    #   在到达稳定等待入口时验证控制回调身份，避免用替身模拟整个飞行流程。
    # 输入：
    #   kwargs：替换检查点传入的稳定等待配置。
    # 输出：
    #   None：不返回业务数据。
    async def inspect_settle(**kwargs):
        assert kwargs.get("setpoint_refresh") is refresh
        raise RuntimeError("checked stable boundary")

    monkeypatch.setattr(executor, "_wait_checkpoint_stable", inspect_settle)
    with pytest.raises(RuntimeError, match="checked stable boundary"):
        asyncio.run(
            executor._review_replacement_checkpoint(
                args=SimpleNamespace(setpoint_rate_hz=20, run_dir=tmp_path),
                base=None,
                client=None,
                setpoint=None,
                checkpoint=None,
                checkpoint_contract=None,
                runtime_action_contract=None,
                track=_two_segment_track(),
                completed_action_step_ids=set(),
                completed_action_task_ids=set(),
                runtime_interrupt_probe=lambda: None,
                setpoint_refresh=refresh,
                sample_observer=None,
                target_frame_position_resolver=None,
                timing={},
            )
        )


# 功能：
#   验证请求回显及证据内的成功字样不会覆盖真正的服务失败状态，保留含括号的完整证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_ros_response_ignores_request_and_quoted_success():
    executor = _load_executor()
    result = executor._parse_ros2_domain_action_response(
        "requester: Request(success=True)\nresponse:\n"
        "pkg.Response(details_json='success=true', success=False, "
        "evidence=['check [door]'], issue_code='not ready')"
    )
    assert result["confirmed"] is False
    assert result["evidence"] == ["check [door]"]
    assert result["details_json"] == "success=true"


# 功能：
#   验证缺少独立响应、重复成功字段及非字面调用均被拒绝，不执行响应中的代码。
# 输入：
#   output：损坏的 ROS 输出文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "output",
    [
        "requester: Request(success=True)",
        "response: pkg.Response(success=True, success=False)",
        "response: pkg.Response(success=True, evidence=list())",
        "response: pkg.Response(success='true')",
    ],
)
def test_ros_response_rejects_ambiguous_or_executable_fields(output):
    with pytest.raises(RuntimeError):
        _load_executor()._parse_ros2_domain_action_response(output)


# 功能：
#   验证调度器可直接解析冻结快照，后续文件变化不会改变已构建计划。
# 输入：
#   tmp_path：隔离轨迹文件目录。
# 输出：
#   None：不返回业务数据。
def test_reference_plan_and_contract_share_one_snapshot(tmp_path):
    executor = _load_executor()
    base = executor._load_base(Path(__file__).parents[1] / "runtime/px4_offboard_track_executor.py")
    track = _two_segment_track()
    path = tmp_path / "track.json"
    path.write_text(track.model_dump_json(), encoding="utf-8")
    payload = executor.read_runtime_object(path)
    path.write_text("{}", encoding="utf-8")
    reference = base.parse_reference_track_plan(payload)
    frozen = executor.Px4Track.model_validate(payload)
    payload["points"][0]["x"] = 99
    assert reference.points[0].x == frozen.points[0].x == track.points[0].x


# 功能：
#   验证电池采样等待期间不断刷新控制，不重复发送已经失去实时意义的固定计划目标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_battery_wait_renews_live_control_each_tick():
    executor = _load_executor()
    base = executor._load_base(Path(__file__).parents[1] / "runtime/px4_offboard_track_executor.py")
    client = base.FakeOffboardClient()
    refreshed = []

    # 功能：
    #   模拟有延迟但有界的真实电池读取，让等待器有机会多次维持控制。
    # 输入：
    #   timeout：本次采样时限。
    # 输出：
    #   battery：实际返回给等待器的电池样本。
    async def battery(timeout):
        await asyncio.sleep(0.09)
        return {"remaining_percent": 0.7, "voltage_v": 15.7}

    # 功能：
    #   为每轮维持操作提供可区别的目标，检查等待器确实发送了该轮刷新结果。
    # 输入：
    #   planned：不应被原样发送的原始计划目标。
    # 输出：
    #   setpoint：本轮刷新目标。
    async def refresh(planned):
        setpoint = base.Setpoint(0, 0, -1, len(refreshed) + 1)
        refreshed.append(setpoint)
        return setpoint

    client.sample_battery = battery
    result = asyncio.run(
        executor._sample_checkpoint_battery(
            base=base,
            client=client,
            setpoint=base.Setpoint(9, 9, -9, 0),
            rate_hz=100,
            timeout_seconds=1,
            sample_timeout_seconds=0.5,
            setpoint_refresh=refresh,
        )
    )
    assert result["remaining_percent"] == 0.7
    assert len(refreshed) >= 3 and client.setpoints == refreshed


# 功能：
#   验证新落盘的 RGB 记录仍保留旧采样年龄，不能通过重新记录绕过图像失效门槛。
# 输入：
#   tmp_path：隔离机载数据集目录。
#   monkeypatch：只把当前测试指向该隔离数据集。
#   sample_time：原图像的单调采样时间。
#   status_age：记录快照中报告的原始传感器年龄。
#   accepted：是否应被接纳。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "sample_time,status_age,accepted",
    [
        (10, 0.01, True),
        (5, 0.01, False),
        (10, 4, False),
        (11, 0.01, False),
    ],
)
def test_rgb_capture_age_includes_sensor_source(
    tmp_path, monkeypatch, sample_time, status_age, accepted
):
    import hashlib

    executor = _load_executor()
    rgb = b"fixture hash binding, not a decoded image"
    digest = hashlib.sha256(rgb).hexdigest()
    (tmp_path / "rgb").mkdir()
    (tmp_path / "rgb" / f"{digest}.png").write_bytes(rgb)
    payload = {
        "sample_id": "sensor-sample-" + "a" * 24,
        "flight_id": "fixture",
        "map_sha256": "b" * 64,
        "recorded_at_unix_ms": 10000,
        "recorded_at_monotonic_seconds": 10,
        "rgb_sample_monotonic_seconds": sample_time,
        "rgb_relative_path": f"rgb/{digest}.png",
        "rgb_sha256": digest,
        "depth_frame_sha256": "c" * 64,
        "state": {},
        "previous_record_sha256": "0" * 64,
        "sensor_snapshot": {
            "captured_at_monotonic_seconds": 10,
            "contract_set_sha256": "d" * 64,
            "ready_for_motion": True,
            "active_sensor_ids": ["rgb"],
            "statuses": [
                {
                    "sensor_id": "rgb",
                    "modality": "rgb-camera",
                    "required_for_motion": True,
                    "latest_sequence": 1,
                    "sample_age_seconds": status_age,
                    "health": "healthy",
                }
            ],
        },
        "record_sha256": "0" * 64,
    }
    # 先补齐契约默认字段，再计算真实记录摘要，确保失败来自时效而不是损坏散列。
    # 先建立合法记录并补齐默认字段，再显式注入未来时钟；不能在夹具建立时提前失败。
    record = executor.RuntimeMultimodalDatasetRecord.model_validate(
        {**payload, "rgb_sample_monotonic_seconds": 10}
    ).model_dump(mode="json")
    record["rgb_sample_monotonic_seconds"] = float(sample_time)
    record.pop("record_sha256")
    record["record_sha256"] = sha256_json(record)
    (tmp_path / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setenv("DRONEDREAM_SIMULATION_ONBOARD_RGB_DATASET", str(tmp_path))
    parameters = {"command": "take_photo", "component_id": 100}
    if accepted:
        result = executor._verified_simulation_onboard_rgb_capture(parameters, now_unix_ms=10500)
        assert result["frame_age_seconds"] == pytest.approx(0.51)
    else:
        with pytest.raises(
            RuntimeError, match="source.*(stale|invalid)|capture failed: ValidationError"
        ):
            executor._verified_simulation_onboard_rgb_capture(parameters, now_unix_ms=10500)


# 功能：
#   验证很大沿线误差下仍保留厘米级横向偏差，避免平方相减把净空消耗算为零。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cross_track_error_preserves_small_perpendicular_component():
    result = _load_executor()._route_tracking_error_components(
        observed_route_frame_ned=(0.0, 0.0, 0.0),
        planned_setpoint=SimpleNamespace(north_m=1e8, east_m=0.01, down_m=0.0),
        planned_velocity_ned_mps=(1.0, 0.0, 0.0),
    )
    assert result["cross_track_error_m"] == pytest.approx(0.01)
    assert result["along_track_lag_m"] == pytest.approx(1e8)


# 功能：
#   验证原始速度足够大时限幅仍按正确范数缩放，不因平方溢出退化成错误的零速度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_feedforward_bounds_large_representable_velocity():
    result = _load_executor()._schedule_velocity_feedforward(
        current_setpoint=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=0.0),
        next_setpoint=SimpleNamespace(north_m=1e200, east_m=1e200, down_m=0.0),
        rate_hz=1.0,
        speed_limit_mps=0.2,
        recovery_active=False,
    )
    assert math.hypot(*result) == pytest.approx(0.2)
    assert result[0] == pytest.approx(result[1])


# 功能：
#   为三种恢复窗口提供独立的最小合法输入，便于检查共同的数值边界。
# 输入：
#   executor：当前检查点执行器。
#   kind：语义进度、跟踪恢复或局部修复窗口。
# 输出：
#   fixture：待调用窗口函数、参数及其主要时限字段的三元组。
def _progress_fixture(executor, kind):
    if kind == "semantic":
        return (
            executor._advance_model_semantic_progress_window,
            dict(
                now=0.0,
                recovery_after_seconds=3.0,
                abort_after_seconds=9.0,
                state=None,
                navigation_goal_id="goal",
                model_goal_distance_m=1.0,
            ),
            "recovery_after_seconds",
        )
    if kind == "tracking":
        return (
            executor._advance_tracking_recovery_window,
            dict(
                now=0.0,
                timeout_seconds=3.0,
                state=None,
                tracking_error_m=0.2,
                model_goal_distance_m=1.0,
            ),
            "timeout_seconds",
        )
    return (
        executor._advance_local_repair_progress,
        dict(
            now=0.0,
            observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=0.0),
            minimum_clearance_m=0.2,
            stall_timeout_seconds=3.0,
            state=None,
        ),
        "stall_timeout_seconds",
    )


# 功能：
#   验证窗口不能接受非有限、布尔或不可表示的时间输入，不能把坏时限钳成默认值。
# 输入：
#   kind：待检查的窗口。
#   value：非法秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["semantic", "tracking", "local"])
@pytest.mark.parametrize(
    "value",
    [True, "1", float("nan"), float("inf"), 10**400],
    ids=["boolean", "text", "nan", "infinity", "huge-integer"],
)
def test_progress_windows_reject_invalid_time_inputs(kind, value):
    function, kwargs, budget = _progress_fixture(_load_executor(), kind)
    with pytest.raises(ValueError):
        function(**{**kwargs, "now": value})
    with pytest.raises(ValueError):
        function(**{**kwargs, budget: value})


# 功能：
#   验证窗口记录上次检查时刻，系统时钟回退不能悄悄恢复已消耗的等待预算。
# 输入：
#   kind：待检查的窗口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["semantic", "tracking", "local"])
def test_progress_windows_reject_clock_reversal(kind):
    function, kwargs, _ = _progress_fixture(_load_executor(), kind)
    result = function(**kwargs)
    state = result if kind == "local" else result[0]
    with pytest.raises(ValueError, match="clock"):
        function(**{**kwargs, "now": -1.0, "state": state})


# 功能：
#   验证真正超期后的迟到进展不能再次续期；语义授权标志也不能复活旧窗口。
# 输入：
#   kind：待检查的窗口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["semantic", "tracking", "local"])
def test_expired_progress_window_cannot_be_revived(kind):
    function, kwargs, _ = _progress_fixture(_load_executor(), kind)
    initial = function(**kwargs)
    state = initial if kind == "local" else initial[0]
    changed = {"model_goal_distance_m": 0.0}
    if kind == "semantic":
        changed["authorized_schedule_advance"] = True
    elif kind == "local":
        changed = {"minimum_clearance_m": 2.0}
    result = function(**{**kwargs, "now": 10.0, "state": state, **changed})
    if kind == "local":
        assert result["stall_deadline"] == 3.0
    else:
        assert result[1] is True
    assert state["progress_revision"] == 0


# 功能：
#   验证语义进展不能使用字符串真假值冒充当前已授权的路线推进。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_semantic_progress_requires_boolean_advance_authority():
    function, kwargs, _ = _progress_fixture(_load_executor(), "semantic")
    with pytest.raises(ValueError):
        function(**{**kwargs, "authorized_schedule_advance": "false"})


# 功能：
#   验证人工指令来自未来或净空检查消耗完授权期限时，实际飞控不会收到运动指令。
# 输入：
#   tmp_path：隔离人工接管目录。
#   monkeypatch：控制本测试的时钟与净空检查耗时。
#   future_command：是否测试未来命令，而非计算期间授权过期。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("future_command", [False, True])
def test_operator_authorization_rechecked_before_dispatch(tmp_path, monkeypatch, future_command):
    from test_runtime_follow import _modules, _write_assets

    base, executor = _modules()
    semantic, vehicle = _write_assets(tmp_path)
    clock = [datetime.now(UTC)]
    grant = executor.RuntimeOperatorTakeoverGrant(
        message_id="runtime-msg-" + "b" * 32,
        execution_id="execution-" + "c" * 32,
        operator_id="operator-a",
        message_sha256="1" * 64,
        hold_ack_sha256="2" * 64,
        decision_sha256="3" * 64,
        grant_token_sha256="4" * 64,
        maximum_horizontal_speed_mps=1.0,
        maximum_vertical_speed_mps=1.0,
        maximum_yaw_rate_dps=90.0,
        deterministic_gates={"stable_hold": True},
        issued_at=clock[0],
        expires_at=clock[0] + timedelta(seconds=10),
    )
    command = executor.RuntimeOperatorControlCommand(
        message_id=grant.message_id,
        execution_id=grant.execution_id,
        grant_sha256=sha256_json(grant),
        sequence=1,
        action="velocity",
        velocity_ned_mps=Vector3(x=0.2, y=0, z=0),
        yaw_rate_dps=0,
        duration_seconds=0.1,
        issued_at=clock[0] + timedelta(seconds=1 if future_command else 0),
    )
    control = tmp_path / "control"
    (control / "operator-commands").mkdir(parents=True)
    (control / "operator-commands/00000001.json").write_text(
        command.model_dump_json(),
        encoding="utf-8",
    )

    # 功能：
    #   模拟净空计算结束时授权已经过期，用真实发送记录验证最终门槛。
    # 输入：
    #   args：路线和地图参数。
    #   kwargs：车辆碰撞尺寸。
    # 输出：
    #   clearance：几何层认为可以通过的结果，不能替代时间授权。
    def expire_during_clearance(*args, **kwargs):
        clock[0] = grant.expires_at + timedelta(milliseconds=1)
        return SimpleNamespace(accepted=True)

    monkeypatch.setattr(executor, "datetime", SimpleNamespace(now=lambda zone: clock[0]))
    monkeypatch.setattr(executor, "validate_route_clearance", expire_during_clearance)
    client = base.FakeOffboardClient()
    with pytest.raises(executor.UserDirectedLanding, match="fresh|expired during clearance"):
        asyncio.run(
            executor._run_operator_takeover(
                base=base,
                client=client,
                hold_setpoint=base.Setpoint(0, 0, -0.8, 0),
                interruption=None,
                grant=grant,
                control_dir=control,
                abort_file=tmp_path / "abort",
                rate_hz=20,
                runtime_session=None,
                active_track_sha256="5" * 64,
                params=None,
                hold_timeout_seconds=1,
                decision_timeout_seconds=1,
                replan_hold_seconds=1,
                semantic_path=semantic,
                vehicle_metadata_path=vehicle,
                coordinate_contract=executor.Px4CoordinateContract(
                    model_root_world_enu_m=[0.0, 0.0, 0.0],
                    collision_center_offset_model_m=[0.0, 0.0, 0.2],
                ),
            )
        )
    assert client.setpoints == []
    assert (control / "operator-commands/00000001.json").exists()


# 功能：
#   验证入口任一通道关闭失败也会尝试关闭另一通道，并返回失败退出码而非假成功。
# 输入：
#   monkeypatch：用隔离通道替身替代实际本地通信资源。
#   tmp_path：当前测试的暂停诊断目录。
# 输出：
#   None：不返回业务数据。
def test_executor_main_closes_all_channels_after_cleanup_failure(monkeypatch, tmp_path):
    from dronedream_agent_core import local_safety_channel, native_state_channel

    executor = _load_executor()
    closed = []
    args = SimpleNamespace(
        base_executor=Path("fixture"),
        native_state_channel=Path("native"),
        local_safety_channel=Path("safety"),
        local_safety_command=Path("command"),
        local_safety_observation=Path("observation"),
        log=Path("log"),
        run_dir=tmp_path,
        runtime_phase_channel=[],
    )

    # 功能：
    #   模拟先关闭的安全通道失败，后续仍必须尝试释放原生发布通道。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def failed_close():
        closed.append("safety")
        raise OSError("fixture close failed")

    # 功能：
    #   只验证入口资源生命周期，不运行设备或仿真。
    # 输入：
    #   args：隔离通道配置。
    #   base：隔离日志端口。
    # 输出：
    #   None：不返回业务数据。
    async def isolated_run(args, base):
        closed.append("run")

    monkeypatch.setattr(executor, "_parse_args", lambda: args)
    monkeypatch.setattr(executor, "_load_base", lambda _: SimpleNamespace(_log=lambda *a: None))
    monkeypatch.setattr(executor, "importlib", SimpleNamespace(import_module=lambda _: None))
    monkeypatch.setattr(executor, "configure_sensor_thread_handoff", lambda: .001)
    monkeypatch.setattr(executor, "_run", isolated_run)
    monkeypatch.setattr(
        native_state_channel,
        "NativeStatePublisher",
        lambda _: SimpleNamespace(close=lambda: closed.append("native")),
    )
    monkeypatch.setattr(
        local_safety_channel, "LocalSafetyReceiver", lambda _: SimpleNamespace(close=failed_close)
    )
    assert executor.main() == 2
    assert closed == ["run", "safety", "native"]
