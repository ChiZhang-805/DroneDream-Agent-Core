import asyncio
import gc
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from clock_fixtures import isolate_time
from test_runtime_commands import _load_executor

from dronedream_agent_core import runtime_phase_channel
from dronedream_agent_core.executor_snapshots import ACTIVE_SNAPSHOTS, ExecutorSnapshots


# 功能：
#   验证后台快照通过实际发布器恢复一次临时读锁，不刷新载荷来源时间。
# 输入：
#   tmp_path、monkeypatch：隔离路径及一次原子替换读锁注入。
# 输出：
#   None：载荷最终写出且收尾完整。
def test_snapshot_writer_recovers_transient_read_lock(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    original = Path.replace
    calls = []

    # 功能：
    #   只让第一次替换遇到读锁，其余调用使用真实文件系统。
    # 输入：
    #   self、target：暂存文件和明确目标。
    # 输出：
    #   path：成功替换后的路径。
    def transient_lock(self, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError('synthetic transient reader')
        path = original(self, target)
        return path

    monkeypatch.setattr(Path, 'replace', transient_lock)
    snapshots = ExecutorSnapshots(tmp_path)
    snapshots.submit(tmp_path / 'runtime-phase.json', {'phase': 'PREFLIGHT', 'source_time': 1000})
    summary = snapshots.close(timeout_seconds=2)
    assert summary['complete'] and len(calls) == 2
    assert json.loads((tmp_path / 'runtime-phase.json').read_text())['source_time'] == 1000
from dronedream_agent_core.local_packet_channel import LatestPacketPublisher
from dronedream_agent_core.runtime_phase_channel import PHASE_CONTRACT, RuntimePhaseBroadcaster
from dronedream_agent_core.simulation_phase_monitor import SimulationPhaseMonitor


# 功能：
#   阻塞真正后台写线程时，阶段读回、局部阶段恢复和遥测快照提交都不依赖磁盘完成。
# 输入：
#   tmp_path：测试隔离目录。
# 输出：
#   None：不返回业务数据。
def test_slow_disk_never_stalls_native_snapshot_submission_or_phase_restore(tmp_path):
    entered, release = threading.Event(), threading.Event()
    written = []

    # 功能：
    #   等待测试释放磁盘并保存实际后台写入内容。
    # 输入：
    #   path、payload、kwargs：原子发布器的路径、独占快照和预算。
    # 输出：
    #   None：不返回业务数据。
    def writer(path, payload, **kwargs):
        entered.set()
        assert release.wait(2.)
        written.append((path, payload))

    snapshots = ExecutorSnapshots(tmp_path, writer=writer)
    token = ACTIVE_SNAPSHOTS.set(snapshots)
    executor = _load_executor()
    try:
        executor._atomic_json(snapshots.phase_path, {"phase": "TRACK"})
        assert entered.wait(1.)
        for index in range(10):
            executor._atomic_json(tmp_path / "runtime-state" / "px4-identity-telemetry.json",
                                  {"sequence": index})
        executor._publish_local_control_phase(
            snapshots.phase_path, local_phase="PERCEPTION_REFRESH_HOLD", details={})
        assert snapshots.phase()["enclosing_executor_state"]["phase"] == "TRACK"
        executor._clear_local_control_phase(snapshots.phase_path)
        assert snapshots.phase() == {"phase": "TRACK"}
        assert written == []
        with snapshots._condition:
            assert len(snapshots._pending) == 2
    finally:
        ACTIVE_SNAPSHOTS.reset(token)
        release.set()
        summary = snapshots.close()
    assert summary["complete"]
    assert summary["submitted"] == summary["written"] + summary["coalesced"]
    assert written[-1][1] == {"phase": "TRACK"}
    assert any(value == {"sequence": 9} for _, value in written)


# 功能：
#   写入失败后不能伪报完整，非白名单路径不被后台写入器接管。
# 输入：
#   tmp_path：测试隔离目录。
# 输出：
#   None：不返回业务数据。
def test_snapshot_writer_failure_and_nonowned_paths_are_explicit(tmp_path):
    snapshots = ExecutorSnapshots(tmp_path, writer=Mock(side_effect=OSError("disk failed")))
    assert snapshots.submit(tmp_path / "operator-command.json", {}) is False
    snapshots.submit(snapshots.phase_path, {"phase": "TRACK"})
    summary = snapshots.close()
    assert not summary["complete"] and "disk failed" in summary["issue"]
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        snapshots.submit(snapshots.phase_path, {"phase": "COMPLETE"})


# 功能：
#   阶段通信采用独立失联预算，来源年龄不因重复读取、迟到报文或旧文件而更新。
# 输入：
#   tmp_path：当前回合端点目录。
#   monkeypatch：控制发送来源时钟和接收端时钟。
# 输出：
#   None：不返回业务数据。
def test_phase_channel_retains_original_packet_age_and_never_falls_back(tmp_path, monkeypatch):
    wall = Mock(return_value=10.08)
    mono = Mock(return_value=20.)
    isolate_time(monkeypatch, runtime_phase_channel, time=wall)
    descriptor = tmp_path / "phase-channel.json"
    phase_file = tmp_path / "runtime-phase.json"
    phase_file.write_text('{"phase":"TRACK"}')
    monitor = SimulationPhaseMonitor(channel=descriptor, clock=mono)
    monitor.open()
    publisher = LatestPacketPublisher(descriptor, contract=PHASE_CONTRACT)
    try:
        assert monitor.read(phase_file)["phase"] is None
        assert publisher.send({"schema_version": PHASE_CONTRACT.payload_schema,
                               "updated_at_unix_ms": 10_000,
                               "state": {"phase": "TRACK"}})
        for _ in range(50):
            if monitor.read(phase_file)["phase"] == "TRACK":
                break
            time.sleep(.002)
        assert monitor.read(phase_file)["read_age_seconds"] == pytest.approx(.08)
        wall.return_value, mono.return_value = 10.09, 20.01
        assert monitor.read(phase_file)["read_age_seconds"] == pytest.approx(.09)
        wall.return_value, mono.return_value = 10.101, 20.021
        assert monitor.read(phase_file)["phase"] == "TRACK"
        wall.return_value, mono.return_value = 10.499, 20.419
        assert monitor.read(phase_file)["read_age_seconds"] == pytest.approx(.499)
        wall.return_value, mono.return_value = 10.501, 20.421
        result = monitor.read(phase_file)
        assert result["phase"] is None and result["issue"] == "LIVE_RUNTIME_PHASE_UNAVAILABLE"
        # 新信封包着旧来源时间仍不能恢复阶段，背景收包也不是观测续期。
        assert publisher.send({"schema_version": PHASE_CONTRACT.payload_schema,
                               "updated_at_unix_ms": 10_000,
                               "state": {"phase": "TAKEOFF"}})
        time.sleep(.025)
        assert monitor.read(phase_file)["phase"] is None
    finally:
        publisher.close()
        monitor.close()


# 功能：
#   真套接字广播来自当前执行器状态；落盘阻塞不妨碍接收新阶段，停止后不再发送。
# 输入：
#   tmp_path：本次测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_phase_broadcast_works_while_diagnostic_disk_is_blocked(tmp_path):
    release = threading.Event()

    # 功能：
    #   阻塞诊断存储，直到测试明确释放。
    # 输入：
    #   args、kwargs：原子快照写入参数。
    # 输出：
    #   None：不返回业务数据。
    def writer(*args, **kwargs):
        assert release.wait(2.)

    # 功能：
    #   启动并消费实际阶段报文，然后按所有权关闭发布器、写入器和接收端。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        snapshots = ExecutorSnapshots(tmp_path, writer=writer)
        descriptor = tmp_path / "phase-channel.json"
        monitor = SimulationPhaseMonitor(channel=descriptor)
        monitor.open()
        broadcaster = RuntimePhaseBroadcaster([descriptor], snapshots)
        try:
            broadcaster.start()
            for phase in ("TAKEOFF", "TRACK", "LANDING"):
                snapshots.submit(snapshots.phase_path, {"phase": phase})
                for _ in range(40):
                    await asyncio.sleep(.005)
                    if monitor.read(Path("unused.json"))["phase"] == phase:
                        break
                else:
                    pytest.fail(f"live phase was not received: {phase}")
            # 连接 SDK 等非控制操作短暂占住事件循环时，阶段仍来自已提交的真实内存状态。
            time.sleep(.45)
            assert monitor.read(Path("unused.json"))["phase"] == "LANDING"
            snapshots.submit(snapshots.phase_path, {"phase": "LANDED"})
            await broadcaster.close()
            for _ in range(50):
                if monitor.read(Path("unused.json"))["phase"] == "LANDED":
                    break
                await asyncio.sleep(.002)
            assert monitor.read(Path("unused.json"))["phase"] == "LANDED"
            await asyncio.sleep(.11)
            assert monitor.read(Path("unused.json"))["phase"] == "LANDED"
        finally:
            await broadcaster.close()
            release.set()
            assert snapshots.close()["complete"]
            monitor.close()

    asyncio.run(scenario())


# 功能：
#   验证真实主入口在启动实时链之前保留 SDK 对象基线，退出恢复作用域并保存暂停记录。
# 输入：
#   tmp_path：测试证据目录。
#   monkeypatch：替换实际飞行和 SDK 导入，不建立硬件连接。
# 输出：
#   None：不返回业务数据。
def test_executor_main_freezes_sdk_baseline_before_realtime_work(tmp_path, monkeypatch):
    executor = _load_executor()
    steps = []
    previous_frozen = gc.get_freeze_count()
    args = SimpleNamespace(base_executor=tmp_path / "base.py", run_dir=tmp_path,
                           log=tmp_path / "log", native_state_channel=None,
                           local_safety_channel=None, runtime_phase_channel=[])
    monkeypatch.setattr(executor, "_parse_args", lambda: args)
    monkeypatch.setattr(executor, "_load_base", lambda _: SimpleNamespace(_log=lambda *a: None))
    monkeypatch.setattr(executor, "importlib", SimpleNamespace(
        import_module=lambda name: steps.append(name)))
    monkeypatch.setattr(executor, "configure_sensor_thread_handoff", lambda: .001)

    # 功能：
    #   在飞行替身入口检查 SDK 已初始化且循环回收保持开启。
    # 输入：
    #   args、base：当前执行器配置和非硬件基础替身。
    # 输出：
    #   None：不返回业务数据。
    async def run(args, base):
        assert steps == ["mavsdk"]
        assert gc.isenabled() and gc.get_freeze_count() > 0
        steps.append("runtime-started")

    monkeypatch.setattr(executor, "_run_with_live_snapshots", run)
    assert executor.main() == 0
    assert steps == ["mavsdk", "runtime-started"]
    if previous_frozen == 0:
        assert gc.get_freeze_count() == 0
    path = tmp_path / "runtime-state" / "executor-interpreter-pauses.json"
    report = json.loads(path.read_text())
    assert report["startup_frozen_objects"] > 0
