"""Independent control rejects missing evidence, not missing learned actors."""

import io
import json
import subprocess
import time
from types import SimpleNamespace

import pytest
from test_runtime_manager import _execution_fixture, _manager, _runtime_resources
from test_runtime_commands import _load_executor
from test_executed_control_training import teacher_evidence

from dronedream_agent_app.runtime_manager import RuntimeBridgeError
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.route_control_mode import route_observation_eligible, validate_route_control_mode

PROFILE = {"schema_version": "dronedream.control-profile.v1",
           "mode": "independent-route-v1", "simulation_only": True}
pytestmark = pytest.mark.usefixtures("isolated_wsl_host_paths")


# 功能：验证独立控制无需训练包，但不得与模型、教师、无融合模式混用。
# 输入：非法模式参数；输出：拒绝与正常路线模式的断言。
@pytest.mark.parametrize("option", [dict(provider="local-policy"), dict(model_required=True),
    dict(hybrid=True), dict(teacher=True), dict(training=True), dict(fusion=False),
    dict(heading="measured-hold"), dict(enabled="true")])
def test_route_mode_is_explicit_and_exclusive(option):
    validate_route_control_mode(enabled=True)
    with pytest.raises(ValueError):
        validate_route_control_mode(**{"enabled": True, **option})


# 功能：路线不能冒充传感器定位，缺少健康状态不通过。
# 输入：来源与健康组合；输出：仅当前机载来源被接纳。
@pytest.mark.parametrize("source", ["onboard", "external", "simulation-ground-truth", None])
@pytest.mark.parametrize("healthy", [True, False, None])
def test_onboard_evidence_remains_required(source, healthy):
    assert route_observation_eligible(source, healthy) is (source == "onboard" and healthy is True)


# 功能：飞控独立复验控制来源与坐标转换，拒绝真值补丁。
# 输入：合成安全命令；输出：原生许可与错误来源的断言，无真实飞行。
def test_executor_rejects_truth_and_wrong_authority():
    executor = _load_executor()
    _, command, _, _ = teacher_evidence()
    args = SimpleNamespace(independent_route_control=True)
    command = command.model_copy(update={"source": "onboard"})
    check = lambda c: executor._local_safety_command_matches_required_authority(args=args, command=c)
    assert check(command)
    assert not check(command.model_copy(update={"source": "simulation-ground-truth"}))
    assert not check(command.model_copy(update={"navigation_control_authority": "model-required"}))
    assert not check(command.model_copy(update={"estimator_to_world_position_offset_m": Vector3(x=.1,y=0,z=0)}))


# 功能：局部控制等待必须立即尊重停止请求，不能困在悬停/重规划内循环。
# 输入：真实执行函数与录制型停止检查；输出：读取任何控制状态前退出。
def test_local_wait_observes_abort_before_polling(tmp_path):
    import asyncio
    executor = _load_executor()
    args = SimpleNamespace(local_safety_target=None, local_safety_command=tmp_path / 'cmd',
        local_safety_repair_timeout_seconds=15., abort_file=tmp_path / 'abort')
    executor._publish_local_safety_target = lambda **kw: None
    def abort(path):
        assert path == args.abort_file
        raise RuntimeError('STOP_REQUESTED')
    with pytest.raises(RuntimeError, match='STOP_REQUESTED'):
        asyncio.run(executor._apply_local_safety(args=args,
            base=SimpleNamespace(_raise_if_external_abort_requested=abort), client=None,
            planned_setpoint=None, coordinate_contract=None, phase_path=tmp_path / 'phase'))


# 功能：保护性纯位置传输成功后必须有回执；传输失败不得记录成成功。
# 输入：录制型飞控和固定时钟；输出：成功时间与原始失败保留。
@pytest.mark.parametrize('fails', [False, True])
def test_position_only_protective_transport_receipt(fails):
    import asyncio
    executor = _load_executor()
    executor.time = SimpleNamespace(time=lambda: 1234.5)
    sent = []
    async def send(setpoint):
        if fails:
            raise RuntimeError('TRANSPORT_FAILED')
        sent.append(setpoint)
    setpoint = SimpleNamespace(north_m=1., east_m=2., down_m=-1., yaw_deg=0.)
    operation = executor._send_position_with_velocity(base=SimpleNamespace(),
        client=SimpleNamespace(set_position_ned=send), setpoint=setpoint,
        velocity_ned_mps=(0., 0., 0.))
    if fails:
        with pytest.raises(RuntimeError, match='TRANSPORT_FAILED'):
            asyncio.run(operation)
        assert sent == []
    else:
        assert asyncio.run(operation) == 1234500
        assert sent == [setpoint]


# 功能：验证损坏的发布模式不能默认为路线飞行或旧模型飞行。
# 输入：隔离资源；输出：默认关闭、合法开启及损坏拒绝。
def test_packaged_profile_is_strict(tmp_path):
    resources = _runtime_resources(tmp_path / "resources")
    manager = _manager(tmp_path / "store", resources)
    assert not manager._independent_route_control_enabled()
    path = resources / "runtime/control-profile.json"
    path.write_text(json.dumps(PROFILE), encoding="utf8")
    assert manager._independent_route_control_enabled()
    path.write_text(json.dumps({**PROFILE, "simulation_only": False}), encoding="utf8")
    with pytest.raises(RuntimeBridgeError, match="RUNTIME_CONTROL_PROFILE_INVALID"):
        manager._independent_route_control_enabled()


# 功能：把默认控制配置固定为通用发布资源，防止下次构建遗漏、仅本机可用。
# 输入：仓库配置和两条构建入口；输出：配置与融合执行器一起分发。
def test_build_entries_ship_profile_and_fusion_wrapper():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / 'runtime/control-profile.json').read_text()) == PROFILE
    for name in ('build-autonomy-windows.ps1', 'stage-development-runtime-resources.ps1'):
        script = (root / 'scripts' / name).read_text(encoding='utf8')
        assert 'control-profile.json' in script
        assert 'px4_map_fusion_experiment_executor.py' in script


# 功能：通过桌面实际 execute 入口验证整段 WSL 参数，禁止携带导航模型依赖。
# 输入：隔离资产/任务、录制型进程；输出：保留大模型检查点、融合和权限绑定。
def test_desktop_launches_independent_controller_without_actor(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, "_independent_route_control_enabled", lambda: True)
    def actor_forbidden():
        raise AssertionError("Independent control tried to load actor weights")
    monkeypatch.setattr(manager, "_local_policy_runtime_resources", actor_forbidden)
    class RecordingInput(io.StringIO):
        def close(self):
            pass
    class Process:
        stdin = RecordingInput()
        def wait(self):
            time.sleep(.05)
            return 1
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    manager.execute(**arguments)
    command = process.stdin.getvalue()
    for flag in ("--independent-route-control", "--simulation-map-fusion",
                 "--heading-policy route-tangent-relative", "--execution-authority",
                 "--checkpoint-provider", "--runtime-interrupt-provider"):
        assert flag in command
    for flag in ("--local-navigation-provider", "--local-policy-package",
                 "--local-policy-trial", "--require-local-navigation-control-authority"):
        assert flag not in command
