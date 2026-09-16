"""Read and teardown tests only; this module never launches a simulator."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_geometry_motion_fixture import CAMERA, WORLD


# 功能：
#   加载采集脚本供单元测试调用，不执行命令行入口或导入原生 Gazebo 模块。
# 输入：
#   无。
# 输出：
#   module：本次测试独立的脚本模块。
def script():
    path = Path(__file__).parents[1] / "scripts/run_geometry_motion_fixture.py"
    spec = importlib.util.spec_from_file_location("geometry_capture_script_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：
#   在隔离临时目录创建四项小型输入及输出参数，供纯文件准备测试使用。
# 输入：
#   tmp_path：pytest 的隔离目录。
# 输出：
#   args：未启动仿真的采集参数对象。
def source_args(tmp_path):
    args = SimpleNamespace(world=tmp_path / "world.sdf", camera=tmp_path / "camera.sdf",
        semantic=tmp_path / "semantic.json", route=tmp_path / "route.json",
        output=tmp_path / "output", duration=16., wall_timeout=180., debugger=False,
        retain_wsl_d3d12=False)
    args.world.write_bytes(WORLD)
    args.camera.write_bytes(CAMERA)
    args.semantic.write_bytes(b"{}")
    args.route.write_text(json.dumps({"positions_m": [
        {"x": 1., "y": 2., "z": 3.}, {"x": 2., "y": 2., "z": 3.}]}), encoding="utf-8")
    return args


# 功能：
#   来源读取后变化不能影响已验证的解析结果，结束复核必须识别这种变化。
# 输入：
#   tmp_path：pytest 隔离目录。
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_source_snapshot_parses_the_bytes_it_hashed(tmp_path, monkeypatch):
    module, args = script(), source_args(tmp_path)
    original = module.load_source

    # 功能：
    #   在读取路线后改写其文件，模拟两次打开同一路径会得到不同字节的情形。
    # 输入：
    #   path：读取路径。
    #   maximum：字节预算。
    # 输出：
    #   raw：改写之前已经读取的字节。
    def changing_source(path, maximum):
        raw = original(path, maximum)
        if path == args.route:
            path.write_text('{"positions_m":[]}', encoding="utf-8")
        return raw

    monkeypatch.setattr(module, "load_source", changing_source)
    snapshot = module.source_snapshot(args)
    assert snapshot["origin"] == [1., 2., 3.]
    assert snapshot["direction"] == [1., 0., 0.]
    assert module.sources_match(snapshot["sources"]) is False


# 功能：
#   路线来源缺失、类型混淆和重复键在任何原生启动之前被拒绝。
# 输入：
#   tmp_path：pytest 隔离目录。
#   raw：候选路线字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("raw", [b'[]', b'{"positions_m":[]}',
    b'{"positions_m":[],"positions_m":[]}',
    b'{"positions_m":[{"x":true,"y":0,"z":1},{"x":2,"y":0,"z":1}]}',
    b'{"positions_m":[{"x":"1","y":0,"z":1},{"x":2,"y":0,"z":1}]}'])
def test_source_snapshot_rejects_invalid_route(tmp_path, raw):
    module, args = script(), source_args(tmp_path)
    args.route.write_bytes(raw)
    with pytest.raises(ValueError):
        module.source_snapshot(args)
    assert not args.output.exists()


# 功能：
#   删除来源文件后应记录不一致而非中止其余证据收尾。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_missing_source_is_an_explicit_mismatch(tmp_path):
    module, args = script(), source_args(tmp_path)
    snapshot = module.source_snapshot(args)
    assert module.sources_match(snapshot["sources"]) is True
    args.route.unlink()
    assert module.sources_match(snapshot["sources"]) is False


# 功能：
#   绑定后修改世界必须在渲染准备和原生导入之前拒绝，不生成采用错误来源的夹具。
# 输入：
#   tmp_path：pytest 隔离目录。
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_run_binds_rendering_to_verified_world(tmp_path, monkeypatch):
    module, args = script(), source_args(tmp_path)
    original = module.source_snapshot

    # 功能：
    #   模拟生成源快照后世界内容发生变化。
    # 输入：
    #   parameters：采集参数。
    # 输出：
    #   snapshot：变化前的已验证来源。
    def changed_world(parameters):
        snapshot = original(parameters)
        parameters.world.write_bytes(WORLD.replace(b'name="map"', b'name="changed-map"'))
        return snapshot

    monkeypatch.setattr(module, "source_snapshot", changed_world)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux", byteorder="little"))
    with pytest.raises(ValueError, match="SOURCE_BINDING"):
        module.run(args)
    assert not (args.output / "render").exists()


# 功能：
#   序列化错误不创建半文件；排他写入也不能覆盖已有证据。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_json_is_validated_before_exclusive_creation(tmp_path):
    module, path = script(), tmp_path / "record.json"
    with pytest.raises(ValueError):
        module.write_json(path, {"value": float("nan")})
    assert not path.exists()
    module.write_json(path, {"cell": (1, 2, 3)})
    assert json.loads(path.read_text()) == {"cell": [1, 2, 3]}
    with pytest.raises(FileExistsError):
        module.write_json(path, {"changed": True})
    assert json.loads(path.read_text()) == {"cell": [1, 2, 3]}


class SimulatedProcess:
    # 功能：
    #   建立不关联操作系统进程的等待状态，模拟第几次等待才能结束。
    # 输入：
    #   self：当前测试对象。
    #   finish_after：可以结束的等待次数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, finish_after):
        self.finish_after = finish_after
        self.pid, self.returncode, self.waits = 123456, None, []

    # 功能：
    #   返回测试对象当前退出码，不查询真实进程。
    # 输入：
    #   self：当前测试对象。
    # 输出：
    #   code：退出码或仍存活标记 None。
    def poll(self):
        code = self.returncode
        return code

    # 功能：
    #   按预设次数制造超时，不进行实际睡眠。
    # 输入：
    #   self：当前测试对象。
    #   timeout：生产代码申请的等待秒数。
    # 输出：
    #   code：最终退出码。
    def wait(self, *, timeout):
        self.waits.append(timeout)
        if len(self.waits) < self.finish_after:
            raise subprocess.TimeoutExpired("fake-simulator", timeout)
        if self.returncode is None:
            self.returncode = 0
        code = self.returncode
        return code


# 功能：
#   诊断读取失败后仍执行强制回收，且同时保留诊断失败和强制退出记录。
# 输入：
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_diagnostic_failure_cannot_skip_forced_cleanup(monkeypatch):
    module, process = script(), SimulatedProcess(3)
    signals, errors = [], []

    # 功能：
    #   记录信号但不向系统发送，强制终止时更新假进程退出码。
    # 输入：
    #   target：假进程。
    #   signum：拟发送的信号编号。
    # 输出：
    #   None：不返回业务数据。
    def record_signal(target, signum):
        assert target is process
        signals.append(signum)
        if signum == 9:
            target.returncode = -9

    # 功能：
    #   模拟读取 proc 状态时进程诊断文件缺失。
    # 输入：
    #   target：假进程。
    # 输出：
    #   None：不返回业务数据。
    def failed_diagnostic(target):
        raise FileNotFoundError("proc status disappeared")

    monkeypatch.setattr(module, "signal", SimpleNamespace(SIGINT=2, SIGKILL=9))
    monkeypatch.setattr(module, "signal_owned_process", record_signal)
    monkeypatch.setattr(module, "stop_diagnostic", failed_diagnostic)
    assert module.stop_owned_simulator(process, False, errors) is None
    assert signals == [2, 9] and process.waits == [8, 12, 5]
    assert any(error.startswith("STOP_DIAGNOSTIC:") for error in errors)
    assert "SIMULATOR_REQUIRED_FORCED_TERMINATION" in errors


# 功能：
#   已确认停止且正常退出时不发额外信号、不记录清理失败，重复检查已退出对象不再等待。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_acknowledged_server_exits_without_forcing():
    module, process, errors = script(), SimulatedProcess(1), []
    assert module.stop_owned_simulator(process, True, errors) is None
    assert process.waits == [8] and errors == []
    assert module.stop_owned_simulator(process, True, errors) is None
    assert process.waits == [8]


# 功能：
#   虚拟诊断读取也必须保持真实字节上限，不允许负预算退化为无限量读取。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_proc_diagnostics_have_a_real_read_bound(tmp_path):
    module, path = script(), tmp_path / "status"
    path.write_bytes(b"12345")
    assert module.proc_text(path, 5) == "12345"
    with pytest.raises(ValueError, match="TOO_LARGE"):
        module.proc_text(path, 4)
    with pytest.raises(ValueError, match="BUDGET_INVALID"):
        module.proc_text(path, -1)


# 功能：
#   在完整脚本调用链中注入主循环、诊断和退订失败，核对仍回收自有进程并落盘失败报告。
# 输入：
#   tmp_path：pytest 隔离目录。
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_run_preserves_failure_report_and_completes_cleanup(tmp_path, monkeypatch):
    module, args, process = script(), source_args(tmp_path), SimulatedProcess(1)
    calls = []

    # 功能：
    #   假订阅只记录主题，不调用原生库或启动消息接收线程。
    # 输入：
    #   kind：消息类型。
    #   topic：订阅主题。
    #   callback：生产回调函数。
    # 输出：
    #   accepted：固定为 True，允许测试进入主循环。
    def subscribe(kind, topic, callback):
        calls.append(("subscribe", topic))
        accepted = True
        return accepted

    # 功能：
    #   在第一项退订抛错，第二项返回无确认，验证两类失败都可被保存且不会跳过后续步骤。
    # 输入：
    #   topic：正在取消的主题。
    # 输出：
    #   None：不返回业务数据。
    def unsubscribe(topic):
        calls.append(("unsubscribe", topic))
        if topic == module.POSE_TOPIC:
            raise RuntimeError("unsubscribe failed")

    # 功能：
    #   在主循环触发可定位的测试异常，启动完整异常收尾路径。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def services():
        raise RuntimeError("injected capture-loop failure")

    # 功能：
    #   返回无原生行为的消息节点。
    # 输入：
    #   无。
    # 输出：
    #   node：假订阅、退订与服务发现节点。
    def node_factory():
        node = SimpleNamespace(subscribe=subscribe, unsubscribe=unsubscribe, service_list=services)
        return node

    # 功能：
    #   模拟停止服务器没有得到确认，不发出真实停止请求。
    # 输入：
    #   无。
    # 输出：
    #   accepted：固定为 False。
    def stop_server():
        calls.append(("stop_server",))
        accepted = False
        return accepted

    # 功能：
    #   记录通信进程正常关闭。
    # 输入：
    #   无。
    # 输出：
    #   complete：固定为 True。
    def close_transport():
        calls.append(("close_transport",))
        complete = True
        return complete

    # 功能：
    #   返回只提供清理接口的假通信客户端。
    # 输入：
    #   partition：脚本分配的隔离分区。
    #   world：夹具世界名。
    # 输出：
    #   client：没有原生进程的测试客户端。
    def transport_factory(partition, world):
        client = SimpleNamespace(stop_server=stop_server, close=close_transport)
        return client

    # 功能：
    #   核对独立会话参数后返回假仿真进程，禁止运行任何实际命令。
    # 输入：
    #   command：脚本准备的命令参数。
    #   options：日志、环境及进程隔离参数。
    # 输出：
    #   process：受控进程对象。
    def spawn(command, **options):
        assert options["start_new_session"] is True
        calls.append(("spawn",))
        return process

    # 功能：
    #   模拟停止前内存映射读取失败，验证该失败不能中断退订及进程清理。
    # 输入：
    #   target：假进程。
    #   debugger：调试器开关。
    #   output：证据目录。
    # 输出：
    #   None：不返回业务数据。
    def failed_maps(target, debugger, output):
        raise OSError("diagnostic unavailable")

    # 功能：
    #   记录退出信号，不调用系统信号接口。
    # 输入：
    #   target：假进程。
    #   signum：待发送信号。
    # 输出：
    #   None：不返回业务数据。
    def record_signal(target, signum):
        assert target is process
        calls.append(("signal", signum))

    image_type = SimpleNamespace(DESCRIPTOR=SimpleNamespace(fields_by_name={
        "pixel_format_type": SimpleNamespace(enum_type=SimpleNamespace(values_by_number={}))}))
    for name in ("gz", "gz.msgs10"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, exports in {
        "gz.msgs10.image_pb2": {"Image": image_type},
        "gz.msgs10.pose_pb2": {"Pose": object()},
        "gz.transport13": {"Node": node_factory},
    }.items():
        replacement = ModuleType(name)
        replacement.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, replacement)
    monkeypatch.setenv("GZ_PARTITION", "test-before-capture")
    monkeypatch.setattr(module, "sys", SimpleNamespace(
        platform="linux", byteorder="little", path=[]))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(
        Popen=spawn, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(module, "signal", SimpleNamespace(SIGINT=2, SIGKILL=9))
    monkeypatch.setattr(module, "FixturePoseRequests", transport_factory)
    monkeypatch.setattr(module, "capture_process_maps", failed_maps)
    monkeypatch.setattr(module, "signal_owned_process", record_signal)
    assert module.run(args) == 1
    report = json.loads((args.output / "capture.json").read_text())
    assert report["complete"] is False and report["sources_unchanged"] is True
    assert report["issue"] == "RuntimeError:injected capture-loop failure"
    assert any(error.startswith("PRE_STOP_MAPS:") for error in report["close_errors"])
    assert any(error.startswith("UNSUBSCRIBE:") for error in report["close_errors"])
    assert ("unsubscribe", module.DEPTH_TOPIC) in calls
    assert ("stop_server",) in calls and ("close_transport",) in calls and ("signal", 2) in calls
    assert process.waits == [8]
