"""旁路模型协调器拥有的连接不能在停止或初始化失败后继续留存。"""

import threading
from types import SimpleNamespace

import pytest
from test_checkpointing import _ready_coordinator
from test_mission_verification import _artifacts

from dronedream_agent_core import checkpointing


# 功能：
#   检查未启动、正在退出两种生命周期都回收模型端口，而不请求真实供应商。
# 输入：
#   tmp_path：隔离的执行目录。
#   phase：停止未启动协调器，或已经进入运行循环后退出。
# 输出：
#   None：端口至少被关闭一次，线程不产生额外控制决定。
@pytest.mark.parametrize("phase", ["before-start", "worker-exit"])
def test_checkpoint_lifetime_closes_owned_port(tmp_path, phase):
    coordinator = _ready_coordinator(tmp_path)
    closed = []
    coordinator.port.close = lambda: closed.append(True)
    coordinator._thread = threading.Thread(target=coordinator._run)
    if phase == "before-start":
        coordinator.stop(timeout_seconds=0.1)
    else:
        coordinator._stop.set()
        coordinator._run()
    assert closed
    assert not coordinator.decisions


# 功能：
#   插件注册失败应先于模型连接创建，不能在构造器中途退出时泄漏连接。
# 输入：
#   tmp_path：隔离运行目录。
#   monkeypatch：替换外部连接与插件加载，保留真实构造顺序。
# 输出：
#   None：注册失败后未创建模型端口。
def test_registry_failure_precedes_transport_allocation(tmp_path, monkeypatch):
    values = _artifacts()
    prepared = SimpleNamespace(runtime_checkpoints=values[8], contract=values[0])
    prepared.model_copy = lambda **kwargs: prepared
    created = []
    monkeypatch.setattr(checkpointing, "StructuredModelPort", lambda *a, **kw: created.append(True))

    # 功能：
    #   用可识别的错误模拟失效插件清单，避免实际导入第三方插件。
    # 输入：
    #   mission：构造器传来的准备任务。
    # 输出：
    #   None：此测试函数始终抛出指定错误。
    def unavailable_registry(mission):
        raise ValueError("invalid plugin snapshot")

    monkeypatch.setattr(checkpointing, "runtime_extension_registry", unavailable_registry)
    with pytest.raises(ValueError, match="invalid plugin snapshot"):
        checkpointing.CheckpointCoordinator(
            prepared=prepared,
            run_dir=tmp_path,
            provider="kimi",
            abort_file=tmp_path / "abort",
            model_timeout_seconds=1,
        )
    assert not created


# 功能：
#   检查点索引不能把布尔值隐式当成整数，以免错绑定第一个实际轨迹点。
# 输入：
#   无。
# 输出：
#   None：错误类型必须被拒绝。
def test_checkpoint_contract_rejects_boolean_index():
    values = _artifacts()
    values[8].checkpoints[0] = (
        values[8].checkpoints[0].model_copy(update={"track_point_index": True})
    )
    prepared = SimpleNamespace(runtime_checkpoints=values[8], contract=values[0])
    with pytest.raises(ValueError):
        checkpointing.checkpoint_contract_for(prepared)


# 功能：
#   请求文件中的整数不能被转换成安全门控真值后进入模型调用。
# 输入：
#   tmp_path：隔离的请求与决定目录。
# 输出：
#   None：错误请求触发拒绝，不进入模型调用。
def test_checkpoint_request_requires_actual_boolean_gates(tmp_path):
    import json

    coordinator = _ready_coordinator(tmp_path)
    path = tmp_path / "checkpoints" / "checkpoint-001.request.json"
    data = json.loads(path.read_text())
    data["deterministic_gates"]["no_collision"] = 1
    path.write_text(json.dumps(data))
    called = []

    # 功能：
    #   若错误请求进入调用就立即结束循环，避免回归失败导致测试无限等待。
    # 输入：
    #   kwargs：协调器尝试发送的模型请求。
    # 输出：
    #   response：不含可用决策的哨兵对象。
    def unexpected_call(**kwargs):
        called.append(True)
        coordinator._stop.set()
        response = object()
        return response

    coordinator.port.call = unexpected_call
    coordinator._run()
    assert not called
    assert coordinator.error is not None
    assert not coordinator.decisions
