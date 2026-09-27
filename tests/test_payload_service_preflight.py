"""载荷服务必须在起飞前完成有界、只读的类型发现。"""

import subprocess

import pytest

from dronedream_agent_core import gazebo_adapter as adapter

SERVICE = "/world/school/model/parcel/place_detached"
PROVIDER = "  tcp://127.0.0.1:40000, gz.msgs.Pose, gz.msgs.Boolean\n"


# 功能：
#   模拟单调时钟和只读发现命令，使重试与超时测试不依赖真实进程。
# 输入：
#   monkeypatch：测试替换器；responses：逐次发现结果。
# 输出：
#   calls：已执行命令与各自超时上限。
def discovery_fixture(monkeypatch, responses):
    now = [0.0]
    calls = []
    pending = iter(responses)

    # 功能：
    #   验证服务发现只读且保留隔离环境，按序返回预设结果。
    # 输入：
    #   argv：命令；env：隔离环境；timeout：本次等待上限。
    # 输出：
    #   result：模拟进程结果。
    def run(argv, *, env, timeout):
        assert argv == ["gz", "service", "-i", "-s", SERVICE]
        assert env == {"GZ_PARTITION": "isolated"}
        assert 0 < timeout <= 3
        calls.append((argv, timeout))
        response = next(pending, (0, ""))
        if response is None:
            now[0] += timeout
            raise subprocess.TimeoutExpired(argv, timeout)
        result = subprocess.CompletedProcess(argv, response[0], response[1], "")
        return result

    monkeypatch.setattr(adapter, "_run", run)
    # 替换模块时钟对象，避免全局 time 模块污染其他测试组件。
    from types import SimpleNamespace

    monkeypatch.setattr(adapter, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    ))
    return calls


# 功能：
#   验证暂未注册后可恢复，并且确认过程不发布控制消息。
# 输入：
#   monkeypatch：测试替换器。
# 输出：
#   无。
def test_payload_service_recovers_before_flight(monkeypatch):
    calls = discovery_fixture(monkeypatch, [(0, ""), (1, PROVIDER), (0, PROVIDER)])
    evidence = adapter._require_payload_placement_service(
        "gz", service=SERVICE, env={"GZ_PARTITION": "isolated"}, timeout=1,
    )
    assert evidence["discovered"] is True
    assert evidence["attempts"] == len(calls) == 3


# 功能：
#   确认错误类型、重复提供者不会被误认成可用组件。
# 输入：
#   monkeypatch：测试替换器；provider：异常服务提供者描述。
# 输出：
#   无。
@pytest.mark.parametrize("provider", [PROVIDER.replace("Pose", "StringMsg"), PROVIDER * 2, PROVIDER.replace(", gz.msgs.Boolean", "")])
def test_payload_service_rejects_wrong_contract(monkeypatch, provider):
    discovery_fixture(monkeypatch, [(0, provider)])
    with pytest.raises(adapter.SimulationRuntimeError, match="incompatible"):
        adapter._require_payload_placement_service("gz", service=SERVICE, env={"GZ_PARTITION": "isolated"})


# 功能：
#   确认缺失、非零退出和进程超时均受整个发现阶段的期限约束。
# 输入：
#   monkeypatch：测试替换器；response：无法确认就绪的返回值。
# 输出：
#   无。
@pytest.mark.parametrize("response", [(0, ""), (1, PROVIDER), None])
def test_payload_service_has_bounded_wait(monkeypatch, response):
    calls = discovery_fixture(monkeypatch, [response] * 10)
    with pytest.raises(adapter.SimulationRuntimeError, match="not ready"):
        adapter._require_payload_placement_service("gz", service=SERVICE, env={"GZ_PARTITION": "isolated"}, timeout=0.5)
    assert 1 <= len(calls) <= 3


# 功能：
#   错误服务名必须在启动外部命令前被拒绝。
# 输入：
#   monkeypatch：测试替换器。
# 输出：
#   无。
def test_payload_service_rejects_invalid_endpoint(monkeypatch):
    calls = discovery_fixture(monkeypatch, [])
    with pytest.raises(adapter.SimulationRuntimeError, match="name is invalid"):
        adapter._require_payload_placement_service("gz", service="/world/test/set_pose", env={})
    assert calls == []
