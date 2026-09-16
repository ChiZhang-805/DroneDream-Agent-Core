"""Declarative plugins must not lend mutable bindings or pretend a receipt is actuation."""

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_plugins.harness_policy_plugins import _constant, _event_bus, _observer
from dronedream_agent_plugins.runtime_action_adapter_plugins import _adapter, _declare


# 功能：
#   验证注册后修改原配置或一次调用结果，不会改写后续调用的策略默认值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_policy_defaults_are_frozen_and_each_invocation_is_detached():
    source = {"options": {"enabled": True}}
    hook = _constant(source)
    source["options"]["enabled"] = False
    first = hook()
    assert first["options"]["enabled"] is True
    first["options"]["enabled"] = False
    assert hook()["options"]["enabled"] is True


# 功能：
#   验证适配器默认参数和声明结果均不共享可变数据，避免目标车辆主题被后续修改。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_action_adapter_bindings_do_not_borrow_defaults_or_returned_declarations():
    defaults = {"topic": "/selected-vehicle/payload", "nested": {"required": True}}
    adapter = _adapter(
        adapter_id="runtime.test",
        executors=["native.payload.pickup"],
        driver="gazebo-payload",
        defaults=defaults,
        schema={"type": "object"},
        authority="actuate",
    )
    defaults["nested"]["required"] = False
    assert adapter["default_parameters"]["nested"]["required"] is True
    hook = _declare([adapter])
    adapter["default_parameters"]["topic"] = "/wrong-vehicle"
    returned = hook()
    assert returned["adapters"][0]["default_parameters"]["topic"] == "/selected-vehicle/payload"
    returned["adapters"].clear()
    assert len(hook()["adapters"]) == 1


# 功能：
#   验证事件封装与观察回执绑定相同原始内容，且不会冒称已执行或改写输入。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_event_envelope_and_observer_bind_the_same_finite_payload_without_mutating_it():
    payload = {"nested": {"value": 1}}
    envelope = _event_bus(event="prepared", payload=payload)
    receipt = _observer(event="prepared", payload=payload)
    assert envelope["payload_sha256"] == receipt["payload_sha256"] == sha256_json(payload)
    envelope["payload"]["nested"]["value"] = 9
    assert payload["nested"]["value"] == 1
    assert "executed" not in receipt


# 功能：
#   验证事件与观察钩子在摘要生成前拒绝非有限载荷，不产生有效封装或回执。
# 输入：
#   hook：待测试的事件封装器或观察器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("hook", [_event_bus, _observer])
def test_event_hooks_reject_nonfinite_payload_before_hashing(hook):
    with pytest.raises(ValueError):
        hook(event="prepared", payload={"value": float("nan")})
