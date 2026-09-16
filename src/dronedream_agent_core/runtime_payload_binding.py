"""Resolve payload transport from frozen mission actions, never model-generated addresses."""

from __future__ import annotations

from dronedream_plugin_sdk.protocol import copy_json

from .contracts import PreparedMission
from .hashing import sha256_json


# 功能：
#   1. 从选定任务已冻结的载荷动作中解析唯一接口，不接受模型临时编造或替换端点。
#   2. 相同绑定按内容去重；缺失、冲突或参数越权直接拒绝，不回退到演示无人机。
#   3. 解析不授予执行权限；悬停、消息身份、核心授权与设备读回由调用链独立验证。
# 输入：
#   prepared：当前冻结任务，缺失时不得解析设备接口。
#   parameters：所需 attach/detach 操作及可选的已冻结参数引用。
# 输出：
#   binding：唯一且解除可变共享引用的资产派生载荷绑定。
def resolve_runtime_payload(
    prepared: PreparedMission | None,
    parameters: dict[str, object],
) -> dict[str, object]:
    operation = parameters.get("operation")
    if not isinstance(operation, str) or operation not in {"attach", "detach"}:
        raise ValueError("PAYLOAD_OPERATION_REQUIRED")
    actions = getattr(prepared, "runtime_actions", None)
    if actions is None or actions.contract_id != prepared.contract.contract_id:
        raise ValueError("PAYLOAD_FROZEN_BINDING_REQUIRED")
    candidates: dict[str, dict[str, object]] = {}
    for step in actions.steps:
        if (
            step.driver != "gazebo-payload"
            or step.authority != "actuate"
            or step.parameters.get("operation") != operation
        ):
            continue
        binding = copy_json(step.parameters)
        # 不依赖字典插入顺序；同一资产端点跨任务步骤的重复引用按规范化内容识别。
        candidates[sha256_json(binding)] = binding
    if len(candidates) != 1:
        raise ValueError("PAYLOAD_FROZEN_BINDING_REQUIRED")
    binding = next(iter(candidates.values()))
    if any(key not in binding or value != binding[key] for key, value in parameters.items()):
        raise ValueError("PAYLOAD_FROZEN_PARAMETER_OVERRIDE")
    return binding
