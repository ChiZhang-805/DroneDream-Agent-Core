"""Build-only probes of actual ONNX computation; not samples or qualification evidence."""

from .causal_control import CONTROL_HISTORY_WIDTH
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .local_expert_harness import NAVIGATION_EXPERT_ROLES
from .local_policy_packages import (
    LOCAL_POLICY_MANEUVER_FEATURE_COUNT,
    LOCAL_POLICY_MAXIMUM_CANDIDATES,
    LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
    LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,
    LOCAL_POLICY_SENSOR_FEATURE_COUNT,
    LOCAL_POLICY_STATE_FEATURE_COUNT,
    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
    LocalPolicyPackageManifest,
)
from .local_policy_tensors import bounded_scalar, float_output
from .pilot_control_mapping import ACTION_RISK_FEATURE_COUNT


# 功能：
#   核对推理入口接收当前单批次 float32 形状，只允许批次维动态，拒绝按模型声明分配任意数组。
# 输入：
#   role：正在验证的专家名称。
#   node：ONNX Runtime 返回的输入节点描述。
#   shape：运行代码实际会提供的有界形状。
# 输出：
#   None：不返回业务数据。
def _check_input(role, node, shape):
    observed = node.shape
    if (
        node.type != "tensor(float)"
        or not isinstance(observed, list)
        or len(observed) != len(shape)
        or any(actual != required for actual, required in zip(observed[1:], shape[1:], strict=True))
        or not (observed[0] == 1 or observed[0] is None or isinstance(observed[0], str))
    ):
        raise RuntimeError(f"LOCAL_POLICY_PROBE_INPUT_MISMATCH:{role}:{node.name}")


# 功能：
#   用一次实际计算验证专家所需输出，复用实时推理的数值校验，不将合成输入计入训练或验收。
# 输入：
#   role：专家名称。
#   session：已按内容摘要加载的 ONNX 会话。
#   shapes：由当前运行代码定义的输入形状表。
#   outputs：任务实际使用的输出名称及元素数量。
# 输出：
#   result：该专家执行过的输入与输出名称记录。
def _probe_session(role, session, shapes, outputs):
    import numpy as np

    feeds = {}
    for node in session.get_inputs():
        if node.name not in shapes:
            raise RuntimeError(f"LOCAL_POLICY_PROBE_INPUT_UNKNOWN:{role}:{node.name}")
        shape = shapes[node.name]
        _check_input(role, node, shape)
        # 掩码置一仅为触发完整计算路径；没有传感器时间戳，不能进入真实控制历史。
        feeds[node.name] = np.full(
            shape, 1.0 if node.name.endswith("mask") else 0.0, dtype=np.float32,
        )
    values = session.run(list(outputs), feeds)
    if not isinstance(values, list) or len(values) != len(outputs):
        raise RuntimeError(f"LOCAL_POLICY_PROBE_OUTPUT_COUNT_INVALID:{role}")
    for (name, count), value in zip(outputs.items(), values, strict=True):
        label = f"PROBE:{role}:{name}"
        if name in {"risk_score", "anomaly_score", "controller_step_scale"}:
            bounded_scalar(value, label, minimum=0.1 if name == "controller_step_scale" else 0.0)
        else:
            vector = float_output(value, count, label)
            if name == "pilot_control" and np.any(np.abs(vector) > 1.0):
                raise RuntimeError(f"LOCAL_POLICY_PROBE_PILOT_CONTROL_OUT_OF_RANGE:{role}")
    result = {"inputs": sorted(feeds), "outputs": list(outputs), "computed": True}
    return result


# 功能：
#   1. 按当前控制契约依次检查全部十专家，使用固定上限的单批次输入，不做模型训练或飞行。
#   2. 不从旧清单猜测缺失形状，不把接口可计算性当成模型能力或发布许可。
# 输入：
#   manifest：已经来源绑定的完整模型包清单。
#   sessions：角色到当前已加载会话的映射。
# 输出：
#   report：接口检查记录，资格字段固定为 False。
def verify_ensemble_io(manifest: LocalPolicyPackageManifest, sessions: dict) -> dict:
    manifest = LocalPolicyPackageManifest.model_validate(manifest.model_dump())
    if (
        manifest.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
        or manifest.navigation_architecture != "causal-gru-control"
        or manifest.pilot_control_mode != "normalized-body-velocity"
    ):
        raise RuntimeError("LOCAL_POLICY_PROBE_CURRENT_CAUSAL_CONTROL_REQUIRED")
    outputs = {
        role: {
            "candidate_scores": LOCAL_POLICY_MAXIMUM_CANDIDATES,
            "action_scores": 4, "risk_score": 1,
            "pilot_control": LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,
        }
        for role in NAVIGATION_EXPERT_ROLES
    }
    outputs.update({
        "perception-encoder": {"visual_features": manifest.visual_feature_count},
        "risk-critic": {"risk_score": 1},
        "perception-health-critic": {"risk_score": 1},
        "settle-stability-critic": {"risk_score": 1},
        "payload-dynamics-adapter": {"risk_score": 1, "controller_step_scale": 1},
        "state-anomaly-detector": {"anomaly_score": 1},
        "cross-modal-consistency-critic": {"risk_score": 1},
    })
    if set(sessions) != set(outputs) or any(session is None for session in sessions.values()):
        raise RuntimeError("LOCAL_POLICY_PROBE_COMPLETE_ENSEMBLE_REQUIRED")
    if any(value is None for value in (
        manifest.visual_width, manifest.visual_height, manifest.visual_feature_count,
        manifest.realtime_feature_count, manifest.navigation_history_length,
    )):
        raise RuntimeError("LOCAL_POLICY_PROBE_INPUT_DIMENSIONS_MISSING")
    shapes = {
        "state_features": (1, LOCAL_POLICY_STATE_FEATURE_COUNT),
        "realtime_features": (1, manifest.realtime_feature_count),
        "realtime_valid_mask": (1, manifest.realtime_feature_count),
        "visual_features": (1, manifest.visual_feature_count),
        "forward_rgb": (1, 3, manifest.visual_height, manifest.visual_width),
        "control_history": (1, manifest.navigation_history_length, CONTROL_HISTORY_WIDTH),
        "control_history_mask": (1, manifest.navigation_history_length),
        "proposed_control": (1, ACTION_RISK_FEATURE_COUNT),
        "state_history": (
            1, LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH, LOCAL_POLICY_STATE_FEATURE_COUNT,
        ),
        "payload_history": (
            1, LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH, LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
        ),
        "history_mask": (1, LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH),
        "payload_features": (1, LOCAL_POLICY_PAYLOAD_FEATURE_COUNT),
        "maneuver_features": (1, LOCAL_POLICY_MANEUVER_FEATURE_COUNT),
        "sensor_features": (1, LOCAL_POLICY_SENSOR_FEATURE_COUNT),
    }
    checked = {
        role: _probe_session(role, sessions[role],
            {**shapes, 'heading_context': (1, 23)}
            if manifest.heading_context_for_role(role) is not None
            else shapes, names)
        for role, names in outputs.items()
    }
    report = {"experts": checked, "expert_count": len(checked), "qualification_granted": False}
    return report
