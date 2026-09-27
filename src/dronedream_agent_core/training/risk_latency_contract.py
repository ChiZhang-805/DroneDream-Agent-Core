"""Decision-time nominal latency envelope, independent of future acceptance time."""

import math

from ..control_timing import LOCAL_CONTROL_MAXIMUM_AGE_SECONDS


# 功能：给出所有合法源年龄的名义等待位移余量，而非把最晚执行误当作必然最危险。
# 输入：current_speed、requested_speed、obstacle_speed：非负米每秒；integration_seconds：积分步长。
# 输出：contract：可核验的固定延迟区间与额外扫掠余量，不代表实测动力学保证。
def decision_latency_contract(current_speed, requested_speed, obstacle_speed, integration_seconds):
    values = (current_speed, requested_speed, obstacle_speed, integration_seconds)
    if (
        any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
        or not 0.005 <= integration_seconds <= 0.02
    ):
        raise ValueError("ACTION_RISK_LATENCY_BOUND_INPUT_INVALID")
    uncertainty = (current_speed + obstacle_speed) * LOCAL_CONTROL_MAXIMUM_AGE_SECONDS + 2 * max(
        current_speed, requested_speed
    ) * integration_seconds
    if not math.isfinite(uncertainty):
        raise ValueError("ACTION_RISK_LATENCY_BOUND_NONFINITE")
    return {
        "kind": "bounded-source-age-v1",
        "minimum_seconds": 0.0,
        "maximum_seconds": LOCAL_CONTROL_MAXIMUM_AGE_SECONDS,
        "additional_swept_uncertainty_m": uncertainty,
        "uses_future_actuation_latency_for_prediction": False,
        "nominal_dynamics_only": True,
    }


# 功能：按原教师输入重算新延迟契约，拒绝只改版本串、删除余量或把旧预测冒充新区间预测。
# 输入：prediction：已绑定摘要的教师回执；config：经过严格解析的教师配置。
# 输出：None；配置和实际预测不一致时失败，旧版本不得携带新版契约。
def validate_decision_latency_prediction(prediction, config):
    if config.decision_latency_policy == "measured-actuation-v1":
        if "decision_latency_contract" in prediction:
            raise ValueError("ACTION_RISK_LATENCY_POLICY_MIXED")
        return
    from ..contracts import DynamicObstacleObservation, Vector3

    state = prediction["state"]
    velocity = Vector3.model_validate(state["velocity"])
    obstacles = [DynamicObstacleObservation.model_validate(o) for o in state["dynamic_obstacles"]]
    requested = prediction["physical_request"]
    expected = decision_latency_contract(
        math.hypot(velocity.x, velocity.y, velocity.z),
        math.hypot(*requested[:3]),
        max(
            (math.hypot(o.velocity_mps.x, o.velocity_mps.y, o.velocity_mps.z) for o in obstacles),
            default=0.0,
        ),
        config.integration_seconds,
    )
    # JSON 身份比较区分 bool 与 0/1，不能用 Python 宽松相等接纳伪造字段。
    from ..hashing import sha256_json

    if (
        sha256_json(prediction.get("decision_latency_contract")) != sha256_json(expected)
        or type(prediction.get("effective_latency_seconds")) not in (int, float)
        or prediction["effective_latency_seconds"] != expected["maximum_seconds"]
    ):
        raise ValueError("ACTION_RISK_LATENCY_CONTRACT_MISMATCH")
