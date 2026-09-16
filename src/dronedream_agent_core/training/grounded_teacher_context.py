"""Label-only native pose/timing binding shared by step and stream imitation."""

from dataclasses import replace

from dronedream_plugin_sdk.protocol import copy_json

from ..contracts import QuaternionWxyz, RuntimeLocalSafetyObservation, Vector3
from ..control_execution_evidence import ControlApplicationRecord
from ..hashing import sha256_json
from ..realtime_feature_encoders import RealtimeFeatureSnapshot
from .counterfactual_teacher import CounterfactualConfig, CounterfactualState
from .flight_environment import FlightObservation


# 功能：
#   复制教师上下文中可变的坐标、姿态和动态障碍，不让调用方污染缓存的独立真值。
# 输入：
#   state：只供离线监督的已验证上下文。
# 输出：
#   owned：独立持有所有嵌套模型的上下文。
def copy_grounded_state(state: CounterfactualState) -> CounterfactualState:
    owned = replace(state, position=state.position.model_copy(deep=True),
                    velocity=state.velocity.model_copy(deep=True),
                    orientation=state.orientation.model_copy(deep=True),
                    goal=state.goal.model_copy(deep=True),
                    dynamic_obstacles=tuple(row.model_copy(deep=True)
                                            for row in state.dynamic_obstacles))
    return owned


# 功能：
#   1. 绑定原始观测与快照，将及时的独立仿真见证外推到源观测时刻并保留不确定度。
#   2. 构造只供离线监督的上下文及摘要回执，不将特权真值合并进学生输入。
# 输入：
#   observation：学生当时获得的原始观测。
#   snapshot：与观测摘要一致的原输入快照。
#   witness：独立仿真真值，不是模型预测。
#   application：实际接受控制的回执。
#   teacher：提供名义加速度边界的离线教师。
#   binding：上层完成停止和来源验证后传入的严格 JSON 绑定信息。
# 输出：
#   state：与来源时刻对齐的独立教师上下文。
#   context：包含原始见证、时间偏移和绑定信息的独立回执。
def grounded_teacher_context(observation, snapshot, witness, application, teacher, *, binding):
    if (not isinstance(observation, FlightObservation)
            or not isinstance(witness, RuntimeLocalSafetyObservation)
            or not isinstance(application, ControlApplicationRecord)
            or type(snapshot) is not dict or type(binding) is not dict):
        raise ValueError("DAGGER_NATIVE_CONTEXT_INPUT_INVALID")
    observation = FlightObservation.model_validate(observation.model_dump())
    witness = RuntimeLocalSafetyObservation.model_validate(witness.model_dump())
    application = ControlApplicationRecord.model_validate(application.model_dump())
    snapshot, binding = copy_json(snapshot), copy_json(binding)
    config = CounterfactualConfig.model_validate(teacher.config.model_dump())
    content = dict(snapshot)
    digest = content.pop("snapshot_sha256", None)
    if digest != sha256_json(content) or digest != observation.sample.source_snapshot_sha256:
        raise ValueError("DAGGER_NATIVE_CONTEXT_SNAPSHOT_MISMATCH")
    source_ms = observation.sample.temporal_evidence.observed_at_unix_ms
    reference = snapshot.get("control_reference_observed_at_unix_ms")
    if type(reference) is not int or reference != source_ms:
        raise ValueError("DAGGER_NATIVE_CONTEXT_SOURCE_TIME_MISMATCH")
    if (
        not 0 <= source_ms - witness.observed_at_unix_ms <= 100
        or witness.source != "simulation-ground-truth"
        or not witness.stream_healthy
        or witness.stream_age_seconds > 0.1
    ):
        raise ValueError("DAGGER_NATIVE_INITIAL_WITNESS_INVALID")
    latency_ms = application.accepted_at_unix_ms - source_ms
    if not 0 <= latency_ms <= 250:
        # An expired safety hold is legitimate containment, but it is not a
        # timely continuous-action latency example for counterfactual training.
        raise ValueError("DAGGER_NATIVE_ACTION_LATENCY_OUTSIDE_MODEL")
    features = RealtimeFeatureSnapshot.model_validate(snapshot["realtime_feature_snapshot"])
    flight = next((row for row in features.encodings
                   if row.encoder_role == "flight-state-encoder"), None)
    if flight is None or flight.valid_mask[:4] != [1.0] * 4:
        raise ValueError("DAGGER_NATIVE_ORIENTATION_MISSING")
    # Advance a recent witness to the observation time under constant velocity;
    # retain an acceleration-derived uncertainty rather than claiming exactness.
    dt = (source_ms - witness.observed_at_unix_ms) / 1000
    p, v = witness.current_position_m, witness.current_velocity_mps
    context = {
        "witness": witness.model_dump(mode="json"),
        "source_observation_sha256": sha256_json(observation),
        "source_ms": source_ms,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "pose_prediction_seconds": dt,
        "application_sha256": sha256_json(application),
        "observed_action_latency_seconds": latency_ms / 1000,
        "binding": binding,
        "scope": "privileged-offline-supervision-only; never actor input",
    }
    state = CounterfactualState(
        position=Vector3(x=p.x + v.x * dt, y=p.y + v.y * dt, z=p.z + v.z * dt),
        velocity=v,
        orientation=QuaternionWxyz(**dict(zip("wxyz", flight.features[:4], strict=True))),
        goal=Vector3.model_validate(snapshot["goal_position_m"]),
        dynamic_obstacles=tuple(
            # 教师预测按 age+t 推进障碍；这里只累加原始年龄，不能再次预推进位置。
            row.model_copy(update={"age_seconds": row.age_seconds + dt})
            for row in witness.dynamic_obstacles
        ),
        context_sha256=sha256_json(context),
        additional_position_uncertainty_m=0.5 * config.acceleration_mps2 * dt * dt,
        observed_action_latency_seconds=latency_ms / 1000,
    )
    return state, context
