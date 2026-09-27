"""Compile real runtime inputs for learning without manufacturing supervision."""

import io
from dataclasses import dataclass, field
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, decode_json

from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_expert_harness import requested_navigation_expert
from ..local_policy_port import compile_local_policy_features
from ..local_policy_training import LocalPolicyObservation
from ..plugin_files import read_plugin_file
from ..realtime_feature_encoders import RealtimeFeatureSnapshot, parse_realtime_control_input
from ..simulation_teacher import teacher_input_deadline


class TrainingObservationError(ValueError):
    """Reject stale or inconsistent actor inputs with a stable machine-readable reason."""

    # 功能：
    #   保留稳定的观测接纳错误标识，供训练诊断区分时效、身份和特征错误。
    # 输入：
    #   self：待初始化的异常实例。
    #   reason_code：不包含原始敏感载荷的错误标识。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


# 功能：
#   有界读取非空观测历史，逐行严格验证 JSON 和观测契约，不跳过空白样本或重复键。
# 输入：
#   path：最多 256 MiB、每行最多 4 MiB、最多二十五万条记录的历史文件。
# 输出：
#   rows：按原文件顺序验证后的观测列表。
def load_policy_observations(path: Path) -> list[LocalPolicyObservation]:
    content = read_plugin_file(path, limit=256 * 1024 * 1024)
    rows = []
    with io.BytesIO(content) as handle:
        while line := handle.readline(4 * 1024 * 1024 + 1):
            if not line.strip() or len(rows) >= 250_000:
                raise ValueError("TRAINING_OBSERVATION_HISTORY_RECORD_INVALID")
            decode_json(line, limit=4 * 1024 * 1024)
            # JSON 模式保留 dataclass 控制限额的合法解析，前一步独立拒绝重复键。
            rows.append(LocalPolicyObservation.model_validate_json(line))
    if not rows:
        raise ValueError("TRAINING_OBSERVATION_HISTORY_EMPTY")
    return rows


# 功能：
#   检查 Unix 毫秒时刻是限定范围内的整数，防止非有限比较绕过传感器过期判断。
# 输入：
#   value：当前时刻或源观测截止毫秒值。
# 输出：
#   None：不返回业务数据。
def _validate_clock(value) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise TrainingObservationError("TRAINING_INPUT_CLOCK_INVALID")


# 功能：
#   复用部署特征编译器生成学习器观测，核对快照摘要、连续控制模式和当前传感器时效。
# 输入：
#   snapshot：原始运行快照，不包含人为添加的真值标签。
#   now_unix_ms：接纳时的当前 Unix 毫秒值。
# 输出：
#   observation：绑定原快照和当前特征契约的无监督标签观测。
def compile_training_observation(snapshot: dict, *, now_unix_ms: int) -> LocalPolicyObservation:
    return compile_training_input(snapshot, now_unix_ms=now_unix_ms)[0]


# 功能：单次入口同时返回观测、独立验证的特征及原始期限，避免调用方重复解析同一特征。
# 输入：未信任的导航快照和当前UNIX毫秒时钟；输出：观测、特征、绝对截止毫秒。
# 返回对象仅属于本次调用，不缓存许可；再次使用仍须核对时效和内容身份。
def compile_training_input(snapshot: dict, *, now_unix_ms: int):
    _validate_clock(now_unix_ms)
    if not isinstance(snapshot, dict):
        raise TrainingObservationError("TRAINING_INPUT_SNAPSHOT_NOT_OBJECT")
    snapshot = copy_json(snapshot, limit=4 * 1024 * 1024)
    content = dict(snapshot)
    digest = content.pop("snapshot_sha256", None)
    if digest != sha256_json(content):
        raise TrainingObservationError("TRAINING_INPUT_SNAPSHOT_HASH_MISMATCH")
    context = snapshot.get("strategic_context")
    if not isinstance(context, dict) or not isinstance(context.get("task"), dict):
        raise TrainingObservationError("TRAINING_INPUT_TASK_CONTEXT_INVALID")
    task = context["task"]
    if (task.get("local_navigation_output_mode") != "normalized-body-velocity"
            or not task.get("control_session_id")
            or snapshot.get("authorized_candidate_paths")):
        raise TrainingObservationError("TRAINING_INPUT_MUST_BE_CONTINUOUS_SENSOR_CONTROL")
    features, fresh, deadline = parse_realtime_control_input(
        snapshot.get("realtime_feature_snapshot"), now_unix_ms=now_unix_ms
    )
    if not fresh or deadline <= now_unix_ms:
        raise TrainingObservationError("TRAINING_INPUT_SENSOR_EVIDENCE_EXPIRED")
    batch = compile_local_policy_features(snapshot, include_candidate_features=False)
    if (not batch.realtime_features_ready or batch.temporal_evidence is None
            or batch.pilot_control_limits is None
            or batch.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256):
        raise TrainingObservationError("TRAINING_INPUT_CURRENT_FEATURES_REQUIRED")
    observation = LocalPolicyObservation(
        temporal_evidence=batch.temporal_evidence,
        pilot_control_limits=batch.pilot_control_limits,
        control_feature_contract_sha256=batch.control_feature_contract_sha256,
        source_snapshot_sha256=digest,
        navigation_expert_role=requested_navigation_expert(snapshot),
        state_features=list(batch.state_features),
        candidate_features=[list(row) for row in batch.candidate_features],
        candidate_mask=list(batch.candidate_mask),
        realtime_features=list(batch.realtime_features),
        realtime_valid_mask=list(batch.realtime_valid_mask),
    )
    return observation, features, deadline


@dataclass(frozen=True)
class PreparedTrainingInput:
    """One verified packet, reusable only with unchanged contents and source time.

    Compilation is independent of the admission clock. The freshness check is
    not: it must run again immediately before giving the observation to an actor.
    This object has a single-request lifetime, not a cache of old permissions.
    """

    request_sha256: str
    sample: LocalPolicyObservation
    features: RealtimeFeatureSnapshot
    _sample_sha256: str = field(init=False, repr=False)
    _features_sha256: str = field(init=False, repr=False)

    # 功能：
    #   验证并独立持有样本和融合快照，保存内容指纹以检测冻结对象内部列表的后续修改。
    # 输入：
    #   self：包含请求身份、样本和传感器特征的准备对象。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self):
        if not isinstance(self.sample, LocalPolicyObservation) or not isinstance(
            self.features, RealtimeFeatureSnapshot
        ):
            raise ValueError("TRAINING_PREPARED_INPUT_TYPES_INVALID")
        sample = LocalPolicyObservation.model_validate_json(self.sample.model_dump_json())
        features = RealtimeFeatureSnapshot.model_validate(self.features.model_dump(mode="json"))
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "_sample_sha256", sha256_json(sample))
        object.__setattr__(self, "_features_sha256", sha256_json(features))

    # 功能：
    #   在源观测时刻编译一次快照，要求请求附带的特征与实际运行编译结果完全一致。
    # 输入：
    #   cls：准备输入类。
    #   request：完整、可序列化的源请求。
    # 输出：
    #   prepared：持有独立观测、特征和请求指纹的单请求准备对象。
    @classmethod
    def from_request(cls, request: dict):
        request = copy_json(request, limit=4 * 1024 * 1024)
        if not isinstance(request, dict) or not isinstance(request.get("snapshot"), dict):
            raise ValueError("TRAINING_PREPARED_REQUEST_INVALID")
        snapshot = request["snapshot"]
        compiled, features, _ = compile_training_input(
            snapshot, now_unix_ms=snapshot.get("control_reference_observed_at_unix_ms")
        )
        supplied = LocalPolicyObservation.model_validate(request.get("observation"))
        if supplied != compiled:
            raise ValueError("PX4_TRAINING_FEATURES_DIFFER_FROM_RUNTIME_SNAPSHOT")
        # 复用同一次验证返回的特征；构造器仍独立复制并记录指纹，不缓存旧时效许可。
        prepared = cls(sha256_json(request), compiled, features)
        return prepared

    # 功能：
    #   使用前重查原始请求、准备内容指纹和当前时效，仅在全部保持一致时返回独立样本。
    # 输入：
    #   self：先前为同一请求创建的准备对象。
    #   request：此刻实际准备交给学习器的请求。
    #   now_unix_ms：真正使用观测时的 Unix 毫秒值。
    # 输出：
    #   sample：不共享准备对象内部容器的有效观测副本。
    def admit(self, request: dict, *, now_unix_ms: int) -> LocalPolicyObservation:
        _validate_clock(now_unix_ms)
        request = copy_json(request, limit=4 * 1024 * 1024)
        if not isinstance(request, dict):
            raise ValueError("TRAINING_PREPARED_REQUEST_INVALID")
        if sha256_json(request) != self.request_sha256:
            raise ValueError("PX4_TRAINING_PREPARED_INPUT_CHANGED")
        if (sha256_json(self.sample) != self._sample_sha256
                or sha256_json(self.features) != self._features_sha256):
            raise ValueError("PX4_TRAINING_PREPARED_STATE_CHANGED")
        _validate_clock(request.get("valid_until_unix_ms"))
        if (request["valid_until_unix_ms"] <= now_unix_ms
                or teacher_input_deadline(self.features, now_ms=now_unix_ms) <= now_unix_ms):
            raise TrainingObservationError("TRAINING_INPUT_SENSOR_EVIDENCE_EXPIRED")
        # Neither visual attachment nor a caller can mutate retained history.
        sample = self.sample.model_copy(deep=True)
        return sample
