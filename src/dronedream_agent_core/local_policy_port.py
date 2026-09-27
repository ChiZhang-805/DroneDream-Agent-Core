"""Millisecond-scale local policy adapter for the structured navigation boundary."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from dronedream_plugin_sdk.protocol import copy_json

from .causal_control import (
    CONTROL_HISTORY_CONTRACT_SHA256,
    CONTROL_HISTORY_WIDTH,
    CausalControlHistory,
)
from .contracts import (
    LocalExpertInferenceTrace,
    ModelCallRecord,
    NormalizedPilotControl,
    QuaternionWxyz,
    StrictModel,
    TextNavigationDecision,
    Vector3,
)
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .hashing import sha256_json
from .latest_inference_worker import LatestInferenceWorker
from .local_expert_harness import (
    NAVIGATION_EXPERT_ROLES,
    AdvisoryExpertRole,
    LocalExpertRoutingDecision,
    NavigationExpertRole,
    route_local_experts,
)
from .local_policy_packages import (
    LOCAL_POLICY_CANDIDATE_FEATURE_COUNT,
    LOCAL_POLICY_MANEUVER_FEATURE_COUNT,
    LOCAL_POLICY_MAXIMUM_CANDIDATES,
    LOCAL_POLICY_PAYLOAD_FEATURE_COUNT,
    LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,
    LOCAL_POLICY_SENSOR_FEATURE_COUNT,
    LOCAL_POLICY_STATE_FEATURE_COUNT,
    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
    LoadedLocalPolicyPackage,
)
from .local_policy_tensors import bounded_scalar as _bounded_scalar
from .local_policy_tensors import float_output as _float_output
from .model_harness.model_port import (
    ModelInvocationError,
    ProviderSettings,
    StructuredCallResult,
)
from .pilot_control_mapping import (
    ACTION_RISK_FEATURE_COUNT,
    PilotControlLimits,
    action_risk_features,
)
from .plugin_files import read_plugin_file
from .policy_observation_buffer import PolicyObservationBuffer
from .precision_heading_input import (
    PRECISION_HEADING_ARCHITECTURE,
    current_precision_heading_input,
    validate_current_precision_heading_input,
)
from .realtime_feature_encoders import (
    POLICY_CONTROL_REFERENCE_FEATURE_COUNT,
    POLICY_REALTIME_FEATURE_COUNT,
    REALTIME_CONTROL_FEATURE_COUNT,
    RealtimeFeatureSnapshot,
    world_enu_to_body,
)
from .temporal_evidence import ObservationHistory, TemporalEvidence
from .visual_encoding_input import VisualEncodingInput, freeze_visual_input

_SECTOR_LABELS = (
    "front",
    "front-left",
    "left",
    "back-left",
    "back",
    "back-right",
    "right",
    "front-right",
)
_CANDIDATE_KINDS = (
    "qualified-route-local-lookahead",
    "qualified-route-lookahead",
    "goal-direct",
    "goal",
    "goal-dynamic-detour",
    "observed-frontier",
)

_ACCELERATED_EXECUTION_PROVIDER_ORDER = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
)


# 功能：
#   按明确配置选择可用执行后端；自动选择一个加速器及可用 CPU 回退，TensorRT 可再接 CUDA。
# 输入：
#   available_providers：当前 ONNX Runtime 实际支持的后端名称集合。
#   requested_providers：可选的有序用户配置，必须全部可用且不重复。
# 输出：
#   selected：本次会话要使用的有序后端列表。
def select_onnx_execution_providers(
    available_providers: list[str] | tuple[str, ...] | set[str],
    requested_providers: list[str] | None = None,
) -> list[str]:
    if not isinstance(available_providers, (list, tuple, set)) or any(
        type(item) is not str or not item for item in available_providers
    ):
        raise RuntimeError("ONNX_RUNTIME_EXECUTION_PROVIDER_LIST_INVALID")
    available = set(available_providers)
    if requested_providers is not None:
        if (
            not isinstance(requested_providers, list)
            or not requested_providers
            or any(type(item) is not str or not item for item in requested_providers)
            or len(set(requested_providers)) != len(requested_providers)
        ):
            raise RuntimeError("ONNX_RUNTIME_EXECUTION_PROVIDER_REQUEST_INVALID")
        if any(provider not in available for provider in requested_providers):
            raise RuntimeError("ONNX_RUNTIME_EXECUTION_PROVIDER_UNAVAILABLE")
        selected = list(requested_providers)
        return selected

    selected: list[str] = []
    accelerator = next(
        (provider for provider in _ACCELERATED_EXECUTION_PROVIDER_ORDER if provider in available),
        None,
    )
    if accelerator is not None:
        selected.append(accelerator)
        if accelerator == "TensorrtExecutionProvider" and "CUDAExecutionProvider" in available:
            selected.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        selected.append("CPUExecutionProvider")
    if not selected:
        raise RuntimeError("ONNX_RUNTIME_HAS_NO_EXECUTION_PROVIDER")
    return selected


class LocalPolicyRawInference(StrictModel):
    """Numerical output contract shared by test and ONNX inference backends."""

    candidate_scores: list[float] = Field(
        min_length=LOCAL_POLICY_MAXIMUM_CANDIDATES,
        max_length=LOCAL_POLICY_MAXIMUM_CANDIDATES,
    )
    action_scores: list[float] = Field(min_length=3, max_length=4)
    risk_score: float = Field(ge=0.0, le=1.0)
    navigation_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    advisory_risk_scores: dict[AdvisoryExpertRole, float] = Field(
        default_factory=dict,
        max_length=6,
    )
    invoked_advisory_roles: list[AdvisoryExpertRole] = Field(
        default_factory=list,
        max_length=6,
    )
    expert_latency_ms: dict[str, float] = Field(default_factory=dict, max_length=8)
    pipeline_latency_ms: dict[str, float] = Field(default_factory=dict, max_length=8)
    controller_step_scale: float = Field(default=1.0, ge=0.1, le=1.0)
    pilot_control: NormalizedPilotControl | None = None

    # 功能：
    #   检查分数及耗时有效、每个顾问有且只有一项风险记录，聚合不能压低任一风险分量。
    # 输入：
    #   self：后端原始分数、顾问调用记录与时间统计。
    # 输出：
    #   self：通过数值和顾问记账一致性校验的结果。
    @model_validator(mode="after")
    def validate_finite_scores(self) -> LocalPolicyRawInference:
        if any(not math.isfinite(value) for value in self.candidate_scores):
            raise ValueError("local policy candidate scores must be finite")
        if any(not math.isfinite(value) for value in self.action_scores):
            raise ValueError("local policy action scores must be finite")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in [
                *self.expert_latency_ms.values(),
                *self.pipeline_latency_ms.values(),
            ]
        ):
            raise ValueError("local policy latencies must be finite and non-negative")
        if len(self.invoked_advisory_roles) != len(set(self.invoked_advisory_roles)):
            raise ValueError("LOCAL_POLICY_ADVISORY_EXECUTION_MISMATCH")
        if set(self.invoked_advisory_roles) != set(self.advisory_risk_scores):
            raise ValueError("each invoked advisor requires exactly one risk score")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.advisory_risk_scores.values()
        ):
            raise ValueError("each advisor risk score must be a finite probability")
        if self.risk_score < max(
            self.navigation_risk_score or 0.0,
            *self.advisory_risk_scores.values(),
            0.0,
        ):
            raise ValueError("aggregate risk cannot hide a navigation or advisor veto")
        return self


@dataclass(frozen=True)
class LocalPolicyFeatureBatch:
    """Owned fixed-layout tensors plus their source/contract and expert-routing identity.

    Zero defaults are masked/unready placeholders, not synthesized observations.
    A batch is model input only; dispatch obtains a separate live safety lease.
    """

    state_features: tuple[float, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    candidate_mask: tuple[float, ...]
    candidate_ids: tuple[str, ...]
    temporal_evidence: TemporalEvidence | None = None
    maneuver_features: tuple[float, ...] = (0.0,) * LOCAL_POLICY_MANEUVER_FEATURE_COUNT
    payload_features: tuple[float, ...] = (0.0,) * LOCAL_POLICY_PAYLOAD_FEATURE_COUNT
    sensor_features: tuple[float, ...] = (0.0,) * LOCAL_POLICY_SENSOR_FEATURE_COUNT
    realtime_features: tuple[float, ...] = (0.0,) * POLICY_REALTIME_FEATURE_COUNT
    realtime_valid_mask: tuple[float, ...] = (0.0,) * POLICY_REALTIME_FEATURE_COUNT
    realtime_features_ready: bool = False
    realtime_snapshot_sha256: str | None = None
    control_feature_contract_sha256: str | None = None
    precomputed_visual_features: tuple[float, ...] | None = None
    navigation_expert_role: NavigationExpertRole = "local-navigation-policy"
    expert_routing: LocalExpertRoutingDecision | None = None
    pilot_control_limits: PilotControlLimits | None = None
    precision_heading_context: tuple[float, ...] | None = None


@dataclass(frozen=True)
class _VisualEncodingResult:
    source_sha256: str
    features: tuple[float, ...]
    preprocess_latency_ms: float
    encoder_latency_ms: float


# 功能：
#   在支持线程亲和性的系统绑定当前视觉线程，不改变传感器或控制线程的 CPU 分配。
# 输入：
#   cpu_ids：初始化阶段已验证、去重的 CPU 序号。
# 输出：
#   None：不返回业务数据。
def _set_current_thread_cpu_affinity(cpu_ids: tuple[int, ...]) -> None:
    setter = getattr(os, "sched_setaffinity", None)
    if callable(setter) and cpu_ids:
        setter(0, set(cpu_ids))


class LocalPolicyInferenceBackend(Protocol):
    """Numerical expert execution seam; the port validates every implementation's output."""

    # 功能：
    #   声明本地专家推理接口；实现只产生数值结果，不直接取得执行器控制权。
    # 输入：
    #   self：后端实现对象。
    #   batch：固定布局的模型特征与路由身份。
    #   multimodal：本次明确绑定的视觉来源。
    # 输出：
    #   result：等待端口重验及仲裁的原始推理结果。
    def infer(
        self,
        batch: LocalPolicyFeatureBatch,
        *,
        multimodal: list[dict[str, object]],
    ) -> LocalPolicyRawInference: ...


# 功能：
#   读取可选描述数值；无效值回到调用方明确的缺省值，可用性仍由独立掩码表达。
# 输入：
#   value：未校验的描述字段，不接受布尔值或数字字符串。
#   default：字段缺失、溢出或非有限时使用的明确缺省值。
# 输出：
#   number：有效数值或给定缺省值，不代表实际测量成功。
def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    try:
        number = float(value)
    except OverflowError:
        return default
    return number if math.isfinite(number) else default


# 功能：
#   为可选描述对象取得一层独立映射，缺失时使用空对象，不推断传感器内容。
# 输入：
#   value：候选映射值。
# 输出：
#   mapping：浅层复制的映射或空字典。
def _mapping(value: object) -> dict[str, object]:
    mapping = dict(value) if isinstance(value, dict) else {}
    return mapping


# 功能：
#   按具名字段读取 ENU 三轴描述，避免依赖 JSON 字段插入顺序。
# 输入：
#   value：含可选 x、y、z 数值的对象。
# 输出：
#   vector：三轴描述元组，无效分量为零但不获得测量有效标记。
def _vector(value: object) -> tuple[float, float, float]:
    item = _mapping(value)
    vector = (_number(item.get("x")), _number(item.get("y")), _number(item.get("z")))
    return vector


# 功能：
#   将已验证有限的归一化数值限制在明确区间，不用于替代原传感器质量校验。
# 输入：
#   value：需要限幅的有限数值。
#   minimum：闭区间下界。
#   maximum：闭区间上界。
# 输出：
#   bounded：区间内的限幅结果。
def _clip(value: float, minimum: float, maximum: float) -> float:
    bounded = min(maximum, max(minimum, value))
    return bounded


_SENSOR_MODALITY_ORDER = (
    "rgb-camera",
    "depth-camera",
    "lidar",
    "rangefinder",
    "imu",
    "magnetometer",
    "barometer",
    "gnss",
    "odometry",
    "optical-flow",
    "state-estimate",
    "powertrain",
)


# 功能：
#   按固定模态顺序编码传感器存在性、健康、时间和覆盖；缺失样本不能算作健康。
# 输入：
#   snapshot：已绑定的导航快照，包含可选多模态健康汇总。
# 输出：
#   result：固定宽度的健康特征元组，不包含伪造的物理测量。
def compile_multimodal_sensor_features(snapshot: dict[str, object]) -> tuple[float, ...]:

    multimodal = _mapping(snapshot.get("multimodal_sensor_snapshot"))
    statuses = [_mapping(item) for item in multimodal.get("statuses", []) if isinstance(item, dict)]
    by_modality: dict[str, list[dict[str, object]]] = {}
    for status in statuses:
        modality = str(status.get("modality", ""))
        # Preserve the qualified fixed-width policy tensor while admitting
        # calibrated radar contracts. Radar and LiDAR both populate the
        # geometric-ranging health channel; their metric content is fused into
        # tracked obstacles before candidate scoring, not silently discarded.
        feature_modality = "lidar" if modality == "radar" else modality
        by_modality.setdefault(feature_modality, []).append(status)
    features: list[float] = []
    required_count = 0
    required_healthy_count = 0
    for modality in _SENSOR_MODALITY_ORDER:
        items = by_modality.get(modality, [])
        present_items = [item for item in items if _number(item.get("latest_sequence")) > 0]
        healthy_items = [item for item in present_items if item.get("health") == "healthy"]
        required_items = [item for item in items if item.get("required_for_motion") is True]
        required_count += len(required_items)
        required_healthy_count += sum(
            item.get("health") == "healthy" and _number(item.get("latest_sequence")) > 0
            for item in required_items
        )
        age = max(
            (_number(item.get("sample_age_seconds"), 2.0) for item in present_items),
            default=2.0,
        )
        quality = min(
            (_number(item.get("quality")) for item in present_items),
            default=0.0,
        )
        coverage = min(
            (_number(item.get("coverage")) for item in present_items),
            default=0.0,
        )
        features.extend(
            (
                1.0 if present_items else 0.0,
                1.0 if present_items and len(healthy_items) == len(present_items) else 0.0,
                _clip(age / 2.0, 0.0, 4.0),
                _clip(quality, 0.0, 1.0),
                _clip(coverage, 0.0, 1.0),
            )
        )
    issue_codes = multimodal.get("issue_codes", [])
    issue_count = len(issue_codes) if isinstance(issue_codes, list) else 0
    features.extend(
        (
            1.0 if multimodal.get("ready_for_motion") is True else 0.0,
            _clip(len(statuses) / 16.0, 0.0, 1.0),
            _clip(issue_count / 16.0, 0.0, 1.0),
            (required_healthy_count / required_count if required_count > 0 else 0.0),
        )
    )
    if len(features) != LOCAL_POLICY_SENSOR_FEATURE_COUNT:
        raise AssertionError("local policy sensor feature contract drifted")
    result = tuple(features)
    return result


# 功能：
#   1. 将导航、载荷及实时传感器快照编码为部署与训练共用的固定布局特征。
#   2. 连续控制禁用坐标候选，机体参考表示目标意图而非执行器命令；缺测保持掩码。
# 输入：
#   snapshot：本次拥有的导航快照。
#   continuous_control_speed_mps：可选的明确控制速度尺度。
#   include_candidate_features：仅明确候选选择模式可开启的兼容特征。
# 输出：
#   batch：特征、来源时序及控制单位契约组成的输入批次，不授予飞行权限。
def compile_local_policy_features(
    snapshot: dict[str, object],
    *,
    continuous_control_speed_mps: float | None = None,
    include_candidate_features: bool = True,
) -> LocalPolicyFeatureBatch:
    if type(include_candidate_features) is not bool:
        raise ValueError("candidate feature mode must be an explicit boolean")
    current = _vector(snapshot.get("current_position_m"))
    velocity = _vector(snapshot.get("current_velocity_mps"))
    goal = _vector(snapshot.get("goal_position_m"))
    local_radius = max(0.1, _number(snapshot.get("local_radius_m"), 8.0))
    envelope = _mapping(snapshot.get("vehicle_envelope_m"))
    candidate_speed = max(0.05, _number(envelope.get("candidate_speed"), 0.5))
    strategic = _mapping(snapshot.get("strategic_context"))
    task = _mapping(strategic.get("task"))
    normalized_limits = _mapping(task.get("normalized_pilot_control_limits"))
    declared_control_speed = _number(normalized_limits.get("horizontal_speed_mps"), 0.0)
    if continuous_control_speed_mps is not None:
        if (
            type(continuous_control_speed_mps) not in (int, float)
            or not math.isfinite(_number(continuous_control_speed_mps, math.nan))
            or continuous_control_speed_mps <= 0.0
        ):
            raise ValueError("continuous control speed must be finite and positive")
        control_speed = continuous_control_speed_mps
    elif (
        task.get("local_navigation_output_mode") == "normalized-body-velocity"
        and declared_control_speed > 0.0
    ):
        control_speed = declared_control_speed
    else:
        control_speed = candidate_speed
    state: list[float] = [
        *(_clip(value / control_speed, -4.0, 4.0) for value in velocity),
        *(_clip((goal[index] - current[index]) / local_radius, -2.0, 2.0) for index in range(3)),
        _clip(_number(snapshot.get("goal_distance_m")) / local_radius, 0.0, 4.0),
    ]
    health = _mapping(snapshot.get("perception_health"))
    state.extend(
        (
            1.0 if health.get("stream_healthy") is True else 0.0,
            _clip(_number(health.get("stream_age_seconds")) / 0.5, 0.0, 4.0),
            _clip(_number(health.get("localization_covariance_m2")) / 0.25, 0.0, 4.0),
            _clip(_number(envelope.get("body_radius")) / 5.0, 0.0, 1.0),
            _clip(_number(envelope.get("body_height")) / 10.0, 0.0, 1.0),
            _clip(control_speed / 20.0, 0.0, 1.0),
            _clip(_number(envelope.get("required_local_clearance")) / 20.0, 0.0, 1.0),
        )
    )
    sectors_by_name = {
        str(item.get("sector")): item
        for item in snapshot.get("egocentric_sectors", [])
        if isinstance(item, dict)
    }
    status_value = {"unknown-blocked": 0.0, "blocked": -1.0, "observed-partial": 1.0}
    for name in _SECTOR_LABELS:
        sector = sectors_by_name.get(name, {})
        free = max(0.0, _number(sector.get("observed_free_voxels")))
        occupied = max(0.0, _number(sector.get("occupied_voxels")))
        total = max(1.0, free + occupied)
        nearest = sector.get("nearest_occupied_clearance_m")
        state.extend(
            (
                free / total,
                occupied / total,
                1.0 if nearest is None else _clip(_number(nearest) / local_radius, 0.0, 1.0),
                status_value.get(str(sector.get("status")), 0.0),
            )
        )
    if len(state) != LOCAL_POLICY_STATE_FEATURE_COUNT:
        raise AssertionError("local policy state feature contract drifted")

    candidates = (
        [
            dict(item)
            for item in snapshot.get("authorized_candidate_paths", [])
            if isinstance(item, dict)
        ][:LOCAL_POLICY_MAXIMUM_CANDIDATES]
        if include_candidate_features
        else []
    )
    candidate_features: list[tuple[float, ...]] = []
    candidate_ids: list[str] = []
    for candidate in candidates:
        endpoint = _vector(candidate.get("endpoint_m"))
        features = [
            *(
                _clip((endpoint[index] - current[index]) / local_radius, -2.0, 2.0)
                for index in range(3)
            ),
            _clip(_number(candidate.get("path_length_m")) / local_radius, 0.0, 4.0),
            _clip(
                _number(candidate.get("minimum_predicted_dynamic_clearance_m")) / local_radius,
                0.0,
                4.0,
            ),
            _clip(
                _number(candidate.get("time_to_minimum_dynamic_clearance_seconds")) / 5.0,
                0.0,
                4.0,
            ),
            _clip(_number(candidate.get("required_clearance_m")) / local_radius, 0.0, 2.0),
            1.0 if candidate.get("deterministic_metric_path_validated") is True else 0.0,
            1.0 if candidate.get("dynamic_path_validated") is True else 0.0,
        ]
        kind = str(candidate.get("kind", ""))
        features.extend(1.0 if kind == known else 0.0 for known in _CANDIDATE_KINDS)
        if len(features) != LOCAL_POLICY_CANDIDATE_FEATURE_COUNT:
            raise AssertionError("local policy candidate feature contract drifted")
        candidate_features.append(tuple(features))
        candidate_ids.append(str(candidate.get("candidate_id", "")))
    candidate_mask = [1.0] * len(candidate_features)
    # Padding keeps ONNX dimensions fixed. A padded row cannot become an eligible
    # candidate just because its numerical score exceeds a real candidate's.
    while len(candidate_features) < LOCAL_POLICY_MAXIMUM_CANDIDATES:
        candidate_features.append((0.0,) * LOCAL_POLICY_CANDIDATE_FEATURE_COUNT)
        candidate_mask.append(0.0)
    phase = str(task.get("phase", "UNKNOWN")).strip().upper()
    profile = str(task.get("control_profile", "unknown")).strip().lower()
    trigger = str(task.get("decision_trigger", "unknown")).strip().lower()
    stationary_phase = phase in {
        "CHECKPOINT",
        "ACTION",
        "PICKUP",
        "HOVER",
        "WAYPOINT_SETTLE",
        "LAND",
        "LANDING",
        "COMPLETE",
    }
    speed_mps = math.sqrt(sum(value * value for value in velocity))
    settle_speed_limit_mps = (
        max(0.12, min(0.25, control_speed * 0.35))
        if stationary_phase
        else max(0.25, control_speed * 0.9)
    )
    maneuver_features = [
        1.0 if profile == "precision" else 0.0,
        1.0 if task.get("action_checkpoint_goal") is True else 0.0,
        1.0 if phase == "TAKEOFF" else 0.0,
        1.0 if phase in {"CHECKPOINT", "ACTION", "PICKUP", "HOVER"} else 0.0,
        1.0 if phase == "WAYPOINT_SETTLE" else 0.0,
        1.0 if phase in {"LAND", "LANDING", "COMPLETE"} else 0.0,
        1.0 if phase == "LOCAL_SLOW" else 0.0,
        1.0 if trigger in {"goal-amended", "progress-stalled", "dynamic-obstacle"} else 0.0,
        1.0 if stationary_phase else 0.0,
        _clip(speed_mps / control_speed, 0.0, 4.0),
        _clip(abs(velocity[2]) / control_speed, 0.0, 4.0),
        _clip(settle_speed_limit_mps / control_speed, 0.0, 4.0),
        _clip(speed_mps / settle_speed_limit_mps, 0.0, 4.0),
    ]
    if len(maneuver_features) != LOCAL_POLICY_MANEUVER_FEATURE_COUNT:
        raise AssertionError("local policy maneuver feature contract drifted")
    payload = _mapping(strategic.get("payload"))
    payload_state = str(payload.get("state", "no-runtime-payload-evidence"))
    known_payload_states = (
        "no-runtime-payload-evidence",
        "detached",
        "attached",
        "custody-confirmed",
        "loaded-stable",
    )
    observed_mass = payload.get("observed_payload_mass_kg")
    maximum_mass = max(1e-6, _number(payload.get("maximum_payload_kg"), 0.0))
    # An invalid numeric payload is unknown, not a measured zero-gram load.
    mass_value = _number(observed_mass, math.nan)
    mass_known = math.isfinite(mass_value) and mass_value >= 0.0
    within_limit = payload.get("within_declared_payload_limit")
    transition_steps = payload.get("accepted_transition_steps")
    dynamics = _mapping(payload.get("dynamics"))
    payload_features = [
        *(1.0 if payload_state == state_name else 0.0 for state_name in known_payload_states),
        1.0 if mass_known else 0.0,
        _clip(mass_value / maximum_mass, 0.0, 2.0) if mass_known else 0.0,
        1.0 if within_limit is True else (-1.0 if within_limit is False else 0.0),
        _clip(
            len(transition_steps) / 8.0 if isinstance(transition_steps, list) else 0.0,
            0.0,
            1.0,
        ),
        1.0 if str(task.get("control_profile", "")).lower() == "precision" else 0.0,
        1.0 if dynamics.get("available") is True else 0.0,
        1.0 if dynamics.get("ready") is True else 0.0,
        _clip(_number(dynamics.get("acceleration_forward_m_s2")) / 9.80665, -4.0, 4.0),
        _clip(_number(dynamics.get("acceleration_right_m_s2")) / 9.80665, -4.0, 4.0),
        _clip(_number(dynamics.get("acceleration_down_m_s2")) / 9.80665, -4.0, 4.0),
        _clip(_number(dynamics.get("angular_velocity_forward_rad_s")) / 5.0, -4.0, 4.0),
        _clip(_number(dynamics.get("angular_velocity_right_rad_s")) / 5.0, -4.0, 4.0),
        _clip(_number(dynamics.get("angular_velocity_down_rad_s")) / 5.0, -4.0, 4.0),
        _clip(_number(dynamics.get("roll_deg")) / 45.0, -2.0, 2.0),
        _clip(_number(dynamics.get("pitch_deg")) / 45.0, -2.0, 2.0),
        _clip(_number(dynamics.get("current_battery_a")) / 20.0, 0.0, 4.0),
        _clip(_number(dynamics.get("voltage_v")) / 25.0, 0.0, 2.0),
        _clip(_number(dynamics.get("actuator_mean")), -2.0, 2.0),
        _clip(_number(dynamics.get("actuator_max_abs")), 0.0, 2.0),
    ]
    if len(payload_features) != LOCAL_POLICY_PAYLOAD_FEATURE_COUNT:
        raise AssertionError("local policy payload feature contract drifted")
    realtime_payload = snapshot.get("realtime_feature_snapshot")
    realtime_features = (0.0,) * POLICY_REALTIME_FEATURE_COUNT
    realtime_valid_mask = (0.0,) * POLICY_REALTIME_FEATURE_COUNT
    realtime_features_ready = False
    realtime_snapshot_sha256 = None
    control_feature_contract_sha256 = None
    temporal_evidence = None
    if realtime_payload is not None:
        realtime_snapshot = RealtimeFeatureSnapshot.model_validate(realtime_payload)
        control_feature_contract_sha256 = realtime_snapshot.policy_feature_contract_sha256()
        if not include_candidate_features and (
            control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
        ):
            raise ValueError("continuous control feature semantics are missing or incompatible")
        if len(realtime_snapshot.required_roles) != 3:
            raise ValueError("local control requires all three realtime encoder roles")
        if (
            len(realtime_snapshot.fused_features) != REALTIME_CONTROL_FEATURE_COUNT
            or len(realtime_snapshot.fused_valid_mask) != REALTIME_CONTROL_FEATURE_COUNT
        ):
            raise ValueError("realtime control feature snapshot has an invalid width")
        sensor_features = tuple(realtime_snapshot.fused_features)
        sensor_valid_mask = tuple(realtime_snapshot.fused_valid_mask)
        flight_state = next(
            (
                encoding
                for encoding in realtime_snapshot.encodings
                if encoding.encoder_role == "flight-state-encoder"
            ),
            None,
        )
        if flight_state is not None:
            # A physical state sample owns the history row. Repeated model calls
            # or a changed mission/session must not silently extend that history.
            temporal_evidence = TemporalEvidence(
                stream_id=sha256_json(
                    {
                        "sources": flight_state.source_ids,
                        "execution": task.get("execution_id"),
                        "mission": task.get("mission_id"),
                        "control_session": task.get("control_session_id"),
                    }
                ),
                sample_sha256=flight_state.source_sha256,
                history_slot_revision=flight_state.history_slot_revision,
                observed_at_unix_ms=flight_state.observed_at_unix_ms,
                reset_history=any(
                    code in flight_state.issue_codes
                    for code in (
                        "FLIGHT_STATE_IMU_CLOCK_RESET",
                        "FLIGHT_STATE_SOURCE_CHANGED",
                    )
                ),
            )
        control_reference = (0.0,) * POLICY_CONTROL_REFERENCE_FEATURE_COUNT
        control_reference_mask = (0.0,) * POLICY_CONTROL_REFERENCE_FEATURE_COUNT
        control_reference_ready = False
        if (
            flight_state is not None
            and len(flight_state.features) >= 4
            and flight_state.valid_mask[:4] == [1.0, 1.0, 1.0, 1.0]
        ):
            orientation = QuaternionWxyz(
                w=flight_state.features[0],
                x=flight_state.features[1],
                y=flight_state.features[2],
                z=flight_state.features[3],
            )
            body_velocity = world_enu_to_body(
                orientation,
                Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
            )
            body_goal = world_enu_to_body(
                orientation,
                Vector3(
                    x=goal[0] - current[0],
                    y=goal[1] - current[1],
                    z=goal[2] - current[2],
                ),
            )
            goal_magnitude = math.sqrt(body_goal.x**2 + body_goal.y**2 + body_goal.z**2)
            inverse_goal_magnitude = 1.0 / max(goal_magnitude, 1e-9)
            heading_error = math.atan2(body_goal.y, body_goal.x)
            control_reference = (
                _clip(body_velocity.x / control_speed, -4.0, 4.0),
                _clip(body_velocity.y / control_speed, -4.0, 4.0),
                _clip(body_velocity.z / control_speed, -4.0, 4.0),
                _clip(body_goal.x / local_radius, -2.0, 2.0),
                _clip(body_goal.y / local_radius, -2.0, 2.0),
                _clip(body_goal.z / local_radius, -2.0, 2.0),
                body_goal.x * inverse_goal_magnitude,
                body_goal.y * inverse_goal_magnitude,
                body_goal.z * inverse_goal_magnitude,
                _clip(heading_error / math.pi, -1.0, 1.0),
            )
            control_reference_mask = (1.0,) * POLICY_CONTROL_REFERENCE_FEATURE_COUNT
            control_reference_ready = True
        realtime_features = (*sensor_features, *control_reference)
        realtime_valid_mask = (*sensor_valid_mask, *control_reference_mask)
        realtime_features_ready = realtime_snapshot.ready_for_control and control_reference_ready
        realtime_snapshot_sha256 = realtime_snapshot.snapshot_sha256
    if len(realtime_features) != POLICY_REALTIME_FEATURE_COUNT:
        raise AssertionError("policy realtime feature contract drifted")
    batch = LocalPolicyFeatureBatch(
        temporal_evidence=temporal_evidence,
        state_features=tuple(state),
        candidate_features=tuple(candidate_features),
        candidate_mask=tuple(candidate_mask),
        candidate_ids=tuple(candidate_ids),
        maneuver_features=tuple(maneuver_features),
        payload_features=tuple(payload_features),
        sensor_features=compile_multimodal_sensor_features(snapshot),
        realtime_features=realtime_features,
        realtime_valid_mask=realtime_valid_mask,
        realtime_features_ready=realtime_features_ready,
        realtime_snapshot_sha256=realtime_snapshot_sha256,
        control_feature_contract_sha256=control_feature_contract_sha256,
        pilot_control_limits=(
            PilotControlLimits(
                **{
                    key: normalized_limits[key]
                    for key in (
                        "horizontal_speed_mps",
                        "vertical_speed_mps",
                        "yaw_rate_dps",
                    )
                }
            )
            if all(
                key in normalized_limits
                for key in (
                    "horizontal_speed_mps",
                    "vertical_speed_mps",
                    "yaw_rate_dps",
                )
            )
            else None
        ),
    )
    return batch


class OnnxLocalPolicyBackend:
    """Optional ONNX Runtime backend loaded only in Runtime distributions that ship it."""

    # 功能：
    #   1. 用同次有界读取的模型字节建立单线程会话，核对角色、实时特征及因果历史接口。
    #   2. 显式离线模式才可读取预编码视觉；视觉任务独立串行，不替代实时感知期限。
    # 输入：
    #   self：待初始化的本地执行后端。
    #   package：上游已选择的内容绑定模型包。
    #   execution_providers：可选的有序加速后端选择。
    #   allow_precomputed_visual_features：是否明确允许离线已编码视觉特征。
    #   visual_worker_cpu_ids：可选视觉线程 CPU 亲和性列表。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        package: LoadedLocalPolicyPackage,
        *,
        execution_providers: list[str] | None = None,
        allow_precomputed_visual_features: bool = False,
        visual_worker_cpu_ids: tuple[int, ...] | None = None,
    ) -> None:
        if type(allow_precomputed_visual_features) is not bool:
            raise ValueError("precomputed visual mode must be an explicit boolean")
        if visual_worker_cpu_ids is not None and (
            not isinstance(visual_worker_cpu_ids, (tuple, list))
            or any(type(cpu_id) is not int or cpu_id < 0 for cpu_id in visual_worker_cpu_ids)
            or len(set(visual_worker_cpu_ids)) != len(visual_worker_cpu_ids)
        ):
            raise ValueError("visual worker CPU IDs must be unique non-negative integers")
        normalized_visual_cpu_ids = tuple(sorted(visual_worker_cpu_ids or ()))
        # 自己持有经过重验的清单，外部后续修改原对象不能改变已加载会话的控制阈值。
        package = replace(
            package,
            artifact_paths=dict(package.artifact_paths),
            manifest=type(package.manifest).model_validate(package.manifest.model_dump()),
        )
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("ONNX_RUNTIME_NOT_INSTALLED") from error
        policy_path = package.artifact_paths["local-navigation-policy"]
        providers = select_onnx_execution_providers(
            ort.get_available_providers(),
            execution_providers,
        )
        # These experts execute beside Gazebo, sensor projection and the PX4
        # control loop.  ONNX Runtime otherwise creates a full-sized CPU thread
        # pool for every session; a multi-expert package can then starve the
        # camera producer it depends on.  The networks are intentionally small,
        # so sequential single-thread execution has lower contention and keeps
        # the control pipeline's end-to-end cadence deterministic.  Hardware
        # execution providers may still accelerate operators while this bounds
        # their CPU-side scheduling work.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        loaded_digests = {}
        remaining_source_bytes = 2 * 1024 * 1024 * 1024

        # 功能：
        #   从普通文件边界读取有界权重，核对摘要后将同一不可变字节交给会话，拒绝加载期替换。
        # 输入：
        #   path：清单中明确角色对应的模型文件。
        # 输出：
        #   session：已绑定本次权重及执行后端的 ONNX 会话。
        def load_session(path: Path) -> Any:
            nonlocal remaining_source_bytes
            content = read_plugin_file(path, limit=min(1024**3, remaining_source_bytes))
            remaining_source_bytes -= len(content)
            digest = hashlib.sha256(content).hexdigest()
            expected = next(
                (
                    artifact.sha256
                    for artifact in package.manifest.artifacts
                    if package.artifact_paths.get(artifact.role) == path
                ),
                None,
            )
            if digest != expected:
                raise RuntimeError("LOCAL_POLICY_ARTIFACT_CHANGED_DURING_LOAD")
            loaded_digests[path] = digest
            session = ort.InferenceSession(
                content,
                sess_options=session_options,
                providers=providers,
            )
            return session

        self._session = load_session(policy_path)
        self.execution_providers = tuple(self._session.get_providers())
        self._navigation_sessions = {"local-navigation-policy": self._session}
        self._perception_session = None
        self._risk_session = None
        self._perception_health_session = None
        self._settle_session = None
        self._payload_session = None
        self._anomaly_session = None
        self._cross_modal_session = None
        self._allow_precomputed_visual_features = allow_precomputed_visual_features
        self._visual_worker_cpu_ids = normalized_visual_cpu_ids
        self._visual_worker: LatestInferenceWorker | None = None
        self._observation_history = ObservationHistory(LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH)
        self._control_history = (
            CausalControlHistory(package.manifest.navigation_history_length)
            if package.manifest.navigation_architecture == "causal-gru-control"
            else None
        )
        self._history_lock = Lock()
        self._payload_history_input_name: str | None = None
        self._payload_uses_maneuver_features = False
        observed_inputs = {item.name for item in self._session.get_inputs()}
        observed_outputs = {item.name for item in self._session.get_outputs()}
        base_navigation_inputs = {
            "state_features",
            "candidate_features",
            "candidate_mask",
        }
        expected_inputs = set(base_navigation_inputs)
        if self._control_history is not None:
            expected_inputs = {"state_features", "control_history", "control_history_mask"}
        if package.manifest.realtime_feature_count is not None:
            expected_inputs.update({"realtime_features", "realtime_valid_mask"})
        if package.manifest.heading_context_for_role('local-navigation-policy') is not None:
            expected_inputs.add('heading_context')
        expected_outputs = {"candidate_scores", "action_scores", "risk_score"}
        if package.manifest.pilot_control_mode is not None:
            expected_outputs.add("pilot_control")
        for expert_role in NAVIGATION_EXPERT_ROLES[1:]:
            expert_path = package.artifact_paths.get(expert_role)
            if expert_path is None:
                continue
            expert_session = load_session(expert_path)
            expert_inputs = {item.name for item in expert_session.get_inputs()}
            expert_outputs = {item.name for item in expert_session.get_outputs()}
            expected_expert_inputs = set(observed_inputs) - {'heading_context'}
            if package.manifest.heading_context_for_role(expert_role) is not None:
                expected_expert_inputs.add('heading_context')
            if expert_inputs != expected_expert_inputs or not expected_outputs.issubset(expert_outputs):
                raise RuntimeError("LOCAL_POLICY_EXPERT_TENSOR_CONTRACT_MISMATCH")
            self._navigation_sessions[expert_role] = expert_session
        if self._control_history is not None:
            for role, session in self._navigation_sessions.items():
                metadata = session.get_modelmeta().custom_metadata_map
                shapes = {item.name: item.shape for item in session.get_inputs()}
                heading_contract = package.manifest.heading_context_for_role(role)
                composed_heading = heading_contract is not None
                expected_architecture = PRECISION_HEADING_ARCHITECTURE if composed_heading else 'causal-gru-control'
                if composed_heading and (
                    metadata.get('heading_context_sha256') != heading_contract
                    or metadata.get('yaw_limit_dps') != '20.0'
                    or shapes.get('heading_context') != [1, 23]
                ):
                    raise RuntimeError('LOCAL_POLICY_PRECISION_HEADING_CONTRACT_MISMATCH')
                if (
                    metadata.get("architecture") != expected_architecture
                    or metadata.get("history_contract_sha256") != CONTROL_HISTORY_CONTRACT_SHA256
                    or metadata.get("history_length")
                    != str(package.manifest.navigation_history_length)
                    or metadata.get("feature_contract_sha256")
                    != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                    or shapes.get("control_history")
                    != [
                        1,
                        package.manifest.navigation_history_length,
                        CONTROL_HISTORY_WIDTH,
                    ]
                    or shapes.get("control_history_mask")
                    != [
                        1,
                        package.manifest.navigation_history_length,
                    ]
                ):
                    raise RuntimeError("LOCAL_POLICY_CAUSAL_TENSOR_CONTRACT_MISMATCH")
        perception_path = package.artifact_paths.get("perception-encoder")
        if perception_path is not None:
            expected_inputs.add("visual_features")
            self._perception_session = load_session(perception_path)
            self._perception_sha256 = loaded_digests[perception_path]
            perception_inputs = {item.name for item in self._perception_session.get_inputs()}
            perception_outputs = {item.name for item in self._perception_session.get_outputs()}
            if perception_inputs != {"forward_rgb"} or "visual_features" not in perception_outputs:
                raise RuntimeError("LOCAL_POLICY_VISUAL_TENSOR_CONTRACT_MISMATCH")
        risk_path = package.artifact_paths.get("risk-critic")
        if risk_path is not None:
            self._risk_session = load_session(risk_path)
            risk_inputs = {item.name for item in self._risk_session.get_inputs()}
            risk_outputs = {item.name for item in self._risk_session.get_outputs()}
            accepted_risk_inputs = {frozenset(base_navigation_inputs)}
            if package.manifest.realtime_feature_count is not None:
                accepted_risk_inputs.add(
                    frozenset(
                        {
                            *base_navigation_inputs,
                            "realtime_features",
                            "realtime_valid_mask",
                        }
                    )
                )
            if perception_path is not None:
                accepted_risk_inputs.add(frozenset({*base_navigation_inputs, "visual_features"}))
                accepted_risk_inputs.add(frozenset(expected_inputs))
            if package.manifest.pilot_control_mode is not None:
                accepted_risk_inputs = {
                    frozenset(
                        {
                            *(set(inputs) - {"candidate_features", "candidate_mask"}),
                            "realtime_features",
                            "realtime_valid_mask",
                            "proposed_control",
                        }
                    )
                    for inputs in accepted_risk_inputs
                }
                action_input = next(
                    (
                        item
                        for item in self._risk_session.get_inputs()
                        if item.name == "proposed_control"
                    ),
                    None,
                )
                if (
                    action_input is None
                    or len(action_input.shape) != 2
                    or (action_input.shape[1:] != [ACTION_RISK_FEATURE_COUNT])
                ):
                    raise RuntimeError("LOCAL_POLICY_ACTION_RISK_TENSOR_SHAPE_MISMATCH")
            if frozenset(risk_inputs) not in accepted_risk_inputs or risk_outputs != {"risk_score"}:
                raise RuntimeError("LOCAL_POLICY_RISK_TENSOR_CONTRACT_MISMATCH")
            required_risk = (
                {"state_features", "proposed_control", "realtime_features", "realtime_valid_mask"}
                if package.manifest.pilot_control_mode is not None
                else base_navigation_inputs
            )
            if not required_risk.issubset(risk_inputs):
                raise RuntimeError("LOCAL_POLICY_RISK_TENSOR_CONTRACT_MISMATCH")
        perception_health_path = package.artifact_paths.get("perception-health-critic")
        if perception_health_path is not None:
            self._perception_health_session = load_session(perception_health_path)
            perception_inputs = {item.name for item in self._perception_health_session.get_inputs()}
            perception_outputs = {
                item.name for item in self._perception_health_session.get_outputs()
            }
            if perception_inputs != {"state_features"} or perception_outputs != {"risk_score"}:
                raise RuntimeError("LOCAL_POLICY_PERCEPTION_HEALTH_TENSOR_CONTRACT_MISMATCH")
        payload_path = package.artifact_paths.get("payload-dynamics-adapter")
        if payload_path is not None:
            self._payload_session = load_session(payload_path)
            payload_input_items = {item.name: item for item in self._payload_session.get_inputs()}
            payload_outputs = {item.name for item in self._payload_session.get_outputs()}
            payload_inputs = set(payload_input_items)
            accepted_payload_inputs = {
                frozenset({"payload_features", "state_history", "history_mask"}),
                frozenset({"payload_features", "payload_history", "history_mask"}),
                frozenset({"payload_features", "maneuver_features", "payload_history", "state_history", "history_mask"}),
            }
            if frozenset(payload_inputs) not in accepted_payload_inputs or payload_outputs != {
                "risk_score",
                "controller_step_scale",
            }:
                raise RuntimeError("LOCAL_POLICY_PAYLOAD_TENSOR_CONTRACT_MISMATCH")
            payload_shape = payload_input_items["payload_features"].shape
            if len(payload_shape) != 2 or payload_shape[1:] != [LOCAL_POLICY_PAYLOAD_FEATURE_COUNT]:
                raise RuntimeError("LOCAL_POLICY_PAYLOAD_TENSOR_SHAPE_MISMATCH")
            self._payload_uses_maneuver_features = "maneuver_features" in payload_input_items
            if self._payload_uses_maneuver_features:
                shape = payload_input_items["maneuver_features"].shape
                if len(shape) != 2 or shape[1:] != [LOCAL_POLICY_MANEUVER_FEATURE_COUNT]:
                    raise RuntimeError("LOCAL_POLICY_PAYLOAD_MOTION_TENSOR_SHAPE_MISMATCH")
                shape = payload_input_items["state_history"].shape
                if len(shape) != 3 or shape[1:] != [LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH, LOCAL_POLICY_STATE_FEATURE_COUNT]:
                    raise RuntimeError("LOCAL_POLICY_PAYLOAD_MOTION_HISTORY_SHAPE_MISMATCH")
            self._payload_history_input_name = (
                "payload_history" if "payload_history" in payload_input_items else "state_history"
            )
            payload_history_shape = payload_input_items[self._payload_history_input_name].shape
            expected_history_width = (
                LOCAL_POLICY_PAYLOAD_FEATURE_COUNT
                if self._payload_history_input_name == "payload_history"
                else LOCAL_POLICY_STATE_FEATURE_COUNT
            )
            if len(payload_history_shape) != 3 or payload_history_shape[1:] != [
                LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                expected_history_width,
            ]:
                raise RuntimeError("LOCAL_POLICY_PAYLOAD_HISTORY_TENSOR_SHAPE_MISMATCH")
        settle_path = package.artifact_paths.get("settle-stability-critic")
        if settle_path is not None:
            self._settle_session = load_session(settle_path)
            settle_input_items = {item.name: item for item in self._settle_session.get_inputs()}
            settle_outputs = {item.name for item in self._settle_session.get_outputs()}
            if set(settle_input_items) != {
                "maneuver_features",
                "state_history",
                "history_mask",
            } or settle_outputs != {"risk_score"}:
                raise RuntimeError("LOCAL_POLICY_SETTLE_TENSOR_CONTRACT_MISMATCH")
            settle_history_shape = settle_input_items["state_history"].shape
            settle_mask_shape = settle_input_items["history_mask"].shape
            settle_maneuver_shape = settle_input_items["maneuver_features"].shape
            if (
                len(settle_maneuver_shape) != 2
                or settle_maneuver_shape[1:] != [LOCAL_POLICY_MANEUVER_FEATURE_COUNT]
                or len(settle_history_shape) != 3
                or settle_history_shape[1:]
                != [
                    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                    LOCAL_POLICY_STATE_FEATURE_COUNT,
                ]
                or len(settle_mask_shape) != 2
                or settle_mask_shape[1:] != [LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH]
            ):
                raise RuntimeError("LOCAL_POLICY_SETTLE_TENSOR_SHAPE_MISMATCH")
        anomaly_path = package.artifact_paths.get("state-anomaly-detector")
        if anomaly_path is not None:
            self._anomaly_session = load_session(anomaly_path)
            anomaly_input_items = {item.name: item for item in self._anomaly_session.get_inputs()}
            anomaly_inputs = set(anomaly_input_items)
            anomaly_outputs = {item.name for item in self._anomaly_session.get_outputs()}
            if anomaly_inputs != {
                "state_history",
                "history_mask",
            } or anomaly_outputs != {"anomaly_score"}:
                raise RuntimeError("LOCAL_POLICY_ANOMALY_TENSOR_CONTRACT_MISMATCH")
            state_history_shape = anomaly_input_items["state_history"].shape
            history_mask_shape = anomaly_input_items["history_mask"].shape
            if (
                len(state_history_shape) != 3
                or state_history_shape[1:]
                != [
                    LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                    LOCAL_POLICY_STATE_FEATURE_COUNT,
                ]
                or len(history_mask_shape) != 2
                or history_mask_shape[1:] != [LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH]
            ):
                raise RuntimeError("LOCAL_POLICY_ANOMALY_TENSOR_SHAPE_MISMATCH")
        cross_modal_path = package.artifact_paths.get("cross-modal-consistency-critic")
        if cross_modal_path is not None:
            self._cross_modal_session = load_session(cross_modal_path)
            cross_modal_inputs = {
                item.name: item for item in self._cross_modal_session.get_inputs()
            }
            cross_modal_outputs = {item.name for item in self._cross_modal_session.get_outputs()}
            if set(cross_modal_inputs) != {"sensor_features"} or cross_modal_outputs != {
                "risk_score"
            }:
                raise RuntimeError("LOCAL_POLICY_CROSS_MODAL_TENSOR_CONTRACT_MISMATCH")
            sensor_shape = cross_modal_inputs["sensor_features"].shape
            if len(sensor_shape) != 2 or sensor_shape[1:] != [LOCAL_POLICY_SENSOR_FEATURE_COUNT]:
                raise RuntimeError("LOCAL_POLICY_CROSS_MODAL_TENSOR_SHAPE_MISMATCH")
        if observed_inputs != expected_inputs or not expected_outputs.issubset(observed_outputs):
            raise RuntimeError("LOCAL_POLICY_ONNX_TENSOR_CONTRACT_MISMATCH")
        self._manifest = package.manifest
        if self._perception_session is not None:
            self._visual_worker = LatestInferenceWorker(
                self._encode_visual,
                name="dronedream-local-visual-encoder",
                initializer=(
                    lambda: (
                        _set_current_thread_cpu_affinity(self._visual_worker_cpu_ids)
                        if self._visual_worker_cpu_ids
                        else None
                    )
                ),
            )

    # 功能：
    #   对全部已加载专家做安装前的真实张量计算检查，不注入历史、不发送飞控指令。
    # 输入：
    #   self：已绑定权重及当前清单的后端，只能在进入任务前调用。
    # 输出：
    #   report：各专家接口检查结果，不表示模型质量、延迟准入或飞行资格。
    def verify_runtime_io(self) -> dict:
        from .local_policy_runtime_probe import verify_ensemble_io

        sessions = {
            **self._navigation_sessions,
            "perception-encoder": self._perception_session,
            "risk-critic": self._risk_session,
            "perception-health-critic": self._perception_health_session,
            "settle-stability-critic": self._settle_session,
            "payload-dynamics-adapter": self._payload_session,
            "state-anomaly-detector": self._anomaly_session,
            "cross-modal-consistency-critic": self._cross_modal_session,
        }
        report = verify_ensemble_io(self._manifest, sessions)
        return report

    # 功能：
    #   在异步任务开始前同时绑定 RGB 内容、预处理参数和当前编码器身份。
    # 输入：
    #   self：具有固定视觉清单的后端。
    #   multimodal：当前明确传入的视觉来源列表。
    # 输出：
    #   source：拥有实际字节与联合缓存键的不可变视觉输入。
    def _freeze_visual(self, multimodal: list[dict[str, object]]) -> VisualEncodingInput:
        source = freeze_visual_input(
            multimodal,
            width=self._manifest.visual_width,
            height=self._manifest.visual_height,
            normalization=self._manifest.visual_normalization,
            encoder_sha256=self._perception_sha256,
        )
        return source

    # 功能：
    #   实际预处理 RGB 并运行编码器，核对 float32 特征宽度及有限值，分别记录耗时。
    # 输入：
    #   self：已加载当前视觉编码器的后端。
    #   source：绑定编码器及预处理语义的固定图像。
    # 输出：
    #   result：图像身份、视觉特征以及预处理和推理毫秒耗时。
    def _encode_visual(self, source: VisualEncodingInput) -> _VisualEncodingResult:
        if self._perception_session is None:
            raise RuntimeError("LOCAL_POLICY_VISUAL_ENCODER_NOT_CONFIGURED")
        preprocess_started_ns = time.perf_counter_ns()
        image_tensor = source.tensor()
        preprocess_latency_ms = (time.perf_counter_ns() - preprocess_started_ns) / 1_000_000.0
        encoder_started_ns = time.perf_counter_ns()
        visual = self._perception_session.run(["visual_features"], {"forward_rgb": image_tensor})[0]
        encoder_latency_ms = (time.perf_counter_ns() - encoder_started_ns) / 1_000_000.0
        features = _float_output(visual, self._manifest.visual_feature_count, "VISUAL_FEATURE")
        result = _VisualEncodingResult(
            source_sha256=source.source_sha256,
            features=tuple(float(value) for value in features),
            preprocess_latency_ms=preprocess_latency_ms,
            encoder_latency_ms=encoder_latency_ms,
        )
        return result

    # 功能：
    #   提交一个固定视觉来源，工作队列最多保留运行中任务和一个可替换的待处理来源。
    # 输入：
    #   self：本地后端及其拥有的视觉线程。
    #   multimodal：本次视觉来源，不运行导航决策。
    # 输出：
    #   None：不返回业务数据。
    def prime_visual(self, multimodal: list[dict[str, object]]) -> None:
        if self._visual_worker is None:
            if multimodal:
                raise RuntimeError("LOCAL_POLICY_VISUAL_ENCODER_NOT_CONFIGURED")
            return
        source = self._freeze_visual(multimodal)
        self._visual_worker.submit(source.key, source)

    # 功能：
    #   仅复用完全相同输入的预取结果；等待消耗原始时间预算，不延长观测有效期。
    # 输入：
    #   self：本地编码器和预取队列。
    #   multimodal：要解析的当前视觉来源。
    # 输出：
    #   result：与当前输入身份匹配的视觉编码。
    #   wait_latency_ms：等待已有预取结果的毫秒数，直接执行时为零。
    def _resolve_visual(
        self,
        multimodal: list[dict[str, object]],
    ) -> tuple[_VisualEncodingResult, float]:
        source = self._freeze_visual(multimodal)
        future = self._visual_worker.cached(source.key) if self._visual_worker is not None else None
        if future is None:
            return self._encode_visual(source), 0.0
        wait_started_ns = time.perf_counter_ns()
        # A late result has no right to renew the coordinator's source lease.
        result = future.result(timeout=0.25)
        wait_latency_ms = (time.perf_counter_ns() - wait_started_ns) / 1_000_000.0
        if result.source_sha256 != source.source_sha256:
            raise RuntimeError("LOCAL_POLICY_VISUAL_PREFETCH_IDENTITY_MISMATCH")
        return result, wait_latency_ms

    # 功能：
    #   关闭后端拥有的预取线程，不删除模型、传感器数据或用户实验。
    # 输入：
    #   self：可能创建视觉线程的本地后端。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        if self._visual_worker is not None:
            self._visual_worker.close()

    # 功能：
    #   报告必须使用完整窗口、但实际状态历史尚未填满的异常检测角色。
    # 输入：
    #   self：具有已发生观测历史的后端。
    # 输出：
    #   roles：当前不可执行的异常检测角色集合。
    def not_ready_advisory_roles(self) -> set[AdvisoryExpertRole]:
        if self._anomaly_session is None:
            roles = set()
            return roles
        with self._history_lock:
            history_ready = self._observation_history.ready
        roles = {"state-anomaly-detector"} if not history_ready else set()
        return roles

    # 功能：
    #   按真实来源更新状态与因果历史；重复观测不增加样本，缺少来源时清空历史。
    # 输入：
    #   self：拥有受锁保护历史的后端。
    #   batch：本次已编译的状态、载荷、实时特征及时间身份。
    # 输出：
    #   None：不返回业务数据。
    def prepare_temporal_context(self, batch: LocalPolicyFeatureBatch) -> None:
        with self._history_lock:
            if batch.temporal_evidence is None:
                self._observation_history.clear()
                if self._control_history is not None:
                    self._control_history.clear()
                return
            self._observation_history.append(
                batch.temporal_evidence, batch.state_features, batch.payload_features
            )
            if self._control_history is not None:
                self._control_history.append(
                    batch.temporal_evidence,
                    batch.state_features,
                    batch.realtime_features,
                    batch.realtime_valid_mask,
                )

    # 功能：
    #   当前因果操纵模式要求完整历史；不使用因果网络的明确兼容模式不额外要求该窗口。
    # 输入：
    #   self：已确认模型架构的后端。
    # 输出：
    #   ready：当前因果历史是否满足操纵模型窗口要求。
    def motion_history_ready(self) -> bool:
        with self._history_lock:
            ready = self._control_history is None or self._control_history.ready
        return ready

    # 功能：
    #   1. 执行一个路由操纵专家及适用顾问，风险取最大、负载缩放取最小，不能相互抵消风险。
    #   2. 动作风险专家最后评估缩放后的实际物理请求；本函数不向飞控发送命令。
    # 输入：
    #   self：已加载的会话与来源历史。
    #   batch：同次控制特征、路由与物理单位限制。
    #   multimodal：当前视觉来源，离线特征模式不能同时提供图像。
    # 输出：
    #   result：当前操纵输出、顾问风险、缩放及真实执行耗时。
    def infer(
        self,
        batch: LocalPolicyFeatureBatch,
        *,
        multimodal: list[dict[str, object]],
    ) -> LocalPolicyRawInference:
        import numpy as np

        backend_started_ns = time.perf_counter_ns()
        self.prepare_temporal_context(batch)
        expert_latency_ms: dict[str, float] = {}
        pipeline_latency_ms: dict[str, float] = {}

        # 功能：
        #   调用明确会话并记录实际运行耗时，异常时仍执行计时，输出数量必须吻合具名请求。
        # 输入：
        #   role：本次执行的专家角色。
        #   session：该专家已加载的会话。
        #   output_names：明确需要读取的输出名称。
        #   inputs：具名输入数组映射。
        # 输出：
        #   outputs：与具名输出一一对应的原始数组列表。
        def timed_run(
            role: str,
            session: object,
            output_names: list[str],
            inputs: dict[str, object],
        ) -> list[object]:
            started_ns = time.perf_counter_ns()
            try:
                outputs = session.run(output_names, inputs)  # type: ignore[attr-defined]
                if not isinstance(outputs, list) or len(outputs) != len(output_names):
                    raise RuntimeError("LOCAL_POLICY_SESSION_OUTPUT_COUNT_INVALID")
                return outputs
            finally:
                expert_latency_ms[role] = (time.perf_counter_ns() - started_ns) / 1_000_000.0

        tensor_started_ns = time.perf_counter_ns()
        feeds = {
            "state_features": np.asarray([batch.state_features], dtype=np.float32),
            "candidate_features": np.asarray([batch.candidate_features], dtype=np.float32),
            "candidate_mask": np.asarray([batch.candidate_mask], dtype=np.float32),
        }
        if self._control_history is not None:
            # A causal pilot must not regain the old coordinate-candidate input
            # through an otherwise compatible backend call or training fixture.
            if any(batch.candidate_mask) or any(v for row in batch.candidate_features for v in row):
                raise RuntimeError("LOCAL_POLICY_CAUSAL_COORDINATE_CANDIDATES_FORBIDDEN")
            feeds.pop("candidate_features")
            feeds.pop("candidate_mask")
            with self._history_lock:
                history, history_mask = self._control_history.values()
            feeds["control_history"] = np.asarray([history], dtype=np.float32)
            feeds["control_history_mask"] = np.asarray([history_mask], dtype=np.float32)
        if self._manifest.realtime_feature_count is not None:
            if not batch.realtime_features_ready or batch.realtime_snapshot_sha256 is None:
                raise RuntimeError("LOCAL_POLICY_REALTIME_FEATURES_NOT_READY")
            if self._manifest.pilot_control_mode is not None and (
                self._manifest.control_feature_contract_sha256
                != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                or batch.control_feature_contract_sha256
                != self._manifest.control_feature_contract_sha256
            ):
                raise ValueError("continuous control feature semantics do not match this package")
            realtime_array = np.asarray([batch.realtime_features], dtype=np.float32)
            realtime_mask_array = np.asarray([batch.realtime_valid_mask], dtype=np.float32)
            expected_realtime_shape = (
                1,
                self._manifest.realtime_feature_count,
            )
            if (
                realtime_array.shape != expected_realtime_shape
                or realtime_mask_array.shape != expected_realtime_shape
                or not np.isfinite(realtime_array).all()
                or not np.isfinite(realtime_mask_array).all()
                or not np.isin(realtime_mask_array, (0.0, 1.0)).all()
            ):
                raise RuntimeError("LOCAL_POLICY_REALTIME_FEATURE_CONTRACT_MISMATCH")
            feeds["realtime_features"] = realtime_array
            feeds["realtime_valid_mask"] = realtime_mask_array
        pipeline_latency_ms["tensor-assembly"] = (
            time.perf_counter_ns() - tensor_started_ns
        ) / 1_000_000.0
        precomputed_visual = batch.precomputed_visual_features
        if self._perception_session is None:
            if precomputed_visual is not None:
                # Offline held-out evaluation may use explicitly enabled cached
                # embeddings. Production RGB cannot silently be replaced by them.
                raise RuntimeError("LOCAL_POLICY_PRECOMPUTED_VISUAL_UNEXPECTED")
            if multimodal:
                raise RuntimeError("LOCAL_POLICY_VISUAL_ENCODER_NOT_CONFIGURED")
        else:
            if precomputed_visual is not None:
                if not self._allow_precomputed_visual_features:
                    raise RuntimeError("LOCAL_POLICY_PRECOMPUTED_VISUAL_DISABLED")
                if multimodal:
                    raise RuntimeError("LOCAL_POLICY_VISUAL_INPUT_CONFLICT")
                visual_array = np.asarray([precomputed_visual], dtype=np.float32)
            else:
                visual, wait_latency_ms = self._resolve_visual(multimodal)
                visual_array = np.asarray([visual.features], dtype=np.float32)
                expert_latency_ms["perception-encoder"] = visual.encoder_latency_ms
                pipeline_latency_ms["visual-input-preprocess"] = visual.preprocess_latency_ms
                pipeline_latency_ms["visual-prefetch-wait"] = wait_latency_ms
            if (
                visual_array.shape != (1, self._manifest.visual_feature_count)
                or not np.isfinite(visual_array).all()
            ):
                raise RuntimeError("LOCAL_POLICY_VISUAL_FEATURE_COUNT_MISMATCH")
            feeds["visual_features"] = visual_array
        navigation_session = self._navigation_sessions.get(batch.navigation_expert_role)
        if navigation_session is None:
            raise RuntimeError("LOCAL_POLICY_ROUTED_EXPERT_NOT_LOADED")
        navigation_output_names = ["candidate_scores", "action_scores", "risk_score"]
        if self._manifest.pilot_control_mode is not None:
            navigation_output_names.append("pilot_control")
        navigation_feeds = feeds
        if self._manifest.heading_context_for_role(batch.navigation_expert_role) is not None:
            if batch.pilot_control_limits is None:
                raise RuntimeError('LOCAL_POLICY_PRECISION_HEADING_LIMITS_MISSING')
            heading_values = validate_current_precision_heading_input(batch.precision_heading_context,
                yaw_limit_dps=batch.pilot_control_limits.yaw_rate_dps)
            # 额外输入仅交给显式声明的当前控制分支；不能污染独立风险图或其他专家。
            navigation_feeds = {**feeds, 'heading_context': np.asarray([heading_values], dtype=np.float32)}
        outputs = timed_run(
            batch.navigation_expert_role,
            navigation_session,
            navigation_output_names,
            navigation_feeds,
        )
        navigation_risk_score = _bounded_scalar(outputs[2], "NAVIGATION_RISK")
        candidate_scores = _float_output(outputs[0], LOCAL_POLICY_MAXIMUM_CANDIDATES, "CANDIDATE")
        action_count = 4 if self._manifest.pilot_control_mode is not None else 3
        action_scores = _float_output(outputs[1], action_count, "ACTION")
        risk_score = navigation_risk_score
        advisory_risk_scores: dict[AdvisoryExpertRole, float] = {}
        invoked_advisors: list[AdvisoryExpertRole] = []
        controller_step_scale = 1.0
        requested_advisors = (
            set(batch.expert_routing.advisory_roles)
            if batch.expert_routing is not None
            else {
                role
                for role, session in (
                    ("risk-critic", self._risk_session),
                    ("perception-health-critic", self._perception_health_session),
                    ("settle-stability-critic", self._settle_session),
                    ("payload-dynamics-adapter", self._payload_session),
                    ("state-anomaly-detector", self._anomaly_session),
                    ("cross-modal-consistency-critic", self._cross_modal_session),
                )
                if session is not None
            }
        )
        if (
            self._perception_health_session is not None
            and "perception-health-critic" in requested_advisors
        ):
            perception_output = timed_run(
                "perception-health-critic",
                self._perception_health_session,
                ["risk_score"],
                {"state_features": feeds["state_features"]},
            )[0]
            perception_score = _bounded_scalar(perception_output, "PERCEPTION_HEALTH")
            advisory_risk_scores["perception-health-critic"] = perception_score
            risk_score = max(risk_score, perception_score)
            invoked_advisors.append("perception-health-critic")
        temporal_inputs = None
        anomaly_history_ready = False
        if (
            self._anomaly_session is not None
            or (
                self._payload_session is not None
                and "payload-dynamics-adapter" in requested_advisors
            )
            or (
                self._settle_session is not None and "settle-stability-critic" in requested_advisors
            )
        ):
            with self._history_lock:
                history = [row[0] for row in self._observation_history.rows]
                payload_history = [row[1] for row in self._observation_history.rows]
            anomaly_history_ready = len(history) == LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
            history_padding = LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH - len(history)
            anomaly_history = np.asarray(
                [
                    [
                        *(
                            (0.0,) * LOCAL_POLICY_STATE_FEATURE_COUNT
                            for _ in range(history_padding)
                        ),
                        *history,
                    ]
                ],
                dtype=np.float32,
            )
            history_mask = np.asarray(
                [[0.0] * history_padding + [1.0] * len(history)],
                dtype=np.float32,
            )
            temporal_inputs = {
                "state_history": anomaly_history,
                "history_mask": history_mask,
            }
        if self._settle_session is not None and "settle-stability-critic" in requested_advisors:
            if temporal_inputs is None:
                raise AssertionError("settle critic requires temporal inputs")
            settle_output = timed_run(
                "settle-stability-critic",
                self._settle_session,
                ["risk_score"],
                {
                    **temporal_inputs,
                    "maneuver_features": np.asarray([batch.maneuver_features], dtype=np.float32),
                },
            )[0]
            settle_score = _bounded_scalar(settle_output, "SETTLE")
            advisory_risk_scores["settle-stability-critic"] = settle_score
            risk_score = max(risk_score, settle_score)
            invoked_advisors.append("settle-stability-critic")
        if self._payload_session is not None and "payload-dynamics-adapter" in requested_advisors:
            if temporal_inputs is None:
                raise AssertionError("payload adapter requires temporal inputs")
            if self._payload_history_input_name is None:
                raise AssertionError("payload adapter history input is unavailable")
            payload_temporal_input = temporal_inputs["state_history"]
            if self._payload_history_input_name == "payload_history":
                payload_padding = LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH - len(payload_history)
                payload_temporal_input = np.asarray(
                    [
                        [
                            *(
                                (0.0,) * LOCAL_POLICY_PAYLOAD_FEATURE_COUNT
                                for _ in range(payload_padding)
                            ),
                            *payload_history,
                        ]
                    ],
                    dtype=np.float32,
                )
            payload_outputs = timed_run(
                "payload-dynamics-adapter",
                self._payload_session,
                ["risk_score", "controller_step_scale"],
                {
                    self._payload_history_input_name: payload_temporal_input,
                    "history_mask": temporal_inputs["history_mask"],
                    "payload_features": np.asarray([batch.payload_features], dtype=np.float32),
                    **({"maneuver_features": np.asarray([batch.maneuver_features], dtype=np.float32),
                        "state_history": temporal_inputs["state_history"]}
                       if self._payload_uses_maneuver_features else {}),
                },
            )
            payload_risk = _bounded_scalar(payload_outputs[0], "PAYLOAD")
            payload_scale = _bounded_scalar(payload_outputs[1], "PAYLOAD_STEP_SCALE", minimum=0.1)
            advisory_risk_scores["payload-dynamics-adapter"] = payload_risk
            risk_score = max(risk_score, payload_risk)
            controller_step_scale = min(controller_step_scale, payload_scale)
            invoked_advisors.append("payload-dynamics-adapter")
        if (
            self._anomaly_session is not None
            and "state-anomaly-detector" in requested_advisors
            and anomaly_history_ready
        ):
            if temporal_inputs is None:
                raise AssertionError("anomaly detector requires temporal inputs")
            anomaly_output = timed_run(
                "state-anomaly-detector",
                self._anomaly_session,
                ["anomaly_score"],
                temporal_inputs,
            )[0]
            anomaly_score = _bounded_scalar(anomaly_output, "ANOMALY")
            advisory_risk_scores["state-anomaly-detector"] = anomaly_score
            risk_score = max(risk_score, anomaly_score)
            invoked_advisors.append("state-anomaly-detector")
        if (
            self._cross_modal_session is not None
            and "cross-modal-consistency-critic" in requested_advisors
        ):
            cross_modal_output = timed_run(
                "cross-modal-consistency-critic",
                self._cross_modal_session,
                ["risk_score"],
                {"sensor_features": np.asarray([batch.sensor_features], dtype=np.float32)},
            )[0]
            cross_modal_score = _bounded_scalar(cross_modal_output, "CROSS_MODAL")
            advisory_risk_scores["cross-modal-consistency-critic"] = cross_modal_score
            risk_score = max(risk_score, cross_modal_score)
            invoked_advisors.append("cross-modal-consistency-critic")
        pilot_control = None
        if self._manifest.pilot_control_mode is not None:
            pilot_values = _float_output(
                outputs[3], LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT, "PILOT_CONTROL"
            )
            if (
                pilot_values.shape != (LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT,)
                or not np.isfinite(pilot_values).all()
                or np.any(pilot_values < -1.0)
                or np.any(pilot_values > 1.0)
            ):
                raise RuntimeError("LOCAL_POLICY_PILOT_CONTROL_OUTPUT_INVALID")
            pilot_control = NormalizedPilotControl(
                forward_axis=float(pilot_values[0]),
                right_axis=float(pilot_values[1]),
                up_axis=float(pilot_values[2]),
                yaw_axis=float(pilot_values[3]),
            )
        if self._risk_session is not None and "risk-critic" in requested_advisors:
            # Run last: this critic must assess the actual scaled axes, not a
            # hypothetical unscaled or candidate-selected action from earlier.
            critic_input_names = {item.name for item in self._risk_session.get_inputs()}
            critic_feeds = {
                name: value for name, value in feeds.items() if name in critic_input_names
            }
            if "proposed_control" in critic_input_names:
                if pilot_control is None or batch.pilot_control_limits is None:
                    raise RuntimeError("LOCAL_POLICY_ACTION_RISK_CONTEXT_MISSING")
                critic_feeds["proposed_control"] = np.asarray(
                    [
                        action_risk_features(
                            pilot_control,
                            batch.pilot_control_limits,
                            harness_scale=controller_step_scale,
                        )
                    ],
                    dtype=np.float32,
                )
            critic_output = timed_run(
                "risk-critic", self._risk_session, ["risk_score"], critic_feeds
            )[0]
            critic_score = _bounded_scalar(critic_output, "RISK_CRITIC")
            advisory_risk_scores["risk-critic"] = critic_score
            risk_score = max(risk_score, critic_score)
            invoked_advisors.append("risk-critic")
        pipeline_latency_ms["backend-wall"] = (
            time.perf_counter_ns() - backend_started_ns
        ) / 1_000_000.0
        result = LocalPolicyRawInference(
            candidate_scores=candidate_scores.tolist(),
            action_scores=action_scores.tolist(),
            risk_score=risk_score,
            navigation_risk_score=navigation_risk_score,
            advisory_risk_scores=advisory_risk_scores,
            invoked_advisory_roles=invoked_advisors,
            expert_latency_ms=expert_latency_ms,
            pipeline_latency_ms=pipeline_latency_ms,
            controller_step_scale=controller_step_scale,
            pilot_control=pilot_control,
        )
        return result


class LocalPolicyPort:
    """Expose one qualified local policy through the existing structured model port API."""

    # 功能：
    #   接入上游已选择的本地包；开发采集需明确开启，调用耗时宽限不放宽观测有效期。
    # 输入：
    #   self：待初始化的结构化本地模型端口。
    #   package：已由上游选择并绑定资格的包。
    #   backend：可选显式后端；省略则创建实际 ONNX 后端。
    #   development_payload_collection：明确的非资格采集开关。
    #   visual_worker_cpu_ids：可选视觉工作线程的 CPU 序号。
    #   scheduling_jitter_grace_ms：零到五十毫秒的调用调度宽限。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        package: LoadedLocalPolicyPackage,
        *,
        backend: LocalPolicyInferenceBackend | None = None,
        development_payload_collection: bool = False,
        visual_worker_cpu_ids: tuple[int, ...] | None = None,
        scheduling_jitter_grace_ms: float = 0.0,
    ) -> None:
        if type(development_payload_collection) is not bool:
            raise ValueError("development collection must be an explicit boolean")
        if (
            type(scheduling_jitter_grace_ms) not in (int, float)
            or not math.isfinite(_number(scheduling_jitter_grace_ms, math.nan))
        ) or not (0.0 <= scheduling_jitter_grace_ms <= 50.0):
            raise ValueError("local policy scheduling jitter grace must be in [0, 50] ms")
        # 端口仲裁与后端加载均持有自己的已校验描述，调用者不能通过原对象改变阈值或角色。
        package = replace(
            package,
            artifact_paths=dict(package.artifact_paths),
            manifest=type(package.manifest).model_validate(package.manifest.model_dump()),
        )
        self.package = package
        self.backend = (
            backend
            if backend is not None
            else OnnxLocalPolicyBackend(
                package,
                visual_worker_cpu_ids=visual_worker_cpu_ids,
            )
        )
        self.development_payload_collection = development_payload_collection
        self._observations = PolicyObservationBuffer()
        self._prepared_inputs: OrderedDict[str, LocalPolicyFeatureBatch] = OrderedDict()
        self._prepared_inputs_lock = Lock()
        self.scheduling_jitter_grace_ms = float(scheduling_jitter_grace_ms)
        self.settings = ProviderSettings(
            name="local-policy",
            model=package.manifest.package_id,
            api_key_env="",
            base_url=None,
            api_style="chat-completions",
        )

    # 功能：
    #   提供一次本地调用的包级时间预算，不作为可延长的控制动作持续时间。
    # 输入：
    #   self：持有已选择包的端口。
    # 输出：
    #   timeout_seconds：由清单毫秒预算换算的秒数。
    @property
    def invocation_timeout_seconds(self) -> float:
        timeout_seconds = self.package.manifest.maximum_inference_latency_ms / 1_000.0
        return timeout_seconds

    # 功能：声明端口可独立接收真实连续观测；输入：当前包；输出：只对因果操纵模式启用。
    @property
    def supports_observation_history(self) -> bool:
        return self.package.manifest.pilot_control_mode is not None

    # 功能：仅连续本地策略可使用上游原始来源截止时间；不会把云端超时作为动作寿命。
    # 输入：当前模型包；输出：是否接受本次调用独立的单调时钟期限。
    @property
    def supports_control_deadline(self) -> bool:
        return self.package.manifest.pilot_control_mode is not None

    # 功能：校验并积累真实观测，不推理、不刷新源钟、不授权动作；模型忙时也可调用。
    # 输入：地图所有者线程生成的独立导航快照；输出：是否保存一条新的来源观测。
    def observe_navigation_snapshot(self, snapshot: dict) -> bool:
        if snapshot.get('snapshot_sha256') != sha256_json(
                {key: value for key, value in snapshot.items() if key != 'snapshot_sha256'}):
            raise ValueError('LOCAL_POLICY_SNAPSHOT_HASH_INVALID')
        batch = compile_local_policy_features(snapshot, include_candidate_features=False)
        accepted = self._observations.stage(batch)
        if self.supports_observation_history:
            # 历史队列已复制其需要的字段；该完整 batch 只由本缓存持有，
            # 后续相同内容的推理一次性移交所有权，不反复编译同一张量。
            with self._prepared_inputs_lock:
                key = snapshot['snapshot_sha256']
                self._prepared_inputs[key] = batch
                self._prepared_inputs.move_to_end(key)
                while len(self._prepared_inputs) > 4:
                    self._prepared_inputs.popitem(last=False)
        return accepted

    # 功能：
    #   声明本地因果历史按传感器来源维护，不支持云端会话缓存接口。
    # 输入：
    #   self：本地模型端口。
    # 输出：
    #   supported：固定为否的云端上下文能力标记。
    @property
    def supports_provider_context(self) -> bool:
        supported = False
        return supported

    # 功能：
    #   满足通用端口重置接口；本地没有远程传输，不因此清除真实来源历史。
    # 输入：
    #   self：本地模型端口。
    # 输出：
    #   None：不返回业务数据。
    def reset_transport(self) -> None:
        return None

    # 功能：
    #   满足通用失败处理接口；不自动重试或换成另一套未选定模型包。
    # 输入：
    #   self：本次已固定模型来源的端口。
    # 输出：
    #   None：不返回业务数据。
    def advance_after_failure(self) -> None:
        return None

    # 功能：
    #   将视觉预取请求交给支持该能力的后端，仅预处理图像、不执行操纵动作。
    # 输入：
    #   self：可能具备预取能力的端口。
    #   multimodal：明确传入的本次多模态来源。
    # 输出：
    #   None：不返回业务数据。
    def prime_multimodal(self, multimodal: list[dict[str, object]]) -> None:
        prime = getattr(self.backend, "prime_visual", None)
        if callable(prime):
            prime([dict(item) for item in multimodal])

    # 功能：
    #   通过统一抽象关闭后端工作线程，不清理用户文件或改动模型资格。
    # 输入：
    #   self：持有本次后端的端口。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()

    # 功能：
    #   计算本次准备、推理与仲裁总耗时；超预算拒绝，不声称可以中断正在执行的 ONNX。
    # 输入：
    #   self：声明时间预算与显式宽限的端口。
    #   started：本次调用的原始单调时钟起点。
    #   raw：已有推理结果时附上模型耗时证据，缺省表示尚未发生实际调用。
    #   control_deadline_monotonic：已扣除发令余量的原始来源期限；不允许给旧帧续期。
    # 输出：
    #   elapsed_ms：仍处于本次调用预算内的毫秒耗时。
    def _check_call_budget(
        self,
        started: float,
        raw: LocalPolicyRawInference | None = None,
        *, control_deadline_monotonic: float | None = None,
    ) -> float:
        now = time.monotonic()
        elapsed_ms = (now - started) * 1_000.0
        limit_ms = float(self.package.manifest.maximum_inference_latency_ms)
        grace_ms = self.scheduling_jitter_grace_ms
        if control_deadline_monotonic is not None:
            # 性能标称值不是新鲜度截止时间。最多容纳 50 ms 调度波动，且必须
            # 在上游扣除发令保留时间后的原期限内完成；旧传感器不会因此续龄。
            grace_ms = min(50., max(0., (control_deadline_monotonic - started) * 1000 - limit_ms))
            if now >= control_deadline_monotonic:
                error = TimeoutError("LOCAL_POLICY_CONTROL_DEADLINE_EXPIRED")
                error.reason_code = "LOCAL_POLICY_CONTROL_DEADLINE_EXPIRED"
                error.diagnostic_metrics = {
                    'total-latency-ms': elapsed_ms,
                    'source-budget-at-entry-ms': (control_deadline_monotonic - started) * 1000,
                    'inference-completed': int(raw is not None),
                    **({f'pipeline-{k}-ms': v for k, v in raw.pipeline_latency_ms.items()}
                       if raw is not None else {}),
                }
                raise error
        if elapsed_ms > limit_ms + grace_ms:
            if raw is None:
                # No network was invoked. Do not report a physical model call.
                error = TimeoutError(
                    f"LOCAL_POLICY_INPUT_PREPARATION_EXCEEDED_LATENCY_BOUND: {elapsed_ms:.3f} ms"
                )
                error.reason_code = "LOCAL_POLICY_INPUT_PREPARATION_EXCEEDED_LATENCY_BOUND"
                raise error
            metrics = {
                "total-latency-ms": elapsed_ms,
                "qualified-limit-ms": limit_ms,
                "scheduling-jitter-grace-ms": grace_ms,
            }
            if raw is not None:
                metrics.update({f"pipeline-{k}-ms": v for k, v in raw.pipeline_latency_ms.items()})
                metrics.update({f"expert-{k}-ms": v for k, v in raw.expert_latency_ms.items()})
            raise ModelInvocationError(
                "local-policy/local_navigation_advisor exceeded its qualified latency bound",
                attempts_used=1,
                reason_code="LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED",
                diagnostic_metrics=dict(list(metrics.items())[:16]),
            )
        return elapsed_ms

    # 功能：
    #   1. 固定并校验输入，按真实时序路由专家，一次推理后重验输出，再仲裁动作和完整记账。
    #   2. 所有准备与仲裁计入本次时限；本地调用的零云端 token 不代表已完成飞行或免费 API。
    # 输入：
    #   self：本次固定的模型包和后端。
    #   role：只允许 local_navigation_advisor。
    #   output_type：只允许 TextNavigationDecision。
    #   instructions：兼容通用端口的指令文本，本地固定模型不消费提示词。
    #   input_artifact：包含当前导航快照与内容摘要的输入对象。
    #   context_id：通用云端上下文标识，本地不使用。
    #   multimodal：可选的当前图像来源。
    #   maximum_physical_attempts：省略或显式整数一，不自动重试。
    #   control_deadline_monotonic：连续控制传入的剩余来源期限，省略时维持包声明时限。
    # 输出：
    #   result：本次导航决策及来源、角色、耗时的完整调用记录。
    def call(
        self,
        *,
        role: str,
        output_type: type[BaseModel],
        instructions: str,
        input_artifact: BaseModel | dict[str, object],
        context_id: str | None = None,
        multimodal: list[dict[str, object]] | None = None,
        maximum_physical_attempts: int | None = None,
        control_deadline_monotonic: float | None = None,
    ) -> StructuredCallResult[TextNavigationDecision]:
        del instructions, context_id
        if role != "local_navigation_advisor" or output_type is not TextNavigationDecision:
            raise ValueError("local policy port is restricted to local navigation decisions")
        if maximum_physical_attempts is not None and (
            type(maximum_physical_attempts) is not int or maximum_physical_attempts != 1
        ):
            raise ValueError("local policy inference permits one physical attempt")
        started_at = datetime.now(UTC)
        started = time.monotonic()
        if control_deadline_monotonic is not None and (
            not self.supports_control_deadline
            or type(control_deadline_monotonic) not in (int, float)
            or not started < control_deadline_monotonic <= started + .25
            or not math.isfinite(control_deadline_monotonic)
        ):
            raise ValueError("LOCAL_POLICY_CONTROL_DEADLINE_INVALID")
        payload = copy_json(
            (
                input_artifact.model_dump(mode="json")
                if isinstance(input_artifact, BaseModel)
                else dict(input_artifact)
            ),
            limit=16 * 1024 * 1024,
        )
        snapshot = payload.get("text_navigation_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("local policy input is missing the navigation snapshot")
        snapshot_payload = {
            key: value for key, value in snapshot.items() if key != "snapshot_sha256"
        }
        if snapshot.get("snapshot_sha256") != sha256_json(snapshot_payload):
            raise ValueError("LOCAL_POLICY_SNAPSHOT_HASH_INVALID")
        # 上方已对本次输入独立复制并重新计算内容摘要。缓存不保存权限或
        # 到期判定；当前时钟、路由、历史及调用期限仍在每次推理重新检查。
        # pop 将唯一的可变子对象所有权交给本次调用，后端不能污染下次输入。
        with self._prepared_inputs_lock:
            batch = (self._prepared_inputs.pop(snapshot['snapshot_sha256'], None)
                     if self.supports_observation_history else None)
        if batch is None:
            batch = compile_local_policy_features(
                snapshot,
                include_candidate_features=self.package.manifest.pilot_control_mode is None,
            )
        available_roles = set(self.package.artifact_paths)
        prepare_context = getattr(self.backend, "prepare_temporal_context", None)
        if callable(prepare_context):
            # 仅推理线程修改后端历史。传感器线程保存的新帧最多排队，不能越过本次来源时刻。
            for observation in self._observations.take_through(batch.temporal_evidence):
                prepare_context(observation)
            prepare_context(batch)
        not_ready_roles = set()
        readiness = getattr(self.backend, "not_ready_advisory_roles", None)
        if callable(readiness):
            not_ready_roles = readiness()
            available_roles.difference_update(not_ready_roles)
        routing = route_local_experts(
            snapshot,
            available_roles=available_roles,
            allow_general_fallback=self.package.manifest.pilot_control_mode is None,
        )
        motion_history = getattr(self.backend, "motion_history_ready", None)
        motion_history_ready = not callable(motion_history) or motion_history()
        if (not_ready_roles or not motion_history_ready) and (
            self.package.manifest.pilot_control_mode is not None
        ):
            routing = LocalExpertRoutingDecision.model_validate(
                {
                    **routing.model_dump(),
                    "motion_permitted": False,
                    "reason_codes": [
                        *routing.reason_codes,
                        "LOCAL_EXPERT_TEMPORAL_HISTORY_WARMING",
                    ],
                }
            )
        batch = replace(
            batch,
            navigation_expert_role=routing.selected_navigation_role,
            expert_routing=routing,
        )
        if self.package.manifest.heading_context_for_role(routing.selected_navigation_role) is not None:
            batch = replace(batch, precision_heading_context=current_precision_heading_input(
                snapshot, now_unix_ms=int(time.time() * 1000)))
        preparation_ms = self._check_call_budget(started, control_deadline_monotonic=control_deadline_monotonic)
        try:
            raw = self.backend.infer(batch, multimodal=list(multimodal or []))
            # Pydantic assignment checks do not cover model_copy(update=...) or
            # mutated nested containers. Own and revalidate the backend graph.
            raw = LocalPolicyRawInference.model_validate(raw.model_dump(mode="python"))
            if len(raw.invoked_advisory_roles) != len(set(raw.invoked_advisory_roles)) or set(
                raw.invoked_advisory_roles
            ) != set(routing.advisory_roles):
                raise RuntimeError("LOCAL_POLICY_ADVISORY_EXECUTION_MISMATCH")
        except Exception as error:
            raise ModelInvocationError(
                f"local-policy/{role} inference failed: {type(error).__name__}: {error}",
                attempts_used=1,
                reason_code="LOCAL_POLICY_INFERENCE_FAILED",
            ) from error
        routing_reasons = set(routing.reason_codes)
        strategic_context = snapshot.get("strategic_context")
        payload_context = (
            strategic_context.get("payload") if isinstance(strategic_context, dict) else None
        )
        payload_state = (
            str(payload_context.get("state", "")).lower()
            if isinstance(payload_context, dict)
            else ""
        )
        development_payload_cap = bool(
            self.development_payload_collection
            and payload_state in {"attached", "custody-confirmed", "loaded-stable"}
            and "LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY" not in routing_reasons
        )
        if development_payload_cap:
            # Development collection is explicitly non-qualifying. Whether it
            # bootstraps the first adapter or gathers independent evidence for
            # a replacement, every payload motion target remains capped at one
            # fifth of the ordinary controller step. Stale dynamics still hold.
            raw = raw.model_copy(
                update={"controller_step_scale": min(raw.controller_step_scale, 0.2)}
            )
        inference_finished_ms = self._check_call_budget(started, raw, control_deadline_monotonic=control_deadline_monotonic)
        navigation_risk_score = (
            raw.navigation_risk_score if raw.navigation_risk_score is not None else raw.risk_score
        )
        navigation_decision = self._decision(
            snapshot=snapshot,
            batch=batch,
            raw=raw.model_copy(
                update={
                    "risk_score": navigation_risk_score,
                    "invoked_advisory_roles": [],
                    "advisory_risk_scores": {},
                    "controller_step_scale": 1.0,
                }
            ),
        )
        decision = self._decision(snapshot=snapshot, batch=batch, raw=raw)
        record = ModelCallRecord(
            call_id=f"model-{uuid4().hex[:24]}",
            role="local_navigation_advisor",
            attempt=1,
            input_sha256=sha256_json(payload),
            output_sha256=sha256_json(decision),
            output_schema=TextNavigationDecision.__name__,
            provider="local-policy",
            model="+".join(
                (
                    f"{self.package.manifest.package_id}/{routing.selected_navigation_role}",
                    *(
                        ("perception-encoder",)
                        if "perception-encoder" in self.package.artifact_paths
                        else ()
                    ),
                    *raw.invoked_advisory_roles,
                )
            ),
            input_tokens=0,
            output_tokens=0,
            latency_ms=round(inference_finished_ms),
            created_at=started_at,
            local_expert_trace=LocalExpertInferenceTrace(
                requested_navigation_role=routing.requested_navigation_role,
                selected_navigation_role=routing.selected_navigation_role,
                fallback_used=routing.fallback_used,
                decision_reason_codes=list(decision.risk_notes),
                temporal_history_ready=motion_history_ready and not not_ready_roles,
                navigation_action=navigation_decision.action,
                navigation_selected_candidate_id=(
                    navigation_decision.selected_candidate_id
                    if navigation_decision.action == "select-candidate"
                    else None
                ),
                candidate_scores=raw.candidate_scores,
                action_scores=raw.action_scores,
                navigation_risk_score=navigation_risk_score,
                advisory_risk_scores=raw.advisory_risk_scores,
                expert_latency_ms=raw.expert_latency_ms,
                pipeline_latency_ms=raw.pipeline_latency_ms,
                aggregate_risk_score=raw.risk_score,
                controller_step_scale=raw.controller_step_scale,
                pilot_control=navigation_decision.pilot_control,
            ),
        )
        result = StructuredCallResult(artifact=decision, record=record)
        elapsed_ms = self._check_call_budget(started, raw, control_deadline_monotonic=control_deadline_monotonic)
        # These are nested wall intervals, not independently additive P99s.
        record.local_expert_trace.pipeline_latency_ms = {
            **raw.pipeline_latency_ms,
            "port-input-preparation": preparation_ms,
            "port-decision-record": elapsed_ms - inference_finished_ms,
            "port-wall": elapsed_ms,
            "qualified-latency-limit": float(self.package.manifest.maximum_inference_latency_ms),
            "deadline-bound-scheduling-grace": max(0., elapsed_ms - self.package.manifest.maximum_inference_latency_ms),
        }
        # Include trace validation as well; only the final scalar stamp/return
        # remain outside the measured interval. Never round before admission.
        record.latency_ms = round(self._check_call_budget(started, raw, control_deadline_monotonic=control_deadline_monotonic))
        return result

    # 功能：
    #   1. 在包声明的输出模式内仲裁：缺少必需专家、风险过高或动作分数接近时保持不动。
    #   2. 连续模式输出学习得到的四轴幅度；明确兼容模式只能选择已授权候选，不自行编坐标。
    # 输入：
    #   self：提供阈值、模式与明确开发开关的端口。
    #   snapshot：已验证的当前导航快照。
    #   batch：该快照的特征、候选掩码和专家路由。
    #   raw：已重验的操纵及顾问输出。
    # 输出：
    #   decision：等待外层安全控制权限检查的结构化决策。
    def _decision(
        self,
        *,
        snapshot: dict[str, object],
        batch: LocalPolicyFeatureBatch,
        raw: LocalPolicyRawInference,
    ) -> TextNavigationDecision:
        snapshot_sha256 = str(snapshot.get("snapshot_sha256", ""))
        routing = batch.expert_routing
        if routing is not None and not routing.motion_permitted:
            decision = TextNavigationDecision(
                snapshot_sha256=snapshot_sha256,
                action="hold",
                selected_candidate_id=None,
                risk_notes=[
                    code
                    for code in routing.reason_codes
                    if "UNAVAILABLE" in code or "WARMING" in code
                ],
                rationale_summary=(
                    "Required specialist or independent temporal evidence is unavailable."
                ),
            )
            return decision
        expert_summary = (
            routing.selected_navigation_role
            if routing is not None
            else batch.navigation_expert_role
        )
        fallback_notes = (
            ["LOCAL_EXPERT_GENERAL_FALLBACK"]
            if routing is not None and routing.fallback_used
            else []
        )
        advisory_notes = [
            *(fallback_notes),
            *(
                ["LOCAL_POLICY_CONTROLLER_STEP_RESTRICTED"]
                if raw.controller_step_scale < 0.999
                else []
            ),
            *(
                ["LOCAL_EXPERT_DEVELOPMENT_PAYLOAD_COLLECTION_FALLBACK"]
                if self.development_payload_collection
                and routing is not None
                and "LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE" in routing.reason_codes
                and "LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY" not in routing.reason_codes
                else []
            ),
        ]
        payload_gate_reasons = [
            reason
            for reason in (
                "LOCAL_EXPERT_PAYLOAD_LIMIT_NOT_VERIFIED",
                "LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE",
                "LOCAL_EXPERT_PAYLOAD_DYNAMICS_NOT_READY",
            )
            if routing is not None and reason in routing.reason_codes
        ]
        development_payload_fallback = bool(
            self.development_payload_collection
            and payload_gate_reasons == ["LOCAL_EXPERT_PAYLOAD_ADAPTER_UNAVAILABLE"]
        )
        if payload_gate_reasons and not development_payload_fallback:
            decision = TextNavigationDecision(
                snapshot_sha256=snapshot_sha256,
                action="hold",
                rationale_summary=(
                    "Payload motion was held until its qualified local dynamics "
                    "expert and fresh PX4 telemetry were both available."
                ),
                risk_notes=[*payload_gate_reasons, *advisory_notes],
            )
            return decision
        if raw.risk_score >= self.package.manifest.risk_hold_threshold:
            decision = TextNavigationDecision(
                snapshot_sha256=snapshot_sha256,
                action="hold",
                rationale_summary=(
                    f"{expert_summary} was vetoed by the aggregated navigation/advisor risk gate."
                ),
                risk_notes=["LOCAL_POLICY_RISK_THRESHOLD_REACHED", *advisory_notes],
            )
            return decision
        valid_indices = [index for index, mask in enumerate(batch.candidate_mask) if mask == 1.0]
        if len(valid_indices) != len(batch.candidate_ids):
            raise ValueError("local policy candidate identities do not match the mask")
        valid_candidates = [raw.candidate_scores[index] for index in valid_indices]
        if self.package.manifest.pilot_control_mode is not None:
            if raw.pilot_control is None:
                raise RuntimeError("LOCAL_POLICY_PILOT_CONTROL_OUTPUT_MISSING")
            if len(raw.action_scores) != 4:
                raise RuntimeError("LOCAL_POLICY_PILOT_ACTION_OUTPUT_MISSING")
            action_names = ("hold", "request-new-scan", "abort", "pilot-control")
            options = sorted(
                zip(raw.action_scores, action_names, strict=True),
                key=lambda item: item[0],
                reverse=True,
            )
            best = options[0]
            margin = best[0] - options[1][0]
            if margin < self.package.manifest.minimum_selection_margin:
                decision = TextNavigationDecision(
                    snapshot_sha256=snapshot_sha256,
                    action="hold",
                    rationale_summary=(
                        f"{expert_summary} control-mode confidence was below its bound."
                    ),
                    risk_notes=["LOCAL_POLICY_CONTROL_MODE_AMBIGUOUS", *advisory_notes],
                )
                return decision
            if best[1] != "pilot-control":
                decision = TextNavigationDecision(
                    snapshot_sha256=snapshot_sha256,
                    action=best[1],
                    rationale_summary=(f"{expert_summary} selected a bounded non-motion action."),
                    risk_notes=advisory_notes,
                )
                return decision
            decision = TextNavigationDecision(
                snapshot_sha256=snapshot_sha256,
                action="pilot-control",
                rationale_summary=(f"{expert_summary} issued bounded body-frame pilot control."),
                risk_notes=advisory_notes,
                controller_step_scale=raw.controller_step_scale,
                pilot_control=raw.pilot_control,
            )
            return decision
        candidate_options = [
            (score, "select-candidate", index) for index, score in enumerate(valid_candidates)
        ]
        action_names = ("hold", "request-new-scan", "abort")
        action_options = [
            (score, action, None)
            for score, action in zip(raw.action_scores, action_names, strict=True)
        ]
        options = sorted(
            [*candidate_options, *action_options],
            key=lambda item: item[0],
            reverse=True,
        )
        best = options[0]
        margin = best[0] - options[1][0] if len(options) > 1 else math.inf
        if margin < self.package.manifest.minimum_selection_margin:
            decision = TextNavigationDecision(
                snapshot_sha256=snapshot_sha256,
                action="hold",
                rationale_summary=(
                    f"{expert_summary} confidence margin was below its qualified bound."
                ),
                risk_notes=["LOCAL_POLICY_SELECTION_AMBIGUOUS", *advisory_notes],
            )
            return decision
        if best[1] != "select-candidate":
            decision = TextNavigationDecision(
                snapshot_sha256=snapshot_sha256,
                action=best[1],
                rationale_summary=(f"{expert_summary} selected a bounded non-motion action."),
                risk_notes=advisory_notes,
            )
            return decision
        candidate_index = best[2]
        if candidate_index is None or candidate_index >= len(batch.candidate_ids):
            raise ValueError("local policy selected an unavailable candidate")
        decision = TextNavigationDecision(
            snapshot_sha256=snapshot_sha256,
            action="select-candidate",
            selected_candidate_id=batch.candidate_ids[candidate_index],
            rationale_summary=(
                f"{expert_summary} selected a deterministically authored candidate."
            ),
            risk_notes=advisory_notes,
            controller_step_scale=raw.controller_step_scale,
            pilot_control=raw.pilot_control,
        )
        return decision
