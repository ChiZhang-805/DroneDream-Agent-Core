"""Content-bound local flight-policy packages and qualification selection.

Local neural policies are proposals inside the existing deterministic flight
envelope.  A package cannot grant actuator authority: it is selected only when
its files, vehicle/sensor bindings, optional map binding, and an independently
written qualification receipt all agree.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from .causal_control import CONTROL_HISTORY_CONTRACT_SHA256
from .contracts import StrictModel
from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from .hashing import sha256_json
from .local_expert_harness import NAVIGATION_EXPERT_ROLES
from .local_policy_quality import LocalPolicyTrainingMetrics, navigation_quality_issues
from .plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from .realtime_feature_encoders import (
    POLICY_REALTIME_FEATURE_COUNT,
    REALTIME_CONTROL_FEATURE_COUNT,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PolicyScope = Literal["general", "map-specialist"]
PolicyArtifactRole = Literal[
    "perception-encoder",
    "local-navigation-policy",
    "precision-maneuver-policy",
    "recovery-policy",
    "risk-critic",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
]

LOCAL_POLICY_STATE_FEATURE_COUNT = 46
LOCAL_POLICY_CANDIDATE_FEATURE_COUNT = 15
LOCAL_POLICY_MAXIMUM_CANDIDATES = 8
LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH = 8
LOCAL_POLICY_PAYLOAD_FEATURE_COUNT = 24
LOCAL_POLICY_MANEUVER_FEATURE_COUNT = 13
LOCAL_POLICY_SENSOR_FEATURE_COUNT = 64
LOCAL_POLICY_PILOT_CONTROL_AXIS_COUNT = 4
LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE = 0.2


class LocalPolicyArtifact(StrictModel):
    """One immutable inference artifact stored underneath a policy package."""

    role: PolicyArtifactRole
    relative_path: str = Field(min_length=1, max_length=240)
    sha256: Sha256
    format: Literal["onnx"] = "onnx"
    input_names: list[str] = Field(min_length=1, max_length=16)
    output_names: list[str] = Field(min_length=1, max_length=16)

    # 功能：
    #   规范反斜杠后验证跨平台包内路径，拒绝点段、设备名、数据流及不可见非法字符。
    # 输入：
    #   cls：模型产物字段类型。
    #   value：清单中的原始相对路径。
    # 输出：
    #   normalized：含义唯一的正斜杠相对路径。
    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        portable_plugin_path(normalized)
        return normalized

    # 功能：
    #   输入输出张量名在各自方向内必须非空且唯一，避免一项覆盖另一项接口。
    # 输入：
    #   self：已解析的模型产物声明。
    # 输出：
    #   self：通过张量名检查的产物声明。
    @model_validator(mode="after")
    def validate_tensor_names(self) -> LocalPolicyArtifact:
        if any(
            not name.strip() or len(name) > 240 or any(ord(char) < 32 for char in name)
            for name in (*self.input_names, *self.output_names)
        ):
            raise ValueError("policy tensor names must be nonempty bounded identifiers")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("policy artifact input names must be unique")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("policy artifact output names must be unique")
        return self


class LocalPolicyPackageManifest(StrictModel):
    """Stable package identity for a general or exact-map specialist policy."""

    package_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", min_length=3, max_length=120)
    display_name: str = Field(min_length=1, max_length=120)
    scope: PolicyScope
    base_package_sha256: Sha256 | None = None
    map_sha256: Sha256 | None = None
    vehicle_sha256: Sha256
    sensor_contract_sha256: Sha256
    state_feature_count: Literal[46] = LOCAL_POLICY_STATE_FEATURE_COUNT
    candidate_feature_count: Literal[15] = LOCAL_POLICY_CANDIDATE_FEATURE_COUNT
    maximum_candidates: Literal[8] = LOCAL_POLICY_MAXIMUM_CANDIDATES
    maximum_inference_latency_ms: int = Field(ge=1, le=2_000)
    minimum_selection_margin: float = Field(default=0.05, ge=0.0, le=20.0)
    risk_hold_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_width: int | None = Field(default=None, ge=32, le=1_024)
    visual_height: int | None = Field(default=None, ge=32, le=1_024)
    visual_feature_count: int | None = Field(default=None, ge=1, le=65_536)
    realtime_feature_count: int | None = Field(default=None, ge=1, le=2_048)
    control_feature_contract_sha256: Sha256 | None = None
    pilot_control_mode: Literal["normalized-body-velocity"] | None = None
    navigation_architecture: Literal["feedforward-control", "causal-gru-control"] = (
        "feedforward-control"
    )
    navigation_history_length: int | None = Field(default=None, ge=4, le=32)
    navigation_history_contract_sha256: Sha256 | None = None
    visual_normalization: Literal[
        "zero-to-one",
        "minus-one-to-one",
        "imagenet",
    ] = "zero-to-one"
    artifacts: list[LocalPolicyArtifact] = Field(min_length=1, max_length=12)

    # 功能：
    #   1. 验证通用或地图范围、唯一角色及路径，禁止不兼容历史权重只改声明后冒充当前控制器。
    #   2. 按候选或因果结构核对导航、独立风险、视觉和各顾问的专用张量及历史契约。
    # 输入：
    #   self：包的设备绑定、控制语义及完整产物清单。
    # 输出：
    #   self：静态声明一致的清单；实际图计算与飞行资格由后续独立验证。
    @model_validator(mode="after")
    def validate_scope_and_roles(self) -> LocalPolicyPackageManifest:
        if self.scope == "general":
            if self.map_sha256 is not None or self.base_package_sha256 is not None:
                raise ValueError("general policies cannot bind a map or a base policy")
        elif self.map_sha256 is None or self.base_package_sha256 is None:
            raise ValueError("map-specialist policies require map and base-policy hashes")
        roles = [artifact.role for artifact in self.artifacts]
        if len(set(roles)) != len(roles):
            raise ValueError("policy artifact roles must be unique")
        if len({artifact.relative_path.casefold() for artifact in self.artifacts}) != len(roles):
            raise ValueError("policy artifact paths must be unique across platforms")
        if "local-navigation-policy" not in roles:
            raise ValueError("a policy package requires one local-navigation policy")
        navigation = next(
            artifact for artifact in self.artifacts if artifact.role == "local-navigation-policy"
        )
        base_navigation_inputs = {
            "state_features",
            "candidate_features",
            "candidate_mask",
        }
        required_inputs = set(base_navigation_inputs)
        if self.navigation_architecture == "causal-gru-control":
            if self.pilot_control_mode is None or self.navigation_history_length is None:
                raise ValueError("causal navigation requires continuous axes and bounded history")
            if self.navigation_history_contract_sha256 != CONTROL_HISTORY_CONTRACT_SHA256:
                raise ValueError("causal navigation history semantics do not match")
            required_inputs = {"state_features", "control_history", "control_history_mask"}
            if {"candidate_features", "candidate_mask"} & set(navigation.input_names):
                raise ValueError("causal navigation cannot accept coordinate candidates")
        elif (
            self.navigation_history_length is not None
            or self.navigation_history_contract_sha256 is not None
        ):
            raise ValueError("feedforward architecture cannot advertise recurrent history")
        realtime_inputs = {"realtime_features", "realtime_valid_mask"}
        if self.realtime_feature_count is not None:
            required_inputs.update(realtime_inputs)
        elif realtime_inputs.intersection(navigation.input_names):
            raise ValueError("realtime policy inputs require a declared realtime feature count")
        required_outputs = {"candidate_scores", "action_scores", "risk_score"}
        if self.pilot_control_mode is not None:
            required_outputs.add("pilot_control")
        elif "pilot_control" in navigation.output_names:
            raise ValueError("pilot control output requires a declared pilot control mode")
        if not required_inputs.issubset(navigation.input_names):
            raise ValueError("local-navigation policy is missing a required input tensor")
        if self.realtime_feature_count not in {
            None,
            REALTIME_CONTROL_FEATURE_COUNT,
            POLICY_REALTIME_FEATURE_COUNT,
        }:
            raise ValueError("local-navigation realtime feature width is incompatible")
        if (
            self.pilot_control_mode is not None
            and self.realtime_feature_count != POLICY_REALTIME_FEATURE_COUNT
        ):
            raise ValueError(
                "pilot control requires sensor features plus an egocentric control reference"
            )
        if not required_outputs.issubset(navigation.output_names):
            raise ValueError("local-navigation policy is missing a required output tensor")
        for expert_role in NAVIGATION_EXPERT_ROLES[1:]:
            expert = next(
                (artifact for artifact in self.artifacts if artifact.role == expert_role),
                None,
            )
            if expert is not None and (
                set(expert.input_names) != set(navigation.input_names)
                or set(expert.output_names) != set(navigation.output_names)
            ):
                raise ValueError(f"{expert_role} tensor contract is incompatible")
        risk_critic = next(
            (artifact for artifact in self.artifacts if artifact.role == "risk-critic"),
            None,
        )
        if risk_critic is not None:
            accepted_risk_inputs = {frozenset(base_navigation_inputs)}
            if self.realtime_feature_count is not None:
                accepted_risk_inputs.add(frozenset({*base_navigation_inputs, *realtime_inputs}))
            if (self.visual_feature_count or 0) > 0:
                accepted_risk_inputs.add(frozenset({*base_navigation_inputs, "visual_features"}))
                if self.realtime_feature_count is not None:
                    accepted_risk_inputs.add(
                        frozenset(
                            {
                                *base_navigation_inputs,
                                *realtime_inputs,
                                "visual_features",
                            }
                        )
                    )
            if self.pilot_control_mode is not None:
                # Candidate/state-only critics cannot assess a continuous
                # action. No implicit compatibility with legacy risk heads.
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
            if frozenset(risk_critic.input_names) not in accepted_risk_inputs or set(
                risk_critic.output_names
            ) != {"risk_score"}:
                raise ValueError("risk critic tensor contract is incompatible")
        perception_health_critic = next(
            (
                artifact
                for artifact in self.artifacts
                if artifact.role == "perception-health-critic"
            ),
            None,
        )
        if perception_health_critic is not None and (
            set(perception_health_critic.input_names) != {"state_features"}
            or set(perception_health_critic.output_names) != {"risk_score"}
        ):
            raise ValueError("perception health critic tensor contract is incompatible")
        settle_stability_critic = next(
            (artifact for artifact in self.artifacts if artifact.role == "settle-stability-critic"),
            None,
        )
        if settle_stability_critic is not None and (
            set(settle_stability_critic.input_names)
            != {"maneuver_features", "state_history", "history_mask"}
            or set(settle_stability_critic.output_names) != {"risk_score"}
        ):
            raise ValueError("settle stability critic tensor contract is incompatible")
        anomaly_detector = next(
            (artifact for artifact in self.artifacts if artifact.role == "state-anomaly-detector"),
            None,
        )
        if anomaly_detector is not None and (
            set(anomaly_detector.input_names) != {"state_history", "history_mask"}
            or set(anomaly_detector.output_names) != {"anomaly_score"}
        ):
            raise ValueError("state anomaly detector tensor contract is incompatible")
        payload_adapter = next(
            (
                artifact
                for artifact in self.artifacts
                if artifact.role == "payload-dynamics-adapter"
            ),
            None,
        )
        if payload_adapter is not None:
            accepted_payload_inputs = {
                frozenset({"payload_features", "state_history", "history_mask"}),
                frozenset({"payload_features", "payload_history", "history_mask"}),
            }
            if frozenset(payload_adapter.input_names) not in accepted_payload_inputs or set(
                payload_adapter.output_names
            ) != {"risk_score", "controller_step_scale"}:
                raise ValueError("payload dynamics adapter tensor contract is incompatible")
        cross_modal_critic = next(
            (
                artifact
                for artifact in self.artifacts
                if artifact.role == "cross-modal-consistency-critic"
            ),
            None,
        )
        if cross_modal_critic is not None and (
            set(cross_modal_critic.input_names) != {"sensor_features"}
            or set(cross_modal_critic.output_names) != {"risk_score"}
        ):
            raise ValueError("cross-modal consistency critic tensor contract is incompatible")
        perception = next(
            (artifact for artifact in self.artifacts if artifact.role == "perception-encoder"),
            None,
        )
        visual_dimensions = (
            self.visual_width,
            self.visual_height,
            self.visual_feature_count,
        )
        if perception is None:
            if any(value is not None for value in visual_dimensions):
                raise ValueError("visual dimensions require a perception encoder")
            if "visual_features" in navigation.input_names:
                raise ValueError("visual policy input requires a perception encoder")
        else:
            if any(value is None for value in visual_dimensions):
                raise ValueError("perception encoder requires complete visual dimensions")
            if set(perception.input_names) != {"forward_rgb"}:
                raise ValueError("perception encoder input must be forward_rgb")
            if "visual_features" not in perception.output_names:
                raise ValueError("perception encoder must output visual_features")
            if "visual_features" not in navigation.input_names:
                raise ValueError("visual policy is missing the encoder feature input")
        return self


class LocalPolicyQualificationReceipt(StrictModel):
    """Independent simulation evidence required before a policy can be selected."""

    control_feature_contract_sha256: Sha256 | None = None
    receipt_id: str = Field(pattern=r"^policy-qualification-[0-9a-f]{32}$")
    policy_package_sha256: Sha256
    navigation_expert_metrics: dict[str, LocalPolicyTrainingMetrics] = Field(
        default_factory=dict, max_length=3
    )
    evaluation_suite_sha256: Sha256
    map_sha256: Sha256 | None = None
    vehicle_sha256: Sha256
    sensor_contract_sha256: Sha256
    trial_count: int = Field(ge=1, le=1_000_000)
    successful_trial_count: int = Field(ge=0, le=1_000_000)
    collision_count: int = Field(ge=0, le=1_000_000)
    deterministic_safety_intervention_count: int = Field(ge=0, le=1_000_000)
    p50_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)
    p95_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)
    p99_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)
    expert_expected_invocation_counts: dict[PolicyArtifactRole, int] = Field(default_factory=dict)
    expert_observed_invocation_counts: dict[PolicyArtifactRole, int] = Field(default_factory=dict)
    local_policy_call_count: int | None = Field(default=None, ge=0, le=100_000_000)
    realtime_feature_ready_call_count: int | None = Field(default=None, ge=0, le=100_000_000)
    model_motion_decision_count: int | None = Field(default=None, ge=0, le=100_000_000)
    pilot_control_command_count: int | None = Field(default=None, ge=0, le=100_000_000)
    pilot_control_mean_absolute_error: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )
    qualified: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=64)

    # 功能：
    #   核对试验计数、延迟分位、实际专家调用和连续控制证据，合格声明不能覆盖失败指标。
    # 输入：
    #   self：独立生成的资格回执数值与状态。
    # 输出：
    #   self：满足计数一致性及其资格声明门槛的回执。
    @model_validator(mode="after")
    def validate_metrics(self) -> LocalPolicyQualificationReceipt:
        if self.successful_trial_count > self.trial_count:
            raise ValueError("successful policy trials cannot exceed total trials")
        if self.collision_count > self.trial_count:
            raise ValueError("policy collision count cannot exceed total trials")
        if not (
            self.p50_inference_latency_ms
            <= self.p95_inference_latency_ms
            <= self.p99_inference_latency_ms
        ):
            raise ValueError("policy latency percentiles must be monotonic")
        if self.qualified and (
            self.collision_count != 0
            or self.successful_trial_count / self.trial_count < 0.95
            or self.issue_codes
        ):
            raise ValueError("qualified policy evidence does not satisfy hard gates")
        if any(
            value < 0
            for value in (
                *self.expert_expected_invocation_counts.values(),
                *self.expert_observed_invocation_counts.values(),
            )
        ):
            raise ValueError("expert invocation counts cannot be negative")
        if self.qualified and (
            self.expert_expected_invocation_counts != self.expert_observed_invocation_counts
        ):
            raise ValueError("qualified policy evidence has unmatched expert invocations")
        continuous_fields = (
            self.local_policy_call_count,
            self.realtime_feature_ready_call_count,
            self.model_motion_decision_count,
            self.pilot_control_command_count,
            self.pilot_control_mean_absolute_error,
        )
        if any(value is not None for value in continuous_fields) and not all(
            value is not None for value in continuous_fields
        ):
            raise ValueError("continuous-control qualification evidence is incomplete")
        if self.local_policy_call_count is not None:
            assert self.realtime_feature_ready_call_count is not None
            assert self.model_motion_decision_count is not None
            assert self.pilot_control_command_count is not None
            assert self.pilot_control_mean_absolute_error is not None
            if self.realtime_feature_ready_call_count > self.local_policy_call_count:
                raise ValueError("ready realtime calls cannot exceed local-policy calls")
            if self.pilot_control_command_count > self.model_motion_decision_count:
                raise ValueError("pilot-control commands cannot exceed motion decisions")
            if self.model_motion_decision_count > self.local_policy_call_count:
                raise ValueError("motion decisions cannot exceed local-policy calls")
            if self.qualified and (
                self.local_policy_call_count == 0
                or self.model_motion_decision_count == 0
                or self.realtime_feature_ready_call_count != self.local_policy_call_count
                or self.pilot_control_command_count != self.model_motion_decision_count
                or self.pilot_control_mean_absolute_error > LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE
            ):
                raise ValueError(
                    "qualified continuous-control evidence does not satisfy hard gates"
                )
        return self


class LocalPolicyAdvisorAdmissionMetrics(StrictModel):
    """Independent held-out metrics for one packaged advisory expert."""

    sample_count: int = Field(ge=1, le=10_000_000)
    risky_sample_count: int = Field(ge=0, le=10_000_000)
    safe_sample_count: int = Field(ge=0, le=10_000_000)
    risk_hold_recall: float = Field(ge=0.0, le=1.0)
    safe_motion_recall: float = Field(ge=0.0, le=1.0)
    risk_mean_absolute_error: float = Field(ge=0.0, le=1.0)
    controller_step_scale_mean_absolute_error: float | None = Field(default=None, ge=0.0, le=0.9)
    p99_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)

    # 功能：
    #   确保顾问风险双类支持总量与被评估的样本总数一致。
    # 输入：
    #   self：单个顾问的留出指标。
    # 输出：
    #   self：类别数量一致的指标对象。
    @model_validator(mode="after")
    def validate_counts(self) -> LocalPolicyAdvisorAdmissionMetrics:
        if self.risky_sample_count + self.safe_sample_count != self.sample_count:
            raise ValueError("advisor admission class counts do not sum to sample count")
        return self


class LocalPolicySimulationAdmissionReceipt(StrictModel):
    """Offline evidence that permits an unqualified package in simulation only."""

    control_feature_contract_sha256: Sha256 | None = None
    receipt_id: str = Field(pattern=r"^policy-simulation-admission-[0-9a-f]{32}$")
    policy_package_sha256: Sha256
    navigation_expert_metrics: dict[str, LocalPolicyTrainingMetrics] = Field(
        default_factory=dict, max_length=3
    )
    dataset_receipt_sha256: Sha256
    evaluation_suite_sha256: Sha256
    map_sha256: Sha256 | None = None
    vehicle_sha256: Sha256
    sensor_contract_sha256: Sha256
    sample_count: int = Field(ge=1, le=10_000_000)
    motion_sample_count: int = Field(ge=0, le=10_000_000)
    non_motion_sample_count: int = Field(ge=0, le=10_000_000)
    motion_authorization_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_selection_accuracy: float = Field(ge=0.0, le=1.0)
    risk_hold_recall: float = Field(ge=0.0, le=1.0)
    safe_motion_recall: float = Field(ge=0.0, le=1.0)
    risk_mean_absolute_error: float | None = Field(default=None, ge=0.0, le=1.0)
    realtime_feature_ready_sample_count: int | None = Field(default=None, ge=0, le=10_000_000)
    pilot_control_target_sample_count: int | None = Field(default=None, ge=0, le=10_000_000)
    pilot_control_mean_absolute_error: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )
    p50_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)
    p95_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)
    p99_inference_latency_ms: float = Field(ge=0.0, le=60_000.0)
    visual_encoding_receipt_sha256: Sha256 | None = None
    perception_encoder_sha256: Sha256 | None = None
    visual_preprocess_p99_latency_ms: float | None = Field(default=None, ge=0.0, le=60_000.0)
    visual_encoder_p99_inference_latency_ms: float | None = Field(default=None, ge=0.0, le=60_000.0)
    advisor_evaluation_sha256: Sha256 | None = None
    advisor_metrics: dict[PolicyArtifactRole, LocalPolicyAdvisorAdmissionMetrics] = Field(
        default_factory=dict
    )
    combined_p99_inference_latency_ms: float | None = Field(default=None, ge=0.0, le=60_000.0)
    artifact_preservation_receipt_sha256: Sha256 | None = None
    inherited_admission_receipt_sha256: Sha256 | None = None
    navigation_training_dataset_receipt_sha256: Sha256 | None = None
    navigation_admission_split: Literal["training", "validation"] | None = None
    navigation_admission_source_evidence_sha256: list[Sha256] = Field(
        default_factory=list, max_length=1_024
    )
    admission_scope: Literal[
        "standard-simulation",
        "recovery-bootstrap-simulation",
    ] = "standard-simulation"
    minimum_motion_sample_count: int = Field(default=20, ge=1, le=10_000_000)
    minimum_non_motion_sample_count: int = Field(default=20, ge=1, le=10_000_000)
    admitted_to_simulation: bool
    issue_codes: list[str] = Field(default_factory=list, max_length=64)

    # 功能：
    #   1. 验证离线样本、视觉和继承来源证据完整，标准与恢复引导通道使用各自固定门槛。
    #   2. 准入声明必须满足质量和连续控制覆盖要求，不能据此取得正式飞行资格。
    # 输入：
    #   self：完整仿真准入回执。
    # 输出：
    #   self：数值、来源字段及准入声明一致的回执。
    @model_validator(mode="after")
    def validate_admission_metrics(self) -> LocalPolicySimulationAdmissionReceipt:
        if self.motion_sample_count + self.non_motion_sample_count != self.sample_count:
            raise ValueError("simulation admission sample classes do not sum to total")
        continuous_fields = (
            self.realtime_feature_ready_sample_count,
            self.pilot_control_target_sample_count,
            self.pilot_control_mean_absolute_error,
        )
        if any(value is not None for value in continuous_fields) and not all(
            value is not None for value in continuous_fields
        ):
            raise ValueError("continuous-control simulation evidence is incomplete")
        if self.realtime_feature_ready_sample_count is not None:
            assert self.pilot_control_target_sample_count is not None
            assert self.pilot_control_mean_absolute_error is not None
            if self.realtime_feature_ready_sample_count > self.sample_count:
                raise ValueError("ready realtime samples cannot exceed total samples")
            if self.pilot_control_target_sample_count > self.sample_count:
                raise ValueError("pilot-control targets cannot exceed total samples")
        if not (
            self.p50_inference_latency_ms
            <= self.p95_inference_latency_ms
            <= self.p99_inference_latency_ms
        ):
            raise ValueError("simulation admission latency percentiles must be monotonic")
        if (
            self.combined_p99_inference_latency_ms is not None
            and self.combined_p99_inference_latency_ms < self.p99_inference_latency_ms
        ):
            raise ValueError("combined inference latency cannot be below navigation latency")
        visual_fields = (
            self.visual_encoding_receipt_sha256,
            self.perception_encoder_sha256,
            self.visual_preprocess_p99_latency_ms,
            self.visual_encoder_p99_inference_latency_ms,
        )
        if any(item is not None for item in visual_fields) and not all(
            item is not None for item in visual_fields
        ):
            raise ValueError("visual simulation admission evidence is incomplete")
        navigation_evidence = (
            self.navigation_training_dataset_receipt_sha256,
            self.navigation_admission_split,
        )
        if any(item is not None for item in navigation_evidence) != all(
            item is not None for item in navigation_evidence
        ):
            raise ValueError("navigation admission provenance is incomplete")
        if (
            self.navigation_admission_split is not None
            and not self.navigation_admission_source_evidence_sha256
        ):
            raise ValueError("navigation admission has no source evidence")
        if (
            self.navigation_admission_split is None
            and self.navigation_admission_source_evidence_sha256
        ):
            raise ValueError("navigation source evidence has no admission split")
        preservation_evidence = (
            self.artifact_preservation_receipt_sha256,
            self.inherited_admission_receipt_sha256,
        )
        if any(item is not None for item in preservation_evidence) != all(
            item is not None for item in preservation_evidence
        ):
            raise ValueError("inherited artifact admission evidence is incomplete")
        if self.admission_scope == "standard-simulation" and (
            self.minimum_motion_sample_count != 20 or self.minimum_non_motion_sample_count != 20
        ):
            raise ValueError("standard simulation admission thresholds are immutable")
        if self.admission_scope == "recovery-bootstrap-simulation":
            if self.minimum_motion_sample_count != 20 or self.minimum_non_motion_sample_count != 3:
                raise ValueError("recovery bootstrap admission thresholds are immutable")
            if self.navigation_training_dataset_receipt_sha256 is None:
                raise ValueError("recovery bootstrap admission requires training provenance")
        if self.admitted_to_simulation and (
            self.motion_sample_count < self.minimum_motion_sample_count
            or self.non_motion_sample_count < self.minimum_non_motion_sample_count
            or self.motion_authorization_accuracy < 0.95
            or (
                self.realtime_feature_ready_sample_count is None
                and self.candidate_selection_accuracy < 0.9
            )
            or self.risk_hold_recall < 0.95
            or self.safe_motion_recall < 0.95
            or (self.risk_mean_absolute_error is not None and self.risk_mean_absolute_error > 0.2)
            or (
                self.realtime_feature_ready_sample_count is not None
                and (
                    self.realtime_feature_ready_sample_count != self.sample_count
                    or self.pilot_control_target_sample_count != self.sample_count
                    or self.pilot_control_mean_absolute_error
                    > LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE
                )
            )
            or self.issue_codes
        ):
            raise ValueError("simulation admission does not satisfy offline hard gates")
        return self


@dataclass(frozen=True)
class LoadedLocalPolicyPackage:
    root: Path
    manifest_path: Path
    manifest: LocalPolicyPackageManifest
    package_sha256: str
    artifact_paths: dict[PolicyArtifactRole, Path]


class LocalPolicySelection(StrictModel):
    package_id: str
    package_sha256: Sha256
    scope: PolicyScope
    qualification_receipt_id: str
    selection_reason: Literal[
        "exact-map-specialist-qualified",
        "general-qualified",
        "exact-map-specialist-simulation-admitted",
        "general-simulation-admitted",
    ]
    simulation_only: bool = False
    rollback_package_sha256: Sha256 | None = None


# 功能：
#   有界流式读取普通模型文件，检查读取前后身份、大小和修改时间，不把路径别名当成已验模型。
# 输入：
#   path：包内模型路径。
# 输出：
#   digest：至多 2 GiB 的同次读取内容摘要。
def _file_sha256(path: Path) -> str:
    digest = hash_plugin_file(path, limit=2 * 1024 * 1024 * 1024)
    return digest


# 功能：
#   1. 严格有界读取清单，先查路径再规范化，逐份检查模型摘要及清单读取期间是否变化。
#   2. 包身份绑定实际发布字段，不因新增 schema 默认项而漂移；本函数不执行图或授予资格。
# 输入：
#   root：已存在的普通包目录。
# 输出：
#   package：静态内容检查通过的包及角色路径。
def load_local_policy_package(root: Path) -> LoadedLocalPolicyPackage:
    check_plain_plugin_path(root)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("local policy package root must be a directory")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("local policy package manifest is unavailable")
    manifest_content = read_plugin_file(manifest_path, limit=2 * 1024 * 1024)
    manifest_payload = decode_json(manifest_content, limit=2 * 1024 * 1024, node_limit=100_000)
    if not isinstance(manifest_payload, dict):
        raise ValueError("local policy package manifest must be an object")
    manifest = LocalPolicyPackageManifest.model_validate(manifest_payload)
    artifact_paths: dict[PolicyArtifactRole, Path] = {}
    package_entries: list[dict[str, str]] = []
    for artifact in manifest.artifacts:
        path = root / Path(*PurePosixPath(artifact.relative_path).parts)
        check_plain_plugin_path(path)
        path = path.resolve(strict=True)
        if path == root or root not in path.parents or not path.is_file():
            raise ValueError("policy artifact escapes its package root")
        size = path.stat().st_size
        if size <= 0 or size > 2 * 1024 * 1024 * 1024:
            raise ValueError("policy artifact has an invalid bounded size")
        observed_sha256 = _file_sha256(path)
        if observed_sha256 != artifact.sha256:
            raise ValueError(f"policy artifact hash mismatch: {artifact.relative_path}")
        artifact_paths[artifact.role] = path
        package_entries.append(
            {
                "role": artifact.role,
                "relative_path": artifact.relative_path,
                "sha256": observed_sha256,
            }
        )
    if read_plugin_file(manifest_path, limit=2 * 1024 * 1024) != manifest_content:
        raise ValueError("policy manifest changed while loading artifacts")
    package_sha256 = sha256_json(
        {
            # Bind the exact validated manifest fields that were actually
            # published. Adding an optional schema field later must not mutate
            # the identity of an immutable historical package.
            "manifest": manifest_payload,
            "artifacts": sorted(package_entries, key=lambda item: item["role"]),
        }
    )
    package = LoadedLocalPolicyPackage(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        package_sha256=package_sha256,
        artifact_paths=artifact_paths,
    )
    return package


# 功能：
#   重验回执及清单，连续控制还必须绑定当前特征语义、全部导航专家质量和四轴覆盖证据。
# 输入：
#   package：静态加载的包；历史格式可读不等于支持当前控制。
#   receipt：资格或仿真准入回执。
# 输出：
#   supported：上述控制契约证据是否匹配，不独立代表可选择或可执行。
def local_policy_receipt_supports_control_contract(
    package: LoadedLocalPolicyPackage,
    receipt: LocalPolicyQualificationReceipt | LocalPolicySimulationAdmissionReceipt,
) -> bool:
    supported = False
    if not isinstance(
        receipt, (LocalPolicyQualificationReceipt, LocalPolicySimulationAdmissionReceipt)
    ):
        return supported
    try:
        manifest = LocalPolicyPackageManifest.model_validate(package.manifest.model_dump())
        receipt = type(receipt).model_validate(receipt.model_dump())
    except ValidationError:
        return supported
    if manifest.pilot_control_mode is None:
        supported = True
        return supported
    if manifest.control_feature_contract_sha256 != CURRENT_POLICY_FEATURE_CONTRACT_SHA256:
        return supported
    if receipt.control_feature_contract_sha256 != manifest.control_feature_contract_sha256:
        return supported
    expected_roles = set(NAVIGATION_EXPERT_ROLES) & {item.role for item in manifest.artifacts}
    if set(receipt.navigation_expert_metrics) != expected_roles or any(
        navigation_quality_issues(metrics, continuous_control=True)
        for metrics in receipt.navigation_expert_metrics.values()
    ):
        return supported
    if isinstance(receipt, LocalPolicySimulationAdmissionReceipt):
        supported = bool(
            receipt.realtime_feature_ready_sample_count == receipt.sample_count
            and receipt.pilot_control_target_sample_count == receipt.sample_count
            and receipt.pilot_control_mean_absolute_error is not None
            and receipt.pilot_control_mean_absolute_error <= LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE
        )
    else:
        supported = bool(
            receipt.local_policy_call_count is not None
            and receipt.local_policy_call_count > 0
            and receipt.realtime_feature_ready_call_count == receipt.local_policy_call_count
            and receipt.model_motion_decision_count is not None
            and receipt.model_motion_decision_count > 0
            and receipt.pilot_control_command_count == receipt.model_motion_decision_count
            and receipt.pilot_control_mean_absolute_error is not None
            and receipt.pilot_control_mean_absolute_error <= LOCAL_POLICY_MAXIMUM_PILOT_CONTROL_MAE
        )
    return supported


# 功能：
#   选择与包身份、设备、地图及控制契约一致且延迟达标的正式回执，必须包含独立风险角色。
# 输入：
#   package：候选包。
#   receipts：可能包含其他包证据的资格回执列表。
# 输出：
#   selected：最快匹配回执，无合格证据时为空。
def _qualified_receipt(
    package: LoadedLocalPolicyPackage,
    receipts: list[LocalPolicyQualificationReceipt],
) -> LocalPolicyQualificationReceipt | None:
    selected = None
    if "risk-critic" not in package.artifact_paths:
        return selected
    matching = [
        receipt
        for receipt in receipts
        if receipt.policy_package_sha256 == package.package_sha256
        and receipt.vehicle_sha256 == package.manifest.vehicle_sha256
        and receipt.sensor_contract_sha256 == package.manifest.sensor_contract_sha256
        and receipt.map_sha256 == package.manifest.map_sha256
        and receipt.qualified
        and receipt.p99_inference_latency_ms <= package.manifest.maximum_inference_latency_ms
        and local_policy_receipt_supports_control_contract(package, receipt)
    ]
    if matching:
        selected = min(matching, key=lambda receipt: receipt.p99_inference_latency_ms)
    return selected


# 功能：
#   合并回执中的导航、视觉及已报告端到端延迟，选择时不能只看最快的导航头。
# 输入：
#   receipt：已解析的仿真准入回执。
# 输出：
#   latency：用于准入门槛和排序的保守毫秒延迟。
def _admission_latency(receipt: LocalPolicySimulationAdmissionReceipt) -> float:
    visual_latency = (receipt.visual_preprocess_p99_latency_ms or 0.0) + (
        receipt.visual_encoder_p99_inference_latency_ms or 0.0
    )
    latency = max(
        receipt.p99_inference_latency_ms + visual_latency,
        receipt.combined_p99_inference_latency_ms or 0.0,
    )
    return latency


# 功能：
#   查找身份、设备、地图、控制证据及完整延迟满足条件的仿真回执，不把它当正式飞行资格。
# 输入：
#   package：待选择的模型包。
#   receipts：已有离线准入回执。
# 输出：
#   selected：全链路延迟最小的匹配回执，无符合项时为空。
def _simulation_admission(
    package: LoadedLocalPolicyPackage,
    receipts: list[LocalPolicySimulationAdmissionReceipt],
) -> LocalPolicySimulationAdmissionReceipt | None:
    matching = [
        receipt
        for receipt in receipts
        if receipt.policy_package_sha256 == package.package_sha256
        and receipt.vehicle_sha256 == package.manifest.vehicle_sha256
        and receipt.sensor_contract_sha256 == package.manifest.sensor_contract_sha256
        and receipt.map_sha256 == package.manifest.map_sha256
        and receipt.admitted_to_simulation
        and local_policy_receipt_supports_control_contract(package, receipt)
        and _admission_latency(receipt) <= package.manifest.maximum_inference_latency_ms
    ]
    selected = min(matching, key=_admission_latency) if matching else None
    return selected


# 功能：
#   回退基座不仅要自身有资格，也必须与当前专家的设备、连续控制模式和时序语义兼容。
# 输入：
#   base：专家声明的通用回退包。
#   specialist：将使用该回退的地图专家。
# 输出：
#   compatible：是否能在同一设备和控制接口下替换，具体资格仍另行核验。
def _rollback_compatible(
    base: LoadedLocalPolicyPackage, specialist: LoadedLocalPolicyPackage
) -> bool:
    fields = (
        "vehicle_sha256",
        "sensor_contract_sha256",
        "pilot_control_mode",
        "realtime_feature_count",
        "control_feature_contract_sha256",
        "navigation_architecture",
        "navigation_history_length",
        "navigation_history_contract_sha256",
    )
    compatible = base.manifest.scope == "general" and all(
        getattr(base.manifest, field) == getattr(specialist.manifest, field) for field in fields
    )
    return compatible


# 功能：
#   选择前重读当前模型字节和清单，拒绝加载后身份变化或内存字段被改写，固定本次候选列表。
# 输入：
#   packages：先前加载的模型包列表；此检查发生于选择阶段，不放入逐帧控制热路径。
# 输出：
#   current_packages：重新核验的独立包对象，不允许重复内容身份。
def _current_package_inventory(
    packages: list[LoadedLocalPolicyPackage],
) -> list[LoadedLocalPolicyPackage]:
    current_packages = []
    seen = set()
    for original in tuple(packages):
        current = load_local_policy_package(original.root)
        if (
            current.package_sha256 != original.package_sha256
            or current.manifest != original.manifest
            or current.manifest_path != original.manifest_path
            or current.artifact_paths != original.artifact_paths
        ):
            raise ValueError("loaded policy package identity changed before selection")
        if current.package_sha256 in seen:
            raise ValueError("duplicate local policy package content")
        seen.add(current.package_sha256)
        current_packages.append(current)
    return current_packages


# 功能：
#   从当前设备绑定的合格包中优先选择准确地图专家，并要求兼容且合格的通用回退。
# 输入：
#   packages：内容身份不重复的已加载包。
#   receipts：独立资格证据。
#   map_sha256：当前地图身份。
#   vehicle_sha256：当前载具身份。
#   sensor_contract_sha256：当前传感器契约。
#   prefer_specialist：是否优先地图专用包，否则仅选择通用包。
# 输出：
#   selection：明确包、回执及可选回退的选择，不直接授予执行器权限。
def select_local_policy(
    *,
    packages: list[LoadedLocalPolicyPackage],
    receipts: list[LocalPolicyQualificationReceipt],
    map_sha256: str,
    vehicle_sha256: str,
    sensor_contract_sha256: str,
    prefer_specialist: bool = True,
) -> LocalPolicySelection:
    if not re.fullmatch(r"[0-9a-f]{64}", map_sha256):
        raise ValueError("active map hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", vehicle_sha256):
        raise ValueError("active vehicle hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", sensor_contract_sha256):
        raise ValueError("active sensor contract hash is invalid")
    packages = _current_package_inventory(packages)
    package_by_sha256 = {package.package_sha256: package for package in packages}

    eligible_general: list[tuple[LoadedLocalPolicyPackage, LocalPolicyQualificationReceipt]] = []
    eligible_specialist: list[
        tuple[LoadedLocalPolicyPackage, LocalPolicyQualificationReceipt, LoadedLocalPolicyPackage]
    ] = []
    for package in packages:
        manifest = package.manifest
        if (
            manifest.vehicle_sha256 != vehicle_sha256
            or manifest.sensor_contract_sha256 != sensor_contract_sha256
            or "risk-critic" not in package.artifact_paths
        ):
            continue
        receipt = _qualified_receipt(package, receipts)
        if receipt is None:
            continue
        if manifest.scope == "general":
            eligible_general.append((package, receipt))
            continue
        if manifest.map_sha256 != map_sha256 or manifest.base_package_sha256 is None:
            continue
        base = package_by_sha256.get(manifest.base_package_sha256)
        if (
            base is None
            or not _rollback_compatible(base, package)
            or _qualified_receipt(base, receipts) is None
        ):
            continue
        eligible_specialist.append((package, receipt, base))

    if prefer_specialist and eligible_specialist:
        package, receipt, base = min(
            eligible_specialist,
            key=lambda item: (item[1].p99_inference_latency_ms, item[0].manifest.package_id),
        )
        selection = LocalPolicySelection(
            package_id=package.manifest.package_id,
            package_sha256=package.package_sha256,
            scope=package.manifest.scope,
            qualification_receipt_id=receipt.receipt_id,
            selection_reason="exact-map-specialist-qualified",
            rollback_package_sha256=base.package_sha256,
        )
        return selection
    if eligible_general:
        package, receipt = min(
            eligible_general,
            key=lambda item: (item[1].p99_inference_latency_ms, item[0].manifest.package_id),
        )
        selection = LocalPolicySelection(
            package_id=package.manifest.package_id,
            package_sha256=package.package_sha256,
            scope=package.manifest.scope,
            qualification_receipt_id=receipt.receipt_id,
            selection_reason="general-qualified",
        )
        return selection
    raise ValueError("NO_QUALIFIED_LOCAL_POLICY_FOR_ACTIVE_BINDINGS")


# 功能：
#   按完整延迟和设备绑定选择仅限仿真的包，地图专家的兼容回退可为正式合格或仿真准入。
# 输入：
#   packages：内容身份不重复的已加载包。
#   admissions：离线仿真准入证据。
#   qualification_receipts：可用于通用回退的正式资格回执。
#   map_sha256：当前仿真地图摘要。
#   vehicle_sha256：当前载具摘要。
#   sensor_contract_sha256：当前传感器契约摘要。
#   prefer_specialist：是否优先专用包，否则仅选择通用包。
# 输出：
#   selection：强制标记 simulation_only 的选择及其回退，不代表真机资格。
def select_local_policy_for_simulation(
    *,
    packages: list[LoadedLocalPolicyPackage],
    admissions: list[LocalPolicySimulationAdmissionReceipt],
    qualification_receipts: list[LocalPolicyQualificationReceipt],
    map_sha256: str,
    vehicle_sha256: str,
    sensor_contract_sha256: str,
    prefer_specialist: bool = True,
) -> LocalPolicySelection:
    if not re.fullmatch(r"[0-9a-f]{64}", map_sha256):
        raise ValueError("active map hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", vehicle_sha256):
        raise ValueError("active vehicle hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", sensor_contract_sha256):
        raise ValueError("active sensor contract hash is invalid")
    packages = _current_package_inventory(packages)
    package_by_sha256 = {package.package_sha256: package for package in packages}
    general: list[tuple[LoadedLocalPolicyPackage, LocalPolicySimulationAdmissionReceipt]] = []
    specialist: list[
        tuple[
            LoadedLocalPolicyPackage,
            LocalPolicySimulationAdmissionReceipt,
            LoadedLocalPolicyPackage,
        ]
    ] = []
    for package in packages:
        manifest = package.manifest
        if (
            manifest.vehicle_sha256 != vehicle_sha256
            or manifest.sensor_contract_sha256 != sensor_contract_sha256
            or (manifest.pilot_control_mode is None and "risk-critic" not in package.artifact_paths)
        ):
            continue
        admission = _simulation_admission(package, admissions)
        if admission is None:
            continue
        if manifest.scope == "general":
            general.append((package, admission))
            continue
        if manifest.map_sha256 != map_sha256 or manifest.base_package_sha256 is None:
            continue
        base = package_by_sha256.get(manifest.base_package_sha256)
        if (
            base is None
            or not _rollback_compatible(base, package)
            or (
                _qualified_receipt(base, qualification_receipts) is None
                and _simulation_admission(base, admissions) is None
            )
        ):
            continue
        specialist.append((package, admission, base))
    if prefer_specialist and specialist:
        package, admission, base = min(
            specialist,
            key=lambda item: (_admission_latency(item[1]), item[0].manifest.package_id),
        )
        selection = LocalPolicySelection(
            package_id=package.manifest.package_id,
            package_sha256=package.package_sha256,
            scope=package.manifest.scope,
            qualification_receipt_id=admission.receipt_id,
            selection_reason="exact-map-specialist-simulation-admitted",
            simulation_only=True,
            rollback_package_sha256=base.package_sha256,
        )
        return selection
    if general:
        package, admission = min(
            general,
            key=lambda item: (_admission_latency(item[1]), item[0].manifest.package_id),
        )
        selection = LocalPolicySelection(
            package_id=package.manifest.package_id,
            package_sha256=package.package_sha256,
            scope=package.manifest.scope,
            qualification_receipt_id=admission.receipt_id,
            selection_reason="general-simulation-admitted",
            simulation_only=True,
        )
        return selection
    raise ValueError("NO_SIMULATION_ADMITTED_LOCAL_POLICY_FOR_ACTIVE_BINDINGS")


# 功能：
#   按最近秩计算延迟百分位，拒绝空样本、布尔、非有限或不能表示的数值。
# 输入：
#   values_ms：非负有限的毫秒样本。
#   percentile：零至一百的百分位，零对应最小样本。
# 输出：
#   latency：排序后对应秩的观测延迟。
def latency_percentile(values_ms: list[float], percentile: float) -> float:
    if not values_ms:
        raise ValueError("latency evidence cannot be empty")
    if type(percentile) not in (int, float) or not 0.0 <= percentile <= 100.0:
        raise ValueError("latency percentile must be within [0, 100]")
    try:
        invalid = any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0.0
            for value in values_ms
        )
    except (OverflowError, TypeError) as error:
        raise ValueError("latency evidence must be finite and non-negative") from error
    if invalid:
        raise ValueError("latency evidence must be finite and non-negative")
    ordered = sorted(values_ms)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    latency = ordered[rank - 1]
    return latency
