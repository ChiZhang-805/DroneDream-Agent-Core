"""Simulation-only learner port inside the ordinary bounded control coordinator."""

import base64
import time
from datetime import UTC, datetime
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json

from ..contracts import (
    ModelCallRecord,
    NormalizedPilotControl,
    SimulationTrainingTrace,
    TextNavigationDecision,
)
from ..hashing import sha256_json
from ..model_harness.model_port import ProviderSettings, StructuredCallResult
from ..realtime_feature_encoders import RealtimeFeatureSnapshot
from ..simulation_teacher import teacher_input_deadline
from .observations import compile_training_observation
from .policy_exchange import MAX_PACKET_BYTES, TrainingPolicyClient, _clock_seconds


class SimulationTrainingPolicyPort:
    """Adapt an offline learner proposal to the usual safety-gated runtime decision."""

    simulation_only = True
    supports_provider_context = False
    invocation_timeout_seconds = .25

    # 功能：
    #   连接明确选择的本地仿真学习器，声明训练专用提供方，不连接云端模型。
    # 输入：
    #   self：待初始化的训练策略入口。
    #   descriptor：本地训练运行描述文件。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, descriptor: Path):
        self.client = TrainingPolicyClient(descriptor)
        self.settings = ProviderSettings(name="simulation-training", model="offline-learner",
                                         api_key_env="", base_url=None,
                                         api_style="chat-completions", supports_image_input=True)

    # 功能：
    #   保持原训练运行绑定；协调器请求重置时不擅自更换学习器或重放旧动作。
    # 输入：
    #   self：当前训练策略入口。
    # 输出：
    #   None：不返回业务数据。
    def reset_transport(self):
        pass

    # 功能：
    #   1. 冻结源输入，发送当前传感器与图像数据，接收绑定同一观测和操纵角色的提案。
    #   2. 将四轴提案转换为普通协调器决策和可追溯调用记录，不越过下游安全授权。
    # 输入：
    #   self：仿真专用本地学习器入口。
    #   role：仅允许 local_navigation_advisor。
    #   output_type：仅允许 TextNavigationDecision。
    #   instructions：统一模型接口参数，此入口不执行自然语言提示。
    #   input_artifact：包含当前导航快照的输入制品。
    #   context_id：统一接口的上下文标识，此入口不维护提供方会话。
    #   multimodal：可选相机媒体，原始字节在发送时编码为 Base64。
    # 输出：
    #   result：结构化动作提案与本地训练调用记录组成的结果。
    def call(self, *, role, output_type, instructions, input_artifact,
             context_id=None, multimodal=None):
        if role != "local_navigation_advisor" or output_type is not TextNavigationDecision:
            raise ValueError("SIMULATION_TRAINING_ROLE_NOT_ALLOWED")
        started = _clock_seconds(time.monotonic())
        now = int(_clock_seconds(time.time()) * 1000)
        input_artifact = copy_json(input_artifact, limit=4 * 1024 * 1024)
        if (not isinstance(input_artifact, dict)
                or not isinstance(input_artifact.get("text_navigation_snapshot"), dict)):
            raise ValueError("SIMULATION_TRAINING_SNAPSHOT_REQUIRED")
        snapshot = input_artifact["text_navigation_snapshot"]
        observation = compile_training_observation(snapshot, now_unix_ms=now)
        realtime = RealtimeFeatureSnapshot.model_validate(snapshot["realtime_feature_snapshot"])
        visual = []
        if multimodal is not None and (
            not isinstance(multimodal, (list, tuple)) or len(multimodal) > 1
            or any(not isinstance(item, dict) for item in multimodal)
        ):
            raise ValueError("SIMULATION_TRAINING_MEDIA_INVALID")
        for item in multimodal or []:
            if any(isinstance(value, bytes) and 4 * ((len(value) + 2) // 3) > MAX_PACKET_BYTES - 32
                   for value in item.values()):
                raise ValueError("SIMULATION_TRAINING_MEDIA_TOO_LARGE")
            # Raw model-sized RGB stays byte-identical across the boundary.
            # No URLs are fetched by an input request and no path is executed.
            visual.append({key: ({"base64_bytes": base64.b64encode(value).decode("ascii")}
                                  if isinstance(value, bytes) else value)
                           for key, value in item.items()})
        preparation_ms = (_clock_seconds(time.monotonic()) - started) * 1000
        if preparation_ms < 0:
            raise ValueError("SIMULATION_TRAINING_CLOCK_REGRESSED")
        request = {"snapshot": snapshot, "observation": observation.model_dump(mode="json"),
                   "multimodal": visual,
                   "valid_until_unix_ms": teacher_input_deadline(realtime, now_ms=now),
                   "runtime_timing": {
                       "port_started_at_unix_ms": now,
                       "request_prepared_at_unix_ms": int(_clock_seconds(time.time()) * 1000),
                       "input_preparation_ms": preparation_ms,
                   }}
        proposal = self.client.propose(request)
        if proposal.expert_role != observation.navigation_expert_role:
            raise ValueError("SIMULATION_TRAINING_EXPERT_ROUTING_MISMATCH")
        action = proposal.action
        decision = TextNavigationDecision(
            snapshot_sha256=snapshot["snapshot_sha256"], action=action.mode,
            rationale_summary="Offline simulation learner control proposal.",
            pilot_control=(NormalizedPilotControl(**dict(zip(
                ("forward_axis", "right_axis", "up_axis", "yaw_axis"), action.axes, strict=True,
            ))) if action.mode == "pilot-control" else None),
        )
        # Deterministic join identity binds the learner reply to runtime receipts.
        call_id = "model-" + sha256_json(proposal)[:24]
        record = ModelCallRecord(
            call_id=call_id, role=role, attempt=1, input_sha256=sha256_json(input_artifact),
            output_sha256=sha256_json(decision), output_schema="TextNavigationDecision",
            provider="simulation-training", model=proposal.policy_sha256,
            latency_ms=round((_clock_seconds(time.monotonic()) - started) * 1000),
            created_at=datetime.now(UTC),
            simulation_training_trace=SimulationTrainingTrace(
                request_sha256=proposal.request_sha256, policy_sha256=proposal.policy_sha256,
                selected_navigation_role=proposal.expert_role,
            ),
        )
        result = StructuredCallResult(artifact=decision, record=record)
        return result
