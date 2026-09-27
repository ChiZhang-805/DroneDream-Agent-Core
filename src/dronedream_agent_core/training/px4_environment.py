"""Actual PX4/Gazebo reset/step adapter, using the product's only control outlet.

Run the trainer in the simulation host (Linux/WSL), not in a real aircraft.
Each reset launches the existing native-sensor mission runner with an explicit
training-only port. Observations come from the deployed feature compiler;
actions are joined to acknowledged executor receipts, never echoed as applied.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import StrictModel, Vector3, VehicleAsset
from ..control_timing import TRAINING_DISPATCH_RESERVE_MS as LOCAL_DISPATCH_RESERVE_MS
from ..hashing import sha256_json
from ..local_expert_harness import NavigationExpertRole
from ..local_policy_training import LocalPolicyObservation
from ..plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from ..runtime_phase import ENDING_PHASES, phase_context
from ..runtime_phase_observer import RuntimePhaseObserver
from ..simulation_camera_profile import validate_camera_profile_choice
from .capture_archive import MAXIMUM_ARCHIVE_BYTES, PackedNativeTransition, pack_native_transition
from .evidence_files import MAX_TRAINING_ROW_BYTES, read_evidence_object
from .evidence_publication import write_evidence_object as _write_new
from .evidence_snapshot import detach_evidence
from .executed_control import executed_training_action
from .flight_environment import (
    ControlPreparationExpired,
    FlightObservation,
    FlightStep,
    PilotAction,
)
from .mission_groups import mission_group_evidence
from .native_transition import CapturedNativeTransition, evaluate_native_transition
from .observations import PreparedTrainingInput, TrainingObservationError
from .outcome_channel import descriptor_path as outcome_descriptor_path
from .outcome_verifier import OutcomeEnvelope, SimulationOutcomeVerifier
from .policy_exchange import TrainingPolicyExchange, TrainingProposal
from .runtime_evidence import (
    ControlReceiptMonitor,
    IndependentPoseMonitor,
    OutcomeWindowError,
    read_object,
)

# 流式序列化器在起飞前加载，避免首条动作在短租约内承担模块初始化成本。
from .stream_capture import StreamActionCapture, pack_stream_capture
from .transition_writer import TransitionWriter
from .visual_lineage import VisualInputContract
from .visual_observation import FrozenVisualEncoder

ASSET_FIELDS = ("route", "semantic", "world_sdf", "vehicle_sdf", "vehicle", "controller_params")
# Includes visual encoding, verified transition return, recurrent inference,
# signed reply and the runtime's post-reply safety evaluation. The measured
# learner path can consume 45 ms before reply; 35 ms admitted doomed commands.
# Owned numeric-container snapshots reduced measured preparation to 34 ms.
# Keep that predictive reserve for unselected startup and the reward-step API. Streaming
# imitation instead attempts preparation against the actual remaining deadline,
# then discards an output if it lacks the shared 50 Hz executor dispatch reserve.
# A prediction of preparation time never substitutes for final admission.
TRAINING_REPLY_PREPARATION_RESERVE_MS = 50
# 仅旧请求可丢弃并等待新帧；超长租约、身份不符和损坏内容必须继续上抛。
EXPIRED_POLICY_REQUESTS = frozenset({
    'TRAINING_POLICY_INPUT_EXPIRED', 'TRAINING_POLICY_REQUEST_EXPIRED_OR_REPLAYED',
})


# 功能：
#   在创建飞行回合前拒绝只有接收时钟的视觉控制配置，避免起飞后永久等待不可能就绪的输入。
# 输入：
#   config：已验证的仿真训练配置；原生渲染制品和实际报头仍由运行器另外核验。
# 输出：
#   None：来源配置可用于后续校验，否则抛出明确错误。
def require_visual_control_source(config) -> None:
    runtime = getattr(config, 'native_camera_clock_runtime', None)
    if runtime is not None:
        from ..simulation_camera_clock import validate_native_camera_clock

        # 地图融合也消费原生源钟；不能等到子进程创建后才发现源/二进制版本不一致。
        validate_native_camera_clock(Path(runtime),
            source_root=Path(__file__).resolve().parents[3]/'native/camera_clock')
    if ((config.visual_encoder is not None or config.visual_package is not None)
            and config.render_replica_runtime is None):
        if runtime is None:
            raise ValueError("PX4_VISUAL_CONTROL_REQUIRES_SOURCE_CLOCK_RENDERER")


class Px4TrainingConfig(StrictModel):
    """Explicit simulation assets, role, holdout exclusions and bounded collection settings."""

    runner: Path
    output_root: Path
    mission_id: str = Field(min_length=1, max_length=128)
    expert_role: NavigationExpertRole
    route: Path
    semantic: Path
    world_sdf: Path
    vehicle_sdf: Path
    vehicle: Path
    controller_params: Path
    asset_sha256: dict[str, str]
    minimum_enu_m: tuple[float, float, float]
    maximum_enu_m: tuple[float, float, float]
    speed_limit_mps: float = Field(default=0.4, gt=0, le=2)
    required_clearance_m: float = Field(default=0.25, gt=0, le=2)
    episode_steps: int = Field(default=256, ge=2, le=10000)
    # 显式流式回合从首帧采用流式准入，避免必须先收到首帧才能选择模式的循环依赖。
    # 仅省去预测准备余量；准备完成后的真实截止时间和派发安全余量不变。
    initial_collection_mode: Literal['stream-imitation', 'reward-step'] | None = None
    startup_timeout_seconds: float = Field(default=240, ge=30, le=600)
    quiesce_timeout_seconds: float = Field(default=180, ge=30, le=300)
    visual_package: Path | None = None
    visual_encoder: Path | None = None
    visual_input_contract: VisualInputContract | None = None
    record_multimodal_training_dataset: bool = False
    batch_static_world_visuals: bool = False
    simulation_camera_profile: str = "native"
    camera_source_model_sha256: str | None = None
    preflight_render_warmup: bool = False
    preflight_depth_warmup: bool = False
    render_preparation_runtime: Path | None = None
    render_cache_bundle: Path | None = None
    render_replica_runtime: Path | None = None
    native_sensor_runtime: Path | None = None
    native_camera_clock_runtime: Path | None = None
    experimental_map_fusion: bool = False
    # 仅阶段决策实验可显式启用已存在的有界衔接；回执仍保留独立控制器来源。
    bounded_hybrid_control: bool = False
    # 仅阶段决策流式采集可显式跨专家；普通单专家训练仍禁止角色切换。
    collection_expert_roles: list[NavigationExpertRole] = Field(default_factory=list, max_length=3)
    held_out_missions: list[str] = Field(default_factory=list)
    held_out_route_groups: list[str] = Field(default_factory=list)

    # 功能：
    #   拒绝相互竞争的渲染路径，要求预热、缓存和相机来源使用同一明确配置。
    # 输入：
    #   self：完成字段解析的仿真训练配置。
    # 输出：
    #   self：通过相机与渲染组合约束的配置。
    @model_validator(mode="after")
    def bound_camera_profile(self):
        if self.collection_expert_roles and (
            self.initial_collection_mode != "stream-imitation"
            or self.expert_role not in self.collection_expert_roles
            or len(set(self.collection_expert_roles)) != len(self.collection_expert_roles)
        ):
            raise ValueError("PX4_STAGE_COLLECTION_REQUIRES_EXPLICIT_UNIQUE_STREAM_ROLES")
        if self.native_camera_clock_runtime is not None and (
                self.render_replica_runtime is not None or self.native_sensor_runtime is None):
            raise ValueError('native camera clock requires exclusive native sensor runtime')
        if self.experimental_map_fusion and (
                self.native_sensor_runtime is None or self.simulation_camera_profile != 'responsive-control'
                or self.native_camera_clock_runtime is None or self.render_replica_runtime is not None
                or not self.record_multimodal_training_dataset):
            raise ValueError('map fusion collection requires native responsive camera and raw media')
        if (self.visual_encoder is None) != (self.visual_input_contract is None):
            raise ValueError("standalone visual encoder requires its input contract")
        if self.visual_package is not None and self.visual_encoder is not None:
            raise ValueError("choose a visual package or a standalone encoder, not both")
        validate_camera_profile_choice(
            self.simulation_camera_profile, self.camera_source_model_sha256
        )
        if (self.simulation_camera_profile != 'native'
                and self.visual_package is None and self.visual_encoder is None
                and not self.experimental_map_fusion):
            raise ValueError('non-native camera profile requires explicit visual training input')
        if self.render_replica_runtime is not None and (
            (self.visual_package is None and self.visual_encoder is None)
            or self.simulation_camera_profile == "native"
            or self.preflight_render_warmup
            or self.preflight_depth_warmup
            or self.render_preparation_runtime is not None
            or self.render_cache_bundle is not None
        ):
            raise ValueError("isolated rendering requires exclusive source-bound visual training")
        if self.preflight_render_warmup and self.simulation_camera_profile == "native":
            raise ValueError("render warmup requires the source-bound camera profile")
        if self.preflight_depth_warmup and not self.preflight_render_warmup:
            raise ValueError("depth warmup requires explicit preflight render warmup")
        if self.render_cache_bundle is not None and self.render_preparation_runtime is None:
            raise ValueError("render cache requires verified native runtime")
        if self.render_preparation_runtime is not None and not self.preflight_render_warmup:
            raise ValueError("render cache requires explicit preflight preparation")
        return self


# 功能：
#   核对六项普通资产的原始摘要，语义解析保留同次读取字节，大型 SDF 只流式计算摘要。
# 输入：
#   config：已固定路径和摘要的配置。
#   error_prefix：构造或重置阶段的错误前缀。
# 输出：
#   contents：路线、车辆和语义地图的已核对字节，总量不超过 72 MiB。
def _validated_asset_bytes(config, *, error_prefix: str) -> dict[str, bytes]:
    if set(config.asset_sha256) != set(ASSET_FIELDS):
        raise ValueError("PX4_TRAINING_ALL_ASSET_IDENTITIES_REQUIRED")
    contents = {}
    for name in ASSET_FIELDS:
        digest = config.asset_sha256[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(error_prefix + name)
        if name in {"route", "vehicle", "semantic"}:
            limit = 64 * 1024 * 1024 if name == "semantic" else MAX_TRAINING_ROW_BYTES
            content = read_plugin_file(getattr(config, name), limit=limit)
            contents[name] = content
            actual = hashlib.sha256(content).hexdigest()
        else:
            actual = hash_plugin_file(getattr(config, name), limit=256 * 1024 * 1024)
        if actual != digest:
            raise ValueError(error_prefix + name)
    return contents


# 功能：
#   拒绝非有限、布尔或超大等待期限；已经到期的合法期限由调用方报告超时。
# 输入：
#   deadline：与当前单调时钟同源的截止秒数，最多允许未来六百秒。
# 输出：
#   None：不返回业务数据。
def _check_wait_deadline(deadline) -> None:
    if type(deadline) not in (int, float) or not 0 < deadline <= time.monotonic() + 600:
        raise ValueError("PX4_TRAINING_WAIT_DEADLINE_INVALID")


class Px4GazeboTrainingEnvironment:
    """Bridge learner proposals to the guarded simulator and separately join actual execution."""

    simulation_only = True
    evidence_kind = "px4-gazebo"

    # 功能：
    #   复制并重验 Linux 仿真配置，绑定当前运行器、资产及独立结果验证器，不启动飞行。
    # 输入：
    #   self：待初始化的仿真训练环境。
    #   config：明确资产、任务、限额及留出分区的配置。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, config: Px4TrainingConfig):
        if not sys.platform.startswith("linux"):
            raise ValueError("PX4_TRAINER_MUST_RUN_IN_LINUX_SIMULATION_HOST")
        if not isinstance(config, Px4TrainingConfig):
            raise ValueError("PX4_TRAINING_CONFIG_INVALID")
        config = Px4TrainingConfig.model_validate(config.model_dump(mode="python"), strict=True)
        require_visual_control_source(config)
        # 固定工作目录解释；后续修改调用方配置不会改变这一回合的资产选择。
        for name in (
            "runner",
            "output_root",
            *ASSET_FIELDS,
            "visual_package",
            "visual_encoder",
            "render_preparation_runtime",
            "render_cache_bundle",
            "render_replica_runtime",
            "native_sensor_runtime",
            "native_camera_clock_runtime",
        ):
            path = getattr(config, name)
            if path is not None:
                path = path.absolute()
                check_plain_plugin_path(path)
                setattr(config, name, path)
        if config.mission_id in config.held_out_missions:
            raise ValueError("TRAINING_HELD_OUT_MISSION_FORBIDDEN")
        current_runner = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "run_school_map_depth_qualification.py"
        )
        if config.runner.resolve() != current_runner or not current_runner.is_file():
            raise ValueError("PX4_TRAINING_REQUIRES_NATIVE_MISSION_RUNNER")
        contents = _validated_asset_bytes(config, error_prefix="PX4_TRAINING_ASSET_CHANGED:")
        self.config = config
        self.mission_split = mission_group_evidence(
            contents["route"],
            config.asset_sha256["semantic"],
            expected_route_sha256=config.asset_sha256["route"],
        )
        if self.mission_split.group_sha256 in config.held_out_route_groups:
            raise ValueError("TRAINING_HELD_OUT_SPATIAL_ROUTE_FORBIDDEN")
        vehicle = VehicleAsset.model_validate(
            decode_json(contents["vehicle"], limit=MAX_TRAINING_ROW_BYTES, node_limit=1_000_000)
        )
        semantic = decode_json(contents["semantic"], limit=64 * 1024 * 1024, node_limit=2_000_000)
        if type(semantic) is not dict:
            raise ValueError("PX4_TRAINING_SEMANTIC_OBJECT_REQUIRED")
        self.verifier = SimulationOutcomeVerifier(
            semantic["runtime_collision_primitives"]
            if "runtime_collision_primitives" in semantic
            else semantic["collision_primitives"],
            OutcomeEnvelope(
                vehicle.body_radius_m,
                vehicle.body_height_m,
                config.minimum_enu_m,
                config.maximum_enu_m,
            ),
        )
        self.visual = (FrozenVisualEncoder.from_artifact(
                           config.visual_encoder, config.visual_input_contract)
                       if config.visual_encoder else
                       FrozenVisualEncoder(config.visual_package)
                       if config.visual_package else None)
        self._policy_sha256: str | None = None
        self._process = self._exchange = self._monitor = self._log = None
        self._transition_writer = None
        self._receipt_monitor = None
        self._phase_observer = None
        self._pending = self._observation = None
        self._following_request = None
        self._outcome_window_failure = None
        self._rollout_stop_reason = None
        self._collection_mode = self.config.initial_collection_mode
        self._stream_captures = []
        self._quiesced = True
        self._closed = False
        self.episode_path: Path | None = None

    # 功能：
    #   只在确认静止且环境未关闭时更换策略身份，非法摘要不改动原绑定。
    # 输入：
    #   self：当前训练环境。
    #   digest：实际策略权重的小写十六进制 SHA-256。
    # 输出：
    #   None：不返回业务数据。
    def bind_policy_identity(self, digest: str) -> None:
        if (
            self._closed
            or not self._quiesced
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("TRAINING_POLICY_IDENTITY_REQUIRES_QUIESCED_VALID_WEIGHTS")
        self._policy_sha256 = digest

    # 功能：
    #   1. 确认上次停止后重验资产和路线分区，再启动独立仿真回合。
    #   2. 启动失败仍请求安全停止；异常和进程退出不能替代原生落地证明。
    # 输入：
    #   self：未关闭且已绑定策略的训练环境。
    #   seed：学习器非负整数种子，不代表原生仿真的随机化种子。
    # 输出：
    #   observation：首次通过真实来源与时效检查的控制观测。
    def reset(self, *, seed: int) -> FlightObservation:
        if self._closed or self._policy_sha256 is None or type(seed) is not int or seed < 0:
            raise ValueError("PX4_TRAINING_RESET_STATE_INVALID")
        self.quiesce()
        contents = _validated_asset_bytes(
            self.config, error_prefix="PX4_TRAINING_ASSET_CHANGED_BEFORE_RESET:"
        )
        if (
            mission_group_evidence(
                contents["route"],
                self.config.asset_sha256["semantic"],
                expected_route_sha256=self.config.asset_sha256["route"],
            )
            != self.mission_split
        ):
            raise ValueError("PX4_TRAINING_MISSION_CHANGED_BEFORE_RESET")
        try:
            observation = self._start_episode(seed)
            return observation
        except BaseException as error:
            # No descriptor, observer or control authority survives a failed reset.
            # If a simulator was launched, quiesce still requires native landing
            # evidence; a Python exception is never proof that it is grounded.
            try:
                self.quiesce()
            except BaseException as cleanup_error:
                error.add_note("PX4 reset cleanup also failed: " + repr(cleanup_error))
            raise

    # 功能：
    #   1. 创建独立回合和通信／证据写入器，启动唯一的原生传感器任务运行器。
    #   2. 等待有独立真值基线的跟踪阶段；启动期间中性动作不作为学生执行标签。
    # 输入：
    #   self：已完成资产与停止检查的环境。
    #   seed：写入回合记录的学习器种子。
    # 输出：
    #   observation：已接纳的首个跟踪阶段观测。
    def _start_episode(self, seed: int) -> FlightObservation:
        self._rollout_stop_reason = None
        self.episode_path = self.config.output_root / ("episode-" + uuid.uuid4().hex)
        self.episode_path.mkdir(parents=True, exist_ok=False)
        self.simulation = self.episode_path / "flight" / "simulation"
        self._quiesced = False
        self._exchange = TrainingPolicyExchange(self.episode_path / "policy-channel.json")
        self._receipt_monitor = ControlReceiptMonitor(
            self.simulation / "depth-local-safety-history.jsonl",
            self.simulation / "runtime-state" / "control-applications.jsonl",
        )
        self._last_application = None
        self._collection_mode = self.config.initial_collection_mode
        self._stream_captures = []
        self._sequence = 0
        self._rollout_stop_reason = None
        self._retained_capture_bytes = 0
        self._interface_timings = deque(maxlen=512)
        self._input_rejection_counts = {}
        self._proposal_preparation_expirations = 0
        self._input_received_count = 0
        self._source_history = deque(maxlen=32)
        self._prepared_input = None
        self._following_request = None
        self._outcome_window_failure = None
        self._transition_writer = TransitionWriter(_write_new)
        self._monitor = IndependentPoseMonitor(outcome_descriptor_path(self._exchange.path))
        self._phase_observer = RuntimePhaseObserver(self.simulation / "runtime-phase.json")
        args = [
            sys.executable,
            str(self.config.runner),
            *(str(getattr(self.config, name)) for name in ASSET_FIELDS),
            str(self.episode_path / "flight"),
            "--training-data-collection",
            "--simulation-training-channel",
            str(self._exchange.path),
            "--speed-limit-mps",
            str(self.config.speed_limit_mps),
            "--required-clearance-m",
            str(self.config.required_clearance_m),
        ]
        if self.config.batch_static_world_visuals:
            args.append("--batch-static-world-visuals")
        if self.config.experimental_map_fusion:
            args.append('--experimental-map-fusion')
        if self.config.bounded_hybrid_control:
            args.append('--bounded-hybrid-control')
        if self.config.preflight_render_warmup:
            args.append("--preflight-render-warmup")
        if self.config.preflight_depth_warmup:
            args.append("--preflight-depth-warmup")
        if self.config.render_preparation_runtime is not None:
            args.extend(
                ["--render-preparation-runtime", str(self.config.render_preparation_runtime)]
            )
        if self.config.render_cache_bundle is not None:
            args.extend(["--render-cache-bundle", str(self.config.render_cache_bundle)])
        if self.config.render_replica_runtime is not None:
            args.extend(["--render-replica-runtime", str(self.config.render_replica_runtime)])
        if self.config.native_sensor_runtime is not None:
            args.extend(["--native-sensor-runtime", str(self.config.native_sensor_runtime)])
        if self.config.simulation_camera_profile != "native":
            args.extend(
                [
                    "--simulation-camera-profile",
                    self.config.simulation_camera_profile,
                    "--camera-source-model-sha256",
                    self.config.camera_source_model_sha256,
                ]
            )
        if self.config.record_multimodal_training_dataset:
            args.extend(
                [
                    "--multimodal-dataset-root",
                    str(self.episode_path / "media"),
                    "--multimodal-flight-id",
                    self.episode_path.name,
                ]
            )
        if self.visual:
            args.extend(
                [
                    "--local-navigation-visual-enabled",
                    "--learning-image-size",
                    str(self.visual.manifest.visual_width),
                    str(self.visual.manifest.visual_height),
                ]
            )
        _write_new(
            self.episode_path / "reset.json",
            {
                "mission_id": self.config.mission_id,
                "learner_seed": seed,
                "seed_scope": "learner-only; native simulator reset has no random perturbation",
                "config": self.config.model_dump(mode="json"),
                "policy_sha256": self._policy_sha256,
                "visual_encoder_sha256": self.visual.sha256 if self.visual else None,
                "visual_input_contract": self.visual.input_contract if self.visual else None,
                "simulation_only": True,
                "qualification_granted": False,
            },
        )
        self._log = (self.episode_path / "runner.log").open("xb")
        child_env = os.environ.copy()
        if self.config.native_camera_clock_runtime is not None:
            child_env['DRONEDREAM_NATIVE_CAMERA_CLOCK_RUNTIME'] = str(
                self.config.native_camera_clock_runtime)
        child_env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[2]) + os.pathsep + child_env.get("PYTHONPATH", "")
        )
        self._process = subprocess.Popen(
            args, stdout=self._log, stderr=subprocess.STDOUT, env=child_env
        )
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            request = self._next_request(deadline=deadline)
            task = request["snapshot"]["strategic_context"]["task"]
            phase = task.get("executor_phase", task.get("phase", "UNKNOWN"))
            if phase in {"TRACK", "WAYPOINT_SETTLE", "CHECKPOINT"}:
                source_ms = request["observation"]["temporal_evidence"]["observed_at_unix_ms"]
                if not self._monitor.has_start_baseline(source_ms):
                    self._record_rejected_input(request, "independent-start-baseline-missing")
                    self._exchange.discard_pending()
                    continue
                try:
                    observation = self._adopt(request)
                    return observation
                except TrainingObservationError as error:
                    if error.reason_code != "TRAINING_INPUT_SENSOR_EVIDENCE_EXPIRED":
                        raise
                    self._record_rejected_input(request, "input-expired-at-adoption")
                    self._exchange.discard_pending()
                    continue
            # Startup uses explicit neutral requests, never student training
            # labels on a grounded/arming vehicle. Native takeoff stays guarded.
            try:
                self._exchange.reply(
                    TrainingProposal(
                        request_sha256=sha256_json(request),
                        policy_sha256=self._policy_sha256,
                        expert_role=request["observation"]["navigation_expert_role"],
                        action=PilotAction(mode="hold", axes=[0.0] * 4),
                    )
                )
            except (TimeoutError, ConnectionError):
                # A cancelled startup request has no action label. Await a new
                # source; neither its old reply nor its lease is reused.
                continue
        raise TimeoutError("PX4_TRAINING_STARTUP_TIMEOUT")

    # 功能：
    #   接收已鉴权请求，重建数值观测并保留真实源历史；视觉预取不授予控制权限。
    # 输入：
    #   self：具有当前通信端点的环境。
    #   timeout_seconds：本次接收的最大等待秒数。
    # 输出：
    #   request：关联准备对象的原始请求，截止时间保持不变。
    def _receive_source_request(self, *, timeout_seconds: float) -> dict:
        request = self._exchange.wait_request(timeout_seconds=timeout_seconds)
        preparation_started = time.monotonic()
        received_ms = int(time.time() * 1000)
        visual = getattr(self, "visual", None)
        if visual is not None and request["valid_until_unix_ms"] - received_ms >= (
            self._input_admission_budget_ms()
        ):
            # Overlap this frame's immutable pixel encoding with numeric source
            # validation. Neither prefetch nor cache hits admit an observation.
            visual.prime(request["multimodal"])
        self._input_received_count = getattr(self, "_input_received_count", 0) + 1
        timing = {
            "stage": "request-received",
            "remaining_input_lease_ms": request["valid_until_unix_ms"] - received_ms,
            "received_at_unix_ms": received_ms,
            "capture_reference_unix_ms": request["snapshot"][
                "control_reference_observed_at_unix_ms"
            ],
            "runtime_timing": detach_evidence(request.get("runtime_timing")),
        }
        self._interface_timings.append(timing)
        # Historical perception is not a control permission. Preserve the
        # source clocks even for requests rejected while awaiting execution.
        self._prepared_input = PreparedTrainingInput.from_request(request)
        self._retain_source_observation(self._prepared_input.sample)
        timing["input_preparation_ms"] = (time.monotonic() - preparation_started) * 1000
        timing["remaining_after_preparation_ms"] = request["valid_until_unix_ms"] - int(
            time.time() * 1000
        )
        return request

    # 功能：
    #   记录输入被拒原因的累计数和有界明细，不重发动作或延长原输入有效期。
    # 输入：
    #   self：持有诊断队列和计数的环境。
    #   request：被拒绝的源请求。
    #   reason：内部固定的拒绝原因。
    # 输出：
    #   None：不返回业务数据。
    def _record_rejected_input(self, request: dict, reason: str) -> None:
        counts = getattr(self, "_input_rejection_counts", None)
        if counts is None:
            counts = self._input_rejection_counts = {}
        counts[reason] = counts.get(reason, 0) + 1
        self._interface_timings.append(
            {
                "stage": "request-rejected",
                "reason": reason,
                "remaining_input_lease_ms": request["valid_until_unix_ms"]
                - int(time.time() * 1000),
                "snapshot_sha256": request.get("snapshot", {}).get("snapshot_sha256"),
            }
        )

    # 功能：
    #   读取非阻塞阶段缓存，在结束阶段撤销待处理输入；阶段名称不作为物理落地证明。
    # 输入：
    #   self：具有阶段观察器与训练端点的环境。
    # 输出：
    #   None：不返回业务数据。
    def _check_executor_ending(self) -> None:
        observer = getattr(self, "_phase_observer", None)
        # No synchronous filesystem read in the actor's input/reply deadline.
        # UNKNOWN is only absent lifecycle context, never motion permission.
        context = observer.latest() if observer is not None else phase_context(None)
        phase = context["executor_phase"]
        if phase not in ENDING_PHASES:
            return
        self._exchange.discard_pending()
        self._following_request = self._pending = self._prepared_input = None
        if getattr(self, "_rollout_stop_reason", None) is None:
            self._rollout_stop_reason = {
                "reason": "executor-ending",
                "executor_phase": phase,
                "phase_context": context,
                "source": "runtime-phase-context",
                "qualified_for_flight": False,
            }
        raise RuntimeError("PX4_TRAINING_EXECUTOR_ENDING:" + str(phase))

    # 功能：
    #   按采集模式提供输入预算，流式用实际准备耗时，同步模式另留预测准备余量。
    # 输入：
    #   self：持有本回合采集模式的环境。
    # 输出：
    #   budget_ms：原有派发余量或派发与准备余量之和，单位毫秒。
    def _input_admission_budget_ms(self) -> int:
        if getattr(self, "_collection_mode", None) == "stream-imitation":
            budget_ms = LOCAL_DISPATCH_RESERVE_MS
        else:
            budget_ms = LOCAL_DISPATCH_RESERVE_MS + TRAINING_REPLY_PREPARATION_RESERVE_MS
        return budget_ms

    # 功能：
    #   在同一截止时刻内等待新输入，只跳过明确过期或断开的请求，鉴权和来源错误上抛。
    # 输入：
    #   self：已启动原生仿真进程的环境。
    #   deadline：单调时钟的绝对截止秒数，不因重试更新。
    # 输出：
    #   request：具有足够准备余量的新请求，尚未发送控制动作。
    def _next_request(self, *, deadline: float) -> dict:
        _check_wait_deadline(deadline)
        while time.monotonic() < deadline:
            self._check_executor_ending()
            if self._process.poll() is not None:
                raise RuntimeError("PX4_SIMULATION_STOPPED_BEFORE_NEXT_OBSERVATION")
            try:
                request = getattr(self, "_following_request", None)
                self._following_request = None
                if request is None:
                    request = self._receive_source_request(
                        timeout_seconds=max(0.001, min(1.0, deadline - time.monotonic()))
                    )
                self._check_executor_ending()
                remaining = request["valid_until_unix_ms"] - int(time.time() * 1000)
                # Streaming may attempt inference inside the remaining lease;
                # it grants no action until _submit_action checks the actual
                # completed preparation against the original dispatch deadline.
                if remaining < self._input_admission_budget_ms():
                    self._record_rejected_input(request, "insufficient-input-budget")
                    self._exchange.discard_pending(reason="insufficient-input-budget")
                    continue
                return request
            except TimeoutError:
                continue
            except ConnectionError:
                # The runtime can expire/cancel a connection before its packet
                # arrives. There is no admitted input or applied action here.
                # Authentication, provenance and replay errors still fail closed.
                continue
            except ValueError as error:
                if str(error) in EXPIRED_POLICY_REQUESTS:
                    continue  # Already expired requests cannot receive replacement actions.
                raise
        self._check_executor_ending()
        if getattr(self, "_rollout_stop_reason", None) is None:
            self._rollout_stop_reason = {
                "reason": "next-observation-timeout",
                "required_remaining_input_ms": self._input_admission_budget_ms(),
                "collection_mode": getattr(self, "_collection_mode", None),
                "qualified_for_flight": False,
            }
        raise TimeoutError("PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT")

    # 功能：
    #   按原生时钟保留独立历史；同槽显式递增修订只替换末行，不增加样本或续期。
    #   换流或大间隔重建历史；与部署历史一致，拒绝重标时间、修订冲突及真实时钟倒退。
    # 输入：
    #   self：持有最多三十二条源历史的环境。
    #   sample：已经由准备阶段验证的源观测。
    # 输出：
    #   None：不返回业务数据。
    def _retain_source_observation(self, sample: LocalPolicyObservation) -> None:
        prior = self._source_history[-1].temporal_evidence if self._source_history else None
        current = sample.temporal_evidence
        if prior == current:
            # The history clock is native flight-state evidence, just as in
            # deployed CausalControlHistory. New geometry/context or changing
            # age fields do not make repeated native state an independent row.
            # Every snapshot was already content-validated before this point.
            return
        owned = detach_evidence(sample)
        if prior is not None:
            if current.stream_id == prior.stream_id:
                if current.sample_sha256 == prior.sample_sha256:
                    raise ValueError('PX4_TRAINING_OBSERVATION_SAMPLE_REDATED')
                delta = current.observed_at_unix_ms - prior.observed_at_unix_ms
                if delta == 0:
                    if (current.history_slot_revision > prior.history_slot_revision
                            and current.reset_history == prior.reset_history):
                        self._source_history[-1] = owned
                        return
                    raise ValueError('PX4_TRAINING_OBSERVATION_SAME_TIME_CONFLICT')
                if delta < 0 and not current.reset_history:
                    self._source_history.clear()
                    raise ValueError('PX4_TRAINING_OBSERVATION_CLOCK_REGRESSED')
            if (
                current.stream_id != prior.stream_id
                or current.reset_history
                or current.observed_at_unix_ms - prior.observed_at_unix_ms > 250
            ):
                self._source_history.clear()
        self._source_history.append(owned)

    # 功能：
    #   1. 再验来源时效、执行阶段和专家角色，按明确视觉配置生成控制观测。
    #   2. 内部待执行证据与学习器返回值分别持有容器，外部修改不能回写历史。
    # 输入：
    #   self：持有本请求准备对象的环境。
    #   request：对应同一准备对象的完整请求。
    # 输出：
    #   observation：学习器可以读取的独立观测副本。
    def _adopt(self, request: dict) -> FlightObservation:
        started = time.perf_counter()
        if self._prepared_input is None:
            raise ValueError("PX4_TRAINING_PREPARED_INPUT_MISSING")
        sample = self._prepared_input.admit(request, now_unix_ms=int(time.time() * 1000))
        task = request.get("snapshot", {}).get("strategic_context", {}).get("task", {})
        phase = task.get("executor_phase", task.get("phase", "UNKNOWN"))
        if phase in ENDING_PHASES:
            self._exchange.discard_pending()
            self._rollout_stop_reason = {
                "reason": "executor-ending",
                "executor_phase": phase,
                "source_snapshot_sha256": sample.source_snapshot_sha256,
                "qualified_for_flight": False,
            }
            raise RuntimeError("PX4_TRAINING_EXECUTOR_ENDING:" + phase)
        allowed_roles = getattr(self.config, "collection_expert_roles", [])
        stage_collection = (
            getattr(self, "_collection_mode", None) == "stream-imitation"
            and sample.navigation_expert_role in allowed_roles
        )
        if sample.navigation_expert_role != self.config.expert_role and not stage_collection:
            self._exchange.discard_pending()
            self._rollout_stop_reason = {
                "reason": "role-specific-episode-required",
                "executor_phase": phase,
                "source_snapshot_sha256": sample.source_snapshot_sha256,
                "configured_expert_role": self.config.expert_role,
                "requested_expert_role": sample.navigation_expert_role,
                "qualified_for_flight": False,
            }
            raise ValueError("PX4_TRAINING_REQUIRES_ROLE_SPECIFIC_EPISODE")
        if self._source_history and (
            self._source_history[-1].navigation_expert_role != sample.navigation_expert_role
        ):
            # 角色标签保留真实请求值；跨角色不能把原专家的循环状态当作新专家历史。
            self._source_history.clear()
        compiled_at = time.perf_counter()
        if self.visual:
            sample = self.visual.attach(sample, request["multimodal"])
        elif request["multimodal"]:
            raise ValueError("PX4_TRAINING_CANNOT_SILENTLY_DISCARD_CAMERA")
        observation = FlightObservation(
            mission_id=self.config.mission_id,
            episode_id=self.episode_path.name,
            map_sha256=self.config.asset_sha256["semantic"],
            sequence=self._sequence,
            sample=sample,
            prior_observations=[
                s
                for s in self._source_history
                if s.temporal_evidence.observed_at_unix_ms
                < sample.temporal_evidence.observed_at_unix_ms
            ],
        )
        # 完整构造成功后才改变待执行状态；返回值也不能共享历史队列内部对象。
        pending = detach_evidence(request)
        owned = detach_evidence(observation)
        observation = detach_evidence(observation)
        self._pending, self._observation = pending, owned
        self._interface_timings.append(
            {
                "stage": "observation-adoption",
                "sequence": self._sequence,
                "compile_ms": (compiled_at - started) * 1000,
                "visual_and_pack_ms": (time.perf_counter() - compiled_at) * 1000,
                "remaining_input_lease_ms": request["valid_until_unix_ms"]
                - int(time.time() * 1000),
            }
        )
        return observation

    # 功能：
    #   向仿真研究策略公开当前已接纳快照及严格过去回执，复制返回，不允许修改待执行请求。
    # 输入：
    #   self：当前具有待响应原生观测的仿真环境。
    # 输出：
    #   snapshot、application：同回合源快照与已绑定实际回执，回执未齐全时为 None。
    def current_control_context(self):
        snapshot = self.current_navigation_snapshot()
        application = self._receipt_monitor.latest_before(
            snapshot['control_reference_observed_at_unix_ms'])
        return snapshot, application

    # 功能：只复制当前已接纳的导航输入，不同步扫描执行账本；适用于结果稍后归档的阶段教师。
    # 输入：活跃采集环境；输出：独立快照副本，身份/生命周期校验与完整上下文入口一致。
    def current_navigation_snapshot(self):
        if self._pending is None or self._observation is None or self._quiesced or self._closed:
            raise RuntimeError('PX4_TRAINING_CONTEXT_REQUIRES_LIVE_OBSERVATION')
        snapshot = self._pending['snapshot']
        if snapshot['snapshot_sha256'] != self._observation.sample.source_snapshot_sha256:
            raise ValueError('PX4_TRAINING_CONTEXT_IDENTITY_MISMATCH')
        return detach_evidence(snapshot)

    # 功能：
    #   读取当前已接纳观测的位置副本，不复制整幅感知快照，也不查询不需要的执行回执。
    # 输入：
    #   self：当前具有待响应原生观测的仿真环境。
    # 输出：
    #   position：与当前观测身份一致、有限且独立持有的三维位置。
    def current_position_m(self):
        if self._pending is None or self._observation is None or self._quiesced or self._closed:
            raise RuntimeError('PX4_TRAINING_CONTEXT_REQUIRES_LIVE_OBSERVATION')
        snapshot = self._pending['snapshot']
        if snapshot['snapshot_sha256'] != self._observation.sample.source_snapshot_sha256:
            raise ValueError('PX4_TRAINING_CONTEXT_IDENTITY_MISMATCH')
        position = Vector3.model_validate(snapshot['current_position_m']).model_copy(deep=True)
        return position

    # 功能：
    #   等待指定动作的执行回执，同时接收新传感器历史；最多保留一个可用的执行后请求。
    # 输入：
    #   self：持有执行回执监视器的环境。
    #   call_id：待关联的控制调用标识。
    #   deadline：本次关联的单调时钟绝对截止秒数。
    # 输出：
    #   receipt：真实命令及其执行接纳记录组成的二元组。
    def _application(self, call_id: str, *, deadline: float):
        _check_wait_deadline(deadline)
        while time.monotonic() < deadline:
            receipt = self._receipt_monitor.poll(call_id)
            if receipt is not None:
                return receipt
            if self._process.poll() is not None:
                raise RuntimeError("PX4_SIMULATION_STOPPED_WITHOUT_ACTION_RECEIPT")
            retain_following = False
            try:
                request = self._receive_source_request(timeout_seconds=0.005)
                # The acknowledgment may have arrived while reading this new
                # source. Don't throw away a usable post-application observation
                # and wait another sensor cycle just because the first poll lost
                # that race. Keep at most this one authenticated request; never
                # renew its timestamps, publish a reply, or replay the old action.
                receipt = self._receipt_monitor.poll(call_id)
                if receipt is not None:
                    accepted_ms = receipt[1].accepted_at_unix_ms
                    observed_ms = request["observation"]["temporal_evidence"]["observed_at_unix_ms"]
                    remaining = request["valid_until_unix_ms"] - int(time.time() * 1000)
                    if (
                        observed_ms > accepted_ms
                        and remaining
                        >= LOCAL_DISPATCH_RESERVE_MS + TRAINING_REPLY_PREPARATION_RESERVE_MS
                    ):
                        self._following_request = request
                        retain_following = True
                    return receipt
            except (TimeoutError, ConnectionError):
                continue
            except ValueError as error:
                if str(error) not in EXPIRED_POLICY_REQUESTS:
                    raise
            finally:
                # Observe while the previous action's receipt is being joined;
                # do not send a replacement action or renew its lease.
                if not retain_following:
                    self._exchange.discard_pending()
        raise TimeoutError("PX4_TRAINING_PROPOSAL_WAS_NOT_ACKNOWLEDGED")

    # 功能：
    #   仅追加合法、有限的推理耗时诊断，不更新源时间或控制截止时刻。
    # 输入：
    #   self：持有有界诊断队列的环境。
    #   sequence：本次非负整数观测序号。
    #   input_preparation_ms：输入准备耗时，单位毫秒。
    #   actor_sampling_ms：策略采样耗时，单位毫秒。
    # 输出：
    #   None：不返回业务数据。
    def record_actor_timing(self, *, sequence, input_preparation_ms, actor_sampling_ms):
        if (
            type(sequence) is not int
            or not 0 <= sequence < 2**63
            or any(
                type(value) not in (int, float) or not 0 <= value <= 600_000
                for value in (input_preparation_ms, actor_sampling_ms)
            )
        ):
            raise ValueError("PX4_TRAINING_ACTOR_TIMING_INVALID")
        self._interface_timings.append(
            {
                "stage": "actor-inference",
                "sequence": sequence,
                "input_preparation_ms": input_preparation_ms,
                "actor_sampling_ms": actor_sampling_ms,
            }
        )

    # 功能：
    #   1. 重验四轴动作并固定提案，保持原始派发余量，过期时明确标记未提交。
    #   2. 发送提案不等于执行成功，实际动作必须稍后关联原生回执。
    # 输入：
    #   self：具有活跃观测、通信端点及写入器的环境。
    #   action：归一化机体速度与偏航轴的控制提案，不是目标坐标或电机推力。
    # 输出：
    #   previous：本动作之前的环境观测。
    #   request：固定的原始输入请求。
    #   proposal：已发送的独立控制提案。
    def _submit_action(self, action: PilotAction):
        if self._pending is None or self._quiesced or self._closed:
            raise RuntimeError("PX4_TRAINING_STEP_REQUIRES_LIVE_OBSERVATION")
        self._transition_writer.check()
        if not isinstance(action, PilotAction):
            raise ValueError("PX4_TRAINING_ACTION_INVALID")
        action = PilotAction.model_validate(action.model_dump(mode="python"), strict=True)
        previous, request = self._observation, self._pending
        preparing_reply = time.perf_counter()
        proposal = TrainingProposal(
            request_sha256=sha256_json(request),
            policy_sha256=self._policy_sha256,
            expert_role=previous.sample.navigation_expert_role,
            action=action,
        )
        # 推理期间执行器可能已经进入降落；采用时的阶段不能授权此刻的新动作。
        self._check_executor_ending()
        remaining_ms = request["valid_until_unix_ms"] - int(time.time() * 1000)
        if remaining_ms < LOCAL_DISPATCH_RESERVE_MS:
            # A locally serialized reply is not an executor acknowledgment.
            # Do not spend seconds waiting for a request already known to lack
            # the runtime's unchanged dispatch reserve, nor renew its deadline.
            self._proposal_preparation_expirations = (
                getattr(self, "_proposal_preparation_expirations", 0) + 1
            )
            self._interface_timings.append(
                {
                    "stage": "proposal-rejected-before-send",
                    "sequence": self._sequence,
                    "remaining_input_lease_ms": remaining_ms,
                    "issue_code": "PX4_TRAINING_PROPOSAL_BUDGET_EXHAUSTED_BEFORE_SEND",
                    "proposal_sha256": sha256_json(proposal),
                    "source_snapshot_sha256": request["snapshot"]["snapshot_sha256"],
                    "not_submitted": True,
                }
            )
            self._exchange.discard_pending(reason="proposal-budget-exhausted")
            self._pending = self._prepared_input = None
            raise ControlPreparationExpired("PX4_TRAINING_PROPOSAL_BUDGET_EXHAUSTED_BEFORE_SEND")
        self._exchange.reply(detach_evidence(proposal))
        self._interface_timings.append(
            {
                "stage": "proposal-submitted",
                "sequence": self._sequence,
                "hash_and_reply_ms": (time.perf_counter() - preparing_reply) * 1000,
                "remaining_input_lease_ms": request["valid_until_unix_ms"]
                - int(time.time() * 1000),
            }
        )
        self._pending = None
        self._prepared_input = None
        return previous, request, proposal

    # 功能：
    #   锁定本回合唯一采集语义，拒绝未知模式或将流式模仿与同步奖励混用。
    # 输入：
    #   self：当前回合环境。
    #   mode：stream-imitation 或 reward-step。
    # 输出：
    #   None：不返回业务数据。
    def _select_collection_mode(self, mode: str) -> None:
        if type(mode) is not str or mode not in {"stream-imitation", "reward-step"}:
            raise ValueError("PX4_TRAINING_COLLECTION_MODE_INVALID")
        if getattr(self.config, "collection_expert_roles", []) and mode != "stream-imitation":
            raise ValueError("PX4_STAGE_COLLECTION_CANNOT_PRODUCE_REWARD_STEPS")
        current = getattr(self, "_collection_mode", None)
        if current is not None and current != mode:
            raise ValueError("PX4_TRAINING_COLLECTION_MODE_CHANGED_DURING_FLIGHT")
        self._collection_mode = mode

    # 功能：
    #   1. 发布提案后立即保留不可变模仿捕获，再回到感知，不等待本次执行回执。
    #   2. 限制捕获数量及总字节；发布后的归档错误不能被解释为动作未发送。
    # 输入：
    #   self：已选择流式采集的活跃环境。
    #   action：本观测对应的四轴提案。
    # 输出：
    #   receipt：提案及捕获身份，仅证明提交，不证明已经实际执行。
    def submit_stream_action(self, action: PilotAction) -> dict:
        self._select_collection_mode("stream-imitation")
        if self._observation is None or self._pending is None or self._quiesced:
            raise RuntimeError("PX4_TRAINING_STEP_REQUIRES_LIVE_OBSERVATION")
        if self._sequence >= self.config.episode_steps or len(self._stream_captures) >= 512:
            raise ValueError("PX4_TRAINING_STREAM_ACTION_LIMIT_REACHED")
        witness = self._monitor.initial_witness(
            self._observation.sample.temporal_evidence.observed_at_unix_ms
        )
        previous, request, proposal = self._submit_action(action)
        started = time.perf_counter()
        packed = pack_stream_capture(StreamActionCapture(previous, request, proposal, witness))
        total_bytes = self._retained_capture_bytes + len(packed.content)
        if total_bytes > MAXIMUM_ARCHIVE_BYTES or len(self._stream_captures) >= 512:
            raise ValueError("PX4_TRAINING_CAPTURE_ARCHIVE_FULL")
        self._stream_captures.append(packed)
        self._retained_capture_bytes = total_bytes
        self._interface_timings.append(
            {
                "stage": "stream-capture-retained",
                "sequence": previous.sequence,
                "packing_ms": (time.perf_counter() - started) * 1000,
                "capture_bytes": len(packed.content),
                "archive_bytes": total_bytes,
                "application_not_yet_asserted": True,
            }
        )
        self._sequence += 1
        return {"call_id": "model-" + sha256_json(proposal)[:24],
                "capture_sha256": packed.sha256,
                "source_snapshot_sha256": request["snapshot"]["snapshot_sha256"],
                "application_not_yet_asserted": True}

    # 功能：把独立仿真见证单独交给标签记录器，不合并到策略观测或控制上下文。
    # 输入：UNIX毫秒起止时刻，窗口不超过2秒且不得跨回合；输出：原始连续见证副本。
    # 采集失败只影响标签，调用者不得把它解释成机载感知失效。
    def label_witness_window(self, start_ms: int, end_ms: int):
        if self._monitor is None or self._quiesced or self._closed:
            raise RuntimeError("PX4_LABEL_WITNESS_REQUIRES_ACTIVE_EPISODE")
        if (type(start_ms) is not int or type(end_ms) is not int
                or not 0 < end_ms-start_ms <= 2000):
            raise ValueError("PX4_LABEL_WITNESS_WINDOW_INVALID")
        return self._monitor.window(start_ms, end_ms)

    # 功能：
    #   跳过重复或采用时已过期的源状态，始终使用采集器原截止时刻，不续租旧动作。
    # 输入：
    #   self：保存上一原生观测时刻的环境。
    #   deadline：可选绝对单调时钟截止秒数，不晚于当前时刻后两秒。
    # 输出：
    #   observation：时间严格更新且重新接纳的观测。
    def next_stream_observation(self, *, deadline: float | None = None) -> FlightObservation:
        self._select_collection_mode("stream-imitation")
        previous_stamp = self._observation.sample.temporal_evidence.observed_at_unix_ms
        maximum_deadline = time.monotonic() + 2.0
        if deadline is None:
            deadline = maximum_deadline
        elif type(deadline) not in (int, float) or not 0 < deadline <= maximum_deadline:
            raise ValueError("PX4_TRAINING_STREAM_WAIT_DEADLINE_INVALID")
        while True:
            request = self._next_request(deadline=deadline)
            stamp = request["observation"]["temporal_evidence"]["observed_at_unix_ms"]
            if stamp > previous_stamp:
                try:
                    observation = self._adopt(request)
                    return observation
                except TrainingObservationError as error:
                    if error.reason_code != "TRAINING_INPUT_SENSOR_EVIDENCE_EXPIRED":
                        raise
                    self._record_rejected_input(request, "stream-input-expired-at-adoption")
                    self._exchange.discard_pending(reason="stream-input-expired-at-adoption")
                    self._pending = self._prepared_input = None
                    previous_stamp = stamp
                    continue
            self._record_rejected_input(request, "stream-native-state-not-new")
            self._exchange.discard_pending(reason="stream-native-state-not-new")

    # 功能：
    #   只在确认静止且原生进程已结束后，关联流式捕获与实际执行账本。
    # 输入：
    #   self：持有本回合不可变流式捕获的环境。
    # 输出：
    #   visits：有实际执行归属的离线访问列表，不包含伪造的下一状态奖励。
    def finalize_stream_captures(self):
        if not self._quiesced or self._process is not None:
            raise ValueError("PX4_DEFERRED_OUTCOME_REQUIRES_CONFIRMED_QUIESCENCE")
        from .stream_capture import finalize_stream_captures

        visits = finalize_stream_captures(
            self.episode_path, self._stream_captures, write_new=_write_new
        )
        return visits

    # 功能：
    #   同步关联提案、实际动作和执行后的新观测，再采集同一时间窗的独立真值。
    # 输入：
    #   self：处于同步奖励采集模式的环境。
    #   action：本步四轴提案。
    # 输出：
    #   capture：包含真实执行和真值窗口的原生转移，尚未计算奖励。
    def capture_step(self, action: PilotAction) -> CapturedNativeTransition:
        self._select_collection_mode("reward-step")
        if self._sequence >= self.config.episode_steps:
            raise ValueError("PX4_TRAINING_REWARD_ACTION_LIMIT_REACHED")
        previous, request, proposal = self._submit_action(action)
        call_id = "model-" + sha256_json(proposal)[:24]
        awaiting_receipt = time.perf_counter()
        command, application = self._application(call_id, deadline=time.monotonic() + 2.0)
        self._interface_timings.append(
            {
                "stage": "execution-receipt",
                "sequence": self._sequence,
                "wait_ms": (time.perf_counter() - awaiting_receipt) * 1000,
            }
        )
        applied, deadline_intervened = executed_training_action(
            request["snapshot"], command, application, limits=previous.sample.pilot_control_limits
        )
        deadline = time.monotonic() + 2.0
        while True:
            following = self._next_request(deadline=deadline)
            stamp = following["observation"]["temporal_evidence"]["observed_at_unix_ms"]
            if stamp > application.accepted_at_unix_ms:
                break
            self._exchange.discard_pending()
        self._sequence += 1
        observation = self._adopt(following)
        start = previous.sample.temporal_evidence.observed_at_unix_ms
        end = observation.sample.temporal_evidence.observed_at_unix_ms
        try:
            witnesses = self._monitor.window(start, end)
        except OutcomeWindowError as error:
            self._outcome_window_failure = error.diagnostics
            raise
        capture = CapturedNativeTransition(
            previous,
            observation,
            request["snapshot"],
            proposal,
            command,
            application,
            self._last_application,
            applied,
            deadline_intervened,
            tuple(witnesses),
            self._sequence >= self.config.episode_steps,
        )
        self._last_application = application
        return capture

    # 功能：
    #   验证真实转移并排队持久化证据，任务终止或达到步数限制后请求安全停止。
    # 输入：
    #   self：同步原生训练环境。
    #   action：本步四轴控制提案。
    # 输出：
    #   step：带独立结果证据、终止与截断状态的训练步骤。
    def step(self, action: PilotAction) -> FlightStep:
        capture = self.capture_step(action)
        verifying = time.perf_counter()
        step, record = evaluate_native_transition(capture, self.verifier)
        transition_path = self.episode_path / f"transition-{self._sequence:06d}.json"
        self._transition_writer.submit(transition_path, record)
        self._interface_timings.append(
            {
                "stage": "transition-return",
                "sequence": self._sequence,
                "outcome_and_submission_ms": (time.perf_counter() - verifying) * 1000,
                "remaining_input_lease_ms": self._pending["valid_until_unix_ms"]
                - int(time.time() * 1000),
            }
        )
        if step.terminated or step.truncated:
            self.quiesce()
        return step

    # 功能：
    #   把已验证捕获编码为不可变字节并累计检查归档预算，不保留扩张的运行对象图。
    # 输入：
    #   self：持有本回合归档预算的环境。
    #   capture：同步采集得到的原生转移。
    # 输出：
    #   packed：内容绑定摘要的捕获包。
    def retain_capture(self, capture: CapturedNativeTransition) -> PackedNativeTransition:
        started = time.perf_counter()
        packed = pack_native_transition(capture)
        total_bytes = getattr(self, "_retained_capture_bytes", 0) + len(packed.content)
        if total_bytes > MAXIMUM_ARCHIVE_BYTES:
            raise ValueError("PX4_TRAINING_CAPTURE_ARCHIVE_FULL")
        self._retained_capture_bytes = total_bytes
        self._interface_timings.append(
            {
                "stage": "capture-retained",
                "sequence": capture.source_observation.sequence,
                "packing_ms": (time.perf_counter() - started) * 1000,
                "capture_bytes": len(packed.content),
                "archive_bytes": total_bytes,
            }
        )
        return packed

    # 功能：
    #   1. 原生确认落地后，固定有界捕获列表，逐步验证身份、计算奖励和保存结果。
    #   2. 仅全部成功才写完整回执；多个转移文件不是跨文件原子事务。
    # 输入：
    #   self：已确认静止且进程结束的环境。
    #   captures：最多二百五十六份不可变同步捕获包。
    # 输出：
    #   visits：按原始执行顺序得到的有奖励访问列表。
    def finalize_captures(self, captures: list[PackedNativeTransition]):
        if not self._quiesced or self._process is not None:
            raise ValueError("PX4_DEFERRED_OUTCOME_REQUIRES_CONFIRMED_QUIESCENCE")
        from .student_collection import StudentVisit

        if type(captures) not in (list, tuple) or not 1 <= len(captures) <= 256:
            raise ValueError("PX4_DEFERRED_CAPTURE_COUNT_INVALID")
        captures = tuple(captures)
        if any(
            type(packed) is not PackedNativeTransition or type(packed.content) is not bytes
            for packed in captures
        ):
            raise ValueError("PX4_DEFERRED_CAPTURE_NOT_PACKED")
        archive_bytes = sum(len(packed.content) for packed in captures)
        if archive_bytes > MAXIMUM_ARCHIVE_BYTES:
            raise ValueError("PX4_TRAINING_CAPTURE_ARCHIVE_FULL")
        visits = []
        for index, packed in enumerate(captures):
            capture = packed.unpack()
            if (
                capture.source_observation.episode_id != self.episode_path.name
                or capture.source_observation.sequence != index
            ):
                raise ValueError("PX4_DEFERRED_CAPTURE_IDENTITY_MISMATCH")
            step, record = evaluate_native_transition(capture, self.verifier)
            record["outcome_evaluation_phase"] = "after-confirmed-native-landing"
            _write_new(self.episode_path / f"transition-{index + 1:06d}.json", record)
            visits.append(StudentVisit(capture.source_observation, capture.proposal.action, step))
        _write_new(
            self.episode_path / "deferred-outcome-receipt.json",
            {
                "captured": len(captures),
                "evaluated": len(visits),
                "persisted": len(visits),
                "complete": True,
                "phase": "after-confirmed-native-landing",
                "qualified_for_flight": False,
                "retention": "content-bound-json-bytes",
                "archive_bytes": archive_bytes,
            },
        )
        return visits

    # 功能：
    #   1. 撤销学习器控制，等待原生落地或明确未起飞证明，再排空证据写入器。
    #   2. 超时或证明缺失时保持未静止状态，禁止重置或优化器据此开始训练。
    # 输入：
    #   self：可能仍具有原生进程和控制通道的环境。
    # 输出：
    #   None：不返回业务数据。
    def quiesce(self) -> None:
        if self._quiesced:
            return
        if self._exchange:
            self._exchange.close()  # Immediately revoke all future learner replies.
        self._pending = None
        self._following_request = None
        deadline = time.monotonic() + self.config.quiesce_timeout_seconds
        abort = self.simulation / "live_abort.request.json"
        while self._process is not None and self._process.poll() is None:
            if (self.simulation / "runtime-phase.json").exists() and not abort.exists():
                with suppress(FileExistsError):  # Preserve an existing safety/operator abort.
                    _write_new(
                        abort, {"reason": "offline-training-rollout-quiesce", "world_paused": False}
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError("PX4_TRAINING_SAFE_LANDING_NOT_CONFIRMED")
            time.sleep(0.05)
        if self._process is not None:
            preflight_path = self.simulation / "simulation-preflight-stop.json"
            preflight = read_object(preflight_path) if preflight_path.is_file() else {}
            never_started = (
                preflight.get("run_directory") == str(self.simulation.resolve())
                and preflight.get("flight_vehicle_spawn_attempted") is False
                and preflight.get("all_started_processes_exited") is True
                and preflight.get("terminal_state") == "NOT_STARTED"
                and preflight.get("qualification_granted") is False
            )
            lifecycle = (
                {}
                if never_started
                else read_object(self.simulation / "native-terminal-lifecycle.json")
            )
            if not never_started and (
                lifecycle.get("terminal_state") != "ON_GROUND"
                or lifecycle.get("landing_confirmed") is not True
                or lifecycle.get("safe_to_stop_watchdog") is not True
            ):
                raise RuntimeError("PX4_TRAINING_NATIVE_STOP_EVIDENCE_MISSING")
        if self._monitor:
            self._monitor.close()
        if getattr(self, "_phase_observer", None) is not None:
            summary = self._phase_observer.close()
            path = self.episode_path / "phase-observer-receipt.json"
            if not path.exists():
                _write_new(path, summary)
            self._phase_observer = None
        if getattr(self, "_rollout_stop_reason", None) is not None:
            # Read final diagnostics only after native ground/NOT_STARTED proof.
            # They explain the stop; they cannot grant that proof or a reward.
            timing_path = self.simulation / "offboard_timing.json"
            try:
                timing, timing_sha256 = read_evidence_object(
                    timing_path, limit=MAX_TRAINING_ROW_BYTES
                )
                failure = timing.get("failure")
                status = timing.get("status")
                self._rollout_stop_reason["executor_outcome"] = {
                    "timing_sha256": timing_sha256,
                    "failure": failure[:2048] if isinstance(failure, str) else None,
                    "failure_truncated": isinstance(failure, str) and len(failure) > 2048,
                    "status": status
                    if isinstance(status, str)
                    and status in {"passed", "failed", "completed", "aborted"}
                    else "UNKNOWN",
                }
            except (OSError, ValueError, UnicodeError, RecursionError):
                self._rollout_stop_reason["executor_outcome"] = {"status": "unavailable"}
            stop_path = self.episode_path / "rollout-stop-reason.json"
            if not stop_path.exists():
                _write_new(stop_path, self._rollout_stop_reason)
        if getattr(self, "_outcome_window_failure", None) is not None:
            path = self.episode_path / "independent-witness-window-failure.json"
            if not path.exists():
                _write_new(path, self._outcome_window_failure)
        if self._log:
            self._log.close()
        if getattr(self, "_transition_writer", None) is not None:
            # Optimizer updates cannot proceed before every transition is
            # durable. Drain only after the native vehicle is confirmed down.
            summary = self._transition_writer.close()
            path = self.episode_path / "transition-writer-receipt.json"
            if not path.exists():
                _write_new(path, summary)
            self._transition_writer = None
        if getattr(self, "_receipt_monitor", None) is not None:
            self._receipt_monitor.close()
            self._receipt_monitor = None
        if getattr(self, "_interface_timings", None) and self.episode_path is not None:
            path = self.episode_path / "training-interface-timing.json"
            if not path.exists():
                _write_new(
                    path,
                    {
                        "rows": list(self._interface_timings),
                        "retention": "latest-512-interface-events",
                        "requests_received": getattr(self, "_input_received_count", 0),
                        "input_rejection_counts": getattr(self, "_input_rejection_counts", {}),
                        "proposal_preparation_expirations": getattr(
                            self, "_proposal_preparation_expirations", 0
                        ),
                        "collection_mode": getattr(self, "_collection_mode", None),
                        "input_admission_budget_ms": self._input_admission_budget_ms(),
                        "post_preparation_dispatch_reserve_ms": LOCAL_DISPATCH_RESERVE_MS,
                    },
                )
        self._process = self._exchange = self._monitor = self._log = None
        self._quiesced = True

    # 功能：
    #   关闭控制环境并释放视觉资源；保留首个停止错误及后续清理错误，不伪造静止状态。
    # 输入：
    #   self：需要最终关闭的环境。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        error = None
        try:
            self.quiesce()
        except BaseException as caught:
            error = caught
            raise
        finally:
            self._closed = True
            visual = getattr(self, "visual", None)
            if visual is not None:
                try:
                    visual.close()
                except BaseException as cleanup_error:
                    if error is None:
                        raise
                    error.add_note("PX4 visual cleanup also failed: " + repr(cleanup_error))


# 功能：
#   解析序列化配置后构造 Linux 专用仿真环境，不在此入口启动飞行。
# 输入：
#   config：标准配置字段组成的字典。
# 输出：
#   environment：已绑定当前资产与验证器的原生仿真训练环境。
def create_environment(config: dict) -> Px4GazeboTrainingEnvironment:
    environment = Px4GazeboTrainingEnvironment(Px4TrainingConfig.model_validate(config))
    return environment
