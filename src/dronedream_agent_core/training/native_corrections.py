"""Bind grounded student visits to independent native correction witnesses."""

from pathlib import Path

from ..contracts import (
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
)
from ..control_execution_evidence import ControlApplicationRecord, validate_application_binding
from ..hashing import sha256_json
from ..plugin_files import check_plain_plugin_path, portable_plugin_path
from .counterfactual_teacher import CounterfactualTeacher
from .evidence_files import read_evidence_object
from .executed_control import executed_training_action
from .flight_environment import FlightObservation, FlightStep
from .grounded_teacher_context import copy_grounded_state, grounded_teacher_context
from .observations import compile_training_observation
from .policy_exchange import TrainingProposal


class GroundedNativeCorrections:
    """Annotate ended native episodes with independent geometry, never during live motion."""
    # 功能：
    #   绑定离线回合根目录、独立教师、地图和确切学生权重，初始化单上下文缓存。
    # 输入：
    #   self：原生教师纠正器。
    #   episode_root：停止后证据所在根目录。
    #   teacher：只做反事实评估、不发送控制的教师。
    #   map_sha256：该批采集使用的语义地图摘要。
    #   student_policy_sha256：学生策略权重摘要。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, episode_root: Path, teacher: CounterfactualTeacher, *,
                 map_sha256: str, student_policy_sha256: str):
        if not isinstance(episode_root, Path):
            raise ValueError("DAGGER_NATIVE_ROOT_INVALID")
        for digest in (map_sha256, student_policy_sha256):
            if (type(digest) is not str or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("DAGGER_NATIVE_IDENTITY_INVALID")
        self.root = episode_root.absolute()
        check_plain_plugin_path(self.root)
        self.teacher, self.map_sha256 = teacher, map_sha256
        self.student_policy_sha256 = student_policy_sha256
        self.receipts: list[dict] = []
        self._key, self._state = None, None

    # 功能：
    #   1. 每次读取都重查停止状态、策略地图、原输入、执行动作和独立见证，即使缓存命中。
    #   2. 将及时的仿真真值投影为离线教师上下文，不合并到学生传感器输入。
    # 输入：
    #   self：包含来源约束和单上下文缓存的纠正器。
    #   observation：需要生成监督信号的原始学生观测。
    # 输出：
    #   state：独立持有内容的、与证据绑定的离线上下文。
    def _context(self, observation):
        if not isinstance(observation, FlightObservation):
            raise ValueError("DAGGER_NATIVE_OBSERVATION_INVALID")
        observation = FlightObservation.model_validate(observation.model_dump())
        # 先检查原路径再打开，不能 resolve 掉链接后丢失其真实来源。
        name = portable_plugin_path(observation.episode_id)
        if "/" in name:
            raise ValueError("DAGGER_NATIVE_EPISODE_OUTSIDE_ROOT")
        episode = self.root / name
        terminal, _ = read_evidence_object(
            episode / "flight/simulation/native-terminal-lifecycle.json")
        if (
            terminal.get("terminal_state") != "ON_GROUND"
            or terminal.get("landing_confirmed") is not True
            or terminal.get("safe_to_stop_watchdog") is not True
        ):
            raise ValueError("DAGGER_NATIVE_LANDING_NOT_CONFIRMED")
        if observation.map_sha256 != self.map_sha256:
            raise ValueError("DAGGER_NATIVE_MAP_MISMATCH")
        reset, _ = read_evidence_object(episode / "reset.json")
        if (
            reset["policy_sha256"] != self.student_policy_sha256
            or reset["config"]["asset_sha256"]["semantic"] != self.map_sha256
            or reset["mission_id"] != observation.mission_id
        ):
            raise ValueError("DAGGER_NATIVE_COLLECTION_IDENTITY_MISMATCH")
        path = episode / f"transition-{observation.sequence + 1:06d}.json"
        transition, content_digest = read_evidence_object(path, limit=4 * 1024 * 1024)
        # 配置也是上下文条件：加速度边界改变时不能复用按旧边界算出的预测不确定度。
        key = (sha256_json(observation), content_digest, sha256_json(terminal), sha256_json(reset),
               sha256_json(self.teacher.config), self.teacher.geometry_sha256)
        if key == self._key:
            state = copy_grounded_state(self._state)
            return state
        if sha256_json(transition["source_observation"]) != key[0]:
            raise ValueError("DAGGER_NATIVE_STUDENT_STATE_MISMATCH")
        proposal = TrainingProposal.model_validate(transition["proposal"])
        if proposal.policy_sha256 != self.student_policy_sha256:
            raise ValueError("DAGGER_NATIVE_STUDENT_WEIGHTS_MISMATCH")
        step = FlightStep.model_validate(transition["step"])
        step.validate_transition(observation, proposal.action)
        command = RuntimeLocalSafetyCommand.model_validate(transition["command"])
        application = ControlApplicationRecord.model_validate(transition["application"])
        validate_application_binding(command, application)
        if sha256_json(command) != step.applied_command_sha256:
            raise ValueError("DAGGER_NATIVE_APPLIED_COMMAND_MISMATCH")
        snapshot = transition["source_snapshot"]
        compiled = compile_training_observation(
            snapshot, now_unix_ms=snapshot["control_reference_observed_at_unix_ms"]
        )
        # The frozen visual encoder adds image features separately. Its identity
        # is preserved in the observation and collection manifest, not invented
        # by this map/physics teacher.
        exclude = {"visual_features", "source_visual_sha256"}
        if compiled.model_dump(exclude=exclude) != observation.sample.model_dump(exclude=exclude):
            raise ValueError("DAGGER_NATIVE_FEATURE_SNAPSHOT_MISMATCH")
        applied, expired = executed_training_action(
            snapshot, command, application, limits=observation.sample.pilot_control_limits)
        if applied != step.applied_action:
            raise ValueError("DAGGER_NATIVE_APPLIED_ACTION_MISMATCH")
        declared_expired = transition.get("expired_model_proposal_replaced_by_safety_hold", False)
        if type(declared_expired) is not bool or declared_expired != expired:
            raise ValueError("DAGGER_NATIVE_DEADLINE_INTERVENTION_MISMATCH")
        if application.transport == "velocity-ned" and not application.model_authorized:
            raise ValueError("DAGGER_NATIVE_STUDENT_MOTION_NOT_AUTHORIZED")
        receipt = transition["outcome_receipt"]
        if (
            sha256_json(receipt) != transition["step"]["evidence"]["verifier_receipt_sha256"]
            or receipt["geometry_sha256"] != self.teacher.geometry_sha256
        ):
            raise ValueError("DAGGER_NATIVE_OUTCOME_BINDING_MISMATCH")
        source_ms = observation.sample.temporal_evidence.observed_at_unix_ms
        if (
            receipt["start_ms"] != source_ms
            or receipt["end_ms"] != step.observation.sample.temporal_evidence.observed_at_unix_ms
            or receipt["goal_revision"] != sha256_json(snapshot["goal_position_m"])
        ):
            raise ValueError("DAGGER_NATIVE_OUTCOME_WINDOW_MISMATCH")
        witnesses = [
            RuntimeLocalSafetyObservation.model_validate(row) for row in receipt["observations"]
        ]
        witnesses = [row for row in witnesses if row.observed_at_unix_ms <= source_ms]
        if not witnesses:
            raise ValueError("DAGGER_NATIVE_INITIAL_WITNESS_MISSING")
        witness = max(witnesses, key=lambda row: row.observed_at_unix_ms)
        state, context = grounded_teacher_context(
            observation, snapshot, witness, application, self.teacher,
            binding={"terminal_receipt_sha256": sha256_json(terminal),
                     "transition_content_sha256": key[1], "reset_sha256": sha256_json(reset)})
        self.receipts.append({"context": context, "context_sha256": state.context_sha256})
        self._key, self._state = key, state
        state = copy_grounded_state(state)
        return state

    # 功能：
    #   以重新核对过的落地证据生成教师纠正，并保存对应独立回执。
    # 输入：
    #   self：离线纠正器。
    #   observation：原始学生观测。
    # 输出：
    #   correction：与该观测及教师回执绑定的替代动作标签。
    def correction(self, observation):
        correction, receipt = self.teacher.correction(observation, self._context(observation))
        self.receipts.append({"receipt": receipt, "receipt_sha256": sha256_json(receipt)})
        return correction

    # 功能：
    #   评估学生原提案而非教师替代动作，保存可提供的反事实风险回执。
    # 输入：
    #   self：离线纠正器。
    #   observation：原始学生观测。
    #   action：需要评估的学生提案。
    # 输出：
    #   assessment：名义动力学与几何假设下的风险标签。
    def risk(self, observation, action):
        assessment, receipt = self.teacher.risk(observation, self._context(observation), action)
        if receipt is not None:
            self.receipts.append({"receipt": receipt, "receipt_sha256": sha256_json(receipt)})
        return assessment
