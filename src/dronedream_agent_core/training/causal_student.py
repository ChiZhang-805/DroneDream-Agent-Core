"""Deterministic recurrent student for offline simulation data collection."""

import hashlib
import time

import numpy as np
import torch

from dronedream_plugin_sdk.protocol import decode_json

from ..causal_control import (
    CONTROL_HISTORY_CONTRACT_SHA256,
    CONTROL_HISTORY_WIDTH,
    CONTROL_REALTIME_WIDTH,
    CONTROL_STATE_WIDTH,
    CausalControlHistory,
)
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..local_expert_harness import NAVIGATION_EXPERT_ROLES
from .causal_policy import CausalExample, CausalPilotPolicy, _validated_observation, example_tensors
from .flight_environment import MODES, FlightObservation, PilotAction


# 功能：
#   在改变历史或提交序号前重新验证并冻结当前与先前观测，拒绝旧语义及损坏的嵌套实例。
# 输入：
#   observation：回合、任务、地图及独立观测绑定的输入。
# 输出：
#   validated：已重新检查并独立持有记录容器的飞行观测。
def _validated_flight_observation(observation: FlightObservation) -> FlightObservation:
    if (not isinstance(observation, FlightObservation)
            or not isinstance(observation.prior_observations, list)
            or len(observation.prior_observations) > 32):
        raise ValueError("CAUSAL_STUDENT_OBSERVATION_INVALID")
    payload = observation.model_dump(mode="python", exclude={"sample", "prior_observations"})
    payload["sample"] = _validated_observation(observation.sample)
    payload["prior_observations"] = [_validated_observation(prior)
                                     for prior in observation.prior_observations]
    validated = FlightObservation.model_validate(payload, strict=True)
    return validated


# 功能：
#   按独立来源时序加入已经验证的先前及当前记录，重复旧行不用于填满窗口。
# 输入：
#   history：训练和部署共同的因果历史。
#   observation：已通过本模块入口验证的观测。
# 输出：
#   None：不返回业务数据。
def _append_validated_history(history: CausalControlHistory,
                              observation: FlightObservation) -> None:
    for prior in observation.prior_observations:
        latest = history.history.latest
        if latest is None or prior.temporal_evidence.observed_at_unix_ms > (
            latest.observed_at_unix_ms
        ):
            history.append(prior.temporal_evidence, prior.state_features,
                           prior.realtime_features, prior.realtime_valid_mask)
    sample = observation.sample
    history.append(sample.temporal_evidence, sample.state_features,
                   sample.realtime_features, sample.realtime_valid_mask)


# 功能：
#   为 PPO 与确定性学生提供相同的观测重验和来源排序入口，拒绝旧契约或坐标候选。
# 输入：
#   history：目标因果窗口。
#   observation：可能经过 model_copy 或外部修改的观测实例。
# 输出：
#   None：不返回业务数据。
def append_flight_history(history: CausalControlHistory, observation: FlightObservation) -> None:
    validated = _validated_flight_observation(observation)
    _append_validated_history(history, validated)


class CausalStudent:
    """One routed expert with current-source recurrent history and deterministic inference."""

    # 功能：
    #   将指定专家模型设置为 CPU 评价模式并预热；预热零张量不进入观测历史或提供移动许可。
    # 输入：
    #   self：新建的离线学生采集器。
    #   model：已配置的因果策略模型。
    #   expert_role：本采集器负责的导航、精细机动或恢复专家角色。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, model: CausalPilotPolicy, *, expert_role: str):
        if not isinstance(expert_role, str) or expert_role not in NAVIGATION_EXPERT_ROLES:
            raise ValueError("CAUSAL_STUDENT_EXPERT_ROLE_INVALID")
        self.model, self.expert_role = model.cpu().eval(), expert_role
        self.history = CausalControlHistory(model.config.history_length)
        self._episode = None
        self._last_sequence = -1
        self._last_observation_stamp = -1
        self.last_timing = None
        # Initialize kernels before any environment starts. These tensors are
        # not observations, training examples, or movement authorizations.
        config = model.config
        columns = [torch.zeros(1, CONTROL_STATE_WIDTH),
                   torch.zeros(1, CONTROL_REALTIME_WIDTH),
                   torch.zeros(1, CONTROL_REALTIME_WIDTH),
                   torch.zeros(1, config.history_length, CONTROL_HISTORY_WIDTH),
                   torch.zeros(1, config.history_length)]
        if config.visual_feature_count:
            columns.append(torch.zeros(1, config.visual_feature_count))
        with torch.inference_mode():
            self.model(*columns)

    # 功能：
    #   运行离线 Torch 参考推理；ONNX 子类使用自己的实现，不在失败后回退到这里。
    # 输入：
    #   self：当前学生采集器。
    #   columns：按共用输入契约排序的张量。
    # 输出：
    #   outputs：候选分数、模式、风险和四轴输出的元组。
    def _predict(self, columns):
        outputs = self.model(*columns)
        return outputs

    # 功能：
    #   消费更新的物理观测并提议控制；历史未准备好时保持悬停，坏输入不得推进提交序号。
    # 输入：
    #   self：当前学生采集器。
    #   observation：与任务、回合及地图绑定的实时观测。
    # 输出：
    #   result：悬停或模型提议的四轴动作，仍需外部安全层和实际执行回执。
    @torch.inference_mode()
    def __call__(self, observation: FlightObservation) -> PilotAction:
        started = time.perf_counter()
        self.last_timing = None
        observation = _validated_flight_observation(observation)
        if observation.sample.navigation_expert_role != self.expert_role:
            raise ValueError("CAUSAL_STUDENT_EXPERT_ROUTING_MISMATCH")
        identity = observation.episode_id, observation.mission_id, observation.map_sha256
        if self._episode != identity:
            self.history.clear()
            self._episode, self._last_sequence = identity, -1
            self._last_observation_stamp = -1
        stamp = observation.sample.temporal_evidence.observed_at_unix_ms
        # Sequence indexes submitted proposals, not discarded computations.
        # A newer source can replace an expired preparation at that same index;
        # repeated/regressing sensor time can never advance recurrent memory.
        if observation.sequence < self._last_sequence or stamp <= self._last_observation_stamp:
            raise ValueError("CAUSAL_STUDENT_OBSERVATION_REPLAYED")
        _append_validated_history(self.history, observation)
        self._last_sequence = observation.sequence
        self._last_observation_stamp = stamp
        if not self.history.ready:
            self.last_timing = {
                "sequence": observation.sequence,
                "input_preparation_ms": (time.perf_counter() - started) * 1000,
                "actor_sampling_ms": 0.,
            }
            result = PilotAction(mode="hold", axes=[0.] * 4)
            return result
        rows, mask = self.history.values()
        columns = example_tensors([CausalExample(observation.sample, rows, mask,
                                                 "student-live-observation")],
                                  self.model.config.visual_feature_count)
        prepared = time.perf_counter()
        _, modes, risk, axes = self._predict(columns)
        if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
               for value, shape in ((modes, (1, 4)), (risk, (1, 1)), (axes, (1, 4)))):
            raise ValueError("CAUSAL_STUDENT_OUTPUT_SHAPE_INVALID")
        if not all(torch.isfinite(value).all() for value in (modes, risk, axes)):
            raise ValueError("CAUSAL_STUDENT_NONFINITE_OUTPUT")
        if not ((risk >= 0) & (risk <= 1)).all() or not ((axes >= -1) & (axes <= 1)).all():
            raise ValueError("CAUSAL_STUDENT_OUTPUT_RANGE_INVALID")
        mode = MODES[int(modes[0].argmax())]
        result = PilotAction(mode=mode,
                             axes=axes[0].tolist() if mode == "pilot-control" else [0.] * 4)
        self.last_timing = {
            "sequence": observation.sequence,
            "input_preparation_ms": (prepared - started) * 1000,
            "actor_sampling_ms": (time.perf_counter() - prepared) * 1000,
        }
        return result


class OnnxCausalStudent(CausalStudent):
    """Collect with the actual CPU deployment graph, not the training executor.

    Checkpoint and ONNX are jointly bound by the producer receipt. Startup
    probes check conversion before physics; no optimizer or graph export runs
    during collection. Torch remains only the offline reference, never a
    silent inference fallback when the selected graph fails.
    """

    # 功能：
    #   先验证严格回执、图结构、元数据及与基座的数值一致性，再允许开始离线采集。
    # 输入：
    #   self：新建的 ONNX 学生采集器。
    #   model：用于数值对照的因果基座。
    #   expert_role：当前专家角色。
    #   onnx_content：最多二百五十六 MiB 的实际模型字节。
    #   receipt_content：最多四 MiB、无重复键或非有限数值的配套回执字节。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, model, *, expert_role, onnx_content: bytes, receipt_content: bytes):
        import onnxruntime as ort

        from .artifact_assembly import validate_embedded_graph

        if (not isinstance(onnx_content, bytes) or not 0 < len(onnx_content) <= 256 * 1024 * 1024
                or not isinstance(receipt_content, bytes)):
            raise ValueError("CAUSAL_STUDENT_ONNX_CONTENT_INVALID")
        receipt = decode_json(receipt_content, limit=4 * 1024 * 1024)
        self.artifact_sha256 = hashlib.sha256(onnx_content).hexdigest()
        if (not isinstance(receipt, dict)
                or receipt.get("artifact_sha256") != self.artifact_sha256
                or receipt.get("expert_role") != expert_role
                or receipt.get("config") != model.config.model_dump()
                or receipt.get("feature_contract_sha256")
                != CURRENT_POLICY_FEATURE_CONTRACT_SHA256):
            raise ValueError("CAUSAL_STUDENT_ONNX_RECEIPT_MISMATCH")
        self.input_names = ["state_features", "realtime_features", "realtime_valid_mask",
                            "control_history", "control_history_mask"]
        if model.config.visual_feature_count:
            self.input_names.append("visual_features")
        self.output_names = ["candidate_scores", "action_scores", "risk_score", "pilot_control"]
        validate_embedded_graph(onnx_content, input_names=self.input_names,
                                output_names=self.output_names)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(onnx_content, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        expected = {"architecture": "causal-gru-control",
                    "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
                    "history_contract_sha256": CONTROL_HISTORY_CONTRACT_SHA256,
                    "history_length": str(model.config.history_length),
                    "config_sha256": sha256_json(model.config)}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("CAUSAL_STUDENT_ONNX_METADATA_MISMATCH")
        super().__init__(model, expert_role=expert_role)
        shapes = [(1, CONTROL_STATE_WIDTH), (1, CONTROL_REALTIME_WIDTH),
                  (1, CONTROL_REALTIME_WIDTH),
                  (1, model.config.history_length, CONTROL_HISTORY_WIDTH),
                  (1, model.config.history_length)]
        if model.config.visual_feature_count:
            shapes.append((1, model.config.visual_feature_count))
        # Deterministic numerical probes are not fabricated sensor observations.
        random = np.random.default_rng(805)
        self.startup_maximum_absolute_error = 0.
        for _ in range(3):
            columns = [torch.from_numpy(random.uniform(-.5, .5, shape).astype(np.float32))
                       for shape in shapes]
            columns[2].fill_(1.)
            columns[4].fill_(1.)
            with torch.inference_mode():
                reference, actual = self.model(*columns), self._predict(columns)
            for original, exported in zip(reference, actual, strict=True):
                error = float((original - exported).abs().max())
                if (not torch.isfinite(original).all() or not torch.isfinite(exported).all()
                        or error > 1e-5):
                    raise ValueError("CAUSAL_STUDENT_ONNX_DIFFERS_FROM_CHECKPOINT")
                self.startup_maximum_absolute_error = max(
                    self.startup_maximum_absolute_error, error)

    # 功能：
    #   只运行所选 CPU 部署图并还原张量结果；推理失败不能静默切回 Torch。
    # 输入：
    #   self：当前 ONNX 学生。
    #   columns：按固定名称顺序排列的输入张量。
    # 输出：
    #   outputs：实际 ONNX 输出转换成的 Torch 张量元组。
    def _predict(self, columns):
        inputs = {name: value.detach().cpu().numpy()
                  for name, value in zip(self.input_names, columns, strict=True)}
        outputs = tuple(torch.from_numpy(value)
                        for value in self.session.run(self.output_names, inputs))
        return outputs
