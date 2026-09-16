"""Bounded proposed-action counterfactuals on authentic student observations.

These labels train only an action-risk critic, never behaviour cloning. They
are nominal swept-geometry supervision, not executed counterfactual flights or
calibrated collision probabilities. All counterfactual work runs after landing.
"""

from itertools import product

from ..contracts import NormalizedPilotControl
from ..hashing import sha256_json
from ..local_policy_training import LocalPolicyTrainingSample
from ..pilot_control_mapping import action_risk_features
from .dagger import ProposedActionRisk
from .evidence_snapshot import detach_evidence
from .flight_environment import FlightObservation, PilotAction


# 功能：
#   枚举零动作及两档幅度的四轴组合，用于离线风险比较，不产生坐标目标或飞控调用。
# 输入：
#   无。
# 输出：
#   probes：一百六十一种不重复的归一化连续动作。
def bounded_action_probes() -> tuple[PilotAction, ...]:
    # Include true zero velocity, both signs and two nonzero magnitudes on
    # every translational/yaw axis, including coupled manoeuvres. No positions.
    axes = {(0.0, 0.0, 0.0, 0.0)}
    for magnitude in (0.5, 1.0):
        axes.update(
            tuple(magnitude * v for v in direction)
            for direction in product((-1.0, 0.0, 1.0), repeat=4)
        )
    probes = tuple(PilotAction(mode="pilot-control", axes=list(row)) for row in sorted(axes))
    return probes


# 功能：
#   1. 在评估前冻结并重验整批四轴探针与观测，隔离回调对来源及后续动作的改写。
#   2. 将名义未执行探针绑定真实观测和物理尺度，只生成风险监督，不授予行为模仿资格。
# 输入：
#   observation：不含教师真值的学生观测。
#   actions：至多二百五十六个不重复的连续动作，使用列表或元组。
#   assessor：对独立输入副本产生名义几何风险回执的离线评估器。
# 输出：
#   result：逐次产出的风险样本与对应来源记录二元组。
def counterfactual_risk_samples(observation, actions, assessor):
    if type(actions) not in (list, tuple) or not 1 <= len(actions) <= 256:
        raise ValueError("ACTION_RISK_PROBE_COUNT_INVALID")
    if not isinstance(observation, FlightObservation) or not callable(assessor):
        raise ValueError("ACTION_RISK_OBSERVATION_OR_ASSESSOR_INVALID")
    observation = FlightObservation.model_validate(observation.model_dump())
    frozen = []
    seen = set()
    for action in actions:
        if not isinstance(action, PilotAction):
            raise ValueError("ACTION_RISK_PROBE_TYPE_INVALID")
        action = PilotAction.model_validate(action.model_dump())
        action_sha = sha256_json(action)
        if action.mode != "pilot-control" or action_sha in seen:
            raise ValueError("ACTION_RISK_CONTINUOUS_UNIQUE_PROBES_REQUIRED")
        seen.add(action_sha)
        frozen.append((action, action_sha))
    limits = observation.sample.pilot_control_limits
    if limits is None:
        raise ValueError("ACTION_RISK_PHYSICAL_LIMITS_REQUIRED")
    digest = sha256_json(observation)
    for action, action_sha in frozen:
        # 保留一份不外借的基准；评估器和生成器消费者都不能改写下一条监督的输入。
        risk = assessor(detach_evidence(observation), detach_evidence(action))
        if not isinstance(risk, ProposedActionRisk):
            raise ValueError("ACTION_RISK_COUNTERFACTUAL_BINDING_MISMATCH")
        risk = ProposedActionRisk.model_validate(risk.model_dump())
        if (
            risk.source != "swept-geometry"
            or risk.observation_sha256 != digest
            or risk.proposed_action_sha256 != action_sha
        ):
            raise ValueError("ACTION_RISK_COUNTERFACTUAL_BINDING_MISMATCH")
        physical = action_risk_features(
            NormalizedPilotControl(
                **dict(
                    zip(
                        ("forward_axis", "right_axis", "up_axis", "yaw_axis"),
                        action.axes,
                        strict=True,
                    )
                )
            ),
            limits,
            harness_scale=1.0,
        )
        label = LocalPolicyTrainingSample.model_validate(
            {
                **observation.sample.model_dump(),
                "target_action_index": 11,
                "target_pilot_control": action.axes,
                "risk_target": risk.risk,
                "risk_proposed_control": list(physical),
            }
        )
        result = (
            label,
            {
                "observation_sha256": digest,
                "episode_id": observation.episode_id,
                "sequence": observation.sequence,
                "proposed_action": action.model_dump(),
                "assessment": risk.model_dump(),
                "label_sha256": sha256_json(label),
                "counterfactual_action_executed": False,
                "behavior_cloning_label": False,
            },
        )
        yield result
