"""Independent geometric/telemetry reward evidence, excluded from learner inputs.

No model declares its own success. This per-step verifier does not grant mission
completion: that remains the full runtime verifier's job after confirmed land.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..collision import vehicle_clearance
from ..contracts import RuntimeLocalSafetyObservation
from ..hashing import sha256_json
from .evidence_snapshot import detach_evidence
from .flight_environment import RewardEvidence
from .geometry_inputs import finite_metric, metric_vector
from .swept_geometry import SweptMapGeometry


@dataclass(frozen=True)
class OutcomeEnvelope:
    body_radius_m: float
    body_height_m: float
    minimum_enu_m: tuple[float, float, float]
    maximum_enu_m: tuple[float, float, float]

    # 功能：
    #   验证正机体尺度和有序围栏，固定边界为独立元组，不允许布尔值充当物理量。
    # 输入：
    #   self：待验证的机体及围栏包络。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self):
        minimum = metric_vector(self.minimum_enu_m)
        maximum = metric_vector(self.maximum_enu_m)
        if (not all(finite_metric(v) for v in (self.body_radius_m, self.body_height_m))
                or self.body_radius_m <= 0 or self.body_height_m <= 0
                or any(a >= b for a, b in zip(minimum, maximum, strict=True))):
            raise ValueError("OUTCOME_VEHICLE_OR_GEOFENCE_INVALID")
        object.__setattr__(self, "minimum_enu_m", minimum)
        object.__setattr__(self, "maximum_enu_m", maximum)


# 功能：
#   以至多两厘米采样间距求扫掠安全距离下界，并减去半间隔以覆盖采样点之间的薄障碍。
# 输入：
#   start、end：线段起止三维米制位置。
#   primitives：当前地图的非空几何基元列表。
#   radius_m：无人机正水平半径，米。
#   half_height_m：无人机正半高，米。
# 输出：
#   minimum：包含采样间隙保守扣减的最小有限距离下界，米。
def swept_clearance_bound(start, end, primitives, *, radius_m, half_height_m):
    start, end = metric_vector(start), metric_vector(end)
    if (not all(finite_metric(v) and v > 0 for v in (radius_m, half_height_m))
            or type(primitives) is not list or not primitives):
        raise ValueError("OUTCOME_GEOMETRY_INVALID")
    distance = math.dist(start, end)
    if not math.isfinite(distance) or distance > 10:
        raise ValueError("OUTCOME_POSE_JUMP_NOT_VERIFIABLE")
    count = max(1, math.ceil(distance / .02))
    minimum = math.inf
    for i in range(count + 1):
        point = tuple(a + (b - a) * i / count for a, b in zip(start, end, strict=True))
        for primitive in primitives:
            value = vehicle_clearance(point, primitive, radius_m=radius_m,
                                      half_height_m=half_height_m)
            if not math.isfinite(value):
                raise ValueError("OUTCOME_GEOMETRY_INVALID")
            minimum = min(minimum, value - distance / (2 * count))
    return minimum


class SimulationOutcomeVerifier:
    # 功能：
    #   固定独立评分地图及包络，编译全部障碍并绑定源地图摘要，不将特权真值交给策略。
    # 输入：
    #   self：待初始化评分器。
    #   static_primitives：静态地图几何基元。
    #   envelope：机体与围栏边界。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, static_primitives: list[dict], envelope: OutcomeEnvelope):
        if type(static_primitives) is not list or not static_primitives:
            raise ValueError("OUTCOME_VERIFIER_REQUIRES_BOUND_MAP_GEOMETRY")
        if type(envelope) is not OutcomeEnvelope:
            raise ValueError("OUTCOME_ENVELOPE_INVALID")
        self.primitives = detach_evidence(static_primitives)
        self.envelope = envelope = replace(envelope)
        self.geometry_sha256 = sha256_json(self.primitives)
        self.geometry = SweptMapGeometry(self.primitives, radius_m=envelope.body_radius_m,
                                         half_height_m=envelope.body_height_m / 2)

    # 功能：
    #   1. 重验并冻结有界连续见证，拒绝过旧观测、缺帧和动态身份冲突。
    #   2. 计算静态与动态扫掠距离、围栏及目标距离，将全部奖励事实绑定到同一回执。
    #   3. 未知能耗保持未知，步骤评分不宣布整项任务完成。
    # 输入：
    #   self：持有独立地图和包络的评分器。
    #   observations：按序保留的两至五百一十二帧独立仿真见证。
    #   start_ms、end_ms：真实动作窗口的起止 Unix 毫秒。
    #   stage_id、goal_revision：当前阶段及目标版本的身份。
    #   goal_position_m：目标 ENU 三维位置，米。
    #   safety_intervened：实际安全接管事实。
    #   commanded_change_squared：实际控制变化平方成本。
    #   power_joules：有测量依据的耗能，未知时为 None。
    #   premature_landing：实际提前落地事实。
    # 输出：
    #   reward：由独立事实生成的步骤奖励证据。
    #   receipt：绑定来源、地图及所有评分输入的独立回执。
    def evaluate(self, *, observations: list[RuntimeLocalSafetyObservation],
                 start_ms: int, end_ms: int, stage_id: str, goal_revision: str,
                 goal_position_m: tuple[float, float, float], safety_intervened: bool,
                 commanded_change_squared: float, power_joules: float | None,
                 premature_landing: bool = False) -> tuple[RewardEvidence, dict]:
        if (type(observations) not in (list, tuple) or not 2 <= len(observations) <= 512
                or type(start_ms) is not int or type(end_ms) is not int
                or not 0 <= start_ms < end_ms < 2**63
                or type(safety_intervened) is not bool or type(premature_landing) is not bool
                or not finite_metric(commanded_change_squared) or commanded_change_squared < 0
                or (power_joules is not None
                    and (not finite_metric(power_joules) or power_joules < 0))):
            raise ValueError("OUTCOME_EVIDENCE_WINDOW_INVALID")
        goal_position_m = metric_vector(goal_position_m)
        if not all(isinstance(row, RuntimeLocalSafetyObservation) for row in observations):
            raise ValueError("OUTCOME_WITNESS_TYPE_INVALID")
        observations = [RuntimeLocalSafetyObservation.model_validate(row.model_dump(), strict=True)
                        for row in observations]
        if any(r.source != "simulation-ground-truth" or not r.stream_healthy
               or r.stream_age_seconds > .1 for r in observations):
            raise ValueError("OUTCOME_REQUIRES_INDEPENDENT_HEALTHY_WITNESS")
        if any(b.sequence != a.sequence + 1
               for a, b in zip(observations, observations[1:], strict=False)):
            raise ValueError("OUTCOME_WITNESS_SEQUENCE_GAP")
        if any(len({o.obstacle_id for o in row.dynamic_obstacles}) != len(row.dynamic_obstacles)
               for row in observations):
            raise ValueError("OUTCOME_DYNAMIC_IDENTITY_DUPLICATE")
        stamps = [r.observed_at_unix_ms for r in observations]
        if (not 0 <= start_ms - stamps[0] <= 100 or not 0 <= end_ms - stamps[-1] <= 100
                or not stamps[-1] > start_ms
                or any(not 0 < b-a <= 250 for a, b in zip(stamps, stamps[1:], strict=False))):
            raise ValueError("OUTCOME_EVIDENCE_TIME_BINDING_INVALID")
        positions = [tuple(getattr(r.current_position_m, axis) for axis in "xyz")
                     for r in observations]
        envelope = self.envelope
        extents = (envelope.body_radius_m, envelope.body_radius_m, envelope.body_height_m / 2)
        violation = any(not all(lo + extent <= v <= hi - extent for v, lo, hi, extent in zip(
            p, envelope.minimum_enu_m, envelope.maximum_enu_m, extents, strict=True,
        )) for p in positions)
        minimum = self.geometry.clearance(positions)
        # Dynamic witnesses use measured relative motion over each observed
        # interval. Losing an actor is unknown, not a free corridor assertion.
        for first, second, a, b in zip(observations, observations[1:],
                                        positions, positions[1:], strict=False):
            old = {o.obstacle_id: o for o in first.dynamic_obstacles}
            new = {o.obstacle_id: o for o in second.dynamic_obstacles}
            if set(old) != set(new):
                raise ValueError("OUTCOME_DYNAMIC_WITNESS_APPEARED_OR_DISAPPEARED")
            for key, previous in old.items():
                current = new[key]
                if previous.age_seconds > .1 or current.age_seconds > .1:
                    raise ValueError("OUTCOME_DYNAMIC_WITNESS_STALE")
                old_center = tuple(getattr(previous.position_m, axis) for axis in "xyz")
                new_center = tuple(getattr(current.position_m, axis) for axis in "xyz")
                relative_a = tuple(x-y for x, y in zip(a, old_center, strict=True))
                relative_b = tuple(x-y for x, y in zip(b, new_center, strict=True))
                cylinder = {"shape": "cylinder", "center_x": 0., "center_y": 0., "center_z": 0.,
                            "radius_m": max(previous.radius_m, current.radius_m),
                            "height_m": max(previous.height_m, current.height_m)}
                minimum = min(minimum, swept_clearance_bound(
                    relative_a, relative_b, [cylinder], radius_m=envelope.body_radius_m,
                    half_height_m=envelope.body_height_m / 2,
                ))
        receipt = {"purpose": "independent-simulation-step-outcome",
                   "geometry_sha256": self.geometry_sha256,
                   "observations": [r.model_dump(mode="json") for r in observations],
                   "start_ms": start_ms, "end_ms": end_ms,
                   "minimum_clearance_lower_bound_m": minimum,
                   "geofence_violation": violation, "stage_id": stage_id,
                   "goal_revision": goal_revision, "goal_position_m": list(goal_position_m),
                   "power_joules": power_joules,
                   "safety_intervened": safety_intervened,
                   "premature_landing": premature_landing,
                   "commanded_change_squared": commanded_change_squared,
                   "qualification_granted": False}
        reward = RewardEvidence(
            stage_id=stage_id, goal_revision=goal_revision, elapsed_seconds=(end_ms-start_ms)/1000,
            goal_distance_m=math.dist(positions[-1], goal_position_m),
            power_joules=power_joules, commanded_change_squared=commanded_change_squared,
            safety_intervened=safety_intervened, collision=minimum <= 0,
            geofence_violation=violation, premature_landing=premature_landing,
            verifier_receipt_sha256=sha256_json(receipt),
        )
        return reward, receipt
