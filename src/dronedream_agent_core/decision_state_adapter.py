"""Causal, versioned UAV decision input. No flight transport or model imports."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_shadow import ACTIONS, DESCRIPTIONS

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=160)]
Nonnegative = Annotated[float, Field(ge=0)]
Action = Literal["follow_route", "slow_down", "wait", "replan", "request_observation"]


class DecisionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class MetricVector(DecisionContract):
    x: float
    y: float
    z: float


class SourceStamp(DecisionContract):
    source_id: Identifier
    evidence_sha256: Digest
    observed_at_ms: Annotated[int, Field(ge=0)]
    maximum_age_ms: Annotated[int, Field(gt=0, le=10000)]

    # 功能：按同一时钟域判断源数据时效，不把未来时间或处理时间当观测时间。
    # 输入：now_ms：当前时钟域毫秒；输出：是否在原始有效期内。
    def fresh(self, now_ms: int) -> bool:
        return 0 <= now_ms - self.observed_at_ms <= self.maximum_age_ms


class DynamicTrack(DecisionContract):
    track_id: Identifier
    relative_position_body_frd_m: MetricVector
    relative_velocity_body_frd_mps: MetricVector | None
    confidence: Annotated[float, Field(ge=0, le=1)]
    source: SourceStamp


class DecisionFrame(DecisionContract):
    observed_at_ms: Annotated[int, Field(ge=0)]
    position_world_enu_m: MetricVector | None
    velocity_world_enu_mps: MetricVector | None
    velocity_body_frd_mps: MetricVector | None = None
    goal_relative_body_frd_m: MetricVector | None
    position_uncertainty_m: Nonnegative | None
    pose_source: SourceStamp | None
    geometry_source: SourceStamp | None
    route_source: SourceStamp | None
    # 六向距离顺序在字段中固定；None 表示没有覆盖，不等于无限净空。
    clearance_front_m: Nonnegative | None
    clearance_back_m: Nonnegative | None
    clearance_left_m: Nonnegative | None
    clearance_right_m: Nonnegative | None
    clearance_up_m: Nonnegative | None
    clearance_down_m: Nonnegative | None
    local_route_verified: bool | None
    crossing_obstacle: bool | None
    persistent_blockage: bool | None
    caution_required: bool | None
    can_hold_position: bool | None
    can_brake: bool | None
    payload_attached: bool | None
    optional_media_available: bool | None = None
    tracks: tuple[DynamicTrack, ...] = Field(default=(), max_length=12)
    applied_action: Action | None = None

    # 功能：拒绝跨时刻拼接的未来观测和无来源的位置/空间数据。
    # 输入：完整帧；输出：已校验帧，缺失保持未知而非补安全值。
    @model_validator(mode="after")
    def validate_sources(self):
        stamps = [self.pose_source, self.geometry_source, self.route_source]
        stamps.extend(track.source for track in self.tracks)
        if any(s and s.observed_at_ms > self.observed_at_ms for s in stamps):
            raise ValueError("DECISION_FUTURE_SOURCE")
        if (self.position_world_enu_m is not None or self.velocity_world_enu_mps is not None
                or self.goal_relative_body_frd_m is not None
                or self.velocity_body_frd_mps is not None
                or self.position_uncertainty_m is not None) and self.pose_source is None:
            raise ValueError("DECISION_POSE_SOURCE_REQUIRED")
        if (any(getattr(self, f"clearance_{d}_m") is not None
                for d in ("front", "back", "left", "right", "up", "down"))
                and self.geometry_source is None):
            raise ValueError("DECISION_GEOMETRY_SOURCE_REQUIRED")
        if self.local_route_verified is not None and self.route_source is None:
            raise ValueError("DECISION_ROUTE_SOURCE_REQUIRED")
        if len({t.track_id for t in self.tracks}) != len(self.tracks):
            raise ValueError("DECISION_DUPLICATE_TRACK")
        return self


class DecisionStateV2(DecisionContract):
    schema_version: Literal["dronedream.decision-state.v2"] = "dronedream.decision-state.v2"
    mission_id: Identifier
    goal_id: Identifier
    route_sha256: Digest
    map_sha256: Digest
    vehicle_sha256: Digest
    calibration_sha256: Digest
    clock_domain: Identifier
    sequence: Annotated[int, Field(ge=0)]
    task: Literal["navigate", "takeoff", "pickup", "return", "land"]
    scene: Literal["office", "corridor", "door", "stairs", "transition", "outdoor", "pickup"]
    braking_acceleration_mps2: Annotated[float, Field(gt=0, le=20)]
    body_radius_m: Annotated[float, Field(gt=0, le=5)]
    preferred_height_above_floor_m: Nonnegative | None
    frame: DecisionFrame
    history: tuple[DecisionFrame, ...] = Field(default=(), max_length=10)

    # 功能：保证历史严格早于当前帧，且只含约两秒的因果状态。
    # 输入：状态及有序历史；输出：有效状态；拒绝未来标签伪装成历史。
    @model_validator(mode="after")
    def validate_history(self):
        times = [f.observed_at_ms for f in self.history]
        if times != sorted(set(times)) or any(
                not 0 < self.frame.observed_at_ms - t <= 2500 for t in times):
            raise ValueError("DECISION_HISTORY_NOT_CAUSAL")
        return self


# 功能：对版本化 JSON 建立稳定内容身份，不包含 NaN 或依赖字典插入顺序。
# 输入：JSON 兼容对象；输出：SHA256，供证据/训练/运行时绑定。
def decision_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# 功能：剥离样本身份和证据路径，生成只有当前可见事实的紧凑模型输入。
# 输入：严格验证状态；输出：固定英文/数字模板，界面语言不影响行为 ID。
def model_state(state: DecisionStateV2, *, history_limit: int = 10) -> str:
    state = DecisionStateV2.model_validate_json(state.model_dump_json())
    if type(history_limit) is not int or not 0 <= history_limit <= 10:
        raise ValueError("DECISION_HISTORY_LIMIT_INVALID")

    # 功能：保留数值与缺失掩码，仅将来源身份压缩为年龄和时效。
    # 输入：同域帧；输出：无未来结果、无地图/样本 ID 的状态字典。
    def compact(frame):
        data = frame.model_dump(mode="json", exclude={"optional_media_available"})
        for name in ("pose_source", "geometry_source", "route_source"):
            stamp = getattr(frame, name)
            data[name] = None if stamp is None else {
                "age_ms": state.frame.observed_at_ms - stamp.observed_at_ms,
                "fresh": stamp.fresh(state.frame.observed_at_ms),
            }
        for track in data["tracks"]:
            track.pop("track_id")
            source = track.pop("source")
            track["age_ms"] = state.frame.observed_at_ms - source["observed_at_ms"]
            track["fresh"] = 0 <= track["age_ms"] <= source["maximum_age_ms"]
        data["age_ms"] = state.frame.observed_at_ms - data.pop("observed_at_ms")
        # 全局绝对坐标易形成地图记忆；局部目标向量已表达路线关系。
        data.pop("position_world_enu_m")
        # 与局部目标使用同一机体系；不把 ENU 速度当成机头前进速度。
        data.pop("velocity_world_enu_mps")
        return data

    value = {"task": state.task, "scene": state.scene,
             "braking_acceleration_mps2": state.braking_acceleration_mps2,
             "body_radius_m": state.body_radius_m,
             "preferred_height_above_floor_m": state.preferred_height_above_floor_m,
             "current": compact(state.frame),
             "history": [compact(f) for f in state.history[-history_limit:]]
             if history_limit else []}
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


# 功能：构造训练和推理共同使用的五选项问题，选项换序时严格保持身份映射。
# 输入：完整选项排列；输出：Laya 内部问题格式，不授予执行权限。
def decision_question(order: tuple[str, ...] = ACTIONS) -> dict:
    if len(order) != len(ACTIONS) or set(order) != set(ACTIONS):
        raise ValueError("DECISION_OPTION_ORDER_INVALID")
    return {"t": "choice", "ins": "Choose the appropriate UAV behavior from current "
            "observations and recent history. Unknown is not clear space. Preserve the mission "
            "through temporary difficulties. This is advice, not a motor command.",
            "crit": {a: DESCRIPTIONS[a] for a in order}}
