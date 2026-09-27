"""Small causal decision baseline; never controls an aircraft or creates labels."""

import json
import math

from .decision_shadow import ACTIONS
from .decision_state_adapter import DecisionStateV2, decision_digest

TASKS = ("navigate", "takeoff", "pickup", "return", "land")
SCENES = ("office", "corridor", "door", "stairs", "transition", "outdoor", "pickup")
VECTORS = ("velocity_body_frd_mps", "goal_relative_body_frd_m")
NUMBERS = ("position_uncertainty_m", *(f"clearance_{d}_m" for d in
           ("front", "back", "left", "right", "up", "down")))
FLAGS = ("local_route_verified", "crossing_obstacle", "persistent_blockage",
         "caution_required", "can_hold_position", "can_brake", "payload_attached")
SOURCES = ("pose_source", "geometry_source", "route_source")
FEATURE_CONTRACT = {"schema": "dronedream.decision-student-features.v1",
    "tasks": TASKS, "scenes": SCENES, "vectors": VECTORS, "numbers": NUMBERS,
    "flags": FLAGS, "sources": SOURCES, "history_slots": 10, "track_slots": 12,
    "units": "metres,metres-per-second,seconds",
    "scalar": "x/(1+abs(x)),observed-mask", "history": "oldest-to-newest-left-pad",
    "track_order": "current-range-then-canonical-observation-without-id",
    "privileged_or_media_features": False}
FEATURE_SHA256 = decision_digest(FEATURE_CONTRACT)


# 功能：为轻量对照编码同一因果状态，保留未知掩码，不使用绝对位置/身份/未来真值。
# 输入：v2 状态；米、米每秒、毫秒年龄明确转换；输出：固定宽度有限特征。
def numeric_state(state: DecisionStateV2) -> list[float]:
    state = DecisionStateV2.model_validate_json(state.model_dump_json())
    output = []

    # 功能：保留缺失与零值的区别；输入：有限数值或 None；输出：数值/掩码两维。
    def scalar(value):
        if value is None:
            return [0., 0.]
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("DECISION_STUDENT_NONFINITE")
        return [value / (1. + abs(value)), 1.]

    # 功能：同一机体系向量逐轴编码；输入：FRD 向量；输出：六维数值/掩码。
    def vector(value):
        return [v for axis in "xyz" for v in scalar(getattr(value, axis) if value else None)]

    # 功能：固定槽位编码历史帧，不用全零冒充观测到静止；输入：帧或 None；输出：帧特征。
    def frame_features(frame):
        result = [float(frame is not None)]
        result += scalar((state.frame.observed_at_ms-frame.observed_at_ms)/1000 if frame else None)
        for name in VECTORS:
            result += vector(getattr(frame, name) if frame else None)
        for name in (*NUMBERS, *FLAGS):
            result += scalar(getattr(frame, name) if frame else None)
        for name in SOURCES:
            source = getattr(frame, name) if frame else None
            result += scalar((state.frame.observed_at_ms-source.observed_at_ms)/1000
                             if source else None)
            result += scalar(source.fresh(state.frame.observed_at_ms) if source else None)
        action = frame.applied_action if frame else None
        result += [float(action == a) for a in ACTIONS] + [float(action is not None)]
        return result

    output += [float(state.task == t) for t in TASKS]
    output += [float(state.scene == s) for s in SCENES]
    for value in (state.braking_acceleration_mps2, state.body_radius_m,
                  state.preferred_height_above_floor_m):
        output += scalar(value)
    output += frame_features(state.frame)
    for frame in (None,) * (10-len(state.history)) + state.history:
        output += frame_features(frame)
    # 目标标识符不作为训练特征，避免模型记住人员/日志 ID；同距离用观测内容稳定排序。
    tracks = sorted(state.frame.tracks, key=lambda track: (
        sum(getattr(track.relative_position_body_frd_m, axis)**2 for axis in "xyz"),
        json.dumps(track.model_dump(mode="json", exclude={"track_id", "source"}), sort_keys=True)))
    for index in range(12):
        track = tracks[index] if index < len(tracks) else None
        output += [float(track is not None)]
        output += vector(track.relative_position_body_frd_m if track else None)
        output += vector(track.relative_velocity_body_frd_mps if track else None)
        output += scalar(track.confidence if track else None)
        output += scalar((state.frame.observed_at_ms-track.source.observed_at_ms)/1000
                         if track else None)
        output += scalar(track.source.fresh(state.frame.observed_at_ms) if track else None)
    return output


# 功能：创建两层轻量对照，不把它当作已微调的 Laya 或已合格控制模型。
# 输入：固定输入宽度和隐藏宽度；输出：CPU PyTorch 网络，五类原始 logits。
def make_student(width: int, hidden: int = 64):
    from torch import nn

    if type(width) is not int or width <= 0 or type(hidden) is not int or not 8 <= hidden <= 256:
        raise ValueError("DECISION_STUDENT_SHAPE_INVALID")
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, len(ACTIONS)))
