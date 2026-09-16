"""Deeply immutable in-process perception, with unchanged wire contracts.

Fusion validates and owns this graph once. Internal readers may retain it
across source replacement without borrowing a mutable ray list or vector.
External callers still receive ordinary detached OnboardPerceptionFrame data.
No clocks, measurements, masks or validation constraints are weakened here.
"""

from copy import copy

from pydantic import ConfigDict

from .contracts import (
    DynamicObstacleObservation,
    OnboardPerceptionFrame,
    RangeRayObservation,
    Vector3,
)


class FrozenVector3(Vector3):
    model_config = ConfigDict(frozen=True)


class FrozenRangeRay(RangeRayObservation):
    model_config = ConfigDict(frozen=True)
    origin_m: FrozenVector3
    endpoint_m: FrozenVector3


class FrozenDynamicObstacle(DynamicObstacleObservation):
    model_config = ConfigDict(frozen=True)
    position_m: FrozenVector3
    velocity_mps: FrozenVector3


# 功能：
#   复制线协议列表字段的约束，仅将内部默认容器改为元组，不放宽必填与容量限制。
# 输入：
#   name：感知帧中已有列表字段的名称。
# 输出：
#   field：保留原始约束的独立字段描述。
def _immutable_sequence_field(name):
    # Keep the authoritative field's bounds, metadata and required/default
    # behavior. Only the container representation changes inside the process.
    field = copy(OnboardPerceptionFrame.model_fields[name])
    if field.default_factory is not None:
        field.default_factory = tuple
    return field


class FrozenPerceptionFrame(OnboardPerceptionFrame):
    """Own an immutable observation graph, including every nested vector."""

    model_config = ConfigDict(frozen=True)
    localization_position_m: FrozenVector3
    localization_velocity_mps: FrozenVector3
    range_rays: tuple[FrozenRangeRay, ...] = _immutable_sequence_field("range_rays")
    dynamic_obstacles: tuple[FrozenDynamicObstacle, ...] = _immutable_sequence_field(
        "dynamic_obstacles"
    )


# 功能：
#   从原始帧生成独立且深层不可变的观测图，在任何状态计算前严格重验所有字段。
# 输入：
#   frame：原始感知帧或待重新检查的内部冻结帧。
# 输出：
#   frozen：保持原始时钟、数值与 JSON 协议的不可变帧。
def freeze_perception_frame(frame: OnboardPerceptionFrame) -> FrozenPerceptionFrame:
    if not isinstance(frame, OnboardPerceptionFrame):
        raise ValueError("PERCEPTION_FRAME_TYPE_INVALID")
    payload = frame.model_dump(mode="python")
    # 只改变两处容器表示。严格验证仍逐一检查射线、目标及向量，不沿用旧对象的信任。
    for name in ("range_rays", "dynamic_obstacles"):
        if not isinstance(payload[name], (list, tuple)):
            raise ValueError("PERCEPTION_FRAME_COLLECTION_INVALID")
        payload[name] = tuple(payload[name])
    frozen = FrozenPerceptionFrame.model_validate(payload, strict=True)
    return frozen
