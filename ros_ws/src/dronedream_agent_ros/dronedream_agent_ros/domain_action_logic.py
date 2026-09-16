"""Pure validation logic for simulation-backed ROS 2 domain actions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_io import decode_object, read_object

# 这些名称保留明确的服务拒绝回执；接近目标不构成扫码或收件人认证实现。
DECLARED_ACTIONS = ("native.payload.scan-code", "native.payload.verify-recipient")


@dataclass(frozen=True)
class TargetPoint:
    east_m: float
    north_m: float
    up_m: float
    checkpoint_id: str
    track_point_index: int


# 功能：
#   校验有限真实数值，拒绝布尔、文本及超大整数溢出被当作物理观测。
# 输入：
#   value：待校验标量。
# 输出：
#   valid：输入为可表示的有限实数时为真。
def finite_number(value: Any) -> bool:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   从同次执行的轨迹及检查点建立目标坐标，拒绝错误类型、重复检查点与冲突目标。
# 输入：
#   track_path：当前飞控轨迹路径。
#   checkpoint_contract_path：冻结检查点合同路径。
#   expected_contract_id：可选的当前任务身份；运行服务必须显式提供。
# 输出：
#   result：目标节点到世界 ENU 坐标及检查点的映射。
def load_target_points(
    track_path: Path, checkpoint_contract_path: Path, *, expected_contract_id: str | None = None
) -> dict[str, TargetPoint]:
    track = read_object(track_path)
    checkpoints = read_object(checkpoint_contract_path)
    if expected_contract_id is not None and checkpoints.get("contract_id") != expected_contract_id:
        raise ValueError("DOMAIN_ACTION_CHECKPOINT_CONTRACT_MISMATCH")
    points = track.get("source_world_points")
    if not isinstance(points, list) or not points:
        raise ValueError("PX4 track has no source_world_points")
    result: dict[str, TargetPoint] = {}
    raw_checkpoints = checkpoints.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or not 1 <= len(raw_checkpoints) <= 128:
        raise ValueError("checkpoint contract has invalid checkpoint count")
    seen = set()
    for checkpoint in raw_checkpoints:
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint contract contains a non-object entry")
        index = checkpoint.get("track_point_index")
        if type(index) is not int or not 1 <= index < len(points):
            raise ValueError("checkpoint track point index is outside the PX4 track")
        raw = points[index]
        if not isinstance(raw, dict) or not all(
            finite_number(raw.get(axis)) for axis in ("east_m", "north_m", "up_m")
        ):
            raise ValueError("checkpoint target contains invalid coordinates")
        checkpoint_id, target_node = checkpoint.get("checkpoint_id"), checkpoint.get("target_node")
        if not isinstance(checkpoint_id, str) or not checkpoint_id or checkpoint_id in seen:
            raise ValueError("checkpoint identity missing or duplicated")
        if not isinstance(target_node, str) or not target_node.strip():
            raise ValueError("checkpoint target identity missing")
        seen.add(checkpoint_id)
        target = TargetPoint(
            east_m=float(raw["east_m"]),
            north_m=float(raw["north_m"]),
            up_m=float(raw["up_m"]),
            checkpoint_id=checkpoint_id,
            track_point_index=index,
        )
        if not all(math.isfinite(value) for value in (target.east_m, target.north_m, target.up_m)):
            raise ValueError("checkpoint target contains non-finite coordinates")
        previous = result.get(target_node)
        if (
            previous is not None
            and math.dist(
                (previous.east_m, previous.north_m, previous.up_m),
                (target.east_m, target.north_m, target.up_m),
            )
            > 0.01
        ):
            raise ValueError(f"target node is bound to conflicting track points: {target_node}")
        result[target_node] = target
    if not result:
        raise ValueError("checkpoint contract contains no target bindings")
    return result


# 功能：
#   1. 检查服务身份、目标绑定、观测时效及距离，只证明到达前置条件。
#   2. 未接入实际解码或身份验证器时必须返回失败，禁止用目标接近代替扫码与认证结果。
# 输入：
#   expected_contract_id：服务绑定的任务合同。
#   expected_action：服务入口绑定的动作。
#   request_contract_id：请求声明的任务合同。
#   task_id：请求任务节点。
#   request_action：请求的具体动作。
#   target_node：目标语义节点。
#   arguments_json：动作参数 JSON。
#   target_points：本次执行的目标坐标映射。
#   observation_contract_id：遥测所属任务。
#   observation_sequence：真实观测递增序号。
#   observation_position_enu_m：当前世界 ENU 三维位置。
#   observation_age_seconds：观测在本机经过的秒数。
#   maximum_observation_age_seconds：允许的观测年龄上限。
#   maximum_target_distance_m：允许的目标距离上限。
# 输出：
#   result：包含成功标识、实际证据、诊断 JSON 与失败原因的服务结果。
def evaluate_simulation_action(
    *,
    expected_contract_id: str,
    expected_action: str,
    request_contract_id: str,
    task_id: str,
    request_action: str,
    target_node: str,
    arguments_json: str,
    target_points: dict[str, TargetPoint],
    observation_contract_id: str,
    observation_sequence: int,
    observation_position_enu_m: tuple[float, float, float],
    observation_age_seconds: float,
    maximum_observation_age_seconds: float,
    maximum_target_distance_m: float,
) -> dict[str, Any]:
    issue_code = ""
    if (
        not finite_number(maximum_observation_age_seconds)
        or maximum_observation_age_seconds <= 0
        or not finite_number(maximum_target_distance_m)
        or maximum_target_distance_m <= 0
    ):
        issue_code = "DOMAIN_ACTION_LIMIT_INVALID"
    elif type(observation_sequence) is not int or observation_sequence <= 0:
        issue_code = "DOMAIN_ACTION_OBSERVATION_SEQUENCE_INVALID"
    elif not expected_contract_id or request_contract_id != expected_contract_id:
        issue_code = "DOMAIN_ACTION_CONTRACT_MISMATCH"
    elif observation_contract_id != expected_contract_id:
        issue_code = "DOMAIN_ACTION_OBSERVATION_CONTRACT_MISMATCH"
    elif request_action != expected_action:
        issue_code = "DOMAIN_ACTION_EXECUTOR_MISMATCH"
    elif not isinstance(task_id, str) or not task_id.strip():
        issue_code = "DOMAIN_ACTION_TASK_ID_MISSING"
    elif expected_action not in DECLARED_ACTIONS:
        issue_code = "SIMULATION_ACTION_BACKEND_UNIMPLEMENTED"
    elif target_node not in target_points:
        issue_code = "DOMAIN_ACTION_TARGET_UNBOUND"
    elif not finite_number(observation_age_seconds) or (
        not 0 <= observation_age_seconds <= maximum_observation_age_seconds
    ):
        issue_code = "DOMAIN_ACTION_OBSERVATION_STALE"

    try:
        arguments = decode_object(arguments_json)
    except ValueError:
        arguments = None
        issue_code = issue_code or "DOMAIN_ACTION_ARGUMENTS_INVALID_JSON"
    if not isinstance(arguments, dict):
        issue_code = issue_code or "DOMAIN_ACTION_ARGUMENTS_NOT_OBJECT"

    target = target_points.get(target_node)
    distance_m = math.inf
    position_valid = (
        isinstance(observation_position_enu_m, (tuple, list))
        and len(observation_position_enu_m) == 3
        and all(finite_number(value) for value in observation_position_enu_m)
    )
    if not position_valid:
        issue_code = issue_code or "DOMAIN_ACTION_POSITION_INVALID"
    if target is not None and position_valid:
        distance_m = math.dist(
            observation_position_enu_m,
            (target.east_m, target.north_m, target.up_m),
        )
        if not math.isfinite(distance_m) or distance_m > maximum_target_distance_m:
            issue_code = issue_code or "DOMAIN_ACTION_TARGET_NOT_REACHED"

    # 当前后端只有到达观察，没有二维码解码或收件人认证事实，不能签发这些成功证据。
    proximity_verified = not issue_code
    issue_code = issue_code or "DOMAIN_ACTION_SENSOR_VERIFIER_UNAVAILABLE"
    success = False
    details = {
        "schema_version": "dronedream.simulation-domain-action.v1",
        "backend": "semantic-target-proximity",
        "proximity_verified": proximity_verified,
        "sensor_verifier_available": False,
        "task_id": task_id,
        "action": request_action,
        "target_node": target_node,
        "observation_sequence": observation_sequence,
        "observation_age_seconds": observation_age_seconds
        if finite_number(observation_age_seconds)
        else None,
        "target_distance_m": distance_m if math.isfinite(distance_m) else None,
        "maximum_target_distance_m": maximum_target_distance_m
        if finite_number(maximum_target_distance_m)
        else None,
        "target_binding": (
            {
                "checkpoint_id": target.checkpoint_id,
                "track_point_index": target.track_point_index,
                "east_m": target.east_m,
                "north_m": target.north_m,
                "up_m": target.up_m,
            }
            if target is not None
            else None
        ),
        "arguments_sha256": hashlib.sha256(arguments_json.encode("utf-8")).hexdigest(),
    }
    result = {
        "success": success,
        "evidence": [],
        "details_json": json.dumps(
            details,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        "issue_code": issue_code,
    }
    return result
