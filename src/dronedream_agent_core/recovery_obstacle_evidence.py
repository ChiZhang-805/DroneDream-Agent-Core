"""Offline challenge witnesses; never provide truth observations to control."""

import bisect
import json
import math
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from .contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation, VehicleAsset
from .hashing import sha256_json
from .plugin_files import read_plugin_file
from .runtime_evidence import BoundedRuntimeEvidenceWriter, reserve_new_evidence_files

MAX_WITNESS_RECORDS = 20_000


# 功能：
#   为后台见证队列生成严格 JSON，不把仿真真值变为控制指令。
# 输入：
#   payload：已冻结的观测字典。
# 输出：
#   text：单行 JSON 文本。
def _serialize(payload: object) -> str:
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return text


class RecoveryObstacleWitness:
    """Single observer owner, five-Hz bounded independent obstacle evidence."""

    # 功能：
    #   独占建立离线障碍见证文件与有界写入队列，不连接执行器。
    # 输入：
    #   root：当前运行目录。
    #   publisher：原子保存回执的函数。
    # 输出：
    #   None：无返回数据。
    def __init__(self, root: Path, publisher) -> None:
        self.path = root / "recovery-obstacle-witness.jsonl"
        self.summary_path = root / "recovery-obstacle-witness-summary.json"
        reserve_new_evidence_files(self.path, self.summary_path)
        self.count, self.last_ms, self.issue = 0, None, None
        self.closed_summary = None
        self.writer = BoundedRuntimeEvidenceWriter(self.summary_path,
            summary_publisher=publisher, serializer=_serialize, maximum_pending_records=16)
        self.publisher = publisher

    # 功能：
    #   采样有原始时间的仿真障碍观测；空场景不记录，回退时钟和超量记录锁定失败。
    # 输入：
    #   observation：独立观察器的真值观测，不能来自原生控制器。
    # 输出：
    #   accepted：记录成功入队时为真。
    def record(self, observation: RuntimeLocalSafetyObservation) -> bool:
        if self.issue is not None or self.closed_summary is not None:
            return False
        if observation.source != "simulation-ground-truth":
            self.issue = "RECOVERY_WITNESS_SOURCE_INVALID"
            return False
        stamp = observation.observed_at_unix_ms
        if self.last_ms is not None and stamp < self.last_ms:
            self.issue = "RECOVERY_WITNESS_CLOCK_REGRESSED"
            return False
        if not observation.dynamic_obstacles or (self.last_ms is not None and stamp - self.last_ms < 200):
            return False
        if self.count >= MAX_WITNESS_RECORDS or len(observation.dynamic_obstacles) > 32:
            self.issue = "RECOVERY_WITNESS_CAPACITY_EXCEEDED"
            return False
        accepted = self.writer.submit(self.path, observation.model_dump(mode="json"))
        if accepted:
            self.count += 1
            self.last_ms = stamp
        else:
            self.issue = "RECOVERY_WITNESS_WRITE_REJECTED"
        return accepted

    # 功能：
    #   关闭见证流并保留真实写入状态，采集失败不能被底层队列成功掩盖。
    # 输入：
    #   self：独立见证记录器。
    # 输出：
    #   summary：采集数、错误及排空结果。
    def close(self) -> dict:
        if self.closed_summary is not None:
            return dict(self.closed_summary)
        summary = self.writer.close()
        summary = {**summary, "witness_count": self.count, "witness_issue": self.issue,
            "complete": summary["complete"] and self.issue is None,
            "truth_used_as_control_input": False}
        self.publisher(self.summary_path, summary)
        self.closed_summary = dict(summary)
        return summary


# 功能：
#   严格读取有界、已完整结束的 JSONL，不接受截断行或非对象记录。
# 输入：
#   path：固定证据路径。
# 输出：
#   rows：逐行解析的对象迭代器。
def _rows(path: Path):
    content = read_plugin_file(path, limit=256 * 1024 * 1024)
    if content and not content.endswith(b"\n"):
        raise ValueError("RECOVERY_EVIDENCE_TRUNCATED")
    for index, line in enumerate(content.splitlines()):
        if index >= 250_000:
            raise ValueError("RECOVERY_EVIDENCE_CAPACITY")
        row = decode_json(line, limit=4 * 1024 * 1024)
        if type(row) is not dict:
            raise ValueError("RECOVERY_EVIDENCE_OBJECT_REQUIRED")
        yield row


# 功能：
#   1. 离线匹配独立仿真见证与原生动态感知，不要求两个系统使用相同目标名称。
#   2. 只接受时间相近、位置一致且几何唯一关联的目标；威胁还须有同观测摘要的原生命令。
#   3. 本统计不授予控制权；实际执行标签仍由独立动作回执验收。
# 输入：
#   root：已关闭的仿真运行目录。
#   entity_names：挑战场景要求核对的独立实体名称。
# 输出：
#   metrics：每个实体的匹配观测数、原生威胁数及最小水平距离。
def recovery_obstacle_metrics(root: Path, entity_names: list[str]) -> dict:
    if (type(entity_names) is not list or not 1 <= len(entity_names) <= 32
            or any(type(name) is not str or not name or len(name) > 256 for name in entity_names)
            or len(set(entity_names)) != len(entity_names)):
        raise ValueError("RECOVERY_CHALLENGE_ENTITIES_INVALID")
    witnesses, stamps = _load_witnesses(root)
    metrics = {name: {"observation_count": 0, "threat_count": 0, "moving_observation_count": 0,
        "minimum_horizontal_distance_m": None} for name in entity_names}
    seen, threats_seen, motion_seen = set(), set(), set()
    for row in _rows(root / "depth-local-safety-history.jsonl"):
        observation = RuntimeLocalSafetyObservation.model_validate(row["observation"], strict=True)
        if (observation.source != "onboard" or not observation.stream_healthy
                or observation.stream_age_seconds > .25 or observation.localization_covariance_m2 > .25):
            continue
        stamp = observation.observed_at_unix_ms
        offset = bisect.bisect_left(stamps, stamp)
        candidates = [i for i in (offset - 1, offset) if 0 <= i < len(stamps)]
        if not candidates:
            continue
        index = min(candidates, key=lambda i: abs(stamps[i] - stamp))
        witness = witnesses[index]
        if (abs(stamps[index] - stamp) > 250 or not witness.stream_healthy
                or witness.stream_age_seconds > .1
                or math.dist(tuple(observation.current_position_m.model_dump().values()),
                             tuple(witness.current_position_m.model_dump().values())) > 1.):
            continue
        command = (RuntimeLocalSafetyCommand.model_validate(row["command"], strict=True)
                   if row.get("command") is not None else None)
        bound = (command is not None and command.source == "onboard"
            and command.observation_sequence == observation.sequence
            and command.observation_sha256 == sha256_json(observation)
            and stamp <= command.generated_at_unix_ms < command.valid_until_unix_ms <= stamp + 250)
        for native in observation.dynamic_obstacles:
            if native.age_seconds > .25 or native.confidence < .35:
                continue
            # 不靠名字猜关联，也不允许一个大感知包络同时为两个相邻挑战实体作证。
            matches = [truth for truth in witness.dynamic_obstacles
                if truth.confidence == 1. and truth.age_seconds <= .1
                and math.hypot(native.position_m.x - truth.position_m.x,
                               native.position_m.y - truth.position_m.y) <= truth.radius_m + .35
                and abs(native.position_m.z - truth.position_m.z) <= truth.height_m / 2 + .35]
            if len(matches) != 1 or matches[0].obstacle_id not in metrics:
                continue
            truth = matches[0]
            identity = observation.sequence, stamp, truth.obstacle_id
            entry = metrics[truth.obstacle_id]
            if identity not in seen:
                seen.add(identity)
                entry["observation_count"] += 1
            if (math.hypot(truth.velocity_mps.x, truth.velocity_mps.y) >= .12
                    and math.hypot(native.velocity_mps.x, native.velocity_mps.y) >= .12
                    and identity not in motion_seen):
                motion_seen.add(identity)
                entry["moving_observation_count"] += 1
            distance = math.hypot(observation.current_position_m.x - truth.position_m.x,
                                  observation.current_position_m.y - truth.position_m.y)
            previous = entry["minimum_horizontal_distance_m"]
            entry["minimum_horizontal_distance_m"] = distance if previous is None else min(previous, distance)
            # 同一实体可被深度分割成多个簇；先遇到非威胁簇不能吞掉另一簇的真实威胁。
            if bound and command.decision.avoidance_obstacle_id == native.obstacle_id and identity not in threats_seen:
                threats_seen.add(identity)
                entry["threat_count"] += 1
    return metrics


# 功能：
#   完整读取并核对独立见证的关闭回执、严格类型、时间顺序及数量，复用于感知关联和距离复核。
# 输入：
#   root：已经关闭的仿真运行目录。
# 输出：
#   records：观测列表与原始毫秒时间列表组成的二元组。
def _load_witnesses(root: Path) -> tuple[list[RuntimeLocalSafetyObservation], list[int]]:
    summary = decode_json(read_plugin_file(root / "recovery-obstacle-witness-summary.json",
        limit=4 * 1024 * 1024), limit=4 * 1024 * 1024)
    if (type(summary) is not dict or summary.get("complete") is not True
            or summary.get("truth_used_as_control_input") is not False
            or summary.get("witness_issue") is not None
            or summary.get("closed") is not True or summary.get("thread_finished") is not True
            or summary.get("thread_alive") is not False or summary.get("issue_code") is not None):
        raise ValueError("RECOVERY_WITNESS_INCOMPLETE")
    for field in ("witness_count", "completed_count", "submitted_count", "rejected_count", "pending_count"):
        if type(summary.get(field)) is not int or summary[field] < 0:
            raise ValueError("RECOVERY_WITNESS_COUNT_INVALID")
    if summary["rejected_count"] or summary["pending_count"]:
        raise ValueError("RECOVERY_WITNESS_INCOMPLETE")
    witnesses, stamps = [], []
    for row in _rows(root / "recovery-obstacle-witness.jsonl"):
        witness = RuntimeLocalSafetyObservation.model_validate(row, strict=True)
        if (witness.source != "simulation-ground-truth" or len(witness.dynamic_obstacles) > 32
                or len({item.obstacle_id for item in witness.dynamic_obstacles}) != len(witness.dynamic_obstacles)
                or len(witnesses) >= MAX_WITNESS_RECORDS
                or (stamps and witness.observed_at_unix_ms <= stamps[-1])):
            raise ValueError("RECOVERY_WITNESS_STREAM_INVALID")
        witnesses.append(witness)
        stamps.append(witness.observed_at_unix_ms)
    if (len(witnesses) != summary.get("witness_count")
            or len(witnesses) != summary.get("completed_count")
            or len(witnesses) != summary.get("submitted_count")):
        raise ValueError("RECOVERY_WITNESS_COUNT_MISMATCH")
    records = witnesses, stamps
    return records


# 功能：
#   1. 独立核对动态障碍与完整机体的三维包围体净空，不把中心距离或静态路线净空冒充动态净空。
#   2. 使用本轮实际实时安全余量；任何采样点违反余量均拒绝，不按原生感知是否发现障碍筛选。
#   3. 仅证明采样点的保守距离，不能证明两帧之间无碰撞；同时公开见证断档与不可用数量。
# 输入：
#   root：已关闭的独立仿真见证目录。
#   entity_names：本轮要求复核的障碍身份。
#   vehicle：本轮真实机体包围尺寸。
#   required_clearance_m：实时控制器使用的正净空余量，单位米。
# 输出：
#   report：各障碍最小净空、覆盖质量与采样点检查结果。
def recovery_clearance_metrics(root: Path, entity_names: list[str], vehicle: VehicleAsset, required_clearance_m: float) -> dict:
    if (type(entity_names) is not list or not 1 <= len(entity_names) <= 32
            or any(type(name) is not str or not name or len(name) > 256 for name in entity_names)
            or len(set(entity_names)) != len(entity_names)
            or type(required_clearance_m) not in (int, float)
            or not 0 < required_clearance_m <= 100):
        raise ValueError("RECOVERY_CLEARANCE_CONTRACT_INVALID")
    vehicle = VehicleAsset.model_validate(vehicle.model_dump(mode="python"), strict=True)
    witnesses, stamps = _load_witnesses(root)
    metrics = {name: {"sample_count": 0, "usable_sample_count": 0,
        "minimum_clearance_m": None, "minimum_at_unix_ms": None} for name in entity_names}
    for witness in witnesses:
        for obstacle in witness.dynamic_obstacles:
            if obstacle.obstacle_id not in metrics:
                continue
            entry = metrics[obstacle.obstacle_id]
            entry["sample_count"] += 1
            usable = (witness.stream_healthy and witness.stream_age_seconds <= .1
                      and obstacle.age_seconds <= .1 and obstacle.confidence == 1.)
            entry["usable_sample_count"] += int(usable)
            # 陈旧/不健康记录不证明安全，但其中已出现的近距离也不能被过滤后抹去。
            horizontal = math.hypot(witness.current_position_m.x - obstacle.position_m.x,
                                    witness.current_position_m.y - obstacle.position_m.y)
            horizontal -= vehicle.body_radius_m + obstacle.radius_m
            vertical = abs(witness.current_position_m.z - obstacle.position_m.z)
            vertical -= (vehicle.body_height_m + obstacle.height_m) / 2
            clearance = (math.hypot(horizontal, vertical) if horizontal > 0 and vertical > 0
                         else max(horizontal, vertical))
            if not math.isfinite(clearance):
                raise ValueError("RECOVERY_CLEARANCE_NONFINITE")
            if entry["minimum_clearance_m"] is None or clearance < entry["minimum_clearance_m"]:
                entry["minimum_clearance_m"] = clearance
                entry["minimum_at_unix_ms"] = witness.observed_at_unix_ms
    report = {"required_clearance_m": required_clearance_m,
        "vehicle_radius_m": vehicle.body_radius_m, "vehicle_height_m": vehicle.body_height_m,
        "witness_count": len(witnesses),
        "maximum_witness_gap_ms": max((right - left for left, right in zip(stamps, stamps[1:], strict=False)), default=None),
        "entities": metrics, "continuous_clearance_verified": False,
        "sampled_clearance_respected": all(entry["usable_sample_count"] > 0
            and entry["minimum_clearance_m"] is not None
            and entry["minimum_clearance_m"] >= required_clearance_m for entry in metrics.values())}
    return report
