"""Bind perception health to completion of controlled motion, before shutdown.

The final mutable health file may correctly report expired sources after the
executor closes telemetry. It is not the health evidence for earlier motion.
Capture at the actual control-to-landing transition; never search backwards for
an arbitrarily old good frame or change a failed flight into a success.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from dronedream_plugin_sdk.protocol import copy_json, decode_json

from .hashing import sha256_json
from .perception_health_channel import PerceptionHealthReceiver
from .plugin_files import read_plugin_file

_HEALTH_BYTES = 262_144
_RECEIPT_BYTES = _HEALTH_BYTES + 4096
_MAXIMUM_UNIX_MS = ((1 << 63) - 1) // 1_000_000


# 功能：
#   检查完成时刻为可与来源纳秒契约相容的非负整数毫秒，拒绝布尔值和隐式转换。
# 输入：
#   value：候选 UNIX 毫秒。
# 输出：
#   valid：类型及范围均合法时为 True。
def _valid_unix_ms(value) -> bool:
    valid = type(value) is int and 0 <= value <= _MAXIMUM_UNIX_MS
    return valid


# 功能：
#   检查轨迹 SHA-256 的完整小写十六进制表示，不将普通名称或短摘要作为身份。
# 输入：
#   value：候选轨迹摘要。
# 输出：
#   valid：摘要格式合法时为 True。
def _valid_track_digest(value) -> bool:
    valid = type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    return valid


# 功能：
#   根据完成时刻复核来源年龄、原生定位绑定与实时特征状态，不以健康文件写入时间代替。
# 输入：
#   health：已经严格解析的健康字典。
#   completion_ms：控制结束时的 UNIX 毫秒。
# 输出：
#   ready：各项健康条件在完成时刻仍同时成立时为 True。
def _health_ready(health: dict, *, completion_ms: int) -> bool:
    updated = health.get("updated_at_unix_ms")
    source_age = health.get("stream_age_seconds")
    localized = health.get("localization_observed_at_unix_ms")
    ready = bool(
        _valid_unix_ms(completion_ms) and _valid_unix_ms(updated)
        and 0 <= completion_ms - updated <= 250
        # 先比较范围再转浮点，超大 JSON 整数不能在失败收尾时引起 OverflowError。
        and type(source_age) in (int, float) and 0 <= source_age <= .25
        and math.isfinite(source_age)
        and source_age + (completion_ms - updated) / 1000 <= .25
        and _valid_unix_ms(localized) and 0 <= completion_ms - localized <= 250
        and health.get("stream_healthy") is True
        and health.get("identity_accepted") is True
        and health.get("truth_correction_applied") is False
        and health.get("realtime_features_ready") is True
        and health.get("pose_source") == "native-estimator-fixed-deployment-binding"
    )
    return ready


# 功能：
#   1. 在控制切换至降落时冻结最新已收健康快照并绑定轨迹，后来的健康数据不能修补此事件。
#   2. 非法身份或读取失败生成拒绝回执，不阻止调用方继续降落清理，不复制内部异常详情。
# 输入：
#   path：当前感知健康普通文件的路径。
#   track_sha256：刚结束控制的活动轨迹摘要。
#   completed_at_unix_ms：控制结束的 UNIX 毫秒。
#   receiver：当前回合健康接收端；提供后不再回退读取磁盘。
# 输出：
#   receipt：健康快照、接受结果、失败类别及内容摘要组成的独立回执。
def capture_control_completion(
    path: Path, *, track_sha256: str, completed_at_unix_ms: int,
    receiver: PerceptionHealthReceiver | None = None,
) -> dict:
    track = track_sha256 if _valid_track_digest(track_sha256) else None
    completed = completed_at_unix_ms if _valid_unix_ms(completed_at_unix_ms) else None
    try:
        if track is None or completed is None:
            raise ValueError("PERCEPTION_COMPLETION_IDENTITY_INVALID")
        if not isinstance(path, Path):
            raise ValueError("PERCEPTION_COMPLETION_PATH_INVALID")
        if receiver is not None:
            health = receiver.latest_snapshot()
        else:
            packet = read_plugin_file(path, limit=_HEALTH_BYTES)
            health = decode_json(packet, limit=_HEALTH_BYTES)
        if type(health) is not dict:
            raise ValueError("PERCEPTION_HEALTH_OBJECT_REQUIRED")
        # 读取字节、严格 JSON 与摘要共用同一份内容；重复键不能在生成摘要前被静默覆盖。
        sha256_json(health)
        issue = None
    except (OSError, ValueError, RecursionError, RuntimeError) as error:
        health, issue = {}, "PERCEPTION_COMPLETION_REJECTED:" + type(error).__name__[:80]
    content = {
        "event": "controlled-motion-completed-before-landing",
        "track_sha256": track,
        "completed_at_unix_ms": completed,
        "health": health,
        "accepted": issue is None and _health_ready(health, completion_ms=completed),
        "read_issue": issue,
    }
    receipt = {**content, "receipt_sha256": sha256_json(content)}
    return receipt


# 功能：
#   独立复核回执预算、摘要、活动轨迹及当时健康条件，后续停流不改写此前控制结果。
# 输入：
#   timing：执行器计时与完成状态，包含控制结束时冻结的感知回执。
# 输出：
#   valid：执行已完成且回执完整、身份一致、当时健康时为 True；非法输入为 False。
def verify_control_completion(timing: dict) -> bool:
    if type(timing) is not dict:
        valid = False
        return valid
    record = timing.get("perception_control_completion")
    if type(record) is not dict or timing.get("status") != "complete":
        valid = False
        return valid
    try:
        # 先复核标准 JSON 类型，不能让历史摘要函数的键字符串化行为替非法输入消歧。
        content = copy_json(record, limit=_RECEIPT_BYTES)
    except (TypeError, ValueError, RecursionError):
        valid = False
        return valid
    digest = content.pop("receipt_sha256", None)
    health = content.get("health")
    completed = content.get("completed_at_unix_ms")
    track = content.get("track_sha256")
    try:
        expected = sha256_json(content)
    except (TypeError, ValueError, RecursionError):
        valid = False
        return valid
    valid = bool(
        digest == expected
        and content.get("event") == "controlled-motion-completed-before-landing"
        and content.get("accepted") is True
        and content.get("read_issue") is None
        and _valid_track_digest(track)
        and track == timing.get("active_track_sha256")
        and type(health) is dict and _valid_unix_ms(completed)
        and _health_ready(health, completion_ms=completed)
    )
    return valid
