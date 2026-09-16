"""Require an independently progressing native perception chain before arming."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .collision import build_tracking_corridor_budget
from .perception_health_channel import PerceptionHealthReceiver
from .runtime_control_io import read_runtime_object
from .sensor_diagnostics import sensor_issue_codes

NATIVE_PREFLIGHT_MAXIMUM_AGE_MS = 250
PREFLIGHT_STABLE_WINDOW_MS = 1000


# 功能：
#   在预送控制设定值之后、解锁之前重查原始感知时刻，不以重新发布的时刻延长授权。
# 输入：
#   receipt：感知就绪检查返回的原始回执。
#   now_unix_ms：实际准备解锁时的 UNIX 毫秒时刻。
# 输出：
#   None：不返回业务数据。
def assert_native_preflight_current(receipt: dict, *, now_unix_ms: int) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("NATIVE_PREFLIGHT_EVIDENCE_EXPIRED_OR_INVALID_BEFORE_ARM")
    observed = receipt.get("source_observed_at_unix_ms")
    deadline = receipt.get("valid_until_unix_ms")
    if (receipt.get("ready") is not True or type(observed) is not int
            or type(deadline) is not int or type(now_unix_ms) is not int
            or not 0 <= observed <= now_unix_ms <= deadline < 2**63
            or deadline != observed + NATIVE_PREFLIGHT_MAXIMUM_AGE_MS):
        raise ValueError("NATIVE_PREFLIGHT_EVIDENCE_EXPIRED_OR_INVALID_BEFORE_ARM")


@dataclass
class NativePerceptionReadiness:
    """Accumulate independent fresh frames, clearing qualification after interruption."""

    minimum_route_clearance_m: float | None = None
    stable_window_ms: int = 100
    require_source_timestamps: bool = False
    last_sequence: int = -1
    independent_frames: int = 0
    first_frame_time: int | None = None
    last_frame_time: int | None = None
    last_health_time: int | None = None
    source_observed_at_unix_ms: int | None = None
    tracking_budget: dict[str, float] | None = None
    _worst_variance_m2: float | None = None
    last_issue: str | None = None

    # 功能：
    #   校验连续稳定窗口，稳定时长与单份观测有效期分别管理，不能互相替代。
    # 输入：
    #   self：已配置等待窗口和来源要求的检查器。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self) -> None:
        if (type(self.stable_window_ms) is not int or not 100 <= self.stable_window_ms <= 5000
                or type(self.require_source_timestamps) is not bool):
            raise ValueError("NATIVE_PREFLIGHT_STABILITY_CONFIGURATION_INVALID")

    # 功能：
    #   清空已积累帧数、来源时间和不确定性预算，保留最近失败原因供调用方报告。
    # 输入：
    #   self：当前就绪窗口。
    # 输出：
    #   None：不返回业务数据。
    def _reset_window(self) -> None:
        self.last_sequence, self.independent_frames, self.first_frame_time = -1, 0, None
        self.last_frame_time = self.last_health_time = None
        self.source_observed_at_unix_ms = None
        self._worst_variance_m2 = None
        self.tracking_budget = None

    # 功能：
    #   1. 只累计来源序列与时刻共同推进的新鲜帧，至少三帧且达到配置的稳定窗口才就绪。
    #   2. 若要求路线余量，持续使用窗口内最差实测协方差；中断或回退使窗口失效。
    # 输入：
    #   self：感知就绪窗口及可选的路线净空约束。
    #   payload：同一部署原生感知链路的健康快照。
    #   now_unix_ms：消费这份快照时的 UNIX 毫秒时刻。
    # 输出：
    #   ready：独立帧、原始时效及所需余量均满足时为 True。
    def observe(self, payload: dict, *, now_unix_ms: int) -> bool:
        self.last_issue = None
        self.tracking_budget = None
        if (not isinstance(payload, dict) or type(now_unix_ms) is not int
                or not 0 <= now_unix_ms < 2**63):
            self.last_issue = "NATIVE_PREFLIGHT_CLOCK_OR_PAYLOAD_INVALID"
            self._reset_window()
            return False
        sequence = payload.get("latest_sequence")
        updated = payload.get("updated_at_unix_ms")
        valid = (
            payload.get("schema_version") == "dronedream.perception-fusion-health.v1"
            and payload.get("stream_healthy") is True
            and payload.get("identity_accepted") is True
            and payload.get("truth_correction_applied") is False
            and payload.get("pose_source") == "native-estimator-fixed-deployment-binding"
            and payload.get("realtime_features_ready") is True
            and type(updated) is int and 0 <= updated < 2**63
            and 0 <= now_unix_ms - updated <= NATIVE_PREFLIGHT_MAXIMUM_AGE_MS
            and type(sequence) is int and 0 < sequence < 2**63
        )
        frame_time = updated
        if not valid:
            # 只传递固定诊断代码；发布端的不健康原因优先于后续缺字段症状。
            codes = sensor_issue_codes(payload.get("issue_codes"))
            if payload.get("stream_healthy") is not True:
                self.last_issue = codes[0] if codes else "NATIVE_PERCEPTION_STREAM_NOT_HEALTHY"
            elif payload.get("identity_accepted") is not True:
                self.last_issue = "NATIVE_PERCEPTION_IDENTITY_NOT_READY"
            elif payload.get("realtime_features_ready") is not True:
                self.last_issue = "NATIVE_PERCEPTION_FEATURES_NOT_READY"
            elif (type(updated) is not int or not 0 <= updated < 2**63
                    or not 0 <= now_unix_ms-updated <= NATIVE_PREFLIGHT_MAXIMUM_AGE_MS):
                self.last_issue = "NATIVE_PERCEPTION_HEALTH_CLOCK_INVALID_OR_EXPIRED"
            else:
                self.last_issue = "NATIVE_PERCEPTION_CONTRACT_INVALID"
        if valid and self.require_source_timestamps:
            depth_time = payload.get("perception_observed_at_unix_ms")
            if (type(depth_time) is not int or not 0 <= depth_time <= updated
                    or now_unix_ms-depth_time > NATIVE_PREFLIGHT_MAXIMUM_AGE_MS):
                valid = False
                self.last_issue = "NATIVE_PERCEPTION_DEPTH_SOURCE_INVALID_OR_EXPIRED"
            else:
                frame_time = min(updated, depth_time)
        if valid and self.minimum_route_clearance_m is not None:
            variance = payload.get("localization_covariance_m2")
            observed = payload.get("localization_observed_at_unix_ms")
            if (type(observed) is not int or observed < 0 or not 0 <= now_unix_ms - observed
                    <= NATIVE_PREFLIGHT_MAXIMUM_AGE_MS
                    or variance is None):
                valid = False
                self.last_issue = "NATIVE_ROUTE_LOCALIZATION_EVIDENCE_UNAVAILABLE"
            else:
                try:
                    # 先验证本次方差，再取窗口最差值；否则 NaN 可能被以前的合法值遮住。
                    self.tracking_budget = build_tracking_corridor_budget(
                        self.minimum_route_clearance_m, localization_covariance_m2=variance)
                    self._worst_variance_m2 = max(variance, self._worst_variance_m2 or 0.)
                    self.tracking_budget = build_tracking_corridor_budget(
                        self.minimum_route_clearance_m,
                        localization_covariance_m2=self._worst_variance_m2)
                    frame_time = min(observed, frame_time)
                except (ValueError, TypeError, OverflowError) as error:
                    valid = False
                    self.last_issue = ("NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED:"
                                       + type(error).__name__)
        regressed = valid and (
            sequence < self.last_sequence
            or (self.last_health_time is not None and updated < self.last_health_time)
            or (self.last_frame_time is not None and frame_time < self.last_frame_time)
        )
        interrupted = (self.last_frame_time is not None
                       and now_unix_ms - self.last_frame_time > NATIVE_PREFLIGHT_MAXIMUM_AGE_MS)
        if not valid or regressed or interrupted:
            if self.last_issue is None:
                self.last_issue = ("NATIVE_PREFLIGHT_SOURCE_REGRESSED" if regressed
                                   else "NATIVE_PREFLIGHT_SOURCE_INTERRUPTED")
            self._reset_window()
            return False
        distinct = sequence > self.last_sequence and (
            self.last_frame_time is None or frame_time > self.last_frame_time)
        self.last_sequence = sequence
        self.last_health_time = updated
        if distinct:
            self.independent_frames += 1
            self.last_frame_time = frame_time
            if self.first_frame_time is None:
                self.first_frame_time = frame_time
        # 重写健康文件或重复轮询不能刷新真正获得独立帧计数的来源时刻。
        self.source_observed_at_unix_ms = self.last_frame_time
        ready = (
            self.independent_frames >= 3
            and self.last_frame_time - self.first_frame_time >= self.stable_window_ms
        )
        return ready


# 功能：
#   有界读取当前运行的普通健康文件，拒绝静态链接、读取中替换、重复键和非有限数值。
# 输入：
#   path：当前运行健康快照的固定路径。
# 输出：
#   value：最多二百五十六 KiB 的有限 JSON 对象。
def _read_native_health(path: Path) -> dict:
    try:
        value = read_runtime_object(path, maximum_bytes=262_144)
    except ValueError as error:
        raise ValueError("NATIVE_PERCEPTION_HEALTH_INVALID") from error
    return value


# 功能：
#   1. 在总期限内等待原生健康快照持续推进，并返回保持原始有效期的就绪回执。
#   2. 同时最多一个后台文件读取；取消只停止接收读取结果，不冒充能中断阻塞的系统读取。
# 输入：
#   path：本次运行的健康快照路径。
#   timeout_seconds：总等待秒数，上限六十秒。
#   minimum_route_clearance_m：可选的路线最小净空米数，用实测协方差重新核算余量。
#   receiver：可选的运行专属接收器，由调用者持有并回收；配置后不回退到磁盘旧快照。
#   evidence：可选的实时诊断容器，失败和取消也保留统计。
#   stable_window_ms：连续独立观测所需的毫秒跨度，不改变单份观测有效期。
#   require_source_timestamps：是否强制检查深度的原始观测时刻。
# 输出：
#   receipt：含来源时刻、有效期、独立帧数量及跟踪余量的就绪回执。
async def wait_for_native_perception(
    path: Path, *, timeout_seconds: float = 30., minimum_route_clearance_m: float | None = None,
    receiver: PerceptionHealthReceiver | None = None,
    evidence: dict | None = None, stable_window_ms: int = 100,
    require_source_timestamps: bool = False,
) -> dict:
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
        raise ValueError("NATIVE_PERCEPTION_PREFLIGHT_TIMEOUT_INVALID")
    started = time.monotonic()
    deadline = started + timeout_seconds
    if minimum_route_clearance_m is not None:
        build_tracking_corridor_budget(minimum_route_clearance_m)
    readiness = NativePerceptionReadiness(minimum_route_clearance_m=minimum_route_clearance_m,
        stable_window_ms=stable_window_ms, require_source_timestamps=require_source_timestamps)
    if evidence is not None and type(evidence) is not dict:
        raise ValueError("NATIVE_PREFLIGHT_EVIDENCE_INVALID")
    diagnostics = evidence if evidence is not None else {}
    diagnostics.clear()
    diagnostics.update(status="waiting", stable_window_ms=stable_window_ms,
                       maximum_source_age_ms=NATIVE_PREFLIGHT_MAXIMUM_AGE_MS,
                       checks=0, interruption_count=0, issue_counts={})
    last_issue = "NATIVE_PERCEPTION_NOT_RECEIVED"

    pending = None
    latest = None
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                if receiver is not None:
                    packet = receiver.read_latest()
                    if packet is not None:
                        latest = packet
                    # 重复缓存不增加独立帧，过期、异常包和流中断仍由同一窗口拒绝。
                    payload = latest if latest is not None else {}
                else:
                    if pending is None:
                        pending = asyncio.create_task(asyncio.to_thread(_read_native_health, path))
                    # 慢读取仍占用唯一槽位，单轮超时不会堆积后台线程任务。
                    payload = await asyncio.wait_for(
                        asyncio.shield(pending), timeout=min(.25, remaining))
                    pending = None
                if time.monotonic() >= deadline:
                    last_issue = "NATIVE_PERCEPTION_READ_DEADLINE_EXCEEDED"
                    diagnostics["last_issue"] = last_issue
                    counts = diagnostics["issue_counts"]
                    counts[last_issue] = counts.get(last_issue, 0) + 1
                    break  # 总期限之后返回的磁盘结果不能取得就绪资格。
                now = int(time.time() * 1000)
                diagnostics["checks"] += 1
                previous_frames = readiness.independent_frames
                if readiness.observe(payload, now_unix_ms=now):
                    if time.monotonic() >= deadline:
                        break
                    receipt = {"ready": True, "independent_frames": readiness.independent_frames,
                            "latest_sequence": readiness.last_sequence,
                            "confirmed_at_unix_ms": now,
                            "source_observed_at_unix_ms": readiness.source_observed_at_unix_ms,
                            "valid_until_unix_ms": (readiness.source_observed_at_unix_ms
                                                    + NATIVE_PREFLIGHT_MAXIMUM_AGE_MS),
                            "elapsed_seconds": time.monotonic() - started,
                            "transport": "run-scoped-datagram" if receiver is not None else "file",
                            "tracking_budget": readiness.tracking_budget,
                            "uncertainty_basis": ("live-native-localization"
                                if readiness.tracking_budget is not None else "not-assessed"),
                            "truth_correction_applied": False}
                    diagnostics.update(receipt)
                    diagnostics["status"] = "ready"
                    return receipt
                last_issue = readiness.last_issue or "NATIVE_INDEPENDENT_FRAMES_NOT_READY"
                if previous_frames and readiness.independent_frames == 0:
                    diagnostics["interruption_count"] += 1
            except (OSError, ValueError, TimeoutError, RecursionError) as error:
                if pending is not None and pending.done():
                    pending = None
                readiness.observe({}, now_unix_ms=int(time.time() * 1000))
                last_issue = ("NATIVE_PERCEPTION_NOT_RECEIVED"
                              if isinstance(error, FileNotFoundError)
                              else "NATIVE_PERCEPTION_READ_" + type(error).__name__)
            diagnostics["last_issue"] = last_issue
            counts = diagnostics["issue_counts"]
            key = last_issue if last_issue in counts or len(counts) < 32 else "OTHER"
            counts[key] = counts.get(key, 0) + 1
            await asyncio.sleep(max(0., min(.05, deadline - time.monotonic())))
        diagnostics["status"] = "timeout"
    except BaseException as error:
        diagnostics["status"] = (
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
        diagnostics["failure_type"] = type(error).__name__
        raise
    finally:
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        if diagnostics["status"] != "ready":
            diagnostics["last_issue"] = last_issue
        if pending is not None:
            if not pending.done():
                pending.cancel()
            # 取走已完成异常或取消结果，避免无人接收的后台 Task 异常泄漏到事件循环。
            with suppress(asyncio.CancelledError, Exception):
                await pending
    raise RuntimeError("NATIVE_PERCEPTION_NOT_READY_BEFORE_ARM:" + last_issue)
