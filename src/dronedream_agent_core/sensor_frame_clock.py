"""Original scene clocks for isolated Gazebo rendering, not flight authority.

The source identity must be selected before subscribing. A header cannot opt a
consumer into trusting an arbitrary renderer. Direct (non-replica) streams keep
their explicitly weaker host-receipt clock; mixed or partial clocks are rejected.
No Gazebo dependency is needed to validate the protobuf-shaped boundary.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass

_PREFIX = "dronedream_scene_"
_FIELDS = {"epoch", "sha256", "sequence", "source_unix_ns", "simulation_ns"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DECIMAL = re.compile(r"[0-9]{1,19}\Z")
MAXIMUM_FRAME_AGE_NS = 250_000_000
_MAXIMUM_NS = (1 << 63) - 1


# 功能：
#   将有界十进制文本解析为非负 64 位整数，不接受指数、符号或隐式类型转换。
# 输入：
#   value：场景报头中的整数文本。
# 输出：
#   number：通过范围和格式校验的整数。
def _integer(value: str) -> int:
    if (type(value) is not str or not _DECIMAL.fullmatch(value)
            or (number := int(value)) > _MAXIMUM_NS):
        raise ValueError("SENSOR_SOURCE_INTEGER_INVALID")
    return number


# 功能：
#   判断单调钟是否为可安全转换的非负有限数，拒绝布尔值及超大整数转换溢出。
# 输入：
#   value：来源或接收侧的单调时钟秒数。
# 输出：
#   valid：数值类型、符号及有限性均合法时为 True。
def _valid_monotonic(value) -> bool:
    try:
        valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
    except OverflowError:
        valid = False
    return valid


# 功能：
#   有界读取场景时钟元数据，拒绝畸形行、重复字段及将单字符串误作值数组的结构。
# 输入：
#   message：包含 protobuf 形状报头的图像消息。
# 输出：
#   values：场景前缀字段的字符串映射；直连图像可为空映射。
def _source_metadata(message) -> dict[str, str]:
    rows = getattr(getattr(message, "header", None), "data", ())
    values = {}
    try:
        if len(rows) > 32:
            raise ValueError("SENSOR_SOURCE_HEADER_CAPACITY")
        for index, item in enumerate(rows):
            key = getattr(item, "key", None)
            if type(key) is not str:
                raise ValueError("SENSOR_SOURCE_HEADER_INVALID")
            if index >= 32 or len(key) > 128:
                raise ValueError("SENSOR_SOURCE_HEADER_CAPACITY")
            if not key.startswith(_PREFIX):
                continue
            name = key[len(_PREFIX):]
            parts = getattr(item, "value", None)
            if (name not in _FIELDS | {"basis"} or name in values or isinstance(parts, (str, bytes))
                    or len(parts) != 1 or type(parts[0]) is not str or len(parts[0]) > 64):
                raise ValueError("SENSOR_SOURCE_HEADER_AMBIGUOUS")
            values[name] = parts[0]
    except (TypeError, AttributeError, IndexError, KeyError, OverflowError) as error:
        raise ValueError("SENSOR_SOURCE_HEADER_INVALID") from error
    return values


# 功能：
#   尊重 protobuf 字段存在标记，区分未填充的默认 stamp 与真实仿真零时刻。
# 输入：
#   message：可能包含 header.stamp 的图像消息。
# 输出：
#   stamp：实际存在的时间戳对象；未填充时为 None。
def _publisher_stamp(message):
    header = getattr(message, "header", None)
    if (hasattr(message, "HasField") and not message.HasField("header")) or (
        hasattr(header, "HasField") and not header.HasField("stamp")
    ):
        stamp = None
    else:
        stamp = getattr(header, "stamp", None)
    return stamp


@dataclass(frozen=True, slots=True)
class SensorFrameTime:
    """Original/receipt UNIX nanoseconds plus local monotonic seconds.

    Scene-source clocks retain render age; host-receipt clocks cannot establish
    exposure time. Neither kind alone certifies geometry or control validity.
    """
    source_unix_ns: int
    received_unix_ns: int
    sample_monotonic_seconds: float
    received_monotonic_seconds: float
    clock_kind: str
    epoch: str | None = None
    scene_sha256: str | None = None
    sequence: int | None = None
    simulation_ns: int | None = None

    # 功能：
    #   计算图像到达时距原始来源的纳秒年龄，不将主机接收钟误称为硬件曝光钟。
    # 输入：
    #   self：已接入的来源与接收时间记录。
    # 输出：
    #   age：接收 UNIX 纳秒减去来源 UNIX 纳秒。
    @property
    def age_at_receipt_ns(self) -> int:
        age = self.received_unix_ns - self.source_unix_ns
        return age


# 功能：
#   1. 在模型编码边界复核来源时钟，不允许丢失场景标识或把来源时间改成处理完成时间。
#   2. 场景与直连记录均按原始接收时钟重建并比较，拒绝仅数值相等的错误字段类型。
# 输入：
#   message：正在编码的图像消息。
#   frame_time：接入时保留的完整时钟；无场景报头的旧直连调用允许为 None。
#   sample_mono：本次编码传入的来源单调钟秒数。
#   sample_unix_ms：本次编码传入的来源 UNIX 毫秒数。
# 输出：
#   None：不返回业务数据。
def require_model_frame_time(
    message, frame_time: SensorFrameTime | None, sample_mono: float, sample_unix_ms: int
) -> None:
    has_scene = bool(_source_metadata(message))
    if (not _valid_monotonic(sample_mono) or type(sample_unix_ms) is not int
            or sample_unix_ms < 0):
        raise ValueError("MODEL_RGB_SOURCE_TIME_INVALID")
    if frame_time is None:
        if has_scene:
            raise ValueError("MODEL_RGB_SCENE_CLOCK_REQUIRED")
        return
    if (not isinstance(frame_time, SensorFrameTime)
            or type(frame_time.source_unix_ns) is not int
            or type(frame_time.received_unix_ns) is not int
            or any(value is not None and type(value) is not int
                   for value in (frame_time.sequence, frame_time.simulation_ns))
            or not _valid_monotonic(frame_time.sample_monotonic_seconds)
            or not _valid_monotonic(frame_time.received_monotonic_seconds)):
        raise ValueError("MODEL_RGB_FRAME_CLOCK_INVALID")
    if (frame_time.sample_monotonic_seconds != sample_mono
            or frame_time.source_unix_ns // 1_000_000 != sample_unix_ms):
        raise ValueError("MODEL_RGB_SCENE_CLOCK_REDATED")
    if not has_scene and frame_time.clock_kind != "host-receipt":
        raise ValueError("MODEL_RGB_SCENE_SOURCE_MISSING")
    if not has_scene and any(value is not None for value in (
        frame_time.epoch, frame_time.scene_sha256, frame_time.sequence
    )):
        raise ValueError("MODEL_RGB_FRAME_CLOCK_INVALID")
    checked = SensorFrameClock(
        expected_scene_epoch=frame_time.epoch if frame_time.clock_kind != "native-simulation" else None,
        expected_native_epoch=frame_time.epoch if frame_time.clock_kind == "native-simulation" else None,
    ).admit(message,
        received_unix_ns=frame_time.received_unix_ns,
        received_monotonic_seconds=frame_time.received_monotonic_seconds)
    if checked != frame_time:
        raise ValueError("MODEL_RGB_SCENE_SOURCE_CHANGED")


class SensorFrameClock:
    """One ordered camera stream; caller owns synchronization of admit()."""

    # 功能：
    #   固定单一相机流的场景身份并清空顺序状态，消息自身不能启用对未知场景的信任。
    # 输入：
    #   self：待初始化的时钟实例。
    #   expected_scene_epoch：订阅前选定的场景摘要；None 表示较弱的主机接收钟。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, expected_scene_epoch: str | None = None,
                 expected_native_epoch: str | None = None):
        if expected_native_epoch is not None:
            if (expected_scene_epoch is not None or type(expected_native_epoch) is not str
                    or not _DIGEST.fullmatch(expected_native_epoch)):
                raise ValueError("SENSOR_EXPECTED_NATIVE_EPOCH_INVALID")
        self.expected_native_epoch = expected_native_epoch
        selected_epoch = expected_native_epoch or expected_scene_epoch
        if expected_scene_epoch is not None and (
            type(expected_scene_epoch) is not str or not _DIGEST.fullmatch(expected_scene_epoch)
        ):
            raise ValueError("SENSOR_EXPECTED_SCENE_EPOCH_INVALID")
        self.expected_scene_epoch = selected_epoch
        self._previous: SensorFrameTime | None = None
        self._direct_simulation_ns: int | None = None

    # 功能：
    #   1. 校验接收钟、完整场景标识及发布者顺序，拒绝过期、未来、重复和回退图像。
    #   2. 用原始 UNIX 年龄反推本地单调来源时刻；所有检查通过后才推进顺序状态。
    # 输入：
    #   self：调用方独占或已加锁的单相机时钟。
    #   message：待接入的图像消息。
    #   received_unix_ns：实际收到图像时的 UNIX 纳秒。
    #   received_monotonic_seconds：实际收到图像时的本地单调钟秒数。
    # 输出：
    #   result：通过身份、时效和顺序校验的来源时钟记录。
    def admit(
        self, message, *, received_unix_ns: int, received_monotonic_seconds: float
    ) -> SensorFrameTime:
        if (type(received_unix_ns) is not int or not 0 < received_unix_ns <= _MAXIMUM_NS
                or not _valid_monotonic(received_monotonic_seconds)):
            raise ValueError("SENSOR_RECEIPT_CLOCK_INVALID")
        values = _source_metadata(message)
        stamp = _publisher_stamp(message)
        if self.expected_scene_epoch is None:
            if values:
                raise ValueError("SENSOR_SCENE_SOURCE_NOT_CONFIGURED")
            if stamp is not None:
                sec, ns = getattr(stamp, "sec", None), getattr(stamp, "nsec", None)
                if (type(sec) is not int or type(ns) is not int or sec < 0
                        or not 0 <= ns < 1_000_000_000
                        or sec * 1_000_000_000 + ns > _MAXIMUM_NS):
                    raise ValueError("SENSOR_DIRECT_STAMP_INVALID")
                sim = sec * 1_000_000_000 + ns
                if self._direct_simulation_ns is not None and sim <= self._direct_simulation_ns:
                    raise ValueError("SENSOR_DIRECT_FRAME_REPLAY_OR_REGRESSION")
            elif self._direct_simulation_ns is not None:
                raise ValueError("SENSOR_DIRECT_STAMP_MISSING")
            else:
                sim = None
            result = SensorFrameTime(received_unix_ns, received_unix_ns,
                received_monotonic_seconds, received_monotonic_seconds, "host-receipt",
                simulation_ns=sim)
            if self._previous is not None and (
                received_unix_ns < self._previous.received_unix_ns
                or received_monotonic_seconds <= self._previous.received_monotonic_seconds
            ):
                raise ValueError("SENSOR_DIRECT_RECEIPT_REGRESSED")
            self._direct_simulation_ns = sim
            self._previous = result
            return result
        basis = values.get("basis", "scene-snapshot")
        required_basis = "native-preupdate" if self.expected_native_epoch else "scene-snapshot"
        if (set(values) - {"basis"} != _FIELDS or basis != required_basis
                or values["epoch"] != self.expected_scene_epoch
                or not _DIGEST.fullmatch(values["sha256"])):
            raise ValueError("SENSOR_SCENE_IDENTITY_MISMATCH")
        source, sequence, sim = (_integer(values[key])
                                 for key in ("source_unix_ns", "sequence", "simulation_ns"))
        sec, ns = getattr(stamp, "sec", None), getattr(stamp, "nsec", None)
        if (type(sec) is not int or type(ns) is not int or sec < 0
                or not 0 <= ns < 1_000_000_000 or sec * 1_000_000_000 + ns != sim):
            raise ValueError("SENSOR_SCENE_STAMP_MISMATCH")
        age = received_unix_ns - source
        if source <= 0 or sequence <= 0 or not 0 <= age <= MAXIMUM_FRAME_AGE_NS:
            raise ValueError("SENSOR_SCENE_FRAME_EXPIRED_OR_FUTURE")
        sample_monotonic = received_monotonic_seconds - age / 1e9
        if sample_monotonic < 0:
            raise ValueError("SENSOR_SCENE_MONOTONIC_ORIGIN_INVALID")
        result = SensorFrameTime(source, received_unix_ns, sample_monotonic,
            received_monotonic_seconds,
            "native-simulation" if self.expected_native_epoch else "scene-source", values["epoch"],
            values["sha256"], sequence, sim)
        previous = self._previous
        if previous is not None and (
            sequence <= previous.sequence or sim <= previous.simulation_ns
            or source <= previous.source_unix_ns
            or received_unix_ns < previous.received_unix_ns
            or received_monotonic_seconds < previous.received_monotonic_seconds
            or sample_monotonic <= previous.sample_monotonic_seconds
        ):
            raise ValueError("SENSOR_SCENE_CLOCK_REPLAY_OR_REGRESSION")
        # 允许跳过中间完整帧，但下一帧仍必须携带自己的原始年龄；拒绝帧不能推进基线。
        self._previous = result
        return result


class SensorImageIngress:
    """Bounded RGB/depth clock history and diagnostics, no frame or control queue."""

    # 功能：
    #   为 RGB 与深度分别建立顺序时钟，共享锁保护固定容量的来源历史和诊断计数。
    # 输入：
    #   self：待初始化的图像入口。
    #   expected_scene_epoch：订阅前选定的场景身份，直连流为 None。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, expected_scene_epoch: str | None = None,
                 expected_native_epoch: str | None = None):
        self._clocks = {kind: SensorFrameClock(expected_scene_epoch=expected_scene_epoch,
                                             expected_native_epoch=expected_native_epoch)
                        for kind in ("rgb", "depth")}
        self._lock = threading.Lock()
        self._history: OrderedDict[tuple[str, float], SensorFrameTime] = OrderedDict()
        self._accepted = Counter()
        self._rejected = Counter()
        self._epoch = expected_scene_epoch
        self._native_epoch = expected_native_epoch
        self._cadence_previous: dict[str, SensorFrameTime] = {}
        self._cadence = {kind: dict(interval_count=0, source_span_ns=0,
            receive_span_seconds=0., simulation_interval_count=0, simulation_span_ns=0)
            for kind in self._clocks}

    # 功能：
    #   1. 串行接入指定相机流，记录拒绝原因但不排队坏帧、不刷新旧帧时间。
    #   2. 仅用成功接入的原始时间累计频率诊断，并保留最近四十八份精确来源记录。
    # 输入：
    #   self：维护两种图像流状态的入口实例。
    #   kind：rgb 或 depth 流类型。
    #   message：待验证的图像消息。
    #   received_unix_ns：实际接收 UNIX 纳秒。
    #   received_monotonic_seconds：实际接收单调钟秒数。
    # 输出：
    #   value：接受的原始时钟记录；输入被拒绝时为 None。
    def admit(
        self, kind, message, *, received_unix_ns: int, received_monotonic_seconds: float
    ) -> SensorFrameTime | None:
        with self._lock:
            if type(kind) is not str or kind not in self._clocks:
                self._rejected["unknown:SENSOR_STREAM_KIND_INVALID"] += 1
                value = None
                return value
            try:
                value = self._clocks[kind].admit(message, received_unix_ns=received_unix_ns,
                    received_monotonic_seconds=received_monotonic_seconds)
            except ValueError as error:
                self._rejected[kind + ":" + str(error)] += 1
                value = None
                return value
            self._accepted[kind] += 1
            # 只统计已接入来源间隔，不把重复帧的迟到当作新采样，也不据此授予运动权限。
            previous = self._cadence_previous.get(kind)
            if previous is not None:
                cadence = self._cadence[kind]
                cadence["interval_count"] += 1
                cadence["source_span_ns"] += value.source_unix_ns - previous.source_unix_ns
                cadence["receive_span_seconds"] += (
                    value.received_monotonic_seconds - previous.received_monotonic_seconds)
                if value.simulation_ns is not None and previous.simulation_ns is not None:
                    cadence["simulation_interval_count"] += 1
                    cadence["simulation_span_ns"] += value.simulation_ns - previous.simulation_ns
            self._cadence_previous[kind] = value
            self._history[(kind, value.sample_monotonic_seconds)] = value
            while len(self._history) > 48:
                self._history.popitem(last=False)
            return value

    # 功能：
    #   按流类型及原始单调时刻查回同一帧，未命中时不返回其他较新图像的时钟。
    # 输入：
    #   self：保存有限来源历史的入口实例。
    #   kind：rgb 或 depth 流类型。
    #   source_monotonic_seconds：需要匹配的原始来源单调钟秒数。
    # 输出：
    #   value：精确命中的来源记录；参数非法或已淘汰时为 None。
    def lookup(self, kind, source_monotonic_seconds) -> SensorFrameTime | None:
        if (type(kind) is not str or kind not in self._clocks
                or not _valid_monotonic(source_monotonic_seconds)):
            value = None
            return value
        with self._lock:
            value = self._history.get((kind, source_monotonic_seconds))
            return value

    # 功能：
    #   在锁内复制接入、拒绝及间隔诊断，修改快照不改变内部统计；诊断不授予飞行资质。
    # 输入：
    #   self：当前图像入口实例。
    # 输出：
    #   result：包含有限历史数量及频率统计的独立字典。
    def summary(self) -> dict:
        with self._lock:
            result = {"scene_epoch": self._epoch, "native_epoch": self._native_epoch,
                    "accepted": dict(self._accepted),
                    "rejected": dict(self._rejected), "history_entries": len(self._history),
                    "accepted_frame_cadence": {kind: dict(values)
                                               for kind, values in self._cadence.items()},
                    "cadence_scope": "accepted frame intervals only; no freshness or qualification",
                    "qualification_granted": False}
            return result
