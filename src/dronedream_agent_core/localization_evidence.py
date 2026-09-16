"""Native estimator uncertainty, never a substitute for missing localization.

MAVLink ODOMETRY packs the upper triangle of a 6x6 pose covariance.
Its leading 3x3 block is position covariance in square metres. We return
a conservative direction-independent variance bound, including correlations,
not a fabricated diagonal or an exact eigenvalue. No transport or clocks live
here; producers preserve the original packet age for this consumer.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence

from .source_clock import source_received_at_unix_ms


# 功能：
#   将原生协方差的 NaN 转为未知值，不把未报告的相关项伪造为零；首项未知表示整体未知。
# 输入：
#   values：6×6 位姿协方差上三角按行打包的 21 项序列。
# 输出：
#   normalized：规范化数值或 None 元素列表；整体未知时为 None。
def normalize_native_pose_covariance(values: Sequence[object]) -> list[float | None] | None:
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray))
            or len(values) != 21):
        raise ValueError("LOCALIZATION_COVARIANCE_SHAPE_INVALID")
    normalized = []
    for value in values:
        if type(value) is float and math.isnan(value):
            normalized.append(None)
            continue
        if (type(value) not in (int, float)
                or not -sys.float_info.max <= value <= sys.float_info.max):
            raise ValueError("LOCALIZATION_COVARIANCE_VALUE_INVALID")
        normalized.append(float(value))
    # 仍检查其余字段，不能让首项 NaN 隐藏 Infinity 或错误类型。
    if normalized[0] is None:
        normalized = None
    return normalized


# 功能：
#   1. 校验位置协方差块的非负性及半正定约束，返回方向无关的保守方差上界。
#   2. 完整相关项使用绝对行和，缺失相关项使用迹，不假设位置误差相互独立。
# 输入：
#   packed_covariance：包含 21 项的规范化位姿协方差，只读取其中的位置块。
# 输出：
#   bound：位置误差最大方向方差的保守上界，单位为平方米。
def position_variance_bound_m2(packed_covariance: Sequence[object]) -> float:
    if (not isinstance(packed_covariance, Sequence)
            or isinstance(packed_covariance, (str, bytes, bytearray))
            or len(packed_covariance) != 21):
        raise ValueError("LOCALIZATION_COVARIANCE_SHAPE_INVALID")
    values = []
    for index in (0, 1, 2, 6, 7, 11):
        value = packed_covariance[index]
        if value is None and index in {1, 2, 7}:
            values.append(None)
            continue
        if (type(value) not in (int, float)
                or not -sys.float_info.max <= value <= sys.float_info.max):
            raise ValueError("LOCALIZATION_COVARIANCE_UNAVAILABLE")
        values.append(float(value))
    xx, xy, xz, yy, yz, zz = values
    scale = max(abs(value) for value in values if value is not None)
    # 全零不是本链路认可的理想传感器证明；先缩放再计算小行列式避免数值溢出。
    if scale == 0.0 or min(xx, yy, zz) < 0.0:
        raise ValueError("LOCALIZATION_COVARIANCE_INVALID")
    if any(value is None for value in (xy, xz, yz)):
        # 半正定矩阵的最大特征值不超过迹；未知交叉项不能简单按零处理。
        for first, second, cross in ((xx, yy, xy), (xx, zz, xz), (yy, zz, yz)):
            if (cross is not None
                    and (cross / scale) ** 2 > (first / scale) * (second / scale) + 1e-10):
                raise ValueError("LOCALIZATION_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE")
        bound = xx + yy + zz
        if not math.isfinite(bound) or bound > 10_000.0:
            raise ValueError("LOCALIZATION_COVARIANCE_OUT_OF_RANGE")
        return bound
    a, b, c, d, e, f = (value / scale for value in values)
    principal_minors = (a * d - b * b, a * f - c * c, d * f - e * e,
                        a * d * f + 2 * b * c * e - a * e * e - d * c * c - f * b * b)
    if min(principal_minors) < -1e-10:
        raise ValueError("LOCALIZATION_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE")
    bound = max(xx + abs(xy) + abs(xz), yy + abs(xy) + abs(yz),
                zz + abs(xz) + abs(yz))
    if not math.isfinite(bound) or bound > 10_000.0:
        raise ValueError("LOCALIZATION_COVARIANCE_OUT_OF_RANGE")
    return bound


# 功能：
#   1. 读取原生里程计的位置不确定性与最初接收时刻，不生成缺失的测量。
#   2. 区分未收到可选信息和已收到但损坏的信息，后者明确拒绝。
# 输入：
#   identity：包含动力学和可选里程计来源的遥测对象。
#   now_unix_ms：当前消费 UNIX 毫秒时刻。
#   maximum_age_ms：允许的最大年龄，不超过 250 毫秒。
# 输出：
#   evidence：位置方差上界与接收时刻组成的二元组，缺失部分为 None。
def native_localization_evidence(
    identity: Mapping[str, object], *, now_unix_ms: int, maximum_age_ms: int,
) -> tuple[float | None, int | None]:
    if type(maximum_age_ms) is not int or not 0 < maximum_age_ms <= 250:
        raise ValueError("LOCALIZATION_MAXIMUM_AGE_INVALID")
    if type(now_unix_ms) is not int or not 0 <= now_unix_ms < 2**63:
        raise ValueError("LOCALIZATION_CONSUMER_TIME_INVALID")
    if not isinstance(identity, Mapping):
        raise ValueError("LOCALIZATION_IDENTITY_INVALID")
    if "dynamics" not in identity:
        evidence = (None, None)
        return evidence
    dynamics = identity["dynamics"]
    if not isinstance(dynamics, Mapping):
        raise ValueError("LOCALIZATION_DYNAMICS_INVALID")
    if "sources" not in dynamics:
        evidence = (None, None)
        return evidence
    sources = dynamics["sources"]
    if not isinstance(sources, Mapping):
        raise ValueError("LOCALIZATION_SOURCES_INVALID")
    if "odometry" not in sources:
        evidence = (None, None)
        return evidence
    packet = sources["odometry"]
    if not isinstance(packet, Mapping):
        raise ValueError("LOCALIZATION_PACKET_INVALID")
    # 此处仅使用方向无关的度量方差上界，不用里程计标签推测地图航向或旋转位置。
    # 实际位姿仍来自独立 position_velocity_ned 流及固定部署坐标契约。
    frame = packet.get("frame_id")
    if type(frame) is not str or frame not in {
        "ESTIM_NED", "LOCAL_NED", "LOCAL_FRD", "BODY_NED", "VISION_NED",
    }:
        raise ValueError("LOCALIZATION_FRAME_UNSUPPORTED")
    timestamp = packet.get("timestamp_us")
    collected = dynamics.get("collected_at_unix_ms")
    age = packet.get("sample_age_seconds")
    if any(type(value) is not int or not 0 <= value < 2**63
           for value in (timestamp, collected)):
        raise ValueError("LOCALIZATION_TIMESTAMP_INVALID")
    if type(age) not in (int, float) or not 0 <= age <= sys.float_info.max:
        raise ValueError("LOCALIZATION_AGE_INVALID")
    if age > maximum_age_ms / 1_000:
        raise ValueError("LOCALIZATION_SAMPLE_EXPIRED")
    if collected > now_unix_ms:
        raise ValueError("LOCALIZATION_TIMESTAMP_IN_FUTURE")
    observed = source_received_at_unix_ms(
        packet, collected_at_unix_ms=collected, now_unix_ms=now_unix_ms,
        maximum_age_ms=maximum_age_ms,
    )
    covariance = packet.get("pose_covariance_upper_m2")
    if covariance is None:
        evidence = (None, observed)
        return evidence
    if not isinstance(covariance, list | tuple):
        raise ValueError("LOCALIZATION_COVARIANCE_SHAPE_INVALID")
    evidence = (position_variance_bound_m2(covariance), observed)
    return evidence
