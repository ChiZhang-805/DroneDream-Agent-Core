"""Bounded sensor failure vocabulary shared by sampling and preflight reports.

Only known codes cross the diagnostic channel. Raw exception messages, file
paths and sensor payloads are never copied into these health summaries.
"""

_ALIASES = {
    "native IMU source has expired": "NATIVE_IMU_EXPIRED",
    "native state source has expired": "NATIVE_STATE_EXPIRED",
    "native pose source has expired": "NATIVE_POSE_EXPIRED",
    "native pose has a future or invalid source time": "NATIVE_POSE_TIME_INVALID",
    "native state contains a future or invalid source timestamp": "NATIVE_STATE_TIME_INVALID",
    "flight state timestamps must increase strictly": "FLIGHT_STATE_TIMESTAMP_NOT_INCREASING",
}
_CODES = frozenset({
    *_ALIASES.values(), "NATIVE_ATTITUDE_EXPIRED", "NATIVE_POSE_IMU_NOT_SYNCHRONIZED",
    "SOURCE_SAMPLE_EXPIRED", "SOURCE_TIME_IN_FUTURE", "SOURCE_RECEIVE_TIME_INVALID",
    "LOCALIZATION_SAMPLE_EXPIRED", "LOCALIZATION_COVARIANCE_UNAVAILABLE",
    "LOCALIZATION_TIMESTAMP_IN_FUTURE", "NATIVE_LOCALIZATION_COVARIANCE_UNAVAILABLE",
    "NATIVE_MAP_BINDING_CHANGED_DURING_EXECUTION", "NATIVE_MAP_FRAME_BINDING_HASH_MISMATCH",
    "NATIVE_POSE_CLOCK_REGRESSED", "NATIVE_FLIGHT_STATE_TIMESTAMP_REGRESSED",
    "NATIVE_STATE_STREAM_EXPIRED", "NATIVE_STATE_STREAM_WARMING",
    "NATIVE_STATE_STREAM_NOT_YET_AVAILABLE", "SAMPLER_STOPPED",
    "NATIVE_ODOMETRY_RESET_WAITING_FOR_COHERENT_STATE",
    "IMAGE_NATIVE_SOURCE_CLOCK_UNAVAILABLE", "IMAGE_SOURCE_HISTORY_UNAVAILABLE",
    "IMAGE_SOURCE_RECEIPT_EXPIRED", "IMAGE_SOURCE_ODOMETRY_EXPIRED",
    "FLIGHT_STATE_REFRESH_NOT_SAME_HISTORY_SLOT",
    "IMAGE_NATIVE_STATE_UNAVAILABLE", "IMAGE_NATIVE_POSE_NOT_SYNCHRONIZED",
    "IMAGE_NATIVE_ALIGNMENT_TIME_INVALID", "IMAGE_NATIVE_ALIGNMENT_UNCERTAINTY_INVALID",
    "NATIVE_CONTROL_REFRESH_BINDING_CHANGED", "NATIVE_CONTROL_REFRESH_CLOCK_REGRESSED",
    "PERCEPTION_STREAM_STALE", "LOCALIZATION_COVARIANCE_EXCEEDED",
    "METRIC_MAP_EVIDENCE_INSUFFICIENT", "PERCEPTION_FRAME_NOT_RECEIVED",
})


# 功能：
#   将已知传感器拒收原因保留为固定代码，未知异常只保留通用类别，不传递原文。
# 输入：
#   value：异常对象或感知链已经产生的错误代码。
# 输出：
#   code：有界、无传感器原始内容的诊断字符串。
def sensor_issue_code(value: object) -> str:
    fallback = "SENSOR_INPUT_REJECTED"
    if isinstance(value, (OSError, ValueError, TypeError, RuntimeError)):
        fallback = ("OSError" if isinstance(value, OSError) else
                    "ValueError" if isinstance(value, ValueError) else
                    "TypeError" if isinstance(value, TypeError) else "RuntimeError")
        value = value.args[0] if len(value.args) == 1 else None
    if type(value) is not str or len(value) > 192:
        return fallback
    for prefix in ("ValueError:", "RuntimeError:", "NATIVE_STATE_STREAM_UNAVAILABLE:"):
        value = value.removeprefix(prefix)
    code = _ALIASES.get(value, value)
    return code if code in _CODES else fallback


# 功能：
#   把健康消息中的错误压缩为最多四个固定代码，异常输入不能扩张实时通信报文。
# 输入：
#   values：候选错误数组。
# 输出：
#   codes：去重、有界的错误代码列表。
def sensor_issue_codes(values: object) -> list[str]:
    if type(values) not in (list, tuple):
        return []
    codes = list(dict.fromkeys(sensor_issue_code(value) for value in values[:4]))
    return codes
