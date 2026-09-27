"""Project bounded, allowlisted executor failures into desktop status messages."""

from pathlib import Path

from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   从本次执行器失败回执提取已知原因；不公开原始异常，不把诊断记录当作执行授权。
# 输入：
#   run_dir：本次运行独占目录；english：对话是否使用英语。
# 输出：
#   detail：白名单错误码与用户可读说明；缺失、损坏或未知原因返回 None。
def executor_failure_detail(run_dir: Path, *, english: bool) -> tuple[str, str] | None:
    try:
        value = decode_json(read_plugin_file(run_dir / "offboard_timing.json", limit=2 * 1024**2),
                            limit=2 * 1024**2, node_limit=100_000)
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("status") != "failed":
        return None
    failure = value.get("failure")
    # 功能：把已确认的定位启动失败与运行中失联分开显示；不公开任意异常或改变起飞权限。
    # 输入：本次执行器的白名单失败；输出：用户可理解的中英文根因。
    map_failures = {
        "RuntimeError: LIVE_MAP_STARTUP_GEOMETRY_INCOMPLETE": (
            "LIVE_MAP_STARTUP_GEOMETRY_INCOMPLETE",
            "The initial camera view cannot constrain all required position directions. "
            "Takeoff has not been authorized; improve the initial view or localization source.",
            "初始相机视角缺少必要方向的定位参照，尚未允许起飞。需要改善初始视角或定位来源。"),
        "RuntimeError: LIVE_MAP_STREAM_OBSERVATIONS_LOST": (
            "LIVE_MAP_STREAM_OBSERVATIONS_LOST",
            "Valid visual-map localization did not remain available. "
            "Check camera geometry, observation timing, and flight-controller delivery.",
            "有效的视觉地图定位未能持续提供。需要检查相机视角、观测时效及飞控接收链路。"),
    }
    if isinstance(failure, str) and failure in map_failures:
        code, en, zh = map_failures[failure]
        return code, en if english else zh
    prefix = "RuntimeError: NATIVE_PERCEPTION_NOT_READY_BEFORE_ARM:"
    if not isinstance(failure, str) or not failure.startswith(prefix):
        return None
    reason = failure[len(prefix):]
    if reason in {"NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED:ValueError",
                  "NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED:TypeError",
                  "NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED:OverflowError"}:
        return ("NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED",
                "The route clearance cannot cover the current localization uncertainty. "
                "Use a wider validated route or improve localization; restarting alone will not fix this."
                if english else
                "路线净空无法覆盖当前定位误差。需要重新校验更宽的路线或改善定位；仅重启或删除对话不能解决。")
    if reason in {"NATIVE_PERCEPTION_FEATURES_NOT_READY", "NATIVE_PERCEPTION_STREAM_NOT_HEALTHY",
                  "NATIVE_PERCEPTION_DEPTH_SOURCE_INVALID_OR_EXPIRED",
                  "NATIVE_PERCEPTION_HEALTH_CLOCK_INVALID_OR_EXPIRED",
                  "NATIVE_INDEPENDENT_FRAMES_NOT_READY", "NATIVE_PREFLIGHT_SOURCE_INTERRUPTED",
                  "NATIVE_PERCEPTION_NOT_RECEIVED"}:
        return ("NATIVE_PERCEPTION_NOT_READY_BEFORE_ARM",
                "Native sensor observations did not become continuously ready before takeoff. "
                "Check simulator rendering performance and sensor delivery."
                if english else
                "起飞前原生传感器数据未能持续就绪。需要检查仿真渲染性能和传感器数据链路。")
    return None
