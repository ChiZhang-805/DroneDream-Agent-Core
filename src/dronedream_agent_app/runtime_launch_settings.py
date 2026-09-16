"""Validate launch-time numeric settings and packaged executable paths."""

import math
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path, portable_plugin_path


# 功能：
#   校验整数配置及闭区间范围，不把字符串、布尔值或小数隐式转换成整数。
# 输入：
#   value：待校验的配置值。
#   lower：允许的最小整数。
#   upper：允许的最大整数。
# 输出：
#   integer：通过类型和范围校验的原值。
def bounded_integer(value: object, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("runtime integer setting is outside its contract")
    integer = value
    return integer


# 功能：
#   校验正数时间配置，拒绝布尔值、非有限值及不能表示为有限秒数的数值。
# 输入：
#   value：待校验的秒数。
# 输出：
#   seconds：有限且大于零的浮点秒数。
def positive_seconds(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("runtime time setting must be numeric")
    try:
        seconds = float(value)
    except OverflowError as error:
        raise ValueError("runtime time setting is too large") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("runtime time setting must be finite and positive")
    return seconds


# 功能：
#   将插件声明的可移植相对路径限定在产品 Runtime 资源内，拒绝链接或外部路径。
# 输入：
#   root：已校验的 Runtime 资源目录。
#   relative：插件声明的执行器相对路径。
# 输出：
#   executable：存在于指定资源目录内的普通执行器文件路径。
def packaged_executor(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("runtime executor path must be a string")
    member = portable_plugin_path(relative)
    executable = root / member
    check_plain_plugin_path(executable)
    if not executable.resolve().is_relative_to(root.resolve()) or not executable.is_file():
        raise ValueError("runtime executor is missing or outside packaged resources")
    return executable
