"""SemVer precedence and the deliberately small plugin dependency range language.

Precedence follows https://semver.org/; caret and prerelease admission follow
https://github.com/npm/node-semver#caret-ranges-123-025-004 . No PEP 440 coercion,
partial versions, disjunctions or implicit prerelease promotion are supported.
"""

from __future__ import annotations

import re

_NUMBER = r"(?:0|[1-9][0-9]*)"
_PRE = rf"(?:{_NUMBER}|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER_PATTERN = re.compile(
    rf"^({_NUMBER})\.({_NUMBER})\.({_NUMBER})"
    rf"(?:-({_PRE}(?:\.{_PRE})*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


# 功能：
#   1. 严格解析完整 SemVer，拒绝前导零、不完整版本及过长输入。
#   2. 按主次修订号和预发布标识生成排序键，构建元数据不参与版本优先级比较。
# 输入：
#   value：待解析的插件版本字符串。
# 输出：
#   precedence：包含版本号、正式发布标记和预发布分段的排序元组。
def version_precedence(value: str) -> tuple:
    if not isinstance(value, str) or len(value) > 160:
        raise ValueError("PLUGIN_VERSION_INVALID")
    match = SEMVER_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("PLUGIN_VERSION_INVALID")
    major, minor, patch = (int(match[index]) for index in (1, 2, 3))
    prerelease = match[4]
    parts = (
        tuple((0, int(part)) if part.isdigit() else (1, part) for part in prerelease.split("."))
        if prerelease
        else ()
    )
    precedence = major, minor, patch, 0 if prerelease else 1, parts
    return precedence


# 功能：
#   1. 判断插件是否满足通配、精确、下限或兼容范围要求，不做隐式版本格式转换。
#   2. 只有要求显式指定相同基础版本的预发布时才接纳预发布，防止提前装入未稳定依赖。
# 输入：
#   version：候选插件的完整 SemVer。
#   requirement：*、完整版本、>=版本或 ^版本形式的依赖约束。
# 输出：
#   matches：候选版本满足约束时为 True，否则为 False。
def version_matches(version: str, requirement: str) -> bool:
    actual = version_precedence(version)
    if not isinstance(requirement, str) or len(requirement) > 162:
        raise ValueError("PLUGIN_VERSION_REQUIREMENT_INVALID")
    if requirement == "*":
        matches = actual[3] == 1
        return matches
    operator = ">=" if requirement.startswith(">=") else "^" if requirement.startswith("^") else ""
    minimum = version_precedence(requirement[len(operator) :])
    # 即使数字更大，另一基础版本的预发布也不代表当前依赖已兼容。
    if actual[3] == 0 and (minimum[3] != 0 or actual[:3] != minimum[:3]):
        matches = False
        return matches
    if operator == ">=":
        matches = actual >= minimum
    elif operator == "^":
        major, minor, patch = minimum[:3]
        # 主版本为零时，以首个非零段约束兼容范围，不能一律放宽到下一个主版本。
        upper = (major + 1, 0, 0) if major else (0, minor + 1, 0) if minor else (0, 0, patch + 1)
        matches = actual >= minimum and actual[:3] < upper
    else:
        matches = actual == minimum
    return matches
