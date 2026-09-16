from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dronedream_agent_core.source_identity import (
    github_repository_identity,
    source_repository_identity,
)


# 功能：
#   验证公开仓库、原私有仓库和合法 fork 均返回实际来源，而非固定产品名称。
# 输入：
#   remote：允许的 GitHub Git 地址。
#   expected：对应的仓库身份。
# 输出：
#   无；身份提取错误时断言失败。
@pytest.mark.parametrize(("remote", "expected"), [
    ("https://github.com/ChiZhang-805/DroneDream-Agent-Core.git",
     "ChiZhang-805/DroneDream-Agent-Core"),
    ("git@github.com:ChiZhang-805/DroneDream-Flight-Agent-Core.git",
     "ChiZhang-805/DroneDream-Flight-Agent-Core"),
    ("ssh://git@github.com/contributor/core.git", "contributor/core"),
    ("https://github.com/contributor/core", "contributor/core"),
])
def test_supported_origin(remote: str, expected: str) -> None:
    assert github_repository_identity(remote) == expected


# 功能：
#   验证本机路径、嵌入凭证及伪装主机不会进入公开构建清单。
# 输入：
#   remote：不可信或含敏感内容的地址样例。
# 输出：
#   无；未能拒绝地址时断言失败。
@pytest.mark.parametrize("remote", [
    "Q:/private/core", "https://github.com.evil.test/owner/repo",
    "https://example-user:example-password@github.com/owner/repo",
    "https://example-token@github.com/owner/repo", "https://github.com/o/r?token=example",
    "https://github.com/o/r#private", "http://github.com/o/r",
    "https://github.com:invalid/o/r", "https://github.com/o/../r",
    "git@github.com:o/r/extra", "https://[invalid/o/r", "",
])
def test_unverified_origin_is_not_disclosed(remote: str) -> None:
    assert github_repository_identity(remote) is None


# 功能：
#   用真实临时 Git 配置验证 origin 的读取和未配置分支，不修改开发仓库。
# 输入：
#   tmp_path：隔离的临时目录。
# 输出：
#   无；来源读取错误时断言失败。
def test_reads_actual_git_origin(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    assert source_repository_identity(tmp_path) is None
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin",
                    "https://github.com/test-owner/public-core.git"],
                   check=True, capture_output=True)
    assert source_repository_identity(tmp_path) == "test-owner/public-core"
