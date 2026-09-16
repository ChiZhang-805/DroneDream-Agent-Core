"""Public repository identity without leaking local paths or remote credentials."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


# 功能：
#   从 GitHub HTTPS/SSH 地址提取仓库身份；其他地址不猜测，也不返回凭证或本机路径。
# 输入：
#   remote：Git origin 地址。
# 输出：
#   identity：owner/repository；无法验证时为 None。
def github_repository_identity(remote: str) -> str | None:
    if remote.startswith("git@github.com:"):
        candidate = remote.removeprefix("git@github.com:")
    else:
        try:
            parsed = urlsplit(remote)
            if (parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com"
                    or parsed.password is not None or parsed.query or parsed.fragment
                    or parsed.port is not None
                    or (parsed.username is not None
                        and not (parsed.scheme == "ssh" and parsed.username == "git"))):
                return None
            candidate = parsed.path.removeprefix("/")
        except ValueError:
            return None
    candidate = candidate.removesuffix(".git")
    identity = candidate if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", candidate,
    ) else None
    return identity


# 功能：
#   有界读取实际 origin，不再把不同公开仓库或分支来源写成旧私有仓库。
# 输入：
#   repository：正在构建的 Git 仓库目录。
# 输出：
#   identity：可验证的 GitHub 仓库身份；缺失或读取失败时为 None。
def source_repository_identity(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    identity = github_repository_identity(result.stdout.strip())
    return identity
