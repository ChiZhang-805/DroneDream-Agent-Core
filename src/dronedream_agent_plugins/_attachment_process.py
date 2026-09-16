"""Bounded, owned preview subprocesses; not a general-purpose execution sandbox."""

from __future__ import annotations

import os
from collections.abc import Sequence

from dronedream_agent_core.plugin_contracts import PluginResourcePolicy
from dronedream_agent_core.process_capture import capture_process


# 功能：
#   1. 在共享进程边界内生成附件预览，仅成功退出时返回标准输出，不附加错误输出。
#   2. PDF 工作者在收到请求后才解析文档；Windows Job 在输送请求前完成分配。
#   3. POSIX 的 PDF 工作者自行设置资源限制，本函数只限制运行时长和通信字节数。
#   4. 环境继承用于本地解码依赖，因此命令必须来自可信程序；此处不是执行沙箱。
# 输入：
#   command：调用方选定的可信解码器及独立命令参数。
#   request：不超过 4096 字节的请求，通常为包含本地路径的 JSON，而非附件内容。
#   timeout：进程运行及通信的总秒数预算，清理另外进行有界等待。
#   output_limit：输入及标准输出、错误输出合计的字节预算。
# 输出：
#   output：成功解码器返回的原始标准输出字节。
def preview_process(
    command: Sequence[str],
    *,
    request: bytes = b"",
    timeout: float = 20,
    output_limit: int = 2 * 1024 * 1024,
) -> bytes:
    # A request is a tiny JSON path, never the attachment itself. Validate it
    # before launch so malformed requests cannot start a decoder unnecessarily.
    if type(request) is not bytes or len(request) > 4096:
        raise ValueError("ATTACHMENT_PROCESS_REQUEST_LIMIT")
    result = capture_process(
        command,
        stdin=request,
        maximum_bytes=output_limit,
        timeout=timeout,
        environment=dict(os.environ),
        resource_policy=PluginResourcePolicy(
            memory_limit_mb=384,
            cpu_time_limit_seconds=15,
            # Windows venv redirectors and frozen bootloaders need a child slot.
            process_limit=4,
        ),
    )
    if result.returncode != 0:
        raise RuntimeError("ATTACHMENT_PROCESS_FAILED")
    output = result.stdout
    return output
