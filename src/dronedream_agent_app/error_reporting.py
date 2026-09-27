"""Bounded local error evidence without credentials, request bodies or exception values."""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse

from .harness_design_service import HarnessDesignServiceError
from dronedream_agent_core.model_harness.model_port import ModelInvocationError, _safe_attempt_diagnostic


# 功能：
#   为后端异常保存有界诊断记录，返回可关联的错误编号；不记录请求正文、令牌和异常参数。
# 输入：
#   root：当前应用数据目录。
# 输出：
#   handler：供 HTTP 异常处理调用的异步函数。
def error_handler(root: Path):
    logger = logging.Logger("dronedream.local-errors", level=logging.ERROR)
    directory = root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    sink = RotatingFileHandler(directory / "backend-errors.jsonl", maxBytes=2_000_000,
                               backupCount=3, encoding="utf-8", delay=True)
    logger.addHandler(sink)
    logger.propagate = False

    # 功能：
    #   捕获异常链中的类型和代码位置，将已知配置错误与未知内部错误分别反馈给界面。
    # 输入：
    #   request：失败请求；error：捕获的异常。
    # 输出：
    #   response：不包含机密数据的结构化错误响应。
    async def handler(request: Request, error: Exception):
        error_id = uuid4().hex
        chain = []
        seen = set()
        current = error
        while current is not None and id(current) not in seen and len(chain) < 8:
            seen.add(id(current))
            frames = []
            trace = current.__traceback__
            while trace is not None and len(frames) < 64:
                frames.append({"file": Path(trace.tb_frame.f_code.co_filename).name,
                               "function": trace.tb_frame.f_code.co_name, "line": trace.tb_lineno})
                trace = trace.tb_next
            item = {"type": type(current).__name__, "frames": frames}
            if isinstance(error, ModelInvocationError) and isinstance(current, Exception):
                item["diagnostic"] = _safe_attempt_diagnostic(current)
            chain.append(item)
            current = current.__cause__ or current.__context__
        code = ("MODEL_INVOCATION_FAILED" if isinstance(error, ModelInvocationError)
                else "HARNESS_CONFIGURATION_UPGRADE_REQUIRED" if isinstance(error, HarnessDesignServiceError)
                else "AGENT_CORE_INTERNAL_ERROR")
        route = request.scope.get("route")
        logger.error(json.dumps({"error_id": error_id, "code": code,
                                "route": getattr(route, "path", "unknown"), "chain": chain}))
        # 每次释放文件句柄，支持多实例、退出后备份和 Windows 文件轮换。
        sink.close()
        response = JSONResponse(status_code=409 if isinstance(error, (HarnessDesignServiceError, ModelInvocationError)) else 500,
                                content={"detail": {"code": code, "error_id": error_id}})
        return response

    return handler
