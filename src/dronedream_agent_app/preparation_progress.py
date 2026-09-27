"""Bounded account/request-isolated progress mailbox for desktop polling."""
import threading
import time
from collections import OrderedDict

from dronedream_agent_core.model_harness.progress import progress_sink


class PreparationProgress:
    # 功能：
    #   初始化有界临时事件缓存，不将进度日志当作飞行授权或会话回复。
    # 输入：
    #   无。
    # 输出：
    #   self：进度邮箱。
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = OrderedDict()

    # 功能：
    #   在请求范围内捕获真实阶段事件，结束或异常时清理上下文；跨模型线程仍绑定同一邮箱。
    # 输入：
    #   key：已验证账户、会话和请求组合；operation：实际准备工作。
    # 输出：
    #   result：原工作结果，不改变业务异常。
    def run(self, key: tuple, operation):
        started = time.monotonic()
        job = {"events": [], "sequence": 0, "state": "running", "updated": started}
        with self._lock:
            if key in self._jobs and self._jobs[key]["state"] == "running":
                raise ValueError("PREPARATION_REQUEST_ALREADY_RUNNING")
            self._jobs[key] = job
            while len(self._jobs) > 128:
                self._jobs.popitem(last=False)

        # 功能：
        #   追加当前请求的公开摘要，限制长度和数量，禁止写入已结束请求。
        # 输入：
        #   stage：阶段；zh、en：公开摘要。
        # 输出：
        #   无返回值。
        def emit(stage, zh, en):
            with self._lock:
                if self._jobs.get(key) is not job or job["state"] != "running":
                    return
                job["sequence"] += 1
                job["updated"] = time.monotonic()
                job["events"].append({"sequence": job["sequence"], "stage": stage[:64], "zh": zh[:700], "en": en[:1000], "elapsed_ms": round((job["updated"] - started) * 1000)})
                job["events"] = job["events"][-128:]

        token = progress_sink.set(emit)
        try:
            emit("assets", "开始核对所选地图、无人机、账户与任务约束；当前阶段不会启动飞行。", "Checking selected map, aircraft, account and mission constraints. This stage does not start a flight.")
            result = operation()
            emit("ready", "规划处理已返回，正在整理正式回复。", "Planning returned; preparing the final reply.")
            with self._lock:
                job["state"] = "completed"
            return result
        except Exception:
            emit("failed", "当前阶段未能完成；具体原因将显示在正式错误或追问中。", "This stage could not complete; the final error or clarification will explain the issue.")
            with self._lock:
                job["state"] = "failed"
            raise
        finally:
            progress_sink.reset(token)

    # 功能：
    #   只读取已验证账户对应请求的新事件，不暴露其他账户或旧请求内容。
    # 输入：
    #   key：账户/会话/请求键；after：已收到的最大序号。
    # 输出：
    #   snapshot：新增事件及状态。
    def read(self, key: tuple, after: int) -> dict:
        with self._lock:
            job = self._jobs.get(key)
            if job is None or time.monotonic() - job["updated"] > 3 * 3600:
                return {"state": "pending", "events": []}
            return {"state": job["state"], "events": [dict(event) for event in job["events"] if event["sequence"] > after]}
