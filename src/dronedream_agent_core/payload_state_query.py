"""Fresh, request-bound reads of simulated parcel joints, not event-cache guesses."""

import json
import re
import subprocess
import time
import uuid


# 功能：验证当前请求的原生关节快照，拒绝缺失、未知、错配和非法迭代号。
# 输入：raw：Gazebo StringMsg 文本；token：本次随机请求标识。
# 输出：真实 detached 状态和仿真迭代号；无可靠回读时抛错。
def parse_attachment_snapshot(raw: str, token: str) -> dict:
    match = re.fullmatch(r'\s*data:\s*("(?:[^"\\]|\\.)*")\s*', raw)
    if not match:
        raise RuntimeError("PAYLOAD_STATE_QUERY_INVALID_RESPONSE")
    parts = json.loads(match[1]).split("|")
    if (len(parts) != 3 or parts[0] != token or parts[1] not in ("attached", "detached")
            or not parts[2].isdigit() or not 0 < int(parts[2]) < 2**64):
        raise RuntimeError("PAYLOAD_STATE_QUERY_UNCONFIRMED")
    return {"detached": parts[1] == "detached", "iteration": int(parts[2]), "request_id": token}


# 功能：通过独立 CLI 进程查询下一仿真时步，避免 Gazebo 同步绑定占用控制线程 GIL。
# 输入：gz_binary、service、env：当前隔离运行；timeout：墙钟上限。
# 输出：绑定到当前请求的关节状态；不发送 attach/detach，不改变任何实体。
def query_attachment_state(gz_binary: str, *, service: str, env: dict, timeout: float = 2.5) -> dict:
    if re.fullmatch(r"/world/[^/\s]+/model/[^/\s]+/attachment_state", service) is None:
        raise ValueError("PAYLOAD_STATE_QUERY_SERVICE_INVALID")
    if not 0 < timeout <= 30:
        raise ValueError("PAYLOAD_STATE_QUERY_TIMEOUT_INVALID")
    token = uuid.uuid4().hex
    result = subprocess.run([
        gz_binary, "service", "-s", service, "--reqtype", "gz.msgs.StringMsg",
        "--reptype", "gz.msgs.StringMsg", "--timeout", str(max(1, int(timeout * 800))),
        "--req", f'data: "{token}"',
    ], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError("PAYLOAD_STATE_QUERY_TRANSPORT_FAILED: " + result.stderr[-300:])
    snapshot = parse_attachment_snapshot(result.stdout, token)
    snapshot.update(service=service, observed_at_unix_ms=int(time.time() * 1000),
                    source="gazebo-ecm-joint-query")
    return snapshot
