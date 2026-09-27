"""Non-blocking, bounded Laya advice process. It has no flight command authority."""

from __future__ import annotations

import math
import multiprocessing
import time
from contextlib import suppress
from pathlib import Path
from queue import Empty, Full

from .decision_shadow import ACTIONS, validate_probabilities
from .decision_state_adapter import DecisionStateV2, decision_digest


# 功能：在隔离子进程加载权重，串行处理显式请求；不接触飞控或调用网络。
# 输入：有界请求/结果队列、本地模型；输出：有身份的概率/错误响应，异常不伪装成功。
def stage_worker(requests, results, model_path: str):
    try:
        import torch

        from .laya_uav_training import encode_decision, load_laya, model_batch, verify_package

        torch.set_num_threads(2)
        receipt = verify_package(Path(model_path))
        if receipt.get("state_schema") != "dronedream.decision-state.v2":
            raise ValueError("STAGE_PACKAGE_SCHEMA_MISMATCH")
        if receipt.get("actions") != list(ACTIONS):
            raise ValueError("STAGE_PACKAGE_ACTIONS_MISMATCH")
        temperature = receipt["calibration"]["temperature"]
        if type(temperature) not in (int, float) or not .5 <= temperature <= 5:
            raise ValueError("STAGE_PACKAGE_TEMPERATURE_INVALID")
        agent = load_laya(Path(model_path), "cpu")
        agent.model.eval()
        results.put({"kind": "ready", "smoke_only": receipt.get("smoke_only") is not False})
        while True:
            message = requests.get()
            if message is None:
                break
            request_id, state_json = message
            tick = time.perf_counter()
            try:
                state = DecisionStateV2.model_validate_json(state_json)
                item = encode_decision(agent.tok, state)
                with torch.inference_mode():
                    logits, _ = agent.model(**model_batch([item], agent.tok, agent.device))
                    raw = torch.softmax(logits[0].float(), dim=0).cpu().tolist()
                    calibrated = torch.softmax(
                        logits[0].float() / temperature, dim=0).cpu().tolist()
                results.put({"kind": "advice", "request_id": request_id,
                                 "raw": dict(zip(ACTIONS, raw, strict=True)),
                                 "calibrated": dict(zip(ACTIONS, calibrated, strict=True)),
                                 "compute_ms": (time.perf_counter() - tick) * 1000})
            except Exception as error:
                results.put({"kind": "error", "request_id": request_id,
                                 "error_type": type(error).__name__, "reason": str(error)[:512]})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        with suppress(EOFError, BrokenPipeError, OSError):
            results.put({"kind": "fatal", "error_type": type(error).__name__,
                             "reason": str(error)[:512]})
    finally:
        requests.close()
        results.close()


class LocalStageDecisionPort:
    """One in-flight request plus one replaceable pending state; poll never waits."""

    # 功能：创建仅属于本实例的推理子进程；不阻塞等待模型加载。
    # 输入：模型目录、建议 TTL、独立启动/推理超时；输出：建议通道，无飞行权限。
    def __init__(self, model_path: Path, *, ttl_ms: int = 750, startup_timeout_s: float = 60,
                 inference_timeout_s: float = 10, worker=stage_worker):
        if (type(ttl_ms) is not int or not 1 <= ttl_ms <= 750
                or not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0
                or not math.isfinite(inference_timeout_s) or inference_timeout_s <= 0):
            raise ValueError("STAGE_PORT_BUDGET_INVALID")
        context = multiprocessing.get_context("spawn")
        # Queue.put_nowait 使用独立 feeder；大型状态不能把控制线程卡在 Pipe.send。
        self.requests, self.results = context.Queue(maxsize=1), context.Queue(maxsize=2)
        self.process = context.Process(target=worker,
            args=(self.requests, self.results, str(model_path)), daemon=True)
        self.process.start()
        self.ttl_ms, self.startup_timeout_s = ttl_ms, startup_timeout_s
        self.inference_timeout_s = inference_timeout_s
        self.started = time.monotonic()
        self.ready, self.closed = False, False
        self.pending, self.inflight, self.identity = None, None, None
        self.generation = 0
        self.risk_signature = None
        self.last_sequence = -1
        self.last_reason = "model-loading"
        self.last_error_detail = None
        self.smoke_only = True

    # 功能：投递最新因果状态，等待队列只保留一个；目标变化使旧建议失效。
    # 输入：已验证状态；输出：None，不发送任何无人机指令。
    def submit(self, state: DecisionStateV2):
        if self.closed:
            raise ValueError("STAGE_PORT_CLOSED")
        state = DecisionStateV2.model_validate_json(state.model_dump_json())
        if state.clock_domain not in {"unix-ms", "synthetic-only"}:
            raise ValueError("STAGE_PORT_CLOCK_DOMAIN_UNSUPPORTED")
        submitted_at = time.monotonic()
        identity = (state.mission_id, state.goal_id, state.route_sha256, state.map_sha256,
                    state.vehicle_sha256, state.calibration_sha256, state.clock_domain)
        if identity == self.identity and state.sequence <= self.last_sequence:
            raise ValueError("STAGE_STATE_SEQUENCE_NOT_INCREASING")
        frame = state.frame
        # 目标A→B→A仍是不同代；新横穿/堵塞/缺观测使旧建议作废，位置微抖则不必饿死推理。
        risk = tuple(getattr(frame, key) for key in (
            "crossing_obstacle", "persistent_blockage", "local_route_verified",
            "caution_required", "can_hold_position", "can_brake")) + tuple(
                source is not None and source.fresh(frame.observed_at_ms)
                for source in (frame.pose_source, frame.geometry_source, frame.route_source))
        if identity != self.identity or risk != self.risk_signature:
            self.generation += 1
        self.identity, self.last_sequence, self.risk_signature = identity, state.sequence, risk
        source_age_ms = 0
        if state.clock_domain == "unix-ms":
            source_age_ms = int(time.time() * 1000) - frame.observed_at_ms
            if not 0 <= source_age_ms < self.ttl_ms:
                self.generation += 1
                self.pending = None
                self.last_reason = "input-frame-expired-or-future"
                return
        raw = state.model_dump_json()
        self.pending = {"request_id": decision_digest(state.model_dump(mode="json")),
                        "state_json": raw, "identity": identity, "sequence": state.sequence,
                        "observed_at_ms": state.frame.observed_at_ms,
                        "submitted_at": submitted_at, "source_age_ms": source_age_ms,
                        "generation": self.generation}

    # 功能：零等待提取结果并发送最新待处理状态；过期/旧目标结果不能延长有效期。
    # 输入：无；输出：有效建议或 None；失败原因可读取 last_reason。
    def poll(self) -> dict | None:
        if self.closed:
            return None
        now = time.monotonic()
        advice = None
        try:
            while True:
                try:
                    result = self.results.get_nowait()
                except Empty:
                    break
                if type(result) is not dict:
                    raise ValueError("STAGE_WORKER_MESSAGE_INVALID")
                kind = result.get("kind")
                if kind == "ready":
                    if self.ready or type(result.get("smoke_only")) is not bool:
                        raise ValueError("STAGE_WORKER_READY_INVALID")
                    self.ready, self.smoke_only = True, result.get("smoke_only", True)
                    self.last_reason = "ready"
                elif kind in {"advice", "error"}:
                    request = self.inflight
                    self.inflight = None
                    if request is None or result.get("request_id") != request["request_id"]:
                        self.last_reason = "result-identity-invalid"
                        continue
                    age_ms = request["source_age_ms"] + (now - request["submitted_at"]) * 1000
                    if request["identity"] != self.identity:
                        self.last_reason = "goal-or-contract-changed"
                    elif request["generation"] != self.generation:
                        self.last_reason = "decision-context-superseded"
                    elif not 0 <= age_ms < self.ttl_ms:
                        self.last_reason = "advice-expired"
                    elif kind == "error":
                        self.last_reason = "model-error:" + result.get("error_type", "unknown")
                        self.last_error_detail = result.get("reason")
                    else:
                        raw = validate_probabilities(result["raw"])
                        calibrated = validate_probabilities(result["calibrated"])
                        advice = {"schema_version": "dronedream.decision-advice.v1",
                                  "input_sha256": request["request_id"],
                                  "sequence": request["sequence"],
                                  "mission_id": request["identity"][0],
                                  "goal_id": request["identity"][1],
                                  "route_sha256": request["identity"][2],
                                  "map_sha256": request["identity"][3],
                                  "vehicle_sha256": request["identity"][4],
                                  "calibration_sha256": request["identity"][5],
                                  "clock_domain": request["identity"][6],
                                  "input_observed_at_ms": request["observed_at_ms"],
                                  "raw_probabilities": raw, "probabilities": calibrated,
                                  "choice": max(ACTIONS, key=calibrated.get),
                                  "age_ms": age_ms, "valid_for_ms": self.ttl_ms - age_ms,
                                  "expires_at_host_monotonic_s":
                                      request["submitted_at"]
                                      + (self.ttl_ms - request["source_age_ms"])/1000,
                                  "execution_authority": False, "smoke_only": self.smoke_only}
                        self.last_reason = "advice-available"
                elif kind == "fatal":
                    self.last_reason = "model-load-failed:" + result.get("error_type", "unknown")
                    self.last_error_detail = result.get("reason")
                    self._abort()
                    return None
                else:
                    raise ValueError("STAGE_WORKER_MESSAGE_INVALID")
            if not self.process.is_alive():
                self.last_reason = "model-process-exited"
                self._abort()
                return None
            if not self.ready and now - self.started > self.startup_timeout_s:
                self.last_reason = "model-startup-timeout"
                self._abort()
            elif self.inflight and now - self.inflight["sent_at"] > self.inference_timeout_s:
                self.last_reason = "model-inference-timeout"
                self._abort()
            elif self.ready and self.inflight is None and self.pending:
                request, self.pending = self.pending, None
                age_ms = request["source_age_ms"] + (now - request["submitted_at"]) * 1000
                if request["generation"] == self.generation and 0 <= age_ms < self.ttl_ms:
                    request["sent_at"] = now
                    try:
                        self.requests.put_nowait((request["request_id"], request["state_json"]))
                    except Full:
                        self.pending = request
                    else:
                        self.inflight = request
                else:
                    self.last_reason = "pending-state-expired"
        except (EOFError, BrokenPipeError, OSError, ValueError, KeyError, TypeError):
            self.last_reason = "model-channel-invalid"
            self._abort()
            return None
        return advice

    # 功能：无等待撤销本通道，终止仅属于本实例的进程；不阻塞控制轮询。
    # 输入：无；输出：None，后续由 close 回收系统句柄。
    def _abort(self):
        if self.closed:
            return
        self.closed = True
        self.pending, self.inflight = None, None
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()
        if self.process.is_alive():
            self.process.terminate()

    # 功能：回收本实例推理进程，不影响 Runtime、飞控或其他进程。
    # 输入：无；输出：None；close 幂等，不能从飞控高频回调同步调用。
    def close(self):
        self._abort()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
