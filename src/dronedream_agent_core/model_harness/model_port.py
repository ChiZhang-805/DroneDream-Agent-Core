"""Real structured model APIs used by bounded flight-agent roles."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar
from uuid import uuid4

from openai import APIStatusError, OpenAI
from pydantic import BaseModel, ValidationError

from ..contracts import ModelCallRecord, ModelRole
from ..hashing import canonical_json, sha256_json

ProviderName: TypeAlias = str
OutputT = TypeVar("OutputT", bound=BaseModel)


class ModelConfigurationError(RuntimeError):
    """Raised when a selected real provider is not configured."""


class ModelInvocationError(RuntimeError):
    """Raised after all bounded real-provider attempts fail validation."""

    # 功能：
    #   保存实际尝试次数和有限的机器诊断，不保存供应商原始响应或密钥。
    # 输入：
    #   message：面向调用方的安全错误说明。
    #   attempts_used：已经消耗的物理调用次数，必须为正整数。
    #   reason_code：有长度上限的大写机器原因码。
    #   diagnostic_metrics：可选的非负有限计时指标。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        message: str,
        *,
        attempts_used: int,
        reason_code: str = "MODEL_INVOCATION_FAILED",
        diagnostic_metrics: dict[str, float] | None = None,
    ) -> None:
        super().__init__(message)
        if type(attempts_used) is not int or attempts_used < 1:
            raise ValueError("attempts_used must be a positive integer")
        if (
            not isinstance(reason_code, str)
            or not reason_code
            or len(reason_code) > 160
            or not reason_code.isascii()
            or not reason_code[0].isalpha()
            or not reason_code.replace("_", "").isalnum()
            or reason_code != reason_code.upper()
        ):
            raise ValueError("reason_code must be a bounded uppercase machine code")
        self.attempts_used = attempts_used
        self.reason_code = reason_code
        metrics = dict(diagnostic_metrics or {})
        if len(metrics) > 16 or any(
            not isinstance(key, str)
            or not key
            or len(key) > 80
            or not key.isascii()
            or not key[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in key)
            or type(value) not in {int, float}
            or value < 0.0
            or value > 60_000.0
            or not math.isfinite(value)
            for key, value in metrics.items()
        ):
            raise ValueError("diagnostic metrics must be bounded non-negative timings")
        self.diagnostic_metrics = metrics


@dataclass(frozen=True)
class ProviderSettings:
    name: ProviderName
    model: str
    api_key_env: str
    base_url: str | None
    api_style: Literal["responses", "chat-completions"]
    supports_image_input: bool = False

    # 功能：
    #   读取明确配置的模型与协议；兼容某种协议不等于该模型具备视觉能力。
    # 输入：
    #   name：供应商标识，用于选择配置或自定义环境变量前缀。
    # 输出：
    #   settings：供应商、模型、凭证变量名及已声明能力。
    @classmethod
    def from_env(cls, name: ProviderName) -> ProviderSettings:
        if name == "openai":
            api_style = os.getenv("OPENAI_API_STYLE") or "responses"
            if api_style not in {"responses", "chat-completions"}:
                raise ModelConfigurationError("OPENAI_API_STYLE is invalid")
            model = os.getenv("OPENAI_MODEL") or "gpt-5.4"
            settings = cls(
                name=name,
                model=model,
                api_key_env="OPENAI_API_KEY",
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                api_style=api_style,
                supports_image_input=model in {"gpt-4.1", "gpt-5.1", "gpt-5.4"},
            )
            return settings
        if name == "deepseek":
            settings = cls(
                name=name,
                model=os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash",
                api_key_env="DEEPSEEK_API_KEY",
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                api_style="chat-completions",
                supports_image_input=False,
            )
            return settings
        if name == "kimi":
            settings = cls(
                name=name,
                model=os.getenv("KIMI_MODEL") or "kimi-k2.6",
                api_key_env="KIMI_API_KEY",
                base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
                api_style="chat-completions",
                supports_image_input=False,
            )
            return settings
        prefix = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_") or "CUSTOM"
        api_style = os.getenv(f"{prefix}_API_STYLE", "chat-completions")
        if api_style not in {"responses", "chat-completions"}:
            raise ModelConfigurationError("CUSTOM_API_STYLE is invalid")
        base_url = os.getenv(f"{prefix}_BASE_URL")
        if not base_url:
            raise ModelConfigurationError("CUSTOM_BASE_URL is required")
        settings = cls(
            name=name,
            model=os.getenv(f"{prefix}_MODEL", ""),
            api_key_env=f"{prefix}_API_KEY",
            base_url=base_url,
            api_style=api_style,
            supports_image_input=False,
        )
        return settings


@dataclass(frozen=True)
class StructuredCallResult(Generic[OutputT]):
    artifact: OutputT
    record: ModelCallRecord
    supporting_records: tuple[ModelCallRecord, ...] = ()
    attempt_failures: tuple[str, ...] = ()


# 功能：
#   提取失败类型及有限诊断，排除供应商正文和校验器可能回显的输入值。
# 输入：
#   error：一次模型调用或输出校验的异常。
# 输出：
#   diagnostic：用于受限重试的诊断文本。
def _safe_attempt_diagnostic(error: Exception) -> str:
    if isinstance(error, ValidationError):
        # Custom validator messages can interpolate the rejected secret/prompt.
        # Error kinds and locations support repair without retaining those messages.
        details = [
            {
                "location": [str(item) for item in entry.get("loc", ())],
                "type": str(entry.get("type", "validation_error")),
            }
            for entry in error.errors()
        ][:12]
        encoded = json.dumps(details, ensure_ascii=True, separators=(",", ":"))
        diagnostic = f"ValidationError:{encoded[:1600]}"
    elif isinstance(error, APIStatusError):
        diagnostic = f"{type(error).__name__}:status={error.status_code}"
    elif isinstance(error, json.JSONDecodeError):
        diagnostic = (
            f"JSONDecodeError:line={error.lineno}:column={error.colno}:reason={error.msg[:120]}"
        )
    elif isinstance(error, TimeoutError):
        diagnostic = f"{type(error).__name__}:provider-call-timeout"
    elif isinstance(error, ValueError):
        diagnostic = "ValueError:model-output-or-contract-invalid"
    else:
        diagnostic = type(error).__name__
    return diagnostic


# 功能：
#   在创建传输前拒绝布尔、非有限值及无法表示的超时。
# 输入：
#   value：以秒计的候选超时。
# 输出：
#   converted：有限且严格大于零的秒数。
def _positive_timeout(value: float) -> float:
    if type(value) not in {int, float}:
        raise ValueError("timeout_seconds must be finite and positive")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("timeout_seconds must be finite and positive") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    return converted


class StructuredModelPort:
    """Provider-neutral boundary that returns only schema-validated artifacts."""

    # 功能：
    #   建立供应商边界和恢复状态；关闭 SDK 隐式重试，使 Harness 掌握调用预算。
    # 输入：
    #   provider：被选定的供应商标识。
    #   max_attempts：每次逻辑调用的物理尝试硬上限。
    #   timeout_seconds：每次物理请求的超时硬上限，不是整次任务的截止时间。
    #   settings：可选的供应商配置。
    #   api_key：可选密钥，未提供时读取配置声明的环境变量。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        provider: ProviderName,
        *,
        max_attempts: int = 3,
        timeout_seconds: float = 90.0,
        settings: ProviderSettings | None = None,
        api_key: str | None = None,
    ) -> None:
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        timeout_seconds = _positive_timeout(timeout_seconds)
        self.settings = settings or ProviderSettings.from_env(provider)
        if self.settings.name != provider:
            raise ValueError("provider settings do not match the selected provider")
        api_key = api_key or os.getenv(self.settings.api_key_env)
        if not api_key:
            raise ModelConfigurationError(
                f"{self.settings.api_key_env} is required for provider {provider}"
            )
        client_args: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if self.settings.base_url:
            client_args["base_url"] = self.settings.base_url
        self._client_args = client_args
        self._client = OpenAI(**self._client_args)
        self._maximum_attempt_ceiling = max_attempts
        self._timeout_ceiling_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._previous_response_by_context: dict[str, str] = {}
        self._transport_lock = threading.Lock()
        self._transport_ready = threading.Event()
        self._transport_ready.set()
        self._transport_generation = 0
        self._transport_changed = threading.Condition(self._transport_lock)
        self._transport_worker: threading.Thread | None = None
        self._transport_reset_requested = False
        self._transport_closed = False
        self._transport_creation_failed = False

    # 功能：
    #   撤销当前传输代次并异步替换卡住的连接；合并重复恢复，禁止关闭后复活。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def reset_transport(self) -> None:
        with self._transport_changed:
            if self._transport_closed or self._transport_reset_requested:
                return
            self._transport_generation += 1
            self._transport_ready.clear()
            self._transport_creation_failed = False
            self._transport_reset_requested = True
            self._start_transport_worker_locked()
            self._transport_changed.notify_all()

    # 功能：
    #   持有条件锁时启动唯一清理线程；启动失败撤销就绪状态，允许后续显式重试。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _start_transport_worker_locked(self) -> None:
        # One worker owns construction and retirement, with at most one pending
        # reset. A stuck SDK close cannot spawn unbounded threads / clients.
        if self._transport_worker is not None:
            return
        worker = threading.Thread(
            target=self._replace_transport,
            name=f"dronedream-model-transport-reset-{self.settings.name}",
            daemon=True,
        )
        self._transport_worker = worker
        try:
            worker.start()
        except Exception:
            self._transport_worker = None
            self._transport_reset_requested = False
            self._transport_creation_failed = True
            self._transport_changed.notify_all()
            raise

    # 功能：
    #   仅发布仍有效代次的新客户端，随后清理旧连接及失效的候选客户端。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def _replace_transport(self) -> None:
        while True:
            with self._transport_changed:
                if not self._transport_reset_requested:
                    self._transport_worker = None
                    self._transport_changed.notify_all()
                    return
                self._transport_reset_requested = False
                previous, self._client = self._client, None
                generation = self._transport_generation
                closing = self._transport_closed
            replacement: OpenAI | None = None
            if not closing:
                with suppress(Exception):
                    replacement = OpenAI(**self._client_args)
            with self._transport_changed:
                if not self._transport_closed and generation == self._transport_generation:
                    if replacement is not None:
                        self._client = replacement
                        self._transport_ready.set()
                    else:
                        # Construction failed: the old client is retired, NOT ready.
                        self._transport_creation_failed = True
                    replacement = None
                self._transport_changed.notify_all()
            if replacement is not None:
                with suppress(Exception):
                    replacement.close()
            if previous is not None:
                with suppress(Exception):
                    previous.close()

    # 功能：
    #   立即撤销调用资格并由唯一线程清理连接；不把返回视为第三方清理已经完成。
    #   Python 无法强制终止卡住的第三方关闭函数，线程结束才表示正常清理完成。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        with self._transport_changed:
            if self._transport_closed and (
                self._transport_worker is not None or self._client is None
            ):
                return
            if not self._transport_closed:
                self._transport_closed = True
                self._transport_generation += 1
            self._transport_ready.clear()
            self._transport_reset_requested = True
            self._start_transport_worker_locked()
            self._transport_changed.notify_all()

    # 功能：
    #   有界等待连接就绪并取得代次；关闭或构造失败时拒绝调用。
    # 输入：
    #   无。
    # 输出：
    #   generation：后续请求及回执必须绑定的传输代次。
    def _transport_snapshot(self) -> int:
        with self._transport_changed:
            self._transport_changed.wait_for(
                lambda: (
                    self._transport_ready.is_set()
                    or self._transport_closed
                    or self._transport_creation_failed
                ),
                timeout=self._timeout_seconds,
            )
            if (
                self._transport_closed
                or self._transport_creation_failed
                or not self._transport_ready.is_set()
                or self._client is None
            ):
                raise TimeoutError("MODEL_TRANSPORT_UNAVAILABLE")
            generation = self._transport_generation
            return generation

    # 功能：
    #   在 SDK 请求前拒绝失效连接，并给本次请求配置当前超时。
    # 输入：
    #   generation：可选的请求所属代次。
    # 输出：
    #   request_client：共享有效传输的请求配置视图。
    def _request_client(self, generation: int | None = None):
        with self._transport_lock:
            if (
                self._transport_closed
                or not self._transport_ready.is_set()
                or self._client is None
                or (generation is not None and generation != self._transport_generation)
            ):
                raise TimeoutError("MODEL_TRANSPORT_RETIRED")
            client = self._client
        request_client = client.with_options(timeout=self._timeout_seconds)
        return request_client

    # 功能：
    #   在构造上限内应用任务预算；完整校验后修改，保留任务要求的亚秒级截止时间。
    # 输入：
    #   maximum_attempts：任务允许的尝试次数。
    #   timeout_seconds：任务要求的单次物理调用超时。
    # 输出：
    #   None：不返回业务数据。
    def configure_execution_policy(self, *, maximum_attempts: int, timeout_seconds: float) -> None:
        if type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("maximum_attempts must be a positive integer")
        timeout_seconds = _positive_timeout(timeout_seconds)
        self._max_attempts = max(1, min(self._maximum_attempt_ceiling, maximum_attempts))
        self._timeout_seconds = min(self._timeout_ceiling_seconds, timeout_seconds)

    # 功能：
    #   判断当前协议能否恢复供应商侧响应链，不将该链当作账户长期记忆。
    # 输入：
    #   无。
    # 输出：
    #   supported：是否使用 Responses 协议。
    @property
    def supports_provider_context(self) -> bool:
        supported = self.settings.api_style == "responses"
        return supported

    # 功能：
    #   返回明确声明的图像输入能力，未知协议兼容端点不自动获得视觉能力。
    # 输入：
    #   无。
    # 输出：
    #   supported：供应商配置是否声明支持图像。
    @property
    def supports_image_input(self) -> bool:
        supported = self.settings.supports_image_input
        return supported

    # 功能：
    #   为外层调度公开当前请求超时，不承诺网络调用必定在该时间内被线程终止。
    # 输入：
    #   无。
    # 输出：
    #   timeout_seconds：当前物理请求的超时秒数。
    @property
    def invocation_timeout_seconds(self) -> float:
        timeout_seconds = self._timeout_seconds
        return timeout_seconds

    # 功能：
    #   为调用方指定的上下文恢复响应链；此映射不是身份认证或任务执行授权。
    # 输入：
    #   context_id：调用上下文标识。
    #   response_id：供应商已保存的响应标识。
    # 输出：
    #   None：不返回业务数据。
    def restore_provider_context(self, context_id: str, response_id: str) -> None:
        if not self.supports_provider_context:
            raise ValueError(f"provider {self.settings.name} has no Responses context chain")
        if not context_id or not response_id:
            raise ValueError("context_id and response_id must be non-empty")
        with self._transport_lock:
            if self._transport_closed:
                raise TimeoutError("MODEL_TRANSPORT_UNAVAILABLE")
            self._previous_response_by_context[context_id] = response_id

    # 功能：
    #   使用固定输入快照执行受限调用与修复重试，返回结构化产物及真实尝试记录。
    #   拒绝被撤销的回包；结构化校验成功不等于计划安全或任务已完成。
    # 输入：
    #   role：本次调用承担的角色。
    #   output_type：输出必须符合的 Pydantic 类型。
    #   instructions：该角色的指令。
    #   input_artifact：任务输入，发送前序列化成固定文本。
    #   context_id：可选的供应商上下文链标识。
    #   multimodal：可选的已准入图像描述。
    #   maximum_physical_attempts：外层剩余调用预算。
    # 输出：
    #   result：结构化产物、成功调用记录及此前尝试的有限错误信息。
    def call(
        self,
        *,
        role: ModelRole,
        output_type: type[OutputT],
        instructions: str,
        input_artifact: BaseModel | dict[str, object],
        context_id: str | None = None,
        multimodal: list[dict[str, object]] | None = None,
        maximum_physical_attempts: int | None = None,
    ) -> StructuredCallResult[OutputT]:
        if multimodal and not self.supports_image_input:
            raise ModelConfigurationError(
                f"{self.settings.name}/{self.settings.model} does not declare image input support"
            )
        if maximum_physical_attempts is not None and (
            type(maximum_physical_attempts) is not int or maximum_physical_attempts < 1
        ):
            raise ValueError("maximum_physical_attempts must be positive")
        input_json = canonical_json(input_artifact)
        input_digest = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
        media = [dict(item) for item in (multimodal or [])]
        generation = self._transport_snapshot()
        attempt_limit = min(
            self._max_attempts,
            maximum_physical_attempts
            if maximum_physical_attempts is not None
            else self._max_attempts,
        )
        with self._transport_lock:
            previous_response_id = (
                self._previous_response_by_context.get(context_id) if context_id else None
            )
        errors: list[str] = []
        for attempt in range(1, attempt_limit + 1):
            started_at = datetime.now(UTC)
            started = time.monotonic()
            try:
                if self.settings.api_style == "responses":
                    artifact, response_id, input_tokens, output_tokens = self._responses_call(
                        output_type=output_type,
                        instructions=instructions,
                        input_json=input_json,
                        previous_response_id=previous_response_id,
                        repair_errors=errors,
                        multimodal=media,
                        transport_generation=generation,
                    )
                else:
                    artifact, response_id, input_tokens, output_tokens = self._chat_call(
                        output_type=output_type,
                        instructions=instructions,
                        input_json=input_json,
                        repair_errors=errors,
                        multimodal=media,
                        transport_generation=generation,
                    )
            except Exception as exc:  # provider and validation failures share retry policy
                with self._transport_lock:
                    retired = self._transport_closed or generation != self._transport_generation
                if retired:
                    raise ModelInvocationError(
                        "MODEL_TRANSPORT_RETIRED",
                        attempts_used=attempt,
                        reason_code="MODEL_TRANSPORT_RETIRED",
                    ) from exc
                if isinstance(exc, APIStatusError) and exc.status_code in {400, 401, 403, 404, 422}:
                    raise ModelInvocationError(
                        f"{self.settings.name}/{role} rejected request with "
                        f"{type(exc).__name__} status={exc.status_code}",
                        attempts_used=attempt,
                    ) from exc
                errors.append(_safe_attempt_diagnostic(exc))
                if attempt == attempt_limit:
                    joined = " | ".join(errors)
                    raise ModelInvocationError(
                        f"{self.settings.name}/{role} failed after {attempt} attempts: {joined}",
                        attempts_used=attempt,
                    ) from exc
                continue

            record = ModelCallRecord(
                call_id=f"model-{uuid4().hex[:24]}",
                role=role,
                attempt=attempt,
                input_sha256=input_digest,
                output_sha256=sha256_json(artifact),
                output_schema=output_type.__name__,
                provider=self.settings.name,
                model=self.settings.model,
                response_id=response_id,
                previous_response_id=previous_response_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round((time.monotonic() - started) * 1000),
                created_at=started_at,
            )
            # An old provider response cannot revive an invalidated connection,
            # update the current conversation chain or authorize a new task.
            with self._transport_lock:
                if self._transport_closed or generation != self._transport_generation:
                    raise ModelInvocationError(
                        "MODEL_TRANSPORT_RETIRED",
                        attempts_used=attempt,
                        reason_code="MODEL_TRANSPORT_RETIRED",
                    )
                if context_id and response_id and self.settings.api_style == "responses":
                    self._previous_response_by_context[context_id] = response_id
                result = StructuredCallResult(
                    artifact=artifact, record=record, attempt_failures=tuple(errors)
                )
                return result
        raise AssertionError("unreachable bounded model loop")

    # 功能：
    #   按输出结构选择严格解析或 JSON 模式；自由参数字典保留原业务契约并本地校验。
    # 输入：
    #   output_type：预期产物类型。
    #   instructions：角色指令。
    #   input_json：已冻结的输入文本。
    #   previous_response_id：可选的上次响应标识。
    #   repair_errors：此前尝试的安全诊断。
    #   multimodal：已准入图像描述。
    #   transport_generation：本次调用所属的有效代次。
    # 输出：
    #   response_result：产物、响应标识、输入 token 数和输出 token 数。
    def _responses_call(
        self,
        *,
        output_type: type[OutputT],
        instructions: str,
        input_json: str,
        previous_response_id: str | None,
        repair_errors: list[str],
        multimodal: list[dict[str, object]],
        transport_generation: int | None = None,
    ) -> tuple[OutputT, str | None, int | None, int | None]:
        schema = output_type.model_json_schema()
        strict_schema = self._supports_strict_structured_output(schema)
        request: dict[str, object] = {
            "model": self.settings.model,
            "instructions": instructions,
            "input": self._responses_input(
                self._user_payload(
                    input_json,
                    output_type,
                    repair_errors,
                    schema=None if strict_schema else schema,
                ),
                multimodal,
            ),
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        client = self._request_client(transport_generation)
        if strict_schema:
            response = client.responses.parse(text_format=output_type, **request)
            artifact = response.output_parsed
            if artifact is None:
                raise ValueError("Responses API returned no parsed structured artifact")
        else:
            # OpenAI strict Structured Outputs deliberately rejects free-form object maps
            # (for example TaskNode.arguments). Keep that domain contract intact, request
            # valid JSON, and apply the same Pydantic validation locally.
            response = client.responses.create(
                text={"format": {"type": "json_object"}},
                **request,
            )
            if not response.output_text:
                raise ValueError("Responses API returned no JSON artifact")
            artifact = output_type.model_validate_json(response.output_text)
        usage = getattr(response, "usage", None)
        response_result = (
            artifact,
            getattr(response, "id", None),
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )
        return response_result

    # 功能：
    #   为 Chat Completions 协议生成结构化请求，再按业务模型校验响应正文。
    # 输入：
    #   output_type：预期产物类型。
    #   instructions：角色指令。
    #   input_json：已冻结的输入文本。
    #   repair_errors：此前尝试的安全诊断。
    #   multimodal：已准入图像描述。
    #   transport_generation：本次调用所属的有效代次。
    # 输出：
    #   response_result：产物、响应标识、输入 token 数和输出 token 数。
    def _chat_call(
        self,
        *,
        output_type: type[OutputT],
        instructions: str,
        input_json: str,
        repair_errors: list[str],
        multimodal: list[dict[str, object]],
        transport_generation: int | None = None,
    ) -> tuple[OutputT, str | None, int | None, int | None]:
        schema_json = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
        system = (
            f"{instructions}\nReturn one JSON object only. It must validate against this JSON "
            f"Schema: {schema_json}"
        )
        request: dict[str, object] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": self._chat_input(
                        self._user_payload(input_json, output_type, repair_errors), multimodal
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        if self.settings.name == "kimi" and self.settings.model == "kimi-k2.6":
            # K2.6 enables thinking by default.  A bounded control-plane call needs the
            # schema artifact, not a long hidden chain that can consume the complete
            # output allowance before ``message.content`` is emitted.
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self._request_client(transport_generation).chat.completions.create(
            **request,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("provider returned empty JSON content")
        artifact = output_type.model_validate_json(content)
        usage = response.usage
        response_result = (
            artifact,
            getattr(response, "id", None),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return response_result

    # 功能：
    #   将任务数据、可选输出契约和最近两次修复诊断组成用户输入，不改写角色指令。
    # 输入：
    #   input_json：已冻结的任务数据。
    #   output_type：请求生成的产物类型。
    #   repair_errors：有限且已去敏的尝试诊断。
    #   schema：JSON 模式需要显式携带的可选契约。
    # 输出：
    #   payload：发送到模型的用户输入文本。
    @staticmethod
    def _user_payload(
        input_json: str,
        output_type: type[BaseModel],
        repair_errors: list[str],
        *,
        schema: dict[str, object] | None = None,
    ) -> str:
        repair = ""
        if repair_errors:
            repair = (
                "\nPrevious attempts failed validation. Repair all of these errors: "
                + " | ".join(repair_errors[-2:])
            )
        schema_requirement = ""
        if schema is not None:
            schema_requirement = (
                "\nReturn exactly one JSON object matching this schema; do not add prose or "
                f"extra fields. OUTPUT_SCHEMA={json.dumps(schema, ensure_ascii=False)}"
            )
        payload = (
            f"Produce a {output_type.__name__} JSON artifact from this validated input.\n"
            f"INPUT_JSON={input_json}{schema_requirement}{repair}"
        )
        return payload

    # 功能：
    #   检查契约中是否存在自由对象字典；存在时选择 JSON 模式加本地校验。
    # 输入：
    #   schema：由 Pydantic 生成的输出契约。
    # 输出：
    #   supported：所有对象节点是否都明确禁止额外属性。
    @staticmethod
    def _supports_strict_structured_output(schema: dict[str, object]) -> bool:
        # 功能：
        #   遍历包括定义区在内的契约容器，识别严格模式不能接受的对象节点。
        # 输入：
        #   value：当前契约节点。
        # 输出：
        #   supported：当前节点及后代是否满足对象封闭条件。
        def visit(value: object) -> bool:
            if isinstance(value, dict):
                if value.get("type") == "object" and value.get("additionalProperties") is not False:
                    supported = False
                else:
                    supported = all(visit(item) for item in value.values())
            elif isinstance(value, list):
                supported = all(visit(item) for item in value)
            else:
                supported = True
            return supported

        supported = visit(schema)
        return supported

    # 功能：
    #   有界读取并核对图像字节摘要，再编码为数据 URL，防止同一路径的新帧冒充旧帧。
    #   内容类型来自准入描述或扩展名，此处摘要校验不能替代上游图像解码质量检查。
    # 输入：
    #   item：带摘要的不可变图像字节或本地文件描述。
    # 输出：
    #   data_url：包含声明媒体类型及 Base64 图像字节的数据 URL。
    @staticmethod
    def _media_data_url(item: dict[str, object]) -> str:
        if item.get("kind") != "image-file":
            raise ValueError("unsupported multimodal media kind")
        digests = [item[key] for key in ("sha256", "content_sha256") if key in item]
        if (
            not digests
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in digests
            )
            or len(set(digests)) != 1
        ):
            raise ValueError("multimodal image digest is missing or inconsistent")
        inline = item.get("content_bytes")
        if "content_bytes" in item:
            if not isinstance(inline, bytes):
                raise ValueError("multimodal image bytes must be immutable")
            content = inline
            fallback_type = None
        else:
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
                raise ValueError("multimodal image is unavailable")
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise ValueError("multimodal image is unavailable")
            size = path.stat().st_size
            if size <= 0 or size > 12 * 1024 * 1024:
                raise ValueError("multimodal image exceeds bounded size")
            fallback_type = mimetypes.guess_type(path.name)[0]
            # The file can grow after stat; enforce the actual read bound too.
            with path.open("rb") as stream:
                content = stream.read(12 * 1024 * 1024 + 1)
        content_type = item.get("content_type") or fallback_type or ""
        if not isinstance(content_type, str) or content_type not in {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        }:
            raise ValueError("unsupported multimodal image content type")
        if not content or len(content) > 12 * 1024 * 1024:
            raise ValueError("multimodal image exceeds bounded size")
        if hashlib.sha256(content).hexdigest() != digests[0]:
            raise ValueError("multimodal image digest does not match the admitted bytes")
        encoded = base64.b64encode(content).decode("ascii")
        data_url = f"data:{content_type};base64,{encoded}"
        return data_url

    # 功能：
    #   使用 Responses 内容类型编码同一组文本和图像，保留纯文本调用的线格式。
    # 输入：
    #   text：用户输入文本。
    #   multimodal：已准入图像描述。
    # 输出：
    #   request_input：纯文本或含用户角色的多模态消息列表。
    @classmethod
    def _responses_input(
        cls, text: str, multimodal: list[dict[str, object]]
    ) -> str | list[dict[str, object]]:
        if not multimodal:
            request_input = text
            return request_input
        content: list[dict[str, object]] = [{"type": "input_text", "text": text}]
        content.extend(
            {"type": "input_image", "image_url": cls._media_data_url(item)} for item in multimodal
        )
        request_input = [{"role": "user", "content": content}]
        return request_input

    # 功能：
    #   使用 Chat Completions 内容类型编码已准入图像，不改变其字节身份。
    # 输入：
    #   text：用户输入文本。
    #   multimodal：已准入图像描述。
    # 输出：
    #   content：纯文本或该协议的内容块列表。
    @classmethod
    def _chat_input(
        cls, text: str, multimodal: list[dict[str, object]]
    ) -> str | list[dict[str, object]]:
        if not multimodal:
            content = text
            return content
        content: list[dict[str, object]] = [{"type": "text", "text": text}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": cls._media_data_url(item), "detail": "auto"},
            }
            for item in multimodal
        )
        return content


class FailoverStructuredModelPort:
    """Prefer one provider while keeping a bounded text-only fallback available."""

    # 功能：
    #   建立显式主备调度，只有外层报告失败或超时才切换，不增加隐藏调用。
    # 输入：
    #   primary：首选模型端口。
    #   fallback：不同供应商或本地角色的备用端口。
    #   fallback_accepts_multimodal：是否允许向备用端口传图像。
    #   primary_probe_after_fallback_successes：备用连续成功多少次后探测主端口。
    #   retry_primary_after_returned_failure：本地主端口返回拒绝时是否继续优先本地重试。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self,
        primary: StructuredModelPort,
        fallback: StructuredModelPort,
        *,
        fallback_accepts_multimodal: bool = False,
        primary_probe_after_fallback_successes: int = 3,
        retry_primary_after_returned_failure: bool = False,
    ) -> None:
        if primary.settings.name == fallback.settings.name:
            raise ValueError("model failover providers must be distinct")
        if (
            type(primary_probe_after_fallback_successes) is not int
            or not 1 <= primary_probe_after_fallback_successes <= 10
        ):
            raise ValueError("primary probe interval must be between 1 and 10 successes")
        if type(fallback_accepts_multimodal) is not bool:
            raise ValueError("fallback_accepts_multimodal must be a boolean")
        if type(retry_primary_after_returned_failure) is not bool:
            raise ValueError("retry_primary_after_returned_failure must be a boolean")
        self.primary = primary
        self.fallback = fallback
        self.fallback_accepts_multimodal = fallback_accepts_multimodal
        self.primary_probe_after_fallback_successes = primary_probe_after_fallback_successes
        self.retry_primary_after_returned_failure = retry_primary_after_returned_failure
        self.settings = primary.settings
        self._active_index = 0
        self._generation = 0
        self._fallback_successes_remaining = 0
        self._lock = threading.Lock()
        self._closed = False

    # 功能：
    #   调用当前端口并检查结果仍属于有效代次；文本备用端口不接收图像。
    # 输入：
    #   kwargs：传给实际模型端口的结构化调用参数。
    # 输出：
    #   result：当前有效调用结果。
    def call(self, **kwargs: Any) -> Any:
        with self._lock:
            if self._closed:
                raise TimeoutError("MODEL_TRANSPORT_UNAVAILABLE")
            active_index = self._active_index
            generation = self._generation
        port = self.primary if active_index == 0 else self.fallback
        request = dict(kwargs)
        if active_index == 1 and not self.fallback_accepts_multimodal:
            request["multimodal"] = []
        result = port.call(**request)
        with self._lock:
            if self._closed or self._generation != generation:
                attempts = getattr(getattr(result, "record", None), "attempt", 1)
                raise ModelInvocationError(
                    "MODEL_TRANSPORT_RETIRED",
                    attempts_used=attempts,
                    reason_code="MODEL_TRANSPORT_RETIRED",
                )
            if active_index == 1 and self._active_index == active_index:
                if self._fallback_successes_remaining > 1:
                    self._fallback_successes_remaining -= 1
                else:
                    self._fallback_successes_remaining = 0
                    self._active_index = 0
        return result

    # 功能：
    #   暴露下一次调用所选端口的超时，供外层调度规划等待预算。
    # 输入：
    #   无。
    # 输出：
    #   timeout_seconds：当前所选端口的物理调用超时秒数。
    @property
    def invocation_timeout_seconds(self) -> float:
        with self._lock:
            active_index = self._active_index
        port = self.primary if active_index == 0 else self.fallback
        timeout_seconds = port.invocation_timeout_seconds
        return timeout_seconds

    # 功能：
    #   撤销当前调用代次并为下一周期切换端口，同时请求旧端口恢复传输。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def reset_transport(self) -> None:
        with self._lock:
            if self._closed:
                return
            active_index = self._active_index
            self._generation += 1
            if active_index == 0:
                # Open a small circuit after a primary failure. A successful
                # text-only peer remains selected for several calls before a
                # visual-primary probe, so a known primary outage cannot be
                # paid again on every navigation cycle.
                self._active_index = 1
                self._fallback_successes_remaining = self.primary_probe_after_fallback_successes
            else:
                # A failed fallback is not a viable circuit.  Re-probe the
                # primary on the next bounded call instead of becoming stuck
                # on a slow or schema-incompatible peer indefinitely.
                self._active_index = 0
                self._fallback_successes_remaining = 0
            port = self.primary if active_index == 0 else self.fallback
        port.reset_transport()

    # 功能：
    #   处理截止前已返回的失败；本地主端口的拒绝可保留本地优先而不立即切至云端。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def advance_after_failure(self) -> None:
        with self._lock:
            if self._closed:
                return
            active_index = self._active_index
            if active_index == 0 and self.retry_primary_after_returned_failure:
                self._generation += 1
        if active_index == 0 and self.retry_primary_after_returned_failure:
            # A bounded local inference that completed with a rejected result
            # is not evidence that its transport is unavailable.  Real-time
            # navigation must hold for one cycle and retry the millisecond
            # primary rather than handing the high-rate loop to a multi-second
            # cloud fallback.  True timeouts still enter reset_transport and
            # retain the ordinary failover path.
            self.primary.reset_transport()
            return
        self.reset_transport()

    # 功能：
    #   为所选端口预取图像描述；不向文本备用端口传图，也不在关闭后预取。
    # 输入：
    #   multimodal：上游已准入图像描述。
    # 输出：
    #   None：不返回业务数据。
    def prime_multimodal(self, multimodal: list[dict[str, object]]) -> None:
        with self._lock:
            if self._closed:
                return
            active_index = self._active_index
        if active_index == 1 and not self.fallback_accepts_multimodal:
            return
        port = self.primary if active_index == 0 else self.fallback
        prime = getattr(port, "prime_multimodal", None)
        if callable(prime):
            prime([dict(item) for item in multimodal])

    # 功能：
    #   先撤销所有新调用及迟到结果，再尝试清理两端；保留首个清理异常交给调用方。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._generation += 1
                self._closed = True
        first_error: Exception | None = None
        for port in (self.primary, self.fallback):
            close = getattr(port, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error
