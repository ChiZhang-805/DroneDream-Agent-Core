"""Authenticated console adapter; all mission decisions stay in the desktop services.

This module never imports the development runner, a flight controller, or model
weights. Planning and execution go through the same HTTP contracts as the UI.
"""

from __future__ import annotations

import re
import time
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, SecretStr, field_validator

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .models import (
    AppModel,
    MissionExecuteRequest,
    MissionPrepareRequest,
    RuntimeMessageRequest,
    ThreadCreate,
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_NODES = 524288
THREAD_ID = re.compile(r"thread-[0-9a-f]{32}")
PLAN_ID = re.compile(r"plan-[0-9a-f]{32}")
_SAFE_REMOTE_ERRORS = frozenset(
    {
        "LOCAL_SESSION_REQUIRED",
        "VERIFIED_IDENTITY_REQUIRED",
        "IDENTITY_TOKEN_EXPIRED",
        "RESOURCE_NOT_FOUND",
        "RUNTIME_PROVISION_REQUIRED",
        "RUNTIME_RESOURCES_MISSING",
        "TASK_NOT_AWAITING_CONFIRMATION",
        "EXECUTION_PLAN_REVISION_MISMATCH",
        "EXECUTION_THREAD_OWNER_MISMATCH",
        "EXECUTION_THREAD_NOT_BOUND",
        "PREPARATION_MODEL_MISMATCH",
        "MODEL_GATEWAY_REQUIRED",
        "MODEL_PROVIDER_UNAVAILABLE",
        "MODEL_PROVIDER_FAILED",
        "MODEL_USAGE_SETTLEMENT_FAILED",
        "INSUFFICIENT_ALLOWANCE",
    }
)


class ConsoleError(RuntimeError):
    """Public error codes contain neither service response bodies nor credentials."""


class ConsolePlanningSelection(AppModel):
    """Only public task inputs; authorization and execution knobs are not accepted."""

    message: str = Field(min_length=3, max_length=4000)
    map_id: str = Field(min_length=1, max_length=120)
    map_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_id: str = Field(min_length=1, max_length=120)
    vehicle_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_entity: str = Field(default="__auto__", min_length=1, max_length=160)


class ConsoleSession(AppModel):
    """In-memory credentials supplied by the operator, never a saved login file."""

    core_url: str
    supabase_url: str
    local_token: SecretStr
    identity_token: SecretStr
    publishable_key: SecretStr
    source_edition: Literal["universal", "sim", "lab", "field", "autonomy"]

    # 功能：
    #   固定本机 IPv4 回环端点，禁止代理主机、用户信息、路径和重定向式配置。
    # 输入：
    #   cls：会话配置类型。
    #   value：Core HTTP 根地址。
    # 输出：
    #   origin：规范化后的固定回环地址。
    @field_validator("core_url")
    @classmethod
    def validate_core_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
            or value != f"http://127.0.0.1:{parsed.port}{parsed.path}"
        ):
            raise ValueError("CONSOLE_CORE_ORIGIN_INVALID")
        origin = value.rstrip("/")
        return origin

    # 功能：
    #   将账户服务固定为明确配置的 Supabase HTTPS 项目，不信任服务返回的新主机。
    # 输入：
    #   cls：会话配置类型。
    #   value：账户所属项目根地址。
    # 输出：
    #   origin：通过校验的项目根地址。
    @field_validator("supabase_url")
    @classmethod
    def validate_supabase_url(cls, value: str) -> str:
        if re.fullmatch(r"https://[a-z0-9]{20}\.supabase\.co/?", value) is None:
            raise ValueError("CONSOLE_CLOUD_ORIGIN_INVALID")
        origin = value.rstrip("/")
        return origin

    # 功能：
    #   拒绝空白、控制字符及超长秘密，防止 HTTP 头注入；不在错误中包含原值。
    # 输入：
    #   cls：会话配置类型。
    #   value：保存在内存中的秘密值。
    # 输出：
    #   value：可安全放入请求头的秘密对象。
    @field_validator("local_token", "identity_token", "publishable_key")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not 16 <= len(raw) <= 16384 or any(not 33 <= ord(char) <= 126 for char in raw):
            raise ValueError("CONSOLE_CREDENTIAL_INVALID")
        return value


class ProductConsoleClient:
    """Thin, non-retrying HTTP client for account-bound planning and simulation."""

    # 功能：
    #   建立不继承环境代理、不跟随重定向的连接池；构造过程不发起请求。
    # 输入：
    #   self：命令行业务适配器。
    #   session：内存会话配置。
    #   transport：测试专用传输注入点，命令行不对用户开放此选项。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, session: ConsoleSession, *, transport: httpx.BaseTransport | None = None):
        self.session = session
        self.gateway = session.supabase_url + "/functions/v1/model-gateway"
        self._http = httpx.Client(trust_env=False, follow_redirects=False, transport=transport)
        self._secrets = [
            session.local_token.get_secret_value(),
            session.identity_token.get_secret_value(),
            session.publishable_key.get_secret_value(),
        ]

    # 功能：
    #   关闭连接池；退出控制台不等于停止后台任务或向飞控发出降落命令。
    # 输入：
    #   self：当前客户端。
    # 输出：
    #   None：不返回业务数据。
    def close(self) -> None:
        self._http.close()

    # 功能：
    #   1. 单次请求共享服务，限制响应大小并拒绝压缩、重定向和不合法 JSON。
    #   2. 写请求超时后报告结果未知，不重试、伪造成功或取消已经接收的后台工作。
    # 输入：
    #   self：当前客户端。
    #   method：HTTP 方法。
    #   path：由本模块固定的服务路径。
    #   body：已验证请求对象。
    #   cloud：是否访问账户网关。
    #   timeout：单次慢速业务请求秒数，不用于实时飞行控制。
    # 输出：
    #   result：服务返回的 JSON 对象，不代表飞行验收通过。
    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        cloud: bool = False,
        timeout: float = 30.0,
    ) -> dict:
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if cloud:
            headers["Authorization"] = "Bearer " + self.session.identity_token.get_secret_value()
            base = self.gateway
        else:
            headers["Authorization"] = "Bearer " + self.session.local_token.get_secret_value()
            headers["X-DroneDream-Identity-Token"] = self.session.identity_token.get_secret_value()
            headers["X-DroneDream-Supabase-Publishable-Key"] = (
                self.session.publishable_key.get_secret_value()
            )
            base = self.session.core_url
        raw = None
        if body is not None:
            raw = encode_json(body, limit=MAX_RESPONSE_BYTES, node_limit=65536).encode("utf-8")
            headers["Content-Type"] = "application/json"
        deadline = time.monotonic() + timeout
        try:
            with self._http.stream(
                method,
                base + path,
                headers=headers,
                content=raw,
                timeout=httpx.Timeout(timeout, connect=5.0),
            ) as response:
                if response.is_redirect:
                    raise ConsoleError("CONSOLE_REDIRECT_REJECTED")
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ConsoleError("CONSOLE_COMPRESSED_RESPONSE_REJECTED")
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise httpx.ReadTimeout("deadline")
                    if len(chunks) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise ConsoleError("CONSOLE_RESPONSE_TOO_LARGE")
                    chunks.extend(chunk)
                value = decode_json(
                    bytes(chunks), limit=MAX_RESPONSE_BYTES, node_limit=MAX_RESPONSE_NODES
                )
                if not isinstance(value, dict):
                    raise ConsoleError("CONSOLE_RESPONSE_INVALID")
                if not response.is_success:
                    detail = value.get("detail")
                    error = value.get("error")
                    if cloud and isinstance(error, dict):
                        detail = error.get("code")
                    code = (
                        detail
                        if isinstance(detail, str) and detail in _SAFE_REMOTE_ERRORS
                        else (f"CONSOLE_HTTP_{response.status_code}")
                    )
                    raise ConsoleError(code)
                result = value.get("data") if cloud else value
                if not isinstance(result, dict):
                    raise ConsoleError("CONSOLE_RESPONSE_INVALID")
                return result
        except httpx.RequestError as error:
            code = "CONSOLE_WRITE_OUTCOME_UNKNOWN" if method != "GET" else "CONSOLE_NETWORK_ERROR"
            raise ConsoleError(code) from error
        except (ValueError, UnicodeError) as error:
            raise ConsoleError("CONSOLE_RESPONSE_INVALID") from error

    # 功能：
    #   校验任务标识，避免任意路径、查询字符串或另一类资源被拼入业务端点。
    # 输入：
    #   thread_id：任务标识。
    # 输出：
    #   path：当前任务的固定 HTTP 路径。
    @staticmethod
    def _thread_path(thread_id: str) -> str:
        if THREAD_ID.fullmatch(thread_id) is None:
            raise ConsoleError("CONSOLE_THREAD_ID_INVALID")
        path = "/v1/threads/" + thread_id
        return path

    # 功能：
    #   读取桌面共享的资产、模型及任务目录，不猜测默认地图或替换模型包。
    # 输入：
    #   self：当前客户端。
    # 输出：
    #   catalog：当前 Core 目录。
    def bootstrap(self) -> dict:
        catalog = self._request("GET", "/v1/bootstrap")
        return catalog

    # 功能：
    #   读取账户云端真实用量，不根据调用次数在本地估算或修改余额。
    # 输入：
    #   self：当前客户端。
    # 输出：
    #   usage：云端用量快照。
    def usage(self) -> dict:
        usage = self._request("GET", "/usage", cloud=True)
        return usage

    # 功能：
    #   读取 Runtime 安装和连接状态；安装成功不代表传感器或模型已经就绪。
    # 输入：
    #   self：当前客户端。
    # 输出：
    #   runtime：后端报告的 Runtime 状态。
    def runtime(self) -> dict:
        runtime = self._request("GET", "/v1/runtime/status")
        return runtime

    # 功能：
    #   创建共享后端的任务会话，不调用模型、不产生飞行权限。
    # 输入：
    #   self：当前客户端。
    #   request：选择的模型、标题和语言。
    # 输出：
    #   thread：后端创建的任务。
    def create(self, request: ThreadCreate) -> dict:
        thread = self._request("POST", "/v1/threads", request.model_dump(mode="json"))
        return thread

    # 功能：
    #   读取任务及消息；不把传输成功解释成任务成功。
    # 输入：
    #   self：当前客户端。
    #   thread_id：任务标识。
    # 输出：
    #   thread：服务中的当前任务快照。
    def status(self, thread_id: str) -> dict:
        thread = self._request("GET", self._thread_path(thread_id))
        return thread

    # 功能：
    #   为任务当前选中的托管模型申请与桌面同范围的短期授权；不读取供应商 API Key。
    # 输入：
    #   self：当前客户端。
    #   thread：服务返回的任务及当前选中模型。
    # 输出：
    #   connection：只用于下一次业务请求的模型授权字段。
    def _connection(self, thread: dict) -> dict:
        thread_id = thread.get("thread_id", "")
        self._thread_path(thread_id)
        selected = thread.get("selected_model")
        catalog = self._request("GET", "/v1/models").get("models")
        if not isinstance(catalog, list):
            raise ConsoleError("CONSOLE_MODEL_CATALOG_INVALID")
        matches = [
            item for item in catalog if isinstance(item, dict) and item.get("id") == selected
        ]
        if len(matches) != 1:
            raise ConsoleError("CONSOLE_SELECTED_MODEL_UNAVAILABLE")
        model = matches[0]
        if model.get("source") != "default":
            raise ConsoleError("CONSOLE_MANAGED_MODEL_REQUIRED")
        if not isinstance(model.get("provider"), str) or not isinstance(model.get("model"), str):
            raise ConsoleError("CONSOLE_MODEL_CATALOG_INVALID")
        grant = self._request(
            "POST",
            "/grants",
            {
                "scope": "assistant",
                "scope_reference": thread_id,
                "provider": model["provider"],
                "model": model["model"],
            },
            cloud=True,
        )
        secret = grant.get("grant")
        if (
            not isinstance(secret, str)
            or re.fullmatch(r"ddg_[A-Za-z0-9_-]{20,124}", secret) is None
        ):
            raise ConsoleError("CONSOLE_MODEL_GRANT_INVALID")
        self._secrets.append(secret)
        if grant.get("gateway_base_url") != self.gateway:
            raise ConsoleError("CONSOLE_GATEWAY_ORIGIN_MISMATCH")
        connection = {"model_id": selected, "model_grant": secret, "gateway_base_url": self.gateway}
        return connection

    # 功能：
    #   1. 将自然语言和精确资产摘要提交给桌面共用的真实模型规划服务。
    #   2. 规划与执行分离，输入不允许指定教师控制、绕过计费或直接执行开发脚本。
    # 输入：
    #   self：当前客户端。
    #   thread_id：已有任务标识。
    #   selection：自然语言、地图、飞机及来源摘要字段。
    # 输出：
    #   plan：后端生成的待确认计划及真实调用计数。
    def prepare(self, thread_id: str, selection: dict) -> dict:
        # 精确内容绑定必须由目录选择，不能在自动化中默默改成另一份地图或飞机。
        selected = ConsolePlanningSelection.model_validate(selection)
        path = self._thread_path(thread_id)
        thread = self.status(thread_id)
        payload = {
            **selected.model_dump(mode="json"),
            **self._connection(thread),
            "source_edition": self.session.source_edition,
            "locale": thread.get("locale"),
            "input_channel": "api",
        }
        request = MissionPrepareRequest.model_validate(payload)
        plan = self._request(
            "POST", path + "/prepare", request.model_dump(mode="json"), timeout=600
        )
        return plan

    # 功能：
    #   明确确认一个计划修订后请求共享仿真执行器；旧计划、身份及资产仍由后端复验。
    # 输入：
    #   self：当前客户端。
    #   thread_id：任务标识。
    #   plan_revision_id：操作者已阅读并明确确认的计划标识。
    # 输出：
    #   execution：后端启动回执，不代表起飞或任务成功。
    def execute(self, thread_id: str, plan_revision_id: str) -> dict:
        path = self._thread_path(thread_id)
        if PLAN_ID.fullmatch(plan_revision_id) is None:
            raise ConsoleError("CONSOLE_EXPLICIT_PLAN_CONFIRMATION_REQUIRED")
        thread = self.status(thread_id)
        if thread.get("state") != "awaiting_confirmation":
            raise ConsoleError("TASK_NOT_AWAITING_CONFIRMATION")
        plans = [
            item.get("metadata")
            for item in thread.get("messages", [])
            if isinstance(item, dict)
            and item.get("kind") == "plan"
            and item.get("role") == "assistant"
        ]
        if (
            not plans
            or not isinstance(plans[-1], dict)
            or (plans[-1].get("plan_revision_id") != plan_revision_id)
        ):
            raise ConsoleError("CONSOLE_PLAN_CHANGED_REVIEW_AGAIN")
        request = MissionExecuteRequest.model_validate(
            {
                **self._connection(thread),
                "source_edition": self.session.source_edition,
                "plan_revision_id": plan_revision_id,
            }
        )
        execution = self._request("POST", path + "/execute", request.model_dump(mode="json"))
        return execution

    # 功能：
    #   读取后端独立执行证据，不根据日志文字自行授予飞行成功。
    # 输入：
    #   self：当前客户端。
    #   thread_id：任务标识。
    # 输出：
    #   evidence：共享执行器保存的真实证据。
    def evidence(self, thread_id: str) -> dict:
        evidence = self._request("GET", self._thread_path(thread_id) + "/execution-evidence")
        return evidence

    # 功能：
    #   将临时自然语言要求提交到当前任务的运行消息链路，不直接写飞控设定值。
    # 输入：
    #   self：当前客户端。
    #   thread_id：任务标识。
    #   text：临时要求；此通道不是紧急制动接口。
    # 输出：
    #   receipt：消息接收回执，不代表要求已经执行。
    def message(self, thread_id: str, text: str) -> dict:
        request = RuntimeMessageRequest(text=text)
        receipt = self._request(
            "POST",
            self._thread_path(thread_id) + "/runtime-message",
            request.model_dump(mode="json"),
        )
        return receipt

    # 功能：
    #   将输出序列化并隐藏已知令牌和权限对象，防止重定向终端输出时泄露执行凭据。
    # 输入：
    #   self：当前客户端。
    #   value：待展示的业务结果。
    # 输出：
    #   rendered：可写入终端或验收文件的 JSON 文本。
    def render(self, value: dict) -> str:
        hidden = {
            "grant",
            "model_grant",
            "api_key",
            "access_token",
            "refresh_token",
            "authorization",
            "execution_authority",
            "local_token",
            "identity_token",
            "publishable_key",
        }

        # 功能：
        #   复制有界 JSON 树并遮蔽权限字段，保留原始后台证据不变。
        # 输入：
        #   item：当前 JSON 节点。
        # 输出：
        #   redacted：脱敏后的节点。
        def redact(item: object) -> object:
            if isinstance(item, dict):
                return {
                    key: "[redacted]" if key.lower() in hidden else redact(child)
                    for key, child in item.items()
                }
            if isinstance(item, list):
                return [redact(child) for child in item]
            return item

        # 先验证深度与大小，再递归，避免任意嵌套输入造成递归异常。
        encode_json(value, limit=MAX_RESPONSE_BYTES, node_limit=MAX_RESPONSE_NODES)
        rendered = encode_json(
            redact(value), limit=MAX_RESPONSE_BYTES, node_limit=MAX_RESPONSE_NODES
        )
        for secret in self._secrets:
            rendered = rendered.replace(secret, "[redacted]")
        return rendered
