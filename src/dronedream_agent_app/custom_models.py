"""User-owned model profiles with local encrypted credentials."""

from __future__ import annotations

import ctypes
import os
import re
import secrets
import tempfile
import threading
from contextlib import suppress
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from openai import APIStatusError, OpenAI

from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file

from .models import CustomModelCreateRequest, CustomModelDiscoverRequest
from .storage import AppStore

ApiStyle = Literal["responses", "chat-completions"]


class CredentialVault(Protocol):
    # 功能：
    #   约定按不透明标识保存秘密，持久化实现不得写出明文。
    # 输入：
    #   self：凭证库实现。
    #   name：不透明凭证标识。
    #   secret：要保存的秘密原文。
    # 输出：
    #   None：不返回业务数据。
    def put(self, name: str, secret: str) -> None: ...

    # 功能：
    #   约定读取凭证原文，缺失时由实现抛出 KeyError。
    # 输入：
    #   self：凭证库实现。
    #   name：不透明凭证标识。
    # 输出：
    #   secret：对应秘密原文。
    def get(self, name: str) -> str: ...

    # 功能：
    #   约定幂等删除一个秘密；调用方另外负责撤销其元数据和授权。
    # 输入：
    #   self：凭证库实现。
    #   name：不透明凭证标识。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, name: str) -> None: ...


# 功能：
#   保存失败后补偿删除独立秘密；补偿失败只附固定说明，不覆盖根因或记录秘密。
# 输入：
#   vault：负责实际删除的凭证库。
#   name：本次保存使用的凭证标识。
#   error：需要继续抛出的原始保存异常。
# 输出：
#   None：不返回业务数据。
def rollback_credential(vault: CredentialVault, name: str, error: BaseException) -> None:
    try:
        vault.delete(name)
    except Exception:
        error.add_note("CREDENTIAL_CLEANUP_FAILED")


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


class WindowsCredentialVault:
    """DPAPI CurrentUser encrypted storage; it does not protect against this user's processes."""

    # 功能：
    #   1. 建立 Windows 当前用户 DPAPI 凭证目录，其他平台不降级为明文文件。
    #   2. 明确原生函数指针类型，拒绝静态目录链接；不替代同用户进程间的系统隔离。
    # 输入：
    #   self：待初始化的凭证库。
    #   root：应用管理的凭证目录。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, root: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("CUSTOM_MODEL_CREDENTIAL_VAULT_UNAVAILABLE")
        self.root = root
        self._lock = threading.RLock()
        check_plain_plugin_path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        check_plain_plugin_path(self.root)
        self._crypt32 = ctypes.windll.crypt32
        self._kernel32 = ctypes.windll.kernel32
        self._crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            wintypes.LPCWSTR,
            ctypes.POINTER(_DataBlob),
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        self._crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.POINTER(wintypes.LPWSTR),
            ctypes.POINTER(_DataBlob),
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        self._crypt32.CryptProtectData.restype = wintypes.BOOL
        self._crypt32.CryptUnprotectData.restype = wintypes.BOOL
        self._kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
        self._kernel32.LocalFree.restype = wintypes.HLOCAL

    # 功能：
    #   将合法不透明标识映射到固定目录中的密文文件，拒绝路径字符及静态链接。
    # 输入：
    #   self：持有规范目录的凭证库。
    #   name：凭证标识，不是用户指定的相对路径。
    # 输出：
    #   path：检查后的密文文件路径。
    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9-]{8,80}", name):
            raise ValueError("CUSTOM_MODEL_PROFILE_ID_INVALID")
        path = self.root / f"{name}.dpapi"
        check_plain_plugin_path(path)
        return path

    # 功能：
    #   用当前 Windows 用户上下文加密并复制结果，复制后始终释放 DPAPI 原生缓冲区。
    # 输入：
    #   self：已绑定原生加密函数的凭证库。
    #   value：待加密的 UTF-8 秘密字节。
    # 输出：
    #   encrypted：可落盘的 DPAPI 密文字节。
    def _protect(self, value: bytes) -> bytes:
        source_buffer = ctypes.create_string_buffer(value)
        source = _DataBlob(len(value), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
        output = _DataBlob()
        if not self._crypt32.CryptProtectData(
            ctypes.byref(source), "DroneDream AGENT", None, None, None, 1, ctypes.byref(output)
        ):
            raise ctypes.WinError()
        try:
            encrypted = ctypes.string_at(output.pbData, output.cbData)
            return encrypted
        finally:
            self._kernel32.LocalFree(output.pbData)

    # 功能：
    #   在内存中解密当前用户的密文，复制后释放原生输出；不将明文写入临时文件。
    # 输入：
    #   self：已绑定原生解密函数的凭证库。
    #   value：有界读取的 DPAPI 密文字节。
    # 输出：
    #   plaintext：解密后的秘密字节。
    def _unprotect(self, value: bytes) -> bytes:
        source_buffer = ctypes.create_string_buffer(value)
        source = _DataBlob(len(value), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
        output = _DataBlob()
        if not self._crypt32.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)
        ):
            raise ctypes.WinError()
        try:
            plaintext = ctypes.string_at(output.pbData, output.cbData)
            return plaintext
        finally:
            self._kernel32.LocalFree(output.pbData)

    # 功能：
    #   1. 在同实例锁内检查并替换密文目标，避免 Windows 并发打开或替换的共享冲突。
    #   2. 发布前核对暂存仍为当前写入创建的文件；不声称提供跨进程文件锁。
    # 输入：
    #   self：凭证库实例。
    #   temporary：已关闭并同步的独占暂存文件。
    #   owned：创建暂存时取得的文件身份。
    #   target：准备替换的密文目标。
    # 输出：
    #   None：不返回业务数据。
    def _publish(self, temporary: Path, owned: os.stat_result, target: Path) -> None:
        with self._lock:
            check_plain_plugin_path(target)
            check_plain_plugin_path(temporary)
            if not os.path.samestat(owned, temporary.stat()):
                raise ValueError("CREDENTIAL_STAGING_CHANGED")
            temporary.replace(target)

    # 功能：
    #   1. 限制秘密为非空且不超过 64 KiB 的 UTF-8，先加密再独占创建同目录暂存。
    #   2. 写入同步后原子替换目标；失败只回收本次仍属自己的暂存，不删除旧密文。
    #   3. 并发写入互不共用暂存路径；同标识最终值取决于最后一次成功发布。
    # 输入：
    #   self：Windows 当前用户凭证库。
    #   name：要创建或替换的凭证标识。
    #   secret：秘密原文，保留首尾空格。
    # 输出：
    #   None：不返回业务数据。
    def put(self, name: str, secret: str) -> None:
        target = self._path(name)
        if not isinstance(secret, str) or not secret or len(secret) > 65_536:
            raise ValueError("CREDENTIAL_SECRET_INVALID")
        plaintext = secret.encode("utf-8")
        if len(plaintext) > 65_536:
            raise ValueError("CREDENTIAL_SECRET_TOO_LARGE")
        encrypted = self._protect(plaintext)
        temporary = None
        owned = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{name}-", suffix=".tmp", dir=self.root, delete=False
            ) as output:
                temporary = Path(output.name)
                owned = os.fstat(output.fileno())
                output.write(encrypted)
                output.flush()
                os.fsync(output.fileno())
            self._publish(temporary, owned, target)
        except BaseException as error:
            if temporary is not None:
                try:
                    if owned is not None and os.path.samestat(owned, temporary.lstat()):
                        temporary.unlink()
                    else:
                        error.add_note("CREDENTIAL_STAGING_OWNERSHIP_CHANGED")
                except FileNotFoundError:
                    pass
                except Exception:
                    error.add_note("CREDENTIAL_STAGING_CLEANUP_FAILED")
            raise

    # 功能：
    #   有界稳定读取最多 128 KiB 密文后解密，区分缺失、读取损坏和原生解密失败。
    # 输入：
    #   self：Windows 当前用户凭证库。
    #   name：需要读取的凭证标识。
    # 输出：
    #   secret：精确恢复的秘密文本。
    def get(self, name: str) -> str:
        try:
            with self._lock:
                encrypted = read_plugin_file(self._path(name), limit=128 * 1024)
            secret = self._unprotect(encrypted).decode("utf-8")
            return secret
        except FileNotFoundError as error:
            raise KeyError(name) from error

    # 功能：
    #   幂等删除指定密文文件；服务层须先撤销元数据和临时授权。
    # 输入：
    #   self：Windows 当前用户凭证库。
    #   name：需要删除的凭证标识。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, name: str) -> None:
        with self._lock:
            self._path(name).unlink(missing_ok=True)


@dataclass(frozen=True)
class ModelConnection:
    """Internal connection material; never serialize this object into a public profile."""

    selection_id: str
    provider: str
    model_id: str
    api_key: str = field(repr=False)
    base_url: str
    api_style: ApiStyle
    capability_id: str
    source: Literal["default", "custom"]
    supports_image_input: bool = False


@dataclass(frozen=True)
class _Grant:
    connection: ModelConnection
    thread_id: str
    expires_at: datetime


_PROVIDERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("openai", "openai", ("api.openai.com",)),
    ("azure-openai", "azure-openai", ("openai.azure.com", "services.ai.azure.com")),
    ("anthropic", "claude", ("api.anthropic.com",)),
    ("deepseek", "deepseek", ("api.deepseek.com",)),
    ("kimi", "kimi", ("api.moonshot.ai", "moonshot.cn")),
    ("openrouter", "openrouter", ("openrouter.ai",)),
    ("xai", "xai", ("api.x.ai",)),
    ("groq", "groq", ("api.groq.com",)),
    ("mistral", "mistral", ("api.mistral.ai",)),
    ("together", "together", ("api.together.xyz",)),
    ("qwen", "qwen", ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com")),
    ("zhipu", "zhipu", ("open.bigmodel.cn",)),
    ("minimax", "minimax", ("api.minimax.chat", "api.minimaxi.com")),
    ("doubao", "doubao", ("volces.com", "volcengineapi.com")),
    ("hunyuan", "hunyuan", ("hunyuan.tencentcloudapi.com",)),
    ("baichuan", "baichuan", ("api.baichuan-ai.com",)),
    ("stepfun", "stepfun", ("api.stepfun.com",)),
    ("siliconflow", "siliconflow", ("api.siliconflow.cn", "api.siliconflow.com")),
    ("gemini", "gemini", ("generativelanguage.googleapis.com",)),
    ("vertexai", "vertexai", ("aiplatform.googleapis.com",)),
    ("cerebras", "cerebras", ("api.cerebras.ai",)),
    ("cohere", "cohere", ("api.cohere.ai",)),
    ("perplexity", "perplexity", ("api.perplexity.ai",)),
    ("nvidia", "nvidia", ("integrate.api.nvidia.com",)),
    ("cloudflare", "cloudflare", ("api.cloudflare.com",)),
    ("replicate", "replicate", ("api.replicate.com",)),
    ("huggingface", "huggingface", ("huggingface.co",)),
    ("github-models", "github", ("models.github.ai", "models.inference.ai.azure.com")),
    ("fireworks", "fireworks", ("api.fireworks.ai",)),
    ("sambanova", "sambanova", ("api.sambanova.ai",)),
    ("novita", "novita", ("api.novita.ai",)),
    ("ai21", "ai21", ("api.ai21.com",)),
    ("upstage", "upstage", ("api.upstage.ai",)),
    ("modelscope", "modelscope", ("modelscope.cn",)),
    ("yi", "yi", ("api.lingyiwanwu.com",)),
    ("ollama", "ollama", ("127.0.0.1", "localhost", "::1")),
)

_PROVIDER_ICON_BY_NAME = {provider: icon for provider, icon, _hosts in _PROVIDERS}
_PROVIDER_ICON_BY_NAME.update(
    {
        "claude": "claude",
        "grok": "grok",
        "glm": "zhipu",
        "lmstudio": "lmstudio",
        "moonshot": "kimi",
        "yi": "yi",
    }
)

_MODEL_PROVIDER_HINTS: tuple[tuple[str, str, str], ...] = (
    (r"(^|[/._-])claude([/._-]|$)", "anthropic", "claude"),
    (r"(^|[/._-])gemini([/._-]|$)", "gemini", "gemini"),
    (r"(^|[/._-])grok([/._-]|$)", "xai", "grok"),
    (r"(^|[/._-])(gpt|o1|o3|o4|codex)([/._-]|$)", "openai", "openai"),
    (r"deepseek", "deepseek", "deepseek"),
    (r"(^|[/._-])(kimi|moonshot)([/._-]|$)", "kimi", "kimi"),
    (r"qwen", "qwen", "qwen"),
    (r"(^|[/._-])(glm|chatglm)([/._-]|$)", "zhipu", "zhipu"),
    (r"minimax", "minimax", "minimax"),
    (r"(^|[/._-])doubao([/._-]|$)", "doubao", "doubao"),
    (r"hunyuan", "hunyuan", "hunyuan"),
    (r"baichuan", "baichuan", "baichuan"),
    (r"(^|[/._-])step([/._-]|$)", "stepfun", "stepfun"),
    (r"(mistral|mixtral|codestral)", "mistral", "mistral"),
    (r"(^|[/._-])command([/._-]|$)", "cohere", "cohere"),
    (r"(^|[/._-])sonar([/._-]|$)", "perplexity", "perplexity"),
    (r"(^|[/._-])yi([/._-]|$)", "yi", "yi"),
    (r"(jamba|jurassic)", "ai21", "ai21"),
)


# 功能：
#   规范化模型地址，拒绝夹带凭证、查询、片段及非法端口；仅显式回环允许明文 HTTP。
# 输入：
#   value：用户选择的模型 API 基础地址。
# 输出：
#   normalized：移除外围空白及尾斜杠后的地址。
def validate_custom_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("CUSTOM_MODEL_BASE_URL_INVALID")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlparse(normalized)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            parsed.scheme not in ({"http", "https"} if loopback else {"https"})
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "?" in normalized
            or "#" in normalized
            or "\\" in normalized
            or parsed.port == 0
            or any(character.isspace() or ord(character) < 32 for character in normalized)
        ):
            raise ValueError("CUSTOM_MODEL_BASE_URL_INVALID")
    except ValueError as error:
        raise ValueError("CUSTOM_MODEL_BASE_URL_INVALID") from error
    return normalized


# 功能：
#   按端点、密钥前缀及模型名称推测显示品牌；推测结果不证明端点可信或模型能力。
# 输入：
#   base_url：模型 API 基础地址。
#   api_key：只用于本地前缀识别的秘密，不写入输出。
#   model_id：供兜底品牌识别的模型标识。
# 输出：
#   detected：包含提供商品牌及图标名称的字典。
def detect_provider(base_url: str, api_key: str = "", model_id: str = "") -> dict[str, str]:
    host = (urlparse(base_url).hostname or "").lower()
    if host.startswith("bedrock-runtime.") and host.endswith(".amazonaws.com"):
        detected = {"provider": "bedrock", "icon": "bedrock"}
        return detected
    for provider, icon, hosts in _PROVIDERS:
        if any(host == value or host.endswith(f".{value}") for value in hosts):
            if provider == "ollama" and urlparse(base_url).port == 1234:
                detected = {"provider": "lmstudio", "icon": "lmstudio"}
            else:
                detected = {"provider": provider, "icon": icon}
            return detected
    prefix = api_key[:12].lower()
    if prefix.startswith("sk-or-"):
        detected = {"provider": "openrouter", "icon": "openrouter"}
        return detected
    if prefix.startswith("gsk_"):
        detected = {"provider": "groq", "icon": "groq"}
        return detected
    if prefix.startswith("aiza"):
        detected = {"provider": "gemini", "icon": "gemini"}
        return detected
    normalized_model = model_id.strip().lower()
    for pattern, provider, icon in _MODEL_PROVIDER_HINTS:
        if re.search(pattern, normalized_model):
            detected = {"provider": provider, "icon": icon}
            return detected
    detected = {"provider": "openai-compatible", "icon": "generic"}
    return detected


class CustomModelService:
    # 功能：
    #   将可公开模型资料、独立秘密和短时一次性授权分开管理，测试可注入隔离凭证库。
    # 输入：
    #   self：待初始化的模型服务。
    #   store：本机模型与任务元数据存储。
    #   vault：可选凭证库，缺省使用当前 Windows 用户加密存储。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, store: AppStore, vault: CredentialVault | None = None) -> None:
        self.store = store
        self.vault = (
            vault
            if vault is not None
            else WindowsCredentialVault(store.root / "credentials" / "models")
        )
        self._grants: dict[str, _Grant] = {}
        self._lock = threading.RLock()

    # 功能：
    #   为自定义资料建立独立选择命名空间，不与平台模型标识冲突。
    # 输入：
    #   profile_id：自定义连接资料标识。
    # 输出：
    #   selection_id：界面和任务使用的模型选择标识。
    @staticmethod
    def _selection_id(profile_id: str) -> str:
        selection_id = f"custom:{profile_id}"
        return selection_id

    # 功能：
    #   发布已启用模型的显示资料，不读取秘密或返回临时授权令牌。
    # 输入：
    #   self：模型服务。
    # 输出：
    #   catalog：可选模型的非秘密资料列表。
    def catalog(self) -> list[dict[str, object]]:
        catalog = [
            {
                "id": self._selection_id(str(profile["profile_id"])),
                "model": profile["model_id"],
                "label": profile["display_name"],
                "provider": profile["provider"],
                "icon": profile["icon"],
                "source": "custom",
                "profile_id": profile["profile_id"],
            }
            for profile in self.store.list_custom_models()
            if bool(profile["enabled"])
        ]
        return catalog

    # 功能：
    #   1. 在用户明确请求时查询提供商模型目录，最多展示 500 项，所有路径关闭客户端。
    #   2. 缺少目录不是认证成功，401/403 继续抛出；目录内容不证明推理质量或视觉能力。
    # 输入：
    #   self：模型服务。
    #   base_url：用户指定的模型目录基础地址。
    #   api_key：仅交给本次客户端的秘密。
    # 输出：
    #   result：提供商品牌、地址、目录及可选提醒。
    def discover(self, *, base_url: str, api_key: str) -> dict[str, object]:
        request = CustomModelDiscoverRequest(base_url=base_url, api_key=api_key)
        base_url, api_key = request.base_url, request.api_key
        base_url = validate_custom_base_url(base_url)
        detected = detect_provider(base_url, api_key)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)
        try:
            models = sorted(
                {item.id.removeprefix("models/") for item in client.models.list().data if item.id}
            )
        except APIStatusError as error:
            if error.status_code in {401, 403}:
                raise
            result = {
                **detected,
                "base_url": base_url,
                "models": [],
                "warning": "MODEL_CATALOG_UNAVAILABLE_ENTER_ID_MANUALLY",
            }
            return result
        finally:
            # 本次查询拥有连接池，成功、目录不可用和认证错误都必须关闭。
            client.close()
        if detected["provider"] == "openai-compatible":
            for model_id in models:
                candidate = detect_provider(base_url, api_key, model_id)
                if candidate["provider"] != "openai-compatible":
                    detected = candidate
                    break
        result = {**detected, "base_url": base_url, "models": models[:500]}
        return result

    # 功能：
    #   1. 验证连接资料，先保存独立加密秘密，再发布可选择的模型元数据。
    #   2. 任一步保存失败均尝试回收秘密，补偿失败不掩盖根因；两种存储不是同一事务。
    # 输入：
    #   self：模型服务。
    #   display_name：界面显示名称。
    #   base_url：模型 API 地址。
    #   model_id：提供商的模型标识。
    #   api_key：保留原文的秘密。
    #   api_style：Responses 或 Chat Completions 调用协议。
    #   provider：可选显示品牌，缺省从端点和模型名称推测。
    # 输出：
    #   result：非秘密模型资料、选择标识和已保存密钥标志。
    def create(
        self,
        *,
        display_name: str,
        base_url: str,
        model_id: str,
        api_key: str,
        api_style: ApiStyle,
        provider: str | None = None,
    ) -> dict[str, object]:
        request = CustomModelCreateRequest(
            display_name=display_name,
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            api_style=api_style,
            provider=provider,
        )
        display_name, model_id = request.display_name, request.model_id
        base_url, api_key, provider = request.base_url, request.api_key, request.provider
        base_url = validate_custom_base_url(base_url)
        detected = detect_provider(base_url, api_key, model_id)
        selected_provider = (provider or detected["provider"]).strip().lower()
        profile_id = f"cmp-{uuid4().hex[:24]}"
        try:
            self.vault.put(profile_id, api_key)
            profile = self.store.save_custom_model(
                {
                    "profile_id": profile_id,
                    "display_name": display_name,
                    "provider": selected_provider,
                    "icon": _PROVIDER_ICON_BY_NAME.get(selected_provider, detected["icon"]),
                    "base_url": base_url,
                    "api_style": api_style,
                    "model_id": model_id,
                    "enabled": True,
                }
            )
        except BaseException as error:
            rollback_credential(self.vault, profile_id, error)
            raise
        result = {**profile, "selection_id": self._selection_id(profile_id), "has_api_key": True}
        return result

    # 功能：
    #   1. 先撤销模型资料与缓存授权再删除秘密，删除秘密失败也不能继续使用旧授权。
    #   2. 元数据已撤销时仍允许重试回收密文，不让一次文件锁故障永久遗留秘密。
    # 输入：
    #   self：模型服务。
    #   profile_id：待删除的模型资料标识。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, profile_id: str) -> None:
        with self._lock:
            with suppress(KeyError):
                self.store.delete_custom_model(profile_id)
            selection_id = self._selection_id(profile_id)
            self._grants = {
                token: grant
                for token, grant in self._grants.items()
                if grant.connection.selection_id != selection_id
            }
            self.vault.delete(profile_id)

    # 功能：
    #   检查已保存模型是否出现在提供商目录；此操作不是推理质量或飞行资格验收。
    # 输入：
    #   self：模型服务。
    #   profile_id：要检查的已保存模型。
    # 输出：
    #   result：目录匹配结果、品牌、模型标识及目录计数。
    def test(self, profile_id: str) -> dict[str, object]:
        profile = self.store.get_custom_model(profile_id)
        discovered = self.discover(
            base_url=str(profile["base_url"]), api_key=self.vault.get(profile_id)
        )
        result = {
            "ok": str(profile["model_id"]) in discovered["models"],
            "provider": discovered["provider"],
            "model_id": profile["model_id"],
            "model_count": len(discovered["models"]),
        }
        return result

    # 功能：
    #   为存在的任务和已启用模型签发十分钟一次性授权，长期秘密仅保留在服务内部。
    # 输入：
    #   self：持有模型资料、凭证库及授权缓存的服务。
    #   profile_id：已保存的模型资料标识。
    #   thread_id：授权要绑定的现有任务。
    # 输出：
    #   result：短期授权令牌及到期时间。
    def issue_grant(self, profile_id: str, thread_id: str) -> dict[str, object]:
        with self._lock:
            self.store.get_thread(thread_id)
            profile = self.store.get_custom_model(profile_id)
            if not bool(profile["enabled"]):
                raise ValueError("CUSTOM_MODEL_DISABLED")
            if profile["api_style"] not in {"responses", "chat-completions"}:
                raise ValueError("CUSTOM_MODEL_API_STYLE_INVALID")
            grant = f"ddc_{secrets.token_urlsafe(32)}"
            expires_at = datetime.now(UTC) + timedelta(minutes=10)
            connection = ModelConnection(
                selection_id=self._selection_id(profile_id),
                provider="custom",
                model_id=str(profile["model_id"]),
                api_key=self.vault.get(profile_id),
                base_url=validate_custom_base_url(profile["base_url"]),
                api_style=str(profile["api_style"]),  # type: ignore[arg-type]
                capability_id="model.custom.openai-compatible",
                source="custom",
                supports_image_input=False,
            )
            # 和本服务删除路径共用锁，避免删除后重新发布含旧秘密的授权。
            self._grants = {
                token: record
                for token, record in self._grants.items()
                if record.expires_at > datetime.now(UTC)
            }
            self._grants[grant] = _Grant(connection, thread_id, expires_at)
        result = {"grant": grant, "expires_at": expires_at.isoformat()}
        return result

    # 功能：
    #   1. 一次性消费授权并核对期限、任务和模型范围，错误尝试同样使该令牌失效。
    #   2. 重新检查撤销、连接资料和秘密变化，禁止授权后仍调用旧模型或旧凭证。
    # 输入：
    #   self：模型服务及当前授权缓存。
    #   grant：前端提交的短期授权令牌。
    #   thread_id：实际准备执行的任务标识。
    #   selection_id：实际准备使用的模型选择标识。
    # 输出：
    #   connection：仅供内部调用器使用的含秘密连接对象。
    def consume_grant(self, grant: str, thread_id: str, selection_id: str) -> ModelConnection:
        with self._lock:
            record = self._grants.pop(grant, None)
            if record is None:
                raise ValueError("CUSTOM_MODEL_GRANT_INVALID")
            if record.expires_at <= datetime.now(UTC):
                raise ValueError("CUSTOM_MODEL_GRANT_EXPIRED")
            if record.thread_id != thread_id or record.connection.selection_id != selection_id:
                raise ValueError("CUSTOM_MODEL_GRANT_SCOPE_MISMATCH")
            try:
                self.store.get_thread(thread_id)
                profile = self.store.get_custom_model(selection_id.removeprefix("custom:"))
            except KeyError as error:
                raise ValueError("CUSTOM_MODEL_GRANT_INVALID") from error
            if not bool(profile["enabled"]):
                raise ValueError("CUSTOM_MODEL_DISABLED")
            connection = record.connection
            if any(
                profile[name] != getattr(connection, name)
                for name in ("base_url", "model_id", "api_style")
            ):
                raise ValueError("CUSTOM_MODEL_GRANT_STALE")
            try:
                secret = self.vault.get(str(profile["profile_id"]))
            except KeyError as error:
                raise ValueError("CUSTOM_MODEL_GRANT_STALE") from error
            if not isinstance(secret, str) or not secrets.compare_digest(
                secret.encode("utf-8"), connection.api_key.encode("utf-8")
            ):
                raise ValueError("CUSTOM_MODEL_GRANT_STALE")
            return connection
