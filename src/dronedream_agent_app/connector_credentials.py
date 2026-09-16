"""Opaque, plugin-scoped connector credentials backed by the OS user vault."""

from __future__ import annotations

import re
from collections.abc import Iterable
from contextlib import suppress
from uuid import uuid4

from dronedream_agent_core.capability_broker import CapabilityBrokerError, CredentialResolver

from .custom_models import CredentialVault, WindowsCredentialVault, rollback_credential
from .storage import AppStore

_PLUGIN_ID = re.compile(r"^[a-z][a-z0-9._-]{2,119}$")


class ConnectorCredentialService(CredentialResolver):
    """Own connector secrets while exposing only revocable opaque references."""

    # 功能：
    #   分离连接器非秘密资料与用户凭证库，测试可显式注入内存实现。
    # 输入：
    #   self：待初始化的连接器凭证服务。
    #   store：连接器资料与已安装插件存储。
    #   vault：可选秘密存储，缺省使用 Windows 当前用户 DPAPI。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, store: AppStore, vault: CredentialVault | None = None) -> None:
        self.store = store
        self.vault = (
            vault
            if vault is not None
            else WindowsCredentialVault(store.root / "credentials" / "connectors")
        )

    # 功能：
    #   列出不透明凭证引用与插件范围，不读取任何秘密。
    # 输入：
    #   self：连接器凭证服务。
    # 输出：
    #   records：可公开的凭证元数据列表。
    def list(self) -> list[dict[str, object]]:
        records = self.store.list_connector_credentials()
        return records

    # 功能：
    #   1. 有界验证已安装插件范围，先存秘密，再发布非秘密引用与显示名称。
    #   2. 保存失败时补偿删除秘密，回收再失败保留原异常并附固定说明。
    # 输入：
    #   self：连接器凭证服务。
    #   display_name：用户可见的凭证名称。
    #   secret：保留原文的连接器秘密。
    #   allowed_plugin_ids：最多 32 项的插件标识输入，验证后去重。
    # 输出：
    #   record：已保存的不透明引用、名称和允许插件范围。
    def create(
        self,
        *,
        display_name: str,
        secret: str,
        allowed_plugin_ids: Iterable[str],
    ) -> dict[str, object]:
        if not isinstance(display_name, str):
            raise ValueError("CONNECTOR_CREDENTIAL_NAME_INVALID")
        name = display_name.strip()
        if not name or len(name) > 80:
            raise ValueError("CONNECTOR_CREDENTIAL_NAME_INVALID")
        if not isinstance(secret, str) or not secret or len(secret) > 16_384:
            raise ValueError("CONNECTOR_CREDENTIAL_SECRET_INVALID")
        if isinstance(allowed_plugin_ids, (str, bytes, dict)):
            raise ValueError("CONNECTOR_CREDENTIAL_SCOPE_INVALID")
        plugin_ids = []
        try:
            # 在去重之前限制消费数量，重复值或无限迭代器也不能绕过输入预算。
            for index, plugin_id in enumerate(allowed_plugin_ids):
                if (
                    index >= 32
                    or not isinstance(plugin_id, str)
                    or _PLUGIN_ID.fullmatch(plugin_id) is None
                ):
                    raise ValueError("CONNECTOR_CREDENTIAL_SCOPE_INVALID")
                plugin_ids.append(plugin_id)
        except TypeError as error:
            raise ValueError("CONNECTOR_CREDENTIAL_SCOPE_INVALID") from error
        if not plugin_ids:
            raise ValueError("CONNECTOR_CREDENTIAL_SCOPE_INVALID")
        plugin_ids = sorted(set(plugin_ids))
        installed = {str(item["plugin_id"]) for item in self.store.list_plugins()}
        if any(plugin_id not in installed for plugin_id in plugin_ids):
            raise ValueError("CONNECTOR_CREDENTIAL_PLUGIN_NOT_INSTALLED")
        reference = f"cred-{uuid4().hex[:24]}"
        try:
            self.vault.put(reference, secret)
            record = self.store.save_connector_credential(
                {
                    "reference": reference,
                    "display_name": name,
                    "allowed_plugin_ids": plugin_ids,
                }
            )
        except BaseException as error:
            rollback_credential(self.vault, reference, error)
            raise
        return record

    # 功能：
    #   1. 先撤销元数据再删除秘密，即使后者失败也不能继续通过引用取得秘密。
    #   2. 已撤销的引用仍可重复删除，以便恢复上一次失败的密文回收。
    # 输入：
    #   self：连接器凭证服务。
    #   reference：需要撤销的不透明引用。
    # 输出：
    #   None：不返回业务数据。
    def delete(self, reference: str) -> None:
        with suppress(KeyError):
            self.store.delete_connector_credential(reference)
        self.vault.delete(reference)

    # 功能：
    #   根据当前保存的插件范围读取秘密，已撤销或范围不符的引用统一拒绝；不信任调用方缓存。
    # 输入：
    #   self：连接器凭证服务。
    #   reference：代理请求解析的不透明引用。
    #   plugin_id：已经由能力代理确定的插件身份。
    # 输出：
    #   secret：仅供代理调用目标连接器的秘密。
    def resolve(self, reference: str, *, plugin_id: str) -> str:
        try:
            metadata = self.store.get_connector_credential(reference)
        except KeyError as error:
            raise CapabilityBrokerError("BROKER_CREDENTIAL_UNAVAILABLE") from error
        allowed = metadata.get("allowed_plugin_ids", [])
        if not isinstance(allowed, list) or plugin_id not in allowed:
            raise CapabilityBrokerError("BROKER_CREDENTIAL_SCOPE_DENIED")
        try:
            secret = self.vault.get(reference)
        except KeyError as error:
            raise CapabilityBrokerError("BROKER_CREDENTIAL_UNAVAILABLE") from error
        return secret
