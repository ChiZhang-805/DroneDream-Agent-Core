"""Merge and validate mission-domain action packs without weakening core boundaries."""

from __future__ import annotations

from typing import Any

import jsonschema

from dronedream_plugin_sdk.protocol import copy_json, validate_local_schema

from .contracts import ActionDefinition, DomainActionCatalog
from .hashing import sha256_json


class DomainActionError(RuntimeError):
    """Reject inconsistent action catalogs before mission planning uses them."""


# 功能：
#   校验并合并领域动作声明，拒绝同名不同义、跨领域错配及需要远程读取的 Schema。
#   目录描述可计划的动作，不授予飞控执行权限。
# 输入：
#   values：插件返回的领域动作声明包。
# 输出：
#   catalog：与声明内容绑定、按稳定顺序排列的独立动作目录。
def merge_action_packs(values: list[Any]) -> DomainActionCatalog:
    actions: dict[str, ActionDefinition] = {}
    domains: set[str] = set()
    for raw_pack in values:
        if not isinstance(raw_pack, dict):
            raise DomainActionError("DOMAIN_ACTION_PACK_INVALID")
        try:
            raw_pack = copy_json(raw_pack)
        except ValueError as error:
            raise DomainActionError("DOMAIN_ACTION_PACK_INVALID") from error
        domain_id = raw_pack.get("domain_id")
        raw_actions = raw_pack.get("actions")
        if not isinstance(domain_id, str) or not isinstance(raw_actions, list):
            raise DomainActionError("DOMAIN_ACTION_PACK_INVALID")
        domains.add(domain_id)
        for raw_action in raw_actions:
            action = ActionDefinition.model_validate(raw_action)
            try:
                action.input_schema = validate_local_schema(action.input_schema)
            except (jsonschema.SchemaError, ValueError) as error:
                raise DomainActionError(
                    f"DOMAIN_ACTION_INPUT_SCHEMA_INVALID:{action.action_id}"
                ) from error
            if action.domain_id != domain_id:
                raise DomainActionError(f"DOMAIN_ACTION_PACK_DOMAIN_MISMATCH:{action.action_id}")
            previous = actions.get(action.action_id)
            # Identical shared declarations are harmless; silently overwriting
            # a different action would change what an approved plan means.
            if previous is not None and previous != action:
                raise DomainActionError(f"DOMAIN_ACTION_CONFLICT:{action.action_id}")
            actions[action.action_id] = action
    if "takeoff" not in actions or "land" not in actions:
        raise DomainActionError("DOMAIN_ACTION_CORE_BOUNDARY_MISSING")
    # Catalog identity must not depend on filesystem/import order.
    ordered = [actions[action_id] for action_id in sorted(actions)]
    catalog_payload = {
        "domains": sorted(domains),
        "actions": [item.model_dump(mode="json") for item in ordered],
    }
    catalog = DomainActionCatalog(
        catalog_id=f"actions.{sha256_json(catalog_payload)[:24]}",
        domain_ids=sorted(domains),
        actions=ordered,
    )
    return catalog


# 功能：
#   提取已声明动作名称用于任务合法性判断，不生成命令或扩展权限。
# 输入：
#   catalog：当前领域动作目录。
# 输出：
#   identifiers：不重复的动作标识集合。
def action_ids(catalog: DomainActionCatalog) -> set[str]:
    identifiers = {item.action_id for item in catalog.actions}
    return identifiers


# 功能：
#   选出需要独立运动验证和控制链路的动作，避免普通业务动作替代飞行控制。
# 输入：
#   catalog：当前领域动作目录。
# 输出：
#   identifiers：声明为移动动作的标识集合。
def movement_action_ids(catalog: DomainActionCatalog) -> set[str]:
    identifiers = {item.action_id for item in catalog.actions if item.movement}
    return identifiers


# 功能：
#   精确查询已声明动作，缺失时拒绝，不从名称猜测或发明替代动作。
# 输入：
#   catalog：当前领域动作目录。
#   action_id：待查询的动作标识。
# 输出：
#   action：目录中的动作声明，调用方只读使用。
def action_by_id(catalog: DomainActionCatalog, action_id: str) -> ActionDefinition:
    for action in catalog.actions:
        if action.action_id == action_id:
            return action
    raise DomainActionError(f"DOMAIN_ACTION_UNKNOWN:{action_id}")
