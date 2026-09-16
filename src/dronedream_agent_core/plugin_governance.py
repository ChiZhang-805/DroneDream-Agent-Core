"""Deterministic enterprise policy evaluation for plugin lifecycle mutations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from .hashing import sha256_json
from .plugin_contracts import (
    PluginGovernanceDecision,
    PluginGovernanceOperation,
    PluginGovernancePolicy,
    PluginManifest,
)


# 功能：
#   1. 重新校验并隔离可变策略和清单，再评估标识、发布者、权限、更新通道与信任限制。
#   2. 区分仅导入检查和实际执行，撤销优先于允许名单；空允许名单表示不限制该维度。
#   3. 用本次实际参与判断的策略和清单生成摘要，决策本身不执行或启用插件。
# 输入：
#   policy：待应用的治理策略。
#   manifest：待评估的插件清单。
#   operation：计划执行的生命周期操作。
#   trust_status：调用方已核验的包信任状态。
#   installed_external_plugins：不含本次目标的已安装外部插件数量。
# 输出：
#   decision：包含接受结果、拒绝原因和输入摘要的治理决策。
def evaluate_plugin_governance(
    *,
    policy: PluginGovernancePolicy,
    manifest: PluginManifest,
    operation: PluginGovernanceOperation,
    trust_status: str,
    installed_external_plugins: int,
) -> PluginGovernanceDecision:
    if type(installed_external_plugins) is not int or installed_external_plugins < 0:
        raise ValueError("PLUGIN_GOVERNANCE_COUNT_INVALID")
    # Pydantic 实例和嵌套列表仍可修改，不能把曾经通过构造校验当成本次调用的保证。
    policy = PluginGovernancePolicy.model_validate(policy.model_dump(mode="python"), strict=True)
    manifest = PluginManifest.model_validate(manifest.model_dump(mode="python"), strict=True)
    issue_codes: list[str] = []
    if policy.allowed_plugin_ids and manifest.plugin_id not in policy.allowed_plugin_ids:
        issue_codes.append("GOVERNANCE_PLUGIN_NOT_ALLOWED")
    if policy.allowed_publishers and manifest.publisher not in policy.allowed_publishers:
        issue_codes.append("GOVERNANCE_PUBLISHER_NOT_ALLOWED")
    denied = sorted(set(manifest.permissions).intersection(policy.denied_permissions))
    issue_codes.extend(f"GOVERNANCE_PERMISSION_DENIED:{permission}" for permission in denied)
    if manifest.provenance.update_ring not in policy.allowed_update_rings:
        issue_codes.append("GOVERNANCE_UPDATE_RING_DENIED")
    if operation == "import" and installed_external_plugins >= policy.maximum_external_plugins:
        issue_codes.append("GOVERNANCE_EXTERNAL_PLUGIN_LIMIT")
    # 导入可保留不可信包供检查；启用和提升执行更严格的签名规则，导入不代表执行许可。
    if (
        operation in {"enable", "promote"}
        and policy.require_verified_signatures
        and trust_status != "verified"
    ):
        issue_codes.append("GOVERNANCE_VERIFIED_SIGNATURE_REQUIRED")
    if operation == "trust-local-package" and not policy.allow_local_approval:
        issue_codes.append("GOVERNANCE_LOCAL_APPROVAL_DISABLED")
    # 本地批准和宽松允许列表均不能覆盖明确撤销。
    if trust_status == "revoked":
        issue_codes.append("GOVERNANCE_PACKAGE_REVOKED")
    decision = PluginGovernanceDecision(
        decision_id=f"plugin-policy-{uuid4().hex[:24]}",
        policy_id=policy.policy_id,
        operation=operation,
        plugin_id=manifest.plugin_id,
        version=manifest.version,
        accepted=not issue_codes,
        issue_codes=issue_codes,
        policy_sha256=sha256_json(policy.model_dump(mode="json")),
        manifest_sha256=sha256_json(manifest.model_dump(mode="json")),
        trust_status=trust_status,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
    )
    return decision
