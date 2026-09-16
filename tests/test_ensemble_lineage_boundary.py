"""Composition metadata is an object keyed by the complete current expert set."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from dronedream_agent_core.training.ensemble_lineage import load_base_lineage
from dronedream_agent_core.training.policy_packaging import REQUIRED_RECURRENT_ENSEMBLE_ROLES


# 功能：
#   验证完整专家列表必须为按角色索引的对象，不能以空值、布尔值或数组代替。
# 输入：
#   tmp_path：测试独立目录。
#   entries：错误的专家证据容器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("entries", [None, True, [], sorted(REQUIRED_RECURRENT_ENSEMBLE_ROLES)])
def test_base_lineage_rejects_non_object_expert_evidence(tmp_path, entries):
    base = SimpleNamespace(manifest_path=tmp_path / "manifest.json", package_sha256="a" * 64)
    (tmp_path / "assembly-receipt.json").write_text(json.dumps({
        "package_sha256": base.package_sha256, "expert_evidence": entries,
    }))
    with pytest.raises(ValueError, match="BASE_LINEAGE_MISMATCH"):
        load_base_lineage(base)


# 功能：
#   验证来源读取保留十角色映射，并返回实际读取字节的摘要。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_base_lineage_preserves_complete_mapping_and_content_identity(tmp_path):
    base = SimpleNamespace(manifest_path=tmp_path / "manifest.json", package_sha256="a" * 64)
    receipt = {"package_sha256": base.package_sha256,
               "expert_evidence": {role: {} for role in REQUIRED_RECURRENT_ENSEMBLE_ROLES}}
    content = json.dumps(receipt).encode("utf-8")
    (tmp_path / "assembly-receipt.json").write_bytes(content)
    actual, digest = load_base_lineage(base)
    assert actual == receipt
    assert digest == hashlib.sha256(content).hexdigest()
