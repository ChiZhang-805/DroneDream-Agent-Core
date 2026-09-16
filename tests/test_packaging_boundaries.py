"""Offline package boundaries; synthetic receipts never grant flight qualification."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_causal_packaging import (
    base_package,  # noqa: F401
    complete_current_base,
    replacements,
    training_receipts,
)
from test_complete_artifact_assembly import recipe  # noqa: F401

from dronedream_agent_core.training import artifact_assembly as assembly
from dronedream_agent_core.training.ensemble_lineage import load_base_lineage
from dronedream_agent_core.training.policy_packaging import (
    REQUIRED_RECURRENT_ENSEMBLE_ROLES,
    assemble_causal_package,
    validate_causal_base,
)


# 功能：
#   验证小文件不会按模型最大预算预分配读取缓冲，实际读取量至多为文件大小加一。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：临时替换文件打开入口的夹具。
# 输出：
#   None：不返回业务数据。
def test_small_source_read_uses_actual_size(tmp_path, monkeypatch):
    path = tmp_path / "small.bin"
    data = b"model"
    path.write_bytes(data)
    original_open = Path.open
    reads = []

    # 功能：
    #   包装真实句柄以记录读取预算，同时保留文件描述符和正常关闭行为。
    # 输入：
    #   target：待打开的文件路径。
    #   args：原打开调用的位置参数。
    #   kwargs：原打开调用的关键字参数。
    # 输出：
    #   proxy：委托真实文件操作的上下文管理器。
    def tracked_open(target, *args, **kwargs):
        stream = original_open(target, *args, **kwargs)
        proxy = MagicMock(wraps=stream)
        proxy.__enter__.return_value = proxy
        proxy.__exit__.side_effect = stream.__exit__
        reads.append(proxy.read)
        return proxy

    monkeypatch.setattr(Path, "open", tracked_open)
    assert assembly.read_bound_content(
        path, hashlib.sha256(data).hexdigest(), maximum_bytes=1024**3
    ) == data
    assert reads[0].call_args.args[0] <= len(data) + 1


# 功能：
#   验证打开前被替换成相同字节的文件也会因身份改变而拒绝，不仅核对摘要。
# 输入：
#   tmp_path：测试独立目录。
#   monkeypatch：临时插入替换操作的夹具。
# 输出：
#   None：不返回业务数据。
def test_same_content_replacement_before_open_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "source.bin"
    data = b"unchanged bytes"
    path.write_bytes(data)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(data)
    original_open = Path.open

    # 功能：
    #   仅对本测试源路径注入一次替换，再打开新文件以模拟检查与打开之间的竞争。
    # 输入：
    #   target：待打开路径。
    #   args：打开模式等位置参数。
    #   kwargs：打开调用的关键字参数。
    # 输出：
    #   stream：替换后的真实文件句柄。
    def replaced_open(target, *args, **kwargs):
        if target == path and replacement.exists():
            replacement.replace(path)
        stream = original_open(target, *args, **kwargs)
        return stream

    monkeypatch.setattr(Path, "open", replaced_open)
    with pytest.raises(ValueError, match="CHANGED"):
        assembly.read_bound_content(path, hashlib.sha256(data).hexdigest(), maximum_bytes=100)


# 功能：
#   验证有正确摘要的回执也不能包含重复关键字段或非有限数值。
# 输入：
#   recipe：十角色合成打包配方。
#   tmp_path：测试独立目录。
#   invalid：需要注入的 JSON 歧义类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", ["duplicate", "nan", "infinity"])
def test_assembly_rejects_ambiguous_receipt_bytes(recipe, tmp_path, invalid):  # noqa: F811
    source = next(s for s in recipe.sources if s.role == "local-navigation-policy")
    original = source.training_receipt_path.read_text(encoding="utf-8")
    prefix = {
        "duplicate": '"qualified_for_flight":true,',
        "nan": '"untrusted_metric":NaN,',
        "infinity": '"untrusted_metric":1e999,',
    }[invalid]
    content = ("{" + prefix + original[1:]).encode("utf-8")
    source.training_receipt_path.write_bytes(content)
    source.training_receipt_sha256 = hashlib.sha256(content).hexdigest()
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        assembly.assemble_complete_ensemble(recipe=recipe, source_root=tmp_path, output_root=output)
    assert not output.exists()


# 功能：
#   验证直接传入内存对象的回执也必须满足有限 JSON 和失败列表结构。
# 输入：
#   recipe：十角色合成配方。
#   field：回执中需要破坏的字段。
#   value：非法字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("untrusted_metric", float("nan")), ("untrusted_metric", float("inf")),
    ("issue_codes", None), ("issue_codes", 0), ("issue_codes", ""),
])
def test_memory_receipt_cannot_bypass_json_boundary(recipe, field, value):  # noqa: F811
    source = next(s for s in recipe.sources if s.role == "local-navigation-policy")
    artifact = next(a for a in recipe.manifest.artifacts if a.role == source.role)
    receipt_value = json.loads(source.training_receipt_path.read_bytes())
    receipt_value[field] = value
    with pytest.raises(ValueError):
        assembly.validate_expert_training_receipt(
            source.role, artifact.sha256, receipt_value, recipe.manifest
        )


# 功能：
#   验证基座来源回执拒绝重复键，不能用最后的匹配摘要掩盖前面的矛盾身份。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_base_lineage_rejects_duplicate_package_identity(tmp_path):
    lineage = {
        "package_sha256": "a" * 64,
        "expert_evidence": {role: {} for role in REQUIRED_RECURRENT_ENSEMBLE_ROLES},
    }
    content = '{"package_sha256":"wrong",' + json.dumps(lineage)[1:]
    (tmp_path / "assembly-receipt.json").write_text(content, encoding="utf-8")
    base = SimpleNamespace(manifest_path=tmp_path / "manifest.json", package_sha256="a" * 64)
    with pytest.raises(ValueError):
        load_base_lineage(base)


# 功能：
#   验证视觉特征数只接受正整数，不把布尔值和相等的浮点值当成维度契约。
# 输入：
#   base_package：合成十角色基座包。
#   invalid：非法视觉维度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", [True, 1.0])
def test_visual_feature_count_requires_integer(base_package, invalid):  # noqa: F811
    with pytest.raises(ValueError, match="VISUAL"):
        validate_causal_base(base_package, visual_feature_count=invalid)


# 功能：
#   验证替换期间调用方改写回执不会改变已经开始组装的证据快照。
# 输入：
#   base_package：合成十角色基座。
#   tmp_path：测试独立目录。
#   monkeypatch：在读取模型之后模拟调用方修改的夹具。
# 输出：
#   None：不返回业务数据。
def test_causal_replacement_owns_receipt_snapshot(base_package, tmp_path, monkeypatch):  # noqa: F811
    base = complete_current_base(base_package, tmp_path)
    paths = replacements(tmp_path)
    receipts = training_receipts(paths, base)
    read = assembly.read_bound_content

    # 功能：
    #   在捕获第一份模型字节后改写外部回执，检验组装器是否仍引用调用方字典。
    # 输入：
    #   path：当前源文件。
    #   expected：预期内容摘要。
    #   maximum_bytes：读取字节预算。
    # 输出：
    #   content：原读取器完成校验的字节。
    def mutate_after_read(path, expected, *, maximum_bytes):
        content = read(path, expected, maximum_bytes=maximum_bytes)
        receipts["local-navigation-policy"]["qualified_for_flight"] = True
        return content

    monkeypatch.setattr(assembly, "read_bound_content", mutate_after_read)
    output = tmp_path / "owned-receipt"
    assemble_causal_package(base_root=base, navigation_models=paths, navigation_receipts=receipts,
                           output_root=output, package_id="test.owned", history_length=4)
    saved = json.loads((output / "training-evidence/local-navigation-policy.json").read_bytes())
    assert saved["qualified_for_flight"] is False
    assert receipts["local-navigation-policy"]["qualified_for_flight"] is True


# 功能：
#   验证实际发布瞬间出现的既有目录不会被覆盖，空目录也享有相同保护。
# 输入：
#   recipe：十角色合成配方。
#   tmp_path：测试独立目录。
#   monkeypatch：在发布入口注入竞争的夹具。
#   populated：竞争目录内是否包含需保留的文件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("populated", [False, True])
def test_publication_preserves_late_destination(
    recipe, tmp_path, monkeypatch, populated  # noqa: F811 - imported pytest fixture
):
    publish = assembly.publish_asset_directory
    output = tmp_path / "late-destination"
    identities = []

    # 功能：
    #   在图和回执验证完毕之后占用最终目录，交由实际无覆盖发布器处理冲突。
    # 输入：
    #   source：已经验证的暂存目录。
    #   destination：即将发布的最终路径。
    # 输出：
    #   None：不返回业务数据。
    def concurrent_publish(source, destination):
        assert (source / "assembly-receipt.json").is_file()
        destination.mkdir()
        identities.append(destination.stat())
        if populated:
            (destination / "preserved.txt").write_text("keep", encoding="utf-8")
        publish(source, destination)

    monkeypatch.setattr(assembly, "publish_asset_directory", concurrent_publish)
    with pytest.raises(FileExistsError):
        assembly.assemble_complete_ensemble(recipe=recipe, source_root=tmp_path, output_root=output)
    assert os.path.samestat(identities[0], output.stat())
    assert sorted(p.name for p in output.iterdir()) == (["preserved.txt"] if populated else [])
    if populated:
        assert (output / "preserved.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".complete-ensemble-*"))


# 功能：
#   验证字节预算边界、内容不符和非法摘要均明确拒绝，恰好达到预算的合法文件可读。
# 输入：
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_bound_source_checks_digest_and_exact_budget(tmp_path):
    path = tmp_path / "bounded.bin"
    content = b"12345678"
    digest = hashlib.sha256(content).hexdigest()
    path.write_bytes(content)
    assert assembly.read_bound_content(path, digest, maximum_bytes=8) == content
    with pytest.raises(ValueError, match="LARGE"):
        assembly.read_bound_content(path, digest, maximum_bytes=7)
    with pytest.raises(ValueError, match="CONTENT_CHANGED"):
        assembly.read_bound_content(path, "0" * 64, maximum_bytes=8)
    with pytest.raises(ValueError, match="DIGEST_INVALID"):
        assembly.read_bound_content(tmp_path / "not-opened", "wrong", maximum_bytes=8)
