"""Synthetic contract fixtures only; these tests never add formal flight data."""

import copy
import json

import pytest
from test_decision_assembly import collection
from test_decision_label_evidence import reference

from dronedream_agent_core.decision_dataset import audit_records, read_records
from dronedream_agent_core.decision_layout_lineage import collection_split, load_layout_families
from scripts.assemble_decision_corpus import assemble
from scripts.merge_decision_corpora import merge


# 功能：采集时预设集合必须保留，旧回合仅默认train；损坏元数据不能偷偷改成训练集。
# 输入：不同采集上下文；输出：原集合或明确错误，不执行飞行。
@pytest.mark.parametrize("split", ["train", "development", "calibration", "test", "stress"])
def test_collection_split_roundtrip(tmp_path, split):
    assert collection_split(tmp_path) == "train"
    (tmp_path / "decision-collection-context.json").write_text(json.dumps({
        "schema_version": "dronedream.decision-collection-context.v1", "split": split,
    }))
    assert collection_split(tmp_path) == split


# 功能：禁止损坏/未来格式被解释成旧版缺省；输入：非法上下文；输出：拒绝。
@pytest.mark.parametrize("doc", [{}, {"schema_version": "v9", "split": "train"},
    {"schema_version": "dronedream.decision-collection-context.v1", "split": "validation"}])
def test_bad_collection_split_is_not_defaulted(tmp_path, doc):
    (tmp_path / "decision-collection-context.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CONTEXT_INVALID"):
        collection_split(tmp_path)


# 功能：创建带原始证据的测试语料和对应布局收据；输出：仅用于接口测试的文件路径。
def source(root):
    root.mkdir()
    manifest, _ = collection(root)
    assemble(manifest, root / "assembled")
    corpus = root / "assembled/decisions.jsonl"
    rows = read_records(corpus)
    receipt = {
        "scene_family": "straight-corridor-width-variants-must-stay-in-same-split",
        "distinct_layout_claim": False,
        "semantic_sha256": rows[0]["state"]["map_sha256"],
    }
    reference(root, "scene-receipt.json", receipt)
    return corpus, root / "scene-receipt.json"


# 功能：合并后证书相对路径可独立重放；输入：合成来源；输出：一条且只认一个布局族。
def test_portable_merge_reaudits_and_preserves_identity(tmp_path):
    corpus, receipt = source(tmp_path / "source")
    output = tmp_path / "merged"
    result = merge([corpus], [receipt], output)
    assert result["formal_windows"] == 1
    assert result["verified_layout_families"] == 1
    assert result["unverified_layout_groups"] == 0
    assert not result["gpu_data_ready"]
    old, new = read_records(corpus)[0], read_records(output / "decisions.jsonl")[0]
    assert {k: v for k, v in old.items() if k != "evidence"} == {
        k: v for k, v in new.items() if k != "evidence"
    }
    assert audit_records([new], output)["formal_windows"] == 1


# 功能：重复来源不能增加数量；输入：两次同一源；输出：不发布看似可训练的语料。
def test_duplicate_merge_not_published(tmp_path):
    corpus, receipt = source(tmp_path / "source")
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="DUPLICATE_SAMPLE"):
        merge([corpus, corpus], [receipt, receipt], output)
    assert not (output / "decisions.jsonl").exists()


# 功能：缺失布局来源不取消物理证据，但不能借地图哈希数量宣称GPU数据齐备。
def test_missing_lineage_blocks_readiness_not_physics(tmp_path):
    corpus, _ = source(tmp_path / "source")
    result = audit_records(read_records(corpus), corpus.parent)
    assert result["formal_windows"] == 1
    assert result["verified_layout_families"] == 0
    assert "LAYOUT_LINEAGE_REQUIRED" in result["readiness_issues"]


# 功能：宽度/移动障碍变体即使地图哈希不同，也不能跨集合。
# 输入：不同父组和地图的同生成器族；输出：在后续标签核验前即拒绝泄漏。
def test_variant_cross_split_rejected(tmp_path):
    corpus, receipt = source(tmp_path / "source")
    output = tmp_path / "merged"
    merge([corpus], [receipt], output)
    rows = read_records(output / "decisions.jsonl")
    other = copy.deepcopy(rows[0])
    other.update(
        sample_id="second", parent_group="other-width", episode_id="other-episode", split="test"
    )
    other["state"]["map_sha256"] = "b" * 64
    manifest = json.loads((output / "layout-lineage.json").read_text("utf-8"))
    second = {
        "scene_family": "straight-corridor-crossing-variants-must-stay-in-same-split",
        "distinct_layout_claim": False,
        "static_map_sha256": "b" * 64,
    }
    manifest["entries"].append(
        {
            "parent_group": "other-width",
            "map_sha256": "b" * 64,
            "receipt": reference(output, "second-receipt.json", second),
        }
    )
    (output / "layout-lineage.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="LAYOUT_FAMILY_LEAKAGE"):
        audit_records(rows + [other], output)


# 功能：未知生成器、伪地图、重复条目与篡改收据不能声称独立布局。
@pytest.mark.parametrize("fault", ["generator", "map", "duplicate", "digest", "escape"])
def test_lineage_invalid_receipt(tmp_path, fault):
    corpus, receipt = source(tmp_path / "source")
    output = tmp_path / "merged"
    merge([corpus], [receipt], output)
    path = output / "layout-lineage.json"
    manifest = json.loads(path.read_text("utf-8"))
    entry = manifest["entries"][0]
    if fault == "map":
        entry["map_sha256"] = "b" * 64
    elif fault == "duplicate":
        manifest["entries"].append(copy.deepcopy(entry))
    elif fault == "digest":
        entry["receipt"]["sha256"] = "b" * 64
    elif fault == "escape":
        entry["receipt"]["path"] = "../scene-receipt.json"
    else:
        doc = json.loads((output / entry["receipt"]["path"]).read_text("utf-8"))
        doc["scene_family"] = "invented-independent-layout"
        entry["receipt"] = reference(output, "invented.json", doc)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_layout_families(output)
