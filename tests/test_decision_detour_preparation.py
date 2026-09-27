"""Detour materials preserve preassigned splits and exact simulated geometry."""

import json

import pytest

from dronedream_agent_core.decision_dataset import file_digest
from scripts.prepare_decision_detour_suite import prepare
from scripts.prepare_decision_native_scene import make_world
from scripts.prepare_decision_topology_suite import prepare_suite


# 功能：生成独立的合成测试套件；输入：临时目录；输出：无任何实际飞行或正式数据的材料。
@pytest.fixture
def suite(tmp_path):
    folder = tmp_path / "source"
    prepare_suite(
        {"required_clearance_m": 0.25, "asset_sha256": {}}, {}, folder, radius_m=0.38, height_m=0.43
    )
    return folder / "suite.json"


# 功能：逐份校验20种材料的真实碰撞世界、来源摘要及原集合；输出：正式样本始终0。
def test_prepared_detour_suite_is_not_formal_training(suite, tmp_path):
    output = tmp_path / "detours"
    result = prepare(suite, output, radius_m=0.38, height_m=0.43, clearance_m=0.25)
    source = json.loads(suite.read_text())
    assert result["materials"] == 20
    assert result["formal_training_additions"] == 0
    assert not result["gpu_data_ready"]
    for entry, parent in zip(result["layouts"], source["layouts"], strict=True):
        path = output / entry["receipt"]
        assert file_digest(path) == entry["receipt_sha256"]
        receipt = json.loads(path.read_text())
        assert receipt["split"] == parent["split"]
        assert receipt["parent_family"] == parent["family"]
        assert receipt["checks"]["initial_route_clearance_m"] <= 0
        assert receipt["checks"]["replacement_route_clearance_m"] >= 0.25
        for name, digest in receipt["assets"].items():
            assert file_digest(path.parent / name) == digest
        for name in ("before", "blocked"):
            semantic = json.loads((path.parent / (name + ".json")).read_text())
            world = (path.parent / (name + ".sdf")).read_bytes().decode()
            assert world == make_world(semantic["collision_primitives"])
    with pytest.raises(FileExistsError):
        prepare(suite, output, radius_m=0.38, height_m=0.43, clearance_m=0.25)


# 功能：采集前发现来源/划分修改时完全不发布新套件；输出：明确拒绝。
@pytest.mark.parametrize("fault", ["split", "family", "recipe"])
def test_source_mutation_rejected_before_output(suite, tmp_path, fault):
    doc = json.loads(suite.read_text())
    if fault in {"split", "family"}:
        doc["layouts"][0][fault] = "stress" if fault == "split" else "invented"
        suite.write_text(json.dumps(doc))
    else:
        path = suite.parent / doc["layouts"][0]["receipt"]
        receipt = json.loads(path.read_text())
        receipt["recipe"]["cell_m"] = 5.0
        path.write_text(json.dumps(receipt))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        prepare(suite, output, radius_m=0.38, height_m=0.43, clearance_m=0.25)
    assert not output.exists()
