"""Synthetic scene generation tests, never formal decision labels."""

import copy
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter

import pytest

from dronedream_agent_core.decision_dataset import decode_object, file_digest
from dronedream_agent_core.decision_scene_topology import (
    generate_edges,
    scene_documents,
    verified_scene_family,
)
from dronedream_plugin_sdk.protocol import encode_json
from scripts.prepare_decision_native_scene import make_world
from scripts.prepare_decision_topology_suite import prepare_suite


# 功能：固定可复现的合成场景配方；输入：种子/尺寸；输出：只用于测试的配方。
def recipe(seed=0, cell=4.0):
    return {"size": 4, "edges": generate_edges(4, seed), "cell_m": cell, "flight_z_m": 1.6}


# 功能：同形拓扑即使旋转或改宽度也保持同族；输入：已知树；输出：族身份一致。
def test_topology_identity_ignores_geometry_variants_and_numbering():
    original = recipe()
    _, _, family = scene_documents(original, {})
    assert scene_documents(recipe(cell=3.2), {})[2] == family
    rotated = copy.deepcopy(original)
    # 功能：将格点旋转90度；输入/输出：同一网格的房间编号。
    def rotate(n):
        return (n % 4) * 4 + 3 - n // 4
    rotated["edges"] = sorted(sorted([rotate(a), rotate(b)]) for a, b in original["edges"])
    assert scene_documents(rotated, {})[2] == family


# 功能：断开、重复边、非法坐标/非有限尺度不能产生地图；输出：明确拒绝。
@pytest.mark.parametrize("fault", ["duplicate", "disconnected", "long-edge", "size", "nan", "bool"])
def test_bad_topology_or_metric_rejected(fault):
    value = recipe()
    if fault == "duplicate":
        value["edges"][-1] = value["edges"][0]
    elif fault == "disconnected":
        value["edges"] = [[i, i + 1] for i in range(15)]
    elif fault == "long-edge":
        value["edges"][0] = [0, 15]
    elif fault == "size":
        value["size"] = 1000000
    elif fault == "nan":
        value["cell_m"] = float("nan")
    else:
        value["edges"][0][0] = True
    with pytest.raises(ValueError):
        scene_documents(value, {})


# 功能：收据重算完整地图而非相信自填family；输入：合法及篡改配方；输出：摘要绑定验证。
def test_receipt_binds_regenerated_map():
    value = recipe()
    semantic, _, family = scene_documents(value, {})
    digest = hashlib.sha256(encode_json(semantic).encode()).hexdigest()
    receipt = {
        "scene_family": "room-tree-generator-v1",
        "distinct_layout_claim": True,
        "recipe": value,
        "vehicle_clearance": {},
        "semantic_sha256": digest,
    }
    assert verified_scene_family(receipt, digest) == family
    receipt["recipe"]["cell_m"] = 4.1
    with pytest.raises(ValueError, match="MAP_MISMATCH"):
        verified_scene_family(receipt, digest)


# 功能：碰撞与视觉逐项使用同一几何；输出：每个障碍同时出现在SDF碰撞和视觉中。
def test_sdf_and_semantic_share_every_obstacle():
    semantic, _, _ = scene_documents(recipe(), {})
    root = ET.fromstring(make_world(semantic["collision_primitives"]))
    for kind in ("collision", "visual"):
        items = root.findall(f".//{kind}")
        assert len(items) == len(semantic["collision_primitives"])
        for primitive, element in zip(semantic["collision_primitives"], items, strict=True):
            assert list(map(float, element.findtext("geometry/box/size").split())) == [
                primitive["size_" + axis] for axis in "xyz"
            ]


# 功能：20个不同树形布局先验分组，静态验证不冒充飞行；输入：隔离临时目录；输出：清单。
def test_full_suite_is_independent_and_not_formal(tmp_path):
    base = {"asset_sha256": {}, "required_clearance_m": 0.25}
    report = prepare_suite(base, {}, tmp_path / "suite", radius_m=0.3, height_m=0.3)
    assert len({row["family"] for row in report["layouts"]}) == 20
    assert Counter(row["split"] for row in report["layouts"]) == {
        "train": 12,
        "development": 2,
        "calibration": 2,
        "test": 2,
        "stress": 2,
    }
    assert not report["gpu_data_ready"]
    assert report["formal_training_additions"] == report["physically_verified_layouts"] == 0
    for row in report["layouts"]:
        path = tmp_path / "suite" / row["config"]
        assert file_digest(path) == row["config_sha256"]
        config = decode_object(path.read_text("utf-8"))
        receipt = decode_object((tmp_path / "suite" / row["receipt"]).read_text("utf-8"))
        assert verified_scene_family(receipt, config["asset_sha256"]["semantic"]) == row["family"]
        assert row["static_route_clearance_m"] >= 0.25
    with pytest.raises(FileExistsError):
        prepare_suite(base, {}, tmp_path / "suite", radius_m=0.3, height_m=0.3)
