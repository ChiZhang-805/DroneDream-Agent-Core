"""Native scene assets must use the product map contract, not an experiment dialect."""

import json
import xml.etree.ElementTree as ET

import pytest

from dronedream_agent_core.runtime_bindings import load_map_runtime_bindings
from scripts.prepare_decision_native_scene import prepare


# 功能：核验同源真实几何和地图接口；输入：临时实验目录；输出：逐盒体相等断言。
def test_scene_runtime_bindings_and_render_collision_identity(tmp_path):
    output = tmp_path / "scene"
    config = prepare({"asset_sha256": {}}, {}, output, 3.0)
    semantic = json.loads((output / "semantic.json").read_text("utf-8"))
    bindings = load_map_runtime_bindings(semantic)
    assert bindings.mission_launch_waypoint.z == 1.6
    root = ET.parse(output / "world.sdf")
    primitives = semantic["runtime_collision_primitives"]
    assert len(root.findall(".//collision")) == len(primitives) == 8
    assert len(root.findall(".//visual")) == len(primitives)
    for primitive in primitives:
        for kind in ("collision", "visual"):
            item = root.find(f".//{kind}[@name='{primitive['name']}-{kind}']")
            assert list(map(float, item.findtext("pose").split()))[:3] == [
                primitive["center_" + axis] for axis in "xyz"
            ]
            assert list(map(float, item.findtext("geometry/box/size").split())) == [
                primitive["size_" + axis] for axis in "xyz"
            ]
    assert config["batch_static_world_visuals"] is False
    assert config["expert_role"] == "local-navigation-policy"
    assert json.loads((output / "scene-receipt.json").read_text())["formal_training_additions"] == 0


# 功能：禁止错误宽度和复用已有证据目录；输入：非法参数；输出：明确拒绝。
def test_scene_rejects_invalid_width_and_existing_output(tmp_path):
    for width in (0, 6, float("nan"), True):
        with pytest.raises(ValueError):
            prepare({}, {}, tmp_path / "invalid", width)
    with pytest.raises(FileExistsError):
        prepare({}, {}, tmp_path, 3.0)


# 功能：定位结构只能来自场景真实实物；输入：带梁实验；输出：渲染/碰撞/标签三者同源。
def test_beams_are_real_geometry_not_a_localization_override(tmp_path):
    prepare({"asset_sha256": {}}, {}, tmp_path / "beams", 3.0, ceiling_beams=True)
    semantic = json.loads((tmp_path / "beams/semantic.json").read_text("utf-8"))
    beams = [p for p in semantic["collision_primitives"] if p["name"].startswith("ceiling-beam-")]
    assert len(beams) == 3
    root = ET.parse(tmp_path / "beams/world.sdf")
    for beam in beams:
        assert beam["center_z"] - beam["size_z"] / 2 == 3.0
        for kind in ("collision", "visual"):
            assert root.find(f".//{kind}[@name='{beam['name']}-{kind}']") is not None
    receipt = json.loads((tmp_path / "beams/scene-receipt.json").read_text())
    assert not receipt["qualification_granted"] and not receipt["distinct_layout_claim"]
