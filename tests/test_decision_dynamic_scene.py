"""Physical scene lineage tests, not flight qualification."""

import json
import xml.etree.ElementTree as ET

import pytest

from dronedream_agent_core.decision_dataset import file_digest
from dronedream_agent_core.simulation_dynamic_obstacles import load_dynamic_geometry
from scripts.prepare_decision_dynamic_scene import prepare_dynamic
from scripts.prepare_decision_native_scene import prepare


# 功能：核对两种横穿真实碰撞与渲染一致且运动参数不泄露给静态地图；输入：独立目录；输出：断言。
@pytest.mark.parametrize("clockwise", [True, False])
def test_crossing_world_is_physical_and_label_separated(tmp_path, clockwise):
    base = prepare({"asset_sha256": {}}, {}, tmp_path / "base", 3.0)
    unrelated = tmp_path / "base" / "native-run"
    unrelated.mkdir()
    (unrelated / "private-session.json").write_text('{"private":true}')
    output = tmp_path / "dynamic"
    result = prepare_dynamic(base, output, clockwise=clockwise)
    assert result["semantic"] == base["semantic"]
    assert result["asset_sha256"]["semantic"] == base["asset_sha256"]["semantic"]
    assert result["asset_sha256"]["world_sdf"] == file_digest(output / "world.sdf")
    assert "motion" not in json.dumps(result)
    root = ET.parse(output / "world.sdf")
    model = root.find(".//model[@name='dronedream_dynamic_decision-crossing-box']")
    collision = model.findtext("link/collision/geometry/box/size")
    assert collision == model.findtext("link/visual/geometry/box/size")
    assert model.findtext("static") == "false"
    assert model.findtext("link/gravity") == "false"
    assert float(model.findtext("plugin/initial_angular").split()[-1]) == (
        -0.5 if clockwise else 0.5
    )
    assert load_dynamic_geometry(output / "world.sdf")
    receipt = json.loads((output / "scene-receipt.json").read_text())
    assert receipt["scenario_not_a_behavior_label"] and not receipt["distinct_layout_claim"]
    assert receipt["formal_training_additions"] == 0
    assert not (output / "native-run").exists()
    assert (
        json.loads((output / "physical-world-receipt.json").read_text())["relative_resource_count"]
        == 0
    )


# 功能：源世界被改动或目录被占用时禁止生成掩盖版本差异的场景；输入：合成源；输出：明确拒绝。
def test_changed_source_and_existing_destination_are_rejected(tmp_path):
    base = prepare({"asset_sha256": {}}, {}, tmp_path / "base", 3.0)
    with pytest.raises(FileExistsError):
        prepare_dynamic(base, tmp_path)
    base["asset_sha256"]["world_sdf"] = "a" * 64
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        prepare_dynamic(base, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


# 功能：较大轨道提供真实离开通道的机会，但不预先声明恢复成功。
# 输入：两种方向和米制半径；输出：物理参数一致且部署地图不含未来轨迹。
@pytest.mark.parametrize("clockwise", [True, False])
def test_wider_crossing_orbit_keeps_source_separation(tmp_path, clockwise):
    base = prepare({"asset_sha256": {}}, {}, tmp_path / "base", 3.0)
    output = tmp_path / "wide"
    result = prepare_dynamic(base, output, clockwise=clockwise, radius_m=1.1, forward_m=3.0)
    model = ET.parse(output / "world.sdf").find(
        ".//model[@name='dronedream_dynamic_decision-crossing-box']"
    )
    assert float(model.findtext("pose").split()[1]) == (1.1 if clockwise else -1.1)
    assert float(model.findtext("pose").split()[0]) == 3.0
    assert float(model.findtext("plugin/initial_angular").split()[-1]) == pytest.approx(
        (-1 if clockwise else 1) * 0.3 / 1.1
    )
    assert result["semantic"] == base["semantic"]
    assert "radius_m" not in json.dumps(result)


# 功能：拒绝非有限半径和会使旋转方盒撞墙的轨道；输入：无效配置；输出：不生成目录。
@pytest.mark.parametrize("radius", [True, float("nan"), float("inf"), 0.1, 1.6, 1.2])
def test_invalid_or_wall_intersecting_orbits_rejected(tmp_path, radius):
    base = prepare({"asset_sha256": {}}, {}, tmp_path / "base", 3.0)
    with pytest.raises(ValueError, match="DECISION_DYNAMIC_(RADIUS_INVALID|ORBIT_DOES_NOT_FIT)"):
        prepare_dynamic(base, tmp_path / "invalid", radius_m=radius)
    assert not (tmp_path / "invalid").exists()


# 功能：前移障碍仍须位于物理边界内，非法距离不能产生场景；输入：边界/类型；输出：拒绝。
@pytest.mark.parametrize("forward", [True, float("nan"), 1.9, 5.1])
def test_invalid_forward_position(tmp_path, forward):
    base = prepare({"asset_sha256": {}}, {}, tmp_path / "base", 3.0)
    with pytest.raises(ValueError, match="DECISION_DYNAMIC_FORWARD_INVALID"):
        prepare_dynamic(base, tmp_path / "invalid", forward_m=forward)
    assert not (tmp_path / "invalid").exists()
