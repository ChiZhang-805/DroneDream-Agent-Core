from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.build_dynamic_obstacle_world import build_world
from scripts.build_dynamic_obstacle_world import _challenge_model, _motion
from scripts.run_school_map_depth_qualification import (
    _load_dynamic_obstacle_challenge,
)


# 功能：
#   确保异常挑战参数在写入场景或复制资源前被拒绝，不把字符串或布尔值当作物理量。
# 输入：
#   tmp_path：隔离目录。
#   field、value：待验证字段和非法值。
# 输出：
#   无：断言配置被拒绝且没有生成场景、回执或资源。
@pytest.mark.parametrize("field,value", [
    ("center_enu_m", [True, 0, 0]),
    ("center_enu_m", ["1", 0, 0]),
    ("center_enu_m", [float("nan"), 0, 0]),
    ("size_m", [0.2, False, 0.8]),
    ("color_rgba", [True, 0, 0, 1]),
    ("color_rgba", [0, 0, 0, "1"]),
    ("required_encounter_distance_m", True),
    ("required_encounter_distance_m", "3"),
    ("required_encounter_distance_m", float("inf")),
    ("required_encounter_distance_m", 10 ** 400),
])
def test_invalid_challenge_numbers_do_not_create_outputs(tmp_path: Path, field: str, value: object) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    _spec(spec)
    content = json.loads(spec.read_text())
    target = content if field == "required_encounter_distance_m" else content["obstacles"][0]
    target[field] = value
    spec.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        build_world(base_world=base, challenge_spec=spec, output_world=output, receipt_output=receipt)
    assert not output.parent.exists()


# 功能：
#   阻止回执覆盖场景，并拒绝非对象配置，保留全部源输入。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   无：断言异常输入不产生输出。
def test_rejects_output_alias_and_non_object_spec(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    _base_world(base)
    _spec(spec)
    with pytest.raises(ValueError, match="outputs must differ"):
        build_world(base_world=base, challenge_spec=spec, output_world=output, receipt_output=output)
    spec.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        build_world(base_world=base, challenge_spec=spec, output_world=output, receipt_output=output.with_suffix(".json"))
    assert not output.parent.exists()


# 功能：
#   拒绝源目录递归复制和输出覆盖资源，防止生成损坏但带有效回执的世界。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   无：断言冲突在文件复制之前被拒绝。
def test_rejects_resource_tree_and_output_collisions(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "challenge.json"
    _base_world(base)
    _spec(spec)
    nested = base.parent / "nested" / "world.sdf"
    with pytest.raises(ValueError, match="outside the base"):
        build_world(base_world=base, challenge_spec=spec, output_world=nested, receipt_output=tmp_path / "receipt.json")
    assert not nested.parent.exists()
    output = tmp_path / "derived" / "world.sdf"
    receipt = output.parent / "meshes" / "gate.obj"
    with pytest.raises(ValueError, match="overlap copied resources"):
        build_world(base_world=base, challenge_spec=spec, output_world=output, receipt_output=receipt)
    assert not output.parent.exists()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_world(path: Path) -> None:
    meshes = path.parent / "meshes"
    meshes.mkdir(parents=True)
    (meshes / "gate.obj").write_text("o gate\n", encoding="utf-8")
    textures = path.parent / "materials" / "textures"
    textures.mkdir(parents=True)
    (textures / "surface.ppm").write_text("P3\n1 1\n255\n255 255 255\n", encoding="utf-8")
    path.write_text(
        '<sdf version="1.9"><world name="school_map_world">'
        '<model name="school_map"><static>true</static><link name="map">'
        '<visual name="gate"><geometry><mesh><uri>meshes/gate.obj</uri>'
        "</mesh></geometry></visual></link></model>"
        "</world></sdf>",
        encoding="utf-8",
    )


def _spec(path: Path, *, obstacle_id: str = "corridor-cart") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.dynamic-obstacle-challenge.v1",
                "challenge_id": "corridor-recovery",
                "world_name": "school_map_world",
                "required_encounter_distance_m": 3.0,
                "obstacles": [
                    {
                        "obstacle_id": obstacle_id,
                        "center_enu_m": [-20.0, 9.85, 9.0],
                        "size_m": [0.2, 0.2, 0.8],
                        "color_rgba": [0.95, 0.32, 0.12, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_builds_content_bound_physical_dynamic_obstacle(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "derived" / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt_path = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    spec.parent.mkdir()
    _spec(spec)

    receipt = build_world(
        base_world=base,
        challenge_spec=spec,
        output_world=output,
        receipt_output=receipt_path,
    )

    model = ET.parse(output).getroot().find(
        "./world/model[@name='dronedream_dynamic_corridor-cart']"
    )
    assert model is not None
    assert model.findtext("static") == "false"
    assert model.findtext("./link/gravity") == "false"
    assert model.findtext("./link/kinematic") == "true"
    assert model.findtext("./link/collision/geometry/box/size") == "0.2 0.2 0.8"
    assert model.findtext("./link/visual/geometry/box/size") == "0.2 0.2 0.8"
    assert receipt["output_world_sha256"] == _sha256(output)
    assert receipt["qualification_granted"] is False
    assert receipt["required_encounter_distance_m"] == 3.0
    assert receipt["obstacles"][0]["published_as_dynamic_obstacle"] is True
    assert receipt["obstacles"][0]["gazebo_static"] is False
    assert receipt["relative_resource_count"] == 2
    assert (tmp_path / "derived" / "meshes" / "gate.obj").is_file()
    assert (tmp_path / "derived" / "materials" / "textures" / "surface.ppm").is_file()
    assert json.loads(receipt_path.read_text())["challenge_id"] == "corridor-recovery"


def test_rejects_duplicate_output_and_invalid_identity(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "derived" / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    spec.parent.mkdir()
    _spec(spec, obstacle_id="INVALID ID")

    with pytest.raises(ValueError, match="obstacle_id"):
        build_world(
            base_world=base,
            challenge_spec=spec,
            output_world=output,
            receipt_output=receipt,
        )

    _spec(spec)
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_world(
            base_world=base,
            challenge_spec=spec,
            output_world=output,
            receipt_output=receipt,
        )


# 功能：
#   核对挑战回执绑定实际世界摘要，不把旧真值控制历史作为原生感知证明。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：无业务数据。
def test_challenge_receipt_is_bound_to_world(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "derived" / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt_path = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    spec.parent.mkdir()
    _spec(spec)
    build_world(
        base_world=base,
        challenge_spec=spec,
        output_world=output,
        receipt_output=receipt_path,
    )

    challenge = _load_dynamic_obstacle_challenge(receipt_path, world_sdf=output)
    assert challenge["world_sha256"] == _sha256(output)
    assert challenge["entity_names"] == ["dronedream_dynamic_corridor-cart"]

    output.write_text(output.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt is invalid"):
        _load_dynamic_obstacle_challenge(receipt_path, world_sdf=output)


# 功能：
#   回执必须使用有限距离数值且保留唯一 JSON 字段，不能把布尔值或字符串作为米数。
# 输入：
#   tmp_path：隔离路径。
#   fault：接纳边界的异常类型。
# 输出：
#   None：无业务数据。
@pytest.mark.parametrize("fault", [True, "3", None, 10 ** 400, "duplicate", "array"])
def test_challenge_receipt_rejects_ambiguous_inputs(tmp_path, fault):
    world = tmp_path / "world.sdf"
    world.write_text("<sdf/>")
    receipt = tmp_path / "receipt.json"
    payload = {"schema_version": "dronedream.dynamic-obstacle-world-receipt.v1",
        "intended_use": "simulation-recovery-challenge", "qualification_granted": False,
        "output_world_sha256": _sha256(world), "required_encounter_distance_m": fault,
        "obstacles": [{"entity_name": "dronedream_dynamic_box", "physical_collision": True,
            "visible_to_camera": True, "published_as_dynamic_obstacle": True}]}
    text = json.dumps(payload)
    if fault == "array":
        text = "[]"
    if fault == "duplicate":
        text = text[:-1] + ',"required_encounter_distance_m":3}'
    receipt.write_text(text)
    with pytest.raises(ValueError):
        _load_dynamic_obstacle_challenge(receipt, world_sdf=world)


# 功能：
#   圆周挑战保留物理碰撞并声明实际插件和速度，不把静止或瞬移当作运动配置。
# 输入：
#   clockwise：绕行方向。
# 输出：
#   None：无业务数据。
@pytest.mark.parametrize("clockwise", [False, True])
def test_circular_obstacle_uses_bounded_velocity_plugin(clockwise):
    model, receipt = _challenge_model({"obstacle_id": "moving-box", "center_enu_m": [0, 0, 5],
        "size_m": [.6, .6, 1.], "motion": {"kind": "circle", "radius_m": .8, "speed_mps": .35, "clockwise": clockwise}})
    assert model.findtext("./link/kinematic") == "false"
    assert model.find("./link/collision") is not None
    assert model.findtext("./link/inertial/mass") == "1"
    plugin = model.find("plugin")
    assert plugin.attrib["filename"] == "gz-sim-velocity-control-system"
    assert plugin.findtext("initial_linear") == "0.35 0 0"
    assert float(plugin.findtext("initial_angular").split()[-1]) == pytest.approx((-.35 if clockwise else .35) / .8)
    assert not receipt["stationary"] and receipt["motion"]["kind"] == "circle"


# 功能：
#   拒绝含糊字段、非有限值、隐式转换及过快运动，避免生成不可用的恢复标签。
# 输入：
#   update：注入的非法运动参数。
# 输出：
#   None：无业务数据。
@pytest.mark.parametrize("update", [{"radius_m": True}, {"speed_mps": "0.3"}, {"clockwise": 1},
    {"radius_m": float("nan")}, {"speed_mps": 2.}, {"kind": "teleport"}, {"extra": 0},
    {"radius_m": .5, "speed_mps": 1.}])
def test_motion_contract_rejects_invalid_parameters(update):
    with pytest.raises(ValueError, match="motion"):
        _motion({"kind": "circle", "radius_m": .8, "speed_mps": .35, "clockwise": False, **update})
