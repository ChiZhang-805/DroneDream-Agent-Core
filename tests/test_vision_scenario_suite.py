"""Procedural data is authored supervision, not fabricated flight evidence."""

import hashlib
import math
import xml.etree.ElementTree as ET

import pytest

from dronedream_agent_core.training.vision_render_world import index_static_visuals
from dronedream_agent_core.training.vision_scenario_suite import (
    SceneBuilder,
    build_scenario,
    look_at,
)


# 功能：
#   检查本轮全部独立布局的几何、完整标签、分组隔离和单位四元数。
# 输入：
#   无。
# 输出：
#   None：任何不合法布局使检查失败。
def test_full_planned_suite_is_deterministic_and_geometrically_distinct():
    world_hashes, group_splits = set(), {}
    assignments = ((1000, 20, "training"), (2000, 8, "validation"), (3000, 8, "test"))
    for offset, count, split in assignments:
        for index in range(1, count + 1):
            seed = offset + index
            world, labels, plan, receipt = build_scenario(seed, split)
            assert labels["world_sha256"] == hashlib.sha256(world).hexdigest()
            assert labels["world_sha256"] not in world_hashes
            world_hashes.add(labels["world_sha256"])
            xml = ET.fromstring(world)
            assert set(index_static_visuals(xml.find("world"))) == set(labels["visuals"])
            assert set(labels["visuals"].values()) == set(range(1, 8))
            assert not xml.findall(".//sensor") and not xml.findall(".//plugin")
            assert len(plan["views"]) == 128
            assert receipt["physical_flight_evidence"] is False
            poses = set()
            for view in plan["views"]:
                assert view["split"] == split
                group = view["scene_group_id"]
                assert group_splits.setdefault(group, split) == split
                assert math.isclose(sum(v * v for v in view["orientation_wxyz"]), 1, abs_tol=1e-12)
                pose = tuple(view["position_m"])
                assert pose not in poses
                poses.add(pose)
            assert build_scenario(seed, split) == (world, labels, plan, receipt)


# 功能：
#   用真实 XYZ 前向约定验证上仰和左右转向，防止训练视角被四元数符号反转。
# 输入：
#   target：希望相机前向轴指向的三维方向。
# 输出：
#   None：旋转后的前向单位向量必须匹配目标方向。
@pytest.mark.parametrize("target", [(1, 0, 0), (0, 1, 0), (1, 0, 1), (-1, -1, -1)])
def test_look_at_matches_gazebo_forward_axis(target):
    w, x, y, z = look_at([0, 0, 0], target, 0.1)
    forward = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    norm = math.sqrt(sum(v * v for v in target))
    assert forward == pytest.approx([v / norm for v in target], abs=1e-12)


# 功能：
#   验证相机实体间隙及严格的输入预算，不允许使用盒内或地面以下的光学中心。
# 输入：
#   无。
# 输出：
#   None：边界拒绝规则必须生效。
def test_camera_exclusion_and_request_bounds():
    builder = SceneBuilder(1)
    builder.box("wall", 2, [1, 2, 1], [2, 2, 2])
    assert not builder.camera_clear([1, 2, 1])
    assert not builder.camera_clear([2.1, 2, 1])
    assert builder.camera_clear([2.2, 2, 1])
    assert not builder.camera_clear([5, 5, -1])
    for args in ((True, "training", 16), (1, "unknown", 16), (1, "test", 129)):
        with pytest.raises(ValueError):
            build_scenario(*args)
    with pytest.raises(ValueError):
        look_at([0, 0, 0], [0, 0, 0])


# 功能：
#   对生成的每个视角重建实体体积，核验拒绝穿墙的规则确实用于实际扫描计划。
# 输入：
#   无。
# 输出：
#   None：所有视角具有至少 15 厘米的相机几何间距。
def test_generated_views_remain_outside_solid_geometry():
    world, _, plan, _ = build_scenario(1001, "training")
    xml = ET.fromstring(world)
    volumes = []
    for model in xml.findall("world/model"):
        position = list(map(float, model.findtext("pose").split()))[:3]
        size_node = model.find("link/visual/geometry/box/size")
        if size_node is not None:
            volumes.append((position, list(map(float, size_node.text.split()))))
    for view in plan["views"]:
        assert all(not all(abs(p - c) <= s / 2 + 0.15 for p, c, s in
                           zip(view["position_m"], center, size, strict=True))
                   for center, size in volumes)
