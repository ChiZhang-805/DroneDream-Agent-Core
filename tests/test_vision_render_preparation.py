"""Camera-only render contracts; fixtures are not a product training corpus."""

import hashlib
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dronedream_agent_core.local_vision_training import LOCAL_VISION_SEMANTIC_CLASSES
from dronedream_agent_core.training.vision_render_world import build_labelled_render_world

WORLD = b'''<sdf version="1.9"><world name="fixture">
<model name="wall"><static>true</static><link name="link"><visual name="visual">
<geometry><box><size>1 1 1</size></box></geometry>
</visual></link></model></world></sdf>'''
CAMERA = b'''<sdf version="1.9"><model name="camera"><link name="link">
<sensor name="forward" type="camera"><camera><horizontal_fov>1.2</horizontal_fov>
<image><width>64</width><height>64</height><format>R8G8B8</format></image>
<clip><near>0.1</near><far>30</far></clip></camera></sensor></link></model></sdf>'''


# 功能：
#   为已知测试几何生成逐可见面的显式类别表，不推测真实学校地图语义。
# 输入：
#   无。
# 输出：
#   labels：与测试 SDF 完整摘要绑定的类别定义。
def label_fixture():
    labels = {"schema": "dronedream.vision-world-labels.v1",
              "world_sha256": hashlib.sha256(WORLD).hexdigest(),
              "classes": list(LOCAL_VISION_SEMANTIC_CLASSES),
              "visuals": {"wall::link::visual": 2}}
    return labels


# 功能：
#   核对派生世界保留原图、同位同内参相机及类别，且没有飞机或控制插件。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_labelled_render_world_preserves_camera_alignment():
    derived, record = build_labelled_render_world(WORLD, label_fixture(), CAMERA)
    tree = ET.fromstring(derived)
    rgb, semantic = tree.findall(".//sensor")
    assert rgb.get("type") == "camera"
    assert semantic.get("type") == "segmentation"
    assert rgb.findtext("pose") == semantic.findtext("pose") == "0 0 0 0 0 0"
    assert ET.tostring(rgb.find("camera/image")) == ET.tostring(semantic.find("camera/image"))
    assert tree.findtext(".//visual/plugin/label") == "2"
    assert b"PX4" not in derived and b"motor" not in derived
    assert record["world_sha256"] == hashlib.sha256(WORLD).hexdigest()
    assert record["physical_flight_evidence"] is False


# 功能：
#   缺标签、地图版本错误或非法类别不能悄悄变成背景类别。
# 输入：
#   bad：待注入的问题种类。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["missing", "world", "class", "extra"])
def test_unreviewed_label_assignment_is_rejected(bad):
    labels = label_fixture()
    if bad == "missing":
        labels["visuals"].clear()
    elif bad == "world":
        labels["world_sha256"] = "0" * 64
    elif bad == "class":
        labels["visuals"]["wall::link::visual"] = True
    else:
        labels["visuals"]["unknown"] = 3
    with pytest.raises(ValueError):
        build_labelled_render_world(WORLD, labels, CAMERA)


# 功能：
#   视角计划中的同一场景组不能通过更换视角标识混入不同划分。
# 输入：
#   monkeypatch：按真实模块导入注册类型解析命名空间，结束后恢复。
# 输出：
#   None：不返回业务数据。
def test_render_plan_requires_group_disjointness(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/capture_labelled_vision_views.py"
    spec = importlib.util.spec_from_file_location("render_plan_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    view = {"view_id": "first", "scene_group_id": "corridor", "split": "training",
            "position_m": [0.0, 0.0, 1.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    payload = {"schema": "dronedream.vision-view-plan.v1", "views": [view]}
    assert len(module.checked_views(payload)) == 1
    payload["views"].append({**view, "view_id": "second", "split": "validation"})
    with pytest.raises(ValueError, match="SPATIAL_GROUP_LEAKAGE"):
        module.checked_views(payload)
