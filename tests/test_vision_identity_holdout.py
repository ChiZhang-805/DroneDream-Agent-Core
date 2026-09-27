"""Identity holdouts remain unseen and source-bound instead of being training duplicates."""

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dronedream_agent_core.training.vision_scenario_suite import build_scenario


# 功能：
#   隔离导入明确的身份留出工具，禁止执行命令行采集入口。
# 输入：
#   monkeypatch：恢复脚本搜索路径的测试夹具。
# 输出：
#   module：可直接测试的身份计划模块。
def load_tool(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("identity_holdout_test",
                                                scripts / "prepare_vision_identity_holdout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：
#   核查仅人物网格被替换，几何、标签表和插件状态均不被隐式改写。
# 输入：
#   monkeypatch：隔离脚本导入。
# 输出：
#   None：两个人形必须完整替换，其他内容与固定场景一致。
def test_identity_replacement_changes_only_two_mesh_paths(monkeypatch):
    module = load_tool(monkeypatch)
    world, _, _, _ = build_scenario(4001, "test", 16)
    result = module.replace_person_mesh(world)
    old, new = ET.fromstring(world), ET.fromstring(result)
    assert not new.findall(".//plugin")
    assert len(new.findall(".//mesh/uri")) == 2
    for uri in new.findall(".//mesh/uri"):
        assert uri.text == "person/meshes/casual_female.dae"
        uri.text = "person/meshes/standing.dae"
    assert ET.tostring(new) == ET.tostring(old)
    with pytest.raises(ValueError, match="PERSON_COUNT_CHANGED"):
        module.replace_person_mesh(b"<sdf><world/></sdf>")
    with pytest.raises(ValueError, match="UNEXPECTED_MESH"):
        module.replace_person_mesh(result)


# 功能：
#   用明确标记的最小资源夹具验证 512 个计划全属测试，损坏资源不能发布新计划。
# 输入：
#   tmp_path、monkeypatch：独立文件夹及仅测试使用的网格摘要替代。
# 输出：
#   None：正式固定摘要不被写入假资源，生产文件和既有计划不被修改。
def test_holdout_plan_stays_test_only_and_rejects_resource_changes(tmp_path, monkeypatch):
    module = load_tool(monkeypatch)
    source = tmp_path / "source"
    mesh = source / "meshes/casual_female.dae"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"identity test fixture, not a renderable human")
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "IDENTITY_MESH_SHA256", digest)
    metadata = {"version": 4, "intended_split": "test-only",
        "metadata_url": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Casual%20female",
        "license_name": "Creative Commons Zero v1.0 Universal",
        "files_sha256": {"meshes/casual_female.dae": digest}}
    (source / "source.json").write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "holdout"
    receipt = module.prepare(source, output)
    assert receipt["planned_native_frames"] == 512 and receipt["rendered_frames"] == 0
    assert receipt["identity_held_out_from_training_and_validation"] is True
    for entry in receipt["layouts"]:
        root = output / entry["directory"]
        plan = json.loads((root / "views.json").read_bytes())
        assert all(view["split"] == "test" for view in plan["views"])
        assert entry["person_identity_training_allowed"] is False
        actual_world_hash = hashlib.sha256((root / "world.sdf").read_bytes()).hexdigest()
        assert entry["world_sha256"] == actual_world_hash
    with pytest.raises(FileExistsError):
        module.prepare(source, output)
    mesh.write_bytes(b"changed after metadata was saved")
    with pytest.raises(ValueError, match="RESOURCE_CHANGED"):
        module.prepare(source, tmp_path / "must-not-be-published")
    assert not (tmp_path / "must-not-be-published").exists()
