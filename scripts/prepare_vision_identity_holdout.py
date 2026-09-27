"""Freeze an unseen human identity and independent geometry exclusively in the test split."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_vision_scenario_suite import write_new

from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    portable_plugin_path,
    read_plugin_file,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_scenario_suite import build_scenario
from dronedream_plugin_sdk.protocol import decode_json

IDENTITY_MESH_SHA256 = "b0f47228b4a1cfa5026f24f5955d5f2b7cc7f7a770888951041a787b3e51d994"


# 功能：
#   只替换两个已标注站立人物的网格，不改变相机、标签编号或加入可执行插件。
# 输入：
#   world：明确场景生成器产生的世界字节。
# 输出：
#   result：具有第二个人物身份的静态世界字节。
def replace_person_mesh(world):
    root = ET.fromstring(world)
    replacements = 0
    for uri in root.findall(".//visual/geometry/mesh/uri"):
        if uri.text != "person/meshes/standing.dae":
            raise ValueError("VISION_IDENTITY_UNEXPECTED_MESH")
        uri.text = "person/meshes/casual_female.dae"
        replacements += 1
    if replacements != 2:
        raise ValueError("VISION_IDENTITY_PERSON_COUNT_CHANGED")
    result = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return result


# 功能：
#   固定四个新几何布局和未参加训练的人物身份，保留全部原始资源及许可摘要。
# 输入：
#   person_source、destination：已下载的明确 CC0 人物源和新的留出场景目录。
# 输出：
#   receipt：仅测试用途的 512 个计划视角，不是采集或模型能力回执。
def prepare(person_source, destination):
    metadata = read_plugin_file(person_source / "source.json", limit=64 * 1024)
    attribution = decode_json(metadata)
    if (type(attribution) is not dict or attribution.get("version") != 4
            or attribution.get("intended_split") != "test-only"
            or attribution.get("metadata_url") !=
            "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Casual%20female"
            or attribution.get("license_name") != "Creative Commons Zero v1.0 Universal"):
        raise ValueError("VISION_IDENTITY_SOURCE_INVALID")
    hashes = attribution.get("files_sha256")
    if (type(hashes) is not dict or not 1 <= len(hashes) <= 32
            or hashes.get("meshes/casual_female.dae") != IDENTITY_MESH_SHA256):
        raise ValueError("VISION_IDENTITY_FILES_INVALID")
    files, total = {}, 0
    for relative, digest in hashes.items():
        path = portable_plugin_path(relative)
        raw = read_plugin_file(person_source / path, limit=32 * 1024**2)
        total += len(raw)
        if total > 64 * 1024**2 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("VISION_IDENTITY_RESOURCE_CHANGED_OR_OVERSIZED")
        files[relative] = raw
    if "meshes/casual_female.dae" not in files:
        raise ValueError("VISION_IDENTITY_MESH_MISSING")
    destination = destination.absolute()
    check_plain_plugin_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for seed in range(4001, 4005):
        world, labels, plan, layout = build_scenario(seed, "test", 16)
        world = replace_person_mesh(world)
        world_hash = hashlib.sha256(world).hexdigest()
        labels["world_sha256"] = layout["world_sha256"] = world_hash
        layout.update({"held_out_person_identity": True,
                       "person_asset_sha256": hashes["meshes/casual_female.dae"],
                       "person_identity_training_allowed": False})
        root = destination / f"layout-{seed}"
        root.mkdir()
        written = {"world.sdf": write_new(root / "world.sdf", world)}
        written["person-source.json"] = write_new(root / "person-source.json", metadata)
        for relative, raw in files.items():
            target = root / "person" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            written["person/" + relative] = write_new(target, raw)
        for name, payload in (("labels.json", labels), ("views.json", plan)):
            publish_runtime_json(root / name, payload, replace_existing=False)
            written[name] = hashlib.sha256(read_plugin_file(root / name, limit=1024**2)).hexdigest()
        layout["files_sha256"] = written
        publish_runtime_json(root / "layout-receipt.json", layout, replace_existing=False)
        entries.append({"directory": root.name, **layout})
    receipt = {"schema": "dronedream.vision-suite-plan.v1", "complete": True,
               "planned_native_frames": sum(entry["views"] for entry in entries),
               "layouts": entries, "rendered_frames": 0, "ready_for_training": False,
               "identity_held_out_from_training_and_validation": True}
    publish_runtime_json(destination / "suite-plan.json", receipt, replace_existing=False)
    return receipt


# 功能：
#   发布固定身份留出计划，不启动 GPU 租用、训练或实际采集。
# 输入：
#   命令行参数：明确人物来源和新输出目录。
# 输出：
#   exit_code：来源和完整测试计划保存成功为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.person_source, args.output)
    print(json.dumps({"planned_views": receipt["planned_native_frames"], "split": "test"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
