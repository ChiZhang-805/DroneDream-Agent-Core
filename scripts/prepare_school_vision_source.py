"""Bind authored School Map semantics to render labels without fabricating people."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dronedream_agent_core.asset_package_storage import extract_verified_asset
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.local_vision_training import LOCAL_VISION_SEMANTIC_CLASSES
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_render_world import index_static_visuals
from dronedream_agent_core.xml_values import parse_xml
from dronedream_plugin_sdk.protocol import decode_json

# 门框、环框表示入口结构而不是可通行体积；任何类别均不能单独放行控制动作。
# 只接受已逐项核对的作者语义，不将未知标签默认当作地面或背景。
SEMANTIC_CLASSES = {
    "terrain": 1, "floor": 1, "launch-pad": 1,
    "window-glazing": 6, "pickup-pad": 7,
    "door-frame": 3, "door-header": 3, "door-threshold": 3, "training-gate-ring": 3,
    "stair-tread": 4, "stair-landing": 4, "entrance-step": 4,
    "furniture-support": 2, "chair": 2, "window-frame": 2, "window-mullion": 2,
    "fence-rail": 2, "fence-post": 2, "tree-crown": 2, "interior-wall": 2,
    "student-desktop": 2, "training-gate-ring-collision": 2, "exterior-wall": 2,
    "stair-handrail-post": 2, "tree-trunk": 2, "cafeteria-table": 2,
    "street-light-base": 2, "street-light-pole": 2, "street-light-arm": 2,
    "street-light-lamp": 2, "facade-structure": 2, "bookshelf": 2,
    "closed-door-leaf": 2, "stair-handrail": 2, "canopy-column": 2,
    "blackboard": 2, "teacher-desk": 2, "podium": 2,
    "training-gate-support": 2, "training-gate-base": 2,
    "conservative-bicycle-and-rack-envelope": 2, "canopy": 2, "office-desk": 2,
    "conservative-bike-rack-envelope": 2, "roof": 2, "open-door-leaf": 2,
    "plant-pot": 2, "plant-crown": 2, "service-counter": 2, "gate-post": 2,
    "pickup-shelf": 2, "guard-booth": 2, "gate-header": 2,
}


# 功能：
#   将作者的逐物体语义严格对应到实际 SDF 可见面，拒绝缺失、重复及未知类别。
# 输入：
#   world_bytes、semantics：已从同一校验资产包取得的世界字节与语义对象。
# 输出：
#   labels、counts：完整可见面标签表及对象类别统计，不是像素覆盖或飞行资格。
def authored_labels(world_bytes, semantics):
    root = parse_xml(world_bytes, maximum_bytes=32 * 1024**2)
    worlds = root.findall("world")
    if len(worlds) != 1 or type(semantics) is not dict:
        raise ValueError("VISION_AUTHORED_WORLD_INVALID")
    by_name = {}
    for key in ("collision_primitives", "visual_only_primitives"):
        values = semantics.get(key)
        if type(values) is not list or len(values) > 20_000:
            raise ValueError("VISION_AUTHORED_PRIMITIVES_INVALID")
        for value in values:
            if type(value) is not dict or type(value.get("name")) is not str:
                raise ValueError("VISION_AUTHORED_PRIMITIVE_INVALID")
            if value["name"] in by_name:
                raise ValueError("VISION_AUTHORED_NAME_DUPLICATE")
            semantic = value.get("semantic")
            if type(semantic) is not str or semantic not in SEMANTIC_CLASSES:
                raise ValueError("VISION_AUTHORED_SEMANTIC_NOT_REVIEWED")
            by_name[value["name"]] = SEMANTIC_CLASSES[semantic]
    assignments = {}
    for key, visual in index_static_visuals(worlds[0]).items():
        name = visual.get("name", "")
        if not name.endswith("-visual") or name[:-7] not in by_name:
            raise ValueError("VISION_AUTHORED_VISUAL_UNBOUND")
        assignments[key] = by_name[name[:-7]]
    if not assignments:
        raise ValueError("VISION_AUTHORED_VISUALS_EMPTY")
    labels = {"schema": "dronedream.vision-world-labels.v1",
              "world_sha256": hashlib.sha256(world_bytes).hexdigest(),
              "classes": list(LOCAL_VISION_SEMANTIC_CLASSES), "visuals": assignments}
    counts = Counter(assignments.values())
    return labels, counts


# 功能：
#   校验并隔离解包明确选择的地图，生成标签和来源回执，不修改软件安装资产。
# 输入：
#   命令行参数：当前地图 DDPKG 和尚不存在的准备目录。
# 输出：
#   exit_code：准备成功返回零；缺少行人等类别仍在回执中列明。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspected = inspect_ddpkg(args.package)
    if inspected.manifest.asset_id != "dronedream.school-map.v1":
        raise ValueError("VISION_SCHOOL_SOURCE_WRONG_ASSET")
    output = args.output.absolute()
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "asset").mkdir()
    extract_verified_asset(args.package, output / "asset", inspected)
    source = output / "asset/normalized/map/gazebo"
    world_bytes = read_plugin_file(source / "world.sdf", limit=32 * 1024**2)
    semantic_bytes = read_plugin_file(source / "semantic.json", limit=8 * 1024**2)
    semantics = decode_json(semantic_bytes, limit=8 * 1024**2, node_limit=2_000_000)
    labels, counts = authored_labels(world_bytes, semantics)
    publish_runtime_json(output / "labels.json", labels, replace_existing=False)
    receipt = {"schema": "dronedream.authored-vision-source.v1",
               "asset_content_sha256": inspected.manifest.content_sha256,
               "world_sha256": labels["world_sha256"],
               "semantic_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
               "labelled_visual_count": sum(counts.values()),
               "class_object_counts": {name: counts[index]
                                       for index, name in enumerate(LOCAL_VISION_SEMANTIC_CLASSES)},
               "missing_object_classes": [name for index, name in
                   enumerate(LOCAL_VISION_SEMANTIC_CLASSES) if not counts[index]],
               "background_may_be_rendered_without_geometry": True,
               "dynamic_people_not_spawned": True, "ready_for_training": False,
               "flight_qualification_granted": False}
    publish_runtime_json(output / "source-receipt.json", receipt, replace_existing=False)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
