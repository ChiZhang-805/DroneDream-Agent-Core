"""Sample feasible native camera views around the current authored School Map graph."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from dronedream_agent_core.collision_batch import static_point_clearances
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_render_world import bind_render_resources
from dronedream_agent_core.training.vision_scenario_suite import look_at
from dronedream_plugin_sdk.protocol import decode_json

SCHOOL_WORLD_SHA256 = "ef9d4e976dbd2b128f80aa2ff11f41a1cfffb8ccb58e763d206d9f9163ce7e8c"


# 功能：
#   从显式图节点及边生成均匀距离采样，不把图内旧资格字符串当作当前飞行验收结果。
# 输入：
#   graph、spacing_m：来源导航图和采样间距。
# 输出：
#   positions、directions：每个采样点的米制坐标及对应边前向方向。
def route_candidates(graph, spacing_m):
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if (type(nodes) is not list or type(edges) is not list
            or not 1 <= len(nodes) <= 2000 or not 1 <= len(edges) <= 4000
            or type(spacing_m) not in (int, float) or not 0.3 <= spacing_m <= 3):
        raise ValueError("SCHOOL_VIEW_GRAPH_INVALID")
    indexed = {}
    for node in nodes:
        key = node["node_id"]
        point = [node["position_m"][axis] for axis in "xyz"]
        if key in indexed or not all(type(v) in (int, float) and math.isfinite(v) for v in point):
            raise ValueError("SCHOOL_VIEW_NODE_INVALID")
        indexed[key] = point
    positions, directions, seen = [], [], set()
    for edge in edges:
        start, end = indexed[edge["from_node"]], indexed[edge["to_node"]]
        distance = math.dist(start, end)
        if not 0.05 <= distance <= 200:
            raise ValueError("SCHOOL_VIEW_EDGE_INVALID")
        direction = [(b - a) / distance for a, b in zip(start, end, strict=True)]
        count = math.ceil(distance / spacing_m)
        for step in range(count):
            point = [a + (b - a) * step / count for a, b in zip(start, end, strict=True)]
            key = tuple(round(v, 3) for v in point)
            if key not in seen:
                seen.add(key)
                positions.append(point)
                directions.append(direction)
            if len(positions) > 6000:
                raise ValueError("SCHOOL_VIEW_POINT_BUDGET")
    return positions, directions


# 功能：
#   根据当前建筑的实体包围区域确定观察点室内外属性，边界附近保持未知而非猜测。
# 输入：
#   point：地图 ENU 米制坐标。
# 输出：
#   setting：明确室内、明确室外或边界未知。
def school_setting(point):
    x, y, z = point
    buildings = [(-53, 3, 2, 24, 10.8), (13, 47, 7.5, 32.5, 7.2)]
    for x0, x1, y0, y1, top in buildings:
        if x0 + 0.3 < x < x1 - 0.3 and y0 + 0.3 < y < y1 - 0.3 and 0.3 < z < top:
            return "indoor"
        if x0 - 0.3 < x < x1 + 0.3 and y0 - 0.3 < y < y1 + 0.3 and z < top + 0.3:
            return "unknown"
    return "outdoor"


# 功能：
#   以微小位置扰动和双向航向采样航线周边，复用产品几何检查排除不可达的拍摄位置。
# 输入：
#   graph、spacing_m、primitives：当前地图导航图、距离间隔及完整检查几何。
# 输出：
#   views、summary：可采视角和因实体间隙不足被拒绝的数量。
def make_school_views(graph, spacing_m, primitives):
    positions, directions = route_candidates(graph, spacing_m)
    rng, candidates, descriptions = random.Random(9172026), [], []
    for point_index, (point, direction) in enumerate(zip(positions, directions, strict=True)):
        heading = math.atan2(direction[1], direction[0])
        for angle_index, offset in enumerate((0, -0.45, 0.45, math.pi, math.pi - 0.45,
                                               math.pi + 0.45, -1.2, 1.2)):
            camera = [point[0] + rng.uniform(-0.12, 0.12), point[1] + rng.uniform(-0.12, 0.12),
                      point[2] + rng.uniform(-0.15, 0.18)]
            yaw, dz = heading + offset + rng.uniform(-0.08, 0.08), rng.uniform(-0.45, 0.3)
            aim = [camera[0] + math.cos(yaw), camera[1] + math.sin(yaw), camera[2] + dz]
            corruption = {}
            mode = (point_index * 8 + angle_index) % 32
            if mode == 1:
                corruption = {"exposure_multiplier": 0.06}
            elif mode == 2:
                corruption = {"exposure_multiplier": 3.8}
            elif mode in (3, 4):
                corruption = {"blur_sigma_px": rng.uniform(1.2, 3.5)}
            elif mode in (5, 6):
                left, top = rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.4)
                corruption = {"occlusion_rect": [left, top, left + 0.25, top + 0.25]}
            candidates.append(camera)
            descriptions.append({"view_id": f"route-{point_index:04d}-{angle_index}",
                "scene_group_id": "school-map-whole-layout-v1", "split": "training",
                "position_m": camera,
                "orientation_wxyz": look_at(camera, aim, rng.uniform(-0.08, 0.08)),
                "setting": school_setting(camera), "corruption": corruption})
    # 此处使用真实可见几何而非简化运行碰撞包；玻璃等视觉实体也不能被相机穿透。
    clearances, _ = static_point_clearances(candidates, primitives,
                                           radius_m=0.25, half_height_m=0.18)
    views = [view for view, clearance in zip(descriptions, clearances, strict=True)
             if clearance >= 0.05]
    if not views:
        raise ValueError("SCHOOL_VIEW_NO_CLEAR_SAMPLES")
    summary = {"route_points": len(positions), "candidate_views": len(candidates),
               "accepted_views": len(views),
               "rejected_geometry_views": len(candidates) - len(views),
               "minimum_checked_clearance_m": float(min(v for v in clearances if v >= 0.05)),
               "spacing_m": spacing_m, "physical_flight_evidence": False}
    return views, summary


# 功能：
#   为显式 OBJ 网格计算保守外包围盒，环框内部也保守排除，不静默忽略不支持的几何。
# 输入：
#   semantics、resources：作者几何表及摘要绑定的资源字节。
# 输出：
#   primitives：可由共享几何检查器处理的全部实体与网格外包围盒。
def checked_camera_geometry(semantics, resources):
    primitives = []
    for source in semantics["collision_primitives"] + semantics["visual_only_primitives"]:
        if "uri" not in source:
            primitives.append(source)
            continue
        if (not source["uri"].endswith(".obj")
                or any(abs(source.get(key, 0)) > 1e-12 for key in ("roll_rad", "pitch_rad"))):
            raise ValueError("SCHOOL_CAMERA_MESH_UNSUPPORTED")
        vertices = []
        for line in resources[source["uri"]].decode("utf-8").splitlines():
            fields = line.split()
            if fields and fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError("SCHOOL_CAMERA_MESH_VERTEX_INVALID")
                values = [float(value) for value in fields[1:]]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("SCHOOL_CAMERA_MESH_VERTEX_INVALID")
                vertices.append(values)
        if not vertices or len(vertices) > 100_000:
            raise ValueError("SCHOOL_CAMERA_MESH_BUDGET")
        minimum, maximum = [], []
        for index, axis in enumerate("xyz"):
            scale = source["scale_" + axis]
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("SCHOOL_CAMERA_MESH_SCALE_INVALID")
            minimum.append(min(v[index] for v in vertices) * scale)
            maximum.append(max(v[index] for v in vertices) * scale)
        local = [(a + b) / 2 for a, b in zip(minimum, maximum, strict=True)]
        yaw = source.get("yaw_rad", 0)
        c, s = math.cos(yaw), math.sin(yaw)
        center = [c * local[0] - s * local[1] + source["center_x"],
                  s * local[0] + c * local[1] + source["center_y"], local[2] + source["center_z"]]
        box = {"name": source["name"], "yaw_rad": yaw}
        for index, axis in enumerate("xyz"):
            box["center_" + axis] = center[index]
            box["size_" + axis] = maximum[index] - minimum[index]
        primitives.append(box)
    return primitives


# 功能：
#   绑定已审查地图、语义和导航图后分批写出扫描计划，整个 School Map 只进入训练集合。
# 输入：
#   命令行参数：已解包校验的学校源目录、新输出目录及距离采样间距。
# 输出：
#   exit_code：准备完整为零，错误地图版本或空数据直接停止。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.6)
    args = parser.parse_args()
    map_root = args.source / "asset/normalized/map"
    world = read_plugin_file(map_root / "gazebo/world.sdf", limit=32 * 1024**2)
    raw_semantics = read_plugin_file(map_root / "gazebo/semantic.json", limit=16 * 1024**2)
    raw_graph = read_plugin_file(map_root / "navigation-graph.json", limit=2 * 1024**2)
    raw_labels = read_plugin_file(args.source / "labels.json", limit=2 * 1024**2)
    if hashlib.sha256(world).hexdigest() != SCHOOL_WORLD_SHA256:
        raise ValueError("SCHOOL_VIEW_REVIEWED_WORLD_MISMATCH")
    semantics = decode_json(raw_semantics, limit=16 * 1024**2, node_limit=2_000_000)
    graph = decode_json(raw_graph, limit=2 * 1024**2)
    _, resource_hashes = bind_render_resources(world, map_root / "gazebo")
    resources = {relative: read_plugin_file(map_root / "gazebo" / relative, limit=32 * 1024**2)
                 for relative in resource_hashes}
    if any(hashlib.sha256(raw).hexdigest() != resource_hashes[name]
           for name, raw in resources.items()):
        raise ValueError("SCHOOL_CAMERA_RESOURCE_CHANGED")
    primitives = checked_camera_geometry(semantics, resources)
    views, summary = make_school_views(graph, args.spacing_m, primitives)
    output = args.output.absolute()
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for first in range(0, len(views), 512):
        directory = f"school-route-{first // 512:03d}"
        root = output / directory
        root.mkdir()
        hashes = {}
        for name, raw in [("world.sdf", world), ("labels.json", raw_labels), *resources.items()]:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            with (root / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        plan = {"schema": "dronedream.vision-view-plan.v1", "views": views[first:first + 512]}
        publish_runtime_json(root / "views.json", plan, replace_existing=False)
        hashes["views.json"] = hashlib.sha256((root / "views.json").read_bytes()).hexdigest()
        entries.append({"directory": directory, "views": len(plan["views"]),
                        "world_sha256": SCHOOL_WORLD_SHA256, "files_sha256": hashes})
    receipt = {"schema": "dronedream.vision-suite-plan.v1", "complete": True,
               "planned_native_frames": len(views), "layouts": entries,
               "source_semantics_sha256": hashlib.sha256(raw_semantics).hexdigest(),
               "source_graph_sha256": hashlib.sha256(raw_graph).hexdigest(),
               "sampling": summary, "rendered_frames": 0, "ready_for_training": False,
               "whole_school_map_training_only": True}
    publish_runtime_json(output / "suite-plan.json", receipt, replace_existing=False)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
