"""Create an explicit native corridor experiment; not a School Map qualification."""

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from dronedream_agent_core.decision_dataset import decode_object, file_digest
from dronedream_agent_core.runtime_bindings import load_map_runtime_bindings
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：从同一几何表产生真实可碰撞/可见模型，防止标签地图与渲染地图各用一套坐标。
# 输入：米制盒体清单；输出：新的SDF世界，不改变用户校园地图或受管Runtime。
def make_world(primitives):
    root = ET.Element("sdf", version="1.9")
    world = ET.SubElement(root, "world", name="decision_corridor_world")
    ET.SubElement(world, "gravity").text = "0 0 -9.80665"
    ET.SubElement(world, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"
    physics = ET.SubElement(world, "physics", name="native", type="ode")
    for key, value in (
        ("max_step_size", "0.004"),
        ("real_time_factor", "1"),
        ("real_time_update_rate", "250"),
    ):
        ET.SubElement(physics, key).text = value
    coordinates = ET.SubElement(world, "spherical_coordinates")
    for key, value in (
        ("surface_model", "EARTH_WGS84"),
        ("world_frame_orientation", "ENU"),
        ("latitude_deg", "30.27415"),
        ("longitude_deg", "120.15515"),
        ("elevation", "12"),
    ):
        ET.SubElement(coordinates, key).text = value
    scene = ET.SubElement(world, "scene")
    ET.SubElement(scene, "ambient").text = "0.65 0.65 0.65 1"
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "direction").text = "-0.4 0.2 -0.8"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    model = ET.SubElement(world, "model", name="decision_corridor")
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", name="geometry")
    for primitive in primitives:
        for kind in ("collision", "visual"):
            item = ET.SubElement(link, kind, name=primitive["name"] + "-" + kind)
            ET.SubElement(item, "pose").text = (
                " ".join(str(primitive["center_" + axis]) for axis in "xyz") + " 0 0 0"
            )
            geometry = ET.SubElement(item, "geometry")
            box = ET.SubElement(geometry, "box")
            ET.SubElement(box, "size").text = " ".join(
                str(primitive["size_" + axis]) for axis in "xyz"
            )
            if kind == "visual":
                material = ET.SubElement(item, "material")
                for channel in ("ambient", "diffuse"):
                    ET.SubElement(material, channel).text = "0.65 0.55 0.4 1"
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


# 功能：创建独立布局、路线和来源冻结配置；只产生实验输入，不增加正式照片/决策数。
# 输入：现有机型配置、语义机型净空、输出目录、真实通道宽度；输出：新的采集配置。
def prepare(
    base,
    vehicle_clearance,
    output,
    width,
    *,
    bounded_hybrid=False,
    stage_roles=False,
    ceiling_beams=False,
):
    if type(width) not in (float, int) or not 1.6 <= width <= 5:
        raise ValueError("DECISION_SCENE_WIDTH_INVALID")
    output.mkdir(parents=True, exist_ok=False)
    specifications = [
        ("floor", (4, 0, 0.11), (16, 10, 0.22)),
        ("ceiling", (4, 0, 3.9), (16, width + 0.4, 0.2)),
        ("left-wall", (4, width / 2 + 0.1, 2), (16, 0.2, 3.6)),
        ("right-wall", (4, -width / 2 - 0.1, 2), (16, 0.2, 3.6)),
        ("front-wall", (12, 0, 2), (0.2, width + 0.4, 3.6)),
        ("back-wall", (-4, 0, 2), (0.2, width + 0.4, 3.6)),
        # 不规则墙柱提供定位几何；它们是真实碰撞物，不能只在标签中出现。
        ("left-pier", (2.8, width / 2 - 0.08, 2), (0.45, 0.16, 3.6)),
        ("right-pier", (6.2, -width / 2 + 0.08, 2), (0.35, 0.16, 3.6)),
    ]
    if ceiling_beams:
        # 梁体同时出现在碰撞、视觉和地图中；它们是新场景实物，不是虚构的定位证据。
        # 梁下净高3.0米，不进入1.6米目标高度；原无梁退化场景仍保留作压力测试。
        specifications.extend(
            (f"ceiling-beam-{i}", (x, 0, 3.35), (0.35, width, 0.7))
            for i, x in enumerate((1.4, 4.8, 8.0))
        )
    primitives = [
        {
            "shape": "box",
            "name": name,
            **dict(zip(("center_x", "center_y", "center_z"), center, strict=True)),
            **dict(zip(("size_x", "size_y", "size_z"), size, strict=True)),
            "yaw_rad": 0.0,
            "pitch_rad": 0.0,
            "roll_rad": 0.0,
        }
        for name, center, size in specifications
    ]
    start, goal = {"x": 0.0, "y": 0.0, "z": 1.6}, {"x": 4.0, "y": 0.0, "z": 1.6}
    semantic = {
        "schema_version": "dronedream.map-semantic.v1",
        "name": "Independent decision corridor",
        "coordinate_frame": "ENU",
        "collision_primitives": primitives,
        "runtime_collision_primitives": primitives,
        "vehicle_clearance": vehicle_clearance,
        "runtime_bindings": {
            "schema_version": "dronedream.map-runtime-bindings.v1",
            "simulator": "gazebo-harmonic",
            "coordinate_frame": "ENU",
            "mission_launch_waypoint": start,
            "vehicle_spawn": {"x": 0.0, "y": 0.0, "z": 0.209},
        },
        "qualification_granted": False,
    }
    route = {
        "schema_version": "dronedream.graph-route.v1",
        "start_node": "launch",
        "goal_node": "corridor-goal",
        "node_ids": ["launch", "corridor-goal"],
        "edge_ids": ["corridor"],
        "positions_m": [start, goal],
        "route_length_m": 4.0,
        "all_edges_flight_verified": False,
    }
    load_map_runtime_bindings(semantic)
    write_evidence_object(output / "semantic.json", semantic)
    write_evidence_object(output / "route.json", route)
    with (output / "world.sdf").open("x", encoding="utf-8") as stream:
        stream.write(make_world(primitives))
    config = copy.deepcopy(base)
    for key, filename in (
        ("semantic", "semantic.json"),
        ("route", "route.json"),
        ("world_sdf", "world.sdf"),
    ):
        config[key] = str(output / filename)
        config["asset_sha256"][key] = file_digest(output / filename)
    config.update(
        output_root=str(output / "native-run"),
        mission_id="decision-corridor-" + str(width),
        expert_role="local-navigation-policy",
        bounded_hybrid_control=bounded_hybrid,
        collection_expert_roles=(
            ["local-navigation-policy", "precision-maneuver-policy", "recovery-policy"]
            if stage_roles
            else []
        ),
        minimum_enu_m=[-2, -width / 2, 0],
        maximum_enu_m=[8, width / 2, 3.6],
        batch_static_world_visuals=False,
    )
    write_evidence_object(output / "config.json", config)
    write_evidence_object(
        output / "scene-receipt.json",
        {
            "simulation_only": True,
            "formal_training_additions": 0,
            "qualification_granted": False,
            "distinct_layout_claim": False,
            "scene_family": "straight-corridor-width-variants-must-stay-in-same-split",
            "world_sha256": config["asset_sha256"]["world_sdf"],
            "semantic_sha256": config["asset_sha256"]["semantic"],
            "primitive_count": len(primitives),
            "ceiling_beams": ceiling_beams,
        },
    )
    return config


# 功能：限定在调用方明确指定的新实验目录产生文件；输出：冻结配置路径而非飞行成功。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=3)
    parser.add_argument("--bounded-hybrid-control", action="store_true")
    parser.add_argument("--stage-collection-roles", action="store_true")
    parser.add_argument("--ceiling-beams", action="store_true")
    args = parser.parse_args()
    base = decode_object(args.base_config.read_text("utf-8"))
    semantic = decode_object(Path(base["semantic"]).read_text("utf-8"))
    prepare(
        base,
        semantic["vehicle_clearance"],
        args.output,
        args.width,
        bounded_hybrid=args.bounded_hybrid_control,
        stage_roles=args.stage_collection_roles,
        ceiling_beams=args.ceiling_beams,
    )
    print(json.dumps({"config": str(args.output / "config.json"), "formal_training_additions": 0}))


if __name__ == "__main__":
    main()
