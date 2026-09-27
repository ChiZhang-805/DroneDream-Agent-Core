#!/usr/bin/env python3
"""Derive a Gazebo world with content-bound physical obstacle challenges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "dronedream.dynamic-obstacle-challenge.v1"
RECEIPT_SCHEMA_VERSION = "dronedream.dynamic-obstacle-world-receipt.v1"
_OBSTACLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


# 功能：
#   计算实际资源文件摘要，用于绑定派生场景及回执。
# 输入：
#   path：资源文件路径。
# 输出：
#   digest：文件内容 SHA-256。
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 功能：
#   校验三维米制参数，拒绝布尔值、字符串、非有限值和不可用的极大坐标。
# 输入：
#   value：三项数值列表。
#   field：错误定位使用的字段名。
# 输出：
#   converted：有限三元浮点数组。
def _numbers(value: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers")
    if any(type(item) not in (int, float) or not -10_000_000 <= item <= 10_000_000 for item in value):
        raise ValueError(f"{field} must contain bounded numbers without coercion")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{field} must contain only finite numbers")
    return converted  # type: ignore[return-value]


# 功能：
#   生成明确的默认颜色或校验四个零到一分量，不进行隐式类型转换。
# 输入：
#   value：可选 RGBA 列表。
# 输出：
#   converted：四元浮点颜色。
def _color(value: object) -> tuple[float, float, float, float]:
    if value is None:
        return 0.95, 0.32, 0.12, 1.0
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("color_rgba must contain exactly four numbers")
    if any(type(item) not in (int, float) or not 0 <= item <= 1 for item in value):
        raise ValueError("color_rgba must contain bounded numbers without coercion")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in converted):
        raise ValueError("color_rgba values must be finite and within [0, 1]")
    return converted  # type: ignore[return-value]


# 功能：
#   将已核验的数值转换为 SDF 空格分隔文本。
# 输入：
#   values：已验证的有限数值序列。
# 输出：
#   text：SDF 向量字符串。
def _format_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


# 功能：
#   校验可选的低速圆周障碍运动，显式限制速度、转速与向心加速度；不控制无人机。
# 输入：
#   value：圆周运动配置，None 表示静止场景。
# 输出：
#   motion：规范化参数或 None。
def _motion(value: object) -> dict | None:
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"kind", "radius_m", "speed_mps", "clockwise"}
            or value["kind"] != "circle" or type(value["clockwise"]) is not bool):
        raise ValueError("challenge motion contract is invalid")
    radius, speed = value["radius_m"], value["speed_mps"]
    if (type(radius) not in (int, float) or not .5 <= radius <= 3.
            or type(speed) not in (int, float) or not .15 <= speed <= 1.
            or speed / radius > 1. or speed * speed / radius > .75):
        raise ValueError("challenge motion exceeds bounded speed, radius or acceleration")
    motion = {**value, "yaw_rate_rad_s": (-1 if value["clockwise"] else 1) * speed / radius}
    return motion


# 功能：
#   创建具有真实碰撞体和可见外观的障碍；可选运动只驱动此障碍，不向飞行控制器透露轨迹。
# 输入：
#   item：障碍身份、位置、尺寸和颜色。
# 输出：
#   model、receipt：SDF 模型和明确的能力回执。
def _challenge_model(item: dict[str, Any]) -> tuple[ET.Element, dict[str, object]]:
    obstacle_id = item.get("obstacle_id")
    if not isinstance(obstacle_id, str) or not _OBSTACLE_ID.fullmatch(obstacle_id):
        raise ValueError("obstacle_id must be a lowercase stable identifier")
    center = _numbers(item.get("center_enu_m"), field="center_enu_m")
    size = _numbers(item.get("size_m"), field="size_m")
    if not all(0.05 <= value <= 3.0 for value in size):
        raise ValueError("each obstacle size must be within [0.05, 3.0] metres")
    color = _color(item.get("color_rgba"))
    motion = _motion(item.get("motion"))
    entity_name = f"dronedream_dynamic_{obstacle_id}"

    model = ET.Element("model", {"name": entity_name})
    # 独立见证需要连续位姿：两类挑战都不声明 static。
    # 静止挑战使用无重力运动学刚体；运动挑战改用带惯量的刚体和速度插件。
    ET.SubElement(model, "static").text = "false"
    ET.SubElement(model, "pose").text = _format_vector((*center, 0.0, 0.0, 0.0))
    link = ET.SubElement(model, "link", {"name": "obstacle_link"})
    ET.SubElement(link, "gravity").text = "false"
    ET.SubElement(link, "kinematic").text = "true" if motion is None else "false"
    if motion is not None:
        # 由已安装 Gazebo VelocityControl 施加模型自身坐标系速度，而非外部瞬移位姿。
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass").text = "1"
        inertia = ET.SubElement(inertial, "inertia")
        for name, value in (("ixx", (size[1] ** 2 + size[2] ** 2) / 12),
                            ("iyy", (size[0] ** 2 + size[2] ** 2) / 12),
                            ("izz", (size[0] ** 2 + size[1] ** 2) / 12)):
            ET.SubElement(inertia, name).text = str(value)
        plugin = ET.SubElement(model, "plugin", filename="gz-sim-velocity-control-system", name="gz::sim::systems::VelocityControl")
        ET.SubElement(plugin, "topic").text = f"/dronedream/challenges/{obstacle_id}/velocity"
        ET.SubElement(plugin, "initial_linear").text = _format_vector((motion["speed_mps"], 0., 0.))
        ET.SubElement(plugin, "initial_angular").text = _format_vector((0., 0., motion["yaw_rate_rad_s"]))
    collision = ET.SubElement(link, "collision", {"name": "obstacle_collision"})
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_box = ET.SubElement(collision_geometry, "box")
    ET.SubElement(collision_box, "size").text = _format_vector(size)
    visual = ET.SubElement(link, "visual", {"name": "obstacle_visual"})
    visual_geometry = ET.SubElement(visual, "geometry")
    visual_box = ET.SubElement(visual_geometry, "box")
    ET.SubElement(visual_box, "size").text = _format_vector(size)
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = _format_vector(color)
    ET.SubElement(material, "diffuse").text = _format_vector(color)

    return model, {
        "obstacle_id": obstacle_id,
        "entity_name": entity_name,
        "center_enu_m": list(center),
        "size_m": list(size),
        "stationary": motion is None,
        "motion": motion,
        "gazebo_static": False,
        "kinematic": motion is None,
        "physical_collision": True,
        "visible_to_camera": True,
        "published_as_dynamic_obstacle": True,
    }


# 功能：
#   校验恢复场景并复制其实际资源，保留源世界，输出绑定摘要的派生世界和回执。
# 输入：
#   base_world、challenge_spec：源世界和显式挑战配置。
#   output_world、receipt_output：互不相同且不存在的输出文件。
# 输出：
#   receipt：派生场景、资源及障碍身份回执，不授予飞行资格。
def build_world(
    *,
    base_world: Path,
    challenge_spec: Path,
    output_world: Path,
    receipt_output: Path,
) -> dict[str, object]:
    if output_world.resolve() == receipt_output.resolve():
        raise ValueError("challenge world and receipt outputs must differ")
    if output_world.exists() or receipt_output.exists():
        raise FileExistsError("challenge outputs must not already exist")
    spec = json.loads(challenge_spec.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dynamic obstacle challenge schema is unsupported")
    challenge_id = spec.get("challenge_id")
    if not isinstance(challenge_id, str) or not _OBSTACLE_ID.fullmatch(challenge_id):
        raise ValueError("challenge_id must be a lowercase stable identifier")
    obstacle_items = spec.get("obstacles")
    if not isinstance(obstacle_items, list) or not 1 <= len(obstacle_items) <= 32:
        raise ValueError("challenge must contain between one and 32 obstacles")
    required_encounter_distance_m = spec.get("required_encounter_distance_m", 12.0)
    if not (
        type(required_encounter_distance_m) in (int, float)
        and 0.5 <= required_encounter_distance_m <= 50.0
    ):
        raise ValueError(
            "required_encounter_distance_m must be finite and within [0.5, 50]"
        )

    tree = ET.parse(base_world)
    root = tree.getroot()
    worlds = root.findall("world")
    if root.tag != "sdf" or len(worlds) != 1:
        raise ValueError("base SDF must contain exactly one Gazebo world")
    world = worlds[0]
    expected_world_name = spec.get("world_name")
    if expected_world_name is not None and world.get("name") != expected_world_name:
        raise ValueError("challenge world_name does not match the base SDF")

    resource_roots: dict[Path, Path] = {}
    base_resource_root = base_world.parent.resolve()
    output_resource_root = output_world.parent.resolve()
    # 输出嵌入源资源树会让邻接资源递归包含自身；不能依赖 copytree 在写到一半后失败。
    if output_resource_root.is_relative_to(base_resource_root):
        raise ValueError("challenge output directory must be outside the base resource tree")
    # Gazebo worlds may reference assets outside ``<uri>`` nodes (for example
    # PBR texture maps), while OBJ files can in turn reference sibling MTL
    # files.  Preserve every adjacent resource directory so the derived world
    # remains visually and physically identical to the base world apart from
    # the explicitly appended challenge models.
    for adjacent in sorted(base_resource_root.iterdir()):
        if not adjacent.is_dir():
            continue
        destination_root = (output_resource_root / adjacent.name).resolve()
        if destination_root.exists():
            raise FileExistsError(
                f"challenge resource output already exists: {destination_root}"
            )
        resource_roots[adjacent.resolve()] = destination_root
    for uri_node in root.findall(".//uri"):
        uri = (uri_node.text or "").strip()
        if not uri or "://" in uri or Path(uri).is_absolute():
            continue
        relative = Path(uri)
        source = (base_resource_root / relative).resolve()
        try:
            source.relative_to(base_resource_root)
        except ValueError as error:
            raise ValueError("relative world resource escapes the base directory") from error
        if not source.is_file():
            raise FileNotFoundError(f"relative world resource is missing: {uri}")
        destination = (output_resource_root / relative).resolve()
        try:
            destination.relative_to(output_resource_root)
        except ValueError as error:
            raise ValueError("relative world resource escapes the output directory") from error
        if len(relative.parts) > 1:
            source_root = (base_resource_root / relative.parts[0]).resolve()
            destination_root = (output_resource_root / relative.parts[0]).resolve()
        else:
            source_root = source
            destination_root = destination
        if destination_root.exists():
            raise FileExistsError(
                f"challenge resource output already exists: {destination_root}"
            )
        resource_roots[source_root] = destination_root

    # 场景或回执不能覆盖将要复制的网格/纹理；否则回执摘要与最终资源不一致。
    for source_root, destination_root in resource_roots.items():
        for output in (output_world.resolve(), receipt_output.resolve()):
            if output == destination_root or (
                source_root.is_dir() and output.is_relative_to(destination_root)
            ):
                raise ValueError("challenge outputs overlap copied resources")

    receipts: list[dict[str, object]] = []
    entity_names = {model.get("name") for model in world.findall("model")}
    obstacle_ids: set[str] = set()
    for raw_item in obstacle_items:
        if not isinstance(raw_item, dict):
            raise ValueError("each obstacle must be an object")
        model, receipt = _challenge_model(raw_item)
        obstacle_id = str(receipt["obstacle_id"])
        entity_name = str(receipt["entity_name"])
        if obstacle_id in obstacle_ids or entity_name in entity_names:
            raise ValueError("challenge obstacle identities must be unique")
        obstacle_ids.add(obstacle_id)
        entity_names.add(entity_name)
        world.append(model)
        receipts.append(receipt)

    output_world.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    resource_receipts: list[dict[str, object]] = []
    for source_root, destination_root in sorted(
        resource_roots.items(), key=lambda item: str(item[0])
    ):
        if source_root.is_dir():
            shutil.copytree(source_root, destination_root)
            sources = sorted(path for path in source_root.rglob("*") if path.is_file())
        else:
            destination_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root, destination_root)
            sources = [source_root]
        for source in sources:
            relative = source.relative_to(base_resource_root)
            destination = output_resource_root / relative
            resource_receipts.append(
                {
                    "uri": relative.as_posix(),
                    "source": str(source),
                    "source_sha256": _sha256(source),
                    "output": str(destination),
                    "output_sha256": _sha256(destination),
                }
            )
    tree.write(output_world, encoding="utf-8", xml_declaration=True)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "challenge_id": challenge_id,
        "challenge_spec": str(challenge_spec.resolve()),
        "challenge_spec_sha256": _sha256(challenge_spec),
        "base_world": str(base_world.resolve()),
        "base_world_sha256": _sha256(base_world),
        "output_world": str(output_world.resolve()),
        "output_world_sha256": _sha256(output_world),
        "world_name": world.get("name"),
        "required_encounter_distance_m": required_encounter_distance_m,
        "obstacle_count": len(receipts),
        "obstacles": receipts,
        "relative_resource_count": len(resource_receipts),
        "relative_resources": resource_receipts,
        "qualification_granted": False,
        "intended_use": "simulation-recovery-challenge",
    }
    receipt_output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


# 功能：
#   读取明确的场景输入和输出路径并执行构建，输出内容绑定回执。
# 输入：
#   无：读取命令行参数。
# 输出：
#   exit_code：构建成功为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_world", type=Path)
    parser.add_argument("challenge_spec", type=Path)
    parser.add_argument("output_world", type=Path)
    parser.add_argument("receipt_output", type=Path)
    args = parser.parse_args()
    receipt = build_world(
        base_world=args.base_world,
        challenge_spec=args.challenge_spec,
        output_world=args.output_world,
        receipt_output=args.receipt_output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
