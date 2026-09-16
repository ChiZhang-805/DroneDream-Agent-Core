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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numbers(value: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{field} must contain only finite numbers")
    return converted  # type: ignore[return-value]


def _color(value: object) -> tuple[float, float, float, float]:
    if value is None:
        return 0.95, 0.32, 0.12, 1.0
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("color_rgba must contain exactly four numbers")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in converted):
        raise ValueError("color_rgba values must be finite and within [0, 1]")
    return converted  # type: ignore[return-value]


def _format_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _challenge_model(item: dict[str, Any]) -> tuple[ET.Element, dict[str, object]]:
    obstacle_id = item.get("obstacle_id")
    if not isinstance(obstacle_id, str) or not _OBSTACLE_ID.fullmatch(obstacle_id):
        raise ValueError("obstacle_id must be a lowercase stable identifier")
    center = _numbers(item.get("center_enu_m"), field="center_enu_m")
    size = _numbers(item.get("size_m"), field="size_m")
    if not all(0.05 <= value <= 3.0 for value in size):
        raise ValueError("each obstacle size must be within [0.05, 3.0] metres")
    color = _color(item.get("color_rgba"))
    entity_name = f"dronedream_dynamic_{obstacle_id}"

    model = ET.Element("model", {"name": entity_name})
    # Keep the obstruction stationary without declaring it static. Gazebo's
    # dynamic pose stream may publish static entities only once, while the
    # runtime safety adapter needs the challenge in every observation. A
    # gravity-free kinematic link remains collision-bearing and camera-visible
    # but is continuously represented as a dynamic entity.
    ET.SubElement(model, "static").text = "false"
    ET.SubElement(model, "pose").text = _format_vector((*center, 0.0, 0.0, 0.0))
    link = ET.SubElement(model, "link", {"name": "obstacle_link"})
    ET.SubElement(link, "gravity").text = "false"
    ET.SubElement(link, "kinematic").text = "true"
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
        "stationary": True,
        "gazebo_static": False,
        "kinematic": True,
        "physical_collision": True,
        "visible_to_camera": True,
        "published_as_dynamic_obstacle": True,
    }


def build_world(
    *,
    base_world: Path,
    challenge_spec: Path,
    output_world: Path,
    receipt_output: Path,
) -> dict[str, object]:
    if output_world.exists() or receipt_output.exists():
        raise FileExistsError("challenge outputs must not already exist")
    spec = json.loads(challenge_spec.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dynamic obstacle challenge schema is unsupported")
    challenge_id = spec.get("challenge_id")
    if not isinstance(challenge_id, str) or not _OBSTACLE_ID.fullmatch(challenge_id):
        raise ValueError("challenge_id must be a lowercase stable identifier")
    obstacle_items = spec.get("obstacles")
    if not isinstance(obstacle_items, list) or not 1 <= len(obstacle_items) <= 32:
        raise ValueError("challenge must contain between one and 32 obstacles")
    required_encounter_distance_m = float(
        spec.get("required_encounter_distance_m", 12.0)
    )
    if not (
        math.isfinite(required_encounter_distance_m)
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
