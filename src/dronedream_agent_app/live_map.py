"""Read the selected simulator map's image registration for the Live view."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from dronedream_agent_core.asset_packages import AssetIR
from dronedream_agent_core.assets import read_map_semantic_object
from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_agent_core.xml_values import parse_xml

from .asset_runtime_resolver import _declared_file, resolve_versioned_map


@dataclass(frozen=True)
class LiveMap:
    metadata: dict[str, object]
    image: bytes
    media_type: str


def _finite_numbers(value: object, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
    return result


def _planar_registration(root: Path, world: Path, asset_ir: AssetIR) -> dict[str, object]:
    """Legacy fallback: verify a single untransformed rectangular textured plane.

    Do not infer registration from a map name, a route's bounds, or an image's
    subject. Nontrivial meshes require an explicit affine registration.
    """
    document = parse_xml(
        read_plugin_file(world, limit=16 * 1024 * 1024), maximum_bytes=16 * 1024 * 1024
    )
    candidates = []
    for model in document.findall("./world/model"):
        for link in model.findall("link"):
            for visual in link.findall("visual"):
                image_name = visual.findtext("material/pbr/metal/albedo_map")
                mesh_name = visual.findtext("geometry/mesh/uri")
                if not image_name or not mesh_name:
                    continue
                # A transformed/nested visual cannot use the simple legacy mapping.
                if any(element.find("pose") is not None for element in (model, link, visual)):
                    continue
                if visual.find("geometry/mesh/scale") is not None:
                    continue
                mesh_relative = (world.parent / mesh_name).relative_to(root).as_posix()
                mesh = _declared_file(root, asset_ir, mesh_relative)
                vertices, texture, faces = [], [], []
                for line in read_plugin_file(mesh, limit=1024 * 1024).decode().splitlines():
                    parts = line.split()
                    if parts[:1] == ["v"]:
                        vertices.append(_finite_numbers([float(v) for v in parts[1:]], 3))
                    elif parts[:1] == ["vt"]:
                        texture.append(_finite_numbers([float(v) for v in parts[1:]], 2))
                    elif parts[:1] == ["f"]:
                        faces.append(parts[1:])
                if len(vertices) != 4 or len(texture) != 4 or len(faces) != 1 or len(faces[0]) != 4:
                    continue
                xs, ys, zs = zip(*vertices, strict=True)
                if max(zs) != min(zs) or max(xs) <= min(xs) or max(ys) <= min(ys):
                    continue
                if set((x, y) for x, y, _ in vertices) != {
                    (x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))
                }:
                    continue
                mapped = set()
                for token in faces[0]:
                    indices = token.split("/")
                    if len(indices) < 2:
                        raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
                    vi, ti = int(indices[0]) - 1, int(indices[1]) - 1
                    if not 0 <= vi < 4 or not 0 <= ti < 4:
                        raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
                    x, y, _ = vertices[vi]
                    expected = [
                        (x - min(xs)) / (max(xs) - min(xs)),
                        (y - min(ys)) / (max(ys) - min(ys)),
                    ]
                    if texture[ti] != expected:
                        break
                    mapped.add(vi)
                if len(mapped) != 4:
                    continue
                candidates.append(
                    {
                        "image_path": (world.parent / image_name).relative_to(root).as_posix(),
                        "bounds": (min(xs), min(ys), max(xs), max(ys)),
                    }
                )
    if len(candidates) != 1:
        raise ValueError("LIVE_MAP_REGISTRATION_REQUIRED")
    return candidates[0]


def resolve_live_map(record: dict[str, object]) -> LiveMap:
    """Resolve hash-verified pixels and the exact ENU-to-image affine transform."""
    selected = resolve_versioned_map(record)
    asset_ir = AssetIR.model_validate(record["asset_ir"])
    semantic = read_map_semantic_object(selected.semantic)
    if not isinstance(semantic, dict) or semantic.get("coordinate_frame") not in {
        "ENU",
        "map_enu",
        "gazebo-enu",
    }:
        raise ValueError("LIVE_MAP_COORDINATE_FRAME_UNSUPPORTED")
    registration = semantic.get("live_map")
    explicit = registration is not None
    if explicit:
        if not isinstance(registration, dict) or registration.get("schema_version") != (
            "dronedream.live-map-georeference.v1"
        ):
            raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
        if registration.get("coordinate_frame") not in {"ENU", "map_enu", "gazebo-enu"}:
            raise ValueError("LIVE_MAP_COORDINATE_FRAME_UNSUPPORTED")
    else:
        registration = _planar_registration(selected.root, selected.world_sdf, asset_ir)
    image_path = _declared_file(selected.root, asset_ir, str(registration["image_path"]))
    image_bytes = read_plugin_file(image_path, limit=32 * 1024 * 1024)
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        media_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(
            image.format
        )
        if media_type is None or not 0 < width <= 16384 or not 0 < height <= 16384:
            raise ValueError("LIVE_MAP_IMAGE_INVALID")
        image.verify()
    if explicit:
        if registration.get("width_px") != width or registration.get("height_px") != height:
            raise ValueError("LIVE_MAP_IMAGE_DIMENSIONS_MISMATCH")
        affine = _finite_numbers(registration.get("world_to_image"), 6)
    else:
        min_east, min_north, max_east, max_north = registration["bounds"]
        sx, sy = width / (max_east - min_east), height / (max_north - min_north)
        affine = [sx, 0, -min_east * sx, 0, -sy, max_north * sy]
    if abs(affine[0] * affine[4] - affine[1] * affine[3]) < 1e-12:
        raise ValueError("LIVE_MAP_REGISTRATION_INVALID")
    return LiveMap(
        {
            "schema_version": "dronedream.live-map.v1",
            "map_id": selected.asset_id,
            "map_content_sha256": selected.content_sha256,
            "coordinate_frame": "gazebo-enu",
            "width_px": width,
            "height_px": height,
            # x=a*east+b*north+c; y=d*east+e*north+f, in image pixels.
            "world_to_image": affine,
        },
        image_bytes,
        media_type,
    )
