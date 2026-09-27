"""Exact static box tessellation for an explicitly selected Gazebo render path.

Only opaque, untextured, link-local visuals in plugin-free static models are
eligible. Collision geometry, dynamics, sensors, lights, textures, transparent
objects and unfamiliar SDF constructs are untouched. This is not decimation or
a simpler physics world. Runtime qualification still has to measure the result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shlex
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file


@dataclass(frozen=True)
class BoxVisual:
    element: ET.Element
    size: tuple[float, ...]
    pose: tuple[float, ...]
    style: bytes


@dataclass(frozen=True)
class BatchedWorld:
    content: bytes
    meshes: dict[str, bytes]
    receipt: dict


# 功能：
#   为实际转换的字节生成摘要，供源世界、网格及输出回执相互绑定。
# 输入：
#   value：原始字节。
# 输出：
#   digest：SHA-256 十六进制摘要。
def _digest(value: bytes) -> str:
    digest = hashlib.sha256(value).hexdigest()
    return digest


# 功能：
#   解析 SDF 空格分隔数值并检查维数和有限性，不补缺失的坐标或颜色分量。
# 输入：
#   text：SDF 数值文本。
#   length：期望分量个数。
# 输出：
#   values：按原顺序排列的有限浮点元组。
def _numbers(text: str, length: int) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split())
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise ValueError("invalid finite SDF vector")
    return values


# 功能：
#   1. 仅接收语义明确、无纹理且完全不透明的局部盒体，保留不认识的结构不作猜测。
#   2. 将材质、阴影等样式冻结为字节，只有样式完全相同的盒体才允许合并。
# 输入：
#   visual：候选 SDF visual 元素。
# 输出：
#   result：可合并盒体及原因；不适用时盒体为 None 并保留具体拒绝原因。
def _box(visual: ET.Element) -> tuple[BoxVisual | None, str]:
    children = [child.tag for child in visual]
    allowed = {"pose", "geometry", "material", "cast_shadows", "transparency"}
    if (set(visual.attrib) != {"name"} or not set(children) <= allowed
            or len(children) != len(set(children))):
        result = None, "visual-extension-or-duplicate"
        return result
    geometry = visual.find("geometry")
    if (geometry is None or geometry.attrib or len(geometry) != 1
            or geometry[0].tag != "box" or geometry[0].attrib
            or len(geometry[0]) != 1 or geometry[0][0].tag != "size"
            or geometry[0][0].attrib):
        result = None, "non-simple-box"
        return result
    pose_element = visual.find("pose")
    if pose_element is not None and pose_element.attrib:
        # 不猜测 relative_to、角度制或四元数格式；这些 visual 由原 SDF 继续表达。
        result = None, "non-default-pose-convention"
        return result
    material = visual.find("material")
    if (material is None or material.attrib or not len(material)
            or any(child.tag not in {"ambient", "diffuse", "specular", "emissive"}
                   or child.attrib or len(child) for child in material)
            or len(material) != len({child.tag for child in material})):
        result = None, "textured-or-extended-material"
        return result
    try:
        if float(visual.findtext("transparency", "0")) != 0:
            result = None, "transparent"
            return result
        if any(_numbers(child.text or "", 4)[3] != 1. for child in material):
            result = None, "transparent"
            return result
        size = _numbers(geometry[0][0].text or "", 3)
        if min(size) <= 0:
            result = None, "invalid-size"
            return result
        pose = _numbers(visual.findtext("pose", "0 0 0 0 0 0"), 6)
    except (TypeError, ValueError):
        result = None, "invalid-numeric-value"
        return result
    style = ET.Element("style")
    for child in visual:
        if child.tag not in {"geometry", "pose"}:
            style.append(copy.deepcopy(child))
    result = BoxVisual(visual, size, pose, ET.tostring(style)), "eligible"
    return result


# 功能：
#   按 SDF 默认 RzRyRx 顺序旋转顶点或法向，不施加平移。
# 输入：
#   vector：盒体局部三维向量。
#   angles：滚转、俯仰和偏航弧度角。
# 输出：
#   result：旋转后位于 link 坐标系的三维向量。
def _rotate(vector: tuple[float, ...], angles: tuple[float, ...]) -> tuple[float, ...]:
    x, y, z = vector
    roll, pitch, yaw = angles
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    y, z = cr * y - sr * z, sr * y + cr * z
    x, z = cp * x + sp * z, -sp * x + cp * z
    result = cy * x - sy * y, sy * x + cy * y, z
    return result


# Outward winding, separate vertices for flat normals (no smoothed cube edges).
_FACES = (
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)


# 功能：
#   1. 为每个盒体保留六面和十二个朝外三角形，面间不共享平滑法向。
#   2. 将姿态烘入顶点并拒绝非有限结果，不以删面或降低精度换取合并。
# 输入：
#   boxes：同一局部空间分组内、样式一致的已验证盒体列表。
# 输出：
#   content：ASCII OBJ 网格字节。
def _mesh(boxes: list[BoxVisual]) -> bytes:
    lines = ["# Exact opaque static box tessellation", "o static_boxes"]
    vertex_index = normal_index = 1
    for box in boxes:
        for normal, corners in _FACES:
            rotated_normal = _rotate(normal, box.pose[3:])
            lines.append("vn " + " ".join(format(v, ".17g") for v in rotated_normal))
            for corner in corners:
                point = _rotate(tuple(c * s * .5 for c, s in zip(corner, box.size, strict=True)),
                                box.pose[3:])
                point = tuple(p + t for p, t in zip(point, box.pose[:3], strict=True))
                if not all(math.isfinite(value) for value in point):
                    raise ValueError("STATIC_RENDER_MESH_NONFINITE")
                lines.append("v " + " ".join(format(v, ".17g") for v in point))
            for triangle in ((0, 1, 2), (0, 2, 3)):
                lines.append("f " + " ".join(f"{vertex_index + k}//{normal_index}"
                                               for k in triangle))
            vertex_index += 4
            normal_index += 1
    content = ("\n".join(lines) + "\n").encode("ascii")
    return content


# 功能：
#   去除视觉元素后规范化剩余 XML，绑定碰撞、动力学、传感器及插件配置。
# 输入：
#   content：待核对的 SDF 字节，最多 32 MiB。
# 输出：
#   digest：非视觉世界的 SHA-256 摘要。
def nonvisual_world_digest(content: bytes) -> str:
    if not isinstance(content, bytes) or len(content) > 32 * 1024 * 1024:
        raise ValueError("STATIC_RENDER_BATCH_LIMIT_INVALID")
    root = ET.fromstring(content)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "visual":
                parent.remove(child)
    normalized = ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)
    digest = _digest(normalized.encode("utf-8"))
    return digest


# 功能：
#   1. 在不改变非视觉世界的前提下，按局部网格与样式合并静态不透明盒体。
#   2. 每组最多 256 个盒体，小于四个的组和所有不支持的 visual 原样保留。
#   3. 返回转换回执与全部网格，不据转换成功授予飞行验收资格。
# 输入：
#   content：原始 SDF 世界字节，最多 32 MiB。
#   cell_size_m：局部分组边长，单位米，范围一至三十。
# 输出：
#   batched：派生世界、内容寻址网格及转换统计。
def batch_static_box_visuals(content: bytes, *, cell_size_m: float = 10.) -> BatchedWorld:
    if (type(cell_size_m) not in (int, float) or not math.isfinite(cell_size_m)
            or not isinstance(content, bytes)
            or not 1 <= cell_size_m <= 30 or len(content) > 32 * 1024 * 1024):
        raise ValueError("STATIC_RENDER_BATCH_LIMIT_INVALID")
    root = ET.fromstring(content)
    if root.tag != "sdf" or len(root.findall("world")) != 1:
        raise ValueError("STATIC_RENDER_BATCH_REQUIRES_ONE_SDF_WORLD")
    before = len(root.findall(".//visual"))
    meshes, mapping, skipped = {}, [], Counter()
    for model in root.findall("world/model"):
        if (model.findtext("static", "false").strip() not in {"true", "1"}
                or any(model.findall(".//" + tag)
                       for tag in ("plugin", "sensor", "model", "joint", "frame"))):
            skipped["dynamic-or-extended-model"] += len(model.findall(".//visual"))
            continue
        for link in model.findall("link"):
            groups = defaultdict(list)
            existing_names = {visual.get("name") for visual in link.findall("visual")}
            for visual in link.findall("visual"):
                box, reason = _box(visual)
                if box is None:
                    skipped[reason] += 1
                    continue
                cell = tuple(math.floor(value / cell_size_m) for value in box.pose[:3])
                groups[(cell, box.style)].append(box)
            for (cell, style), boxes in groups.items():
                for start in range(0, len(boxes), 256):
                    chunk = boxes[start:start + 256]
                    if len(chunk) < 4:
                        skipped["small-spatial-group"] += len(chunk)
                        continue
                    mesh = _mesh(chunk)
                    digest = _digest(mesh)
                    relative = f"render-meshes/{digest}.obj"
                    name = f"static-render-batch-{len(mapping):06d}"
                    while name in existing_names:
                        name += "-mesh"
                    existing_names.add(name)
                    combined = ET.Element("visual", name=name)
                    geometry = ET.SubElement(combined, "geometry")
                    mesh_element = ET.SubElement(geometry, "mesh")
                    ET.SubElement(mesh_element, "uri").text = relative
                    ET.SubElement(mesh_element, "scale").text = "1 1 1"
                    for element in ET.fromstring(style):
                        combined.append(element)
                    for box in chunk:
                        link.remove(box.element)
                    link.append(combined)
                    meshes[relative] = mesh
                    mapping.append({"model": model.get("name"), "link": link.get("name"),
                        "visual": name, "source_visuals": [b.element.get("name") for b in chunk],
                        "cell": list(cell), "mesh": relative, "mesh_sha256": digest,
                        "box_count": len(chunk), "triangle_count": 12 * len(chunk)})
    result = ET.tostring(root, encoding="utf-8", xml_declaration=True) if mapping else content
    physics = nonvisual_world_digest(content)
    if nonvisual_world_digest(result) != physics:
        raise ValueError("STATIC_RENDER_BATCH_CHANGED_NONVISUAL_WORLD")
    batched = BatchedWorld(result, meshes, {
        "algorithm": "static-opaque-box-spatial-batching", "cell_size_m": cell_size_m,
        "source_world_sha256": _digest(content), "render_world_sha256": _digest(result),
        "nonvisual_world_sha256": physics, "nonvisual_world_unchanged": True,
        "source_visual_count": before, "render_visual_count": len(root.findall(".//visual")),
        "merged_box_count": sum(item["box_count"] for item in mapping),
        "mesh_bytes": sum(len(value) for value in meshes.values()),
        "skipped_visuals": dict(skipped), "batches": mapping,
        "flight_qualification_granted": False,
    })
    return batched


# 功能：
#   读取 OBJ 材质引用及 MTL 简单纹理引用，未知纹理选项拒绝而非静默遗漏依赖。
# 输入：
#   relative：相对于世界目录的资源路径。
#   data：已受大小限制的资源字节。
# 输出：
#   dependencies：相对于世界目录的嵌套依赖路径列表。
def _resource_dependencies(relative: str, data: bytes) -> list[str]:
    path = Path(relative)
    dependencies = []
    suffix = path.suffix.lower()
    if suffix not in {".obj", ".mtl"}:
        return dependencies
    for line in data.decode("utf-8-sig").splitlines():
        command = line.lstrip().split(maxsplit=1)
        if not command:
            continue
        is_material = suffix == ".obj" and command[0] == "mtllib"
        is_texture = suffix == ".mtl" and (
            command[0].startswith("map_") or command[0] in {"bump", "disp", "decal", "refl", "norm"})
        if not (is_material or is_texture):
            continue
        fields = shlex.split(line, comments=True, posix=True)[1:]
        # 不猜测带选项纹理的文件名，避免把参数当成文件或复制后丢失贴图。
        if not fields or (is_texture and (len(fields) != 1 or fields[0].startswith("-"))):
            raise ValueError("STATIC_RENDER_RELATIVE_RESOURCE_UNSUPPORTED:" + relative)
        for dependency in fields:
            if (not dependency or urlsplit(dependency).scheme or Path(dependency).is_absolute()
                    or "\\" in dependency or "\x00" in dependency):
                raise ValueError("STATIC_RENDER_RELATIVE_RESOURCE_UNSUPPORTED:" + dependency)
            dependencies.append((path.parent / dependency).as_posix())
            if len(dependencies) > 4096:
                raise ValueError("STATIC_RENDER_RESOURCE_BUDGET_EXCEEDED")
    return dependencies


# 功能：
#   1. 复制 SDF 相对资源及 OBJ/MTL 依赖并绑定实际字节，不遍历目录、不联网下载。
#   2. 拒绝越界、链接及超量资源；显式 model、file、网络或绝对引用保留原有解析方式。
# 输入：
#   content：包含资源引用的 SDF 字节。
#   source_parent：源 SDF 所在目录。
#   output：新派生世界所在目录。
# 输出：
#   resources：按原始相对引用记录摘要与字节数的资源表。
def copy_relative_render_resources(content: bytes, source_parent: Path, output: Path) -> dict:
    if not isinstance(content, bytes) or len(content) > 32 * 1024 * 1024:
        raise ValueError("STATIC_RENDER_BATCH_LIMIT_INVALID")
    check_plain_plugin_path(source_parent)
    check_plain_plugin_path(output)
    source_parent, output = source_parent.resolve(), output.resolve()
    resources, total, pending = {}, 0, deque()
    names = {"uri", "albedo_map", "normal_map", "roughness_map", "metalness_map",
             "emissive_map", "environment_map", "light_map"}
    for element in ET.fromstring(content).iter():
        if element.tag not in names or not element.text or not element.text.strip():
            continue
        value = element.text.strip()
        if urlsplit(value).scheme or Path(value).is_absolute():
            continue  # Existing explicit model://, file://, URL or absolute reference.
        pending.append(value)
        if len(pending) > 4096:
            raise ValueError("STATIC_RENDER_RESOURCE_BUDGET_EXCEEDED")
    while pending:
        value = pending.popleft()
        source, destination = source_parent / value, output / value
        check_plain_plugin_path(source)
        check_plain_plugin_path(destination)
        if (not source.resolve().is_relative_to(source_parent)
                or not destination.resolve().is_relative_to(output) or not source.is_file()):
            raise ValueError("STATIC_RENDER_RELATIVE_RESOURCE_UNSUPPORTED:" + value)
        # 同一材质可能被多个网格引用；规范化后只复制一次，仍保留原引用可解析的路径。
        value = source.resolve().relative_to(source_parent).as_posix()
        if value in resources:
            continue
        if len(resources) >= 4096:
            raise ValueError("STATIC_RENDER_RESOURCE_BUDGET_EXCEEDED")
        # 大小检查不能代替实际读取上限；总预算也在读取前扣除已使用的字节。
        data = read_plugin_file(source, limit=min(128 * 1024 * 1024, 512 * 1024 * 1024-total))
        total += len(data)
        if total > 512 * 1024 * 1024:
            raise ValueError("STATIC_RENDER_RESOURCE_BUDGET_EXCEEDED")
        check_plain_plugin_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        check_plain_plugin_path(destination)
        with destination.open("xb") as handle:
            handle.write(data)
        resources[value] = {"sha256": _digest(data), "bytes": len(data)}
        pending.extend(_resource_dependencies(value, data))
        if len(pending) > 4096:
            raise ValueError("STATIC_RENDER_RESOURCE_BUDGET_EXCEEDED")
    return resources


# 功能：
#   1. 在新目录独占落盘派生世界、网格与相对资源，不覆盖原图或旧实验。
#   2. 绑定实际使用的源字节及调用者可选摘要，落盘后再次确认源内容未变化。
# 输入：
#   source：源世界 SDF 普通文件路径。
#   output：必须尚不存在的派生目录。
#   expected_source_sha256：调用者先前读取并验证的源摘要，没有上游快照时为 None。
# 输出：
#   prepared：派生世界路径与转换回执。
def prepare_static_render_world(source: Path, output: Path, *,
                               expected_source_sha256: str | None = None) -> tuple[Path, dict]:
    content = read_plugin_file(source, limit=32 * 1024 * 1024)
    if expected_source_sha256 is not None and (
            type(expected_source_sha256) is not str
            or re.fullmatch(r"[a-f0-9]{64}", expected_source_sha256) is None
            or _digest(content) != expected_source_sha256):
        raise ValueError("STATIC_RENDER_SOURCE_BINDING_MISMATCH")
    result = batch_static_box_visuals(content)
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    resources = copy_relative_render_resources(content, source.parent, output)
    for relative, data in result.meshes.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    world = output / "world.sdf"
    with world.open("xb") as handle:
        handle.write(result.content)
    receipt = {**result.receipt, "source_path": str(source.resolve()),
               "preserved_relative_resources": resources,
               "implementation_sha256": hash_plugin_file(Path(__file__), limit=1024 * 1024)}
    with (output / "render-batching-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, allow_nan=False, indent=2)
    if read_plugin_file(source, limit=32 * 1024 * 1024) != content:
        raise ValueError("STATIC_RENDER_BATCH_SOURCE_CHANGED")
    prepared = world, receipt
    return prepared
