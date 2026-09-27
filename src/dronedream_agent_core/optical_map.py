"""Compile localization geometry from SDF VISUALS, never collision envelopes.

This is a bounded, offline optical map compiler. Unsupported geometry/frame
semantics fail explicitly. Transparent surfaces remain occluders but cannot
support localization. The result does not grant calibrated covariance or flight.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .local_map_alignment import MapSurfaceIndex
from .plugin_files import portable_plugin_path, read_plugin_file


# 功能：严格读取有限米制几何向量，拒绝缺项、无穷大及超大坐标。
# 输入：text：SDF 数值文本；count：精确分量数。
# 输出：有限 float64 向量。
def _numbers(text, count):
    try:
        parts = str(text).split()
        if len(parts) != count:
            raise ValueError()
        value = np.array([float(p) for p in parts])
        if not np.isfinite(value).all() or np.max(np.abs(value)) > 1e6:
            raise ValueError()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("OPTICAL_MAP_NUMBERS_INVALID") from exc
    return value


# 功能：解析默认 SDF 米/弧度局部姿态；拒绝未实现的 relative_to 和四元数约定。
# 输入：node：模型、连接或视觉节点。
# 输出：父坐标系到该节点的齐次变换。
def _pose(node):
    pose = node.find("pose")
    if pose is not None and pose.attrib:
        raise ValueError("OPTICAL_MAP_POSE_CONVENTION_UNSUPPORTED")
    xyzrpy = _numbers(node.findtext("pose", "0 0 0 0 0 0"), 6)
    r, p, y = xyzrpy[3:]
    cr, cp, cy, sr, sp, sy = (
        math.cos(r),
        math.cos(p),
        math.cos(y),
        math.sin(r),
        math.sin(p),
        math.sin(y),
    )
    transform = np.eye(4)
    transform[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    transform[:3, 3] = xyzrpy[:3]
    return transform


# 功能：从闭合、有向 OBJ 三角网格读取实际表面；不以碰撞胶囊或外包盒代替孔洞。
# 输入：raw：有界 OBJ 内容；scale：三个正缩放因子。
# 输出：缩放后的三角形数组；开放、反向连接、非三角形或退化表面被明确拒绝。
def _obj_triangles(raw, scale):
    vertices, faces = [], []
    for line in raw.decode("utf-8").splitlines():
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(_numbers(" ".join(fields[1:]), 3))
            if len(vertices) > 8192:
                raise ValueError("OPTICAL_MAP_MESH_BUDGET_EXCEEDED")
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("OPTICAL_MAP_TRIANGLES_REQUIRED")
            try:
                ids = [int(f.split("/")[0]) for f in fields[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
            except ValueError as exc:
                raise ValueError("OPTICAL_MAP_MESH_INDEX_INVALID") from exc
            if len(set(ids)) != 3 or any(i < 0 or i >= len(vertices) for i in ids):
                raise ValueError("OPTICAL_MAP_MESH_INDEX_INVALID")
            faces.append(ids)
            if len(faces) > 8192:
                raise ValueError("OPTICAL_MAP_MESH_BUDGET_EXCEEDED")
        elif fields[0] not in {"vn", "vt", "o", "g", "s", "usemtl", "mtllib"}:
            raise ValueError("OPTICAL_MAP_OBJ_DIRECTIVE_UNSUPPORTED")
    if not vertices or not faces:
        raise ValueError("OPTICAL_MAP_MESH_EMPTY")
    edges = {}
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges[(i, j)] = edges.get((i, j), 0) + 1
    if any(n != 1 or edges.get((j, i)) != 1 for (i, j), n in edges.items()):
        raise ValueError("OPTICAL_MAP_CLOSED_ORIENTED_MESH_REQUIRED")
    adjacency = {}
    for i, j in edges:
        adjacency.setdefault(i, set()).add(j)
    seen, pending = set(), [faces[0][0]]
    while pending:
        vertex = pending.pop()
        if vertex in seen:
            continue
        seen.add(vertex)
        pending.extend(adjacency[vertex] - seen)
    if len(seen) != len(adjacency):
        # Nested shells need an explicit solid/void contract. Never guess which
        # component is a cavity or merge their normals by total signed volume.
        raise ValueError("OPTICAL_MAP_DISCONNECTED_MESH_REQUIRES_SOLID_CONTRACT")
    triangles = (np.asarray(vertices) * scale)[np.asarray(faces)]
    if np.any(
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        < 1e-12
    ):
        raise ValueError("OPTICAL_MAP_MESH_DEGENERATE")
    centered = triangles - triangles[0, 0]
    volume = float(
        np.sum(np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2]))) / 6
    )
    if not math.isfinite(volume) or abs(volume) < 1e-12:
        raise ValueError("OPTICAL_MAP_MESH_VOLUME_INVALID")
    # Preserve authoring winding: single-sided rendering culls its backfaces.
    # Normalizing winding here would change the surface visible to the camera.
    return triangles


# 功能：核验包内资源路径和内容身份；禁止绝对路径、链接、越界读取及累积超额资源。
# 输入：root：地图包目录；relative：包内路径；dependencies：本次编译的内容身份表。
# 输出：原始资源字节，身份表记录其 SHA256 和长度。
def _resource(root, relative, dependencies):
    relative = portable_plugin_path(relative)
    data = read_plugin_file(root / relative, limit=8 * 1024 * 1024)
    digest = hashlib.sha256(data).hexdigest()
    previous = dependencies.get(relative)
    record = {"sha256": digest, "bytes": len(data)}
    if previous is not None and previous != record:
        raise ValueError("OPTICAL_MAP_RESOURCE_CHANGED")
    dependencies[relative] = record
    if len(dependencies) > 128 or sum(d["bytes"] for d in dependencies.values()) > 64 * 1024 * 1024:
        raise ValueError("OPTICAL_MAP_RESOURCE_BUDGET_EXCEEDED")
    return data


# 功能：检查网格材料透明度；含透明或未支持材质指令时不授予定位支持。
# 输入：OBJ 原文、包内网格路径、资源根及摘要表。
# 输出：opaque：全部引用材料均为明确支持的非透明材料。
def _mesh_opaque(raw, relative, root, dependencies):
    opaque = True
    for line in raw.decode("utf-8").splitlines():
        fields = line.partition("#")[0].split()
        if not fields or fields[0] != "mtllib":
            continue
        if len(fields) != 2:
            raise ValueError("OPTICAL_MAP_MATERIAL_PATH_INVALID")
        path = (Path(relative).parent / portable_plugin_path(fields[1])).as_posix()
        material = _resource(root, path, dependencies)
        for row in material.decode("utf-8").splitlines():
            parts = row.partition("#")[0].split()
            if not parts:
                continue
            if parts[0] in {"d", "Tr"}:
                alpha = _numbers(" ".join(parts[1:]), 1)[0]
                opaque &= alpha == (1.0 if parts[0] == "d" else 0.0)
            elif parts[0] not in {"newmtl", "Ka", "Kd", "Ks", "Ns", "Ni", "illum"}:
                opaque = False
    return bool(opaque)


# 功能：从版本绑定的静态 SDF 视觉表面构建定位地图，同时保留透明表面的保守遮挡。
# 输入：world_path：实际渲染前的 SDF；expected_world_sha256：调用方所绑定的世界身份。
# 输出：只读索引、实际视觉原语与来源回执；动态模型明确列为排除项，不授予定位或飞行资格。
def compile_optical_map(
    world_path: Path, *, expected_world_sha256: str, expected_resources: dict | None = None
):
    raw = read_plugin_file(world_path, limit=32 * 1024 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_world_sha256:
        raise ValueError("OPTICAL_MAP_WORLD_DIGEST_MISMATCH")
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("OPTICAL_MAP_XML_DIRECTIVE_FORBIDDEN")
    root = ET.fromstring(raw)
    worlds = root.findall("world")
    if root.tag != "sdf" or len(worlds) != 1:
        raise ValueError("OPTICAL_MAP_SINGLE_WORLD_REQUIRED")
    if any(root.find(".//" + tag) is not None for tag in ("include", "frame", "actor")):
        raise ValueError("OPTICAL_MAP_EXTERNAL_FRAME_OR_ACTOR_UNSUPPORTED")
    primitives, dependencies, excluded = [], {}, []
    mesh_triangle_count = 0

    # 功能：按模型/连接/视觉三层真实姿态组合，不把碰撞项混入视觉索引。
    # 输入：model：静态或动态模型节点；parent：父世界变换；prefix：唯一作用域。
    # 输出：累积视觉原语及明确排除项。
    def visit(model, parent, prefix, inherited_static=False, depth=0):
        nonlocal mesh_triangle_count
        # Reject deep user-supplied nesting before Python's recursion limit;
        # rendering and matching must share a bounded, explicit transform tree.
        if depth > 32:
            raise ValueError("OPTICAL_MAP_MODEL_DEPTH_EXCEEDED")
        name = prefix + "/" + model.get("name", "")
        if not inherited_static and model.findtext("static", "false").strip().lower() not in {
            "true",
            "1",
        }:
            excluded.append(name)
            return
        transform = parent @ _pose(model)
        if model.find("plugin") is not None:
            raise ValueError("OPTICAL_MAP_STATIC_MODEL_PLUGIN_UNSUPPORTED")
        for child in model.findall("model"):
            visit(child, transform, name, inherited_static=True, depth=depth + 1)
        for link in model.findall("link"):
            if link.find("plugin") is not None:
                raise ValueError("OPTICAL_MAP_LINK_PLUGIN_UNSUPPORTED")
            link_transform = transform @ _pose(link)
            for visual in link.findall("visual"):
                matrix = link_transform @ _pose(visual)
                geometry = visual.find("geometry")
                if geometry is None or len(geometry) != 1:
                    raise ValueError("OPTICAL_MAP_VISUAL_GEOMETRY_INVALID")
                r = matrix[:3, :3]
                angles = (
                    math.atan2(r[2, 1], r[2, 2]),
                    math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])),
                    math.atan2(r[1, 0], r[0, 0]),
                )
                if math.hypot(r[0, 0], r[1, 0]) < 1e-10:
                    angles = (0.0, angles[1], math.atan2(-r[0, 1], r[1, 1]))
                opaque = _numbers(visual.findtext("transparency", "0"), 1)[0] == 0.0
                for color in ("ambient", "diffuse"):
                    text = visual.findtext("material/" + color)
                    if text is not None:
                        opaque &= _numbers(text, 4)[3] == 1.0
                if visual.find("material/script") is not None or visual.find("plugin") is not None:
                    raise ValueError("OPTICAL_MAP_VISUAL_PLUGIN_OR_SCRIPT_UNSUPPORTED")
                primitive = {
                    "name": name + "/" + link.get("name", "") + "/" + visual.get("name", ""),
                    **dict(
                        zip(
                            ("center_x", "center_y", "center_z"),
                            map(float, matrix[:3, 3]),
                            strict=True,
                        )
                    ),
                    **dict(zip(("roll_rad", "pitch_rad", "yaw_rad"), angles, strict=True)),
                }
                shape = geometry[0]
                if shape.tag == "box":
                    primitive.update(
                        zip(
                            ("size_x", "size_y", "size_z"),
                            map(float, _numbers(shape.findtext("size"), 3)),
                            strict=True,
                        )
                    )
                elif shape.tag in {"sphere", "cylinder"}:
                    primitive["radius_m"] = float(_numbers(shape.findtext("radius"), 1)[0])
                    if shape.tag == "cylinder":
                        primitive["length_m"] = float(_numbers(shape.findtext("length"), 1)[0])
                elif shape.tag == "mesh":
                    if set(c.tag for c in shape) - {"uri", "scale"}:
                        raise ValueError("OPTICAL_MAP_MESH_OPTIONS_UNSUPPORTED")
                    relative = shape.findtext("uri", "")
                    if not relative.endswith(".obj"):
                        raise ValueError("OPTICAL_MAP_MESH_FORMAT_UNSUPPORTED")
                    mesh = _resource(world_path.parent, relative, dependencies)
                    scale = _numbers(shape.findtext("scale", "1 1 1"), 3)
                    if np.min(scale) <= 0:
                        raise ValueError("OPTICAL_MAP_MESH_SCALE_INVALID")
                    triangles = _obj_triangles(mesh, scale)
                    mesh_triangle_count += len(triangles)
                    # Enforce the index-wide budget BEFORE allocating nested
                    # Python lists. Repeated instances count even when they
                    # share one small OBJ resource.
                    if mesh_triangle_count > 65_536:
                        raise ValueError("OPTICAL_MAP_MESH_TRIANGLE_BUDGET_EXCEEDED")
                    primitive["triangles"] = triangles.tolist()
                    double_sided = visual.findtext("material/double_sided", "false").strip().lower()
                    if double_sided not in {"true", "false", "1", "0"}:
                        raise ValueError("OPTICAL_MAP_DOUBLE_SIDED_INVALID")
                    primitive["mesh_double_sided"] = double_sided in {"true", "1"}
                    opaque &= _mesh_opaque(mesh, relative, world_path.parent, dependencies)
                else:
                    raise ValueError("OPTICAL_MAP_VISUAL_SHAPE_UNSUPPORTED:" + shape.tag)
                primitive["registration_eligible"] = bool(opaque)
                primitives.append(primitive)
                if len(primitives) > 20_000:
                    raise ValueError("OPTICAL_MAP_VISUAL_BUDGET_EXCEEDED")

    for model in worlds[0].findall("model"):
        visit(model, np.eye(4), "")
    if expected_resources is not None and (
        not isinstance(expected_resources, dict)
        or any(expected_resources.get(k) != v for k, v in dependencies.items())
    ):
        raise ValueError("OPTICAL_MAP_RESOURCE_DIGEST_MISMATCH")
    index = MapSurfaceIndex(primitives)
    receipt = {
        "schema": "dronedream.optical-map.v1",
        "source_world_sha256": digest,
        "resource_dependencies": dependencies,
        "primitive_counts": index.primitive_counts,
        "resource_lineage_checked": expected_resources is not None,
        "ineligible_occluder_count": sum(not p["registration_eligible"] for p in primitives),
        "excluded_dynamic_models": excluded,
        "covariance_qualified": False,
        "flight_qualification_granted": False,
        "collision_geometry_used": False,
    }
    return index, primitives, receipt
