"""Resolve installed SDF frames for calibration and independent simulation truth.

SDFormat is imported only inside Runtime. No estimator, command or covariance
is modified. The declared collision centre remains distinct from base_link.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from dronedream_plugin_sdk.protocol import decode_json

from .hashing import sha256_json
from .plugin_files import read_plugin_file


# 功能：
#   1. 有界读取启动坐标契约并核对摘要、飞行器和碰撞中心，不回退到旧固定偏移。
#   2. 校验真实数值与安装位姿；该契约仅用于观测，不授予运动权限或控制真值权限。
# 输入：
#   path：启动时生成的契约普通文件。
#   vehicle_name：调用方当前使用的飞行器模型名称。
#   collision_center_model_m：调用方绑定的模型坐标系碰撞中心，单位米。
# 输出：
#   value：读取和几何校验通过的契约字典。
def load_simulation_frame_contract(
    path: Path, *, vehicle_name: str, collision_center_model_m
) -> dict:
    if path.stat().st_size > 128 * 1024:
        raise ValueError("SIMULATION_FRAME_CONTRACT_TOO_LARGE")
    raw = read_plugin_file(path, limit=128 * 1024)
    value = decode_json(raw, limit=128 * 1024)
    expected_center = _vector(collision_center_model_m, 3)
    if (not isinstance(value, dict)
            or value.get("record_sha256") != sha256_json(
                {k: v for k, v in value.items() if k != "record_sha256"})
            or value.get("schema_version") != "dronedream.simulation-sensor-frames.v1"
            or value.get("vehicle_model_name") != vehicle_name
            or value.get("collision_center_model_m") != expected_center.tolist()
            or value.get("motion_permission_granted") is not False
            or value.get("truth_used_as_control_input") is not False):
        raise ValueError("SIMULATION_FRAME_CONTRACT_BINDING_INVALID")
    # 摘要只证明字段一致；合法摘要仍可能覆盖布尔坐标、空位姿或非单位四元数。
    _vector(value.get("collision_center_model_m"), 3)
    _pose_components(value.get("canonical_at_rest"))
    _validate_frame_names(vehicle_name, value.get("canonical_link_name"))
    return value


# 功能：
#   读取发布者仿真时间，保留整数纳秒精度；该时间只用于仿真导数，不作为 UTC 或延迟钟。
# 输入：
#   message：带有 header.stamp.sec 和 nsec 的仿真消息。
# 输出：
#   timestamp_ns：存在且处于非负有符号 64 位范围内的仿真纳秒时间。
def simulation_pose_time_ns(message) -> int:
    try:
        if (hasattr(message, "HasField") and (not message.HasField("header")
                or not message.header.HasField("stamp"))):
            raise ValueError("SIMULATION_POSE_TIME_MISSING")
        sec, nsec = message.header.stamp.sec, message.header.stamp.nsec
        if (type(sec) is not int or type(nsec) is not int
                or sec < 0 or not 0 <= nsec < 10**9 or sec * 10**9 + nsec > 2**63 - 1):
            raise ValueError("SIMULATION_POSE_TIME_INVALID")
        timestamp_ns = sec * 10**9 + nsec
        return timestamp_ns
    except AttributeError as exc:
        raise ValueError("SIMULATION_POSE_TIME_MISSING") from exc


# 功能：
#   验证指定维数的有限几何数值，拒绝布尔值、隐式字符串转换和超出坐标预算的分量。
# 输入：
#   value：由整数或浮点数组成的列表或元组。
#   length：需要的分量数量。
# 输出：
#   result：校验后的浮点 NumPy 一维数组。
def _vector(value, length):
    if (not isinstance(value, list | tuple) or len(value) != length
            or any(type(v) not in (int, float) for v in value)):
        raise ValueError("SIMULATION_FRAME_VECTOR_INVALID")
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, OverflowError) as exc:
        raise ValueError("SIMULATION_FRAME_VECTOR_INVALID") from exc
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e6:
        raise ValueError("SIMULATION_FRAME_VECTOR_INVALID")
    return result


# 功能：
#   将 wxyz 四元数转换为旋转矩阵，只修正既有百万分之一容差以内的模长偏差。
# 输入：
#   value：安装或观测位姿的四元数分量。
# 输出：
#   result：保持米制距离的三乘三旋转矩阵。
def _rotation(value):
    q = _vector(value, 4)
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > 1e-6:
        raise ValueError("SIMULATION_FRAME_QUATERNION_INVALID")
    w, x, y, z = q / norm
    result = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    return result


# 功能：
#   使用相同的类型、距离和姿态约束解析启动契约及实时观测中的位姿。
# 输入：
#   value：含 position_m 与 orientation_wxyz 的位姿字典。
# 输出：
#   components：由三维位置数组和三乘三旋转矩阵组成的二元组。
def _pose_components(value):
    if not isinstance(value, dict):
        raise ValueError("SIMULATION_FRAME_POSE_INVALID")
    components = _vector(value.get("position_m"), 3), _rotation(value.get("orientation_wxyz"))
    return components


# 功能：
#   精确合成世界、模型及机体坐标变换，计算随机体旋转的碰撞中心，不使用固定 Z 偏移。
# 输入：
#   model_world：模型在世界坐标系中的实时位姿。
#   canonical_in_model：规范机体链接在模型坐标系中的实时位姿。
#   canonical_at_rest：规范机体链接的初始安装位姿。
#   collision_center_model_m：模型初始碰撞中心坐标，单位米。
# 输出：
#   result：碰撞中心在世界坐标系中的三维位置列表，单位米。
def collision_center_from_canonical(
    *, model_world: dict, canonical_in_model: dict, canonical_at_rest: dict,
    collision_center_model_m
):
    model_position, model_rotation = _pose_components(model_world)
    link_position, link_rotation = _pose_components(canonical_in_model)
    initial_position, initial_rotation = _pose_components(canonical_at_rest)
    center = _vector(collision_center_model_m, 3)
    # 先回到初始机体坐标系，再随实时机体和外层模型旋转，避免把安装偏移当作世界偏移。
    center_in_link = initial_rotation.T @ (center - initial_position)
    result = (model_position + model_rotation @
              (link_position + link_rotation @ center_in_link)).tolist()
    return result


# 功能：
#   验证模型与直接机体链接名称，拒绝缺少父级变换信息的嵌套链接。
# 输入：
#   vehicle_name：当前飞行器模型名称。
#   canonical_link_name：直接位于模型下的规范机体链接名称。
# 输出：
#   None：不返回业务数据。
def _validate_frame_names(vehicle_name, canonical_link_name):
    if (not isinstance(vehicle_name, str) or not vehicle_name
            or not isinstance(canonical_link_name, str) or not canonical_link_name
            or "::" in canonical_link_name or "/" in canonical_link_name):
        raise ValueError("SIMULATION_FRAME_ENTITY_CONTRACT_UNSUPPORTED")


# 功能：
#   1. 按精确身份选择唯一的模型及机体位姿，不按相似名称或消息顺序猜测。
#   2. 启动时两种实体都不存在则不生成观测；出现缺失、重复或非法几何时拒绝。
# 输入：
#   poses：同一消息中的实体位姿集合。
#   vehicle_name：待观测飞行器的精确模型名称。
#   canonical_link_name：已绑定的直接机体链接名称。
# 输出：
#   result：模型位姿、机体位姿、命中名称和身份证据组成的元组；尚无实体时为 None。
def select_canonical_poses(poses, *, vehicle_name: str, canonical_link_name: str):
    _validate_frame_names(vehicle_name, canonical_link_name)
    aliases = {canonical_link_name, f"{vehicle_name}::{canonical_link_name}",
               f"{vehicle_name}/{canonical_link_name}"}
    models, links = [], []
    for entity in poses:
        if entity.name == vehicle_name:
            models.append(entity)
        elif entity.name in aliases:
            links.append(entity)
    if not models and not links:
        result = None
        return result
    if len(models) != 1 or len(links) != 1:
        raise ValueError("SIMULATION_FRAME_ENTITY_AMBIGUOUS_OR_MISSING")

    # 功能：
    #   先检查实时消息的原始几何类型，再形成可序列化位姿，避免 float 掩盖非法输入。
    # 输入：
    #   entity：已经匹配到唯一名称的模型或机体实体。
    # 输出：
    #   value：包含位置和 wxyz 四元数的位姿字典。
    def serialize(entity):
        try:
            position = [getattr(entity.position, a) for a in "xyz"]
            orientation = [getattr(entity.orientation, a) for a in "wxyz"]
        except AttributeError as exc:
            raise ValueError("SIMULATION_FRAME_POSE_INVALID") from exc
        value = {"position_m": position, "orientation_wxyz": orientation}
        _pose_components(value)
        return value

    result = (serialize(models[0]), serialize(links[0]), str(links[0].name),
              sorted([vehicle_name, str(links[0].name)]))
    return result


# 功能：
#   1. 按显式资源根顺序解析已安装模型，固定各 SDF 首次摘要并复核解析期间的变化。
#   2. 解析实际规范机体及相机安装，核对当前深度内外参；缺失或未标定配置直接拒绝。
#   3. 生成独立仿真观测契约，不修改估计器、协方差或控制指令，也不授予飞行权限。
# 输入：
#   vehicle_sdf：已安装飞行器 SDF 文件路径。
#   model_roots：按优先级排列的本地模型资源目录，最多八个。
#   collision_center_model_m：模型初始碰撞中心，单位米。
#   require_depth：是否强制校验当前深度传感器安装与光学参数。
# 输出：
#   contract：带来源摘要、安装位姿、深度校验结果及自身摘要的观测契约。
def inspect_simulation_sensor_frames(
    vehicle_sdf: Path, model_roots: list[Path], *, collision_center_model_m,
    require_depth: bool = True
) -> dict:
    center = _vector(collision_center_model_m, 3)
    if type(require_depth) is not bool:
        raise ValueError("SIMULATION_FRAME_DEPTH_REQUIREMENT_INVALID")
    if not model_roots or len(model_roots) > 8:
        raise ValueError("SIMULATION_FRAME_RESOURCE_ROOTS_INVALID")
    roots = [p.resolve(strict=True) for p in model_roots]
    if any(not root.is_dir() for root in roots):
        raise ValueError("SIMULATION_FRAME_RESOURCE_ROOTS_INVALID")
    sources = {}

    # 功能：
    #   有界读取并固定一个 SDF 的首次摘要，拒绝重复包含或最后复核时发生的内容变化。
    # 输入：
    #   path：已定位的模型普通文件。
    # 输出：
    #   None：不返回业务数据。
    def bind(path):
        if len(sources) >= 32 and str(path) not in sources:
            raise ValueError("SIMULATION_FRAME_INCLUDE_LIMIT")
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("SIMULATION_FRAME_SOURCE_TOO_LARGE")
        content = read_plugin_file(path, limit=1024 * 1024)
        digest = hashlib.sha256(content).hexdigest()
        previous = sources.setdefault(str(path), digest)
        if previous != digest:
            raise ValueError("SIMULATION_FRAME_SOURCE_CHANGED")

    # 功能：
    #   按资源根优先级查找本地包含模型，限制路径归属并在交给解析器前绑定其 SDF。
    # 输入：
    #   uri：解析器请求的模型 URI 或相对资源路径。
    # 输出：
    #   resolved：匹配模型的绝对目录；未找到时为空字符串，由解析器报告缺失。
    def find(uri):
        relative = uri.removeprefix("model://")
        for root in roots:
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError("SIMULATION_FRAME_RESOURCE_ESCAPES_ROOT")
            if candidate.is_dir() and (candidate / "model.sdf").is_file():
                bind(candidate / "model.sdf")
                resolved = str(candidate)
                return resolved
        resolved = ""
        return resolved

    system_packages = "/usr/lib/python3/dist-packages"
    if system_packages not in sys.path:
        sys.path.append(system_packages)
    import sdformat14 as sdf

    path = vehicle_sdf.resolve(strict=True)
    bind(path)
    config, root = sdf.ParserConfig(), sdf.Root()
    config.set_find_callback(find)
    root.load(str(path), config)
    model = root.model()
    if model is None:
        raise ValueError("SIMULATION_FRAME_MODEL_MISSING")
    canonical, canonical_name = model.canonical_link_and_relative_name()
    if canonical is None or not canonical_name:
        raise ValueError("SIMULATION_FRAME_CANONICAL_LINK_MISSING")
    if "::" in canonical_name or "/" in canonical_name:
        raise ValueError("SIMULATION_FRAME_NESTED_CANONICAL_LINK_UNSUPPORTED")
    _validate_frame_names(model.name(), canonical_name)

    # 功能：
    #   将解析器已求解的位姿转为严格校验的米制位置与 wxyz 四元数字典。
    # 输入：
    #   pose：SDFormat 在指定参考系中求解的位姿。
    # 输出：
    #   value：校验后的安装位姿字典。
    def serialize(pose):
        value = {"position_m": [pose.pos().x(), pose.pos().y(), pose.pos().z()],
                 "orientation_wxyz": [pose.rot().w(), pose.rot().x(),
                                      pose.rot().y(), pose.rot().z()]}
        _pose_components(value)
        return value

    body_pose = serialize(canonical.semantic_pose().resolve("__model__"))
    depth_fields = {}
    if require_depth:
        camera = model.link_by_name("camera_link")
        sensor = camera.sensor_by_name("StereoOV7251") if camera is not None else None
        if sensor is None:
            raise ValueError("SIMULATION_FRAME_DEPTH_SENSOR_MISSING")
        optics = sensor.camera_sensor()
        if optics is None:
            raise ValueError("SIMULATION_FRAME_DEPTH_CAMERA_MISSING")
        intrinsic_values = (optics.horizontal_fov().radian(), optics.near_clip(), optics.far_clip())
        if (any(not math.isclose(v, expected, rel_tol=0., abs_tol=1e-9)
                for v, expected in zip(intrinsic_values, (1.274, .2, 19.1), strict=True))
                or optics.has_lens_intrinsics() or optics.has_lens_projection()
                or optics.has_depth_near_clip() or optics.has_depth_far_clip()):
            raise ValueError("SIMULATION_FRAME_DEPTH_OPTICS_REQUIRE_CALIBRATION")
        optical_pose = serialize(sensor.semantic_pose().resolve("__model__"))
        rotation = _rotation(body_pose["orientation_wxyz"])
        sensor_offset = rotation.T @ (_vector(optical_pose["position_m"], 3) - center)
        sensor_rotation = rotation.T @ _rotation(optical_pose["orientation_wxyz"])
        from .runtime_sensor_contracts import oakd_lite_depth_sensor_contract

        mount = oakd_lite_depth_sensor_contract()
        expected_offset = np.array([mount.translation_body_m.x, mount.translation_body_m.y,
                                    mount.translation_body_m.z])
        q = mount.orientation_body_from_sensor
        if (not np.allclose(sensor_offset, expected_offset, rtol=0., atol=1e-7)
                or not np.allclose(sensor_rotation, _rotation([q.w, q.x, q.y, q.z]),
                                   rtol=0., atol=1e-7)):
            raise ValueError("SIMULATION_FRAME_DEPTH_MOUNT_REQUIRES_CALIBRATION")
        depth_fields = {"optical_at_rest": optical_pose,
            "depth_optics": dict(zip(("horizontal_fov_rad", "near_m", "far_m"),
                                     intrinsic_values, strict=True)),
            "sensor_translation_from_collision_center_body_m": sensor_offset.tolist()}
    # 最后的复核仍使用同一有界普通文件读取器；不覆盖首次摘要，也不分配无限量内容。
    # 这是读取期间的点时一致性检查，不声称阻止其他进程在检查结束后再替换文件。
    for name in sources:
        bind(Path(name))
    result = {"schema_version": "dronedream.simulation-sensor-frames.v1",
              "vehicle_model_name": model.name(), "sources_sha256": sources,
              "canonical_link_name": canonical_name, "canonical_at_rest": body_pose,
              "collision_center_model_m": center.tolist(), **depth_fields,
              "depth_mount_verified": require_depth, "motion_permission_granted": False,
              "truth_used_as_control_input": False}
    contract = {**result, "record_sha256": sha256_json(result)}
    return contract
