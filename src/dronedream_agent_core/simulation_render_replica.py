"""Explicit run-local rendering isolation; physical sensors stay in Gazebo.

The launch contract binds an already verified native build and the selected SDF
include graph. Only the Sensors system is replaced in the copied server config.
This is neither an installed Runtime mutation nor flight qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from .plugin_files import check_plain_plugin_path, read_plugin_file
from .render_replica_runtime import (
    EXECUTABLE,
    RECEIPT,
    SOURCE_LIBRARY,
    file_hash,
    validate_replica_runtime,
)
from .simulation_camera_profile import camera_configuration
from .xml_values import parse_xml

_SENSORS = "gz::sim::systems::Sensors"
_SOURCE_BYTES = 32 * 1024 * 1024
_REPORT_BYTES = 1024 * 1024


# 功能：
#   在独占创建文件前验证报告 JSON 并完成序列化，错误数据不留下半份输出。
# 输入：
#   path：必须尚不存在的报告路径。
#   data：允许一 MiB 以内的标准 JSON 报告。
# 输出：
#   None：不返回业务数据。
def _write_new(path: Path, data: dict) -> None:
    encode_json(data, limit=_REPORT_BYTES)
    rendered = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    if len(rendered.encode("utf-8")) > _REPORT_BYTES:
        raise ValueError("REPLICA_REPORT_TOO_LARGE")
    check_plain_plugin_path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)


# 功能：
#   有界读取和解析普通 XML 文件，固定首次读取摘要，拒绝重复读取时的来源变化。
# 输入：
#   path：待解析的 SDF 或服务器配置文件。
#   inventory：本次准备过程共享的来源摘要表，最多 128 份文件。
# 输出：
#   result：同一份原始字节对应的 XML 根节点和字节串。
def _xml(path: Path, *, inventory: dict[str, str]) -> tuple[ET.Element, bytes]:
    try:
        content = read_plugin_file(path, limit=_SOURCE_BYTES)
        key = str(path.resolve(strict=True))
        digest = hashlib.sha256(content).hexdigest()
        if key in inventory and inventory[key] != digest:
            raise ValueError("REPLICA_SOURCE_CHANGED")
        if key not in inventory and len(inventory) >= 128:
            raise ValueError("REPLICA_SDF_INCLUDE_CAPACITY")
        root = parse_xml(content, maximum_bytes=_SOURCE_BYTES)
    except (OSError, ValueError, ET.ParseError) as error:
        raise ValueError("REPLICA_SDF_INPUT_INVALID_OR_CHANGED") from error
    inventory[key] = digest
    result = root, content
    return result


# 功能：
#   按实际资源搜索顺序遍历本地包含图，拒绝循环、远程获取与模型内重新引入的旧渲染系统。
# 输入：
#   paths：世界和飞机模型等入口文件。
#   resources：按原始优先级排列的本地模型资源目录。
#   inventory：本次准备过程共享的已绑定来源，重复访问仍复核内容。
# 输出：
#   sources：包含图来源摘要表的独立副本。
def _include_graph(paths: tuple[Path, ...], resources: tuple[Path, ...],
                   *, inventory: dict[str, str]) -> dict[str, str]:
    visited: set[Path] = set()

    # 功能：
    #   校验单个模型及其 model.config 和嵌套包含，重复包含复核摘要但不重复遍历子树。
    # 输入：
    #   path：当前模型来源路径。
    #   ancestors：当前递归路径上的来源集合。
    # 输出：
    #   None：不返回业务数据。
    def visit(path: Path, ancestors: frozenset[Path]) -> None:
        check_plain_plugin_path(path)
        path = path.resolve(strict=True)
        if path in ancestors or len(ancestors) >= 32:
            raise ValueError("REPLICA_SDF_INCLUDE_CYCLE")
        root, _ = _xml(path, inventory=inventory)
        if path in visited:
            return
        visited.add(path)
        # 服务器配置独占物理世界渲染系统；模型包含不能偷偷恢复旧的渲染链路。
        for plugin in root.findall(".//plugin"):
            name, library = plugin.get("name", ""), plugin.get("filename", "").lower()
            if (name == _SENSORS or "sensors-system" in library
                    or "render-preparation" in library or "render-source" in library):
                raise ValueError("REPLICA_SDF_CONTAINS_RENDER_SYSTEM")
        for uri_element in root.findall(".//include/uri"):
            uri = (uri_element.text or "").strip()
            if uri.startswith("model://"):
                relative = Path(uri.removeprefix("model://"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("REPLICA_MODEL_URI_INVALID")
                candidates = [p / relative for p in resources]
                candidate = next((p for p in candidates if p.exists()), None)
            elif "://" not in uri and uri:
                candidate = path.parent / uri
                if not candidate.exists():
                    # SDFormat also resolves bare model names (PX4's x500 and
                    # OakD includes) through the registered resource roots.
                    candidate = next((p / uri for p in resources if (p / uri).exists()), None)
            else:
                raise ValueError("REPLICA_REMOTE_INCLUDE_FORBIDDEN")
            if candidate is None or not candidate.exists():
                raise ValueError("REPLICA_MODEL_INCLUDE_UNRESOLVED:" + uri)
            if candidate.is_dir():
                config = candidate / "model.config"
                if config.is_file():
                    model_config, _ = _xml(config, inventory=inventory)
                    descriptions = model_config.findall("sdf")
                    names = {(element.text or "").strip() for element in descriptions}
                    if names != {"model.sdf"}:
                        raise ValueError("REPLICA_MODEL_CONFIG_SELECTION_UNSUPPORTED")
                candidate = candidate / "model.sdf"
            visit(candidate, ancestors | {path})

    for path in paths:
        visit(path, frozenset())
    sources = dict(inventory)
    return sources


# 功能：
#   1. 绑定已验收原生渲染包及本次世界、飞机、相机来源，在运行专属目录生成隔离渲染配置。
#   2. 只替换 Sensors 系统并保留其他系统；生成前复核全部来源，不修改已安装 Runtime。
# 输入：
#   runtime_root：已验证的原生隔离渲染构建目录。
#   server_config：当前物理世界的服务器配置。
#   world_sdf：任务选定世界文件。
#   vehicle_sdf：任务选定飞机模型。
#   camera_sdf：必须确实位于飞机包含图中的相机模型。
#   resource_paths：保持当前搜索顺序的本地模型资源目录。
#   world_name：实际世界名称。
#   vehicle_name：实际飞机实体名称。
#   output：必须尚不存在的运行专属输出目录。
# 输出：
#   result：启动命令、环境、相机配置及来源绑定回执，不授予飞行资格。
def prepare_render_replica(*, runtime_root: Path, server_config: Path, world_sdf: Path,
                           vehicle_sdf: Path, camera_sdf: Path, resource_paths: tuple[Path, ...],
                           world_name: str, vehicle_name: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if any(type(name) is not str or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", name) is None
           for name in (world_name, vehicle_name)):
        raise ValueError("REPLICA_ENTITY_NAME_INVALID")
    runtime_root = runtime_root.resolve(strict=True)
    runtime = validate_replica_runtime(runtime_root)
    inventory: dict[str, str] = {}
    world_tree, _ = _xml(world_sdf, inventory=inventory)
    world = world_tree.find("world")
    # 世界显式声明系统时会覆盖 server.config，不能假装新隔离渲染系统已经被选中。
    if world is None or world.get("name") != world_name or world.findall("plugin"):
        raise ValueError("REPLICA_WORLD_SERVER_CONFIG_NOT_EXCLUSIVE")
    sdf_sources = _include_graph((world_sdf, vehicle_sdf), resource_paths, inventory=inventory)
    if str(camera_sdf.resolve(strict=True)) not in inventory:
        raise ValueError("REPLICA_CAMERA_NOT_IN_ACTUAL_INCLUDE_GRAPH")
    camera_tree, camera_bytes = _xml(camera_sdf, inventory=inventory)
    camera = camera_configuration(camera_bytes)
    tree, server_bytes = _xml(server_config, inventory=inventory)
    plugins = tree.find("plugins")
    matches = tree.findall(f"./plugins/plugin[@name='{_SENSORS}']")
    if tree.tag != "server_config" or plugins is None or len(matches) != 1:
        raise ValueError("REPLICA_SERVER_SENSOR_SYSTEM_NOT_UNIQUE")
    for item in plugins:
        if item is matches[0]:
            continue
        library = item.get("filename", "").lower()
        if any(word in library for word in (
                "sensors-system", "render-preparation", "render-source")):
            raise ValueError("REPLICA_SERVER_CONTAINS_OTHER_RENDER_SYSTEM")
    selected = matches[0]
    if (selected.get("entity_type") != "world"
            or selected.get("entity_name") not in ("*", world_name)
            or selected.findtext("render_engine", "ogre2") != "ogre2"):
        raise ValueError("REPLICA_SERVER_SENSOR_CONFIGURATION_UNSUPPORTED")
    before = [ET.tostring(item) for item in plugins if item is not selected]
    plugins.remove(selected)
    epoch = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    check_plain_plugin_path(output)
    output = output.resolve()
    scene_topic = "/dronedream/replica/" + epoch + "/scene"
    plugin = ET.SubElement(plugins, "plugin", entity_name=world_name, entity_type="world",
        filename=str(runtime_root / SOURCE_LIBRARY), name="dronedream::RenderSceneSource")
    for key, value in {"epoch": epoch, "snapshot_topic": scene_topic,
                       "receipt": str(output / "source.json")}.items():
        ET.SubElement(plugin, key).text = value
    if before != [ET.tostring(item) for item in plugins if item is not plugin]:
        raise ValueError("REPLICA_NON_RENDER_SYSTEMS_CHANGED")
    rgb_topic = f"/world/{world_name}/model/{vehicle_name}/link/camera_link/sensor/IMX214/image"
    # 主题和相机标定使用同一次读取的字节，不从可能已经变化的磁盘文件再次提取字段。
    sensors = {s.get("name"): s for s in camera_tree.findall("model/link/sensor")}
    if ((sensors["IMX214"].findtext("topic", "").strip() not in ("", rgb_topic))
            or sensors["StereoOV7251"].findtext("topic", "").strip() not in (
                "depth_camera", "/depth_camera")):
        raise ValueError("REPLICA_CAMERA_TOPICS_UNSUPPORTED")
    for name, digest in inventory.items():
        try:
            current = hashlib.sha256(read_plugin_file(Path(name), limit=_SOURCE_BYTES)).hexdigest()
        except (OSError, ValueError) as error:
            raise ValueError("REPLICA_SOURCE_CHANGED") from error
        if current != digest:
            raise ValueError("REPLICA_SOURCE_CHANGED")
    output.mkdir(parents=True, exist_ok=False)
    config = output / "server.config"
    with config.open("xb") as stream:
        stream.write(ET.tostring(tree, encoding="utf-8", xml_declaration=True))
    command = [str(runtime_root / EXECUTABLE), "--epoch", epoch, "--world", world_name,
               "--snapshot-topic", scene_topic, "--rgb-topic", rgb_topic,
               "--depth-topic", "/depth_camera", "--sensors-plugin", runtime["sensors_plugin"],
               "--receipt", str(output / "replica.json"), "--duration-seconds", "0"]
    result = {"epoch": epoch, "command": command, "rgb_topic": rgb_topic,
              "depth_topic": "/depth_camera", "camera_configuration": camera,
              "input_sdf_files": sdf_sources,
              "input_server_config_sha256": hashlib.sha256(server_bytes).hexdigest(),
              "server_config_sha256": file_hash(config),
              "runtime_receipt_sha256": file_hash(runtime_root / RECEIPT),
              "binaries": runtime["binaries"], "non_render_systems_preserved": True,
              "qualification_granted": False, "installed_runtime_modified": False,
              "environment": {"GZ_SIM_SERVER_CONFIG_PATH": str(config)}}
    _write_new(output / "deployment.json", result)
    return result


# 功能：
#   在未解锁启动阶段验证 RGB 和深度图像的真实布局及连续来源身份，并退订、排空回调。
# 输入：
#   deployment：选定渲染纪元、主题和相机尺寸的部署描述。
#   timeout_seconds：等待两路图像就绪的最大秒数。
# 输出：
#   result：启动就绪、接收和拒绝统计及关闭回执，不代表持续控制权限。
def wait_for_replica_images(deployment: dict, *, timeout_seconds: float = 90) -> dict:
    if (type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= threading.TIMEOUT_MAX):
        raise ValueError("REPLICA_STARTUP_TIMEOUT_INVALID")
    deployment = copy_json(deployment, limit=_REPORT_BYTES)
    if type(deployment) is not dict:
        raise ValueError("REPLICA_STARTUP_DEPLOYMENT_INVALID")
    epoch = deployment.get("epoch")
    topics = deployment.get("rgb_topic"), deployment.get("depth_topic")
    cameras = deployment.get("camera_configuration")
    if (type(epoch) is not str or re.fullmatch(r"[0-9a-f]{64}", epoch) is None
            or any(type(topic) is not str or not 1 <= len(topic) <= 1024
                   or not topic.strip() or "\0" in topic for topic in topics)
            or topics[0] == topics[1] or type(cameras) is not dict):
        raise ValueError("REPLICA_STARTUP_DEPLOYMENT_INVALID")
    for name in ("IMX214", "StereoOV7251"):
        settings = cameras.get(name)
        if type(settings) is not dict or any(
            type(settings.get(axis)) is not int or not 1 <= settings[axis] <= 8192
            for axis in ("width", "height")
        ):
            raise ValueError("REPLICA_STARTUP_DEPLOYMENT_INVALID")
    # 所有静态输入先复核完毕，非法纪元、重复主题或缺失尺寸不能创建半初始化的原生节点。
    from gz.msgs10.image_pb2 import Image
    from gz.transport13 import Node

    from .gazebo_subscriptions import GazeboSubscriptions, subscription_shutdown_is_complete
    from .sensor_frame_clock import SensorFrameClock

    lock = threading.Lock()
    node = Node()
    subscriptions = GazeboSubscriptions(node)
    clocks = {k: SensorFrameClock(expected_scene_epoch=deployment["epoch"])
              for k in ("rgb", "depth")}
    counts, consecutive, last_source, rejected, failure = {}, {}, {}, {}, []

    # 功能：
    #   在接收锁内检验当前纪元、原始时效及实际像素布局，累计连续有效帧并保留拒绝原因。
    # 输入：
    #   kind：固定为 rgb 或 depth 的本地订阅类别。
    #   message：原生相机消息。
    # 输出：
    #   None：不返回业务数据。
    def received(kind, message):
        arrival, mono = time.time_ns(), time.monotonic()
        with lock:
            try:
                frame = clocks[kind].admit(message, received_unix_ns=arrival,
                                           received_monotonic_seconds=mono)
                name = "IMX214" if kind == "rgb" else "StereoOV7251"
                expected = deployment["camera_configuration"][name]
                if (message.width, message.height) != (expected["width"], expected["height"]):
                    raise RuntimeError("REPLICA_ACTUAL_IMAGE_DIMENSIONS_MISMATCH")
                channels = 3 if kind == "rgb" else 4
                pixel_format = "RGB_INT8" if kind == "rgb" else "R_FLOAT32"
                if (message.pixel_format_type != Image.DESCRIPTOR.fields_by_name[
                        "pixel_format_type"].enum_type.values_by_name[pixel_format].number
                        or message.step < message.width * channels
                        or len(message.data) != message.step * message.height):
                    raise RuntimeError("REPLICA_ACTUAL_IMAGE_FORMAT_INVALID")
                counts[kind] = counts.get(kind, 0) + 1
                consecutive[kind] = (consecutive.get(kind, 0) + 1
                    if kind in last_source and 0 < frame.source_unix_ns - last_source[kind]
                        <= 250_000_000 else 1)
                last_source[kind] = frame.source_unix_ns
            except ValueError as error:
                # 被拒绝的帧会打断连续有效序列，不能用失败前后的零散好帧拼出启动就绪。
                consecutive[kind] = 0
                key = str(error)[:128]
                rejected[key] = rejected.get(key, 0) + 1
            except (RuntimeError, KeyError) as error:
                if not failure:
                    failure.append(str(error)[:256])

    ready = False
    deadline = time.monotonic() + timeout_seconds
    try:
        for kind in ("rgb", "depth"):
            subscriptions.subscribe(Image, deployment[kind + "_topic"],
                                    lambda message, kind=kind: received(kind, message))
        while time.monotonic() < deadline:
            with lock:
                now = time.time_ns()
                ready = not failure and all(consecutive.get(kind, 0) >= 5 and
                    0 <= now - last_source[kind] <= 250_000_000 for kind in ("rgb", "depth"))
                if ready or failure:
                    break
            time.sleep(.02)
    finally:
        shutdown = subscriptions.close()
    with lock:
        result = {"ready": ready and subscription_shutdown_is_complete(shutdown),
            "epoch": deployment["epoch"],
            "accepted": dict(counts), "rejected": dict(rejected), "failure": list(failure),
            "consecutive_source_frames": dict(consecutive),
            "shutdown": shutdown, "qualification_granted": False,
            "startup_only_not_control_authority": True}
    return result


# 功能：
#   有界读取严格部署 JSON，执行未解锁图像就绪检查并独占保存回执。
# 输入：
#   无：部署路径与回执路径从命令行参数读取。
# 输出：
#   status：就绪为 0，未就绪为 1；非法输入抛出异常。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    deployment = decode_json(read_plugin_file(args.deployment, limit=_REPORT_BYTES),
                             limit=_REPORT_BYTES)
    result = wait_for_replica_images(deployment)
    _write_new(args.receipt, result)
    status = 0 if result["ready"] else 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
