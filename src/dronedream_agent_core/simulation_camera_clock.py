"""Prepare a run-owned native-camera clock relay, never a ground-truth pose source."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

from dronedream_plugin_sdk.protocol import decode_json
from .plugin_files import read_plugin_file, check_plain_plugin_path
from .xml_values import parse_xml

CAMERA_CLOCK_FILES = ("camera-clock-runtime.json", "libdronedream-camera-clock.so")


# 功能：有界读取并核对时钟插件、配置格式和当前源码身份，不启动仿真。
# 输入：runtime_root：组件目录；source_root：构建时必需的源码核对目录，安装预检可省略。
# 输出：receipt、raw：同次核验的契约与二进制字节，供暂存和独占运行副本使用。
def validate_native_camera_clock(runtime_root: Path, *, source_root: Path | None = None):
    receipt = decode_json(read_plugin_file(runtime_root / CAMERA_CLOCK_FILES[0], limit=65536),
                          limit=65536)
    raw = read_plugin_file(runtime_root / CAMERA_CLOCK_FILES[1], limit=16*1024*1024)
    if (type(receipt) is not dict
            or receipt.get("schema_version") != "dronedream.native-camera-clock.v1"
            or receipt.get("library_sha256") != hashlib.sha256(raw).hexdigest()
            or receipt.get("clock_contract") != "exact-native-sim-tick-preupdate-v1"):
        raise ValueError("NATIVE_CAMERA_CLOCK_RUNTIME_INVALID")
    sources = receipt.get("sources")
    if (type(sources) is not dict or set(sources) != {
            "camera_clock.cpp", "capture_clock.hpp", "CMakeLists.txt", "capture_clock_test.cpp"}
            or any(type(v) is not str or not re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values())):
        raise ValueError("NATIVE_CAMERA_CLOCK_SOURCE_MANIFEST_INVALID")
    if source_root is not None:
        for name, digest in sources.items():
            if hashlib.sha256(read_plugin_file(source_root/name, limit=1024*1024)).hexdigest() != digest:
                raise ValueError("NATIVE_CAMERA_CLOCK_SOURCE_CHANGED")
    return receipt, raw


# 功能：核验相机时钟二进制和来源摘要，生成独立运行配置，不修改 Runtime 安装目录。
# 输入：已构建组件根目录、原生服务器配置、两个真实相机主题及新的输出目录。
# 输出：唯一 epoch、转发主题、组件摘要和启动环境；缺组件或混源配置在起飞前失败。
def prepare_native_camera_clock(*, runtime_root: Path, server_config: Path,
                                rgb_topic: str, depth_topic: str, output: Path,
                                source_root: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    check_plain_plugin_path(output)
    for topic in (rgb_topic, depth_topic):
        if type(topic) is not str or not re.fullmatch(r"/[A-Za-z0-9_/-]{1,510}", topic):
            raise ValueError("NATIVE_CAMERA_CLOCK_TOPIC_INVALID")
    if rgb_topic == depth_topic:
        raise ValueError("NATIVE_CAMERA_CLOCK_TOPIC_ALIAS")
    receipt, raw = validate_native_camera_clock(runtime_root, source_root=source_root)
    library_sha = hashlib.sha256(raw).hexdigest()
    config_bytes = read_plugin_file(server_config, limit=1024*1024)
    tree = ET.ElementTree(parse_xml(config_bytes, maximum_bytes=1024*1024, maximum_elements=4096))
    plugins = tree.find("plugins")
    if (tree.getroot().tag != 'server_config' or len(tree.findall('plugins')) != 1
            or plugins is None or tree.findall(".//plugin[@name='dronedream::NativeCameraClock']")):
        raise ValueError("NATIVE_CAMERA_CLOCK_SERVER_CONFIG_INVALID")
    epoch = hashlib.sha256(uuid4().bytes + config_bytes + raw).hexdigest()
    topics = {kind: f"/dronedream/native_camera/{epoch}/{kind}" for kind in ("rgb", "depth")}
    # 加载同次校验的字节副本，避免部署后源构建目录发生变化造成摘要与实际加载物不同。
    library = output / CAMERA_CLOCK_FILES[1]
    plugin = ET.SubElement(plugins, "plugin", entity_name="*", entity_type="world",
                           filename=str(library.resolve()), name="dronedream::NativeCameraClock")
    for name, value in {"epoch": epoch, "rgb_input": rgb_topic, "depth_input": depth_topic,
                        "rgb_output": topics["rgb"], "depth_output": topics["depth"]}.items():
        ET.SubElement(plugin, name).text = value
    output.mkdir(parents=True)
    with library.open('xb') as stream:
        stream.write(raw)
    config = output / "server.config"
    tree.write(config, encoding="utf-8", xml_declaration=True)
    result = {"epoch": epoch, "rgb_topic": topics["rgb"], "depth_topic": topics["depth"],
              "library_sha256": library_sha, "clock_contract": receipt["clock_contract"],
              "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "qualification_granted": False, "truth_pose_used": False,
              "environment": {"GZ_SIM_SERVER_CONFIG_PATH": str(config)}}
    (output/"camera-clock-deployment.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result
