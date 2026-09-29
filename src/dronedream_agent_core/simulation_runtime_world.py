"""Create a run-owned Gazebo world whose required server plugins really load."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from .plugin_files import read_plugin_file, check_plain_plugin_path
from .xml_values import parse_xml

_CUSTOM_MAGNETOMETER = "dronedream::Magnetometer"
_STOCK_MAGNETOMETER = "gz::sim::systems::Magnetometer"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# 功能：把最终服务器配置中的系统插件合并进本次运行的世界副本。
# 背景：Gazebo Harmonic 在世界已声明任一插件时会忽略 GZ_SIM_SERVER_CONFIG_PATH。
# 输入：只读世界、已逐层生成并验证的最终 server.config、尚不存在的输出目录。
# 输出：运行世界路径和可审计回执；不修改用户地图、Runtime 或原始配置。
def materialize_runtime_world_plugins(*, world_sdf: Path, server_config: Path,
                                      output: Path) -> tuple[Path, dict[str, object]]:
    if output.exists():
        raise FileExistsError(output)
    check_plain_plugin_path(output)
    world_raw = read_plugin_file(world_sdf, limit=64 * 1024**2)
    config_raw = read_plugin_file(server_config, limit=4 * 1024**2)
    world_tree = ET.ElementTree(parse_xml(
        world_raw, maximum_bytes=64 * 1024**2, maximum_elements=1_000_000))
    config_tree = ET.ElementTree(parse_xml(
        config_raw, maximum_bytes=4 * 1024**2, maximum_elements=16_384))
    worlds = world_tree.findall("./world")
    plugins = config_tree.findall("./plugins/plugin")
    if world_tree.getroot().tag != "sdf" or len(worlds) != 1 or not plugins:
        raise ValueError("SIMULATION_RUNTIME_WORLD_INPUT_INVALID")
    world = worlds[0]
    existing = world.findall("./plugin")
    names = [plugin.get("name") for plugin in existing]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("SIMULATION_RUNTIME_WORLD_PLUGIN_INVALID")
    custom_magnetometer = any(p.get("name") == _CUSTOM_MAGNETOMETER for p in plugins)
    if custom_magnetometer:
        for plugin in list(world.findall("./plugin")):
            if plugin.get("name") in {_CUSTOM_MAGNETOMETER, _STOCK_MAGNETOMETER}:
                world.remove(plugin)
        names = [plugin.get("name") for plugin in world.findall("./plugin")]
    inserted: list[str] = []
    for source in plugins:
        name = source.get("name")
        filename = source.get("filename")
        if not isinstance(name, str) or not name or not isinstance(filename, str) or not filename:
            raise ValueError("SIMULATION_RUNTIME_WORLD_PLUGIN_INVALID")
        if name in names:
            continue
        plugin = copy.deepcopy(source)
        # entity_name/entity_type are server-config routing attributes, not world-plugin fields.
        plugin.attrib.pop("entity_name", None)
        plugin.attrib.pop("entity_type", None)
        world.append(plugin)
        names.append(name)
        inserted.append(name)
    if custom_magnetometer and _STOCK_MAGNETOMETER in names:
        raise ValueError("SIMULATION_RUNTIME_WORLD_MAGNETOMETER_CONFLICT")
    required_custom = sorted(p.get("name") for p in plugins
                             if str(p.get("name", "")).startswith("dronedream::"))
    if any(name not in names for name in required_custom):
        raise ValueError("SIMULATION_RUNTIME_WORLD_PLUGIN_MISSING")
    output.mkdir(parents=True)
    runtime_world = output / "world.sdf"
    world_tree.write(runtime_world, encoding="utf-8", xml_declaration=True)
    result: dict[str, object] = {
        "schema_version": "dronedream.simulation-runtime-world.v1",
        "source_world_sha256": _sha(world_raw),
        "source_server_config_sha256": _sha(config_raw),
        "runtime_world_sha256": _sha(runtime_world.read_bytes()),
        "inserted_plugins": inserted,
        "required_custom_plugins": required_custom,
        "stock_magnetometer_removed": custom_magnetometer,
        "qualification_granted": False,
    }
    (output / "runtime-world.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runtime_world, result
