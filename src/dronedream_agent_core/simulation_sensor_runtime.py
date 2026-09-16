"""Explicit simulated sensor transport; never a live estimator correction.

World field is computed from the same geographical table as PX4. The native plugin
simulates sensor mount, attitude, sampling and noise before adapting the known
PX4 GZ wire convention. A different bridge implementation must be qualified
explicitly rather than silently inheriting this conversion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from dronedream_plugin_sdk.protocol import decode_json

from .plugin_files import MAX_PLUGIN_FILE_BYTES, hash_plugin_file, read_plugin_file

MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_WORLD_BYTES = 32 * 1024 * 1024
MAX_BRIDGE_SOURCE_BYTES = 8 * 1024 * 1024

MAGNETIC_SENSOR_CONTRACT = {
    "field": "PX4-identical-world-magnetic-model-ENU-tesla",
    "measurement": "native-mounted-sensor-FLU-tesla-with-noise",
    "noise": "PX4-Harmonic-source-gauss-converted-to-tesla",
    "wire": "px4-gz-fimex-gauss",
    "decode_FRD_gauss": ["-wire.y", "-wire.x", "wire.z"],
    "estimator_heading_fit": False,
}
MAGNETIC_SENSOR_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    MAGNETIC_SENSOR_CONTRACT, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()


# 功能：
#   复用有界普通文件读取规则流式计算摘要，检测读取过程中的文件替换或增长。
# 输入：
#   path：准备核对的运行包、源文件或证据文件。
# 输出：
#   digest：实际读取内容的 SHA-256 摘要。
def _sha(path: Path) -> str:
    digest = hash_plugin_file(path, limit=MAX_PLUGIN_FILE_BYTES)
    return digest


# 功能：
#   从同一份有界字节快照解析无歧义 JSON 对象并计算摘要，拒绝重复键与非有限数。
# 输入：
#   path：运行包或部署回执的路径。
# 输出：
#   result：已验证的对象及其原始字节摘要。
def _receipt_snapshot(path: Path) -> tuple[dict, str]:
    raw = read_plugin_file(path, limit=MAX_RECEIPT_BYTES)
    receipt = decode_json(raw, limit=MAX_RECEIPT_BYTES)
    if not isinstance(receipt, dict):
        raise ValueError("SIMULATION_SENSOR_RECEIPT_OBJECT_REQUIRED")
    result = receipt, hashlib.sha256(raw).hexdigest()
    return result


# 功能：
#   核对已解析回执的协议、测试标记、运行文件摘要及可选当前源码清单，不运行传感器。
# 输入：
#   runtime_root：待检查的原生运行包目录。
#   receipt：已经通过有界、无歧义解析的回执对象。
#   source_root：可选的当前源码目录，用于拒绝未经重新构建的旧包。
# 输出：
#   None：不返回业务数据。
def _validate_native_receipt(runtime_root: Path, receipt: dict, source_root: Path | None) -> None:
    library = runtime_root / "libdronedream-magnetometer.so"
    if (not isinstance(receipt, dict)
            or receipt.get("wire_contract") != "px4-gz-fimex-gauss"
            or receipt.get("sensor_contract_sha256") != MAGNETIC_SENSOR_CONTRACT_SHA256
            or receipt.get("library_sha256") != _sha(library)
            or receipt.get("field_probe_sha256") != _sha(runtime_root / "magnetic-field-probe")
            or receipt.get("px4_magnetic_table_sha256") != _sha(
                runtime_root / "geo_magnetic_tables.hpp")
            or receipt.get("native_tests_passed") is not True):
        raise ValueError("SIMULATION_NATIVE_SENSOR_RUNTIME_NOT_VERIFIED")
    if source_root is not None:
        sources = {path.name: _sha(path)
                   for path in sorted(source_root.iterdir()) if path.is_file()}
        if receipt.get("sources") != sources:
            raise ValueError("SIMULATION_NATIVE_SENSOR_SOURCE_MISMATCH")


# 功能：
#   从运行包回执快照核对原生插件、磁场探针及地磁表，必要时核对当前源码。
# 输入：
#   runtime_root：待检查的原生运行包目录。
#   source_root：可选的当前原生源码目录。
# 输出：
#   receipt：内容与已核对文件一致的回执。
def validate_native_sensor_runtime(runtime_root: Path, *, source_root: Path | None = None) -> dict:
    receipt, _ = _receipt_snapshot(runtime_root / "native-sensor-runtime.json")
    _validate_native_receipt(runtime_root, receipt, source_root)
    return receipt


# 功能：
#   从有界 SDF 字节快照读取唯一世界节点，使解析内容与证据摘要对应同一次读取。
# 输入：
#   world_sdf：仿真世界描述文件。
# 输出：
#   result：唯一世界节点及源字节摘要。
def _world_snapshot(world_sdf: Path) -> tuple[ET.Element, str]:
    raw = read_plugin_file(world_sdf, limit=MAX_WORLD_BYTES)
    worlds = ET.fromstring(raw).findall("world")
    if len(worlds) != 1:
        raise ValueError("SIMULATION_MAGNETIC_WORLD_MISSING")
    result = worlds[0], hashlib.sha256(raw).hexdigest()
    return result


# 功能：
#   核对世界为真北对齐 ENU，校验磁场量纲和水平分量，并计算真北以东为正的磁偏角。
# 输入：
#   world：已经从唯一世界快照获得的节点。
# 输出：
#   result：世界磁场三元组和磁偏角度数。
def _declared_field(world: ET.Element) -> tuple[tuple[float, ...], float]:
    spherical = world.find("spherical_coordinates")
    if (spherical is None or spherical.findtext("world_frame_orientation", "ENU") != "ENU"
            or float(spherical.findtext("heading_deg", "0")) != 0):
        raise ValueError("SIMULATION_MAGNETIC_WORLD_NOT_NORTH_ALIGNED_ENU")
    field = tuple(float(value) for value in world.findtext("magnetic_field", "").split())
    if (len(field) != 3 or not all(math.isfinite(value) for value in field)
            or not 1e-6 <= math.hypot(*field) <= 1e-3
            or math.hypot(*field[:2]) < 1e-6):
        raise ValueError("SIMULATION_MAGNETIC_FIELD_INVALID")
    # Declination is east of true north, not ENU yaw and not a fitted heading.
    result = field, math.degrees(math.atan2(field[0], field[1]))
    return result


# 功能：
#   从世界文件读取声明的物理磁场，不用飞机真实朝向拟合磁偏角。
# 输入：
#   world_sdf：唯一世界的 SDF 文件。
# 输出：
#   result：世界 ENU 磁场与真北以东为正的磁偏角。
def declared_magnetic_field(world_sdf: Path) -> tuple[tuple[float, ...], float]:
    world, _ = _world_snapshot(world_sdf)
    result = _declared_field(world)
    return result


# 功能：
#   1. 绑定当前世界、PX4 桥接源码、原生插件和相同地磁表，生成本次运行专用配置。
#   2. 仅替换唯一磁力计发布系统；核对探针结果及准备期间的文件变化，不覆盖已安装资产。
# 输入：
#   world_sdf：本次仿真使用的世界文件。
#   px4_root：当前 PX4 源码目录。
#   runtime_root：待使用的已构建原生传感器包。
#   output：必须尚不存在的运行配置目录。
#   source_root：可选的当前原生源码目录。
# 输出：
#   result：本次部署的输入摘要、实际磁场和启动环境配置。
def prepare_sensor_runtime(
    *, world_sdf: Path, px4_root: Path, runtime_root: Path, output: Path,
    source_root: Path | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    world, world_digest = _world_snapshot(world_sdf)
    declared_field, _ = _declared_field(world)
    source = px4_root / "src/modules/simulation/gz_bridge/GZBridge.cpp"
    bridge_raw = read_plugin_file(source, limit=MAX_BRIDGE_SOURCE_BYTES)
    bridge = bridge_raw.decode("utf-8")
    if bridge.count("void GZBridge::magnetometerCallback(") != 1:
        raise ValueError("SIMULATION_MAGNETIC_PX4_WIRE_CONTRACT_UNSUPPORTED")
    body = bridge.split("void GZBridge::magnetometerCallback(", 1)[-1].split(
        "void GZBridge::", 1)[0]
    assignments = {axis: re.findall(rf"report\.{axis}\s*=\s*([^;]+);", body)
                   for axis in "xyz"}
    expected = {"x": ["-msg.field_tesla().y()"], "y": ["-msg.field_tesla().x()"],
                "z": ["msg.field_tesla().z()"]}
    if assignments != expected:
        raise ValueError("SIMULATION_MAGNETIC_PX4_WIRE_CONTRACT_UNSUPPORTED")
    receipt_path = runtime_root / "native-sensor-runtime.json"
    runtime_receipt, receipt_digest = _receipt_snapshot(receipt_path)
    _validate_native_receipt(runtime_root, runtime_receipt, source_root)
    table = px4_root / "src/lib/world_magnetic_model/geo_magnetic_tables.hpp"
    if runtime_receipt.get("px4_magnetic_table_sha256") != _sha(table):
        raise ValueError("SIMULATION_NATIVE_MAGNETIC_MODEL_MISMATCH")
    spherical = world.find("spherical_coordinates")
    latitude = float(spherical.findtext("latitude_deg", "nan"))
    longitude = float(spherical.findtext("longitude_deg", "nan"))
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("SIMULATION_MAGNETIC_GEOGRAPHIC_ORIGIN_INVALID")
    probe = subprocess.run([str(runtime_root / "magnetic-field-probe"),
                            str(latitude), str(longitude)], check=True, capture_output=True,
                           text=True, timeout=5)
    magnetic = decode_json(probe.stdout, limit=64 * 1024)
    if not isinstance(magnetic, dict):
        raise ValueError("SIMULATION_NATIVE_MAGNETIC_PROBE_INVALID")
    declination = magnetic.get("declination_deg")
    field = magnetic.get("field_enu_tesla")
    if (type(declination) not in (float, int) or not -180 <= declination <= 180
            or not isinstance(field, list) or len(field) != 3
            or any(type(value) not in (float, int) or not math.isfinite(value) for value in field)
            or not 1e-6 <= math.hypot(*field) <= 1e-3):
        raise ValueError("SIMULATION_NATIVE_MAGNETIC_PROBE_INVALID")
    library = runtime_root / "libdronedream-magnetometer.so"
    config_source = px4_root / "src/modules/simulation/gz_bridge/server.config"
    config_raw = read_plugin_file(config_source, limit=MAX_RECEIPT_BYTES)
    tree = ET.ElementTree(ET.fromstring(config_raw))
    matches = tree.findall("./plugins/plugin[@name='gz::sim::systems::Magnetometer']")
    if len(matches) != 1:
        raise ValueError("SIMULATION_MAGNETIC_SYSTEM_NOT_UNIQUE")
    # The replacement is exclusive: never run two publishers on the same wire.
    plugin = matches[0]
    plugin.clear()
    plugin.attrib.update(entity_name="*", entity_type="world", filename=str(library.resolve()),
                         name="dronedream::Magnetometer")
    ET.SubElement(plugin, "wire_contract").text = "px4-gz-fimex-gauss"
    ET.SubElement(plugin, "source_noise_units").text = "gauss"
    ET.SubElement(plugin, "field_provider").text = "px4-world-magnetic-model"
    # 摘要来自真正参与解析的字节；准备期间输入变化则拒绝，而不是事后散列新文件冒充旧输入。
    if any(_sha(path) != expected_digest for path, expected_digest in (
        (world_sdf, world_digest), (source, hashlib.sha256(bridge_raw).hexdigest()),
        (receipt_path, receipt_digest), (config_source, hashlib.sha256(config_raw).hexdigest()),
        (table, runtime_receipt["px4_magnetic_table_sha256"]),
    )):
        raise ValueError("SIMULATION_SENSOR_INPUT_CHANGED")
    _validate_native_receipt(runtime_root, runtime_receipt, source_root)
    output.mkdir(parents=True)
    config = output / "server.config"
    tree.write(config, encoding="utf-8", xml_declaration=True)
    result = {
        # 内存回执直接参与严格 JSON 证据；在数值生产边界明确使用数组，不放宽下游校验。
        "world_sha256": world_digest, "declared_static_field_enu_tesla": list(declared_field),
        "active_field_provider": "px4-world-magnetic-model",
        "world_field_enu_tesla": magnetic["field_enu_tesla"],
        "px4_magnetic_table_sha256": runtime_receipt["px4_magnetic_table_sha256"],
        "declination_deg": declination,
        "px4_bridge_source_sha256": hashlib.sha256(bridge_raw).hexdigest(),
        "native_runtime_receipt_sha256": receipt_digest,
        "native_library_sha256": runtime_receipt["library_sha256"],
        "server_config_sha256": _sha(config),
        "native_sources_verified": source_root is not None,
        "truth_heading_fit": False, "source_noise_units": "gauss",
        "environment": {"GZ_SIM_SERVER_CONFIG_PATH": str(config),
                        "PX4_PARAM_EKF2_DECL_TYPE": "3",
                        "PX4_PARAM_EKF2_MAG_DECL": format(declination, ".12g")},
    }
    (output / "sensor-deployment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


# 功能：
#   1. 复核配置中的有限磁偏角及飞控实际读回值，拒绝无效类型或超出误差范围的参数。
#   2. 读回后确认部署文件仍与初始快照相同，不能用旧配置比较却记录新配置摘要。
# 输入：
#   client：提供整型、浮点参数异步读回接口的飞控客户端。
#   deployment_path：本次仿真部署回执路径。
# 输出：
#   result：参数读回结果及与本次比较对应的部署文件摘要。
async def verify_magnetic_parameters(client, deployment_path: Path) -> dict[str, object]:
    deployment, deployment_digest = _receipt_snapshot(deployment_path)
    declination = deployment.get("declination_deg")
    if type(declination) not in (int, float) or not -180 <= declination <= 180:
        raise ValueError("SIMULATION_SENSOR_DECLINATION_INVALID")
    actual_type = await client.get_param_int("EKF2_DECL_TYPE")
    actual_declination = await client.get_param_float("EKF2_MAG_DECL")
    if (type(actual_type) is not int or actual_type != 3
            or type(actual_declination) not in (int, float)
            or not -180 <= actual_declination <= 180
            or abs(actual_declination - declination) > .01):
        raise RuntimeError("NATIVE_MAGNETIC_CONFIGURATION_READBACK_FAILED:"
                           f"source={actual_type},declination={actual_declination}")
    # 这是读回结束时的一致性核对，不声称能阻止检查后恶意进程再次替换文件。
    if _sha(deployment_path) != deployment_digest:
        raise RuntimeError("SIMULATION_SENSOR_DEPLOYMENT_CHANGED")
    result = {"verified": True, "declination_source": actual_type,
            "declination_deg": actual_declination, "truth_heading_fit": False,
              "deployment_sha256": deployment_digest}
    return result
