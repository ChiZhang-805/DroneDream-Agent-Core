import asyncio
import hashlib
import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from dronedream_agent_core.simulation_sensor_runtime import (
    MAGNETIC_SENSOR_CONTRACT_SHA256,
    declared_magnetic_field,
    prepare_sensor_runtime,
    validate_native_sensor_runtime,
    verify_magnetic_parameters,
)
from dronedream_plugin_sdk.protocol import encode_json


# 功能：
#   用固定输出替代原生磁场程序，只测试解析和打包契约，不把此结果当作原生或飞行验收。
# 输入：
#   monkeypatch：替换子进程调用的测试工具。
# 输出：
#   None：不返回业务数据。
@pytest.fixture(autouse=True)
def compiled_field_probe(monkeypatch):
    # Parser/packaging unit fixture only. Native CTest and Gazebo exercise the
    # compiled field provider independently; this fixture proves no flight.
    monkeypatch.setattr("dronedream_agent_core.simulation_sensor_runtime.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(stdout=json.dumps({
                            "declination_deg": -6.129664,
                            "field_enu_tesla": [-3e-6, 23e-6, -42e-6]})))


# 功能：
#   构造世界、PX4 桥接、地磁表及摘要一致的原生包夹具，不使用实际安装目录。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   result：准备传感器运行环境所需的路径参数。
def inputs(tmp_path):
    world = tmp_path / "world.sdf"
    world.write_text('<sdf><world><magnetic_field>0.000006 0.000023 -0.000042</magnetic_field>'
                     '<spherical_coordinates><world_frame_orientation>ENU</world_frame_orientation>'
                     '<latitude_deg>30.27415</latitude_deg><longitude_deg>120.15515</longitude_deg>'
                     '</spherical_coordinates></world></sdf>')
    px4 = tmp_path / "px4"
    bridge = px4 / "src/modules/simulation/gz_bridge"
    bridge.mkdir(parents=True)
    (bridge / "GZBridge.cpp").write_text('void GZBridge::magnetometerCallback() { '
        'report.x = -msg.field_tesla().y(); report.y = -msg.field_tesla().x(); '
        'report.z = msg.field_tesla().z(); }')
    (bridge / "server.config").write_text('<server_config><plugins>'
        '<plugin name="gz::sim::systems::Physics" filename="physics"/>'
        '<plugin name="gz::sim::systems::Magnetometer" filename="old"/>'
        '</plugins></server_config>')
    table = px4 / "src/lib/world_magnetic_model/geo_magnetic_tables.hpp"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"unit-model")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "libdronedream-magnetometer.so").write_bytes(b"unit-binary-not-real-plugin")
    (runtime / "magnetic-field-probe").write_bytes(b"unit-probe")
    (runtime / "geo_magnetic_tables.hpp").write_bytes(table.read_bytes())
    (runtime / "native-sensor-runtime.json").write_text(json.dumps({
        "wire_contract": "px4-gz-fimex-gauss", "native_tests_passed": True,
        "sensor_contract_sha256": MAGNETIC_SENSOR_CONTRACT_SHA256,
        "field_probe_sha256": hashlib.sha256(b"unit-probe").hexdigest(),
        "px4_magnetic_table_sha256": hashlib.sha256(b"unit-model").hexdigest(),
        "library_sha256": hashlib.sha256(b"unit-binary-not-real-plugin").hexdigest()}))
    result = {"world_sdf": world, "px4_root": px4, "runtime_root": runtime,
            "output": tmp_path / "run-sensors"}
    return result


# 功能：
#   核对静态磁场声明与探针结果各自留痕，唯一替换磁力计系统，并保持源文件不变。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   directory：包含普通名称或 hold 子串的运行目录名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("directory", ["sensor-run", "camera-hold-run"])
def test_magnetic_declination_is_declared_field_not_fitted_pose(tmp_path, directory):
    # The legitimate absolute plugin path may contain "old" (e.g. "hold").
    # Validate the actual plugin binding, not substrings anywhere in the XML.
    tmp_path = tmp_path / directory
    tmp_path.mkdir()
    args = inputs(tmp_path)
    field, declination = declared_magnetic_field(args["world_sdf"])
    assert field == (6e-6, 23e-6, -42e-6)
    assert declination == pytest.approx(math.degrees(math.atan2(6, 23)))
    old_world = args["world_sdf"].read_bytes()
    result = prepare_sensor_runtime(**args)
    # 内存返回值也会直接嵌入最终严格证据，不能仅验证默认 json.dumps 写盘成功。
    assert json.loads(encode_json(result)) == result
    assert result["truth_heading_fit"] is False
    assert result["environment"]["PX4_PARAM_EKF2_DECL_TYPE"] == "3"
    assert result["declination_deg"] == -6.129664
    assert result["active_field_provider"] == "px4-world-magnetic-model"
    assert args["world_sdf"].read_bytes() == old_world
    plugins = ET.parse(args["output"] / "server.config").findall("./plugins/plugin")
    assert [p.get("name") for p in plugins] == ["gz::sim::systems::Physics",
                                              "dronedream::Magnetometer"]
    assert [p.get("filename") for p in plugins] == [
        "physics", str((args["runtime_root"] / "libdronedream-magnetometer.so").resolve())]
    assert plugins[1].findtext("wire_contract") == "px4-gz-fimex-gauss"
    with pytest.raises(FileExistsError):
        prepare_sensor_runtime(**args)


# 功能：
#   检查缺失、非有限、无水平分量或单位量级错误的磁场在创建部署目录前被拒绝。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   field：替换 SDF 磁场声明的非法文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["", "NaN 1 0", "0 0 0", "0 0 .00004", "1 2 3", "1 2"])
def test_invalid_physical_field_rejected(tmp_path, field):
    args = inputs(tmp_path)
    tree = ET.parse(args["world_sdf"])
    tree.find("./world/magnetic_field").text = field
    tree.write(args["world_sdf"])
    with pytest.raises(ValueError, match="MAGNETIC_FIELD_INVALID"):
        prepare_sensor_runtime(**args)
    assert not args["output"].exists()


# 功能：
#   检查原生文件被改写或 PX4 接线符号变化时不会继续沿用旧校准协议。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_unrecognized_bridge_and_tampered_plugin_are_not_silently_used(tmp_path):
    args = inputs(tmp_path)
    (args["runtime_root"] / "libdronedream-magnetometer.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="NOT_VERIFIED"):
        prepare_sensor_runtime(**args)
    bridge = args["px4_root"] / "src/modules/simulation/gz_bridge/GZBridge.cpp"
    bridge.write_text(bridge.read_text().replace("-msg.field_tesla().x()", "msg.field_tesla().x()"))
    with pytest.raises(ValueError, match="WIRE_CONTRACT_UNSUPPORTED"):
        prepare_sensor_runtime(**args)
    assert not args["output"].exists()


# 功能：
#   用客户端读回验证磁偏角来源和误差门限，写出配置本身不能代替实际读回。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   source：模拟的飞控磁偏角来源参数。
#   offset：实际读回值与预期值的差。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source,offset", [(3, 0), (2, 0), (3, 1), (3, float("nan"))])
def test_actual_px4_parameter_readback_required(tmp_path, source, offset):
    args = inputs(tmp_path)
    result = prepare_sensor_runtime(**args)

    class Client:
        # 功能：
        #   验证被查询的参数名称，并返回本例的来源代码。
        # 输入：
        #   self：测试客户端实例。
        #   name：整型参数名。
        # 输出：
        #   source：本例设定的来源代码。
        async def get_param_int(self, name):
            assert name == "EKF2_DECL_TYPE"
            return source

        # 功能：
        #   返回带指定偏差的磁偏角，以隔离读回误差判断。
        # 输入：
        #   self：测试客户端实例。
        #   name：浮点参数名。
        # 输出：
        #   value：合成的磁偏角读回值。
        async def get_param_float(self, name):
            assert name == "EKF2_MAG_DECL"
            value = result["declination_deg"] + offset
            return value

    operation = verify_magnetic_parameters(Client(), args["output"] / "sensor-deployment.json")
    if source == 3 and offset == 0:
        assert asyncio.run(operation)["verified"]
    else:
        with pytest.raises(RuntimeError, match="READBACK_FAILED"):
            asyncio.run(operation)


# 功能：
#   检查 PX4 与运行包使用不同地磁表时拒绝生成配置。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_different_magnetic_table_cannot_be_mixed_with_runtime(tmp_path):
    args = inputs(tmp_path)
    table = args["px4_root"] / "src/lib/world_magnetic_model/geo_magnetic_tables.hpp"
    table.write_bytes(b"other")
    with pytest.raises(ValueError, match="MAGNETIC_MODEL_MISMATCH"):
        prepare_sensor_runtime(**args)
    assert not args["output"].exists()


# 功能：
#   检查运行包内地磁表被替换后回执摘要不能继续授权该包。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_packaged_native_table_is_bound_to_runtime(tmp_path):
    args = inputs(tmp_path)
    (args["runtime_root"] / "geo_magnetic_tables.hpp").write_bytes(b"other table")
    with pytest.raises(ValueError, match="NOT_VERIFIED"):
        validate_native_sensor_runtime(args["runtime_root"])


# 功能：
#   检查暂存仅复制四份已验证运行文件，保留地磁表，并拒绝覆盖目录或沿用旧源码产物。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_stage_only_verified_current_sources_and_retain_license_table(tmp_path):
    args = inputs(tmp_path)
    stage = runpy.run_path(str(Path(__file__).resolve().parents[1]
                              / "scripts/stage_native_sensor_runtime.py"))[
                                  "stage_native_sensor_runtime"]
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "magnetometer_system.cpp"
    source.write_bytes(b"unit-source")
    receipt_path = args["runtime_root"] / "native-sensor-runtime.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sources"] = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    output = tmp_path / "staged"
    stage(source=args["runtime_root"], output=output, source_root=source_root)
    assert {path.name for path in output.iterdir()} == {
        "native-sensor-runtime.json", "libdronedream-magnetometer.so",
        "magnetic-field-probe", "geo_magnetic_tables.hpp"}
    assert (output / "geo_magnetic_tables.hpp").read_bytes() == b"unit-model"
    with pytest.raises(FileExistsError):
        stage(source=args["runtime_root"], output=output, source_root=source_root)
    source.write_bytes(b"new source must be rebuilt")
    with pytest.raises(ValueError, match="SOURCE_MISMATCH"):
        stage(source=args["runtime_root"], output=tmp_path / "other", source_root=source_root)
    assert not (tmp_path / "other").exists()


# 功能：
#   检查启动前源码核对能识别未重新构建的变化，拒绝时不创建部署目录。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   changed：是否在生成回执后修改原生源码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changed", [False, True])
def test_launch_checks_native_sources_before_creating_deployment(tmp_path, changed):
    args = inputs(tmp_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "magnetometer_system.cpp"
    source.write_bytes(b"current-source")
    receipt_path = args["runtime_root"] / "native-sensor-runtime.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sources"] = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    if changed:
        source.write_bytes(b"new-source")
        with pytest.raises(ValueError, match="SOURCE_MISMATCH"):
            prepare_sensor_runtime(**args, source_root=source_root)
        assert not args["output"].exists()
    else:
        result = prepare_sensor_runtime(**args, source_root=source_root)
        assert result["native_sources_verified"] is True


# 功能：
#   检查非法或有歧义的预期磁偏角不能通过异步参数读回验收。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   value：含非有限数、重复字段、布尔或字符串的配置文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [
    '{"declination_deg": NaN}', '{"declination_deg": Infinity}',
    '{"declination_deg": 1e999}', '{"declination_deg": true}',
    '{"declination_deg": "1"}', '{"declination_deg": 999}',
    '{"declination_deg": 999, "declination_deg": 1}',
])
def test_invalid_expected_declination_cannot_qualify(tmp_path, value):
    from unittest.mock import AsyncMock

    path = tmp_path / "deployment.json"
    path.write_text(value, encoding="utf-8")
    client = SimpleNamespace(get_param_int=AsyncMock(return_value=3),
                             get_param_float=AsyncMock(return_value=1.))
    with pytest.raises(ValueError):
        asyncio.run(verify_magnetic_parameters(client, path))
    client.get_param_int.assert_not_called()
    client.get_param_float.assert_not_called()


# 功能：
#   模拟参数读取期间的部署配置替换，禁止旧值验收却记录新摘要。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_deployment_changed_during_parameter_readback_is_rejected(tmp_path):
    path = tmp_path / "deployment.json"
    path.write_text('{"declination_deg": 1}', encoding="utf-8")

    class Client:
        # 功能：
        #   在第一个异步读回点改写部署文件，模拟检查与使用之间的配置变化。
        # 输入：
        #   self：测试客户端实例。
        #   name：待读取的整型参数名称。
        # 输出：
        #   value：合法的手动磁偏角来源代码。
        async def get_param_int(self, name):
            assert name == "EKF2_DECL_TYPE"
            path.write_text('{"declination_deg": 2}', encoding="utf-8")
            value = 3
            return value

        # 功能：
        #   返回与旧文件相符的参数，确保失败来自文件替换而非数值超限。
        # 输入：
        #   self：测试客户端实例。
        #   name：待读取的浮点参数名称。
        # 输出：
        #   value：旧配置中的磁偏角。
        async def get_param_float(self, name):
            assert name == "EKF2_MAG_DECL"
            value = 1.
            return value

    with pytest.raises(RuntimeError, match="DEPLOYMENT_CHANGED"):
        asyncio.run(verify_magnetic_parameters(Client(), path))


# 功能：
#   检查飞控参数读回的错误类型不会通过相等比较伪装成合法参数。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   source：模拟的磁偏角来源返回值。
#   declination：模拟的磁偏角返回值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("source,declination", [(3., 1.), (3, True), (3, "1")])
def test_parameter_readback_requires_numeric_contract(tmp_path, source, declination):
    from unittest.mock import AsyncMock

    path = tmp_path / "deployment.json"
    path.write_text('{"declination_deg": 1}', encoding="utf-8")
    client = SimpleNamespace(get_param_int=AsyncMock(return_value=source),
                             get_param_float=AsyncMock(return_value=declination))
    with pytest.raises(RuntimeError, match="READBACK_FAILED"):
        asyncio.run(verify_magnetic_parameters(client, path))


# 功能：
#   检查原生运行包回执不接受重复字段，避免摘要校验与后续读取产生不同解释。
# 输入：
#   tmp_path：当前测试独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_native_receipt_rejects_duplicate_fields(tmp_path):
    args = inputs(tmp_path)
    receipt = args["runtime_root"] / "native-sensor-runtime.json"
    receipt.write_text('{"native_tests_passed": false,' + receipt.read_text()[1:])
    with pytest.raises(ValueError):
        validate_native_sensor_runtime(args["runtime_root"])


# 功能：
#   模拟探针执行期间已验证输入变化，检查准备流程拒绝混合新旧资产且不生成部署目录。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   monkeypatch：在探针返回点注入文件变化。
#   target：发生变化的世界、源码、回执、地磁表或原生程序。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("target", ["world", "bridge", "receipt", "table", "library", "probe"])
def test_prepare_rejects_inputs_changed_during_native_probe(tmp_path, monkeypatch, target):
    args = inputs(tmp_path)
    paths = {
        "world": args["world_sdf"],
        "bridge": args["px4_root"] / "src/modules/simulation/gz_bridge/GZBridge.cpp",
        "receipt": args["runtime_root"] / "native-sensor-runtime.json",
        "table": args["px4_root"] / "src/lib/world_magnetic_model/geo_magnetic_tables.hpp",
        "library": args["runtime_root"] / "libdronedream-magnetometer.so",
        "probe": args["runtime_root"] / "magnetic-field-probe",
    }

    # 功能：
    #   返回正常磁场前改变一个输入文件，避免把失败归因于探针输出格式错误。
    # 输入：
    #   command：原流程准备执行的命令参数。
    #   kwargs：原流程提供的子进程选项。
    # 输出：
    #   result：具有合法 JSON 标准输出的探针替身。
    def changed_probe(command, **kwargs):
        path = paths[target]
        path.write_bytes(path.read_bytes() + b"\n")
        result = SimpleNamespace(stdout=json.dumps({
            "declination_deg": 1., "field_enu_tesla": [1e-6, 23e-6, -42e-6]}))
        return result

    monkeypatch.setattr("dronedream_agent_core.simulation_sensor_runtime.subprocess.run",
                        changed_probe)
    with pytest.raises(ValueError, match="INPUT_CHANGED|NOT_VERIFIED"):
        prepare_sensor_runtime(**args)
    assert not args["output"].exists()


# 功能：
#   检查原生探针结果同样遵守严格 JSON 与物理数值契约，不能靠 Python 自动转换混入坏值。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   monkeypatch：替换原生探针输出。
#   payload：非法探针 JSON 文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("payload", [
    '[]', '{"declination_deg": true, "field_enu_tesla": [0, 0.000023, -0.000042]}',
    '{"declination_deg": "1", "field_enu_tesla": [0, 0.000023, -0.000042]}',
    '{"declination_deg": 1, "field_enu_tesla": [true, 0.000023, -0.000042]}',
    '{"declination_deg": NaN, "field_enu_tesla": [0, 0.000023, -0.000042]}',
    '{"declination_deg": 999, "declination_deg": 1, "field_enu_tesla": [0, 0.000023, 0]}',
])
def test_prepare_rejects_malformed_probe_output(tmp_path, monkeypatch, payload):
    args = inputs(tmp_path)
    monkeypatch.setattr("dronedream_agent_core.simulation_sensor_runtime.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(stdout=payload))
    with pytest.raises(ValueError):
        prepare_sensor_runtime(**args)
    assert not args["output"].exists()


# 功能：
#   检查多个世界或缺少真实磁力计回调时不能靠挑选首个世界、扫描其它源码段通过协议检查。
# 输入：
#   tmp_path：当前测试独占的临时目录。
#   case：双世界或回调缺失场景。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("case", ["multiple-worlds", "missing-callback"])
def test_prepare_requires_unique_world_and_actual_bridge_callback(tmp_path, case):
    args = inputs(tmp_path)
    if case == "multiple-worlds":
        path = args["world_sdf"]
        path.write_text(path.read_text().replace("</sdf>", "<world/></sdf>"))
    else:
        path = args["px4_root"] / "src/modules/simulation/gz_bridge/GZBridge.cpp"
        path.write_text(path.read_text().replace("magnetometerCallback", "otherCallback"))
    with pytest.raises(ValueError, match="WORLD_MISSING|WIRE_CONTRACT_UNSUPPORTED"):
        prepare_sensor_runtime(**args)
    assert not args["output"].exists()
