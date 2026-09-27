import hashlib
import json
from xml.etree import ElementTree as ET

import pytest
from dronedream_agent_core.simulation_camera_clock import (
    prepare_native_camera_clock, validate_native_camera_clock,
)


# 功能：创建最小可验证时钟组件，不使用真实安装文件。输入：临时目录。输出：组件与配置路径。
def fixture_runtime(root):
    runtime = root/'runtime'
    runtime.mkdir()
    (runtime/'libdronedream-camera-clock.so').write_bytes(b'test-only')
    receipt = {'schema_version':'dronedream.native-camera-clock.v1',
        'clock_contract':'exact-native-sim-tick-preupdate-v1',
        'library_sha256':hashlib.sha256(b'test-only').hexdigest(),
        'sources':dict.fromkeys(['camera_clock.cpp','capture_clock.hpp','CMakeLists.txt','capture_clock_test.cpp'], 'a'*64)}
    (runtime/'camera-clock-runtime.json').write_text(json.dumps(receipt))
    config = root/'server.config'
    config.write_text('<server_config><plugins><plugin name="Sensors" /></plugins></server_config>')
    return runtime, config


# 功能：证明配置是每次运行隔离的，原配置不变且输入主题不被覆盖。输入：临时目录。输出：断言。
def test_run_owned_clock_configuration(tmp_path):
    runtime, config = fixture_runtime(tmp_path)
    original = config.read_bytes()
    kwargs = dict(runtime_root=runtime, server_config=config, rgb_topic='/camera/rgb', depth_topic='/depth_camera')
    a = prepare_native_camera_clock(**kwargs, output=tmp_path/'a')
    b = prepare_native_camera_clock(**kwargs, output=tmp_path/'b')
    assert a['epoch'] != b['epoch']
    assert config.read_bytes() == original
    plugin = ET.parse(a['environment']['GZ_SIM_SERVER_CONFIG_PATH']).find(".//plugin[@name='dronedream::NativeCameraClock']")
    assert plugin.findtext('rgb_input') == '/camera/rgb'
    assert plugin.findtext('rgb_output') == a['rgb_topic']
    assert plugin.attrib['filename'] == str((tmp_path/'a/libdronedream-camera-clock.so').resolve())
    assert (tmp_path/'a/libdronedream-camera-clock.so').read_bytes() == b'test-only'
    assert not a['qualification_granted'] and not a['truth_pose_used']
    with pytest.raises(FileExistsError):
        prepare_native_camera_clock(**kwargs, output=tmp_path/'a')


# 功能：损坏二进制应在仿真启动前拒绝，不能运行后悬停等待永远不存在的有效图像。
# 输入：临时目录。输出：拒绝异常，未创建部署目录。
def test_bad_component_fails_before_deployment(tmp_path):
    runtime, config = fixture_runtime(tmp_path)
    (runtime/'libdronedream-camera-clock.so').write_bytes(b'changed')
    with pytest.raises(ValueError, match='RUNTIME_INVALID'):
        prepare_native_camera_clock(runtime_root=runtime, server_config=config,
            rgb_topic='/rgb', depth_topic='/depth', output=tmp_path/'deploy')
    assert not (tmp_path/'deploy').exists()


# 功能：拒绝数组/标量与重复字段格式，错误必须发生在任何运行目录创建之前。
# 输入：tmp_path：隔离目录；content：非法收据内容。
# 输出：None；断言明确拒绝而不是抛出属性读取异常。
@pytest.mark.parametrize('content', ['[]', 'null', '2', '{"sources":{},"sources":{}}'])
def test_malformed_receipt_is_rejected(tmp_path, content):
    runtime, _ = fixture_runtime(tmp_path)
    (runtime/'camera-clock-runtime.json').write_text(content)
    with pytest.raises(ValueError):
        validate_native_camera_clock(runtime)


# 功能：拒绝外部实体、错误根元素和重复插件容器，防止不可信配置替换运行环境。
# 输入：tmp_path：隔离目录；xml：非法服务器配置。
# 输出：None；断言未创建任何新运行文件。
@pytest.mark.parametrize('xml', [
    '<other><plugins/></other>', '<server_config><plugins/><plugins/></server_config>',
    '<!DOCTYPE x [<!ENTITY xx "expanded">]><server_config><plugins>&xx;</plugins></server_config>',
])
def test_invalid_xml_is_rejected_before_writes(tmp_path, xml):
    runtime, config = fixture_runtime(tmp_path)
    config.write_text(xml)
    with pytest.raises(ValueError):
        prepare_native_camera_clock(runtime_root=runtime, server_config=config,
            rgb_topic='/rgb', depth_topic='/depth', output=tmp_path/'deployment')
    assert not (tmp_path/'deployment').exists()


# 功能：验证发布暂存只包括运行组件，源码不匹配与覆盖请求均会被拒绝。
# 输入：tmp_path：合成源码与组件目录。
# 输出：None；验证新副本哈希和严格来源检查。
def test_verified_clock_staging(tmp_path):
    import runpy
    from pathlib import Path
    runtime, _ = fixture_runtime(tmp_path)
    source = tmp_path/'source'
    source.mkdir()
    receipt = json.loads((runtime/'camera-clock-runtime.json').read_bytes())
    for name in receipt['sources']:
        (source/name).write_bytes(name.encode())
        receipt['sources'][name] = hashlib.sha256(name.encode()).hexdigest()
    (runtime/'camera-clock-runtime.json').write_text(json.dumps(receipt))
    stage = runpy.run_path(str(Path(__file__).parents[1]/'scripts/stage_native_camera_clock.py'))['stage_native_camera_clock']
    target = tmp_path/'staged'
    stage(source=runtime, output=target, source_root=source)
    assert {path.name for path in target.iterdir()} == {'camera-clock-runtime.json', 'libdronedream-camera-clock.so'}
    assert validate_native_camera_clock(target, source_root=source)[0] == receipt
    with pytest.raises(FileExistsError):
        stage(source=runtime, output=target, source_root=source)
    (source/'camera_clock.cpp').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        stage(source=runtime, output=tmp_path/'bad', source_root=source)
    assert not (tmp_path/'bad').exists()
