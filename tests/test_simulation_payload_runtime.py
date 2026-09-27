"""Synthetic packaging boundaries for the native payload service; no flight evidence."""

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from dronedream_agent_core.simulation_payload_runtime import (
    CONTRACT,
    LIBRARY,
    RECEIPT,
    prepare_payload_runtime,
    validate_payload_runtime,
)


# 功能：
#   创建明确合成的 ELF 头、源码和单链接资产，用于校验拒绝边界，不执行原生代码。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   fixture：运行库、源码、载荷路径三元组。
@pytest.fixture
def fixture(tmp_path):
    runtime, source = tmp_path / 'runtime', tmp_path / 'source'
    runtime.mkdir()
    source.mkdir()
    data = b'\x7fELF\x02\x01' + b'synthetic-not-executable'
    (runtime / LIBRARY).write_bytes(data)
    hashes = {}
    for name in ('CMakeLists.txt', 'PayloadPlacement.cc'):
        (source / name).write_text(name)
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    (runtime / RECEIPT).write_text(json.dumps(dict(contract=CONTRACT,
        library_sha256=hashlib.sha256(data).hexdigest(), sources=hashes)))
    payload = tmp_path / 'payload.sdf'
    payload.write_text('<sdf version="1.9"><model name="parcel"><link name="body">'
                       '<inertial><mass>0.04</mass></inertial></link></model></sdf>')
    return runtime, source, payload


# 功能：
#   检查只改变运行副本的插件声明，原物理资产和质量不变，重复输出不得覆盖。
# 输入：
#   fixture：合成组件；tmp_path：隔离输出根。
# 输出：
#   None：由副本内容、摘要、质量和覆盖拒绝断言给出结果。
def test_preparation_preserves_original_physics(fixture, tmp_path):
    runtime, source, payload = fixture
    original = payload.read_bytes()
    output = tmp_path / 'prepared'
    result = prepare_payload_runtime(payload_sdf=payload, runtime_root=runtime,
                                     output=output, source_root=source)
    assert payload.read_bytes() == original
    root = ET.parse(result['sdf_path']).getroot()
    assert root.findtext('./model/link/inertial/mass') == '0.04'
    assert len(list(root.iter('plugin'))) == 1
    assert root.find('./model/plugin').get('filename') == str(output / LIBRARY)
    assert result['original_sha256'] == hashlib.sha256(original).hexdigest()
    with pytest.raises(FileExistsError):
        prepare_payload_runtime(payload_sdf=payload, runtime_root=runtime, output=output)


# 功能：
#   拒绝旧回执、改动库和源码；未知控制插件或多链接资产不被隐式接管。
# 输入：
#   fixture：合成组件；tmp_path：输出位置；mode：需要破坏的边界。
# 输出：
#   None：所有破坏都在产生运行副本前拒绝。
@pytest.mark.parametrize('mode', ['library', 'source', 'contract', 'plugin', 'links', 'entity'])
def test_invalid_placement_components_fail_before_staging(fixture, tmp_path, mode):
    runtime, source, payload = fixture
    if mode == 'library':
        (runtime / LIBRARY).write_bytes(b'changed')
    elif mode == 'source':
        (source / 'PayloadPlacement.cc').write_text('changed')
    elif mode == 'contract':
        receipt = json.loads((runtime / RECEIPT).read_text())
        receipt['contract'] = 'legacy'
        (runtime / RECEIPT).write_text(json.dumps(receipt))
    else:
        content = payload.read_text()
        if mode == 'plugin':
            content = content.replace('</model>', '<plugin name="unknown"/></model>')
        elif mode == 'links':
            content = content.replace('</model>', '<link name="other"/></model>')
        else:
            content = '<!DOCTYPE sdf [<!ENTITY value "unsafe">]>' + content
        payload.write_text(content)
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError):
        prepare_payload_runtime(payload_sdf=payload, runtime_root=runtime,
                                output=output, source_root=source)
    assert not output.exists()


# 功能：
#   发布读取可不带源码，但不把组件摘要核验冒充原生运行或飞行测试。
# 输入：
#   fixture：明确合成的包校验夹具。
# 输出：
#   None：只确认回执合同值。
def test_runtime_validation_needs_no_source_checkout(fixture):
    receipt = validate_payload_runtime(fixture[0])
    assert receipt['contract'] == CONTRACT
