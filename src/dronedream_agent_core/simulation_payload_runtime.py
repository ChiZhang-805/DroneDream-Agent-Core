"""Hash-bound native placement runtime for detached simulated parcels only."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .plugin_files import check_plain_plugin_path, read_plugin_file

LIBRARY = 'libdronedream-payload-placement.so'
RECEIPT = 'payload-placement-runtime.json'
CONTRACT = 'detached-parcel-one-step-pose-velocity-placement-v1'


# 功能：
#   核对载荷放置组件的文件摘要、有限 ELF 文件和可选当前源码，拒绝旧组件或失配来源。
# 输入：
#   runtime_root：独立原生载荷组件目录；source_root：开发模式下的当前源码目录。
# 输出：
#   receipt：已验证的组件构建信息，不授予飞行资格。
def validate_payload_runtime(runtime_root: Path, *, source_root: Path | None = None) -> dict:
    from dronedream_plugin_sdk.protocol import decode_json

    root = Path(runtime_root).absolute()
    check_plain_plugin_path(root)
    if not root.is_dir():
        raise ValueError('PAYLOAD_RUNTIME_DIRECTORY_MISSING')
    receipt = decode_json(read_plugin_file(root / RECEIPT, limit=65536), limit=65536)
    if (type(receipt) is not dict or set(receipt) != {'contract', 'library_sha256', 'sources'}
            or receipt.get('contract') != CONTRACT or type(receipt.get('sources')) is not dict
            or set(receipt['sources']) != {'CMakeLists.txt', 'PayloadPlacement.cc'}):
        raise ValueError('PAYLOAD_RUNTIME_RECEIPT_INVALID')
    library = read_plugin_file(root / LIBRARY, limit=16 * 1024**2)
    if (not library.startswith(b'\x7fELF\x02\x01')
            or hashlib.sha256(library).hexdigest() != receipt.get('library_sha256')):
        raise ValueError('PAYLOAD_RUNTIME_LIBRARY_MISMATCH')
    for name, digest in receipt['sources'].items():
        if (type(digest) is not str or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('PAYLOAD_RUNTIME_SOURCE_IDENTITY_INVALID')
        if source_root is not None and hashlib.sha256(read_plugin_file(
                Path(source_root).absolute() / name, limit=1024**2)).hexdigest() != digest:
            raise ValueError('PAYLOAD_RUNTIME_SOURCE_CHANGED')
    return receipt


# 功能：
#   在当前运行的独立副本加入经验证的放置服务；不改原资产或质量、惯量、碰撞形状。
# 输入：
#   payload_sdf：原始载荷资产；runtime_root：已构建的组件；output：不存在的运行副本目录。
#   source_root：开发模式下需要匹配的源码目录。
# 输出：
#   prepared：运行 SDF 路径及原资产、组件和衍生文件摘要。
def prepare_payload_runtime(*, payload_sdf: Path, runtime_root: Path, output: Path,
                            source_root: Path | None = None) -> dict:
    receipt = validate_payload_runtime(runtime_root, source_root=source_root)
    content = read_plugin_file(Path(payload_sdf).absolute(), limit=4 * 1024**2)
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        raise ValueError('PAYLOAD_RUNTIME_SDF_ENTITIES_FORBIDDEN')
    root = ET.fromstring(content)
    models = root.findall('model')
    if (root.tag != 'sdf' or len(models) != 1 or len(list(root.iter('model'))) != 1
            or len(models[0].findall('link')) != 1 or list(root.iter('plugin'))):
        raise ValueError('PAYLOAD_RUNTIME_REQUIRES_SINGLE_UNCONTROLLED_LINK')
    library = read_plugin_file(Path(runtime_root).absolute() / LIBRARY, limit=16 * 1024**2)
    if hashlib.sha256(library).hexdigest() != receipt['library_sha256']:
        raise ValueError('PAYLOAD_RUNTIME_CHANGED_DURING_PREPARATION')
    output = Path(output).absolute()
    check_plain_plugin_path(output)
    if not output.parent.is_dir():
        raise ValueError('PAYLOAD_RUNTIME_OUTPUT_PARENT_MISSING')
    output.mkdir(exist_ok=False)
    with (output / LIBRARY).open('xb') as stream:
        stream.write(library)
    ET.SubElement(models[0], 'plugin', filename=str(output / LIBRARY),
                  name='dronedream::PayloadPlacement')
    derived = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    with (output / 'payload.sdf').open('xb') as stream:
        stream.write(derived)
    prepared = dict(sdf_path=str(output / 'payload.sdf'), original_sha256=hashlib.sha256(content).hexdigest(),
                    staged_sha256=hashlib.sha256(derived).hexdigest(), library_sha256=receipt['library_sha256'],
                    contract=CONTRACT, sources=receipt['sources'])
    with (output / 'deployment.json').open('x', encoding='utf-8') as stream:
        json.dump(prepared, stream, sort_keys=True, indent=2)
    return prepared
