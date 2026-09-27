"""Verify an isolated installed wheel, not the editable source checkout or a product deployment."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path


# 功能：
#   确认实际导入位置和源码摘要，拒绝把工作区源码或其他版本误当作安装包验收。
# 输入：
#   installation：隔离安装根目录；source：构建来源仓库；binary：已验证的编译内核。
# 输出：
#   receipt：导入位置、文件摘要和协议验收记录。
def verify(installation, source, binary):
    installation, source, binary = installation.resolve(), source.resolve(), binary.resolve()
    receipt = {'files': {}, 'training_rows_promoted': 0, 'qualified_for_flight': False}
    for name in ('local_world_model', 'metric_ray_sampling', 'metric_scan_native',
                 'depth_projection', 'depth_projection_native'):
        module = importlib.import_module('dronedream_agent_core.' + name)
        path = Path(module.__file__).resolve()
        expected = source / 'src/dronedream_agent_core' / (name + '.py')
        if not path.is_relative_to(installation) or path.read_bytes() != expected.read_bytes():
            raise RuntimeError('METRIC_INSTALLED_SOURCE_MISMATCH:' + name)
        receipt['files'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    loader = importlib.import_module('dronedream_agent_core.metric_scan_native')
    if not loader.NATIVE_SCAN_AVAILABLE:
        raise RuntimeError('METRIC_INSTALLED_NATIVE_MISSING')
    backend = loader.backend
    path = Path(backend.__file__).resolve()
    if not path.is_relative_to(installation) or path.read_bytes() != binary.read_bytes():
        raise RuntimeError('METRIC_INSTALLED_BINARY_MISMATCH')
    receipt['files']['native'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt['contract'] = backend.CONTRACT
    depth = importlib.import_module('dronedream_agent_core.depth_projection_native')
    if not depth.NATIVE_DEPTH_AVAILABLE:
        raise RuntimeError('DEPTH_INSTALLED_NATIVE_MISSING')
    receipt['depth_contract'] = backend.DEPTH_CONTRACT
    # 小型直接调用检查符号可用性；数值覆盖仍由独立安装目录下的完整对照测试负责。
    result = backend.integrate([((0., 0., 0.), (.25, 0., 0.), 3, .7, True)], (0., 0., 0.), .25)
    if result != ((((1, 0, 0), .7),), (((0, 0, 0), .7),)):
        raise RuntimeError('METRIC_INSTALLED_SMOKE_MISMATCH')
    receipt['verified'] = True
    return receipt


# 功能：
#   将隔离安装验收写入独占新文件，不修改产品安装或替换已有验收证据。
# 输入：
#   installation、source、binary、output：命令行指定的安装、来源、内核和证据路径。
# 输出：
#   receipt：完整安装核验结果。
def main():
    parser = argparse.ArgumentParser()
    for name in ('installation', 'source', 'binary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.installation, args.source, args.binary)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
