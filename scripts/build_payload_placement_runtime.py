"""Build or stage the source-bound native parcel placement component, not a flight approval."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from dronedream_agent_core.simulation_payload_runtime import (
    CONTRACT,
    LIBRARY,
    RECEIPT,
    validate_payload_runtime,
)


# 功能：
#   冻结源码并构建独立原生载荷组件，或验证后暂存已有组件；禁止覆盖已冻结目录。
# 输入：
#   命令行 output：新目录；stage_from：可选已有组件目录；check_only：只检查来源。
# 输出：
#   result：组件目录及校验结果写入标准输出。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage-from', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'native/payload_placement'
    if args.check_only:
        if args.stage_from is None or args.output is not None:
            parser.error('--check-only requires --stage-from and forbids --output')
        result = validate_payload_runtime(args.stage_from, source_root=source)
        print(json.dumps(dict(verified=result, qualification_granted=False)))
        return
    if args.output is None:
        parser.error('--output is required for build or staging')
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(output)
    original = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ('CMakeLists.txt', 'PayloadPlacement.cc')}
    if args.stage_from is not None:
        validate_payload_runtime(args.stage_from, source_root=source)
        output.mkdir(parents=True)
        for name in (LIBRARY, RECEIPT):
            shutil.copyfile(args.stage_from / name, output / name)
    else:
        subprocess.run(['cmake', '-S', str(source), '-B', str(output), '-DCMAKE_BUILD_TYPE=Release'], check=True)
        subprocess.run(['cmake', '--build', str(output), '-j', '2'], check=True)
        if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
               for name, digest in original.items()):
            raise ValueError('PAYLOAD_RUNTIME_BUILD_SOURCE_CHANGED')
        receipt = dict(contract=CONTRACT, sources=original,
                       library_sha256=hashlib.sha256((output / LIBRARY).read_bytes()).hexdigest())
        with (output / RECEIPT).open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
    result = validate_payload_runtime(output, source_root=source)
    print(json.dumps(dict(output=str(output), verified=result, qualification_granted=False)))


if __name__ == '__main__':
    main()
