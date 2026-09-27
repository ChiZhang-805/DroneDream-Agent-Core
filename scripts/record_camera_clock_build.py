"""Record the actually built camera-clock component after native tests pass."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


# 功能：运行本次构建的原生时钟测试，固定源码及二进制摘要；不安装或授予飞行资格。
# 输入：--build 指定独立 CMake 构建目录。输出：组件清单，用于启动前验证。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]/'native/camera_clock'
    subprocess.run(['ctest', '--test-dir', str(args.build), '--output-on-failure'], check=True)
    receipt = {'schema_version': 'dronedream.native-camera-clock.v1',
        'clock_contract': 'exact-native-sim-tick-preupdate-v1',
        'library_sha256': hashlib.sha256((args.build/'libdronedream-camera-clock.so').read_bytes()).hexdigest(),
        'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source/'camera_clock.cpp', source/'capture_clock.hpp', source/'CMakeLists.txt', source/'capture_clock_test.cpp')}}
    (args.build/'camera-clock-runtime.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
