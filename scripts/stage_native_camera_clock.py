"""Stage only verified native camera-clock runtime bytes, without build debris."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.simulation_camera_clock import (
    CAMERA_CLOCK_FILES, validate_native_camera_clock,
)


# 功能：核对当前源码后写出同次验证的时钟组件，再复查新副本，禁止覆盖已有目录。
# 输入：source：已构建组件；output：新目录；source_root：当前时钟源码目录。
# 输出：None；仅暂存运行文件，不更改安装状态或授予飞行权限。
def stage_native_camera_clock(*, source: Path, output: Path, source_root: Path):
    check_plain_plugin_path(output)
    if output.exists():
        raise FileExistsError(output)
    receipt, raw = validate_native_camera_clock(source, source_root=source_root)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in ((CAMERA_CLOCK_FILES[0], json.dumps(receipt, sort_keys=True).encode()),
                          (CAMERA_CLOCK_FILES[1], raw)):
        with (output / name).open('xb') as stream:
            stream.write(content)
    validate_native_camera_clock(output, source_root=source_root)


# 功能：供正式及开发构建调用相同时钟组件预检/暂存入口，缺文件在修改输出前失败。
# 输入：命令行源目录、输出目录或只读检查开关。
# 输出：None；失败抛出明确异常。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1] / 'native/camera_clock'
    if args.check_only:
        if args.output is not None:
            parser.error('--check-only cannot write --output')
        validate_native_camera_clock(args.source, source_root=source_root)
    else:
        if args.output is None:
            parser.error('--output is required')
        stage_native_camera_clock(source=args.source, output=args.output, source_root=source_root)


if __name__ == '__main__':
    main()
