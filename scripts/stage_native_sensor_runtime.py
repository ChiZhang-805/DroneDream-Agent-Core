"""Stage the verified current native plugin, without CMake/cache build debris."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dronedream_agent_core.simulation_sensor_runtime import validate_native_sensor_runtime


# 功能：
#   1. 对照当前源码验证运行包，只复制运行所需文件并再次验证副本，拒绝覆盖旧暂存。
#   2. 失败时保留新目录用于诊断，不删除用户文件；该步骤不构建、安装或授权飞行。
# 输入：
#   source：待暂存的原生运行包。
#   output：必须尚不存在的暂存目录。
#   source_root：当前原生插件源码目录。
# 输出：
#   None：不返回业务数据。
def stage_native_sensor_runtime(*, source: Path, output: Path, source_root: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    validate_native_sensor_runtime(source, source_root=source_root)
    output.mkdir(parents=True)
    for name in ("native-sensor-runtime.json", "libdronedream-magnetometer.so",
                 "magnetic-field-probe", "geo_magnetic_tables.hpp"):
        shutil.copy2(source / name, output / name)
    validate_native_sensor_runtime(output, source_root=source_root)


# 功能：
#   按当前原生源码校验显式运行包；只读预检不写文件，暂存模式复制并复核完整运行包。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        if args.output is not None:
            parser.error("--check-only does not accept --output")
        validate_native_sensor_runtime(
            args.source, source_root=Path(__file__).resolve().parents[1] / "native/gazebo_sensors",
        )
        return
    if args.output is None:
        parser.error("--output is required unless --check-only is selected")
    stage_native_sensor_runtime(source=args.source, output=args.output,
                                source_root=Path(__file__).resolve().parents[1]
                                / "native/gazebo_sensors")


if __name__ == "__main__":
    main()
