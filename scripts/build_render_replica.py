"""Build/test/freeze a new native sensor-replica directory in the project workspace."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.render_replica_runtime import (
    BINARIES,
    RECEIPT,
    SOURCE_ROOT,
    file_hash,
    source_files,
    validate_replica_runtime,
)
from dronedream_agent_core.simulation_render_cache import validate_ogre_dependencies


# 功能：
#   在独立新目录编译渲染副本并运行原生测试，冻结源码、组件头文件、运行库和渲染器搜索路径。
#   复核构建期间输入未变化后生成回执，不修改已安装运行环境或授予飞行资格。
# 输入：
#   argv：可选命令行参数，--output 指定新构建目录；None 时读取进程命令行。
# 输出：
#   None：不返回业务数据。
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    check_plain_plugin_path(args.output)
    output = args.output.resolve()
    if output.is_relative_to(SOURCE_ROOT.resolve()):
        raise ValueError("REPLICA_BUILD_OUTPUT_INSIDE_SOURCE")
    output.mkdir(parents=True, exist_ok=False)
    original = source_files()
    header_root = Path("/usr/include/gz/sim8/gz/sim/components")
    headers = {str(p): file_hash(p) for p in sorted(header_root.glob("*.hh"))}
    if not headers:
        raise RuntimeError("REPLICA_COMPONENT_HEADERS_MISSING")
    subprocess.run(["cmake", "-S", str(SOURCE_ROOT), "-B", str(output),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(output), "--parallel", "2"], check=True)
    subprocess.run(["ctest", "--test-dir", str(output), "--output-on-failure"], check=True)
    plugin = Path("/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins/"
                  "libgz-sim8-sensors-system.so").resolve(strict=True)
    engine = Path("/usr/lib/x86_64-linux-gnu/gz-rendering-8/engine-plugins/"
                  "libgz-rendering-ogre2.so").resolve(strict=True)
    dynamic = subprocess.run(["readelf", "-d", str(engine)], check=True,
                             capture_output=True, text=True, timeout=30).stdout
    engine_paths = re.findall(r"\(RUNPATH\).*?\[(.*?)\]", dynamic)
    ogre_directory = "/usr/lib/x86_64-linux-gnu/OGRE-2.3"
    if engine_paths != [ogre_directory]:
        raise RuntimeError("REPLICA_RENDER_ENGINE_SEARCH_PATH_UNEXPECTED")
    # 单独对 Ogre 子库执行 ldd 会丢失父引擎 RUNPATH，因此仅补入已核对的实际目录。
    ldd_environment = {**os.environ, "LD_LIBRARY_PATH": ogre_directory}
    libraries = {plugin, engine}
    libraries.update(p.resolve(strict=True)
        for p in Path("/usr/lib/x86_64-linux-gnu/OGRE-2.3").rglob("*.so*"))
    # WSL 专用驱动也参与绑定，退出映射保护不能悄悄作用于另一套驱动。
    libraries.update(p.resolve(strict=True)
        for p in Path("/usr/lib/wsl/lib").glob("libd3d12*.so"))
    for binary in [*(output / name for name in BINARIES), *sorted(libraries)]:
        result = subprocess.run(["ldd", str(binary)], check=True, capture_output=True,
                                text=True, timeout=30, env=ldd_environment).stdout
        if "not found" in result:
            raise RuntimeError("REPLICA_LINKED_DEPENDENCY_MISSING:" + str(binary))
        libraries.update(Path(name).resolve(strict=True)
                         for name in re.findall(r"(/[^\s()]+)", result))
    validate_ogre_dependencies([str(p) for p in libraries])
    if (original != source_files()
            or headers != {str(p): file_hash(p) for p in sorted(header_root.glob("*.hh"))}):
        raise RuntimeError("REPLICA_BUILD_INPUT_CHANGED")
    data = {"sources": original, "binaries": {n: file_hash(output / n) for n in BINARIES},
            "libraries": {str(p): file_hash(p) for p in sorted(libraries)},
            "component_headers": headers, "sensors_plugin": str(plugin),
            "render_engine_runpath": ogre_directory,
            "native_tests_passed": True, "qualification_granted": False}
    with (output / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
    validate_replica_runtime(output)


if __name__ == "__main__":
    main()
