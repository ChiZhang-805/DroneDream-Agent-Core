"""Build the explicit simulation render helper without changing installed files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.simulation_render_cache import (
    LIBRARY,
    file_sha,
    source_inventory,
    validate_ogre_dependencies,
)


# 功能：
#   在全新、源码树之外的目录编译并运行原生契约测试，绑定当前源码、库字节与实际依赖。
#   回执只声明此构建步骤通过，不替代真实渲染、传感器等价或飞行验收。
# 输入：
#   argv：可选命令行参数，--output 指定全新构建目录；None 时读取进程命令行。
# 输出：
#   None：不返回业务数据。
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = (Path(__file__).resolve().parents[1] / "native/render_preparation").resolve()
    check_plain_plugin_path(args.output)
    output = args.output.resolve()
    if output.is_relative_to(source):
        raise ValueError("RENDER_BUILD_OUTPUT_INSIDE_SOURCE")
    # CMake 缓存可继承旧配置：即使没有成功回执，也不能继续复用已有构建目录。
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "render-preparation-runtime.json"
    original = source_inventory(source)
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release"], check=True
    )
    subprocess.run(["cmake", "--build", str(output), "-j", "2"], check=True)
    subprocess.run(["ctest", "--test-dir", str(output), "--output-on-failure"], check=True)
    library = output / LIBRARY
    dependencies = subprocess.run(
        ["ldd", str(library)], check=True, text=True, capture_output=True, timeout=30
    ).stdout
    if "not found" in dependencies:
        raise RuntimeError("native render dependency missing")
    files = {Path(name).resolve(strict=True) for name in re.findall(r"(/[^\s()]+)", dependencies)}
    # 动态加载的 RenderSystem 不出现在 ldd 中，必须单独冻结实际模块。
    files.update(
        path.resolve(strict=True)
        for path in Path("/usr/lib/x86_64-linux-gnu/OGRE-2.3").rglob("*.so*")
    )
    files.add(Path("/usr/lib/x86_64-linux-gnu/gz-rendering-8/engine-plugins/"
                   "libgz-rendering-ogre2.so").resolve(strict=True))
    validate_ogre_dependencies([str(path) for path in files])
    if not files or source_inventory(source) != original:
        raise RuntimeError("native render build inputs changed or dependencies missing")
    value = {
        "library_sha256": file_sha(library),
        "sources": original,
        "native_tests_passed": True,
        "libraries": {str(path): file_sha(path) for path in sorted(files)},
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
