"""Source-bound native renderer artifacts; never model or flight qualification."""

from __future__ import annotations

from pathlib import Path

from .render_artifact_io import file_sha as file_hash
from .render_artifact_io import read_object_snapshot, source_inventory, validate_dependencies
from .simulation_render_cache import validate_ogre_dependencies

SOURCE_LIBRARY = "libdronedream-render-source.so"
EXECUTABLE = "dronedream-render-replica"
RECEIPT = "render-replica-runtime.json"
BINARIES = (SOURCE_LIBRARY, EXECUTABLE, "replica-contract-test", "replica-scene-test")
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "native/render_replica"


# 功能：
#   枚举当前原生副本源码的完整目录树，不读取旧构建回执来冒充当前源码。
# 输入：
#   无。
# 输出：
#   sources：当前源码相对路径与内容摘要的映射。
def source_files() -> dict[str, str]:
    sources = source_inventory(SOURCE_ROOT)
    return sources


# 功能：
#   复核副本构建的当前源码、明确二进制集合及真实依赖，拒绝旧内容或混合渲染 ABI。
#   此检查仅验证制品绑定，不证明传感器等价、原生测试确实执行或获得飞行资格。
# 输入：
#   root：本次准备使用的原生渲染副本构建目录。
# 输出：
#   data：内容绑定通过且不授予飞行资格的构建回执。
def validate_replica_runtime(root: Path) -> dict:
    path = root / RECEIPT
    data, _ = read_object_snapshot(path, limit=1024 * 1024,
                                   error_code="REPLICA_RUNTIME_RECEIPT_UNAVAILABLE")
    if (not isinstance(data, dict) or data.get("sources") != source_files()
            or data.get("native_tests_passed") is not True
            or data.get("qualification_granted") is not False
            or not isinstance(data.get("binaries"), dict)
            or set(data["binaries"]) != set(BINARIES)):
        raise ValueError("REPLICA_RUNTIME_NOT_SOURCE_BOUND")
    # 精确文件名集合限制可执行制品，清单不能借路径穿越指定另一个程序。
    for name, digest in data["binaries"].items():
        binary = root / name
        if binary.is_symlink() or not binary.is_file() or file_hash(binary) != digest:
            raise ValueError("REPLICA_RUNTIME_BINARY_CHANGED:" + name)
    for group in ("libraries", "component_headers"):
        validate_dependencies(data.get(group), error_code="REPLICA_RUNTIME_DEPENDENCY_CHANGED")
    plugin = data.get("sensors_plugin")
    if (not isinstance(plugin, str) or plugin not in data["libraries"]
            or not Path(plugin).name.startswith("libgz-sim8-sensors-system.so")):
        raise ValueError("REPLICA_RUNTIME_SENSOR_PLUGIN_UNBOUND")
    validate_ogre_dependencies(list(data["libraries"]))
    if not any(Path(name).name.startswith("libgz-rendering8-ogre2.so")
               for name in data["libraries"]):
        raise ValueError("REPLICA_RUNTIME_RENDER_ENGINE_UNBOUND")
    return data
