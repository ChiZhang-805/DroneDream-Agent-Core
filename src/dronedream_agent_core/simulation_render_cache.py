"""Opt-in, source-bound Ogre preparation, never a flight qualification.

Cache input is copied and verified before simulator launch. The native plugin
checks the actual graphics context and loads only on its first render callback.
It serializes only on render teardown. Promotion requires native landing (or a
vehicle that was never spawned) AND exit of every owned simulator process.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from dronedream_plugin_sdk import protocol
from dronedream_plugin_sdk.protocol import copy_json, encode_json

from . import plugin_files, render_artifact_io, xml_values
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .render_artifact_io import (
    file_sha,
    inventory_tree,
    read_object_snapshot,
    source_inventory,
    validate_dependencies,
)
from .xml_values import parse_xml

LIBRARY = "libdronedream-render-preparation.so"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 65536
MAX_CONFIG_BYTES = 2 * 1024 * 1024
GRAPHICS_ENVIRONMENT = (
    "GALLIUM_DRIVER",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME",
    "EGL_PLATFORM",
    "LIBGL_ALWAYS_SOFTWARE",
    "LP_NUM_THREADS",
    "LP_NATIVE_VECTOR_WIDTH",
    "GALLIVM_PERF",
    "MESA_SHADER_CACHE_DISABLE",
    "MESA_SHADER_CACHE_MAX_SIZE",
)


# 功能：
#   校验标准 JSON 值并按稳定键序计算渲染身份摘要，拒绝非有限数字及超限结构。
# 输入：
#   value：待绑定的身份字典。
# 输出：
#   digest：紧凑稳定 JSON 对应的 SHA-256 摘要。
def object_sha(value: dict) -> str:
    encode_json(value, limit=MAX_METADATA_BYTES, node_limit=2_000_000)
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return digest


# 功能：
#   绑定当前渲染准备入口及直接参与文件、JSON、XML 验证的宿主源码，避免只更新公共模块时复用旧身份。
# 输入：
#   无。
# 输出：
#   digest：当前宿主校验源码集合的稳定摘要。
def _host_implementation_sha() -> str:
    sources = {"simulation_render_cache": file_sha(Path(__file__))}
    for module in (render_artifact_io, plugin_files, xml_values, protocol):
        sources[module.__name__] = file_sha(Path(module.__file__))
    digest = object_sha(sources)
    return digest


# 功能：
#   在独占创建文件前完成 JSON 校验与编码，不覆盖现有回执或留下序列化错误的半份文档。
# 输入：
#   path：必须尚不存在的输出路径。
#   value：需要保存的标准 JSON 字典。
# 输出：
#   None：不返回业务数据。
def _new_json(path: Path, value: dict) -> None:
    encode_json(value, limit=MAX_METADATA_BYTES, node_limit=2_000_000)
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if len(rendered.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("RENDER_METADATA_TOO_LARGE")
    check_plain_plugin_path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


# 功能：
#   完整绑定真实材质、网格及模板内容，跨所有资源根累计条目与字节预算。
# 输入：
#   roots：资源标签到本地目录的映射。
# 输出：
#   result：每个资源标签对应的相对文件路径摘要表。
def resource_inventory(roots: dict[str, Path]) -> dict:
    if not isinstance(roots, dict) or any(not isinstance(label, str) for label in roots):
        raise ValueError("RENDER_RESOURCE_ROOT_INVALID")
    result, count, size = {}, 0, 0
    for label, root in sorted(roots.items()):
        entries, used_count, used_bytes = inventory_tree(
            root, maximum_entries=32768 - count, maximum_bytes=4 * 1024**3 - size
        )
        count += used_count
        size += used_bytes
        result[label] = entries
    return result


# 功能：
#   只允许当前适配的 Ogre Next 与动态渲染模块组合，阻止混入旧 Ogre ABI。
# 输入：
#   paths：已绑定依赖的路径列表。
# 输出：
#   None：不返回业务数据。
def validate_ogre_dependencies(paths: list[str]) -> None:
    names = {Path(path).name for path in paths}
    if (
        any(name.startswith("libOgreMain.so") for name in names)
        or "libOgreNextMain.so.2.3.1" not in names
        or "RenderSystem_GL3Plus.so.2.3.1" not in names
    ):
        raise ValueError("RENDER_RUNTIME_MIXED_OR_UNSUPPORTED_OGRE_ABI")


# 功能：
#   严格读取渲染准备构建回执，核对当前源码、二进制和实际依赖，不授予飞行资格。
# 输入：
#   root：准备使用的独立原生构建目录。
# 输出：
#   receipt：通过内容绑定检查的构建记录。
def validate_runtime(root: Path) -> dict:
    receipt, _ = read_object_snapshot(root / "render-preparation-runtime.json",
                                      limit=MAX_METADATA_BYTES,
                                      error_code="RENDER_RUNTIME_NOT_VERIFIED")
    source = Path(__file__).resolve().parents[2] / "native/render_preparation"
    if (
        receipt.get("library_sha256") != file_sha(root / LIBRARY)
        or receipt.get("sources") != source_inventory(source)
        or receipt.get("native_tests_passed") is not True
        or not receipt.get("libraries")
    ):
        raise ValueError("RENDER_RUNTIME_NOT_VERIFIED")
    validate_dependencies(receipt.get("libraries"), error_code="RENDER_RUNTIME_DEPENDENCY_CHANGED")
    validate_ogre_dependencies(list(receipt["libraries"]))
    return receipt


# 功能：
#   校验缓存类型、文件名、声明大小与实际摘要，拒绝重复类型、超限与仅有 microcode 的集合。
# 输入：
#   rows：清单中的缓存文件记录。
#   directory：存放这些缓存文件的普通目录。
# 输出：
#   checked：按缓存类型稳定排序的独立记录列表。
def _file_rows(rows: list[dict], directory: Path) -> list[dict]:
    checked, types, total = [], set(), 0
    if not isinstance(rows, list) or not 1 <= len(rows) <= 15:
        raise ValueError("RENDER_CACHE_FILE_SET_INVALID")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"type", "name", "bytes", "sha256"}:
            raise ValueError("RENDER_CACHE_FILE_INVALID")
        kind, size = row["type"], row["bytes"]
        # 宿主只限制协议预算；具体 HLMS 类型是否被引擎支持仍由原生加载器决定。
        if (
            type(kind) is not int
            or not 0 <= kind < 16
            or kind in types
            or type(size) is not int
            or not 0 < size <= MAX_FILE_BYTES
        ):
            raise ValueError("RENDER_CACHE_FILE_INVALID")
        expected = "microcode.bin" if kind == 0 else f"hlms-{kind}.bin"
        if (
            row["name"] != expected
            or not isinstance(row["sha256"], str)
            or re.fullmatch("[a-f0-9]{64}", row["sha256"]) is None
        ):
            raise ValueError("RENDER_CACHE_FILE_INVALID")
        path = directory / expected
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or hash_plugin_file(path, limit=size) != row["sha256"]
        ):
            raise ValueError("RENDER_CACHE_FILE_CHANGED")
        types.add(kind)
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("RENDER_CACHE_SIZE_LIMIT")
        checked.append(dict(row))
    if not (types - {0}):
        raise ValueError("RENDER_CACHE_HLMS_MISSING")
    checked = sorted(checked, key=lambda item: item["type"])
    return checked


# 功能：
#   有界读取并校验原生 XML 回执，拒绝 DTD、异常状态和身份不匹配，不凭状态文字授权飞行。
# 输入：
#   path：ready 或 finished 回执路径。
#   status：本次要求的明确状态。
#   identity：本次部署身份摘要。
# 输出：
#   receipt：校验通过的 XML 根元素及同一份原文的 SHA-256 摘要。
def _receipt(path: Path, *, status: str, identity: str) -> tuple[ET.Element, str]:
    try:
        content = read_plugin_file(path, limit=MAX_RECEIPT_BYTES)
        root = parse_xml(content, maximum_bytes=MAX_RECEIPT_BYTES, maximum_elements=32)
    except (OSError, ValueError, ET.ParseError) as error:
        raise ValueError("RENDER_CACHE_RECEIPT_UNAVAILABLE") from error
    if (
        root.tag != "receipt"
        or root.get("status") != status
        or root.get("identity_sha256") != identity
        or root.get("error") != ""
        or root.get("qualification_granted") != "false"
        or root.get("mode") not in {"capture", "load"}
        or root.get("microcode_supported") not in {"true", "false"}
        or re.fullmatch("[a-f0-9]{64}", root.get("renderer_sha256", "")) is None
    ):
        raise ValueError("RENDER_CACHE_RECEIPT_REJECTED")
    receipt = root, hashlib.sha256(content).hexdigest()
    return receipt


# 功能：
#   将原生回执的文件节点转为清单记录，拒绝未定义子节点或缺失字段。
# 输入：
#   root：经过结构预算限制的 XML 回执根节点。
# 输出：
#   rows：仍需实际文件校验的类型、大小、文件名和摘要记录。
def _xml_rows(root: ET.Element) -> list[dict]:
    if any(child.tag != "file" for child in root):
        raise ValueError("RENDER_CACHE_RECEIPT_UNEXPECTED_CHILD")
    rows = [
        {
            "type": int(node.attrib["type"]),
            "bytes": int(node.attrib["bytes"]),
            "name": node.attrib["name"],
            "sha256": node.attrib["sha256"],
        }
        for node in root
    ]
    return rows


# 功能：
#   按已经校验的清单限量复制缓存，同时校验实际复制字节，拒绝来源被替换或增长。
# 输入：
#   rows：经过文件类型与路径校验的清单记录。
#   source：当前缓存来源目录。
#   destination：本次新建的复制目录。
# 输出：
#   None：不返回业务数据。
def _copy_cache_files(rows: list[dict], source: Path, destination: Path) -> None:
    for row in rows:
        digest = hash_plugin_file(source / row["name"], limit=row["bytes"],
                                  destination=destination / row["name"])
        if digest != row["sha256"]:
            raise ValueError("RENDER_CACHE_FILE_CHANGED")
    _file_rows(rows, destination)


# 功能：
#   固定部署清单并复核内容身份、当前宿主实现与实际配置，防止旧 ready 回执证明新内容。
# 输入：
#   output：本次运行独占的渲染准备目录。
# 输出：
#   deployment：通过身份和配置核对的部署字典。
def _deployment(output: Path) -> dict:
    deployment, _ = read_object_snapshot(output / "deployment.json", limit=MAX_METADATA_BYTES,
                                         error_code="RENDER_DEPLOYMENT_INVALID")
    identity = deployment.get("identity")
    if (not isinstance(identity, dict) or deployment.get("identity_sha256") != object_sha(identity)
            or identity.get("host_implementation_sha256") != _host_implementation_sha()
            or deployment.get("qualification_granted") is not False
            or deployment.get("mode") not in {"load", "capture"}
            or not isinstance(deployment.get("input_files"), list)):
        raise ValueError("RENDER_DEPLOYMENT_IDENTITY_MISMATCH")
    config = read_plugin_file(output / "server.config", limit=MAX_CONFIG_BYTES)
    if hashlib.sha256(config).hexdigest() != deployment.get("server_config_sha256"):
        raise ValueError("RENDER_DEPLOYMENT_CONFIG_CHANGED")
    expected_renderer = deployment.get("expected_renderer_sha256")
    if deployment["mode"] == "capture":
        if deployment["input_files"] or expected_renderer is not None:
            raise ValueError("RENDER_DEPLOYMENT_CAPTURE_INVALID")
    else:
        if (not isinstance(expected_renderer, str)
                or re.fullmatch(r"[a-f0-9]{64}", expected_renderer) is None):
            raise ValueError("RENDER_DEPLOYMENT_RENDERER_INVALID")
        _file_rows(deployment["input_files"], output / "input")
    return deployment


# 功能：
#   在新运行目录生成源码、资源、设备环境绑定的渲染准备配置，按已验证身份复制可选缓存。
#   不修改安装目录；不完整复制不能形成成功部署回执。
# 输入：
#   runtime_root：已绑定源码与依赖的原生构建目录。
#   input_bundle：可选的已封存缓存目录。
#   output：必须尚不存在的运行专属输出目录。
#   server_config：实际服务器 XML 配置。
#   resources：本次使用的资源标签与目录。
#   assets：本次相机和其他渲染资产的身份数据。
#   environment：本次显卡和驱动相关环境变量。
# 输出：
#   deployment：启动配置环境和渲染缓存部署身份，不含飞行授权。
def prepare_render_cache(
    *,
    runtime_root: Path,
    input_bundle: Path | None,
    output: Path,
    server_config: Path,
    resources: dict[str, Path],
    assets: dict,
    environment: dict[str, str],
) -> dict:
    check_plain_plugin_path(output)
    if output.exists():
        raise FileExistsError(output)
    runtime = validate_runtime(runtime_root)
    original_config = read_plugin_file(server_config, limit=MAX_CONFIG_BYTES)
    tree = parse_xml(original_config, maximum_bytes=MAX_CONFIG_BYTES)
    plugins = tree.find("plugins")
    if tree.tag != "server_config" or plugins is None or any(
        node.get("name") == "dronedream::RenderPreparation" for node in plugins
    ):
        raise ValueError("RENDER_PREPARATION_PLUGIN_CONFIG_INVALID")
    identity = {
        "assets": copy_json(assets, limit=MAX_METADATA_BYTES),
        "resources": resource_inventory(resources),
        "runtime": runtime,
        "host_implementation_sha256": _host_implementation_sha(),
        "graphics_environment": {name: environment.get(name, "") for name in GRAPHICS_ENVIRONMENT},
    }
    digest = object_sha(identity)
    bundle, rows = None, []
    if input_bundle is not None:
        if input_bundle.is_symlink() or not input_bundle.is_dir():
            raise ValueError("RENDER_CACHE_BUNDLE_INVALID")
        manifest = input_bundle / "cache-bundle.json"
        bundle, _ = read_object_snapshot(manifest, limit=MAX_METADATA_BYTES,
                                         error_code="RENDER_CACHE_MANIFEST_INVALID")
        if (
            bundle.get("identity") != identity
            or bundle.get("identity_sha256") != digest
            or bundle.get("qualification_granted") is not False
            or bundle.get("all_owned_processes_exited") is not True
            or bundle.get("terminal_state") not in {"ON_GROUND", "NOT_STARTED"}
            or not isinstance(bundle.get("renderer_sha256"), str)
            or re.fullmatch("[a-f0-9]{64}", bundle["renderer_sha256"]) is None
        ):
            raise ValueError("RENDER_CACHE_IDENTITY_MISMATCH")
        rows = _file_rows(bundle.get("files"), input_bundle)
    output.mkdir(parents=True)
    (output / "output").mkdir()
    plugin = ET.SubElement(
        plugins,
        "plugin",
        {
            "entity_name": "*",
            "entity_type": "world",
            "name": "dronedream::RenderPreparation",
            "filename": str((runtime_root / LIBRARY).resolve()),
        },
    )
    ET.SubElement(plugin, "output").text = str(output.resolve())
    ET.SubElement(plugin, "identity_sha256").text = digest
    if bundle:
        copied = output / "input"
        copied.mkdir()
        _copy_cache_files(rows, input_bundle, copied)
        ET.SubElement(plugin, "input").text = str(copied.resolve())
        ET.SubElement(plugin, "renderer_sha256").text = bundle["renderer_sha256"]
        for row in rows:
            item = ET.SubElement(plugin, "file")
            for key, value in row.items():
                ET.SubElement(item, key).text = str(value)
    config = output / "server.config"
    with config.open("xb") as stream:
        stream.write(ET.tostring(tree, encoding="utf-8", xml_declaration=True))
    result = {
        "identity": identity,
        "identity_sha256": digest,
        "mode": "load" if bundle else "capture",
        "input_files": rows,
        "expected_renderer_sha256": bundle["renderer_sha256"] if bundle else None,
        "server_config_sha256": file_sha(config),
        "qualification_granted": False,
    }
    _new_json(output / "deployment.json", result)
    deployment = {"environment": {"GZ_SIM_SERVER_CONFIG_PATH": str(config)}, **result}
    return deployment


# 功能：
#   对同一份已验证部署核对 ready 状态、缓存文件及显卡身份，并保留原文摘要。
# 输入：
#   output：运行专属渲染准备目录。
#   deployment：本次已验证且独立持有的部署字典。
# 输出：
#   result：准备阶段验证结果及 ready 原文摘要。
def _verified_ready(output: Path, deployment: dict) -> tuple[dict, str]:
    root, digest = _receipt(output / "ready.xml", status="ready",
                            identity=deployment["identity_sha256"])
    if (
        root.get("mode") != deployment["mode"]
        or _xml_rows(root) != deployment["input_files"]
        or (
            deployment["expected_renderer_sha256"] is not None
            and root.get("renderer_sha256") != deployment["expected_renderer_sha256"]
        )
    ):
        raise ValueError("RENDER_PREPARATION_READBACK_MISMATCH")
    elapsed = float(root.get("prepare_ms", "nan"))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("RENDER_PREPARATION_TIMING_INVALID")
    verification = {
        "verified": True,
        "mode": root.get("mode"),
        "renderer_sha256": root.get("renderer_sha256"),
        "prepare_ms": elapsed,
        "loaded_files": len(root),
        "microcode_supported": root.get("microcode_supported") == "true",
        "qualification_granted": False,
    }
    result = verification, digest
    return result


# 功能：
#   复核当前部署与原生 ready 回执，返回渲染准备证据而非飞行或模型资格。
# 输入：
#   output：运行专属渲染准备目录。
# 输出：
#   verification：模式、渲染器身份、准备耗时与已加载文件数。
def verify_render_preparation(output: Path) -> dict:
    deployment = _deployment(output)
    verification, _ = _verified_ready(output, deployment)
    return verification


# 功能：
#   1. 所有自有仿真进程退出且原生确认落地或从未尝试生成飞机后，才封存渲染缓存。
#   2. 部署、ready、finished 及落地文件均以本次读取字节绑定，不混用再次读取的新内容。
#   3. 复制并复核缓存后最后创建清单，失败现场保留但不能冒充有效缓存包或飞行验收。
# 输入：
#   output：本次运行专属渲染准备目录。
#   all_owned_processes_exited：必须为字面 True 的自有进程退出确认。
#   flight_vehicle_spawn_attempted：是否实际尝试过生成飞行实体。
#   timing_path：生成过飞机时必须读取的原生落地计时回执路径。
# 输出：
#   result：封存目录、文件数、字节数和终态，不授予飞行资格。
def finalize_render_cache(
    output: Path,
    *,
    all_owned_processes_exited: bool,
    flight_vehicle_spawn_attempted: bool,
    timing_path: Path,
) -> dict:
    if all_owned_processes_exited is not True:
        raise ValueError("RENDER_CACHE_SIMULATION_NOT_STOPPED")
    if type(flight_vehicle_spawn_attempted) is not bool:
        raise ValueError("RENDER_CACHE_SPAWN_STATE_UNKNOWN")
    state, terminal_hash = "NOT_STARTED", None
    if flight_vehicle_spawn_attempted:
        timing, timing_content = read_object_snapshot(timing_path, limit=MAX_METADATA_BYTES,
                                                      error_code="RENDER_CACHE_TIMING_INVALID")
        cleanup = timing.get("cleanup", {})
        observation = cleanup.get("landing_observation") if isinstance(cleanup, dict) else None
        if (not isinstance(observation, dict) or observation.get("state") != "ON_GROUND"
                or observation.get("confirmed") is not True):
            raise ValueError("RENDER_CACHE_NATIVE_LANDING_NOT_CONFIRMED")
        state, terminal_hash = "ON_GROUND", hashlib.sha256(timing_content).hexdigest()
    deployment = _deployment(output)
    ready, ready_digest = _verified_ready(output, deployment)
    finish, finished_digest = _receipt(
        output / "finished.xml", status="complete", identity=deployment["identity_sha256"]
    )
    if (
        finish.get("renderer_sha256") != ready["renderer_sha256"]
        or finish.get("mode") != deployment["mode"]
    ):
        raise ValueError("RENDER_CACHE_FINALIZATION_MISMATCH")
    rows = _file_rows(_xml_rows(finish), output / "output")
    bundle = output / "bundle"
    bundle.mkdir(exist_ok=False)
    _copy_cache_files(rows, output / "output", bundle)
    manifest = {
        "identity": deployment["identity"],
        "identity_sha256": deployment["identity_sha256"],
        "renderer_sha256": ready["renderer_sha256"],
        "files": rows,
        "all_owned_processes_exited": True,
        "terminal_state": state,
        "terminal_timing_sha256": terminal_hash,
        "qualification_granted": False,
        "ready_receipt_sha256": ready_digest,
        "finished_receipt_sha256": finished_digest,
    }
    _new_json(bundle / "cache-bundle.json", manifest)
    result = {
        "bundle": str(bundle),
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "terminal_state": state,
        "qualification_granted": False,
    }
    return result
