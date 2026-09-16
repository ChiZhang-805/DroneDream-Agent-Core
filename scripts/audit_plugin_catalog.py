from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore

_DIRECT_EXTENSION_DISPATCH_METHODS = {
    "_invoke_single_extension",
    "_invoke_multiple_extensions",
    "_invoke_extension_pipeline",
    "invoke_single",
    "invoke_multiple",
    "invoke_pipeline",
    "invoke_single_slot",
}


# 功能：
#   从宿主 Python 源码识别显式分派、策略映射及中间件的插槽与钩子，不补造默认绑定。
# 输入：
#   source_root：准备扫描的生产源码根目录。
# 输出：
#   bindings：语法模式匹配到的插槽与钩子集合，不代表运行可达性或功能验收。
def discover_host_hook_bindings(source_root: Path) -> set[tuple[str, str]]:
    bindings: set[tuple[str, str]] = set()
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target_names = {
                    target.id for target in node.targets if isinstance(target, ast.Name)
                }
                if any(name.endswith("_hooks") for name in target_names) and isinstance(
                    node.value, ast.Dict
                ):
                    for value in node.value.values:
                        if (
                            isinstance(value, (ast.Tuple, ast.List))
                            and len(value.elts) == 2
                            and all(
                                isinstance(item, ast.Constant) and isinstance(item.value, str)
                                for item in value.elts
                            )
                        ):
                            slot, hook = (str(item.value) for item in value.elts)
                            bindings.add((slot, hook))
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if (
                name in _DIRECT_EXTENSION_DISPATCH_METHODS
                and len(node.args) >= 2
                and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in node.args[:2]
                )
            ):
                bindings.add((str(node.args[0].value), str(node.args[1].value)))
            # 中间件的钩子名在调用点选择，插槽固定封装在 ToolRegistry._middleware 中。
            if (
                name == "_middleware"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                bindings.add(("tools.middleware", str(node.args[0].value)))

    return bindings


REQUIRED_CAPABILITY_KINDS = {
    "action-pack",
    "anomaly-detector",
    "attachment-decoder",
    "budget-policy",
    "cache-policy",
    "checkpoint-policy",
    "clock-policy",
    "consensus-policy",
    "context-retriever",
    "context-store",
    "context-strategy",
    "controller",
    "entity-resolver",
    "evaluator",
    "event-bus",
    "evidence-exporter",
    "fallback-policy",
    "fault-injector",
    "harness-profile",
    "harness-scheduler",
    "input-channel",
    "locale-policy",
    "localization",
    "model-provider",
    "model-router",
    "monte-carlo-policy",
    "multimodal-preprocessor",
    "observer",
    "payload-driver",
    "perception",
    "physics-model",
    "plan-optimizer",
    "plan-scorer",
    "plan-validator",
    "planner",
    "result-fusion",
    "retry-policy",
    "runtime-adapter",
    "runtime-amendment",
    "runtime-replanner",
    "runtime-watchdog",
    "scenario-generator",
    "sensor-model",
    "simulator-adapter",
    "state-estimator",
    "structured-decoder",
    "task-decomposer",
    "telemetry-adapter",
    "timeout-policy",
    "token-meter",
    "tool-execution-policy",
    "tool-middleware",
    "tool-router",
    "transport",
    "workflow-topology",
}


# 功能：
#   1. 读取实际插件目录及冻结扩展目录，检查插槽排他性、必需能力类别和宿主钩子绑定。
#   2. 回收本次存储与管理器，返回用于静态目录检查的报告，不执行飞行或模型调用。
# 输入：
#   store_root：目录检查使用的应用存储位置。
#   official_plugins_root：可选的官方插件根目录。
#   plugin_isolator_path：可选的隔离器路径，供管理器检查运行条件。
#   source_root：可选宿主源码根目录，默认使用当前仓库的 src。
# 输出：
#   report：检查状态、问题、能力统计、插槽分组与缺失绑定明细。
def audit(
    store_root: Path,
    *,
    official_plugins_root: Path | None = None,
    plugin_isolator_path: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, object]:
    with ExitStack() as resources:
        store = AppStore(store_root)
        # 先登记存储回收；管理器构造失败或关闭报错，都不能跳过数据库关闭。
        resources.callback(store.close)
        manager = PluginManager(
            store,
            official_plugins_root=official_plugins_root,
            plugin_isolator_path=plugin_isolator_path,
        )
        resources.callback(manager.close)
        plugins = manager.list_plugins()
        snapshot = manager.snapshot()
        extension_catalog = manager.build_extension_registry(snapshot=snapshot).catalog()

    slots: dict[str, list[dict[str, object]]] = defaultdict(list)
    category_labels: dict[str, str] = {}
    capability_kinds: Counter[str] = Counter()
    runtime_kinds: Counter[str] = Counter()
    activation_modes: Counter[str] = Counter()

    for plugin in plugins:
        manifest = plugin["manifest"]
        placement = manifest["placement"]
        category_labels[str(placement["category_id"])] = str(placement["category_label"])
        slots[str(placement["slot_id"])].append(plugin)
        runtime_kinds.update([str(manifest["runtime"]["kind"])])
        activation_modes.update([str(placement["activation_mode"])])
        capability_kinds.update(str(item["kind"]) for item in manifest["capabilities"])

    slot_rows: list[dict[str, object]] = []
    issues: list[str] = []
    for slot_id, members in sorted(slots.items()):
        placements = [member["manifest"]["placement"] for member in members]
        modes = {str(placement["activation_mode"]) for placement in placements}
        enabled = [str(member["plugin_id"]) for member in members if member["enabled"]]
        if len(modes) != 1:
            issues.append(f"SLOT_ACTIVATION_MODE_MISMATCH:{slot_id}")
        mode = sorted(modes)[0]
        if mode == "single" and len(enabled) > 1:
            issues.append(f"SINGLE_SLOT_MULTIPLE_ENABLED:{slot_id}")
        slot_rows.append(
            {
                "slot_id": slot_id,
                "slot_label": str(placements[0]["slot_label"]),
                "category_id": str(placements[0]["category_id"]),
                "activation_mode": mode,
                "plugin_ids": sorted(str(member["plugin_id"]) for member in members),
                "enabled_plugin_ids": sorted(enabled),
                "failure_modes": sorted({str(item["failure_mode"]) for item in placements}),
                "swap_policies": sorted({str(item["swap_policy"]) for item in placements}),
            }
        )

    missing_capability_kinds = sorted(REQUIRED_CAPABILITY_KINDS - set(capability_kinds))
    issues.extend(f"REQUIRED_CAPABILITY_KIND_MISSING:{kind}" for kind in missing_capability_kinds)
    resolved_source_root = (
        source_root.resolve()
        if source_root is not None
        else (Path(__file__).resolve().parents[1] / "src")
    )
    host_hook_bindings = discover_host_hook_bindings(resolved_source_root)
    extension_hook_bindings = {
        (str(row["slot_id"]), str(hook)) for row in extension_catalog for hook in row["hooks"]
    }
    missing_host_bindings = sorted(extension_hook_bindings - host_hook_bindings)
    issues.extend(
        f"PLUGIN_EXTENSION_HOST_BINDING_MISSING:{slot_id}:{hook}"
        for slot_id, hook in missing_host_bindings
    )
    categories = {
        category_id: {
            "label": category_labels[category_id],
            "plugin_count": sum(
                1
                for plugin in plugins
                if plugin["manifest"]["placement"]["category_id"] == category_id
            ),
            "slot_count": sum(1 for row in slot_rows if row["category_id"] == category_id),
        }
        for category_id in sorted(category_labels)
    }
    report = {
        "schema_version": "dronedream.plugin-catalog-audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "store_root": str(store_root.resolve()),
        "status": "accepted" if not issues else "rejected",
        "issues": issues,
        "summary": {
            "plugin_count": len(plugins),
            "enabled_plugin_count": sum(bool(plugin["enabled"]) for plugin in plugins),
            "slot_count": len(slots),
            "category_count": len(categories),
            "slots_with_alternatives": sum(len(members) > 1 for members in slots.values()),
            "capability_kind_count": len(capability_kinds),
            "extension_capability_count": len(extension_catalog),
            "extension_hook_binding_count": len(extension_hook_bindings),
            "host_hook_binding_count": len(host_hook_bindings),
        },
        "activation_modes": dict(sorted(activation_modes.items())),
        "runtime_kinds": dict(sorted(runtime_kinds.items())),
        "capability_kinds": dict(sorted(capability_kinds.items())),
        "categories": categories,
        "slots": slot_rows,
        "extension_host_wiring": {
            "source_root": str(resolved_source_root),
            "status": "accepted" if not missing_host_bindings else "rejected",
            "missing_bindings": [
                {"slot_id": slot_id, "hook": hook} for slot_id, hook in missing_host_bindings
            ],
        },
    }
    return report


# 功能：
#   解析目录检查参数，生成 JSON 报告并打印摘要，将接受或拒绝状态映射为退出码。
# 输入：
#   无；参数由 argparse 从命令行读取。
# 输出：
#   exit_code：目录检查接受时为 0，拒绝时为 1。
def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the resolved plugin catalog.")
    parser.add_argument("store_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--official-plugins-root", type=Path)
    parser.add_argument("--plugin-isolator-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    result = audit(
        args.store_root,
        official_plugins_root=args.official_plugins_root,
        plugin_isolator_path=args.plugin_isolator_path,
        source_root=args.source_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PLUGIN_CATALOG_AUDIT "
        f"status={result['status']} plugins={result['summary']['plugin_count']} "
        f"slots={result['summary']['slot_count']} output={args.output}"
    )
    exit_code = 0 if result["status"] == "accepted" else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
