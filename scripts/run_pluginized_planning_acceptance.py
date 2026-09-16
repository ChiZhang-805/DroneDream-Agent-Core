"""Run a real model-planning workflow with the isolated official MCP plugin enabled."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.asset_runtime_resolver import (
    assert_development_map_binding,
    resolve_development_mission_input,
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from dronedream_agent_app.mission_service import _validate_gateway
from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.assets import (
    load_map_catalog,
    map_entity_is_mentioned,
    resolve_map_entity,
)
from dronedream_agent_core.context import ContextStore
from dronedream_agent_core.contracts import MapAsset, MissionRequest, VehicleAsset
from dronedream_agent_core.model_harness.boundary import (
    HarnessInputEnvelope,
    autonomy_runtime_envelope,
    compile_autonomy_control_plane_receipt,
    selections_from_plugin_snapshot,
)
from dronedream_agent_core.model_harness.memory import MemoryOwnerScope
from dronedream_agent_core.model_harness.model_port import ProviderSettings, StructuredModelPort
from dronedream_agent_core.orchestrator import MissionOrchestrator, PreparationConfig
from dronedream_agent_core.plugin_api import ToolEnvironment


def _current_bundled_pair(
    store: AppStore, default_assets_root: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Resolve the exact pair named current by the bundled release index."""

    try:
        index = json.loads((default_assets_root / "index.json").read_text(encoding="utf-8"))
        pair = index["qualified_pair"]
        packages = pair["packages"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("CURRENT_BUNDLED_ASSET_INDEX_INVALID") from error
    if index.get("schema_version") != "dronedream.bundled-assets.v2" or not isinstance(
        packages, list
    ):
        raise ValueError("CURRENT_BUNDLED_ASSET_INDEX_INVALID")
    by_kind = {str(entry.get("kind")): entry for entry in packages if isinstance(entry, dict)}
    if set(by_kind) != {"map", "vehicle"}:
        raise ValueError("CURRENT_BUNDLED_ASSET_PAIR_INCOMPLETE")
    records: dict[str, dict[str, object]] = {}
    for kind in ("map", "vehicle"):
        entry = by_kind[kind]
        asset_id = entry.get("asset_id")
        content_sha256 = entry.get("content_sha256")
        if not isinstance(asset_id, str) or not isinstance(content_sha256, str):
            raise ValueError("CURRENT_BUNDLED_ASSET_ENTRY_INVALID")
        records[kind] = store.get_asset_version(asset_id, content_sha256)
    job = store.qualified_asset_pair(
        map_asset_id=str(records["map"]["asset_id"]),
        map_content_sha256=str(records["map"]["content_sha256"]),
        vehicle_asset_id=str(records["vehicle"]["asset_id"]),
        vehicle_content_sha256=str(records["vehicle"]["content_sha256"]),
    )
    if job is None:
        raise ValueError("CURRENT_BUNDLED_ASSET_PAIR_NOT_QUALIFIED")
    binding = store.verified_asset_pair_execution_binding(job).model_dump(mode="json")
    return records["map"], records["vehicle"], binding


def run(
    work_root: Path,
    resource_root: Path,
    official_plugins_root: Path,
    *,
    provider: str,
    model: str,
    gateway_base_url: str | None = None,
    grant_env: str = "DRONEDREAM_MODEL_GRANT",
    profile: str | None = None,
    message: str | None = None,
    start_entity: str = "office-launch-pad",
    expected_target_entity: str | None = None,
    plugin_isolator: Path | None = None,
    development_input: Path | None = None,
) -> dict[str, object]:
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"acceptance directory is not empty: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    provider_settings = {
        "openai": ("OPENAI_API_KEY", None, "responses"),
        "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "chat-completions"),
        "kimi": ("KIMI_API_KEY", "https://api.moonshot.ai/v1", "chat-completions"),
    }
    api_key_env, base_url, api_style = provider_settings[provider]
    if gateway_base_url is not None:
        if not grant_env or not grant_env.replace("_", "").isalnum() or not grant_env.isupper():
            raise ValueError("grant environment variable name is invalid")
        api_key_env = grant_env
        base_url = _validate_gateway(gateway_base_url)
        api_style = "chat-completions"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not configured")
    store = AppStore(work_root / "app-store")
    context: ContextStore | None = None
    try:
        default_assets_root = resource_root / "default-assets"
        AssetImportService(store).seed_bundled_sources(default_assets_root)
        map_record, vehicle_record, pair_binding = _current_bundled_pair(store, default_assets_root)
        map_selection = resolve_versioned_map(map_record)
        vehicle_selection = resolve_versioned_vehicle(vehicle_record)
        plugin_manager = PluginManager(
            store,
            official_plugins_root=official_plugins_root,
            plugin_isolator_path=plugin_isolator,
        )
        plugin_manager.enable("dronedream.mission-evidence-gate")
        if profile is not None:
            plugin_manager.apply_profile(profile)
        graph = MapAsset.model_validate_json(map_selection.graph.read_text(encoding="utf-8"))
        semantic_path = map_selection.semantic
        vehicle_sdf = vehicle_selection.vehicle_sdf
        vehicle_metadata_path = vehicle_selection.vehicle_metadata
        asset_input_mode = "qualified-bundled-pair"
        asset_input_binding: dict[str, object] = {
            "mode": asset_input_mode,
            "qualification_granted": True,
            "asset_pair_qualification": pair_binding,
        }
        if development_input is not None:
            development_mission = resolve_development_mission_input(development_input)
            assert_development_map_binding(development_mission, map_selection)
            vehicle_sdf = development_mission.vehicle_sdf
            vehicle_metadata_path = development_mission.vehicle_metadata
            asset_input_mode = "current-development-mission"
            asset_input_binding = {
                "mode": asset_input_mode,
                "qualification_granted": False,
                "base_qualified_pair": pair_binding,
                "manifest_sha256": development_mission.manifest_sha256,
                "manifest_path": str(development_mission.manifest_path),
                "map_asset_id": development_mission.map_asset_id,
                "map_content_sha256": development_mission.map_content_sha256,
                "controller_params_sha256": hashlib.sha256(
                    development_mission.controller_params.read_bytes()
                ).hexdigest(),
                "vehicle_summary_sha256": hashlib.sha256(
                    development_mission.vehicle_summary.read_bytes()
                ).hexdigest(),
                "payload_sdf_sha256": hashlib.sha256(
                    development_mission.payload_sdf.read_bytes()
                ).hexdigest(),
                "payload_mass_kg": development_mission.payload_mass_kg,
            }
        vehicle = VehicleAsset.model_validate_json(
            vehicle_metadata_path.read_text(encoding="utf-8")
        )
        if development_input is not None and vehicle.asset_id != (development_mission.asset_id):
            raise ValueError("DEVELOPMENT_VEHICLE_IDENTITY_MISMATCH")
        map_catalog = load_map_catalog(semantic_path, qualified_graph=graph)
        mission_message = message or (
            "从办公室起降坪起飞，到外卖取餐点取一份外卖，"
            "安全返回办公室起降坪并降落；先只生成计划，不要开始执行。"
        )
        effective_expected_target = expected_target_entity
        if effective_expected_target is None:
            if message is not None:
                raise ValueError("ACCEPTANCE_EXPECTED_TARGET_REQUIRED")
            effective_expected_target = "takeout-pickup-pad"
        resolve_map_entity(start_entity, map_catalog, graph)
        expected_target_node = resolve_map_entity(effective_expected_target, map_catalog, graph)
        if not map_entity_is_mentioned(mission_message, effective_expected_target, map_catalog):
            raise ValueError("ACCEPTANCE_EXPECTED_TARGET_NOT_MENTIONED")
        snapshot = plugin_manager.snapshot()
        control_plane = compile_autonomy_control_plane_receipt(
            selections_from_plugin_snapshot(snapshot),
            effective_maximum_model_calls=48,
            effective_maximum_repair_cycles=6,
        )
        runtime_envelope = autonomy_runtime_envelope(
            MemoryOwnerScope(
                owner_account_id="pluginized-acceptance-account",
                source_edition="autonomy",
            ),
            control_plane,
            snapshot,
            selected_maximum_model_calls=48,
            maximum_intent_rounds=3,
            maximum_planning_rounds=5,
        )
        extension_registry = plugin_manager.build_extension_registry(snapshot=snapshot)
        registry = plugin_manager.build_tool_registry(
            environment=ToolEnvironment(
                map_graph=graph,
                semantic_path=semantic_path,
                vehicle_diameter_m=vehicle.body_radius_m * 2,
                vehicle_height_m=vehicle.body_height_m,
                waypoint_hold_seconds=0.4,
                vehicle=vehicle,
            ),
            snapshot=snapshot,
        )
        settings = ProviderSettings(
            name=provider,  # type: ignore[arg-type]
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            api_style=api_style,  # type: ignore[arg-type]
        )
        primary = StructuredModelPort(
            provider,  # type: ignore[arg-type]
            settings=settings,
            api_key=api_key,
            max_attempts=3,
            timeout_seconds=180,
        )
        critic = StructuredModelPort(
            provider,  # type: ignore[arg-type]
            settings=settings,
            api_key=api_key,
            max_attempts=3,
            timeout_seconds=180,
        )
        context = ContextStore(work_root / "mission-context.sqlite3")
        orchestrator = MissionOrchestrator(
            config=PreparationConfig(
                provider=provider,  # type: ignore[arg-type]
                critic_provider=provider,  # type: ignore[arg-type]
                vehicle_diameter_m=vehicle.body_radius_m * 2,
                vehicle_height_m=vehicle.body_height_m,
            ),
            map_catalog=map_catalog,
            map_graph=graph,
            semantic_path=semantic_path,
            vehicle_sdf=vehicle_sdf,
            vehicle_asset_id=vehicle.asset_id,
            vehicle=vehicle,
            context_store=context,
            primary_port=primary,
            critic_port=critic,
            tool_registry=registry,
            extension_registry=extension_registry,
            plugin_snapshot=snapshot,
        )
        conversation_id = "pluginized-user-acceptance-001"
        vehicle_content_sha256 = (
            str(vehicle_record["content_sha256"])
            if development_input is None
            else str(asset_input_binding["manifest_sha256"])
        )
        asset_versions = {
            "map": {
                "asset_id": map_selection.asset_id,
                "content_sha256": map_selection.content_sha256,
            },
            "vehicle": {
                "asset_id": vehicle.asset_id,
                "content_sha256": vehicle_content_sha256,
            },
        }
        input_envelope = HarnessInputEnvelope(
            request_id=f"request-{uuid4().hex}",
            task_id=conversation_id,
            thread_id=conversation_id,
            owner_binding_sha256=runtime_envelope.owner_binding_sha256,
            tenant_binding_sha256=runtime_envelope.tenant_binding_sha256,
            source_edition=runtime_envelope.source_edition,
            control_plane_selection_sha256=control_plane.selection_sha256,
            current_request={
                "message": mission_message,
                "locale": "zh-CN",
                "input_channel": "text",
                "start_entity": start_entity,
                "attachment_ids": [],
                "map_asset_id": asset_versions["map"]["asset_id"],
                "map_content_sha256": asset_versions["map"]["content_sha256"],
                "vehicle_asset_id": asset_versions["vehicle"]["asset_id"],
                "vehicle_content_sha256": asset_versions["vehicle"]["content_sha256"],
                "model_selection_id": model,
            },
            session_context={
                "asset_versions": asset_versions,
                "asset_input_binding": asset_input_binding,
            },
        )
        request_input_metadata: dict[str, object] = {
            "asset_versions": asset_versions,
            "asset_input_binding": asset_input_binding,
            "model_harness_input": input_envelope.model_dump(mode="json"),
            "model_harness_control_plane": control_plane.model_dump(mode="json"),
            "model_harness_runtime": runtime_envelope.model_dump(mode="json"),
        }
        if asset_input_binding["qualification_granted"] is True:
            request_input_metadata["asset_pair_qualification"] = pair_binding
        prepared = orchestrator.prepare(
            MissionRequest(
                conversation_id=conversation_id,
                message=mission_message,
                start_entity=start_entity,
                locale="zh-CN",
                input_metadata=request_input_metadata,
            ),
            work_root / "plan",
        )
        if (
            resolve_map_entity(prepared.intent.target_entity, map_catalog, graph)
            != expected_target_node
        ):
            raise ValueError("ACCEPTANCE_PLANNED_TARGET_MISMATCH")
        roles = [record.role for record in prepared.model_calls]
        plugin_receipt = next(
            receipt
            for receipt in prepared.tool_receipts
            if receipt.tool_id == "mission.evidence-requirements"
        )
        assert "plugin_router" in roles
        assert plugin_receipt.outcome == "accepted"
        assert plugin_receipt.plugin_package_sha256
        summary = {
            "schema_version": "dronedream.pluginized-planning-acceptance.v1",
            "status": prepared.status,
            "conversation_id": prepared.contract.conversation_id,
            "contract_id": prepared.contract.contract_id,
            "provider": provider,
            "model": model,
            "harness_profile": profile or "harness.profile-balanced",
            "active_plugin_ids": sorted(item.plugin_id for item in snapshot.plugins),
            "model_roles": roles,
            "model_calls": len(prepared.model_calls),
            "model_input_tokens": sum(record.input_tokens or 0 for record in prepared.model_calls),
            "model_output_tokens": sum(
                record.output_tokens or 0 for record in prepared.model_calls
            ),
            "plugin_snapshot_id": prepared.plugin_snapshot.snapshot_id,
            "plugin_catalog_sha256": prepared.plugin_snapshot.catalog_sha256,
            "plugin_id": plugin_receipt.plugin_id,
            "plugin_package_sha256": plugin_receipt.plugin_package_sha256,
            "plugin_output_sha256": plugin_receipt.output_sha256,
            "route_node_count": len(prepared.execution_route.node_ids),
            "minimum_clearance_m": prepared.route_clearance.minimum_clearance_m,
            "planning_attempts": prepared.planning_attempts,
            "asset_input_mode": asset_input_mode,
            "flight_qualification_granted": bool(asset_input_binding["qualification_granted"]),
            "map_asset_id": map_selection.asset_id,
            "map_content_sha256": map_selection.content_sha256,
            "vehicle_asset_id": vehicle.asset_id,
            "vehicle_content_sha256": vehicle_content_sha256,
            "map_graph_sha256": hashlib.sha256(map_selection.graph.read_bytes()).hexdigest(),
            "map_semantic_sha256": hashlib.sha256(semantic_path.read_bytes()).hexdigest(),
            "vehicle_sdf_sha256": hashlib.sha256(vehicle_sdf.read_bytes()).hexdigest(),
            "vehicle_metadata_sha256": hashlib.sha256(
                vehicle_metadata_path.read_bytes()
            ).hexdigest(),
            "expected_target_entity": effective_expected_target,
            "development_input_binding": asset_input_binding,
        }
        (work_root / "acceptance-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        if context is not None:
            context.close()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_root", type=Path)
    parser.add_argument("resource_root", type=Path)
    parser.add_argument("official_plugins_root", type=Path)
    parser.add_argument("--provider", choices=["openai", "deepseek", "kimi"], default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--gateway-base-url",
        default=None,
        help="Use an authenticated DroneDream model-gateway grant instead of a provider key.",
    )
    parser.add_argument(
        "--grant-env",
        default="DRONEDREAM_MODEL_GRANT",
        help="Environment variable containing the short-lived gateway grant.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Apply one Harness profile before freezing the plugin snapshot.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Override the natural-language acceptance mission.",
    )
    parser.add_argument("--start-entity", default="office-launch-pad")
    parser.add_argument("--expected-target-entity")
    parser.add_argument("--plugin-isolator", type=Path)
    parser.add_argument(
        "--development-input",
        type=Path,
        help=(
            "Hash-pinned current-development manifest. It never grants release "
            "or hardware flight qualification."
        ),
    )
    args = parser.parse_args()
    print(
        "PLANNING_ACCEPTANCE_START "
        f"provider={args.provider} model={args.model} "
        f"profile={args.profile or 'harness.profile-balanced'}",
        flush=True,
    )
    try:
        summary = run(
            args.work_root,
            args.resource_root,
            args.official_plugins_root,
            provider=args.provider,
            model=args.model,
            gateway_base_url=args.gateway_base_url,
            grant_env=args.grant_env,
            profile=args.profile,
            message=args.message,
            start_entity=args.start_entity,
            expected_target_entity=args.expected_target_entity,
            plugin_isolator=args.plugin_isolator,
            development_input=args.development_input,
        )
    except Exception as exc:
        failure = {
            "schema_version": "dronedream.pluginized-planning-acceptance-failure.v1",
            "status": "failed",
            "provider": args.provider,
            "model": args.model,
            "harness_profile": args.profile or "harness.profile-balanced",
            "exception_type": type(exc).__name__,
            "error": str(exc),
        }
        args.work_root.mkdir(parents=True, exist_ok=True)
        (args.work_root / "acceptance-failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
