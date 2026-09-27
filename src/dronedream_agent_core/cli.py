"""Development CLI for real model and runtime acceptance operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .assets import load_map_catalog
from .collision import validate_route_clearance
from .context import ContextStore
from .contracts import (
    GraphRoute,
    IntentArtifact,
    IntentCritique,
    MapAsset,
    MapCatalog,
    MissionRequest,
    RouteQuery,
    StrictModel,
    VehicleAsset,
)
from .execution import execute_prepared_mission, reverify_prepared_run
from .gazebo_adapter import run_px4_gazebo_track
from .hashing import sha256_json
from .model_harness.boundary import (
    HarnessInputEnvelope,
    autonomy_runtime_envelope,
    compile_autonomy_control_plane_receipt,
    selections_from_plugin_snapshot,
)
from .model_harness.memory import MemoryOwnerScope
from .model_harness.model_port import StructuredModelPort
from .navigation import shortest_route
from .orchestrator import MissionOrchestrator, PreparationConfig
from .plugin_api import ToolEnvironment, build_discovered_tool_registry
from .plugin_files import check_plain_plugin_path, read_plugin_file
from .prompts import INTENT_CRITIC, INTENT_PARSER
from .px4_track import route_to_px4_track
from .runtime_interrupt import submit_runtime_message

MAX_CLI_ARTIFACT_BYTES = 16 * 1024 * 1024
CliArtifact = TypeVar("CliArtifact", bound=StrictModel)


class CliProfileError(RuntimeError):
    """A direct development/runtime command was invoked outside its explicit profile."""


# 功能：
#   限制直接调用模型和底层运行器的入口，桌面产品不能隐式走无账户的开发通道。
# 输入：
#   args：包含显式运行 profile 的命令参数。
# 输出：
#   None：允许时继续，否则在产生副作用前抛出错误。
def _require_development_profile(args: argparse.Namespace) -> None:
    if getattr(args, "profile", "locked") not in {"development", "test"}:
        raise CliProfileError("DEVELOPMENT_OR_TEST_PROFILE_REQUIRED")


# 功能：
#   为受管理的任务执行检查命令入口；实际账户授权仍由执行凭据单独验证。
# 输入：
#   args：运行 profile 参数。
# 输出：
#   None：通过入口检查，不代表任务执行已获授权。
def _require_runtime_manager_profile(args: argparse.Namespace) -> None:
    if getattr(args, "profile", "locked") not in {"runtime-manager", "development", "test"}:
        raise CliProfileError("RUNTIME_MANAGER_PROFILE_REQUIRED")


# 功能：
#   有界读取独立 JSON 快照，拒绝重复键、非有限数、链接以及读取期间被替换的文件。
# 输入：
#   path：命令行明确指定的输入文件。
# 输出：
#   value：标准 JSON 对象。
def _read_json(path: Path) -> dict[str, Any]:
    raw = read_plugin_file(path, limit=MAX_CLI_ARTIFACT_BYTES)
    value = decode_json(raw, limit=MAX_CLI_ARTIFACT_BYTES, node_limit=1_000_000)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


# 功能：
#   在普通 JSON 检查后按精确契约解析，禁止把字符串开关或布尔坐标自动转换成合法输入。
# 输入：
#   path：待读取的契约文件。
#   contract：期望的契约模型类。
# 输出：
#   artifact：经过严格校验的独立模型。
def _read_contract(path: Path, contract: type[CliArtifact]) -> CliArtifact:
    artifact = contract.model_validate_json(
        encode_json(_read_json(path), limit=MAX_CLI_ARTIFACT_BYTES, node_limit=1_000_000),
        strict=True,
    )
    return artifact


# 功能：
#   在付费调用之前检查输出位置，防止完成调用后才发现会覆盖已有证据。
# 输入：
#   path：可选输出路径，未提供表示仅打印。
# 输出：
#   None：路径可用时继续；真正写入仍用独占创建防止并发覆盖。
def _check_output(path: Path | None) -> None:
    if path is not None:
        check_plain_plugin_path(path)
        if path.exists():
            raise FileExistsError("CLI_OUTPUT_ALREADY_EXISTS")


# 功能：
#   将标准 JSON 产物独占写入新文件并刷新磁盘，不覆盖历史证据。
# 输入：
#   path：尚不存在的输出路径。
#   value：可序列化为标准 JSON 的产物。
# 输出：
#   None：写入成功；失败的部分文件保留现场，不能当成完成回执。
def _write_output(path: Path, value: dict[str, Any]) -> None:
    raw = (encode_json(value, limit=MAX_CLI_ARTIFACT_BYTES, node_limit=1_000_000) + "\n").encode(
        "utf-8"
    )
    _check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    check_plain_plugin_path(path)
    with path.open("xb") as destination:
        destination.write(raw)
        destination.flush()
        os.fsync(destination.fileno())


# 功能：
#   为显式开发或测试准备同一 Harness 输入结构，隔离的开发身份不能充当用户账户或飞行授权。
# 输入：
#   request：当前自然语言请求。
#   config：调用及修复预算。
#   plugin_snapshot：本次发现的插件快照。
# 输出：
#   bound_request：带控制面和运行边界的请求，现有边界仍由下游严格校验。
def _bind_development_harness_request(
    request: MissionRequest, *, config: PreparationConfig, plugin_snapshot: Any
) -> MissionRequest:
    if isinstance(request.input_metadata.get("model_harness_input"), dict):
        return request
    repair_cycles = config.max_intent_rounds + config.max_planning_rounds - 2
    selections = selections_from_plugin_snapshot(plugin_snapshot)
    receipt = compile_autonomy_control_plane_receipt(
        selections,
        effective_maximum_model_calls=config.maximum_model_calls,
        effective_maximum_repair_cycles=repair_cycles,
    )
    runtime = autonomy_runtime_envelope(
        MemoryOwnerScope(
            owner_account_id="development-cli",
            source_edition="autonomy",
        ),
        receipt,
        plugin_snapshot,
        selected_maximum_model_calls=config.maximum_model_calls,
        maximum_intent_rounds=config.max_intent_rounds,
        maximum_planning_rounds=config.max_planning_rounds,
    )
    envelope = HarnessInputEnvelope(
        request_id=f"request-{uuid4().hex}",
        task_id=request.conversation_id,
        thread_id=request.conversation_id,
        owner_binding_sha256=runtime.owner_binding_sha256,
        tenant_binding_sha256=runtime.tenant_binding_sha256,
        source_edition="autonomy",
        control_plane_selection_sha256=receipt.selection_sha256,
        current_request={
            "message": request.message,
            "locale": request.locale,
            "input_channel": request.input_channel,
            "start_entity": request.start_entity,
            "attachment_ids": [item.attachment_id for item in request.attachments],
            "planning_preferences": {
                key: value
                for key, value in request.input_metadata.items()
                if key
                not in {
                    "model_harness_input",
                    "model_harness_control_plane",
                    "model_harness_runtime",
                }
            },
        },
    )
    metadata = dict(request.input_metadata)
    metadata["model_harness_control_plane"] = receipt.model_dump(mode="json")
    metadata["model_harness_runtime"] = runtime.model_dump(mode="json")
    metadata["model_harness_input"] = envelope.model_dump(mode="json")
    bound_request = request.model_copy(update={"input_metadata": metadata})
    return bound_request


# 功能：
#   用真实供应商执行单次意图角色调用，成功或异常都关闭端口，不执行飞行。
# 输入：
#   args：开发 profile、供应商、请求、目录和可选输出路径。
# 输出：
#   exit_code：成功为零，模型和文件错误向调用方传播。
def _model_probe(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    _check_output(args.output)
    request = _read_contract(args.request, MissionRequest)
    map_catalog = _read_contract(args.map_catalog, MapCatalog).model_dump(mode="json")
    port = StructuredModelPort(args.provider, max_attempts=args.max_attempts)
    try:
        result = port.call(
            role="intent_parser",
            output_type=IntentArtifact,
            instructions=INTENT_PARSER,
            input_artifact={
                "mission_request": request.model_dump(mode="json"),
                "map_catalog": map_catalog,
            },
            context_id=request.conversation_id,
        )
    finally:
        port.close()
    output = {
        "artifact": result.artifact.model_dump(mode="json"),
        "model_call": result.record.model_dump(mode="json"),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        _write_output(args.output, output)
    else:
        print(rendered, end="")
    exit_code = 0
    return exit_code


# 功能：
#   导出当前语义目录，不把无图语义升级成可飞路线，也不覆盖原有输出。
# 输入：
#   args：语义路径及新输出路径。
# 输出：
#   exit_code：成功写入目录时为零。
def _export_map_catalog(args: argparse.Namespace) -> int:
    _check_output(args.output)
    catalog = load_map_catalog(args.semantic)
    _write_output(args.output, catalog.model_dump(mode="json"))
    print(
        f"MAP_CATALOG_EXPORTED entities={len(catalog.entities)} "
        f"topology_available={catalog.topology_available} output={args.output}"
    )
    exit_code = 0
    return exit_code


# 功能：
#   独立评审当前候选意图，散列绑定实际候选而非沿用输入包装内未经检查的散列。
# 输入：
#   args：开发 profile、供应商、原请求、候选、目录及新输出位置。
# 输出：
#   exit_code：成功写入评审回执时为零；退出前关闭供应商端口。
def _intent_critic(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    _check_output(args.output)
    request = _read_contract(args.request, MissionRequest)
    intent_document = _read_json(args.intent)
    intent_value = intent_document.get("artifact", intent_document)
    intent = IntentArtifact.model_validate_json(encode_json(intent_value), strict=True)
    map_catalog = _read_contract(args.map_catalog, MapCatalog).model_dump(mode="json")
    port = StructuredModelPort(args.provider, max_attempts=args.max_attempts)
    try:
        result = port.call(
            role="intent_critic",
            output_type=IntentCritique,
            instructions=INTENT_CRITIC,
            input_artifact={
                "mission_request": request.model_dump(mode="json"),
                "candidate_intent": intent.model_dump(mode="json"),
                "map_catalog": map_catalog,
            },
            context_id=request.conversation_id,
        )
    finally:
        port.close()
    output = {
        "artifact": result.artifact.model_dump(mode="json"),
        "candidate_intent_sha256": sha256_json(intent),
        "model_call": result.record.model_dump(mode="json"),
    }
    _write_output(args.output, output)
    exit_code = 0
    return exit_code


# 功能：
#   拒绝旧学校地图生成入口，不能凭历史轨迹和固定高度生成貌似已核验的新地图。
# 输入：
#   args：保留旧命令参数，仅用于返回明确的迁移错误。
# 输出：
#   result：不产生地图，始终抛出停用错误；当前图须由资产资格流程提供。
def _export_navigation_graph(args: argparse.Namespace) -> int:
    raise CliProfileError("RETIRED_MAP_GRAPH_EXPORT_USE_QUALIFIED_ASSET_GRAPH")


# 功能：
#   在显式所选图中解析起终点并求宏观候选路线，不沿用内置路线或直接下发运动。
# 输入：
#   args：图、起终点、边资格要求及输出路径。
# 输出：
#   exit_code：新路线成功导出时为零。
def _plan_graph_route(args: argparse.Namespace) -> int:
    _check_output(args.output)
    graph = _read_contract(args.graph, MapAsset)
    start_node = graph.named_entities.get(args.start, args.start)
    goal_node = graph.named_entities.get(args.goal, args.goal)
    route = shortest_route(
        graph,
        RouteQuery(
            start_node=start_node,
            goal_node=goal_node,
            require_flight_verified_edges=args.require_flight_verified,
        ),
    )
    _write_output(args.output, route.model_dump(mode="json"))
    print(
        f"GRAPH_ROUTE_PLANNED points={len(route.node_ids)} "
        f"length_m={route.route_length_m:.3f} "
        f"all_edges_flight_verified={route.all_edges_flight_verified} output={args.output}"
    )
    exit_code = 0
    return exit_code


# 功能：
#   用当前车辆包络和地图碰撞原语检查候选路线，保留通过或拒绝报告。
# 输入：
#   args：路线、语义、车辆元数据、采样间隔及输出路径。
# 输出：
#   exit_code：通过为零，拒绝为二；报告不能代替实时感知安全检查。
def _validate_route(args: argparse.Namespace) -> int:
    _check_output(args.output)
    route = _read_contract(args.route, GraphRoute)
    vehicle = _read_contract(args.vehicle_metadata, VehicleAsset)
    report = validate_route_clearance(
        route,
        args.semantic,
        vehicle_diameter_m=vehicle.body_radius_m * 2.0,
        vehicle_height_m=vehicle.body_height_m,
        sample_interval_m=args.sample_interval_m,
    )
    _write_output(args.output, report.model_dump(mode="json"))
    print(
        f"ROUTE_CLEARANCE accepted={report.accepted} samples={report.sample_count} "
        f"collisions={report.collision_count} minimum_m={report.minimum_clearance_m:.6f} "
        f"output={args.output}"
    )
    exit_code = 0 if report.accepted else 2
    return exit_code


# 功能：
#   从当前路线和车辆几何导出 PX4 参考轨迹；实时控制由执行器和本地专家另行完成。
# 输入：
#   args：显式开发 profile、路线图、语义、车辆及停留时间。
# 输出：
#   exit_code：参考轨迹成功导出时为零。
def _export_px4_track(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    _check_output(args.output)
    route = _read_contract(args.route, GraphRoute)
    graph = _read_contract(args.graph, MapAsset)
    vehicle = _read_contract(args.vehicle_metadata, VehicleAsset)
    track = route_to_px4_track(
        route,
        graph,
        args.semantic,
        vehicle=vehicle,
        waypoint_hold_seconds=args.waypoint_hold_seconds,
    )
    _write_output(args.output, track.model_dump(mode="json"))
    print(f"PX4_TRACK_EXPORTED points={len(track.points)} output={args.output}")
    exit_code = 0
    return exit_code


# 功能：
#   进入真实 PX4 SITL/Gazebo 开发运行链，传递显式资产、本地模型包及资格凭据。
# 输入：
#   args：运行目录、仿真资源、执行器、专家包及时间预算。
# 输出：
#   exit_code：运行证据为 verified 时为零，其余状态为二。
def _run_px4_track(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    evidence = run_px4_gazebo_track(
        run_dir=args.run_dir,
        world_sdf=args.world_sdf,
        semantic_path=args.semantic,
        vehicle_sdf=args.vehicle_sdf,
        vehicle_metadata_path=args.vehicle_metadata,
        route_path=args.route,
        track_path=args.track,
        clearance_path=args.clearance,
        controller_params_path=args.controller_params,
        px4_root=args.px4_root,
        executor_path=args.executor,
        ros_workspace=args.ros_workspace,
        local_navigation_provider=args.local_navigation_provider,
        local_navigation_fallback_provider=args.local_navigation_fallback_provider,
        local_navigation_model_timeout_seconds=args.local_navigation_model_timeout_seconds,
        local_navigation_fallback_model_timeout_seconds=(
            args.local_navigation_fallback_model_timeout_seconds
        ),
        local_navigation_period_seconds=args.local_navigation_period_seconds,
        local_navigation_context_id=args.local_navigation_context_id,
        local_navigation_visual_enabled=args.local_navigation_visual_enabled,
        local_navigation_control_authority_required=(
            args.require_local_navigation_control_authority
        ),
        bounded_hybrid_control=args.bounded_hybrid_control,
        heading_policy=args.heading_policy,
        maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
        multimodal_dataset_root=args.multimodal_dataset_root,
        multimodal_flight_id=args.multimodal_flight_id,
        multimodal_dataset_maximum_mib=args.multimodal_dataset_maximum_mib,
        multimodal_record_period_seconds=args.multimodal_record_period_seconds,
        local_policy_package_paths=tuple(args.local_policy_package),
        local_policy_qualification_paths=tuple(args.local_policy_qualification),
        local_policy_simulation_admission_paths=tuple(args.local_policy_simulation_admission),
        local_policy_trial_path=args.local_policy_trial,
        development_payload_collection=args.development_payload_collection,
    )
    print(
        f"PX4_GAZEBO_RUN status={evidence['status']} "
        f"output={args.run_dir / 'mission_evidence.json'}"
    )
    exit_code = 0 if evidence["status"] == "verified" else 2
    return exit_code


# 功能：
#   通过真实模型、当前地图和插件准备任务，开发身份仅用于隔离规划，不能签发账户授权。
# 输入：
#   args：请求、资产、供应商、预算、上下文数据库及新计划目录。
# 输出：
#   exit_code：准备成功为零，阻断或失败传播异常；数据库在退出时关闭。
def _prepare_mission(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    request = _read_contract(args.request, MissionRequest)
    graph = _read_contract(args.graph, MapAsset)
    vehicle = _read_contract(args.vehicle_metadata, VehicleAsset)
    catalog = load_map_catalog(args.semantic, qualified_graph=graph)
    config = PreparationConfig(
        provider=args.provider,
        critic_provider=args.critic_provider,
        max_provider_attempts=args.max_provider_attempts,
        max_intent_rounds=args.max_intent_rounds,
        max_planning_rounds=args.max_planning_rounds,
        model_timeout_seconds=args.model_timeout_seconds,
        vehicle_diameter_m=vehicle.body_radius_m * 2.0,
        vehicle_height_m=vehicle.body_height_m,
        waypoint_hold_seconds=args.waypoint_hold_seconds,
    )
    tool_registry, plugin_snapshot = build_discovered_tool_registry(
        ToolEnvironment(
            map_graph=graph,
            semantic_path=args.semantic,
            vehicle_diameter_m=vehicle.body_radius_m * 2.0,
            vehicle_height_m=vehicle.body_height_m,
            waypoint_hold_seconds=config.waypoint_hold_seconds,
            vehicle=vehicle,
            planning_phase="initial",
        )
    )
    request = _bind_development_harness_request(
        request,
        config=config,
        plugin_snapshot=plugin_snapshot,
    )
    context = ContextStore(args.context_db)
    orchestrator = None
    try:
        orchestrator = MissionOrchestrator(
            config=config,
            map_catalog=catalog,
            map_graph=graph,
            semantic_path=args.semantic,
            vehicle_sdf=args.vehicle_sdf,
            vehicle_asset_id=vehicle.asset_id,
            vehicle=vehicle,
            context_store=context,
            tool_registry=tool_registry,
            plugin_snapshot=plugin_snapshot,
        )
        prepared = orchestrator.prepare(request, args.output_dir)
    finally:
        try:
            if orchestrator is not None:
                orchestrator.close()
        finally:
            context.close()
    print(
        f"MISSION_PREPARED status={prepared.status} "
        f"model_calls={len(prepared.model_calls)} "
        f"planning_attempts={prepared.planning_attempts} "
        f"route_points={len(prepared.execution_route.node_ids)} "
        f"output={args.output_dir / 'prepared-mission.json'}"
    )
    exit_code = 0
    return exit_code


# 功能：
#   把已确认计划和执行凭据交给受管理执行链，继续核验资源、权限与真实回执。
# 输入：
#   args：计划、账户授权、确认标识、仿真资源和实时模型配置。
# 输出：
#   exit_code：任务由证据验证成功时为零，否则为二。
def _execute_prepared_mission(args: argparse.Namespace) -> int:
    _require_runtime_manager_profile(args)
    context = ContextStore(args.context_db)
    try:
        result = execute_prepared_mission(
            prepared_path=args.prepared,
            execution_authority_path=args.execution_authority,
            confirm_contract_id=args.confirm_contract_id,
            run_dir=args.run_dir,
            world_sdf=args.world_sdf,
            semantic_path=args.semantic,
            vehicle_sdf=args.vehicle_sdf,
            controller_params_path=args.controller_params,
            executor_path=args.executor,
            px4_root=args.px4_root,
            ros_workspace=args.ros_workspace,
            completion_provider=args.completion_provider,
            context_store=context,
            model_timeout_seconds=args.model_timeout_seconds,
            checkpoint_provider=args.checkpoint_provider,
            checkpoint_executor_path=args.checkpoint_executor,
            checkpoint_timeout_seconds=args.checkpoint_timeout_seconds,
            runtime_interrupt_provider=args.runtime_interrupt_provider,
            runtime_hold_timeout_seconds=args.runtime_hold_timeout_seconds,
            runtime_decision_timeout_seconds=args.runtime_decision_timeout_seconds,
            runtime_replan_hold_seconds=args.runtime_replan_hold_seconds,
            local_navigation_provider=args.local_navigation_provider,
            local_navigation_fallback_provider=(args.local_navigation_fallback_provider),
            local_navigation_model_timeout_seconds=(args.local_navigation_model_timeout_seconds),
            local_navigation_fallback_model_timeout_seconds=(
                args.local_navigation_fallback_model_timeout_seconds
            ),
            local_navigation_period_seconds=args.local_navigation_period_seconds,
            local_navigation_context_id=args.local_navigation_context_id,
            local_navigation_visual_enabled=args.local_navigation_visual_enabled,
            local_navigation_control_authority_required=(
                args.require_local_navigation_control_authority
            ),
            bounded_hybrid_control=args.bounded_hybrid_control,
            independent_route_control=args.independent_route_control,
            heading_policy=args.heading_policy,
            maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
            multimodal_dataset_root=args.multimodal_dataset_root,
            multimodal_flight_id=args.multimodal_flight_id,
            multimodal_dataset_maximum_mib=args.multimodal_dataset_maximum_mib,
            multimodal_record_period_seconds=args.multimodal_record_period_seconds,
            local_policy_package_paths=tuple(args.local_policy_package),
            local_policy_qualification_paths=tuple(args.local_policy_qualification),
            local_policy_simulation_admission_paths=tuple(args.local_policy_simulation_admission),
            local_policy_trial_path=args.local_policy_trial,
            development_payload_collection=args.development_payload_collection,
            map_graph_path=args.map_graph,
            vehicle_metadata_path=args.vehicle_metadata,
            simulation_map_fusion=args.simulation_map_fusion,
        )
    finally:
        context.close()
    print(
        f"MISSION_WORKFLOW status={result.status} contract={result.contract_id} "
        f"output={args.run_dir / 'workflow-result.json'}"
    )
    exit_code = 0 if result.status == "verified" else 2
    return exit_code


# 功能：
#   给精确绑定的当前执行提交自然语言变更，先触发受控中断再由模型判断变更内容。
# 输入：
#   args：显式开发 profile、运行目录与用户文本。
# 输出：
#   exit_code：提交成功为零，不代表变更已被批准或执行。
def _submit_runtime_message(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    message = submit_runtime_message(control_dir=args.run_dir / "runtime-control", text=args.text)
    print(
        f"RUNTIME_MESSAGE_ACCEPTED message_id={message.message_id} "
        f"mission_id={message.mission_id} execution_id={message.execution_id}"
    )
    exit_code = 0
    return exit_code


# 功能：
#   对已结束运行的不可变证据重新执行完成评审，不重飞也不制造运行证据。
# 输入：
#   args：开发 profile、计划、原运行目录、资产和评审模型。
# 输出：
#   exit_code：重新验证成功为零，否则为二。
def _reverify_prepared_run(args: argparse.Namespace) -> int:
    _require_development_profile(args)
    context = ContextStore(args.context_db)
    try:
        result = reverify_prepared_run(
            prepared_path=args.prepared,
            confirm_contract_id=args.confirm_contract_id,
            run_dir=args.run_dir,
            semantic_path=args.semantic,
            vehicle_sdf=args.vehicle_sdf,
            completion_provider=args.completion_provider,
            context_store=context,
            model_timeout_seconds=args.model_timeout_seconds,
        )
    finally:
        context.close()
    print(
        f"MISSION_REVERIFIED status={result.status} contract={result.contract_id} "
        f"output={args.run_dir / 'workflow-result.json'}"
    )
    exit_code = 0 if result.status == "verified" else 2
    return exit_code


# 功能：
#   构造明确区分开发诊断与受管理执行的命令接口，资产和模型包由参数指定。
# 输入：
#   无。
# 输出：
#   parser：默认 locked 的参数解析器；旧地图导出命令仅返回停用错误。
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dronedream-agent")
    parser.add_argument(
        "--profile",
        choices=("locked", "runtime-manager", "development", "test"),
        default="locked",
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("model-probe", help="call one real structured intent role")
    probe.add_argument("--provider", required=True)
    probe.add_argument("--request", type=Path, required=True)
    probe.add_argument("--map-catalog", type=Path, required=True)
    probe.add_argument("--output", type=Path)
    probe.add_argument("--max-attempts", type=int, default=3)
    probe.set_defaults(handler=_model_probe)
    catalog = subparsers.add_parser(
        "export-map-catalog", help="import a qualified map semantic artifact"
    )
    catalog.add_argument("--semantic", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.set_defaults(handler=_export_map_catalog)
    critic = subparsers.add_parser(
        "intent-critic", help="independently review a structured intent with a real model"
    )
    critic.add_argument("--provider", required=True)
    critic.add_argument("--request", type=Path, required=True)
    critic.add_argument("--intent", type=Path, required=True)
    critic.add_argument("--map-catalog", type=Path, required=True)
    critic.add_argument("--output", type=Path, required=True)
    critic.add_argument("--max-attempts", type=int, default=3)
    critic.set_defaults(handler=_intent_critic)
    graph = subparsers.add_parser(
        "export-navigation-graph", help="retired: use the current qualified asset graph"
    )
    graph.add_argument("--semantic", type=Path, required=True)
    graph.add_argument("--verified-track", type=Path, required=True)
    graph.add_argument("--mission-evidence", type=Path, required=True)
    graph.add_argument("--output", type=Path, required=True)
    graph.set_defaults(handler=_export_navigation_graph)
    route = subparsers.add_parser("plan-graph-route", help="run generic Dijkstra routing")
    route.add_argument("--graph", type=Path, required=True)
    route.add_argument("--start", required=True)
    route.add_argument("--goal", required=True)
    route.add_argument("--require-flight-verified", action="store_true")
    route.add_argument("--output", type=Path, required=True)
    route.set_defaults(handler=_plan_graph_route)
    clearance = subparsers.add_parser(
        "validate-route", help="sample a route against qualified map collision primitives"
    )
    clearance.add_argument("--route", type=Path, required=True)
    clearance.add_argument("--semantic", type=Path, required=True)
    clearance.add_argument("--vehicle-metadata", type=Path, required=True)
    clearance.add_argument("--sample-interval-m", type=float, default=0.1)
    clearance.add_argument("--output", type=Path, required=True)
    clearance.set_defaults(handler=_validate_route)
    track = subparsers.add_parser(
        "export-px4-track", help="convert a validated ENU route for the real PX4 executor"
    )
    track.add_argument("--route", type=Path, required=True)
    track.add_argument("--graph", type=Path, required=True)
    track.add_argument("--semantic", type=Path, required=True)
    track.add_argument("--vehicle-metadata", type=Path, required=True)
    track.add_argument("--waypoint-hold-seconds", type=float, default=0.4)
    track.add_argument("--output", type=Path, required=True)
    track.set_defaults(handler=_export_px4_track)
    runtime = subparsers.add_parser(
        "run-px4-track", help="execute a validated track in real PX4 SITL and Gazebo"
    )
    runtime.add_argument("--run-dir", type=Path, required=True)
    runtime.add_argument("--world-sdf", type=Path, required=True)
    runtime.add_argument("--semantic", type=Path, required=True)
    runtime.add_argument("--vehicle-sdf", type=Path, required=True)
    runtime.add_argument("--vehicle-metadata", type=Path, required=True)
    runtime.add_argument("--route", type=Path, required=True)
    runtime.add_argument("--track", type=Path, required=True)
    runtime.add_argument("--clearance", type=Path, required=True)
    runtime.add_argument("--controller-params", type=Path, required=True)
    runtime.add_argument("--px4-root", type=Path, default=Path("/opt/PX4-Autopilot"))
    runtime.add_argument(
        "--executor",
        type=Path,
        required=True,
    )
    runtime.add_argument(
        "--ros-workspace",
        type=Path,
        required=True,
    )
    runtime.add_argument("--local-navigation-provider")
    runtime.add_argument("--local-navigation-fallback-provider")
    runtime.add_argument("--local-policy-package", type=Path, action="append", default=[])
    runtime.add_argument("--local-policy-trial", type=Path)
    runtime.add_argument("--local-policy-qualification", type=Path, action="append", default=[])
    runtime.add_argument(
        "--local-policy-simulation-admission",
        type=Path,
        action="append",
        default=[],
    )
    runtime.add_argument("--local-navigation-model-timeout-seconds", type=float, default=10.0)
    runtime.add_argument(
        "--local-navigation-fallback-model-timeout-seconds", type=float, default=10.0
    )
    runtime.add_argument("--local-navigation-period-seconds", type=float, default=3.0)
    runtime.add_argument("--local-navigation-context-id")
    runtime.add_argument("--local-navigation-visual-enabled", action="store_true")
    runtime.add_argument(
        "--heading-policy",
        choices=("measured-hold", "route-tangent-relative"),
        default="measured-hold",
    )
    runtime.add_argument("--maximum-yaw-rate-deg-s", type=float, default=20.0)
    runtime.add_argument("--multimodal-dataset-root", type=Path)
    runtime.add_argument("--multimodal-flight-id")
    runtime.add_argument("--multimodal-dataset-maximum-mib", type=int, default=5_120)
    runtime.add_argument("--multimodal-record-period-seconds", type=float, default=0.1)
    runtime.add_argument("--require-local-navigation-control-authority", action="store_true")
    runtime.add_argument("--bounded-hybrid-control", action="store_true")
    runtime.add_argument("--development-payload-collection", action="store_true")
    runtime.set_defaults(handler=_run_px4_track)
    prepare = subparsers.add_parser(
        "prepare-mission",
        help="run the real multi-call model and qualified-tool preparation workflow",
    )
    prepare.add_argument("--provider", required=True)
    prepare.add_argument("--critic-provider", required=True)
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--graph", type=Path, required=True)
    prepare.add_argument("--semantic", type=Path, required=True)
    prepare.add_argument("--vehicle-sdf", type=Path, required=True)
    prepare.add_argument("--vehicle-metadata", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--context-db", type=Path, default=Path("artifacts/state/conversations.sqlite3")
    )
    prepare.add_argument("--max-provider-attempts", type=int, default=3)
    prepare.add_argument("--max-intent-rounds", type=int, default=3)
    prepare.add_argument("--max-planning-rounds", type=int, default=5)
    prepare.add_argument("--model-timeout-seconds", type=float, default=180.0)
    prepare.add_argument("--waypoint-hold-seconds", type=float, default=0.4)
    prepare.set_defaults(handler=_prepare_mission)
    execute = subparsers.add_parser(
        "execute-prepared-mission",
        help="confirm a hash-bound package and run it in real PX4 SITL and Gazebo",
    )
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--execution-authority", type=Path, required=True)
    execute.add_argument("--confirm-contract-id", required=True)
    execute.add_argument(
        "--completion-provider",
        required=True,
    )
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--world-sdf", type=Path, required=True)
    execute.add_argument("--semantic", type=Path, required=True)
    execute.add_argument("--map-graph", type=Path)
    execute.add_argument("--vehicle-sdf", type=Path, required=True)
    execute.add_argument("--vehicle-metadata", type=Path, required=True)
    execute.add_argument("--controller-params", type=Path, required=True)
    execute.add_argument("--px4-root", type=Path, default=Path("/opt/PX4-Autopilot"))
    execute.add_argument("--executor", type=Path, required=True)
    execute.add_argument(
        "--ros-workspace",
        type=Path,
        required=True,
    )
    execute.add_argument(
        "--context-db", type=Path, default=Path("artifacts/state/conversations.sqlite3")
    )
    execute.add_argument("--model-timeout-seconds", type=float, default=180.0)
    execute.add_argument("--checkpoint-provider")
    execute.add_argument("--checkpoint-executor", type=Path)
    execute.add_argument("--checkpoint-timeout-seconds", type=float, default=180.0)
    execute.add_argument("--runtime-interrupt-provider")
    execute.add_argument("--runtime-hold-timeout-seconds", type=float, default=12.0)
    execute.add_argument("--runtime-decision-timeout-seconds", type=float, default=180.0)
    execute.add_argument("--runtime-replan-hold-seconds", type=float, default=60.0)
    execute.add_argument("--local-navigation-provider")
    execute.add_argument("--local-navigation-fallback-provider")
    execute.add_argument("--local-policy-package", type=Path, action="append", default=[])
    execute.add_argument("--local-policy-trial", type=Path)
    execute.add_argument("--simulation-map-fusion", action="store_true")
    execute.add_argument("--independent-route-control", action="store_true")
    execute.add_argument("--local-policy-qualification", type=Path, action="append", default=[])
    execute.add_argument(
        "--local-policy-simulation-admission",
        type=Path,
        action="append",
        default=[],
    )
    execute.add_argument("--local-navigation-model-timeout-seconds", type=float, default=10.0)
    execute.add_argument(
        "--local-navigation-fallback-model-timeout-seconds",
        type=float,
        default=10.0,
    )
    execute.add_argument("--local-navigation-period-seconds", type=float, default=3.0)
    execute.add_argument("--local-navigation-context-id")
    execute.add_argument("--local-navigation-visual-enabled", action="store_true")
    execute.add_argument(
        "--heading-policy",
        choices=("measured-hold", "route-tangent-relative"),
        default="measured-hold",
    )
    execute.add_argument("--maximum-yaw-rate-deg-s", type=float, default=20.0)
    execute.add_argument("--multimodal-dataset-root", type=Path)
    execute.add_argument("--multimodal-flight-id")
    execute.add_argument("--multimodal-dataset-maximum-mib", type=int, default=5_120)
    execute.add_argument("--multimodal-record-period-seconds", type=float, default=0.1)
    execute.add_argument("--require-local-navigation-control-authority", action="store_true")
    execute.add_argument("--bounded-hybrid-control", action="store_true")
    execute.add_argument("--development-payload-collection", action="store_true")
    execute.set_defaults(handler=_execute_prepared_mission)
    interrupt = subparsers.add_parser(
        "submit-runtime-message",
        help="atomically interrupt the exact active task execution before model reasoning",
    )
    interrupt.add_argument("--run-dir", type=Path, required=True)
    interrupt.add_argument("--text", required=True)
    interrupt.set_defaults(handler=_submit_runtime_message)
    reverify = subparsers.add_parser(
        "reverify-prepared-run",
        help="re-run completion review over an immutable finished PX4/Gazebo run",
    )
    reverify.add_argument("--prepared", type=Path, required=True)
    reverify.add_argument("--confirm-contract-id", required=True)
    reverify.add_argument(
        "--completion-provider",
        required=True,
    )
    reverify.add_argument("--run-dir", type=Path, required=True)
    reverify.add_argument("--semantic", type=Path, required=True)
    reverify.add_argument("--vehicle-sdf", type=Path, required=True)
    reverify.add_argument(
        "--context-db", type=Path, default=Path("artifacts/state/conversations.sqlite3")
    )
    reverify.add_argument("--model-timeout-seconds", type=float, default=180.0)
    reverify.set_defaults(handler=_reverify_prepared_run)
    return parser


# 功能：
#   解析命令并调用唯一处理器，不吞掉运行失败或把异常转换成成功状态。
# 输入：
#   无：参数由进程命令行提供。
# 输出：
#   exit_code：处理器返回的进程退出码。
def main() -> int:
    args = _parser().parse_args()
    exit_code = args.handler(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
