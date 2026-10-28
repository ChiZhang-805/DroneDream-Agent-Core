"""Run production preparation with offline model fixtures; no simulated flight is claimed."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from test_map_semantic_boundaries import _graph, _semantic
from test_planning_energy_gates import _vehicle

from dronedream_agent_core.assets import load_map_catalog
from dronedream_agent_core.cli import _bind_development_harness_request
from dronedream_agent_core.context import ContextStore
from dronedream_agent_core.contracts import MissionRequest, ModelCallRecord, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.model_port import StructuredCallResult
from dronedream_agent_core.orchestrator import (
    MissionClarificationRequired,
    MissionOrchestrator,
    MissionPreparationBlocked,
    PreparationConfig,
)
from dronedream_agent_core.tools import ToolPlugin


class _PlanningPort:
    # 功能：
    #   创建明确标记为离线夹具的角色端口，保留真实编排器、插件、几何与证据链。
    # 输入：
    #   alter_asset：可选的测试注入，在模型响应期间替换资产。
    # 输出：
    #   None。
    def __init__(self, alter_asset=None, clarification=False):
        self.settings = SimpleNamespace(name="offline-fixture", model="contract-test-only")
        self.supports_provider_context = False
        self.supports_image_input = False
        self.calls = []
        self.alter_asset = alter_asset
        self.clarification = clarification
        self.intent_inputs = []
        self.router_inputs = []
        self.planning_inputs = []
        self.tool_responder = None

    # 功能：
    #   为每个角色提供结构化测试答案，不替换生产路径求解器、门控或任何飞行执行器。
    # 输入：
    #   kwargs：真实编排传入的角色、结构合同、当前材料和预算。
    # 输出：
    #   result：明确标注供应商为 offline-fixture 的响应及记录。
    def call(self, **kwargs):
        role = kwargs["role"]
        self.calls.append(role)
        if role == "intent_parser":
            self.intent_inputs.append(kwargs["input_artifact"])
            value = dict(
                goal="Visit target and return without a payload",
                start_entity="start",
                target_entity="target",
                return_entity="start",
                payload_action="none",
            )
            if self.clarification:
                value["missing_critical_fields"] = ["要送回办公室，还是送到其他位置？"]
        elif role in {"intent_critic", "plan_critic"}:
            value = {"accepted": True}
        elif role == "plugin_router":
            self.router_inputs.append(kwargs["input_artifact"])
            value = (
                self.tool_responder(kwargs["input_artifact"])
                if self.tool_responder else {"calls": []}
            )
        elif role == "task_decomposer":
            self.planning_inputs.append(kwargs["input_artifact"])
            value = {
                "graph": {
                    "nodes": [
                        {
                            "task_id": "launch",
                            "action": "takeoff",
                            "target_node": "a",
                            "success_evidence": ["to be bound"],
                            "fallback": "hold",
                        },
                        {
                            "task_id": "outbound",
                            "action": "navigate",
                            "target_node": "b",
                            "depends_on": ["launch"],
                            "success_evidence": ["to be bound"],
                            "fallback": "hold",
                        },
                        {
                            "task_id": "inbound",
                            "action": "return",
                            "target_node": "a",
                            "depends_on": ["outbound"],
                            "success_evidence": ["to be bound"],
                            "fallback": "hold",
                        },
                        {
                            "task_id": "landing",
                            "action": "land",
                            "target_node": "a",
                            "depends_on": ["inbound"],
                            "success_evidence": ["to be bound"],
                            "fallback": "land",
                        },
                    ]
                }
            }
        elif role == "global_planner":
            self.planning_inputs.append(kwargs["input_artifact"])
            speed_limit = kwargs["input_artifact"]["vehicle_speed_envelope"]["cruise_max_mps"]
            value = {
                "ordered_targets": ["b", "a"],
                "route_policy": "balanced",
                "speed_preferences": [
                    {
                        "target_node": "b",
                        "cruise_speed_mps": min(3.6, speed_limit),
                        "approach_speed_mps": min(0.9, speed_limit),
                        "rationale": "Controlled destination approach.",
                    },
                    {
                        "target_node": "a",
                        "cruise_speed_mps": min(4.5, speed_limit),
                        "approach_speed_mps": min(0.7, speed_limit),
                        "rationale": "Open return then landing.",
                    },
                ],
                "rationale_summary": "Offline test intent; route geometry remains tool-owned.",
            }
        else:
            raise AssertionError("Unexpected planning role: " + role)
        artifact = kwargs["output_type"].model_validate(value)
        record = ModelCallRecord(
            call_id="model-" + uuid4().hex[:24],
            role=role,
            attempt=1,
            input_sha256=sha256_json(kwargs["input_artifact"]),
            output_sha256=sha256_json(artifact),
            output_schema=type(artifact).__name__,
            provider=self.settings.name,
            model=self.settings.model,
            input_tokens=1,
            output_tokens=1,
            latency_ms=0,
            created_at=datetime.now(UTC),
        )
        if role == "plan_critic" and self.alter_asset is not None:
            self.alter_asset()
        result = StructuredCallResult(artifact=artifact, record=record)
        return result


# 功能：
#   用测试所有的地图、机型和模型端口运行完整生产准备链，检查成功或资产中途变化的拒绝路径。
# 输入：
#   tmp_path：隔离测试目录。
#   change_asset：是否在计划评审响应时改写机型文件。
# 输出：
#   None。
@pytest.mark.parametrize(
    "change_asset", [False, True, "clarification", "tool_chain", "tool_repair", "tool_budget"]
)
def test_full_preparation_uses_real_plugins_and_geometry(tmp_path, change_asset, monkeypatch):
    graph = _graph()
    vehicle = _vehicle()
    vehicle.collision_center_offset_model_m = Vector3(x=0, y=0, z=0.1)
    semantic = _semantic()
    semantic["runtime_bindings"] = {
        "schema_version": "dronedream.map-runtime-bindings.v1",
        "simulator": "gazebo-harmonic",
        "coordinate_frame": "ENU",
        "vehicle_spawn": {"x": 0, "y": 0, "z": 0},
        "mission_launch_waypoint": {"x": 0, "y": 0, "z": 2},
    }
    semantic["collision_primitives"] = [
        {
            "name": "fixture-floor",
            "semantic": "floor",
            "center_x": 1.5,
            "center_y": 0,
            "center_z": -1,
            "size_x": 20,
            "size_y": 20,
            "size_z": 1,
        }
    ]
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    sdf = tmp_path / "vehicle.sdf"
    sdf.write_text('<sdf version="1.9"><model name="fixture"/></sdf>', encoding="utf-8")
    port = _PlanningPort(
        alter_asset=(lambda: sdf.write_text("changed test asset", encoding="utf-8"))
        if change_asset is True
        else None,
        clarification=change_asset == "clarification",
    )
    context = ContextStore(tmp_path / "context.sqlite")
    instance = None
    output_dir = tmp_path / "prepared"
    try:
        config = PreparationConfig(
            provider="kimi", critic_provider="kimi", persisted_task_context=False,
            maximum_plugin_calls=1 if change_asset == "tool_budget" else 8,
        )
        instance = MissionOrchestrator(
            config=config,
            map_catalog=load_map_catalog(semantic_path, qualified_graph=graph),
            map_graph=graph,
            semantic_path=semantic_path,
            vehicle_sdf=sdf,
            vehicle_asset_id=vehicle.asset_id,
            vehicle=vehicle,
            context_store=context,
            primary_port=port,
            critic_port=port,
        )
        executed_tools = []
        if change_asset in {"tool_chain", "tool_repair", "tool_budget"}:
            # More than 32 tools ensures the model also sees the last catalog entries.
            for index in range(35):
                instance.tool_registry.register(ToolPlugin(
                    tool_id=f"zz-fixture.tool-{index:02d}",
                    version="1.0.0",
                    authority="read",
                    input_type=None,
                    output_type=None,
                    input_schema={
                        "type": "object", "properties": {"node": {"type": "string"}},
                        "required": ["node"], "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object", "properties": {"node": {"type": "string"}},
                        "required": ["node"], "additionalProperties": False,
                    },
                    handler=lambda value, tool=index: (
                        executed_tools.append(tool) or {"node": "b"}
                    ),
                ))
            def respond_to_tools(payload):
                assert "zz-fixture.tool-34" in {
                    item["tool_id"] for item in payload["optional_tool_catalog"]
                }
                if payload["round"] == 1:
                    return {"calls": [{
                        "tool_id": "zz-fixture.tool-34",
                        "arguments_json": '{}' if change_asset == "tool_repair" else '{"node":"a"}',
                        "purpose": "Resolve a destination from tool evidence",
                    }, *([{
                        "tool_id": "zz-fixture.tool-33", "arguments_json": '{"node":"a"}',
                        "purpose": "Excess call must not execute",
                    }] if change_asset == "tool_budget" else [])]}
                previous = payload["tool_results"][-1]
                if change_asset == "tool_repair":
                    assert previous["accepted"] is False
                    assert previous["issue_codes"][0].startswith("TOOL_INPUT_INVALID")
                    tool_id = "zz-fixture.tool-34"
                    node = "a"
                else:
                    assert previous["accepted"] is True
                    tool_id = "zz-fixture.tool-33"
                    node = previous["output"]["node"]
                return {"calls": [{
                    "tool_id": tool_id, "arguments_json": json.dumps({"node": node}),
                    "purpose": "Use the previous tool result or corrected schema",
                }]}

            port.tool_responder = respond_to_tools
        request = _bind_development_harness_request(
            MissionRequest(
                conversation_id="offline-pipeline",
                message="Visit target and return without a payload",
                start_entity="start",
            ),
            config=config,
            plugin_snapshot=instance.plugin_snapshot,
        )
        # 与桌面准备使用相同的 canonical session context，验证解析不会只停留在缓存。
        envelope = request.input_metadata["model_harness_input"]
        envelope["session_context"]["asset_understanding"] = {
            "map": {
                "asset_id": graph.asset_id,
                "authority": "advisory-only",
                "understanding": {"summary": "离线测试的地图解析上下文"},
            },
        }
        if change_asset == "clarification":
            with pytest.raises(MissionClarificationRequired) as pending:
                instance.prepare(request, output_dir)
            assert pending.value.fields == ["要送回办公室，还是送到其他位置？"]
            assert "global_planner" not in port.calls
            assert not (output_dir / "prepared-mission.json").exists()
        elif change_asset is True:
            with pytest.raises(MissionPreparationBlocked, match="PREPARATION_ASSET_CHANGED"):
                instance.prepare(request, output_dir)
            assert not (output_dir / "prepared-mission.json").exists()
        else:
            prepared = instance.prepare(request, output_dir)
            if change_asset in {"tool_chain", "tool_repair", "tool_budget"}:
                assert executed_tools == ([34, 33] if change_asset == "tool_chain" else [34])
                assert len(port.router_inputs) == (1 if change_asset == "tool_budget" else 2)
                for planning_input in port.planning_inputs:
                    advice = planning_input["tool_advice"]
                    assert any(item["accepted"] and item["output"] == {"node": "b"}
                               for item in advice)
            assert prepared.execution_route.node_ids == ["a", "b", "a"]
            assert prepared.route_clearance.accepted
            assert len(prepared.plan.segments) == 2
            assert prepared.contract.map_semantic_sha256 == prepared.route_clearance.semantic_sha256
            assert all(record.provider == "offline-fixture" for record in prepared.model_calls)
            assert {
                "intent_parser",
                "intent_critic",
                "task_decomposer",
                "global_planner",
                "plan_critic",
            } <= set(port.calls)
            assert (output_dir / "prepared-mission.json").is_file()
            assert (output_dir / "mission-lifecycle.json").is_file()
            from test_execution_integrity import _exercise_failed_execution_and_reverification

            _exercise_failed_execution_and_reverification(tmp_path, prepared, context, monkeypatch)
        assert (
            port.intent_inputs[0]["mission_request"]["session_context"]["asset_understanding"][
                "map"
            ]["understanding"]["summary"]
            == "离线测试的地图解析上下文"
        )
    finally:
        try:
            if instance is not None:
                instance.close()
        finally:
            context.close()
