from __future__ import annotations

from copy import deepcopy

from dronedream_agent_app.harness_design_service import HarnessDesignService
from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.model_harness.design import HarnessEditOperation


# 功能：
#   建立真实存储、插件管理器及设计服务，构建中途失败时回收已经创建的资源。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   services：由调用测试负责最终关闭的存储、管理器和设计服务。
def _service(tmp_path):
    store = AppStore(tmp_path)
    manager = None
    try:
        manager = PluginManager(store)
        services = store, manager, HarnessDesignService(store, manager)
        return services
    except BaseException:
        try:
            if manager is not None:
                manager.close()
        finally:
            store.close()
        raise


# 功能：
#   核对默认配置的语义与布局摘要，确认结构预演不实际执行外部调用。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_visual_harness_initializes_with_separate_semantic_and_layout_hashes(tmp_path) -> None:
    store, manager, service = _service(tmp_path)
    try:
        current = service.current()["active"]
        assert current["revision"] == 1
        assert current["validation"]["valid"] is True
        assert current["validation"]["semantic_sha256"]
        assert current["validation"]["layout_sha256"]
        assert service.dry_run()["external_calls_executed"] == 0
        assert service.freeze_active_for_task()["revision"] == 1
    finally:
        manager.close()
        store._connection.close()


# 功能：
#   检查展示目录的父子关系、类别和外观均绑定真实节点及插件槽位。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_catalog_exposes_three_level_composition_bound_to_real_nodes_and_slots(tmp_path) -> None:
    store, manager, service = _service(tmp_path)
    try:
        catalog = service.catalog()
        items = {value["item_id"]: value for value in catalog["composition_items"]}
        levels = {value["level"] for value in items.values()}
        assert levels == {1, 2, 3}

        phase = items["composition.phase.intent"]
        assert phase["granularity"] == "large"
        assert phase["category_id"] == "reasoning"
        assert phase["child_item_ids"]
        stage = items[phase["child_item_ids"][0]]
        assert stage["parent_item_id"] == phase["item_id"]
        assert stage["member_node_ids"][0].startswith("mission.")
        assert stage["child_item_ids"]
        assert all(items[item_id]["level"] == 3 for item_id in stage["child_item_ids"])

        intent_parse = items["composition.stage.mission.intent-parse"]
        assert intent_parse["category_id"] == "model"
        assert intent_parse["color_token"] == "magenta"
        assert intent_parse["visual_kind"] == "model"
        assert intent_parse["aspect_ratio"] == "model-bar"
        timeout = items["composition.policy.mission.intent-parse.timeout"]
        cache = items["composition.policy.mission.intent-parse.cache"]
        assert timeout["category_id"] == "orchestration"
        assert cache["category_id"] == "memory"
        assert timeout["color_token"] != cache["color_token"]
        assert {timeout["aspect_ratio"], cache["aspect_ratio"]} <= {"1:1", "1.5:1"}

        plugin_slots = {value["slot_id"]: value for value in catalog["plugins"]}
        assert plugin_slots["harness.workflow-topology"]["granularity"] == "large"
        assert plugin_slots["harness.retry-policy"]["granularity"] == "small"
        assert plugin_slots["harness.retry-policy"]["owner_item_ids"]
    finally:
        manager.close()
        store._connection.close()


# 功能：
#   验证纯布局变化只改变布局摘要，撤销与重做正确恢复活动历史。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_layout_edit_activates_without_changing_semantics_and_supports_undo_redo(tmp_path) -> None:
    store, manager, service = _service(tmp_path)
    try:
        before = service.current()["active"]
        result = service.apply_operation(
            HarnessEditOperation(
                client_operation_id="move-operation-0001",
                base_revision=1,
                operation="move_node",
                payload={
                    "node_id": "mission.intent-parse",
                    "x": 812.5,
                    "y": 240.0,
                },
            )
        )
        after = result["revision"]
        assert after["revision"] == 2
        assert after["state"] == "active"
        assert after["validation"]["semantic_sha256"] == before["validation"]["semantic_sha256"]
        assert after["validation"]["layout_sha256"] != before["validation"]["layout_sha256"]

        undone = service.undo(2)["revision"]
        assert undone["revision"] == 1
        redone = service.redo(1)["revision"]
        assert redone["revision"] == 2
    finally:
        manager.close()
        store._connection.close()


# 功能：
#   断开必需输入后保存拒绝历史，同时保持原活动配置供新任务冻结。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_visual_edit_is_a_rejected_revision_and_keeps_active_runtime(tmp_path) -> None:
    store, manager, service = _service(tmp_path)
    try:
        current = service.current()["active"]
        edge = next(
            value
            for value in current["candidate"]["edges"]
            if value["target"]["node_id"] == "mission.contract-freeze"
        )
        result = service.apply_operation(
            HarnessEditOperation(
                client_operation_id="disconnect-operation-0001",
                base_revision=1,
                operation="disconnect",
                payload={"edge": edge},
            )
        )
        assert result["revision"]["state"] == "rejected"
        assert result["revision"]["validation"]["valid"] is False
        assert service.current()["active"]["revision"] == 1
        assert service.freeze_active_for_task()["revision"] == 1
    finally:
        manager.close()
        store._connection.close()


# 功能：
#   以测试目录中的兼容描述替换可选节点，验证身份和既有连线保持不变。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_replace_node_is_atomic_and_preserves_node_identity_and_connections(tmp_path) -> None:
    store, manager, service = _service(tmp_path)
    try:
        catalog = service.catalog()
        source = next(
            value
            for value in catalog["node_descriptors"]
            if value["descriptor_id"] == "mission.tool-advice"
        )
        alternative = deepcopy(source)
        alternative["descriptor_id"] = "mission.tool-advice-alternative"
        catalog["node_descriptors"].append(alternative)
        service.catalog = lambda: catalog  # type: ignore[method-assign]

        before = service.current()["active"]["candidate"]
        incident_edges = [
            value
            for value in before["edges"]
            if "mission.tool-advice" in {value["source"]["node_id"], value["target"]["node_id"]}
        ]
        result = service.apply_operation(
            HarnessEditOperation(
                client_operation_id="replace-operation-0001",
                base_revision=1,
                operation="replace_node",
                payload={
                    "node_id": "mission.tool-advice",
                    "descriptor_id": "mission.tool-advice-alternative",
                },
            )
        )

        revision = result["revision"]
        assert revision["state"] == "active"
        replacement = next(
            value
            for value in revision["candidate"]["nodes"]
            if value["node_id"] == "mission.tool-advice"
        )
        assert replacement["descriptor_id"] == "mission.tool-advice-alternative"
        assert [
            value
            for value in revision["candidate"]["edges"]
            if "mission.tool-advice" in {value["source"]["node_id"], value["target"]["node_id"]}
        ] == incident_edges
    finally:
        manager.close()
        store._connection.close()


# 功能：
#   检查配置档切换同时改变实际插件包选择和编译拓扑，而不只是界面标题。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   None：不返回业务数据。
def test_profile_operation_switches_plugin_bundle_and_runtime_topology(tmp_path) -> None:
    store, manager, service = _service(tmp_path)
    try:
        result = service.apply_operation(
            HarnessEditOperation(
                client_operation_id="profile-operation-0001",
                base_revision=1,
                operation="apply_profile",
                payload={"profile_id": "harness.profile-evaluation-lab"},
            )
        )
        revision = result["revision"]
        assert revision["validation"]["valid"] is True
        assert revision["candidate"]["profile_id"] == "harness.profile-evaluation-lab"
        assert revision["candidate"]["topology_id"] == "topology.committee-closed-loop"
        review_nodes = [
            node
            for node in revision["candidate"]["nodes"]
            if node["node_id"].startswith("mission.intent-review-")
        ]
        assert len(review_nodes) == 3
        enabled_profiles = [value for value in service.profiles() if value["enabled"]]
        assert [value["profile_id"] for value in enabled_profiles] == [
            "harness.profile-evaluation-lab"
        ]
        service.undo(2)
        assert [value["profile_id"] for value in service.profiles() if value["enabled"]] == [
            "harness.profile-balanced"
        ]
        service.redo(1)
        assert [value["profile_id"] for value in service.profiles() if value["enabled"]] == [
            "harness.profile-evaluation-lab"
        ]
    finally:
        manager.close()
        store._connection.close()
