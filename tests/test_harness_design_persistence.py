"""设计器历史与磁盘边界；模型、插件与飞行器均不执行。"""

import json
from unittest.mock import Mock

import pytest

from dronedream_agent_app.harness_design_service import (
    HarnessDesignService,
    HarnessDesignServiceError,
)
from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.model_harness.design import HarnessEditOperation


# 功能：
#   创建真实临时存储和设计服务，插件动作由替身记录，退出时关闭数据库。
# 输入：
#   tmp_path：本用例独占的临时目录。
# 输出：
#   service：只在隔离目录中发布设计记录的服务实例。
@pytest.fixture
def service(tmp_path):
    store = AppStore(tmp_path)
    manager = Mock(spec=PluginManager)
    manager.list_plugins.return_value = []
    try:
        service = HarnessDesignService(store, manager)
        yield service
    finally:
        store.close()


# 功能：
#   构造只调整画布位置的编辑，以隔离持久化问题与执行语义变化。
# 输入：
#   revision：客户端读取到的基准编号。
#   x：目标画布横坐标。
# 输出：
#   operation：可提交给设计服务的移动操作。
def _move(revision, x=20):
    operation = HarnessEditOperation(
        client_operation_id=f"move-operation-{revision}-{x}",
        base_revision=revision,
        operation="move_node",
        payload={"node_id": "mission.intent-parse", "x": x, "y": 30},
    )
    return operation


# 功能：
#   验证撤销后创建分支保留原历史文件，新编号始终高于所有已经保存的编号。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_edit_after_undo_does_not_overwrite_history(service):
    service.apply_operation(_move(1))
    original = service._revision_path(2).read_bytes()
    service.undo(2)
    result = service.apply_operation(_move(1, 40))
    assert result["revision"]["revision"] == 3
    assert result["revision"]["parent_revision"] == 1
    assert service._revision_path(2).read_bytes() == original
    assert service.current()["can_redo"] is False


# 功能：
#   检查文件名称与内部编号绑定，阻止其他合法历史记录冒充当前记录。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_revision_filename_binds_identity(service):
    service.apply_operation(_move(1))
    service._revision_path(1).write_bytes(service._revision_path(2).read_bytes())
    with pytest.raises(HarnessDesignServiceError, match="HARNESS_REVISION_INVALID"):
        service.get_revision(1)


# 功能：
#   确认冻结前重新核对候选配置、摘要和编译产物，不能信任被改写的缓存结论。
# 输入：
#   service：隔离的设计服务。
#   field：待改写的已持久化字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["candidate", "hash", "compiled"])
def test_freeze_rejects_stale_compilation(service, field):
    path = service._revision_path(1)
    value = json.loads(path.read_text(encoding="utf-8"))
    if field == "candidate":
        value["candidate"]["name"] = "changed candidate"
    elif field == "hash":
        value["validation"]["semantic_sha256"] = "0" * 64
    else:
        value["validation"]["compiled_topology"]["name"] = "changed compiled graph"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(HarnessDesignServiceError, match="HARNESS_REVISION_INVALID"):
        service.freeze_active_for_task()


# 功能：
#   拒绝布尔或字符串编号和歧义索引字段，避免宽松转换读取另一条配置。
# 输入：
#   service：隔离的设计服务。
#   value：非法活动编号。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "1", 0, -1, 1.5])
def test_index_revision_is_strict(service, value):
    index = json.loads(service.index_path.read_text(encoding="utf-8"))
    index["active_revision"] = value
    service.index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(HarnessDesignServiceError, match="HARNESS_INDEX_INVALID"):
        service.current()


# 功能：
#   拒绝重复 JSON 键，即使解析器默认使用的末值看起来合法。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_index_duplicate_keys_rejected(service):
    original = service.index_path.read_text(encoding="utf-8")
    service.index_path.write_text('{"active_revision": 999,' + original[1:], encoding="utf-8")
    with pytest.raises(HarnessDesignServiceError, match="HARNESS_INDEX_INVALID"):
        service.current()


# 功能：
#   验证布局操作不把文本 false 当成启用，也不把布尔坐标当成数字坐标。
# 输入：
#   service：隔离的设计服务。
#   field：被破坏的布局字段。
#   value：错误类型的字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("field", "value"), [("pinned", "false"), ("x", True), ("x", "20")])
def test_layout_values_are_not_coerced(service, field, value):
    operation = _move(1)
    operation.payload[field] = value
    with pytest.raises((ValueError, HarnessDesignServiceError)):
        service.apply_operation(operation)
    assert service.current()["active"]["revision"] == 1


# 功能：
#   确认自身或向前的父编号在读取时被拒绝，撤销不能陷入损坏的父链。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_parent_revision_must_precede_child(service):
    path = service._revision_path(1)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["parent_revision"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(HarnessDesignServiceError, match="HARNESS_REVISION_INVALID"):
        service.get_revision(1)


# 功能：
#   模拟插件切换或索引发布中断，确认新任务被阻止；重开服务后显式激活可恢复。
# 输入：
#   service：隔离的设计服务。
#   monkeypatch：局部故障注入夹具。
#   fault：发生中断的发布环节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["plugin", "index"])
def test_partial_profile_transition_blocks_freeze_and_can_recover(service, monkeypatch, fault):
    operation = HarnessEditOperation(
        client_operation_id="profile-failure-operation",
        base_revision=1,
        operation="apply_profile",
        payload={"profile_id": "harness.profile-evaluation-lab"},
    )
    with monkeypatch.context() as patch:
        if fault == "plugin":
            patch.setattr(
                service.plugin_manager,
                "apply_profile",
                Mock(side_effect=RuntimeError("plugin failure")),
            )
        else:
            patch.setattr(service, "_save_index", Mock(side_effect=OSError("index failure")))
        with pytest.raises((RuntimeError, OSError)):
            service.apply_operation(operation)
    reopened = HarnessDesignService(service.store, service.plugin_manager)
    with pytest.raises(HarnessDesignServiceError, match="HARNESS_TRANSITION_INCOMPLETE"):
        reopened.freeze_active_for_task()
    reopened.activate(1)
    assert reopened.freeze_active_for_task()["revision"] == 1
    assert service.plugin_manager.apply_profile.call_args.args[0] == "harness.profile-balanced"


# 功能：
#   检查回执按记录时间而不是随机文件名排序，超出条数时保留最新记录。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_receipts_are_ordered_by_timestamp(service):
    for receipt_id, created_at in [
        ("harness-receipt-" + "f" * 24, "2025-01-01T00:00:00+00:00"),
        ("harness-receipt-" + "0" * 24, "2099-01-01T00:00:00+00:00"),
    ]:
        service._write_json(
            service.receipts_root / f"{receipt_id}.json",
            {"receipt_id": receipt_id, "created_at": created_at},
        )
    assert service.receipts(limit=1)[0]["receipt_id"] == "harness-receipt-" + "0" * 24


# 功能：
#   检查历史独占发布不能覆盖既有配置，且失败不会残留本次暂存。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_revision_publication_is_create_only(service):
    original = service._revision_path(1).read_bytes()
    with pytest.raises(FileExistsError):
        service._save_revision(service.get_revision(1))
    assert service._revision_path(1).read_bytes() == original
    assert not list(service.revisions_root.glob("*.tmp"))


# 功能：
#   验证激活旧历史后编辑基准也随之恢复，新编辑仍分配未用过的较大编号。
# 输入：
#   service：隔离的设计服务。
# 输出：
#   None：不返回业务数据。
def test_activate_restores_edit_base_without_reusing_numbers(service):
    service.apply_operation(_move(1))
    service.activate(1)
    assert service.current()["current"]["revision"] == 1
    assert service.apply_operation(_move(1, 50))["revision"]["revision"] == 3


# 功能：
#   模拟索引替换失败，验证旧文件不变、独占暂存被回收、原始错误保留。
# 输入：
#   service：隔离的设计服务。
#   monkeypatch：替换系统发布调用的夹具。
# 输出：
#   None：不返回业务数据。
def test_index_replace_failure_preserves_previous_file(service, monkeypatch):
    original = service.index_path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("dronedream_agent_app.harness_design_service.os.replace",
                      Mock(side_effect=OSError("replace failure")))
        with pytest.raises(OSError, match="replace failure"):
            service._write_json(service.index_path, {"example": True})
    assert service.index_path.read_bytes() == original
    assert not list(service.root.glob("*.tmp"))
