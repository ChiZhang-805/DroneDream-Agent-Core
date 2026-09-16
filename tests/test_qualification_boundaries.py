"""Isolated qualification boundary fixtures; no simulator, account or model API is used."""

import json
import os
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from io import BytesIO
from uuid import UUID

import pytest
from test_asset_pair_qualification import _fixtures, _runtime_evidence

import dronedream_agent_core.asset_pair_qualification as qualification


# 功能：
#   构造两个最小资产及模拟回执，供准入边界测试使用，不生成真实飞行证据。
# 输入：
#   tmp_path：pytest 为本次测试分配的独立目录。
# 输出：
#   fixture：地图包、计划目录、计划与模拟回执组成的元组。
def _prepared_pair(tmp_path):
    map_archive, vehicle_archive = _fixtures(tmp_path)
    work_root = tmp_path / "run"
    plan = qualification.prepare_asset_pair_qualification(
        map_archive=map_archive, vehicle_archive=vehicle_archive, work_root=work_root
    )
    receipt = qualification.build_pair_qualification_receipt(
        plan=plan,
        work_root=work_root,
        runtime_evidence=_runtime_evidence(plan, work_root),
        environment_versions={"gazebo": "fixture"},
    )
    fixture = map_archive, work_root, plan, receipt
    return fixture


# 功能：
#   验证删除计划中的必需门禁，不能让缺少落地证据的执行获得验收回执。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_mutated_plan_cannot_remove_required_runtime_gate(tmp_path):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    plan.required_runtime_gates.remove("landing_confirmed")
    evidence = _runtime_evidence(plan, work_root)
    with pytest.raises(ValueError):
        qualification.build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=evidence,
            environment_versions={"gazebo": "fixture"},
        )


# 功能：
#   验证回执生成后再篡改门禁并重算摘要，不能绕过发布边界的重新验证。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_mutated_receipt_cannot_be_bound_even_with_recomputed_hash(tmp_path):
    source, _, _, receipt = _prepared_pair(tmp_path)
    receipt.runtime_evidence["gates"].pop("landing_confirmed")
    # 故意使用不校验的复制入口，模拟嵌套容器被外部代码篡改后的对象。
    # 普通字段赋值已在更早的校验器中被拒绝，不能用它跳过发布边界测试。
    receipt = receipt.model_copy(
        update={
            "runtime_evidence_sha256": qualification._sha256_bytes(
                qualification._json_bytes(receipt.runtime_evidence)
            ),
        }
    )
    target = tmp_path / "result.ddpkg"
    with pytest.raises(ValueError):
        qualification.bind_qualification_receipt(
            source_archive=source, destination_archive=target, receipt=receipt
        )
    assert not target.exists()


# 功能：
#   验证发布失败不得覆盖已有目标，也不得删除其他流程创建的同名临时文件。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_publication_preserves_existing_files(tmp_path):
    source, _, _, receipt = _prepared_pair(tmp_path)
    target = tmp_path / "result.ddpkg"
    old_temporary = tmp_path / "result.ddpkg.tmp"
    target.write_bytes(b"existing result")
    old_temporary.write_bytes(b"another writer")
    with pytest.raises((ValueError, FileExistsError)):
        qualification.bind_qualification_receipt(
            source_archive=source, destination_archive=target, receipt=receipt
        )
    assert target.read_bytes() == b"existing result"
    assert old_temporary.read_bytes() == b"another writer"


# 功能：
#   验证控制参数文件拒绝重复键和非有限扩展值，不按解析器的覆盖顺序选配置。
# 输入：
#   tmp_path：隔离文件目录。
#   raw：刻意构造的歧义 JSON。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "raw",
    [
        '{"vel_limit":1,"vel_limit":2,"accel_limit":1}',
        '{"vel_limit":1,"accel_limit":1,"other":NaN}',
    ],
)
def test_controller_contract_rejects_ambiguous_json(tmp_path, raw):
    target = tmp_path / "controller.json"
    target.write_text(raw, encoding="utf-8")
    with pytest.raises(qualification.AssetPairQualificationError):
        qualification._controller_contract(target)


# 功能：
#   验证资格任务不接受布尔进度、字符串问题列表及倒退的更新时间。
# 输入：
#   override：需要单独检验的非法状态迁移参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "override",
    [
        {"progress_percent": True},
        {"issue_codes": "not-a-list"},
        {"now": datetime(2026, 9, 1, tzinfo=UTC) - timedelta(seconds=1)},
    ],
)
def test_job_transition_rejects_coercion_and_time_regression(override):
    now = datetime(2026, 9, 1, tzinfo=UTC)
    job = qualification.AssetPairQualificationJob(
        job_id="asset-qualification-job-" + "a" * 24,
        owner_id="fixture",
        map_asset_id="map",
        map_content_sha256="a" * 64,
        vehicle_asset_id="vehicle",
        vehicle_content_sha256="b" * 64,
        created_at=now,
        updated_at=now,
    )
    arguments = {"progress_percent": 5, "now": now, **override}
    with pytest.raises(ValueError):
        qualification.transition_pair_qualification_job(job, "preparing", **arguments)
    assert job.state == "created"


# 功能：
#   验证容器中合法的嵌套模型不会被误认为多个顶层待验收飞机。
# 输入：
#   tmp_path：隔离 XML 文件目录。
# 输出：
#   None：不返回业务数据。
def test_xml_identity_uses_top_level_entity(tmp_path):
    source = tmp_path / "model.sdf"
    source.write_text('<sdf><model name="vehicle"><model name="part"/></model></sdf>')
    assert qualification._xml_entity_name(source, "model") == "vehicle"


# 功能：
#   验证非有限数值不会被写入资格回执的 JSON 摘要材料。
# 输入：
#   无输入参数；使用一个含 NaN 的隔离字典。
# 输出：
#   None：不返回业务数据。
def test_qualification_json_does_not_emit_nonfinite_numbers():
    with pytest.raises(ValueError):
        qualification._json_bytes({"value": float("nan")})


# 功能：
#   验证函数对重复数据契约不会按文件顺序选第一个。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：不返回业务数据。
def test_contract_selection_requires_one_matching_file(tmp_path):
    source, work_root, _, _ = _prepared_pair(tmp_path)
    inspected = qualification.inspect_ddpkg(source)
    graph = next(entry for entry in inspected.asset_ir.files if "navigation-graph" in entry.path)
    duplicate = graph.model_copy(update={"path": "normalized/duplicate.json"})
    inspected.asset_ir.files.append(duplicate)
    (work_root / "map" / duplicate.path).write_text(
        json.dumps(
            {
                "schema_version": "dronedream.map-graph.v1",
            }
        )
    )
    with pytest.raises(qualification.AssetPairQualificationError, match="AMBIGUOUS"):
        qualification._json_contract_path(work_root / "map", inspected, "dronedream.map-graph.v1")


# 功能：
#   验证常规字段赋值会立即拒绝丢失落地门禁的回执。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_assignment_revalidates_required_gates(tmp_path):
    _, _, _, receipt = _prepared_pair(tmp_path)
    receipt.runtime_evidence["gates"].pop("landing_confirmed")
    digest = qualification._sha256_bytes(qualification._json_bytes(receipt.runtime_evidence))
    with pytest.raises(ValueError, match="gates"):
        receipt.runtime_evidence_sha256 = digest


# 功能：
#   验证空字典、布尔时间等错误输入不会因真假值判断被当成未提供参数。
# 输入：
#   tmp_path：隔离资产目录。
#   override：本次需要拒绝的错误可选参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "override",
    [
        {"plugin_checks": {}},
        {"plugin_hook_receipts": ""},
        {"qualified_at": False},
    ],
)
def test_receipt_rejects_falsey_nonoptional_values(tmp_path, override):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    with pytest.raises(ValueError):
        qualification.build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=_runtime_evidence(plan, work_root),
            environment_versions={"gazebo": "fixture"},
            **override,
        )


# 功能：
#   验证生成回执后修改调用方原始证据，不会连带改变已生成回执及其摘要。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_detaches_caller_owned_evidence(tmp_path):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    evidence = _runtime_evidence(plan, work_root)
    evidence["extra"] = {"samples": [1, 2]}
    environment = {"gazebo": "fixture"}
    receipt = qualification.build_pair_qualification_receipt(
        plan=plan,
        work_root=work_root,
        runtime_evidence=evidence,
        environment_versions=environment,
    )
    evidence["gates"].clear()
    evidence["extra"]["samples"].append(3)
    environment["gazebo"] = "different"
    assert receipt.runtime_evidence["gates"]["landing_confirmed"] is True
    assert receipt.runtime_evidence["extra"]["samples"] == [1, 2]
    assert receipt.environment_versions == {"gazebo": "fixture"}
    qualification.AssetPairQualificationReceipt.model_validate(receipt.model_dump(mode="python"))


# 功能：
#   验证准备文件被替换后，即使运行结果仍引用旧摘要，也不能生成新回执。
# 输入：
#   tmp_path：隔离资产目录。
#   name：被替换的计划材料文件名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("name", ["route.json", "track.json", "clearance.json"])
def test_receipt_rechecks_prepared_artifact_bytes(tmp_path, name):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    evidence = _runtime_evidence(plan, work_root)
    (work_root / name).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="PLAN_ARTIFACT_CHANGED"):
        qualification.build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=evidence,
            environment_versions={"gazebo": "fixture"},
        )


# 功能：
#   验证第二个无名称的顶层模型不能被忽略，从而伪装成唯一模型入口。
# 输入：
#   tmp_path：隔离 XML 目录。
# 输出：
#   None：不返回业务数据。
def test_xml_identity_rejects_unnamed_second_entity(tmp_path):
    source = tmp_path / "model.sdf"
    source.write_text('<sdf><model name="vehicle"/><model/></sdf>', encoding="utf-8")
    with pytest.raises(ValueError, match="NAME_AMBIGUOUS"):
        qualification._xml_entity_name(source, "model")


# 功能：
#   验证并发发布者先创建目标时，本次发布只清理自己创建的临时包。
# 输入：
#   tmp_path：隔离资产目录。
#   monkeypatch：pytest 提供的可恢复替换工具。
# 输出：
#   None：不返回业务数据。
def test_receipt_publication_does_not_replace_concurrent_winner(tmp_path, monkeypatch):
    source, _, _, receipt = _prepared_pair(tmp_path)
    target = tmp_path / "result.ddpkg"
    real_link = os.link

    # 功能：
    #   在原子发布的最后一步模拟另一位写入者已经创建目标。
    # 输入：
    #   temporary：本次准备发布的临时包。
    #   destination：待发布目标。
    # 输出：
    #   None：不返回业务数据。
    def publish_after_competitor(temporary, destination):
        destination.write_bytes(b"concurrent winner")
        real_link(temporary, destination)

    monkeypatch.setattr(qualification.os, "link", publish_after_competitor)
    with pytest.raises(FileExistsError):
        qualification.bind_qualification_receipt(
            source_archive=source,
            destination_archive=target,
            receipt=receipt,
        )
    assert target.read_bytes() == b"concurrent winner"
    assert list(tmp_path.glob(".result.ddpkg-*.tmp")) == []


# 功能：
#   验证即使随机临时名称发生碰撞，也不能删除属于其他流程的文件。
# 输入：
#   tmp_path：隔离资产目录。
#   monkeypatch：pytest 提供的可恢复替换工具。
# 输出：
#   None：不返回业务数据。
def test_receipt_publication_preserves_unowned_temporary(tmp_path, monkeypatch):
    source, _, _, receipt = _prepared_pair(tmp_path)
    target = tmp_path / "result.ddpkg"
    collision = UUID(int=1)
    temporary = tmp_path / f".result.ddpkg-{collision.hex}.tmp"
    temporary.write_bytes(b"owned by another writer")
    monkeypatch.setattr(qualification, "uuid4", lambda: collision)
    with pytest.raises(FileExistsError):
        qualification.bind_qualification_receipt(
            source_archive=source,
            destination_archive=target,
            receipt=receipt,
        )
    assert temporary.read_bytes() == b"owned by another writer"
    assert not target.exists()


# 功能：
#   验证复制入口拿到的源元数据与先前检查不一致时，不能发布验收包。
# 输入：
#   tmp_path：隔离资产目录。
#   monkeypatch：pytest 提供的可恢复替换工具。
# 输出：
#   None：不返回业务数据。
def test_receipt_publication_rejects_changed_source_snapshot(tmp_path, monkeypatch):
    source, _, _, receipt = _prepared_pair(tmp_path)
    target = tmp_path / "result.ddpkg"
    verified_open = qualification.open_verified_ddpkg

    # 功能：
    #   模拟复制前发现了不同的归档元数据，不修改实际源包。
    # 输入：
    #   archive：测试资产包路径。
    # 输出：
    #   changed_bundle：读取器与发生变化的测试元数据。
    @contextmanager
    def changed_snapshot(archive):
        with verified_open(archive) as (bundle, inspected):
            inspected.asset_ir.name = "changed source metadata"
            changed_bundle = bundle, inspected
            yield changed_bundle

    monkeypatch.setattr(qualification, "open_verified_ddpkg", changed_snapshot)
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        qualification.bind_qualification_receipt(
            source_archive=source,
            destination_archive=target,
            receipt=receipt,
        )
    assert not target.exists()
    assert list(tmp_path.glob(".result.ddpkg-*.tmp")) == []


# 功能：
#   验证前面的候选只有去路时，仍会继续寻找后面能够往返的候选。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_round_trip_skips_outbound_only_candidate(tmp_path):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    graph_data = qualification._load_json(work_root / plan.inputs.graph, "fixture")
    dead_end = {
        **graph_data["nodes"][1],
        "node_id": "dead-end",
        "label": "Dead end",
        "position_m": {"x": 0.0, "y": 3.0, "z": 2.0},
    }
    graph_data["nodes"].insert(1, dead_end)
    graph_data["edges"].append(
        {
            **graph_data["edges"][0],
            "edge_id": "outbound-only",
            "to_node": "dead-end",
            "distance_m": 3.0,
            "bidirectional": False,
        }
    )
    graph = qualification.MapAsset.model_validate(graph_data)
    vehicle = qualification._validated_contract(
        work_root / plan.inputs.vehicle_metadata,
        qualification.VehicleAsset,
        "fixture",
    )
    route, clearance = qualification._round_trip_route(
        graph,
        work_root / plan.inputs.semantic,
        vehicle,
        minimum_translation_m=1.0,
        maximum_translation_m=30.0,
    )
    assert route.node_ids == ["launch", "turn", "launch"]
    assert clearance.accepted


# 功能：
#   验证显式起飞语义不因另一个节点使用旧学校专用别名而被覆盖。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_launch_selection_does_not_prefer_school_specific_alias(tmp_path):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    graph_data = qualification._load_json(work_root / plan.inputs.graph, "fixture")
    graph_data["named_entities"]["office-launch-pad"] = "turn"
    graph = qualification.MapAsset.model_validate(graph_data)
    assert qualification._launch_node(graph) == "launch"


# 功能：
#   验证已校验归档在读取作用域中发生时间戳变化时，不会静默接受该读取。
# 输入：
#   tmp_path：隔离资产目录。
# 输出：
#   None：不返回业务数据。
def test_verified_archive_context_detects_change_before_return(tmp_path):
    source, _, _, _ = _prepared_pair(tmp_path)
    with (
        pytest.raises(ValueError, match="ARCHIVE_CHANGED"),
        qualification.open_verified_ddpkg(source) as (bundle, inspected),
    ):
        assert bundle.read("manifest.json")
        assert inspected.manifest.asset_kind == "map"
        previous = source.stat()
        os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))


# 功能：
#   验证成员实际内容与已验证索引不一致时，复制过程不能仅相信旧索引。
# 输入：
#   mismatch：本次模拟大小变更或同大小内容变更。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mismatch", ["size", "hash"])
def test_member_copy_rechecks_actual_bytes(mismatch):
    expected = b"abc"
    actual = b"abcd" if mismatch == "size" else b"xyz"
    entry = qualification.AssetFile(
        path="payload.bin",
        role="metadata",
        media_type="application/octet-stream",
        sha256=qualification._sha256_bytes(expected),
        size_bytes=len(expected),
    )
    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as output:
        output.writestr(entry.path, actual)
    archive_bytes.seek(0)
    with (
        zipfile.ZipFile(archive_bytes) as bundle,
        pytest.raises(ValueError, match="MEMBER_CHANGED"),
    ):
        qualification._copy_verified_member(bundle, entry, BytesIO())
