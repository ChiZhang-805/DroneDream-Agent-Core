"""Synthetic orchestration failures; no live simulator or cloud account is used."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_asset_pair_qualification import (
    _fixtures,
    _install_fixture_version,
    _runtime_evidence,
    _wait_for_job,
)

import dronedream_agent_app.asset_qualification_service as services
from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.asset_pair_qualification import AssetPairQualificationPlan
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_contracts import PluginSnapshot


# 功能：
#   建立包含两个未验收资产的隔离服务，运行器只生成单元测试证据。
# 输入：
#   tmp_path：本次测试的独立目录。
# 输出：
#   fixture：服务、任务标识和可供故障注入的环境字典。
def _service_fixture(tmp_path):
    map_archive, vehicle_archive = _fixtures(tmp_path)
    store = AppStore(tmp_path / "store")
    map_id, map_hash = _install_fixture_version(store, map_archive)
    vehicle_id, vehicle_hash = _install_fixture_version(store, vehicle_archive)
    environment = {"gazebo": "test-runtime"}
    service = services.AssetQualificationService(
        store,
        AssetImportService(store),
        runtime_runner=_fixture_runner,
        runtime_canceller=lambda _job_id: False,
        environment_versions_provider=lambda: environment,
    )
    created = service.create(
        map_asset_id=map_id,
        map_content_sha256=map_hash,
        vehicle_asset_id=vehicle_id,
        vehicle_content_sha256=vehicle_hash,
    )
    fixture = service, str(created["job_id"]), environment
    return fixture


# 功能：
#   从当前准备材料生成模拟通过结果，不执行飞行或授予真实产品验收资格。
# 输入：
#   kwargs：服务传递的当前尝试目录等参数。
# 输出：
#   evidence：与测试计划绑定的模拟证据。
def _fixture_runner(**kwargs):
    root = Path(kwargs["work_root"])
    plan = AssetPairQualificationPlan.model_validate_json(
        (root / "qualification-plan.json").read_bytes()
    )
    evidence = _runtime_evidence(plan, root)
    return evidence


# 功能：
#   验证线程无法启动时，任务恢复到可重试状态，而不是永久显示正在准备。
# 输入：
#   tmp_path：隔离数据库和资产目录。
#   monkeypatch：用于注入线程启动异常的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_failed_worker_start_leaves_resumable_job(tmp_path, monkeypatch):
    service, job_id, _ = _service_fixture(tmp_path)

    # 功能：
    #   模拟操作系统拒绝创建线程。
    # 输入：
    #   self：尚未启动的测试线程。
    # 输出：
    #   None：不返回业务数据。
    def fail_start(self):
        raise RuntimeError("thread resources exhausted")

    monkeypatch.setattr(services.threading.Thread, "start", fail_start)
    with pytest.raises(services.AssetQualificationServiceError, match="WORKER_START_FAILED"):
        service.start(job_id)
    assert service.store.load_asset_pair_qualification_job(job_id).state == "paused"
    assert job_id not in service._workers


# 功能：
#   验证运行中修改环境标识不能把旧环境的结果归到新环境下。
# 输入：
#   tmp_path：隔离数据库和资产目录。
# 输出：
#   None：不返回业务数据。
def test_environment_change_during_run_cannot_qualify(tmp_path):
    service, job_id, environment = _service_fixture(tmp_path)

    # 功能：
    #   在模拟运行结果返回前切换共享环境字典，复现原地修改问题。
    # 输入：
    #   kwargs：当前运行参数。
    # 输出：
    #   evidence：切换前材料生成的测试证据。
    def changing_runner(**kwargs):
        evidence = _fixture_runner(**kwargs)
        environment["gazebo"] = "replacement-runtime"
        return evidence

    service.runtime_runner = changing_runner
    service.start(job_id)
    result = _wait_for_job(service.store, job_id, "qualified", "failed")
    assert result["state"] == "failed"
    assert result["issue_codes"] == ["ASSET_QUALIFICATION_ENVIRONMENT_CHANGED"]
    assert len(service.store.list_asset_versions()) == 2


# 功能：
#   验证附加检查只能提供自己的结果，不能重写原始飞行证据或共享环境。
# 输入：
#   tmp_path：隔离数据库和资产目录。
# 输出：
#   None：不返回业务数据。
def test_plugin_receives_detached_evidence_and_environment(tmp_path):
    service, job_id, environment = _service_fixture(tmp_path)

    # 功能：
    #   修改传入对象并返回空插件快照，用于检测宿主是否错误共享可变容器。
    # 输入：
    #   kwargs：宿主交给插件的检查材料。
    # 输出：
    #   checks：没有附加检查项的有效测试快照。
    def mutating_checker(**kwargs):
        kwargs["runtime_evidence"]["plugin_rewrote_source"] = True
        kwargs["environment_versions"]["gazebo"] = "plugin-invented-runtime"
        snapshot = PluginSnapshot(
            snapshot_id="plugin-snapshot-" + "a" * 24,
            catalog_sha256="b" * 64,
            plugins=[],
            created_at=datetime.now(UTC),
        )
        checks = {
            "plugin_snapshot": snapshot.model_dump(mode="json"),
            "plugin_snapshot_sha256": sha256_json(snapshot),
            "plugin_checks": [],
            "plugin_hook_receipts": [],
        }
        return checks

    service.qualification_checker = mutating_checker
    service.start(job_id)
    result = _wait_for_job(service.store, job_id, "qualified", "failed")
    assert result["state"] == "qualified", result
    receipt, _ = service.store.verified_asset_pair_receipt(
        service.store.load_asset_pair_qualification_job(job_id)
    )
    assert environment == {"gazebo": "test-runtime"}
    assert receipt.environment_versions == environment
    assert "plugin_rewrote_source" not in receipt.runtime_evidence


# 功能：
#   验证重打包失败不留下貌似可用的半成品，也不覆盖已存在的用户目标。
# 输入：
#   tmp_path：隔离数据库和资产目录。
# 输出：
#   None：不返回业务数据。
def test_failed_archive_does_not_publish_partial_output(tmp_path):
    service, job_id, _ = _service_fixture(tmp_path)
    job = service.store.load_asset_pair_qualification_job(job_id)
    _, map_root, _ = service.store.asset_pair_qualification_paths(job_id)
    (map_root / "unindexed.txt").write_text("must not be silently ignored")
    destination = tmp_path / "export.ddpkg"
    with pytest.raises(ValueError):
        service._archive_version(
            map_root,
            destination,
            expected_asset_id=job.map_asset_id,
            expected_content_sha256=job.map_content_sha256,
        )
    assert not destination.exists()
