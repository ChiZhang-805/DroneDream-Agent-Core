"""Prepared-input boundary tests with synthetic evidence and no simulator process."""

import pytest
from test_asset_pair_qualification import _runtime_evidence
from test_qualification_boundaries import _prepared_pair

from dronedream_agent_core import asset_pair_qualification as qualification
from dronedream_agent_core import asset_qualification_cli as cli


# 功能：
#   调用真实入口校验，但以捕获器替换飞行适配器，避免启动任何仿真。
# 输入：
#   tmp_path：隔离测试目录。
#   work_root：已准备的资产目录。
#   monkeypatch：替换适配器的测试工具。
#   calls：记录进入执行阶段的参数。
# 输出：
#   evidence：模拟适配器的结果。
def _run_without_flight(tmp_path, work_root, monkeypatch, calls):
    # 功能：
    #   记录调用并返回模拟结果，只用于检查是否跨过执行边界。
    # 输入：
    #   kwargs：运行入口传来的参数。
    # 输出：
    #   evidence：最小模拟结果。
    def run_adapter(**kwargs):
        calls.append(kwargs)
        evidence = {"status": "verified"}
        return evidence

    monkeypatch.setattr(cli, "run_px4_gazebo_track", run_adapter)
    evidence = cli.run_prepared_asset_qualification(
        work_root=work_root,
        run_dir=tmp_path / "evidence",
        px4_root=tmp_path / "px4",
        executor=tmp_path / "executor.py",
        ros_workspace=tmp_path / "ros",
    )
    return evidence


# 功能：
#   验证准备后的任一关键资产变化，都不能仅凭路线摘要正确就进入执行。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换适配器的测试工具。
#   field：需要篡改的计划输入字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field",
    ["world_sdf", "semantic", "graph", "vehicle_sdf", "vehicle_metadata", "controller_params"],
)
def test_changed_prepared_asset_is_rejected_before_execution(tmp_path, monkeypatch, field):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    path = work_root / getattr(plan.inputs, field)
    path.write_bytes(path.read_bytes() + b"\n")
    calls = []
    with pytest.raises(cli.AssetQualificationCliError, match="PREPARED_ASSET"):
        _run_without_flight(tmp_path, work_root, monkeypatch, calls)
    assert calls == []


# 功能：
#   验证 JSON 重复键不能通过后一个合法值掩盖前一个非法值。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换适配器的测试工具。
# 输出：
#   None：不返回业务数据。
def test_duplicate_plan_key_is_rejected_before_execution(tmp_path, monkeypatch):
    _, work_root, _, _ = _prepared_pair(tmp_path)
    plan_path = work_root / "qualification-plan.json"
    original = plan_path.read_bytes()
    plan_path.write_bytes(b'{"qualification_id":"invalid",' + original[1:])
    calls = []
    with pytest.raises(cli.AssetQualificationCliError, match="PLAN_INVALID"):
        _run_without_flight(tmp_path, work_root, monkeypatch, calls)
    assert calls == []


# 功能：
#   验证仍落在目录内但含点段的路径不能绕过输入规范性检查。
# 输入：
#   tmp_path：隔离测试目录。
#   relative：含歧义的相对路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("relative", ["./track.json", "map/../track.json"])
def test_prepared_path_must_have_canonical_spelling(tmp_path, relative):
    _, work_root, _, _ = _prepared_pair(tmp_path)
    with pytest.raises(cli.AssetQualificationCliError, match="INPUT_PATH_INVALID"):
        cli._bound_path(work_root, relative)


# 功能：
#   验证运行证据即使与修改后的世界一致，也不能替原包内容签发资格回执。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_rejects_source_change_even_when_runtime_hash_agrees(tmp_path):
    _, work_root, plan, _ = _prepared_pair(tmp_path)
    world = work_root / plan.inputs.world_sdf
    world.write_bytes(world.read_bytes() + b"\n")
    runtime_evidence = _runtime_evidence(plan, work_root)
    with pytest.raises(qualification.AssetPairQualificationError, match="PREPARED_ASSET"):
        qualification.build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=runtime_evidence,
            environment_versions={"gazebo": "fixture"},
        )
