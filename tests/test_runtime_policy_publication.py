"""Runtime staging/rebinding file boundaries, using non-executed model fixtures."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from test_local_policy_runtime_staging import _admission, _package

from dronedream_agent_core.plugin_files import hash_plugin_file
from scripts import rebind_local_policy_package as rebinding
from scripts import stage_local_policy_runtime as staging


# 功能：
#   创建当前四轴合同的十角色占位包及合成准入回执，测试文件发布而非推理或飞行资格。
# 输入：
#   tmp_path：测试来源与输出目录。
#   monkeypatch：设置暂存命令行参数的工具。
# 输出：
#   case：源包、回执、输出和原始参数。
@pytest.fixture
def staging_case(tmp_path, monkeypatch):
    package = _package(tmp_path / "package", continuous_control=True)
    receipt = tmp_path / "admission.json"
    _admission(receipt, package, continuous_evidence=True, expert_evidence=True)
    output = tmp_path / "runtime-policy"
    argv = [
        "stage",
        "--package",
        str(package.root),
        "--simulation-admission",
        str(receipt),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    case = SimpleNamespace(package=package, receipt=receipt, output=output, argv=argv)
    return case


# 功能：
#   运行时暂存不能接受尾延迟超预算的回执，不能仅看准入布尔标记。
# 输入：
#   staging_case：具有 250 毫秒模型预算的完整包。
#   kind：单头延迟或端到端延迟超限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["navigation", "combined"])
def test_staging_checks_receipt_latency(staging_case, kind):
    case = staging_case
    payload = json.loads(case.receipt.read_bytes())
    payload[
        "p99_inference_latency_ms" if kind == "navigation" else "combined_p99_inference_latency_ms"
    ] = 251.0
    case.receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        staging.main()
    assert not case.output.exists()


# 功能：
#   暂存回执重复字段必须被拒绝，不能覆盖先前的拒绝准入标记。
# 输入：
#   staging_case：普通完整包和回执。
# 输出：
#   None：不返回业务数据。
def test_staging_rejects_duplicate_receipt_fields(staging_case):
    case = staging_case
    case.receipt.write_bytes(b'{"admitted_to_simulation":false,' + case.receipt.read_bytes()[1:])
    with pytest.raises(ValueError):
        staging.main()


# 功能：
#   Runtime 输出不能写入原包内部，避免以后重新加载时混入暂存及目录自嵌套内容。
# 输入：
#   staging_case：源包及命令行上下文。
#   monkeypatch：将输出改到源包内部的工具。
# 输出：
#   None：不返回业务数据。
def test_staging_rejects_output_inside_source(staging_case, monkeypatch):
    case = staging_case
    argv = list(case.argv)
    argv[-1] = str(case.package.root / "nested-runtime")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        staging.main()


# 功能：
#   来源回执在模型复制期间变化时，暂存结果及目录摘要仍绑定校验过的原始字节。
# 输入：
#   staging_case：有效包与回执。
#   monkeypatch：在复制边界模拟外部回执变化的工具。
# 输出：
#   None：不返回业务数据。
def test_staging_preserves_validated_receipt_bytes(staging_case, monkeypatch):
    case = staging_case
    content = case.receipt.read_bytes()
    changed = False

    # 功能：
    #   首次复制模型后替换外部回执，模拟校验和最终发布之间的文件变化。
    # 输入：
    #   无：使用闭包内的回执和单次替换标志。
    # 输出：
    #   None：不返回业务数据。
    def change_once():
        nonlocal changed
        if not changed:
            case.receipt.write_bytes(b"changed externally")
            changed = True

    # 功能：
    #   在共享有界复制器完成模型复制后注入相同变化，验证修复后的发布路径。
    # 输入：
    #   source：实际模型源。
    #   limit：最大字节数。
    #   destination：可选独占复制目标。
    # 输出：
    #   digest：实际复制或散列内容摘要。
    def changing_hash(source, *, limit, destination=None):
        digest = hash_plugin_file(source, limit=limit, destination=destination)
        if destination is not None:
            change_once()
        return digest

    monkeypatch.setattr(staging, "hash_plugin_file", changing_hash)
    assert staging.main() == 0
    assert changed
    assert (case.output / "receipts/active.json").read_bytes() == content
    catalog = json.loads((case.output / "catalog.json").read_bytes())
    assert catalog["entries"][0]["receipt"]["sha256"] == hashlib.sha256(content).hexdigest()


# 功能：
#   使用仓库现有合格载具包和测试模型构造重绑定入口，不修改载具资产或安装软件。
# 输入：
#   staging_case：占位模型及原准入回执。
#   tmp_path：新模型包和重绑定回执目录。
#   monkeypatch：切换为重绑定参数的工具。
# 输出：
#   case：完整重绑定来源及输出上下文。
@pytest.fixture
def rebinding_case(staging_case, tmp_path, monkeypatch):
    from pathlib import Path

    source = staging_case
    vehicle = (
        Path(__file__).parents[1] / "app/desktop/src-tauri/resources/default-assets/my-drone.ddpkg"
    )
    output, receipt = tmp_path / "rebound", tmp_path / "rebinding.json"
    argv = [
        "rebind",
        "--source-package",
        str(source.package.root),
        "--source-simulation-admission",
        str(source.receipt),
        "--qualified-vehicle-package",
        str(vehicle),
        "--output-package",
        str(output),
        "--rebinding-receipt",
        str(receipt),
        "--package-id",
        "test.rebound",
        "--display-name",
        "Rebound test",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    case = SimpleNamespace(source=source, argv=argv, output=output, receipt=receipt)
    return case


# 功能：
#   原准入回执在载具检查期间变化后，重绑定来源记录仍须使用先前核验的同次字节。
# 输入：
#   rebinding_case：有效来源和新输出。
#   monkeypatch：在实际载具检查之后注入回执变化的工具。
# 输出：
#   None：不返回业务数据。
def test_rebinding_records_consumed_admission(rebinding_case, monkeypatch):
    case = rebinding_case
    original = case.source.receipt.read_bytes()
    qualified_vehicle = rebinding._qualified_vehicle

    # 功能：
    #   检查真实载具后仅替换本测试生成的原准入回执，仓库载具资产保持不变。
    # 输入：
    #   path：本次载具包路径。
    # 输出：
    #   result：真实载具及其来源证据。
    def mutate_admission(path):
        result = qualified_vehicle(path)
        case.source.receipt.write_bytes(b"another admission")
        return result

    monkeypatch.setattr(rebinding, "_qualified_vehicle", mutate_admission)
    assert rebinding.main() == 0
    result = json.loads(case.receipt.read_bytes())
    assert result["source_simulation_admission_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["qualification_granted"] is False


# 功能：
#   在任何重绑定发布之前拒绝原准入回执中的重复键，不能用最后的准入标记遮盖前值。
# 输入：
#   rebinding_case：原包、回执及新输出。
# 输出：
#   None：不返回业务数据。
def test_rebinding_rejects_duplicate_source_receipt(rebinding_case):
    case = rebinding_case
    path = case.source.receipt
    path.write_bytes(b'{"admitted_to_simulation":false,' + path.read_bytes()[1:])
    with pytest.raises(ValueError):
        rebinding.main()


# 功能：
#   新载具仅引用旧准入证明来源，不复用旧顾问分数；原回执摘要漂移必须拒绝。
# 输入：
#   rebinding_case：文件真实但权重不执行的重绑定夹具。
# 输出：
#   None：不返回业务数据。
def test_rebound_vehicle_does_not_inherit_old_advisor_metrics(rebinding_case):
    from scripts import evaluate_local_policy_offline as evaluation

    case = rebinding_case
    original = json.loads(case.source.receipt.read_bytes())
    original["advisor_metrics"] = {
        "state-anomaly-detector": {
            "sample_count": 40,
            "risky_sample_count": 20,
            "safe_sample_count": 20,
            "risk_hold_recall": 1.0,
            "safe_motion_recall": 1.0,
            "risk_mean_absolute_error": 0.0,
            "p99_inference_latency_ms": 1.0,
        }
    }
    original["advisor_evaluation_sha256"] = "e" * 64
    case.source.receipt.write_text(json.dumps(original), encoding="utf-8")
    assert rebinding.main() == 0
    args = dict(
        package_path=case.output,
        preservation_receipt_path=case.receipt,
        inherited_admission_receipt_path=case.source.receipt,
        trainable_advisor_roles={"state-anomaly-detector"},
    )
    assert evaluation._load_inherited_advisor_evidence(**args) == ({}, [])
    record = json.loads(case.receipt.read_bytes())
    record["source_simulation_admission_sha256"] = "f" * 64
    case.receipt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="rebinding receipt is invalid"):
        evaluation._load_inherited_advisor_evidence(**args)


# 功能：
#   载具重绑定不得在原模型包内创建输出，避免输入和候选相互嵌套。
# 输入：
#   rebinding_case：有效来源参数。
#   monkeypatch：设置嵌套输出的工具。
# 输出：
#   None：不返回业务数据。
def test_rebinding_rejects_nested_output(rebinding_case, monkeypatch):
    case = rebinding_case
    argv = list(case.argv)
    argv[argv.index("--output-package") + 1] = str(case.source.package.root / "nested")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError):
        rebinding.main()


# 功能：
#   最后发布前同名目录被占用时保留外部文件，不能把成功校验当成覆盖许可。
# 输入：
#   staging_case：已验证的来源包及尚未出现的输出。
#   monkeypatch：在真实发布函数之前注入目录竞争的工具。
# 输出：
#   None：不返回业务数据。
def test_staging_preserves_competing_output(staging_case, monkeypatch):
    case = staging_case
    publish = staging.publish_asset_directory

    # 功能：
    #   在发布边界创建另一方拥有的目录后调用真实无覆盖发布器。
    # 输入：
    #   staged：本次私有暂存目录。
    #   destination：最终发布目录。
    # 输出：
    #   result：真实发布器的返回值；本测试预期其抛出冲突异常。
    def competing_publish(staged, destination):
        destination.mkdir()
        (destination / "foreign.txt").write_bytes(b"preserve me")
        result = publish(staged, destination)
        return result

    monkeypatch.setattr(staging, "publish_asset_directory", competing_publish)
    with pytest.raises(FileExistsError):
        staging.main()
    assert (case.output / "foreign.txt").read_bytes() == b"preserve me"
    assert list(case.output.iterdir()) == [case.output / "foreign.txt"]


# 功能：
#   候选已发布但外部回执出现竞争时，保留竞争文件并报错，不宣告完成或赋予资格。
# 输入：
#   rebinding_case：原准入、载具及候选输出。
#   monkeypatch：在核验载具后写入另一方回执的工具。
# 输出：
#   None：不返回业务数据。
def test_rebinding_preserves_competing_receipt(rebinding_case, monkeypatch):
    case = rebinding_case
    qualified_vehicle = rebinding._qualified_vehicle

    # 功能：
    #   核验真实载具后模拟另一任务抢先占用最终回执。
    # 输入：
    #   path：已冻结的载具包。
    # 输出：
    #   result：真实载具及其证据。
    def occupy_receipt(path):
        result = qualified_vehicle(path)
        case.receipt.write_bytes(b"foreign receipt")
        return result

    monkeypatch.setattr(rebinding, "_qualified_vehicle", occupy_receipt)
    with pytest.raises(FileExistsError):
        rebinding.main()
    assert case.receipt.read_bytes() == b"foreign receipt"
    assert case.output.is_dir()
    assert not (case.output / "qualification.json").exists()


# 功能：
#   包加载后模型被改写时，真正复制的摘要必须被拒绝，不能发布损坏模型目录。
# 输入：
#   staging_case：完整源包、回执和输出。
#   monkeypatch：在模型复制前改变测试文件的工具。
# 输出：
#   None：不返回业务数据。
def test_staging_rejects_model_changed_before_copy(staging_case, monkeypatch):
    case = staging_case

    # 功能：
    #   在复制前替换测试模型字节，再让真实有界复制器计算新摘要。
    # 输入：
    #   source：被替换的测试模型。
    #   limit：有界读取预算。
    #   destination：独占目标路径。
    # 输出：
    #   digest：被改写模型的真实摘要。
    def changing_model(source, *, limit, destination=None):
        if destination is not None:
            source.write_bytes(b"modified model")
        digest = hash_plugin_file(source, limit=limit, destination=destination)
        return digest

    monkeypatch.setattr(staging, "hash_plugin_file", changing_model)
    with pytest.raises(RuntimeError, match="COPY_HASH_MISMATCH"):
        staging.main()
    assert not case.output.exists()
