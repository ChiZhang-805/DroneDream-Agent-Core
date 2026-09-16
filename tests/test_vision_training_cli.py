"""Offline publication-boundary tests; stub metrics do not qualify a product model."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dronedream_agent_core.local_vision_training import (
    LocalVisionTrainingMetrics,
    LocalVisionTrainingSample,
)


# 功能：
#   独立加载训练 CLI，测试不触发命令行主入口或预训练模型下载。
# 输入：
#   无。
# 输出：
#   module：本次测试独立持有的命令行模块。
@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_local_vision.py"
    spec = importlib.util.spec_from_file_location("vision_training_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：
#   创建仅用于发布边界的摘要绑定样本，不将这些字节当作真实可训练图像。
# 输入：
#   root：隔离目录。
#   flight：互不重叠的飞行标识。
# 输出：
#   path：完整换行结尾的单行样本清单。
def _dataset(root, flight):
    image = root / f"{flight}.png"
    image.write_bytes(f"publication-fixture-{flight}".encode())
    sample = LocalVisionTrainingSample(
        flight_id=flight,
        map_sha256="0" * 64,
        image_relative_path=image.name,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        traversability_target=0.5,
        scene_targets=[0.0] * 6,
        quality_targets=[0.0] * 4,
    )
    path = root / f"{flight}.jsonl"
    path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    return path


# 功能：
#   隔离训练计算，仅运行真实清单读取、来源复核、模型与回执发布代码。
# 输入：
#   cli：待测命令行模块。
#   tmp_path：本次测试专属目录。
#   monkeypatch：局部替换计算依赖和参数。
# 输出：
#   state：输入、输出路径及可注入发布竞争的测试状态。
@pytest.fixture
def run_state(cli, tmp_path, monkeypatch):
    train = _dataset(tmp_path, "train")
    validation = _dataset(tmp_path, "validation")
    model = tmp_path / "expert.onnx"
    receipt = tmp_path / "receipt.json"
    state = {"train": train, "validation": validation, "model": model, "receipt": receipt}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_local_vision",
            "--training-root",
            str(tmp_path),
            "--training-data",
            str(train),
            "--validation-root",
            str(tmp_path),
            "--validation-data",
            str(validation),
            "--output-model",
            str(model),
            "--training-receipt",
            str(receipt),
            "--benchmark-iterations",
            "5",
        ],
    )
    metrics = LocalVisionTrainingMetrics(
        sample_count=20,
        semantic_sample_count=20,
        mean_loss=0.1,
        semantic_pixel_accuracy=0.9,
        semantic_class_target_pixels=[100] * 8,
        semantic_class_iou=[0.8] * 8,
        semantic_mean_iou=0.8,
        traversability_mean_absolute_error=0.1,
        scene_binary_accuracy=0.9,
        quality_binary_accuracy=0.9,
    )
    monkeypatch.setattr(cli, "local_vision_semantic_class_weights", lambda *_: [1.0] * 8)
    monkeypatch.setattr(cli, "build_local_vision_model", lambda *_, **__: object())
    monkeypatch.setattr(cli, "train_local_vision_model", lambda model, *_: (model, metrics))
    monkeypatch.setattr(cli, "evaluate_local_vision_model", lambda *_: metrics)
    monkeypatch.setattr(cli, "count_trainable_parameters", lambda *_: 1)

    # 功能：
    #   在真实独立暂存目录写入小型占位制品，仅供文件发布测试。
    # 输入：
    #   model、config：本用例不执行计算的模型和配置。
    #   path：CLI 选择的尚不存在暂存路径。
    # 输出：
    #   digest：占位制品的实际摘要。
    def export(model, path, config):
        path.write_bytes(b"isolated-test-artifact")
        state["staged"] = path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest

    # 功能：
    #   在模拟评测阶段注入用户文件变化，并提供有限时延测试值。
    # 输入：
    #   path、config、iterations：被测入口传来的评测参数。
    # 输出：
    #   result：仅用于门控测试的基准字典。
    def benchmark(path, config, *, iterations):
        hook = state.get("hook")
        if hook is not None:
            hook()
        result = {"p99_inference_latency_ms": state.get("latency", 1.0)}
        return result

    monkeypatch.setattr(cli, "export_local_vision_onnx", export)
    monkeypatch.setattr(cli, "benchmark_local_vision_onnx", benchmark)
    return state


# 功能：
#   已通过门控的发布必须绑定解析清单及制品摘要，并保持飞行资格为假。
# 输入：
#   cli、run_state：真实发布入口及隔离计算环境。
# 输出：
#   None：不返回业务数据。
def test_publication_binds_original_inputs(cli, run_state):
    assert cli.main() == 0
    receipt = json.loads(run_state["receipt"].read_text())
    assert (
        receipt["training_data_sha256"]
        == hashlib.sha256(run_state["train"].read_bytes()).hexdigest()
    )
    assert receipt["artifact_sha256"] == hashlib.sha256(run_state["model"].read_bytes()).hexdigest()
    assert receipt["flight_qualification_granted"] is False


# 功能：
#   清单重复键或截断末行不得被静默当作有效监督输入。
# 输入：
#   cli、tmp_path：解析器及隔离目录。
#   kind：清单损坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["duplicate", "truncated"])
def test_manifest_rejects_ambiguous_or_incomplete_lines(cli, tmp_path, kind):
    path = _dataset(tmp_path, "train")
    raw = path.read_text()
    raw = (
        raw.replace('{"flight_id":', '{"flight_id":"ignored","flight_id":')
        if (kind == "duplicate")
        else raw.rstrip("\n")
    )
    path.write_text(raw)
    with pytest.raises(ValueError):
        cli._samples(path)


# 功能：
#   训练期间清单或图像被改动后，不能发布原评测结果对应的合格回执。
# 输入：
#   cli、run_state：入口与隔离环境。
#   kind：变化的输入类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["manifest", "image"])
def test_changed_sources_cannot_be_published(cli, run_state, kind):
    path = run_state["train"] if kind == "manifest" else run_state["train"].with_suffix(".png")
    run_state["hook"] = lambda: path.write_bytes(b"changed during training")
    with pytest.raises(ValueError):
        cli.main()
    assert not run_state["model"].exists()
    assert not run_state["receipt"].exists()


# 功能：
#   后到的训练输出不得覆盖先出现的用户模型或回执。
# 输入：
#   cli、run_state：入口与隔离环境。
#   kind：被用户抢先写入的输出类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["model", "receipt"])
def test_output_race_preserves_existing_file(cli, run_state, kind):
    path = run_state[kind]
    run_state["hook"] = lambda: path.write_bytes(b"user-owned data")
    with pytest.raises(FileExistsError):
        cli.main()
    assert path.read_bytes() == b"user-owned data"


# 功能：
#   非有限、布尔或负时延不能通过性能门控并发布模型。
# 输入：
#   cli、run_state：入口与隔离环境。
#   latency：非法评测时延。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1.0, True])
def test_invalid_latency_does_not_pass_gate(cli, run_state, latency):
    run_state["latency"] = latency
    with pytest.raises(ValueError):
        cli.main()
    assert not run_state["model"].exists()


# 功能：
#   性能不达标仅保存拒绝回执，不发布或保留已知归属的暂存模型。
# 输入：
#   cli、run_state：入口与隔离环境。
# 输出：
#   None：不返回业务数据。
def test_failed_gate_retains_rejection_only(cli, run_state):
    run_state["latency"] = 101.0
    assert cli.main() == 1
    assert not run_state["model"].exists()
    assert not run_state["staged"].exists()
    receipt = json.loads(run_state["receipt"].read_text())
    assert not receipt["training_accepted"]
    assert receipt["artifact_sha256"] is None


# 功能：
#   回执发布失败时保留已发布的模型供恢复，不虚构成功回执或删除模型。
# 输入：
#   cli、run_state、monkeypatch：入口、隔离状态和故障注入工具。
# 输出：
#   None：不返回业务数据。
def test_receipt_failure_preserves_recoverable_model(cli, run_state, monkeypatch):
    # 功能：
    #   模拟回执落盘错误，不影响模型发布路径。
    # 输入：
    #   args、kwargs：回执发布调用参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_receipt(*args, **kwargs):
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(cli, "publish_runtime_json", fail_receipt)
    with pytest.raises(OSError):
        cli.main()
    assert run_state["model"].read_bytes() == b"isolated-test-artifact"
    assert not run_state["receipt"].exists()


# 功能：
#   暂存目录出现未知资料时，完成或拒绝训练都不进行递归删除。
# 输入：
#   cli、run_state：入口与隔离状态。
# 输出：
#   None：不返回业务数据。
def test_staging_cleanup_preserves_unowned_material(cli, run_state):
    run_state["hook"] = lambda: (run_state["staged"].parent / "unowned.txt").write_text("keep")
    assert cli.main() == 0
    assert (run_state["staged"].parent / "unowned.txt").read_text() == "keep"


# 功能：
#   被替换的暂存模型不能发布，也不能在异常清理时误删替代文件。
# 输入：
#   cli、run_state：入口与隔离状态。
# 输出：
#   None：不返回业务数据。
def test_staging_replacement_is_not_deleted(cli, run_state):
    # 功能：
    #   保留原 inode 后创建不同内容的同名文件，模拟评测期间替换。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def replace_staged():
        path = run_state["staged"]
        path.rename(path.with_suffix(".original"))
        path.write_bytes(b"replacement")

    run_state["hook"] = replace_staged
    with pytest.raises(ValueError):
        cli.main()
    assert run_state["staged"].read_bytes() == b"replacement"
    assert not run_state["model"].exists()
