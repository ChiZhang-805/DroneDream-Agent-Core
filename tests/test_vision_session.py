"""Real optimizer recovery with synthetic images; never evidence of flight capability."""

import json

import pytest
import torch
from test_local_vision_training import _dataset

from dronedream_agent_core import local_vision_training as training
from dronedream_agent_core.training.vision_session import (
    TrainingBudgetReached,
    VisionTrainingSession,
    cpu_snapshot,
)


# 功能：
#   限制小型回归的线程开销，结束后恢复调用方设置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# 功能：
#   创建物理分离且图像内容不同的合成训练与验证集合。
# 输入：
#   root：本测试专属目录。
# 输出：
#   config、train、validation：小型真实网络配置及不重叠样本。
def datasets(root):
    from PIL import Image

    train_root, val_root = root / "train", root / "val"
    train_root.mkdir()
    val_root.mkdir()
    train = training.resolve_local_vision_samples(train_root, _dataset(train_root))
    val_samples = _dataset(val_root)
    for index, sample in enumerate(val_samples):
        import hashlib

        image = val_root / sample.image_relative_path
        Image.new("RGB", (64, 64), (15 + index, 40, 80)).save(image)
        sample.flight_id = f"validation-{index}"
        sample.image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    validation = training.resolve_local_vision_samples(val_root, val_samples)
    config = training.LocalVisionTrainingConfig(width=64, height=64, batch_size=2,
        epoch_count=2, freeze_backbone_epochs=1)
    return config, train, validation


# 功能：
#   比较不中断训练与批次后中断恢复，要求最佳权重逐张量一致。
# 输入：
#   tmp_path：训练、检查点隔离目录。
# 输出：
#   None：不返回业务数据。
def test_resume_matches_uninterrupted_optimizer_and_weights(tmp_path):
    config, train, validation = datasets(tmp_path)
    binding = {"config": config.model_dump(mode="json"), "dataset": "synthetic-fixed"}
    full = VisionTrainingSession(tmp_path / "full", binding)
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    expected, _ = training.train_local_vision_model(model, train, config, session=full,
                                                   validation_samples=validation)
    interrupted = VisionTrainingSession(tmp_path / "resume", binding)
    interrupted.started -= 3601
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    with pytest.raises(TrainingBudgetReached):
        training.train_local_vision_model(model, train, config, session=interrupted,
                                          validation_samples=validation)
    assert interrupted.steps == 1
    restored = VisionTrainingSession(tmp_path / "resume", binding, resume=True)
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    actual, _ = training.train_local_vision_model(model, train, config, session=restored,
                                                 validation_samples=validation)
    assert restored.steps == full.steps == 4
    assert restored.history == full.history
    assert actual.initialization_record == expected.initialization_record
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], atol=0, rtol=0)


# 功能：
#   损坏文件和改变数据绑定都必须在模型状态恢复前失败。
# 输入：
#   tmp_path、bad：隔离目录和损坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", ["binding", "bytes", "path"])
def test_resume_rejects_wrong_or_corrupt_state(tmp_path, bad):
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    optimizer = torch.optim.AdamW(model.parameters())
    session = VisionTrainingSession(tmp_path / "run", {"dataset": "original"})
    record = session.save(model, optimizer, "fixture")
    binding = {"dataset": "original"}
    if bad == "binding":
        binding["dataset"] = "changed"
    elif bad == "bytes":
        (session.directory / record["file"]).write_bytes(b"broken")
    else:
        record["file"] = "../outside.pt"
        (session.directory / "latest.json").write_text(json.dumps(record))
    restored = VisionTrainingSession(session.directory, binding, resume=True)
    with pytest.raises(ValueError):
        restored.restore(model, optimizer)


# 功能：
#   拒绝覆盖非空作业目录，快照不得包含非有限参数或任意 Python 对象。
# 输入：
#   tmp_path：隔离作业目录。
# 输出：
#   None：不返回业务数据。
def test_sessions_and_snapshots_fail_closed(tmp_path):
    (tmp_path / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError):
        VisionTrainingSession(tmp_path, {})
    for value in (object(), torch.tensor([float("nan")]), {"invalid": float("inf")}):
        with pytest.raises(ValueError):
            cpu_snapshot(value)
    assert (tmp_path / "user-file").read_text() == "preserve"


# 功能：
#   第二个进程式会话不能同时进入同一作业，异常退出后锁仍可被合法恢复者获取。
# 输入：
#   tmp_path：锁文件隔离目录。
# 输出：
#   None：不返回业务数据。
def test_session_writer_excludes_concurrent_recovery(tmp_path):
    session = VisionTrainingSession(tmp_path / "run", {})
    second = VisionTrainingSession(tmp_path / "run", {}, resume=True)
    with session.writer(), pytest.raises(OSError), second.writer():
        pytest.fail("second writer entered")
    with second.writer():
        assert (second.directory / ".writer.lock").is_file()


# 功能：
#   检查点即使具有合法摘要，也不能带着不一致的轮次、最佳状态或优化步数恢复。
# 输入：
#   tmp_path、fault：隔离检查点目录和要注入的状态不一致。
# 输出：
#   None：错误历史或进度均须在下一次训练之前拒绝。
@pytest.mark.parametrize("fault", ["history", "best", "steps", "pointer"])
def test_resume_validates_internal_progress_and_history(tmp_path, fault):
    config = training.LocalVisionTrainingConfig(width=64, height=64)
    model = training.build_local_vision_model(config, pretrained_backbone=False)
    optimizer = torch.optim.AdamW(model.parameters())
    session = VisionTrainingSession(tmp_path / "run", {})
    if fault == "history":
        session.next_epoch = 1
    elif fault == "best":
        session.best_score = 1.0
    elif fault == "steps":
        session.steps = 12
    record = session.save(model, optimizer, "corrupted-state-test")
    if fault == "pointer":
        record["steps"] = 1
        (session.directory / "latest.json").write_text(json.dumps(record))
    resumed = VisionTrainingSession(session.directory, {}, resume=True)
    with pytest.raises(ValueError, match="(CHECKPOINT_|PROGRESS_)"):
        resumed.restore(model, optimizer)
        resumed.validate_progress(2, 20)
