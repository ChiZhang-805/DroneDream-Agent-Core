"""Export is exclusive even when a competing writer wins after the existence check."""

from pathlib import Path

import pytest

from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    export_causal_policy,
)


# 功能：
#   即使早期存在性检查失真，最后的独占创建仍保护其他写入者的模型文件。
# 输入：
#   tmp_path：隔离的模型目录。
#   monkeypatch：模拟存在性检查与实际创建之间的竞争。
# 输出：
#   None：不返回业务数据。
def test_export_never_overwrites_after_stale_existence_check(tmp_path, monkeypatch):
    model = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1))
    target = tmp_path / "existing.onnx"
    target.write_bytes(b"preserve-existing-user-model")
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == target else real_exists(path))
    with pytest.raises(FileExistsError):
        export_causal_policy(model, target)
    assert target.read_bytes() == b"preserve-existing-user-model"


# 功能：
#   ONNX 导出临时切换评价模式后，必须恢复调用者原有训练模式。
# 输入：
#   tmp_path：隔离的模型输出目录。
# 输出：
#   None：不返回业务数据。
def test_export_restores_the_callers_training_mode(tmp_path):
    model = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1)).train()
    export_causal_policy(model, tmp_path / "new.onnx")
    assert model.training
