"""Synthetic binding tests; these fixtures never contribute to production counts."""

import json

import pytest
from test_decision_input_evidence import input_fixture

from dronedream_agent_core.decision_state_adapter import decision_digest
from scripts import prepare_native_decision_manifest as binder


# 功能：构造明确合成回合，不伪造落地校验；输入：测试目录；输出：输入/候选/结果文件。
def episode_fixture(root):
    root.mkdir()
    windows = root / "decision-windows"
    windows.mkdir()
    state, proof = input_fixture()
    (root / "decision-input-source.json").write_text(json.dumps(proof), encoding="utf-8")
    candidate = {"state": state}
    (windows / "000-candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    outcome = {
        "state_sha256": decision_digest(state),
        "parent_group": "synthetic",
        "episode_id": "synthetic-episode",
    }
    (windows / "000-outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
    return root


# 功能：绑定成功只表示输入可重放，不能直接变成正式数据；输入：合成回合；输出：零准入。
def test_binding_is_portable_but_does_not_grant_labels(tmp_path, monkeypatch):
    source = episode_fixture(tmp_path / "episode")
    checked = []
    monkeypatch.setattr(binder, "require_grounded_stream", lambda p: checked.append(p))
    output = tmp_path / "bound"
    report = binder.prepare(source, output, "train")
    assert checked == [source]
    assert report["bound_windows"] == 1 and report["formal_decision_windows"] == 0
    assert report["gpu_ready"] is False
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    candidate = json.loads(
        (output / manifest["entries"][0]["candidate"]["path"]).read_text("utf-8")
    )
    assert candidate["label_status"] == "awaiting-independent-outcome"
    assert not candidate["input_evidence"]["path"].startswith(str(source))
    with pytest.raises(FileExistsError):
        binder.prepare(source, output, "train")


# 功能：阻止活动/未落地回合被打包；输入：生命周期校验失败；输出：无新目录。
def test_requires_confirmed_grounded_stream(tmp_path, monkeypatch):
    source = episode_fixture(tmp_path / "episode")

    def reject(_):
        raise ValueError("NOT_GROUNDED")

    monkeypatch.setattr(binder, "require_grounded_stream", reject)
    with pytest.raises(ValueError, match="NOT_GROUNDED"):
        binder.prepare(source, tmp_path / "bound", "train")
    assert not (tmp_path / "bound").exists()


# 功能：损坏输入与错配结果均隔离并保留原因；输入：两种篡改；输出：无可用窗口。
@pytest.mark.parametrize("fault", ["sequence", "state-binding"])
def test_binding_quarantines_mismatch(tmp_path, monkeypatch, fault):
    source = episode_fixture(tmp_path / "episode")
    monkeypatch.setattr(binder, "require_grounded_stream", lambda _: None)
    target = (
        source
        / "decision-windows"
        / ("000-candidate.json" if fault == "sequence" else "000-outcome.json")
    )
    document = json.loads(target.read_text("utf-8"))
    if fault == "sequence":
        document["state"]["sequence"] = 100
    else:
        document["state_sha256"] = "b" * 64
    target.write_text(json.dumps(document), encoding="utf-8")
    report = binder.prepare(source, tmp_path / "bound", "train")
    assert report["bound_windows"] == 0
    assert len(report["input_quarantine"]) == 1
    assert target.exists()
