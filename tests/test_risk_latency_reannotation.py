"""Synthetic provenance tests for actual re-integration, not merely new labels."""

import json

import pytest
from test_native_action_risk_artifacts import dataset

from dronedream_agent_core.training.action_risk_artifacts import load_action_risk_dataset
from dronedream_agent_core.training.risk_latency_reannotation import reannotate_decision_latency


# 功能：确认所有原观测和分组保留，新的轨迹确实来自新积分策略，原始文件不被覆盖。
# 输入：tmp_path、monkeypatch：合成课程夹具。输出：验证来源、重新积分及完整数据接纳。
def test_reprediction_preserves_sources_and_recomputes_dynamics(tmp_path, monkeypatch):
    source = dataset(tmp_path, monkeypatch)
    original = load_action_risk_dataset(source)
    geometry = json.loads((tmp_path / "semantic.json").read_bytes())["collision_primitives"]
    destination = tmp_path / "bounded"
    report = reannotate_decision_latency(source, destination, geometry)
    updated = load_action_risk_dataset(destination)
    assert load_action_risk_dataset(source).receipt_sha256 == original.receipt_sha256
    assert updated.observations == original.observations and updated.groups == original.groups
    assert len(updated.samples) == len(original.samples)
    assert report["reannotation"]["new_physical_observation_count"] == 0
    assert report["reannotation"]["changed_physical_prediction"] is True
    before = [
        json.loads(line)["receipt"]
        for line in (source / "counterfactual-receipts.jsonl").read_text().splitlines()
        if "receipt" in json.loads(line)
    ]
    after = [
        json.loads(line)["receipt"]
        for line in (destination / "counterfactual-receipts.jsonl").read_text().splitlines()
        if "receipt" in json.loads(line)
    ]
    assert any(a["positions_m"] != b["positions_m"] for a, b in zip(before, after, strict=True))
    assert all(row["effective_latency_seconds"] == 0.25 for row in after)
    with pytest.raises(ValueError, match="DESTINATION_MUST_BE_NEW"):
        reannotate_decision_latency(source, destination, geometry)


# 功能：错误地图不得用于重新生成标签，即使形状和字段完全合法。
# 输入：tmp_path、monkeypatch：合成来源。输出：不创建目标，不改原始数据。
def test_reprediction_rejects_other_geometry(tmp_path, monkeypatch):
    source = dataset(tmp_path, monkeypatch)
    geometry = json.loads((tmp_path / "semantic.json").read_bytes())["collision_primitives"]
    geometry[0]["center_x"] += 0.1
    destination = tmp_path / "wrong"
    with pytest.raises(ValueError, match="GEOMETRY_MISMATCH"):
        reannotate_decision_latency(source, destination, geometry)
    assert not destination.exists()
