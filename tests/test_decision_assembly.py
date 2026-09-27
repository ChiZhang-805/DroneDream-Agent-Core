"""Assembly tests use explicit synthetic physical fixtures, never production counts."""

import json

from test_decision_input_evidence import input_fixture
from test_decision_label_evidence import native_fixture, reference

from dronedream_agent_core.decision_state_adapter import decision_digest
from scripts.assemble_decision_corpus import assemble


# 功能：构建格式正确的合成来源，仅测证据链；输入：临时目录；输出：清单路径及内容。
def collection(root):
    row, outcome = native_fixture()
    _, proof = input_fixture()
    raw = proof["snapshots"][0]
    raw["strategic_context"]["task"]["navigation_goal_id"] = row["state"]["goal_id"]
    raw["current_position_m"] = {"x": 0.0, "y": 0.0, "z": 1.5}
    raw["goal_position_m"] = outcome["goal_world_enu_m"]
    raw.pop("snapshot_sha256")
    raw["snapshot_sha256"] = decision_digest(raw)
    from dronedream_agent_core.decision_collection_teacher import CollectionTeacher

    teacher = CollectionTeacher(
        map_sha256=raw["known_static_map"]["source_sha256"],
        primitives=json.loads(proof["semantic_text"])["runtime_collision_primitives"],
        **proof["teacher"],
    )
    state, _, issues = teacher.observe(raw, sequence=0, **proof["identity"])
    row["state"] = state.model_dump(mode="json")
    outcome["state_sha256"] = outcome["behavior_application"]["state_sha256"] = decision_digest(
        row["state"]
    )
    outcome["map_sha256"] = state.map_sha256
    outcome["static_primitives"] = json.loads(proof["semantic_text"])[
        "runtime_collision_primitives"
    ]
    outcome["geometry_sha256"] = decision_digest(outcome["static_primitives"])
    input_reference = reference(root, "input.json", proof)
    candidate = {
        "schema_version": "dronedream.decision-candidate.v2",
        "sample_id": decision_digest(row["state"]),
        "state": row["state"],
        "source_sha256": input_reference["sha256"],
        "input_evidence": input_reference,
        "coverage_issues": issues,
        "label_status": "awaiting-independent-outcome",
        "formal_training_eligible": False,
    }
    entry = {
        "candidate": reference(root, "candidate.json", candidate),
        "outcomes": [reference(root, "outcome.json", outcome)],
        **{k: row[k] for k in ("parent_group", "episode_id", "split", "license")},
    }
    payload = {"schema_version": "dronedream.decision-collection-manifest.v1", "entries": [entry]}
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


# 功能：整链结果必须重新校验且少量数据不触发租卡；输入：合成夹具；输出：一条/未准入。
def test_assembly_recomputes_evidence(tmp_path):
    path, _ = collection(tmp_path)
    result = assemble(path, tmp_path / "output")
    assert result["formal_decision_windows"] == 1  # 测试夹具，绝不计入产品训练清单。
    assert not result["gpu_ready"]
    assert (tmp_path / "output/decisions.jsonl").exists()


# 功能：缺原生结果不能用规则补标签；输入：空结果列表；输出：隔离且零正式记录。
def test_missing_outcome_is_quarantined(tmp_path):
    path, payload = collection(tmp_path)
    payload["entries"][0]["outcomes"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = assemble(path, tmp_path / "output")
    assert result["formal_decision_windows"] == 0
    assert result["rejected_entries"] == 1
    assert not (tmp_path / "output/decisions.jsonl").exists()


# 功能：重复样本不能绕过全局检查；输入：重复来源；输出：整包不产生可训练文件。
def test_duplicates_not_published(tmp_path):
    path, payload = collection(tmp_path)
    payload["entries"].append(payload["entries"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = assemble(path, tmp_path / "output")
    assert result["formal_decision_windows"] == 0
    assert "GLOBAL_INTEGRITY" in result["remaining"][0]
    assert not (tmp_path / "output/decisions.jsonl").exists()


# 功能：非必要方向缺失不自动否定独立验证的正向行为；输入：覆盖缺口；输出：保留示范。
def test_known_noncritical_gap_is_not_blanket_rejection(tmp_path):
    path, payload = collection(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate = json.loads(candidate_path.read_text("utf-8"))
    assert "GOAL_ALIGNED_SECTORS_NOT_BODY_CLEARANCES" in candidate["coverage_issues"]
    payload["entries"][0]["candidate"] = reference(tmp_path, "candidate.json", candidate)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert assemble(path, tmp_path / "output")["verified_candidates"] == 1


# 功能：仅填写来源哈希不能让自声明的安全状态成为训练输入；输出：缺原始来源时零准入。
def test_hash_string_without_source_is_rejected(tmp_path):
    path, payload = collection(tmp_path)
    candidate = json.loads((tmp_path / "candidate.json").read_text("utf-8"))
    candidate.pop("input_evidence")
    payload["entries"][0]["candidate"] = reference(tmp_path, "candidate.json", candidate)
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = assemble(path, tmp_path / "output")
    assert result["formal_decision_windows"] == 0
    assert any("INPUT_EVIDENCE_REQUIRED" in key for key in result["rejection_reasons"])


# 功能：独立结果不能偷换成更空的地图；输入：重签但换图的结果；输出：零正式记录。
def test_outcome_must_use_input_geometry(tmp_path):
    path, payload = collection(tmp_path)
    outcome = json.loads((tmp_path / "outcome.json").read_text("utf-8"))
    outcome["static_primitives"][0]["center_x"] += 10
    outcome["geometry_sha256"] = decision_digest(outcome["static_primitives"])
    payload["entries"][0]["outcomes"] = [reference(tmp_path, "outcome.json", outcome)]
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = assemble(path, tmp_path / "output")
    assert result["formal_decision_windows"] == 0
    assert any("GEOMETRY_MISMATCH" in key for key in result["rejection_reasons"])
