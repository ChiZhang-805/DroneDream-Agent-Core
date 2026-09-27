"""Portable decision-corpus integrity and pre-GPU readiness; no invented labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .decision_shadow import ACTIONS
from .decision_state_adapter import DecisionStateV2, decision_digest, model_state

SPLITS = ("train", "development", "calibration", "test", "stress")
LICENSES = {"MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0", "proprietary-user-owned"}


# 功能：按块核对真实文件内容，不把已有 receipt 当文件仍正确的证据。
# 输入：文件路径；输出：SHA256；读取失败直接传播。
def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 功能：限定证据在语料根内，拒绝跨平台绝对路径、目录穿越和符号链接逃逸。
# 输入：语料根与 POSIX 相对路径；输出：解析后的普通文件。
def evidence_path(root: Path, relative: str) -> Path:
    if (type(relative) is not str or not relative or "\\" in relative or ":" in relative
            or "\x00" in relative or relative.startswith("/")):
        raise ValueError("DECISION_EVIDENCE_PATH_INVALID")
    parts = PurePosixPath(relative).parts
    if any(p in {"..", "."} for p in relative.split("/")):
        raise ValueError("DECISION_EVIDENCE_PATH_INVALID")
    resolved = root.joinpath(*parts).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("DECISION_EVIDENCE_PATH_INVALID")
    return resolved


# 功能：严格解析 JSON 对象，证据文件与语料使用同一规则，禁止后键覆盖前键。
# 输入：已由调用方限定大小的文本；输出：无重复键/非有限常量的 JSON 字典。
def decode_object(text: str) -> dict:
    # 功能：重复键不能覆盖身份/标签；输入：JSON 键值对；输出：唯一键字典。
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DECISION_JSON_DUPLICATE_KEY")
            result[key] = value
        return result

    # 功能：拒绝 JSON 扩展 NaN/Infinity；输入：常量；输出：始终抛错。
    def invalid(value):
        raise ValueError("DECISION_JSON_NONFINITE:" + value)

    value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError("DECISION_RECORD_NOT_OBJECT")
    return value


# 功能：读取有行长/总行数上限的 JSONL，拒绝重复键及非有限 JSON 常量。
# 输入：文件；输出：记录列表；损坏数据不静默跳过。
def read_records(path: Path, maximum_rows: int = 150000) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        while line := stream.readline(262145):
            if len(line) > 262144 or len(rows) >= maximum_rows:
                raise ValueError("DECISION_CORPUS_LIMIT_EXCEEDED")
            rows.append(decode_object(line))
    if not rows:
        raise ValueError("DECISION_CORPUS_EMPTY")
    return rows


# 功能：验证状态、标签、来源内容绑定及跨集合泄漏，计算而非信任正式资格。
# 输入：语料与根目录，smoke 仅允许合成接口用例；输出：不含训练权重的审计报告。
def audit_records(rows: list[dict], root: Path, *, smoke: bool = False) -> dict:
    if not rows:
        raise ValueError("DECISION_CORPUS_EMPTY")
    identities, group_splits, input_splits, checked_files = set(), {}, {}, {}
    episode_groups, unique_inputs = {}, set()
    map_groups = {}
    from .decision_layout_lineage import load_layout_families

    assignments = {}
    lineage, _ = ({}, {}) if smoke else load_layout_families(root, assignments=assignments)
    family_splits, missing_lineage = {}, set()
    coverage = {s: Counter() for s in SPLITS}
    unique_coverage = {s: Counter() for s in SPLITS}
    effective_train = Counter()
    counts, groups, episodes = Counter(), defaultdict(set), set()
    formal = 0
    for row in rows:
        required = {"sample_id", "parent_group", "episode_id", "split", "state",
                    "acceptable_actions", "license", "label_source", "evidence",
                    "formal_training_eligible"}
        if set(row) != required:
            raise ValueError("DECISION_RECORD_FIELDS_INVALID")
        for key in ("sample_id", "parent_group", "episode_id"):
            if type(row[key]) is not str or not row[key].strip():
                raise ValueError("DECISION_RECORD_ID_INVALID")
        if row["sample_id"] in identities:
            raise ValueError("DECISION_DUPLICATE_SAMPLE")
        identities.add(row["sample_id"])
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("DECISION_SPLIT_INVALID")
        group = row["parent_group"]
        if episode_groups.setdefault(row["episode_id"], group) != group:
            raise ValueError("DECISION_EPISODE_RELABELLED")
        if group_splits.setdefault(group, split) != split:
            raise ValueError("DECISION_GROUP_LEAKAGE")
        if row["license"] not in LICENSES:
            raise ValueError("DECISION_LICENSE_NOT_APPROVED")
        state = DecisionStateV2.model_validate_json(json.dumps(row["state"], allow_nan=False))
        family = group
        if not smoke:
            if group in lineage:
                if group in assignments and split != assignments[group]:
                    raise ValueError("DECISION_LAYOUT_PREASSIGNED_SPLIT_CHANGED")
                lineage_map, family = lineage[group]
                if lineage_map != state.map_sha256:
                    raise ValueError("DECISION_LAYOUT_MAP_BINDING_MISMATCH")
                if family_splits.setdefault(family, split) != split:
                    raise ValueError("DECISION_LAYOUT_FAMILY_LEAKAGE")
            else:
                missing_lineage.add(group)
        if not smoke and map_groups.setdefault(state.map_sha256, family) != family:
            raise ValueError("DECISION_MAP_RELABELLED_AS_NEW_LAYOUT")
        # 相同完整因果输入跨集合也是泄漏，即使人为换了 group 名字。
        input_hash = decision_digest(model_state(state))
        if input_splits.setdefault(input_hash, split) != split:
            raise ValueError("DECISION_INPUT_LEAKAGE")
        if not smoke and input_hash in unique_inputs:
            raise ValueError("DECISION_DUPLICATE_FORMAL_INPUT")
        unique_inputs.add(input_hash)
        labels = row["acceptable_actions"]
        if (type(labels) is not list or not labels or len(set(labels)) != len(labels)
                or any(a not in ACTIONS for a in labels)):
            raise ValueError("DECISION_LABEL_INVALID")
        if type(row["formal_training_eligible"]) is not bool:
            raise ValueError("DECISION_ELIGIBILITY_INVALID")
        if smoke:
            if (row["label_source"] != "synthetic-contract"
                    or row["formal_training_eligible"] or row["evidence"]):
                raise ValueError("DECISION_SMOKE_SOURCE_INVALID")
        else:
            if (row["label_source"] not in {"reviewed-expert", "verified-simulation-outcome"}
                    or row["formal_training_eligible"] is not True):
                raise ValueError("DECISION_FORMAL_LABEL_REQUIRED")
            references = row["evidence"]
            if type(references) is not list or not references:
                raise ValueError("DECISION_EVIDENCE_REQUIRED")
            documents = []
            for ref in references:
                if type(ref) is not dict or set(ref) != {"path", "sha256"}:
                    raise ValueError("DECISION_EVIDENCE_REFERENCE_INVALID")
                path = evidence_path(root, ref["path"])
                if path not in checked_files:
                    checked_files[path] = file_digest(path)
                if checked_files[path] != ref["sha256"]:
                    raise ValueError("DECISION_EVIDENCE_HASH_MISMATCH")
                if path.stat().st_size > 2 * 1024**2:
                    raise ValueError("DECISION_LABEL_CERTIFICATE_TOO_LARGE")
                documents.append(decode_object(path.read_text(encoding="utf-8")))
            # 此证书必须由独立校验器/审核器产生；字段存在本身不是物理真实性证明。
            certificates = [d for d in documents if d.get("schema_version")
                            == "dronedream.decision-label-certificate.v1"]
            if len(certificates) != 1:
                raise ValueError("DECISION_LABEL_CERTIFICATE_REQUIRED")
            certificate = certificates[0]
            if (certificate.get("state_sha256") != decision_digest(row["state"])
                    or certificate.get("parent_group") != group
                    or certificate.get("episode_id") != row["episode_id"]
                    or certificate.get("label_source") != row["label_source"]
                    or set(certificate.get("acceptable_actions", [])) != set(labels)
                    or certificate.get("verified") is not True
                    or not certificate.get("verifier_version")
                    or not certificate.get("outcome_references")):
                raise ValueError("DECISION_LABEL_CERTIFICATE_MISMATCH")
            for ref in certificate["outcome_references"]:
                path = evidence_path(root, ref["path"])
                if path not in checked_files:
                    checked_files[path] = file_digest(path)
                if checked_files[path] != ref["sha256"]:
                    raise ValueError("DECISION_OUTCOME_HASH_MISMATCH")
            from .decision_label_evidence import verify_label_evidence

            verify_label_evidence(row, certificate, root)
            formal += 1
        counts[split] += 1
        coverage[split].update(labels)
        if len(labels) == 1:
            unique_coverage[split].update(labels)
        if split == "train":
            for label in labels:
                effective_train[label] += 1 / len(labels)
        groups[split].add(family)
        episodes.add((group, row["episode_id"]))
    issues = []
    if missing_lineage:
        issues.append("LAYOUT_LINEAGE_REQUIRED")
    for split in SPLITS:
        if set(coverage[split]) != set(ACTIONS):
            issues.append("ACTION_COVERAGE:" + split)
        if len(groups[split]) < 2:
            issues.append("INDEPENDENT_GROUPS_BELOW_2:" + split)
    # 压力集是额外独立集合，不能拿它补足主语料 30,000 条。
    primary_formal = formal - counts["stress"] if not smoke else 0
    if primary_formal < 30000:
        issues.append("PRIMARY_FORMAL_WINDOWS_BELOW_30000")
    if len(episodes) < 1000:
        issues.append("EPISODES_BELOW_1000")
    if len(family_splits) < 20:
        issues.append("INDEPENDENT_LAYOUT_GROUPS_BELOW_20")
    if counts["stress"] < 3000 or any(coverage["stress"][a] < 300 for a in ACTIONS):
        issues.append("STRESS_COVERAGE_INCOMPLETE")
    if counts["train"] and any(not .1 <= effective_train[a]/counts["train"] <= .5
                              for a in ACTIONS):
        issues.append("TRAIN_ACTION_BALANCE_OUTSIDE_10_TO_50_PERCENT")
    # 集合标签不能用于唯一正确类别的召回率/ECE，必须另有足够独立唯一标签。
    for split in ("calibration", "test", "stress"):
        if any(unique_coverage[split][a] < 30 for a in ACTIONS):
            issues.append("UNIQUE_LABEL_METRIC_COVERAGE:" + split)
    return {"schema_version": "dronedream.decision-corpus-audit.v2", "rows": len(rows),
            "formal_windows": formal, "smoke_only": smoke,
            "primary_formal_windows": primary_formal,
            "unique_label_coverage": {k: dict(v) for k, v in unique_coverage.items()},
            "split_counts": dict(counts), "coverage": {k: dict(v) for k, v in coverage.items()},
            "group_counts": {k: len(v) for k, v in groups.items()}, "episodes": len(episodes),
            "verified_layout_families": len(family_splits),
            "unverified_layout_groups": len(missing_lineage),
            "verified_files": len(checked_files), "gpu_data_ready": not issues and not smoke,
            "readiness_issues": issues, "flight_authority": False}
