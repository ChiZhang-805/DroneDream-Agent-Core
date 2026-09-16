"""Synthetic source replacement/leakage regressions, not flight evidence."""

import hashlib
import json

import pytest
from test_local_advisor_dataset import _write_run
from test_local_advisor_training import _sample
from test_mission_groups import route, split_fixture

from dronedream_agent_core.training.advisor_sources import (
    RecordedSource,
    validate_advisor_spatial_splits,
)
from dronedream_agent_core.training.mission_groups import mission_group_evidence
from scripts.build_local_advisor_dataset import _collect_run_samples, _source_receipt
from scripts.train_local_advisors import _load_samples, _sha256


# 功能：
#   将已构造的合成路线证据与同一语义地图摘要组织为训练来源记录。
# 输入：
#   split：有效路线分组证据。
# 输出：
#   row：供顾问训练划分使用的来源记录。
def source(split):
    row = {"mission_split": split.model_dump(), "semantic_sha256": split.semantic_sha256}
    return row


# 功能：
#   验证改名和反向路线仍与训练组重叠，而另一条空间路线可以作为独立组。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_renamed_reversed_advisor_flights_cannot_become_independent_validation():
    reversed_split = mission_group_evidence(json.dumps({
        **route([(2, 0, 1), (1, 0, 1), (0, 0, 1)]), "name": "another-run", "seed": 13,
    }).encode(), "a" * 64)
    with pytest.raises(ValueError, match="SPATIAL_ROUTE_OVERLAP"):
        validate_advisor_spatial_splits([source(split_fixture())], [source(reversed_split)])
    groups = validate_advisor_spatial_splits([source(split_fixture())],
                                            [source(split_fixture(y=2))])
    assert not groups[0] & groups[1]


# 功能：
#   验证缺少路线证据、原文变化、地图变化或分组摘要变化均阻止训练划分通过。
# 输入：
#   mutation：损坏来源记录的方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["missing", "route", "map", "group"])
def test_advisor_sources_revalidate_original_route_and_map(mutation):
    item = source(split_fixture())
    if mutation == "missing":
        item.pop("mission_split")
    elif mutation == "map":
        item["semantic_sha256"] = "b" * 64
    elif mutation == "route":
        item["mission_split"]["route_source_utf8"] = "{}"
    else:
        item["mission_split"]["group_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        validate_advisor_spatial_splits([item], [source(split_fixture(y=2))])


# 功能：
#   验证原路径被改写后，训练加载与摘要仍消费最初同一份冻结字节。
# 输入：
#   tmp_path：独占来源目录。
# 输出：
#   None：不返回业务数据。
def test_training_consumes_same_bytes_even_if_original_path_is_replaced(tmp_path):
    path = tmp_path / "samples.jsonl"
    expected = _sample(role="perception-health-critic", risky=False)
    original = (expected.model_dump_json() + "\n").encode()
    path.write_bytes(original)
    frozen = RecordedSource.read(path)
    path.write_text('{}\n')
    assert _load_samples(frozen) == [expected]
    assert _sha256(frozen) == hashlib.sha256(original).hexdigest()


# 功能：
#   验证采集快照、控制周期和深度历史改写后，数据编译结果仍对应已验证的原始来源。
# 输入：
#   tmp_path：合成运行记录目录。
# 输出：
#   None：不返回业务数据。
def test_dataset_compiles_the_verified_source_not_later_path_contents(tmp_path):
    root = tmp_path / "run"
    _write_run(root, sensor_rate_history=True, multimodal_evidence=True)
    receipt, snapshots, cycles, depth, multimodal = _source_receipt(
        root, require_verified=True, allow_verified_fault_recovery=False,
        allow_verified_payload_collection=False, allow_verified_payload_recovery=False)
    expected = _collect_run_samples(snapshots, cycles, depth, multimodal,
        roles=("perception-health-critic",), source_identity=receipt["mission_evidence_sha256"])
    for filename in ("model-navigation-snapshots.jsonl", "model-navigation-cycles.jsonl",
                     "depth-local-safety-history.jsonl"):
        (root / filename).write_text('{"changed":true}\n')
    actual = _collect_run_samples(snapshots, cycles, depth, multimodal,
        roles=("perception-health-critic",), source_identity=receipt["mission_evidence_sha256"])
    assert actual == expected and actual
