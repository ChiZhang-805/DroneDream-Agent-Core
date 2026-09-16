import copy

import pytest
from test_causal_policy import samples
from test_training_observation_boundary import observation

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.evidence_snapshot import detach_evidence
from dronedream_agent_core.training.flight_environment import FlightObservation


# 功能：
#   验证真实观测模型的当前特征、历史行与来源对象全部独立，复制不改变证据摘要。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_all_nested_observation_containers_are_detached_without_changing_hash():
    rows = [observation(s) for s in samples()[:4]]
    value = FlightObservation(mission_id="m", episode_id="e", map_sha256="a"*64,
        sequence=0, sample=rows[-1], prior_observations=rows[:-1])
    expected = copy.deepcopy(value)
    detached = detach_evidence(value)
    assert sha256_json(detached) == sha256_json(expected)
    value.sample.state_features[0] = 99.
    value.sample.realtime_features[0] = 99.
    value.sample.candidate_features[0][0] = 99.
    value.sample.temporal_evidence.stream_id = "changed"
    value.prior_observations[0].realtime_valid_mask[0] = .5
    value.prior_observations[-1].candidate_mask[0] = 1.
    value.prior_observations.pop()
    assert detached == expected


# 功能：
#   验证嵌套列表与元组保持形状且不共享可变子项，循环和集合明确拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_json_containers_and_immutable_atoms_have_correct_ownership():
    source = {"nested": [{"axes": [0., .2]}, [1, 2, 3]], "tuple": ([1, 2], "id")}
    detached = detach_evidence(source)
    source["nested"][0]["axes"][0] = 7.
    source["nested"][1].append(4)
    source["tuple"][0].append(9)
    assert detached == {"nested": [{"axes": [0., .2]}, [1, 2, 3]], "tuple": ([1, 2], "id")}
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="NESTING_EXCEEDED"):
        detach_evidence(cycle)
    with pytest.raises(ValueError, match="UNSUPPORTED_VALUE"):
        detach_evidence({"not-json": {1, 2}})
