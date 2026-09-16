"""Synthetic action probes/grounding tests, never physical flight evidence."""

import pytest
from test_native_corrections import fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.action_risk_dataset import (
    bounded_action_probes,
    counterfactual_risk_samples,
)
from dronedream_agent_core.training.flight_environment import PilotAction


# 功能：
#   验证离线探针覆盖四轴正负方向、两档幅度和零动作，不包含位置控制。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_probes_have_both_signs_and_magnitudes_without_any_coordinate():
    actions = bounded_action_probes()
    assert len(actions) == 161
    assert len({sha256_json(row) for row in actions}) == 161
    for axis in range(4):
        assert {row.axes[axis] for row in actions} == {-1.0, -0.5, 0.0, 0.5, 1.0}
    assert all(row.mode == "pilot-control" for row in actions)


# 功能：
#   验证风险样本对应原始观测与不同物理动作，保留未执行标记，且落地证明失效后拒绝标注。
# 输入：
#   tmp_path：合成原生证据目录。
# 输出：
#   None：不返回业务数据。
def test_counterfactuals_bind_each_actual_observation_and_distinct_physical_request(tmp_path):
    oracle, observation, _, terminal = fixture(tmp_path)
    before = sha256_json(observation)
    actions = [PilotAction(mode="pilot-control", axes=[a, 0, 0, 0]) for a in (0.0, 0.5, 1.0)]
    rows = list(counterfactual_risk_samples(observation, actions, oracle.risk))
    assert len(rows) == 3
    assert sha256_json(observation) == before
    assert [row[0].risk_proposed_control[0] for row in rows] == [0.0, 0.01, 0.02]
    for sample, record in rows:
        assert record["label_sha256"] == sha256_json(sample)
        assert record["observation_sha256"] == before
        assert record["counterfactual_action_executed"] is False
        assert record["behavior_cloning_label"] is False
    terminal.write_text('{"terminal_state":"IN_AIR"}')
    with pytest.raises(ValueError, match="LANDING_NOT_CONFIRMED"):
        list(counterfactual_risk_samples(observation, actions, oracle.risk))


# 功能：
#   验证重复动作、非连续控制动作或缺少绑定评估结果时无法生成风险标签。
# 输入：
#   tmp_path：合成原生证据目录。
# 输出：
#   None：不返回业务数据。
def test_duplicate_nonvelocity_and_wrong_assessment_probes_are_rejected(tmp_path):
    oracle, observation, _, _ = fixture(tmp_path)
    action = PilotAction(mode="pilot-control", axes=[0.5, 0, 0, 0])
    with pytest.raises(ValueError, match="UNIQUE_PROBES"):
        list(counterfactual_risk_samples(observation, [action, action], oracle.risk))
    with pytest.raises(ValueError, match="UNIQUE_PROBES"):
        list(
            counterfactual_risk_samples(
                observation, [PilotAction(mode="hold", axes=[0.0] * 4)], oracle.risk
            )
        )
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        list(counterfactual_risk_samples(observation, [action], lambda *_: None))
