import numpy as np
import pytest
from test_offline_flight_learning import UnitEnvironment, replay

from dronedream_agent_core.local_policy_training import (
    LocalPolicyObservation,
    LocalPolicyTrainingSample,
    _arrays,
    policy_input_arrays,
)
from dronedream_agent_core.training.flight_environment import (
    FlightObservation,
    FlightReward,
    PilotAction,
    RewardConfig,
)


# 功能：
#   从带标签的样本中只投影实际可观测字段，构造供在线策略使用的无监督标签观测。
# 输入：
#   sample：带离线监督信息的测试样本。
# 输出：
#   result：按观测契约重新验证的传感器输入。
def observation(sample):
    result = LocalPolicyObservation.model_validate(
        {name: getattr(sample, name) for name in LocalPolicyObservation.model_fields}
    )
    return result


# 功能：
#   验证在线观测不需要虚构风险或动作标签，并与离线训练使用相同特征顺序。
# 输入：
#   无：使用合成回放样本与真实张量编译函数。
# 输出：
#   None：不返回业务数据。
def test_live_policy_features_need_no_fabricated_labels_and_match_training_order():
    labeled = replay()
    live = [observation(sample) for sample in labeled]
    np.testing.assert_array_equal(policy_input_arrays(live), _arrays(labeled).inputs)
    assert "risk_target" not in live[0].model_dump()
    assert "target_action_index" not in live[0].model_dump()
    with pytest.raises(ValueError):
        LocalPolicyTrainingSample.model_validate(live[0].model_dump())
    with pytest.raises(ValueError, match="ASSESSED_PROPOSAL"):
        policy_input_arrays(live, include_proposed_control=True)
    wrapped = FlightObservation(mission_id="m", episode_id="e", map_sha256="a" * 64,
                                sequence=0, sample=live[0])
    np.testing.assert_array_equal(wrapped.policy_input(), _arrays(labeled).inputs[0])


# 功能：
#   验证在线观测拒绝非有限特征及长度不匹配的有效性掩码。
# 输入：
#   无：测试自行破坏一份合成观测。
# 输出：
#   None：不返回业务数据。
def test_live_observation_rejects_nonfinite_or_malformed_features():
    payload = observation(replay()[0]).model_dump()
    payload["state_features"][0] = float("nan")
    with pytest.raises(ValueError):
        LocalPolicyObservation.model_validate(payload)
    payload["state_features"][0] = 0.
    payload["realtime_valid_mask"].pop()
    with pytest.raises(ValueError):
        LocalPolicyObservation.model_validate(payload)


# 功能：
#   验证缺少实测耗电不能视为零耗电或消耗奖励事件，只有关闭能量项时才允许未知值。
# 输入：
#   无：使用受控环境构造缺少能量测量的步骤。
# 输出：
#   None：不返回业务数据。
def test_missing_battery_current_is_unknown_energy_not_zero_consumption():
    env = UnitEnvironment()
    env.reset(seed=3)
    step = env.step(PilotAction(mode="hold", axes=[0.] * 4))
    step = step.model_copy(update={"evidence": step.evidence.model_copy(
        update={"power_joules": None})})
    reward = FlightReward()
    with pytest.raises(ValueError, match="MEASURED_CONSUMPTION"):
        reward.score(step)
    assert reward.stage is None  # Failed validation cannot consume rewards/events.
    assert FlightReward(RewardConfig(energy_per_joule=0)).score(step)["energy"] == 0


# 功能：
#   验证因果历史只包含当前时刻之前独立、有序且不带标签的传感器观测。
# 输入：
#   无：测试自行构造四个连续观测。
# 输出：
#   None：不返回业务数据。
def test_live_causal_history_contains_only_prior_independent_sensors():
    from test_causal_policy import samples

    rows = [observation(s) for s in samples()[:4]]
    args = dict(mission_id="m", episode_id="e", map_sha256="a" * 64,
                sequence=0, sample=rows[-1])
    packet = FlightObservation(**args, prior_observations=rows[:-1])
    assert len(packet.prior_observations) == 3
    assert all(not hasattr(p, "target_action_index") for p in packet.prior_observations)
    for invalid in ([rows[-1]], [rows[0], rows[0]], [rows[1], rows[0]]):
        with pytest.raises(ValueError, match="PRIOR_OBSERVATION_IDENTITY"):
            FlightObservation(**args, prior_observations=invalid)


# 功能：
#   构造当前连续操纵模式的快照与真实编译观测，使用固定时钟供生命周期边界测试。
# 输入：
#   无：使用合成完整传感器快照及固定操纵限额。
# 输出：
#   request：有效至指定时刻且绑定快照内容的训练请求。
def current_request():
    from control_fixtures import complete_feature_snapshot
    from test_local_policy_packages import _snapshot

    from dronedream_agent_core.hashing import sha256_json
    from dronedream_agent_core.training.observations import compile_training_observation

    snapshot = _snapshot()
    snapshot.pop("snapshot_sha256")
    snapshot["authorized_candidate_paths"] = []
    snapshot["control_reference_observed_at_unix_ms"] = 1000
    snapshot["realtime_feature_snapshot"] = complete_feature_snapshot().model_dump(mode="json")
    snapshot["strategic_context"] = {"task": {
        "control_session_id": "session", "control_profile": "cruise",
        "local_navigation_output_mode": "normalized-body-velocity",
        "normalized_pilot_control_limits": {"horizontal_speed_mps": .4,
                                            "vertical_speed_mps": .4, "yaw_rate_dps": 20.},
    }}
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    sample = compile_training_observation(snapshot, now_unix_ms=1000)
    request = {"snapshot": snapshot, "observation": sample.model_dump(mode="json"),
            "valid_until_unix_ms": 1250, "multimodal": []}
    return request


# 功能：
#   验证准备输入只编译一次，但每次使用重查原请求和期限，并返回不能修改缓存的副本。
# 输入：
#   monkeypatch：统计编译次数的夹具。
# 输出：
#   None：不返回业务数据。
def test_prepared_input_compiles_once_but_rechecks_identity_and_source_expiry(monkeypatch):
    from dronedream_agent_core.training import observations as module

    request = current_request()
    original = module.compile_training_input
    calls = []

    # 功能：
    #   记录真实观测编译器的调用参数，原样委托业务实现。
    # 输入：
    #   args：编译入口的位置参数。
    #   kwargs：编译入口的关键字参数。
    # 输出：
    #   result：真实编译器生成的观测。
    def counted(*args, **kwargs):
        calls.append(kwargs)
        result = original(*args, **kwargs)
        return result

    monkeypatch.setattr(module, "compile_training_input", counted)
    prepared = module.PreparedTrainingInput.from_request(request)
    admitted = prepared.admit(request, now_unix_ms=1100)
    assert len(calls) == 1
    admitted.state_features[0] = 999.
    assert prepared.admit(request, now_unix_ms=1101).state_features[0] != 999.
    with pytest.raises(ValueError, match="SENSOR_EVIDENCE_EXPIRED"):
        prepared.admit(request, now_unix_ms=1251)
    request["snapshot"]["goal_position_m"]["x"] += 1.
    with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
        prepared.admit(request, now_unix_ms=1102)


# 功能：
#   验证请求附带的特征必须与真实快照编译结果一致，不能通过外部声明替换实际传感器值。
# 输入：
#   无：测试自行改写一份请求的特征值。
# 输出：
#   None：不返回业务数据。
def test_prepared_input_rejects_claimed_features_not_in_actual_snapshot():
    from dronedream_agent_core.training.observations import PreparedTrainingInput

    request = current_request()
    request["observation"]["state_features"][0] += .01
    with pytest.raises(ValueError, match="FEATURES_DIFFER"):
        PreparedTrainingInput.from_request(request)
