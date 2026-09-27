"""Explicit stage collection preserves real specialist identity; no flight launched."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_px4_environment_boundaries import adopting_environment

from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.px4_environment import ASSET_FIELDS, Px4TrainingConfig


# 功能：验证多专家配置只用于显式流式采集；输入：合成路径及配置组合；输出：约束断言。
@pytest.mark.parametrize(
    "mode,roles,accepted",
    [
        (None, [], True),
        ("stream-imitation", ["local-navigation-policy", "recovery-policy"], True),
        (None, ["local-navigation-policy", "recovery-policy"], False),
        ("reward-step", ["local-navigation-policy", "recovery-policy"], False),
        ("stream-imitation", ["recovery-policy"], False),
        ("stream-imitation", ["local-navigation-policy", "local-navigation-policy"], False),
    ],
)
def test_config_limits(mode, roles, accepted):
    fields = dict(
        runner="runner",
        output_root="runs",
        mission_id="synthetic",
        expert_role="local-navigation-policy",
        asset_sha256={},
        minimum_enu_m=(0, 0, 0),
        maximum_enu_m=(1, 1, 1),
        initial_collection_mode=mode,
        collection_expert_roles=roles,
        **{name: "asset" for name in ASSET_FIELDS},
    )
    if accepted:
        assert Px4TrainingConfig(**fields).collection_expert_roles == roles
    else:
        with pytest.raises(ValueError, match="EXPLICIT_UNIQUE_STREAM_ROLES"):
            Px4TrainingConfig(**fields)


# 功能：真实请求角色必须保留且不能混入另一专家历史；输入：准备对象夹具；输出：发送身份断言。
def test_explicit_roles_preserve_request_identity_and_reset_history(tmp_path, monkeypatch):
    env, request = adopting_environment(tmp_path, monkeypatch)
    requested = env.config.expert_role
    other = "recovery-policy" if requested != "recovery-policy" else "local-navigation-policy"
    env.config.expert_role = other
    env.config.collection_expert_roles = [requested, other]
    env._collection_mode = "stream-imitation"
    env._source_history.append(SimpleNamespace(navigation_expert_role=other))
    result = env._adopt(request)
    assert result.sample.navigation_expert_role == requested
    assert not result.prior_observations
    env._quiesced = False
    env._transition_writer = SimpleNamespace(check=Mock())
    env._exchange = SimpleNamespace(reply=Mock())
    _, _, proposal = env._submit_action(PilotAction(mode="pilot-control", axes=[0.0] * 4))
    assert proposal.expert_role == requested != env.config.expert_role
    assert env._exchange.reply.call_args.args[0].expert_role == requested
    with pytest.raises(ValueError, match="CANNOT_PRODUCE_REWARD_STEPS"):
        env._select_collection_mode("reward-step")


# 功能：无显式授权或非流式模式继续拒绝专家切换；输入：不匹配请求；输出：保持单专家训练边界。
@pytest.mark.parametrize("explicit,mode", [(False, "stream-imitation"), (True, "reward-step")])
def test_unapproved_role_switch_still_stops(tmp_path, monkeypatch, explicit, mode):
    env, request = adopting_environment(tmp_path, monkeypatch)
    requested = env.config.expert_role
    env.config.expert_role = (
        "recovery-policy" if requested != "recovery-policy" else "local-navigation-policy"
    )
    env.config.collection_expert_roles = [requested, env.config.expert_role] if explicit else []
    env._collection_mode = mode
    env._exchange = SimpleNamespace(discard_pending=Mock())
    with pytest.raises(ValueError, match="ROLE_SPECIFIC_EPISODE"):
        env._adopt(request)
    env._exchange.discard_pending.assert_called_once()
