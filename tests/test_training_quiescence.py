"""Lifecycle contract tests use synthetic, non-physical environments."""

from types import SimpleNamespace

import pytest
from test_offline_flight_learning import UnitEnvironment, base, replay

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.dagger import TeacherCorrection, collect_dagger
from dronedream_agent_core.training.flight_environment import (
    PilotAction,
    quiesced_simulation_rollout,
)
from dronedream_agent_core.training.ppo import PPOConfig, PPOTrainer


class QuiescenceFixture(UnitEnvironment):
    def __init__(self):
        self.stops = 0
        self.resets = 0
        self.fail_stop = False

    def reset(self, *, seed):
        self.resets += 1
        return super().reset(seed=seed)

    def quiesce(self):
        self.stops += 1
        if self.fail_stop:
            raise RuntimeError("fixture stop failed")


def test_physical_interface_missing_stop_is_rejected_before_entering_rollout():
    env = SimpleNamespace(simulation_only=True, evidence_kind="px4-gazebo")
    with pytest.raises(ValueError, match="QUIESCE_REQUIRED"), quiesced_simulation_rollout(env):
        pytest.fail("must not arm or start collecting")


def test_ppo_success_and_exception_cannot_return_with_active_training_control():
    learner = PPOTrainer(base(), replay(), PPOConfig(rollout_steps=2))
    env = QuiescenceFixture()
    assert learner.collect(env).records
    assert env.stops == 1
    env.fail_stop = True
    with pytest.raises(RuntimeError, match="stop failed"):
        learner.collect(env)
    assert env.stops == 2
    env.fail_stop = False
    env.step = lambda _: (_ for _ in ()).throw(ValueError("fixture sensor failed"))
    with pytest.raises(ValueError, match="sensor failed"):
        learner.collect(env)
    assert env.stops == 3


def test_dagger_always_quiesces_and_does_not_reset_an_unused_final_episode():
    env = QuiescenceFixture()
    original_step = env.step

    def terminal(action):
        return original_step(action).model_copy(update={"truncated": True})

    env.step = terminal
    hold = PilotAction(mode="hold", axes=[0.] * 4)

    def teacher(observation):
        return TeacherCorrection(observation_sha256=sha256_json(observation), action=hold,
                                 verified_action_risk=0., verifier_receipt_sha256="f" * 64)

    kwargs = dict(seed=1, steps=2, teacher_probability=1., held_out_missions=set())
    result = collect_dagger(env, lambda _: hold, teacher, lambda *_: None, **kwargs)
    assert len(result.records) == 2
    assert env.resets == 2  # No unnecessary third launch at the final boundary.
    assert env.stops == 1
    kwargs["held_out_missions"] = {"unit-mission"}
    with pytest.raises(ValueError, match="HELD_OUT"):
        collect_dagger(env, lambda _: hold, teacher, lambda *_: None, **kwargs)
    assert env.stops == 2
