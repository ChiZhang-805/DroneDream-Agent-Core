from itertools import chain, repeat
from types import SimpleNamespace

import pytest
from test_deferred_student_collection import Lifecycle
from test_stream_collection import StreamLifecycle

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training import stream_collection
from dronedream_agent_core.training.dagger import TeacherCorrection
from dronedream_agent_core.training.flight_environment import PilotAction
from dronedream_agent_core.training.stream_capture import StreamControlVisit
from dronedream_agent_core.training.student_collection import (
    collect_student_visits,
    label_student_visits,
)


# 功能：
#   生成固定合成四轴动作，仅供调度与所有权测试，不连接飞控。
# 输入：
#   observation：模拟环境当前观测。
# 输出：
#   proposal：前向幅度零点二、其他轴为零的动作。
def actor(observation):
    proposal = PilotAction(mode="pilot-control", axes=[0.2, 0., 0., 0.])
    return proposal


# 功能：
#   验证显式终止提案结束当前采集，不为填满步数继续控制或重置回合。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_deferred_abort_stops_collection_even_without_terminal_step():
    env = Lifecycle()
    visits = collect_student_visits(
        env, lambda _: PilotAction(mode="abort", axes=[0.] * 4),
        seed=1, steps=8, held_out_missions=set(), student_policy_sha256="a" * 64,
    )
    assert env.grounded and env.resets == 1
    assert len(visits) == env.sequence == 1


# 功能：
#   验证环境回调清空外部留出集合后，采集器仍拒绝原本禁止训练的任务。
# 输入：
#   streaming：是否使用连续控制流入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("streaming", [False, True])
def test_collection_owns_held_out_missions_before_reset(streaming):
    env = StreamLifecycle() if streaming else Lifecycle()
    held_out = {"unit-mission"}
    reset = env.reset

    # 功能：
    #   在环境重置回调中修改调用方集合，复现外部配置的别名问题。
    # 输入：
    #   seed：本轮随机种子。
    # 输出：
    #   observation：合成环境的初始观测。
    def change_held_out(*, seed):
        held_out.clear()
        observation = reset(seed=seed)
        return observation

    env.reset = change_held_out
    collect = (stream_collection.collect_native_control_stream if streaming
               else collect_student_visits)
    with pytest.raises(ValueError, match="HELD_OUT"):
        collect(env, actor, seed=1, steps=2, held_out_missions=held_out,
                student_policy_sha256="a" * 64)
    assert env.grounded


# 功能：
#   验证非法回合步数在绑定模型或重置环境前被拒绝，不能空飞后正常返回。
# 输入：
#   steps：环境配置中错误的回合步数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("steps", [0, -1, True, 1.5, "3", 257])
def test_invalid_episode_limit_is_rejected_before_reset(steps):
    env = StreamLifecycle()
    env.config = SimpleNamespace(episode_steps=steps)
    with pytest.raises(ValueError, match="EPISODE_STEPS_INVALID"):
        stream_collection.collect_native_control_stream(
            env, actor, seed=1, steps=2, held_out_missions=set(), student_policy_sha256="a" * 64,
        )
    assert env.events == [] and env.grounded


# 功能：
#   验证无效单调时钟不能绕过实时采集期限，也不能提交动作。
# 输入：
#   monkeypatch：替换采集模块时钟的工具。
#   now：非法时钟值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now", [float("nan"), float("inf"), -1., True, 10**400])
def test_stream_invalid_clock_never_submits(monkeypatch, now):
    monkeypatch.setattr(stream_collection, "time", SimpleNamespace(monotonic=lambda: now))
    env = StreamLifecycle()
    with pytest.raises(ValueError, match="COLLECTION_CLOCK_INVALID"):
        stream_collection.collect_native_control_stream(
            env, actor, seed=1, steps=2, held_out_missions=set(), student_policy_sha256="a" * 64,
        )
    assert "submit-without-wait" not in env.events and env.grounded


# 功能：
#   验证回退的时钟不增加控制机会或延长进度期限。
# 输入：
#   monkeypatch：时钟替换器。
# 输出：
#   None：不返回业务数据。
def test_stream_clock_cannot_move_backwards(monkeypatch):
    times = chain([10.], repeat(9.))
    monkeypatch.setattr(stream_collection, "time", SimpleNamespace(monotonic=lambda: next(times)))
    env = StreamLifecycle()
    with pytest.raises(ValueError, match="COLLECTION_CLOCK_INVALID"):
        stream_collection.collect_native_control_stream(
            env, actor, seed=1, steps=2, held_out_missions=set(), student_policy_sha256="a" * 64,
        )
    assert "submit-without-wait" not in env.events and env.grounded


# 功能：
#   验证离线教师修改外部原始访问记录时，不会改变已固定的训练动作与批次大小。
# 输入：
#   streaming：是否转换为无奖励的连续控制记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("streaming", [False, True])
def test_labels_hold_independent_visits_before_callbacks(streaming):
    visits = collect_student_visits(Lifecycle(), actor, seed=1, steps=2,
                                   held_out_missions=set(), student_policy_sha256="a" * 64)
    if streaming:
        visits = [StreamControlVisit(
            v.observation, v.proposal, v.step.applied_action, "b" * 64, "c" * 64, "d" * 64, False,
        ) for v in visits]

    # 功能：
    #   返回合法教师标签，同时篡改外部访问列表以检验标注输入的所有权。
    # 输入：
    #   observation：标注器独立交给教师的观测。
    # 输出：
    #   correction：与该观测绑定的保持动作标签。
    def teacher(observation):
        if visits:
            visits[0].proposal.axes[0] = 0.9
            visits.clear()
        correction = TeacherCorrection(
            observation_sha256=sha256_json(observation),
            action=PilotAction(mode="hold", axes=[0.] * 4),
            verified_action_risk=0., verifier_receipt_sha256="f" * 64,
        )
        return correction

    if streaming:
        result = stream_collection.label_stream_visits(
            visits, teacher, lambda *_: None, held_out_missions=set(),
        )
    else:
        result = label_student_visits(
            visits, teacher, lambda *_: None, held_out_missions=set(), evidence_kind="unit-test",
        )
    assert len(result.records) == 2
    assert all(row["student_proposal"]["axes"][0] == 0.2 for row in result.records)
