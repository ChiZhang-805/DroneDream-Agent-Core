import pytest
from test_deferred_student_collection import Lifecycle

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.dagger import TeacherCorrection, collect_dagger
from dronedream_agent_core.training.flight_environment import PilotAction


# 功能：
#   生成与当前观测绑定的合成终止纠正，不产生真实飞控指令。
# 输入：
#   observation：当前合成观测。
# 输出：
#   correction：零轴终止动作的教师回执。
def abort_teacher(observation):
    correction = TeacherCorrection(
        observation_sha256=sha256_json(observation),
        action=PilotAction(mode="abort", axes=[0.] * 4),
        verified_action_risk=0., verifier_receipt_sha256="f" * 64,
    )
    return correction


# 功能：
#   验证同步混合采集实际选中终止动作后也停止，不为填满步数重启。
# 输入：
#   teacher_probability：选择教师提案的概率，零和一分别覆盖学生及教师终止。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("teacher_probability", [0., 1.])
def test_selected_abort_ends_synchronous_collection(teacher_probability):
    env = Lifecycle()
    result = collect_dagger(
        env, lambda _: PilotAction(mode="abort", axes=[0.] * 4), abort_teacher,
        lambda *_: None, seed=1, steps=8, teacher_probability=teacher_probability,
        held_out_missions=set(),
    )
    assert len(result.records) == 1 and env.resets == 1 and env.grounded


# 功能：
#   验证后续教师改动其持有的原始学生动作时，不能改写已固定的提案与实际执行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_teacher_cannot_change_student_proposal_through_shared_return_value():
    env = Lifecycle()
    proposed = PilotAction(mode="pilot-control", axes=[0.2, 0., 0., 0.])

    # 功能：
    #   模拟教师拥有外部学生动作引用并修改该对象的情况。
    # 输入：
    #   observation：教师收到的独立观测。
    # 输出：
    #   correction：合法的终止纠正回执，本例不选择执行它。
    def teacher(observation):
        proposed.axes[0] = 0.9
        correction = abort_teacher(observation)
        return correction

    result = collect_dagger(
        env, lambda _: proposed, teacher, lambda *_: None, seed=1, steps=1,
        teacher_probability=0., held_out_missions=set(),
    )
    assert result.records[0]["student_proposal"]["axes"][0] == 0.2
    assert result.records[0]["applied_action"]["axes"][0] == 0.2


# 功能：
#   验证同步入口同样固定留出集合，重置回调不能取消任务隔离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_synchronous_collection_freezes_holdout_before_environment_callback():
    env, held_out = Lifecycle(), {"unit-mission"}
    reset = env.reset

    # 功能：
    #   清空外部配置后产生属于原留出集合的任务，复现配置别名问题。
    # 输入：
    #   seed：合成回合种子。
    # 输出：
    #   observation：本应被隔离的任务观测。
    def resetting(*, seed):
        held_out.clear()
        observation = reset(seed=seed)
        return observation

    env.reset = resetting
    with pytest.raises(ValueError, match="HELD_OUT"):
        collect_dagger(env, lambda _: PilotAction(mode="hold", axes=[0.] * 4),
                       abort_teacher, lambda *_: None, seed=1, steps=1,
                       teacher_probability=0., held_out_missions=held_out)
    assert env.sequence == 0 and env.grounded


# 功能：
#   验证负数、布尔和非整数种子在环境开始之前被拒绝。
# 输入：
#   seed：非法随机种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("seed", [-1, True, 1.5, "1"])
def test_synchronous_collection_rejects_bad_seed_before_reset(seed):
    env = Lifecycle()
    with pytest.raises(ValueError, match="COLLECTION_CONFIGURATION_INVALID"):
        collect_dagger(env, lambda _: PilotAction(mode="hold", axes=[0.] * 4),
                       abort_teacher, lambda *_: None, seed=seed, steps=1,
                       teacher_probability=0., held_out_missions=set())
    assert env.resets == 0 and env.grounded
