"""Shared collection bounds, policy identity and held-out task isolation."""


# 功能：
#   在启动环境前检查采集规模和种子，所有入口统一采用有界的单次采集。
# 输入：
#   steps：本次最多记录的步数。
#   seed：非负整数随机种子。
# 输出：
#   None：不返回业务数据。
def validate_collection_size(*, steps, seed):
    # 更长训练分成多次各自安全停止的采集，不能在一次飞行中无限保存列表。
    if type(steps) is not int or not 1 <= steps <= 256 or type(seed) is not int or seed < 0:
        raise ValueError("STUDENT_COLLECTION_CONFIGURATION_INVALID")


# 功能：
#   检查采集规模、种子和必须绑定的策略身份，不以缺失摘要默认为某个模型。
# 输入：
#   steps：本次最多记录的步数。
#   seed：非负整数随机种子。
#   policy_sha256：六十四位小写十六进制策略摘要。
# 输出：
#   None：不返回业务数据。
def validate_collection_arguments(*, steps, seed, policy_sha256):
    validate_collection_size(steps=steps, seed=seed)
    if (not isinstance(policy_sha256, str) or len(policy_sha256) != 64
            or any(c not in "0123456789abcdef" for c in policy_sha256)):
        raise ValueError("STUDENT_COLLECTION_POLICY_IDENTITY_INVALID")


# 功能：
#   校验并固定留出任务集合，避免环境或模型回调改变本轮训练隔离边界。
# 输入：
#   missions：文本任务标识组成的集合或冻结集合。
# 输出：
#   result：独立且不可变的留出任务集合。
def frozen_held_out_missions(missions):
    if type(missions) not in (set, frozenset) or any(
        type(mission) is not str or not mission for mission in missions
    ):
        raise ValueError("DAGGER_HELD_OUT_MISSIONS_INVALID")
    result = frozenset(missions)
    return result
