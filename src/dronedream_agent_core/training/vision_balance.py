"""Training-only group loss weights; never resplit, duplicate or relabel images."""

import math
from collections import Counter

from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample


# 功能：
#   限制单个空间组的累计损失权重，避免大量相邻帧压过其他地图；不改变验证和测试。
# 输入：
#   samples：筛选后仅训练划分的样本。
#   maximum_share：任一空间组允许占有的累计样本权重比例。
# 输出：
#   balanced、receipt：只调整 sample_weight 的新样本，以及原始和实际组权重。
def cap_training_group_weights(samples, maximum_share):
    if (isinstance(maximum_share, bool) or not isinstance(maximum_share, (float, int))
            or not math.isfinite(maximum_share) or not 0 < maximum_share <= 1
            or not 1 <= len(samples) <= 100_000):
        raise ValueError("VISION_BALANCE_POLICY_INVALID")
    masses, counts = Counter(), Counter()
    for sample in samples:
        if not isinstance(sample, LocalVisionTrainingSample) or not sample.scene_group_id:
            raise ValueError("VISION_BALANCE_EXPLICIT_SCENE_GROUP_REQUIRED")
        masses[sample.scene_group_id] += sample.sample_weight
        counts[sample.scene_group_id] += 1
    if maximum_share * len(masses) < 1 - 1e-12:
        raise ValueError("VISION_BALANCE_TOO_FEW_GROUPS_FOR_SHARE_CAP")
    total = sum(masses.values())
    ceiling = max(masses.values())
    if ceiling / total > maximum_share + 1e-12:
        # 固定 80 步二分求共享上限；只下调过量来源，不复制稀少样本。
        # 当要求恰好均分时，最小组的原权重就是可行且不放大的上限。
        low, high = min(masses.values()), ceiling
        for _ in range(80):
            middle = (low + high) / 2
            share = middle / sum(min(value, middle) for value in masses.values())
            if share <= maximum_share:
                low = middle
            else:
                high = middle
        ceiling = low
    factors = {group: min(1.0, ceiling / mass) for group, mass in masses.items()}
    balanced = [LocalVisionTrainingSample.model_validate({**sample.model_dump(mode="json"),
        "sample_weight": sample.sample_weight * factors[sample.scene_group_id]})
        for sample in samples]
    actual = {group: mass * factors[group] for group, mass in masses.items()}
    actual_total = sum(actual.values())
    if any(mass / actual_total > maximum_share + 1e-10 for mass in actual.values()):
        raise ValueError("VISION_BALANCE_SHARE_INVARIANT_FAILED")
    receipt = {"schema": "dronedream.vision-group-loss-balance.v1",
        "maximum_group_weight_share": maximum_share,
        "sample_count": len(samples), "group_count": len(masses),
        "groups": {group: {"sample_count": counts[group], "original_weight": masses[group],
            "weight_multiplier": factors[group], "balanced_weight": actual[group],
            "balanced_weight_share": actual[group] / actual_total} for group in sorted(masses)},
        "image_sampling_order_changed": False, "validation_or_test_modified": False,
        "model_improvement_measured": False}
    return balanced, receipt
