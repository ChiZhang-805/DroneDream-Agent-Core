"""A large map cannot dominate the declared loss objective just by emitting more frames."""

import pytest

from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.training.vision_balance import cap_training_group_weights


# 功能：
#   建立仅用于组权重计算的样本，不作为真实采集或训练资格证据。
# 输入：
#   group、weight：空间组和原样本权重。
# 输出：
#   sample：具有合法格式但不读取图像的测试样本。
def example(group, weight=1.0):
    sample = LocalVisionTrainingSample(flight_id="test-" + group, source_kind="rendered-view",
        scene_group_id=group, map_sha256="a" * 64, image_relative_path="rgb.png",
        image_sha256="b" * 64, source_record_sha256="c" * 64, sample_weight=weight,
        semantic_mask_relative_path="mask.png", semantic_mask_sha256="d" * 64,
        traversability_target=0.5, scene_targets=[0.0] * 6, quality_targets=[0.0] * 4)
    return sample


# 功能：
#   验证大量同图帧只降低训练损失权重，输入、标签和来源信息均保持不变。
# 输入：
#   无。
# 输出：
#   None：单组不得超过声明的四分之一权重，原样本不可被原地修改。
def test_dominant_map_is_capped_without_changing_observations_or_sources():
    samples = [example("school") for _ in range(1000)]
    samples += [example(f"independent-{index}") for index in range(20)]
    balanced, receipt = cap_training_group_weights(samples, 0.25)
    assert receipt["groups"]["school"]["balanced_weight_share"] == pytest.approx(0.25)
    assert all(sample.sample_weight == 1 for sample in samples)
    for old, new in zip(samples, balanced, strict=True):
        assert (old.model_dump(exclude={"sample_weight"})
                == new.model_dump(exclude={"sample_weight"}))
        assert 0 < new.sample_weight <= old.sample_weight
    assert receipt["validation_or_test_modified"] is False


# 功能：
#   多个过量来源和已有样本权重都按同一约束处理，不把权重上限误算成帧数上限。
# 输入：
#   weights、share：测试组原权重和期望限制。
# 输出：
#   None：全部组满足限制，并保持可行时不放大任何样本。
@pytest.mark.parametrize("weights,share", [([100, 80, 2, 1, 1], .25), ([10, 2, 1, 1], .25),
                                          ([1, 1, 1, 1], .25), ([1, 2], 1.0)])
def test_multiple_dominant_groups_and_equal_share_boundary(weights, share):
    balanced, report = cap_training_group_weights(
        [example(str(index), weight) for index, weight in enumerate(weights)], share)
    assert all(group["balanced_weight_share"] <= share + 1e-10
               for group in report["groups"].values())
    assert all(0 < sample.sample_weight <= original
               for sample, original in zip(balanced, weights, strict=True))


# 功能：
#   不可行的组数、非法上限及缺少显式组身份必须拒绝，不能为凑均衡伪造分组。
# 输入：
#   share：非法或无法满足的权重上限。
# 输出：
#   None：所有不合法政策都在改写任何样本前失败。
@pytest.mark.parametrize("share", [True, 0, -1, 1.1, float("nan"), float("inf"), .2])
def test_invalid_or_infeasible_policy_is_rejected(share):
    with pytest.raises(ValueError, match="VISION_BALANCE_"):
        cap_training_group_weights([example("one"), example("two")], share)
