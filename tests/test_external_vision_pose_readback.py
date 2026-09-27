"""Non-finite or ambiguous transport echoes must never certify delivery."""
import pytest

from dronedream_agent_core.external_vision_clock import validate_external_vision_pose_readback


# 功能：覆盖有效打印精度及非有限、重复、缺失和畸形回读，保留所有错误。
# 输入：readback：构造的飞控文本；valid：是否为唯一、有限且正确的回读。
# 输出：无；只有有效回读成功。
@pytest.mark.parametrize("readback,valid", [
    ("position: [0.10000, -0.20000, 0.30000]\n q: [1.00000, 0, 0, 0]", True),
    ("position: [nan, -0.2, 0.3]\nq: [1, 0, 0, 0]", False),
    ("position: [inf, -0.2, 0.3]\nq: [1, 0, 0, 0]", False),
    ("position: [0.1, -0.2, 0.3]\nq: [nan, 0, 0, 0]", False),
    ("position: [0.1, -0.2, 0.3]\nq: [1, 0, 0, 0]\nq: [1, 0, 0, 0]", False),
    ("position: [0.1, -0.2, 0.3]\nq: [1, 0, 0]", False),
    ("position: [0.1, -0.2, 0.3]\nq: [true, 0, 0, 0]", False),
    ("position: [0.1, -0.2, 0.3]\nq: [1, 0, 0, 0.001]", False),
    ("position: [0.101, -0.2, 0.3]\nq: [1, 0, 0, 0]", False),
    ("q: [1, 0, 0, 0]", False),
    ("x"*32769, False),
    (None, False),
], ids=["finite", "nan-position", "inf-position", "nan-quaternion", "duplicate",
        "short-vector", "boolean-text", "quaternion-error", "position-error",
        "missing-position", "oversize", "not-text"])
def test_pose_readback_rejects_false_success(readback, valid):
    args = dict(readback=readback, position=(.1,-.2,.3), quaternion=(1.,0.,0.,0.))
    if valid:
        validate_external_vision_pose_readback(**args)
    else:
        with pytest.raises(ValueError, match="EXTERNAL_VISION"):
            validate_external_vision_pose_readback(**args)


@pytest.mark.parametrize("position,quaternion", [
    ((True,0.,0.), (1.,0.,0.,0.)),
    ((float('nan'),0.,0.), (1.,0.,0.,0.)),
    ((0.,0.,0.), (0.,0.,0.,0.)),
    ((0.,0.,0.), (1.,False,0.,0.)),
    ((0.,0.), (1.,0.,0.,0.)),
])
def test_expected_pose_cannot_bypass_validation(position, quaternion):
    with pytest.raises(ValueError, match="EXPECTED_POSE_INVALID"):
        validate_external_vision_pose_readback(
            readback="position: [0, 0, 0]\nq: [1, 0, 0, 0]",
            position=position, quaternion=quaternion)
