"""Real registration outputs must satisfy strict in-process and JSON audit consumers."""
import dataclasses
import json

import numpy as np
import pytest

from dronedream_agent_core.local_map_alignment import fit_map_translation
from dronedream_agent_core.local_pose_alignment import fit_map_pose
from dronedream_agent_core.localization_truth_audit import _numeric_array
from test_local_pose_alignment import scene


# 功能：真实几何求解结果在序列化前后均可严格消费，避免 NumPy 标量触发类型合同错误。
# 输入：joint：选择平移或联合姿态求解；planes：覆盖满秩及退化场景。
# 输出：无；本测试不授予定位或飞行资格。
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("planes", [1, 2, 3])
def test_real_fit_has_plain_numeric_contract(joint, planes):
    points, index = scene(planes)
    bias = np.array([.04, -.03, .02])
    fit = (fit_map_pose(points + bias, index, sensor_origins_world_m=bias,
                        reference_position_world_m=bias) if joint else
           fit_map_translation(points + bias, index, sensor_origins_world_m=bias))
    assert fit.usable_candidate
    raw = dataclasses.asdict(fit)
    for candidate in (raw, json.loads(json.dumps(raw, allow_nan=False))):
        value = _numeric_array(candidate["correction_world_m"], (3,), "INVALID")
        np.testing.assert_allclose(value, fit.correction_world_m)
        if joint:
            _numeric_array(candidate["rotation_world_from_input"], (3, 3), "INVALID")
    assert all(type(v) is float for v in fit.correction_world_m)
