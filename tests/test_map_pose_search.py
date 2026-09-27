"""Multi-start cannot reset its reference, conceal ambiguity or grant flight."""

from dataclasses import replace

import numpy as np
import pytest
from test_local_pose_alignment import scene

from dronedream_agent_core.local_pose_alignment import fit_map_pose
from dronedream_agent_core.map_pose_search import search_map_pose


# 功能：验证粗初值不会变成新参考，最终修正仍包含全部平移。
# 输入：无。
# 输出：无。
def test_seed_keeps_original_reference_and_full_correction():
    points, index = scene()
    bias = np.array([0.04, -0.03, 0.02])
    fit = fit_map_pose(
        points + bias,
        index,
        sensor_origins_world_m=bias,
        reference_position_world_m=bias,
        initial_correction_world_m=(-0.02, 0.0, 0.0),
        initial_rotation_vector_world_rad=(0.0, 0.0, 0.02),
    )
    assert fit.usable_candidate
    np.testing.assert_allclose(fit.reference_position_world_m, bias)
    np.testing.assert_allclose(fit.correction_world_m, -bias, atol=1e-6)


# 功能：初值缺一项、超范围或类型非法时拒绝，不借初值放宽安全包络。
# 输入：kwargs：非法初值。
# 输出：无。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_correction_world_m": (0, 0, 0)},
        {"initial_rotation_vector_world_rad": (0, 0, 0)},
        {"initial_correction_world_m": (0.3, 0, 0), "initial_rotation_vector_world_rad": (0, 0, 0)},
        {"initial_correction_world_m": (0, 0, 0), "initial_rotation_vector_world_rad": (0, 0, 0.2)},
        {
            "initial_correction_world_m": (True, 0, 0),
            "initial_rotation_vector_world_rad": (0, 0, 0),
        },
    ],
)
def test_invalid_seed_rejected(kwargs):
    points, index = scene()
    with pytest.raises(ValueError):
        fit_map_pose(
            points,
            index,
            sensor_origins_world_m=[0, 0, 0],
            reference_position_world_m=[0, 0, 0],
            **kwargs,
        )


# 功能：对无歧义平面场景保留全部尝试，且不创建协方差或运动权限。
# 输入：无。
# 输出：无。
def test_consistent_search():
    points, index = scene()
    result = search_map_pose(
        points, index, sensor_origins_world_m=[0, 0, 0], reference_position_world_m=[0, 0, 0]
    )
    assert result.candidate is not None
    assert len(result.attempts) == 13
    assert not result.covariance_qualified and not result.motion_permission_granted


# 功能：即使低残差候选存在，只要另有可信但不一致的解，也不能选最好看的那个。
# 输入：monkeypatch：注入已求解但有歧义的分支。
# 输出：无。
@pytest.mark.parametrize("kind", ["position", "rank"])
def test_competing_solution_not_hidden(monkeypatch, kind):
    points, index = scene()
    base = fit_map_pose(
        points, index, sensor_origins_world_m=[0, 0, 0], reference_position_world_m=[0, 0, 0]
    )
    changed = (
        replace(base, correction_world_m=(0.05, 0.0, 0.0))
        if kind == "position"
        else replace(base, observed_translation_rank=2)
    )
    results = iter([base, changed] + [base] * 11)
    monkeypatch.setattr(
        "dronedream_agent_core.map_pose_search.fit_map_pose", lambda *a, **kw: next(results)
    )
    result = search_map_pose(
        points, index, sensor_origins_world_m=[0, 0, 0], reference_position_world_m=[0, 0, 0]
    )
    assert result.candidate is None
    assert result.issue == (
        "MAP_POSE_SEARCH_AMBIGUOUS"
        if kind == "position"
        else "MAP_POSE_SEARCH_OBSERVABILITY_UNSTABLE"
    )
