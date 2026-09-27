"""Exact visibility parity for the allocation-reduced box intersection kernel."""

import numpy as np
import pytest

from dronedream_agent_core.map_surface_geometry import _box_hits


# 功能：保存优化前逐张量求交作为独立测试参照，不用于生产选择或补全结果。
# 输入：有限原点、方向、半尺寸、边缘距离。
# 输出：原算法的首交点、法向、面内部性、面编号及内部起点。
def reference(origin, direction, half, margin):
    parallel = np.abs(direction) < 1e-12
    divisor = np.where(parallel, 1., direction)
    first, second = (-half - origin) / divisor, (half - origin) / divisor
    entry = np.where(parallel, -np.inf, np.minimum(first, second))
    exit = np.where(parallel, np.inf, np.maximum(first, second))
    enter, leave = entry.max(axis=-1), exit.min(axis=-1)
    hit = ((leave >= np.maximum(enter, 0.)) & (enter > 0.)
           & ~np.any(parallel & (np.abs(origin) > half), axis=-1))
    enter = np.where(hit, enter, np.inf)
    axis = np.argmax(entry, axis=-1)
    sign = -np.sign(np.take_along_axis(direction, axis[..., None], axis=-1)[..., 0])
    normals = np.eye(3)[axis] * sign[..., None]
    point = origin + direction * np.where(hit, enter, 0.)[..., None]
    gaps = half - np.abs(point)
    interior = np.min(np.where(np.arange(3) == axis[..., None], np.inf, gaps), axis=-1) > margin
    return enter, normals, interior, axis * 2 + (sign > 0), np.all(np.abs(origin) <= half, axis=-1)


# 功能：覆盖批量/单盒、近平行、边界、同值选面和内部起点，要求所有返回值逐元素一致。
# 输入：不同广播布局与固定随机样本。
# 输出：优化前后完全相同，不仅比较成功射线或最终位姿。
@pytest.mark.parametrize('shape', [(512, 128, 3), (123, 3), (3,)])
@pytest.mark.parametrize('margin', [0., .02, .2])
def test_box_kernel_preserves_all_intersection_outputs(shape, margin):
    rng = np.random.default_rng(827)
    origin = rng.uniform(-4, 4, size=shape)
    direction = rng.uniform(-1, 1, size=shape)
    half = np.ones(shape[-2:] if len(shape) == 3 else (3,))
    flat_d, flat_o = direction.reshape(-1, 3), origin.reshape(-1, 3)
    flat_d[::5, 0] = 0.
    flat_d[::7, 1] = 1e-14
    flat_d[::11] = [1., 1., 1.]
    flat_o[::11] = [-2., -2., -2.]
    flat_o[::13] = [1., 1., 1.]
    for actual, expected in zip(_box_hits(origin, direction, half, margin),
                                reference(origin, direction, half, margin), strict=True):
        np.testing.assert_array_equal(actual, expected)
