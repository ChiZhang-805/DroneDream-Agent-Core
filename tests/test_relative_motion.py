import math
import random
from decimal import Decimal, localcontext

import pytest

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.relative_motion import (
    closest_center_distance_m,
    cylinder_contact_seconds,
)


# 功能：
#   验证远距离细小圆柱的首次接触时间不被二次方程相减消去误差改变。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_small_radius_at_large_distance_keeps_first_surface_contact():
    contact = cylinder_contact_seconds(
        Vector3(x=1.0e9, y=0, z=0),
        Vector3(x=-1, y=0, z=0),
        radius_m=1.0,
        height_m=2.0,
        horizon_seconds=2.0e9,
    )
    assert contact == pytest.approx(999_999_999.0, rel=0.0, abs=1.0e-6)


# 功能：
#   验证大但有限的相对位置与速度不会因平方溢出返回立即接触或错误最近距离。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_large_finite_motion_remains_geometrically_meaningful():
    p, v = Vector3(x=1.0e200, y=0, z=0), Vector3(x=-1.0e200, y=0, z=0)
    assert cylinder_contact_seconds(p, v, radius_m=1.0e199, height_m=2.0) == pytest.approx(0.9)
    assert closest_center_distance_m(p, v) == 0.0


# 功能：
#   验证低速仍能在预测窗口内触碰近表面，不能用固定速度阈值误认为完全静止。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_very_slow_horizontal_motion_is_not_discarded():
    p, v = Vector3(x=1.0000001, y=0, z=0), Vector3(x=-1.0e-7, y=0, z=0)
    assert cylinder_contact_seconds(p, v, radius_m=1.0, height_m=2.0) == pytest.approx(1.0)
    assert closest_center_distance_m(p, v, horizon_seconds=2.0) < p.x


# 功能：
#   验证竖直低速接触也使用实际相对运动，不把近表面缓慢下降当成永不相交。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_very_slow_vertical_motion_retains_contact():
    p, v = Vector3(x=0, y=0, z=1.00000000001), Vector3(x=0, y=0, z=-1.0e-12)
    expected = (p.z - 1.0) / -v.z
    assert cylinder_contact_seconds(p, v, radius_m=1.0, height_m=2.0) == pytest.approx(expected)


# 功能：
#   验证两种相对运动特征均拒绝非法预测时域，不产生正常格式的伪特征。
# 输入：
#   horizon：非法时间值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("horizon", [True, -1.0, 0.0, math.nan, math.inf])
def test_invalid_horizon_is_rejected(horizon):
    p, v = Vector3(x=0, y=0, z=0), Vector3(x=0, y=0, z=0)
    with pytest.raises(ValueError):
        cylinder_contact_seconds(p, v, radius_m=1.0, height_m=2.0, horizon_seconds=horizon)
    with pytest.raises(ValueError):
        closest_center_distance_m(p, v, horizon_seconds=horizon)


# 功能：
#   验证圆柱尺寸必须为有限正数，布尔与非有限值不能进入解析计算。
# 输入：
#   value：无效半径或高度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, -1.0, 0.0, math.nan, math.inf])
def test_invalid_cylinder_dimensions_are_rejected(value):
    p, v = Vector3(x=0, y=0, z=0), Vector3(x=0, y=0, z=0)
    for dimensions in ({"radius_m": value, "height_m": 2.0}, {"radius_m": 1.0, "height_m": value}):
        with pytest.raises(ValueError):
            cylinder_contact_seconds(p, v, **dimensions)


# 功能：
#   验证事后篡改的向量字段必须被重新校验，不能输出伪造的安全时间。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_vector_is_revalidated():
    p = Vector3(x=0, y=0, z=0).model_copy(update={"x": True})
    v = Vector3(x=0, y=0, z=0)
    with pytest.raises(ValueError):
        cylinder_contact_seconds(p, v, radius_m=1.0, height_m=2.0)
    with pytest.raises(ValueError):
        closest_center_distance_m(p, v)


# 功能：
#   用高精度二次方程作为独立接触时间基准，不调用生产中的投影弦长算法。
# 输入：
#   p、v：三轴相对位置与速度序列。
#   radius、height、horizon：圆柱半径、全高及预测秒数。
# 输出：
#   expected：高精度基准接触秒数。
def _decimal_contact(p, v, radius, height, horizon):
    expected = horizon
    with localcontext() as context:
        context.prec = 70
        px, py, pz, vx, vy, vz, r, h, limit = [
            Decimal(str(x)) for x in (*p, *v, radius, height, horizon)
        ]
        a, b, c = vx * vx + vy * vy, 2 * (px * vx + py * vy), px * px + py * py - r * r
        if a == 0:
            if c > 0:
                return expected
            lo, hi = Decimal(0), limit
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return expected
            root = discriminant.sqrt()
            lo, hi = (-b - root) / (2 * a), (-b + root) / (2 * a)
        if vz == 0:
            if abs(pz) > h / 2:
                return expected
            bottom, top = Decimal(0), limit
        else:
            bottom, top = sorted(((-h / 2 - pz) / vz, (h / 2 - pz) / vz))
        start, end = max(Decimal(0), lo, bottom), min(limit, hi, top)
        expected = float(start if start <= end else limit)
        return expected


# 功能：
#   将五百组固定随机场景与独立高精度基准对比，覆盖斜向接近、错层和离开等情况。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_contact_matches_independent_high_precision_reference():
    rng = random.Random(805)
    for _ in range(500):
        p = tuple(rng.uniform(-10.0, 10.0) for _ in range(3))
        v = tuple(rng.uniform(-4.0, 4.0) for _ in range(3))
        radius, height, horizon = (
            rng.uniform(0.1, 4.0),
            rng.uniform(0.1, 8.0),
            rng.uniform(0.1, 30.0),
        )
        actual = cylinder_contact_seconds(
            Vector3(x=p[0], y=p[1], z=p[2]),
            Vector3(x=v[0], y=v[1], z=v[2]),
            radius_m=radius,
            height_m=height,
            horizon_seconds=horizon,
        )
        assert actual == pytest.approx(_decimal_contact(p, v, radius, height, horizon), abs=1.0e-9)


# 功能：
#   验证统一缩放长度与速度不改变接触时间，同时最近中心距离按长度比例变化。
# 输入：
#   scale：跨越不同数值量级的长度缩放系数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("scale", [1.0e-100, 1.0, 1.0e100])
def test_motion_features_respect_length_and_speed_scale(scale):
    p = Vector3(x=4 * scale, y=0.3 * scale, z=0)
    v = Vector3(x=-2 * scale, y=0, z=0)
    expected = (4 - math.sqrt(0.5**2 - 0.3**2)) / 2
    assert cylinder_contact_seconds(
        p, v, radius_m=0.5 * scale, height_m=2 * scale
    ) == pytest.approx(expected)
    assert closest_center_distance_m(p, v) == pytest.approx(0.3 * scale, rel=1.0e-12, abs=0.0)


# 功能：
#   验证水平区间端点的差溢出时，不能把实际已离开的圆柱误认为仍与晚到竖直段重叠。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_overflowing_boundary_difference_does_not_extend_contact_window():
    p, v = Vector3(x=1.0e308, y=0, z=5.0), Vector3(x=-1.0e308, y=0, z=-1.0)
    assert cylinder_contact_seconds(p, v, radius_m=1.0e308, height_m=2.0) == 30.0


# 功能：
#   验证恰好相切、已重叠、静止及从圆柱离开等边界都有确定的有限结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tangent_overlap_stationary_and_receding_cases():
    still = Vector3(x=0, y=0, z=0)
    assert (
        cylinder_contact_seconds(
            Vector3(x=2, y=1, z=0), Vector3(x=-1, y=0, z=0), radius_m=1.0, height_m=2.0
        )
        == 2.0
    )
    assert cylinder_contact_seconds(still, still, radius_m=1.0, height_m=2.0) == 0.0
    assert (
        cylinder_contact_seconds(Vector3(x=2, y=0, z=0), still, radius_m=1.0, height_m=2.0) == 30.0
    )
    assert (
        cylinder_contact_seconds(
            Vector3(x=2, y=0, z=0), Vector3(x=1, y=0, z=0), radius_m=1.0, height_m=2.0
        )
        == 30.0
    )
