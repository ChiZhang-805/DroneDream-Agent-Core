"""Synthetic export-boundary checks, not qualified training records."""

from scripts.export_decision_stage_candidate import bounded_displacement_witnesses


# 功能：保留每次桥接的真实起点及两端见证，只裁掉与本窗口无关的前后回合。
# 输入：合成时间序列和早于窗口的租期；输出：不降采样的连续子序列。
def test_displacement_slice_includes_lease_origin_and_handback():
    rows = [dict(observed_at_unix_ms=t) for t in range(0, 1000, 20)]
    commands = [dict(hybrid_lease=dict(started_at_unix_ms=135))]
    original = rows.copy()
    actual = bounded_displacement_witnesses(rows, commands, 200, 405)
    assert actual == rows[6:22]
    assert rows == original and all(a is b for a, b in zip(actual, rows[6:22]))


# 功能：原始缺口和缺失边界必须保留以供后续拒收，不补充不存在的见证。
# 输入：稀疏原记录；输出：原值不变，没有插值或复制时间戳。
def test_displacement_slice_never_fills_missing_boundary_or_gap():
    rows = [dict(observed_at_unix_ms=t) for t in (220, 240, 380)]
    assert bounded_displacement_witnesses(rows, [], 200, 400) == rows


# 功能：端点精确重合只出现一次，重复命令租期不重复见证；输出：严格连续原序列。
def test_exact_boundary_no_duplicate_witness():
    rows = [dict(observed_at_unix_ms=t) for t in range(0, 501, 100)]
    assert bounded_displacement_witnesses(rows, [], 100, 400) == rows[1:5]
