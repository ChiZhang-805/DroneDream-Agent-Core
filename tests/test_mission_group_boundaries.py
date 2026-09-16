"""Adversarial route evidence; no simulator, model fitting or flight acceptance."""

import hashlib
import json

import pytest
from test_mission_groups import route, split_fixture

from dronedream_agent_core.training.mission_groups import (
    MissionGroupEvidence,
    mission_group_evidence,
    recorded_mission_group,
    spatial_mission_group,
)


# 功能：
#   验证路线摘要即使重新匹配，歧义键和非标准数值仍不能成为训练划分证据。
# 输入：
#   fault：重复路径字段或无关元数据中的非有限数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["duplicate", "nan"])
def test_route_json_requires_unambiguous_finite_values(fault):
    content = json.dumps(route([(0, 0, 1), (2, 0, 1)])).encode()
    prefix = b'"positions_m":[], ' if fault == "duplicate" else b'"unused":NaN, '
    content = b"{" + prefix + content[1:]
    with pytest.raises(ValueError):
        mission_group_evidence(content, "a" * 64)


# 功能：
#   验证反序列化已有证据时也严格解析嵌套路线原文，不能绕过创建入口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_embedded_route_is_strict_on_model_validation():
    original = split_fixture()
    content = '{"positions_m":[], ' + original.route_source_utf8[1:]
    payload = original.model_dump()
    payload.update(route_source_utf8=content,
                   route_sha256=hashlib.sha256(content.encode()).hexdigest())
    with pytest.raises(ValueError):
        MissionGroupEvidence.model_validate(payload)


# 功能：
#   验证路线按实际 UTF-8 字节数限制，而非只按 Python 字符数放行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_route_byte_budget_precedes_hash_and_parse():
    content = json.dumps({**route([(0, 0, 1), (2, 0, 1)]), "title": "航" * 1_500_000},
                         ensure_ascii=False).encode()
    assert len(content) > 4 * 1024 * 1024
    with pytest.raises(ValueError):
        mission_group_evidence(content, "a" * 64)


# 功能：
#   验证历史路线入口执行普通文件路径检查，不只比较读取内容的摘要。
# 输入：
#   tmp_path：临时历史运行目录。
#   monkeypatch：用受控读取边界代替不可移植的符号链接创建。
# 输出：
#   None：不返回业务数据。
def test_recorded_route_uses_shared_plain_file_reader(tmp_path, monkeypatch):
    from dronedream_agent_core import plugin_files

    original = split_fixture()
    (tmp_path / "mission-route.json").write_text(original.route_source_utf8, encoding="utf-8")

    # 功能：
    #   模拟路径校验发现链接或重解析点时的明确拒绝。
    # 输入：
    #   args、kwargs：读取器的原始实参。
    # 输出：
    #   None：不返回业务数据。
    def reject_path(*args, **kwargs):
        raise ValueError("TEST_ROUTE_PATH_REFUSED")

    monkeypatch.setattr(plugin_files, "check_plain_plugin_path", reject_path)
    with pytest.raises(ValueError, match="TEST_ROUTE_PATH_REFUSED"):
        recorded_mission_group(tmp_path, {"semantic_sha256": original.semantic_sha256,
                                         "route_sha256": original.route_sha256})


# 功能：
#   验证合法的明确兄弟计划可以读回，而已存在但损坏的运行路线不能被旧计划替换。
# 输入：
#   tmp_path：同时含运行目录与计划目录的临时根。
# 输出：
#   None：不返回业务数据。
def test_explicit_sibling_fallback_never_masks_corrupt_primary(tmp_path):
    original = split_fixture()
    run = tmp_path / "run"
    plan = tmp_path / "plan"
    run.mkdir()
    plan.mkdir()
    (plan / "route.json").write_text(original.route_source_utf8, encoding="utf-8")
    artifacts = {"semantic_sha256": original.semantic_sha256,
                 "route_sha256": original.route_sha256}
    assert recorded_mission_group(run, artifacts) == original
    (run / "mission-route.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        recorded_mission_group(run, artifacts)


# 功能：
#   验证共线点细分、相邻重复点及反向遍历保持相同路线空间身份。
# 输入：
#   axis：三维空间中的一个方向轴。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_aligned_route_subdivision_preserves_identity(axis):
    points = []
    for value in (-3, -2, 0, 2, 3):
        point = [1, 1, 1]
        point[axis] = value
        points.append(tuple(point))
    expected = spatial_mission_group(route([points[0], points[-1]]), "a" * 64)
    actual = spatial_mission_group(route([points[0], *points, *points[::-1]]), "a" * 64)
    assert actual == expected
