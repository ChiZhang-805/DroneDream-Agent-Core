import json

import pytest

from dronedream_agent_core.training.mission_groups import (
    MissionGroupEvidence,
    MissionGroupManifest,
    merge_mission_group_manifests,
    mission_group_evidence,
    spatial_mission_group,
)


# 功能：
#   将测试三维元组构造成米制路线，严格要求每个点具有三个分量。
# 输入：
#   points：三维坐标序列。
# 输出：
#   payload：含 positions_m 的路线对象。
def route(points):
    payload = {"positions_m": [dict(zip(("x", "y", "z"), p, strict=True)) for p in points]}
    return payload


# 功能：
#   构造可由固定直线路线原文重算的合成分组证据。
# 输入：
#   y：直线的横向位置，用于构造不同空间路线。
#   semantic：语义地图摘要。
# 输出：
#   evidence：带原文和一致摘要的测试证据。
def split_fixture(*, y=0., semantic="a" * 64):
    evidence = mission_group_evidence(json.dumps(route([(0, y, 1), (2, y, 1)])).encode(), semantic)
    return evidence


# 功能：
#   构造训练流和验证流分别对应不同直线路线的合成清单。
# 输入：
#   无。
# 输出：
#   manifest：两条流和两份路线证据的分组清单。
def group_fixture():
    train, validation = split_fixture(), split_fixture(y=2.)
    manifest = MissionGroupManifest(
        groups={"train": train.group_sha256, "val": validation.group_sha256},
        evidence=[train, validation])
    return manifest


# 功能：
#   验证改名、反向和重复遍历或共线细分不改变分组，而高度或语义地图变化会改变分组。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_names_reversal_repeated_traversals_and_intermediate_points_are_one_route():
    path = [(0, 0, 1), (2, 0, 1), (2, 2, 1)]
    group = spatial_mission_group(route(path), "a" * 64)
    for points in (path[::-1], [*path, *path[-2::-1]], path + path[-2::-1] + path[1:],
                   [path[0], (1, 0, 1), path[1], (2, 1, 1), path[-1]]):
        content = {**route(points), "mission_id": "renamed", "seed": 900, "vehicle": "another"}
        assert spatial_mission_group(content, "a" * 64) == group
    assert spatial_mission_group(route([(0, 0, 2), (2, 0, 2), (2, 2, 2)]), "a" * 64) != group
    assert spatial_mission_group(route(path), "b" * 64) != group


# 功能：
#   验证旧式纯名称映射和没有路线原文支撑的流分组不能进入当前训练划分。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unsigned_names_only_mapping_cannot_reenter_current_training():
    with pytest.raises(ValueError):
        MissionGroupManifest.model_validate({"train": "route-a", "val": "route-b"})
    one = split_fixture()
    with pytest.raises(ValueError, match="STREAM_EVIDENCE"):
        MissionGroupManifest(groups={"stream": "b" * 64}, evidence=[one])


# 功能：
#   验证绕过赋值检查改写原文或空间摘要后，直接重验和嵌套清单重验都会拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_content_is_verified_again_even_when_given_a_preconstructed_object():
    evidence = split_fixture()
    invalid = evidence.model_copy(update={"route_source_utf8": "{}"})
    with pytest.raises(ValueError, match="CONTENT_CHANGED"):
        MissionGroupEvidence.model_validate(invalid.model_dump())
    with pytest.raises(ValueError, match="CONTENT_CHANGED"):
        MissionGroupManifest(groups={"train": invalid.group_sha256}, evidence=[invalid])
    invalid = evidence.model_copy(update={"group_sha256": "b" * 64})
    with pytest.raises(ValueError, match="SPATIAL_IDENTITY"):
        MissionGroupEvidence.model_validate(invalid.model_dump())


# 功能：
#   验证闭合路线改变起点或部分重走，仍属于相同的无向空间覆盖。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_closed_loop_start_point_and_partial_retracing_do_not_create_a_new_group():
    points = [(0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1), (0, 0, 1)]
    expected = spatial_mission_group(route(points), "a" * 64)
    shifted = [(1, 0, 1), *points[1:], (1, 0, 1)]
    retraced = [*points, (1, 0, 1), (0, 0, 1)]
    for path in (shifted, retraced):
        assert spatial_mission_group(route(path), "a" * 64) == expected


# 功能：
#   验证共线但不相接的覆盖不会误合并为完整线段，补走空隙应产生不同分组。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_disconnected_collinear_segments_do_not_fill_in_a_missing_path():
    path = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (2, 1, 1), (2, 0, 1), (3, 0, 1)]
    gap_filled = [*path, (0, 0, 1)]
    assert spatial_mission_group(route(path), "a" * 64) != spatial_mission_group(
        route(gap_filled), "a" * 64)


# 功能：
#   验证新回合名称和种子可改变原文字节，但不能把相同路线伪装成独立验证组。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_same_route_in_a_new_episode_cannot_become_a_validation_group():
    first = split_fixture()
    renamed = mission_group_evidence(json.dumps({
        **json.loads(first.route_source_utf8), "name": "new-flight", "seed": 998}).encode(),
        first.semantic_sha256)
    assert first.route_sha256 != renamed.route_sha256
    assert first.group_sha256 == renamed.group_sha256
    combined = merge_mission_group_manifests(
        MissionGroupManifest(groups={"train": first.group_sha256}, evidence=[first]),
        MissionGroupManifest(groups={"renamed": renamed.group_sha256}, evidence=[renamed]))
    assert combined.groups["train"] == combined.groups["renamed"]


# 功能：
#   验证嵌套路线中的前后空白和中文经过证据序列化往返后逐字节保留。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_original_route_bytes_keep_whitespace_and_unicode_through_receipt_roundtrip():
    content = (" \r\n" + json.dumps({**route([(0, 0, 1), (1, 0, 1)]),
                                   "name": "校舍"}, ensure_ascii=False) + "\n").encode()
    split = mission_group_evidence(content, "a" * 64)
    restored = MissionGroupEvidence.model_validate_json(split.model_dump_json())
    assert restored.route_source_utf8.encode() == content


# 功能：
#   验证空路线、零位移、非有限坐标、布尔值和极大坐标不能生成合法空间身份。
# 输入：
#   points：违反路线几何或数值契约的点列。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("points", [[], [(0, 0, 0)], [(0, 0, 0), (0, 0, 0)],
    [(0, 0, 0), (float("nan"), 0, 0)], [(0, 0, 0), (True, 0, 0)],
    [(0, 0, 0), (10**1000, 0, 0)], [(0, 0, 0), (1e300, 0, 0)]])
def test_invalid_or_empty_motion_is_not_a_route(points):
    with pytest.raises(ValueError, match="MISSION_GROUP"):
        spatial_mission_group(route(points), "a" * 64)
