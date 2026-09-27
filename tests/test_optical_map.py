"""Optical surfaces must remain distinct from conservative collision envelopes."""

import hashlib

import numpy as np
import pytest

from dronedream_agent_core.local_map_alignment import MapAlignmentLimits, MapSurfaceIndex
from dronedream_agent_core.optical_map import compile_optical_map


# 功能：阻止小型共享网格的大量实例在索引检查前耗尽内存。
# 输入：tmp_path：独立测试目录；monkeypatch：解析器测试替身。
# 输出：无；真实索引预算必须在第九份 8192 三角形实例转换为列表前触发。
def test_total_mesh_budget_is_checked_before_list_expansion(tmp_path, monkeypatch):
    import dronedream_agent_core.optical_map as optical

    expanded = []

    class Triangles:
        # 功能：声明单个实例成本；输入：self；输出：三角形数。
        def __len__(self):
            return 8192

        # 功能：记录昂贵的列表展开；输入：self；输出：无需构建真实巨型数组的替身。
        def tolist(self):
            expanded.append(True)
            return []

    monkeypatch.setattr(optical, "_obj_triangles", lambda *_: Triangles())
    (tmp_path / "mesh.obj").write_bytes(cube())
    path, digest = world(tmp_path, visual('<mesh><uri>mesh.obj</uri></mesh>') * 9)
    with pytest.raises(ValueError, match="MESH_TRIANGLE_BUDGET_EXCEEDED"):
        compile_optical_map(path, expected_world_sha256=digest)
    assert len(expanded) == 8


# 功能：拒绝过深的用户地图模型嵌套，返回确定错误而不是递归崩溃。
# 输入：tmp_path：独立测试目录。
# 输出：无。
def test_deep_model_nesting_is_bounded(tmp_path):
    path = tmp_path / "nested.sdf"
    path.write_text('<sdf><world name="w">' +
                    '<model name="m"><static>true</static>' * 34 +
                    '</model>' * 34 + '</world></sdf>', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="MODEL_DEPTH_EXCEEDED"):
        compile_optical_map(path, expected_world_sha256=digest)


# 功能：构造独立单元测试世界；视觉和碰撞内容分别传入，禁止将测试结果称为飞行验收。
# 输入：临时目录、视觉 XML、附加模型内容和模型姿态。
# 输出：实际文件与预期摘要。
def world(tmp_path, visual, extra="", pose="0 0 0 0 0 0"):
    path = tmp_path / "world.sdf"
    path.write_text(
        f'<sdf><world name="w"><model name="m"><static>true</static><pose>{pose}</pose>'
        f'<link name="l">{visual}{extra}</link></model></world></sdf>',
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


# 功能：封装视觉形状，保留独立姿态和透明度。
# 输入：shape：实际几何；pose：局部姿态；alpha：透明度。
# 输出：测试视觉 XML。
def visual(shape, pose="0 0 0 0 0 0", alpha=0):
    return (f'<visual name="v"><pose>{pose}</pose><transparency>{alpha}</transparency>'
            f'<geometry>{shape}</geometry></visual>')


# 功能：生成十二个有向三角形组成的闭合盒网格，用于验证真实网格求交。
# 输入：无。
# 输出：OBJ 字节。
def cube():
    vertices = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return (
        "\n".join("v " + " ".join(map(str, v)) for v in vertices)
        + "\n"
        + "\n".join("f " + " ".join(str(i + 1) for i in f) for f in faces)
        + "\n"
    ).encode()


# 功能：证明定位只使用真实半径，不匹配两倍大小的避障外包球。
# 输入：pytest 独立临时目录。
# 输出：无。
def test_visual_sphere_not_collision_envelope(tmp_path):
    path, digest = world(
        tmp_path,
        visual("<sphere><radius>1</radius></sphere>", "4 0 0 0 0 0"),
        '<collision name="c"><pose>4 0 0 0 0 0</pose><geometry>'
        '<sphere><radius>2</radius></sphere></geometry></collision>',
    )
    index, primitives, receipt = compile_optical_map(path, expected_world_sha256=digest)
    assert len(primitives) == 1 and primitives[0]["radius_m"] == 1
    result = index.match([[3, 0, 0]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0, 0])
    assert result.valid.tolist() == [True]
    assert result.distances[0] == pytest.approx(0)
    assert receipt["collision_geometry_used"] is False
    assert receipt["flight_qualification_granted"] is False


# 功能：透明对象不能提供定位支持，也不能被跳过后错误匹配到后墙。
# 输入：独立临时目录。
# 输出：无。
def test_transparent_occluder_is_not_free_space(tmp_path):
    shape = "<box><size>.1 4 4</size></box>"
    path, digest = world(tmp_path, visual(shape, "2 0 0 0 0 0", 0.5) + visual(shape, "3 0 0 0 0 0"))
    index, _, receipt = compile_optical_map(path, expected_world_sha256=digest)
    assert receipt["ineligible_occluder_count"] == 1
    assert not index.match(
        [[2.95, 0, 0]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0, 0]
    ).valid[0]


# 功能：复合模型姿态必须转到世界坐标后求交。
# 输入：独立临时目录。
# 输出：无。
def test_visual_pose_composes_with_model_pose(tmp_path):
    path, digest = world(
        tmp_path,
        visual("<box><size>.2 2 2</size></box>", "3 0 0 0 0 0"),
        pose=f"1 2 0 0 0 {np.pi / 2}",
    )
    index, _, _ = compile_optical_map(path, expected_world_sha256=digest)
    result = index.match([[1, 4.9, 0]], MapAlignmentLimits(), sensor_origins_world_m=[1, 2, 0])
    assert result.valid[0]
    np.testing.assert_allclose(result.normals[0], [0, -1, 0], atol=1e-12)


# 功能：真实网格表面应支持前表面、拒绝内部起点，并记录资源身份。
# 输入：独立临时目录。
# 输出：无。
def test_closed_mesh_first_surface_and_inside_rejection(tmp_path):
    (tmp_path / "cube.obj").write_bytes(cube())
    path, digest = world(tmp_path, visual("<mesh><uri>cube.obj</uri></mesh>", "4 0 0 0 0 0"))
    index, _, receipt = compile_optical_map(path, expected_world_sha256=digest)
    match = index.match([[3, 0.2, 0.1]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0.2, 0.1])
    assert match.valid[0] and match.distances[0] == pytest.approx(0)
    np.testing.assert_allclose(match.normals[0], [-1, 0, 0])
    assert not index.match(
        [[5, 0, 0]], MapAlignmentLimits(), sensor_origins_world_m=[4, 0, 0]
    ).valid[0]
    assert (
        receipt["resource_dependencies"]["cube.obj"]["sha256"] == hashlib.sha256(cube()).hexdigest()
    )


# 功能：不支持的形状/坐标约定必须报错，而不是遗漏障碍或改用碰撞外包。
# 输入：具体不支持的视觉 XML。
# 输出：无。
@pytest.mark.parametrize(
    "payload",
    [
        visual("<capsule><radius>1</radius></capsule>"),
        visual("<box><size>1 1 1</size></box>").replace("<pose>", '<pose relative_to="frame">'),
        visual("<mesh><uri>../secret.obj</uri></mesh>"),
        visual("<mesh><uri>cube.dae</uri></mesh>"),
        visual("<box><size>nan 1 1</size></box>"),
        visual("<sphere><radius>-1</radius></sphere>"),
    ],
)
def test_invalid_optical_geometry_fails_closed(tmp_path, payload):
    path, digest = world(tmp_path, payload)
    with pytest.raises(ValueError):
        compile_optical_map(path, expected_world_sha256=digest)


# 功能：检查开放/破损网格、摘要不符和材质透明等常见升级边界。
# 输入：独立临时目录。
# 输出：无。
def test_mesh_integrity_and_material_binding(tmp_path):
    (tmp_path / "cube.obj").write_bytes(cube().rsplit(b"f ", 1)[0])
    path, digest = world(tmp_path, visual("<mesh><uri>cube.obj</uri></mesh>"))
    with pytest.raises(ValueError, match="CLOSED_ORIENTED"):
        compile_optical_map(path, expected_world_sha256=digest)
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        compile_optical_map(path, expected_world_sha256="0" * 64)
    (tmp_path / "cube.obj").write_bytes(b"mtllib cube.mtl\n" + cube())
    (tmp_path / "cube.mtl").write_text("newmtl glass\nd .5\n")
    _, _, receipt = compile_optical_map(path, expected_world_sha256=digest)
    assert receipt["ineligible_occluder_count"] == 1
    assert "cube.mtl" in receipt["resource_dependencies"]


# 功能：保留单面网格的绕序剔除规则；反向壳体从外部可见的是远壁，不能擅自翻面。
# 输入：独立临时目录。
# 输出：无。
def test_inward_mesh_winding_preserves_backface_culling(tmp_path):
    lines = cube().decode().splitlines()
    inverted = "\n".join(
        "f " + " ".join(line.split()[1:][::-1]) if line.startswith("f ") else line for line in lines
    )
    (tmp_path / "cube.obj").write_text(inverted)
    path, digest = world(tmp_path, visual("<mesh><uri>cube.obj</uri></mesh>", "4 0 0 0 0 0"))
    index, _, _ = compile_optical_map(path, expected_world_sha256=digest)
    result = index.match(
        [[5, 0.2, 0.1]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0.2, 0.1]
    )
    assert result.valid[0]
    np.testing.assert_allclose(result.normals[0], [-1, 0, 0])
    content = path.read_text().replace(
        "</visual>", "<material><double_sided>true</double_sided></material></visual>"
    )
    path.write_text(content)
    index, _, _ = compile_optical_map(
        path, expected_world_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    result = index.match(
        [[3, 0.2, 0.1]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0.2, 0.1]
    )
    assert result.valid[0]


# 功能：素材文件在地图 XML 未变时也必须按采集回执校验，禁止悄悄使用升级后的形状。
# 输入：独立临时目录。
# 输出：无。
def test_mesh_digest_bound_independently_of_world(tmp_path):
    (tmp_path / "cube.obj").write_bytes(cube())
    path, digest = world(tmp_path, visual("<mesh><uri>cube.obj</uri></mesh>"))
    _, _, receipt = compile_optical_map(path, expected_world_sha256=digest)
    (tmp_path / "cube.obj").write_bytes(cube() + b"# changed\n")
    with pytest.raises(ValueError, match="RESOURCE_DIGEST_MISMATCH"):
        compile_optical_map(
            path, expected_world_sha256=digest, expected_resources=receipt["resource_dependencies"]
        )


# 功能：验证环形网格孔洞保持可见，不用实心包围盒遮住后方真实墙面。
# 输入：pytest 临时目录。
# 输出：无。
def test_torus_hole_is_not_filled_by_acceleration_bounds(tmp_path):
    vertices = []
    for i in range(16):
        u = 2 * np.pi * i / 16
        for j in range(8):
            v = 2 * np.pi * j / 8
            vertices.append(
                (
                    0.2 * np.cos(v),
                    (2 + 0.2 * np.sin(v)) * np.cos(u),
                    (2 + 0.2 * np.sin(v)) * np.sin(u),
                )
            )
    faces = []
    for i in range(16):
        for j in range(8):
            a, b, c, d = (
                i * 8 + j,
                ((i + 1) % 16) * 8 + j,
                ((i + 1) % 16) * 8 + (j + 1) % 8,
                i * 8 + (j + 1) % 8,
            )
            faces.extend(((a, b, c), (a, c, d)))
    obj = "\n".join("v " + " ".join(map(str, p)) for p in vertices) + "\n"
    obj += "\n".join("f " + " ".join(str(k + 1) for k in f) for f in faces)
    (tmp_path / "ring.obj").write_text(obj)
    ring = visual("<mesh><uri>ring.obj</uri></mesh>", "4 0 0 0 0 0").replace(
        "</visual>", "<material><double_sided>true</double_sided></material></visual>"
    )
    wall = visual("<box><size>.1 10 10</size></box>", "7.05 0 0 0 0 0")
    path, digest = world(tmp_path, ring + wall)
    index, _, _ = compile_optical_map(path, expected_world_sha256=digest)
    result = index.match([[7, 0, 0]], MapAlignmentLimits(), sensor_origins_world_m=[0, 0, 0])
    assert result.valid[0] and result.distances[0] == pytest.approx(0)


# 功能：防止混合布尔与浮点三角形被 NumPy 静默变成有效米制坐标。
# 输入：pytest 临时目录。
# 输出：无。
def test_mesh_rejects_boolean_coordinate_before_numpy_coercion(tmp_path):
    (tmp_path / "cube.obj").write_bytes(cube())
    path, digest = world(tmp_path, visual("<mesh><uri>cube.obj</uri></mesh>"))
    _, primitives, _ = compile_optical_map(path, expected_world_sha256=digest)
    primitives[0]["triangles"][0][0][0] = True
    with pytest.raises(ValueError, match="MESH_INVALID"):
        MapSurfaceIndex(primitives)
