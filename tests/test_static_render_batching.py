import hashlib
import math
from xml.etree import ElementTree as ET

import pytest

from dronedream_agent_core.static_render_batching import (
    batch_static_box_visuals,
    copy_relative_render_resources,
    nonvisual_world_digest,
    prepare_static_render_world,
)


# 功能：
#   构造带固定碰撞、嵌套姿态及可变视觉盒体的 SDF，供渲染等价性测试使用。
# 输入：
#   count：视觉盒体数量。
#   extra：附加到各 visual 的测试 XML。
#   pose：盒体局部六自由度姿态文本。
#   model_extra：附加到模型的测试 XML。
#   static：模型静态标记文本。
# 输出：
#   content：测试世界的 UTF-8 字节。
def world(*, count=8, extra="", pose="0 0 0 0 0 0", model_extra="", static="true"):
    visuals = "".join(f'''<visual name="box-{i}"><pose>{pose}</pose>
        <geometry><box><size>2 4 6</size></box></geometry>
        <material><ambient>.2 .3 .4 1</ambient><diffuse>.2 .3 .4 1</diffuse></material>
        {extra}</visual>''' for i in range(count))
    content = f'''<sdf version="1.9"><world name="test"><physics name="p" type="ode">
        <max_step_size>.004</max_step_size></physics><gravity>0 0 -9.81</gravity>
        <model name="map"><static>{static}</static>{model_extra}<pose>9 8 7 0 0 .4</pose>
        <link name="structure"><pose>1 2 3 0 0 .2</pose>
        <collision name="wall"><geometry><box><size>2 4 6</size></box></geometry></collision>
        {visuals}</link></model></world></sdf>'''.encode()
    return content


# 功能：
#   解析测试 OBJ 中的顶点、法向和索引，用于独立核对面方向与体积。
# 输入：
#   content：生成的 OBJ 字节。
# 输出：
#   result：顶点列表、法向列表及转为零基索引的三角面列表。
def read_mesh(content):
    vertices, normals, faces = [], [], []
    for row in content.decode().splitlines():
        fields = row.split()
        if fields[0] == "v":
            vertices.append(tuple(float(v) for v in fields[1:]))
        elif fields[0] == "vn":
            normals.append(tuple(float(v) for v in fields[1:]))
        elif fields[0] == "f":
            faces.append(tuple(tuple(int(v) - 1 for v in field.split("//"))
                               for field in fields[1:]))
    result = vertices, normals, faces
    return result


# 功能：
#   计算右手三维叉积，作为测试面法向及有向体积的独立算术对照。
# 输入：
#   a：第一个三维向量。
#   b：第二个三维向量。
# 输出：
#   result：a 叉乘 b 的向量。
def cross(a, b):
    result = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0])
    return result


# 功能：
#   核对合并保留每个盒面的几何及全部非视觉世界，改变重力时摘要必须不同。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_batch_keeps_nonvisual_world_and_every_box_face():
    original = world()
    result = batch_static_box_visuals(original)
    assert result.receipt["merged_box_count"] == 8
    assert result.receipt["render_visual_count"] == 1
    assert result.receipt["source_visual_count"] == 8
    assert result.receipt["nonvisual_world_unchanged"]
    assert not result.receipt["flight_qualification_granted"]
    assert nonvisual_world_digest(original) == nonvisual_world_digest(result.content)
    vertices, normals, faces = read_mesh(next(iter(result.meshes.values())))
    assert (len(vertices), len(normals), len(faces)) == (8 * 24, 8 * 6, 8 * 12)
    assert tuple(min(v[k] for v in vertices) for k in range(3)) == (-1., -2., -3.)
    assert tuple(max(v[k] for v in vertices) for k in range(3)) == (1., 2., 3.)
    modified = original.replace(b"0 0 -9.81", b"0 0 -1.6")
    assert nonvisual_world_digest(modified) != nonvisual_world_digest(result.content)


# 功能：
#   对平移和复合旋转后的网格逐面计算法向和有向体积，确认没有反面或几何缩减。
# 输入：
#   pose：测试盒体局部姿态文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("pose", ["1 2 3 0 0 0", "1 2 3 .2 .3 .4", "-2 1 2 0 0 1.5707963267948966"])
def test_transformed_boxes_keep_outward_flat_normals_and_exact_signed_volume(pose):
    result = batch_static_box_visuals(world(pose=pose))
    vertices, normals, faces = read_mesh(next(iter(result.meshes.values())))
    volume = 0.
    for face in faces:
        a, b, c = (vertices[item[0]] for item in face)
        normal = normals[face[0][1]]
        ab, ac = tuple(b[k] - a[k] for k in range(3)), tuple(c[k] - a[k] for k in range(3))
        surface = cross(ab, ac)
        norm = math.sqrt(sum(v * v for v in surface))
        assert tuple(v / norm for v in surface) == pytest.approx(normal)
        volume += sum(a[k] * cross(b, c)[k] for k in range(3)) / 6
    assert volume == pytest.approx(8 * 2 * 4 * 6)


# 功能：
#   透明、带插件或未知属性的视觉元素必须保留，不擅自简化其语义。
# 输入：
#   extra：附加 visual 配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("extra", [
    "<transparency>.1</transparency>", "<plugin name='animated' filename='plugin'/>",
    "<laser_retro>100</laser_retro>", "<visibility_flags>2</visibility_flags>",
])
def test_unknown_or_nonopaque_visuals_are_preserved_byte_for_byte(extra):
    content = world(extra=extra)
    result = batch_static_box_visuals(content)
    assert result.content == content
    assert not result.meshes


# 功能：
#   验证相对姿态、扩展材质、透明通道和非法数值不进入简单盒体合并路径。
# 输入：
#   modifier：修改候选世界的测试函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("modifier", [
    lambda b: b.replace(b"<pose>0", b"<pose relative_to='frame'>0"),
    lambda b: b.replace(b"<material>", b"<material><pbr><metal/></pbr>"),
    lambda b: b.replace(b".4 1", b".4 .5"),
    lambda b: b.replace(b"2 4 6", b"2 4 nan"),
    lambda b: b.replace(b"<size>", b"<size units='other'>"),
])
def test_unsupported_semantics_are_not_guessed(modifier):
    content = modifier(world())
    assert batch_static_box_visuals(content).content == content


# 功能：
#   动态模型、插件、嵌套模型及小规模分组保持原始内容，不跨边界合并。
# 输入：
#   kwargs：世界构造器的测试参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kwargs", [
    {"static": "false"}, {"model_extra": "<plugin name='x' filename='y'/>"},
    {"model_extra": "<model name='nested'/>"}, {"model_extra": "<frame name='f'/>"},
    {"count": 3},
])
def test_dynamic_plugin_or_frame_bound_models_and_small_groups_unchanged(kwargs):
    content = world(**kwargs)
    assert batch_static_box_visuals(content).content == content


# 功能：
#   核对不同空间格或不同阴影设置的盒体分别合并，不混用样式或跨格扩大批次。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_spatial_cells_and_shadow_settings_do_not_merge():
    content = world(count=4, extra="<cast_shadows>false</cast_shadows>")
    root = ET.fromstring(content)
    link = root.find("world/model/link")
    for offset, shadow in ((20, "false"), (0, "true")):
        other = ET.fromstring(content)
        for index, visual in enumerate(other.findall(".//visual")):
            visual.set("name", f"other-{offset}-{index}")
            visual.find("pose").text = f"{offset} 0 0 0 0 0"
            visual.find("cast_shadows").text = shadow
            link.append(visual)
    result = batch_static_box_visuals(ET.tostring(root))
    assert len(result.receipt["batches"]) == 3
    assert result.receipt["merged_box_count"] == 12
    assert result.receipt["render_visual_count"] == 3


# 功能：
#   两个新目录中的相同转换必须具有相同内容和回执，旧目录不可被再次覆盖。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_materialization_is_repeatable_content_bound_and_never_overwrites(tmp_path):
    source = tmp_path / "source.sdf"
    source.write_bytes(world())
    one, receipt = prepare_static_render_world(source, tmp_path / "one")
    two, second = prepare_static_render_world(source, tmp_path / "two")
    assert one.read_bytes() == two.read_bytes()
    assert receipt == second
    assert source.read_bytes() == world()
    for batch in receipt["batches"]:
        mesh = one.parent / batch["mesh"]
        assert hashlib.sha256(mesh.read_bytes()).hexdigest() == batch["mesh_sha256"]
    with pytest.raises(FileExistsError):
        prepare_static_render_world(source, tmp_path / "one")


# 功能：
#   满批 256 个盒体合并后，剩余不足四个的盒体仍留在世界中，不因分批丢失。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_large_groups_are_bounded_without_dropping_remainder():
    result = batch_static_box_visuals(world(count=259))
    assert result.receipt["merged_box_count"] == 256
    assert result.receipt["render_visual_count"] == 4  # One mesh and three retained boxes.
    assert result.receipt["skipped_visuals"] == {"small-spatial-group": 3}


# 功能：
#   即使没有可合并 visual，相对纹理仍应按原路径复制并绑定实际字节摘要。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_relative_texture_is_bound_and_copied_unchanged(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "texture.ppm").write_bytes(b"texture-test-content")
    content = world().replace(b"<material>",
        b"<material><pbr><metal><albedo_map>texture.ppm</albedo_map></metal></pbr>")
    (source / "world.sdf").write_bytes(content)
    derived, receipt = prepare_static_render_world(source / "world.sdf", tmp_path / "derived")
    assert (derived.parent / "texture.ppm").read_bytes() == b"texture-test-content"
    assert receipt["preserved_relative_resources"]["texture.ppm"]["bytes"] == 20
    assert derived.read_bytes() == content  # No eligible visuals: resources still preserved.


# 功能：
#   拒绝越界、缺失或指向目录的相对资源，不能以资源复制为由扫描未知目录。
# 输入：
#   tmp_path：pytest 隔离目录。
#   reference：待测试的相对引用。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reference", ["../secret", "absent", "."])
def test_resource_copy_never_traverses_or_crawls_unclassified_directories(tmp_path, reference):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "secret").write_bytes(b"not-a-texture")
    content = f"<sdf><uri>{reference}</uri></sdf>".encode()
    with pytest.raises(ValueError, match="RESOURCE_UNSUPPORTED"):
        copy_relative_render_resources(content, source, tmp_path / "output")


# 功能：
#   验证有限但极大的盒体坐标不能生成含 Infinity 的 OBJ，并冒充等价渲染结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_batch_rejects_nonfinite_transformed_mesh():
    content = world(count=4, pose="1.7e308 0 0 0 0 0").replace(b"2 4 6", b"1.7e308 4 6")
    with pytest.raises(ValueError, match="MESH_NONFINITE"):
        batch_static_box_visuals(content)


# 功能：
#   为启动脚本绑定已验证的源摘要，拒绝与该快照不同的当前世界，且不创建输出目录。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_materialization_requires_expected_source_digest_before_output(tmp_path):
    source, output = tmp_path / "map.sdf", tmp_path / "derived"
    content = world()
    source.write_bytes(content.replace(b"9 8 7", b"8 8 7"))
    with pytest.raises(ValueError, match="SOURCE_BINDING"):
        prepare_static_render_world(source, output,
                                   expected_source_sha256=hashlib.sha256(content).hexdigest())
    assert not output.exists()


# 功能：
#   保护源文件及相对资源的实际读取上限，禁止先检查大小后调用无限量 read_bytes。
# 输入：
#   tmp_path：pytest 隔离目录。
#   monkeypatch：pytest 替换工具。
# 输出：
#   None：不返回业务数据。
def test_materialization_reads_sources_with_explicit_bounds(tmp_path, monkeypatch):
    from pathlib import Path

    source = tmp_path / "source"
    source.mkdir()
    (source / "texture.ppm").write_bytes(b"texture")
    content = world().replace(b"<material>",
        b"<material><pbr><metal><albedo_map>texture.ppm</albedo_map></metal></pbr>")
    path = source / "map.sdf"
    path.write_bytes(content)
    original_read = Path.read_bytes

    # 功能：
    #   在测试源目录拦截无限量读取；其他测试验证读仍按原方法执行。
    # 输入：
    #   item：读取目标。
    # 输出：
    #   data：非源目录文件的字节。
    def reject_unbounded(item):
        if item.parent == source:
            raise AssertionError("source read requires an explicit bound")
        data = original_read(item)
        return data

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded)
    _, receipt = prepare_static_render_world(path, tmp_path / "derived")
    assert receipt["source_world_sha256"] == hashlib.sha256(content).hexdigest()


# 功能：
#   核对 OBJ 的相邻材质及材质纹理递归复制、去重和绑定，避免派生世界静默丢失颜色。
# 输入：
#   tmp_path：pytest 隔离目录。
# 输出：
#   None：不返回业务数据。
def test_obj_material_and_texture_dependencies_are_preserved(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "meshes").mkdir(parents=True)
    (source / "textures").mkdir()
    (source / "meshes/a.obj").write_bytes(b'mtllib "gate material.mtl"\nv 0 0 0\n')
    (source / "meshes/b.obj").write_bytes(b'mtllib "gate material.mtl"\nv 1 0 0\n')
    (source / "meshes/gate material.mtl").write_bytes(b'newmtl gate\nmap_Kd ../textures/color.png\n')
    (source / "textures/color.png").write_bytes(b"texture-content")
    content = b"<sdf><uri>meshes/a.obj</uri><uri>meshes/b.obj</uri></sdf>"
    resources = copy_relative_render_resources(content, source, output)
    assert len(resources) == 4
    for relative, receipt in resources.items():
        assert (source / relative).read_bytes() == (output / relative).read_bytes()
        assert receipt["sha256"] == hashlib.sha256((output / relative).read_bytes()).hexdigest()


# 功能：
#   嵌套材质引用也必须遵守资源边界，缺失或逃逸时不能继续输出成功回执。
# 输入：
#   tmp_path：pytest 隔离目录。
#   reference：错误材质引用。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reference", ["missing.mtl", "../../secret.mtl", "https://example.test/material.mtl"])
def test_obj_dependencies_reject_missing_external_or_escaping_resources(tmp_path, reference):
    source = tmp_path / "source"
    (source / "meshes").mkdir(parents=True)
    (tmp_path / "secret.mtl").write_bytes(b"newmtl private")
    (source / "meshes/a.obj").write_text(f"mtllib {reference}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RESOURCE_UNSUPPORTED"):
        copy_relative_render_resources(b"<sdf><uri>meshes/a.obj</uri></sdf>", source, tmp_path / "output")
