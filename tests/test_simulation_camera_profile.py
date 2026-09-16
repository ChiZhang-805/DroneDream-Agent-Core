import hashlib
import json
from xml.etree import ElementTree as ET

import pytest

from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.simulation_camera_profile import (
    CameraProfileReadback,
    camera_configuration,
    control_stream_profile,
    prepare_camera_profile,
    validate_camera_profile_choice,
)


# 功能：
#   构造含原生光学参数、位姿及质量字段的双相机 XML，供配置派生前后对照。
# 输入：
#   无。
# 输出：
#   content：独立生成的相机 SDF 字节。
def source_model():
    root = ET.Element("sdf", version="1.9")
    model = ET.SubElement(root, "model", name="OakD-Lite")
    ET.SubElement(model, "static").text = "false"
    link = ET.SubElement(model, "link", name="camera_link")
    ET.SubElement(link, "inertial").text = "preserve native mass and inertia"
    ET.SubElement(link, "collision").text = "preserve native camera collision"
    for name, kind, width, height, fov in [("IMX214", "camera", 1920, 1080, 1.204),
        ("StereoOV7251", "depth_camera", 640, 480, 1.274)]:
        sensor = ET.SubElement(link, "sensor", name=name, type=kind)
        ET.SubElement(sensor, "update_rate").text = "30"
        ET.SubElement(sensor, "pose").text = ".01233 -.03 .01878 0 0 0"
        camera = ET.SubElement(sensor, "camera")
        ET.SubElement(camera, "horizontal_fov").text = str(fov)
        image = ET.SubElement(camera, "image")
        ET.SubElement(image, "width").text = str(width)
        ET.SubElement(image, "height").text = str(height)
        clip = ET.SubElement(camera, "clip")
        ET.SubElement(clip, "near").text = ".1" if name == "IMX214" else ".2"
        ET.SubElement(clip, "far").text = "100" if name == "IMX214" else "19.1"
    content = ET.tostring(root)
    return content


# 功能：
#   验证默认配置只改变流尺寸和频率，保留光学参数且不声称图像等价或飞行通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_profile_is_only_an_explicit_stream_change_not_image_equivalence():
    source = source_model()
    result, receipt = control_stream_profile(source)
    after = camera_configuration(result)
    assert after["IMX214"]["width"] == 640 and after["IMX214"]["height"] == 360
    assert after["StereoOV7251"]["width"] == 320
    assert {s["update_rate_hz"] for s in after.values()} == {20}
    for name, fields in after.items():
        for key in ("horizontal_fov_rad", "near_m", "far_m"):
            assert fields[key] == receipt["before"][name][key]
    assert receipt["only_stream_dimensions_and_rates_changed"] is True
    assert receipt["image_equivalence_claimed"] is False
    assert receipt["flight_qualification_granted"] is False


# 功能：
#   逐项注入不支持的相机类型、光学、比例和嵌套来源，确认不会静默重配。
# 输入：
#   fault：需要注入的模型不兼容类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["wrong-name", "wrong-type", "missing-sensor", "fov",
    "aspect", "rate", "extra-sensor", "include"])
def test_unknown_or_incompatible_camera_is_not_silently_reconfigured(fault):
    root = ET.fromstring(source_model())
    model = root.find("model")
    sensor = root.find("model/link/sensor")
    if fault == "wrong-name":
        model.set("name", "another-camera")
    elif fault == "wrong-type":
        sensor.set("type", "gpu_lidar")
    elif fault == "missing-sensor":
        model.find("link").remove(sensor)
    elif fault == "fov":
        sensor.find("camera/horizontal_fov").text = "nan"
    elif fault == "aspect":
        sensor.find("camera/image/width").text = "1000"
    elif fault == "rate":
        sensor.find("update_rate").text = "0"
    elif fault == "extra-sensor":
        ET.SubElement(model.find("link"), "sensor", name="thermal", type="thermal_camera")
    else:
        ET.SubElement(model, "include")
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_"):
        control_stream_profile(ET.tostring(root))


# 功能：
#   对照源包与派生包字节及摘要，证明模型外资源原样复制，既有输出不能覆盖。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_overlay_binds_copies_and_never_mutates_installed_model_or_mesh(tmp_path):
    models, output = tmp_path / "installed-models", tmp_path / "run"
    bundle = models / "OakD-Lite"
    (bundle / "meshes").mkdir(parents=True)
    original = source_model()
    (bundle / "model.sdf").write_bytes(original)
    (bundle / "meshes/camera.dae").write_bytes(b"native mesh")
    digest = hashlib.sha256(original).hexdigest()
    result = prepare_camera_profile(source_models=models, output=output,
                                    expected_source_sha256=digest)
    receipt = json.loads((output / "camera-profile.json").read_text())
    assert receipt == result["receipt"] and receipt["installed_assets_modified"] is False
    assert (bundle / "model.sdf").read_bytes() == original
    overlay = output / "models/OakD-Lite"
    assert (overlay / "meshes/camera.dae").read_bytes() == b"native mesh"
    assert (receipt["source_files"]["meshes/camera.dae"]
            == receipt["overlay_files"]["meshes/camera.dae"])
    assert receipt["source_files"]["model.sdf"] != receipt["overlay_files"]["model.sdf"]
    with pytest.raises(FileExistsError):
        prepare_camera_profile(source_models=models, output=output, expected_source_sha256=digest)


# 功能：
#   验证调用方绑定的来源摘要不匹配时，在创建派生目录之前拒绝。
# 输入：
#   tmp_path：隔离来源与输出目录。
# 输出：
#   None：不返回业务数据。
def test_source_hash_mismatch_fails_before_overlay_creation(tmp_path):
    models = tmp_path / "models"
    (models / "OakD-Lite").mkdir(parents=True)
    (models / "OakD-Lite/model.sdf").write_bytes(source_model())
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        prepare_camera_profile(source_models=models, output=tmp_path / "run",
                               expected_source_sha256="a" * 64)
    assert not (tmp_path / "run").exists()


# 功能：
#   拒绝未知配置和缺失、错误或不适用于原生模式的来源摘要，避免隐式选用旧模型。
# 输入：
#   profile：候选配置名称。
#   digest：候选来源摘要。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("profile,digest", [("unknown", None), ("native", "a" * 64),
    ("low-latency", None), ("low-latency", "wrong"),
    ("compact-control", None), ("compact-control", "wrong"),
    ("responsive-control", None), ("responsive-control", "wrong"), ([], None)])
def test_profile_never_uses_an_implicit_or_unbound_native_dependency(profile, digest):
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_PROFILE_"):
        validate_camera_profile_choice(profile, digest)


# 功能：
#   从受控来源创建真实派生文件和回执，供回读、资源绑定及故障注入测试使用。
# 输入：
#   tmp_path：隔离测试目录。
#   profile：需要生成的相机配置名称。
#   resources：可选附属文件相对路径与字节，不参与模型字段派生。
# 输出：
#   receipt_path：已生成的相机回执路径。
def readback_fixture(tmp_path, *, profile="low-latency", resources=None):
    models = tmp_path / "models"
    (models / "OakD-Lite").mkdir(parents=True)
    content = source_model()
    (models / "OakD-Lite/model.sdf").write_bytes(content)
    for name, data in (resources or {}).items():
        target = models / "OakD-Lite" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    prepare_camera_profile(source_models=models, output=tmp_path / "run",
                           expected_source_sha256=hashlib.sha256(content).hexdigest(),
                           profile=profile)
    receipt_path = tmp_path / "run/camera-profile.json"
    return receipt_path


# 功能：
#   确认双流必须实际出现且尺寸匹配，结果副本不影响基线，配置失配不会由下一帧抹除。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_actual_camera_streams_must_match_and_both_be_present(tmp_path):
    profile = CameraProfileReadback(readback_fixture(tmp_path))
    assert profile.observe("depth", 320, 240)
    with pytest.raises(ValueError, match="READBACK_PENDING"):
        profile.require_ready()
    assert profile.observe("rgb", 640, 360)
    proof = profile.require_ready()
    assert proof["verified_dimensions"] == {"depth": [320, 240], "rgb": [640, 360]}
    proof["verified_dimensions"]["rgb"][0] = 1
    assert profile.expected["rgb"] == (640, 360)
    assert not profile.observe("rgb", 1920, 1080)  # Native model won search order instead.
    assert not profile.observe("rgb", 640, 360)  # Configuration mismatch stays failed.
    with pytest.raises(ValueError, match="ACTUAL_STREAM_PROFILE_MISMATCH"):
        profile.require_ready()


# 功能：
#   验证派生模型在回执生成后被改变，回读不能仅凭原回执通过。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_changed_overlay_does_not_pass_receipt_readback(tmp_path):
    receipt = readback_fixture(tmp_path)
    model = receipt.parent / "models/OakD-Lite/model.sdf"
    model.write_bytes(model.read_bytes().replace(b"<width>640</width>", b"<width>1920</width>"))
    with pytest.raises(ValueError, match="OVERLAY_CHANGED"):
        CameraProfileReadback(receipt)


# 功能：
#   验证数值合法但与现有编码器标定不同的视场及量程必须拒绝，不能靠摘要自动适配。
# 输入：
#   field：修改的光学字段。
#   value：数值合法但未标定的新值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("horizontal_fov", "1.2"),
    ("clip/near", ".3"), ("clip/far", "30")])
def test_valid_but_different_depth_optics_require_explicit_encoder_calibration(field, value):
    root = ET.fromstring(source_model())
    depth = root.find("model/link/sensor[@name='StereoOV7251']/camera")
    depth.find(field).text = value
    with pytest.raises(ValueError, match="OPTICS_REQUIRE_ENCODER_CALIBRATION"):
        control_stream_profile(ET.tostring(root))


# 功能：
#   确认超大模型在 XML 解析之前因字节预算被拒绝。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_readback_bounds_model_before_xml_parsing(tmp_path):
    receipt = readback_fixture(tmp_path)
    model = receipt.parent / "models/OakD-Lite/model.sdf"
    model.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="MODEL_TOO_LARGE"):
        CameraProfileReadback(receipt)


# 功能：
#   验证紧凑配置按自身实际尺寸回读，不能继承其他配置的流尺寸证据。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_compact_profile_requires_its_own_actual_readback(tmp_path):
    path = readback_fixture(tmp_path, profile="compact-control")
    receipt = json.loads(path.read_text())
    assert receipt["profile"] == "compact-control"
    assert receipt["installed_assets_modified"] is False
    assert receipt["image_equivalence_claimed"] is False
    assert receipt["flight_qualification_granted"] is False
    assert {fields["update_rate_hz"] for fields in receipt["after"].values()} == {20}
    reader = CameraProfileReadback(path)
    assert reader.observe("rgb", 320, 180)
    assert reader.observe("depth", 160, 120)
    assert reader.require_ready()["verified_dimensions"] == {
        "rgb": [320, 180], "depth": [160, 120]}
    assert not reader.observe("depth", 320, 240)
    with pytest.raises(ValueError, match="ACTUAL_STREAM_PROFILE_MISMATCH"):
        reader.require_ready()


# 功能：
#   防止只重命名回执配置而沿用原模型，非法标签和不符的尺寸、频率都需拒绝。
# 输入：
#   tmp_path：隔离测试目录。
#   profile：伪装成的另一种配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("profile", ["low-latency", "responsive-control", "native", [], None])
def test_compact_receipt_cannot_be_relabelled_as_another_profile(tmp_path, profile):
    path = readback_fixture(tmp_path, profile="compact-control")
    receipt = json.loads(path.read_text())
    receipt["profile"] = profile
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_PROFILE_"):
        CameraProfileReadback(path)


# 功能：
#   对照紧凑配置前后光学字段及白名单恢复证明，确认其不扩大为其他模型修改。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_compact_configuration_preserves_optics_pose_and_other_model_fields():
    original = source_model()
    compact, receipt = control_stream_profile(original, profile="compact-control")
    assert receipt["only_stream_dimensions_and_rates_changed"] is True
    assert receipt["same_field_of_view_and_aspect_ratio"] is True
    for name, after in camera_configuration(compact).items():
        before = receipt["before"][name]
        for key in ("horizontal_fov_rad", "near_m", "far_m"):
            assert after[key] == before[key]
    validate_camera_profile_choice("compact-control", hashlib.sha256(original).hexdigest())


# 功能：
#   验证响应优先配置绑定自身频率，相同尺寸不能代替配置频率核验或实际到达频率测量。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_responsive_profile_is_explicit_and_cannot_inherit_compact_rate_receipt(tmp_path):
    path = readback_fixture(tmp_path, profile="responsive-control")
    receipt = json.loads(path.read_text())
    assert {fields["update_rate_hz"] for fields in receipt["after"].values()} == {60}
    reader = CameraProfileReadback(path)
    assert reader.observe("depth", 160, 120) and reader.observe("rgb", 320, 180)
    proof = reader.require_ready()
    assert proof["flight_qualification_granted"] is False
    assert "measured separately" in proof["rate_verification"]
    # Same dimensions are not proof of the intended rate/configuration.
    receipt["profile"] = "compact-control"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="PROFILE_STREAMS_CHANGED"):
        CameraProfileReadback(path)


# 功能：
#   拒绝可被不同 XML 消费者解释成不同配置的重复节点，以及额外的嵌套传感器。
# 输入：
#   node_path：需要复制或追加的 XML 节点位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("node_path", ["model", "camera", "image", "width", "update_rate",
                                      "nested-model"])
def test_ambiguous_xml_nodes_are_rejected(node_path):
    root = ET.fromstring(source_model())
    sensor = root.find("model/link/sensor")
    if node_path == "model":
        root.append(ET.fromstring(ET.tostring(root.find("model"))))
    elif node_path == "nested-model":
        nested = ET.SubElement(root.find("model"), "model", name="hidden")
        nested.append(ET.fromstring(ET.tostring(root.find("model/link"))))
    else:
        parent = {"camera": sensor, "update_rate": sensor,
                  "image": sensor.find("camera"), "width": sensor.find("camera/image")}[node_path]
        parent.append(ET.fromstring(ET.tostring(parent.find(node_path))))
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_"):
        control_stream_profile(ET.tostring(root))


# 功能：
#   确认超大整数尺寸和 DTD 输入稳定拒绝，不发生浮点转换溢出或实体展开。
# 输入：
#   mode：需要注入的非法 XML 形式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["huge-width", "dtd", "underscored-width"])
def test_invalid_numeric_or_declared_xml_is_rejected(mode):
    root = ET.fromstring(source_model())
    if mode == "huge-width":
        root.find("model/link/sensor/camera/image/width").text = "9" * 400
    elif mode == "underscored-width":
        root.find("model/link/sensor/camera/image/width").text = "1_920"
    content = ET.tostring(root)
    if mode == "dtd":
        content = b'<!DOCTYPE sdf [<!ENTITY ignored "value">]>' + content
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_"):
        camera_configuration(content)


# 功能：
#   确认回执中的重复键不能被静默覆盖后用于模型回读。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_duplicate_receipt_keys_are_rejected(tmp_path):
    path = readback_fixture(tmp_path)
    path.write_text(path.read_text()[:-1] + ',"installed_assets_modified":false}')
    with pytest.raises(ValueError):
        CameraProfileReadback(path)


# 功能：
#   验证回执不能改写来源摘要、来源清单或等价性声明，而仍仅凭模型尺寸通过。
# 输入：
#   tmp_path：隔离测试目录。
#   changes：对合法回执注入的内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"source_model_sha256": "bad"}, {"source_files": {}}, {"overlay_files": {}},
    {"image_equivalence_claimed": True}, {"same_field_of_view_and_aspect_ratio": False},
    {"before": {}}, {"source_bundle_bytes": True},
])
def test_inconsistent_receipt_claims_are_rejected(tmp_path, changes):
    path = readback_fixture(tmp_path)
    receipt = json.loads(path.read_text())
    path.write_text(json.dumps({**receipt, **changes}))
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_"):
        CameraProfileReadback(path)


# 功能：
#   确认模型以外的包资源也参与回读绑定，新增的未列举文件不能被忽略。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_unlisted_overlay_resource_is_rejected(tmp_path):
    path = readback_fixture(tmp_path)
    (path.parent / "models/OakD-Lite/unlisted.dae").write_bytes(b"new mesh")
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_"):
        CameraProfileReadback(path)


# 功能：
#   模拟配置派生期间源包发生增删改，确认创建输出前会重新核对来源。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：限定当前测试内的配置派生阶段注入。
# 输出：
#   None：不返回业务数据。
def test_source_bundle_change_during_profile_derivation_is_rejected(tmp_path, monkeypatch):
    from dronedream_agent_core import simulation_camera_profile as module

    models, output = tmp_path / "models", tmp_path / "run"
    bundle = models / "OakD-Lite"
    bundle.mkdir(parents=True)
    source = source_model()
    (bundle / "model.sdf").write_bytes(source)
    original = module.control_stream_profile

    # 功能：
    #   生成合法配置后改变源目录，模拟首次枚举后的资产更新。
    # 输入：
    #   content：原模型字节。
    #   profile：所选相机配置。
    # 输出：
    #   result：修改源目录前生成的派生结果。
    def change_bundle(content, *, profile):
        result = original(content, profile=profile)
        (bundle / "late-resource.dae").write_bytes(b"late mesh")
        return result

    monkeypatch.setattr(module, "control_stream_profile", change_bundle)
    with pytest.raises(ValueError, match="SIMULATION_CAMERA_SOURCE_CHANGED"):
        prepare_camera_profile(source_models=models, output=output,
                               expected_source_sha256=hashlib.sha256(source).hexdigest())
    assert not output.exists()


# 功能：
#   验证读取到的尺寸说明是独立副本，调用方不能通过修改说明使错误尺寸得到授权。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_readback_expected_dimensions_are_not_mutable_authority(tmp_path):
    reader = CameraProfileReadback(readback_fixture(tmp_path))
    expected = reader.expected
    expected["rgb"] = (1, 1)
    assert reader.expected["rgb"] == (640, 360)
    assert not reader.observe("rgb", 1, 1)


# 功能：
#   确认错误类型的流名称会形成持续失败，而不是在回调里抛出未分类的哈希类型异常。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_unhashable_stream_name_fails_without_callback_exception(tmp_path):
    reader = CameraProfileReadback(readback_fixture(tmp_path))
    assert reader.observe([], 640, 360) is False
    with pytest.raises(ValueError, match="ACTUAL_STREAM_PROFILE_MISMATCH"):
        reader.require_ready()


# 功能：
#   验证已列入回执的网格发生改变或丢失时，回读也必须失败而不只是验证模型文件。
# 输入：
#   tmp_path：隔离测试目录。
#   missing：是否删除网格，否则替换成不同内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("missing", [False, True])
def test_listed_mesh_must_still_match_receipt(tmp_path, missing):
    receipt = readback_fixture(tmp_path, resources={"meshes/camera.dae": b"native mesh"})
    mesh = receipt.parent / "models/OakD-Lite/meshes/camera.dae"
    if missing:
        mesh.unlink()
    else:
        mesh.write_bytes(b"changed mesh")
    with pytest.raises(ValueError, match="OVERLAY_CHANGED"):
        CameraProfileReadback(receipt)


# 功能：
#   用缩小的预算验证目录、总字节和回执限额都在生成输出前生效。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：只对当前测试设置小预算。
#   budget：需要触发的预算类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", ["directories", "bytes", "receipt"])
def test_preparation_budgets_reject_before_output(tmp_path, monkeypatch, budget):
    from dronedream_agent_core import simulation_camera_profile as module

    model = tmp_path / "models/OakD-Lite/model.sdf"
    model.parent.mkdir(parents=True)
    source = source_model()
    model.write_bytes(source)
    if budget == "directories":
        monkeypatch.setattr(module, "_MAX_BUNDLE_ENTRIES", 3)
        for index in range(4):
            (model.parent / f"empty-{index}").mkdir()
    elif budget == "bytes":
        monkeypatch.setattr(module, "_MAX_BUNDLE_BYTES", len(source) - 1)
    else:
        monkeypatch.setattr(module, "_MAX_RECEIPT_BYTES", 256)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        prepare_camera_profile(source_models=tmp_path / "models", output=output,
                               expected_source_sha256=hashlib.sha256(source).hexdigest())
    assert not output.exists()


# 功能：
#   模拟输出目录在来源准备期间被其他操作创建，确认不会合并或覆盖已有内容。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：限定当前测试内的派生阶段注入。
# 输出：
#   None：不返回业务数据。
def test_output_created_during_preparation_is_not_merged(tmp_path, monkeypatch):
    from dronedream_agent_core import simulation_camera_profile as module

    model = tmp_path / "models/OakD-Lite/model.sdf"
    model.parent.mkdir(parents=True)
    source = source_model()
    model.write_bytes(source)
    output = tmp_path / "run"
    original = module.control_stream_profile

    # 功能：
    #   生成配置后创建调用方目录和哨兵文件，模拟独占输出路径被抢先占用。
    # 输入：
    #   content：原始模型字节。
    #   profile：需要派生的配置。
    # 输出：
    #   result：已经计算出的派生字节与回执。
    def occupy_output(content, *, profile):
        result = original(content, profile=profile)
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
        return result

    monkeypatch.setattr(module, "control_stream_profile", occupy_output)
    with pytest.raises(FileExistsError):
        prepare_camera_profile(source_models=tmp_path / "models", output=output,
                               expected_source_sha256=hashlib.sha256(source).hexdigest())
    assert (output / "keep.txt").read_text() == "preserve"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


# 功能：
#   将双流回读结果经实际运行时 JSON 发布器落盘，防止 Python 内存对象测试遗漏协议错误。
# 输入：
#   tmp_path：独立来源、派生和回读文件目录。
#   profile：三种显式训练相机配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("profile", ["low-latency", "compact-control", "responsive-control"])
def test_actual_runtime_publisher_accepts_camera_readback(tmp_path, profile):
    reader = CameraProfileReadback(readback_fixture(tmp_path, profile=profile))
    for kind, size in reader.expected.items():
        assert reader.observe(kind, *size)
    proof = reader.require_ready()
    target = tmp_path / "camera-profile-readback.json"
    publish_runtime_json(target, proof)
    assert json.loads(target.read_text(encoding="utf-8")) == proof
    # 调用方修改内层数组也不能改写后续回读，不能只测试字典的浅层复制。
    proof["verified_dimensions"]["depth"][0] = 0
    assert reader.require_ready()["verified_dimensions"]["depth"][0] > 0
