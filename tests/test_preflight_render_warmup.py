import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from test_simulation_camera_profile import source_model

from dronedream_agent_core import preflight_render_warmup as module


# 功能：
#   用固定测试身份和实际来源摘要生成默认 RGB 诊断实体。
# 输入：
#   无。
# 输出：
#   prepared：诊断 SDF 字节及构造回执。
def rig():
    source = source_model()
    prepared = module.prepare_rig(source, hashlib.sha256(source).hexdigest(), "a" * 32)
    return prepared


# 功能：
#   逐节点确认诊断实体只有来源相机，除诊断名称和主题外不改写原生传感器配置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rig_is_sensor_only_and_preserves_exact_native_sensor_configuration():
    source = source_model()
    content, receipt = rig()
    root = ET.fromstring(content)
    assert root.find("model").get("name") == receipt["entity_name"]
    assert root.findtext("model/static") == "true"
    for forbidden in ("collision", "inertial", "visual", "plugin", "include", "joint"):
        assert not root.findall(".//" + forbidden)
    assert len(root.findall(".//sensor")) == 1
    for before, after in zip(ET.fromstring(source).findall('.//sensor[@type="camera"]'),
                             root.findall(".//sensor"), strict=True):
        topic = after.find("topic")
        assert topic.text.startswith("/dronedream/render-warmup/")
        after.remove(topic)
        assert after.get("name").startswith("dronedream_warmup_")
        after.set("name", before.get("name"))
        assert ET.tostring(after) == ET.tostring(before)
    assert receipt["flight_qualification_granted"] is False
    assert receipt["prepared_streams"] == ["rgb"]
    assert receipt["depth_pipeline_prepared"] is False
    assert set(receipt["streams"]) == {"rgb"}
    assert receipt["source_camera_sha256"] == hashlib.sha256(source).hexdigest()
    assert receipt["rig_sha256"] == hashlib.sha256(content).hexdigest()


# 功能：
#   显式深度准备保留两种相机而不带入飞控、碰撞或可见几何，构造不能冒充执行完成。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_explicit_depth_rig_preserves_both_sensors_but_has_no_flight_components():
    source = source_model()
    content, receipt = module.prepare_rig(source, hashlib.sha256(source).hexdigest(),
                                          "b" * 32, include_depth=True)
    root = ET.fromstring(content)
    assert set(receipt["streams"]) == {"rgb", "depth"}
    assert receipt["depth_preparation_requested"] is True
    assert receipt["depth_pipeline_prepared"] is False  # Construction is not execution.
    for before, after in zip(ET.fromstring(source).findall(".//sensor"),
                             root.findall(".//sensor"), strict=True):
        after.remove(after.find("topic"))
        after.set("name", before.get("name"))
        assert ET.tostring(before) == ET.tostring(after)
    assert all(not root.findall(".//" + key)
               for key in ("collision", "visual", "inertial", "joint", "plugin", "include"))
    with pytest.raises(ValueError, match="DEPTH_CHOICE_INVALID"):
        module.prepare_rig(source, hashlib.sha256(source).hexdigest(), "b" * 32,
                            include_depth="true")


# 功能：
#   来源、实体身份、相对位姿或控制插件存在歧义时，必须拒绝构造诊断实体。
# 输入：
#   fault：本次向相机来源注入的单项错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["hash", "name", "relative-pose", "nan", "model-offset",
                                  "plugin", "frame", "second-link", "duplicate-topic"])
def test_unknown_source_frames_plugins_and_arbitrary_target_names_rejected(fault):
    root = ET.fromstring(source_model())
    identity = "a" * 32
    if fault == "name":
        identity = "my_drone"
    elif fault == "relative-pose":
        root.find(".//sensor/pose").set("relative_to", "another_link")
    elif fault == "nan":
        root.find(".//sensor/pose").text = "nan 0 0 0 0 0"
    elif fault == "model-offset":
        ET.SubElement(root.find("model"), "pose").text = "1 0 0 0 0 0"
    elif fault in {"plugin", "frame"}:
        ET.SubElement(root.find("model"), fault)
    elif fault == "second-link":
        ET.SubElement(root.find("model"), "link", name="foreign")
    elif fault == "duplicate-topic":
        for _ in range(2):
            ET.SubElement(root.find(".//sensor"), "topic").text = "/foreign"
    source = ET.tostring(root)
    digest = "0" * 64 if fault == "hash" else hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError, match="RENDER_WARMUP_"):
        module.prepare_rig(source, digest, identity)


# 功能：
#   检查 24 个视角互不重复且位置固定，并拒绝错误形状或非有限位置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_views_are_bounded_and_only_rotate_the_diagnostic_camera():
    views = module.warmup_views((3., 4., 5.))
    assert len(views) == len(set(views)) == 24
    assert {v[:3] for v in views} == {(3., 4., 5.)}
    assert {v[4] for v in views} == {0., math.pi / 6, -math.pi / 6}
    for bad in ((1., 2.), (True, 2, 3), (1, 2, math.inf), (1, 2, "3")):
        with pytest.raises(ValueError):
            module.warmup_views(bad)


# 功能：
#   构造只用于外形与时序验证的微型图像消息，不提供真实传感器或视觉效果证据。
# 输入：
#   stamp：来源时间的纳秒分量。
#   width：消息宽度，默认与测试流配置匹配。
#   data：可替换的测试图像字节。
# 输出：
#   message：带尺寸、行跨度和来源时钟的图像替身。
def frame(stamp, width=2, data=b"12345678"):
    message = SimpleNamespace(width=width, height=1, step=8, data=data,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nsec=stamp)))
    return message


# 功能：
#   两条配置流都必须具有水位后的两帧，重复来源时钟不能追加准备进度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_warmup_requires_both_real_streams_after_pose_readback_and_unique_times():
    frames = module.WarmupFrames({k: {"width": 2, "height": 1} for k in ("rgb", "depth")})
    for key in frames.streams:
        frames.observe(key, frame(1), 1.)
        frames.observe(key, frame(2), 2.)
    with pytest.raises(TimeoutError):
        frames.require_progress(1.5, timeout=.001)
    for key in frames.streams:
        frames.observe(key, frame(3), 3.)
    assert frames.require_progress(1.5) == {"rgb": 2, "depth": 2}
    frames.observe("depth", frame(3), 4.)
    with pytest.raises(ValueError, match="CLOCK_NOT_ADVANCING"):
        frames.require_progress(2.)


# 功能：
#   错误尺寸、字节数或负来源时钟不能形成有效预热帧。
# 输入：
#   bad：本次构造的畸形图像消息。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad", [frame(1, width=4), frame(1, data=b"short"), frame(-1)])
def test_bad_sensor_messages_never_finish_warmup(bad):
    frames = module.WarmupFrames({"depth": {"width": 2, "height": 1}})
    frames.observe("depth", bad, 1.)
    with pytest.raises(ValueError, match="STREAM_INVALID"):
        frames.require_progress(0.)


# 功能：
#   提供不启动仿真的原生消息、服务和进度替身，记录实体操作与退订顺序。
# 输入：
#   monkeypatch：替换模块、服务和观察方法的测试工具。
#   fault：可选的飞机已存在、位姿、图像或删除错误。
# 输出：
#   native：请求历史、实体表、退订列表和进度水位组成的四元组。
def fake_native(monkeypatch, fault):
    from google.protobuf import text_format

    classes, requests, entities = {}, [], {}
    # 功能：
    #   构造带对应类型描述符的消息替身类，供服务参数和解析测试使用。
    # 输入：
    #   kind：Gazebo 消息类型名称。
    # 输出：
    #   message_type：本次生成的消息类。
    def message(kind):
        class Message:
            DESCRIPTOR = SimpleNamespace(full_name="gz.msgs." + kind)
            # 功能：
            #   初始化测试消息的位置、朝向和调用方提供的字段。
            # 输入：
            #   self：待初始化的消息替身。
            #   kw：本次消息需要的关键字字段。
            # 输出：
            #   None：不返回业务数据。
            def __init__(self, **kw):
                self.position = SimpleNamespace(x=0, y=0, z=0)
                self.orientation = SimpleNamespace(w=1, x=0, y=0, z=0)
                if kind in {"EntityFactory", "Scene"}:
                    self.pose = classes["Pose"]()
                self.__dict__.update(kw)
        message_type = Message
        return message_type

    for kind, file in (("Pose", "pose"), ("Boolean", "boolean"), ("Empty", "empty"),
                       ("Entity", "entity"), ("EntityFactory", "entity_factory"),
                       ("Image", "image"), ("Scene", "scene"), ("Pose_V", "pose_v")):
        cls = message(kind)
        cls.MODEL = 2
        classes[kind] = cls
        fake = ModuleType("gz.msgs10." + file + "_pb2")
        setattr(fake, kind, cls)
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    unsubscribed = []
    node = SimpleNamespace(subscribe=lambda *a: True,
                           unsubscribe=lambda topic: unsubscribed.append(topic) is None)
    transport = ModuleType("gz.transport13")
    transport.Node = lambda: node
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    current = [None]
    monkeypatch.setattr(text_format, "MessageToString",
                        lambda req: current.__setitem__(0, req) or "req")
    replies = []
    monkeypatch.setattr(text_format, "Parse", lambda text, response: replies.pop())

    # 功能：
    #   模拟固定世界的查询、创建、移动和删除服务，拒绝任意非自有实体移动。
    # 输入：
    #   command：生产代码实际组装的命令参数。
    #   kw：生产代码提供的超时和输出选项。
    # 输出：
    #   process：成功退出并待由替身解析应答的进程结果。
    def run(command, **kw):
        service = command[command.index("-s")+1].split("/")[-1]
        if command[command.index("-s")+1].endswith("scene/info"):
            service = "scene"
        req = current[0]
        requests.append((service, getattr(req, "name", None)))
        if service == "scene":
            if fault == "aircraft-present":
                entities["my_drone"] = SimpleNamespace(name="my_drone")
            reply = SimpleNamespace(model=list(entities.values()))
        elif service == "create":
            entities[req.name] = SimpleNamespace(name=req.name, id=12, pose=req.pose)
            reply = SimpleNamespace(data=True)
        elif service == "set_pose":
            assert req.name.startswith("dronedream_render_warmup_") and req.id == 12
            entities[req.name].pose = req
            if fault == "pose-mismatch":
                req.position.x += 2
            reply = SimpleNamespace(data=True)
        elif service == "remove":
            assert req.name.startswith("dronedream_render_warmup_") and req.id == 12
            if fault != "remove-rejected":
                entities.pop(req.name)
            reply = SimpleNamespace(data=fault != "remove-rejected")
        else:
            pytest.fail("unexpected RPC: " + service)
        replies.append(reply)
        process = SimpleNamespace(returncode=0, stdout="reply")
        return process

    monkeypatch.setattr(module.subprocess, "run", run)
    progress = []
    # 功能：
    #   检查首次与后续帧预算，并按测试故障返回流计数或缺帧错误。
    # 输入：
    #   self：生产预热帧缓存。
    #   after：位姿回读后的等待水位。
    #   timeout：本次帧等待预算。
    # 输出：
    #   counts：每条配置流的两帧测试计数。
    def fresh(self, after, timeout=5.):
        assert 29. < timeout <= 30. if not progress else 0 < timeout <= 5.
        progress.append(after)
        if fault == "missing-frame":
            raise TimeoutError("missing image")
        counts = {key: 2 for key in self.streams}
        return counts
    monkeypatch.setattr(module.WarmupFrames, "require_progress", fresh)
    # 功能：
    #   单独模拟实时位姿回读，故障时不以场景创建位姿替代真实观察。
    # 输入：
    #   self：生产位姿接收器。
    #   xyz：需要确认的世界位置。
    #   quaternion：需要确认的 wxyz 朝向。
    #   after：设置位姿后的接收水位。
    #   timeout：本次位姿等待预算。
    # 输出：
    #   observation：匹配请求的测试位姿。
    def observed(self, xyz, quaternion, after, timeout=3.):
        # 场景查询中的创建位姿不随移动刷新，必须调用独立的实时回读接口。
        if fault == "pose-mismatch":
            raise TimeoutError("RENDER_WARMUP_LIVE_POSE_NOT_CONFIRMED")
        observation = {"position": xyz, "orientation_wxyz": quaternion}
        return observation
    monkeypatch.setattr(module.WarmupPose, "require_match", observed)
    native = requests, entities, unsubscribed, progress
    return native


# 功能：
#   验证正常和各失败路径只移动自有 RGB 实体、按实际观察删除，并始终退订和记录状态。
# 输入：
#   tmp_path：独占的测试来源与输出目录。
#   monkeypatch：安装原生替身的测试工具。
#   fault：本次生命周期故障类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", [None, "aircraft-present", "pose-mismatch", "missing-frame",
                                  "remove-rejected"])
def test_lifecycle_only_moves_owned_rig_and_requires_observed_removal(tmp_path, monkeypatch, fault):
    requests, entities, unsubscribed, progress = fake_native(monkeypatch, fault)
    source = tmp_path / "camera.sdf"
    source.write_bytes(source_model())
    args = dict(gz_binary="gz", world="school", flight_vehicle="my_drone", source=source,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        position=(3., 4., 5.), output=tmp_path / "output")
    if fault:
        with pytest.raises(RuntimeError, match="RENDER_WARMUP_FAILED"):
            module.execute_warmup(**args)
    else:
        module.execute_warmup(**args)
        assert len(progress) == 24
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["complete"] is (fault is None)
    assert receipt["flight_qualification_granted"] is False
    assert receipt["full_shader_coverage_claimed"] is False
    assert receipt["no_pixels_sent_to_policy"] is True
    if fault == "aircraft-present":
        assert requests == [("scene", None)] and not unsubscribed
        assert receipt["flight_vehicle_absent"] is False
    else:
        assert len(unsubscribed) == 2
        assert any(service == "remove" for service, _ in requests)
        assert bool(entities) is (fault == "remove-rejected")


# 功能：
#   核对真实适配器源码中预热门禁先于生成飞机调用，不把测试替身当成飞行验收。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_adapter_cannot_spawn_aircraft_before_warmup_success():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/dronedream_agent_core/gazebo_adapter.py").read_text(encoding="utf-8")
    gate = source.index('raise SimulationRuntimeError("PREFLIGHT_RENDER_WARMUP_NOT_CONFIRMED")')
    assert source.index('"dronedream_agent_core.preflight_render_warmup"') < gate
    assert source.index("spawn_evidence = _spawn_entity(") > gate


# 功能：
#   深度准备成功或失败都保留自有实体至仿真退出，但必须停止并排空订阅。
# 输入：
#   tmp_path：独占的测试来源与输出目录。
#   monkeypatch：安装原生替身的测试工具。
#   fault：本次深度准备的可选故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", [None, "pose-mismatch", "missing-frame"])
def test_depth_resources_retained_without_subscriptions_even_when_preparation_fails(
    tmp_path, monkeypatch, fault,
):
    requests, entities, unsubscribed, progress = fake_native(monkeypatch, fault)
    source = tmp_path / "camera.sdf"
    source.write_bytes(source_model())
    kwargs = dict(gz_binary="gz", world="school", flight_vehicle="my_drone", source=source,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), position=(3., 4., 5.),
        output=tmp_path / "output", include_depth=True)
    if fault:
        with pytest.raises(RuntimeError, match="RENDER_WARMUP_FAILED"):
            module.execute_warmup(**kwargs)
    else:
        module.execute_warmup(**kwargs)
        assert len(progress) == 24
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["complete"] is (fault is None)
    assert receipt["depth_pipeline_prepared"] is (fault is None)
    assert receipt["retained_until_simulation_exit"] is True
    assert receipt["removed_and_absence_observed"] is False
    assert receipt["rendering_inactivity_verified"] is False
    assert receipt["flight_qualification_granted"] is False
    assert receipt["native_subscriptions"]["complete"] is True
    assert receipt["no_pixels_sent_to_policy"] is True
    assert module.warmup_receipt_ready(receipt,
        source_sha256=kwargs["expected_sha256"], include_depth=True) is (fault is None)
    assert len(entities) == 1 and len(unsubscribed) == 3
    assert not any(service == "remove" for service, _ in requests)


# 功能：
#   来源摘要、模式和终态字段任一缺失或冲突均不能通过预热门禁。
# 输入：
#   include_depth：本次验证的 RGB 或 RGB 加深度模式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("include_depth", [False, True])
def test_receipt_gate_rejects_wrong_mode_incomplete_shutdown_and_source_changes(include_depth):
    receipt = {"complete": True, "source_camera_sha256": "a" * 64,
        "depth_preparation_requested": include_depth, "depth_pipeline_prepared": include_depth,
        "retained_until_simulation_exit": include_depth,
        "removed_and_absence_observed": not include_depth,
        "prepared_streams": ["rgb", "depth"] if include_depth else ["rgb"],
        "native_subscriptions": {"complete": True, "subscribed_count": 3 if include_depth else 2,
            "unsubscribed_count": 3 if include_depth else 2,
            "active_callbacks_at_close": 0, "errors": []}, "flight_vehicle_absent": True,
        "no_pixels_sent_to_policy": True, "flight_qualification_granted": False}
    kwargs = dict(source_sha256="a" * 64, include_depth=include_depth)
    assert module.warmup_receipt_ready(receipt, **kwargs)
    assert not module.warmup_receipt_ready(
        receipt, **{**kwargs, "include_depth": not include_depth})
    assert not module.warmup_receipt_ready(receipt, **{**kwargs, "source_sha256": "b" * 64})
    for key in receipt:
        assert not module.warmup_receipt_ready({**receipt, key: None}, **kwargs)
    assert not module.warmup_receipt_ready([], **kwargs)
    assert not module.warmup_receipt_ready({**receipt, "native_subscriptions": []}, **kwargs)


# 功能：
#   自有实体新位姿必须有有限坐标和单位四元数，朝向符号取反仍表示同一姿态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_live_pose_requires_identity_recency_finite_and_normalized_values():
    # 功能：
    #   构造可单独改写名称、身份、位置及四元数的原生位姿替身。
    # 输入：
    #   name：实体名称。
    #   identity：原生实体 ID。
    #   x：位置的 x 分量。
    #   w：四元数的实部分量。
    # 输出：
    #   candidate：包含一项位姿的消息。
    def message(name="owned", identity=12, x=3, w=1):
        candidate = SimpleNamespace(pose=[SimpleNamespace(name=name, id=identity,
            position=SimpleNamespace(x=x, y=4, z=5),
            orientation=SimpleNamespace(w=w, x=0, y=0, z=0))])
        return candidate

    for candidate, failure in ((message(name="my_drone"), TimeoutError),
        (message(identity=99), TimeoutError), (message(x=math.nan), ValueError),
        (message(w=2), ValueError), (message(x=4), TimeoutError)):
        readback = module.WarmupPose("owned")
        readback.entity_id = 12
        readback.observe(candidate)
        with pytest.raises(failure):
            readback.require_match((3, 4, 5), (1, 0, 0, 0), -1, timeout=.001)
    readback = module.WarmupPose("owned")
    readback.entity_id = 12
    readback.observe(message(w=-1))
    assert readback.require_match((3, 4, 5), (1, 0, 0, 0), -1)["position"] == (3, 4, 5)
    with pytest.raises(ValueError, match="WAIT_BOUNDARY"):
        readback.require_match((3, 4, 5), (1, 0, 0, 0), float("inf"), timeout=.001)


# 功能：
#   调用者修改流配置后，既有帧缓存仍按自己持有的尺寸判断消息。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_frame_buffer_owns_its_dimensions():
    streams = {"rgb": {"width": 2, "height": 1}}
    frames = module.WarmupFrames(streams)
    streams["rgb"]["width"] = 4
    frames.observe("rgb", frame(1), 1.)
    frames.observe("rgb", frame(2), 2.)
    assert frames.require_progress(0.) == {"rgb": 2}


# 功能：
#   非有限、布尔和字符串接收时钟不能通过帧数累计完成准备。
# 输入：
#   received：本次故意构造的非法接收时间。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("received", [float("inf"), float("nan"), True, "1"])
def test_invalid_receive_clocks_cannot_finish_render_preparation(received):
    frames = module.WarmupFrames({"rgb": {"width": 2, "height": 1}})
    frames.observe("rgb", frame(1), received)
    frames.observe("rgb", frame(2), received)
    with pytest.raises(ValueError, match="STREAM_INVALID"):
        frames.require_progress(0., timeout=.001)


# 功能：
#   即使两帧已经就绪，非法等待预算也不能绕过边界检查。
# 输入：
#   timeout：本次故意构造的非法等待秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [True, float("inf"), float("nan"), -1, "1"])
def test_invalid_wait_budget_is_rejected_even_if_frames_are_already_ready(timeout):
    frames = module.WarmupFrames({"rgb": {"width": 2, "height": 1}})
    frames.observe("rgb", frame(1), 1.)
    frames.observe("rgb", frame(2), 2.)
    with pytest.raises(ValueError, match="WAIT_BOUNDARY"):
        frames.require_progress(0., timeout=timeout)


# 功能：
#   超出双精度范围的整数位置须作为受控输入错误拒绝，不能触发隐式转换溢出。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_oversized_integer_position_is_a_controlled_input_error():
    with pytest.raises(ValueError, match="POSITION_INVALID"):
        module.warmup_views((2**4096, 0, 0))
