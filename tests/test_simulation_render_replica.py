"""Deployment fixture tests; do not create a simulator, aircraft or model weights."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from clock_fixtures import isolate_time

from dronedream_agent_core import simulation_render_replica as module


# 功能：
#   建立包含实际相机 XML 的隔离来源夹具，用替身替代原生构建验证，不启动仿真。
# 输入：
#   tmp_path：本次测试的独立目录。
#   monkeypatch：限定当前测试内的原生构建验证替换。
# 输出：
#   arguments：世界、飞机、相机、服务器和运行专属输出路径构成的准备参数。
def inputs(tmp_path, monkeypatch):
    from test_simulation_camera_profile import source_model
    runtime = tmp_path / "native"
    runtime.mkdir()
    (runtime / module.RECEIPT).write_text("fixture")
    monkeypatch.setattr(module, "validate_replica_runtime", lambda root: {
        "sensors_plugin": "/fixture/Sensors.so", "binaries": {"fixture": "a" * 64}})
    world, vehicle, config = (tmp_path / n for n in ("world.sdf", "vehicle.sdf", "server.config"))
    world.write_text('<sdf><world name="campus"><gravity>0 0 -9.8</gravity></world></sdf>')
    vehicle.write_text('<sdf><model name="drone"><include merge="true">'
                       '<uri>model://OakD-Lite</uri></include></model></sdf>')
    camera = tmp_path / "models/OakD-Lite/model.sdf"
    camera.parent.mkdir(parents=True)
    tree = ET.fromstring(source_model())
    depth = tree.find("model/link/sensor[@name='StereoOV7251']")
    ET.SubElement(depth, "topic").text = "depth_camera"
    camera.write_bytes(ET.tostring(tree))
    (camera.parent / "model.config").write_text('<model><sdf version="1.9">model.sdf</sdf></model>')
    config.write_text('''<server_config><plugins>
      <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
      <plugin name="dronedream::Magnetometer" filename="/actual/sensor.so">
        <wire>keep</wire></plugin>
      <plugin entity_name="*" entity_type="world" name="gz::sim::systems::Sensors"
        filename="gz-sim-sensors-system"><render_engine>ogre2</render_engine></plugin>
      </plugins></server_config>''')
    arguments = dict(runtime_root=runtime, server_config=config, world_sdf=world,
        vehicle_sdf=vehicle, camera_sdf=camera, resource_paths=(tmp_path / "models",),
        world_name="campus", vehicle_name="drone", output=tmp_path / "isolated")
    return arguments


# 功能：
#   对照全部输入字节和非渲染系统，验证只在新目录替换 Sensors 系统并保存来源绑定。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：当前测试的原生构建验证替身。
# 输出：
#   None：不返回业务数据。
def test_only_render_system_changes_and_installed_inputs_remain_identical(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    originals = {name: path.read_bytes() for name, path in args.items()
                 if isinstance(path, Path) and path.is_file()}
    result = module.prepare_render_replica(**args)
    for name, content in originals.items():
        assert args[name].read_bytes() == content
    plugins = ET.parse(args["output"] / "server.config").findall("plugins/plugin")
    assert [p.get("name") for p in plugins] == ["gz::sim::systems::Physics",
        "dronedream::Magnetometer", "dronedream::RenderSceneSource"]
    assert plugins[1].findtext("wire") == "keep"
    assert plugins[-1].findtext("epoch") == result["epoch"]
    assert result["non_render_systems_preserved"]
    assert result["qualification_granted"] is False
    assert result["installed_runtime_modified"] is False
    assert result["rgb_topic"] == "/world/campus/model/drone/link/camera_link/sensor/IMX214/image"
    assert result["depth_topic"] == "/depth_camera"
    assert result["command"][-2:] == ["--duration-seconds", "0"]
    assert len(result["input_sdf_files"]) == 4
    assert json.loads((args["output"] / "deployment.json").read_text()) == result
    with pytest.raises(FileExistsError):
        module.prepare_render_replica(**args)


# 功能：
#   逐项注入冲突世界、模型包含、相机主题和旧渲染系统，确认生成输出前就拒绝。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：当前测试的原生构建验证替身。
#   fault：需要注入的来源选择或系统冲突类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["world-name", "world-system", "nested-render", "missing-include",
    "remote-include", "cycle", "camera-not-selected", "depth-topic", "rgb-topic", "other-variant",
    "duplicate-system", "missing-system", "other-engine", "old-cache", "wrong-world-scope"])
def test_ambiguous_or_old_assets_are_rejected_before_output(tmp_path, monkeypatch, fault):
    args = inputs(tmp_path, monkeypatch)
    if fault == "world-name":
        args["world_name"] = "other"
    elif fault == "world-system":
        root = ET.parse(args["world_sdf"])
        ET.SubElement(root.find("world"), "plugin", name="explicit")
        root.write(args["world_sdf"])
    elif fault == "nested-render":
        root = ET.parse(args["camera_sdf"])
        ET.SubElement(root.find("model"), "plugin", name="gz::sim::systems::Sensors")
        root.write(args["camera_sdf"])
    elif fault in ("missing-include", "remote-include", "cycle"):
        root = ET.parse(args["vehicle_sdf"])
        root.find("model/include/uri").text = {
            "missing-include": "model://missing", "remote-include": "https://example.org/model",
            "cycle": "vehicle.sdf"}[fault]
        root.write(args["vehicle_sdf"])
    elif fault == "camera-not-selected":
        other = tmp_path / "other-camera.sdf"
        other.write_bytes(args["camera_sdf"].read_bytes())
        args["camera_sdf"] = other
    elif fault in ("depth-topic", "rgb-topic"):
        root = ET.parse(args["camera_sdf"])
        if fault == "depth-topic":
            root.find("model/link/sensor[@name='StereoOV7251']/topic").text = "other"
        else:
            ET.SubElement(root.find("model/link/sensor[@name='IMX214']"), "topic").text = "other"
        root.write(args["camera_sdf"])
    elif fault == "other-variant":
        (args["camera_sdf"].parent / "model.config").write_text(
            '<model><sdf version="1.9">different.sdf</sdf></model>')
    else:
        root = ET.parse(args["server_config"])
        plugins = root.find("plugins")
        sensor = plugins.find("plugin[@name='gz::sim::systems::Sensors']")
        if fault == "duplicate-system":
            plugins.append(ET.fromstring(ET.tostring(sensor)))
        elif fault == "missing-system":
            plugins.remove(sensor)
        elif fault == "other-engine":
            sensor.find("render_engine").text = "ogre"
        elif fault == "old-cache":
            ET.SubElement(plugins, "plugin", filename="libdronedream-render-preparation.so")
        else:
            sensor.set("entity_type", "model")
        root.write(args["server_config"])
    with pytest.raises(ValueError, match="REPLICA_"):
        module.prepare_render_replica(**args)
    assert not args["output"].exists()


# 功能：
#   证明资源搜索顺序实际选择另一个相机文件时，不能用调用方指定的相机冒充来源。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：当前测试的原生构建验证替身。
# 输出：
#   None：不返回业务数据。
def test_resource_order_cannot_silently_select_old_camera(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    earlier = tmp_path / "earlier/OakD-Lite/model.sdf"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(args["camera_sdf"].read_bytes())
    args["resource_paths"] = (tmp_path / "earlier", *args["resource_paths"])
    with pytest.raises(ValueError, match="ACTUAL_INCLUDE_GRAPH"):
        module.prepare_render_replica(**args)


# 功能：
#   保留 SDFormat 裸模型名沿注册资源目录解析的合法兼容路径。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：当前测试的原生构建验证替身。
# 输出：
#   None：不返回业务数据。
def test_native_bare_model_name_uses_registered_resource_root(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    args["vehicle_sdf"].write_text('<sdf><model name="drone"><include merge="true">'
        '<uri>OakD-Lite</uri></include></model></sdf>')
    result = module.prepare_render_replica(**args)
    assert str(args["camera_sdf"].resolve()) in result["input_sdf_files"]


# 功能：
#   验证训练配置不能混用隔离渲染、旧缓存和预热路径，也不能丢失相机来源绑定。
# 输入：
#   changes：注入的冲突训练字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [dict(preflight_render_warmup=True),
    dict(preflight_depth_warmup=True), dict(render_preparation_runtime="cache"),
    dict(render_cache_bundle="cache"), dict(visual_package=None),
    dict(simulation_camera_profile="native", camera_source_model_sha256=None)])
def test_training_config_cannot_mix_render_paths(changes):
    from dronedream_agent_core.training.px4_environment import ASSET_FIELDS, Px4TrainingConfig
    config = dict(runner="run.py", output_root="runs", mission_id="test",
        expert_role="local-navigation-policy", **{name: "fixture" for name in ASSET_FIELDS},
        asset_sha256={}, minimum_enu_m=(0, 0, 0), maximum_enu_m=(1, 1, 1),
        visual_package="visual-only", simulation_camera_profile="low-latency",
        camera_source_model_sha256="a" * 64, render_replica_runtime="isolated")
    assert Px4TrainingConfig(**config).render_replica_runtime == Path("isolated")
    expected = ("non-native camera profile requires explicit visual training input"
                if changes.get("visual_package", "present") is None else "isolated rendering")
    with pytest.raises(ValueError, match=expected):
        Px4TrainingConfig(**{**config, **changes})


# 功能：
#   按原生注册后的送帧顺序检查双路来源、布局及关闭门控，坏帧后必须重新积累连续好帧。
# 输入：
#   monkeypatch：将原生消息、传输和时间替换为确定性的本地夹具。
#   fault：送帧或退订故障；recovered 表示坏帧后已补足五帧合法数据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", [None, "no-depth", "raw-stream", "wrong-epoch", "replay",
    "expired", "shape", "format", "truncated", "unsubscribe", "gap", "recovered"])
def test_startup_readiness_requires_both_current_source_bound_images(monkeypatch, fault):
    import sys
    import time
    from types import ModuleType
    from types import SimpleNamespace as NS

    from test_sensor_frame_clock import EPOCH, frame

    class Image:
        DESCRIPTOR = NS(fields_by_name={"pixel_format_type": NS(enum_type=NS(values_by_name={
            "RGB_INT8": NS(number=3), "R_FLOAT32": NS(number=6)}))})

    image_module = ModuleType("gz.msgs10.image_pb2")
    image_module.Image = Image
    monkeypatch.setitem(sys.modules, image_module.__name__, image_module)
    received = []
    initial = time.time_ns()
    now = [initial]
    monotonic = [10.]

    # 功能：
    #   将夹具单调时钟推进一毫秒，控制等待循环而不依赖机器运行速度。
    # 输入：
    #   无。
    # 输出：
    #   current：推进后的夹具单调钟秒数。
    def tick():
        monotonic[0] += .001
        current = monotonic[0]
        return current

    isolate_time(monkeypatch, module, monotonic=tick, time_ns=lambda: now[0])

    callbacks = []

    # 功能：
    #   保存注册主题和回调，返回成功后才允许 pump 送帧，不在注册函数内交付图像。
    # 输入：
    #   _type：本替身不使用的消息类型。
    #   topic：订阅主题。
    #   callback：生命周期包装后的回调。
    # 输出：
    #   accepted：订阅成功标记 True。
    def subscribe(_type, topic, callback):
        received.append(topic)
        callbacks.append((topic, callback))
        accepted = True
        return accepted

    # 功能：
    #   根据指定故障构造一串来源绑定图像，模拟重放、过期、格式错误及故障后恢复。
    # 输入：
    #   topic：需要送帧的主题。
    #   callback：已经确认注册的消息处理器。
    # 输出：
    #   None：不返回业务数据。
    def emit(topic, callback):
        if fault == "no-depth" and topic == "/depth":
            return
        frame_count = 9 if fault == "recovered" else 7 if fault == "gap" else 5
        for index in range(frame_count):
            seq = 1 if fault == "replay" else index + 1
            now[0] = initial + (index + 1) * 1_000_000
            source = now[0] - (300_000_000 if fault == "expired" else 100_000_000)
            message = frame(sequence=str(seq), source_unix_ns=str(source),
                            simulation_ns=str(seq * 1_000_000))
            message.header.stamp.nsec = seq * 1_000_000
            if fault == "raw-stream":
                message.header.data = []
            elif fault == "wrong-epoch" or (fault in ("gap", "recovered") and index == 2):
                message.header.data[0].value = ["c" * 64]
            message.width = 3 if fault == "shape" else 4
            message.height = 4
            message.step = 12 if topic == "/rgb" else 16
            message.pixel_format_type = (3 if topic == "/rgb" else 6)
            if fault == "format":
                message.pixel_format_type = 99
            message.data = b"x" * (message.step * message.height - (fault == "truncated"))
            callback(message)

    # 功能：
    #   在等待循环的休眠位置送达已注册主题的数据，每批回调只交付一次。
    # 输入：
    #   _seconds：本夹具不实际休眠的等待秒数。
    # 输出：
    #   None：不返回业务数据。
    def pump(_seconds):
        pending = list(callbacks)
        callbacks.clear()
        for topic, callback in pending:
            emit(topic, callback)

    monkeypatch.setattr(module.time, "sleep", pump)

    transport = ModuleType("gz.transport13")
    transport.Node = lambda: NS(subscribe=subscribe, unsubscribe=lambda _: fault != "unsubscribe")
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    result = module.wait_for_replica_images({"epoch": EPOCH, "rgb_topic": "/rgb",
        "depth_topic": "/depth", "camera_configuration": {
            name: {"width": 4, "height": 4} for name in ("IMX214", "StereoOV7251")}},
        timeout_seconds=.04)
    assert result["ready"] is (fault in (None, "recovered")), result
    assert result["qualification_granted"] is False
    assert result["startup_only_not_control_authority"] is True
    assert received == ["/rgb", "/depth"]


# 功能：
#   在相机参数解析或包含图完成后修改来源，验证部署不得把旧摘要与新内容混用。
# 输入：
#   tmp_path：隔离部署目录。
#   monkeypatch：限制当前测试内的解析阶段故障注入。
#   changed：需要在准备途中替换的来源。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changed", ["camera", "world"])
def test_source_change_during_preparation_is_rejected_before_output(tmp_path, monkeypatch, changed):
    args = inputs(tmp_path, monkeypatch)
    original = module.camera_configuration

    # 功能：
    #   读取合法相机参数后改动一个已检查来源，制造解析内容与最终来源身份不一致。
    # 输入：
    #   content：相机原始 XML 字节。
    # 输出：
    #   result：修改磁盘来源之前解析出的相机配置。
    def changed_source(content):
        result = original(content)
        path = args["camera_sdf" if changed == "camera" else "world_sdf"]
        path.write_bytes(path.read_bytes() + b"\n<!-- source changed -->")
        return result

    monkeypatch.setattr(module, "camera_configuration", changed_source)
    with pytest.raises(ValueError, match="REPLICA_.*CHANGED"):
        module.prepare_render_replica(**args)
    assert not args["output"].exists()


# 功能：
#   确认报告在独占创建前已经完成严格序列化，非法数据不留下半份 JSON 文件。
# 输入：
#   tmp_path：隔离输出目录。
# 输出：
#   None：不返回业务数据。
def test_invalid_report_does_not_leave_partial_output(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        module._write_new(path, {"late-field": float("nan")})
    assert not path.exists()


# 功能：
#   确认非法启动超时在加载原生库或创建订阅前拒绝，避免无界或无意义的就绪等待。
# 输入：
#   timeout：非法等待秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [True, None, "1", 0, -1, float("nan"), float("inf")])
def test_invalid_startup_timeout_is_rejected_before_native_import(timeout):
    with pytest.raises(ValueError, match="REPLICA_STARTUP_TIMEOUT_INVALID"):
        module.wait_for_replica_images({}, timeout_seconds=timeout)


# 功能：
#   验证缺失纪元、重复主题和非法尺寸在原生库导入之前拒绝，不留下半初始化订阅。
# 输入：
#   change：对合法启动配置注入的字段变化。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", [
    {"epoch": None}, {"rgb_topic": "/depth"}, {"depth_topic": "\0"},
    {"camera_configuration": {}},
    {"camera_configuration": {"IMX214": {"width": True, "height": 4},
                              "StereoOV7251": {"width": 4, "height": 4}}},
])
def test_invalid_deployment_does_not_create_native_resources(change):
    deployment = {"epoch": "a" * 64, "rgb_topic": "/rgb", "depth_topic": "/depth",
                  "camera_configuration": {name: {"width": 4, "height": 4}
                                           for name in ("IMX214", "StereoOV7251")}}
    with pytest.raises(ValueError, match="REPLICA_STARTUP_DEPLOYMENT_INVALID"):
        module.wait_for_replica_images({**deployment, **change}, timeout_seconds=1)


# 功能：
#   验证启动命令行拒绝重复 JSON 键，且未解析成功时不触发原生就绪检查或创建回执。
# 输入：
#   tmp_path：隔离的输入与输出目录。
#   monkeypatch：限定当前测试内的命令行与就绪入口替换。
# 输出：
#   None：不返回业务数据。
def test_cli_rejects_duplicate_deployment_keys_before_startup(tmp_path, monkeypatch):
    import sys

    source, receipt = tmp_path / "deployment.json", tmp_path / "receipt.json"
    source.write_text('{"epoch":"one","epoch":"two"}')
    calls = []
    monkeypatch.setattr(sys, "argv", ["replica", "--deployment", str(source),
                                     "--receipt", str(receipt)])
    monkeypatch.setattr(module, "wait_for_replica_images", lambda value: calls.append(value))
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        module.main()
    assert calls == [] and not receipt.exists()


# 功能：
#   确认世界 XML 中的 DTD 在构建树时拒绝，不能将声明展开后的内容当作普通来源。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：当前测试的原生构建验证替身。
# 输出：
#   None：不返回业务数据。
def test_dtd_world_is_rejected_before_deployment_output(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    args["world_sdf"].write_text('<!DOCTYPE sdf [<!ENTITY label "campus">]>'
                                 '<sdf><world name="&label;"/></sdf>')
    with pytest.raises(ValueError, match="REPLICA_SDF_INPUT"):
        module.prepare_render_replica(**args)
    assert not args["output"].exists()
