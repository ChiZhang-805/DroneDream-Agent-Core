import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import dronedream_agent_core.geometry_motion_fixture as fixture
from dronedream_agent_core.contracts import QuaternionWxyz, RawRangeSample, Vector3
from dronedream_agent_core.geometry_fixture_capture import FixtureCapture
from dronedream_agent_core.geometry_fixture_transport import validate_fixture_command
from dronedream_agent_core.geometry_motion_fixture import (
    DEPTH_TOPIC,
    RIG_NAME,
    build_fixture_world,
    camera_calibration,
    euler_quaternion,
    fixture_pose,
    sensor_hits_world,
)

CAMERA = b'''<sdf version="1.9"><model name="camera"><link name="link">
<sensor name="StereoOV7251" type="depth_camera"><pose>1 2 3 0 0 0</pose>
<camera><horizontal_fov>1.274</horizontal_fov><image><width>640</width><height>480</height>
<format>R_FLOAT32</format></image><clip><near>0.2</near><far>19.1</far></clip></camera>
<update_rate>30</update_rate><noise><type>none</type></noise></sensor></link></model></sdf>'''
WORLD = (b'<sdf version="1.9"><world name="fixture"><model name="map">'
         b'<static>true</static></model></world></sdf>')


# 功能：
#   检查夹具轨迹闭合、位移幅度及三轴姿态变化，不以指令作为实测运动证明。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_camera_motion_is_closed_bounded_and_rotates():
    poses = [fixture_pose([10, 20, 3], [0, -1, 0], float(p)) for p in np.linspace(0., 1., 101)]
    np.testing.assert_allclose(poses[0]["position_m"], poses[-1]["position_m"])
    span = np.ptp([p["position_m"] for p in poses], axis=0)
    np.testing.assert_allclose(span, [.24, .8, .2395264], atol=1e-6)
    assert max(2*math.acos(abs(p["orientation_wxyz"][0])) for p in poses) > .5
    for pose in poses:
        assert np.linalg.norm(pose["orientation_wxyz"]) == pytest.approx(1.)


# 功能：验证诊断扫描初始朝向可配置，但不改变位置轨迹或接受错误角度。
# 输入：无。
# 输出：无。
def test_fixture_declared_initial_heading_preserves_translation():
    base = fixture_pose([1,2,3],[1,0,0],0)
    turned = fixture_pose([1,2,3],[1,0,0],0,base_yaw_deg=90.)
    assert turned['position_m'] == base['position_m']
    np.testing.assert_allclose(turned['orientation_wxyz'],[np.sqrt(.5),0,0,np.sqrt(.5)])
    for invalid in (True, float('nan'), 181., -181.):
        with pytest.raises(ValueError):
            fixture_pose([1,2,3],[1,0,0],0,base_yaw_deg=invalid)


# 功能：
#   验证错误维数、竖直方向、非有限坐标及布尔进度不能生成扫描轨迹。
# 输入：
#   origin：候选扫描原点。
#   direction：候选扫描方向。
#   phase：候选扫描进度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("origin,direction,phase", [([1, 2], [1, 0, 0], 0),
    ([1, 2, 3], [0, 0, 1], 0), ([1, float('nan'), 3], [1, 0, 0], 0),
    ([1, 2, 3], [1, 0, 0], True), ([1, 2, 3], [1, 0, 0], float('inf'))])
def test_invalid_trajectory_rejected(origin, direction, phase):
    with pytest.raises(ValueError):
        fixture_pose(origin, direction, phase)


# 功能：
#   核对派生世界保留真实相机内参及更新频率，且未引入飞行器执行机构。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_world_has_real_camera_intrinsics_and_no_aircraft_actuators():
    result = ET.fromstring(build_fixture_world(WORLD, CAMERA, [1, 2, 3]))
    model = result.find(f"world/model[@name='{RIG_NAME}']")
    sensor = model.find("link/sensor")
    assert sensor.findtext("topic") == DEPTH_TOPIC
    assert sensor.findtext("pose") == "0 0 0 0 0 0"
    assert sensor.findtext("update_rate") == "30"
    assert (ET.tostring(sensor.find("camera"))
            == ET.tostring(ET.fromstring(CAMERA).find(".//camera")))
    assert [p.get("name") for p in model.findall("plugin")] == ["gz::sim::systems::PosePublisher"]
    assert model.findtext("plugin/use_pose_vector_msg") == "false"
    assert model.findtext("static") == "true"
    assert camera_calibration(CAMERA).sample_stride_pixels == 32


# 功能：核对第一帧之前的世界初始姿态已应用目标朝向，不依赖后续 set_pose 命令。
# 输入：heading：合法的目标偏航角，单位度。
# 输出：无。
@pytest.mark.parametrize("heading", [-180., -135., -90., 0., 45., 90., 135., 180.])
def test_fixture_spawn_heading_matches_command_origin(heading):
    result = ET.fromstring(build_fixture_world(WORLD, CAMERA, [1, 2, 3],
                                               base_yaw_deg=heading))
    pose = [float(value) for value in result.find(
        f"world/model[@name='{RIG_NAME}']/pose").text.split()]
    assert pose[:5] == [1., 2., 3., 0., 0.]
    assert pose[5] == pytest.approx(math.radians(heading))


# 功能：禁止在初始化地图时绕过动态轨迹相同的朝向输入校验。
# 输入：heading：非法偏航值。
# 输出：无。
@pytest.mark.parametrize("heading", [True, "90", float("nan"), float("inf"), 181., -181.])
def test_fixture_spawn_rejects_invalid_heading(heading):
    with pytest.raises(ValueError):
        build_fixture_world(WORLD, CAMERA, [1, 2, 3], base_yaw_deg=heading)


# 功能：
#   验证动态模型、引用及任意插件不能作为静态校准地图启动。
# 输入：
#   replacement：地图中拟注入的实体或执行内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("replacement", [b'<model name="map"><static>false</static></model>',
    b'<include><uri>model://anything</uri></include>', b'<plugin filename="anything"/>',
    b'<model name="map"><static>true</static><include><uri>model://nested</uri></include></model>'])
def test_world_rejects_dynamic_or_executable_inputs(replacement):
    with pytest.raises(ValueError):
        build_fixture_world(b'<sdf><world name="x">'+replacement+b'</world></sdf>',
                            CAMERA, [1, 2, 3])


# 功能：
#   拒绝不支持的像素格式、投影覆盖及超预算分辨率。
# 输入：
#   camera：候选相机定义字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("camera", [CAMERA.replace(b'R_FLOAT32', b'RGB_INT8'),
    CAMERA.replace(b'</camera>', b'<intrinsics/></camera>'),
    CAMERA.replace(b'<width>640</width>', b'<width>1280</width>')])
def test_unimplemented_camera_calibration_rejected(camera):
    with pytest.raises(ValueError):
        camera_calibration(camera)


# 功能：
#   以九十度机体偏航核对安装平移和 FLU 射线均旋转到正确世界位置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_sensor_hits_rotate_calibrated_optical_offset_and_flu_ray():
    sample = RawRangeSample(direction_sensor=Vector3(x=1, y=0, z=0), range_m=2., hit=True,
                            confidence=.9)
    points, origin = sensor_hits_world([sample], {"position_m": [10, 20, 30],
        "orientation_wxyz": euler_quaternion([0, 0, math.pi/2])})
    np.testing.assert_allclose(origin, [10, 20.13233, 30.03278])
    np.testing.assert_allclose(points[0], [10, 22.13233, 30.03278])


# 功能：
#   生成固定接收时间、可变源时间的测试记录，区分两类时钟。
# 输入：
#   ns：仿真源时间，单位纳秒。
# 输出：
#   record：三种时间字段组成的对象。
def stamp(ns):
    record = {"simulation_time_ns": ns, "received_monotonic_seconds": 5., "received_unix_ms": 1000.}
    return record


# 功能：
#   建立四乘四像素的紧密浮点深度布局，不生成图像内容。
# 输入：
#   ns：仿真源时间，单位纳秒。
# 输出：
#   record：包含时间和像素布局的图像元数据。
def image(ns):
    record = {**stamp(ns), "width": 4, "height": 4, "step": 16, "pixel_format": "R_FLOAT32"}
    return record


# 功能：
#   建立固定位置和单位姿态的采集测试输入。
# 输入：
#   ns：仿真源时间，单位纳秒。
# 输出：
#   record：包含时间、位置和四元数的姿态对象。
def pose(ns):
    record = {**stamp(ns), "position_m": [1, 2, 3], "orientation_wxyz": [1, 0, 0, 0]}
    return record


# 功能：
#   核对采样依据源时间而非接收时间，重复姿态不累积，关闭后忽略新消息。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_capture_samples_source_time_not_wall_receipt_and_closes():
    capture = FixtureCapture(width=4, height=4)
    capture.receive_image(image(0), bytes(64))
    capture.receive_image(image(100_000_000), bytes(64))
    capture.receive_image(image(200_000_000), bytes(64))
    capture.receive_pose(pose(0))
    capture.receive_pose(pose(0))
    queued, last, errors = capture.snapshot()
    assert len(queued) == 2 and last == 0 and not errors
    assert capture.skipped_images == 1 and len(capture.poses) == 1
    capture.close()
    capture.receive_image(image(500_000_000), bytes(64))
    capture.receive_pose(pose(100))
    assert capture.snapshot() == ([], 0, ())


# 功能：
#   核对畸形布局及无效时钟只记录错误，不进入待处理图像队列。
# 输入：
#   mutation：对合法图像元数据的单项破坏。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", [{"step": 20}, {"width": 5}, {"pixel_format": "RGB_INT8"},
    {"simulation_time_ns": True}, {"simulation_time_ns": -1}, {"received_unix_ms": float('nan')},
    {"width": 4.}, {"height": 4.}, {"header_data": {"large": "x" * 17000}}])
def test_capture_rejects_bad_image_layout_or_clock(mutation):
    capture = FixtureCapture(width=4, height=4)
    capture.receive_image({**image(10), **mutation}, bytes(64))
    queued, _, errors = capture.snapshot()
    assert not queued and errors


# 功能：
#   验证满队列及姿态时间倒退被明确记录，不悄悄丢弃后继续声称采集成功。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_capture_queue_limit_and_time_regression_do_not_silently_pass():
    capture = FixtureCapture(width=4, height=4)
    for i in range(5):
        capture.receive_image(image(i*200_000_000), bytes(64))
    assert "QUEUE_FULL" in capture.snapshot()[2][0]
    capture.receive_pose(pose(100))
    capture.receive_pose(pose(90))
    assert "POSE_TIME_REGRESSED" in capture.snapshot()[2][-1]
    assert len(capture.poses) == 1


# 功能：
#   核对夹具命令不能改为真实飞机实体，且拒绝非法坐标与姿态。
# 输入：
#   mutation：对合法目标姿态对象的字段替换。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", [{"name": "aircraft"}, {"position_m": [True, 0, 0]},
    {"position_m": [1e8, 0, 0]}, {"orientation_wxyz": [2., 0., 0., 0.]},
    {"orientation_wxyz": [float('nan'), 0., 0., 0.]}])
def test_command_transport_cannot_retarget_an_aircraft_or_send_invalid_pose(mutation):
    value = {"position_m": [1., 2., 3.], "orientation_wxyz": [1., 0., 0., 0.]}
    validate_fixture_command(value)
    with pytest.raises(ValueError):
        validate_fixture_command({**value, **mutation})


# 功能：
#   检查采集后修改来源姿态或嵌套报头，不会改写已经验证并保存的记录。
# 输入：
#   kind：姿态或深度图像记录类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["pose", "image"])
def test_capture_owns_nested_source_records(kind):
    capture = FixtureCapture(width=4, height=4)
    record = pose(0) if kind == "pose" else image(0)
    record["header_data"] = {"frame_id": ["camera"]}
    if kind == "pose":
        capture.receive_pose(record)
        record["position_m"][0] = 99
        saved = capture.poses[0]
        assert saved["position_m"] == [1, 2, 3]
    else:
        capture.receive_image(record, bytes(64))
        saved = capture.snapshot()[0][0][0]
    record["header_data"]["frame_id"].append("changed")
    assert saved["header_data"] == {"frame_id": ["camera"]}


# 功能：
#   检查非对象记录被转成有界诊断，不从订阅回调漏出属性访问异常。
# 输入：
#   kind：准备调用的采集入口。
#   record：非对象记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["pose", "image"])
@pytest.mark.parametrize("record", [None, []])
def test_capture_rejects_nonobject_records_without_callback_exception(kind, record):
    capture = FixtureCapture(width=4, height=4)
    if kind == "pose":
        capture.receive_pose(record)
    else:
        capture.receive_image(record, bytes(64))
    queued, _, errors = capture.snapshot()
    assert not queued and not capture.poses and errors


# 功能：
#   检查相机定义里的插件与世界中的动画 actor 不能进入只允许静态地图的隔离夹具。
# 输入：
#   case：隐藏插件或动画对象场景。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("case", ["sensor-plugin", "world-actor"])
def test_fixture_rejects_additional_executable_inputs(case):
    world, camera = WORLD, CAMERA
    if case == "sensor-plugin":
        camera = camera.replace(b"</sensor>", b'<plugin filename="unexpected"/></sensor>')
    else:
        world = world.replace(b"</world>", b'<actor name="moving-person"/></world>')
    with pytest.raises(ValueError):
        build_fixture_world(world, camera, [1, 2, 3])


# 功能：
#   核对极大但有限的方向只决定方向，不因平方范数溢出而把测试运动与深度射线变成零。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_fixture_keeps_direction_scale_out_of_motion_and_range():
    position = fixture_pose([0, 0, 0], [1e200, 0, 0], .5)["position_m"]
    assert position[0] == pytest.approx(.8)
    sample = RawRangeSample(direction_sensor=Vector3(x=1e200, y=0, z=0),
                            range_m=2., hit=True, confidence=.9)
    points, origin = sensor_hits_world([sample], {
        "position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]})
    np.testing.assert_allclose(points[0] - origin, [2., 0., 0.])


# 功能：
#   检查夹具几何审计使用实际校准量程，不接受生产桥接层会拒绝的距离。
# 输入：
#   range_m：超出当前安装校准量程的距离。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("range_m", [.1, 100.])
def test_fixture_rejects_uncalibrated_range(range_m):
    sample = RawRangeSample(direction_sensor=Vector3(x=1, y=0, z=0),
                            range_m=range_m, hit=True, confidence=.9)
    with pytest.raises(ValueError, match="RAYS_INVALID"):
        sensor_hits_world([sample], {"position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]})


# 功能：
#   核对非零安装姿态同时用于派生 SDF 与命中点换算，包括俯仰万向节锁位置。
# 输入：
#   monkeypatch：pytest 替换工具。
#   angles：测试安装的 roll、pitch、yaw 弧度角。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("angles", [(0, 0, math.pi/2), (.3, math.pi/2, -.5),
                                    (.3, -math.pi/2, -.5), (.2, .3, .4)])
def test_mount_rotation_agrees_between_world_and_ray_projection(monkeypatch, angles):
    mount = fixture.oakd_lite_depth_sensor_contract()
    quaternion = euler_quaternion(angles)
    mount = mount.model_copy(update={"orientation_body_from_sensor": QuaternionWxyz(
        **dict(zip(("w", "x", "y", "z"), quaternion, strict=True)))})

    # 功能：
    #   返回本用例的明确安装外参，避免使用默认零旋转掩盖测试缺陷。
    # 输入：
    #   无。
    # 输出：
    #   mount：当前测试安装契约。
    def test_mount():
        return mount

    monkeypatch.setattr(fixture, "oakd_lite_depth_sensor_contract", test_mount)
    root = ET.fromstring(build_fixture_world(WORLD, CAMERA, [0, 0, 0]))
    link_pose = [float(value) for value in root.findtext(
        f"world/model[@name='{RIG_NAME}']/link/pose").split()]
    # 四元数 q 与 -q 是同一姿态，使用绝对内积判断重建的旋转是否等价。
    reconstructed = euler_quaternion(link_pose[3:])
    assert abs(np.dot(reconstructed, quaternion)) == pytest.approx(1.)
    sample = RawRangeSample(direction_sensor=Vector3(x=1, y=0, z=0),
                            range_m=2., hit=True, confidence=.9)
    points, origin = sensor_hits_world([sample], {
        "position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]})
    roll, pitch, yaw = angles
    expected_axis = [math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), -math.sin(pitch)]
    np.testing.assert_allclose((points[0] - origin)/2, expected_axis, atol=1e-12)


# 功能：
#   拒绝两个同名 camera 元素，避免解释器选择顺序不同而使用不同内参。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_camera_definition_rejected():
    camera = CAMERA.replace(b"</camera>", b"</camera><camera/>")
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        camera_calibration(camera)


# 功能：
#   错误消息必须同时限制数量与长度，外部非字符串错误也不会引发回调异常。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_capture_errors_are_bounded():
    capture = FixtureCapture(width=4, height=4)
    capture.fail(None)
    for _ in range(20):
        capture.fail("long" * 10000)
    assert len(capture.errors) == 16
    assert all(len(error) <= 512 for error in capture.errors)


# 功能：
#   原始相机的裁剪量程与当前安装契约不同则停止建夹具，禁止采集和审计采用两种边界。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_world_requires_camera_range_bound_to_current_mount():
    camera = CAMERA.replace(b"<far>19.1</far>", b"<far>20.0</far>")
    with pytest.raises(ValueError, match="RANGE_MISMATCH"):
        build_fixture_world(WORLD, camera, [0, 0, 0])
