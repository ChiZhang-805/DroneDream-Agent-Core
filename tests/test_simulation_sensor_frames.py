"""Exact frame identity and geometry in the outcome/training observer path."""

import json
import math
import sys
from types import SimpleNamespace as NS

import pytest

from dronedream_agent_core.gazebo_adapter import _resolve_controlled_vehicle_pose
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.simulation_sensor_frames import (
    inspect_simulation_sensor_frames,
    load_simulation_frame_contract,
    select_canonical_poses,
    simulation_pose_time_ns,
)


# 功能：
#   构造保留原始分量类型的实体消息，供名称选择与位姿边界测试使用。
# 输入：
#   name：实体名称。
#   position：xyz 位置分量。
#   q：wxyz 姿态分量。
# 输出：
#   result：具有实体名称、位置和姿态属性的测试消息。
def entity(name, position=(0., 0., 0.), q=(1., 0., 0., 0.)):
    result = NS(name=name, position=NS(**dict(zip("xyz", position, strict=True))),
                orientation=NS(**dict(zip("wxyz", q, strict=True))))
    return result


# 功能：
#   构造机体安装原点与碰撞中心不同的最小坐标契约，避免测试退化成零偏移。
# 输入：
#   无。
# 输出：
#   contract：每次新建的模型、机体安装及碰撞中心字典。
def frames():
    contract = {"vehicle_model_name": "drone", "canonical_link_name": "base_link",
            "canonical_at_rest": {"position_m": [0., 0., .24],
                                  "orientation_wxyz": [1., 0., 0., 0.]},
            "collision_center_model_m": [0., 0., .228]}
    return contract


# 功能：
#   使用实际结果观测器解析测试消息，只有未提供契约时才建立默认值。
# 输入：
#   poses：当前消息内的实体位姿集合。
#   contract：显式测试契约；None 表示使用默认契约。
# 输出：
#   result：观测器返回的位姿与身份证据；尚无目标实体时为 None。
def resolve(poses, contract=None):
    result = _resolve_controlled_vehicle_pose(poses, vehicle_name="drone",
        frames=frames() if contract is None else contract,
        collision_center_offset_model_m=(0., 0., .228))
    return result


# 功能：
#   验证外层偏航与机体俯仰同时存在时，结果观测器仍按真实机体偏移计算碰撞中心。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_live_outcome_observer_rotates_wrapper_and_body_without_center_approximation():
    h = math.sqrt(.5)
    root, selected, reference, _ = resolve([
        entity("drone", (1., 2., 3.), (h, 0., 0., h)),
        entity("base_link", (2., 0., .7), (h, 0., h, 0.))])
    assert tuple(root[i] + (0., 0., .228)[i] for i in range(3)) == pytest.approx((1., 3.988, 3.7))
    assert selected == "base_link"
    assert reference == "calibrated-collision-center-offset-adjusted"


# 功能：
#   验证实体缺失、相似名称及重复别名不能回退成静态模型位姿或消息顺序选择。
# 输入：
#   names：待注入的实体名称组合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("names", [
    ["drone"], ["base_link"], ["drone", "backup-drone::base_link"],
    ["drone", "base_link", "drone::base_link"], ["drone", "drone", "base_link"],
])
def test_missing_or_ambiguous_entities_never_use_static_wrapper_or_similar_drone(names):
    with pytest.raises(ValueError, match="ENTITY_AMBIGUOUS_OR_MISSING"):
        resolve([entity(name) for name in names])


# 功能：
#   验证允许的三种精确链接别名产生相同碰撞中心，并保留实际命中的名称。
# 输入：
#   name：当前测试使用的规范链接别名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("name", ["base_link", "drone::base_link", "drone/base_link"])
def test_each_exact_link_alias_has_same_calibrated_center(name):
    root, selected, _, _ = resolve([entity("drone", (1., 2., 3.)), entity(name, (0., 0., .24))])
    assert root == pytest.approx((1., 2., 3.))
    assert selected == name


# 功能：
#   验证启动阶段只有其他模型时不会伪造受控飞行器的观测。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_no_vehicle_yet_produces_no_observation():
    assert resolve([entity("other::base_link")]) is None


# 功能：
#   验证车辆名称或碰撞中心与启动绑定不符时，观测器拒绝继续解析。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_contract_binding_mismatch_is_not_a_coordinate_fallback():
    for key, value in (("vehicle_model_name", "other"), ("collision_center_model_m", [0, 0, .24])):
        contract = frames()
        contract[key] = value
        with pytest.raises(ValueError, match="OBSERVER_BINDING_MISMATCH"):
            resolve([], contract)


# 功能：
#   验证缺少完整父级变换的嵌套链接不能被当作模型直属机体处理。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_nested_link_cannot_be_treated_as_direct_model_relative_pose():
    with pytest.raises(ValueError, match="ENTITY_CONTRACT_UNSUPPORTED"):
        select_canonical_poses([], vehicle_name="drone", canonical_link_name="inner::base_link")


# 功能：
#   验证工作进程契约读取同时受大小、车辆、碰撞中心和内容摘要约束。
# 输入：
#   tmp_path：契约文件的独立临时目录。
# 输出：
#   None：不返回业务数据。
def test_worker_frame_contract_is_bounded_bound_to_vehicle_and_not_replaced_by_old_offset(tmp_path):
    value = {**frames(), "schema_version": "dronedream.simulation-sensor-frames.v1",
             "motion_permission_granted": False, "truth_used_as_control_input": False}
    path = tmp_path / "simulation-sensor-frames.json"

    # 功能：
    #   将当前测试契约与其同步计算的摘要写入独立文件。
    # 输入：
    #   value：外层用例建立的契约字段。
    #   path：外层用例分配的临时契约路径。
    # 输出：
    #   None：不返回业务数据。
    def save():
        path.write_text(json.dumps({**value, "record_sha256": sha256_json(value)}),
                        encoding="utf-8")
    save()
    loaded = load_simulation_frame_contract(path, vehicle_name="drone",
                                           collision_center_model_m=(0., 0., .228))
    assert loaded["canonical_at_rest"] == frames()["canonical_at_rest"]
    for vehicle, offset in (("other", (0., 0., .228)), ("drone", (0., 0., .24))):
        with pytest.raises(ValueError, match="CONTRACT_BINDING_INVALID"):
            load_simulation_frame_contract(path, vehicle_name=vehicle,
                                           collision_center_model_m=offset)
    path.write_text(path.read_text().replace('"base_link"', '"wrong"'), encoding="utf-8")
    with pytest.raises(ValueError, match="CONTRACT_BINDING_INVALID"):
        load_simulation_frame_contract(path, vehicle_name="drone",
                                       collision_center_model_m=(0., 0., .228))
    path.write_bytes(b" " * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="CONTRACT_TOO_LARGE"):
        load_simulation_frame_contract(path, vehicle_name="drone",
                                       collision_center_model_m=(0., 0., .228))


# 功能：
#   验证非法秒、纳秒及整数溢出不能转成有效仿真时间。
# 输入：
#   sec：消息中待测试的秒分量。
#   nsec：消息中待测试的纳秒分量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("sec,nsec", [(True, 0), (-1, 0), (0, -1), (0, 10**9), (2**63, 0)])
def test_malformed_publisher_stamp_never_becomes_a_simulation_time(sec, nsec):
    with pytest.raises(ValueError, match="SIMULATION_POSE_TIME_INVALID"):
        simulation_pose_time_ns(NS(header=NS(stamp=NS(sec=sec, nsec=nsec))))


# 功能：
#   验证纳秒不经浮点转换而丢失精度，并拒绝缺失或未实际填充的时间字段。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_publisher_stamp_keeps_integer_precision_and_requires_presence():
    assert simulation_pose_time_ns(NS(header=NS(stamp=NS(sec=123, nsec=4)))) == 123000000004
    with pytest.raises(ValueError, match="SIMULATION_POSE_TIME_MISSING"):
        simulation_pose_time_ns(NS())
    empty_proto = NS(HasField=lambda name: False)
    with pytest.raises(ValueError, match="SIMULATION_POSE_TIME_MISSING"):
        simulation_pose_time_ns(empty_proto)


# 功能：
#   确认真正的 Runtime 解析依赖可用，缺失时明确跳过而不是以假解析器冒充集成通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def runtime_sdformat():
    path = "/usr/lib/python3/dist-packages"
    if sys.platform == "linux" and path not in sys.path:
        sys.path.append(path)
    pytest.importorskip("sdformat14", reason="actual SDFormat parser is installed in Runtime")


# 功能：
#   用实际 SDFormat 解析器验证显式规范链接优先于排列顺序或 base_link 习惯名称。
# 输入：
#   tmp_path：真实解析器读取的独立 SDF 目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_parser_resolves_declared_canonical_link_not_first_or_named_base_link(tmp_path):
    runtime_sdformat()
    path = tmp_path / "model.sdf"
    path.write_text('''<sdf version="1.9"><model name="test" canonical_link="body_b">
      <link name="base_link"><pose>9 9 9 0 0 0</pose></link>
      <link name="body_b"><pose>0.1 0.2 0.3 0 0 1.5707963267948966</pose></link>
    </model></sdf>''', encoding="utf-8")
    result = inspect_simulation_sensor_frames(path, [tmp_path],
        collision_center_model_m=[.1, .2, .28], require_depth=False)
    assert result["canonical_link_name"] == "body_b"
    assert result["canonical_at_rest"]["position_m"] == pytest.approx([.1, .2, .3])
    assert result["canonical_at_rest"]["orientation_wxyz"] == pytest.approx(
        [math.sqrt(.5), 0., 0., math.sqrt(.5)])
    assert result["depth_mount_verified"] is False
    digest = result.pop("record_sha256")
    assert digest == sha256_json(result)
    assert str(path.resolve()) in result["sources_sha256"]
    assert not result["motion_permission_granted"]


# 功能：
#   用实际 SDFormat 解析器验证强制深度标定时缺少相机会拒绝生成观测契约。
# 输入：
#   tmp_path：仅包含机体的测试 SDF 所在目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_parser_requires_camera_when_declared_required(tmp_path):
    runtime_sdformat()
    path = tmp_path / "model.sdf"
    path.write_text('<sdf version="1.9"><model name="test"><link name="body"/></model></sdf>',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="DEPTH_SENSOR_MISSING"):
        inspect_simulation_sensor_frames(path, [tmp_path], collision_center_model_m=[0, 0, 0])


# 功能：
#   为读取边界测试保存摘要自洽的启动契约，不以摘要替代内容有效性检查。
# 输入：
#   tmp_path：当前用例的独立临时目录。
#   changes：覆盖默认契约字段的测试值。
# 输出：
#   path：写入后的契约路径。
def save_frame_contract(tmp_path, **changes):
    value = {**frames(), "schema_version": "dronedream.simulation-sensor-frames.v1",
             "motion_permission_granted": False, "truth_used_as_control_input": False,
             **changes}
    path = tmp_path / "frames.json"
    path.write_text(json.dumps({**value, "record_sha256": sha256_json(value)}), encoding="utf-8")
    return path


# 功能：
#   验证重复 JSON 字段即使最终解析值与摘要一致也必须拒绝。
# 输入：
#   tmp_path：契约文件所在的独立临时目录。
# 输出：
#   None：不返回业务数据。
def test_frame_contract_rejects_duplicate_keys_before_digest_validation(tmp_path):
    path = save_frame_contract(tmp_path)
    path.write_text('{"vehicle_model_name":"wrong",' + path.read_text()[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        load_simulation_frame_contract(path, vehicle_name="drone",
                                       collision_center_model_m=(0, 0, .228))


# 功能：
#   验证重新计算摘要不能使畸形安装位姿、布尔坐标或不支持的实体名称合法化。
# 输入：
#   tmp_path：契约文件所在的独立临时目录。
#   changes：摘要计算前注入的非法字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes", [
    {"collision_center_model_m": [False, 0, .228]},
    {"canonical_at_rest": None},
    {"canonical_at_rest": {"position_m": [False, 0, 0], "orientation_wxyz": [1, 0, 0, 0]}},
    {"canonical_at_rest": {"position_m": [0, 0, 0], "orientation_wxyz": [0, 0, 0, 0]}},
    {"canonical_at_rest": {"position_m": [0, 0, 0], "orientation_wxyz": ["1", 0, 0, 0]}},
    {"canonical_link_name": "inner::body"},
    {"canonical_link_name": ""},
    {"canonical_link_name": None},
])
def test_frame_contract_validates_geometry_even_with_matching_digest(tmp_path, changes):
    path = save_frame_contract(tmp_path, **changes)
    with pytest.raises(ValueError, match="SIMULATION_FRAME_"):
        load_simulation_frame_contract(path, vehicle_name="drone",
                                       collision_center_model_m=(0, 0, .228))


# 功能：
#   验证调用方提供的预期碰撞中心也接受严格类型检查，不能让 False 匹配零坐标。
# 输入：
#   tmp_path：保存合法契约的独立临时目录。
# 输出：
#   None：不返回业务数据。
def test_expected_collision_center_rejects_boolean_alias(tmp_path):
    path = save_frame_contract(tmp_path)
    with pytest.raises(ValueError, match="SIMULATION_FRAME_"):
        load_simulation_frame_contract(path, vehicle_name="drone",
                                       collision_center_model_m=(False, 0, .228))


# 功能：
#   验证实时实体的坐标和四元数先校验再序列化，不把字符串或布尔值转成数字。
# 输入：
#   field：待污染的实体字段。
#   value：布尔值或数字字符串。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("position", True), ("position", "0"),
                                        ("orientation", True), ("orientation", "0")])
def test_live_pose_selection_rejects_coerced_numeric_fields(field, value):
    model = entity("drone")
    getattr(model, field).x = value
    with pytest.raises(ValueError, match="SIMULATION_FRAME_"):
        select_canonical_poses([model, entity("base_link")], vehicle_name="drone",
                               canonical_link_name="base_link")


# 功能：
#   注入最小假解析接口，仅测试宿主文件绑定逻辑，不代表实际 SDFormat 解析通过。
# 输入：
#   monkeypatch：pytest 临时替换工具。
#   on_load：解析阶段调用的受控文件操作，接收包含文件查找回调。
# 输出：
#   None：不返回业务数据。
def install_binding_test_parser(monkeypatch, on_load):
    callbacks = []
    pose = NS(pos=lambda: NS(x=lambda: 0., y=lambda: 0., z=lambda: 0.),
              rot=lambda: NS(w=lambda: 1., x=lambda: 0., y=lambda: 0., z=lambda: 0.))
    link = NS(semantic_pose=lambda: NS(resolve=lambda frame: pose))
    model = NS(canonical_link_and_relative_name=lambda: (link, "body"), name=lambda: "drone")
    parser = NS(
        ParserConfig=lambda: NS(set_find_callback=callbacks.append),
        Root=lambda: NS(load=lambda path, config: on_load(callbacks[0]), model=lambda: model),
    )
    monkeypatch.setitem(sys.modules, "sdformat14", parser)


# 功能：
#   复现同一包含文件在两次加载间变化，要求保留首次摘要而不是覆盖后放行。
# 输入：
#   tmp_path：独立源模型及包含文件目录。
#   monkeypatch：注入仅用于绑定检查的假解析接口。
# 输出：
#   None：不返回业务数据。
def test_repeated_include_cannot_replace_first_source_digest(tmp_path, monkeypatch):
    path = tmp_path / "vehicle.sdf"
    path.write_text("vehicle", encoding="utf-8")
    included = tmp_path / "included"
    included.mkdir()
    source = included / "model.sdf"
    source.write_text("first", encoding="utf-8")

    # 功能：
    #   在两次模型 URI 解析之间改写包含文件，模拟加载期间来源被替换。
    # 输入：
    #   find：生产代码注册的包含模型查找回调。
    # 输出：
    #   None：不返回业务数据。
    def on_load(find):
        assert find("model://included") == str(included)
        source.write_text("second", encoding="utf-8")
        find("model://included")

    install_binding_test_parser(monkeypatch, on_load)
    with pytest.raises(ValueError, match="SIMULATION_FRAME_SOURCE_CHANGED"):
        inspect_simulation_sensor_frames(path, [tmp_path], collision_center_model_m=[0, 0, 0],
                                         require_depth=False)


# 功能：
#   验证末尾来源复核继续使用有界读取，禁止退回 Path.read_bytes 全量分配。
# 输入：
#   tmp_path：独立模型文件目录。
#   monkeypatch：注入假解析接口及禁止无界读取的探针。
# 输出：
#   None：不返回业务数据。
def test_final_source_recheck_does_not_use_unbounded_read(tmp_path, monkeypatch):
    path = tmp_path / "vehicle.sdf"
    path.write_text("vehicle", encoding="utf-8")
    install_binding_test_parser(monkeypatch, lambda find: None)

    # 功能：
    #   将任何无界读取标为测试失败，防止来源复核绕过字节预算。
    # 输入：
    #   self：被读取的路径。
    # 输出：
    #   None：不返回业务数据。
    def forbid_unbounded_read(self):
        pytest.fail("source recheck must retain a byte budget")

    monkeypatch.setattr(type(path), "read_bytes", forbid_unbounded_read)
    result = inspect_simulation_sensor_frames(path, [tmp_path],
        collision_center_model_m=[0, 0, 0], require_depth=False)
    assert result["sources_sha256"]
