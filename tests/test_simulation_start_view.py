"""View planning is not localization evidence and cannot move an existing drone."""
import copy
import math

import pytest

from dronedream_agent_core.local_map_alignment import MapSurfaceIndex
from dronedream_agent_core.simulation_start_view import select_simulation_start_view


# 功能：生成三面室内几何及真实形状的相机安装夹具；输入：无；输出：纯预测测试输入。
def fixture():
    primitives = [dict(center_x=3., center_y=0., center_z=1.5, size_x=.2, size_y=20., size_z=3.),
                  dict(center_x=0., center_y=3., center_z=1.5, size_x=20., size_y=.2, size_z=3.),
                  dict(center_x=0., center_y=0., center_z=-.1, size_x=20., size_y=20., size_z=.2)]
    frames = dict(depth_mount_verified=True, collision_center_model_m=[0.,0.,.228],
        optical_at_rest=dict(position_m=[.13233,0.,.26078], orientation_wxyz=[1.,0.,0.,0.]),
        depth_optics=dict(horizontal_fov_rad=1.274,near_m=.2,far_m=19.1,
                          image_width_px=160,image_height_px=120))
    return primitives, frames


# 功能：缺失侧墙参照时选择更完整视角，不修改起点、输入或授予运动权限。
# 输入：三面场景；输出：非零预测朝向，旧朝向明确缺失一维信息。
def test_selects_informative_heading_without_fabricating_localization():
    primitives, frames = fixture()
    original = copy.deepcopy(frames)
    result = select_simulation_start_view(index=MapSurfaceIndex(primitives),frames=frames,
                                          model_root=[0.,0.,0.])
    assert result['candidates'][0]['predicted_translation_rank'] == 2
    assert 0 < abs(result['selected_yaw_rad']) <= math.pi
    assert result['candidates'][-1]['predicted_pose_rank'] == 6
    assert result['predicted_only'] and not result['localization_qualified']
    assert not result['motion_permission_granted'] and frames == original


# 功能：无限单墙缺失的信息不能通过换朝向或伪造命中生成。
# 输入：单墙；输出：没有可用朝向，仍须运行真实定位检查。
def test_missing_geometry_stays_unobservable():
    primitives, frames = fixture()
    result = select_simulation_start_view(index=MapSurfaceIndex(primitives[:1]),frames=frames,
                                          model_root=[0.,0.,0.])
    assert result['selected_yaw_rad'] is None
    assert result['issue'] == 'START_VIEW_NO_COMPLETE_PREDICTED_CONSTRAINT'
    assert len(result['candidates']) == 14


# 功能：偏心机型不能在原点契约不变时擅自旋转；输入：水平偏心；输出：保留原生成语义。
def test_eccentric_vehicle_does_not_change_bound_origin():
    primitives, frames = fixture()
    frames['collision_center_model_m'][0]=.1
    result=select_simulation_start_view(index=MapSurfaceIndex(primitives),frames=frames,
                                      model_root=[0.,0.,0.])
    assert result['selected_yaw_rad'] is None and not result['candidates']
    assert 'ECCENTRIC' in result['issue']


# 功能：拒绝未验证安装与畸形内参；输入：损坏标定；输出：不可生成预测朝向。
@pytest.mark.parametrize('fault',['mount','dimensions','optics'])
def test_invalid_sensor_does_not_get_a_heading(fault):
    primitives, frames=fixture()
    if fault=='mount':
        frames['depth_mount_verified']=False
    elif fault=='dimensions':
        frames['depth_optics']['image_width_px']=True
    else:
        frames['depth_optics']['horizontal_fov_rad']=float('nan')
    with pytest.raises(ValueError):
        select_simulation_start_view(index=MapSurfaceIndex(primitives),frames=frames,
                                     model_root=[0.,0.,0.])
