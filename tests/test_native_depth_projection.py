"""Native all-pixel depth reduction, portability and numerical contract tests."""

import math
import random
import struct
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from dronedream_agent_core.depth_projection import (
    DepthProjectionCalibration,
    project_metric_depth_frame,
)
from dronedream_agent_core.depth_projection_native import NATIVE_DEPTH_AVAILABLE, backend

pytestmark = pytest.mark.skipif(not NATIVE_DEPTH_AVAILABLE, reason='native depth kernel absent')


# 功能：生成带坏像素和行填充的完整输入；输入：尺寸、步长和无回波语义；输出：冻结字节及内参。
def fixture(width=31, height=19, stride=8, mode='unknown'):
    rng = random.Random(width * height)
    choices = [math.nan, math.inf, -math.inf, -.1, 0., .1, .2, 1., 2., 9., 10., 11.]
    calibration = DepthProjectionCalibration(width, height, 1.274, .2, 10., stride,
        no_return_mode=mode, fx_pixels=width * .63, fy_pixels=height * .71,
        cx_pixels=width * .41, cy_pixels=height * .37)
    values = [rng.choice(choices) for _ in range(width * height)]
    values[0] = 1.
    data = b''.join(struct.pack(f'<{width}f', *values[row*width:(row+1)*width]) + b'pad!'
                    for row in range(height))
    return data, width * 4 + 4, calibration


# 功能：逐条对照独立 NumPy 参考，包含非整块、小图、大步长及两种缺失语义。
# 输入：明确的像素布局；输出：方向、距离、命中、顺序和覆盖率完全等价。
@pytest.mark.parametrize('width,height,stride', [(2, 2, 1), (31, 19, 8), (8, 5, 8192),
    (320, 240, 16), (640, 360, 24)])
@pytest.mark.parametrize('mode', ['unknown', 'gazebo-far-clip'])
def test_native_equals_portable_reference(width, height, stride, mode):
    data, step, calibration = fixture(width, height, stride, mode)
    with patch('dronedream_agent_core.depth_projection.NATIVE_DEPTH_AVAILABLE', False):
        expected = project_metric_depth_frame(data=data, row_step_bytes=step,
                                               calibration=calibration)
    actual = project_metric_depth_frame(data=data, row_step_bytes=step, calibration=calibration)
    assert actual == expected


# 功能：无效直接原生调用必须在读像素/分配前失败；输入：坏字段；输出：明确异常。
@pytest.mark.parametrize('slot,value', [(0, bytearray(64)), (1, True), (1, 8193),
    (1, 10**100), (2, -1), (3, 2), (3, 10**100), (4, 0), (4, True),
    (5, (0., 1., 1., 1.)), (5, (1., 1., math.nan, 1.)), (5, (1e-300, 1., 1., 1.)),
    (5, (True, 1., 1., 1.)), (6, -.1), (7, math.inf), (7, 0.), (8, 1)])
def test_native_rejects_malformed_direct_inputs(slot, value):
    args = [struct.pack('<16f', *([2.] * 16)), 4, 4, 16, 2, (2., 2., 1.5, 1.5), .2, 10., False]
    args[slot] = value
    with pytest.raises((ValueError, OverflowError)):
        backend.project_depth(*args)


# 功能：同时调用不能互改归约缓冲；输入：八个相同完整图；输出：独立且一致的投影。
def test_parallel_native_calls_have_no_shared_accumulator():
    data, step, calibration = fixture(320, 240, 16)
    def project(_):
        return project_metric_depth_frame(data=data, row_step_bytes=step, calibration=calibration)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(project, range(8)))
    assert all(value == outputs[0] for value in outputs)


# 功能：已安装原生内核若失败不得伪装为缺少内核而回退；输入：故障；输出：原异常。
def test_native_failure_is_not_swallowed():
    data, step, calibration = fixture()
    with (patch.object(backend, 'project_depth', side_effect=ValueError('NATIVE_TEST_FAILURE')),
          pytest.raises(ValueError, match='NATIVE_TEST_FAILURE')):
        project_metric_depth_frame(data=data, row_step_bytes=step, calibration=calibration)
