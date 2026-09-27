"""Shared metric-preserving rotation for calibrated sensing and local control."""

from __future__ import annotations

import math

import numpy as np

from .contracts import QuaternionWxyz

Point = tuple[float, float, float]


# 功能：
#   对有界批次共用一次姿态校验，按标量版相同顺序执行两次叉乘，保留米制长度。
#   分量先按行缩放，避免中间乘积溢出；非法输入不返回部分结果。
# 输入：
#   quaternion：标量在前的姿态四元数。
#   vectors：一至 1024 行、每行三分量的 float64 数组。
# 输出：
#   result：独立的旋转后数组，形状与输入一致。
def rotate_vectors(quaternion: QuaternionWxyz, vectors: np.ndarray) -> np.ndarray:
    # 标量入口保持四元数合同及错误码唯一；零向量仅用于姿态校验。
    rotate_vector(quaternion, (0., 0., 0.))
    if (type(vectors) is not np.ndarray or vectors.dtype != np.float64 or vectors.ndim != 2
            or vectors.shape[1] != 3 or not 1 <= len(vectors) <= 1024
            or not np.isfinite(vectors).all()):
        raise ValueError('METRIC_ROTATION_VECTOR_INVALID')
    w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
    norm_squared = sum(value * value for value in (w, x, y, z))
    magnitude = np.maximum(1., np.max(np.abs(vectors), axis=1))
    normalized = vectors / magnitude[:, None]
    a, b, c = normalized.T
    cross = np.column_stack((y*c - z*b, z*a - x*c, x*b - y*a))
    second = np.column_stack((y*cross[:, 2] - z*cross[:, 1],
                              z*cross[:, 0] - x*cross[:, 2],
                              x*cross[:, 1] - y*cross[:, 0]))
    with np.errstate(over='ignore', invalid='ignore'):
        result = (normalized + (2. / norm_squared) * (w*cross + second)) * magnitude[:, None]
    if not np.isfinite(result).all():
        raise ValueError('METRIC_ROTATION_RESULT_NONFINITE')
    return result


# 功能：
#   1. 按右手坐标旋转三维向量，消除契约允许的四元数模长误差，不改变物理长度。
#   2. 拒绝非法姿态及非有限向量；缩放中间计算以避免可表示结果被中间溢出破坏。
#   3. 只做几何转换，不校正姿态观测误差；前右上等左手轴的反射由调用边界处理。
# 输入：
#   quaternion：标量在前、模长位于既有契约容差内的姿态四元数。
#   vector：与该姿态输入坐标系一致的三维有限向量。
# 输出：
#   result：位于目标坐标系、单位与输入一致的旋转向量。
def rotate_vector(quaternion: QuaternionWxyz, vector: Point) -> Point:
    components = (quaternion.w, quaternion.x, quaternion.y, quaternion.z)
    if any(type(value) not in (int, float) for value in components):
        raise ValueError("METRIC_QUATERNION_INVALID")
    try:
        norm_squared = sum(value * value for value in components)
        if not .999**2 <= norm_squared <= 1.001**2:
            raise ValueError("METRIC_QUATERNION_INVALID")
    except OverflowError as error:
        raise ValueError("METRIC_QUATERNION_INVALID") from error
    try:
        if len(vector) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in vector
        ):
            raise ValueError("METRIC_ROTATION_VECTOR_INVALID")
        magnitude = max(1., *(abs(value) for value in vector))
        normalized = tuple(value / magnitude for value in vector)
    except (TypeError, OverflowError) as error:
        raise ValueError("METRIC_ROTATION_VECTOR_INVALID") from error
    w, x, y, z = components
    a, b, c = normalized
    cross = (y*c - z*b, z*a - x*c, x*b - y*a)
    second_cross = (y*cross[2] - z*cross[1], z*cross[0] - x*cross[2],
                    x*cross[1] - y*cross[0])
    # 2/||q||² 与先归一化四元数等价；不修改原始观测及其内容摘要，也不逐射线开方。
    factor = 2. / norm_squared
    result = tuple((normalized[i] + factor * (w*cross[i] + second_cross[i])) * magnitude
                   for i in range(3))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("METRIC_ROTATION_RESULT_NONFINITE")
    return result  # type: ignore[return-value]
