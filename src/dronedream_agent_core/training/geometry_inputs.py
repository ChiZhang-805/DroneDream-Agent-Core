"""Common finite metric inputs for independent scalar and batched scoring."""

import sys


# 功能：
#   判断值是否为可表示成有限双精度数的真实数值，拒绝布尔、字符串、NaN 和巨大整数。
# 输入：
#   value：候选度量数值。
# 输出：
#   valid：数值类型与有限范围均合法时为 True。
def finite_metric(value):
    valid = type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max
    return valid


# 功能：
#   验证三维米制向量并复制成不可变元组，避免借用可变坐标列表。
# 输入：
#   value：三个有限实数构成的列表或元组。
# 输出：
#   vector：验证通过的三维米制坐标。
def metric_vector(value):
    if (type(value) not in (list, tuple) or len(value) != 3
            or not all(finite_metric(item) for item in value)):
        raise ValueError("OUTCOME_METRIC_VECTOR_INVALID")
    vector = tuple(value)
    return vector
