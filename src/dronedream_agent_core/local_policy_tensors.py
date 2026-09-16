"""Shared tensor checks for live inference and build-time interface probes."""


# 功能：
#   验证实际 float32 单向量或单批次张量，不通过隐式转换、任意展平接受错误接口。
# 输入：
#   output：ONNX 会话返回的原始数组。
#   count：合同规定的元素数量。
#   role：用于定位错误的角色名称。
# 输出：
#   values：已经验证的有限一维视图，不修改原数组。
def float_output(output: object, count: int, role: str):
    import numpy as np

    shapes = {(count,), (1, count)}
    if count == 1:
        shapes.add(())
    if (
        not isinstance(output, np.ndarray)
        or output.dtype != np.dtype("float32")
        or output.shape not in shapes
        or not np.isfinite(output).all()
    ):
        raise RuntimeError(f"LOCAL_POLICY_{role}_OUTPUT_INVALID")
    values = output.reshape(count)
    return values


# 功能：
#   读取一个有界概率或缩放系数，拒绝错误类型、秩、非有限值和越界值。
# 输入：
#   output：会话实际返回的标量张量。
#   role：错误定位的顾问角色。
#   minimum：该标量的合法下界，概率为零、负载缩放为零点一。
# 输出：
#   value：通过校验的 Python 浮点值。
def bounded_scalar(output: object, role: str, *, minimum: float = 0.0) -> float:
    values = float_output(output, 1, role)
    value = float(values[0])
    if not minimum <= value <= 1.0:
        raise RuntimeError(f"LOCAL_POLICY_{role}_OUTPUT_OUT_OF_RANGE")
    return value
