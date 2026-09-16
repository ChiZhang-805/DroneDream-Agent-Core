"""Hash optimization must preserve the existing artifact/evidence byte contract."""

import hashlib
import json
import math
import random
from datetime import UTC, datetime
from enum import StrEnum

import pytest
from pydantic import BaseModel, RootModel, computed_field

from dronedream_agent_core.hashing import canonical_json, json_value, sha256_json


# 功能：
#   独立保留原递归规范化算法作为对照，不调用待测实现来生成预期值。
# 输入：
#   value：包含模型、时间、容器或标量的测试数据。
# 输出：
#   normalized：原契约应得到的规范化表示。
def reference_value(value):
    if isinstance(value, BaseModel):
        normalized = reference_value(value.model_dump(mode="json"))
    elif isinstance(value, datetime):
        normalized = value.isoformat()
    elif isinstance(value, dict):
        normalized = {str(key): reference_value(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        normalized = [reference_value(item) for item in value]
    else:
        normalized = value
    return normalized


# 功能：
#   用独立对照规范化结果生成原契约的紧凑 JSON 文本。
# 输入：
#   value：本例待比较的业务值。
# 输出：
#   encoded：原算法期望生成的规范 JSON 文本。
def reference_json(value):
    encoded = json.dumps(
        reference_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoded


class Mode(StrEnum):
    HOLD = "hold"


class Child(BaseModel):
    time: datetime
    floats: list[float]
    mode: Mode
    tuple_value: tuple[int, str]
    keyed: dict[int, str]

    # 功能：
    #   提供一个参与模型序列化的计算字段，用于检查优化没有遗漏此类输出。
    # 输入：
    #   self：包含浮点样本列表的测试模型实例。
    # 输出：
    #   width：当前样本数量。
    @computed_field
    @property
    def width(self) -> int:
        width = len(self.floats)
        return width


# 功能：
#   验证嵌套模型的规范字节与原算法一致，返回容器独立且保留负零等浮点细节。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_nested_pydantic_json_mode_is_already_normalized_and_detached():
    child = Child(
        time=datetime(2026, 9, 8, 9, tzinfo=UTC),
        floats=[-0.0, 1e-15, 3.141592653589793],
        mode=Mode.HOLD,
        tuple_value=(42, "无人机"),
        keyed={1: "one"},
    )
    model = RootModel[list[Child]]([child, child])
    assert canonical_json(model) == reference_json(model)
    result = json_value(model)
    result[0]["floats"][0] = 100
    assert child.floats[0] == 0
    assert math.copysign(1, child.floats[0]) == -1
    child.floats[1] = 5
    assert result[1]["floats"][1] == 1e-15


# 功能：
#   验证历史键转换、标量、元组与日期组合保持相同 JSON 字节和 SHA-256 摘要。
# 输入：
#   value：覆盖不同历史表示边界的测试数据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value",
    [
        {None: 1, "None": 2, False: 3, "False": 4, (1, 2): [3]},
        [True, False, None, "旋转", 0, -0.0, 1e-5, 1e30, 1.2345678901234567],
        (datetime(2026, 9, 8), Mode.HOLD, {5: ("x", 0.05)}),
    ],
)
def test_mixed_keys_scalars_tuples_and_dates_keep_identical_canonical_bytes(value):
    assert canonical_json(value) == reference_json(value)
    assert sha256_json(value) == hashlib.sha256(reference_json(value).encode("utf-8")).hexdigest()


# 功能：
#   使用固定种子生成嵌套传感器形状的数据，逐份比较优化前后的规范字节。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_seeded_nested_sensor_like_payloads_match_reference():
    rng = random.Random(805)
    for _ in range(100):
        value = {
            "samples": [
                {
                    "direction": {"x": rng.uniform(-1, 1), "y": rng.uniform(-1, 1), "z": 0.0},
                    "range": rng.uniform(0.2, 19.1),
                    "hit": rng.random() > 0.5,
                    "time": rng.randrange(100000),
                    "mask": (1.0, 0.0, None),
                }
                for _ in range(32)
            ]
        }
        assert canonical_json(value) == reference_json(value)


# 功能：
#   验证性能优化没有放宽 NaN 和正负无穷值的 JSON 拒绝规则。
# 输入：
#   value：本例的非有限浮点值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_still_fail_closed(value):
    with pytest.raises(ValueError):
        canonical_json({"feature": [value]})


# 功能：
#   验证普通容器的递归转换不会共享嵌套数据，未知对象仍由编码器拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_plain_containers_remain_detached_and_unknown_objects_remain_rejected():
    source = {1: [2, {3: 4}]}
    result = json_value(source)
    result["1"][1]["3"] = 5
    assert source == {1: [2, {3: 4}]}
    with pytest.raises(TypeError):
        canonical_json(object())


# 功能：
#   验证自定义模型忽略 JSON 模式时仍走原递归路径，不能错误使用标准序列化快捷路径。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_overridden_model_dump_keeps_original_normalization():
    class CustomDump(BaseModel):
        # 功能：
        #   故意返回未规范化的键、时间与元组，模拟覆盖标准方法的第三方模型。
        # 输入：
        #   kwargs：本夹具故意忽略的序列化参数。
        # 输出：
        #   payload：需要宿主继续递归规范化的原始值。
        def model_dump(self, **kwargs):
            payload = {3: (datetime(2026, 9, 8), "custom")}
            return payload

    assert canonical_json(CustomDump()) == reference_json(CustomDump())
