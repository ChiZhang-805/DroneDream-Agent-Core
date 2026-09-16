"""Detach validated numeric evidence without deepcopy's per-number dispatch.

The containers must be owned by the recorder; immutable numbers and strings
can be shared. This is not validation, serialization or a trust-boundary bypass.
Only existing validated models and their JSON-shaped dumps belong here.
"""

from pydantic import BaseModel

from ..pilot_control_mapping import PilotControlLimits

_ATOMS = frozenset({str, int, float, bool, type(None)})
MAX_EVIDENCE_NODES = 1_000_000


# 功能：
#   1. 复制已验证证据的可变容器，保留数值和冻结限额，不替代业务契约验证。
#   2. 同时限制递归深度和展开节点数，避免循环或重复子树导致无界复制。
# 输入：
#   value：模型、文本键字典、列表、元组或受支持的不可变值。
# 输出：
#   result：具有独立容器所有权且保留原数据类型的证据快照。
def detach_evidence(value):
    remaining = MAX_EVIDENCE_NODES

    # 功能：
    #   在同一份节点预算内递归复制字段、模型附加状态及容器。
    # 输入：
    #   item：当前待复制子树。
    #   depth：当前嵌套层数。
    # 输出：
    #   copied：当前子树的独立快照或可共享的不可变值。
    def detach(item, depth):
        nonlocal remaining
        if depth > 64:
            raise ValueError("TRAINING_EVIDENCE_NESTING_EXCEEDED")
        remaining -= 1
        if remaining < 0:
            raise ValueError("TRAINING_EVIDENCE_NODE_BUDGET_EXCEEDED")
        kind = type(item)
        if kind in _ATOMS or kind is PilotControlLimits:
            copied = item  # 限额是全标量冻结类；不接受其可能带可变字段的子类。
        elif kind is list or kind is tuple:
            if len(item) > remaining:
                raise ValueError("TRAINING_EVIDENCE_NODE_BUDGET_EXCEEDED")
            if all(type(child) in _ATOMS for child in item):
                # 高频数值向量一次扣除预算，不为每个数字创建递归调用帧。
                remaining -= len(item)
                copied = item.copy() if kind is list else tuple(item)
            else:
                children = [detach(child, depth + 1) for child in item]
                copied = children if kind is list else tuple(children)
        elif kind is dict:
            if len(item) > remaining:
                raise ValueError("TRAINING_EVIDENCE_NODE_BUDGET_EXCEEDED")
            if not all(type(key) is str for key in item):
                raise ValueError("TRAINING_EVIDENCE_KEYS_MUST_BE_TEXT")
            copied = {key: detach(child, depth + 1) for key, child in item.items()}
        elif isinstance(item, BaseModel):
            # model_copy 的浅拷贝会保留 extra/private/cache 的嵌套引用；全部复制后再交出。
            fields = detach(item.__dict__, depth + 1)
            extra = detach(item.__pydantic_extra__, depth + 1)
            private = detach(item.__pydantic_private__, depth + 1)
            copied = item.model_copy()
            object.__setattr__(copied, "__dict__", fields)
            object.__setattr__(copied, "__pydantic_extra__", extra)
            object.__setattr__(copied, "__pydantic_private__", private)
        else:
            raise ValueError("TRAINING_EVIDENCE_UNSUPPORTED_VALUE:" + kind.__name__)
        return copied

    result = detach(value, 0)
    return result
