"""Small learned yaw component; not an aircraft controller or qualified model pack."""

import io
from pathlib import Path

import torch
from torch import nn

from ..heading_observation import HEADING_OBSERVATION_SHA256, HEADING_OBSERVATION_WIDTH


class HeadingPolicy(nn.Module):
    """Learn signed continuous yaw amplitude from measured heading geometry."""

    # 功能：
    #   构建八维感知到连续偏航幅度的独立小网络，不内置教师角速度公式。
    # 输入：
    #   self：新建模型实例。
    # 输出：
    #   None：创建可训练权重；未训练实例不具有控制资格。
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(HEADING_OBSERVATION_WIDTH, 32), nn.SiLU(),
                                     nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

    # 功能：
    #   根据当前感知输出连续转向幅度；部署时仍须乘当前限额并经过安全层。
    # 输入：
    #   self：经过训练且独立验证的候选模型；features：批次八维观测。
    # 输出：
    #   yaw_axis：负一至一、顺时针为正的归一化偏航请求。
    def forward(self, features):
        yaw_axis = self.network(features)
        return yaw_axis


# 功能：
#   只载入已绑定感知契约、完整键集合和有限浮点参数的候选，不接受旧输入权重。
# 输入：
#   content：调用方已核对文件摘要的检查点字节。
# 输出：
#   model：CPU 评价模式的偏航候选，不授予整机飞行资格。
def load_heading_policy(content: bytes) -> HeadingPolicy:
    if type(content) is not bytes or not 0 < len(content) <= 1024*1024:
        raise ValueError('HEADING_CHECKPOINT_SIZE_INVALID')
    payload = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
    if (type(payload) is not dict
            or set(payload) != {'architecture', 'heading_observation_sha256', 'state_dict', 'qualified_for_flight'}
            or payload.get('architecture') != 'heading-observation-mlp-v1'
            or payload.get('heading_observation_sha256') != HEADING_OBSERVATION_SHA256
            or payload.get('qualified_for_flight') is not False):
        raise ValueError('HEADING_CHECKPOINT_CONTRACT_MISMATCH')
    model = HeadingPolicy()
    state = payload.get('state_dict')
    expected = model.state_dict()
    if (type(state) is not dict or set(state) != set(expected)
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                   or value.shape != expected[key].shape or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError('HEADING_CHECKPOINT_WEIGHTS_INVALID')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# 功能：
#   独占保存语义绑定的有限权重，不覆盖原候选或正式软件模型。
# 输入：
#   model：训练得到的偏航候选；path：尚不存在的检查点路径。
# 输出：
#   None：写入带明确未授予飞行资格标志的检查点。
def save_heading_policy(model: HeadingPolicy, path: Path) -> None:
    if not isinstance(model, HeadingPolicy):
        raise ValueError('HEADING_MODEL_TYPE_INVALID')
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('HEADING_CHECKPOINT_WEIGHTS_INVALID')
    with path.open('xb') as stream:
        torch.save(dict(architecture='heading-observation-mlp-v1', state_dict=state,
            heading_observation_sha256=HEADING_OBSERVATION_SHA256, qualified_for_flight=False), stream)
