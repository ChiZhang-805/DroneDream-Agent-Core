"""Shared strict inputs and scoped CPU budget for offline causal training commands."""

from contextlib import contextmanager
from pathlib import Path

import torch

from dronedream_plugin_sdk.protocol import decode_json

from ..plugin_files import read_plugin_file
from .causal_policy import CausalPolicyConfig
from .causal_replay import (
    MAX_REPLAY_FILE_BYTES,
    MAX_REPLAY_MANIFEST_BYTES,
    MAX_REPLAY_RECORD_BYTES,
    MAX_REPLAY_TOTAL_BYTES,
    decode_replay,
)

SOURCE_TO_REPLAY = {
    "train": "training_replay",
    "validation": "validation_replay",
    "training_observations": "training_observations",
    "validation_observations": "validation_observations",
    "stream_groups": "stream_groups",
}


# 功能：
#   1. 一次读取六项当前训练来源并限制合计字节，重用严格因果回放解码器。
#   2. 配置、标签、无标签历史和路线分区都完成验证后才交给优化入口。
# 输入：
#   paths：恰含五项回放来源和 config 的路径字典，不接受缺失历史。
# 输出：
#   sources：实际参与解析的不可变源字节映射。
#   config：经过严格 JSON 解析和模型校验的训练配置。
#   decoded：已验证的训练／留出标签及历史。
#   manifest：与实际路线证据绑定的分区清单。
#   groups：训练与留出分别使用的独立路线组。
def read_causal_training_inputs(paths: dict[str, Path]):
    if (
        type(paths) is not dict
        or set(paths) != {*SOURCE_TO_REPLAY, "config"}
        or any(not isinstance(path, Path) for path in paths.values())
    ):
        raise ValueError("CAUSAL_TRAINING_REQUIRES_COMPLETE_SOURCE_PATHS")
    paths = {name: path.absolute() for name, path in paths.items()}
    sources = {}
    remaining = MAX_REPLAY_TOTAL_BYTES
    # 先拒绝错误配置，避免为不能启动的训练读取大份数据。
    sources["config"] = read_plugin_file(paths["config"], limit=MAX_REPLAY_RECORD_BYTES)
    config = CausalPolicyConfig.model_validate(
        decode_json(sources["config"], limit=MAX_REPLAY_RECORD_BYTES)
    )
    # 与离线网络构造器的范围一致，先于读取大型样本或创建输出目录拒绝越界。
    if config.seed > 2**63 - 1:
        raise ValueError("CAUSAL_CONFIG_SEED_INVALID")
    remaining -= len(sources["config"])
    for name in SOURCE_TO_REPLAY:
        bound = MAX_REPLAY_MANIFEST_BYTES if name == "stream_groups" else MAX_REPLAY_FILE_BYTES
        content = read_plugin_file(paths[name], limit=min(bound, remaining))
        sources[name] = content
        remaining -= len(content)
    decoded, manifest, groups = decode_replay(
        {target: sources[source] for source, target in SOURCE_TO_REPLAY.items()}
    )
    return sources, config, decoded, manifest, groups


# 功能：
#   临时设置本次本机训练线程数，无论成功或异常都恢复调用方原有设置。
# 输入：
#   count：一至八个 CPU 线程。
# 输出：
#   None：上下文不提供额外业务数据。
@contextmanager
def training_cpu_threads(count: int):
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("CAUSAL_TRAINING_CPU_THREADS_INVALID")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(count)
        yield
    finally:
        torch.set_num_threads(previous)
