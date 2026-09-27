"""Bound training memory with whole observations, never selected by prediction quality."""

import re
from dataclasses import replace

from ..hashing import sha256_json
from .action_risk_artifacts import ActionRiskDataset


# 功能：
#   1. 按原始观测顺序均匀选取固定数量的整组动作，不读取模型预测、不按风险或误差挑样本。
#   2. 保留全部原始观测的来源隔离范围和原回执，另列实际优化子集及未使用数量。
# 输入：
#   dataset：已通过完整来源校验的数据集；maximum：优化中最多采用的独立观测数。
# 输出：
#   selected：使用完整动作组的训练视图，原文件和原数据对象不被修改。
def select_training_observations(dataset: ActionRiskDataset, maximum: int):
    if (not isinstance(dataset, ActionRiskDataset)
            or dataset.training_observation_selection is not None):
        raise ValueError('RISK_TRAINING_SELECTION_SOURCE_INVALID')
    if type(maximum) is not int or not 20 <= maximum <= 10000:
        raise ValueError('RISK_TRAINING_SELECTION_LIMIT_INVALID')
    identities = list(dict.fromkeys(row['observation_sha256'] for row in dataset.records))
    if not identities or len(dataset.samples) != len(dataset.records):
        raise ValueError('RISK_TRAINING_SELECTION_RECORDS_INVALID')
    count = min(maximum, len(identities))
    # 整数等距下标覆盖首尾，结果与风险标签、模型成绩和运行时间完全无关。
    indices = ([0] if count == 1 else
               [index * (len(identities)-1) // (count-1) for index in range(count)])
    chosen = [identities[index] for index in indices]
    keep = set(chosen)
    pairs = [(sample, record) for sample, record in
             zip(dataset.samples, dataset.records, strict=True)
             if record['observation_sha256'] in keep]
    selection = dict(method='uniform-source-order-complete-action-groups-v1',
        dataset_receipt_sha256=dataset.receipt_sha256, maximum_observations=maximum,
        source_observation_count=len(identities), selected_observation_count=len(chosen),
        selected_observation_sha256=chosen,
        source_order_sha256=sha256_json(identities), source_probe_count=len(dataset.samples),
        selected_probe_count=len(pairs), omitted_probe_count=len(dataset.samples)-len(pairs),
        labels_changed=False, outcomes_used_for_selection=False)
    selected = replace(dataset, samples=tuple(row[0] for row in pairs),
                       records=tuple(row[1] for row in pairs),
                       training_observation_selection=selection)
    return selected


# 功能：
#   核对训练视图与选样记录的实际身份和数量一致，不允许凭注记伪造全量训练覆盖。
# 输入：
#   dataset：已选择或未选择的训练数据视图。
# 输出：
#   receipt：经核对的独立选择回执；未选择时为 None。
def training_selection_receipt(dataset: ActionRiskDataset):
    receipt = dataset.training_observation_selection
    if receipt is None:
        return None
    ids = list(dict.fromkeys(row['observation_sha256'] for row in dataset.records))
    integer_fields = ('maximum_observations', 'source_observation_count',
                      'selected_observation_count', 'source_probe_count',
                      'selected_probe_count', 'omitted_probe_count')
    if (type(receipt) is not dict
            or set(receipt) != {*integer_fields, 'method', 'dataset_receipt_sha256',
                               'selected_observation_sha256', 'source_order_sha256',
                               'labels_changed', 'outcomes_used_for_selection'}
            or any(type(receipt.get(key)) is not int for key in integer_fields)
            or receipt.get('method') != 'uniform-source-order-complete-action-groups-v1'
            or receipt.get('dataset_receipt_sha256') != dataset.receipt_sha256
            or receipt.get('selected_observation_sha256') != ids
            or receipt.get('labels_changed') is not False
            or receipt.get('outcomes_used_for_selection') is not False
            or type(receipt.get('source_order_sha256')) is not str
            or re.fullmatch('[0-9a-f]{64}', receipt['source_order_sha256']) is None):
        raise ValueError('RISK_TRAINING_SELECTION_RECEIPT_INVALID')
    if (not 20 <= receipt['maximum_observations'] <= 10000
            or not 1 <= len(ids) == receipt['selected_observation_count']
            <= receipt['source_observation_count']
            or len(ids) != min(receipt['maximum_observations'], receipt['source_observation_count'])
            or not len(dataset.samples) == len(dataset.records) == receipt['selected_probe_count']
            or receipt['source_probe_count'] < receipt['selected_probe_count']
            or receipt['omitted_probe_count'] != (
                receipt['source_probe_count']-len(dataset.samples))):
        raise ValueError('RISK_TRAINING_SELECTION_COUNTS_INVALID')
    receipt = {**receipt, 'selected_observation_sha256': list(ids)}
    return receipt
