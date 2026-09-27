#!/usr/bin/env python3
"""Train the risk-only expert with whole-route holdout, never an actor."""

import argparse
import json
import sys
from pathlib import Path

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingConfig
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.training.action_risk_artifacts import load_action_risk_dataset
from dronedream_agent_core.training.action_risk_training import train_action_risk_expert
from dronedream_agent_core.training.risk_training_sampling import select_training_observations
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   检查配置及独立数据集入口，只训练动作风险评分，不授权控制或覆盖既有训练结果。
# 输入：
#   命令行参数：训练、验证数据集目录，训练配置和新的输出目录。
# 输出：
#   exit_code：离线指标通过时为零，指标未通过时为二。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset", type=Path, action="append", required=True)
    parser.add_argument("--validation-dataset", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classification-margin", type=float, default=0.0)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--physical-context-only", action="store_true")
    parser.add_argument("--monotonic-geometry", action="store_true")
    parser.add_argument("--geometry-distance-cap-m", type=float)
    parser.add_argument("--unsafe-sample-cost", type=float, default=1.)
    parser.add_argument("--contrast-loss-weight", type=float, default=0.)
    parser.add_argument("--diagnostic-dataset", type=Path, action="append", default=[])
    parser.add_argument("--maximum-training-observations-per-dataset", type=int)
    args = parser.parse_args()
    for paths in (args.training_dataset, args.validation_dataset):
        if not 1 <= len(paths) <= 128:
            parser.error("dataset count must be 1..128 per split")
    args.output = args.output.absolute()
    if len(args.diagnostic_dataset) > 16:
        parser.error("diagnostic dataset count must be at most 16")
    if (args.maximum_training_observations_per_dataset is not None
            and not 20 <= args.maximum_training_observations_per_dataset <= 10000):
        parser.error("training observation limit must be 20..10000")
    check_plain_plugin_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = LocalPolicyTrainingConfig.model_validate(
        decode_json(read_plugin_file(args.config, limit=4 * 1024**2), limit=4 * 1024**2)
    )
    training = []
    for index, path in enumerate(args.training_dataset, 1):
        print(f'Loading training source {index}/{len(args.training_dataset)}',
              file=sys.stderr, flush=True)
        dataset = load_action_risk_dataset(path)
        if args.maximum_training_observations_per_dataset is not None:
            dataset = select_training_observations(
                dataset, args.maximum_training_observations_per_dataset)
        training.append(dataset)
        del dataset
    validation = []
    for index, path in enumerate(args.validation_dataset, 1):
        print(f'Loading validation source {index}/{len(args.validation_dataset)}',
              file=sys.stderr, flush=True)
        validation.append(load_action_risk_dataset(path))
    # 诊断旧集逐个核验后只保留身份，不长期占用标签内存，更不交给优化器。
    diagnostics = []
    for index, path in enumerate(args.diagnostic_dataset, 1):
        print(f'Checking excluded diagnostic source {index}/{len(args.diagnostic_dataset)}',
              file=sys.stderr, flush=True)
        diagnostic = load_action_risk_dataset(path)
        diagnostics.append(dict(dataset_receipt_sha256=diagnostic.receipt_sha256,
            teacher_config_sha256=diagnostic.teacher_config_sha256,
            groups=sorted(diagnostic.groups)))
        del diagnostic
    print('Validated source loading complete; starting offline training',
          file=sys.stderr, flush=True)
    receipt = train_action_risk_expert(training, validation, config, args.output,
                                     classification_margin=args.classification_margin,
                                     normalize_inputs=args.normalize_inputs,
                                     physical_context_only=args.physical_context_only,
                                     monotonic_geometry=args.monotonic_geometry,
                                     geometry_distance_cap_m=args.geometry_distance_cap_m,
                                     unsafe_sample_cost=args.unsafe_sample_cost,
                                     contrast_loss_weight=args.contrast_loss_weight,
                                     diagnostic_sources=diagnostics)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt["offline_validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
