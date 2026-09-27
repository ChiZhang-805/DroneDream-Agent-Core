#!/usr/bin/env python3
"""Evaluate an already-trained advisor without touching its weights or flight selection."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.training.advisor_evaluation import evaluate_frozen_advisor
from dronedream_agent_core.training.dagger_artifacts import write_rows


# 功能：
#   核对明确输入后进行独立 ONNX 测试，报告只写入新的路径，不覆盖训练证据。
# 输入：
#   命令行参数：专家、模型、训练回执、新测试数据及输出路径。
# 输出：
#   exit_code：测试达标为零，指标未达标为一；来源不合法直接报错。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True)
    for name in ("artifact", "training-receipt", "dataset-receipt", "test-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    check_plain_plugin_path(args.output.absolute())
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate_frozen_advisor(role=args.role, artifact=args.artifact,
        training_receipt=args.training_receipt, dataset_receipt=args.dataset_receipt,
        test_data=args.test_data)
    write_rows(args.output, [result])
    print(json.dumps(result, allow_nan=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
