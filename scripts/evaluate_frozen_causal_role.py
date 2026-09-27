"""Evaluate only an explicitly frozen control candidate on its untouched test split."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.training.causal_frozen_test import evaluate_frozen_causal_candidate
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：
#   验证独立冻结的候选与数据后输出最终测试报告，不搜索候选、不训练、不覆盖旧证据。
# 输入：
#   无：命令行提供 candidate、dataset、两个预先保存的 SHA-256 及新 output 路径。
# 输出：
#   exit_code：实际测试和报告写入完成时为零，不代表飞行资格通过。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--training-receipt-sha256', required=True)
    parser.add_argument('--assembly-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.absolute()
    check_plain_plugin_path(target)
    if target.exists():
        raise FileExistsError(target)
    report = evaluate_frozen_causal_candidate(args.candidate, args.dataset,
        args.training_receipt_sha256, args.assembly_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_object(target, report)
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
