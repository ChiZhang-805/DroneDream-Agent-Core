"""Reannotate an existing grounded risk corpus; never changes source data."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.training.risk_reannotation import reannotate_clearance_labels


# 功能：运行可追溯净空标签迁移，拒绝覆盖，明确报告没有新增物理观测。
# 输入：命令行指定的源数据与新输出目录。
# 输出：exit_code：迁移并重新准入成功为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = reannotate_clearance_labels(args.source, args.output)
    print(json.dumps({key: receipt[key] for key in (
        'sample_count', 'class_counts', 'qualified_for_flight', 'reannotation')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
