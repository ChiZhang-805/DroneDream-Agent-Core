"""Verify a relocated control dataset against its separately retained assembly digest."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.training.control_shards import verify_assembled_control_data


# 功能：
#   上传前后按独立保存的回执摘要逐文件校验控制数据；不训练、不读取外部来源目录。
# 输入：
#   无：命令行提供 dataset 目录及 sha256 回执摘要。
# 输出：
#   exit_code：完整性检查通过为零，失败抛出错误。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    report = verify_assembled_control_data(args.dataset, args.sha256)
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
