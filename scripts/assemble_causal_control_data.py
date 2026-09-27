"""Assemble exact encoded control shards; this command never fits model weights."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.training.control_shards import assemble_control_shards


# 功能：
#   在训练前生成可重验的三划分组合输入，缺失窗口明确记录，不启用 GPU 或仿真。
# 输入：
#   无：命令行提供冻结索引、新目录和视觉特征维数。
# 输出：
#   report：标准输出的完整合并回执，同时保存到新目录。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visual-feature-count', type=int, required=True)
    args = parser.parse_args()
    report = assemble_control_shards(args.inventory, args.output, feature_count=args.visual_feature_count)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
