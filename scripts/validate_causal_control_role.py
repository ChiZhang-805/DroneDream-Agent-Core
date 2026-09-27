"""Validate a trained control expert's actual ONNX export against frozen validation windows."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.training.causal_export_validation import validate_causal_candidate
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：
#   校验单专家候选的真实部署推理，独占保存证据，不替换软件模型包或授予飞行权限。
# 输入：
#   无：命令行提供候选目录及新报告路径。
# 输出：
#   exit_code：全部一致性检查及报告保存成功时为零。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.absolute()
    check_plain_plugin_path(target)
    if target.exists():
        raise FileExistsError(target)
    report = validate_causal_candidate(args.candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_object(target, report)
    print(json.dumps(report, allow_nan=False))
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
