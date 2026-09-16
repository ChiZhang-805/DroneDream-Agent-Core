#!/usr/bin/env python3
"""Assemble ten explicit experts without relying on a historical base package."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_agent_core.training.artifact_assembly import (
    CompleteEnsembleRecipe,
    assemble_complete_ensemble,
)
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   1. 有界读取显式十专家组装配方，拒绝重复 JSON 键、链接及读取期间文件变化。
#   2. 组装来源绑定的候选包并输出结果，不覆盖已有目录或授予飞行资格。
# 输入：
#   命令行参数：recipe 为配方文件，output 为新的模型包输出目录。
# 输出：
#   exit_code：组装和结果输出完成时为零，输入或组装失败时抛出错误。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe_path = args.recipe.absolute()
    content = read_plugin_file(recipe_path, limit=4 * 1024 * 1024)
    recipe = CompleteEnsembleRecipe.model_validate(decode_json(content, limit=4 * 1024 * 1024))
    package = assemble_complete_ensemble(
        recipe=recipe, source_root=recipe_path.parent, output_root=args.output.absolute()
    )
    print(
        json.dumps(
            {
                "package_sha256": package.package_sha256,
                "expert_count": len(package.artifact_paths),
                "qualified_for_flight": False,
            }
        )
    )
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
