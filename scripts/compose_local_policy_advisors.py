#!/usr/bin/env python3
"""Compose accepted local advisor artifacts into one unqualified policy package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronedream_agent_core.local_policy_composition import (
    COMPOSABLE_ADVISOR_ROLES,
    compose_local_policy_advisors,
    load_local_advisor_artifact_evidence,
)


# 功能：
#   解析显式顾问角色和来源，交由内容绑定组合器产生候选包，不修改当前软件模型选择。
# 输入：
#   无：配置来自命令行参数。
# 输出：
#   exit_code：候选包和独立回执成功发布时为零。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--composition-receipt", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--advisor",
        action="append",
        nargs=3,
        metavar=("ROLE", "ARTIFACT", "TRAINING_RECEIPT"),
        required=True,
        help="Repeat for each advisor that should be added or replaced.",
    )
    args = parser.parse_args()
    additions = []
    for raw_role, raw_artifact, raw_receipt in args.advisor:
        if raw_role not in COMPOSABLE_ADVISOR_ROLES:
            parser.error(f"unsupported composable advisor role: {raw_role}")
        additions.append(
            load_local_advisor_artifact_evidence(
                role=raw_role,
                artifact_path=Path(raw_artifact),
                training_receipt_path=Path(raw_receipt),
            )
        )
    package = compose_local_policy_advisors(
        base_package_path=args.base_package,
        output_package_path=args.output_package,
        composition_receipt_path=args.composition_receipt,
        package_id=args.package_id,
        display_name=args.display_name,
        additions=tuple(additions),
    )
    print(
        json.dumps(
            {
                "package": str(package.root),
                "package_id": package.manifest.package_id,
                "package_sha256": package.package_sha256,
                "roles": sorted(package.artifact_paths),
                "qualification_granted": False,
            },
            sort_keys=True,
        )
    )
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
