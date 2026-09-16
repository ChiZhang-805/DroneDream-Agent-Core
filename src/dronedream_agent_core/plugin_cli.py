"""Command-line developer workflow for DroneDream AGENT plugins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plugin_sdk import (
    build_plugin_bundle,
    generate_publisher_key,
    sandbox_plugin_bundle,
    scaffold_plugin,
    validate_plugin_source,
)


# 功能：
#   输出可机器读取的有限 JSON 回执；密钥创建入口仅传入公开字段，不打印私钥。
# 输入：
#   value：开发命令返回的公开回执。
# 输出：
#   None：不返回业务数据。
def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


# 功能：
#   分派明确指定的开发命令，仅 sandbox 命令可启动包内运行时，其他命令只处理本地材料。
# 输入：
#   无显式参数；从进程命令行读取子命令和路径参数。
# 输出：
#   exit_code：命令成功完成时的零退出码。
def main() -> int:
    parser = argparse.ArgumentParser(prog="dronedream-plugin")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("directory", type=Path)
    init.add_argument("--plugin-id", required=True)
    init.add_argument("--name", required=True)
    init.add_argument("--publisher", required=True)
    init.add_argument("--kind", choices=["mcp", "ui"], default="mcp")
    validate = commands.add_parser("validate")
    validate.add_argument("directory", type=Path)
    keygen = commands.add_parser("keygen")
    keygen.add_argument("output", type=Path)
    keygen.add_argument("--key-id", required=True)
    keygen.add_argument("--publisher", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("directory", type=Path)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--signing-key", type=Path)
    sandbox = commands.add_parser("sandbox")
    sandbox.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        created = scaffold_plugin(
            args.directory,
            plugin_id=args.plugin_id,
            name=args.name,
            publisher=args.publisher,
            kind=args.kind,
        )
        _print({"directory": str(created.resolve())})
    elif args.command == "validate":
        _print(validate_plugin_source(args.directory))
    elif args.command == "keygen":
        _print(generate_publisher_key(args.output, key_id=args.key_id, publisher=args.publisher))
    elif args.command == "pack":
        _print(build_plugin_bundle(args.directory, args.output, signing_key=args.signing_key))
    elif args.command == "sandbox":
        _print(sandbox_plugin_bundle(args.bundle))
    exit_code = 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
