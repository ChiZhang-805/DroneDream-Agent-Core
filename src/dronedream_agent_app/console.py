"""Product console commands, independent of the privileged development CLI.

Credentials arrive through an inherited environment or bounded standard input;
they are not command-line flags, persistent configuration, or printed output.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TextIO

from pydantic import ValidationError

from dronedream_plugin_sdk.protocol import decode_json

from .console_client import ConsoleError, ConsoleSession, ProductConsoleClient
from .models import ThreadCreate

MAX_SESSION_BYTES = 64 * 1024
SESSION_ENV = {
    "core_url": "DRONEDREAM_CONSOLE_CORE_URL",
    "supabase_url": "DRONEDREAM_CONSOLE_SUPABASE_URL",
    "local_token": "DRONEDREAM_CONSOLE_LOCAL_TOKEN",
    "identity_token": "DRONEDREAM_CONSOLE_IDENTITY_TOKEN",
    "publishable_key": "DRONEDREAM_CONSOLE_PUBLISHABLE_KEY",
    "source_edition": "DRONEDREAM_CONSOLE_EDITION",
}


# 功能：
#   声明仅调用产品业务接口的命令；执行命令必须显式给出已确认的计划标识。
# 输入：
#   无。
# 输出：
#   parser：命令行解析器，不解析密码、API Key 或本地会话令牌参数。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dronedream-console",
        description="Account-bound DroneDream planning and simulation via desktop services.",
    )
    parser.add_argument(
        "--session-stdin",
        action="store_true",
        help="Read one bounded JSON session from standard input, never print it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("catalog", "Read current desktop models, assets and tasks."),
        ("usage", "Read the account's actual cloud usage."),
        ("runtime", "Read Runtime status; this does not authorize takeoff."),
    ):
        commands.add_parser(name, help=help_text)
    create = commands.add_parser("create", help="Create a task without planning or executing it.")
    create.add_argument("--model", required=True)
    create.add_argument("--title")
    create.add_argument("--locale", choices=("zh-CN", "en-US"), default="zh-CN")
    prepare = commands.add_parser("prepare", help="Request real cloud planning; may consume quota.")
    prepare.add_argument("thread_id")
    prepare.add_argument("--text", required=True)
    prepare.add_argument("--map", required=True, dest="map_id")
    prepare.add_argument("--map-sha256", required=True)
    prepare.add_argument("--vehicle", required=True, dest="vehicle_id")
    prepare.add_argument("--vehicle-sha256", required=True)
    prepare.add_argument("--start", default="__auto__")
    execute = commands.add_parser("execute", help="Confirm this exact plan and request simulation.")
    execute.add_argument("thread_id")
    execute.add_argument(
        "--confirm-plan",
        required=True,
        dest="plan_revision_id",
        help="Exact plan-... ID that you have reviewed. No automatic confirmation.",
    )
    for name in ("status", "evidence"):
        command = commands.add_parser(name)
        command.add_argument("thread_id")
    message = commands.add_parser("message", help="Submit a runtime amendment; not emergency stop.")
    message.add_argument("thread_id")
    message.add_argument("--text", required=True)
    return parser


# 功能：
#   从明确选定的单一输入来源读取会话；不搜索浏览器、桌面数据库或历史登录文件。
# 输入：
#   from_stdin：是否读取标准输入，否则使用命名环境变量。
#   stream：标准输入流。
# 输出：
#   session：校验通过的内存会话，不代表服务已认可该账户。
def read_session(from_stdin: bool, stream: TextIO) -> ConsoleSession:
    if from_stdin:
        raw = stream.readline(MAX_SESSION_BYTES + 1)
        value = decode_json(raw, limit=MAX_SESSION_BYTES, node_limit=64)
    else:
        value = {name: os.environ.get(variable, "") for name, variable in SESSION_ENV.items()}
    session = ConsoleSession.model_validate(value)
    return session


# 功能：
#   将一个命令映射到一个明确业务操作，不串联自动确认或偷偷切换教师模式。
# 输入：
#   client：使用当前真实服务地址的客户端。
#   args：完成语法解析的命令参数。
# 输出：
#   result：相应服务返回的结果。
def dispatch(client: ProductConsoleClient, args: argparse.Namespace) -> dict:
    if args.command == "catalog":
        return client.bootstrap()
    if args.command == "usage":
        return client.usage()
    if args.command == "runtime":
        return client.runtime()
    if args.command == "create":
        return client.create(
            ThreadCreate(
                title=args.title,
                selected_model=args.model,
                locale=args.locale,
            )
        )
    if args.command == "prepare":
        return client.prepare(
            args.thread_id,
            {
                "message": args.text,
                "map_id": args.map_id,
                "map_content_sha256": args.map_sha256,
                "vehicle_id": args.vehicle_id,
                "vehicle_content_sha256": args.vehicle_sha256,
                "start_entity": args.start,
            },
        )
    if args.command == "execute":
        return client.execute(args.thread_id, args.plan_revision_id)
    if args.command == "status":
        return client.status(args.thread_id)
    if args.command == "evidence":
        return client.evidence(args.thread_id)
    if args.command == "message":
        return client.message(args.thread_id, args.text)
    raise ConsoleError("CONSOLE_COMMAND_INVALID")


# 功能：
#   1. 执行一个产品命令，输出脱敏 JSON 并关闭网络资源。
#   2. 输入错误和异常不打印秘密；中断只退出客户端，不冒充后台任务已经停止。
# 输入：
#   argv：命令参数，缺省使用进程参数。
#   stdin：会话输入流，缺省使用标准输入。
#   stdout：结果输出流，缺省使用标准输出。
#   stderr：错误输出流，缺省使用标准错误。
# 输出：
#   exit_code：零表示 HTTP 业务操作成功，二表示输入错误，一表示服务失败，130 表示中断。
def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    output, errors = stdout or sys.stdout, stderr or sys.stderr
    client = None
    try:
        session = read_session(args.session_stdin, stdin or sys.stdin)
        client = ProductConsoleClient(session)
        result = dispatch(client, args)
        print(client.render(result), file=output)
        return 0
    except (ValidationError, ValueError, TypeError):
        print('{"error":"CONSOLE_INPUT_INVALID"}', file=errors)
        return 2
    except ConsoleError as error:
        # ConsoleError 只由本模块固定分类或远端代码白名单产生。
        print('{"error":"' + str(error) + '","automatic_retry":false}', file=errors)
        return 1
    except KeyboardInterrupt:
        print('{"error":"CONSOLE_INTERRUPTED_CHECK_TASK_STATUS"}', file=errors)
        return 130
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
