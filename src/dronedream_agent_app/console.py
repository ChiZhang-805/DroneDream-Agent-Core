"""Account-bound DroneDream CLI built on the desktop product services.

The CLI is a presentation and orchestration layer.  Natural-language planning,
tool calls, deterministic validation, PX4/Gazebo execution, and evidence all
remain in Agent Core, exactly as they do for the desktop application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from dronedream_plugin_sdk.protocol import decode_json

from .console_client import (
    ConsoleClarificationRequired,
    ConsoleError,
    ConsoleSession,
    ProductConsoleClient,
)
from .console_ui import BLUE, MUTED, TerminalUI
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


def _thread(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("thread_id")


def _prepare_arguments(parser: argparse.ArgumentParser) -> None:
    _thread(parser)
    parser.add_argument("--text", required=True)
    parser.add_argument("--map", required=True, dest="map_id")
    parser.add_argument("--map-sha256", required=True)
    parser.add_argument("--aircraft", "--vehicle", required=True, dest="vehicle_id")
    parser.add_argument(
        "--aircraft-sha256", "--vehicle-sha256", required=True, dest="vehicle_sha256"
    )
    parser.add_argument("--start", default="__auto__")


def _execute_arguments(parser: argparse.ArgumentParser) -> None:
    _thread(parser)
    parser.add_argument(
        "--confirm-plan",
        required=True,
        dest="plan_revision_id",
        help="Exact plan-... revision reviewed by the operator.",
    )


def _frame_arguments(parser: argparse.ArgumentParser) -> None:
    _thread(parser)
    parser.add_argument("--output", type=Path, required=True, help="New PNG path; never overwrite.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dronedream",
        description="Plan, fly, observe, and archive DroneDream missions from one terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""common journeys:
  dronedream doctor
  dronedream models
  dronedream assets maps
  dronedream assets aircraft
  dronedream mission fly "Inspect the library entrance, then return."
  dronedream history list --archived-only

Use --json for automation. Interactive output honors NO_COLOR and --no-color.""",
    )
    parser.add_argument(
        "--session-stdin",
        action="store_true",
        help="Read one bounded JSON session line from stdin.",
    )
    parser.add_argument("--json", action="store_true", help="Emit redacted machine-readable JSON.")
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI color in an interactive terminal."
    )
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("shell", help="Open the interactive DroneDream command console.")

    commands.add_parser("doctor", help="Check account, Core, Runtime, model, and asset readiness.")
    commands.add_parser("catalog", help="Read the shared desktop product catalog.")
    commands.add_parser("account", help="Show the signed-in account and current compute allowance.")
    commands.add_parser("usage", help="Read actual account model usage.")

    models = commands.add_parser("models", help="List the real managed-model catalog.")
    models.add_argument("model_id", nargs="?")

    assets = commands.add_parser(
        "assets", help="Inspect maps, aircraft, and qualified combinations."
    )
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    maps = asset_commands.add_parser("maps")
    maps.add_argument("resource_id", nargs="?")
    aircraft = asset_commands.add_parser("aircraft")
    aircraft.add_argument("resource_id", nargs="?")
    pairs = asset_commands.add_parser("pairs")
    pairs.add_argument("--map", dest="map_id")
    pairs.add_argument("--aircraft", dest="vehicle_id")

    runtime = commands.add_parser("runtime", help="Inspect or prepare the product-managed Runtime.")
    runtime.add_argument(
        "action",
        choices=("status", "setup-status", "setup", "provision"),
        nargs="?",
        default="status",
    )

    mission = commands.add_parser(
        "mission", help="Create, plan, fly, observe, and archive missions."
    )
    mission_commands = mission.add_subparsers(dest="mission_command", required=True)
    mission_list = mission_commands.add_parser("list")
    mission_list.add_argument("--all", action="store_true", dest="include_archived")
    mission_show = mission_commands.add_parser("show")
    _thread(mission_show)
    create = mission_commands.add_parser("create")
    create.add_argument("--model", required=True)
    create.add_argument("--title")
    create.add_argument("--locale", choices=("zh-CN", "en-US"), default="en-US")
    plan = mission_commands.add_parser("plan")
    _prepare_arguments(plan)
    execute = mission_commands.add_parser("execute")
    _execute_arguments(execute)
    fly = mission_commands.add_parser(
        "fly",
        help="Run the reviewed end-to-end product workflow.",
        description=(
            "Turn a natural-language goal into a reviewed, real-model mission "
            "and run it in PX4/Gazebo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  dronedream mission fly "Carry the package to the chemistry building and return."
  dronedream mission fly "Inspect the library entrance." --model gpt-5.4
  dronedream mission fly "Check the roof." --plan-only

When model, map, or aircraft is omitted, an interactive terminal presents a numbered selector.
Execution starts only after the reviewed route is shown and you type FLY.""",
    )
    fly.add_argument("text", help="The operator's natural-language mission.")
    fly.add_argument("--model")
    fly.add_argument("--map", dest="map_id")
    fly.add_argument("--aircraft", dest="vehicle_id")
    fly.add_argument("--title")
    fly.add_argument("--locale", choices=("zh-CN", "en-US"), default="en-US")
    fly.add_argument("--start", default="__auto__")
    fly.add_argument("--plan-only", action="store_true")
    fly.add_argument("--no-archive", action="store_true")
    fly.add_argument("--poll-seconds", type=float, default=3.0)
    fly.add_argument("--timeout-seconds", type=float, default=7200.0)
    for name in ("watch", "evidence", "telemetry", "live-sources"):
        item = mission_commands.add_parser(name)
        _thread(item)
        if name == "watch":
            item.add_argument("--poll-seconds", type=float, default=3.0)
            item.add_argument("--timeout-seconds", type=float, default=7200.0)
    frame = mission_commands.add_parser("frame")
    _frame_arguments(frame)
    message = mission_commands.add_parser("message")
    _thread(message)
    message.add_argument("--text", required=True)
    for name, archived in (
        ("archive", True),
        ("unarchive", False),
        ("pin", False),
        ("unpin", False),
    ):
        item = mission_commands.add_parser(name)
        _thread(item)
        item.set_defaults(patch_field="archived" if "archive" in name else "pinned")
        item.set_defaults(patch_value=archived if "archive" in name else name == "pin")
    export = mission_commands.add_parser("export")
    _thread(export)
    export.add_argument("--output", type=Path, required=True)

    history = commands.add_parser("history", help="Inspect or export saved missions.")
    history_commands = history.add_subparsers(dest="history_command", required=True)
    history_list = history_commands.add_parser("list")
    history_list.add_argument("--archived-only", action="store_true")
    history_show = history_commands.add_parser("show")
    _thread(history_show)
    history_export = history_commands.add_parser("export")
    _thread(history_export)
    history_export.add_argument("--output", type=Path, required=True)

    # Stable flat commands retained for scripts written against the first console release.
    legacy_create = commands.add_parser("create")
    legacy_create.add_argument("--model", required=True)
    legacy_create.add_argument("--title")
    legacy_create.add_argument("--locale", choices=("zh-CN", "en-US"), default="zh-CN")
    legacy_prepare = commands.add_parser("prepare")
    _prepare_arguments(legacy_prepare)
    legacy_execute = commands.add_parser("execute")
    _execute_arguments(legacy_execute)
    for name in ("status", "evidence", "live-sources", "telemetry"):
        item = commands.add_parser(name)
        _thread(item)
    legacy_frame = commands.add_parser("frame")
    _frame_arguments(legacy_frame)
    legacy_message = commands.add_parser("message")
    _thread(legacy_message)
    legacy_message.add_argument("--text", required=True)
    return parser


def read_session(from_stdin: bool, stream: TextIO) -> ConsoleSession:
    if from_stdin:
        value = decode_json(
            stream.readline(MAX_SESSION_BYTES + 1), limit=MAX_SESSION_BYTES, node_limit=64
        )
    else:
        value = {name: os.environ.get(variable, "") for name, variable in SESSION_ENV.items()}
    return ConsoleSession.model_validate(value)


def _selection(args: argparse.Namespace) -> dict:
    return {
        "message": args.text,
        "map_id": args.map_id,
        "map_content_sha256": args.map_sha256,
        "vehicle_id": args.vehicle_id,
        "vehicle_content_sha256": args.vehicle_sha256,
        "start_entity": args.start,
    }


def _filter_resource(catalog: dict, resource_id: str | None) -> dict:
    if not resource_id:
        return catalog
    resources = [
        item
        for item in catalog.get("resources", [])
        if isinstance(item, dict) and item.get("resource_id") == resource_id
    ]
    if len(resources) != 1:
        raise ConsoleError("RESOURCE_NOT_FOUND")
    return {**catalog, "resources": resources}


def _filter_pairs(catalog: dict, map_id: str | None, vehicle_id: str | None) -> dict:
    pairs = [
        item
        for item in catalog.get("pairs", [])
        if isinstance(item, dict)
        and (not map_id or item.get("map_resource_id") == map_id)
        and (not vehicle_id or item.get("vehicle_resource_id") == vehicle_id)
    ]
    counts: dict[str, int] = {}
    for item in pairs:
        status = item.get("status")
        if isinstance(status, str):
            counts[status] = counts.get(status, 0) + 1
    return {**catalog, "counts": counts, "pairs": pairs}


def dispatch(client: ProductConsoleClient, args: argparse.Namespace) -> dict:
    """Dispatch non-interactive commands without silently confirming a plan."""
    if args.command == "catalog":
        return client.bootstrap()
    if args.command == "account":
        return {
            "account": {
                "display_name": os.environ.get("DRONEDREAM_CONSOLE_ACCOUNT_NAME", "Pilot"),
                "email": os.environ.get("DRONEDREAM_CONSOLE_ACCOUNT_EMAIL", ""),
                "edition": client.session.source_edition,
            },
            **client.usage(),
        }
    if args.command == "usage":
        return client.usage()
    if args.command == "models":
        result = client.models()
        if args.model_id:
            matches = [
                item
                for item in result.get("models", [])
                if isinstance(item, dict) and item.get("id") == args.model_id
            ]
            if len(matches) != 1:
                raise ConsoleError("RESOURCE_NOT_FOUND")
            result = {"models": matches}
        return result
    if args.command == "assets":
        if args.asset_command == "maps":
            return _filter_resource(client.maps(), args.resource_id)
        if args.asset_command == "aircraft":
            return _filter_resource(client.aircraft(), args.resource_id)
        return _filter_pairs(client.asset_pairs(), args.map_id, args.vehicle_id)
    if args.command == "runtime":
        return {
            "status": client.runtime,
            "setup-status": client.runtime_setup,
            "setup": client.start_runtime_setup,
            "provision": client.provision_runtime,
        }[args.action]()
    if args.command == "mission":
        sub = args.mission_command
        if sub == "list":
            return client.list_threads(include_archived=args.include_archived)
        if sub == "show":
            return client.status(args.thread_id)
        if sub == "create":
            return client.create(
                ThreadCreate(title=args.title, selected_model=args.model, locale=args.locale)
            )
        if sub == "plan":
            return _require_model_generated_plan(client.prepare(args.thread_id, _selection(args)))
        if sub == "execute":
            return client.execute(args.thread_id, args.plan_revision_id)
        if sub == "evidence":
            return client.evidence(args.thread_id)
        if sub == "telemetry":
            return client.telemetry(args.thread_id)
        if sub == "live-sources":
            return client.live_sources(args.thread_id)
        if sub == "frame":
            return save_overview_frame(client, args.thread_id, args.output)
        if sub == "message":
            return client.message(args.thread_id, args.text)
        if sub in {"archive", "unarchive", "pin", "unpin"}:
            return client.patch_thread(args.thread_id, {args.patch_field: args.patch_value})
        if sub == "export":
            return export_mission(client, args.thread_id, args.output)
        raise ConsoleError("CONSOLE_INTERACTIVE_COMMAND_REQUIRED")
    if args.command == "history":
        if args.history_command == "list":
            result = client.list_threads(include_archived=True)
            if args.archived_only:
                result["threads"] = [
                    item
                    for item in result.get("threads", [])
                    if isinstance(item, dict) and item.get("archived")
                ]
            return result
        if args.history_command == "show":
            return client.status(args.thread_id)
        return export_mission(client, args.thread_id, args.output)
    if args.command == "create":
        return client.create(
            ThreadCreate(title=args.title, selected_model=args.model, locale=args.locale)
        )
    if args.command == "prepare":
        return _require_model_generated_plan(client.prepare(args.thread_id, _selection(args)))
    if args.command == "execute":
        return client.execute(args.thread_id, args.plan_revision_id)
    if args.command == "status":
        return client.status(args.thread_id)
    if args.command == "evidence":
        return client.evidence(args.thread_id)
    if args.command == "live-sources":
        return client.live_sources(args.thread_id)
    if args.command == "telemetry":
        return client.telemetry(args.thread_id)
    if args.command == "frame":
        return save_overview_frame(client, args.thread_id, args.output)
    if args.command == "message":
        return client.message(args.thread_id, args.text)
    raise ConsoleError("CONSOLE_COMMAND_INVALID")


def save_overview_frame(client: ProductConsoleClient, thread_id: str, output: Path) -> dict:
    png = client.live_frame(thread_id)
    try:
        with output.open("xb") as stream:
            stream.write(png)
    except FileExistsError as error:
        raise ConsoleError("CONSOLE_FRAME_OUTPUT_EXISTS") from error
    except OSError as error:
        raise ConsoleError("CONSOLE_FRAME_WRITE_FAILED") from error
    return {
        "thread_id": thread_id,
        "path": str(output.absolute()),
        "sha256": hashlib.sha256(png).hexdigest(),
        "bytes": len(png),
        "view": "runtime-overview",
        "qualification_granted": False,
    }


def export_mission(client: ProductConsoleClient, thread_id: str, output: Path) -> dict:
    thread = client.status(thread_id)
    evidence: dict | None = None
    try:
        evidence = client.evidence(thread_id)
    except ConsoleError as error:
        if str(error) != "EXECUTION_EVIDENCE_NOT_FOUND":
            raise
    bundle = {
        "schema_version": "dronedream.cli-mission-export.v1",
        "thread": thread,
        "execution_evidence": evidence,
    }
    redacted = client.render(bundle).encode("utf-8")
    try:
        with output.open("xb") as stream:
            stream.write(redacted)
            stream.write(b"\n")
    except FileExistsError as error:
        raise ConsoleError("CONSOLE_EXPORT_OUTPUT_EXISTS") from error
    except OSError as error:
        raise ConsoleError("CONSOLE_EXPORT_WRITE_FAILED") from error
    return {
        "thread_id": thread_id,
        "path": str(output.absolute()),
        "sha256": hashlib.sha256(redacted + b"\n").hexdigest(),
        "bytes": len(redacted) + 1,
    }


def _resolve_named(
    items: list[dict],
    requested: str | None,
    kind: str,
    *,
    interactive: bool,
    stdin: TextIO,
    ui: TerminalUI,
) -> dict:
    def names(item: dict) -> set[str]:
        display = item.get("display_name", {})
        localized = (
            set(str(value).casefold() for value in display.values())
            if isinstance(display, dict)
            else {str(display).casefold()}
        )
        identifier = item.get("id") if kind == "model" else item.get("resource_id")
        return localized | {str(identifier).casefold()}

    if not items:
        raise ConsoleError(f"CONSOLE_{kind.upper()}_CATALOG_EMPTY")
    if requested:
        matches = [item for item in items if requested.casefold() in names(item)]
        if len(matches) != 1:
            raise ConsoleError(f"CONSOLE_{kind.upper()}_SELECTION_INVALID")
        return matches[0]
    if len(items) == 1:
        return items[0]
    if not interactive:
        raise ConsoleError(f"CONSOLE_{kind.upper()}_SELECTION_REQUIRED")

    def label(item: dict) -> str:
        identifier = item.get("id") if kind == "model" else item.get("resource_id")
        display_names = item.get("display_name", {})
        if isinstance(display_names, dict):
            display = str(
                display_names.get("en-US")
                or display_names.get("zh-CN")
                or next(iter(display_names.values()), "")
            )
        else:
            display = str(display_names or "")
        return f"{display} ({identifier})" if display and display != identifier else str(identifier)

    ui.choice(kind, [label(item) for item in items])
    ui.prompt("Your choice", f"1–{len(items)}")
    raw = stdin.readline().strip()
    if not raw.isdigit() or not 1 <= int(raw) <= len(items):
        raise ConsoleError(f"CONSOLE_{kind.upper()}_SELECTION_INVALID")
    return items[int(raw) - 1]


def _qualified_selection(
    client: ProductConsoleClient,
    args: argparse.Namespace,
    *,
    interactive: bool,
    stdin: TextIO,
    ui: TerminalUI,
) -> tuple[dict, dict, dict, dict, dict]:
    models = [
        item
        for item in client.models().get("models", [])
        if isinstance(item, dict) and item.get("source") == "default"
    ]
    maps = [item for item in client.maps().get("resources", []) if isinstance(item, dict)]
    aircraft = [item for item in client.aircraft().get("resources", []) if isinstance(item, dict)]
    model = _resolve_named(models, args.model, "model", interactive=interactive, stdin=stdin, ui=ui)
    map_item = _resolve_named(maps, args.map_id, "map", interactive=interactive, stdin=stdin, ui=ui)
    aircraft_item = _resolve_named(
        aircraft, args.vehicle_id, "aircraft", interactive=interactive, stdin=stdin, ui=ui
    )
    pairs = [
        item
        for item in client.asset_pairs().get("pairs", [])
        if isinstance(item, dict)
        and item.get("map_resource_id") == map_item.get("resource_id")
        and item.get("vehicle_resource_id") == aircraft_item.get("resource_id")
    ]
    if len(pairs) != 1:
        raise ConsoleError("CONSOLE_ASSET_PAIR_NOT_FOUND")
    pair = pairs[0]
    if pair.get("status") != "qualified_builtin":
        raise ConsoleError("CONSOLE_ASSET_PAIR_REQUIRES_QUALIFICATION")
    versions = [
        item for item in client.bootstrap().get("asset_versions", []) if isinstance(item, dict)
    ]

    def version(asset_id: object) -> dict:
        matches = [
            item
            for item in versions
            if item.get("asset_id") == asset_id and item.get("maturity") == "qualified"
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("content_sha256"), str):
            raise ConsoleError("CONSOLE_QUALIFIED_ASSET_VERSION_UNAVAILABLE")
        return matches[0]

    return (
        model,
        map_item,
        aircraft_item,
        pair,
        {
            "map": version(pair.get("qualified_map_asset_id")),
            "aircraft": version(pair.get("qualified_vehicle_asset_id")),
        },
    )


def _plan_id(value: dict) -> str:
    direct = value.get("plan_revision_id")
    if isinstance(direct, str):
        return direct
    plan = value.get("mission_plan")
    if isinstance(plan, dict) and isinstance(plan.get("plan_revision_id"), str):
        return str(plan["plan_revision_id"])
    raise ConsoleError("CONSOLE_PLAN_REVISION_MISSING")


def _require_model_generated_plan(value: dict) -> dict:
    """Reject a flight plan that lacks successful model and planning-tool receipts."""
    if (
        type(value.get("model_calls")) is not int
        or value["model_calls"] < 1
        or type(value.get("successful_model_responses")) is not int
        or value["successful_model_responses"] < 1
        or not isinstance(value.get("model_id"), str)
        or not value["model_id"]
        or not isinstance(value.get("model_source"), str)
        or not value["model_source"]
    ):
        raise ConsoleError("CONSOLE_MODEL_PLAN_PROOF_MISSING")
    proof = value.get("model_tool_proof")
    if not isinstance(proof, dict):
        raise ConsoleError("CONSOLE_MODEL_PLAN_PROOF_MISSING")
    requests = proof.get("tool_requests")
    receipts = proof.get("tool_receipts")
    graph = proof.get("task_graph")
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if (
        not isinstance(requests, list)
        or not requests
        or not isinstance(receipts, list)
        or not receipts
        or not isinstance(nodes, list)
        or not nodes
    ):
        raise ConsoleError("CONSOLE_MODEL_PLAN_PROOF_MISSING")
    return value


def _wait_for_terminal(
    client: ProductConsoleClient,
    thread_id: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
    ui: TerminalUI | None = None,
) -> dict:
    if poll_seconds < 0.2 or timeout_seconds < 1:
        raise ConsoleError("CONSOLE_WATCH_INTERVAL_INVALID")
    deadline = time.monotonic() + timeout_seconds
    previous: tuple[object, object] | None = None
    telemetry_samples = 0
    last_live_telemetry: dict | None = None
    live_sources: dict | None = None
    while True:
        thread = client.status(thread_id)
        state = thread.get("state")
        phase: object = "—"
        if state in {"executing", "starting"}:
            try:
                telemetry = client.telemetry(thread_id)
                phase = telemetry.get("phase", telemetry.get("state", "—"))
                last_live_telemetry = telemetry
                telemetry_samples += 1
            except ConsoleError as error:
                if str(error) != "LIVE_TELEMETRY_NOT_READY":
                    raise
            if live_sources is None:
                sources = client.live_sources(thread_id)
                ready_sources = [
                    item
                    for item in sources.get("sources", [])
                    if isinstance(item, dict) and item.get("ready") is True
                ]
                if sources.get("mode") == "simulation" and ready_sources:
                    live_sources = sources
                    if ui is not None:
                        labels = ", ".join(
                            str(item.get("label") or item.get("id")) for item in ready_sources
                        )
                        ui.status(
                            "Live view",
                            "ready",
                            f"{labels}; save a frame with `dronedream mission frame {thread_id}`",
                        )
        current = (state, phase)
        if ui is not None and current != previous:
            ui.status("Mission", state or "unknown", f"flight phase: {phase}")
        if state in {"completed", "failed", "aborted"}:
            return {
                **thread,
                "watch": {
                    "telemetry_samples": telemetry_samples,
                    "last_live_telemetry": last_live_telemetry,
                    "live_sources": live_sources,
                },
            }
        if time.monotonic() >= deadline:
            raise ConsoleError("CONSOLE_WATCH_TIMEOUT")
        previous = current
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _doctor(client: ProductConsoleClient) -> dict:
    checks: list[dict] = []
    for name, call in (
        ("Core session", client.bootstrap),
        ("Managed models", client.models),
        ("Runtime", client.runtime),
        ("Map catalog", client.maps),
        ("Aircraft catalog", client.aircraft),
        ("Qualified pair", client.asset_pairs),
    ):
        try:
            value = call()
            ready = True
            detail = "reachable"
            if name == "Managed models":
                managed = [
                    item
                    for item in value.get("models", [])
                    if isinstance(item, dict) and item.get("source") == "default"
                ]
                ready = bool(managed)
                detail = f"{len(managed)} managed"
            elif name in {"Map catalog", "Aircraft catalog"}:
                count = len([item for item in value.get("resources", []) if isinstance(item, dict)])
                ready, detail = count > 0, f"{count} available"
            elif name == "Qualified pair":
                count = value.get("counts", {}).get("qualified_builtin", 0)
                ready, detail = count > 0, f"{count} executable"
            elif name == "Runtime":
                runtime_available = bool(value.get("runtime_available"))
                resources_ready = bool(value.get("resources_ready"))
                provisioned = bool(value.get("provisioned"))
                ready = runtime_available and resources_ready and provisioned
                detail = str(value.get("issue") or ("ready" if ready else "not ready"))
            checks.append(
                {"name": name, "status": "ready" if ready else "blocked", "detail": detail}
            )
        except ConsoleError as error:
            checks.append({"name": name, "status": "failed", "detail": str(error)})
    return {"checks": checks, "ready": all(item["status"] == "ready" for item in checks)}


def _fly(
    client: ProductConsoleClient,
    args: argparse.Namespace,
    *,
    stdin: TextIO,
    ui: TerminalUI | None,
    show_banner: bool = True,
) -> dict:
    interactive = ui is not None
    model, map_item, aircraft, pair, versions = _qualified_selection(
        client,
        args,
        interactive=interactive,
        stdin=stdin,
        ui=ui or TerminalUI(stream=sys.stderr, color=False),
    )
    if ui:
        if show_banner:
            ui.banner()
            print(file=ui.stream)
        ui.rule("MISSION SETUP")
        print(file=ui.stream)
        ui.status("Model", "ready", str(model.get("id")), blank_after=True)
        ui.status(
            "Map",
            "ready",
            str(map_item.get("display_name", {}).get("en-US", map_item.get("resource_id"))),
            blank_after=True,
        )
        ui.status(
            "Aircraft",
            "ready",
            str(aircraft.get("display_name", {}).get("en-US", aircraft.get("resource_id"))),
            blank_after=True,
        )
        ui.status(
            "Qualification",
            "ready",
            str(pair.get("qualification_id")),
            blank_after=True,
        )
    thread = client.create(
        ThreadCreate(title=args.title, selected_model=str(model["id"]), locale=args.locale)
    )
    thread_id = str(thread["thread_id"])
    message = args.text
    rounds = 0
    while True:
        if ui:
            ui.status(
                "Planner",
                "working",
                f"{model.get('id')} is interpreting the goal and calling mission tools",
                blank_after=True,
            )
        try:
            plan = client.prepare(
                thread_id,
                {
                    "message": message,
                    "map_id": versions["map"]["asset_id"],
                    "map_content_sha256": versions["map"]["content_sha256"],
                    "vehicle_id": versions["aircraft"]["asset_id"],
                    "vehicle_content_sha256": versions["aircraft"]["content_sha256"],
                    "start_entity": args.start,
                },
            )
            _require_model_generated_plan(plan)
            if ui:
                ui.status(
                    "Planner",
                    "ready",
                    f"{plan.get('successful_model_responses', 0)} model responses | "
                    f"{len(plan.get('model_tool_proof', {}).get('tool_receipts', []))} "
                    "tool receipts",
                    blank_after=True,
                )
            break
        except ConsoleClarificationRequired as error:
            if not interactive or rounds >= 3:
                raise
            answers = []
            ui.panel(
                "MODEL NEEDS CLARIFICATION",
                [
                    "The model needs one or more mission details before it can call "
                    "the planning tools."
                ],
                accent=(255, 201, 74),
            )
            for question in error.fields:
                print(ui.rgb("  ? ", (255, 201, 74), bold=True) + question, file=ui.stream)
                ui.prompt("Answer")
                answer = stdin.readline().strip()
                if not answer:
                    raise ConsoleError("CONSOLE_CLARIFICATION_CANCELLED") from error
                answers.append(f"{question}: {answer}")
            message = args.text + "\n\nOperator clarifications:\n" + "\n".join(answers)
            rounds += 1
    plan_revision_id = _plan_id(plan)
    if ui:
        ui.render(plan, "plan", show_banner=False)
    if args.plan_only:
        return {
            "status": "awaiting_confirmation",
            "thread_id": thread_id,
            "plan_revision_id": plan_revision_id,
            "plan": plan,
            "model": model.get("id"),
            "map": map_item.get("resource_id"),
            "aircraft": aircraft.get("resource_id"),
            "qualification_id": pair.get("qualification_id"),
        }
    if not interactive:
        raise ConsoleError("CONSOLE_EXPLICIT_PLAN_CONFIRMATION_REQUIRED")
    ui.confirm_plan(plan_revision_id)
    if stdin.readline().strip().casefold() != "fly":
        raise ConsoleError("CONSOLE_EXPLICIT_PLAN_CONFIRMATION_REQUIRED")
    execution = client.execute(thread_id, plan_revision_id)
    ui.status(
        "Execution accepted",
        execution.get("status", "starting"),
        str(execution.get("execution_id", "")),
    )
    terminal = _wait_for_terminal(
        client,
        thread_id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        ui=ui,
    )
    evidence = client.evidence(thread_id)
    result = evidence.get("result") if isinstance(evidence.get("result"), dict) else {}
    verified = terminal.get("state") == "completed" and result.get("status") == "verified"
    if verified and not args.no_archive:
        client.patch_thread(thread_id, {"archived": True})
    return {
        "status": "verified" if verified else "failed",
        "thread_id": thread_id,
        "plan_revision_id": plan_revision_id,
        "execution_id": execution.get("execution_id"),
        "archived": bool(verified and not args.no_archive),
        "watch": terminal.get("watch"),
        "evidence": evidence,
    }


def _shell_arguments(tokens: list[str]) -> list[str] | None:
    """Expand concise interactive commands into the complete product command tree."""
    if not tokens:
        return None
    command, rest = tokens[0].casefold(), tokens[1:]
    if command == "fly":
        return ["mission", "fly", " ".join(rest)] if rest else None
    if command in {"maps", "aircraft", "pairs"}:
        return ["assets", command, *rest]
    if command == "missions":
        return ["mission", "list", *rest]
    if command == "history" and (not rest or rest[0].startswith("-")):
        return ["history", "list", *rest]
    if command in {"whoami", "profile"}:
        return ["account"]
    if command == "quota":
        return ["usage"]
    if command in {
        "show",
        "watch",
        "telemetry",
        "evidence",
        "frame",
        "archive",
        "unarchive",
        "pin",
        "unpin",
        "export",
    }:
        return ["mission", command, *rest]
    if command in {
        "doctor",
        "catalog",
        "usage",
        "account",
        "models",
        "assets",
        "runtime",
        "mission",
        "history",
        "create",
        "prepare",
        "execute",
        "status",
        "live-sources",
        "message",
    }:
        return tokens
    return None


def _shell_help(ui: TerminalUI) -> None:
    ui.panel(
        "COMMANDS",
        [
            "fly <mission>       Plan a natural-language mission, review it, then type FLY.",
            "maps | aircraft     Browse available maps or aircraft.",
            "models | pairs      Browse managed models or qualified asset pairs.",
            "missions | history  Read active work or saved flights.",
            "show <id>           Open one mission. Add telemetry or evidence for details.",
            "doctor | runtime    Check Core, model, asset, and PX4/Gazebo readiness.",
            "account | usage     Show your account and remaining compute allowance.",
            "clear               Redraw a clean DroneDream console.",
            "logout | exit       Sign out or return to the system terminal.",
        ],
        accent=BLUE,
    )


def _shell_loop(
    *, stdin: TextIO, stdout: TextIO, stderr: TextIO, no_color: bool
) -> int:
    ui = TerminalUI(stream=stdout, color=False if no_color else None)
    if os.environ.get("DRONEDREAM_CONSOLE_BANNER_SHOWN") != "1":
        ui.banner()
    ui.rule("SYSTEM CHECK", width=103)
    print(file=stdout)
    account_name = os.environ.get("DRONEDREAM_CONSOLE_ACCOUNT_NAME", "").strip()
    account_email = os.environ.get("DRONEDREAM_CONSOLE_ACCOUNT_EMAIL", "").strip()
    ui.status("Account", "ready", account_name or account_email or "signed in")
    ui.status("Console", "ready", "interactive command shell")
    ui.status("Commands", "ready", "mission | assets | runtime | history | account")
    display_mode = "gradient color | Unicode" if ui.color else "plain color | Unicode"
    ui.status("Display", "ready", display_mode)
    print(file=stdout)
    while True:
        ui.command_prompt(account_name or None)
        raw = stdin.readline()
        if raw == "":
            print(file=stdout)
            return 0
        try:
            tokens = shlex.split(raw, posix=True)
        except ValueError:
            ui.error("COMMAND_QUOTE_INVALID", "Close the quoted text and try again.")
            continue
        if not tokens:
            continue
        command = tokens[0].casefold()
        if command in {"exit", "quit", "logout"}:
            message = "Account signed out." if command == "logout" else "Flight console closed."
            print(ui.rgb(f"  {message}", MUTED), file=stdout)
            return 0
        if command == "help":
            _shell_help(ui)
            print(file=stdout)
            continue
        if command == "clear":
            if ui.color:
                print("\x1b[2J\x1b[H", end="", file=stdout)
            else:
                print("\n" * 3, file=stdout)
            ui.banner()
            continue
        child_args = _shell_arguments(tokens)
        if child_args is None:
            ui.error("COMMAND_NOT_FOUND", "Type help to see the available commands.")
            print(file=stdout)
            continue
        if no_color:
            child_args.insert(0, "--no-color")
        print(file=stdout)
        try:
            main(
                child_args,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                _shell_child=True,
            )
        except SystemExit:
            ui.error("COMMAND_ARGUMENTS_INVALID", "Type help and check the command values.")
        print(file=stdout)


def _view(args: argparse.Namespace) -> str:
    if args.command == "models":
        return "models"
    if args.command == "assets":
        return args.asset_command
    if args.command == "doctor":
        return "doctor"
    if args.command == "account":
        return "account"
    if args.command == "usage":
        return "usage"
    if args.command == "runtime":
        return "runtime"
    if args.command in {"prepare"}:
        return "plan"
    if args.command in {"evidence", "telemetry"}:
        return args.command
    if args.command in {"status"}:
        return "mission"
    if args.command == "mission":
        return {
            "list": "missions",
            "show": "mission",
            "plan": "plan",
            "watch": "mission",
            "evidence": "evidence",
            "telemetry": "telemetry",
            "fly": "fly",
        }.get(args.mission_command, "generic")
    if args.command == "history":
        return "missions" if args.history_command == "list" else "mission"
    return "generic"


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    _shell_child: bool = False,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if stdout is None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if stderr is None and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    input_stream, output, errors = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    client = None
    interactive = not args.json and hasattr(output, "isatty") and output.isatty()
    ui = TerminalUI(stream=output, color=False if args.no_color else None) if interactive else None
    if args.command is None:
        if not interactive:
            parser.print_help(file=output)
            return 2
        args.command = "shell"
    if args.command == "shell":
        if not interactive:
            print('{"error":"CONSOLE_INTERACTIVE_TERMINAL_REQUIRED"}', file=errors)
            return 2
        return _shell_loop(
            stdin=input_stream,
            stdout=output,
            stderr=errors,
            no_color=args.no_color,
        )
    try:
        session = read_session(args.session_stdin, input_stream)
        client = ProductConsoleClient(session)
        if args.command == "doctor":
            result = _doctor(client)
        elif args.command == "mission" and args.mission_command == "fly":
            result = _fly(
                client,
                args,
                stdin=input_stream,
                ui=ui,
                show_banner=not _shell_child,
            )
        elif args.command == "mission" and args.mission_command == "watch":
            result = _wait_for_terminal(
                client,
                args.thread_id,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
                ui=ui,
            )
        else:
            result = dispatch(client, args)
        rendered = client.render(result)
        if ui:
            redacted = json.loads(rendered)
            if not (args.command == "mission" and args.mission_command in {"fly", "watch"}):
                ui.render(redacted, _view(args), show_banner=not _shell_child)
            elif args.mission_command == "fly":
                ui.render(redacted, "fly", show_banner=False)
        else:
            print(rendered, file=output)
        if args.command == "doctor" and not result.get("ready"):
            return 2
        if (
            args.command == "mission"
            and args.mission_command == "fly"
            and result.get("status") != "verified"
            and not args.plan_only
        ):
            return 2
        return 0
    except ConsoleClarificationRequired as error:
        if ui:
            ui.error(str(error), "The mission needs more information before planning can continue.")
        else:
            print(
                json.dumps(
                    {"error": str(error), "fields": error.fields, "automatic_retry": False},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                file=errors,
            )
        return 3
    except (ValidationError, ValueError, TypeError):
        if ui:
            ui.error("CONSOLE_INPUT_INVALID", "Check the command values and try again.")
        else:
            print('{"error":"CONSOLE_INPUT_INVALID"}', file=errors)
        return 2
    except ConsoleError as error:
        if ui:
            ui.error(str(error))
        else:
            print(
                json.dumps(
                    {"error": str(error), "automatic_retry": False}, separators=(",", ":")
                ),
                file=errors,
            )
        return 1
    except KeyboardInterrupt:
        if ui:
            ui.error(
                "CONSOLE_INTERRUPTED_CHECK_TASK_STATUS",
                "The terminal stopped waiting. The mission may still be active; "
                "check it with `dronedream mission show`.",
            )
        else:
            print('{"error":"CONSOLE_INTERRUPTED_CHECK_TASK_STATUS"}', file=errors)
        return 130
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
