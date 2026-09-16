from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from urllib.request import urlopen

import websockets


async def _request(
    socket: object,
    request_id: int,
    method: str,
    params: dict | None = None,
) -> dict:
    await socket.send(
        json.dumps(
            {"id": request_id, "method": method, "params": params or {}},
            separators=(",", ":"),
        )
    )
    while True:
        response = json.loads(await socket.recv())
        if response.get("id") == request_id:
            return response


async def inspect(args: argparse.Namespace) -> None:
    with urlopen(f"http://127.0.0.1:{args.port}/json", timeout=5) as response:
        targets = json.load(response)
    pages = [target for target in targets if target.get("type") == "page"]
    if not pages:
        raise SystemExit("no WebView page target found")
    target = pages[0]
    async with websockets.connect(
        target["webSocketDebuggerUrl"],
        max_size=32 * 1024 * 1024,
    ) as socket:
        await _request(socket, 1, "Page.enable")
        await _request(socket, 2, "Runtime.enable")
        result = await _request(
            socket,
            3,
            "Runtime.evaluate",
            {
                "expression": args.expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        payload = {
            "target": {key: target.get(key) for key in ("id", "title", "url")},
            "evaluation": result.get("result", {}).get("result", {}),
            "exceptionDetails": result.get("result", {}).get("exceptionDetails"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.output:
            screenshot = await _request(
                socket,
                4,
                "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": False},
            )
            encoded = screenshot.get("result", {}).get("data")
            if not encoded:
                raise SystemExit("CDP did not return screenshot data")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(base64.b64decode(encoded))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9223)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expression",
        default=(
            "({readyState:document.readyState,title:document.title,"
            "bodyText:document.body?.innerText?.slice(0,2000),"
            "bodyHtml:document.body?.innerHTML?.slice(0,2000)})"
        ),
    )
    asyncio.run(inspect(parser.parse_args()))


if __name__ == "__main__":
    main()
