#!/usr/bin/env python3
"""Write the runtime's canonical local-policy sensor binding as JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from dronedream_agent_core.runtime_sensor_contracts import (
    oakd_lite_depth_sensor_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        oakd_lite_depth_sensor_contract().model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
