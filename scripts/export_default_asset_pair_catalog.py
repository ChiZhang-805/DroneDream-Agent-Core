"""Export and validate the complete built-in map/aircraft pairing catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronedream_agent_app.asset_pair_catalog import asset_pair_catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    catalog = asset_pair_catalog()
    if (
        catalog["map_count"] != 8
        or catalog["vehicle_count"] != 13
        or catalog["pair_count"] != 104
        or sum(catalog["counts"].values()) != 104
    ):
        raise RuntimeError("ASSET_PAIR_CATALOG_INCOMPLETE")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(catalog["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
