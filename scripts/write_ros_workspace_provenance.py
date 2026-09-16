from __future__ import annotations

import argparse
from pathlib import Path

from dronedream_agent_core.ros_workspace_provenance import (
    write_ros_workspace_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print(write_ros_workspace_provenance(args.repository, args.workspace))


if __name__ == "__main__":
    main()
