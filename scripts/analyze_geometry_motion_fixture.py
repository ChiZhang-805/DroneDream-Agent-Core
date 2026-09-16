"""Analyze source-stamped moving camera pixels; do not publish control state."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.geometry_motion_audit import compare_moving_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("semantic", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-incomplete-capture", action="store_true",
                        help="Analyze preserved partial evidence without claiming capture passed")
    parser.add_argument("--joint-pose", action="store_true",
                        help="Evaluate joint position/attitude instead of fixed-attitude baseline")
    args = parser.parse_args()
    result = compare_moving_fixture(args.capture, args.semantic,
                                    allow_incomplete_capture=args.allow_incomplete_capture,
                                    joint_pose=args.joint_pose)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}))


if __name__ == "__main__":
    main()
