"""Inspect measured native rays against the bound map; never write control state."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.geometry_alignment_audit import analyze_geometry_capture
from dronedream_agent_core.localization_truth_audit import compare_geometry_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("semantic", type=Path)
    parser.add_argument("output", type=Path)
    solver = parser.add_mutually_exclusive_group()
    solver.add_argument("--fit-translation", action="store_true",
                        help="Compute conditional translation candidates; do not apply to control")
    solver.add_argument("--fit-joint-pose", action="store_true",
                        help="Joint position/attitude candidates; no native state writeback")
    parser.add_argument("--check-perturbations", action="store_true",
                        help="Test six same-frame 4 cm shifts; not independent flight accuracy")
    parser.add_argument("--truth-capture", type=Path,
                        help="Independent timestamped witness; used only AFTER all fitting")
    parser.add_argument("--sensor-frames", type=Path,
                        help="SDF-resolved collision-centre/camera frame receipt for that witness")
    args = parser.parse_args()
    if bool(args.truth_capture) != bool(args.sensor_frames) or (
            args.truth_capture and not (args.fit_translation or args.fit_joint_pose)):
        parser.error("truth comparison requires --truth-capture, --sensor-frames "
                     "and one explicit fitting mode")
    report = analyze_geometry_capture(args.capture, args.semantic,
                                      fit_translation=args.fit_translation,
                                      check_perturbations=args.check_perturbations,
                                      fit_joint_pose=args.fit_joint_pose)
    if args.truth_capture:
        report["independent_truth_comparison"] = compare_geometry_truth(
            args.capture, report, args.truth_capture, args.sensor_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary = {key: value for key, value in report.items()
               if key not in ("frames", "independent_truth_comparison")}
    if args.truth_capture:
        summary["independent_truth_comparison"] = {
            k: v for k, v in report["independent_truth_comparison"].items() if k != "frames"}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
