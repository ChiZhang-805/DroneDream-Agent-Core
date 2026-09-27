"""Profile recorded input preparation without simulation, labels or control permission."""

import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path

from dronedream_agent_core.training.observations import compile_training_observation


# 功能：只读重放一个真实输入以定位CPU热点，不改源时间，不创建训练标签或飞行权限。
# 输入：已结束回合的快照JSONL、重复次数；输出：有界性能统计到标准输出。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        raise ValueError("PROFILE_REPEATS_INVALID")
    with args.snapshots.open(encoding="utf-8") as stream:
        line = stream.readline(4 * 1024 * 1024 + 1)
    if len(line) > 4 * 1024 * 1024:
        raise ValueError("PROFILE_INPUT_TOO_LARGE")
    snapshot = json.loads(line)["snapshot"]
    clock = snapshot["control_reference_observed_at_unix_ms"]
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(args.repeats):
        compile_training_observation(snapshot, now_unix_ms=clock)
    profile.disable()
    output = io.StringIO()
    pstats.Stats(profile, stream=output).strip_dirs().sort_stats("cumulative").print_stats(24)
    print(output.getvalue())


if __name__ == "__main__":
    main()
