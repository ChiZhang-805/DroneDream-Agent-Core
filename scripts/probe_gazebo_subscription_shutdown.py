#!/usr/bin/env python3
"""Native pub/sub shutdown stress probe; no world, aircraft or flight authority."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from dronedream_agent_core.gazebo_subscriptions import GazeboSubscriptions


def child():
    from gz.msgs10.stringmsg_pb2 import StringMsg
    from gz.transport13 import Node

    subscriber, publisher = Node(), Node()
    group = GazeboSubscriptions(subscriber)
    topic = "/dronedream/test/subscription_shutdown"
    received, stop = threading.Event(), threading.Event()
    count = [0]

    def callback(_):
        count[0] += 1
        received.set()
        time.sleep(.001)

    group.subscribe(StringMsg, topic, callback)
    pub = publisher.advertise(topic, StringMsg)

    def publish():
        while not stop.is_set():
            pub.publish(StringMsg(data="test-only"))
            stop.wait(.002)

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        if not received.wait(4):
            raise RuntimeError("NATIVE_SUBSCRIPTION_DID_NOT_RECEIVE")
        time.sleep(.05)
        result = group.close()
        count_at_close = count[0]
        time.sleep(.05)  # Publisher is intentionally still active after unsubscribe.
        result.update(received_count=count[0], unchanged_after_close=count[0] == count_at_close)
        print(json.dumps(result), flush=True)
        return 0 if result["complete"] and result["unchanged_after_close"] else 2
    finally:
        group.close()
        stop.set()
        thread.join(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        return child()
    if not 1 <= args.attempts <= 20 or args.output is None:
        parser.error("provide output and 1..20 attempts")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(args.attempts):
        env = {**os.environ, "GZ_PARTITION": "dronedream-shutdown-probe-" + uuid.uuid4().hex}
        try:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"],
                env=env, capture_output=True, text=True, timeout=15, check=False)
            result = {"attempt": index, "returncode": run.returncode,
                      "stdout": run.stdout[-8000:], "stderr": run.stderr[-8000:]}
        except subprocess.TimeoutExpired:
            result = {"attempt": index, "returncode": None, "error": "child-timeout"}
        results.append(result)
    receipt = {"purpose": "native-transport-shutdown-only", "results": results,
               "passed": all(row["returncode"] == 0 for row in results),
               "qualified_for_flight": False}
    with (args.output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"passed": receipt["passed"], "attempts": args.attempts}), flush=True)
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
