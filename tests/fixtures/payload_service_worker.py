"""Protocol test fixture only; never imports Gazebo or controls an aircraft."""

import json
import sys
import time

print('{"ready":true}', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    mode = request['world']
    if mode == 'delay':
        time.sleep(.15)
    if mode == 'timeout':
        time.sleep(60)
    if mode == 'exit':
        raise SystemExit(1)
    if mode == 'oversized':
        print('x' * 20000, flush=True)
        continue
    print(json.dumps(dict(sequence=request['sequence'] + (mode == 'wrong'),
                          accepted=mode != 'reject')), flush=True)
