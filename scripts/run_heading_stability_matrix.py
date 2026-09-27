"""Two sequential CUDA queues with a predeclared candidate selection budget."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


# 功能：
#   每张卡顺序执行固定实验，所有输出独占保存且训练有硬超时。
# 输入：
#   directory：冻结实验目录；gpu：卡编号；jobs：方案、种子与输出名列表。
# 输出：
#   results：每个实验的退出码。
def queue(directory, gpu, jobs):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(directory/'current-site')}
    results = {}
    for variant, seed, name in jobs:
        with (directory/(name+'.log')).open('x') as stream:
            process = subprocess.run(['timeout', '700', sys.executable, '-B', str(directory/'train_heading_stability.py'),
                '--directory', str(directory), '--variant', variant, '--seed', str(seed), '--output', str(directory/name)],
                env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=720)
        results[name] = process.returncode
    return results


# 功能：
#   对照完成后依据固定误差和噪声门槛选择方案，仅复测获选方案的两个预设种子。
# 输入：
#   directory：实验目录及预先冻结的四组方案。
# 输出：
#   report：全部候选状态、选择依据和复测状态；没有合格候选时停止 GPU 工作。
def main():
    directory = Path(__file__).resolve().parent
    plan = json.loads((directory/'plan.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, directory, gpu, [(v, plan['seed'], v) for v in plan['variants'][gpu::2]]) for gpu in (0, 1)]
        statuses = {k: v for future in futures for k, v in future.result().items()}
    scores, eligible = {}, []
    for name in plan['variants']:
        if statuses[name]:
            scores[name] = dict(eligible=False, exit_code=statuses[name])
            continue
        metrics = json.loads((directory/name/'metrics.json').read_text())
        score, noise = metrics['splits']['validation']['compatible'], metrics['probe']['current_only']
        passed = (score['mae'] <= .021067911759018898*1.05 and score['p95'] <= .12685050070285797*1.05
            and noise['change_p95'] < .015949368476867676 and noise['change_max'] < .028360426425933838)
        scores[name] = dict(eligible=passed, mae=score['mae'], p95=score['p95'], worst=score['worst_group_mae'], noise=noise)
        if passed:
            eligible.append((noise['change_p95'], score['worst_group_mae'], name))
    chosen = min(eligible)[-1] if eligible else None
    selection = dict(selected=chosen, scores=scores, statuses=statuses, flight_authority=False)
    with (directory/'selection.json').open('x') as stream:
        json.dump(selection, stream, indent=2)
    print(json.dumps(selection), flush=True)
    repeats = {}
    if chosen:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(queue, directory, gpu, [(chosen, seed, f'repeat-{seed}')]) for gpu, seed in enumerate(plan['repeat_seeds'])]
            repeats = {k: v for future in futures for k, v in future.result().items()}
    with (directory/'completion.json').open('x') as stream:
        json.dump(dict(selection=selection, repeat_statuses=repeats, gpu_work_complete=True), stream, indent=2)
    print(json.dumps(repeats), flush=True)


if __name__ == '__main__':
    main()
