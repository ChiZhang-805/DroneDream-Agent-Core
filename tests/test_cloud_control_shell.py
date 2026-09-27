"""Exercise Linux orchestration with an explicit fake executor, never as CUDA evidence."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != 'posix' or shutil.which('bash') is None,
                              reason='Linux shell orchestration required')


# 功能：
#   建立隔离解释器替身，仅检查真实 Bash 脚本的参数、路径保护和失败传播，不模拟模型成绩。
# 输入：
#   tmp_path：本用例独占目录。
# 输出：
#   roots、environment：带空格的输入路径及指定替身行为的环境。
def shell_fixture(tmp_path):
    roots = {name: tmp_path / (name + ' with spaces') for name in ('bundle', 'data', 'environment', 'encoder')}
    for path in roots.values():
        path.mkdir()
    interpreter = roots['environment'] / 'venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['SHELL_TEST_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
with open(os.environ['SHELL_TEST_CALLS'] + '.devices', 'a') as log:
    log.write(os.environ.get('CUDA_VISIBLE_DEVICES', 'unset') + '\\n')
if '-c' in args:
    statement = args[args.index('-c') + 1]
    if 'causal_training_device' in statement:
        print('FAKE_EXECUTOR_NO_GPU_EVIDENCE')
        sys.exit(0)
    sys.argv = ['-c', *args[args.index('-c') + 2:]]
    exec(statement)
    sys.exit(0)
entry = next(pathlib.Path(value).name for value in args if value.endswith('.py'))
if entry == os.environ.get('SHELL_TEST_FAIL'):
    print('injected shell test failure', file=sys.stderr)
    sys.exit(7)
if entry == 'train_causal_control_role.py':
    output = pathlib.Path(args[args.index('--output') + 1])
    output.mkdir()
    (output / 'fake-training.txt').write_text('not model weights')
if entry == 'validate_causal_control_role.py':
    pathlib.Path(args[args.index('--output') + 1]).write_text('fake orchestration output only')
''', encoding='utf-8')
    interpreter.chmod(0o755)
    environment = {**os.environ, 'SHELL_TEST_CALLS': str(tmp_path / 'calls.jsonl'),
                   'PYTHONPATH': '/must/be/removed', 'PYTHONHOME': '/must/be/removed'}
    return roots, environment


# 功能：
#   调用仓库真实云端 Bash 入口，以有界时间检查编排，不执行真实训练或访问云端。
# 输入：
#   roots、environment：隔离路径和解释器替身环境。
#   output、gpu_index：请求的输出路径和可选单卡编号。
# 输出：
#   result：真实 shell 的退出码与标准输出、标准错误。
def run_shell(roots, environment, output, gpu_index=None):
    script = Path(__file__).parents[1] / 'training/cloud/run_control.sh'
    args = ['bash', str(script), str(roots['bundle']), str(roots['data']), str(output),
            str(roots['environment']), 'a' * 64, 'b' * 64, 'recovery-policy', 'smoke', str(roots['encoder'])]
    if gpu_index is not None:
        args.append(gpu_index)
    return subprocess.run(args, env=environment, capture_output=True, text=True, timeout=20)


# 功能：
#   检查含空格路径与编码器参数传递正确，且训练只收到训练/验证输入和显式 CUDA 请求。
# 输入：
#   tmp_path：隔离替身目录。
# 输出：
#   None：真实 shell 编排成功，但生成的是明确标识的非模型测试文件。
def test_cloud_shell_preserves_paths_and_training_contract(tmp_path):
    roots, environment = shell_fixture(tmp_path)
    output = tmp_path / 'new run with spaces'
    result = run_shell(roots, environment, output)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(environment['SHELL_TEST_CALLS']).read_text().splitlines()]
    train = next(call for call in calls if any(value.endswith('/train_causal_control_role.py') for value in call))
    assert train[train.index('--device') + 1] == 'cuda'
    assert train[train.index('--encoder-policy') + 1] == str(roots['encoder'] / 'local-navigation-policy.pt')
    assert train[train.index('--train') + 1] == str(roots['data'] / 'training-replay.jsonl')
    assert all('test-replay' not in value for value in train)
    assert (output / 'candidate/fake-training.txt').is_file()
    assert (output / 'validation.json').is_file()
    assert set(Path(environment['SHELL_TEST_CALLS'] + '.devices').read_text().splitlines()) == {'0'}


# 功能：
#   明确选择第二张 GPU 时，校验、训练与导出子进程始终继承同一张卡的可见性。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：原有外部多卡环境被单卡选择替代，不隐式改回 GPU 0。
def test_cloud_shell_selects_single_explicit_gpu(tmp_path):
    roots, environment = shell_fixture(tmp_path)
    environment['CUDA_VISIBLE_DEVICES'] = '0,1'
    result = run_shell(roots, environment, tmp_path / 'gpu1 run', '1')
    assert result.returncode == 0, result.stderr
    assert set(Path(environment['SHELL_TEST_CALLS'] + '.devices').read_text().splitlines()) == {'1'}


# 功能：
#   拒绝多卡列表及歧义编号，不能在非法设备配置下读取数据或创建作业。
# 输入：
#   tmp_path、gpu_index：隔离目录及非法编号。
# 输出：
#   None：入口返回错误且没有执行任何 Python 子进程。
@pytest.mark.parametrize('gpu_index', ['0,1', '-1', '01', '1;echo unsafe', 'all'])
def test_cloud_shell_rejects_invalid_gpu_selection(tmp_path, gpu_index):
    roots, environment = shell_fixture(tmp_path)
    result = run_shell(roots, environment, tmp_path / 'invalid run', gpu_index)
    assert result.returncode == 2
    assert not Path(environment['SHELL_TEST_CALLS']).exists()
    assert not (tmp_path / 'invalid run').exists()


# 功能：
#   上游校验或训练失败不能继续导出验证，错误码保留且输入不会被改动。
# 输入：
#   tmp_path：隔离目录。
#   failing：注入失败的入口名。
# 输出：
#   None：校验失败无作业目录；训练失败仅保留诊断日志，没有成功验证文件。
@pytest.mark.parametrize('failing', ['verify_training_bundle.py', 'verify_causal_control_data.py',
                                    'train_causal_control_role.py'])
def test_cloud_shell_stops_on_failures(tmp_path, failing):
    roots, environment = shell_fixture(tmp_path)
    environment['SHELL_TEST_FAIL'] = failing
    output = tmp_path / 'failed run'
    result = run_shell(roots, environment, output)
    assert result.returncode == 7
    assert not (output / 'validation.json').exists()
    if failing != 'train_causal_control_role.py':
        assert not output.exists()
    else:
        assert 'injected shell test failure' in (output / 'training.log').read_text()


# 功能：
#   阻止输出覆盖输入或已有作业，路径校验不能被空格和父子目录关系绕过。
# 输入：
#   tmp_path：隔离目录。
#   target：受保护输入根或既有作业。
# 输出：
#   None：拒绝执行且原有内容保持不变。
@pytest.mark.parametrize('target', ['bundle', 'data', 'environment', 'encoder', 'existing'])
def test_cloud_shell_rejects_protected_or_existing_output(tmp_path, target):
    roots, environment = shell_fixture(tmp_path)
    output = roots[target] / 'new job' if target in roots else tmp_path / 'existing'
    if target == 'existing':
        output.mkdir()
        (output / 'keep.txt').write_text('preserve')
    result = run_shell(roots, environment, output)
    assert result.returncode != 0
    assert not (output / 'candidate').exists()
    if target == 'existing':
        assert (output / 'keep.txt').read_text() == 'preserve'
    else:
        assert not output.exists()
