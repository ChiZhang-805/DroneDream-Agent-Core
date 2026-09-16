"""Test process-global GC ownership in a separate interpreter, not pytest's."""

import os
import subprocess
import sys
from pathlib import Path


# 功能：
#   在独立子解释器验证启动图保留、运行期循环回收、嵌套与异常退出的所有权恢复。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_worker_baseline_preserves_runtime_collection_and_exception_cleanup():
    program = """
import gc
import weakref
from dronedream_agent_core.runtime_scheduling import retained_interpreter_baseline
# Isolate both ownership cases inside this disposable child process.
gc.unfreeze()
class Cycle:
    # 功能：
    #   构造仅存在于隔离子进程内的可回收循环引用。
    # 输入：
    #   self：测试对象。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self):
        self.me = self
thresholds = gc.get_threshold()
assert gc.isenabled() and gc.get_freeze_count() == 0
try:
    with retained_interpreter_baseline():
        count = gc.get_freeze_count()
        assert count > 0 and gc.isenabled()
        with retained_interpreter_baseline():
            assert gc.get_freeze_count() == count
        assert gc.get_freeze_count() == count
        runtime = Cycle()
        reference = weakref.ref(runtime)
        del runtime
        gc.collect()
        assert reference() is None
        raise RuntimeError('startup failure')
except RuntimeError:
    pass
assert gc.get_freeze_count() == 0 and gc.isenabled()
assert gc.get_threshold() == thresholds
# Some product interpreters already retain the site/bootstrap graph. Extend
# it at process entry without ever unfreezing that other owner's objects.
gc.freeze()
existing = gc.get_freeze_count()
startup = Cycle()
with retained_interpreter_baseline():
    assert gc.get_freeze_count() >= existing
assert gc.get_freeze_count() >= existing
gc.unfreeze()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(Path(__file__).resolve().parents[1] / "src")
                         + os.pathsep + env.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
