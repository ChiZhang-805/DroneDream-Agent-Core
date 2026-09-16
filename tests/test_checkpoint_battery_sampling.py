from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


# 功能：
#   按明确源码路径加载当前测试模块并注册模块身份。
# 输入：
#   name：隔离测试模块名。
#   path：当前工作树中的源码文件。
# 输出：
#   module：已初始化的模块对象。
def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 功能：
#   加载当前基础飞控和检查点执行器，避免使用安装包中的旧实现。
# 输入：
#   无。
# 输出：
#   modules：基础模块和检查点执行器的二元组。
def _modules() -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).parents[1]
    base = _load_module(
        "test_checkpoint_battery_base",
        root / "runtime" / "px4_offboard_track_executor.py",
    )
    executor = _load_module(
        "test_checkpoint_battery_executor",
        root / "scripts" / "px4_checkpoint_executor.py",
    )
    modules = base, executor
    return modules


# 功能：
#   验证首次电池读取超时后重新采样，同时持续发送悬停目标，不以默认电量填补缺失。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_battery_retries_real_samples_while_holding_position() -> None:
    base, executor = _modules()

    class IntermittentBatteryClient(base.FakeOffboardClient):
        # 功能：
        #   初始化隔离飞控并记录实际电池采样尝试次数。
        # 输入：
        #   self：本次测试客户端。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self) -> None:
            super().__init__()
            self.battery_attempts = 0

        # 功能：
        #   首次等待预算耗尽后失败，第二次返回明确样本，模拟真实重订阅读取。
        # 输入：
        #   self：持有尝试次数的电池替身。
        #   timeout_seconds：本次采样允许等待的秒数。
        # 输出：
        #   battery：第二次实际返回的电量和电压字典。
        async def sample_battery(self, timeout_seconds: float) -> dict[str, float]:
            self.battery_attempts += 1
            if self.battery_attempts == 1:
                await asyncio.sleep(timeout_seconds)
                raise TimeoutError("first telemetry subscription produced no sample")
            await asyncio.sleep(0.03)
            return {"remaining_percent": 0.72, "voltage_v": 15.7}

    client = IntermittentBatteryClient()
    setpoint = base.Setpoint(north_m=2.0, east_m=-1.0, down_m=-3.0, yaw_deg=45.0)

    sample = asyncio.run(
        executor._sample_checkpoint_battery(
            base=base,
            client=client,
            setpoint=setpoint,
            rate_hz=50.0,
            timeout_seconds=0.3,
            sample_timeout_seconds=0.05,
        )
    )

    assert sample == {"remaining_percent": 0.72, "voltage_v": 15.7}
    assert client.battery_attempts == 2
    assert len(client.setpoints) >= 3
    assert all(item == setpoint for item in client.setpoints)
