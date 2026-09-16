import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_executor_finalization import resources
from test_runtime_commands import _load_executor

from dronedream_agent_core.executor_finalization import finalize_executor_resources


# 功能：
#   用阻塞的真实后台线程验证控制发送不会等诊断写盘，且慢写期间不会堆积任务。
# 输入：
#   tmp_path：测试私有证据目录。
#   monkeypatch：替换诊断文件发布函数。
# 输出：
#   None：不返回业务数据。
def test_slow_heading_write_never_blocks_next_control_or_queues_more_writes(tmp_path, monkeypatch):
    executor = _load_executor()
    entered, release = threading.Event(), threading.Event()
    writes = []

    # 功能：
    #   阻塞首次诊断写入，记录后台收到的独立报告。
    # 输入：
    #   path：诊断输出路径。
    #   payload：已冻结的标量统计。
    # 输出：
    #   None：不返回业务数据。
    def blocked_write(path, payload):
        writes.append(payload)
        entered.set()
        assert release.wait(2.)

    monkeypatch.setattr(executor, "_atomic_json", blocked_write)

    # 功能：
    #   在写线程被阻塞时发送多个指令，最终放开写入并等待任务收尾。
    # 输入：
    #   无：使用外层线程门和当前执行器。
    # 输出：
    #   None：不返回业务数据。
    async def scenario():
        attitude = {"yaw_deg": 2., "sample_age_seconds": .01, "timestamp_us": 1}
        raw = SimpleNamespace(set_position_ned=AsyncMock(),
                              latest_dynamics_telemetry=lambda age: {
                                  "sources": {"attitude": attitude}})
        client = executor.SpawnRelativeOffboardClient(
            raw, SimpleNamespace(north_m=0., east_m=0., down_m=0.),
            heading_evidence_path=tmp_path / "heading.json",
            heading_evidence_flush_interval_seconds=0.)
        point = SimpleNamespace(north_m=0., east_m=0., down_m=0., yaw_deg=1.)
        try:
            await asyncio.wait_for(client.set_position_ned(point), .2)
            assert await asyncio.to_thread(entered.wait, 1.)
            task = client._pending_heading_publication
            for sequence in range(2, 8):
                attitude["timestamp_us"] = sequence
                await asyncio.wait_for(client.set_position_ned(point), .2)
                assert client._pending_heading_publication is task
            assert raw.set_position_ned.await_count == 7
            assert len(writes) == 1
            assert writes[0]["sample_count"] == 1
        finally:
            release.set()
            result = await client.flush_heading_tracking_evidence()
        assert result["sample_count"] == 7
        assert result["writer_issue"] is None
        assert writes[-1]["sample_count"] == 7
        assert client._pending_heading_publication is None

    asyncio.run(scenario())


# 功能：
#   诊断写入未完成时，即使飞行已经落地，也不能把资源收尾报告为完整成功。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_failed_heading_publication_remains_a_failed_finalization():
    kwargs, calls, receipts = resources(completed=True)
    kwargs["client"].flush_heading_tracking_evidence = AsyncMock(
        return_value={"writer_issue": "TimeoutError"})
    with pytest.raises(RuntimeError, match="heading_tracking"):
        asyncio.run(finalize_executor_resources(**kwargs))
    assert kwargs["timing"]["status"] == "failed"
    assert kwargs["timing"]["flight_execution_status"] == "complete"
    assert "client_close" in calls
    assert receipts[-1]["resource_finalization"]["complete"] is False
