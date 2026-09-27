"""Protective commands retain actual transport evidence without model attribution."""

import asyncio
from types import SimpleNamespace

import pytest
from test_runtime_commands import _load_executor

from dronedream_agent_core.executor_brake_evidence import ExecutorBrakeApplication


# 功能：执行器新增制动统计必须能被Gazebo汇总契约读取，不能导致落地后的汇总失败。
# 输入：新/旧报告；输出：独立制动计数保留，旧报告默认0且不获模型授权。
def test_brake_count_survives_gazebo_authority_contract():
    from dronedream_agent_core.contracts import Px4GazeboModelControlAuthorityEvidence

    parsed = Px4GazeboModelControlAuthorityEvidence.model_validate_json(
        '{"executor_brake_applied_count":113,"authorized_control_applied_count":3}')
    assert parsed.executor_brake_applied_count == 113
    assert parsed.authorized_control_applied_count == 3
    assert parsed.authorized_schedule_advance_count == 0
    assert Px4GazeboModelControlAuthorityEvidence().executor_brake_applied_count == 0
    with pytest.raises(ValueError):
        Px4GazeboModelControlAuthorityEvidence(executor_brake_applied_count=-1)


# 功能：验证每次真实发送均有独立记录，不能被诊断500毫秒节流合并或获得模型权限。
# 输入：合成传输端与写入器；输出：实际时间、来源计数、零速度和独立序号完全一致。
def test_brake_records_every_actual_acceptance(tmp_path):
    executor = _load_executor()
    rows, sends = [], []
    args = SimpleNamespace(run_dir=tmp_path, _model_control_application_counts={"model": 3},
        _control_application_writer=SimpleNamespace(submit=lambda p,r: rows.append((p,r))))

    async def send(**kwargs):
        sends.append(kwargs)
        return 1000+20*len(sends)

    executor._send_position_with_velocity = send
    point = SimpleNamespace(north_m=1., east_m=2., down_m=-3., yaw_deg=45.)
    for reason in ("command-stale", "command-stale", "input-lease-unavailable"):
        asyncio.run(executor._send_executor_brake(args=args, base=None, client=None,
                                                  setpoint=point, reason=reason))
    assert [r[1]["sequence"] for r in rows] == [1,2,3]
    assert [r[1]["accepted_at_unix_ms"] for r in rows] == [1020,1040,1060]
    assert all(r[1]["after_command_application_sequence"] == 3 for r in rows)
    assert all(not r[1]["model_authorized"] for r in rows)
    assert all(r[0].name == "executor-brake-applications.jsonl" for r in rows)
    assert all(s["velocity_ned_mps"] == (0.,0.,0.) for s in sends)


# 功能：传输异常不得生成“已执行”回执；输入：失败发送器；输出：原异常和零记录。
def test_failed_brake_send_is_not_recorded(tmp_path):
    executor = _load_executor()

    async def fail(**kwargs):
        raise RuntimeError("transport unavailable")

    executor._send_position_with_velocity = fail
    args = SimpleNamespace()
    with pytest.raises(RuntimeError, match="transport unavailable"):
        asyncio.run(executor._send_executor_brake(args=args, base=None, client=None,
            setpoint=None, reason="command-stale"))
    assert not hasattr(args, "_executor_brake_application_count")


# 功能：独立制动不得容纳非零速度、伪模型权限或非数值位置。
# 输入：回执字段突变；输出：契约拒绝。
@pytest.mark.parametrize("mutation", [dict(velocity_ned_mps=(.1,0.,0.)),
    dict(model_authorized=True), dict(position_ned_m=(True,0.,0.)),
    dict(position_ned_m=(float("nan"),0.,0.))])
def test_brake_receipt_rejects_false_motion_claim(mutation):
    row = dict(sequence=1, after_command_application_sequence=0, accepted_at_unix_ms=1000,
               position_ned_m=(0.,0.,0.), yaw_heading_deg=0., reason="command-stale")
    with pytest.raises(ValueError):
        ExecutorBrakeApplication(**{**row, **mutation})
