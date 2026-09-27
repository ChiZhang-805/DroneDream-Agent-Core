"""Long real planning runs must not overflow the public receipt envelope."""
import json
import hashlib
from datetime import datetime, UTC
from types import SimpleNamespace

import pytest

from dronedream_agent_app.mission_service import _output_receipt_references, _plan_notification_summary
from dronedream_agent_core.contracts import EvidenceRecord
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   构造可校验的连续证据链，不需要飞行、模型或网络请求。
# 输入：
#   count：记录数量。
# 输出：
#   records：有真实内容散列与前驱绑定的记录。
def records(count):
    result=[]
    previous='0'*64
    for index in range(count):
        payload={'index':index}
        record=EvidenceRecord(sequence=index+1, created_at=datetime.now(UTC),event_type='planning.test',
            artifact_sha256=sha256_json(payload),previous_record_sha256=previous,record_sha256='0'*64,payload=payload)
        previous=sha256_json(record.model_dump(mode='json',exclude={'record_sha256'}))
        result.append(record.model_copy(update={'record_sha256':previous}))
    return result


# 功能：
#   覆盖短链、边界和真实失败规模，确保摘要引用有界而完整索引没有被截断。
# 输入：
#   tmp_path：索引输出目录；count：证据及工具回执数量。
# 输出：
#   None：断言所有记录保留、索引散列正确且返回引用不溢出。
@pytest.mark.parametrize('count',[0,1,128,129,350])
def test_complete_index_with_bounded_references(tmp_path,count):
    evidence=records(count)
    tools=[SimpleNamespace(call_id=f'call-{i}',model_dump=lambda mode,i=i:{'call_id':f'call-{i}'}) for i in range(count)]
    evidence_ids,tool_ids,index=_output_receipt_references(tmp_path,evidence,tools)
    raw=(tmp_path/index['path']).read_bytes()
    saved=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==index['sha256']
    assert len(saved['evidence_record_ids'])==count
    assert len(saved['tool_receipts'])==count
    assert len(evidence_ids)<=1 and len(tool_ids)<=128
    assert evidence_ids==((evidence[-1].record_sha256,) if count else ())


# 功能：
#   确认证据中间记录被改动时拒绝生成可信摘要，不把损坏记录包装成合法索引。
# 输入：
#   tmp_path：索引输出目录。
# 输出：
#   None：断言校验失败且未发布索引。
def test_corrupt_chain_cannot_be_aggregated(tmp_path):
    evidence=records(3)
    evidence[1]=evidence[1].model_copy(update={'payload':{'index':99}})
    with pytest.raises(ValueError):
        _output_receipt_references(tmp_path,evidence,[])
    assert not (tmp_path/'harness-receipt-index.json').exists()


# 功能：
#   确认大量任务证据不会进入通知载荷，同时三个正式通知渲染器仍得到所需字段。
# 输入：
#   无。
# 输出：
#   None：断言完整结果保留，通知载荷小且三个渲染器均成功。
def test_notification_projection_preserves_full_plan():
    from dronedream_agent_plugins.notification_plugins import _plan_ready, _planning_metrics, _operator_checklist
    from dronedream_agent_core.extensions import ExtensionPlugin, ExtensionRegistry
    summary = dict(goal='取餐返回', contract_id='contract', plan_revision_id='revision',
                   plugin_snapshot_id='snapshot', locale='zh-CN', plugin_catalog_sha256='a'*64,
                   minimum_clearance_m=0.8, model_calls=6, planning_attempts=2,
                   target_entity='取餐处', return_entity='办公室', mission_plan={'evidence': list(range(100000))})
    projected = _plan_notification_summary(summary)
    assert len(json.dumps(projected)) < 1024
    assert len(summary['mission_plan']['evidence']) == 100000
    registry = ExtensionRegistry()
    for index, renderer in enumerate((_plan_ready, _planning_metrics, _operator_checklist)):
        registry.register(ExtensionPlugin(
            plugin_id=f'notification.test-{index}', version='1.0.0', package_sha256='b'*64,
            capability_id=f'notification.test-{index}.render', slot_id='notifications.plan-ready',
            activation_mode='multiple', failure_mode='isolate', swap_policy='anytime',
            pipeline_order=index, runs_after=(), runs_before=(), hooks={'render_plan_notification': renderer}))
    outputs, receipts = registry.invoke_multiple('notifications.plan-ready', 'render_plan_notification', summary=projected)
    assert len(outputs) == 3 and all(item['channel'] == 'task-timeline' for item in outputs)
    assert all(receipt.outcome == 'accepted' for receipt in receipts)
    with pytest.raises(ValueError, match='NOTIFICATION_SUMMARY_FIELD_INVALID'):
        _plan_notification_summary({'goal': {'unbounded': []}})
