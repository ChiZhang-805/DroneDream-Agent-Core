from __future__ import annotations

import pytest
from pydantic import ValidationError

from dronedream_agent_core.plugin_panels import materialize_panel, validate_panel_document


# 功能：
#   构造状态、限行日志、配置表单与受限动作的完整声明式面板夹具。
# 输入：
#   无。
# 输出：
#   document：尚未经过契约校验的面板字典。
def _document() -> dict[str, object]:
    document = {
        "schema_version": "dronedream.ui-panel.v1",
        "title": "Mission telemetry",
        "sections": [
            {
                "section_id": "runtime",
                "title": "Runtime",
                "widgets": [
                    {
                        "widget_id": "runtime-ready",
                        "kind": "status",
                        "label": "Ready",
                        "source": "runtime",
                        "path": "ready",
                    },
                    {
                        "widget_id": "recent-events",
                        "kind": "log",
                        "label": "Events",
                        "source": "events",
                        "path": "items",
                        "limit": 2,
                    },
                    {
                        "widget_id": "configuration",
                        "kind": "configuration-form",
                        "label": "Configuration",
                        "source": "configuration",
                    },
                ],
                "actions": [
                    {"action_id": "panel.refresh", "label": "Refresh"},
                    {
                        "action_id": "plugin.disable",
                        "label": "Disable",
                        "style": "danger",
                        "confirmation": "Disable this plugin?",
                    },
                ],
            }
        ],
    }
    return document


# 功能：
#   验证面板从宿主数据源解析状态、截断日志行数，并使用宿主配置 Schema 而非插件自定表单。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_panel_data_binding_is_bounded_and_configuration_schema_is_core_owned():
    document = validate_panel_document(_document())
    value = materialize_panel(
        document,
        sources={
            "runtime": {"ready": True},
            "events": {"items": [{"id": 1}, {"id": 2}, {"id": 3}]},
            "configuration": {"threshold": 3},
        },
        configuration_schema={
            "type": "object",
            "properties": {"threshold": {"type": "integer"}},
        },
    )
    widgets = value["sections"][0]["widgets"]
    assert widgets[0]["resolved"] is True
    assert widgets[1]["resolved"] == [{"id": 1}, {"id": 2}]
    assert widgets[2]["resolved"] == {"threshold": 3}
    assert widgets[2]["schema"]["properties"]["threshold"]["type"] == "integer"


# 功能：
#   验证声明式界面不能请求任意 shell 动作，停用按钮也必须提供用户确认文案。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_panel_rejects_unknown_actions_and_unconfirmed_destructive_actions():
    document = _document()
    document["sections"][0]["actions"][0]["action_id"] = "shell.execute"
    with pytest.raises(ValidationError):
        validate_panel_document(document)

    document = _document()
    del document["sections"][0]["actions"][1]["confirmation"]
    with pytest.raises(ValidationError, match="requires confirmation"):
        validate_panel_document(document)


# 功能：
#   验证旧面板字段不会被猜测迁移成当前控件，避免不明来源的数据获得新界面语义。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_panel_without_the_current_schema_is_rejected_instead_of_migrated():
    with pytest.raises(ValidationError):
        validate_panel_document(
            {"title": "Audit", "sections": [{"title": "Summary", "items": ["Ready"]}]}
        )
