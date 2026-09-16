"""Strict declarative plugin panels with bounded data binding and actions."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dronedream_plugin_sdk.protocol import copy_json, encode_json, validate_local_schema

PanelWidgetKind = Literal[
    "text",
    "status",
    "metric",
    "log",
    "table",
    "replay",
    "telemetry",
    "configuration-form",
]
PanelSource = Literal["static", "plugin", "configuration", "events", "runtime", "task", "evidence"]
PanelActionId = Literal["panel.refresh", "plugin.healthcheck", "plugin.disable"]


class PanelModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PanelAction(PanelModel):
    action_id: PanelActionId
    label: str = Field(min_length=1, max_length=48)
    style: Literal["default", "primary", "danger"] = "default"
    confirmation: str | None = Field(default=None, min_length=1, max_length=160)

    # 功能：
    #   要求停用插件的按钮提供确认文案；执行权限仍由宿主检查，文案不构成授权。
    # 输入：
    #   self：包含动作标识与确认文案的面板动作。
    # 输出：
    #   self：确认要求校验通过的动作模型。
    @model_validator(mode="after")
    def require_destructive_confirmation(self) -> PanelAction:
        if self.action_id == "plugin.disable" and not self.confirmation:
            raise ValueError("destructive panel action requires confirmation")
        return self


class PanelWidget(PanelModel):
    widget_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{1,79}$")
    kind: PanelWidgetKind
    label: str = Field(min_length=1, max_length=80)
    source: PanelSource
    path: str = Field(default="", max_length=160)
    value: str | int | float | bool | None = None
    unit: str | None = Field(default=None, max_length=24)
    limit: int = Field(default=20, ge=1, le=100)
    columns: list[str] = Field(default_factory=list, max_length=12)

    # 功能：
    #   只允许点分隔的字典键或列表索引语法，不允许在数据路径中嵌入可执行表达式。
    # 输入：
    #   value：待绑定的点分隔路径，空值表示根对象。
    # 输出：
    #   value：语法检查通过且未被重写的路径。
    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if value and re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", value) is None:
            raise ValueError("invalid panel data path")
        return value

    # 功能：
    #   区分静态值、动态绑定和配置表单，拒绝缺少取值路径或绑定到错误来源的控件。
    # 输入：
    #   self：待验证的控件模型。
    # 输出：
    #   self：来源与控件类型匹配的模型。
    @model_validator(mode="after")
    def validate_binding(self) -> PanelWidget:
        if self.source == "static" and self.value is None:
            raise ValueError("static widget requires value")
        if self.source != "static" and self.kind != "configuration-form" and not self.path:
            raise ValueError("bound widget requires path")
        if self.kind == "configuration-form" and self.source != "configuration":
            raise ValueError("configuration form must use configuration source")
        return self


class PanelSection(PanelModel):
    section_id: str = Field(pattern=r"^[a-z][a-z0-9._-]{1,79}$")
    title: str = Field(min_length=1, max_length=80)
    widgets: list[PanelWidget] = Field(default_factory=list, max_length=40)
    actions: list[PanelAction] = Field(default_factory=list, max_length=8)

    # 功能：
    #   保证同一分区的动作标识唯一，避免一个宿主动作对应多个歧义按钮。
    # 输入：
    #   self：包含按钮集合的分区模型。
    # 输出：
    #   self：动作标识检查通过的分区模型。
    @model_validator(mode="after")
    def validate_actions(self) -> PanelSection:
        identifiers = [action.action_id for action in self.actions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate panel action id")
        return self


class DeclarativePanelDocument(PanelModel):
    schema_version: Literal["dronedream.ui-panel.v1"] = "dronedream.ui-panel.v1"
    title: str = Field(min_length=1, max_length=120)
    sections: list[PanelSection] = Field(default_factory=list, max_length=24)

    # 功能：
    #   验证分区与控件标识在文档内唯一，并限制完整 UTF-8 文档大小，保护前端的数据绑定。
    # 输入：
    #   self：待验证的完整声明式面板。
    # 输出：
    #   self：标识与大小检查通过的文档模型。
    @model_validator(mode="after")
    def validate_size_and_identity(self) -> DeclarativePanelDocument:
        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("duplicate panel section id")
        widget_ids = [widget.widget_id for section in self.sections for widget in section.widgets]
        if len(widget_ids) != len(set(widget_ids)):
            raise ValueError("duplicate panel widget id")
        encode_json(self.model_dump(mode="python"), limit=256_000)
        return self


# 功能：
#   只按声明式面板契约验证字典，不执行插件提供的界面脚本。
# 输入：
#   value：待验证的面板原始对象。
# 输出：
#   document：验证通过的面板模型。
def validate_panel_document(value: object) -> DeclarativePanelDocument:
    if not isinstance(value, dict):
        raise ValueError("UI_PLUGIN_DOCUMENT_INVALID")
    document = DeclarativePanelDocument.model_validate(value)
    return document


# 功能：
#   按已校验路径读取字典键或列表索引，缺少数据时保留缺失状态而不猜测替代值。
# 输入：
#   value：作为读取起点的数据对象。
#   path：调用方通过面板契约检查后的点分隔路径。
# 输出：
#   current：路径对应的值；无法解析路径时为 None。
def _at_path(value: object, path: str) -> object:
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
        else:
            current = None
            return current
    return current


# 功能：
#   1. 重新校验可变面板模型，再从宿主提供的数据源生成显示数据，不执行按钮动作。
#   2. 对列表限制行数，对嵌套值限制字节数；所有绑定结果及配置 Schema 都隔离可变引用。
# 输入：
#   document：已导入但仍可能被修改的面板模型。
#   sources：宿主允许本次界面读取的命名数据源。
#   configuration_schema：配置表单使用的宿主 Schema。
# 输出：
#   detached：完成数据绑定且与宿主可变对象隔离的面板字典。
def materialize_panel(
    document: DeclarativePanelDocument,
    *,
    sources: dict[str, object],
    configuration_schema: dict[str, Any],
) -> dict[str, object]:
    # 导入时通过不代表当前仍有效，模型字段可能在两次操作之间被修改。
    payload = validate_panel_document(document.model_dump(mode="python")).model_dump(mode="json")
    schema = validate_local_schema(configuration_schema) if configuration_schema else {}
    for section in payload["sections"]:
        for widget in section["widgets"]:
            if widget["source"] == "static":
                continue
            if widget["kind"] == "configuration-form":
                widget["schema"] = copy_json(schema, limit=256_000)
                widget["resolved"] = copy_json(sources.get("configuration", {}), limit=256_000)
                continue
            resolved = _at_path(sources.get(str(widget["source"]), {}), str(widget["path"]))
            if isinstance(resolved, list):
                resolved = resolved[: int(widget["limit"])]
            # 限制行数不能限制单行中的巨大嵌套值，因此绑定后仍做字节预算检查。
            widget["resolved"] = copy_json(resolved, limit=256_000)
    detached = copy_json(payload)
    return detached
