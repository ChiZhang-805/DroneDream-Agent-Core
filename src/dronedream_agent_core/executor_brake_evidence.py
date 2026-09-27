"""Actual protective transport receipts, deliberately separate from model actions."""

from typing import Literal

from pydantic import Field, field_validator

from .contracts import StrictModel
from .control_execution_evidence import ControlApplicationRecord


class ExecutorBrakeApplication(StrictModel):
    """No fabricated sensor lease or model input for a transport-only brake."""

    schema_version: Literal["dronedream.executor-brake-application.v1"] = (
        "dronedream.executor-brake-application.v1"
    )
    sequence: int = Field(ge=1, strict=True)
    after_command_application_sequence: int = Field(ge=0, strict=True)
    accepted_at_unix_ms: int = Field(ge=0, strict=True)
    reason: Literal["input-lease-unavailable", "command-unreadable", "command-stale",
                    "command-missing", "command-startup", "hybrid-execution-contract"]
    position_ned_m: tuple[float, float, float]
    velocity_ned_mps: tuple[float, float, float] = (0., 0., 0.)
    yaw_heading_deg: float = Field(strict=True)
    transport: Literal["position-velocity-ned"] = "position-velocity-ned"
    model_authorized: Literal[False] = False
    execution_success_claimed: Literal[False] = False

    # 功能：保持与真实动作回执相同的数值边界，并禁止制动记录携带非零运动速度。
    # 输入：实际NED位置/速度（米、米每秒）；输出：校验后的向量，不改变控制权限。
    @field_validator("position_ned_m", "velocity_ned_mps", mode="before")
    @classmethod
    def numeric_vector(cls, value, info):
        value = ControlApplicationRecord.physical_vector_is_numeric(value)
        if value is None or (info.field_name == "velocity_ned_mps" and any(value)):
            raise ValueError("EXECUTOR_BRAKE_VECTOR_INVALID")
        return value
