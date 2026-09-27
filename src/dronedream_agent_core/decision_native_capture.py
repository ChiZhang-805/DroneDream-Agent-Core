"""Bounded native capture callbacks; no flight transport and no invented outcome labels.

Call begin when a behavior is selected, accepted only AFTER transport acceptance,
and witness only from the independent simulator observer. Export runs off the
control thread. This module alone does not install callbacks into an executor.
"""

import copy
import math

from .contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from .control_execution_evidence import ControlApplicationRecord, validate_application_binding
from .decision_label_evidence import verify_command_goal, verify_native_label
from .decision_shadow import ACTIONS
from .decision_state_adapter import DecisionStateV2, decision_digest


class NativeDecisionCapture:
    # 功能：创建只属于一个实际行为的有界记录器；不发飞控命令、不写盘。
    # 输入：因果状态、明确行为、任务组及独立见证配置；输出：空记录器。
    # 时间为 UNIX 毫秒，位置/目标为 ENU 米；真值配置仅存标签侧。
    def __init__(
        self,
        *,
        state,
        action,
        episode_id,
        parent_group,
        label_context,
        selected_at_ms,
        requested_speed_limit_mps=None,
        nominal_speed_limit_mps=None,
        split="train",
    ):
        self.state = DecisionStateV2.model_validate_json(state.model_dump_json())
        if (
            action not in ACTIONS
            or type(selected_at_ms) is not int
            or self.state.clock_domain != "unix-ms"
            or not 0 <= selected_at_ms - self.state.frame.observed_at_ms <= 750
            or not episode_id
            or not parent_group
            or split not in {"train", "development", "calibration", "test", "stress"}
        ):
            raise ValueError("DECISION_CAPTURE_SELECTION_INVALID")
        required = {
            "map_sha256",
            "static_primitives",
            "geometry_sha256",
            "envelope",
            "goal_world_enu_m",
        }
        if set(label_context) != required or label_context["map_sha256"] != self.state.map_sha256:
            raise ValueError("DECISION_CAPTURE_CONTEXT_INVALID")
        self.row = {
            "state": self.state.model_dump(mode="json"),
            "episode_id": episode_id,
            "parent_group": parent_group,
            "split": split,
        }
        self.action, self.selected_at_ms = action, selected_at_ms
        self.context = copy.deepcopy(label_context)
        self.speed_limits = {
            "requested_speed_limit_mps": requested_speed_limit_mps,
            "nominal_speed_limit_mps": nominal_speed_limit_mps,
        }
        self.commands, self.applications, self.observations = [], [], []
        self.navigation_snapshots = {}
        self.failure = None
        self.finished = False

    # 功能：永久标记本窗口证据无效，不中止飞行、不丢弃已采集证据。
    # 输入：错误码；输出：False，调用者可继续控制且稍后导出隔离记录。
    def reject(self, code):
        self.failure = self.failure or code
        return False

    # 功能：只接收真正发送完成后的命令/回执；拒绝错目标、倒序和无界积累。
    # 输入：原命令及传输层实际回执；输出：是否记录；拒绝不等于飞机执行失败。
    def accepted(self, command, application, *, snapshot=None):
        if self.finished or self.failure:
            return False
        try:
            command = RuntimeLocalSafetyCommand.model_validate(command.model_dump(mode="python"))
            actual = ControlApplicationRecord.model_validate(application.model_dump(mode="python"))
            validate_application_binding(command, actual)
            stamp = actual.accepted_at_unix_ms
            if (
                len(self.applications) >= 1501
                or not self.selected_at_ms <= stamp <= self.selected_at_ms + 30000
                or command.navigation_goal_id != self.state.goal_id
                or (self.applications and stamp <= self.applications[-1]["accepted_at_unix_ms"])
            ):
                return self.reject("DECISION_CAPTURE_APPLICATION_ORDER_OR_IDENTITY")
            # 局部控制目标会随前视推进，不能要求其等于语义终点。
            # 用原始模型输入摘要绑定语义目标，不以删除身份校验来放行。
            goal = self.context["goal_world_enu_m"]
            verify_command_goal(command, self.row["state"], goal, snapshot)
            if snapshot is not None:
                self.navigation_snapshots[snapshot["snapshot_sha256"]] = copy.deepcopy(snapshot)
            self.commands.append(command.model_dump(mode="json"))
            self.applications.append(actual.model_dump(mode="json"))
            return True
        except ValueError as error:
            return self.reject("DECISION_CAPTURE_APPLICATION_INVALID:" + str(error)[:180])
        except (KeyError, TypeError, AttributeError):
            return self.reject("DECISION_CAPTURE_APPLICATION_INVALID")

    # 功能：采集独立仿真见证，仅用于结果监督；严禁回填到决策输入。
    # 输入：原生真值观测；输出：是否记录；容量/时钟异常隔离本窗口。
    def witness(self, observation):
        if self.finished or self.failure:
            return False
        try:
            observation = RuntimeLocalSafetyObservation.model_validate(
                observation.model_dump(mode="python")
            )
            stamp = observation.observed_at_unix_ms
            if (
                observation.source != "simulation-ground-truth"
                or len(self.observations) >= 1501
                or not self.selected_at_ms - 100 <= stamp <= self.selected_at_ms + 30000
                or (self.observations and stamp <= self.observations[-1]["observed_at_unix_ms"])
            ):
                return self.reject("DECISION_CAPTURE_WITNESS_ORDER_OR_SOURCE")
            self.observations.append(observation.model_dump(mode="json"))
            return True
        except (ValueError, TypeError, AttributeError):
            return self.reject("DECISION_CAPTURE_WITNESS_INVALID")

    # 功能：离线封口并重新算物理结果；缺帧/失败原样隔离，不生成“成功”占位标签。
    # 输入：真实结束时刻、实际接管/提前落地事实及行为特有恢复证据；输出：结果包+资格报告。
    # 此函数会执行几何计算，不应在实时飞控刷新线程调用。
    def finish(self, *, end_ms, overridden, premature_landing, action_evidence=None):
        if self.finished:
            raise ValueError("DECISION_CAPTURE_ALREADY_FINISHED")
        self.finished = True
        if (
            type(end_ms) is not int
            or type(overridden) is not bool
            or type(premature_landing) is not bool
        ):
            self.reject("DECISION_CAPTURE_END_INVALID")
        extra = copy.deepcopy(action_evidence or {})
        allowed = {
            "wait": {"continuation"},
            "request_observation": {"end_decision_frame", "observation_recovery"},
            "replan": {
                "replan_event",
                "replanned_path_world_enu_m",
                "replacement_route",
                "continuation",
            },
        }.get(self.action, set())
        if set(extra) != allowed:
            self.reject("DECISION_CAPTURE_ACTION_EVIDENCE_INVALID")
            extra = {}
        velocities = [a["velocity_ned_mps"] for a in self.applications]
        variation = sum(
            math.dist(a, b) ** 2 for a, b in zip(velocities, velocities[1:], strict=False)
        )
        document = {
            **self.context,
            **extra,
            "schema_version": "dronedream.decision-native-outcome.v1",
            "episode_id": self.row["episode_id"],
            "parent_group": self.row["parent_group"],
            "state_sha256": decision_digest(self.row["state"]),
            "commands": self.commands,
            "control_applications": self.applications,
            "observations": self.observations,
            "commanded_change_squared": variation,
            "premature_landing": premature_landing,
            "behavior_application": {
                "purpose": "native-behavior-application",
                "action": self.action,
                "source": "executor",
                "applied": bool(self.applications),
                "state_sha256": decision_digest(self.row["state"]),
                "overridden": overridden,
                "start_ms": self.selected_at_ms,
                "end_ms": end_ms,
                "control_application_hashes": [decision_digest(a) for a in self.applications],
                **self.speed_limits,
            },
        }
        if self.navigation_snapshots:
            document["navigation_snapshots"] = self.navigation_snapshots
        if not self.failure:
            try:
                verify_native_label(self.row, document)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                self.reject(type(error).__name__ + ":" + str(error)[:300])
        report = {
            "verified": self.failure is None,
            "rejection": self.failure,
            "applications": len(self.applications),
            "witnesses": len(self.observations),
            "execution_authority": False,
        }
        return copy.deepcopy(document), report
