"""Simulation-only observation archive must not relax the flight control boundary."""

from types import SimpleNamespace

import pytest
from control_fixtures import complete_feature_snapshot
from test_perception_runtime import _frame, _world

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    RuntimePerceptionFusion,
)


# 功能：过期输入保留原时间而非运动权限；生产端口不新增该归档，防止额外延迟。
# 输入：合成仿真/生产端口；输出：拒绝回执与可审计原快照，不产生飞行或正式样本。
@pytest.mark.parametrize("simulation", [True, False])
def test_rejected_input_is_visible_but_never_executed(simulation):
    fusion = RuntimePerceptionFusion(
        world=_world(), accepted_sensor_ids={"front-lidar"}, minimum_rays_per_frame=8
    )
    fusion.ingest(_frame(), now_unix_ms=1020)
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=SimpleNamespace(simulation_only=simulation),
        required_clearance_m=0.0,
        control_output_mode="normalized-body-velocity",
    )
    raw = complete_feature_snapshot().model_dump(mode="json")
    try:
        goal = Vector3(x=3.75, y=0.25, z=0.25)
        params = dict(
            goal_position_m=goal,
            navigation_goal_id="test",
            now_unix_ms=2000,
            trigger="initial",
            realtime_feature_snapshot=raw,
        )
        receipt = coordinator.schedule(**params)
        assert receipt.model_action == "not-invoked"
        assert receipt.hold_reason == "CONTINUOUS_CONTROL_SOURCE_NOT_READY"
        snapshot = coordinator.pop_submitted_snapshot()
        if simulation:
            assert snapshot["realtime_feature_snapshot"] == raw
            assert snapshot["control_reference_observed_at_unix_ms"] == 2000
            assert snapshot["snapshot_sha256"] == receipt.snapshot_sha256
            assert (
                sha256_json({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})
                == receipt.snapshot_sha256
            )
            raw["encodings"][0]["source_sha256"] = "f" * 64
            assert snapshot["realtime_feature_snapshot"] != raw
            # 同一高速循环不能反复复制大输入；缺失归档也不会恢复控制权限。
            again = coordinator.schedule(**params)
            assert again.snapshot_sha256 is None
            assert coordinator.pop_submitted_snapshot() is None
        else:
            assert snapshot is None and receipt.snapshot_sha256 is None
        directive = coordinator.controller_directive(
            current_position_m=_frame().localization_position_m,
            fallback_target_m=goal,
            now_unix_ms=2010,
            navigation_goal_id="test",
        )
        assert not directive.model_navigation_authorized
    finally:
        coordinator.close()
