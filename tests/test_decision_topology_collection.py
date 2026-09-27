"""Batch orchestration tests use no aircraft and yield no formal samples."""

from types import SimpleNamespace

import pytest

from scripts import collect_decision_topology_suite as batch


# 功能：隔离批处理外部依赖；输入：临时目录、补丁；输出：不启动物理仿真的受控配置。
def configure(monkeypatch, tmp_path):
    source = tmp_path / "suite.json"
    source.write_text("{}")
    config = SimpleNamespace(model_dump=lambda **kw: {})
    monkeypatch.setattr(
        batch,
        "preflight_suite",
        lambda _: [
            ({"config": f"layout-{i}/config.json", "split": "train", "family": str(i)}, config)
            for i in range(5)
        ],
    )
    monkeypatch.setattr(
        batch,
        "Px4TrainingConfig",
        SimpleNamespace(model_validate=lambda doc: SimpleNamespace(**doc)),
    )
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(batch, "audit", lambda *args, **kwargs: None)
    return source


# 功能：保证连续审计失败不会被下一次成功采集清零；输出：第二轮结束批次。
def test_consecutive_audit_errors_stop(monkeypatch, tmp_path):
    source = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(
        batch,
        "collect",
        lambda config, *a, **kw: {
            "safely_closed": True,
            "episode_path": str(config.output_root / "episode"),
        },
    )

    def fail(*a, **kw):
        raise ValueError("deliberate-audit-failure")

    monkeypatch.setattr(batch, "assemble", fail)
    result = batch.run_suite(source, tmp_path / "result", maximum_layouts=5)
    assert result["attempts"] == 2
    assert result["stop_reason"] == "CONSECUTIVE_COLLECTION_OR_AUDIT_ERRORS"
    assert result["formal_decision_windows"] == 0


# 功能：无安全收尾不得启动下一轮；输出：仅一次实验且资格不放行。
def test_no_safe_terminal_state_stops_immediately(monkeypatch, tmp_path):
    source = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(batch, "collect", lambda *a, **kw: {"safely_closed": False})
    result = batch.run_suite(source, tmp_path / "result", maximum_layouts=5)
    assert result["attempts"] == 1
    assert result["stop_reason"] == "SAFE_TERMINAL_STATE_NOT_CONFIRMED"
    assert not result["gpu_ready"]


# 功能：空产出不能无限继续，也不能伪造正式数据；输出：三轮无正式窗口自动结束。
def test_empty_runs_stop_and_valid_counts_sum(monkeypatch, tmp_path):
    source = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(
        batch,
        "collect",
        lambda config, *a, **kw: {
            "safely_closed": True,
            "episode_path": str(config.output_root / "episode"),
        },
    )
    monkeypatch.setattr(batch, "assemble", lambda *a: {"formal_decision_windows": 0})
    result = batch.run_suite(source, tmp_path / "empty", maximum_layouts=5)
    assert result["attempts"] == 3
    assert result["stop_reason"] == "CONSECUTIVE_COLLECTION_WITHOUT_VERIFIED_DATA"
    monkeypatch.setattr(batch, "assemble", lambda *a: {"formal_decision_windows": 2})
    result = batch.run_suite(source, tmp_path / "nonempty", maximum_layouts=3)
    assert result["attempts"] == 3 and result["formal_decision_windows"] == 6
    assert result["stop_reason"] is None and not result["gpu_ready"]


# 功能：不能把旧回合当作新实验产出；输出：归属不符保留为错误且不授予标签。
def test_episode_outside_attempt_is_rejected(monkeypatch, tmp_path):
    source = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(
        batch,
        "collect",
        lambda *a, **kw: {"safely_closed": True, "episode_path": str(tmp_path / "unrelated")},
    )
    result = batch.run_suite(source, tmp_path / "result", maximum_layouts=5)
    assert result["attempts"] == 2 and result["formal_decision_windows"] == 0


# 功能：资源/时间预算必须有界；输入：非法预算；输出：启动前拒绝。
@pytest.mark.parametrize(
    "kw",
    [
        {"maximum_layouts": True},
        {"seconds": float("nan")},
        {"maximum_layouts": 21},
        {"seconds": 121},
    ],
)
def test_invalid_budget_is_rejected(tmp_path, kw):
    with pytest.raises(ValueError, match="BUDGET_INVALID"):
        batch.run_suite(tmp_path / "missing", tmp_path / "output", **kw)
