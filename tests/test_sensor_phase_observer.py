"""Sensor-worker observer wiring; synthetic faults, no simulator launched."""

import ast
import contextlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import scripts.runtime_depth_safety_worker as worker


@pytest.mark.parametrize("initialization_failed", [False, True])
def test_sensor_worker_joins_phase_reader_and_records_close_even_on_failure(
    tmp_path, monkeypatch, initialization_failed
):
    receipt = {"thread_stopped": True, "confirms_physical_landing": False}
    observer = SimpleNamespace(close=Mock(return_value=receipt))
    constructor = Mock(return_value=observer)
    monkeypatch.setattr(worker, "RuntimePhaseObserver", constructor)
    try:
        with contextlib.ExitStack() as cleanup:
            assert worker._start_phase_observer(tmp_path, cleanup) is observer
            observer.close.assert_not_called()
            if initialization_failed:
                raise RuntimeError("initialization fault")
    except RuntimeError as error:
        assert initialization_failed and str(error) == "initialization fault"
    constructor.assert_called_once_with(
        tmp_path / "runtime-phase.json", reader=worker._runtime_phase_context)
    observer.close.assert_called_once()
    assert json.loads((tmp_path / "sensor-phase-observer-receipt.json").read_bytes()) == receipt


def test_unjoined_phase_thread_cannot_publish_a_successful_shutdown_receipt(tmp_path, monkeypatch):
    observer = SimpleNamespace(close=Mock(side_effect=RuntimeError("reader did not stop")))
    monkeypatch.setattr(worker, "RuntimePhaseObserver", Mock(return_value=observer))
    with (pytest.raises(RuntimeError, match="reader did not stop"),
          contextlib.ExitStack() as cleanup):
        worker._start_phase_observer(tmp_path, cleanup)
    assert not (tmp_path / "sensor-phase-observer-receipt.json").exists()


def test_runtime_loop_uses_bounded_observer_for_both_model_and_teacher_context():
    # Architectural wiring check; the real thread/age/ending behavior is
    # exercised in test_runtime_phase_observer, not proven by this AST test.
    tree = ast.parse(inspect.getsource(worker._run_worker))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Name) and node.func.id == "_runtime_phase_context"
                   for node in calls)
    reads = [node for node in calls if isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "phase_observer" and node.func.attr == "latest"]
    assert len(reads) == 2
