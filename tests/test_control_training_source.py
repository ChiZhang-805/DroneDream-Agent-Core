"""Training source selection cannot fall back to a stale bundled sensor worker."""

import ast
from pathlib import Path

import pytest

from dronedream_agent_core import gazebo_adapter


@pytest.mark.parametrize("bundled_exists", [False, True])
@pytest.mark.parametrize("source_training", [False, True])
def test_current_checkout_always_uses_its_paired_worker(tmp_path, bundled_exists, source_training):
    source, bundled = tmp_path / "source.py", tmp_path / "bundled.py"
    source.write_text("current sensor pipeline")
    if bundled_exists:
        bundled.write_text("old sensor pipeline")
    assert gazebo_adapter._select_depth_worker(
        packaged_worker=bundled, source_worker=source, source_training=source_training,
    ) == source


def test_missing_training_source_is_an_error_not_implicit_rollback(tmp_path):
    bundled = tmp_path / "bundled.py"
    bundled.write_text("old sensor pipeline")
    with pytest.raises(ValueError, match="current source sensor worker"):
        gazebo_adapter._select_depth_worker(packaged_worker=bundled,
            source_worker=tmp_path / "missing.py", source_training=True)
    assert gazebo_adapter._select_depth_worker(packaged_worker=bundled,
        source_worker=tmp_path / "missing.py", source_training=False) == bundled


def test_actual_launcher_classifies_both_explicit_training_modes():
    # Inspect the actual local expression without launching any simulator.
    tree = ast.parse(Path(gazebo_adapter.__file__).read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "run_px4_gazebo_track")
    assignment = next(n for n in run.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "source_training"
                              for t in n.targets))
    expression = compile(ast.Expression(assignment.value), "<source-training>", "eval")
    for learner, teacher, expected in [(None, False, False), (None, True, True),
                                       (Path("learner"), False, True)]:
        assert eval(expression, {"__builtins__": {}}, {
            "simulation_training_channel": learner, "simulation_teacher_control": teacher,
        }) is expected
