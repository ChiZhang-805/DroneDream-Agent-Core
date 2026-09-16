import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from PIL import Image
from test_static_render_batching import world


def script():
    path = Path(__file__).parents[1] / "scripts/verify_static_render_equivalence.py"
    spec = importlib.util.spec_from_file_location("render_equivalence_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_diagnostic_is_static_and_does_not_load_a_flight_controller():
    result = script().add_camera_rigs(world(), [[1., 2., 3., 0., 0., .3]])
    root = ET.fromstring(result)
    camera = root.find("world/model[@name='render_probe_0']")
    assert camera.findtext("static") == "true"
    assert camera.findtext("pose") == "1 2 3 0 0 0.29999999999999999"
    assert camera.find("link/sensor").get("type") == "rgbd_camera"
    assert not camera.findall(".//plugin") and not camera.findall(".//include")
    assert root.findtext("world/physics/max_step_size") == ".004"


@pytest.mark.parametrize("poses", [[], [[1., 2., 3.]], [[1., 2., 3., 0., 0., float('nan')]]])
def test_invalid_camera_configuration_fails_before_process_launch(poses):
    with pytest.raises(ValueError):
        script().add_camera_rigs(world(), poses)


@pytest.mark.parametrize("fault,expected", [("none", True), ("rgb", False),
    ("depth", False), ("missing-surface", False), ("both-blank", False)])
def test_render_comparison_rejects_changes_and_vacuous_blank_matches(tmp_path, fault, expected):
    for kind in ("original", "batched"):
        folder = tmp_path / kind
        folder.mkdir()
        rgb = np.full((128, 224, 3), 80, dtype=np.uint8)
        depth = np.full((128, 224), 2., dtype=np.float32)
        if fault == "both-blank":
            depth[:] = np.inf
        if kind == "batched":
            if fault == "rgb":
                rgb[:20] += 100
            if fault == "depth":
                depth += .01
            if fault == "missing-surface":
                depth[:20] = np.inf
        Image.fromarray(rgb).save(folder / "rgb-0.png")
        np.save(folder / "depth-0.npy", depth, allow_pickle=False)
    result = script().compare(tmp_path / "original", tmp_path / "batched", 1)
    assert result["passed"] is expected
    assert result["flight_qualification_granted"] is False


@pytest.mark.parametrize("count,warmup,sample", [(0, 0, 10), (9, 0, 10),
    (True, 0, 10), (1, -1, 10), (1, float("nan"), 10), (1, 0, float("inf")),
    (1, 0, 4), (1, 0, 121), (1, 121, 10)])
def test_unbounded_or_nonfinite_capture_options_are_rejected(count, warmup, sample):
    with pytest.raises(ValueError, match="render probe requires"):
        script().validate_capture_options(count, warmup, sample)


def test_render_interval_summary_exposes_stalls_not_only_mean_latency():
    result = script().gap_summaries({"camera": [100., 101., 149., 160., 600.], "empty": []})
    assert result["camera"]["interval_count"] == 5
    assert result["camera"]["maximum"] == 600.
    assert result["camera"]["above_150_ms"] == 2
    assert result["camera"]["above_250_ms"] == 1
    assert "empty" not in result
