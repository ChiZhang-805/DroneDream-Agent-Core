from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.build_dynamic_obstacle_world import build_world
from scripts.run_school_map_depth_qualification import (
    _dynamic_obstacle_observation_counts,
    _dynamic_obstacle_observation_metrics,
    _load_dynamic_obstacle_challenge,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_world(path: Path) -> None:
    meshes = path.parent / "meshes"
    meshes.mkdir(parents=True)
    (meshes / "gate.obj").write_text("o gate\n", encoding="utf-8")
    textures = path.parent / "materials" / "textures"
    textures.mkdir(parents=True)
    (textures / "surface.ppm").write_text("P3\n1 1\n255\n255 255 255\n", encoding="utf-8")
    path.write_text(
        '<sdf version="1.9"><world name="school_map_world">'
        '<model name="school_map"><static>true</static><link name="map">'
        '<visual name="gate"><geometry><mesh><uri>meshes/gate.obj</uri>'
        "</mesh></geometry></visual></link></model>"
        "</world></sdf>",
        encoding="utf-8",
    )


def _spec(path: Path, *, obstacle_id: str = "corridor-cart") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.dynamic-obstacle-challenge.v1",
                "challenge_id": "corridor-recovery",
                "world_name": "school_map_world",
                "required_encounter_distance_m": 3.0,
                "obstacles": [
                    {
                        "obstacle_id": obstacle_id,
                        "center_enu_m": [-20.0, 9.85, 9.0],
                        "size_m": [0.2, 0.2, 0.8],
                        "color_rgba": [0.95, 0.32, 0.12, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_builds_content_bound_physical_dynamic_obstacle(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "derived" / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt_path = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    spec.parent.mkdir()
    _spec(spec)

    receipt = build_world(
        base_world=base,
        challenge_spec=spec,
        output_world=output,
        receipt_output=receipt_path,
    )

    model = ET.parse(output).getroot().find(
        "./world/model[@name='dronedream_dynamic_corridor-cart']"
    )
    assert model is not None
    assert model.findtext("static") == "false"
    assert model.findtext("./link/gravity") == "false"
    assert model.findtext("./link/kinematic") == "true"
    assert model.findtext("./link/collision/geometry/box/size") == "0.2 0.2 0.8"
    assert model.findtext("./link/visual/geometry/box/size") == "0.2 0.2 0.8"
    assert receipt["output_world_sha256"] == _sha256(output)
    assert receipt["qualification_granted"] is False
    assert receipt["required_encounter_distance_m"] == 3.0
    assert receipt["obstacles"][0]["published_as_dynamic_obstacle"] is True
    assert receipt["obstacles"][0]["gazebo_static"] is False
    assert receipt["relative_resource_count"] == 2
    assert (tmp_path / "derived" / "meshes" / "gate.obj").is_file()
    assert (tmp_path / "derived" / "materials" / "textures" / "surface.ppm").is_file()
    assert json.loads(receipt_path.read_text())["challenge_id"] == "corridor-recovery"


def test_rejects_duplicate_output_and_invalid_identity(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "derived" / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    spec.parent.mkdir()
    _spec(spec, obstacle_id="INVALID ID")

    with pytest.raises(ValueError, match="obstacle_id"):
        build_world(
            base_world=base,
            challenge_spec=spec,
            output_world=output,
            receipt_output=receipt,
        )

    _spec(spec)
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_world(
            base_world=base,
            challenge_spec=spec,
            output_world=output,
            receipt_output=receipt,
        )


def test_challenge_receipt_and_runtime_observations_are_bound(tmp_path: Path) -> None:
    base = tmp_path / "source" / "base.sdf"
    spec = tmp_path / "derived" / "challenge.json"
    output = tmp_path / "derived" / "challenge.sdf"
    receipt_path = tmp_path / "derived" / "receipt.json"
    _base_world(base)
    spec.parent.mkdir()
    _spec(spec)
    build_world(
        base_world=base,
        challenge_spec=spec,
        output_world=output,
        receipt_output=receipt_path,
    )

    challenge = _load_dynamic_obstacle_challenge(receipt_path, world_sdf=output)
    entity_name = challenge["entity_names"][0]
    history = tmp_path / "local-safety-history.jsonl"
    history.write_text(
        json.dumps(
            {
                "observation": {
                    "current_position_m": {"x": -19.0, "y": 9.85, "z": 9.0},
                    "dynamic_obstacles": [
                        {
                            "obstacle_id": entity_name,
                            "position_m": {"x": -20.0, "y": 9.85, "z": 9.0},
                        }
                    ],
                },
                "command": {
                    "decision": {"threat_obstacle_id": entity_name},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert challenge["world_sha256"] == _sha256(output)
    assert _dynamic_obstacle_observation_counts(
        history,
        entity_names=[entity_name],
    ) == {entity_name: 1}
    assert _dynamic_obstacle_observation_metrics(
        history,
        entity_names=[entity_name],
    ) == {
        entity_name: {
            "observation_count": 1,
            "threat_count": 1,
            "minimum_horizontal_distance_m": 1.0,
        }
    }

    output.write_text(output.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt is invalid"):
        _load_dynamic_obstacle_challenge(receipt_path, world_sdf=output)
