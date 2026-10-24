"""Real SDF/OBJ bindings used by the worker, without starting a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_optical_map import cube

from dronedream_agent_core.gazebo_adapter import _depth_worker_map_arguments
from dronedream_agent_core.local_map_alignment import MapAlignmentLimits
from scripts.runtime_depth_safety_worker import _load_worker_static_surfaces


def _world(tmp_path):
    """The rotated, scaled cube has a physical front face at world x=5."""
    mesh = tmp_path / "canopy.obj"
    mesh.write_bytes(cube())
    world = tmp_path / "world.sdf"
    world.write_text(
        '<sdf><world name="w"><model name="canopy"><static>true</static>'
        '<pose>6 2 3 0 0 1.5707963267948966</pose><link name="l">'
        '<visual name="canopy"><geometry><mesh><uri>canopy.obj</uri>'
        '<scale>2 1 1</scale></mesh></geometry></visual>'
        '</link></model></world></sdf>', encoding="utf-8")
    semantic = {"canopy_collision_meshes": [{"meshes": [{
        "mesh_uri": "canopy.obj", "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}]}],
        # This deliberately huge exclusion is navigation policy, not a visual surface.
        "traversability": {"exclusions": [{"polygon_xy_m": [
            [-100, -100], [100, -100], [100, 100], [-100, 100]]}]}}
    return world, mesh, semantic


def _load(world, semantic, digest=None):
    return _load_worker_static_surfaces(
        world_path=world, semantic=semantic,
        expected_world_sha256=digest or hashlib.sha256(world.read_bytes()).hexdigest())


def test_adapter_world_identity_reaches_actual_transformed_mesh(tmp_path):
    world, mesh, semantic = _world(tmp_path)
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    digest = hashlib.sha256(world.read_bytes()).hexdigest()
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--static-world", required=True)
    parser.add_argument("--static-world-sha256", required=True)
    args = parser.parse_args(_depth_worker_map_arguments(
        world, semantic_path, {world: digest}))
    index, receipt = _load_worker_static_surfaces(
        world_path=Path(args.static_world), expected_world_sha256=args.static_world_sha256,
        semantic=json.loads(Path(args.semantic).read_text(encoding="utf-8")))
    result = index.match([[5., 2.3, 3.1]], MapAlignmentLimits(),
                         sensor_origins_world_m=[0., 2.3, 3.1])
    assert result.valid.tolist() == [True]
    assert result.surface_ranges_m[0] == pytest.approx(5.)
    assert np.allclose(result.normals[0], [-1., 0., 0.], atol=1e-10)
    assert receipt["source_world_sha256"] == digest
    assert receipt["resource_dependencies"]["canopy.obj"]["sha256"] == hashlib.sha256(
        mesh.read_bytes()).hexdigest()
    # A point in front of the surface is not itself explained by the large exclusion.
    independent = index.match([[3., 2.3, 3.1]], MapAlignmentLimits(),
                              sensor_origins_world_m=[0., 2.3, 3.1])
    assert not independent.valid[0]


def test_changed_world_is_rejected_before_becoming_a_static_identity(tmp_path):
    world, _, semantic = _world(tmp_path)
    digest = hashlib.sha256(world.read_bytes()).hexdigest()
    world.write_text(world.read_text().replace("6 2 3", "16 2 3"), encoding="utf-8")
    with pytest.raises(ValueError, match="OPTICAL_MAP_WORLD_DIGEST_MISMATCH"):
        _load(world, semantic, digest)


def test_changed_mesh_is_rejected_even_when_geometry_is_unchanged(tmp_path):
    world, mesh, semantic = _world(tmp_path)
    mesh.write_bytes(mesh.read_bytes() + b"# replaced resource\n")
    with pytest.raises(ValueError, match="DEPTH_STATIC_CANOPY_RESOURCE_MISMATCH"):
        _load(world, semantic)


def test_declared_canopy_absent_from_rendered_world_is_rejected(tmp_path):
    world, _, semantic = _world(tmp_path)
    semantic["canopy_collision_meshes"][0]["meshes"][0]["mesh_uri"] = "missing.obj"
    with pytest.raises(ValueError, match="DEPTH_STATIC_CANOPY_RESOURCE_MISMATCH"):
        _load(world, semantic)


@pytest.mark.parametrize("binding", [None, {}, {"meshes": None}, {"meshes": [{}]}])
def test_incomplete_declared_canopy_binding_is_rejected(tmp_path, binding):
    world, _, semantic = _world(tmp_path)
    semantic["canopy_collision_meshes"] = binding
    with pytest.raises(ValueError, match="DEPTH_STATIC_CANOPY_BINDING_INVALID"):
        _load(world, semantic)


def test_maps_without_canopy_metadata_still_use_their_actual_static_surfaces(tmp_path):
    world, _, _ = _world(tmp_path)
    index, _ = _load(world, {})
    result = index.match([[5., 2.3, 3.1]], MapAlignmentLimits(),
                         sensor_origins_world_m=[0., 2.3, 3.1])
    assert result.valid[0]


@pytest.mark.parametrize("style", ["bundle-object", "legacy-mesh-list"])
def test_other_existing_canopy_receipt_shapes_preserve_same_binding(tmp_path, style):
    world, _, semantic = _world(tmp_path)
    bundle = semantic["canopy_collision_meshes"][0]
    semantic["canopy_collision_meshes"] = (
        bundle if style == "bundle-object" else bundle["meshes"])
    index, _ = _load(world, semantic)
    result = index.match([[5., 2.3, 3.1]], MapAlignmentLimits(),
                         sensor_origins_world_m=[0., 2.3, 3.1])
    assert result.valid[0]
