from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from dronedream_agent_core.simulation_runtime_world import materialize_runtime_world_plugins


def _write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def test_materializes_missing_and_replaces_stock_magnetometer(tmp_path):
    world = _write(tmp_path / "world.sdf", """<sdf version='1.9'><world name='w'>
      <plugin filename='physics' name='gz::sim::systems::Physics'/>
      <plugin filename='stock-mag' name='gz::sim::systems::Magnetometer'/>
    </world></sdf>""")
    config = _write(tmp_path / "server.config", """<server_config><plugins>
      <plugin entity_name='*' entity_type='world' filename='physics' name='gz::sim::systems::Physics'/>
      <plugin entity_name='*' entity_type='world' filename='/runtime/mag.so' name='dronedream::Magnetometer'><wire_contract>x</wire_contract></plugin>
      <plugin entity_name='*' entity_type='world' filename='/runtime/camera.so' name='dronedream::NativeCameraClock'/>
    </plugins></server_config>""")
    runtime, receipt = materialize_runtime_world_plugins(
        world_sdf=world, server_config=config, output=tmp_path / "runtime")
    plugins = ET.parse(runtime).findall("./world/plugin")
    names = [p.get("name") for p in plugins]
    assert names == ["gz::sim::systems::Physics", "dronedream::Magnetometer",
                     "dronedream::NativeCameraClock"]
    assert all("entity_name" not in p.attrib and "entity_type" not in p.attrib for p in plugins)
    assert receipt["stock_magnetometer_removed"] is True


def test_materializes_full_config_when_world_has_no_plugins(tmp_path):
    world = _write(tmp_path / "world.sdf", "<sdf version='1.9'><world name='w'/></sdf>")
    config = _write(tmp_path / "server.config", """<server_config><plugins>
      <plugin filename='physics' name='gz::sim::systems::Physics'/>
      <plugin filename='scene' name='gz::sim::systems::SceneBroadcaster'/>
    </plugins></server_config>""")
    runtime, _ = materialize_runtime_world_plugins(
        world_sdf=world, server_config=config, output=tmp_path / "runtime")
    assert [p.get("name") for p in ET.parse(runtime).findall("./world/plugin")] == [
        "gz::sim::systems::Physics", "gz::sim::systems::SceneBroadcaster"]


def test_rejects_reusing_output(tmp_path):
    world = _write(tmp_path / "world.sdf", "<sdf version='1.9'><world name='w'/></sdf>")
    config = _write(tmp_path / "server.config",
                    "<server_config><plugins><plugin filename='p' name='n'/></plugins></server_config>")
    output = tmp_path / "runtime"
    output.mkdir()
    with pytest.raises(FileExistsError):
        materialize_runtime_world_plugins(world_sdf=world, server_config=config, output=output)


def test_materializes_relative_world_resources(tmp_path):
    materials = tmp_path / "materials" / "textures"
    materials.mkdir(parents=True)
    texture = materials / "campus-surface.ppm"
    texture.write_bytes(b"P6\n1 1\n255\n\x00\x80\x00")
    world = _write(tmp_path / "world.sdf", """<sdf version='1.9'><world name='w'>
      <plugin filename='physics' name='gz::sim::systems::Physics'/>
    </world></sdf>""")
    config = _write(tmp_path / "server.config", """<server_config><plugins>
      <plugin filename='physics' name='gz::sim::systems::Physics'/>
    </plugins></server_config>""")

    _, receipt = materialize_runtime_world_plugins(
        world_sdf=world, server_config=config, output=tmp_path / "runtime")

    copied = tmp_path / "runtime" / "materials" / "textures" / texture.name
    assert copied.read_bytes() == texture.read_bytes()
    assert receipt["copied_resources"]["materials"]["file_count"] == 1
