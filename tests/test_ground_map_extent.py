"""An asymmetric map extension must keep pixel, world and runtime origins aligned."""

import hashlib
import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageChops


def test_extension_preserves_original_pixels_and_geographic_origin(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/ground_map_extent.py"
    spec = importlib.util.spec_from_file_location("ground_map_extent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gazebo = tmp_path / "gazebo"
    (gazebo / "meshes").mkdir(parents=True)
    textures = gazebo / "materials/textures"
    textures.mkdir(parents=True)
    base_image = textures / "kumpula-orthophoto.jpg"
    Image.new("RGB", (2048, 2048), (73, 100, 19)).save(base_image)
    original_bytes = base_image.read_bytes()
    added_image = tmp_path / "source.png"
    Image.new("RGB", (2304, 2304), (19, 70, 90)).save(added_image)
    source = {
        "image_file": added_image.name,
        "image_sha256": hashlib.sha256(added_image.read_bytes()).hexdigest(),
        "crs": "EPSG:3879", "origin": [25497920, 6676880],
        "extent_enu_m": [-281.25, -250, 281.25, 312.5],
        "base_extent_enu_m": [-250, -250, 250, 250],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    model = (
        '<model name="kumpula_campus"><link name="ground">'
        '<collision name="ground-collision"><pose>0 0 -0.1 0 0 0</pose>'
        '<geometry><box><size>500 500 0.2</size></box></geometry></collision>'
        '<visual name="orthophoto-ground"><material><pbr><metal><albedo_map>'
        'materials/textures/kumpula-orthophoto.jpg</albedo_map></metal></pbr>'
        '</material></visual></link></model>'
    )
    for filename in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        content = model if filename == "model.sdf" else f'<world name="campus">{model}</world>'
        if filename == "world.physics.sdf":
            document = ET.fromstring(content)
            ground = document.find("model/link")
            ground.remove(ground.find("visual"))
            content = ET.tostring(document, encoding="unicode")
        (gazebo / filename).write_text(f'<sdf version="1.9">{content}</sdf>')
    semantic = {
        "real_world_crs": "EPSG:3879",
        "origin": {"easting_m": 25497920, "northing_m": 6676880},
        "bounds_m": {"x": 500, "y": 500, "z": 35},
        "runtime_bindings": {"vehicle_spawn": {"x": -41, "y": -83, "z": .067}},
    }
    module.expand_registered_ground(gazebo, semantic, source_path)
    assert base_image.read_bytes() == original_bytes
    assert semantic["runtime_bindings"]["vehicle_spawn"] == {"x": -41, "y": -83, "z": .067}
    with Image.open(textures / "kumpula-orthophoto-expanded.png") as expanded:
        with Image.open(base_image) as original:
            restored = expanded.crop((128, 256, 2176, 2304))
            assert ImageChops.difference(restored, original).getbbox() is None
        assert expanded.getpixel((0, 0)) == (19, 70, 90)
    a, _, tx, _, d, ty = semantic["live_map"]["world_to_image"]
    # Existing map origin and newly included west street retain their true ENU locations.
    assert [tx, ty] == [1152, 1280]
    assert [a * -259 + tx, d * 218 + ty] == [91.13599999999997, 387.072]
    for filename in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        collision = ET.parse(gazebo / filename).getroot().find(".//collision")
        assert collision.find("pose").text == "0 31.25 -0.1 0 0 0"
        assert collision.find("geometry/box/size").text == "562.5 562.5 0.2"
    assert semantic["bounds_m"] == {"x": 562.5, "y": 562.5, "z": 35}
    assert ET.parse(gazebo / "world.physics.sdf").getroot().find(".//visual") is None
