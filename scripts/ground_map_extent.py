"""Register adjoining orthophoto coverage without shifting the existing campus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


def expand_registered_ground(gazebo: Path, semantic: dict, source_path: Path) -> dict:
    """Preserve base pixels; extend the texture, physical floor and Live affine together."""
    source = json.loads(source_path.read_text("utf-8"))
    image_path = source_path.parent / source["image_file"]
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != source["image_sha256"]:
        raise ValueError("Expanded orthophoto does not match its source")
    if source["crs"] != semantic["real_world_crs"] or source["origin"] != [
        semantic["origin"]["easting_m"], semantic["origin"]["northing_m"]
    ]:
        raise ValueError("Expanded orthophoto uses a different geographic reference")
    x0, y0, x1, y1 = source["extent_enu_m"]
    bx0, by0, bx1, by1 = source["base_extent_enu_m"]
    original_path = gazebo / "materials/textures/kumpula-orthophoto.jpg"
    with Image.open(image_path) as image, Image.open(original_path) as original:
        expanded = image.convert("RGB")
        sx, sy = image.width / (x1 - x0), image.height / (y1 - y0)
        placement = [(bx0 - x0) * sx, (y1 - by1) * sy]
        if (
            not (x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1)
            or abs((bx1 - bx0) * sx - original.width) > 1e-7
            or abs((by1 - by0) * sy - original.height) > 1e-7
            or any(abs(v - round(v)) > 1e-7 for v in placement)
        ):
            raise ValueError("Expanded raster must preserve the original campus pixel grid")
        expanded.paste(original.convert("RGB"), tuple(round(v) for v in placement))
        texture = "materials/textures/kumpula-orthophoto-expanded.png"
        expanded.save(gazebo / texture)
        width, height = expanded.size

    # The metric origin is unchanged. A larger asymmetric crop is not centered
    # on (0, 0), so neither the SDF box nor the Live map can assume that center.
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    for filename in ("model.sdf", "world.sdf", "world.physics.sdf", "world.perception.sdf"):
        tree = ET.parse(gazebo / filename)
        root = tree.getroot()
        model = root.find("model")
        if model is None:
            model = root.find("world/model[@name='kumpula_campus']")
        ground = model.find("link[@name='ground']")
        collision = ground.find("collision[@name='ground-collision']")
        collision.find("pose").text = f"{center_x:g} {center_y:g} -0.1 0 0 0"
        collision.find("geometry/box/size").text = f"{x1 - x0:g} {y1 - y0:g} 0.2"
        # The physics-only world deliberately omits the textured visual.
        albedo = ground.find("visual[@name='orthophoto-ground']/material/pbr/metal/albedo_map")
        if albedo is not None:
            albedo.text = texture
        ET.indent(tree, space="  ")
        tree.write(gazebo / filename, encoding="utf-8", xml_declaration=True)
    (gazebo / "meshes/kumpula-ground.obj").write_text(
        "o KumpulaGround\n"
        + "".join(f"v {x:g} {y:g} 0\n" for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        + "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\n",
        encoding="utf-8", newline="\n",
    )
    semantic["bounds_m"].update(x=x1 - x0, y=y1 - y0)
    semantic["extent_enu_m"] = [x0, y0, x1, y1]
    semantic["live_map"] = {
        "schema_version": "dronedream.live-map-georeference.v1",
        "image_path": "normalized/map/gazebo/" + texture,
        "width_px": width, "height_px": height, "coordinate_frame": "ENU",
        "world_to_image": [sx, 0, -x0 * sx, 0, -sy, y1 * sy],
    }
    receipt = {
        **{k: v for k, v in source.items() if k != "image_file"},
        "base_image_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "base_placement_px": list(map(round, placement)),
        "base_pixel_operation": "unchanged pixels; integer translation only",
        "expanded_texture": texture,
        "expanded_texture_sha256": hashlib.sha256((gazebo / texture).read_bytes()).hexdigest(),
    }
    semantic.setdefault("sources", {})["orthophoto_extension"] = receipt
    return receipt
