#!/usr/bin/env python3
"""Generate repository-card previews from reviewed map and vehicle source assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

MAP_SOURCES = {
    "airport-terminal": ("airport_terminal", "airport_terminal.png"),
    "open-test-arena": ("battle_royale", "battle_royale.png"),
    "campus-site": ("campus", "campus_sim.png"),
    "two-level-clinic": ("clinic", "clinic.png"),
    "three-level-hotel": ("hotel", "hotel_L1.png"),
    "office": ("office", "office.png"),
    "triple-h-corridor": ("triple_H", "triple_H.png"),
}

VEHICLE_MODELS = {
    "x500-depth": "x500_depth",
    "x500-flow": "x500_flow",
    "x500-lidar-down": "x500_lidar_down",
    "x500-lidar-front": "x500_lidar_front",
    "x500-mono-front": "x500_mono_cam",
    "x500-mono-down": "x500_mono_cam_down",
    "x500-lidar-2d": "x500_lidar_2d",
    "x500-vision": "x500_vision",
    "x500-gimbal": "x500_gimbal",
    "x500": "x500",
    "standard-vtol": "standard_vtol",
    "tiltrotor": "tiltrotor",
    "quadtailsitter": "quadtailsitter",
}


def _pose(element: ET.Element | None) -> tuple[float, float, float, float]:
    values = [float(value) for value in ((element.text if element is not None else "") or "").split()]
    values.extend([0.0] * (6 - len(values)))
    return values[0], values[1], values[2], values[5]


def _material_color(visual: ET.Element) -> tuple[int, int, int, int]:
    value = visual.findtext("material/diffuse") or visual.findtext("material/ambient")
    if not value:
        return (118, 112, 128, 255)
    channels = [float(channel) for channel in value.split()]
    channels.extend([1.0] * (4 - len(channels)))
    return tuple(max(0, min(255, round(channel * 255))) for channel in channels[:4])


def _school_topdown(world_path: Path, output: Path) -> None:
    root = ET.parse(world_path).getroot()
    shapes: list[dict[str, object]] = []
    for model in root.findall(".//world/model"):
        mx, my, mz, myaw = _pose(model.find("pose"))
        for link in model.findall("link"):
            lx, ly, lz, lyaw = _pose(link.find("pose"))
            for visual in link.findall("visual"):
                vx, vy, vz, vyaw = _pose(visual.find("pose"))
                x, y, z = mx + lx + vx, my + ly + vy, mz + lz + vz
                yaw = myaw + lyaw + vyaw
                geometry = visual.find("geometry")
                if geometry is None:
                    continue
                box = geometry.find("box/size")
                cylinder = geometry.find("cylinder")
                sphere = geometry.find("sphere/radius")
                shape: dict[str, object] | None = None
                if box is not None and box.text:
                    sx, sy, sz = [float(value) for value in box.text.split()]
                    shape = {"type": "box", "x": x, "y": y, "z": z + sz / 2, "sx": sx, "sy": sy, "yaw": yaw}
                elif cylinder is not None:
                    radius = float(cylinder.findtext("radius") or "0")
                    length = float(cylinder.findtext("length") or "0")
                    shape = {"type": "circle", "x": x, "y": y, "z": z + length / 2, "radius": radius}
                elif sphere is not None and sphere.text:
                    radius = float(sphere.text)
                    shape = {"type": "circle", "x": x, "y": y, "z": z + radius, "radius": radius}
                if shape is not None:
                    shape["color"] = _material_color(visual)
                    shapes.append(shape)
    if not shapes:
        raise RuntimeError("SCHOOL_MAP_HAS_NO_RENDERABLE_GEOMETRY")

    bounds: list[tuple[float, float]] = []
    for shape in shapes:
        radius = float(shape.get("radius", max(float(shape.get("sx", 0)), float(shape.get("sy", 0))) / 2))
        bounds.extend([
            (float(shape["x"]) - radius, float(shape["y"]) - radius),
            (float(shape["x"]) + radius, float(shape["y"]) + radius),
        ])
    minimum_x, maximum_x = min(x for x, _ in bounds), max(x for x, _ in bounds)
    minimum_y, maximum_y = min(y for _, y in bounds), max(y for _, y in bounds)
    width, height, padding = 960, 560, 34
    scale = min((width - 2 * padding) / (maximum_x - minimum_x), (height - 2 * padding) / (maximum_y - minimum_y))

    def point(x: float, y: float) -> tuple[float, float]:
        return padding + (x - minimum_x) * scale, height - padding - (y - minimum_y) * scale

    image = Image.new("RGB", (width, height), "#151922")
    draw = ImageDraw.Draw(image, "RGBA")
    for shape in sorted(shapes, key=lambda item: float(item["z"])):
        color = tuple(shape["color"])
        if shape["type"] == "circle":
            cx, cy = point(float(shape["x"]), float(shape["y"]))
            radius = max(1.0, float(shape["radius"]) * scale)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(255, 255, 255, 35), width=1)
            continue
        x, y, sx, sy, yaw = (float(shape[key]) for key in ("x", "y", "sx", "sy", "yaw"))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        polygon = []
        for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
            polygon.append(point(x + dx * cosine - dy * sine, y + dx * sine + dy * cosine))
        draw.polygon(polygon, fill=color, outline=(255, 255, 255, 28))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "WEBP", quality=88, method=6)


def _fit_source_image(source: Path, output: Path) -> None:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = ImageOps.autocontrast(image, cutoff=0.5)
        image = ImageEnhance.Contrast(image).enhance(1.05)
        canvas = ImageOps.fit(image, (960, 560), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, "WEBP", quality=88, method=6)


def _preview_world(model: str, distance: float) -> str:
    horizontal = distance * 0.82
    vertical = distance * 0.78
    pitch = math.atan2(vertical - 0.45, math.sqrt(2) * horizontal)
    return f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="preview">
  <scene><ambient>0.72 0.72 0.76 1</ambient><background>0.075 0.065 0.11 1</background></scene>
  <light name="sun" type="directional"><direction>-0.4 0.3 -1</direction><diffuse>0.95 0.93 1 1</diffuse></light>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <model name="ground"><static>true</static><pose>0 0 -0.035 0 0 0</pose><link name="link"><visual name="visual"><geometry><cylinder><radius>3.8</radius><length>0.06</length></cylinder></geometry><material><ambient>0.16 0.13 0.22 1</ambient><diffuse>0.16 0.13 0.22 1</diffuse></material></visual></link></model>
  <include><uri>model://{model}</uri><name>preview_vehicle</name><pose>0 0 0.22 0 0 -0.3</pose><static>true</static></include>
  <model name="preview_camera"><static>true</static><pose>{horizontal} {-horizontal} {vertical} 0 {pitch} 2.35619449</pose><link name="link"><sensor name="camera" type="camera"><topic>/dronedream/preview/{model}</topic><always_on>true</always_on><update_rate>8</update_rate><camera><horizontal_fov>0.95</horizontal_fov><image><width>960</width><height>560</height><format>R8G8B8</format></image><clip><near>0.05</near><far>30</far></clip></camera></sensor></link></model>
</world></sdf>"""


def _save_gazebo_image(message: object, output: Path) -> None:
    width, height, step = int(message.width), int(message.height), int(message.step)
    data = bytes(message.data)
    channels = max(1, step // width)
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
    if mode is None:
        raise RuntimeError(f"UNSUPPORTED_GAZEBO_IMAGE_STEP:{step}:{width}")
    packed = b"".join(data[row * step:row * step + width * channels] for row in range(height))
    image = Image.frombytes(mode, (width, height), packed).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "WEBP", quality=88, method=6)


def _capture_vehicle(model_root: Path, model: str, output: Path, distance: float) -> None:
    try:
        from gz.msgs10.image_pb2 import Image as GazeboImage
        from gz.transport13 import Node
    except ModuleNotFoundError as error:
        raise RuntimeError("GAZEBO_PYTHON_TRANSPORT_REQUIRED") from error
    import tempfile

    process: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryDirectory(prefix="dronedream-preview-") as temporary_name:
        world = Path(temporary_name) / "world.sdf"
        world.write_text(_preview_world(model, distance), encoding="utf-8")
        environment = dict(os.environ)
        environment["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(filter(None, [str(model_root), environment.get("GZ_SIM_RESOURCE_PATH", "")]))
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        process = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "--headless-rendering", str(world)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            node, deadline = Node(), time.monotonic() + 35
            topic = f"/dronedream/preview/{model}"
            while time.monotonic() < deadline and process.poll() is None:
                if topic in [str(value) for value in node.topic_list()]:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"VEHICLE_PREVIEW_TOPIC_MISSING:{model}")
            received, frame, count = threading.Event(), {}, 0

            def on_image(message: object) -> None:
                nonlocal count
                count += 1
                if count >= 3 and not received.is_set():
                    frame["message"] = message
                    received.set()

            node.subscribe(GazeboImage, topic, on_image)
            if not received.wait(max(0.1, deadline - time.monotonic())):
                raise RuntimeError(f"VEHICLE_PREVIEW_FRAME_MISSING:{model}")
            _save_gazebo_image(frame["message"], output)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-source-root", type=Path, required=True)
    parser.add_argument("--vehicle-model-root", type=Path, required=True)
    parser.add_argument("--school-world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-vehicles", action="store_true")
    args = parser.parse_args()

    map_output = args.output / "maps"
    _school_topdown(args.school_world, map_output / "school-map.webp")
    for key, (directory, filename) in MAP_SOURCES.items():
        source = args.map_source_root / directory / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        _fit_source_image(source, map_output / f"{key}.webp")

    if args.capture_vehicles:
        for key, model in VEHICLE_MODELS.items():
            distance = 2.8 if model in {"standard_vtol", "tiltrotor", "quadtailsitter"} else 1.55
            _capture_vehicle(args.vehicle_model_root, model, args.output / "vehicles" / f"{key}.webp", distance)
    previews = sorted(args.output.rglob("*.webp"))
    expected_count = len(MAP_SOURCES) + 1 + len(VEHICLE_MODELS)
    if len(previews) != expected_count:
        raise RuntimeError(f"RESOURCE_PREVIEW_COUNT_MISMATCH:{len(previews)}:{expected_count}")
    manifest = {
        "schema_version": "dronedream.resource-card-previews.v1",
        "generator": "scripts/generate_resource_card_previews.py",
        "sources": {
            "maps": {
                "project": "open-rmf/rmf_demos",
                "commit": "7851a5792d19a037833292a3e2a823b0f9e0c111",
                "license": "Apache-2.0",
            },
            "school_map": {"project": "DroneDream", "license": "MIT"},
            "vehicles": {
                "project": "PX4/PX4-gazebo-models",
                "commit": "5577035667afb4b63fe1f966fb1a58bbb05d905b",
                "license": "BSD-3-Clause",
            },
        },
        "files": [
            {
                "path": preview.relative_to(args.output).as_posix(),
                "bytes": preview.stat().st_size,
                "sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
            }
            for preview in previews
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
