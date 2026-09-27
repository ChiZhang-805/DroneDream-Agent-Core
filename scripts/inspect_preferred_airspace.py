"""Export and measure current package-bound 3D preferences without launching a vehicle."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from dronedream_agent_app.asset_runtime_resolver import (
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from dronedream_agent_core.asset_package_storage import extract_verified_asset
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.contracts import VehicleAsset
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.preferred_airspace import PreferredAirspace
from dronedream_agent_core.runtime_control_io import publish_runtime_json


# 功能：
#   校验并解包指定资产到新测试目录；不修改产品安装或历史资产。
# 输入：
#   package、root：明确的包和新目录。
# 输出：
#   record：解析器可读取的真实内容绑定。
def extract_record(package, root):
    inspected = inspect_ddpkg(package)
    root.mkdir()
    extract_verified_asset(package, root, inspected)
    record = {"manifest": inspected.manifest.model_dump(mode="json"),
              "asset_ir": inspected.asset_ir.model_dump(mode="json"), "bundle_root": str(root),
              "asset_id": inspected.manifest.asset_id, "kind": inspected.manifest.asset_kind,
              "content_sha256": inspected.manifest.content_sha256,
              "maturity": (inspected.manifest.qualification.maturity
                           if inspected.manifest.qualification is not None else "visual_only")}
    return record


# 功能：
#   从真实软件资产生成全图体积，独立批量检查膨胀后的体积与保守障碍是否相交。
# 输入：
#   参数：地图包、机型包和新输出目录。
# 输出：
#   exit_code：空间绑定、几何独立检查和查询计时均完成时为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-package", type=Path, required=True)
    parser.add_argument("--vehicle-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    map_record = extract_record(args.map_package, output / "map")
    vehicle_record = extract_record(args.vehicle_package, output / "vehicle")
    selected_map = resolve_versioned_map(map_record)
    selected_vehicle = resolve_versioned_vehicle(vehicle_record)
    raw = read_plugin_file(selected_map.semantic, limit=32 * 1024**2)
    semantic = json.loads(raw)
    vehicle = VehicleAsset.model_validate_json(read_plugin_file(
        selected_vehicle.vehicle_metadata, limit=65536))
    started = time.perf_counter()
    field = PreferredAirspace(semantic, {
        "map_asset_id": selected_map.asset_id,
        "map_content_sha256": selected_map.content_sha256,
        "vehicle_asset_id": selected_vehicle.asset_id,
        "vehicle_content_sha256": selected_vehicle.content_sha256,
    }, vehicle.body_radius_m, vehicle.body_height_m)
    snapshot = field.snapshot()
    build_seconds = time.perf_counter() - started
    if not snapshot["volumes"]:
        raise ValueError("AIRSPACE_NO_COVERED_PREFERRED_VOLUMES")
    bounds = np.asarray(snapshot["obstacles"], dtype=np.float64)
    volumes = np.asarray([v[:6] for v in snapshot["volumes"]], dtype=np.float64)
    reach = np.array([vehicle.body_radius_m + field.preferences.margin_m] * 2
                     + [vehicle.body_height_m / 2 + field.preferences.margin_m])
    intersections = 0
    for begin in range(0, len(volumes), 64):
        group = volumes[begin:begin + 64]
        low, high = group[:, :3] - group[:, 3:] / 2 - reach, group[:, :3] + group[:, 3:] / 2 + reach
        overlap = np.all((low[:, None, :] < bounds[None, :, 3:] - 1e-8)
                         & (high[:, None, :] > bounds[None, :, :3] + 1e-8), axis=2)
        intersections += int(overlap.sum())
    if intersections:
        raise ValueError("AIRSPACE_VOLUME_OBSTACLE_INTERSECTION:" + str(intersections))
    timings = []
    for i in range(3000):
        v = snapshot["volumes"][i % len(volumes)]
        started = time.perf_counter_ns()
        context = field.context(tuple(v[:3]))
        timings.append((time.perf_counter_ns() - started) / 1e6)
        if not context["available"]:
            raise ValueError("AIRSPACE_SNAPSHOT_QUERY_DISAGREEMENT")
    publish_runtime_json(output / "snapshot.json", snapshot, replace_existing=False)
    receipt = {"complete": True, "semantic_sha256": hashlib.sha256(raw).hexdigest(),
               "airspace_sha256": field.sha256, "volumes": len(volumes),
               "collision_bounds": len(bounds), "build_seconds": build_seconds,
               "query_p50_p95_p99_ms": np.percentile(timings, [50, 95, 99]).tolist(),
               "independent_box_intersections": intersections,
               "training_photos_added": 0, "flight_tested": False,
               "snapshot_sha256": snapshot["snapshot_sha256"]}
    publish_runtime_json(output / "receipt.json", receipt, replace_existing=False)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
