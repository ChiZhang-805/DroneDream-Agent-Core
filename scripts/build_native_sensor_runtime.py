"""Build and test the native simulator sensor plugin in an explicit staging root."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from dronedream_agent_core.simulation_sensor_runtime import MAGNETIC_SENSOR_CONTRACT_SHA256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--px4-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "native/gazebo_sensors"
    output = args.output.resolve()
    receipt = output / "native-sensor-runtime.json"
    if receipt.exists():
        raise FileExistsError("use a new build root; frozen receipts cannot be overwritten")
    def source_hashes():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source.iterdir()) if path.is_file()}

    original_sources = source_hashes()
    table = args.px4_root / "src/lib/world_magnetic_model/geo_magnetic_tables.hpp"
    original_table = hashlib.sha256(table.read_bytes()).hexdigest()
    subprocess.run(["cmake", "-S", str(source), "-B", str(output),
                    "-DCMAKE_BUILD_TYPE=Release", f"-DDRONEDREAM_PX4_ROOT={args.px4_root}"],
                   check=True)
    subprocess.run(["cmake", "--build", str(output), "-j", "2"], check=True)
    subprocess.run(["ctest", "--test-dir", str(output), "--output-on-failure"], check=True)
    library = output / "libdronedream-magnetometer.so"
    if (source_hashes() != original_sources or hashlib.sha256(table.read_bytes()).hexdigest()
            != original_table or hashlib.sha256(
                (output / "geo_magnetic_tables.hpp").read_bytes()).hexdigest() != original_table):
        raise RuntimeError("native sensor build inputs changed; rebuild in a new directory")
    result = {"wire_contract": "px4-gz-fimex-gauss", "native_tests_passed": True,
              "sensor_contract_sha256": MAGNETIC_SENSOR_CONTRACT_SHA256,
              "px4_magnetic_table_sha256": original_table,
              "field_probe_sha256": hashlib.sha256(
                  (output / "magnetic-field-probe").read_bytes()).hexdigest(),
              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "sources": original_sources}
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
