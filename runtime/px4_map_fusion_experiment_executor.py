"""Explicit isolated SITL experiment; retains native landing and command safety."""

import importlib.util
import sys
from pathlib import Path

from dronedream_agent_core.map_fusion_experiment import MapFusionExperimentMixin

_spec = importlib.util.spec_from_file_location(
    "map_fusion_native_executor", Path(__file__).with_name("px4_offboard_track_executor.py")
)
_native = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _native
_spec.loader.exec_module(_native)
globals().update(
    {name: value for name, value in vars(_native).items() if not name.startswith("__")}
)


class MavsdkOffboardClient(MapFusionExperimentMixin, _native.MavsdkOffboardClient):
    """Starts owned observations before preflight, closes them only after native landing."""


if __name__ == "__main__":
    _native.MavsdkOffboardClient = MavsdkOffboardClient
    raise SystemExit(_native.main())
