"""Version the behavior demonstrated, independently of sensor tensor shape.

Changing the demonstrator's gaze, control units or recovery behavior requires a
new semantic identity. Old recordings remain evidence, not current supervision.
"""

from .hashing import sha256_json

SIMULATION_TEACHER_CONTRACT = {
    'schema_version': 'dronedream.simulation-teacher-contract.v1',
    'translation': 'route teacher with live safety approval; actual velocity-ned receipts only',
    'heading_target': 'nearest measured 3D center minus max(radius,height/2), stable id tie; else semantic goal',
    'heading_stale': 'any track age >0.25s or confidence <=0 requests zero yaw',
    'heading_geometry': 'world ENU horizontal displacement rotated to FRU; body horizontal norm <0.1m is undefined',
    'heading_command': 'clockwise degree angle divided by 1 second, clipped to active yaw limit <=45 deg/s',
    'state_consistency': 'teacher pose and instantaneous flight encoding share measured state; partial refresh preserves oldest deadline and history slot',
    'recovery': 'only independently acknowledged pure-velocity commands, never position labels',
    'authority': 'simulation teacher, never learned-model flight qualification',
}
SIMULATION_TEACHER_CONTRACT_SHA256 = sha256_json(SIMULATION_TEACHER_CONTRACT)
