# Default map and aircraft pairing matrix

The product evaluates every combination of its eight built-in maps and thirteen
reviewed aircraft. The authoritative machine-readable view is returned by
`GET /v1/asset-pair-catalog` and can be exported with
`scripts/export_default_asset_pair_catalog.py`.

The 104 combinations have three states:

- `qualified_builtin` (8): the exact normalized built-in My Drone / PX4
  `x500_depth` package has completed the PX4/Gazebo qualification gates on that
  map. Every record includes its immutable qualification ID.
- `compatible_requires_qualification` (78): the geometry and operating class
  are compatible, but the exact map package, aircraft package, PX4 airframe,
  sensor setup, route, and runtime still need a closed-loop qualification run.
- `incompatible` (18): the three VTOL aircraft are excluded from the six maps
  dominated by rooms, doors, stairs, multilevel interiors, or tight corridors.
  They remain compatible with Campus Site and Open Test Arena.

Compatibility never grants execution authority. A compatible combination is
kept available for packaging and qualification instead of being hidden, while
mission execution still requires an exact current receipt. This prevents an
unverified model alias or a catalog review from being mistaken for flight
evidence.
