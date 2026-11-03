# Default map and aircraft pairing matrix

The desktop product shows six built-in maps and six built-in aircraft. The
authoritative release index is
`app/desktop/src-tauri/resources/default-assets/index.json`; the runtime catalog
is returned by `GET /v1/asset-pair-catalog`.

## Current release result

| Aircraft | Qualified maps | Product status |
| --- | ---: | --- |
| X500 Depth | 6/6 | available |
| X500 Front LiDAR | 6/6 | available |
| X500 Gimbal | 6/6 | available |
| Standard VTOL | 6/6 | available |
| Tiltrotor | 0/6 | unavailable |
| Quadtailsitter | 0/6 | unavailable |

The 24 available combinations use exact, immutable map and vehicle packages.
Each pair has a PX4 SITL and Gazebo Harmonic round trip, landing, zero unsafe
collisions, route and reference-track evidence, qualification-input hashes, and
an immutable qualification ID.

Physical testing produced one additional successful Quadtailsitter round trip
on a simplified, nonrelease Kumpula candidate. It does not qualify the current
release map and is deliberately not promoted. In addition, that vehicle package
does not provide the perception source required by the present autonomous
mission contract. A flight result alone cannot invent a camera, depth sensor, or
LiDAR capability.

Tiltrotor was armed and entered offboard control on the Vuosaari baseline but
did not lift. The remaining five maps were not promoted or presented as tested
successes after that vehicle-level blocker was established.

## Meaning of a qualified pair

Qualification is exact, not inherited from a model family or compatibility
guess. It binds the current map package, aircraft package, PX4 airframe, sensor
setup, route, reference track, Runtime inputs, and physical evidence. Geometry
compatibility keeps an unqualified pair eligible for future work but grants no
execution authority.

The current physical runs used deterministic route inputs and therefore record
`natural_language_acceptance: false`. Real DeepSeek planning acceptance was
attempted with a normal, deliberately imprecise user request, but the dedicated
`DEEPSEEK_API_KEY_DroneDream` account returned HTTP 402. That failed attempt is
retained as evidence and is not counted as natural-language acceptance.
