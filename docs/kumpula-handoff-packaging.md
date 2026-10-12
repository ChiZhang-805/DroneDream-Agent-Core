# Kumpula handoff graph packaging — 2026-10-13

The desktop default package omitted three outdoor destination nodes that already
existed in the product map generator. Copy those geometry-derived endpoints into
the shipped graph and bind the corresponding building names and aliases. Building
centres are not flight targets. These are simulated outdoor handoff positions,
not surveyed real-world entrances.

The current world, collision geometry, semantic bytes, vehicle and controller
parameters are unchanged. The independent pair qualification exercise is exactly
unchanged (390.10988842807024 m). Rebuilt with `scripts/build-default-assets.py`
using the hash-matching source flight in
`Q:/DroneDream-Workspace/TestRuns/kumpula-campus-release-verification-20261012-r1`
and the previous index as `--supersedes-index`. Existing flight evidence applies
to that exercise only; no added edge is marked flight-verified.

Source endpoint graph: software-agent
`backend/app/autonomy/assets/kumpula-campus/navigation-graph.json` from commit
`d90bbb27502ea4d1be75466221e1a7d40a1440c4`. Building aliases come from the packaged
semantic catalog. Reproduction and per-endpoint collision checks are in
`Q:/DroneDream-Workspace/TestRuns/Natural-Language-E2E-20261013/check-map-destinations.py`
and `map-destinations/geometry-check.json`.

New qualification binding: `asset-qualification-57aa3819b5591401b2a92387`.
Supersession preserves previously installed versions and mission evidence.
Installed desktop mission checks for the additional destinations are recorded
separately in the same end-to-end evidence directory.
