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
the dated `kumpula-campus-release-verification` evidence directory under
`Q:/DroneDream-Workspace/TestRuns`
and the previous index as `--supersedes-index`. Existing flight evidence applies
to that exercise only; no added edge is marked flight-verified.

Source endpoint graph: software-agent
`backend/app/autonomy/assets/kumpula-campus/navigation-graph.json` from commit
`d90bbb27502ea4d1be75466221e1a7d40a1440c4`. Building aliases come from the packaged
semantic catalog. Reproduction and per-endpoint collision checks are in
`Q:/DroneDream-Workspace/TestRuns/Natural-Language-E2E-20261013/check-map-destinations.py`
and `map-destinations/geometry-check.json`.

New qualification binding: `asset-qualification-daa2b60cd02725d65b803eba`.
Supersession preserves previously installed versions and mission evidence.
Installed desktop mission checks for the additional destinations are recorded
separately in the same end-to-end evidence directory.

The packaged vehicle range is additionally bound to a 596.5362349350833 m
Gazebo/PX4 full round trip from the launch point to the Chemicum handoff and
back. The run completed with no collision or abort, landed `ON_GROUND`, and
passed every operational qualification gate. It extends the advertised range
only; payload handling remains bound to the original pickup-return evidence.

## Whole-campus geographic review, later on 2026-10-13

The four-name-only graph was insufficient. `build_map_handoffs.py` now audits
all 41 city building records and 10 facility identities, preserving their public
source references. All 41 buildings have exterior simulation handoffs with
collision-free 3-D routes checked for the actual X500 Depth dimensions. Shared
municipal footprints do not merge Physicum, Exactum and the campus library.

The previous Physicum and Exactum positions were flyable but their straight
connections to their own doors crossed the building footprint. Their replacement
positions are `(22.218, 68.372, 2)` and `(-93.680, -24.274, 2)`. Chemicum and
Dynamicum retain their geographically valid positions. Adjacent collision boxes
must be considered as a union: this exposes valid exterior spaces for annexes
29466 and 79952 that sampling each box independently missed.

Recorded doors and facade-only handoffs remain distinct. A generated location
does not invent a surveyed door, permission, real parcel inventory, or a passed
flight. Old unverified walkway candidates are retained for source audit but are
no longer offered to the model as actionable alternatives. Duplicate addresses
remain ambiguous. A genuinely unbound target returns a natural clarification,
not an internal preparation failure.

The revised graph has 49 nodes and 177 aliases. World geometry, controller,
vehicle and the historical qualification exercise are unchanged. New package
binding: `asset-qualification-e237228f40bdc1c2c50ea411`. `handoff-index-qualified.json`
contains every point's building/door relationship and route-clearance receipt;
real desktop flight evidence is recorded separately.
