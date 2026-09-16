# DroneDream Flight Agent Core

MIT-licensed source for DroneDream's natural-language flight-agent core.
It deliberately contains the orchestration, structured contracts, safety gates,
ROS 2 integration, PX4/Gazebo execution adapter, and acceptance tooling without
copying the public product shell.

The first-party code permits modification, redistribution, and commercial use.
Third-party software, model weights, and imported assets retain their own terms;
see [Third-party notices](THIRD_PARTY_NOTICES.md) and
[Open-source publication boundaries](docs/open-source-publication.md).
Production databases, user conversations, credentials, and private training or
experiment records are not part of the source-code license or public export.

Public source: [DroneDream-Agent-Core](https://github.com/ChiZhang-805/DroneDream-Agent-Core).
The clean public repository starts from the current source, not private incubation
history. School Map and My Drone are included with an explicit
[MIT asset grant](docs/default-assets-license.md); no pretrained model weights or
production account data are included. Historical evidence references below describe
development milestones, not evidence bundled in this public snapshot or approval
of the current continuous-control candidate.

This is not a toy simulator and it has no mock model path. The verified milestone
uses real provider calls, ROS 2 Jazzy, `ros_gz_bridge`, Gazebo Sim physics, PX4 SITL,
and MAVSDK offboard control against the qualified School Map world.

## What is implemented

- Multi-call workflow: intent extraction, intent critique, optional-plugin routing, task
  decomposition, route planning, plan critique, runtime checkpoint assessment, and
  completion review.
- Hybrid continuous flight intelligence: a cloud model owns semantic intent, global-map
  reasoning, mission amendments, and completion review; compact local ONNX models propose
  millisecond-scale navigation and risk decisions between cloud calls; the Harness binds
  both to qualified tools, observations, deterministic safety, and evidence.
- Strict Pydantic input/output at every model, tool, process, and evidence boundary.
- Bounded propose/critique/repair loops for intent and planning.
- Two-tier plugin system: 222 first-party implementations across 90 typed slots and
  10 atomic persona profiles, plus transactional user ZIP/MCP tools and certified ROS 2
  `pluginlib`/Lifecycle capabilities. Harness hooks cover prompts, context, planning,
  tools, runtime decisions, simulation campaigns, evidence, evaluation, and staged asset
  conversion with schemas, permissions, dependency guards, hot-swap policies, rollback,
  quarantine, and task-bound plugin snapshots.
- Durable conversation and mission context with bounded compaction.
- Account-scoped memory projection through Supabase RLS with a durable local outbox,
  explicit cloud consent, revision/hash conflict detection, and local safety revalidation.
- Stable task-thread identity with parent-linked plan revisions and a unique execution ID
  for every confirmed launch.
- Deterministic safety gates that a model cannot bypass.
- Segment-boundary runtime checkpoints while PX4 holds position in the real simulator.
- Execution-time user-message ingress inside the 20 Hz PX4 loop: freeze the old track,
  inhibit semantic side effects, establish a telemetry-gated hover, then call the model.
- Append-only hash-linked evidence and immutable-run re-verification.
- Explicit mission-contract confirmation before any execution process is launched.
- Evidence-bound default `School Map` and `My Drone` bundles that seed the desktop map
  and vehicle repositories idempotently on first launch.

Cloud and local models control different time scales of the mission. The cloud model
owns natural-language intent, the coarse route, amendments, and semantic review. Current
local policies directly request normalized forward/right/up velocity and yaw-rate axes;
they do not output world coordinates, motor PWM, or unbounded attitude. The 20 Hz
offboard loop, coordinate conversion, collision prediction, acceleration/jerk limiting,
short leases, and safety exits remain deterministic. Local-model failure, ambiguity,
excessive latency, stale perception, or a broken content binding produces a hold or
controlled landing rather than unbounded motion.

### Hybrid local-policy development status

The dependency-ordered implementation ledger is maintained in
[the autonomy architecture master plan](docs/AUTONOMY_ARCHITECTURE_MASTER_PLAN.md).
[The earlier continuous-control baseline](docs/CONTINUOUS_CONTROL_UPGRADE.md) records the starting evidence.
The current direct-control candidate has nine ONNX artifacts and has not passed
its closed-loop flight trial. Historical candidate-ranking successes below do
not qualify the continuous-control weights. The baseline also identifies missing
yaw and hazardous-state training coverage, specialist admission gaps, and runtime
feature-contract differences that must be resolved before release.

The repository now contains the real package, training, admission, selection, Runtime,
and closed-loop evidence path for compact local policies:

- fixed metric-state, calibrated sensor tensors, missing-data masks, and an egocentric
  coarse-route reference prevent a model from inventing a world coordinate;
- each live, hash-bound continuous decision directly provides a forward-right-up velocity
  and bounded yaw-rate request; the request carries its source
  expert, model call, path hash, acceleration/jerk limits, and expiry, while deterministic
  safety can explicitly override its direction or brake;
- calibrated range, relative dynamic-target, and temporal flight-state encoders now gate
  that control request. They apply sensor extrinsics and fixed physical ranges, keep
  missing-value masks, use vehicle-relative closing speed/TTC, and refuse motion during
  stale data, insufficient field-of-view coverage, timing gaps, or history warm-up;
- independent action-conditioned risk training and inference evaluate the proposed
  physical four-axis request after Harness scaling; unbound or state-only historical
  critics cannot qualify a continuous-control package;
- a deterministic expert router selects the general navigation, precision-maneuver, or
  recovery role from current mission context; recovery is requested only for an explicit
  progress-stall or dynamic-obstacle incident, never from an executor lifecycle name alone;
  an absent specialist in continuous mode withholds motion rather than silently
  substituting the general policy; legacy observer fallback is explicitly separate;
- the three manoeuvre policies can use the shared causal GRU architecture with separate
  weights, original inter-observation timing, 16 independent observations and an immutable
  history contract. Offline BC/PPO, ONNX export, runtime admission and atomic ten-role
  composition share that contract; tests use synthetic weights, not product qualification;
- independent native telemetry publication/history does not depend on image rendering
  or a model finishing. Continuous commands expire with original required sensor evidence,
  and transport acceptance is joined back to the model input and approved command;
- every call verifies that the independent advisors requested by the Harness were actually
  executed by the inference backend; an advisory mismatch fails closed;
- every local call records the navigation expert's pre-advisor action and risk, each
  independent advisor risk score, the aggregate conservative risk, routing fallback, and
  controller scale in a typed evidence trace; qualification rejects missing or mismatched
  traces;
- optional perception-health, settle-stability, and payload-dynamics ONNX advisors have
  dedicated tensor contracts and independent training/export paths; perception can veto
  stale or incoherent sensing, while payload dynamics may only raise risk or shorten an
  already revalidated controller step; settle stability remains active for a precision
  flight profile even when the recovery specialist temporarily owns navigation, and payload
  dynamics is not invoked until runtime payload evidence exists;
- package manifests bind every model file to exact vehicle and sensor-contract hashes;
- the desktop Runtime discovers local packages only through its hash-covered resource
  catalog, runs them as the high-frequency primary provider, and retains the selected
  cloud provider for semantic work and hold-safe inference fallback; a signed installer
  can include a `simulation-only` package for PX4/Gazebo, while only a separately
  `production-qualified` catalog can authorize a real-flight package;
- exact-map specialists additionally bind the semantic-map hash and are selectable only
  with a separately qualified general policy available as rollback;
- offline held-out-campaign admission permits simulation only; production selection
  requires at least 20 distinct full Gazebo/PX4 trials, zero collisions, at least 95%
  success, bounded inference latency, and map-diversity evidence;
- advisor datasets split on complete missions rather than random rows, verify snapshot,
  cycle, and mission-evidence hashes, refuse to invent payload labels without recorded
  payload state and mass, and retain safe/risky class counts for each expert;
- navigation-specialist training additionally requires pre-advisor expert traces from
  verified, mission-separated campaigns. Legacy final ensemble decisions are rejected as
  specialist labels because they entangle navigation with risk, perception, and stability
  vetoes;
- training requires at least 20 safe and 20 risky held-out examples per selected advisor;
  a class with no observed failures cannot pass merely because recall is undefined;
- training never occurs during an active flight. Behavior cloning, distillation, or
  reinforcement learning happens offline, then exports a new immutable package that must
  repeat admission and closed-loop qualification.
- recovery training requires an explicit recovery-episode identity in every source
  snapshot. Progress stalls keep one identity until measured progress resumes; each
  continuous dynamic-obstacle encounter keeps one goal-bound identity across moving
  position signatures and ends only after a bounded quiet interval. Dataset admission then requires
  a material range reduction or semantic-goal advance after the recovery decision and
  keeps at most one sample per episode. Consecutive votes from one stall therefore cannot
  masquerade as independent successes, and an ordinary local replan without a recovery
  trigger is rejected as recovery supervision.

The body-frame control and real-time feature path is currently a source-validated
candidate: repository tests and static checks pass, and a repeatable 320-ray,
16-track, 16-state-frame benchmark records 4.222 ms combined P99. It has not yet been
rebuilt into the Runtime package or rerun through PX4/Gazebo, so it does not inherit the
existing package's simulation admission or any real-flight authority.

Independently trained navigation, precision-maneuver, risk, perception-health, and
settle-stability models have passed their applicable held-out offline gates. The
precision-maneuver expert was bootstrapped only from verified general-expert fallback
traces, with complete source missions kept on one side of the training/validation split.
Its simulation admission used 368 independent validation samples and retained the
independently trained advisors; admission grants no production authority.

A completely newer PX4/Gazebo mission then exercised the admitted package over an
11.78-metre round trip. The Harness made 111 local calls: the general expert handled 28,
the precision expert handled 83 directly, and no navigation fallback occurred. The
precision expert selected bounded legacy candidates 79 times and held four times; risk and
perception-health critics participated in every call, settle stability participated in
all 83 precision calls, P99 local inference was 4 ms, route-fallback schedule
advances remained zero, and the vehicle reached 0.03 m from the final goal before a
confirmed landing. A separate
development-only depth-frame-loss campaign produced 15 sensor-rate unhealthy cycles,
retained fail-closed control, recovered the stream, completed landing, and deliberately
failed the `development_fault_injection_absent` qualification gate. The content-bound
summary is in
[`evidence/local-policy-perception-risk-settle-short-qualification.json`](evidence/local-policy-perception-risk-settle-short-qualification.json).
The content-bound direct precision evidence is in
[`evidence/local-policy-precision-direct-qualification.json`](evidence/local-policy-precision-direct-qualification.json).
The current formal School Map payload round trip covered 413.21 metres and 73 route points
with live vehicle-mounted RGB/depth, 21,047 local decision cycles, 20,038 local-model
calls, 21,048 visual frames, and 18,755 applied authorized paths. All five native payload
and checkpoint actions were accepted; the detachable payload attached, loaded hover was
stable, the return was authorized, and PX4 landed `ON_GROUND` within 0.029 metres of the
goal. No local invocation, timeout, revalidation, cloud-runtime fallback, route-fallback
advance, abort, or qualification-gate failure occurred. All 36 evidence gates passed,
including continuous model participation, controlled-vehicle identity, writer completion,
live perception, schedule authority, ROS observations, PX4 ULog, and landing. The
content-bound full-run summary is
[`evidence/local-policy-full-depth-closed-loop-qualification.json`](evidence/local-policy-full-depth-closed-loop-qualification.json).
The package remains simulation-only: the direct precision mission proves integration, not
production qualification. Production still requires at least 20 distinct trials, at least
two maps, and the separate qualification receipt. The direct specialist traces from this
mission can seed future specialist training; older ensemble-only records remain invalid
specialist supervision.
The temporal anomaly expert now has a real eight-observation ONNX Runtime contract and
conservative risk-veto integration. Perception-health, settle-stability, and
payload-dynamics experts now have trainable compact-network contracts and execute in the
same conservative advisory vote. Qualification now reconstructs the exact expected expert
stack for every recorded cycle and rejects missing, extra, unlinked, or unexercised model
roles. The perception-health and settle-stability artifacts have integration evidence but
still need the complete repeated mission campaign before production qualification.
Recovery, payload dynamics, and temporal anomaly experts still require dedicated
successful campaigns. Precision navigation now has direct closed-loop integration
evidence but also remains unqualified for production until the complete repeated campaign
passes.

## Current verified milestone

The current recorded end-to-end acceptance is the runtime-interruption closed-loop
campaign. A real user-style Chinese request produced a five-call structured plan for `School Map` and
`My Drone`. During real PX4/Gazebo execution, a second user message cancelled the
destination and requested an immediate safe return to the third-floor office pad:

- the 20 Hz executor detected the message in 14 ms and established stable hover in
  1.099 s before the model call;
- DeepSeek classified the amendment in 25.028 s while deterministic code held position
  and inhibited old-plan side effects;
- code resolved the natural-language destination to `verified-000`, built a new
  continuously clearance-checked three-node route, and accepted it only after nine
  replacement gates and a hash-bound executor adoption receipt passed;
- the durable lifecycle automatically resumed from `holding` to `executing`, PX4 landed
  `ON_GROUND`, all nine runtime gates passed, the completion model accepted the evidence,
  and the task ended in `completed`;
- the 20,919,357-byte PX4 ULog and all compact evidence hashes are recorded in
  [`evidence/user-closed-loop-20260819.json`](evidence/user-closed-loop-20260819.json).

The earlier checkpoint-only acceptance remains useful historical evidence:

The recorded checkpoint acceptance evidence shows:

- 10 real DeepSeek calls: 5 planning, 4 in-flight checkpoints, 1 completion review;
- 4/4 checkpoint decisions accepted and independently authorized by code;
- 19,685 Gazebo poses and 6,092 ROS observations;
- minimum goal distance 0.229 m and final PX4 state `ON_GROUND`;
- 91,268,217-byte PX4 ULog with a recorded SHA-256;
- all binding, process, timing, goal, landing, and safety gates passed.

See [`evidence/verified-simulation-run.json`](evidence/verified-simulation-run.json)
for the portable evidence manifest. Large generated run artifacts and flight logs stay
outside Git; the manifest binds their immutable hashes and locations.

The recorded negative-path runtime-interruption acceptance injected a real user message while
the aircraft was tracking. The executor detected it in 43 ms, reached stable hover in
1.617 s, held throughout a 28.272 s real DeepSeek classification, then landed with PX4
state `ON_GROUND`. The original mission was correctly recorded as failed/superseded, not
as a false success. See
[`evidence/runtime-interruption-acceptance.json`](evidence/runtime-interruption-acceptance.json).

## Repository map

```text
src/dronedream_agent_core/
  contracts.py       Versioned structured artifacts
  model_harness/     Model + Harness runtime package
    memory.py        Governed account and mission memory
    memory_projection.py RLS cloud projection and durable outbox
    boundary.py      Structured input/output and execution authority
    design.py        Visual Harness contracts and compiler
    graph.py         Typed bounded Harness execution graph
    model_port.py    Real OpenAI / DeepSeek / Kimi model boundary
  prompts.py         Bounded role prompts
  orchestrator.py    Multi-call planning state machine
  checkpointing.py   Live checkpoint coordinator and hard gates
  lifecycle.py       Stable task, plan-revision, and execution identities
  runtime_interrupt.py Runtime ingress and code-owned authorization
  execution.py       Contract-confirmed execution and completion review
  tools.py           Plugin registry, authority gates, and receipts
  plugin_contracts.py Versioned package, lifecycle, and snapshot contracts
  plugin_process.py  Isolated MCP stdio process boundary
  context.py         Bounded durable conversation context
  navigation.py      Generic graph routing
  collision.py       Static route-clearance validation
  px4_track.py       ENU-to-PX4 track construction
  gazebo_adapter.py  ROS 2 / Gazebo / PX4 lifecycle and evidence
  local_policy_packages.py Immutable local-model packages and qualification selection
  local_policy_port.py Fixed-tensor ONNX inference through the structured model boundary
  local_policy_training.py Reproducible navigation and independent risk-model training
  evidence.py        Append-only hash-linked ledger
ros_ws/              ROS 2 messages and the raw Gazebo observation node
official_plugins/    Source for separately packaged first-party MCP plugins
scripts/             PX4 checkpoint executor, verification, schema export
examples/            Natural-language mission requests
schemas/             Generated JSON Schema boundary definitions
docs/                Architecture and artifact-flow documentation
evidence/            Small, reviewable acceptance manifests
app/desktop/src-tauri/resources/default-assets/
                     Qualified School Map and My Drone seed bundles
tests/               Pure contract, navigation, evidence, and safety tests
```

The qualified world, semantic map, vehicle, controller configuration, navigation graph,
and qualification receipts are bundled into the desktop application as
`School Map` (`dronedream.school-map.v1`) and `My Drone`
(`dronedream.my-drone.v1`). The application validates each native DDPKG, its content
hashes, and the pair qualification receipt before an idempotent first-launch import.
The packages contain one current normalized representation and never embed the retired
mutable ZIP repository. They are evidence-bound release snapshots of the qualified
geometry and physics artifacts rather than untracked copies.

## Development

Python 3.11-3.13 is supported.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\python.exe scripts\export_schemas.py --check
```

Provider configuration is read from environment variables only. Copy `.env.example`
for the variable names, but do not commit secrets. Planning requires a real provider;
there is intentionally no offline fallback.

### Account-memory cloud projection

The desktop frontend forwards the current Supabase access token and the public project
key only across the authenticated loopback API. The sidecar derives the project URL from
the already verified JWT issuer and calls PostgREST with the real user JWT, so Supabase
RLS remains the cloud authority. A `service_role` JWT or `sb_secret_` key is explicitly
rejected and is never required, stored, or bundled by the projection bridge.

Local governed memory remains the runtime source of truth. Candidates, explicit
resolutions, forgets, permanent deletions, and consent changes enter an idempotent SQLite
outbox. A bounded sync runs before mission preparation; accepted cloud records are pulled
only after consent and pass through the same local secret/instruction/authority scrub
before retrieval. Network failures leave operations pending, configuration absence is
reported as `configuration_missing`, and stale revisions or same-revision hash changes are
reported as `conflict` without replacing local state. Changing `memory_enabled` is the only
desktop action that queues account cloud consent; the default setting never grants it.

Cloud writes require the public Supabase migrations through
`20260824060000_add_authenticated_memory_projection.sql`. Deployment of those migrations
and a live authenticated-user RLS check remain release-environment responsibilities; unit
tests do not substitute for that check.

Example preparation command:

```powershell
.venv\Scripts\dronedream-agent.exe prepare-mission `
  --provider deepseek `
  --model deepseek-v4-flash `
  --request examples\campus_gate_one_way.json `
  --graph <qualified-graph.json> `
  --semantic <qualified-semantic-map.json> `
  --vehicle-sdf <qualified-vehicle.sdf> `
  --output-dir artifacts\acceptance\campus-gate
```

Execution is intentionally a separate command and requires the exact generated
`contract_id` to be supplied again. Inspect `dronedream-agent --help` and
[`docs/artifact-flow.md`](docs/artifact-flow.md) before running it.

## Runtime verification

From the `DroneDreamRuntime` WSL distribution:

```bash
cd ros_ws
colcon build --symlink-install
cd ..
bash scripts/verify_ros2_dds.sh /tmp/dronedream-dds
bash scripts/verify_ros_gz_school_map.sh /path/to/world.physics.sdf /path/to/my-drone/model.sdf /tmp/dronedream-ros-gz
bash scripts/verify_ros_gz_raw_observer.sh /path/to/world.physics.sdf /path/to/my-drone/model.sdf /tmp/dronedream-observer
```

These checks exercise actual DDS discovery, the actual School Map world and vehicle,
Gazebo-to-ROS clock and dynamic-pose transport, and the repository's ROS observation node.
They are integration evidence, not substitutes for the full PX4 acceptance run.

## Boundaries

- Current scope is simulation. No physical-aircraft adapter is included yet.
- Scheduled model checkpoints remain at semantic segment boundaries. User-message
  detection and old-track inhibition now happen inside the 20 Hz control loop; model
  reasoning begins only after stable-hold evidence exists.
- Runtime destination amendments block old-plan resume, enter a bounded safe hold, resolve
  the new target against the selected map, generate a continuously clearance-checked
  replacement track from the observed hold position, and hot-swap it only after all identity
  and hash gates pass. Missing, late, ambiguous, or unsafe replacements land fail-closed.
- OpenAI, DeepSeek, and Kimi adapters are implemented; every provider still requires a
  valid account, key, and live verification in the target environment.
- Public source availability is separate from flight qualification and installer
  publication. Unit-test success never replaces closed-loop acceptance, current
  model admission, asset qualification, or release-signing approval.

Read [`docs/architecture.md`](docs/architecture.md) for the authority model and safety
exits before extending the runtime.
Read [`docs/plugin-system.md`](docs/plugin-system.md) for package, lifecycle, isolation,
rollback, Cordis-inspired ownership, and ROS 2 pluginlib details.
