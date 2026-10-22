# DroneDream CLI and terminal acceptance

`dronedream` is the terminal interface to the same account, planning, asset,
Runtime, PX4/Gazebo, telemetry, and evidence services used by the desktop
application. `dronedream-console` remains an equivalent compatibility command.

The CLI is not a scripted flight demo. A valid mission starts with the
operator's natural-language request. Agent Core sends that request to the
selected managed model. The model calls the available map, topology, route,
trajectory, and validation tools; deterministic product gates then validate
the model's proposed plan before the operator may execute it. A route embedded
in CLI source code, a canned response, or a success message without matching
execution evidence fails acceptance.

## Session boundary

A trusted product launcher supplies the following values through inherited
environment variables or one bounded JSON line passed with `--session-stdin`:

- `DRONEDREAM_CONSOLE_CORE_URL`
- `DRONEDREAM_CONSOLE_SUPABASE_URL`
- `DRONEDREAM_CONSOLE_LOCAL_TOKEN`
- `DRONEDREAM_CONSOLE_IDENTITY_TOKEN`
- `DRONEDREAM_CONSOLE_PUBLISHABLE_KEY`
- `DRONEDREAM_CONSOLE_EDITION`

Secrets are never command-line arguments, persisted CLI settings, or terminal
output. The CLI does not scrape browser storage or search desktop databases for
credentials. Use `--json` for automation; an interactive terminal receives the
pink, purple, and blue product theme plus conventional green, red, yellow, and
blue status colors.

## Discover before flying

```text
dronedream doctor
dronedream models
dronedream assets maps
dronedream assets aircraft
dronedream assets pairs
dronedream runtime status
```

Catalog admission and execution qualification are different. The pair catalog
labels combinations as `qualified_builtin` or
`compatible_requires_qualification`. Only an exact qualified pair, including
its qualified package hashes, can enter execution. The current release-qualified
pair is Kumpula Campus with PX4 x500 depth; the remaining aircraft stay visible
for capability discovery and future qualification.

## Guided end-to-end flight

```text
dronedream mission fly \
  "Please collect the instruments from the chemistry building, deliver them, then return and land." \
  --model <managed-model-id> \
  --map dronedream-school-map \
  --aircraft px4-x500-depth
```

The guided command performs these product operations without bypasses:

1. Resolves the selected model, map, aircraft, qualification receipt, and exact
   qualified package hashes from live product catalogs.
2. Creates an account-bound task.
3. Sends the fuzzy request to the real managed model and allows the model to
   call the planning tools. If the intent is materially ambiguous, the CLI
   presents the model's clarification questions and resubmits the answers.
4. Displays the generated goal, task graph, tool-backed planning metadata, and
   immutable `plan-...` revision.
5. Requires the operator to type that exact plan revision. No automatic
   confirmation switch exists.
6. Starts the shared PX4/Gazebo executor, watches real task state and telemetry,
   then loads the execution evidence for the same task and execution.
7. Reports success only when the task completed and the evidence result is
   `verified`; a verified mission is archived unless `--no-archive` is used.

Use `--plan-only` when the operator wants the real model plan without starting
simulation.

## Explicit step-by-step workflow

The individual commands are useful for controlled acceptance and automation:

```text
dronedream mission create --model <model-id> --title "Chemistry delivery"
dronedream mission plan <thread-id> \
  --text "Take the instruments to the destination and come back." \
  --map <qualified-map-asset-id> --map-sha256 <exact-map-hash> \
  --aircraft <qualified-aircraft-asset-id> --aircraft-sha256 <exact-aircraft-hash>
dronedream mission show <thread-id>
dronedream mission execute <thread-id> --confirm-plan <reviewed-plan-id>
dronedream mission watch <thread-id>
dronedream mission telemetry <thread-id>
dronedream mission live-sources <thread-id>
dronedream mission frame <thread-id> --output <new-png-path>
dronedream mission evidence <thread-id>
dronedream mission archive <thread-id>
```

`mission plan` never starts a flight. `mission execute` rechecks identity,
current revision, assets, qualification, and runtime admission. If a write
times out, its outcome is unknown; inspect the task before attempting another
write.

## History and evidence

```text
dronedream history list
dronedream history list --archived-only
dronedream history show <thread-id>
dronedream history export <thread-id> --output <new-json-path>
```

Exports are redacted, hashed, and never overwrite an existing file. A full
acceptance record must tie the original natural-language message, model plan,
tool calls, confirmed plan revision, map and aircraft hashes, task ID,
execution ID, telemetry, gate results, payload actions, return, and landing to
one run. A downloaded frame is useful observation evidence but is never a
flight verdict by itself.

Acceptance requires a real model call, actual tool participation, a completed
PX4/Gazebo run, no failed safety gates, successful mission actions, and a safe
landing. HTTP success, a rendered route, attractive terminal output, or copied
evidence from a different run cannot satisfy those requirements.
