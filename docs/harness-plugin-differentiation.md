# Harness plugin differentiation ledger

This ledger is the admission gate for new first-party Harness plugins. A plugin is not
accepted because it increases the catalog count. It must own a decision axis that is
materially different from the nearest existing implementation, identify missions where
that difference matters, and carry deterministic plus real-model/simulation evidence.

## Current differentiated set

| Plugin | Nearest existing plugin | Distinct decision axis | Primary mission use | Evidence gate |
| --- | --- | --- | --- | --- |
| `planning.candidate-distance` | clearance, energy, and stability route candidates | Pure route length, intentionally ignoring secondary objectives, so ranking can quantify the cost of safer alternatives | evaluation, regression, time-sensitive routing | deterministic route-candidate test; DeepSeek Flash planning snapshot |
| `runtime.checkpoint-risk-adaptive` | every-segment and mission-boundary checkpoints | Inserts internal checks at sharp turns, tight clearance, and unusually long segments while retaining every segment endpoint | indoor flight, payload delivery, inspection, emergency response | geometry test for internal turn/tight checkpoints; runtime mission evidence |
| `validation.clearance-speed-coupling` | global track-stability and energy gates | Rejects only the local combination of narrow clearance and excessive segment speed instead of imposing a global speed cap | indoor, inspection, payload, field-readiness | tight-fast reject/tight-slow accept test; Gazebo narrow-passage run |
| `runtime.replan-mission-continuity` | nearest-anchor and verified-edge anchor policies | Rejects dead-end anchors and jointly scores join distance, reachability to the active target, and return continuity | payload, field-readiness, inspection, emergency amendment | dead-end exclusion test; interrupted Gazebo mission |
| `actions.payload-transition-guards` | delivery identity checks and detachable-joint pickup | Owns two state transitions that neither identity nor actuation can prove: stable separation before contact, then hash-bound payload physics and custody before return authorization | payload pickup and return | negative runtime tests for early contact/detached custody; DeepSeek Flash task graph; Gazebo payload run |
| `runtime-actions.payload-transition-guards` | ROS 2 domain services and Gazebo attach/detach commands | Combines PX4 position/velocity convergence with non-commanding Gazebo joint-state readback; it does not scan identity or issue the attach command | payload pickup and return | adapter contract schema; precontact and custody runtime receipts |
| `actions.loaded-flight-stability` | payload custody confirmation and generic track stability | Adds an explicit post-attachment action between custody and return; it cannot be satisfied by a successful attach acknowledgement or by pre-payload flight stability | loaded payload return, transport, and delivery | task-graph ordering tests; DeepSeek Flash payload plan; loaded Gazebo hover receipt |
| `runtime-actions.loaded-flight-stability` | payload transition readback and generic PX4 anomaly detection | Requires a time-bounded, attached-state hover with simultaneous position-error and speed convergence before it emits return authority | loaded payload return, transport, and delivery | negative detached/unstable tests; direct Gazebo state readback; real PX4 loaded-hover gate |

## Profile combinations

| Harness profile | Checkpoint policy | Replan policy | Local clearance-speed gate | Reason |
| --- | --- | --- | --- | --- |
| Balanced | every segment | nearest anchor | off | predictable baseline |
| Indoor Guardian | risk adaptive | verified anchor | off | dense geometry checks plus previously flight-verified joins |
| Payload Delivery | risk adaptive | mission continuity | on | preserve pickup/return obligations through narrow segments and replans |
| Evaluation Lab | every segment | nearest anchor | off | stable reproducible comparison baseline |
| Field Readiness | risk adaptive | mission continuity | on | expose local clearance/speed and return-continuity failures before real-device review |
| Infrastructure Inspection | risk adaptive | mission continuity | on | protect stable viewpoints and evidence coverage after replans |
| Area Survey | mission boundaries | nearest anchor | off | avoid excessive checks on long open-area coverage tracks |
| Emergency Response | risk adaptive | mission continuity | on | minimize unsafe shortcuts while retaining active objective and return feasibility |
| Plugin Developer | every segment | nearest anchor | off | deterministic integration baseline |
| Privacy Sensitive | every segment | nearest anchor | off | privacy boundaries are dominant; runtime policy remains conservative and simple |

## Admission questions

Before adding another plugin, answer all of these in its change and tests:

1. Which existing plugin is most similar?
2. What independent decision variable does the new plugin control?
3. Which concrete failure cannot the existing plugin handle without becoming less
   coherent for its original users?
4. Which profile or explicit user choice should activate it, and which profiles should
   not?
5. What deterministic counterexample proves the difference?
6. What DeepSeek Flash planning or Gazebo/PX4 run proves that the difference survives
   the real Harness boundary?

If two plugins produce the same choice on every available counterexample, they should
be merged or one should be rejected.

## Real payload acceptance

The current positive acceptance is bound into the default map/vehicle DDPKG pair under
`app/desktop/src-tauri/resources/default-assets`.
DeepSeek Flash produced the pluginized payload plan and the real School Map Gazebo/PX4
run completed with all eight runtime checkpoints authorized, executor return code zero,
PX4 landing state `ON_GROUND`, completion review accepted, and a minimum goal distance
of 0.0227 m. The pickup receipt bound the vehicle-frame mount target to the exact vehicle
and payload SDFs, measured 0.000397 m attachment alignment error against the 0.02 m hard
limit, and read back the attached joint state. The separate loaded-stability action then
observed 0.0707 m position error and 0.0193 m/s speed while the payload remained attached
before granting return authority. This closes the earlier false-positive gap where an
attach acknowledgement could be followed by an unstable rigid-body return.
