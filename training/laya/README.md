# Laya UAV training toolkit

This is a preparation toolkit, **not a flight-qualified model or a software update**.
It neither rents a GPU nor accesses the user's database. Local experiment results
are not uploaded automatically.

## Actual implementation

- `DecisionStateV2`: strict causal state, unknown values, independent source ages,
  ENU/FRD conversions, history, goal/route/map/vehicle/calibration identities.
- `collect_decision_replay.py`: extracts unlabelled candidates from existing
  hash-bound navigation snapshots. Goal-aligned geometry sectors are **not**
  relabelled as body-aligned clearances.
- `audit_decision_candidates.py`: reports actual state/source coverage. It does
  not label data or count candidates as formal training examples.
- `assemble_decision_corpus.py`: independently rechecks native command delivery,
  geometric/telemetry outcomes and action-specific results before making a corpus.
  Missing results are quarantined. It is **not a native scenario collector**.
- `decision_dataset.py`: original file verification, strict JSON, no split/layout/
  input leakage, licenses, labels, group coverage and non-smoke admission.
- `train_laya_uav.py`: real Laya head and encoder optimization, accumulation,
  clipping, CUDA BF16 where supported, reproducible checkpoint continuation,
  development-set selection, independent temperature calibration, export/reload.
- `train_decision_student.py`: separate small MLP baseline using causal numeric
  features and missing masks. This is **not Laya**, nor an already-distilled or
  qualified aircraft controller.
- `LocalStageDecisionPort`: bounded, nonblocking subprocess suggestions. Queue
  replacement, expiry, goal identity, malformed outputs and process failure are
  checked. `execution_authority` is always false. Advice TTL is measured from
  submission, **not a replacement for original sensor freshness checks**.
- `decision_pipeline.py`: allowlisted dependency-closed code bundle with isolated
  extraction tests. Its current preparation report does **not** authorize a GPU
  rental or deployment; native collection readiness is still unfinished.

## Native collection input

The assembler accepts a JSON object:

```json
{
  "schema_version": "dronedream.decision-collection-manifest.v1",
  "entries": [
    {
      "candidate": {"path": "candidate.json", "sha256": "<actual SHA256>"},
      "outcomes": [{"path": "outcome.json", "sha256": "<actual SHA256>"}],
      "parent_group": "<independent base layout identity>",
      "episode_id": "<native episode identity>",
      "split": "train",
      "license": "proprietary-user-owned"
    }
  ]
}
```

Paths are portable relative paths under the manifest directory. Source candidates
must contain the recorded v2 state, original source hash, identity and coverage
status. The native outcome format in `decision_label_evidence.py` requires the
actual command/application sequence plus continuous independent native witnesses,
geometry and vehicle envelope. An arbitrary `success: true` does not qualify.
`slow_down` needs applied speed-limit evidence; `wait` needs verified subsequent
resumption; missing-observation recovery requires a new source; static replanning
requires a checked replacement path. Dynamic replan supervision is not yet
supported by that verifier and is explicitly rejected.

An acceptable-action set may contain multiple **individually verified** actions.
Do not fabricate alternative rollouts or label an unexecuted candidate from a
controller's prediction. Synthetic fixtures in tests are never copied into the
production corpus.

The data gate expects 30,000 primary formal windows **plus** 3,000 independent
stress windows, 1,000 episodes and 20 independent layout groups. It also checks
per-split action coverage, independent groups, train balance and unique labels
needed for class/calibration metrics. These are initial plan targets, not proof
that a trained model will meet flight requirements.

## Portable commands

Run from the extracted toolkit root with Python 3.11 and `PYTHONPATH=src`.
Use new output directories; no command overwrites previous experiments.

```bash
python scripts/assemble_decision_corpus.py --manifest collection/manifest.json --output corpus
python scripts/train_decision_student.py --corpus corpus/decisions.jsonl --output student
python scripts/train_laya_uav.py --corpus corpus/decisions.jsonl --model models/laya \
  --output run --device cuda --epochs 8 --micro-batch 1 --effective-batch 32

# Optional matched-data distillation; never replaces independently verified labels.
python scripts/train_decision_student.py --corpus corpus/decisions.jsonl --output student-distilled \
  --teacher-package run/model --teacher-identity run/training-identity.json --teacher-device cuda
```

For explicit synthetic interface checks only:

```bash
python scripts/prepare_decision_contract_smoke.py --output contract-smoke
python scripts/train_laya_uav.py --corpus contract-smoke/contract-smoke.jsonl \
  --model models/laya --output smoke --smoke --epochs 2 --micro-batch 1 --effective-batch 10
```

`--smoke` cannot produce a formal training or flight qualification receipt.
Real-data training rejects incomplete corpora before starting an optimizer.
Do not use smoke results to claim drone decision accuracy.

Training now permutes option order deterministically per example/epoch and maps
acceptable-action masks into the same order. Evaluation/calibration retain the
canonical action order. `--no-option-order-augmentation` is an explicit ablation,
recorded in the resume identity; it cannot silently change during continuation.
Distillation uses 80% verified-label set loss and 20% teacher KL as an initial
experimental setting, not a proven optimum. Teacher targets are generated only
for the training split, saved with a content hash, and checked against identical
corpus/action/schema/run identities. Development selection and independent
calibration remain unchanged. Teacher disagreements are reported, not used to
rewrite ground-truth labels. Compare this candidate against the supervised MLP;
do not assume distillation improves it.

`NativeDecisionCapture` provides bounded `accepted` and independent `witness`
callbacks. It freezes one causal input and one explicit selected behavior, records
actual transport acceptance separately from physical results, and quarantines
invalid captures without sending any flight command. Its `finish` verifier runs
off the control thread. These callbacks have contract tests but are **not yet
wired into a complete five-behavior native collection runner**. Do not infer that
having a recorder class means the formal demonstrations have been collected.

The Linux installer requires a **new**, experiment-owned `VENV_PATH`, a verified
official `CUDA_WHEEL_INDEX`, and optionally `BOOTSTRAP_PYTHON` selecting Python
3.11. It does not modify the installed product Runtime. `run.sh` supports
`RESUME_CHECKPOINT`; data/model/code/config/dependency identity must match the
original run. CUDA requests fail explicitly when CUDA is unavailable.

## Still required before a justified cloud-training handoff

1. A real native behavior collector producing all five action/result classes,
   causal local-route/dynamic-obstacle evidence and recovery outcomes.
2. Independently grouped scenario collection and accepted formal corpus, with
   frozen calibration/test/stress sets. Existing replay snapshots are insufficient.
3. Linux/CUDA dependency and resource qualification, real sequence memory/throughput
   profiling, and a bounded cost decision. CPU latency alone is not a GPU rental
   specification.
4. Following training: same-data rule/Laya/student comparisons, relevant robustness
   tests, actual safe control integration and full mission evaluation. Offline
   classifier export is not installation or flight acceptance.

The original complete implementation plan remains in
`docs/LAYA_UAV_COMPLETE_IMPLEMENTATION_PLAN.md`; this README does not mark its
unfinished items as complete or weaken those acceptance requirements.
