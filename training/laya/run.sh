#!/usr/bin/env bash
set -euo pipefail
# 功能：使用通过审计的真实领域语料微调，不允许在正式入口附加 smoke 参数。
# 输入：PYTHON、CORPUS、BASE_MODEL、OUTPUT、可选批次；输出：待仿真验收模型包。
: "${PYTHON:?Set virtual environment Python}"
: "${CORPUS:?Set audited corpus path}"
: "${BASE_MODEL:?Set pinned local Laya directory}"
: "${OUTPUT:?Set a new training output path}"
TASK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$TASK_ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RESUME_ARGS=()
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  RESUME_ARGS=(--resume "$RESUME_CHECKPOINT")
fi
"$PYTHON" "$TASK_ROOT/scripts/train_laya_uav.py" --corpus "$CORPUS" \
  --model "$BASE_MODEL" --output "$OUTPUT" --device cuda \
  --micro-batch "${MICRO_BATCH:-1}" --effective-batch "${EFFECTIVE_BATCH:-32}" --epochs 8 \
  "${RESUME_ARGS[@]}"
