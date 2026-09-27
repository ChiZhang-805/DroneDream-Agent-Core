#!/usr/bin/env bash
# 功能：
#   核对实际安装代码和冻结数据后，在单卡上运行一个有界控制专家训练并验证 CPU ONNX 导出。
# 输入：
#   bundle、data、run、environment：上传代码、冻结数据、新作业目录与独立环境。
#   bundle_sha、data_sha：另存的代码和数据清单摘要。
#   role：明确的三个动作专家之一。
#   recipe：smoke、baseline、seed-806 或 seed-807。
#   encoder：可选的导航编码器目录，含原检查点及绑定的训练回执。
#   gpu_index：可选的单张物理 GPU 编号，默认 0；独立作业可明确选择另一张卡。
# 输出：
#   run：候选权重、来源回执和真实导出验证；不读取最终测试预测或替换产品模型。
set -euo pipefail
unset PYTHONPATH PYTHONHOME
bundle_root="$(cd -- "${1:?bundle}" && pwd)"
data_root="$(cd -- "${2:?data}" && pwd)"
run_root="${3:?new job directory}"
environment_root="$(cd -- "${4:?environment}" && pwd)"
bundle_sha="${5:?separately saved bundle manifest sha256}"
data_sha="${6:?separately saved data assembly sha256}"
role="${7:?expert role}"
recipe="${8:?recipe}"
encoder_root="${9:-}"
gpu_index="${10:-0}"
case "$role" in local-navigation-policy|precision-maneuver-policy|recovery-policy) ;; *) exit 2 ;; esac
case "$recipe" in smoke|baseline|seed-806|seed-807) ;; *) exit 2 ;; esac
# 不接受逗号多卡、负数或空值；每个进程只看见一张卡，不改变模型训练契约。
[[ "$gpu_index" =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
py="$environment_root/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$gpu_index"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
"$py" -I -B "$bundle_root/scripts/verify_training_bundle.py" --bundle "$bundle_root" --sha256 "$bundle_sha" --check-installed
"$py" -B "$bundle_root/scripts/verify_causal_control_data.py" --dataset "$data_root" --sha256 "$data_sha"
"$py" -I -c 'from dronedream_agent_core.training.causal_device import causal_training_device; print(causal_training_device("cuda"))'
protected_roots=("$bundle_root" "$data_root" "$environment_root")
encoder_args=()
if test -n "$encoder_root"; then
  test "$role" != local-navigation-policy
  encoder_root="$(cd -- "$encoder_root" && pwd)"
  protected_roots+=("$encoder_root")
  encoder_args=(--encoder-policy "$encoder_root/local-navigation-policy.pt"
    --encoder-training-receipt "$encoder_root/training-receipt.json")
fi
run_root="$("$py" -I -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); roots=[Path(v).resolve() for v in sys.argv[2:]]; assert all(p!=r and r not in p.parents and p not in r.parents for r in roots); print(p)' "$run_root" "${protected_roots[@]}")"
# 既有成功和失败作业都保留；重试必须另用新目录，不覆盖原证据。
mkdir -p -- "$(dirname -- "$run_root")"
mkdir -- "$run_root"
timeout --signal=TERM --kill-after=30s 3600 "$py" -B "$bundle_root/scripts/train_causal_control_role.py" \
  --expert-role "$role" --device cuda --cpu-threads 2 \
  --train "$data_root/training-replay.jsonl" --validation "$data_root/validation-replay.jsonl" \
  --training-observations "$data_root/training-observations.jsonl" \
  --validation-observations "$data_root/validation-observations.jsonl" \
  --stream-groups "$data_root/stream-groups.json" \
  --training-visual-receipt "$data_root/training-visual-receipt.json" \
  --validation-visual-receipt "$data_root/validation-visual-receipt.json" \
  --config "$bundle_root/training/control/$recipe.json" \
  --regularization-config "$bundle_root/training/control/regularization.json" \
  --output "$run_root/candidate" "${encoder_args[@]}" > "$run_root/training.log" 2>&1
timeout --signal=TERM --kill-after=30s 600 "$py" -B "$bundle_root/scripts/validate_causal_control_role.py" \
  --candidate "$run_root/candidate" --output "$run_root/validation.json" > "$run_root/validation.log" 2>&1
printf '%s\n' 'Candidate trained and CPU ONNX validated; no flight qualification or product update.'
