#!/usr/bin/env bash
# 功能：先验证数据再执行一小时预算的单卡训练；不自动循环续跑或更改 GPU 计费。
# 输入：上传目录、数据目录、作业目录、环境目录、代码清单摘要、数据回执摘要、可选 resume。
# 输出：作业检查点及评估回执；退出码二表示已保存且暂停，不代表训练完成。
set -euo pipefail
# 所有解释器调用之前清除宿主模块覆盖；不能先验证安装再用另一份 PYTHONPATH 导入。
unset PYTHONPATH PYTHONHOME
bundle_root="$(cd -- "${1:?bundle}" && pwd)"
dataset_root="$(cd -- "${2:?dataset}" && pwd)"
run_root="${3:?new run directory}"
environment_root="$(cd -- "${4:?environment directory}" && pwd)"
manifest_sha="${5:?separately saved manifest sha256}"
dataset_sha="${6:?separately saved dataset assembly receipt sha256}"
resume_flag="${7:-}"
test -z "$resume_flag" || test "$resume_flag" = "resume"
py="$environment_root/venv/bin/python"
"$py" -I -B "$bundle_root/scripts/verify_training_bundle.py" --bundle "$bundle_root" --sha256 "$manifest_sha" --check-installed
"$py" -B "$bundle_root/scripts/verify_vision_dataset.py" --dataset "$dataset_root" --sha256 "$dataset_sha"
run_root="$("$py" -I -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); roots=[Path(v) for v in sys.argv[2:]]; assert all(p!=r and r not in p.parents and p not in r.parents for r in roots); print(p)' "$run_root" "$bundle_root" "$dataset_root" "$environment_root")"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_HOME="$environment_root/torch-cache"
args=()
if test "$resume_flag" = "resume"; then args+=(--resume); fi
mkdir -p -- "$run_root"
"$py" "$bundle_root/scripts/preflight_vision_data.py" \
  --training-root "$dataset_root/training" --validation-root "$dataset_root/validation" \
  --test-root "$dataset_root/test" --receipt "$run_root/data-preflight-$(date +%s%N).json"
"$py" "$bundle_root/scripts/train_local_vision.py" \
  --training-root "$dataset_root/training" --training-data "$dataset_root/training/samples.jsonl" \
  --validation-root "$dataset_root/validation" --validation-data "$dataset_root/validation/samples.jsonl" \
  --device cuda --batch-size 16 --epoch-count 30 --freeze-backbone-epochs 2 \
  --pretrained-weights "$bundle_root/weights/lraspp_mobilenet_v3_large-d234d4ea.pth" \
  --pretrained-sha256 d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280 \
  --run-directory "$run_root/checkpoints" --checkpoint-batches 500 \
  --max-training-seconds 3600 --early-stop-patience 5 \
  --output-model "$run_root/vision.onnx" --training-receipt "$run_root/training-receipt.json" \
  "${args[@]}"
