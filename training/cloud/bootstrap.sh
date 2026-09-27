#!/usr/bin/env bash
# 功能：在已由用户租好的 Linux Pod 中建立独立训练环境；不创建或续租 Pod。
# 输入：上传目录、另外保存的清单摘要、上传目录之外的新环境目录。
# 输出：环境目录中的 venv 和实际依赖清单；CUDA 不可用时失败，不回退 CPU。
set -euo pipefail
# 环境探测也必须使用干净导入路径，不能等创建 venv 时才清除宿主覆盖。
unset PYTHONPATH PYTHONHOME
bundle_root="$(cd -- "${1:?supply verified bundle directory}" && pwd)"
manifest_sha="${2:?supply separately saved manifest sha256}"
environment_root="${3:?supply new environment directory outside bundle}"
python3 -I -B "$bundle_root/scripts/verify_training_bundle.py" --bundle "$bundle_root" --sha256 "$manifest_sha"
python3 -c 'import platform,sys; assert platform.system()=="Linux" and platform.machine()=="x86_64"; assert (3,11)<=sys.version_info[:2]<(3,14)'
environment_root="$(python3 -I -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(); b=Path(sys.argv[2]); assert p!=b and b not in p.parents and p not in b.parents; print(p)' "$environment_root" "$bundle_root")"
test ! -e "$environment_root"
export PYTHONDONTWRITEBYTECODE=1
python3 -m venv "$environment_root/venv"
py="$environment_root/venv/bin/python"
export PIP_CACHE_DIR="$environment_root/pip-cache"
export TORCH_HOME="$environment_root/torch-cache"
"$py" -m pip install --no-compile torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
"$py" -m pip install --no-compile -r "$bundle_root/training/cloud/requirements.txt"
shopt -s nullglob
wheels=("$bundle_root"/*.whl)
test "${#wheels[@]}" -eq 1
"$py" -m pip install --no-deps --no-compile "${wheels[0]}"
"$py" -m pip check
"$py" -I -B "$bundle_root/scripts/verify_training_bundle.py" --bundle "$bundle_root" --sha256 "$manifest_sha" --check-installed
"$py" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: do not run paid CPU fallback"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$py" -m pip freeze --all > "$environment_root/resolved-linux-environment.txt"
