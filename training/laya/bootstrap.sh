#!/usr/bin/env bash
set -euo pipefail
# 功能：在用户明确指定的新环境安装训练依赖；不替换系统 Python/Runtime。
# 输入：VENV_PATH、CUDA_WHEEL_INDEX 必须由租用后的硬件检查明确给出。
# 输出：独立环境；缺少 GPU/版本不兼容时失败，不静默退回 CPU。
: "${VENV_PATH:?Set an experiment-owned new virtual environment path}"
: "${CUDA_WHEEL_INDEX:?Set the verified official PyTorch CUDA wheel index}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
case "$CUDA_WHEEL_INDEX" in
  https://download.pytorch.org/whl/cu[0-9][0-9][0-9]) ;;
  *) echo "Unsupported wheel index" >&2; exit 2 ;;
esac
if [ -e "$VENV_PATH" ]; then
  echo "Refusing to overwrite an existing environment" >&2
  exit 2
fi
"$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for the validated training toolchain"'
"$BOOTSTRAP_PYTHON" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install torch==2.13.0 --index-url "$CUDA_WHEEL_INDEX"
"$VENV_PATH/bin/python" -m pip install -r "$(dirname "$0")/requirements.txt"
"$VENV_PATH/bin/python" -m pip check
"$VENV_PATH/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
