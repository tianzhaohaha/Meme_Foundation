# Shared environment for the three experiment launchers.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  export PATH="$PROJECT_DIR/.venv/bin:$PATH"
fi
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-tumemo-training-comparison}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-three-strategies}"
