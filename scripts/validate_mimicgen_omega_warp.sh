#!/usr/bin/env bash
# Native VERA validation: nine pinned task loaders plus the original "all" loader.
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
checkpoint=${1:?Usage: validate_mimicgen_omega_warp.sh CHECKPOINT OUTPUT_DIR}
output_dir=${2:?Usage: validate_mimicgen_omega_warp.sh CHECKPOINT OUTPUT_DIR}
checkpoint=$(realpath "$checkpoint")
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export WANDB_MODE=offline
exec bash scripts/train_mimicgen_omega_warp.sh val \
    "load=\"$checkpoint\"" "hydra.run.dir=\"$output_dir\"" "wandb.mode=offline"
