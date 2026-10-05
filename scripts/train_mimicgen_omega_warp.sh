#!/usr/bin/env bash
# Run the Omega-Warp path on the converted official MimicGen demonstrations.
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mode=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in smoke|train|val) ;; *) echo "Usage: $0 [smoke|train|val] [Hydra overrides...]" >&2; exit 2;; esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH="$repo_root/vggt-omega-warp${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export VERA_DATA_PREFIX=${VERA_DATA_PREFIX:-"$repo_root/data"}
export VERA_OMEGA_CHECKPOINT=${VERA_OMEGA_CHECKPOINT:-"$repo_root/vera/checkpoints/vggt_omega_1b_512.pt"}
export VERA_WARP_CHECKPOINT=${VERA_WARP_CHECKPOINT:-"$repo_root/outputs/omega_warp_preflight/best_inference.pt"}
export VERA_WARP_CONFIG=${VERA_WARP_CONFIG:-"$repo_root/outputs/omega_warp_preflight/teacher.toml"}
export WANDB_MODE=${WANDB_MODE:-disabled}
python_bin=${VERA_PYTHON:-"$repo_root/.venv/bin/python"}

for required in "$python_bin" "$VERA_OMEGA_CHECKPOINT" "$VERA_WARP_CHECKPOINT" "$VERA_WARP_CONFIG"; do
    [[ -s "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done

overrides=("wandb.mode=disabled")
if [[ "$mode" == smoke ]]; then
    export VERA_MIMICGEN_ROOT="$repo_root/outputs/omega_warp_preflight/pack_smoke"
    overrides+=("~validation_datasets" "experiment.training.max_steps=2"
        "experiment.training.max_epochs=1" "experiment.training.data.num_workers=0"
        "experiment.training.data.persistent_workers=false"
        "experiment.training.data.prefetch_factor=null"
        "experiment.validation.limit_batch=0" "experiment.validation.num_sanity_val_steps=0"
        "experiment.training.checkpointing.every_n_train_steps=1"
        "experiment.training.log_every_n_steps=1" "algorithm.logging.loss_freq=1")
else
    export VERA_MIMICGEN_ROOT=${VERA_MIMICGEN_ROOT:-"$VERA_DATA_PREFIX/datasets/jacobian/mimicgen_official_rgb_warp"}
fi
if [[ "$mode" == val ]]; then
    overrides+=("experiment.tasks=[validation]")
    [[ " $* " == *" load="* ]] || { echo "Validation requires load=/path/to/checkpoint" >&2; exit 2; }
fi
[[ -s "$VERA_MIMICGEN_ROOT/index.json" ]] || { echo "Dataset conversion has not completed: $VERA_MIMICGEN_ROOT" >&2; exit 2; }
run_dir="$repo_root/outputs/mimicgen_omega_warp/${mode}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir"
printf 'Run directory: %s\nDataset: %s\nGPU selection: %s\n' "$run_dir" "$VERA_MIMICGEN_ROOT" "$CUDA_VISIBLE_DEVICES"
exec "$python_bin" -m vera.main --config-name=config_jacobian_mimicgen_omega_warp_official_rgb \
    "hydra.run.dir=$run_dir" "${overrides[@]}" "$@"
