#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-dual2pose}"
FOLD="${FOLD:-0}"
GPUS="${GPUS:-${GPU:-0,1}}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TIME_WINDOW="${TIME_WINDOW:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SAM3D_HUMAN_KEY="${SAM3D_HUMAN_KEY:-character_cam1}"
DRY_RUN=0
SMOKE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment_matrix.sh [options] [target ...]

Targets:
  recommended   Run E01 E02 E07 E08 (default)
  core          Run E01-E10
  optional      Run E11-E12
  all           Run E01-E12
  smoke         Run one 1-batch smoke check (default E02-style)
  E01..E12      Run specific experiment IDs

Options:
  --dry-run     Print commands without executing them
  --smoke       Apply 1-batch smoke overrides to selected experiments
  -h, --help    Show this help
  --            Treat following arguments as extra Hydra overrides

Extra Hydra overrides:
  Any key=value argument is appended to every experiment command.

Environment overrides:
  CONDA_ENV=dual2pose     # env name, or /absolute/path/to/env
  FOLD=0
  GPU=0              # legacy single-GPU shorthand; GPUS takes precedence
  GPUS=0,1           # comma-separated GPU pool for parallel jobs
  MAX_EPOCHS=50
  BATCH_SIZE=64
  TIME_WINDOW=16
  NUM_WORKERS=16
  SAM3D_HUMAN_KEY=character_cam1

Examples:
  bash scripts/run_experiment_matrix.sh
  bash scripts/run_experiment_matrix.sh --dry-run recommended
  bash scripts/run_experiment_matrix.sh E01 E02
  GPUS=0,1 FOLD=1 bash scripts/run_experiment_matrix.sh core
  GPUS=1 bash scripts/run_experiment_matrix.sh E02
  bash scripts/run_experiment_matrix.sh --smoke E08
  bash scripts/run_experiment_matrix.sh all data.unity.root_path=/path/to/data

Experiment IDs:
  E01 stgcn + unity
  E02 stgcn + sam3d
  E03 mlp + unity
  E04 mlp + sam3d
  E05 tcn + unity
  E06 tcn + sam3d
  E07 skeleton_transformer + unity
  E08 skeleton_transformer + sam3d
  E09 stgcn_query + unity
  E10 stgcn_query + sam3d
  E11 pose2equip + unity + RGB frames
  E12 pose2equip + sam3d + RGB frames
EOF
}

backbone_for() {
  case "$1" in
    E01|E02) echo "stgcn" ;;
    E03|E04) echo "mlp" ;;
    E05|E06) echo "tcn" ;;
    E07|E08) echo "skeleton_transformer" ;;
    E09|E10) echo "stgcn_query" ;;
    E11|E12) echo "pose2equip" ;;
    *) echo "Unknown experiment ID: $1" >&2; return 2 ;;
  esac
}

source_for() {
  case "$1" in
    E01|E03|E05|E07|E09|E11) echo "unity" ;;
    E02|E04|E06|E08|E10|E12) echo "sam3d" ;;
    *) echo "Unknown experiment ID: $1" >&2; return 2 ;;
  esac
}

frames_for() {
  case "$1" in
    E11|E12) echo "true" ;;
    E01|E02|E03|E04|E05|E06|E07|E08|E09|E10) echo "false" ;;
    *) echo "Unknown experiment ID: $1" >&2; return 2 ;;
  esac
}

expand_target() {
  case "$1" in
    recommended) echo "E01 E02 E07 E08" ;;
    core) echo "E01 E02 E03 E04 E05 E06 E07 E08 E09 E10" ;;
    optional) echo "E11 E12" ;;
    all) echo "E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12" ;;
    smoke) echo "E02" ;;
    E01|E02|E03|E04|E05|E06|E07|E08|E09|E10|E11|E12) echo "$1" ;;
    *) echo "Unknown target: $1" >&2; return 2 ;;
  esac
}

run_experiment() {
  local exp_id="$1"
  local visible_gpu="$2"
  local backbone source frames
  backbone="$(backbone_for "${exp_id}")"
  source="$(source_for "${exp_id}")"
  frames="$(frames_for "${exp_id}")"

  local conda_cmd=(conda run)
  if [[ "${CONDA_ENV}" == /* ]]; then
    conda_cmd+=(-p "${CONDA_ENV}")
  else
    conda_cmd+=(-n "${CONDA_ENV}")
  fi

  local cmd=(
    "${conda_cmd[@]}" python -m pose2equip.main
    "model.backbone=${backbone}"
    "data.human_3d_source=${source}"
    "data.load_frames=${frames}"
    "data.load_3d_kpt=true"
    "data.load_2d_kpt=false"
    "data.time_window=${TIME_WINDOW}"
    "data.batch_size=${BATCH_SIZE}"
    "data.num_workers=${NUM_WORKERS}"
    "train.fold=${FOLD}"
    "train.max_epochs=${MAX_EPOCHS}"
    "train.gpu=0"
    "trainer.accelerator=gpu"
    "trainer.devices=1"
  )

  if [[ "${source}" == "sam3d" ]]; then
    cmd+=("data.sam3d_human_key=${SAM3D_HUMAN_KEY}")
  fi

  if [[ "${SMOKE}" == "1" ]]; then
    cmd+=(
      "train.max_epochs=1"
      "data.batch_size=1"
      "data.num_workers=0"
      "trainer.limit_train_batches=1"
      "trainer.limit_val_batches=1"
      "trainer.limit_test_batches=1"
      "trainer.num_sanity_val_steps=0"
      "trainer.test_ckpt_path=null"
    )
  fi

  if [[ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_OVERRIDES[@]}")
  fi

  echo "================================================================"
  echo "Running ${exp_id}: ${backbone} + ${source} (frames=${frames}, visible_gpu=${visible_gpu})"
  echo "================================================================"
  printf 'CUDA_VISIBLE_DEVICES=%q' "${visible_gpu}"
  printf ' %q' "${cmd[@]}"
  echo

  if [[ "${DRY_RUN}" == "0" ]]; then
    CUDA_VISIBLE_DEVICES="${visible_gpu}" "${cmd[@]}"
  fi
}

args=()
EXTRA_OVERRIDES=()
parse_overrides=0
for arg in "$@"; do
  if [[ "${parse_overrides}" == "1" ]]; then
    EXTRA_OVERRIDES+=("$arg")
    continue
  fi

  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --smoke) SMOKE=1 ;;
    -h|--help) usage; exit 0 ;;
    --) parse_overrides=1 ;;
    *=*) EXTRA_OVERRIDES+=("$arg") ;;
    *) args+=("$arg") ;;
  esac
done

if [[ "${#args[@]}" -eq 0 ]]; then
  args=(recommended)
fi

expanded=()
for target in "${args[@]}"; do
  if ! expanded_text="$(expand_target "${target}")"; then
    exit 2
  fi
  read -r -a ids <<< "${expanded_text}"
  expanded+=("${ids[@]}")
  if [[ "${target}" == "smoke" ]]; then
    SMOKE=1
  fi
done

IFS=',' read -r -a gpu_pool <<< "${GPUS}"
if [[ "${#gpu_pool[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi

pids=()
for idx in "${!expanded[@]}"; do
  exp_id="${expanded[$idx]}"
  gpu_idx=$((idx % ${#gpu_pool[@]}))
  visible_gpu="${gpu_pool[$gpu_idx]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_experiment "${exp_id}" "${visible_gpu}"
    continue
  fi

  run_experiment "${exp_id}" "${visible_gpu}" &
  pids+=("$!")

  if [[ "${#pids[@]}" -ge "${#gpu_pool[@]}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

if [[ "${DRY_RUN}" == "0" && "${#pids[@]}" -gt 0 ]]; then
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi
