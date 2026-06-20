#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=72:00:00
#PBS -N pose2equip_matrix
#PBS -t 0-11
#PBS -o logs/pegasus/pose2equip/matrix_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/pose2equip/matrix_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs/pegasus/pose2equip

CONDA_ENV_PATH=${CONDA_ENV_PATH:-/home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body}
SKIP_CONDA=${SKIP_CONDA:-0}
if [[ "${SKIP_CONDA}" != "1" ]]; then
    source ${CONDA_PREFIX}/etc/profile.d/conda.sh
    conda deactivate
    conda activate ${CONDA_ENV_PATH}
    conda env list
fi

# === 2. 训练参数（按需修改） ===
DATA_ROOT=${DATA_ROOT:-/work/SKIING/chenkaixu/data/skiing/skiing_unity_dataset}
INDEX_MAPPING_DIR=${INDEX_MAPPING_DIR:-${DATA_ROOT}/index_mapping}
INDEX_MAPPING_PATH=${INDEX_MAPPING_PATH:-${INDEX_MAPPING_DIR}/use_layer_camera_filter_disabled/camera_pairs_by_action_folds}

EXP_IDS=(E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11 E12)
MODEL_BACKBONES=(stgcn stgcn mlp mlp tcn tcn skeleton_transformer skeleton_transformer stgcn_query stgcn_query pose2equip pose2equip)
HUMAN_3D_SOURCES=(unity sam3d unity sam3d unity sam3d unity sam3d unity sam3d unity sam3d)
LOAD_FRAMES=(false false false false false false false false false false true true)

NUM_WORKERS=${NUM_WORKERS:-16}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_EPOCHS=${MAX_EPOCHS:-50}
TIME_WINDOW=${TIME_WINDOW:-16}
FOLD_ID=${FOLD_ID:-${FOLD:-0}}
SAM3D_HUMAN_KEY=${SAM3D_HUMAN_KEY:-character_cam1}
DRY_RUN=${DRY_RUN:-0}

# experiment assignment:
# - PBS array mode: use PBS_SUBREQNO
# - non-array/manual mode: allow TASK_ID override, default 0
TASK_ID=${PBS_SUBREQNO:-${PBS_ARRAY_INDEX:-${TASK_ID:-0}}}

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "TASK_ID must be a non-negative integer, got: ${TASK_ID}" >&2
    exit 2
fi

if [[ "${TASK_ID}" -ge "${#EXP_IDS[@]}" ]]; then
    echo "TASK_ID ${TASK_ID} is out of range. Valid range: 0-$((${#EXP_IDS[@]} - 1))" >&2
    exit 2
fi

EXP_ID=${EXP_IDS[${TASK_ID}]}
MODEL_BACKBONE=${MODEL_BACKBONES[${TASK_ID}]}
HUMAN_3D_SOURCE=${HUMAN_3D_SOURCES[${TASK_ID}]}
LOAD_FRAME=${LOAD_FRAMES[${TASK_ID}]}

EXTRA_ARGS=()
if [[ "${HUMAN_3D_SOURCE}" == "sam3d" ]]; then
    EXTRA_ARGS+=(data.sam3d_human_key=${SAM3D_HUMAN_KEY})
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(trainer.limit_train_batches=1)
    EXTRA_ARGS+=(trainer.limit_val_batches=1)
    EXTRA_ARGS+=(trainer.limit_test_batches=1)
    EXTRA_ARGS+=(trainer.num_sanity_val_steps=0)
    EXTRA_ARGS+=(trainer.test_ckpt_path=null)
fi

echo "🏁 Matrix job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Index Mapping: ${INDEX_MAPPING_PATH}"
echo "GPU: 0, Epochs: ${MAX_EPOCHS}, Workers: ${NUM_WORKERS}"
echo "Task ID: ${TASK_ID}"
echo "Experiment: ${EXP_ID}"
echo "Backbone: ${MODEL_BACKBONE}"
echo "Human 3D Source: ${HUMAN_3D_SOURCE}"
echo "Load Frames: ${LOAD_FRAME}"
echo "Fold: ${FOLD_ID}"

# === 3. 执行训练（每个 array task 只跑一个实验） ===
CMD=(
    python -m pose2equip.main
    data.unity.root_path=${DATA_ROOT}
    data.index_mapping=${INDEX_MAPPING_DIR}
    data.index_mapping_path=${INDEX_MAPPING_PATH}
    train.gpu=0
    train.max_epochs=${MAX_EPOCHS}
    train.fold=${FOLD_ID}
    data.num_workers=${NUM_WORKERS}
    data.batch_size=${BATCH_SIZE}
    data.time_window=${TIME_WINDOW}
    data.load_3d_kpt=true
    data.load_2d_kpt=false
    data.load_frames=${LOAD_FRAME}
    model.backbone=${MODEL_BACKBONE}
    data.human_3d_source=${HUMAN_3D_SOURCE}
    trainer.accelerator=gpu
    trainer.devices=1
    "${EXTRA_ARGS[@]}"
)

printf '%q ' "${CMD[@]}"
echo

if [[ "${DRY_RUN}" != "1" ]]; then
    "${CMD[@]}"
fi

echo "🏁 Matrix job finished at: $(date)"
