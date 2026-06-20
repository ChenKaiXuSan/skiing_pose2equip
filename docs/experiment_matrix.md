# Pose2Equip Experiment Matrix

This matrix compares clean Unity human-pose input against realistic SAM3D human-pose input for pose-only equipment prediction, then optionally adds the RGB-based Pose2Equip model.

## Fixed Settings

Use these settings unless an experiment overrides them:

```text
train.fold=0
train.max_epochs=50
data.batch_size=16
data.time_window=16
data.load_3d_kpt=true
data.load_2d_kpt=false
train.gpu=0
```

For pose-only backbones, keep `data.load_frames=false`. For `pose2equip`, use `data.load_frames=true`.


## One-Command Runner

The experiment matrix can be launched with:

```bash
bash scripts/run_experiment_matrix.sh
```

By default this runs the recommended first pass: E01, E02, E07, and E08. The runner assigns jobs round-robin across `GPUS=0,1`, so each experiment uses one GPU while different experiments run in parallel. Useful variants:

```bash
# Preview commands without training
bash scripts/run_experiment_matrix.sh --dry-run recommended

# Run one experiment
bash scripts/run_experiment_matrix.sh E02

# Run the full pose-only core matrix
bash scripts/run_experiment_matrix.sh core

# Run all E01-E12 experiments
bash scripts/run_experiment_matrix.sh all

# One-batch smoke check for a selected experiment
bash scripts/run_experiment_matrix.sh --smoke E08

# Change GPU pool/fold without editing the script
GPUS=0,1 FOLD=1 bash scripts/run_experiment_matrix.sh E01 E02

# Force all jobs onto one GPU
GPUS=1 bash scripts/run_experiment_matrix.sh E02
```


## Pegasus All-Matrix Runner

Pegasus 版本模仿 `pegasus/train_pose2equip.sh`：使用一个 PBS array 脚本 `pegasus/run_experiment_matrix_all.sh`，脚本内部用数组按 `PBS_SUBREQNO` 选择实验参数，然后直接调用 `python -m pose2equip.main`。

提交完整 all 矩阵：

```bash
qsub -V pegasus/run_experiment_matrix_all.sh
```

默认映射：

```text
PBS_SUBREQNO=0  -> E01  stgcn + unity
PBS_SUBREQNO=1  -> E02  stgcn + sam3d
PBS_SUBREQNO=2  -> E03  mlp + unity
PBS_SUBREQNO=3  -> E04  mlp + sam3d
PBS_SUBREQNO=4  -> E05  tcn + unity
PBS_SUBREQNO=5  -> E06  tcn + sam3d
PBS_SUBREQNO=6  -> E07  skeleton_transformer + unity
PBS_SUBREQNO=7  -> E08  skeleton_transformer + sam3d
PBS_SUBREQNO=8  -> E09  stgcn_query + unity
PBS_SUBREQNO=9  -> E10  stgcn_query + sam3d
PBS_SUBREQNO=10 -> E11  pose2equip + unity + frames
PBS_SUBREQNO=11 -> E12  pose2equip + sam3d + frames
```

默认设置：

```text
#PBS -t 0-11
FOLD_ID=0
MAX_EPOCHS=50
BATCH_SIZE=64
TIME_WINDOW=16
NUM_WORKERS=16
```

Pegasus 默认数据路径是：

```text
DATA_ROOT=/work/SKIING/chenkaixu/data/skiing/skiing_unity_dataset
INDEX_MAPPING_PATH=/work/SKIING/chenkaixu/data/skiing/skiing_unity_dataset/index_mapping/use_layer_camera_filter_disabled/camera_pairs_by_action_folds
```

常用提交方式：

```bash
# 先手动 dry-run 某个 array task，不提交训练
SKIP_CONDA=1 DRY_RUN=1 PBS_SUBREQNO=1 bash pegasus/run_experiment_matrix_all.sh

# 提交完整 E01-E12：12 个 array tasks，每个 task 一个实验
qsub -V pegasus/run_experiment_matrix_all.sh

# 只跑 E01-E08，对应 task 0-7
qsub -V -t 0-7 pegasus/run_experiment_matrix_all.sh

# 只跑单个实验 E08，对应 task 7
qsub -V -t 7 pegasus/run_experiment_matrix_all.sh

# 改 fold / epoch / batch size，会通过 qsub -V 传到每个 task
FOLD_ID=1 MAX_EPOCHS=80 BATCH_SIZE=32 qsub -V pegasus/run_experiment_matrix_all.sh

# 如果 Pegasus 上环境或数据路径不同
CONDA_ENV_PATH=/path/to/env DATA_ROOT=/path/to/skiing_unity_dataset qsub -V pegasus/run_experiment_matrix_all.sh
```

## Quick Smoke Command

Before launching a long experiment, replace `model.backbone` and `data.human_3d_source` in this command to verify the run starts:

```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=stgcn \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=false \
  train.fold=0 \
  train.max_epochs=1 \
  train.gpu=0 \
  data.num_workers=0 \
  data.batch_size=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.limit_test_batches=1 \
  trainer.num_sanity_val_steps=0 \
  trainer.test_ckpt_path=null
```

## Core Matrix

| ID | Backbone | Human Input | Frames | Purpose |
|---|---|---|---|---|
| E01 | `stgcn` | `unity` | false | Pose-only upper bound |
| E02 | `stgcn` | `sam3d` | false | Pose-only realistic setting |
| E03 | `mlp` | `unity` | false | Weak pose baseline upper bound |
| E04 | `mlp` | `sam3d` | false | Weak pose baseline with estimated pose |
| E05 | `tcn` | `unity` | false | Temporal convolution baseline upper bound |
| E06 | `tcn` | `sam3d` | false | Temporal convolution with estimated pose |
| E07 | `skeleton_transformer` | `unity` | false | Transformer pose baseline upper bound |
| E08 | `skeleton_transformer` | `sam3d` | false | Transformer pose baseline with estimated pose |
| E09 | `stgcn_query` | `unity` | false | Query-decoder pose baseline upper bound |
| E10 | `stgcn_query` | `sam3d` | false | Query-decoder pose baseline with estimated pose |

## Optional Multimodal Matrix

| ID | Backbone | Human Input | Frames | Purpose |
|---|---|---|---|---|
| E11 | `pose2equip` | `unity` | true | RGB + clean pose upper bound |
| E12 | `pose2equip` | `sam3d` | true | RGB + SAM3D realistic setting |

## Commands
### E01: ST-GCN + Unity Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=stgcn \
  data.human_3d_source=unity \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E02: ST-GCN + SAM3D Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=stgcn \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E03: MLP + Unity Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=mlp \
  data.human_3d_source=unity \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E04: MLP + SAM3D Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=mlp \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E05: TCN + Unity Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=tcn \
  data.human_3d_source=unity \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E06: TCN + SAM3D Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=tcn \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E07: Skeleton Transformer + Unity Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=skeleton_transformer \
  data.human_3d_source=unity \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E08: Skeleton Transformer + SAM3D Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=skeleton_transformer \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E09: ST-GCN Query + Unity Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=stgcn_query \
  data.human_3d_source=unity \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E10: ST-GCN Query + SAM3D Body
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=stgcn_query \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=false \
  train.fold=0 \
  train.gpu=0
```
### E11: Pose2Equip + Unity Body + RGB Frames
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=pose2equip \
  data.human_3d_source=unity \
  data.load_frames=true \
  train.fold=0 \
  train.gpu=0
```
### E12: Pose2Equip + SAM3D Body + RGB Frames
```bash
conda run -n dual2pose python -m pose2equip.main \
  model.backbone=pose2equip \
  data.human_3d_source=sam3d \
  data.sam3d_human_key=character_cam1 \
  data.load_frames=true \
  train.fold=0 \
  train.gpu=0
```
## Recommended Run Order

Run this minimal first pass before launching the full matrix:

1. E01: `stgcn + unity`
2. E02: `stgcn + sam3d`
3. E07: `skeleton_transformer + unity`
4. E08: `skeleton_transformer + sam3d`

If those are stable, run E03-E10. Use E11-E12 only after the pose-only baselines look reasonable, because RGB training is heavier and may introduce additional failure modes.

## Outputs To Compare

Each run writes logs under:

```text
logs/train_unity/${model.backbone}/${date}/fold_${train.fold}/
```

The main evaluation files are:

```text
pose_analysis/evaluation_metrics.json
pose_analysis/evaluation_metrics.txt
pose_analysis/pose2equip_outputs.pt
```

Compare at least these JSON fields across experiments:

```text
mpjpe
pa_mpjpe
mpjpe_avg_ski
mpjpe_avg_pole
mpjpe_left_ski
mpjpe_right_ski
mpjpe_left_pole
mpjpe_right_pole
```
