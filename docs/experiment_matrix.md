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
