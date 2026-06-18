# Pose2Equip

Pose2Equip is a PyTorch Lightning project for predicting 3D skiing equipment
keypoints from skier observations. The code supports several backbones:

- `stgcn`: pose-only baseline using 3D human keypoints.
- `pose2equip`: RGB sequence + 3D pose model with a DINO patch encoder, ST-GCN
  pose encoder, and equipment query transformer decoder.
- `3dcnn`: video baseline using a 3D CNN.

The canonical equipment target is shaped as `[B, T, 4, 2, 3]`, where the four
equipment segments are left ski, right ski, left pole, and right pole. Each
segment has two 3D endpoints.

## Repository Layout

```text
configs/
  pose2equip.yaml                 # Hydra training config
pegasus/
  train_pose2equip.sh             # PBS/Pegasus training script
pose2equip/
  dataloader/                     # Unity and Ski-PosePTZ dataset loaders
  eval_data/                      # Visualization scripts
  losses/                         # Equipment losses
  metrics/                        # Equipment metrics
  models/                         # Pose2Equip, ST-GCN, and 3D CNN models
  trainer/                        # PyTorch Lightning modules
  main.py                         # Hydra training entry point
  map_config.py                   # Keypoint and equipment mappings
requirements.txt
environment.yaml
```

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate canonical_dualview_3d_pose
```

Or install Python dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

The `pose2equip` backbone uses Hugging Face `transformers` to load DINO models.
The first real run may download model weights.

## Data

The default config expects Unity skiing data and precomputed fold mappings:

```yaml
data.unity.root_path: /home/kaixu_chen/skiing/data/skiing_unity_dataset
data.index_mapping_path: ${data.index_mapping}/use_layer_camera_filter_enabled/camera_pairs_by_action_folds
```

Each fold should be available as:

```text
fold_00.json
fold_01.json
...
```

Each fold JSON is expected to contain `train`, `val`, and `test` entries that
can be converted to `UnityDataConfig`.

Override paths at runtime if your data lives elsewhere:

```bash
python -m pose2equip.main \
  data.unity.root_path=/path/to/skiing_unity_dataset \
  data.index_mapping_path=/path/to/camera_pairs_by_action_folds
```

## Training

Train the default ST-GCN baseline on one fold:

```bash
python -m pose2equip.main \
  model.backbone=stgcn \
  train.fold=0 \
  train.gpu=0
```

Train the multimodal Pose2Equip model:

```bash
python -m pose2equip.main \
  model.backbone=pose2equip \
  data.load_frames=true \
  data.load_3d_kpt=true \
  train.fold=0 \
  train.gpu=0
```

Run all available folds sequentially:

```bash
python -m pose2equip.main train.fold=-1
```

Logs, checkpoints, TensorBoard files, and test outputs are written under:

```text
logs/train_unity/${model.backbone}/${date}/fold_${train.fold}
```

## Tests and Smoke Checks

Compile all Python files:

```bash
python -m py_compile pose2equip/*.py pose2equip/*/*.py
```

Run equipment geometry helper tests:

```bash
python pose2equip/test_equipment_geometry.py
```

Run the model forward-pass smoke test:

```bash
python pose2equip/test_model_forward.py
```

The forward test patches DINO with a mock when necessary, so it can validate
tensor shapes without downloading model weights.

## Useful Config Options

Common overrides:

```bash
model.backbone=stgcn          # stgcn, pose2equip, or 3dcnn
train.fold=0                  # use -1 for all folds
train.max_epochs=50
data.batch_size=32
data.num_workers=16
data.time_window=16
```

Pose2Equip-specific options:

```bash
pose2equip.dino_model_name=facebook/dinov2-base
pose2equip.dino_freeze=true
pose2equip.loss_w_sym=0.03
pose2equip.loss_w_len_abs=0.2
pose2equip.loss_w_temporal_smooth=0.02
pose2equip.predict_anchor_offsets=true
```

## Notes

- The default training accelerator is GPU. For CPU-only debugging, adjust the
  `Trainer` settings in `pose2equip/main.py`.
- Dataset paths in `configs/pose2equip.yaml` are machine-specific defaults and
  should usually be overridden from the command line or a local config.
- Generated logs, checkpoints, TensorBoard outputs, and visualizations are
  ignored by git.
