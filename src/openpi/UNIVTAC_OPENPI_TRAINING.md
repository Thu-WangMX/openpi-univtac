# OpenPI + UniVTAC Training Guide

This note documents the local workflow for training OpenPI pi0 on UniVTAC HDF5 demonstrations.

Local paths in this machine:

- OpenPI repo: `/vepfs-C区/visuotactile/openpi`
- UniVTAC repo: `/vepfs-C区/visuotactile/UniVTAC`
- Official UniVTAC raw data: `/vepfs-C区/visuotactile/UniVTAC/data/official/<task>/clean`
- LeRobot cache: `/vepfs-C区/visuotactile/openpi/.cache/lerobot`
- OpenPI assets: `/vepfs-C区/visuotactile/openpi/assets`
- OpenPI checkpoints: `/vepfs-C区/visuotactile/openpi/checkpoints`

## Recommended Configs

Use only these two UniVTAC training configs for new runs:

| Config | Use case | Inputs |
| --- | --- | --- |
| `pi0_univtac_vision_low_mem_finetune` | Pure vision | head RGB, wrist RGB, 8D joint state |
| `pi0_univtac_tactile_rgb_low_mem_finetune` | Vision + tactile RGB | head RGB, wrist RGB, left/right tactile `rgb_marker`, 8D joint state |

Do not start new `all_modalities` trainings unless there is a specific reason. The old all-modal setup adds EE pose and tactile pose into state, which is not the default direction now.

## Source Files

Important code paths:

- HDF5 to LeRobot converter: `examples/univtac/convert_univtac_data_to_lerobot.py`
- UniVTAC OpenPI train configs: `src/openpi/training/config.py`
- UniVTAC OpenPI input/output transforms: `src/openpi/policies/univtac_policy.py`
- Norm stats script: `scripts/compute_norm_stats.py`
- Training script: `scripts/train.py`
- Closed-loop deploy adapter: `/vepfs-C区/visuotactile/UniVTAC/policy/OpenPI/deploy_policy.py`

## Raw HDF5 Structure

Example file:

```text
/vepfs-C区/visuotactile/UniVTAC/data/official/grasp_classify/clean/0.hdf5
```

Observed structure:

```text
actor/green_pad: shape=(57, 7), dtype=float32
actor/orange_pad: shape=(57, 7), dtype=float32
actor/plain_prism: shape=(57, 7), dtype=float32
actor/rough_prism: shape=(57, 7), dtype=float32
atom/id: shape=(57,), dtype=int64
atom/tag: shape=(57,), dtype=bytes
embodiment/ee: shape=(57, 7), dtype=float32
embodiment/joint: shape=(57, 9), dtype=float32
observation/head/rgb: shape=(57,), dtype=JPEG bytes, decodes to 270x480x3
observation/wrist/rgb: shape=(57,), dtype=JPEG bytes, decodes to 270x480x3
step: shape=(57,), dtype=int64
tactile/left_gsmini/depth: shape=(57, 240, 320), dtype=float32
tactile/left_gsmini/marker: shape=(57, 2, 1200, 2), dtype=float32
tactile/left_gsmini/pose: shape=(57, 7), dtype=float32
tactile/left_gsmini/rgb: shape=(57,), dtype=JPEG bytes
tactile/left_gsmini/rgb_marker: shape=(57,), dtype=JPEG bytes, decodes to 240x320x3
tactile/right_gsmini/depth: shape=(57, 240, 320), dtype=float32
tactile/right_gsmini/marker: shape=(57, 2, 1200, 2), dtype=float32
tactile/right_gsmini/pose: shape=(57, 7), dtype=float32
tactile/right_gsmini/rgb: shape=(57,), dtype=JPEG bytes
tactile/right_gsmini/rgb_marker: shape=(57,), dtype=JPEG bytes, decodes to 240x320x3
```

In the current OpenPI pipeline, “using tactile” means using the two tactile RGB marker images:

```text
tactile/left_gsmini/rgb_marker
tactile/right_gsmini/rgb_marker
```

The `vision_tactile_rgb` config does not use tactile depth, raw marker arrays, EE pose, or tactile pose. It uses only tactile `rgb_marker` images plus 8D joint state.

## Environment Setup

Run from the OpenPI repo:

```bash
cd /vepfs-C区/visuotactile/openpi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_LEROBOT_HOME=/vepfs-C区/visuotactile/openpi/.cache/lerobot
export HF_HOME=/vepfs-C区/visuotactile/openpi/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-C区/visuotactile/openpi/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/vepfs-C区/visuotactile/openpi/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online
```

Use the OpenPI Python environment:

```bash
/vepfs-C区/visuotactile/openpi/.venv/bin/python --version
```

If W&B is not needed, use:

```bash
export WANDB_MODE=disabled
```

## Task Selection

The task is selected by changing `TASK`, `RAW_DIR`, and `REPO_ID`.

Official raw data path pattern:

```bash
TASK=grasp_classify
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean
```

Common UniVTAC task names:

```text
lift_bottle
lift_can
insert_tube
insert_hole
insert_HDMI
pull_out_key
put_bottle_in_shelf
grasp_classify
```

UniVTAC official policy settings use head-only camera for many tasks and head+wrist for some tasks. In the official `policy/task_settings.json`, `lift_can` and `insert_tube` use `camera_type: all`, while tasks such as `grasp_classify`, `lift_bottle`, `insert_hole`, `insert_HDMI`, `pull_out_key`, and `put_bottle_in_shelf` use `camera_type: head`. The current OpenPI ready-to-run `vision` config uses head+wrist.

## Step 1: Convert HDF5 to LeRobot

### Pure Vision

This creates a LeRobot dataset with visually correct standard RGB head/wrist images, 8D joint state, and 8D qpos action.

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=grasp_classify
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean
REPO_ID=local/univtac_${TASK}_vision

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir ${RAW_DIR} \
  --config.repo-id ${REPO_ID} \
  --config.task-name ${TASK} \
  --config.mode vision \
  --config.state-mode joint \
  --config.use-wrist \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite \
  2>&1 | tee convert_${TASK}_vision.log
```

Output dataset:

```text
/vepfs-C区/visuotactile/openpi/.cache/lerobot/local/univtac_<task>_vision
```

### Vision + Tactile RGB

This creates a LeRobot dataset with visually correct standard RGB head/wrist images, visually correct standard RGB left/right tactile `rgb_marker`, 8D joint state, and 8D qpos action.

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=grasp_classify
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean
REPO_ID=local/univtac_${TASK}_tactile_rgb

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir ${RAW_DIR} \
  --config.repo-id ${REPO_ID} \
  --config.task-name ${TASK} \
  --config.mode visuotactile \
  --config.state-mode joint \
  --config.use-wrist \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite \
  2>&1 | tee convert_${TASK}_tactile_rgb.log
```

Output dataset:

```text
/vepfs-C区/visuotactile/openpi/.cache/lerobot/local/univtac_<task>_tactile_rgb
```

## Step 2: Check Converted Data

Basic checks:

```bash
cd /vepfs-C区/visuotactile/openpi

find .cache/lerobot/local/univtac_${TASK}_vision/data -name 'episode_*.parquet' | wc -l
cat .cache/lerobot/local/univtac_${TASK}_vision/meta/info.json | head -80
```

For tactile RGB datasets, replace `_vision` with `_tactile_rgb`.

Channel check after conversion:

```bash
cd /vepfs-C区/visuotactile
openpi/.venv/bin/python UniVTAC/scripts/check_openpi_rgb_consistency.py
```

The expected best match is `cv_bgr2rgb` for every stream. If the checker reports `cv_no_swap`, the dataset was generated with the old channel convention and must be regenerated before training.

Expected post-transform shapes for the current configs:

| Config | Image keys | State | Actions |
| --- | --- | --- | --- |
| `pi0_univtac_vision_low_mem_finetune` | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` | `(8,)` | `(16, 8)` |
| `pi0_univtac_tactile_rgb_low_mem_finetune` | `base_0_rgb`, `wrist_0_rgb`, `left_tactile_0_rgb`, `right_tactile_0_rgb` | `(8,)` | `(16, 8)` |

For pure vision, `right_wrist_0_rgb` is a dummy zero image with mask `False`; the real images are head and wrist.

## Step 3: Compute Norm Stats

OpenPI training requires norm stats for `state` and `actions`.

### Pure Vision

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=grasp_classify
REPO_ID=local/univtac_${TASK}_vision
CONFIG=pi0_univtac_vision_low_mem_finetune

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name ${CONFIG} \
  --repo-id ${REPO_ID} \
  --max-frames 20000 \
  2>&1 | tee ${TASK}_vision_norm_stats.log
```

Stats path:

```text
/vepfs-C区/visuotactile/openpi/assets/pi0_univtac_vision_low_mem_finetune/local/univtac_<task>_vision/norm_stats.json
```

### Vision + Tactile RGB

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=grasp_classify
REPO_ID=local/univtac_${TASK}_tactile_rgb
CONFIG=pi0_univtac_tactile_rgb_low_mem_finetune

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name ${CONFIG} \
  --repo-id ${REPO_ID} \
  --max-frames 20000 \
  2>&1 | tee ${TASK}_tactile_rgb_norm_stats.log
```

Stats path:

```text
/vepfs-C区/visuotactile/openpi/assets/pi0_univtac_tactile_rgb_low_mem_finetune/local/univtac_<task>_tactile_rgb/norm_stats.json
```

## Step 4: Train pi0

### Pure Vision Training

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=grasp_classify
REPO_ID=local/univtac_${TASK}_vision
CONFIG=pi0_univtac_vision_low_mem_finetune
EXP=${TASK}_vision_lora_h16

.venv/bin/python scripts/train.py ${CONFIG} \
  --data.repo-id=${REPO_ID} \
  --exp-name=${EXP} \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee ${EXP}_train.log
```

Checkpoint path:

```text
/vepfs-C区/visuotactile/openpi/checkpoints/pi0_univtac_vision_low_mem_finetune/<task>_vision_lora_h16/
```

### Vision + Tactile RGB Training

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=grasp_classify
REPO_ID=local/univtac_${TASK}_tactile_rgb
CONFIG=pi0_univtac_tactile_rgb_low_mem_finetune
EXP=${TASK}_tactile_rgb_lora_h16

.venv/bin/python scripts/train.py ${CONFIG} \
  --data.repo-id=${REPO_ID} \
  --exp-name=${EXP} \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee ${EXP}_train.log
```

Checkpoint path:

```text
/vepfs-C区/visuotactile/openpi/checkpoints/pi0_univtac_tactile_rgb_low_mem_finetune/<task>_tactile_rgb_lora_h16/
```

## Resume Training

Do not use `--overwrite` when resuming. Use:

```bash
.venv/bin/python scripts/train.py ${CONFIG} \
  --data.repo-id=${REPO_ID} \
  --exp-name=${EXP} \
  --resume \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee ${EXP}_resume.log
```

## Monitor Training

Tail logs:

```bash
tail -f ${EXP}_train.log
```

Find latest checkpoints:

```bash
find checkpoints/${CONFIG}/${EXP} -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail
```

Extract recent training lines:

```bash
rg "loss|Saving|checkpoint|Traceback|OutOfMemory|nan|NaN" ${EXP}_train.log | tail -80
```

## Action Horizon

Current recommended horizon is 16.

The real action horizon is controlled by the OpenPI model config in `src/openpi/training/config.py`:

```python
pi0_config.Pi0Config(
    action_dim=32,
    action_horizon=16,
)
```

The converter's `action_horizon` field is not the main control for OpenPI action chunk length. The LeRobot loader constructs action sequences using the training config's `model.action_horizon`.

If changing horizon from 16 to another value:

1. Add a new train config or edit the target config in `src/openpi/training/config.py`.
2. Change `model.action_horizon`.
3. Change `policy_metadata["action_horizon"]` to match.
4. Recompute norm stats for the target repo.
5. Retrain.
6. Update closed-loop deploy validation if needed. The current UniVTAC OpenPI deploy adapter expects horizon 16.

## Camera and Modality Notes

Current ready-to-run OpenPI configs:

- `pi0_univtac_vision_low_mem_finetune` expects head + wrist visual input.
- `pi0_univtac_tactile_rgb_low_mem_finetune` expects head + wrist + left/right tactile `rgb_marker`.

The converter switch `--config.use-wrist` controls whether `observation/wrist/rgb` is copied into the LeRobot dataset.

The current ready-to-run pure vision config still uses three pi0 image slots:

```text
base_0_rgb          <- head RGB, mask True
left_wrist_0_rgb    <- wrist RGB, mask True
right_wrist_0_rgb   <- zero image, mask False
```

The current tactile RGB config uses four image keys:

```text
base_0_rgb          <- head RGB
wrist_0_rgb         <- wrist RGB
left_tactile_0_rgb  <- left tactile rgb_marker
right_tactile_0_rgb <- right tactile rgb_marker
```

If strict head-only training is needed, add a separate head-only TrainConfig instead of changing only one command-line flag. Training and serving should use the same config name to avoid train/inference mismatch.

## Serving and Closed-loop Evaluation

Serve a trained OpenPI checkpoint from the OpenPI repo:

```bash
cd /vepfs-C区/visuotactile/openpi

CONFIG=pi0_univtac_vision_low_mem_finetune
EXP=grasp_classify_vision_lora_h16
STEP=29999

.venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config ${CONFIG} \
  --policy.dir checkpoints/${CONFIG}/${EXP}/${STEP}
```

For tactile RGB, use:

```bash
CONFIG=pi0_univtac_tactile_rgb_low_mem_finetune
EXP=grasp_classify_tactile_rgb_lora_h16
```

Then run UniVTAC closed-loop evaluation with a matching deploy YAML under:

```text
/vepfs-C区/visuotactile/UniVTAC/policy/OpenPI/
```

Deploy modality must match the served config:

| OpenPI config | UniVTAC deploy modality |
| --- | --- |
| `pi0_univtac_vision_low_mem_finetune` | `vision` |
| `pi0_univtac_tactile_rgb_low_mem_finetune` | `vision_tactile_rgb` |

The UniVTAC OpenPI deploy adapter validates modality, state dim, action dim, action horizon, and image keys from server metadata.

## Common Failure Cases

- `Norm stats not found`: run `scripts/compute_norm_stats.py` for the exact `CONFIG` and `REPO_ID` first.
- `images dict missing keys`: train config, served checkpoint, or deploy modality do not match.
- `Checkpoint directory already exists`: use `--resume` to continue, or `--overwrite` to delete and restart.
- `Batch size must be divisible by device count`: with 4 GPUs, keep `--batch-size=4`, `8`, etc.
- Wrong server port: keep OpenPI server and UniVTAC deploy YAML on the same port, usually `8000`.

## Minimal End-to-end Example

Pure vision `grasp_classify` on 4 GPUs:

```bash
cd /vepfs-C区/visuotactile/openpi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_LEROBOT_HOME=/vepfs-C区/visuotactile/openpi/.cache/lerobot
export HF_HOME=/vepfs-C区/visuotactile/openpi/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-C区/visuotactile/openpi/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/vepfs-C区/visuotactile/openpi/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online

TASK=grasp_classify
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean
REPO_ID=local/univtac_${TASK}_vision
CONFIG=pi0_univtac_vision_low_mem_finetune
EXP=${TASK}_vision_lora_h16

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir ${RAW_DIR} \
  --config.repo-id ${REPO_ID} \
  --config.task-name ${TASK} \
  --config.mode vision \
  --config.state-mode joint \
  --config.use-wrist \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite \
  2>&1 | tee convert_${TASK}_vision.log

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name ${CONFIG} \
  --repo-id ${REPO_ID} \
  --max-frames 20000 \
  2>&1 | tee ${TASK}_vision_norm_stats.log

.venv/bin/python scripts/train.py ${CONFIG} \
  --data.repo-id=${REPO_ID} \
  --exp-name=${EXP} \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee ${EXP}_train.log
```
