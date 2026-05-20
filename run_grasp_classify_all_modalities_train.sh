#!/usr/bin/env bash
set -euo pipefail
cd /vepfs-C区/visuotactile/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_LEROBOT_HOME=/vepfs-C区/visuotactile/openpi/.cache/lerobot
export HF_HOME=/vepfs-C区/visuotactile/openpi/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-C区/visuotactile/openpi/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/vepfs-C区/visuotactile/openpi/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=${WANDB_MODE:-online}
exec .venv/bin/python scripts/train.py pi0_univtac_all_modalities_low_mem_finetune \
  --data.repo-id=local/univtac_grasp_classify_all_modalities \
  --exp-name=grasp_classify_all_modalities_lora_h16 \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4
