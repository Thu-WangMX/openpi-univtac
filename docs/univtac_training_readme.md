# UniVTAC -> OpenPI π0 训练流程 README

这份文档记录当前仓库里已经跑通的 UniVTAC 全模态 π0 LoRA 训练流程。路径默认使用本机：

- OpenPI: `/vepfs-C区/visuotactile/openpi`
- UniVTAC: `/vepfs-C区/visuotactile/UniVTAC`
- UniVTAC 原始数据: `/vepfs-C区/visuotactile/UniVTAC/data`
- LeRobot 数据缓存: `/vepfs-C区/visuotactile/openpi/.cache/lerobot`

## 0. 当前状态

已完成的 all-modal 训练：

| task | config | exp name | checkpoint |
| --- | --- | --- | --- |
| `lift_bottle` | `pi0_univtac_lift_bottle_all_modalities_low_mem_finetune` | `lift_bottle_all_modalities_lora_h16` | `checkpoints/pi0_univtac_lift_bottle_all_modalities_low_mem_finetune/lift_bottle_all_modalities_lora_h16/29999` |
| `lift_can` | `pi0_univtac_all_modalities_low_mem_finetune` | `lift_can_all_modalities_lora_h16` | `checkpoints/pi0_univtac_all_modalities_low_mem_finetune/lift_can_all_modalities_lora_h16/29999` |
| `insert_tube` | `pi0_univtac_all_modalities_low_mem_finetune` | `insert_tube_all_modalities_lora_h16` | `checkpoints/pi0_univtac_all_modalities_low_mem_finetune/insert_tube_all_modalities_lora_h16/29999` |

当前训练设置：

- 模型：π0 base + LoRA 微调
- `action_horizon=16`
- `action_dim=32`，实际输出 qpos action 前 8 维
- `batch_size=4`
- `num_train_steps=30000`
- `fsdp_devices=4`
- 输入图像为四路全模态：head + wrist + left tactile + right tactile

## 1. 环境说明

训练不要直接用 conda base 运行。当前使用 OpenPI 自己的 `.venv`：

```bash
cd /vepfs-C区/visuotactile/openpi
.venv/bin/python --version
```

当前 `.venv/bin/python` 指向 `/root/miniconda3/bin/python3.11`，依赖安装在 `openpi/.venv` 里。

检查环境：

```bash
cd /vepfs-C区/visuotactile/openpi
.venv/bin/python - <<'PY'
import openpi, jax, lerobot, tyro, pyarrow
print('env ok')
print(jax.devices())
PY
```

如果看到 4 张 `CudaDevice`，说明训练环境可用。

### 为什么之前 `.venv` 会坏

之前有一版 `.venv` 是 uv 创建的，它的 Python 是软链接：

```text
.venv/bin/python -> /root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11
```

后来 `/root/.local/share/uv/python/...` 这个 uv-managed Python 目录消失了，软链接变成 dangling symlink，所以 supervisor 启动后续任务时出现：

```text
.venv/bin/python3: No such file or directory
```

已修复为使用 conda base 的稳定 Python 3.11 作为解释器，但依赖仍在 `openpi/.venv` 中。

## 2. 数据格式

原始 UniVTAC hdf5 放在：

```text
/vepfs-C区/visuotactile/UniVTAC/data/official/<task>/clean/*.hdf5
```

已确认全模态 hdf5 key：

- head RGB: `observation/head/rgb`
- wrist RGB: `observation/wrist/rgb`
- left tactile RGB marker: `tactile/left_gsmini/rgb_marker` 或 `tactile/left_tactile/rgb_marker`
- right tactile RGB marker: `tactile/right_gsmini/rgb_marker` 或 `tactile/right_tactile/rgb_marker`
- joint: `embodiment/joint`
- EE pose: `embodiment/ee`
- tactile pose: `tactile/<left/right>_*/pose`

OpenPI 训练用 state 是 extended 29 维：

```text
joint[:8] + ee[:7] + left_tactile_pose[:7] + right_tactile_pose[:7]
```

OpenPI 模型内部会 pad 到 32 维。

## 3. 一键环境变量

每次手动转换/训练前建议先设置：

```bash
cd /vepfs-C区/visuotactile/openpi

export HF_LEROBOT_HOME=/vepfs-C区/visuotactile/openpi/.cache/lerobot
export HF_HOME=/vepfs-C区/visuotactile/openpi/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-C区/visuotactile/openpi/.cache/huggingface/datasets
export OPENPI_DATA_HOME=/vepfs-C区/visuotactile/openpi/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=online  # 如未登录，先运行：.venv/bin/python -m wandb login

# 如需走代理下载 base checkpoint / git / GCS 资源：
export http_proxy=http://100.68.174.60:3128
export https_proxy=http://100.68.174.60:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
```

## 4. Step 1: HDF5 转 LeRobot

以 `insert_hole` 为例：

```bash
cd /vepfs-C区/visuotactile/openpi

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir /vepfs-C区/visuotactile/UniVTAC/data/official/insert_hole/clean \
  --config.repo-id local/univtac_insert_hole_all_modalities \
  --config.use-wrist \
  --config.state-mode extended \
  --config.mode visuotactile \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite \
  2>&1 | tee convert_insert_hole_all_modalities.log
```

输出目录：

```text
openpi/.cache/lerobot/local/univtac_insert_hole_all_modalities
```

其它任务只需要替换 task 名：

```bash
TASK=insert_HDMI
REPO_ID=local/univtac_${TASK}_all_modalities
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir ${RAW_DIR} \
  --config.repo-id ${REPO_ID} \
  --config.use-wrist \
  --config.state-mode extended \
  --config.mode visuotactile \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite \
  2>&1 | tee convert_${TASK}_all_modalities.log
```

检查转换结果：

```bash
find .cache/lerobot/local/univtac_insert_hole_all_modalities/data -name 'episode_*.parquet' | wc -l
cat .cache/lerobot/local/univtac_insert_hole_all_modalities/meta/info.json | head
```

## 5. Step 2: 计算 norm stats

训练前必须有 norm stats，否则 `scripts/train.py` 会报 `Norm stats not found`。

通用方式：

```bash
cd /vepfs-C区/visuotactile/openpi

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_univtac_all_modalities_low_mem_finetune \
  --max-frames 20000
```

但注意：`compute_norm_stats.py` 默认使用 config 里的 `repo_id`。如果要给某个具体任务算 stats，有两个选择。

### 方式 A: 临时改 config 的 repo-id 参数

用 tyro 覆盖：

```bash
cd /vepfs-C区/visuotactile/openpi

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_univtac_all_modalities_low_mem_finetune \
  --repo-id local/univtac_insert_hole_all_modalities
```

如果 tyro 参数名不兼容，使用方式 B。

### 方式 B: 用已写好的快速 stats 逻辑

supervisor 脚本里已有快速 stats 逻辑，会把输出写到：

```text
openpi/assets/pi0_univtac_all_modalities_low_mem_finetune/local/univtac_<task>_all_modalities/norm_stats.json
```

也可以直接让 supervisor 跑，见第 8 节。

检查 norm stats 是否存在：

```bash
find assets/pi0_univtac_all_modalities_low_mem_finetune -path '*univtac_insert_hole_all_modalities*' -name norm_stats.json -print
```

## 6. Step 3: 启动训练

以 `insert_hole` 为例：

```bash
cd /vepfs-C区/visuotactile/openpi

export HF_LEROBOT_HOME=/vepfs-C区/visuotactile/openpi/.cache/lerobot
export OPENPI_DATA_HOME=/vepfs-C区/visuotactile/openpi/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

.venv/bin/python scripts/train.py pi0_univtac_all_modalities_low_mem_finetune \
  --data.repo-id=local/univtac_insert_hole_all_modalities \
  --exp-name=insert_hole_all_modalities_lora_h16 \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee insert_hole_all_modalities_lora_h16_train.log
```

训练输出：

```text
openpi/checkpoints/pi0_univtac_all_modalities_low_mem_finetune/insert_hole_all_modalities_lora_h16/
```

最终 checkpoint：

```text
openpi/checkpoints/pi0_univtac_all_modalities_low_mem_finetune/insert_hole_all_modalities_lora_h16/29999
```

如果要断点续训，不要用 `--overwrite`，改用：

```bash
--resume
```

## 7. 监控训练

看实时 log：

```bash
tail -f insert_hole_all_modalities_lora_h16_train.log
```

抽取 loss：

```bash
python - <<'PY'
from pathlib import Path
import re, statistics
log = Path('insert_hole_all_modalities_lora_h16_train.log')
text = log.read_text(errors='ignore')
ms = re.findall(r'Step\s+(\d+).*?loss=([0-9.eE+-]+)', text)
print('points:', len(ms))
if ms:
    vals = [float(x[1]) for x in ms[-50:]]
    print('last:', ms[-1])
    print('last50 mean:', statistics.mean(vals))
for needle in ['Traceback', 'OutOfMemory', 'CUDA out of memory', 'nan', 'NaN']:
    if needle in text:
        print('contains', needle)
PY
```

检查 checkpoint：

```bash
find checkpoints/pi0_univtac_all_modalities_low_mem_finetune/insert_hole_all_modalities_lora_h16 \
  -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail
```

## 8. 自动接力训练剩余任务

自动脚本：

```text
/vepfs-C区/visuotactile/.downloads/univtac_training_supervisor.py
```

它会按顺序处理：

```text
lift_can -> insert_tube -> insert_hole -> insert_HDMI -> pull_out_key -> put_bottle_in_shelf -> grasp_classify
```

逻辑是：

1. 等当前训练结束
2. 找到已有 100 条 hdf5 的下一个 task
3. 转 LeRobot
4. 计算 norm stats
5. 启动 30000 step LoRA 训练
6. 训练结束后继续下一个 task

查看状态：

```bash
cat /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_status.json
```

查看日志：

```bash
tail -f /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor.log
```

如果 supervisor 因之前 `.venv` 断链误把任务标成 failed，可以重启或清理状态后再跑。最安全做法是先停掉旧 supervisor：

```bash
pgrep -af univtac_training_supervisor
kill <PID>
```

然后重新启动：

```bash
cd /vepfs-C区/visuotactile/openpi
nohup .venv/bin/python /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor.py \
  > /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor.log 2>&1 &
```

如果状态文件里保留了 failed 列表，需要手动编辑或删除：

```bash
cp /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_status.json \
   /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_status.json.bak

rm /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_status.json
```

删除状态文件后，脚本会重新扫描任务。注意它可能会从已经完成的任务重新开始；如果不想重训，建议改脚本的 `completed` 初始集合或保留 status 里的 completed tasks。

## 9. 已完成 checkpoint 的推理

L20 推理指南在：

```text
/vepfs-C区/visuotactile/UniVTAC/docs/OpenPI_L20_Inference.md
```

启动 OpenPI server 示例：

```bash
cd /vepfs-C区/visuotactile/openpi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_univtac_lift_bottle_all_modalities_low_mem_finetune \
  --policy.dir=checkpoints/pi0_univtac_lift_bottle_all_modalities_low_mem_finetune/lift_bottle_all_modalities_lora_h16/29999 \
  --default-prompt='lift the bottle' \
  --port=8000
```

如果不想用 `uv run`，也可以：

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_univtac_lift_bottle_all_modalities_low_mem_finetune \
  --policy.dir=checkpoints/pi0_univtac_lift_bottle_all_modalities_low_mem_finetune/lift_bottle_all_modalities_lora_h16/29999 \
  --default-prompt='lift the bottle' \
  --port=8000
```

## 10. 常见问题

### `No such file or directory: .venv/bin/python3`

说明 `.venv/bin/python` 软链接断了。当前推荐让它指向 conda base Python 3.11：

```bash
cd /vepfs-C区/visuotactile/openpi
ls -l .venv/bin/python*
```

应看到：

```text
.venv/bin/python -> /root/miniconda3/bin/python3.11
```

### `Norm stats not found`

对应 task 的 norm stats 没算或路径不对。检查：

```bash
find assets -path '*local/univtac_<task>_all_modalities/norm_stats.json' -print
```

### `images dict missing keys`

说明不是 all-modal 数据，或者 config/转换参数不匹配。all-modal 必须有：

```text
head RGB + wrist RGB + left tactile rgb_marker + right tactile rgb_marker
```

训练 config 的 image keys 是：

```text
base_0_rgb, wrist_0_rgb, left_tactile_0_rgb, right_tactile_0_rgb
```

### OOM

当前 4 卡训练用：

```bash
--batch-size=4 --fsdp-devices=4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

如果仍 OOM，先确认没有其它训练/推理进程占 GPU：

```bash
nvidia-smi
pgrep -af 'scripts/train.py|serve_policy.py'
```

## 11. 纯视觉训练优先流程

现在已把 UniVTAC π0 训练做成模态可选：

- `UNIVTAC_MODALITY=vision`：纯视觉，使用 head RGB + wrist RGB，不使用触觉图像，也不使用 tactile pose。
- `UNIVTAC_MODALITY=all_modalities`：全模态，使用 head RGB + wrist RGB + left/right tactile RGB marker + tactile pose。

当前优先推荐纯视觉 config：

```text
pi0_univtac_vision_low_mem_finetune
```

纯视觉 config 的模型图像 key 是：

```text
base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
```

其中：

- `base_0_rgb` = UniVTAC head camera
- `left_wrist_0_rgb` = UniVTAC wrist camera
- `right_wrist_0_rgb` = zero image mask，不输入触觉

纯视觉 state 只用：

```text
joint[:8]
```

### 纯视觉手动转换

以 `lift_bottle` 为例：

```bash
cd /vepfs-C区/visuotactile/openpi

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir /vepfs-C区/visuotactile/UniVTAC/data/lift_bottle_official/lift_bottle/clean \
  --config.repo-id local/univtac_lift_bottle_vision \
  --config.use-wrist \
  --config.state-mode joint \
  --config.mode vision \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite
```

其它 official 任务只需替换：

```bash
TASK=lift_can
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean
REPO_ID=local/univtac_${TASK}_vision

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir ${RAW_DIR} \
  --config.repo-id ${REPO_ID} \
  --config.use-wrist \
  --config.state-mode joint \
  --config.mode vision \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite
```

### 纯视觉训练

```bash
cd /vepfs-C区/visuotactile/openpi
export HF_LEROBOT_HOME=/vepfs-C区/visuotactile/openpi/.cache/lerobot
export OPENPI_DATA_HOME=/vepfs-C区/visuotactile/openpi/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

.venv/bin/python scripts/train.py pi0_univtac_vision_low_mem_finetune \
  --data.repo-id=local/univtac_lift_bottle_vision \
  --exp-name=lift_bottle_vision_lora_h16 \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee lift_bottle_vision_lora_h16_train.log
```

checkpoint 输出：

```text
openpi/checkpoints/pi0_univtac_vision_low_mem_finetune/lift_bottle_vision_lora_h16/29999
```

### 纯视觉自动接力

已支持通过环境变量切换 supervisor：

```bash
cd /vepfs-C区/visuotactile/openpi

export UNIVTAC_MODALITY=vision
export WANDB_MODE=online
nohup bash -lc 'cd /vepfs-C区/visuotactile/openpi && export UNIVTAC_MODALITY=vision && exec .venv/bin/python /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor.py' \
  > /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_vision.log 2>&1 &
```

纯视觉 supervisor 状态文件：

```text
/vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_vision_status.json
```

纯视觉 supervisor 日志：

```text
/vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_vision.log
```

查看进度：

```bash
cat /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_vision_status.json
tail -f /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_vision.log
```

## 9. 纯视觉 π0 训练/推理

现在代码支持 `vision` 和 `all_modalities` 两种 OpenPI 模态。纯视觉不是单相机，而是使用 UniVTAC 的两路视觉：head RGB + wrist RGB；不读取 tactile 图像/pose，state 只用 `joint[:8]`。

### 9.1 纯视觉转换

```bash
cd /vepfs-C区/visuotactile/openpi

TASK=lift_bottle
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/lift_bottle_official/lift_bottle/clean
REPO_ID=local/univtac_${TASK}_vision

.venv/bin/python examples/univtac/convert_univtac_data_to_lerobot.py \
  --config.raw-dir ${RAW_DIR} \
  --config.repo-id ${REPO_ID} \
  --config.use-wrist \
  --config.state-mode joint \
  --config.mode vision \
  --config.fps 10 \
  --config.downsample 1 \
  --config.image-writer-threads 0 \
  --config.overwrite
```

其它官方任务的 `RAW_DIR` 用：

```bash
RAW_DIR=/vepfs-C区/visuotactile/UniVTAC/data/official/${TASK}/clean
```

### 9.2 纯视觉 norm stats 和训练

```bash
cd /vepfs-C区/visuotactile/openpi

.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_univtac_vision_low_mem_finetune \
  --repo-id local/univtac_lift_bottle_vision

.venv/bin/python scripts/train.py pi0_univtac_vision_low_mem_finetune \
  --data.repo-id=local/univtac_lift_bottle_vision \
  --exp-name=lift_bottle_vision_lora_h16 \
  --overwrite \
  --batch-size=4 \
  --fsdp-devices=4 \
  2>&1 | tee lift_bottle_vision_lora_h16_train.log
```

如果使用 supervisor 自动接力纯视觉训练：

```bash
cd /vepfs-C区/visuotactile/openpi
nohup env UNIVTAC_MODALITY=vision WANDB_MODE=online .venv/bin/python /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor.py \
  > /vepfs-C区/visuotactile/.downloads/univtac_training_supervisor_vision.log 2>&1 &
```

### 9.3 纯视觉推理

OpenPI server 必须用纯视觉 config/checkpoint：

```bash
cd /vepfs-C区/visuotactile/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_univtac_vision_low_mem_finetune \
  --policy.dir=checkpoints/pi0_univtac_vision_low_mem_finetune/lift_bottle_vision_lora_h16/29999 \
  --default-prompt="lift the bottle" \
  --port=8000
```

UniVTAC 评测侧使用纯视觉 deploy config，例如：

```bash
cd /vepfs-C区/visuotactile/UniVTAC
python scripts/eval_policy.py lift_bottle demo OpenPI/deploy_lift_bottle_vision
```

`policy/OpenPI/deploy_policy.py` 会读取 deploy yml 里的 `modality`：

- `modality: vision`：发送 8D state，只发送 head/wrist 图像，不发送 tactile。
- `modality: all_modalities`：发送 29D state，并发送 head/wrist/left tactile/right tactile 四路图像。
- 如果 OpenPI server 未暴露 UniVTAC metadata，或 metadata 和 deploy yml 不一致，会直接报错，避免训推混用。
