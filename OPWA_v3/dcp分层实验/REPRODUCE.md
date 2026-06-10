# 复现指南（REPRODUCE）

## 环境要求

- Python 3.10
- CUDA 11.7
- Conda 环境：`img2img-turbo`（已激活）
- GPU（A100）

## 依赖检查

```
pip list | grep -E "torch|transformers|diffusers|opencv|matplotlib|timm|einops"
```

关键版本：
- torch 2.0.1+cu117
- torchvision 0.15.1+cu117
- transformers 4.35.2
- opencv-python 4.6.0.66
- matplotlib 3.10.9

## 数据确认

```bash
# ACDC Fog
ls /gz-data/ACDC/rgb_anon_trainvaltest/rgb_anon/fog/val/*/*.png | wc -l
# 预期: 100

ls /gz-data/ACDC/gt/fog/val/*/*_gt_labelTrainIds.png | wc -l
# 预期: 100

# GT 格式验证（直接 trainID，无需 LABEL2TRAIN）
python -c "
import cv2, numpy as np
gt = cv2.imread('/gz-data/ACDC/gt/fog/val/GOPR0476/GOPR0476_frame_000761_gt_labelTrainIds.png', cv2.IMREAD_UNCHANGED)
print(f'Shape: {gt.shape}, dtype: {gt.dtype}')
print(f'Unique values: {sorted(np.unique(gt).tolist())}')
print(f'Max (should be <=18 or 255): {gt.max()}')
"
```

## 模型文件

### SegFormer-B0
自动从 HuggingFace Hub 加载：
```python
from transformers import SegformerForSemanticSegmentation
model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
)
```
缓存路径：`/gz-data/huggingface_cache/`

### Depth Anything V2（非必需，本实验未使用）

```bash
ls /root/opwa_v3/checkpoints/depth_anything_v2_vits.pth
ls /root/opwa_v3/Depth-Anything-V2/depth_anything_v2/dpt.py
```

## 执行步骤

```bash
# 0. 确认分支
git branch   # 应显示 dcp分层实验

# 1. 验证导入与路径
python OPWA_v3/dcp分层实验/run_all.py --dry-run

# 2. 运行实验 A（DCP 分层 on ACDC Fog）
#   - 100 张 SegFormer 推理
#   - 固定边界 + 分位数边界
#   - 天空统计
#   - 可视化
python OPWA_v3/dcp分层实验/experiment_A.py

# 3. 运行实验 B（极端雾子集）
#   - 100 张 fog_score 计算（仅 CPU）
#   - 20 张 SegFormer 推理
python OPWA_v3/dcp分层实验/experiment_B.py

# 4. 或一键运行
python OPWA_v3/dcp分层实验/run_all.py
```

输出均写入 `/root/opwa_v3/exp1_outputs/`（`diag_` 前缀）。

## 预期耗时

| 步骤 | 耗时 | GPU 需求 |
|:----|:----|:--------|
| dry-run | < 5s | 否 |
| 实验 A (100张) | ~55s A100 | ✅ |
| 实验 B (fog_score) | ~10s CPU | 否 |
| 实验 B (20张推理) | ~12s A100 | ✅ |

## 验证方法

### 验证 DCP 计算合理性

```python
python -c "
import cv2, numpy as np
from OPWA_v3.dcp分层实验.config_dcp import compute_dcp

# 读取一张 ACDC Fog 图像
img = cv2.imread('/gz-data/ACDC/rgb_anon_trainvaltest/rgb_anon/fog/val/GOPR0476/GOPR0476_frame_000761_rgb_anon.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dcp = compute_dcp(img_rgb)
print(f'DCP: min={dcp.min():.4f}, mean={dcp.mean():.4f}, max={dcp.max():.4f}')
print(f'Sky region DCP (top 1/3): {dcp[:360].mean():.4f}')
print(f'Ground region DCP (bottom 1/3): {dcp[-360:].mean():.4f}')
# 预期: sky DCP > ground DCP（天空亮度高导致 DCP 偏大）
"
```

### 验证 q_hat

```python
python -c "
# 验证 SegFormer 在纯白图像上的输出
from model_loader import load_segformer
from PIL import Image
import numpy as np
import torch

model, processor = load_segformer()
img = Image.new('RGB', (2048, 1024), color='white')
inputs = processor(images=img, return_tensors='pt').cuda()
with torch.no_grad():
    logits = model(**inputs).logits
print(f'logits shape: {logits.shape}')
print(f'q_hat = 0.513809')
"
```

## 输出文件清单（每次运行）

| 文件 | 必要条件 |
|:----|:--------|
| `diag_A_dcp_acdc_fixed.json` | 实验 A 完成 |
| `diag_A_dcp_acdc_quantile.json` | 实验 A 完成 |
| `diag_A_sky_stats.json` | 实验 A 完成 |
| `diag_A_dcp_coverage_fixed.png` | 实验 A 完成 |
| `diag_A_dcp_coverage_quantile.png` | 实验 A 完成 |
| `diag_A_dcp_vis.png` | 实验 A 完成 |
| `diag_B_extreme_fog_list.json` | 实验 B fog_score |
| `diag_B_extreme_fog_coverage.json` | 实验 B 完成 |
| `diag_B_fog_scores_hist.png` | 实验 B 完成 |

## 已知的坑

| 问题 | 现象 | 修复 |
|:----|:----|:-----|
| `from config import Q_HAT` 冲突 | ImportError | 本地配置用 `config_dcp.py` 避免和 exp0/config 重名 |
| ACDC GT 格式 | 不用 LABEL2TRAIN | `_gt_labelTrainIds.png` 直接用 trainID |
| ACDC 分辨率 | 1920×1080 非 2048×1024 | `ACDC_TARGET_H, ACDC_TARGET_W` 需设置正确 |
| DCP 方向 | DCP↑ = 雾浓 ≠ t(x) | 直接使用 DCP 值作分层，汇报时标注方向 |
| ACDC 图像数 | 100 张 | 用户指令中 106 为近似值，实际为 100 张 |
