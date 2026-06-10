# 配置参数详解（CONFIG）

## 核心参数（固定不变）

| 参数 | 值 | 来源 |
|:----|:---|:-----|
| 分割模型 | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | HuggingFace Hub |
| 模型模式 | `eval()`，完全冻结 | — |
| q_hat | **0.513809** | Cityscapes 晴天 val 前 250 张校准 |
| 目标覆盖率 | **0.90** (1-α, α=0.1) | — |
| Nonconformity Score | s(x,y) = 1 - softmax[GT] | Split CP 标准 |
| 预测集 | C(x) = {c : softmax[c] ≥ 1 - q_hat} | Split CP 标准 |
| 环境 | img2img-turbo (conda), Python 3.10, CUDA 11.7 | — |

## 数据规格

### ACDC Fog

| 属性 | 值 |
|:----|:----|
| 图像路径 | `/gz-data/ACDC/rgb_anon_trainvaltest/rgb_anon/fog/val/*/*_rgb_anon.png` |
| GT 路径 | `/gz-data/ACDC/gt/fog/val/*/*_gt_labelTrainIds.png` |
| 图像数 | **100**（非 106，用户原始指令中的 106 是近似值） |
| 分辨率 | 1920×1080 |
| GT 格式 | 直接 trainID (0-18)，255=ignore，**不需要 LABEL2TRAIN 映射** |
| 城市/场景 | GOPR0476, GP010476, GP020475 |

### Foggy Cityscapes（参考）

| 属性 | 值 |
|:----|:----|
| 图像路径 | `/gz-data/foggy_cityscapes/leftImg8bit_foggy/val/*/*_foggy_beta_0.02.png` |
| GT 路径 | `/gz-data/cityscapes/gtFine/val/*/*_gtFine_labelIds.png`（去 `_foggy_beta_0.02` 后缀） |
| 图像数 | 500 |
| 分辨率 | 2048×1024 |
| GT 格式 | labelID (0-33)，**需要 LABEL2TRAIN 映射** |

## DCP 计算参数

```python
DCP_BIN_EDGES = [0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
# Bin0=clear (低DCP), Bin4=fog (高DCP)
WINDOW_SIZE = 15  # 形态学腐蚀窗口

def compute_dcp(rgb_norm, window_size=15):
    min_ch = np.min(rgb_norm, axis=2)
    dcp = cv2.erode(min_ch, np.ones((window_size,window_size), np.uint8))
    return dcp  # [0,1], 越大=雾越浓
```

## 两套分层策略

| 策略 | 边界 |
|:----|:-----|
| **固定边界** | `[0.0, 0.10, 0.20, 0.30, 0.50, 1.0]` |
| **分位数边界** | `q20/q40/q60/q80` 基于测试集全体有效像素 DCP 值 |

## 实验 B 筛选参数

```python
fog_score = DCP_mean × (1 - contrast_std_norm)
N_EXTREME = 20  # 取最多的 20 张
```

## 输出目录

所有结果写入 `/root/opwa_v3/exp1_outputs/`（`diag_` 前缀区分）。
