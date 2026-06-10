# CLAUDE.md — Coverage Collapse 实验项目

## 核心目标

验证"恶劣天气下语义分割的 Conformal Prediction 会产生 Coverage Collapse"现象，并提出 PS-CP（Physics-Stratified Conformal Prediction）修复方案。用实验 0-3 的结果投稿 AAAI-27。

**这不是提升 mIoU/跑排行榜/复现 OPWA 的项目。所有代码决策服务于验证 Coverage Collapse 并提出修复方案。**

---

## 代码组织规范（重要）

- **所有实验代码**写在 `/OPWA_v3/` 目录下
- 可以 import/复用其他文件夹的代码（如 `/root/OPWA/`），但**不能在其他文件夹下修改任何文件**
- 输出文件统一放在 `/root/opwa_v3/expX_outputs/` 目录下（exp0_outputs/, exp1_outputs/）

---

## 身份与职责

你是本项目的**代码执行模型**。严格按照本文档规格编写实验代码并汇报结果。

方向决策由上级模型负责。你只负责：写代码 → 跑实验 → 按格式输出结果。自动执行下一步，无需等待确认。

---

## 环境规格

### 硬件
- 可在需要时开启 GPU（A100），当前可能有 GPU 驱动但未分配
- CUDA 11.7
- Python 3.10
- Conda 环境：`img2img-turbo`（已激活）

### 关键包
torch==2.0.1, torchvision==0.15.2, transformers==4.35.2, diffusers==0.25.1
accelerate==1.13.0, opencv-python==4.6.0.66, matplotlib==3.10.9
timm==1.0.27, einops==0.8.2, numpy, Pillow

### 禁止操作
- ❌ `pip install --upgrade transformers`
- ❌ 升级任何已安装包
- ❌ 安装 `mmcv`、`mmsegmentation`
- ❌ 用 `transformers` 加载深度估计模型

### 代码文件开头必加
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 数据路径（实际检测值）

| 规格路径 | 实际路径 |
|:--------|:--------|
| `/data/cityscapes/` | `/gz-data/cityscapes/`（通过 symlink） |
| `/data/foggy_cityscapes/` | `/gz-data/foggy_cityscapes/`（通过 symlink） |
| Cityscapes gtFine | `/gz-data/cityscapes/gtFine/val/`（gtFine/val → ../val 软链） |

### 数据集规格

```
校准集：Cityscapes 晴天 val set 前 250 张（文件名字母序排序后取前 250）
         frankfurt 267 + lindau 59 + munster 174 = 500 张中取前 250
测试集：Foggy Cityscapes val set beta=0.02，全部 500 张
        frankfurt 267 + munster 174 + lindau 59
GT：Cityscapes gtFine_labelIds.png（文件名去掉 _foggy_beta_0.02 后缀）
分辨率：原始 2048×1024
类别：19 类 Cityscapes，忽略标签 255

注意：Foggy Cityscapes 没有自己的 GT，复用 cityscapes gtFine。
      GT 文件名去掉 _foggy_beta_0.02 后缀后匹配。
      两者城市结构完全一致（都是 frankfurt/lindau/munster 三城）。
```

---

## 模型加载规格

### SegFormer（主分割模型，已缓存）
```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
).eval().cuda()
processor = SegformerImageProcessor.from_pretrained(
    'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
)
```
输出 logits (B, 19, H/4, W/4)，需双线性插值回原图大小。

### Depth Anything V2（透射率估计）
```python
import sys
sys.path.insert(0, '/OPWA_v3/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
depth_model = DepthAnythingV2(**model_configs['vits'])
depth_model.load_state_dict(torch.load('/OPWA_v3/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
depth_model = depth_model.eval().cuda()
```

**注意**：DA v2 Small 输出是**视差风格**（值越大=越近），透射率计算时必须反转。

### Depth Anything 输出方向验证（必须做）
```
取一张 Foggy Cityscapes 图像：
- 截取上 1/3 区域（天空/远景）和下 1/3 区域（近景地面）
- 计算 depth_raw 归一化均值：上 1/3 应 < 下 1/3（因视差风格，近处值大）
- 若上 > 下，说明无需反转（非视差风格），去掉 d_distance = 1.0 - d_norm 步骤
- 可视化用 matplotlib colormap='jet' 验证方向
```

---

## 透射率计算

```python
BETA = 3.0

def compute_transmittance(depth_raw):
    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max - d_min < 1e-6:
        return np.ones_like(depth_raw) * 0.5
    d_norm = (depth_raw - d_min) / (d_max - d_min)  # [0, 1]
    d_distance = 1.0 - d_norm  # 反转：远景大、近景小
    t = np.exp(-BETA * d_distance)  # 远景 ~0.05，近景 ~1.0
    return t
```

---

## 实验 0：Coverage Collapse 验证

### CP 设置
- 方法：Split Conformal Prediction
- 目标覆盖率：1 - α = 0.9（α = 0.1）
- Score：s(x,y) = 1 - softmax(logit)[y_gt]（像素级）
- q_hat：np.quantile(all_scores_on_calibration, 0.90)
- 预测集：C(x) = {c : softmax[c] >= 1 - q_hat}

### Bin 划分
```python
BIN_EDGES = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_NAMES = [
    "bin0_t0.00_0.20",   # 浓雾区，预期 Coverage Collapse
    "bin1_t0.20_0.40",   "bin2_t0.40_0.60",
    "bin3_t0.60_0.80",   "bin4_t0.80_1.00",  # 清晰区
]
```

### 输出文件
```
exp0_outputs/
  exp0_results.json       ← 决策树触发文件
  exp0_coverage_gap_plot.png
  exp0_transmittance_vis.png
  exp0_score_distribution.png
  exp0_run.log
```

### exp0_results.json 格式（固定键名，不可更改）
```json
{
  "dataset": "foggy_cityscapes_beta0.02",
  "alpha": 0.1,
  "q_hat": 0.0,
  "calibration_pixel_count": 0,
  "calibration_coverage": 0.0,
  "bins": {
    "bin0_t0.00_0.20": {
      "pixel_count": 0, "covered_count": 0,
      "coverage_rate": 0.0, "gap": 0.0,
      "mean_set_size": 0.0, "mean_score": 0.0
    },
    "bin1_t0.20_0.40": { "...": "同上格式" },
    "bin2_t0.40_0.60": { "...": "同上格式" },
    "bin3_t0.60_0.80": { "...": "同上格式" },
    "bin4_t0.80_1.00": { "...": "同上格式" }
  },
  "overall_test_coverage": 0.0,
  "max_gap": 0.0,
  "bin0_gap": 0.0,
  "bin0_pixel_ratio": 0.0
}
```
其中：
- gap = 0.90 - coverage_rate（正值表示未达标）
- bin0_pixel_ratio = Bin 0 像素数 / 全部有效像素数
- mean_set_size = 该 Bin 内 |C(x)| 的平均值
- mean_score = 该 Bin 内 nonconformity score 的平均值

### 逐像素推理流程（对测试集每张图）
```
1. 加载原始图像 → SegFormer 推理 → logits → softmax → resize 到原图大小 → (H,W,19)
2. 加载原始图像（BGR）→ Depth Anything 推理 → depth_raw → compute_transmittance() → t_map (H,W)
3. 加载 GT → gt_map (H,W), uint8
4. 对每个像素 (i,j):
   if gt_map[i,j] == 255: skip
   score = 1 - softmax[i,j, gt_map[i,j]]
   covered = (gt_map[i,j] in C(x_pixel))
   t = t_map[i,j] → 归入对应 Bin → 累计 pixel_count, covered_count, score_sum
5. rollup: 所有图累计，计算各 Bin 的 coverage_rate/gap/mean_set_size
```

### 图表规格

**exp0_coverage_gap_plot.png**：
- 横轴：5 个 Bin（BIN_NAMES）
- 纵轴：覆盖率（0 到 1）
- 蓝色柱子 = 实际覆盖率，红色虚线 = 0.90
- 标题："Coverage Rate by Transmittance Bin (Standard Split CP, alpha=0.1)"

**exp0_transmittance_vis.png**：
- 测试集前 5 张图
- 每张并排：原始 foggy 图 | 透射率热图（colormap=jet）
- 热图上叠加分 Bin 边界（颜色条标注 0.2/0.4/0.6/0.8）
- 标题显示 mean_t 和 bin0_ratio%

**exp0_score_distribution.png**：
- 各 Bin 的 nonconformity score 分布（violin 或 box plot）

### 决策树
```
exp0_results.json → bin0_pixel_ratio >= 0.05?
├── NO  → Bin 0 样本不足，透射率计算可能有误
└── YES → bin0_gap >= 0.20?
      ├── YES → 现象显著！等待实验 1 指令
      └── NO  → bin0_gap >= 0.10?
            ├── YES → 现象偏弱，可能需换 ACDC 数据集
            └── NO  → 现象不显著，停止
```

---

## 执行顺序（不可跳步）

```
Step 1：模型加载验证
  a. SegFormer：在 Cityscapes 晴天 val set 随机 5 张图上推理，目测输出合理
     （道路/建筑/天空有色块，非全黑或全一种颜色）
  b. 可选：跑 mIoU（预期 ≥ 76%，若 < 70% 说明权重错误，停止）
  c. Depth Anything：输出深度图方向验证
  → 汇报两个模型是否正常加载

Step 2：透射率方向验证
  → 生成 exp0_transmittance_vis.png
  → 汇报近景 t 均值 vs 远景 t 均值（各取图像下 1/3 和上 1/3）

Step 3：CP 校准
  → Cityscapes 晴天 val 前 250 张计算 q_hat
  → 校准集覆盖率必须在 [0.88, 0.92] 之间
  → 若不在此范围，停止

Step 4：分层覆盖率评估（实验 0 主体）
  → Foggy Cityscapes val 全部 500 张推理
  → 输出 exp0_results.json + 3 张图
  → 触发决策树

每步完成后按格式汇报，无需等待确认自动执行下一步。
```

---

## 汇报格式

```
=== Step X 完成 ===
状态：[成功/警告/失败]
关键数字：[- 指标名：值]
异常（如有）：[描述]
输出文件：[文件名列表]
等待指令：[否，自动执行下一步]
```

---

## 常见错误预防

| 错误类型 | 具体表现 | 正确做法 |
|:--------|:--------|:--------|
| GT 对应错误 | Foggy Cityscapes 没有自己的 GT，文件名含 `_foggy_beta_0.02` | GT 路径去掉 `_foggy_beta_0.02` 后缀找对应 Cityscapes GT |
| 分辨率不对齐 | softmax_map 和 t_map 尺寸不同导致像素错位 | 两者都 resize 到 GT 的原始尺寸再做像素匹配 |
| 深度方向错误 | 透射率图近景偏低远景偏高（应该反过来） | 在 compute_transmittance 中先验证方向再反转 |
| 数据泄露 | 用 Foggy Cityscapes 图像参与校准 | 校准集只用纯 Cityscapes 晴天 val 图像 |
| 忽略标签未过滤 | GT=255 的像素参与了覆盖率计算 | 所有像素级计算前加 `valid_mask = (gt != 255)` |
| score 计算错误 | 用 argmax 类别的概率而不是 GT 类别的概率 | `score = 1 - softmax[gt_label]`，GT 标签索引 softmax |
| 内存溢出 | 500 张图一次性加载进 GPU | 逐张推理，结果累计到 CPU numpy 数组 |
| Bin 统计量级错误 | mean_set_size 接近 19（几乎所有类都在预测集里） | 检查 q_hat 是否过大；正常情况 mean_set_size 应在 1-5 之间 |

## AAAI CP+语义分割发表情况（搜索摘要）

搜索结果确认：
- **AAAI-26** 录用了 CP 的有序分类论文、CP 适应性定量论文
- **CVPR-25** 录用了 CP + 测试时增强的论文
- **ICCV-25** 录用了 ConformalSAM（CP + 分割）
- 共同点：**都有清晰的理论贡献，不只是"把 CP 套在应用上"**

## 研究方向参考文档

项目研究规划文档位于：
- `/OPWA_v3/research/实验交接文档_v2.md` — 实验 0 的完整交接规格
- `/OPWA_v3/research/出路B_AAAI27方案.md` — AAAI-27 投稿可行性分析

---

## 实验 1：PS-CP（Physics-Stratified Conformal Prediction）

### 代码路径
```
/root/opwa_v3/
├── OPWA_v3/exp0/             ← 冻结的 baseline（实验 0）
│   ├── config.py              [共享] 路径、参数、常量
│   ├── model_loader.py        [共享] SegFormer + Depth Anything
│   ├── transmittance.py       [共享] 透射率计算
│   ├── data_utils.py          [共享] 数据加载
│   └── gt_utils.py            [共享] GT 加载
├── OPWA_v3/exp1/             ← 实验 1-3（exp1-pscp 分支，新建）
│   ├── pscp_calibration.py     PS-CP 分层校准 + 诊断模式
│   ├── pscp_evaluation.py      PS-CP 测试集评估
│   ├── baselines.py            WCP + Temperature Scaling CP
│   ├── run_all.py              完整执行入口
│   └── visualization.py        对比绘图
├── exp1_outputs/              ← 实验 1-3 输出目录
```

### 核心算法：PS-CP 分层校准

校准阶段，对校准集 250 张晴天图逐像素计算 nonconformity score，并根据透射率 $t(x)$ 将像素归入对应 Bin。每层的 $\hat{q}_k$ 独立计算：

```python
for k in range(N_BINS):
    scores_k = all_scores_by_bin[k]
    n_k = len(scores_k)
    if n_k >= MIN_CALIB_PIXELS_PER_BIN:
        # 有限样本修正（Vovk 2005）
        idx = ceil((n_k + 1) * 0.90) / n_k
        q_hats[k] = np.quantile(scores_k, min(idx, 1.0))
    else:
        q_hats[k] = global_q_hat  # fallback
```

测试阶段，每个像素使用其所在 Bin 的 $\hat{q}_k$ 判断覆盖：

```python
bin_k = digitize(t_map[i,j], BIN_EDGES) - 1
covered = (score <= q_hats[bin_k])
```

### 诊断模式（必做先于实验 1）

```bash
python OPWA_v3/exp1/pscp_calibration.py --diagnose
# 输出：exp1_outputs/calib_bin_stats.json
```

检查校准集各 Bin 的像素分布。若 Bin 0 比例 < 1%，需切换到分位数分层或合并 Bin。

### 输出文件
```
exp1_outputs/
  calib_bin_stats.json          ← 诊断：校准集各 Bin 像素分布
  calib_per_image_t.json        ← 诊断：每张校准图 t_mean
  pscp_calibration.json         ← PS-CP 校准结果（各 Bin 的 q_hat）
  exp1_results.json             ← PS-CP 测试评估结果
  exp1_coverage_gap_plot.png    ← PS-CP 覆盖率柱状图
  exp1_comparison_plot.png      ← Standard CP vs PS-CP 对比图
  exp1_calib_histogram.png      ← 校准集 t 分布直方图
```

### exp1_results.json 格式
```json
{
  "dataset": "foggy_cityscapes_beta0.02",
  "method": "pscp",
  "alpha": 0.1,
  "q_hats": [0.0, 0.0, 0.0, 0.0, 0.0],
  "bins": {
    "bin0_t0.00_0.20": { "coverage_rate": 0.0, "gap": 0.0, ... },
    "bin1_t0.20_0.40": { "coverage_rate": 0.0, "gap": 0.0, ... },
    "bin2_t0.40_0.60": { "coverage_rate": 0.0, "gap": 0.0, ... },
    "bin3_t0.60_0.80": { "coverage_rate": 0.0, "gap": 0.0, ... },
    "bin4_t0.80_1.00": { "coverage_rate": 0.0, "gap": 0.0, ... }
  },
  "overall_test_coverage": 0.0,
  "max_gap": 0.0,
  "bin0_gap": 0.0
}
```

**预期**：各 Bin 的 coverage_rate 应接近 0.90（不再是标准 CP 的 0.78/0.87/0.90 梯度）。

---

## 实验 2：Baseline 对比

### Baselines
| 方法 | 实现 | 校准阶段 |
|:-----|:-----|:---------|
| Standard CP | exp0 已有结果 | 全局 q_hat（单阈值）|
| **PS-CP（你的方法）** | `exp1/pscp_calibration.py` + `evaluation.py` | 逐 Bin 独立 q_hat |
| Weighted CP | `exp1/baselines.py:run_weighted_cp()` | 用 t(x) 的密度比加权（Tibshirani 2019）|
| Temperature Scaling + CP | `exp1/baselines.py:run_temperature_cp()` | 最优温度 T scalse logits 后再算 q_hat |

### Weighted CP 实现细节
```python
# 1. 用直方图估计 p_cal(t) 和 p_test(t)
edges = linspace(t_min, t_max, 50)
cal_hist, _ = histogram(cal_t, bins=edges, density=True)
test_hist, _ = histogram(test_t, bins=edges, density=True)

# 2. 密度比
density_ratio = test_hist / max(cal_hist, 1e-10)

# 3. 每像素权重
weights = density_ratio[digitize(t_values, edges)]

# 4. 加权分位数
q_hat = weighted_quantile(scores, weights, 0.90)
```

### Temperature Scaling 实现细节
```python
# 1. 搜索最优温度 T ∈ [0.1, 10.0]
# 2. logits_scaled = logits / T
# 3. probs_scaled = softmax(logits_scaled)
# 4. 用缩放后的 softmax 计算 nonconformity score
# 5. 标准 Split CP 确定 q_hat
```

### 输出文件
```
exp1_outputs/
  exp2_wcp_results.json             ← Weighted CP 结果
  exp2_temperature_cp_results.json  ← Temperature Scaling + CP 结果
  exp2_baseline_comparison.png      ← 四方法对比图
```

---

## 实验 3：消融实验

### β 扫描（Ablation A）
```python
beta_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# 每个 β 运行完整的 PS-CP pipeline
# 输出：exp3_ablation_beta.json + 折线图
```

### Bin 数量扫描（Ablation B）
```python
bin_configs = [2, 4, 5, 10]
# 创建均匀边界，运行 PS-CP pipeline
# 输出：exp3_ablation_nbins.json + 折线图
```

### 输出文件
```
exp1_outputs/
  exp3_ablation_beta.json           ← β 扫描结果
  exp3_ablation_nbins.json          ← Bin 数扫描结果
  exp3_ablation_beta.png            ← β vs coverage gap 图
  exp3_ablation_nbins.png           ← N_bins vs coverage gap 图
```

---

## 执行顺序（分支 exp1-pscp）

```
Step 0：诊断校准集 Bin 分布
  → python OPWA_v3/exp1/run_all.py --step diagnose
  → 输出 calib_bin_stats.json
  → 若 bin0_ratio < 0.01，需切换到分位数分层

Step 1：PS-CP 校准 + 测试（实验 1）
  → python OPWA_v3/exp1/run_all.py --step exp1
  → 输出 exp1_results.json
  → 验证：各 Bin gap < 0.05

Step 2：Baseline 对比（实验 2）
  → python OPWA_v3/exp1/run_all.py --step exp2
  → 生成对比图 + 各方法 JSON

Step 3：消融实验（实验 3）
  → python OPWA_v3/exp1/run_all.py --step exp3
  → 生成 β 和 Bin 数的消融曲线

Step 4：全部重绘
  → python OPWA_v3/exp1/run_all.py --step all-plots

完整运行：
  → python OPWA_v3/exp1/run_all.py
```

## Git 分支策略
- `main` 分支：实验 0 冻结状态，不再修改
- `exp1-pscp` 分支：实验 1-3 的开发分支
- 输出目录 `exp0_outputs/` 和 `exp1_outputs/` 与代码分离，不跟踪 git
