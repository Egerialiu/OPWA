# 代码架构

## 目录结构

```
OPWA_v3/dcp分层实验/
├── README.md               ← 任务总览、文档索引
├── ARCHITECTURE.md          ← 本文
├── PLAN.md                  ← 方案设计、讨论记录
├── CONFIG.md                ← 配置参数详解
├── RESULTS.md               ← 实验结果
├── ANALYSIS.md              ← 分析解读
├── REPRODUCE.md             ← 复现指南
│
├── __init__.py              ← 空
├── config_dcp.py            ← DCP 专用配置、路径、DCP 计算函数、文件列表获取
├── experiment_A.py          ← 实验 A：DCP 分层 on ACDC Fog
├── experiment_B.py          ← 实验 B：极端雾子集分析
└── run_all.py               ← 统一入口
```

## 模块设计

### 依赖关系

```
run_all.py
  ├── experiment_A.py
  │     ├── config_dcp.py
  │     └── exp0/model_loader.py  ← 复用，不修改
  └── experiment_B.py
        ├── config_dcp.py
        └── exp0/model_loader.py  ← 复用，不修改

config_dcp.py
  └── exp0/config.py              ← 仅读 DEVICE, BETA, LABEL2TRAIN
```

### config_dcp.py — 本地配置

- 所有**固定不可变**参数：`Q_HAT`, `TARGET_COVERAGE`, `ALPHA`
- DCP 专用常量：`DCP_BIN_EDGES`, `DCP_BIN_NAMES`, `NUM_DCP_BINS`
- ACDC 数据路径
- `compute_dcp()` — DCP 计算函数（RGB→min_channel→erode）
- `get_acdc_fog_files()` — 文件列表获取

### experiment_A.py — 实验 A

**pipeline**：
1. 加载 SegFormer（只推理，不训练）
2. 对 100 张 ACDC Fog 逐张处理：
   - SegFormer → logits → softmax → (H,W,19)
   - RGB → DCP → (H,W)
   - GT → trainIDs（无需 LABEL2TRAIN）
   - 像素级：score = 1 - softmax[GT]，covered = score ≤ 0.513809
   - 按 DCP 值归 Bin
3. 累计两套分层（固定 + 分位数）
4. 天空统计
5. 可视化

### experiment_B.py — 实验 B

**pipeline**：
1. 计算 100 张的 fog_score = DCP_mean × (1 - contrast_std_norm)
2. 排序取 top-20
3. 加载 SegFormer，对 20 张推理
4. 无分层，直接全图统计覆盖率

## 与 exp0/exp1 的关系

```
exp0/          ← 冻结的 baseline（不修改）
  ├── config.py
  ├── model_loader.py      ← 复用
  ├── transmittance.py
  ├── data_utils.py        ← 仅作为参考
  ├── gt_utils.py          ← ACDC 不需要（直接 trainID）
  └── run_acdc.py          ← 代码风格参考

exp1/          ← PS-CP 实验代码（不修改）
  ├── pscp_calibration.py
  ├── pscp_evaluation.py
  ├── baselines.py
  └── ...

dcp分层实验/  ← 本分支（新增，独立）
  └── ...
```

## 数据流

```
输入: ACDC Fog (100张)
  │
  ├──→ SegFormer-B0 (冻结)
  │     └── softmax (H×W×19) ──→ per-pixel score
  │                                  │
  ├──→ RGB → DCP ──→ DCP map ──────→ bin assignment
  │                                  │
  ├──→ GT (trainIDs) ──────────────→ valid mask + label
  │                                  │
  └──────────────────────────────────→ covered = score ≤ q_hat
                                      │
                                      ▼
                               fixed bins + quantile bins
                               (pixel_count, covered, score_sum)
```

## 输出

所有文件写入 `exp1_outputs/`（与 exp1 共享输出目录，文件名前缀 `diag_` 区分）。

| 文件 | 来源 |
|:----|:-----|
| `diag_A_dcp_acdc_fixed.json` | 实验 A 固定边界 |
| `diag_A_dcp_acdc_quantile.json` | 实验 A 分位数边界 |
| `diag_A_sky_stats.json` | 实验 A 天空统计 |
| `diag_A_dcp_coverage_fixed.png` | 实验 A 覆盖率柱状图（固定） |
| `diag_A_dcp_coverage_quantile.png` | 实验 A 覆盖率柱状图（分位数） |
| `diag_A_dcp_vis.png` | 实验 A 前 5 张 DCP 可视化 |
| `diag_B_extreme_fog_list.json` | 实验 B 雾浓度排序 |
| `diag_B_extreme_fog_coverage.json` | 实验 B 极端雾覆盖率 |
| `diag_B_fog_scores_hist.png` | 实验 B fog_score 分布 |
