# 实验结果（RESULTS）

## 实验 A：DCP 分层 on ACDC Fog

**运行时间**: 2026-06-10 ~11:42
**模型**: SegFormer-B0 (冻结), q_hat=0.513809
**数据集**: ACDC Fog val, 100张, 1920×1080

### 固定边界结果

| Bin | DCP范围 | 解释 | 像素数 | 像素占比 | 覆盖率 | Gap | Mean Score |
|:---|:--------|:----|------:|:--------:|:-----:|:---:|:---------:|
| 0 | [0.00, 0.10) | 最清晰 | 17,365,561 | 8.85% | **86.08%** | **3.92pp** | 0.165 |
| 1 | [0.10, 0.20) | 轻微雾 | 35,137,440 | 17.91% | **84.04%** | **5.96pp ←max** | 0.182 |
| 2 | [0.20, 0.30) | 中等 | 48,592,013 | 24.77% | 90.12% | 0 | 0.113 |
| 3 | [0.30, 0.50) | 较浓 | 28,496,169 | 14.53% | 91.64% | 0 | 0.103 |
| 4 | [0.50, 1.00] | **最浓雾** | 66,562,819 | **33.93%** | **96.88%** | 0 | 0.044 |
| **全体** | — | — | 196,154,002 | 100% | **91.19%** | — | 0.105 |

### 分位数边界结果

分位数基于全体有效像素 DCP 值：q20=0.165, q40=0.255, q60=0.365, q80=0.776

| Bin | DCP范围 | 像素占比 | 覆盖率 | Gap | Mean Score |
|:---|:--------|:--------:|:-----:|:---:|:---------:|
| qtile0 (最清晰20%) | [0.000, 0.165) | 19.39% | **84.76%** | **5.24pp ←max** | 0.177 |
| qtile1 | [0.165, 0.255) | 20.33% | 86.40% | 3.60pp | 0.154 |
| qtile2 | [0.255, 0.365) | 20.06% | 93.02% | 0 | 0.083 |
| qtile3 | [0.365, 0.776) | 19.73% | 92.43% | 0 | 0.097 |
| qtile4 (最浓雾20%) | [0.776, 1.00] | 20.50% | **99.03%** | 0 | 0.018 |

### 天空统计

| 指标 | 值 |
|:----|:---|
| sky_pixel_ratio | **32.98%**（有效像素中天空占比） |
| sky_DCP_mean | **0.7746**（远超 0.20 阈值） |
| 诊断 | **严重天空误判** |

### 图

- `diag_A_dcp_coverage_fixed.png` — 固定边界覆盖率柱状图
- `diag_A_dcp_coverage_quantile.png` — 分位数边界覆盖率柱状图
- `diag_A_dcp_vis.png` — 前 5 张原始图 + DCP 热图 + Bin 分配

---

## 实验 B：极端雾子集

**运行时间**: 2026-06-10 ~11:44
**筛选**: fog_score = DCP_mean × (1 - contrast_std_norm)

### 雾浓度分布

| 统计 | 值 |
|:----|:---|
| fog_score 范围 | 0.0000 - 0.5166（100张） |
| 阈值（top-20） | ≥ 0.3528 |
| 最浓雾文件 | `GOPR0476_frame_000761` (fog_score=0.517) |

### Top-5 vs Bottom-5

```
Top-5 foggiest:
  GOPR0476_frame_000761  DCP=0.550  contrast=0.218  fog_score=0.517
  GP010476_frame_000199  DCP=0.497  contrast=0.214  fog_score=0.484
  GOPR0476_frame_000798  DCP=0.507  contrast=0.217  fog_score=0.480
  GOPR0476_frame_000982  DCP=0.468  contrast=0.211  fog_score=0.468
  GP010476_frame_000255  DCP=0.513  contrast=0.223  fog_score=0.458

Bottom-5 clearest:
  GP020475_frame_000209  DCP=0.324  contrast=0.304  fog_score=0.059
  GP020475_frame_000179  DCP=0.271  contrast=0.304  fog_score=0.048
  GP020475_frame_000204  DCP=0.326  contrast=0.315  fog_score=0.027
  GP020475_frame_000199  DCP=0.336  contrast=0.317  fog_score=0.022
  GP020475_frame_000229  DCP=0.358  contrast=0.324  fog_score=0.000
```

### 覆盖率统计（top-20）

| 指标 | 值 |
|:----|:---|
| 图像数 | 20 |
| 有效像素 | 39,087,208 |
| **overall_coverage** | **91.47%** |
| **gap** | **0pp** |
| mean_score | 0.098（参考：ACDC全集=0.109，Foggy CS=0.239） |
| empty_set_rate | 1.62% |

**按单张 coverage 排序（最低→最高）**：
- `GOPR0476_frame_001002`: **81.85%**（最高 score=0.198）
- `GP010476_frame_000142`: **83.33%**（score=0.169）
- `GOPR0476_frame_000781`: **86.69%**（score=0.142）
- `GP010476_frame_000125`: **89.84%**（score=0.132）
- 其他 16 张均 ≥ 90%

### 图

- `diag_B_fog_scores_hist.png` — fog_score 分布直方图

---

## 实验 C：ACDC Night

**跳过** — ACDC 数据仅含 fog 条件，无 night/rain/snow。

---

## 输出文件清单

| 文件 | 大小 |
|:----|:----|
| `diag_A_dcp_acdc_fixed.json` | 1.8KB |
| `diag_A_dcp_acdc_quantile.json` | 1.7KB |
| `diag_A_sky_stats.json` | 0.3KB |
| `diag_A_dcp_coverage_fixed.png` | 68KB |
| `diag_A_dcp_coverage_quantile.png` | 68KB |
| `diag_A_dcp_vis.png` | 520KB |
| `diag_B_extreme_fog_list.json` | 23KB |
| `diag_B_extreme_fog_coverage.json` | 8KB |
| `diag_B_fog_scores_hist.png` | 38KB |
