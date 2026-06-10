# 实验进展 — exp1-pscp 分支

> 日期：2026/06/10 | 分支：`exp1-pscp` | commit: `ba72cbe`
> AAAI-27 截稿：2026/07/28（剩余 7 周）

---

## 已完成

### 实验 0：Coverage Collapse 验证（main 分支）

**状态**：✅ 完成，结果冻结在 `main` 分支

| Bin | 实际覆盖率 | Gap |
|:----|:---------:|:---:|
| bin0 t∈[0.0,0.2) | 78.38% | 11.62pp |
| bin1 t∈[0.2,0.4) | 87.07% | 2.93pp |
| bin2 t∈[0.4,0.6) | 89.59% | 0.41pp |
| bin3 t∈[0.6,0.8) | 86.42% | 3.58pp |
| bin4 t∈[0.8,1.0] | 88.20% | 1.80pp |

**决策**：bin0_gap = 11.6pp ≥ 10pp → 现象存在，继续实验 1。

### 诊断：校准集 Bin 分布（exp1-pscp 分支）

**状态**：✅ 完成

校准集（250 张晴天 Cityscapes Vil）透射率分布严重偏斜：

| Bin | 校准集（晴天） | 测试集（雾天） |
|:----|:--------------:|:--------------:|
| bin0 t<0.2 | **88.07%** | 71.13% |
| bin1 0.2-0.4 | **11.69%** | 17.31% |
| bin2 0.4-0.6 | **0.23%**⚠️ | 8.19% |
| bin3 0.6-0.8 | **0.009%**⚠️ | 2.90% |
| bin4 0.8-1.0 | **0%**✗ | 0.47% |

关键数据：
- 校准集 t_mean = 0.109 ± 0.012，范围 [0.081, 0.173]
- **晴天图像的透射率几乎全压在 Bin 0 和 Bin 1**
- Bin 2-4 在校准集中几乎无样本，PS-CP 需要 fallback

> **意义**：晴天校准集在所有透射率 Bin 中都有像素（由于近景/远景差异），但高 t 区域（Bin 2-4）的像素极少。这意味着 PS-CP 只能对 Bin 0-1 做可靠的分层校准，其余 Bin 必须 fallback 到全局 q_hat。

### 实验 1：PS-CP 实现

**状态**：✅ 代码完成，一次运行完成但需后续调优

**PS-CP 算法**（`pscp_calibration.py`）：

```python
# 逐 Bin 独立计算 q_hat
for k in range(N_BINS):
    if n_pixels[k] >= MIN_CALIB_PIXELS_PER_BIN:
        raw_q = quantile(scores_k, ceil((n+1)*0.9)/n)
        q_hats[k] = max(raw_q, global_q_hat * FLOOR)  # FLOOR=0.5
    else:
        q_hats[k] = global_q_hat  # fallback
```

**校准结果**：

| Bin | Cal Pixeles | raw_q | floored to | Cal Cov |
|:----|:---------:|:-----:|:----------:|:-------:|
| bin0 | 395.8M | 0.570 | 0.570 | 90.0% |
| bin1 | 52.5M | 0.003 | **0.257** | 96.8% |
| bin2 | 1.0M | 0.014 | **0.257** | 97.0% |
| bin3 | 39.5K | 0.002 | **0.257** | 99.8% |
| bin4 | 0 | — | 0.514（global） | — |

**测试结果**：

| Method | Bin0 Cov | Bin1 Cov | Bin2 Cov | Bin3 Cov | Bin4 Cov | Max Gap |
|:-------|:--------:|:--------:|:--------:|:--------:|:--------:|:-------:|
| Standard CP | 78.38% | 87.07% | 89.59% | 86.42% | 88.20% | 11.62pp |
| **PS-CP** | **80.07%** | **83.03%** | **86.71%** | **82.28%** | **88.20%** | **9.93pp** |

Bin 0 提升 **1.69pp**（78.38% → 80.07%），但所有其他 Bin 略微下降（因为 q_hat 降低）。

> **分析**：PS-CP 对 Bin 0 有明确帮助，但提升幅度有限（1.7pp）。根本原因是校准集（晴天）和测试集（雾天）在同一透射率 Bin 内的 score 分布不同——校准集的像素来自晴天近景物体（模型极自信），而测试集对应的是雾天中距离物体（模型不确定）。**透射率分层本身不消除域偏移。**

### 实验 2：Baselines

**状态**：🔲 代码完成，尚未运行

文件：`OPWA_v3/exp1/baselines.py`
- `run_weighted_cp()` — Weighted CP（Tibshirani et al. 2019），使用 t(x) 的密度比加权
- `run_temperature_cp()` — Temperature Scaling + CP，搜索最优温度 T

**估算运行时间**：~15 分钟（校准 250 张 + 测试 500 张 × 2 个 baseline）

### 实验 3：消融

**状态**：🔲 代码框架完成，尚未运行

- β 扫描（1.5, 2.0, 2.5, 3.0, 3.5, 4.0）
- Bin 数量扫描（2, 4, 5, 10）

---

## 关键设计决策

### 1. q_hat floor 保护

`Q_HAT_FLOOR = 0.5`：每个 Bin 的 q_hat 不低于全局 q_hat 的 50%。

**动机**：校准集中某 Bin 的 score 分布可能"过于乐观"（晴天近景物体），直接使用 raw_q 会导致该 Bin 在测试集上覆盖率跌破标准 CP。floor 机制确保 PS-CP **不劣于** Standard CP 太多。

**代价**：floor 削弱了 PS-CP 在 score 分布对齐良好的 Bin（如 Bin 0）的优势——但它防止了 Bin 1-3 的灾难性 collapse。

### 2. 校准集不足时的 fallback

Bin 4 校准像素为 0 → 直接使用全局 q_hat。

### 3. 不使用 exp0 的 `config.py` 而是通过 sys.path 引用

所有 exp1 模块在文件头 `sys.path.insert(0, "../exp0")` 后 import exp0 的共享配置。不复制、不修改 exp0 文件。

---

## 待完成的工作

### 高优先级
1. **运行实验 2**：`python OPWA_v3/exp1/run_all.py --step exp2`（~15 min）
2. **运行实验 3**：`python OPWA_v3/exp1/run_all.py --step exp3`（~30 min）
3. **生成全部图表**：`python OPWA_v3/exp1/run_all.py --step all-plots`
4. **检查 PS-CP 的 floor 参数调节**：当前 Q_HAT_FLOOR=0.5 是启发式值，可尝试 0.3 或 0.7 观察 Bin 0 覆盖率变化

### 中等优先级
5. **分位数分层**：诊断已收集 quantile_edges（在 exp0_outputs/），尝试用分位数边界代替固定边界做 PS-CP
6. **ACDC Fog 数据集**：在 exp0 已有 run_acdc.py，将其整合到 exp1 的对比中

### 低优先级
7. **Weighted CP 的密度比可视化**：确认密度比估计是否合理
8. **论文图表美化**：统一调色板、字体、标注

---

## 输出文件索引

```
exp0_outputs/                      ← 实验 0（main 分支）
  exp0_results.json               标准 CP 结果
  exp0_coverage_gap_plot.png      覆盖率柱状图
  exp0_transmittance_vis.png      透射率可视化
  exp0_score_distribution.png     score 分布

exp1_outputs/                      ← 实验 1-3（exp1-pscp 分支）
  calib_bin_stats.json            校准集 Bin 分布诊断
  calib_per_image_t.json          每张校准图 t_mean
  pscp_calibration.json           PS-CP 校准结果（各 Bin q_hat）
  exp1_results.json               PS-CP 测试结果
  [待生成]
  exp1_coverage_gap_plot.png      PS-CP 覆盖率柱状图
  exp1_comparison_plot.png        Standard CP vs PS-CP 对比
  exp1_calib_histogram.png        校准集 t 分布直方图
  exp2_wcp_results.json           Weighted CP 结果
  exp2_temperature_cp_results.json Temp Scaling + CP 结果
  exp2_baseline_comparison.png    四方法对比图
  exp3_ablation_beta.json         β 扫描结果
  exp3_ablation_nbins.json        Bin 数扫描结果
  exp3_ablation_beta.png          β vs coverage gap
  exp3_ablation_nbins.png         N_bins vs coverage gap
```
