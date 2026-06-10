# 出路 B：AAAI-27 投稿可行性分析

> 日期：2026/06/10 → AAAI-27 截稿：2026/07/28（还有约 7 周）

---

## 核心判断

**能投，但有严格前提条件。**

AAAI-26 录用了 CP 的有序分类论文、CP 适应性定量论文
CVPR-25 录用了 CP + 测试时增强的论文
ICCV-25 录用了 ConformalSAM（CP + 分割）

但所有录用有一个共同点：**它们都有清晰的理论贡献，不只是"把 CP 套在应用上"**。

---

## 🚨 错误路径 vs 正确路径

| 错误路径（会被拒） | 正确路径（有录用机会） |
|:----------------|:------------------|
| 先跑 OPWA 实验，再在结果上套 CP 框架，声称"有理论保证" | 从 CP 的核心假设出发，发现它在雾天场景下**系统性失效**，提出修复机制，用 OPWA 的物理约束作为修复工具 |
| 把 $M_{phys}$ 包装成"conformal score" | 严格证明：为什么标准 CP 在 $t(x) < \tau$ 区域的 i.i.d. 假设被大气散射物理破坏，以及如何用透射率分层恢复保证 |
| 贡献 = "我们第一个把 CP 用于天气分割" | 贡献 = "我们发现了天气引起的 **coverage collapse** 现象，并提出第一个有物理解释的条件覆盖修复框架" |

---

## 核心论文命题

> **发现**：标准 Split Conformal Prediction 在恶劣天气语义分割中产生 **Coverage Collapse**——在浓雾区域（$t(x) < \tau$），实际覆盖率系统性地跌破用户设定的 $1-\alpha$ 保证，可低至 $1 - \alpha - \Delta_{fog}$，其中 $\Delta_{fog}$ 可达 **30–40 个百分点**。
>
> **原因**：大气散射使雾天图像分布偏离标定集分布，违反了 CP 的交换性假设（exchangeability）。偏移强度与透射率 $t(x)$ 直接相关，存在可被物理模型量化的结构。
>
> **贡献**：提出 **Physics-Stratified Conformal Risk Control (PS-CRC)**，利用大气散射模型将像素按 $t(x)$ 分层，在每层内单独校准，在 $t(x) < \tau$ 层强制弃权（abstention），在 $t(x) \geq \tau$ 层恢复统计保证。

这个命题的好处：**即使你的 mIoU 不是最高的，Coverage Collapse 这个现象本身就是贡献**。

---

## 实验 0：Coverage Collapse 的定量证明（论文地基）

用标准 Split CP 在 Cityscapes（晴天）上校准 SegFormer-B0，得到阈值 $\hat{q}$，然后在 Foggy Cityscapes / ACDC Fog 上测试。

把每张图的每个像素按估计透射率 $t(x)$ 分成 5 个区间，对每个 Bin 分别计算：
- **实际覆盖率** $\hat{\text{cov}}_k$：GT 类别落在预测集 $\mathcal{C}_{\hat{q}}(x)$ 内的比例
- **目标覆盖率**：$1-\alpha = 0.9$
- **覆盖缺口** $\Delta_k = (1-\alpha) - \hat{\text{cov}}_k$

**预期结果（需实验验证）**：

| 透射率区间 | 含义 | 预期实际覆盖率 | 覆盖缺口 |
|:---------|:-----|:-----------:|:-------:|
| $t \in [0.8, 1.0]$ | 近景清晰 | ~90% | ~0% |
| $t \in [0.6, 0.8)$ | 轻雾 | ~85% | ~5% |
| $t \in [0.4, 0.6)$ | 中雾 | ~75% | ~15% |
| $t \in [0.2, 0.4)$ | 重雾 | ~60% | ~30% |
| $t \in [0.0, 0.2)$ | 浓雾 | ~50% | ~40% ← **Coverage Collapse** |

---

## 实验 1：PS-CRC 三级分层机制

**第一级：透射率分层校准**

不用单一 $\hat{q}$，而是用分层校准集分别计算每个 Bin 的阈值：

$$\hat{q}_k = \text{Quantile}_{1-\alpha}\left(\{s(x_i, y_i) : x_i \in \text{Bin}_k\}\right)$$

其中 $s(x, y)$ 是 nonconformity score。

**第二级：物理硬弃权**

当 $t(x) < \tau_{abs}$（绝对弃权阈值，如 $\tau_{abs} = 0.15$）时，无论 $\hat{q}_k$ 为何，输出弃权：

$$\mathcal{C}(x_{ij}) = \begin{cases} \emptyset & \text{if } t(x_{ij}) < \tau_{abs} \\ \{c : s_c(x_{ij}) \leq \hat{q}_k\} & \text{if } t(x_{ij}) \in \text{Bin}_k \end{cases}$$

**第三级（可选）：自适应集大小控制**

在 $t(x) \in [\tau_{abs}, \tau_{uncertain}]$ 的过渡区，输出包含更多类别的"保守预测集"。

---

## 实验 2：对比基线

| 方法 | 概念 |
|:-----|:----|
| Standard Split CP（基线） | 全局单一阈值，不处理域偏移 |
| Weighted CP（Tibshirani 2019） | 用密度比重加权，处理 covariate shift |
| Temperature Scaling + CP | 先校准置信度，再做 CP |
| **PS-CRC（你的方法）** | 物理分层 + 强制弃权 |

**核心度量**：
- **分层覆盖率** $\hat{\text{cov}}_k$
- **平均预测集大小** $\bar{|\mathcal{C}|}$
- **弃权率** $r_{abs}$

**预期核心表格**：

| 方法 | $\hat{\text{cov}}_{t<0.2}$ | $\hat{\text{cov}}_{t\geq0.6}$ | $\bar{|\mathcal{C}|}$ | $r_{abs}$ |
|:----|:--------------------------:|:----------------------------:|:---------------------:|:---------:|
| Standard CP（目标 90%） | ~50% ❌ | ~90% ✅ | 2.1 | 0% |
| Weighted CP | ~65% ⚠️ | ~89% ✅ | 3.8 | 0% |
| Temp Scaling + CP | ~58% ⚠️ | ~90% ✅ | 2.4 | 0% |
| **PS-CRC（你的）** | **N/A（弃权）** ✅ | **~90%** ✅ | **1.8** | **~35%** |

---

## 实验 3：OPWA 的重新定位

作为 PS-CRC 的**覆盖增强前处理**：

$$\text{OPWA前处理} \xrightarrow{\text{降低域偏移}} \text{更小的 Coverage Collapse} \xrightarrow{\text{更少弃权}} \text{更高有效像素覆盖率}$$

**实验设计**：
- 条件 A：Baseline SegFormer + PS-CRC（无前处理）
- 条件 B：OPWA + PS-CRC（有前处理）
- 对比：弃权率 $r_{abs}$、$\hat{\text{cov}}_{t<0.4}$

**预期**：OPWA 前处理后，Coverage Collapse 的阈值从 $\tau \approx 0.2$ 下移到 $\tau \approx 0.1$。

---

## 理论部分：AAAI 需要的命题

### 命题 1（Coverage Collapse 的量化上界）

设 $P_{fog}$ 是雾天分布，$P_{clear}$ 是标定集（晴天）分布，$d_{TV}$ 是总变差距离。则：

$$\left|(1-\alpha) - \hat{\text{cov}}_{P_{fog}}\right| \leq 2 \cdot d_{TV}(P_{fog}, P_{clear})$$

其中 $d_{TV}$ 可以用透射率 $t(x)$ 的分布差异来估计：

$$d_{TV} \approx \frac{1}{2}\int |p_{fog}(t) - p_{clear}(t)| dt$$

### 命题 2（PS-CRC 的分层覆盖保证）

在每个分层 Bin $k$ 内，若校准集和测试集的透射率分布近似相同，则：

$$\mathbb{P}\left(Y_{ij} \in \mathcal{C}(X_{ij}) \mid t(X_{ij}) \in \text{Bin}_k\right) \geq 1 - \alpha - \epsilon_k$$

其中 $\epsilon_k$ 是 Bin $k$ 内剩余域偏移的误差项，随 Bin 变窄而趋近于 0。

---

## 时间线（7 周）

```
Week 1（6/10–6/16）：Coverage Collapse 实验
Week 2-3（6/17–6/30）：PS-CRC 实现与对比
Week 4（7/1–7/7）：OPWA 整合实验
Week 5-6（7/8–7/21）：写作（摘要截止7/21）
Week 7（7/22–7/28）：完善，全文投稿（7/28）
```

---

## 主要风险

| 风险 | 影响 | 应对 |
|:----|:-----|:----|
| Coverage Collapse < 10pp | 论文动机崩塌 | 先用 Foggy Cityscapes，再试 ACDC Fog；若都不显著则出路 B 不成立 |
| Weighted CP 接近 PS-CRC | 创新性被质疑 | 强调 Weighted CP 需要密度比估计（高维不稳定），PS-CRC 用物理先验替代学习 |
| 理论不够严格 | Reviewer 质疑 | 命题基于现有 CP 理论推论，不需要原创证明技巧 |

---

## 一句话总结

> **出路 B 投 AAAI-27 可行的前提是：实验 0 先做，且缺口 ≥ 20pp。核心贡献是"发现并修复了 CP 在物理可解释的域偏移下的系统性失效"。OPWA 退居为配套贡献。**
