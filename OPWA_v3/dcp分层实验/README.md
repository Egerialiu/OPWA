# DCP 分层诊断实验

## 分支信息

- **分支名**: `dcp分层实验`
- **基于**: `exp1-pscp`（commit `a1b6477`）
- **创建日期**: 2026-06-10
- **状态**: 实验 A+B 已完成

---

## 📚 文档索引

| 文档 | 内容 |
|:----|:-----|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 代码架构、模块设计、路径约定 |
| [PLAN.md](PLAN.md) | 方案设计思路、决策树、讨论记录 |
| [CONFIG.md](CONFIG.md) | 配置参数详解、数据规格 |
| [RESULTS.md](RESULTS.md) | 实验结果、数字、图表 |
| [ANALYSIS.md](ANALYSIS.md) | 分析解读、与外部评估的对话、方向判定 |
| [REPRODUCE.md](REPRODUCE.md) | 可复现指南 |

---

## 任务背景

实验 0 发现 Foggy Cityscapes（合成雾）存在 Coverage Collapse（bin0 gap=14.9pp），但 ACDC Fog（真实雾）的 Depth→透射率分层显示各 Bin 覆盖率均 ≥ 90%（gap=0）。

核心假说：**Depth Anything V2 在真实雾场景存在域偏移，导致透射率分层失效，掩盖了真实存在的 Coverage Collapse。**

本任务通过 DCP（暗通道先验）代替 Depth 作为分层变量来验证这个假说。

---

## 核心结论

两个诊断实验均未发现 ≥10pp 的 Coverage Gap：

| 实验 | 最大 gap | 方向 |
|:----|:--------|:-----|
| A: DCP on ACDC Fog（固定边界） | **5.96pp** | 反向（清晰区更低） |
| A: DCP on ACDC Fog（分位数边界） | **5.24pp** | 反向 |
| B: 极端雾子集（top-20） | **0pp** | — |

详细分析见 `ANALYSIS.md`。**简短判定：Depth 分层并未掩盖 ACDC 上的真实 gap——ACDC Fog 数据集本身对 SegFormer-B0 不产生 Coverage Collapse。**

---

## 严格约束（本分支全程遵守）

- ❌ 不修改 `OPWA_v3/exp0/` 或 `OPWA_v3/exp1/` 中的任何文件
- ❌ 不写 PS-CP 校准代码
- ❌ 不写 Baseline 对比代码
- ❌ 不修改 q_hat（固定 0.513809）
- ✅ 可 import 复用 `exp0/config.py`, `exp0/model_loader.py` 等
