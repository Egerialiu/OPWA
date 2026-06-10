# DCP 分层诊断实验

## 分支信息

- **分支名**: `dcp分层实验`
- **基于**: `exp1-pscp`
- **创建日期**: 2026-06-10
- **状态**: 诊断阶段（不修改 exp0/exp1 任何代码）

---

## 任务背景

实验 0 发现 Foggy Cityscapes（合成雾）存在 Coverage Collapse（bin0 gap=14.9pp），但 ACDC Fog（真实雾）的 Depth→透射率分层显示各 Bin 覆盖率均 ≥ 90%（gap=0）。

核心假说：**Depth Anything V2 在真实雾场景存在域偏移（Domain Shift），导致透射率分层失效，掩盖了真实存在的 Coverage Collapse。**

本任务通过 2 个诊断实验验证这个假说，为 AAAI-27 论文锁定方向。

---

## 实验列表

| 实验 | 数据 | 分层变量 | 目的 |
|:----|:----|:---------|:-----|
| **A** | ACDC Fog (100张) | DCP（暗通道先验） | 检测 Depth 分层是否掩盖了真实 gap |
| **B** | ACDC Fog top-20 极端雾 | 无（整体统计） | 检验"雾不够浓"假说 |

- ACDC Night（实验 C）**跳过并注明**：ACDC 数据集仅含 fog 条件，无 night/rain/snow 数据。

---

## 核心参数（固定不变）

| 参数 | 值 |
|:----|:----|
| 分割模型 | SegFormer-B0 (Cityscapes finetuned, 冻结) |
| q_hat | 0.513809（Cityscapes 校准，不重新计算） |
| 目标覆盖率 | 0.90 |
| Score | s(x,y) = 1 - softmax[GT类别] |

---

## 决策树

```
DCP on ACDC Fog → 最大分层 gap >= 10pp？
├── YES → 主线：混合物理先验的 PS-CP（DCP + Depth 的 max 融合）
├── NO → 极端雾子集 gap >= 10pp？
│     ├── YES → 主线（需补充更大规模真实雾数据）
│     └── NO  → 备选：自适应预测集压缩（Efficiency）
└── 所有输出 → 汇报给决策模型
```

---

## 严格约束

- ❌ 不修改 `OPWA_v3/exp0/` 或 `OPWA_v3/exp1/` 中的任何文件
- ❌ 不写 PS-CP 校准代码
- ❌ 不写 Baseline 对比代码
- ❌ 不修改 q_hat
- ✅ 可 import 复用 `exp0/config.py`, `exp0/model_loader.py` 等
