# OPWA Project — Claude Code Guidelines

## 快速验证阶段

当前处于 A1 快速验证阶段。核心目标是用最短时间判断"方案是否有效"，而非追求收敛。

1. **训练 200-500 步即可判定** — Gate 值前 100 步若无漂移（sigmoid 输出变化 <0.001），直接判定方向无效
2. **优先看 Gate + Loss 趋势**，不等 eval 结束 — Gate 漂移方向 + Loss 下降趋势比最终 mIoU 更重要
3. **双 GPU 同时跑对比实验时，提前判定** — 一个方向明显无效就杀掉，GPU 给另一个方向
4. **每次实验不超过 5-10 分钟**除非有明显的前向趋势,做快速失败的快速验证实验

## 实验记录规则

### 每做完实验立即写入分析
每次完成实验（训练、评估、消融）后，必须立即更新 `docs/a1实验报告.md`：

1. **更新实验状态表**（报告顶部）— 将完成的实验标记为 ✅
2. **添加实验记录日志**（报告 §八）— 日期、操作、结果、关键发现
3. **更新核心贡献点评估**（报告 §四）— 用证据等级标记每个子问题
4. **记录实验配置**（报告 §七附录）— 所有超参数、数据路径、硬件配置

### 证据等级规范

| 标记 | 含义 | 何时使用 |
|:----|:-----|:---------|
| ✅ Hard Data | 数值化、可重现的实验结果 | 评估完成并产出 JSON |
| ⚡ Code Review | 代码审查确认实现正确 | 架构验证经过 |
| ⚠️ Partial | 部分数据但不足以定论 | 实验有结果但有不确定性 |
| ❌ No Data | 尚未验证 | 实验未运行 |
| 🔴 Observed | 已观察到现象但原因不明 | 需要进一步分析 |

### 实验失败时记录什么
1. 失败现象（日志错误、数值异常、OOM）
2. 根因分析（为什么失败）
3. 修复方式（具体修改哪些文件）
4. 对后续架构的影响

## 代码规范

### 关键问题标记
遇到未实现的逻辑时，在代码中添加 `# FIXME: <问题描述>` 或 `# TODO: <待办项>`

### 保存 checkpoint
只保存 LoRA adapter + gate + projection 参数，不保存冻结的 UNet/VAE 参数（从 3.6GB → ~50MB）

## 项目结构
```
OPWA/
├── docs/
│   ├── OPWA和TPSWA实验方案.md    # 主实验方案文档
│   ├── a1实验方案.md              # A1 快速验证方案
│   └── a1实验报告.md              # A1 实验报告（持续更新）
├── scripts/
│   ├── train_opwa_a1.py          # A1 训练脚本
│   ├── eval_opwa_a1.py           # A1 评估脚本
│   └── eval_a1_compare.py        # A1 对比评估
├── opwa/
│   ├── models/
│   │   ├── opwa_a1.py            # OPWA A1 主模型（含 BranchProjection）
│   │   ├── degradation_encoder.py # 退化编码器
│   │   ├── gate.py               # StaticGate
│   │   └── lora.py               # LoRAWrapper
│   ├── training/
│   │   ├── trainer.py            # OPWATrainer（双阶段训练）
│   │   └── dataset.py            # WeatherDataset
│   ├── losses/
│   │   ├── reconstruction.py     # L2 + LPIPS
│   │   └── perception.py         # PerceptionDrivenLoss
│   └── evaluation/
│       └── metrics.py            # Evaluator, CKAAnalyzer
└── configs/
    └── default.yaml              # 默认配置
```
