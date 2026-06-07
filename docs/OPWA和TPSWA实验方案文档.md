# OPWA 与 TPSWA 实验方案文档

> 版本：v2.0 — 融合基线对照体系、数据集使用方案与三阶段验证节奏

---

## A. OPWA 应超越的基线方法（按层级划分）

### A.1 最底层基线

- **无前置模块（Raw）**：直接将恶劣天气图像喂给冻结的检测/分割网络，记录 `mIoU_raw` / `mAP_raw`。
- **纯手工或传统图像增强前置**：CLAHE、Gamma 校正、Retinex 低光增强、白平衡矫正等。这些方法无需训练，直接对退化图像做像素级操作后送入感知模型。

### A.2 中间层基线

- **单帧 All-in-One 恶劣天气恢复网络前置**：使用 TransWeather、AllWeather-Net、Restormer、NAFNet、MAXIM 等统一多退化恢复模型，在配对数据上预训练后作为前置模块，冻结感知模型评估恢复率。这是与 OPWA 最直接的可比基线。

### A.3 上层基线（Plug-and-Play / 感知驱动 SOTA）

- **PIT 类方法**（Perceptual Image Translation）：最小化退化与清晰图像在识别网络高层特征空间的距离，训练 translator。在相同感知模型和数据集上与 OPWA 对比，重点检验 PIT 在极端天气下的幻觉问题。
- **简化版"多专家 + 控制器"系统**（参考 JarvisIR）：选取 1–2 个专家网络（如去雨、去雾各一个）搭配简单路由器，作为多专家 plug-in 基线。
- **AllWeather-Net**：显式为语义分割任务设计的统一多天气增强网络，训练时直接优化 mIoU。与 OPWA 的"正交解耦"路径形成对比。

### A.4 性能上限参照

- **端到端增强-检测/分割模型**：IA-YOLO、D-YOLO、UniDet-D、YOLO-DER、SemOD 等。这些方法改动后端结构并重新训练全网络，代表"若允许修改感知模型"时的性能上界。需强调：OPWA **不触碰后端、只动前端**，而端到端方法需要改动模型结构并重新训练全网络。

---

## B. TPSWA 应超越的基线方法

### B.1 第一层基线

- **单帧 OPWA**（不使用任何时序信息）：将 OPWA 作为逐帧独立处理的基线，直接对比 TPSWA 的时序增益。
- **简单时序平滑**：对 OPWA 输出做像素级滑动平均或中值滤波；或在 backbone 某一层做 temporal pooling（如平均池化 3 帧特征）。

### B.2 第二层基线（视频去雨/去雾 SOTA）

- **CDUN**（含 SADE 模块）：代表性视频去雨深度网络。
- **CAWM-Mamba**：基于小波分解 + Mamba 类状态空间模块的视频恢复方法。

相较于在 RainSyn、Rain100H 视频版、NTURain 等数据集上报告 PSNR/SSIM 的原始论文，TPSWA 的对比应集中在**任务恢复率**指标而非纯重建质量。

### B.3 端到端时序感知模型（参照）

- BEV 阶段或 backbone 上直接做时序融合的多帧检测/分割模型（例如多帧检测网络）。作为参照上限，说明端到端方法与 TPSWA 在"冻结感知模型"前提下的本质差异。

---

## C. 数据集使用方案：快速验证 vs 正式实验

### C.1 快速验证阶段（第 1–2 周）

**OPWA 快速验证数据集**：

| 场景 | 数据集 | 说明 |
|:----|:-------|:-----|
| Rain | WeatherSynthetic Rain → Sunny（已用于 GPPI） | 400 对训练，31 对测试，含 albedo 通道 |
| Fog | **WeatherSynthetic Fog → Sunny** | 使用 WeatherSynthetic 中的雾天子集（替换 Foggy Cityscapes / RESIDE 作为快速验证） |
| Night | **WeatherSynthetic Night → Sunny** | 使用 WeatherSynthetic 中的夜间子集（替换 Dark Zurich / RESIDE 作为快速验证） |
| 通用 | Rain100H / Rain100L 等标准合成去雨数据集 | 补充单退化验证 |

> **变更说明**：相较于原版设计，快速验证阶段的 fog 和 night 场景统一使用 **WeatherSynthetic** 的 fog/night 子集，替换了 Foggy Cityscapes 和 RESIDE 的初始验证角色，以利用已有数据管线、降低第一周数据集准备开销。

**TPSWA 快速验证数据集**：

- 合成雨雪视频数据集：RainSynLight25 / RainSynComplex25
- 从 KITTI / Cityscapes 通过 GAN / 扩散模型合成雨雪视频片段（几十个序列，长度 20–30 帧，可有/无 ground truth）

### C.2 正式图像级 Benchmark（第 3–4 周起）

| 数据集 | 场景 | 用途 |
|:-------|:-----|:-----|
| ACDC | 真实恶劣天气语义分割（夜间、雨天、雾天、雪天） | 核心真实场景评测 |
| Foggy Cityscapes | 合成雾天，保留 Cityscapes 分割标签 | 雾天定量对比 |
| Dark Zurich | 白天/夜间 paired 或粗配对 | 夜间/低光恢复评估 |
| RainCityscapes | 合成雨天 Cityscapes | 雨天定量评估 |
| DAWN | 多天气复杂场景（真实世界） | 多样性验证 |
| S2R-Bench | "Same-to-Robust" 鲁棒性评估基准 | **统一评估协议，以恢复率为核心指标** |

### C.3 正式视频级 Benchmark（第 5–6 周）

| 数据集 | 说明 |
|:-------|:-----|
| BDD100K | 雨/雪天视频片段，带天气标签 |
| nuScenes、Waymo Open Dataset | 筛选雨/雾/夜子集，作为补充 |

### C.4 三阶段验证节奏

```
第一阶段（第 1–2 周）：WeatherSynthetic（rain/fog/night）、Rain100H/L、合成雨雪短序列
  → 结构设计与消融

第二阶段（第 3–4 周）：Foggy Cityscapes、RainCityscapes、少量 ACDC 子集
  → 初步任务恢复率评估，调整损失权重

第三阶段（第 5–6 周）：ACDC 全量、Dark Zurich、S2R-Bench、BDD100K 雨雪视频子集
  → 正式对标实验与消融
```

---

## D. 论文中 OPWA 的创新点陈述框架（三个贡献点）

### 贡献点一：结构解耦设计

将预训练生成模型解码器划分为**主干内容通路**和**辅助天气通路**，通过轻量 MultiScaleEncoder 和门控注入在 skip 连接处形成功能正交的子空间，缓解多损失梯度冲突。相比单分支 All-in-One 恶劣天气恢复网络，在相似参数预算下取得更高 mAP/mIoU 恢复率。

### 贡献点二：表示正交的系统分析

首次在 Plug-and-Play 天气前置场景中结合 **CKA**、**梯度频谱**和 **probe 实验**，证明主干与分支自发形成"结构/语义"与"天气/纹理"的功能分工。架构偏置本身（即使分支输入全零或随机权重冻结）即可带来大部分任务性能增益，超过 PIT 类特征对齐 translator 在特征纠缠和不稳定性上的表现。

### 贡献点三：任务恢复率评估体系

在自动驾驶极端天气感知场景下，首次以**任务恢复率**（mAP/mIoU 相对于干净天气的恢复百分比）系统评估 Plug-and-Play 前置模块。证明 OPWA 在不改动已部署检测/分割模型的前提下，显著优于无前置、简单增强、单分支 All-in-One 和 PIT 风格 translator，部分场景接近或超越端到端增强-检测模型。

---

## E. TPSWA 的定位与相对创新（简述，用于论文）

1. **不追求**长序列、重模型的 PSNR/SSIM 极致，而是围绕**"短窗 + 低延迟 + Plug-in"** 现实约束，使用状态空间结构而非重型 3D CNN 或 Transformer。

2. 与 OPWA 形成**结构互补**：OPWA 处理静态退化（雾、低光），TPSWA 通过时间维背景/退化状态解耦处理动态退化（雨、雪）。

3. 训练目标**直接使用任务恢复率和时间稳定性指标**，而非仅优化视频 PSNR，更贴近自动驾驶感知需求。

---

## F. 工程实现中不可遗漏的基线对照与评估细节

### F.1 评估指标报告规范

评估时需同时报告：

| 符号 | 含义 |
|:----|:-----|
| `mIoU_raw` | 退化图像直接推理 |
| `mIoU_OPWA` | 经 OPWA 翻译后推理 |
| `mIoU_clean` | 晴天图像推理（性能上限） |

并计算恢复率：

$$r_{\text{mIoU}} = \frac{\text{mIoU\_OPWA}}{\text{mIoU\_clean}} \times 100\%$$

$$r_{\text{mAP}} = \frac{\text{mAP\_OPWA}}{\text{mAP\_clean}} \times 100\%$$

### F.2 TPSWA 时序稳定性指标

$$\text{Stability} = 1 - \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{|\text{mIoU}_t - \text{mIoU}_{t+1}|}{\text{mIoU}_{\text{clean}}}$$

### F.3 雨雪视频专项统计

在人工标注的"被雨丝遮挡的行人"子集上比较召回率，额外统计：
- **漏检率**（因雨雪遮挡导致）
- **假阳性率**变化

### F.4 端到端增强-检测模型的标注差异

当以端到端增强-检测模型（IA-YOLO、D-YOLO 等）作为上限参照时，需在论文中明确标注：

> **关键差异**：OPWA **不触碰后端、只动前端**——保持下游感知模型完全冻结且不改动其结构。而端到端方法需要改动模型结构并重新训练全网络。OPWA 与端到端方法的对比本质上是"零成本即插即用"与"全量再训练"之间的对比，两者服务于不同的部署约束场景。

---

## G. 设计总纲：从 GPPI 实验结论出发的设计原则

在展开具体框架之前，先明确从 GPPI 实验中提炼出的五条核心设计原则，它们将贯穿 OPWA 和 TPSWA 的每一个架构决策。

<details>
<summary><strong>原则 1：架构偏置 > 信号内容（zero probe 恢复 152%）</strong></summary>

GPPI 实验中最令人意外的发现是：全零输入的 zero probe 仍能恢复 Baseline 性能的 152%（mIoU 83.11% vs 73.09%），说明 **Add + Gate 的拓扑结构本身**——而非注入的具体信号——是梯度冲突缓解的主因。这意味着在设计 OPWA/TPSWA 时，不应过度纠结于分支输入的"信息量"，而应优先保证**架构层面的梯度流独立性**。分支输入的作用是"锦上添花"而非"雪中送炭"。

</details>

<details>
<summary><strong>原则 2：结构化输入有害（rank1 probe 仅恢复 53%）</strong></summary>

人为引入条纹结构的 rank1 噪声反而严重干扰主干，说明分支输入应保持**无结构或弱结构性**。如果分支输入过于结构化（如深度图、语义分割图），可能通过 Add 操作在 skip 连接处引入与主干冲突的强梯度信号。这一发现直接影响物理先验的注入方式：**物理先验应作为软约束（loss）而非硬输入（tensor）注入分支**。

</details>

<details>
<summary><strong>原则 3：频谱特性影响分工（pink ≈ albedo, 85.51% vs 85.89%）</strong></summary>

粉红噪声（能量集中低频）在 mIoU 上追平了 Albedo 输入，说明分支对输入的**频谱分布**比语义内容更敏感。这提示我们可以通过频域分解来显式引导主干/分支分工：主干处理低频结构，分支处理高频纹理或特定频段的天气残差。

</details>

<details>
<summary><strong>原则 4：Gate 的自适应层级分配（深层小、浅层大）</strong></summary>

所有实验一致显示 gate 值遵循"深层 ≈ 0.45、浅层 ≈ 0.49–0.67"的模式，且 Albedo 输入使浅层 gate 进一步升高至 0.665。这说明网络自发地将分支定位为**浅层纹理/细节补充器**。在 OPWA 中，可以将这一先验编码为 gate 初始化策略或正则约束。

</details>

<details>
<summary><strong>原则 5：随机权重网络具有表征能力（random_gppi 恢复 166%）</strong></summary>

冻结的随机初始化 MultiScaleEncoder 仍能带来显著增益（mIoU 83.44%），验证了 Ramanujan 网络假说。这意味着 GPPI 分支的**卷积拓扑结构**（多尺度下采样 + 投影）本身就构成一个有效的随机特征空间，训练中真正需要学习的参数集中在 Gate 和主干 LoRA 上。这为后续轻量化设计提供了理论支撑。

</details>

---

## H. OPWA：Orthogonal Plug-in Weather Adapter

### H.1 核心定位与问题定义

OPWA 的目标是：**在冻结的下游感知模型前方，插入一个轻量、可插拔、多分支解耦的单帧天气前置模块，使雨/雾/夜等退化图像经过翻译后，下游模型的 mAP/mIoU 恢复到接近晴天水平。**

与 GPPI 的关键差异在于：

| 维度 | GPPI（现有） | OPWA（目标） |
|------|------------|------------|
| 退化类型 | 单一（rain→sunny） | 多种（rain/fog/night/snow） |
| 分支输入 | Albedo（需预计算） | 无需额外物理输入，自适应退化 |
| 正交机制 | 纯结构解耦 | 结构解耦 + 轻量软正则 |
| 训练目标 | 图像重建 + GAN | 重建 + GAN + **感知驱动** |
| 评估体系 | mIoU | **恢复率** $r_{\text{mIoU}}$, $r_{\text{mAP}}$ |

### H.2 OPWA 主框架设计

#### H.2.1 整体架构

OPWA 继承 Pix2Pix-Turbo（SD-Turbo）的一步翻译范式，保留 VAE Encoder → UNet → VAE Decoder 的主干通路，在此基础上做三项关键扩展：

```
退化图像 I_deg (512×512)
  │
  ├──→ VAE Encoder ──→ z ──→ UNet (LoRA) ──→ denoised latent
  │                                              │
  │                                         VAE Decoder
  │                                              │
  │                                   ┌──────────┴──────────┐
  │                                   │  skip + OPWA_feats   │
  │                                   │  × σ(ConditionalGate)│
  │                                   └──────────┬──────────┘
  │                                              │
  │                                     恢复图像 I_rec
  │                                              │
  └──→ 轻量退化编码器 (Degradation Encoder)        │
       ┌─────────────────────────┐                │
       │ D-Enc: 3层下采样         │                │
       │ → 退化特征 f_deg (4尺度)  │                │
       │ → 退化嵌入 e_deg (全局)   │                │
       └────────┬────────────────┘                │
                │                                 │
                ├──→ 4尺度特征 → 注入 skip         │
                └──→ 全局嵌入 → 条件 Gate          │
                                                    │
                    ┌───────────────────────────────┘
                    │
              冻结感知模型 F_percept (SegFormer / YOLO)
                    │
              感知损失 L_percept → 回传至 OPWA
```

#### H.2.2 关键组件一：退化感知分支（Degradation-Aware Branch）

**设计理由**：GPPI 中分支输入是 Albedo（需要额外预计算且仅适用于 rain→sunny），但极端天气场景下需要**通用的、自适应的分支输入**。根据原则 2（结构化输入有害），不应直接将深度图、透射率图等强结构物理量作为分支输入；根据原则 3（频谱敏感性），分支应接收频域特性与退化相关的信号。

**方案**：设计一个轻量退化编码器（Degradation Encoder），从输入退化图像本身提取多尺度退化特征，而非依赖外部物理量。

**退化编码器结构**：
- 输入：退化图像 $I_{\text{deg}}$（$3 \times 512 \times 512$）
- 3 层下采样卷积（与 GPPI 的 MultiScaleEncoder 结构相同），产生 4 个尺度的特征 $f_{\text{deg}}^{(l)}$，$l \in \{1,2,3,4\}$
- 全局平均池化 + 2 层 MLP 产生退化嵌入 $e_{\text{deg}}$（256 维）

**梯度隔离机制**：

```
I_deg → D-Enc → f_deg^(l) → proj_l → σ(gate_l) → Add to skip_l
                                    ↑
                          StopGrad on f_deg^(l)
```

在反向传播时，对 $f_{\text{deg}}^{(l)}$ 施加 stop-gradient（`detach()`），使其仅作为**固定条件信号**注入，梯度只通过 gate 和主干回传。这样做的好处：
1. 退化特征不会与主干在 skip 处产生梯度冲突（因为梯度不流过分支 encoder）
2. 分支提供了一个**退化相关的随机特征空间**（类似 random_gppi 的有效表征），gate 学习选择哪些特征对恢复有用
3. 避免了 GPPI rank1 probe 中"结构化输入直接干扰主干"的问题

> **设计备选**：如果 stop-gradient 导致分支信息利用不足，可以改为**梯度缩放**（如乘以 0.1），允许少量梯度流回分支 encoder，在"信息利用"和"梯度冲突"之间取得平衡。

#### H.2.3 关键组件二：条件自适应 Gate（Weather-Conditional Gate）

**设计理由**：GPPI 中 gate 是 4 个标量参数，对所有输入图像使用相同的 gate 值。但在多种天气场景下，不同退化类型和程度需要不同的分支注入强度。例如：
- 浓雾场景：远处信息严重丢失，浅层 gate 应降低以避免幻觉
- 暴雨场景：局部遮挡为主，浅层 gate 应升高以修复纹理
- 夜间场景：全局低光，深层 gate 应升高以补充语义信息

**方案**：将 gate 从静态标量升级为**退化条件化的动态 gate**。

```
e_deg (256-d) → Gate MLP → [g_1, g_2, g_3, g_4] (4 scalars)
                              ↓
                    σ(g_l) × f_deg^(l) → Add to skip_l
```

Gate MLP 结构：256 → 128 → 4，使用 SiLU 激活，输出经 sigmoid 映射到 (0,1)。

**初始化策略**：基于 GPPI 实验观察到的"深层小、浅层大"先验，将 Gate MLP 最后一层的 bias 初始化为 $[-0.2, -0.1, 0.0, 0.2]$（对应 sigmoid 后约 $[0.45, 0.47, 0.50, 0.55]$），使训练从一个合理的起点开始。

**正则约束**：为防止 gate 在训练中出现极端值（全部关闭或全部打开），施加轻量正则：

$$
\mathcal{L}_{\text{gate}} = \lambda_g \sum_{l=1}^{4} \left( g_l - \bar{g}_l^{\text{prior}} \right)^2
$$

其中 $\bar{g}_l^{\text{prior}}$ 是 GPPI 实验中观察到的各层 gate 均值先验。

#### H.2.4 关键组件三：感知驱动损失（Perception-Driven Loss）

**设计理由**：GPPI 的训练目标是图像重建（L2 + LPIPS + CLIP + GAN），优化的是视觉质量。但 OPWA 的核心目标是**恢复下游感知性能**，需要引入感知驱动损失。PIT 的经验表明，直接最小化翻译前后在冻结感知网络中的特征差异是有效的，但在极端天气下容易产生幻觉。

**方案**：采用**混合损失**策略，将重建损失和感知损失以任务导向的方式结合：

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{rec}}}_{\text{重建}} + \underbrace{\lambda_p \mathcal{L}_{\text{percept}}}_{\text{感知驱动}} + \underbrace{\lambda_g \mathcal{L}_{\text{gate}}}_{\text{gate 正则}} + \underbrace{\lambda_o \mathcal{L}_{\text{ortho}}}_{\text{软正交}}
$$

**$\mathcal{L}_{\text{rec}}$（重建损失）**：与 GPPI 相同，L2 + LPIPS + GAN。这是保证图像不产生严重幻觉的"安全网"。

**$\mathcal{L}_{\text{percept}}$（感知驱动损失）**：

$$
\mathcal{L}_{\text{percept}} = \sum_{k} w_k \cdot \text{CE}\left( F_{\text{percept}}^{(k)}(I_{\text{rec}}),\; Y_{\text{gt}} \right)
$$

其中 $F_{\text{percept}}^{(k)}$ 是冻结感知模型的第 $k$ 个输出头（如 SegFormer 的分割头），$Y_{\text{gt}}$ 是 GT 标签。梯度通过 $I_{\text{rec}}$ 回传到 OPWA，但不更新 $F_{\text{percept}}$。

**关键决策**：$\lambda_p$ 的取值策略。

| 阶段 | $\lambda_p$ | 理由 |
|------|:----------:|------|
| 前 500 步 | 0 | 先让重建损失稳定图像输出，避免感知梯度在训练初期干扰 |
| 500–1500 步 | 线性升温至 $\lambda_p^{\max}$ | 逐步引入感知驱动 |
| 1500 步后 | $\lambda_p^{\max}$ | 全力优化感知恢复率 |

这种 warmup 策略避免了 PIT 在极端天气下的核心失效模式——在训练早期就通过感知损失鼓励幻觉。

#### H.2.5 关键组件四：轻量软正交正则（Soft Orthogonality Regularizer）

**设计理由**：GPPI 实验显示结构解耦自然导致 CKA < 0.1 的功能正交，无需显式正交损失。但在多天气场景下，退化编码器产生的特征可能与主干 encoder 的特征在某些天气类型上重新纠缠。需要一个**轻量**正交正则来维持分离。

**互相关惩罚**：

$$
\mathcal{L}_{\text{ortho}} = \frac{1}{L} \sum_{l=1}^{L} \frac{1}{C_l^2} \left\| \text{corr}\left(\phi_{\text{trunk}}^{(l)},\; \phi_{\text{branch}}^{(l)}\right) - \mathbf{I} \right\|_F^2
$$

其中 $\phi_{\text{trunk}}^{(l)}$ 和 $\phi_{\text{branch}}^{(l)}$ 分别是主干 skip 特征和分支注入特征在空间维度展平后的矩阵，$\text{corr}(\cdot,\cdot)$ 是通道间互相关矩阵。这一损失鼓励两个分支的特征在**通道维度**上去相关。

**权重设置**：$\lambda_o$ 取较小值（如 0.01–0.05），仅作轻度引导，避免过度碎片化。

### H.3 OPWA 的三个变体方案

#### 变体 A1：最小可行版（Minimum Viable OPWA）

**核心思路**：直接复用 GPPI 的 MultiScaleEncoder + Gate 架构，仅更换数据集和损失函数，不引入退化编码器和软正交正则。

| 组件 | 配置 |
|------|------|
| 分支输入 | 退化图像经 stop-gradient 的 D-Enc（或直接用高斯噪声，类似 GPPI noise 模式） |
| Gate | 4 个静态标量 |
| 损失 | $\mathcal{L}_{\text{rec}}$ + $\mathcal{L}_{\text{percept}}$（warmup） |
| 正交正则 | 无 |
| 参数量 | 与 GPPI 相同（~2M 分支参数） |
| 预计实现时间 | 1 周 |

**适用场景**：作为 baseline 验证"GPPI 架构直接迁移到多天气场景是否有效"。如果 A1 在 ACDC/Foggy Cityscapes 上已有显著提升，则论文的核心贡献已成立。

#### 变体 A2：标准版（Standard OPWA）

**核心思路**：在 A1 基础上加入退化编码器的条件 gate 和软正交正则。

| 组件 | 配置 |
|------|------|
| 分支输入 | 退化图像 → D-Enc（stop-gradient） |
| Gate | 条件 gate（退化嵌入 → MLP → 动态 gate） |
| 损失 | $\mathcal{L}_{\text{rec}}$ + $\mathcal{L}_{\text{percept}}$（warmup） + $\mathcal{L}_{\text{gate}}$ |
| 正交正则 | 互相关惩罚 $\mathcal{L}_{\text{ortho}}$ |
| 额外参数 | Gate MLP (~33K) + 正交正则无额外参数 |
| 预计实现时间 | 2 周 |

**适用场景**：作为论文的主要方法。条件 gate 在多天气场景下的自适应行为（如雾天浅层 gate 低、雨天浅层 gate 高）本身就是一个有吸引力的分析点。

#### 变体 A3：频域增强版（Frequency-Enhanced OPWA）

**核心思路**：在 A2 基础上，利用原则 3（频谱敏感性）的发现，对损失函数进行频域分解，显式引导主干/分支的频谱分工。

| 组件 | 在 A2 基础上新增 |
|------|-----------------|
| 频域损失 | FFT 分解 + 分频段 L2 |
| 频域正则 | 分支特征的 HF/LF 能量比约束 |
| 预计实现时间 | 2.5 周 |

**适用场景**：如果 A2 在某些天气类型上（如浓雾，低频信息严重丢失）表现不佳，A3 可以通过频域分解更精确地控制恢复行为。

### H.4 OPWA 训练策略

#### H.4.1 数据策略

| 数据源 | 类型 | 用途 |
|--------|------|------|
| WeatherSynthetic | Paired (rain/fog/night → sunny) | 已有，直接复用 |
| Foggy Cityscapes | Paired (foggy→clear, 合成) | 分割标签来自 Cityscapes |
| ACDC | 真实 (fog/rain/night + GT seg) | 无配对 clear，用作无监督一致性 |
| CARLA 合成 | Paired (多种天气→clear) | 可扩展，生成 snow/night 等 |

**混合训练**：每个 batch 按比例采样不同天气类型（如 rain 30%, fog 30%, night 20%, snow 20%），避免模型偏向某一天气。

#### H.4.2 两阶段训练

**阶段一：重建预训练（1000 步）**
- 仅优化 $\mathcal{L}_{\text{rec}}$ + $\mathcal{L}_{\text{gate}}$ + $\mathcal{L}_{\text{ortho}}$
- 目标：让多分支结构稳定收敛，gate 学到合理的层级分配
- 此阶段等价于 GPPI 的训练方式

**阶段二：感知驱动微调（1000 步）**
- 引入 $\mathcal{L}_{\text{percept}}$，$\lambda_p$ 从 0 线性升温至 $\lambda_p^{\max}$
- 冻结感知模型（SegFormer-b0 或 YOLO），计算分割/检测损失并回传
- 同时保持 $\mathcal{L}_{\text{rec}}$ 作为正则，防止感知驱动导致图像质量崩塌

### H.5 OPWA 评估体系

#### H.5.1 核心指标：恢复率

$$
r_{\text{mIoU}} = \frac{\text{mIoU}_{\text{OPWA}}}{\text{mIoU}_{\text{clean}}} \times 100\%
$$

$$
r_{\text{mAP}} = \frac{\text{mAP}_{\text{OPWA}}}{\text{mAP}_{\text{clean}}} \times 100\%
$$

同时报告：
- $\text{mIoU}_{\text{raw}}$（退化图像直接推理）
- $\text{mIoU}_{\text{OPWA}}$（经 OPWA 翻译后推理）
- $\text{mIoU}_{\text{clean}}$（晴天图像推理，上限）

#### H.5.2 分析工具（复用 GPPI 工具链）

| 工具 | 目的 |
|------|------|
| CKA 分析 | 验证主干/分支在多天气下的正交性 |
| 梯度余弦 | 验证感知损失引入后是否重新引发梯度冲突 |
| Gate 值分布 | 分析条件 gate 对不同天气的自适应行为 |
| Per-class mIoU | 检查安全关键类别（行人、车道线）的恢复情况 |
| 不确定性图 | 对 OPWA 输出计算 MC-Dropout 不确定性，标记不可靠区域 |

### H.6 OPWA 风险评估与缓解

| 风险 | 严重度 | 缓解策略 |
|------|:------:|----------|
| 感知驱动导致幻觉 | 高 | $\mathcal{L}_{\text{rec}}$ 作为安全网 + warmup 策略 |
| 条件 gate 训练不稳定 | 中 | Gate 初始化先验 + $\mathcal{L}_{\text{gate}}$ 正则 |
| 多天气混合训练导致负迁移 | 中 | 天气比例平衡 + 可选天气特定 LoRA adapter |
| 6 周内实验不完 | 高 | A1 作为保底方案，2 周内可出结果 |

---

## I. TPSWA：Temporal-Physical State-Space Weather Adapter

### I.1 核心定位与问题定义

TPSWA 不是 OPWA 的替代品，而是其**时序增强扩展**。它的目标场景非常明确：**在雨/雪等局部遮挡型、时变退化主导的视频场景中，通过短时序信息补全被遮挡的内容，提升 OPWA 单帧恢复的上限和时序稳定性。**

TPSWA 不试图解决大雾等全局退化问题（这超出了时序信息的能力边界），而是专注于雨雪场景中"前后帧可见区域互补"这一独特优势。

### I.2 TPSWA 主框架设计

#### I.2.1 整体架构

TPSWA 作为一个**前置时序模块**，插入在 OPWA 之前。整体流水线为：

```
视频帧序列 [I_{t-1}, I_t, I_{t+1}]
  │
  ├──→ TPSWA 时序模块 ──→ 增强帧 Ĩ_t
  │                          │
  │                    OPWA 单帧恢复 ──→ I_rec,t
  │                                        │
  │                                  冻结感知模型
  │                                        │
  └──→ 场景分类器 ──→ {rain/snow: 启用 TPSWA, fog/night: 跳过}
```

#### I.2.2 关键组件一：双流状态空间模块（Dual-Stream SSM）

**设计理由**：雨雪视频中的信息可以自然分为两类：
1. **背景流（Background Stream）**：场景的静态/慢变内容，在时间上高度一致
2. **退化流（Degradation Stream）**：雨丝/雪花等快速变化的遮挡模式，在帧间位置随机变化

**架构设计**：

```
I_t (当前帧)
  │
  ├──→ 共享特征提取器 (3层CNN) ──→ F_t (C×H×W)
  │                                    │
  │              ┌─────────────────────┤
  │              │                     │
  │    Background SSM            Degradation SSM
  │    (慢变化, 长期记忆)          (快变化, 短期记忆)
  │              │                     │
  │         h_t^bg                 h_t^deg
  │              │                     │
  │              └──────┬──────────────┘
  │                     │
  │              Fusion Gate (可学习)
  │                     │
  │              F_t^enhanced
  │                     │
  │              Decoder (2层CNN)
  │                     │
  └────────────────→ Ĩ_t (增强帧)
```

**Background SSM**：
- 状态空间维度：$d_{\text{state}} = 64$
- 时间常数：大（长记忆），通过初始化 SSM 的 $\mathbf{A}$ 矩阵使衰减慢
- 作用：累积多帧中的稳定背景信息，在被雨丝遮挡区域从历史帧中"借用"可见像素

**Degradation SSM**：
- 状态空间维度：$d_{\text{state}} = 32$
- 时间常数：小（短记忆），快速遗忘旧状态
- 作用：建模雨丝/雪花的运动模式和统计特性，帮助区分"什么是雨"和"什么是背景"

**Fusion Gate**：

$$
\tilde{F}_t = \alpha_t \odot \text{Decode}(h_t^{\text{bg}}) + (1 - \alpha_t) \odot \text{Decode}(h_t^{\text{deg}})
$$

其中 $\alpha_t$ 是空间自适应的注意力图（$1 \times H \times W$），由两路状态的拼接经 1×1 卷积 + sigmoid 产生。在被雨丝遮挡的区域，$\alpha_t$ 应较低（因为背景流在此处信息不完整，需要退化流辅助估计）；在清晰区域，$\alpha_t$ 应较高（直接使用背景流的可靠信息）。

#### I.2.3 关键组件二：物理一致性损失（Physical Consistency Loss）

**雨雪物理先验**：
1. **方向性**：雨丝在图像上近似沿重力方向的细长条纹
2. **稀疏性**：雨丝/雪花在空间上是稀疏的，大部分像素是背景
3. **时变随机性**：雨丝在帧间位置变化近似随机，而背景变化由相机运动决定

**损失设计**：

$$
\mathcal{L}_{\text{phys}} = \lambda_{\text{sparse}} \cdot \mathcal{L}_{\text{sparse}} + \lambda_{\text{dir}} \cdot \mathcal{L}_{\text{dir}} + \lambda_{\text{temp}} \cdot \mathcal{L}_{\text{temp}}
$$

**$\mathcal{L}_{\text{sparse}}$（退化稀疏性）**：

$$
\mathcal{L}_{\text{sparse}} = \| \text{Decode}(h_t^{\text{deg}}) \|_1
$$

**$\mathcal{L}_{\text{dir}}$（方向性约束）**：

$$
\mathcal{L}_{\text{dir}} = \sum_{x} \max\left(0,\; |\nabla_x D_t(x)| - |\nabla_y D_t(x)|\right)
$$

**$\mathcal{L}_{\text{temp}}$（时序一致性）**：

$$
\mathcal{L}_{\text{temp}} = \sum_{t} \| B_t - \text{Warp}(B_{t-1},\; \text{Flow}_{t-1 \to t}) \|_2^2
$$

#### I.2.4 关键组件三：场景自适应启停（Scene-Adaptive Activation）

**方案**：

```
I_t → 轻量分类器 (MobileNetV3-Small) → [p_rain, p_snow, p_fog, p_night, p_clear]
                                            │
                                    if p_rain + p_snow > τ:
                                        启用 TPSWA
                                    else:
                                        跳过 TPSWA, Ĩ_t = I_t
```

分类器在 ACDC + WeatherSynthetic 的天气标签上预训练，参数量 < 1M，推理延迟 < 2ms。阈值 $\tau$ 可设为 0.5。

### I.3 TPSWA 的两个变体方案

#### 变体 B1：轻量版（Light TPSWA）

| 组件 | 配置 |
|------|------|
| 时序建模 | 3D Conv (3×3×3, 2 层) |
| 双流分离 | 无（单流） |
| 物理损失 | 仅 $\mathcal{L}_{\text{temp}}$（时序一致性） |
| 额外参数 | ~200K |
| 额外延迟 | ~5ms |
| 预计实现时间 | 1 周 |

**适用场景**：快速验证"时序信息是否对 OPWA 有增益"。如果 B1 在雨雪视频上已有提升，说明时序方向值得深入。

#### 变体 B2：完整双流 SSM 版（Full TPSWA）

| 组件 | 配置 |
|------|------|
| 时序建模 | 双流 SSM (Mamba-like, 线性复杂度) |
| 双流分离 | Background SSM + Degradation SSM |
| 物理损失 | $\mathcal{L}_{\text{sparse}}$ + $\mathcal{L}_{\text{dir}}$ + $\mathcal{L}_{\text{temp}}$ |
| 额外参数 | ~800K |
| 额外延迟 | ~12ms |
| 预计实现时间 | 2.5 周 |

**适用场景**：论文中的完整方法，双流分离和物理约束是核心创新点。

### I.4 TPSWA 与 OPWA 的联合训练策略

#### I.4.1 三阶段训练

**阶段一：OPWA 单独训练（2000 步）**
- 按 OPWA 的两阶段训练流程完成
- 此阶段 TPSWA 不存在

**阶段二：TPSWA 预训练（1000 步）**
- 冻结 OPWA，仅训练 TPSWA
- 损失：$\mathcal{L}_{\text{rec}}(Ĩ_t, I_{\text{clean}})$ + $\mathcal{L}_{\text{phys}}$
- 目标：让 TPSWA 学会基本的雨雪去除和时序一致性

**阶段三：联合微调（500 步）**
- 解冻 OPWA 的 gate 参数（其余仍冻结），与 TPSWA 联合训练
- 损失：$\mathcal{L}_{\text{rec}}$ + $\mathcal{L}_{\text{percept}}$ + $\mathcal{L}_{\text{phys}}$
- 目标：让 OPWA 的 gate 适应 TPSWA 增强后的输入分布

#### I.4.2 视频数据构造

| 数据源 | 方式 |
|--------|------|
| CARLA 仿真 | 固定相机轨迹 + 多种天气，生成 3–5 帧短序列 |
| 真实视频 | 从 ACDC 等数据集中提取连续帧（如果有） |
| 单帧→伪视频 | 对单帧图像施加随机仿射变换模拟相机运动，生成伪序列 |

### I.5 TPSWA 评估体系

#### I.5.1 核心指标

除恢复率 $r_{\text{mIoU}}$ 外，新增时序稳定性指标：

$$
\text{Stability} = 1 - \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{|\text{mIoU}_t - \text{mIoU}_{t+1}|}{\text{mIoU}_{\text{clean}}}
$$

#### I.5.2 分析维度

| 维度 | 对比 |
|------|------|
| 天气类型 | rain vs snow vs fog（验证 TPSWA 仅在 rain/snow 有效） |
| 雨量级别 | light / moderate / heavy（验证 TPSWA 在 heavy rain 增益最大） |
| 时序稳定性 | 逐帧 mIoU 方差、flickering score |
| 延迟 | 单帧延迟、端到端流水线延迟 |

### I.6 TPSWA 风险评估与缓解

| 风险 | 严重度 | 缓解策略 |
|------|:------:|----------|
| SSM 实现复杂度高 | 高 | B1 (3D Conv) 作为保底 |
| 光流估计不准导致 $\mathcal{L}_{\text{temp}}$ 失效 | 中 | 使用置信度加权光流，低置信区域降低权重 |
| 联合训练不稳定 | 中 | 三阶段训练 + 逐阶段验证 |
| 快速运动场景伪影 | 中 | 场景分类器 + 运动幅度检测，高速时降级为单帧 |

---

## J. OPWA 与 TPSWA 的对比总结

| 维度 | OPWA (A2) | TPSWA (B2) |
|------|-----------|------------|
| **输入** | 单帧退化图像 | 3–5 帧短序列 |
| **核心机制** | 多分支结构解耦 + 条件 gate + 感知驱动 | 双流 SSM 背景/退化分离 + 物理约束 |
| **参数增量** | ~2.1M（分支 + gate MLP） | ~800K（SSM + 场景分类器） |
| **延迟增量** | ~15ms（SD-Turbo 一步推理） | ~12ms（叠加在 OPWA 之上） |
| **适用天气** | 全天气（rain/fog/night/snow） | 仅 rain/snow（其余自动跳过） |
| **创新亮点** | 结构正交 + 感知驱动 + 条件 gate | 双流时序分离 + 物理一致性 |
| **实现难度** | 低（A1 一周可出） | 中（B1 一周，B2 需 2.5 周） |
| **论文角色** | 主方法 | 扩展方法 / 视频增强 |

---

## K. 推荐推进路径（6 周时间线）

| 周次 | OPWA | TPSWA |
|:----:|------|-------|
| W1 | 实现 A1（最小可行版），在 WeatherSynthetic（rain/fog/night）上跑通 | 调研 SSM 实现（Mamba 库），准备 CARLA 数据 |
| W2 | 评估 A1，实现 A2（条件 gate + 软正交） | 实现 B1（3D Conv 轻量版），验证时序增益 |
| W3 | A2 全面实验 + S2R-Bench 评测 | 如有增益，开始实现 B2（双流 SSM） |
| W4 | A3（频域增强）消融实验 | B2 训练 + 联合训练 |
| W5 | 补充实验：CKA、梯度分析、gate 可视化 | 时序稳定性分析 + 消融 |
| W6 | 论文撰写 | 论文撰写 |

**保底策略**：如果 W2 结束时 A1 已有显著恢复率提升且 B1 无明显增益，可以放弃 TPSWA，将全部精力投入 OPWA 的深度分析和论文写作。OPWA 单独作为一篇 AAAI 论文的核心贡献是充分的。

---

## L. A1 版实验方案

### L.1 A1 实验要回答的关键问题

| 问题 | 如何检验 | 对 A2 的影响 |
|:---|:---|:---|
| **GPPI 架构在多天气上是否仍然有效？** | 对比 A1 vs Baseline（无分支）在多天气 test set 上的 mIoU | 如果有效 → A2 架构基础成立；如果无效 → 需要更大的架构改动 |
| **感知驱动损失是否带来额外增益？** | A1（有 L_percept）vs A1-no-percept（无 L_percept） | 如果增益大 → A2 保留并加强；如果无效 → A2 改用其他策略 |
| **不同天气的恢复率差异多大？** | 按天气类型分组统计 r_mIoU | 差异大的天气类型 → A2 条件 gate 的重点优化方向 |
| **静态 gate 是否在不同天气上表现不一致？** | 观察训练后 gate 值 + 分组 mIoU | 如果 fog 和 rain 需要不同的 gate 值 → 条件 gate 的必要性被证明 |

### L.2 实验矩阵

| # | 实验 | 配置 | 目标 | 预计用时 |
|:-:|:---|:----|:----|:-------:|
| 1 | Baseline | 无 OPWA 分支，直接对退化图像推理 | 退化下界 | — |
| 2 | A1 (no percept) | D-Enc + StaticGate + L_rec | GPPI 纯结构迁移验证 | 1 天 |
| 3 | A1 (full) | D-Enc + StaticGate + L_rec + L_percept (warmup) | 感知驱动增益验证 | 1 天 |
| 4 | Clean | 无退化图像直接推理 | 性能上界 | — |

### L.3 实验脚本执行流程

```bash
# A1 完整训练（含两阶段）
python scripts/train_opwa_a1.py \
    --weather_synthetic /data/WeatherSynthetic \
    --foggy_cityscapes /data/FoggyCityscapes \
    --output_dir ./outputs/opwa_a1_full

# A1 no-percept（消融：仅重建损失）
python scripts/train_opwa_a1.py \
    --weather_synthetic /data/WeatherSynthetic \
    --foggy_cityscapes /data/FoggyCityscapes \
    --percept_weight_max 0.0 \
    --output_dir ./outputs/opwa_a1_nopercept

# A1 评估
python scripts/eval_opwa_a1.py \
    --checkpoint ./checkpoints/opwa_a1_final.pt \
    --foggy_cityscapes /data/FoggyCityscapes \
    --output ./eval_results.json
```

### L.4 结果记录模板

```yaml
实验日期: 2026-06-06
模型版本: opwa_a1_step_2000

# 整体恢复率
r_mIoU:
  all: 85.2%    # 全部天气平均
  rain: 82.1%
  fog: 79.8%
  night: 88.5%
  snow: 84.3%

# Gate 值 (训练后)
gate_values: [0.52, 0.48, 0.46, 0.44]  # scale 0→3

# 消融对比
r_mIoU_A1_full: 85.2%
r_mIoU_A1_nopercept: 82.7%
percept_gain: +2.5%    # 感知损失的边际增益

# A2 关键决策
use_conditional_gate: true   # 如果 fog 和 rain 的 gate 需求差异大
use_ortho_reg: false         # 如果 CKA < 0.1 已经成立
focus_weather: fog           # 最难的天气类型
```

### L.5 决策树

```
A1 实验结果出来后的决策逻辑：

1. A1 (full) 的 r_mIoU > Baseline + 5%?
   ├── YES → A2 架构基础成立，继续优化
   └── NO  → 检查梯度冲突，考虑移除 stop-gradient 或改用 gradient scaling

2. A1 (full) 比 A1 (no percept) 的 r_mIoU > 2%?
   ├── YES → 感知驱动损失有效，A2 保留并加强
   └── NO  → 检查 warmup 策略，尝试更高 λ_p 或移除 warmup

3. 不同天气的 r_mIoU 标准差 > 10%?
   ├── YES → 有必要引入条件 gate，A2 重点优化
   └── NO  → 静态 gate 基本够用，条件 gate 降级为消融实验

4. fog + night 的恢复率显著低于 rain + snow?
   ├── YES → A2 需要更强的低频恢复能力（考虑频域增强 A3）
   └── NO  → 标准 A2 方案足够
```

---

## M. 设计理由的系统性总结

### M.1 为什么 OPWA 要在 GPPI 基础上做这三项改动？

**改动 1：分支输入从 Albedo 改为退化图像经 stop-gradient 的 D-Enc**

GPPI 中 Albedo 需要额外预计算（且仅在 rain→sunny 场景有意义），这在多天气场景下不可行。直接用退化图像作为分支输入，虽然引入了结构化信息（违反原则 2），但通过 stop-gradient 切断梯度流，使分支退化为一个**固定的随机特征空间**（类似 random_gppi），gate 学习从中选择有用特征。这既保持了通用性，又避免了梯度冲突。

**改动 2：Gate 从静态标量改为条件动态 gate**

多天气场景下，不同退化的恢复策略截然不同。静态 gate 无法适应这种多样性。条件 gate 通过退化嵌入动态调整注入强度，使同一架构在不同天气下表现出不同的行为模式。这一设计的额外好处是：gate 值分布本身可以作为分析工具，揭示网络对不同天气的"理解"。

**改动 3：引入感知驱动损失**

GPPI 仅优化图像质量，但核心目标是下游感知性能。感知驱动损失直接将冻结感知模型的输出质量作为优化目标，使 OPWA 学到"对特定感知模型最友好"的翻译策略。warmup 策略和重建损失的保留是防止幻觉的关键安全措施。

### M.2 为什么 TPSWA 采用双流 SSM 而非单流 Transformer？

**理由 1：物理可解释性** — 双流设计直接对应"背景"和"退化"两种物理实体，每个流的行为可以被独立分析和验证。单流 Transformer 将所有信息混合处理，无法区分模型是在"恢复背景"还是"去除雨丝"。

**理由 2：线性复杂度** — SSM 的时间复杂度为 $O(T)$，而 Transformer 为 $O(T^2)$。在自动驾驶场景中，即使 $T=5$，SSM 的延迟优势也很明显，且可以自然地以在线模式逐帧处理。

**理由 3：可控的记忆长度** — 通过初始化 SSM 的 $\mathbf{A}$ 矩阵，可以精确控制每个流的记忆长度：背景流需要长记忆（跨多帧积累信息），退化流需要短记忆（只关注当前帧附近的雨丝动态）。这种细粒度的控制在 Transformer 中难以实现。

### M.3 为什么物理先验采用软约束而非硬输入？

GPPI 的 rank1 probe 实验（仅恢复 53%）清楚地表明：**有结构的输入通过 Add 操作会干扰主干**。如果将透射率图、深度图等物理量直接作为分支输入，它们的空间结构（如深度的远小近大）会在 skip 连接处产生与主干冲突的梯度信号。

因此，TPSWA 中的物理先验（方向性、稀疏性、时序一致性）全部设计为**损失函数**而非输入张量。损失函数通过梯度间接引导网络学习物理规律，而不会在正向传播时直接干扰特征图。这一设计选择是 GPPI 实验最重要的启示之一。
