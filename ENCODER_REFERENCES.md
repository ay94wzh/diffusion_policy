# Encoder 参考文献：链接、方法概括与迁移边界

版本：v0.3；创建与更新日期：2026-10-22。  
对应方案：[Encoder 渐进实验计划](ENCODER_EXPERIMENT_PLAN.md)。

本文整理本轮讨论最相关的 12 篇论文。年份使用首次 arXiv 版本年份，可能与正式会议年份不同。链接指向原始论文页面，概括依据摘要和可访问原文；不列出未经核实的性能数字。每项将论文方法与本项目设计建议分开，避免把启发误记成已经验证的结论。

## 阅读导航

| ID | 论文 | 首次年份 | 最相关的问题 |
|---|---|---|---|
| [R01](#r01) | PETR | 2022 | 保留二维空间特征，加入三维位置编码 |
| [R02](#r02) | 3D Diffuser Actor | 2024 | 视觉、机器人状态与动作 tokens 的空间交互 |
| [R03](#r03) | CameraCtrl | 2024 | 独立 ray encoder 与多尺度条件注入 |
| [R04](#r04) | Cameras as Rays | 2024 | Plücker ray bundle、相机预测与坐标歧义 |
| [R05](#r05) | Time-Contrastive Networks | 2017 | 同步多视角的状态语义学习 |
| [R06](#r06) | VICReg | 2021 | 一致性、维度方差与冗余控制 |
| [R07](#r07) | Geometry-biased Transformers | 2023 | 将射线距离直接放入注意力 |
| [R08](#r08) | GTA | 2023 | 利用相对几何变换对齐注意力特征 |
| [R09](#r09) | Cameras as Relative Positional Encoding / PRoPE | 2025 | 内外参进入相对位置编码 |
| [R10](#r10) | CroCo | 2022 | 用 RGB 跨视角补全学习空间关系 |
| [R11](#r11) | FiLM | 2017 | 通道仿射条件化的能力与局限 |
| [R12](#r12) | Diffusion Policy | 2023 | 当前动作生成框架与条件接口 |

建议顺序：先读 R01/R07，理解空间 tokens 和几何偏置；再读 R02，思考机器人状态如何参与读取；R08/R09 用于后续比较相机编码方式；R05/R06/R10 用于独立的表征学习线。

<a id="r01"></a>

## R01. PETR: Position Embedding Transformation for Multi-View 3D Object Detection

- 原文：[arXiv:2203.05625](https://arxiv.org/abs/2203.05625)。
- 方法重点：§3.2 3D Coordinates Generator、§3.3 3D Position Encoder，以及 transformer decoder。

**方法概括。**共享 backbone 为每幅图像提取二维特征图。在图像网格位置沿视锥离散设置候选深度，通过相机标定映射到世界三维坐标；MLP 编码这些坐标，与投影后的二维图像特征相加。多视角空间特征展开为 token 序列，供 object queries 读取并预测三维检测结果。

**需要区分。**候选深度不是传感器测得或网络预测的真实深度图。PETR 不需要逐像素真实深度输入，但确实在表示中使用沿深度方向的候选位置。flatten 主要将 `N×Hf×Wf×C` 变成 `(N·Hf·Wf)×C`，不是把每幅图像平均池化为一个全局向量。

**对本项目的启发。**借鉴“RGB 单独编码、保留空间 tokens、在特征层加入与位置对应的几何地址”。这对应 S/R。

**迁移边界。**把候选三维点编码替换成 Plücker 编码是本项目的新变体。Plücker 表示空间直线，不携带候选深度范围。PETR 的三维检测监督也与本项目的基坐标系动作监督不同，不能直接预期相同效果。本轮计划不使用 PETR 的候选深度采样。

<a id="r02"></a>

## R02. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

- 原文：[arXiv:2402.10885](https://arxiv.org/abs/2402.10885)。
- 方法重点：3D tokenization、3D relative denoising transformer、附录详细结构。

**方法概括。**使用二维视觉 backbone 提取图像 patch 特征，结合对应深度与相机标定计算三维位置，构成带三维位置的 scene tokens。机器人本体感知和带噪动作轨迹也表示为具有空间位置的 tokens。通过相对三维注意力融合这些信息，执行动作扩散去噪。

**需要区分。**视觉 backbone 的 ResNet/ViT 与动作去噪器的 UNet/Transformer 是两个独立选择。该工作可使用 ResNet 提取视觉特征，同时使用 Transformer 做动作去噪。准确理解是利用深度将特征位置提升到三维空间，而非把特征投到 depth map 上。

**对本项目的启发。**让机器人状态参与视觉 tokens 的读取，使用空间关系组织交互。RH 借鉴这种信息交互思路，不要求同时更换动作 UNet。

**迁移边界。**完整方法使用场景深度，其视觉 tokens 具有估计的三维点位置。本项目只有 rays，不能把 ray token 当作已知三维表面点。整段 EEF 轨迹 query 加点到射线偏置是本文方案，不是该论文已验证的无深度版本。

<a id="r03"></a>

## R03. CameraCtrl: Enabling Camera Control for Text-to-Video Generation

- 原文：[arXiv:2404.02101](https://arxiv.org/abs/2404.02101)。
- 方法重点：§3.2 Camera Pose Representation、§3.3 Camera Controllability into Video Generators。

**方法概括。**将相机轨迹表示为逐帧、逐像素的 Plücker maps。独立 camera encoder 仅接收相机表示并生成多尺度特征，结合线性融合层注入视频生成网络的相应模块；主要条件融合路径包含特征相加与可学习线性变换。论文还研究相机分布和数据外观对控制泛化的影响。

**需要区分。**camera encoder 编码的是相机条件，不是机器人控制所需的完整场景状态。其主要注入方式也不应统一描述成 FiLM。

**对本项目的启发。**将视觉内容和相机几何分别编码，在空间特征仍存在时融合；特征层几何编码比只扩宽 conv1 更值得单独比较。对应 R 的设计动机。

**迁移边界。**视频生成目标要求输出体现相机轨迹；同一物理状态的机器人基坐标系动作通常应保持一致，两者的监督压力不同。视频空间 UNet 的层级也不能直接等同于动作一维 UNet 的层级。把 ray encoder 注入动作网络是否有用，需要独立实验。

<a id="r04"></a>

## R04. Cameras as Rays: Pose Estimation via Ray Diffusion

- 原文：[arXiv:2402.14817](https://arxiv.org/abs/2402.14817)。
- 方法重点：§3.1 Representing Cameras with Rays、§3.2 Ray Regression、§3.3 Ray Diffusion、§3.4 Implementation Details。

**方法概括。**使用每个图像 patch 对应的六维 Plücker rays 表示相机。图像特征、像素坐标、加噪 rays 和扩散时间步共同进入 transformer，学习预测干净 rays。推理时由随机 rays 开始条件去噪，再从射线束恢复相机中心、旋转和内参。论文同时有直接 ray regression 版本。

**需要区分。**图像集合不能一般性地唯一确定任意外部世界坐标系中的绝对相机位姿和真实尺度。论文为处理坐标歧义进行坐标/尺度规范化，例如规范第一台相机的旋转和相机平移尺度。它不直接提供 robot-base 标定。

**对本项目的启发。**理解 `m=o×d`、patch 与 ray 的对应关系、射线束为何可以恢复相机参数，以及稀疏多视图中的预测不确定性。

**迁移边界。**本项目已经知道标定，不以重新预测相机为首要目标。由包含已知 ray 的特征重建 ray 很容易成为复制任务；相机可解码性也不等于控制相关几何。该论文不直接证明相机预测辅助 loss 能提升本项目成功率。

<a id="r05"></a>

## R05. Time-Contrastive Networks: Self-Supervised Learning from Video

- 原文：[arXiv:1704.06888](https://arxiv.org/abs/1704.06888)。
- 方法重点：multi-view triplet learning 与时间负样本构造。

**方法概括。**用同一时刻的不同相机观测作为正样本，拉近其表示；用同一序列其他时刻的观测作为负样本，区分外观接近但功能状态不同的画面。目的是获得跨视角、能描述任务状态的表征，并用于模仿或奖励构造。

**需要区分。**时间不同不保证物理状态不同。静止片段、重复动作和接触阶段中的微小变化都可能造成错误负样本，不能把所有相邻帧无条件推远。

**对本项目的启发。**已有同步重渲染视角可提供精确的同状态正样本；正样本语义比普通图像增强更贴近 view-generalization。

**迁移边界。**全局状态语义一致不意味着不同视角的所有局部特征都应一致，遮挡尤其需要考虑。若使用 TCN 风格负样本，应记录时间间隔、运动状态与采样规则。该方向属于表征学习扩展，不混入 B—RHG 主线。

<a id="r06"></a>

## R06. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning

- 原文：[arXiv:2105.04906](https://arxiv.org/abs/2105.04906)。
- 方法重点：invariance、variance、covariance 三项的定义与作用域。

**方法概括。**对成对视图的投影 embedding 施加一致性损失；对每个分支，要求每个维度在一批样本间的标准差高于阈值，并抑制维度间非对角协方差。原论文常用同一图像的两种增强产生配对样本。

**需要区分。**VICReg 不自动识别 state/view 语义，也不自动生成语义解耦的两个子空间。协方差小主要约束线性相关，不能解释为完全统计独立。方差项是跨 batch 样本的维度变化，不是要求同一 state 的不同视角彼此保持不同。

**对本项目的启发。**用同一物理状态的不同相机图像替代增强视图，在视觉共享语义投影上使用完整目标，补足当前纯归一化 MSE 的塌缩风险。该方向作为后续表征学习扩展。

**迁移边界。**归一化 MSE 不能不加调整就直接套用标准 VICReg 的尺度阈值；标准差和协方差应在未做逐样本 L2 归一化的适当 embedding 上定义，并记录尺度。方差/协方差主要在不同 state 的 batch 上计算，不能让多个视角伪装成多个独立 state。

若输入已混入同一样本共享的机器人轨迹，表示可以只编码机器人状态来满足配对一致性与跨样本方差，仍忽略视觉。因此本项目建议把该目标放在轨迹注入前的视觉分支，并检测背景、时间和机器人本体等捷径。

<a id="r07"></a>

## R07. Geometry-biased Transformers for Novel View Synthesis

- 原文：[arXiv:2301.04650](https://arxiv.org/abs/2301.04650)。
- 简称：GBT。
- 方法重点：ray representation 与 geometry-biased attention。

**方法概括。**从给定相机的图像提取 patch tokens，以集合表示场景，并用 query ray 解码新视角颜色。除相机相关输入编码外，在注意力分数中加入由两条空间射线距离构成的偏置，使场景编码与 ray 解码更符合多视角几何。

**对本项目的启发。**这是“几何进入明确 attention 计算，而非只作为额外输入”最直接的参考之一。RHG 将其思路改为已知 EEF 点与图像射线的距离，无需场景深度图。

**迁移边界。**GBT 原工作主要使用 ray-to-ray 关系，本计划使用 point-to-ray 关系。距离小不保证同一表面点、正确对应或可见性。单台相机的 rays 共用相机中心，仅用它们之间的最短直线距离会退化；本计划因此使用机器人位置作为相机外部的空间锚点。不能直接拿 GBT 的新视角合成收益当成控制收益。

<a id="r08"></a>

## R08. GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers

- 原文：[arXiv:2310.10375](https://arxiv.org/abs/2310.10375)。
- 方法重点：Geometric Transform Attention、相对变换及其表示。

**方法概括。**为视觉 tokens 关联几何属性，通过 query 与 key/value 之间的相对几何变换，在注意力特征空间中进行坐标对齐。相比仅将位置编码相加，更直接利用变换的几何结构。论文主要在多视角新视角合成中评估。

**对本项目的启发。**研究相机、robot base、EEF 坐标系的相对关系如何进入注意力，可作为后续结构替代方向；不需要依赖场景深度来为每个像素指定表面点。

**迁移边界。**GTA 需要明确 token 关联的变换和表示选择，不是给任意 transformer 附加一个外参向量就自动等价。采用相对变换也不能直接宣称整个机器人策略具有 SE(3) 等变性；重力、固定工作台、目标布局和基坐标系动作都会影响整体对称性。

<a id="r09"></a>

## R09. Cameras as Relative Positional Encoding

- 原文：[arXiv:2507.10496](https://arxiv.org/abs/2507.10496)。
- 提出方法：Projective Positional Encoding，简称 PRoPE；论文标题不是单独的“PRoPE”。
- 方法重点：pixel-aligned raymap、relative SE(3) encoding、§3.4 PRoPE。

**方法概括。**比较 token 层的 raymap 编码与注意力层的相对相机编码，并提出 PRoPE：将完整相机视锥的内参和外参关系以相对投影编码进入注意力。论文评估新视角合成，以及立体深度估计、空间辨别等任务。

**需要区分。**视锥关系可以由内外参构造，不要求先给视觉 tokens 提供测得的场景深度。论文包含深度估计下游任务，并不等同于该编码必须把深度图作为输入。

**对本项目的启发。**是比较“ray 信息进入 token”与“相机关系进入 attention”的系统参考。后续可与 R/RHG 的绝对基坐标系 ray 编码和点线偏置比较。

**迁移边界。**原文指出其单视角形式可退化为 RoPE。因此对于 N=1 机器人部署，仅靠相机之间的相对关系不足以表达相机相对于 robot base 的关系，需要另外的机器人/基坐标系锚定。世界参考系在机器人控制中也有实际输出含义，不能直接将其全部当作应去除的无关因素。

<a id="r10"></a>

## R10. CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion

- 原文：[arXiv:2210.10716](https://arxiv.org/abs/2210.10716)。
- 方法重点：cross-view completion、masked target image 与 reference image。

**方法概括。**将同一场景的两幅 RGB 图像作为输入，对第一幅大比例遮挡，通过其可见部分与第二视角补全被遮挡内容。该预训练目标鼓励模型利用跨视角的空间关系，获得可迁移到几何视觉任务的表示。

**对本项目的启发。**现有同步多视角 RGB 可以用于补全预训练，不需要在补全目标中提供场景深度标签。相较仅拉近全局向量，它要求恢复更细的空间内容，可作为后续独立训练路线。

**迁移边界。**原论文的预训练样本构造与数据规模不同；三维下游任务表现不直接保证机器人闭环控制提升。重渲染场景背景较固定时，模型可能只利用背景或单视角先验；需比较 reference view 正常、去除、错误配对等条件。预训练 decoder 的使用和丢弃方式也需要明确，不能默认双视角预训练天然解决单视角 policy 泛化。

<a id="r11"></a>

## R11. FiLM: Visual Reasoning with a General Conditioning Layer

- 原文：[arXiv:1709.07871](https://arxiv.org/abs/1709.07871)。
- 方法重点：feature-wise affine modulation。

**方法概括。**由条件输入生成逐通道缩放与偏移，以仿射变换调制视觉特征，使条件参与视觉处理。它已展示可支持视觉推理，不能笼统认为它没有多模态融合能力。

**对本项目的启发。**帮助准确描述当前轨迹 FiLM 分支：它提供条件化，但在当前实现中同一通道的 gamma/beta 在空间位置上共享。

**迁移边界。**空间共享调制不显式编码末端投影位置，并不意味着绝对无法学到空间关系。本文改用轨迹 query 和几何偏置是在改变归纳偏置；B→S 包含多个结构变化，不能单独用于宣布 FiLM 机制无效。

<a id="r12"></a>

## R12. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

- 原文：[arXiv:2303.04137](https://arxiv.org/abs/2303.04137)。
- 项目：[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)。
- 方法重点：条件动作扩散、receding horizon control、视觉条件与时序去噪网络。

**方法概括。**学习给定观测时的动作序列分布，通过条件去噪迭代生成动作块，并在滚动时域执行部分动作后重新观测。论文讨论视觉条件及不同时间序列去噪结构。

**对本项目的启发。**作为动作生成框架保持稳定，使本轮聚焦 encoder 中的空间视觉、相机条件和轨迹交互。本文计划的每帧输出仍为 512+9，保留 To=2 时的 1042 维动作条件接口。

**迁移边界。**条件维度相同只保证接口兼容，不保证 encoder 参数量、FLOPs 或可表达函数相同。动作 diffusion loss 也不唯一要求网络学习显式三维计算，因此本轮必须依靠逐项对照和闭环评价来检验机制。

## 与实验计划的对应关系

| 计划模块 | 主要参考 | 本项目新增或不同的部分 |
|---|---|---|
| B 原动作框架与 FiLM | R12、R11 | 历史协议修正来自项目审计 |
| S 空间 token 读取 | R01 | 用 1 个跨视角共享的 512 维 query、8 heads 读出每视角一个 token，继续使用原一维动作 UNet |
| R ray 编码 | R01、R03、R04 | 无候选深度，ray 地址进 key，视觉内容进 value |
| RH 完整轨迹条件化共享 query | R02 | 汇总全部 4 个有序 EEF 位姿，条件化一个 512 维共享 query；具体组合由本项目提出 |
| RHG 点到射线偏置 | R07 | 当前 EEF 位置作为统一锚点，在共享 query 的 4/8 个 heads 中加入偏置，其余 heads 保留内容读取 |
| 后续：视觉语义正则 | R05、R06 | 约束放在机器人轨迹注入之前，排查状态捷径 |
| 后续：相对几何/预训练 | R08、R09、R10 | 单相机部署与 robot-base 动作需要额外考虑 |

## 概念校准清单

1. **无深度输入不等于无三维几何。**相机标定、射线、已知 EEF 三维位置都可以使用，但不能将 ray 当作物体表面三维点。
2. **flatten 不等于全局池化。**空间序列 `(Hf·Wf)×C` 与单向量 C 的表达不同。
3. **相机可解码不等于几何用于控制。**相机估计辅助目标与控制目标需要分别评价。
4. **VICReg 不自动分离语义。**正则作用域、配对样本和实际视觉依赖都要明确。
5. **不同时间不必然是不同状态。**时间对比学习的负样本可能包含错误负例。
6. **相对几何不是单相机基坐标系锚定的替代品。**需要明确机器人与相机之间的已知关系。
7. **几何距离偏置不提供可见性或真实深度。**遮挡、视线前后方向和投影范围需要分别处理。
8. **整段轨迹编码与几何锚点是独立选择。**本计划用完整 4 步历史条件化一个跨视角共享的 512 维 query，使用当前 EEF 位置作为几何锚点；8 heads 中仅 4 个加入距离偏置。
