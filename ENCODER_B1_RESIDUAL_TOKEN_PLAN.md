# 在 B1 基础上改进空间信息交互：方案与对照实验

日期：2026-10-30（2026-11-02 实现）。状态：编码器、CPU 测试、warm-start 工作区与配置已实现；未训练、未评估。本文仍不修改既有 B/S/R/RH/RHG 实验的定义。

## 1. 目标与已知证据

目标是在保留 B1 现有表现的基础上，改善 **N=1 单相机新视角控制**。本轮首先检验观测编码器，不同时更换 UNet、增加辅助损失或扩大训练相机分布。

根据本地 `eval_log.json` 重新汇总的严格成功率：

| 模型 | 已见方位角，11 个 | 插值方位角，10 个 | 范围外方位角，4 个 | el=-15° | el=+15° |
|---|---:|---:|---:|---:|---:|
| B1 | 0.8127 | 0.7840 | 0.3650 | 0.02 | 0.12 |
| S | 0.6527 | 0.6180 | 0.3550 | 0.04 | 0.24 |
| R | 0.6018 | 0.6040 | 0.3700 | 0.06 | 0.24 |

B1 在这三组中的已见/插值方位角上最好，范围外及俯仰仍弱。结果来自训练 seed 42、每视角 50 个环境 episode。

证据文件：

- [实验记录](PROGRESS.md#the-encoder-ladder-b0b1sr-2026-10-29)。
- `data/eval_interp_square_enc_b1/eval_log.json`
- `data/eval_heldout_square_enc_b1/eval_log.json`
- `data/eval_el_square_enc_b1/eval_log.json`

## 2. B1 

B1 路径：

```text
RGB + 逐像素 Plücker map（共 9 通道）
      ↓
共享完整 ResNet18：stem → layer1 → layer2 → layer3 → layer4
各阶段使用相机坐标系 EEF 历史产生的 FiLM
      ↓
每视角全局平均池化 → 512 维
      ↓
原 masked MHA 跨视角融合 → 512 维
      ↓
拼接当前 9 维机器人状态 → 单时刻 521 维
      ↓
两观测时刻拼接 → 1042 维 global_cond → 原动作 diffusion UNet
```

新方案保留上述全部主路径，包括真实 `episode` 历史、crop/ray 对齐、共享主干、FiLM、平均池化、原融合以及动作接口。只在 layer4 的 FiLM 输出与平均池化之间增加一个残差分支。

LVSM 提供的可检验启发，是在压缩前允许 token 交互。

## 3. 首轮候选：B1 加残差 token 交互

### 3.1 张量与信息流

当前输入 76×76 crop 经过完整 ResNet18 后，layer4 特征预期为每视角 `3×3×512`；实现时应实测并断言。

记第 v 个有效视角的特征为 `F_v ∈ R^(G×512)`，当前 G=9。保留原读出：

```text
z_v_base = mean_spatial(F_v)                     # 原 B1 的 512 维向量
```

新增分支使用一层双向自注意力 Transformer：

```text
U = Transformer(F, attention_mask)
delta_v = mean_spatial(U_v - F_v)
z_v_new = z_v_base + W_out(delta_v)
```

之后仍使用原跨视角融合与 UNet。`W_out` 是无 bias 的 512→512 线性层，权重初始化为零；Transformer 采用常规非零初始化。

不要同时将 Transformer 和输出门控全部置零。验证实际激活情况，而不是要求所有新参数第一步就有非零梯度。

### 3.2 首版规格

| 项目 | 建议 |
|---|---|
| 输入位置 | 原 B1 layer4 的最后一次 FiLM 后、avgpool 前 |
| 宽度 | 512，直接对接原特征 |
| 交互层 | 1 层 Pre-LN Transformer，8 heads，FFN=1024，GELU，dropout=0 |
| 残差读出 | 每视角对 `U-F` 平均，零初始化 `W_out` 后加回原向量 |
| 几何 | 使用 B1 输入层的完整 Plücker map；不加另一条 ray MLP |
| 条件 | 使用 B1 FiLM|
| 位置标记 | 不加相机 slot ID 或新增位置编码|
| 无效视角 | 不参与 attention、均值或读出；每样本至少一个有效视角 |
| 输出 | 每视角仍 512，整体接口仍 521/1042 |
| 新增参数 | 按上述结构约 2.36M，精确值实现后报告 |

N=11 时联合序列最多约 `11×9=99` 个 token；N=1 仍有 9 个空间 token。
layer4 单元已有较大感受野，不能把其 token 解释成一个精确三维点。

## 4. 对照实验：先区分处理能力、空间交互和跨视角交互

同一轮只比较下表，不同时增加层数、query 数、重建损失或新训练视角。

| 组 | B1 主路径 | 新分支的输入与连接 | 回答的问题 |
|---|---|---|---|
| A：B1 | 完整保留 | 无新分支 | 当前基线；短程实验中必须同预算继续训练 |
| B：单向量残差 | 完整保留 | 对每视角 `z_v_base` 独立运行同一个 Transformer 分支，序列长 1；输出差值残差相加 | 增加非线性处理和参数，是否已经足够？ |
| C：视角内空间交互 | 完整保留 | 输入完整空间 tokens；attention 只在各视角内部开放 | 相对 B，保留空间信息并在视角内部交互是否有收益？ |
| D：跨视角空间交互 | 完整保留 | 相同空间 tokens、相同参数，attention 在全部有效视角开放 | 相对 C，训练期跨视角局部信息交换是否有增益？ |

B/C/D 使用相同模块配置、输出投影和对应初始权重，参数个数相同。

C/D 除 attention mask 外保持一致。N=1 时两者的连接结构相同；如果 D 在单相机部署时更好，支持的是训练期跨视角交互带来了迁移收益。

解释顺序：

- B>A，而 C/D 没有进一步收益：优先考虑额外处理能力的解释。
- C>B，且优于 A：支持空间 token 路径有价值；不能仅凭这一项分离空间分辨率与 attention 的作用。
- D>C，且单相机主指标提高：支持训练期跨视角交互的增量价值。
- C/D 只在 N>1 推理改善：可能适合多相机部署，但没有达成本项目 N=1 主目标。
- 新分支学到非零输出而闭环无收益：记为机制生效、行为无收益；不因 attention 图更好看而继续扩大模型。

## 5. 训练

### 5.1 短程筛选：从同一 B1 checkpoint 出发

先在独立实验目录复制 B1 的 model、EMA、normalizer；不覆盖原 checkpoint。 B/C/D 新模块按对应方式初始化，A 同样继续训练。

建议预先固定：额外 30 epochs、batch=64、AdamW 原 beta/weight decay、峰值学习率 1e-5、新 30-epoch cosine schedule、前 500 step warmup；所有组重新创建优化器与 scheduler，保持训练数据与其余动作配置。应在运行前锁定。

model 与 EMA 都要正确初始化新增分支，并保留各自的基线权重。比较同一个固定终点的 EMA，不用测试视角挑 checkpoint。

同时保留不继续训练的原 B1 结果作为参考，但结构收益必须与 **A 的同预算继续训练** 比较。

建议候选晋级条件：相对 A，插值成功率增加至少 0.05，已见方位角均值下降不超过 0.03，且相较 B 有额外收益、推理成本可接受。运行前固定，不是统计显著性阈值。若所有组都无明确收益，停止扩大该分支并检查失败类型。

### 5.2 正式确认：完整预算和独立训练种子

对 A 与筛选出的一个候选，使用训练 seed 42/43/44，从头按原 201-epoch 配方训练，评价 EMA epoch 200。需要解释空间/跨视角机制时，补对应 B 或 C，而不是只与旧的 S/R 比。

每个种子中，相同的 B1 主干与 UNet 使用对应相同初始权重；新模块初始化不要改变主干初始化、数据抽样和 crop 的 RNG 流。仅设置相同整数 seed 不足以自动保证这些条件。

若只在继续训练路线有效、从头训练无收益，应报告为 warm-start 适配结果；不能将两者混合成一次等预算架构比较。若决定只研究 warm-start 路线，应从多个独立训练的 B1 起点重复 A/候选。

## 6. 评价协议与开始前的必要核对

维持任务 square PH、绝对动作、11 个训练视角、N 均匀取 1—11、horizon=16、观测帧数 2、执行动作步数 8。主要部署条件为 N=1。

| 评价集合 | 视角 | 用途 |
|---|---|---|
| 已见方位角 | 0、±15、±30、±45、±60、±75° | 检查现有表现是否退化 |
| 插值方位角 | ±7.5、±22.5、±37.5、±52.5、±67.5° | 主要改进指标 |
| 范围外方位角 | ±82.5、±90° | 单独报告外推，不并入插值 |
| 新俯仰角 | el=-15°、+15° | 两个值及其均值；el=0 作对照，不能混入而掩盖失败 |

开始实验前，需解决或明确下列评价不确定性：

1. B1 本地日志 `az_0` 成功率为 0.80，`el_0` 为 0.90；两者 preset 都声明不施加额外旋转。应核对 checkpoint、真实相机矩阵、episode ID、环境初始化、rollout 分组和动作采样随机性。该差异目前不能直接判定为相机程序错误。
2. 当前 evaluator 显式配对环境 seed，但静态检查未看到按 episode/控制步配对 diffusion 噪声的显式 generator。环境 seed 相同不代表策略采样噪声相同。评价新结构前，应确认能以 episode ID、控制步及扩散步为索引重用对应随机流；只在进程开头设一个 seed 可能受 batch 顺序与不同 episode 长度影响。
3. 若统一采样协议需要改 evaluator，应在之后获得代码实施授权后再做，并用同一新协议重评 B1。本文没有修改该程序；旧结果继续保留原含义。

每视角先沿用 50 个环境 episode。报告逐视角严格成功率、集合平均、相对同预算 A 的差值和训练种子间变化。环境 episode 在不同视角上是配对的，分析区间时保留该结构，不把所有 rollout 当成完全独立样本。

筛选过模型后，正式确认使用预先固定的新一组环境 episode IDs，降低对既有评价 episode 的选择性适配。三种子结果仍应列出每种子的差值；小样本区间不稳定时不强行给显著性结论。

旧实验的 0.15 带是当时的实用差异判据，不能解释成置信区间或统计等价性证明。本轮不据此忽略 5—10 个百分点的可重复提升。

## 7. 诊断与何时换方向

只保留能解释结果的诊断：

- 残差相对强度 `||z_new-z_base|| / (||z_base||+eps)`、新增分支梯度和编码器 across-state spread：判断分支是否工作、有无坍缩。
- N=1 的完整策略延迟，以及训练吞吐/峰值显存：判断性能收益的代价。
- 一组固定 episode 的失败视频：区分看错目标、接近偏差、夹持失败与最后插入失败；这些是待检查类别，不是已识别的 B1 故障。

若 layer4 分支无收益，且视频显示局部对准是主要瓶颈，下一步可保留完整 B1 主路径，只从已有 FiLM 后的 layer3 引出更细网格的残差分支。它与本轮的网格/感受野变化分开比较，不立即再增加几何、历史或新损失。

若主要目标变成俯仰外推，应另立“训练相机分布”实验：固定 B1 结构，对比原方位角环与增加俯仰变化的数据。几何输入不能保证对训练中未覆盖的外观、遮挡与可见性泛化。这条数据轴可能更直接，但不混入本轮架构效果。

## 8. 与现有方案的关系与实施边界

[原联合 Transformer 方案](ENCODER_LVSM_EEF_QUERY_V1.md) 采用 layer3 RGB-only tokenizer、额外 ray embedding、历史 queries 和新的读出路径，是独立候选；本方案不改写或替代那份文档，也不修改 B/S/R/RH/RHG 的含义。

本轮建议优先级：**先核对评价随机性 → A/B/C/D 短程筛选 → A 与候选的独立种子确认 → 有证据后再考虑更细网格或训练相机分布。**

## 9. 实现状态（2026-11-02）

- 编码器：`diffusion_policy/model/vision/residual_token_view_conditioned_obs_encoder.py` 的 `ResidualTokenViewConditionedObsEncoder`，模式 `none|single|intra|cross`；A/B/C/D 参数与初始化控制见 `token_mixer_init_seed`。
- B1 接缝：`view_conditioned_obs_encoder.py` 新增 `_backbone_map` 与 `_encode_active_views`，默认路径与 B1 位级一致。
- 工作区：`diffusion_policy/workspace/train_diffusion_unet_image_workspace_enc_rt.py`；从 B1 checkpoint 复制 model/EMA，恢复 EMA 的 `optimization_step`，并在 loop 前重置 RNG。
- 配置：`train_diffusion_unet_image_workspace_enc_rt_latent.yaml`；30 epochs、batch 64、lr 1e-5、warmup 500、`skip_rollout: true`、`topk.k: 0`。
- CPU 测试：`tests/test_residual_token_view_conditioned_obs_encoder.py`、`tests/test_enc_rt_warmstart.py`；服务器端真实 checkpoint 门：`check_enc_rt_warmstart.py`。
- 驱动：`data/enc_rt_run.sh`（A/B/C/D）、`data/enc_rt_eval.sh`；汇总：`summarize_enc_rt.py`；只读评价噪声审计：`audit_eval_noise.py`。
- 尚未启动任何训练或评估；`AZ_0`/`el_0` 与 diffusion-noise 配对问题仍以只读审计为准。
