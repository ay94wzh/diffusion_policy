1. \(z_v\)：跨视角共享语义对齐监督
目标： 对齐不同视角 latent 中的共享语义，不强制整个 \(z_v\) 一致，保留视角特有信息。
- 在 \(z_v\) 后增加共享语义投影头 \(P_s\)，得到 \(z_s=P_s(z_v)\)。
- 对同一物理状态的不同视角，约束其共享语义表示接近。
- 首先使用归一化 latent 的 MSE Loss，后续可尝试 InfoNCE 对比学习。
\[
L_{\mathrm{align}}
=
\left\|
\operatorname{norm}(P_s(z_v^{(i)}))
-
\operatorname{norm}(P_s(z_v^{(j)}))
\right\|_2^2
\]
总损失：
\[
L=L_{\mathrm{DP}}+\lambda_vL_{\mathrm{align}}
\]
2. \(z_g\)：跨视角动作一致性监督
目标： 让不同视角输入得到的全局表示能够支持一致的机器人动作预测。
- 使用 MHA 融合 \(z_v\)，得到 \(z_g\)。
- 训练时随机采样不同视角子集，分别生成对应的 \(z_g^{(i)}\) 和 \(z_g^{(j)}\)。
- 因为动作均在 robot base frame 下表示，同一物理状态对应的动作目标应一致。
- 对 Diffusion Policy，固定同一真实动作、扩散时间步和噪声，分别预测噪声，并约束预测结果一致。
\[
L_{\mathrm{cons}}
=
\left\|
\epsilon_\psi(A_k,k,z_g^{(i)})
-
\epsilon_\psi(A_k,k,z_g^{(j)})
\right\|_2^2
\]
总损失：
\[
L=L_{\mathrm{DP}}+\lambda_gL_{\mathrm{cons}}
\]
3. 实验顺序
1. Baseline： MHA，不添加额外监督。
2. 实验 A： 仅增加 \(z_v\) 共享语义对齐监督。
3. 实验 B： 仅增加 \(z_g\) 跨视角动作一致性监督。
4. 实验 C： 同时使用两种监督。
保持数据分布、训练预算和随机种子设置一致，比较各任务的成功率、novel-view 泛化曲线及动作预测误差。分别确定两种监督是否有效，再测试联合训练。