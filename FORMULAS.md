# 渲染方程与损失对照：原版 NeRF/NeuRAD vs FoggyFields（LaTeX）

## 1. 原版 NeRF 体渲染（NeuRAD 为其 SDF-alpha 变体）

沿射线 $\mathbf r(s)=\mathbf o+s\,\mathbf d$ 采样 $s_i$，场给出 $(\sigma_i, \mathbf c_i)$：

$$
T_i=\exp\Big(-\sum_{j<i}\sigma_j\,\delta_j\Big),\qquad
w_i=T_i\big(1-e^{-\sigma_i\delta_i}\big),\qquad
\hat C(\mathbf r)=\sum_i w_i\,\mathbf c_i
$$

NeuRAD 差异：$\alpha_i=\mathrm{sigmoid}(-\mathrm{SDF}_i\cdot\beta_{\mathrm{sdf}})$ 代替 $1-e^{-\sigma\delta}$；
逐样本输出 32 维特征 $\mathbf f_i$（无 $\mathbf c_i$），合成后过 CNN；sky 样本吸收剩余透射率
（$\sum_i w_i=1$）。LiDAR：期望深度 $\hat D=\sum_i w_i s_i$，强度/掉线由合成特征过 MLP。

**损失**（NeuRAD）：
$$
\mathcal L=\lambda_{rgb}\,\|\hat C-C\|_1^{\;+\mathrm{VGG}}
+\lambda_d\,\big|\hat D-D^{obs}\big|
+\lambda_i\,\|\hat I-I^{obs}\|^2
+\lambda_{dr}\,\mathrm{BCE}(\hat p_{drop})
+\textstyle\lambda_{carv}\sum w^2_{\text{(非近邻)}}+\dots
$$
观测 $C, D^{obs}, I^{obj}$ 都是**雾天数据**——雾被迫进入 $\sigma,\mathbf f$（实证：进外观）。

## 2. FoggyFields：同一场景 + 显式介质，一个透射指数

介质场 $\beta(\mathbf x)\ge 0$（当前：标量/β(t)/空间场），**与场景密度相加在同一指数里**：

$$
T_{med}(s)=\exp\Big(-\!\int_0^s\!\beta(\mathbf r(t))\,dt\Big)
\quad(\text{MediumIntegrator: 逐射线 } \tau(s)\text{ 查表, 可微})
$$

**相机（单程）**——新增逐样本辐射头 $\mathbf c_i=\mathrm{MLP}_{147}(\mathbf f_i,\mathrm{app})$：
$$
\hat C=\underbrace{\sum_i w_i\,T_{med}(s_i)\,\mathbf c_i}_{\text{衰减的场景}}
+\underbrace{\sum_k W_{sc}(s_k)\big[T_{med}(a_k)-T_{med}(b_k)\big]\,A}_{\text{airlight(精确区间积分, 墙后为0)}}
$$
sky = $s_{\max}$ 处的表面；几何权重 $w_i$ **不乘** $T_{med}$（雾衰减辐射度，不改占据）。

**LiDAR（双程 $T^2$）**：
$$
\hat I = g\cdot a_S\cdot T_{med}(D)^{2k_L},\qquad
E_S=a_S\,T^2(D)\,K(D),\quad
E_F(s_k)=\eta\,\beta_k\,T^2(s_k)\,K(s_k)\,\Delta s_k,\quad
K(s)=\tfrac{1}{s^2+s_0^2}
$$
返回选择 $\pi=\mathrm{softmax}\big(\log\{E_S,E_F(s_{1..K}),\theta\}\big)$；距离似然为高斯混合（绝不软平均）。

**损失**（新增项，全部 flag 控制）：
$$
\mathcal L=\lambda_{rgb}\|\hat C_{ray}-C^{fog}\|^2
+\mathcal L_{\text{NeuRAD-LiDAR}}^{(\text{trusted掩码})}
+\lambda_{nll}\big[-\log L(z)\big]
+\lambda_{dr}\big[-\log\pi_0\big]_{\text{地平线下}}
+\lambda_{int}\big(\log\tfrac{I^{obs}}{g a_S}+2k_L\tau(z)\big)^2
$$
$$
+\;\lambda_{tv}\,\mathrm{TV}(\beta)+\lambda_{sat}\,\overline{\phi^2}
+\lambda_{bi}\,\mathcal H(\alpha_{scene})
+\lambda_{sky}\,\|\mathbf c_{sky}-C^{obs}\|^2_{\,T\to0\text{区}}
$$

## 3. 为什么是"乘"而不是"加/concat"——以及 θ、传感常数

- **乘法来自 Beer–Lambert 定律**：穿过介质的辐射按 $e^{-\int\beta}$ **指数衰减**（每米损失固定比例→积分后是乘子），这不是建模选择而是物理。完整关系其实是"**乘 + 加**"：场景辐射被 $T_{med}$ 乘性衰减，介质自身再**加性**贡献内散射（相机的 airlight 项、LiDAR 的雾回波 $E_F$）。"⊗"是这整套传输方程的简写。加法/concat 没有物理语义——两个密度场**相加在同一个透射指数里**（$\sigma+\beta$）才是 RTE 的正确耦合，也正因如此分解才需要锚点（指数里加法意味着二者可互换吞吐）。
- **θ（噪声底）**：LiDAR 探测器电子学能分辨的最小回波功率。回波强度低于 θ → 射线"掉线"（no return）。它决定了雾天的远距掉点物理（$E_S\propto T^2\to$ 低于 θ）。已在真实清天数据上标定：θ=1.43×10⁻⁵（匹配地平线下 12.5% 掉线率）。
- **传感常数**：$k_L$（LiDAR 波长 905nm 相对可见光的消光比，雾滴大所以 ≈1，固定）；$\eta$（后向散射/消光比——被消光的能量中有多大比例折回传感器方向，钉 Hahner 常数 0.0154）；$g$（电子学增益，与头部尺度简并故冻 1）；$s_0$（近场接收响应衰减尺度，2 m）；$\sigma_S$（表面回波距离噪声宽度 0.1 m）。
