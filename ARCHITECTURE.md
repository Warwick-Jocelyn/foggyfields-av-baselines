# FoggyFields 当前网络结构（截至 2026-11-01）

一句话：**NeuRAD backbone 不动，外挂一个与场景分离、相机/LiDAR 共享的显式雾介质，全部改动藏在
默认关闭的 config flag 后（flag 全关 = 原版 NeuRAD 位元不变）。** 新增可学习参数总计 ~154 个
（均匀雾版）/ +约 60 万（空间场版网格）。

代码位置：
- backbone 改动: `splatad-nff-repro/code/neurad-studio/nerfstudio/models/neurad.py`（flag: `rgb_head`, `medium.*`, `loss.rgb_supervision`）
- 介质包: `foggyfields/medium/foggyfields/{medium.py, transport.py, mixture.py, trusted.py}`（18 项单元测试）
- 方法注册: `method_configs["neurad-foggy"]`

---

## 1. 总数据流

```
                       camera rays + lidar rays  (metric, world frame, 共一个 RayBundle)
                                   │
                 ┌─────────────────┴──────────────────┐
                 │   NeuRAD backbone（原样）           │
                 │   proposal sampling (128,64)→32 样本 │
                 │   neural field → 每样本特征 f_i[32]  │
                 │   SDF-alpha → 权重 w_i (sky 吸收余量) │
                 └─────────────────┬──────────────────┘
          ┌────────────────────────┼─────────────────────────────┐
          │ (原版相机路径, 保留)     │ (新增①)                      │ (原版 LiDAR 头, 保留)
   F=Σw_i f_i → CNN×3上采样 → rgb  │  c_i = σ(Linear[f_i;app]) 147参数│  MLP(48→2): raw强度, drop logit
          │                        │  每样本辐射度(transport 之前)   │
          │                 ┌──────┴──────── 新增② 显式介质 ────────┴──────┐
          │                 │ β field ──MediumIntegrator──► 每射线 τ(s) 查表 LUT │
          │                 │   相机: rgb_ray = Σ w_i·T(s_i)·c_i + 精确区间airlight│
          │                 │        (sky=最大程处表面; W_scene 遮挡=墙后无airlight)│
          │                 │   LiDAR: intensity = g·raw·T(D)^(2k_L)              │
          │                 │   几何权重 w_i 不乘 T —— 雾衰减辐射度, 不改占据      │
          │                 └──────────────────┬───────────────────────────┘
          │                                    │ (新增③, LiDAR 返回模型)
          │                    FogReturnMixture: E_S=a_S·T²·K(D), E_F(s_k)∝η·β·T², θ=噪声底
          │                    π=softmax(logE); NLL(高斯混合, 距离绝不软平均); drop NLL; 强度残差
          ▼
   输出: rgb(CNN), rgb_ray, depth, accumulation, intensity, ray_drop, (训练)mixture 损失项
```

## 2. 介质场 β 的三种形态（flag: `medium.type` 等）
| 形态 | 参数 | 状态 |
|---|---|---|
| 均匀 `HomogeneousFog` | softplus 标量 β₀ (1) | P2/P3 主力, 已验证 |
| 时变 `BetaKnots`（p2_fit 内）| 8 节点分段线性 | P2 ramp 恢复 6–17% ✓ |
| 空间 `SpatialFog` | 多分辨率稠密网格(16/32/64³×2) + 2×32 MLP; β(x)=β_anchor·exp(ε·tanh(f(x))−μ), 均值归一保持全局尺度锚定 | 首版进平场吸引子(tanh 饱和), 修复版(R1'=TV+饱和惩罚)在跑 |

## 3. 介质参数与锚定（可辨识性机制, P3 核心教训）
- **β 锚**: 相机暗通道独立测量（NFF-15: 0.0321, MOR≈93m）; 自由 β 的均衡点 ≈0.0105 (V2b)
- **A 锚**: 暗通道测得 0.544; 可冻 (`airlight_learnable=False`)
- **g (LiDAR 增益)**: 与强度头尺度简并, 自由时漂到 100–1700×(头等比缩小补偿, 乘积不变)——已加 `gain_learnable=False` 冻为 1
- **η (后向散射比)**: 钉 Hahner 常数 0.046/ln20=0.0154; **θ (噪声底)**: 必须真实数据标定
  （清天地平线下 drop 率 12.5% → θ=1.43e-5; 合成标定不可移植, V2c 实证）

## 4. 训练机制（P3 联合反演, flag 控制）
- **监督**: `rgb_supervision="rgb_ray"` — 光度损失在逐样本头的 ray 分辨率输出上
  （CNN 不在 transport 之后, do-not 规则）; LiDAR 原版 depth/intensity/drop 损失 + mixture 三项
- **三阶段调度**（BEFORE_TRAIN_ITERATION 回调, 只翻 requires_grad, 断点安全）:
  A 锚定热身（β 冻在测量值, 场景从头学 → 外观从未学会"画雾"）→ B 释放 → C 场景/介质交替
- **trusted-return 掩码**（标定 s_halo=6m: 清天保留 99.25%）: halo 疑点不进几何损失
  （教训: 有介质时启发式版有害, 最终配方关闭, 等 π_F 版）
- **正则**: R1'(空间场 TV+饱和惩罚), R2(场景 α 双峰: 物体"要么空要么实", 反雾状几何),
  sky 辐射度先验（光学厚区 c_sky 无约束 → 锚到观测, 修去雾黑天）
- **checkpoint 手术** (`p3_seed_ckpt.py`): 老 ckpt 注入新模块参数, 支持雾 ckpt 初始化

## 5. 与文献的对应（物理出身, 非拍脑袋）
- E_S = a_S·T²·K(D) = **NFL** 的主动传感振幅方程（K(s)=1/(s²+s₀²) 有限近场）
- mixture {表面/雾/无回波} = NFL 返回选择的介质推广; θ = NFL 检测阈值
- Koschmieder / Hahner / 晴天渲染 = β=0 或均匀 β 的解析特例
- **DyNFL** 的动态组合 + **SplatAD** 的 actor 处理 → P4 路线（MediumIntegrator 为 backbone
  无关的逐射线 LUT, 可直接乘进 SplatAD 光栅化 fragment——§5.6 插件）
- 升级队列: U1 入射角反射率 a_S=ρ·cos^γ(用 NeuRAD 法线), U2 光束发散, U3 双回波
