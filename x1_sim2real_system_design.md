# X1 参数辨识与 Sim2Real 三方案设计

> 状态：Design / 待实施  
> 版本：v0.2  
> 日期：2026-08-10  
> 候选方案：SPI、UAN、PACE  
> 当前无 `.oma/best.json`，本文不代表已有 checkpoint 可直接上机

## 1. 方案定义

X1 参数辨识保留三个独立候选方案：

1. **SPI / SPI-Active**：用已有安全策略产生自然运动数据，辨识系统级白盒参数。
2. **UAN**：用执行器激励数据学习仿真中的残差力矩网络。
3. **PACE**：固定基座、腿部悬空，用位置 chirp 辨识紧凑关节参数。

CMA-ES 不单列为第四种方案：它是 SPI 和 PACE 内部的黑盒优化器。三个方案先独立实施和对比，不预设必须组合。

| 项目 | SPI / SPI-Active |      | PACE |
|---|---|---|---|
| 主要 Gap | 系统级自然运动轨迹 | 复杂执行器非线性残差 | 关节执行器参数 |
| 需要已有实机 policy | 是 | 否 | 否 |
| 实机数据 | policy/motion-prior 运动 | 方波、正弦、Gaussian | 固定基座位置 chirp |
| 输出 | 质量、COM、惯量、执行器参数 | 残差网络 `delta_tau` | inertia、damping、friction、bias、delay |
| 优化方法 | CMA-ES | PPO | CMA-ES |
| 可解释性 | 高 | 低 | 高 |
| 实机风险 | 中 | 低至中 | 低 |
| X1 主要价值 | Base/整机/接触耦合 | 并联踝等复杂非线性 | 无 policy 冷启动执行器辨识 |

## 2. 三方案共同前置条件

辨识前必须先关闭软件契约 Gap：

- 训练单帧是 51 维，而当前 MuJoCo sim2sim 只填充旧的 47 维。
- `sim2sim.py` 中 body name 判断表达式恒为真。
- 训练和 sim2sim 的 gait phase 计算不一致。
- Actor 的 base 线速度坐标系与奖励/部署语义不统一。
- Isaac、MuJoCo 的关节力矩限制不一致，踝关节尤其明显。
- Torque multiplier 当前在每个 1 ms 物理步重新采样。
- DOF friction/damping 在 reset 时累计相乘。
- Friction curriculum 没有进入 X1 rigid-shape 刷新链。

公共 G0：

1. 建立 `x1_robot_manifest.yaml`，统一 joint order、sign、offset、频率、限制和固件 hash。
2. 训练、MuJoCo、部署复用相同 observation builder 和 phase generator。
3. 对齐 URDF、MJCF、实机的 position/velocity/torque/torque-speed 限制。
4. 修复随机化的每步重采样、累计相乘和无效 curriculum。
5. 用同一 recorded trace 回放，确认 51 维单帧和 66 帧历史逐项一致。

G0 通过标准：观测最大误差目标 `<1e-5`；joint order/sign/offset、phase、delay 语义一致；相同 state/action 下进入 simulator 前的 torque command 一致。

---

## 3. 方案一：SPI / SPI-Active

参考：[SPI-Active](https://arxiv.org/html/2505.14266v1)

### 3.1 适用场景

- 已有一个能在实机安全站立、低速行走的 `pi_collect`。
- 需要辨识 base/pelvis 质量、COM、惯量或整机自然运动中的执行器参数。
- 关注浮动基座、接触和多关节耦合，而不只是单关节响应。

`pi_collect` 只是数据采集工具。它能上机不代表仿真准确；辨识完成后仍需在新模型中重训最终 policy。

### 3.2 辨识流程

```text
选择安全数据采集策略 pi_collect
        ↓
设计普通高层运动片段 c_t
        ↓
实机运行 policy，记录状态 x_t 和实际控制输入 u_t
        ↓
将轨迹切成多个变时域 clips
        ↓
每个 clip 在仿真中对齐实机初始状态
        ↓
回放实机记录的 u_t，不重新运行 policy
        ↓
CMA-ES 优化物理参数 theta，得到 theta_1
        ↓
可选：优化 Fisher 信息最高的安全命令轨迹
        ↓
采集主动数据并再次辨识，得到 theta_active
        ↓
在 identified simulator 中重训最终 policy
```

### 3.3 X1 数据采集

初始运动片段：稳定站立、前向加减速、小幅后退、左右横移、左右转向、前进加转向圆弧，以及 `vx/vy/yaw` 分段组合。命令从已通过实机测试的低速包络开始。

必须记录：

```text
timestamp
vx/vy/yaw command
raw/clipped/scaled action
q_target before/after delay
q, dq, Kp, Kd, tau_pd
base position/orientation/linear velocity/angular velocity
IMU, foot contact, saturation and safety flags
```

关键目标：

\[
x_{t+1}^{sim}=f(x_t^{sim},u_t^{real};\theta)
\]

\[
\theta^*=\arg\min_\theta
\sum_{clips,t}\|x_t^{real}-x_t^{sim}(\theta)\|_{W_x}^{2}
+\|\theta-\theta_0\|_{W_\theta}^{2}
\]

### 3.4 SPI-Active

SPI-Active 在初始参数 `theta_1` 附近计算有限差分灵敏度，使用 CMA-ES 优化高层命令序列：

\[
c_{1:T}^{*}=\arg\min_c\operatorname{tr}[(F(\theta_1,c)+\epsilon I)^{-1}]
\]

X1 只优化 `vx/vy/yaw` 的 Bézier或分段曲线，不优化 12 维原始 action。目标中加入跌倒、倾角、限位、力矩、功率、温度和 command jerk 惩罚。

### 3.5 注意事项

1. 没有安全 `pi_collect` 时不能直接使用 SPI。
2. Replay 必须使用实机最终施加的 `q_target/torque`，不能只记录手柄命令。
3. Base pose/velocity 的测量和时间同步必须可信；否则降低权重或使用外部定位。
4. 不同时优化 PD gain、motor torque constant 和全局 torque scale 等尺度相关参数。
5. 低 loss 不代表真实参数唯一，必须报告置信区间和参数相关矩阵。
6. 主动命令必须先在 identified sim 通过安全约束，再上实机。
7. SPI 自然运动存在跌倒和接触冲击风险，采用“一轮一计划、一轮一结果”。
8. 当前 X1 只有三维速度命令，主动探索能力弱于论文的多行为策略。

### 3.6 输出与验收

输出：`initial_parameters.yaml`、`active_commands.yaml`、`refined_parameters.yaml`、参数协方差和 held-out 报告。

建议验收：held-out 多步误差相对 nominal 改善至少 40%；base 与 joint 指标同时改善；参数区间不覆盖大部分搜索范围；跨不同命令和 session 保持改善。

---

## 4. 方案二：UAN

参考：[Unsupervised Actuator Network](https://arxiv.org/html/2502.10894v1)

### 4.1 适用场景

- 执行器存在齿隙、滞回、死区、负载相关延迟或传动非线性。
- 简单 friction/armature/delay 白盒模型无法拟合 held-out 数据。
- 没有可靠的真实输出 torque 传感器。
- 接受较低可解释性，以换取更强的非线性表达能力。

UAN 不输出物理参数，而学习：

\[
\Delta\tau_t=\pi_{UAN}(e_{t-H:t})
\]

使：

\[
f_{sim}(s_t,\tau_t+\Delta\tau_t)\approx f_{real}(s_t,\tau_t)
\]

### 4.2 辨识流程

```text
固定或可靠支撑机器人
        ↓
采集方波、正弦、Gaussian 执行器数据
        ↓
形成真实 transition (s_t, tau_t, s_t+1)
        ↓
按完整轨迹/session 划分 train/validation
        ↓
仿真重置到真实状态并回放 tau_t
        ↓
UAN 根据 100 ms 历史输出 delta_tau_t
        ↓
仿真施加 tau_t + delta_tau_t
        ↓
以 q_sim-q_real 和输出平滑作为 reward
        ↓
使用 PPO 训练并验证长时 rollout
        ↓
冻结 UAN，放入 policy 训练仿真
        ↓
重训 locomotion policy；实机不运行 UAN
```

### 4.3 数据与网络

论文采集协议：

- 方波和正弦：一次一个 actuator，其他关节保持固定位置。
- 每个 actuator 使用 12 组幅值/频率组合，约 50 s 数据。
- Gaussian：所有关节同时输入，每 5–400 ms 重采样，约 5 min。

初始网络配置：

```text
MLP [128, 128], ELU
20-step history = 100 ms
update period = 5 ms = 200 Hz
4096 parallel environments
long rollout ≈ 20 s
```

记录 `timestamp、commanded torque/q_target、q、dq、Kp/Kd、tau_pd、current/effort、temperature、voltage、saturation flags`。

若 X1 只能发位置目标，仿真必须先复现同一 PD、delay 和 saturation，再叠加 UAN 残差。

### 4.4 X1 并联踝处理

论文按单 actuator 独立运行共享网络。X1 踝 pitch/roll 若存在交叉耦合，建议使用二维模型：

\[
[\Delta\tau_{pitch},\Delta\tau_{roll}]^T
=\pi_{ankle}(e_{pitch,t-H:t},e_{roll,t-H:t})
\]

- 输入 pitch/roll 两轴历史，输出二维残差。
- 左右踝先共享，held-out 数据显示不对称时再拆分。
- 能取得 motor-side 状态时，优先在电机坐标中建模并保留并联运动学映射。

### 4.5 注意事项

1. UAN 不能补偿错误的 joint order、PD、delay、limit 或刚体模型。
2. 网络只在数据覆盖的幅值、频率、负载和温度范围内可信。
3. 验证集按完整轨迹/session 划分，不能随机切相邻帧。
4. 单步误差小不代表稳定，必须验证 20 s 和数分钟长 replay。
5. `delta_tau` 必须有幅值、变化率、torque-speed 和 power 限制。
6. UAN 只在仿真中运行，不部署到实机。
7. UAN 已覆盖的 actuator properties 不再做宽 DR，避免重复建模。
8. 网络不应输入不必要的全身状态，避免记忆数据集和负载耦合。
9. 髋膝白盒模型已足够时，不给全部 12 关节增加 UAN。
10. 直接 torque 激励需要单独安全评审；无安全接口时使用已知 PD 的位置目标。

### 4.6 输出与验收

输出：训练/验证序列、`uan_checkpoint.pt`、normalization、torque limit 和长时 rollout 报告。

建议验收：未见方波/正弦/Gaussian 数据均优于 nominal、DR 和白盒基线；20 s 与 5 min replay 不发散；残差不长期贴限幅；踝 pitch/roll coupled-motion 误差下降；重训 policy 的实机表现同步改善。

---

## 5. 方案三：PACE

参考：[PACE 论文](https://arxiv.org/html/2509.06342v2)；[官方文档](https://pace.filipbjelonic.com/)

PACE：Precise Adaptation through Continuous Evolution。

### 5.1 适用场景

- 尚无能安全上机行走的 policy。
- 希望用少量、重复、低风险数据完成冷启动执行器辨识。
- 主要误差是关节响应、阻尼、低速摩擦、零偏和延迟。
- 希望得到紧凑、可解释的白盒参数。

PACE 解决固定基座关节/执行器层 Gap，不直接辨识 base mass/COM、地面摩擦和接触。

### 5.2 参数

对于 `n` 个关节：

\[
p=[I_a,d,\tau_f,q_{bias},T_d]^T\in\mathbb R^{4n+1}
\]

X1 的 12 个关节对应 49 个参数：每关节等效惯量、粘性阻尼、库仑摩擦、位置偏置，以及一个全局 command delay。

不优化 PD gains、motor torque constant、torque/velocity saturation 和已知 torque-speed curve。它们必须来自实机配置/制造商规格，并在仿真与实机中一致执行。

### 5.3 辨识流程

```text
固定或吊起 X1，base 不运动、腿完全无接触
        ↓
设置已知、保守、较低的 PD gains
        ↓
所有关节执行位置目标 chirp
        ↓
同步记录 q_target、q_real 和时间戳
        ↓
仿真固定相同 base，回放相同 q_target/Kp/Kd
        ↓
4096 个环境运行不同候选参数 p_e
        ↓
计算 joint-position trajectory MSE
        ↓
CMA-ES 优化得到 p_star
        ↓
用未见 random-step/不同 chirp/不同增益验证
        ↓
在 identified simulator 中训练 policy
        ↓
地面零样本部署和验证
```

### 5.4 Chirp 与优化

论文协议：单条 20–60 s，实验常用 20–40 s；起始约 0.1 Hz；最高 2–10 Hz，取决于结构；完整机器人日志和低层控制至少 400 Hz。仿真和实机必须接收完全相同的位置目标。

PD gains 有意设置得较低，使闭环主极点落入可激励频带。高增益会要求悬空硬件难以安全达到的高频激励。

损失：

\[
\ell_e=\frac{1}{k}\sum_i\|q_i^{real}-q_{i,e}^{sim}\|^2
\]

所有参数按物理上下界归一化到 `[-1,1]`；论文设置 CMA-ES `mean=0`、`sigma=0.5`。固定基座和无接触消除了长时 locomotion 漂移，因此可以直接使用时域 joint-position MSE。

### 5.5 X1 并联踝处理

- 若固件已把电机坐标解耦成 ankle pitch/roll，可先在虚拟关节坐标执行 PACE。
- 若仍有明显交叉耦合，应在 motor coordinates 中辨识或引入 2×2 耦合模型。
- 独立的 `I_a/d/tau_f/q_bias` 不能表达所有并联机构非线性。
- 悬空通过后仍按“悬空 → 双脚落地 → 行走 touchdown 前后 100–150 ms”验证；落地数据不混入原始 PACE 拟合。

### 5.6 注意事项

1. Base 必须真正固定，腿、吊带和线缆不能引入未知接触力。
2. 实机与仿真的 `q_target/Kp/Kd/target clipping` 必须一致。
3. 时间戳定义错误会被 `T_d` 吸收，得到错误延迟。
4. 不同时优化 PD gains、torque constant 与动力学参数。
5. Saturation 是已知边界，不是自由参数；辨识尽量远离饱和。
6. 低辨识增益不能在 policy 训练或部署时被静默替换；若更换必须重新验证。
7. PACE 只用无接触数据，不能宣称辨识了 ground friction/contact。
8. 必须用未见 random-step 或不同 chirp 验证，不能只看训练轨迹。
9. 串联仿真与真实并联踝结构不一致时，CMA-ES 只能得到最优近似。
10. 论文不做 dynamics DR；X1 在温度、磨损和装配差异未测清前，建议保留辨识后验的窄 DR。

### 5.7 输出与验收

输出：原始 chirp、parameter bounds、`identified_parameters.yaml`、CMA trace、悬空和落地验证报告。

建议验收：held-out 悬空髋膝 `q RMSE <0.02 rad`、踝 `<0.015 rad`；延迟误差 `<2 ms` 或 `<实测值 20%`；相对 nominal 的 held-out MSE 改善至少 40%；跨幅值、频率和已知 PD gains 保持改善。

---

## 6. 独立对照实验与选择规则

| 组别 | 模型 |
|---|---|
| B0 | 修复契约后的 nominal simulator |
| B1 | 当前 broad DR，先修复随机化 bug |
| S1 | SPI identified simulator |
| S2 | SPI-Active refined simulator |
| U1 | UAN corrected simulator |
| P1 | PACE identified simulator |

各组共享 X1 manifest、URDF/MJCF 基线、policy 架构、reward、训练预算、至少 3 个 seed，以及相同 held-out 实机测试。报告轨迹误差、Isaac-MuJoCo gap、速度/姿态/接触、饱和、功率、温度、多 session 泛化和安全事件。

选择规则：

```text
没有可安全上机的 locomotion policy？
  → PACE

已有安全 policy，主要问题是 base/整机自然运动不一致？
  → SPI；信息不足时增加 SPI-Active

白盒模型仍有稳定滞回、死区或并联耦合残差？
  → UAN
```

当前建议：先完成公共 G0，再将 PACE、SPI、UAN 作为三个独立实验分支。若资源只允许先做一个，PACE 的前置条件最少、安全性最高，可作为 X1 第一优先候选，但不预判最终一定优于 SPI 或 UAN。

## 7. 实机轮次与安全

三个方案都采用“一轮一计划、一轮一结果”：

```text
deploy/sim2real/plans/{spi|uan|pace}_round_NN_*.md
deploy/sim2real/results/{spi|uan|pace}_round_NN_*.md
```

每轮必须写明唯一目标、固定参数、输入幅值/频率/时长、采样字段、Pass/fail、停止条件和本轮禁止修改项。

立即停止条件：超过批准的位置/速度/电流/力矩/温度限制；持续振荡或异常声响；通信/时间戳异常；倾倒或吊架受力异常；operator 急停。发生安全事件后先记录配置 hash、工况、时间戳和恢复过程，不同时修改多个参数继续测试。

## 8. 下一步

完成 G0 后分别准备三个最小实验：

1. **PACE MVP**：一个关节组的小幅 chirp，验证数据链和 CMA replay。
2. **UAN MVP**：一个复杂踝关节的正弦/方波数据，验证 transition replay 和长时稳定性。
3. **SPI MVP**：仅在已有安全 `pi_collect` 时，采集低速前进/横移/转向，验证 full-state clip replay。

使用同一 held-out 指标比较后，再决定 X1 主方案。

