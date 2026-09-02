# X1 SPI / SPI-Active 详细运行步骤

> 状态：Runbook / 分阶段待实施  
> 日期：2026-08-11  
> 依据：`docs/archive/x1_sim2real_system_design.md`、当前 `X1_train` 和 `X1_infer` 代码  
> 参考：[Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning](https://arxiv.org/html/2505.14266v1)

## 0. 结论与当前起点

当前已经存在能够初步实机行走的 policy，因此 X1 已满足 SPI 的关键业务前提：存在可用于自然运动数据采集的 `pi_collect` 候选。当前候选为：

```text
X1_infer/module/control_module/policy/rl_walk_leg.onnx
SHA1: e86691fccc438daa5a1e41f7edac7af67c613963
```

当前部署接口为：

- 下肢 12 维 action。
- 单帧 observation 47 维，历史 66 帧。
- 控制主循环 1000 Hz，policy decimation 为 10，即 policy 100 Hz。
- action scale 为 0.5。
- 高层命令为 `vx/vy/yaw`，控制器实际订阅 `/cmd_vel_limiter`。
- 手柄模块当前以 20 Hz 发布速度命令。
- 当前部署诊断 CSV 已记录 clip 后 action、关节状态、最终位置/力矩目标和 IMU，但单次约 10 秒。

这意味着可以立即开始 SPI Stage 1 的软件准备和低速采集链路验证，但还不能直接宣称已经具备正式 SPI-Active 闭环。当前阻塞项是：

1. 实机没有可直接用于 SPI 的全局 base position 和 base linear velocity 真值。
2. 当前 `sim2sim.py` 会重新运行 policy，不是从完整实机状态回放最终控制输入的 SPI replay。
3. 没有参数化批量 replay、CMA-ES、有限差分 FIM 和 active-command 优化程序。
4. 当前行走诊断日志时长硬编码为 10 秒，不能覆盖完整主动轨迹。
5. 当前 `joy_x1.yaml` 启用了固定 `0.4 m/s` 前进命令，正式采集前必须关闭，避免与脚本命令争抢。
6. `RLController::GetJointCmdData()` 中 position clamp 当前被注释，主动探索前必须确认驱动层有独立可靠限位，或恢复并测试部署侧限位。
7. 当前没有统一记录 ONNX raw action、足端接触、执行器饱和、功率、电流、温度和安全状态。

本文所有步骤使用以下标签：

- **[当前可运行]**：仓库已经存在对应入口。
- **[需扩展]**：在现有模块上补字段或配置。
- **[待实现]**：当前仓库没有对应程序，必须完成后才能通过相关 Gate。

SPI-Active 的正式顺序必须是：

```text
安全 pi_collect
  → 普通运动数据 D0
  → 初始 SPI 得到 theta_1
  → 在 theta_1 及其不确定性范围内优化安全命令
  → 实机主动数据 D1
  → 精化参数 theta_active
  → identified simulator 重训 policy
  → sim2sim 和实机逐级对照
```

禁止跳过初始 SPI，直接在 nominal simulator 上生成激进主动命令上机。

---

## 1. 辨识边界

### 1.1 SPI 中的三个量

必须区分：

```text
c_t：高层 vx/vy/yaw 命令
a_t：policy 输出的 raw/clipped action
u_t：实机最终施加到动力学链的控制输入
```

X1 当前控制链不是简单的 `a_t → torque`：

```text
vx/vy/yaw
  → 47 × 66 observation history
  → ONNX policy
  → clipped action
  → action_scale + init_state
  → LPF
  → 普通关节：position target + Kp/Kd
  → 并联关节：PD torque → LPF → effort command
  → joint offset / transmission / driver
  → hardware
```

因此 replay 必须使用最终 `u_t`：

- 普通关节：最终 `position/velocity/effort/stiffness/damping`。
- 并联踝：最终 LPF 后 effort command 及传动映射。
- 若驱动层还有 torque-speed 限制、电流环限幅、延迟或二次滤波，也必须记录或在 replay 中复现。

只记录 `vx/vy/yaw` 或 raw action 不能完成 SPI。

### 1.2 实机数据的运动边界

PACE 的正式数据来自固定基座悬空试验；SPI 的正式数据来自浮动基座、足端真实接触的自然运动。SPI 数据采集时：

- 机器人必须在地面实际承重行走。
- 肩吊只能作为微松防坠保护，不能持续承重或牵引 base。
- 吊带一旦参与正常受力，测得的是“机器人 + 吊装系统”的动力学，不能进入正式 SPI 数据集。
- 地面材质、污染、平整度和鞋底状态必须记录并保持一致。

### 1.3 第一版参数集合

SPI MVP 不应一次优化所有可随机化参数。推荐分两阶段：

#### Phase A：base/pelvis 惯性参数

- 聚合到与 simulator 一致的 base/pelvis link mass。
- COM 三轴偏移。
- 完整对称 inertia，使用 log-Cholesky/pseudo-inertia 参数化保证物理可行。

#### Phase B：执行器组参数

- 关节组 torque saturation/tanh gain。
- 经独立测量确认后才允许加入 torque scale 或 delay。
- 左右对称关节优先共享参数，只有残差支持时才拆分左右。
- 并联踝单独成组，不与串联髋膝共用参数。

第一轮禁止同时自由优化以下尺度相关参数：

```text
PD gain
全局 torque scale
motor torque constant
per-joint torque saturation
ground friction
```

否则很容易得到低 loss 但不可辨识的参数组合。

地面摩擦、restitution、contact stiffness/damping 在 SPI MVP 中先测量或固定。只有跨地面 held-out 残差明确指向接触参数时，才建立独立接触辨识轮次。

参数上下界必须来自 X1 称重、CAD/BOM、装配公差、执行器台架或已完成的 PACE 结果，不得照搬论文中其他机器人的范围。

---

## 2. 目录和轮次约定

计划目录：

```text
deploy/sim2real/
├── spi_active_runbook.md
├── configs/
│   ├── x1_robot_manifest.yaml
│   ├── spi_parameters_mvp.yaml
│   ├── spi_loss.yaml
│   └── spi_safety.yaml
├── commands/
│   ├── spi_stage1_heuristic.yaml
│   └── spi_active_round_NN.yaml
├── plans/
│   └── spi_round_NN_<name>.md
├── results/
│   └── spi_round_NN_<name>.md
└── artifacts/spi/<run_id>/
    ├── manifest.yaml
    ├── hashes.txt
    ├── raw/
    ├── synced/
    ├── clips/
    ├── cma/
    ├── fim/
    ├── validation/
    └── identified/
```

`run_id` 建议格式：

```text
spi_YYYYMMDD_robotSerial_policyShortHash_roundNN
```

每次实机只允许执行一个已冻结的 `spi_round_NN_*.md`。计划文件至少包含：

- 唯一目标。
- policy、配置、命令文件和代码 hash。
- 地面、鞋底、外部定位和吊装状态。
- 速度、加速度、jerk、时长和重复次数。
- 必录字段。
- Pass/fail 和立即停止条件。
- 本轮禁止修改项。

每轮结束后先写结果文件，再决定下一轮；不得在现场临时提高命令幅值并继续采集。

---

## 3. 角色、场地与安全

每次实机运行至少需要：

- 操作者：负责状态切换和硬件急停，全程不离开急停。
- 安全观察员：只观察机器人、吊带、足端、线缆和环境。
- 数据负责人：核对计划 hash、外部定位、日志和数据质量。

场地要求：

- 平整、清洁、摩擦一致的测试地面。
- 有足够停止距离和转向空间，轨迹边界外无人员和硬物。
- 肩部防坠绳微松，正常行走时不承重、不限制 yaw 和平移。
- 线缆不会牵引机器人或进入足端区域。
- 外部定位在全区域无遮挡，坐标系已标定。
- 硬件急停独立于 command player 和 Linux 进程。

立即停止条件必须写入 `spi_safety.yaml` 并由现场负责人批准。至少覆盖：

- roll/pitch 超限或持续增长。
- base height 低于阈值。
- 关节位置接近硬限位。
- 关节速度、力矩、电流、功率或温度超限。
- action/torque saturation 持续超过批准帧数。
- 足端异常打滑、交叉、绊碰或非预期腾空。
- command、policy、state 或 mocap heartbeat 超时。
- 时间戳倒退、数据丢帧或控制周期异常。
- 异常声响、振动、通信告警或吊带突然受力。
- 操作者主动急停。

所有数值阈值使用整机和执行器批准值。本文不替代硬件安全规范。

---

## 4. 总体 Gate

| Gate | 内容 | 通过条件 |
|---|---|---|
| G0 | 软件契约和版本冻结 | 训练、sim、infer 的 joint/obs/action/time 语义一致 |
| G1 | `pi_collect` 实机资格 | 可重复站立和低速行走，得到批准的安全命令包络 |
| G2 | SPI 采集链 | 全量状态、最终控制输入、外部 base 真值和安全量同步可用 |
| G3 | full-state replay | recorded `u_t` 可开环回放，synthetic recovery 通过 |
| G4 | Stage 1 普通数据 D0 | 覆盖站立、纵向、横向、转向和组合运动，质量通过 |
| G5 | 初始 SPI | 得到稳定 `theta_1`，held-out 明显优于 nominal |
| G6 | Active 命令设计 | FIM 改善且在参数 ensemble 中全部通过安全约束 |
| G7 | 主动实机数据 D1 | 每轮一计划一结果，无安全事件，信息增益有效 |
| G8 | refined SPI | `theta_active` 的 held-out 和置信区间优于 `theta_1` |
| G9 | identified 模型重训 | 至少 3 seeds，训练和导出可复现 |
| G10 | sim2sim/实机对照 | S2 相对 B0/B1/S1 改善且无安全回退 |

任一 Gate 未通过时，不得以“policy 已经能走”为理由跳过。

---

## 5. G0：冻结软件和模型契约

### 5.1 建立本轮 manifest

记录：

```bash
git -C X1_train rev-parse HEAD
shasum X1_infer/module/control_module/policy/rl_walk_leg.onnx
shasum X1_infer/module/control_module/cfg/rl_x1.yaml
shasum X1_infer/module/joy_stick_module/cfg/joy_x1.yaml
shasum X1_infer/module/dcu_driver_module/cfg/dcu_x1.yaml
shasum X1_infer/module/dcu_driver_module/cfg/ankle_trans_x1.yaml
shasum X1_train/resources/robots/x1/urdf/x1.urdf
shasum X1_train/resources/robots/x1/mjcf/xyber_x1_flat.xml
```

同时记录：

- 机器人和执行器序列号。
- DCU、驱动器和 EtherCAT 版本。
- policy 来源 checkpoint；若 checkpoint 不在仓库，记录训练任务、导出日期和交付人。
- 29 关节部署顺序与 12 个 policy 关节顺序。
- joint sign、offset、position/velocity/torque/torque-speed limit。
- 训练/推理 observation 顺序、缩放、clip 和 66 帧历史更新方向。
- 1000 Hz 控制周期、100 Hz policy 周期及实际统计分布。
- LPF 参数、并联踝列表、Kp/Kd 和 action scale。

### 5.2 关闭已知契约 Gap

正式 SPI 前，必须先完成总体方案公共 G0：

1. 训练、MuJoCo 和部署统一到 51 维或明确冻结到同一个 47 维版本，不能混用。
2. 修复 `sim2sim.py` body name 条件和 gait phase 不一致。
3. 对齐 base linear velocity 坐标系。
4. 对齐 Isaac、MuJoCo 和实机 torque limit。
5. 修复 torque multiplier 每 1 ms 重采样问题。
6. 修复 friction/damping reset 累乘问题。
7. 确认 friction curriculum 真正作用到 X1 rigid shape。

SPI replay 的误差必须来自物理参数，而不是 observation/action 契约错误。

### 5.3 验证 ONNX 接口

**[待实现]** 增加离线接口检查：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.verify_policy_contract \
  --onnx X1_infer/module/control_module/policy/rl_walk_leg.onnx \
  --deploy-config X1_infer/module/control_module/cfg/rl_x1.yaml \
  --task x1_dh_stand \
  --manifest deploy/sim2real/configs/x1_robot_manifest.yaml
```

通过条件：

- ONNX input 为 `47 × 66 = 3102` 或与冻结版本完全一致。
- output 为 12。
- 同一 observation trace 下 Python 和 C++ action 最大误差达到约定阈值。
- raw action、clip 后 action、position target 和 offset 逐项一致。

---

## 6. G1：把现有 policy 定义为 `pi_collect`

“能初步走”还不能直接等于“可主动采集”。先生成 `pi_collect_qualification.md`。

### 6.1 关闭固定速度配置

当前文件中启用了：

```yaml
constant_velocity:
  linear-x: 0.4
  linear-y: 0.0
  angular-z: 0.0
```

正式 SPI 采集配置必须删除或注释该段，并生成单独的 `joy_x1_spi.yaml`。禁止修改现场唯一配置后不留 hash。

### 6.2 确定实机安全包络

训练配置范围不是实机批准范围。分别测出：

```text
Vx_safe = [vx_min_safe, vx_max_safe]
Vy_safe = [vy_min_safe, vy_max_safe]
Wz_safe = [wz_min_safe, wz_max_safe]
Ax_safe, Ay_safe, Awz_safe
Jx_safe, Jy_safe, Jwz_safe
```

当前手柄配置的 `[-0.5, 0.5] m/s`、`[-0.3, 0.3] m/s`、`[-0.5, 0.5] rad/s` 只是软件上限，不代表已经验证的安全包络。

资格测试顺序：

1. 原地站立。
2. 低速前进并回零。
3. 低速后退并回零。
4. 低速左右横移并回零。
5. 低速左右转向并回零。
6. 小曲率前进加转向。
7. 重复三次，包含冷机和热机 session。

每一步只使用上一通过幅值的增量，不在一次试验中同时增加速度、时长和转向。

### 6.3 `pi_collect` 通过条件

- 每类基础命令至少 3 次无人工救援完成。
- 无跌倒、非预期吊带受力、明显打滑和异常接触。
- action、关节、力矩、电流和温度均有安全余量。
- 命令回零后可恢复稳定站立。
- 同一命令的 base/joint 轨迹具有可重复性。
- 全程日志无丢帧和时间戳错误。

如 policy 只能稳定前进，不能稳定横移或转向，则先做 SPI-forward 子集，不得让 active optimizer 使用未通过的命令维度。

---

## 7. G2：实现 SPI 采集链

### 7.1 当前已有数据

**[当前可运行]** `walk_diag_*.csv` 当前约以 100 Hz 记录：

- timestamp。
- phase sin/cos。
- `vx/vy/yaw`。
- base Euler 和 angular velocity。
- clip 后 policy action 和 clip count；ONNX raw action 当前未保留。
- 12 关节 position/velocity/effort。
- raw/LPF 后 position target。
- raw/LPF 后 torque target。
- 并联关节标志。
- IMU quaternion、gyro、acceleration。

**[当前限制]**：

- 最大帧数硬编码为 `10 × (1000 / decimation)`，当前约 1000 帧，即 10 秒。
- 进入 RL 状态时触发，达到帧数后不会在同一状态自动开始新文件。
- timestamp 使用本进程时钟，未显式保存各传感器源时间和 sequence。

### 7.2 必须增加的数据字段

**[需扩展]** 正式数据至少包含：

| 类别 | 字段 |
|---|---|
| 身份 | run/session/trial/plan id，robot/policy/config hash |
| 时间 | monotonic timestamp、ROS/source timestamp、sequence、arrival timestamp |
| 高层命令 | planned、raw、limited、actually consumed `vx/vy/yaw` |
| policy | observation hash/可选原始 history、raw action、clipped action、inference latency |
| 最终控制 | `q_des/dq_des/tau_ff/Kp/Kd`，LPF、delay 和限幅后的值 |
| 关节状态 | q、dq、effort、motor-side state、offset 后状态 |
| base 真值 | world position、quaternion、linear velocity、angular velocity |
| IMU | orientation、gyro、accel、source timestamp |
| 接触 | 左右足 contact、法向力/COP（若硬件可用） |
| 执行器 | current、voltage、temperature、fault、saturation |
| 安全 | tilt/base-height/joint-limit/heartbeat/fall/slip/estop flags |
| 环境 | 地面 id、鞋底 id、温度、负载和外部定位质量 |

`JointState.effort` 未经标定时只能作为估计量，不得在 loss 中当作真实输出 torque。

### 7.3 获取实机 floating-base 真值

SPI 论文使用的完整状态包括 base position、orientation、linear/angular velocity 和 joint position/velocity。X1 当前仅凭 IMU 不能稳定恢复长时间全局 position 和低频 linear velocity。

推荐顺序：

1. 光学 motion capture，base/pelvis 上安装刚性 marker cluster。
2. 高质量外部定位融合 IMU，输出带协方差的 `nav_msgs/Odometry`。
3. 仅在已验证的状态估计器不依赖待辨识 nominal 参数时，才将 estimator 输出作为次选。

统一发布到：

```text
/spi/base_state
frame_id: spi_world
child_frame_id: base_link
```

标定内容：

- `spi_world` 到试验场地坐标系。
- marker rigid body 到 `base_link`。
- quaternion 顺序和右手系。
- base linear/angular velocity 的表达坐标系。
- mocap、控制机和 DCU 的时钟偏差与漂移。

### 7.4 实现专用采集器

**[待实现]** 建议新增 `spi_data_recorder`，不要仅依赖 `ros2 bag record -a`：

```text
X1_infer/assistant/spi_data_recorder/
```

采集器要求：

- 原始 topic 同步写入 rosbag，统一数据同时写入列式文件或 HDF5。
- 记录每个来源的 source/arrival/monotonic 时间。
- 保存 joint name，不依赖隐式列序。
- 发现 topic 超时、时间倒退、sequence 跳变立即置 safety flag。
- 日志时长、目录和 trial id 配置化。
- 正常停止和急停都 flush，并生成 `data_quality.json`。

现有 `run_with_recording.sh` 会运行 `ros2 bag record -a`，可保留为原始备份；正式分析以同步后的统一数据为准。

### 7.5 采样与同步规则

- 保留各 topic 原始频率，不先丢数据。
- 最终 motor/joint command 按零阶保持映射到 1 ms physics grid。
- state 只在真实采样点参与损失，或使用明确记录的插值方法。
- policy action 在 100 Hz tick 对齐，不把 20 Hz command 更新误认为 policy tick。
- base quaternion 使用球面插值并统一符号，yaw 只用于展示时展开。
- 不用简单移动均值跨越足端接触切换。
- 估计并报告 command、state、IMU、mocap 的相对延迟。

通过条件：

- 所有必需 topic 覆盖整个 trial。
- 无 timestamp 倒退。
- 控制输入和关节状态丢帧率为 0；其他传感器丢帧率低于项目阈值。
- 时间同步残差小于最小拟合动态的时间尺度。
- 由 CSV、rosbag 和统一文件重建的命令/状态相互一致。

---

## 8. 实现安全 Active Command Player

### 8.1 命令链

**[待实现]** 新增：

```text
active_commands.yaml
  → spi_command_player
  → /spi/cmd_vel_raw
  → spi_safety_supervisor + limiter
  → /cmd_vel_limiter
  → RLController
```

禁止 active player 和 JoyStickModule 同时发布 `/cmd_vel_limiter`。手柄只保留模式切换和独立停止权限。

### 8.2 Player 必须具备

- 只加载带 SHA/hash 的冻结命令文件。
- 显式 `ARM → READY → RUN → RAMP_TO_ZERO → DONE/ABORT` 状态机。
- 默认输出零命令，收到 arm token 和人工确认后才执行。
- 固定发布频率，当前建议与上游命令链一致为 20 Hz。
- 每点经过 position/velocity、acceleration 和 jerk 限制。
- heartbeat 丢失、state 超时或 safety flag 时立即平滑回零并切换 stand/keep。
- 硬件急停优先级最高，不依赖平滑回零。
- 发布 planned、limited 和 executed command，带 sequence 和 plan hash。
- trial 前后自动保留站立零命令段。

### 8.3 仿真优先验证

同一个 player 必须先接入 `x1_cfg_sim.yaml` 干跑，确认：

- 命令方向正确。
- limiter 后波形与计划一致。
- command zero 后 policy 恢复站立。
- 中途 kill player、停止 mocap、注入超时均触发安全回零。
- JoyStickModule 不再输出冲突速度。

---

## 9. G3：建立 full-state clip replay

### 9.1 Replay 的严格定义

对每个 clip：

1. 将 simulator 的 base pose/velocity、joint q/dq 初始化为实机 clip 首帧。
2. 使用与实机一致的地面、鞋底和 contact 设置。
3. 按实机时间轴施加 recorded final `u_t`。
4. 后续不重新运行 ONNX policy，不用模拟状态反馈修正实机 action。
5. 在实机状态采样时刻比较 simulation prediction。

当前 `X1_train/humanoid/scripts/sim2sim.py` 会构造 observation 并重新运行 policy，不满足第 3、4 项，只能作为部署接口参考。

### 9.2 Clip 生成

**[待实现]**：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.spi_prepare_data \
  --manifest deploy/sim2real/artifacts/spi/<run_id>/manifest.yaml \
  --input deploy/sim2real/artifacts/spi/<run_id>/raw \
  --output deploy/sim2real/artifacts/spi/<run_id>/clips \
  --horizon-min 0.5 \
  --horizon-max 2.0
```

规则：

- horizon 采用变长度，避免只适配固定预测长度。
- clip 不跨越急停、数据缺失、状态切换和明显定位跳变。
- 初始状态由真实首帧给定。
- train/validation/test 按完整 session 或完整轨迹划分，不能随机打散相邻 clip。
- 至少保留一个不同日期或不同温度 session 作为最终 held-out。

论文验证平均约 1.5 秒 clip；X1 应在 0.5、1.0、1.5、2.0 秒上同时报告，避免只在短 horizon 有效。

### 9.3 Replay 控制语义

普通关节与并联踝必须分别处理：

```text
普通关节：replay 最终 q_des/dq_des/tau_ff/Kp/Kd
并联踝：replay 最终 joint-side effort，并复现实机传动和 motor limit
```

若 simulator 只接受 torque，则使用 recorded final command 和 recorded q/dq 语义构造与实机一致的 PD/effort torque；不能重复添加已经存在于记录值中的 LPF 或 delay。

### 9.4 Synthetic parameter recovery

正式拟合实机前必须先做合成恢复：

1. 在 simulator 中设置已知 `theta_hidden`。
2. 运行普通命令，生成与实机完全相同 schema 的 synthetic data。
3. 从 nominal `theta_0` 启动 CMA-ES。
4. 检查能否恢复 `theta_hidden`。
5. 分别注入时间偏差、噪声、dropout 和接触差异，确认诊断能发现问题。

如果 synthetic recovery 失败，实机低 loss 没有可信解释。

---

## 10. G4：Stage 1 普通数据 D0

### 10.1 命令设计

从 G1 已批准的安全包络开始，以包络比例表达，不在 runbook 中硬编码未经验证的速度：

| 片段 | 命令 |
|---|---|
| H0 | 原地站立 |
| H1 | `+vx` 25%、50%、75% safe envelope |
| H2 | `-vx` 25%、50% safe envelope |
| H3 | `+vy/-vy` 25%、50% safe envelope |
| H4 | `+wz/-wz` 25%、50% safe envelope |
| H5 | `vx + wz` 左右圆弧 |
| H6 | `vx + vy` 小幅组合 |
| H7 | 分段加速、匀速、减速并回零 |

每个 trial 建议结构：

```text
2 s stand
1–2 s ramp in
2–4 s motion
1–2 s ramp out
2 s stand
```

在现有 10 秒 logger 未扩展前，每个 trial 必须控制在单文件覆盖范围，并在每次进入 walk 状态后检查新日志是否真正触发。正式版本应先把时长配置化，再执行较长轨迹。

### 10.2 重复和划分

- 每种基础片段至少 3 次。
- 至少两个独立上电/温度 session。
- 顺序随机化，避免所有高幅值都出现在热机阶段。
- D0-train、D0-validation、D0-test 按 trial/session 预先冻结。
- 最终 held-out 在任何 loss weight、bound 和参数选择过程中不得使用。

### 10.3 单轮实机步骤

1. 核对 plan、policy、配置、command 和安全阈值 hash。
2. 检查地面、鞋底、marker、吊带、线缆和急停。
3. 上电进入 zero/stand，保持零命令。
4. 检查 joint、IMU、mocap、contact、power 和 temperature heartbeat。
5. 启动 rosbag 和 `spi_data_recorder`。
6. arm command player，但保持 READY。
7. 人工口令确认后进入 RUN。
8. 执行唯一冻结片段。
9. 自动 ramp to zero，稳定站立。
10. 退出 walk，停止记录并 flush。
11. 立即生成 data quality，不合格数据不得进入下一轮。
12. 填写 result 文件并决定 Pass/fail。

### 10.4 数据质量检查

**[待实现]**：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.spi_check_data \
  --run deploy/sim2real/artifacts/spi/<run_id> \
  --manifest deploy/sim2real/artifacts/spi/<run_id>/manifest.yaml
```

至少输出：

- topic 覆盖率和采样频率分布。
- timestamp jitter、drop 和相对延迟。
- command planned/limited/executed 差异。
- action/torque/current/temperature saturation 比例。
- joint limit margin。
- roll/pitch/base height/contact/foot slip。
- mocap residual、遮挡和速度异常点。
- 每个命令维度和状态特征覆盖图。

---

## 11. G5：初始 SPI 得到 `theta_1`

### 11.1 损失函数

使用多步 prediction：

```text
J(theta) = Σclip Σt rho(||x_real - x_sim(theta)||²_Wx)
         + ||theta - theta_0||²_Wtheta
         + physical_constraint_penalty
```

状态至少包含：

```text
base world position
base quaternion
base linear velocity
base angular velocity
joint position
joint velocity
```

建议：

- quaternion 使用符号不变的 orientation distance。
- 各 loss 项先用 nominal validation 数据归一化，再设置全局权重。
- 对 mocap 瞬时异常使用 Huber 等 robust loss，不静默删除大片数据。
- 未校准 joint effort 不进入主损失。
- 参数 prior 反映真实测量不确定性，而不是为追求低 loss 任意放宽。

### 11.2 物理参数化

mass/COM/inertia 使用物理可行参数化，禁止 CMA 直接采样可能非正定的 inertia matrix。每个候选参数加载后检查：

- mass > 0。
- inertia 正定。
- 三角不等式/伪惯量条件满足。
- COM 位于机械允许范围。
- simulator 能稳定初始化。

### 11.3 CMA-ES 运行

**[待实现]**：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.spi_fit \
  --config deploy/sim2real/configs/spi_parameters_mvp.yaml \
  --loss deploy/sim2real/configs/spi_loss.yaml \
  --clips deploy/sim2real/artifacts/spi/<run_id>/clips/train \
  --validation deploy/sim2real/artifacts/spi/<run_id>/clips/validation \
  --output deploy/sim2real/artifacts/spi/<run_id>/cma/stage1 \
  --seed 11
```

要求：

- GPU 并行评估 candidate。
- 至少 3 个 CMA seed。
- 每代保存 distribution mean/covariance、best、validation loss 和 invalid candidate 比例。
- 以 validation 和收敛判据停止，不照搬固定迭代数。
- 单独报告不同 horizon、不同命令和不同 session 的误差。
- 使用 bootstrap 或 profile likelihood 估计置信区间和参数相关性。

### 11.4 `theta_1` 通过条件

- synthetic recovery 已通过。
- 相对 nominal，D0-validation 和 D0-test 的 base 与 joint 指标同时改善。
- 建议 held-out 多步综合误差相对 nominal 改善至少 40%。
- 0.5–2.0 秒 horizon 都改善，不只改善单步。
- 多个 CMA seed 收敛到一致区域。
- 参数未长期贴边界。
- 参数置信区间没有覆盖大部分搜索范围。
- 参数相关矩阵没有无法解释的近完全相关。

若只降低 joint loss 却恶化 base，或只降低短 horizon loss，则不得进入 Active 阶段。

---

## 12. G6：设计 SPI-Active 命令

### 12.1 只优化安全高层命令

X1 第一版只优化：

```text
c_t = [vx, vy, wz]
```

禁止直接优化 12 维 raw action。现有 policy 未验证的横移或转向维度必须锁零。

### 12.2 有限差分灵敏度

围绕 `theta_1` 计算：

```text
S_i(t) ≈ [feature(theta_1 + delta_i) - feature(theta_1 - delta_i)] / (2 delta_i)
F = Σt S(t)^T R^-1 S(t) + epsilon I
```

要求：

- `delta_i` 相对参数尺度设置，并做步长收敛检查。
- feature 与 Stage 1 state loss 一致并按测量噪声 whiten。
- 使用 central difference；边界处使用合法单边差分并标记。
- FIM 的 condition number、eigenvalue 和参数 pair correlation 全部保存。
- 对不敏感参数先冻结，不靠 active 命令强行同时辨识全部参数。

### 12.3 命令参数化

论文使用固定时间段内的 Bézier command。X1 建议从更小的变量数开始：

- 每段 4 秒。
- `vx/vy/wz` 分别使用三次或五次 Bézier曲线。
- 相邻段 position、slope 连续。
- 首尾命令为零。
- 总主动 trial 从 8–12 秒开始，采集链验证后再增加。

只有在低阶曲线不能提供信息增益时，才增加控制点或使用论文中的更高阶参数化。

### 12.4 Active objective

基础目标为 A-optimal criterion：

```text
J_active = trace(F^-1)
         + fall/termination penalty
         + tilt penalty
         + joint-limit penalty
         + torque/current/power/temperature penalty
         + slip/contact penalty
         + command acceleration/jerk penalty
         + uncertainty-robustness penalty
```

不能只在单点 `theta_1` 上评估安全。应从 Stage 1 covariance/bootstrap 中采样 parameter ensemble，对 nominal、置信边界和随机样本全部 rollout；任一关键样本跌倒或超限则 candidate 无效。

### 12.5 Active CMA-ES

**[待实现]**：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.spi_optimize_commands \
  --parameters deploy/sim2real/artifacts/spi/<run_id>/identified/theta_1.yaml \
  --covariance deploy/sim2real/artifacts/spi/<run_id>/identified/theta_1_cov.npy \
  --policy X1_infer/module/control_module/policy/rl_walk_leg.onnx \
  --safety deploy/sim2real/configs/spi_safety.yaml \
  --output deploy/sim2real/commands/spi_active_round_01.yaml \
  --seed 21
```

每个候选命令必须保存：

- FIM score、eigenvalues 和 condition number。
- 相对 D0 heuristic 命令的信息增益。
- ensemble 中最大 roll/pitch、base height、joint margin。
- torque/current/power proxy 和 saturation。
- termination/slip/contact 统计。
- command velocity/acceleration/jerk。

### 12.6 Active 命令的仿真 Gate

候选命令只有同时满足以下条件才可进入实机审批：

1. 信息指标优于同等时长 D0 命令。
2. 所有硬安全约束均有额外 margin。
3. nominal、`theta_1`、置信边界和随机 ensemble 均不跌倒。
4. Isaac 和 MuJoCo 至少进行一次交叉验证，差异可解释。
5. command player 的实际限幅后波形仍保持信息优势。
6. 中途任意时刻 abort 都能安全回零或进入站立。
7. 视频和曲线经过控制、机械、安全三方审核。

FIM 高但依赖单脚极限、接近关节限位或持续饱和的命令必须拒绝。

---

## 13. G7：主动实机数据 D1

### 13.1 渐进放量

不能一次执行 optimizer 的完整幅值。每个 active command 按以下顺序：

1. simulator 100% 幅值。
2. 实机 25% 幅值，只验证方向、节奏和停止。
3. 数据通过后单独批准 50%。
4. 数据通过后单独批准 75%。
5. 只有前三轮安全且信息响应一致时，才考虑 100%。

幅值比例同时作用于速度和曲线变化量；acceleration/jerk 仍受独立硬限制。

### 13.2 一轮一结果

每轮 active 实机只运行一条冻结命令，最多做计划中已批准的重复。结束后检查：

- 实际 limited command 是否与仿真使用的波形一致。
- safety margin 是否符合预测。
- base/joint/contact feature 是否进入预测覆盖范围。
- 估算 FIM 是否确有提升。
- 是否出现新饱和、打滑、温升或定位问题。

未完成结果审查前，不执行下一幅值或下一条命令。

### 13.3 D1 数据隔离

- D1-active-train 用于 refined fit。
- D1-active-validation 用于选择优化设置。
- D0-test 和预先冻结的 D1-test 不得进入拟合。
- 记录每个 active plan 的优化 seed、theta hash、covariance hash 和 simulator hash。

如果实机执行波形因 limiter/safety supervisor 被大幅改写，该 trial 仍可用于安全分析，但不能按原计划 FIM 归类；应以实际 executed command 重新计算信息量。

---

## 14. G8：精化参数 `theta_active`

推荐主结果使用 D0 + D1 的平衡数据拟合，避免主动轨迹过度主导；同时运行 D1-only 作为敏感性分析：

```text
Fit A: D0 → theta_1
Fit B: D0 + D1 → theta_active_main
Fit C: D1 initialized at theta_1 → theta_active_d1
```

比较：

- nominal vs `theta_1` vs `theta_active_main`。
- D0 held-out、D1 held-out 和未见组合命令。
- 0.5/1.0/1.5/2.0 秒 horizon。
- 冷机/热机、不同日期和重复装配 session。
- base position/orientation/velocity 与 joint position/velocity。

`theta_active` 通过条件：

- held-out 综合误差不劣于 `theta_1`，并在主要弱参数相关指标上继续改善。
- 参数置信区间缩小，或最小 FIM eigenvalue 增大。
- 参数不因单条 active trial 大幅漂移。
- 多 seed 和 bootstrap 结果一致。
- 物理参数有效且未贴边界。
- 改善跨普通命令与 active 命令成立，不只记住 D1。

输出：

```text
initial_parameters.yaml
theta_1.yaml
theta_1_cov.npy
active_commands.yaml
theta_active.yaml
theta_active_cov.npy
parameter_correlation.csv
cma_trace.csv
fim_report.json
heldout_report.md
```

---

## 15. G9：回灌 identified simulator 并重训

### 15.1 参数回灌

禁止直接覆盖 nominal URDF/MJCF。建立：

```text
X1_train/resources/robots/x1/identified/spi_active/<run_id>/
```

其中保存：

- 基于 nominal 生成的 identified URDF/MJCF。
- `theta_active.yaml`。
- nominal-to-identified diff。
- 生成脚本版本和输入 hash。
- residual DR distribution。

如果 inertia 聚合到 pelvis/base link，必须保证 URDF 和 MJCF 使用相同聚合语义。

### 15.2 重新设计 DR

identified simulator 不等于关闭全部 DR。建议：

- DR 中心移到 `theta_active`。
- 参数宽度来自 bootstrap/covariance 与跨 session 漂移。
- 已高置信辨识参数收窄。
- 未辨识的接触、传感器噪声、delay 和外扰继续保留合理 DR。
- 先修复总体方案列出的随机化实现 bug，再比较训练结果。

### 15.3 训练对照

至少训练：

| 组别 | 模型 |
|---|---|
| B0 | 修复契约后的 nominal simulator |
| B1 | 修复 bug 后的当前 broad DR |
| S1 | `theta_1` centered simulator |
| S2 | `theta_active` centered simulator |

各组共享 policy 架构、reward、命令 curriculum、训练步数和至少 3 个 seed。

**[当前训练入口]**：

```bash
cd X1_train
python humanoid/scripts/train.py \
  --task x1_dh_stand \
  --experiment_name x1_dh_stand \
  --run_name spi_active_<run_id> \
  --seed 1 \
  --headless
```

在实现 identified config 选择入口前，上述命令仍会使用默认 X1 配置，因此必须先增加显式 `--robot_model/--sysid_config` 或注册独立 task，禁止靠手工覆盖默认 XML 训练。

### 15.4 导出和部署

当前导出入口：

```bash
cd X1_train
python humanoid/scripts/export_onnx_dh.py \
  --task x1_dh_stand \
  --load_run <run_dir>
```

导出后执行：

1. ONNX input/output shape 检查。
2. Python/JIT/ONNX/C++ action parity。
3. observation history、phase、clip、action scale parity。
4. MuJoCo sim2sim。
5. 复制到独立版本化部署路径，不覆盖 `pi_collect`。
6. 修改独立 `rl_x1_spi_active.yaml` 指向新 policy。
7. 保存 policy 和配置 hash。

原 `pi_collect` 必须保留，作为回退和后续数据采集基线。

---

## 16. G10：最终验证顺序

严格按以下顺序：

1. Isaac held-out replay。
2. MuJoCo sim2sim。
3. 仿真中站立、前进、后退、横移、转向、组合命令。
4. 实机防坠条件下站立，不行走。
5. 实机 25% 安全包络前进并回零。
6. 实机低速后退、横移和转向。
7. 实机组合命令。
8. 扩大到原 `pi_collect` 已批准包络。

每一级都比较 B0、B1、S1、S2：

- command tracking。
- base position/orientation/linear/angular velocity。
- joint trajectory 和 foot contact timing。
- action/torque/current/power/temperature saturation。
- 跌倒、打滑、急停和人工救援次数。
- 与 simulator 的 open-loop prediction gap。

S2 必须在相同命令和安全条件下优于或不劣于原 `pi_collect`，才允许替换生产候选。

---

## 17. 建议的首批实际工作单

### SPI-00：冻结当前可走 policy

```text
目标：完成 policy/config/model hash 和实机安全包络记录
实机：需要，普通低速复测
输出：pi_collect_qualification.md、manifest.yaml
```

### SPI-01：修复命令和限位入口

```text
目标：关闭 constant_velocity；确认/实现 position、velocity、torque 安全限位
实机：只做站立和极低速验证
输出：joy_x1_spi.yaml、rl_x1_spi.yaml、安全测试结果
```

### SPI-02：补齐全量采集

```text
目标：增加 external base state、source timestamp、contact、power/temperature、安全状态
实机：站立 + 一条已验证低速前进
输出：统一 schema、data_quality.json
```

### SPI-03：实现 full-state replay

```text
目标：初始化完整 state，开环回放 recorded final u_t
实机：不需要
输出：nominal replay report
```

### SPI-04：synthetic parameter recovery

```text
目标：验证参数化、loss 和 CMA 能恢复隐藏参数
实机：不需要
输出：synthetic_recovery.md
```

### SPI-05：采集 D0

```text
目标：站立、前后、横移、转向、圆弧的多 session 普通数据
实机：需要，多轮低速
输出：D0 train/validation/test
```

### SPI-06：初始 SPI

```text
目标：得到 theta_1、covariance、correlation 和 held-out 报告
实机：不需要
输出：theta_1.yaml
```

### SPI-07：主动命令优化

```text
目标：在 theta_1 ensemble 中生成第一条安全高信息 vx/vy/yaw 曲线
实机：不需要
输出：spi_active_round_01.yaml、FIM/safety report
```

### SPI-08：主动实机采集

```text
目标：25% → 50% → 75% 分轮验证并采集 D1
实机：需要，每个幅值独立计划和结果
输出：D1、active execution report
```

### SPI-09：refined SPI

```text
目标：得到 theta_active 并证明 held-out 和不确定性优于 theta_1
实机：不需要
输出：theta_active.yaml、heldout_report.md
```

### SPI-10：重训与对照

```text
目标：完成 B0/B1/S1/S2 至少 3 seeds 和逐级实机对照
实机：最后阶段需要
输出：SPI-Active 最终报告和是否进入主线的结论
```

---

## 18. 最小可执行路径

基于当前“已经能初步走”的状态，最短路径不是立刻做 FIM，而是：

```text
第 1 周期
  冻结 rl_walk_leg.onnx
  → 关闭 constant_velocity
  → 确认部署限位
  → 标定外部 base pose
  → 把当前 10 s logger 扩展为正式 recorder

第 2 周期
  采集低速 D0
  → 实现 full-state clip replay
  → synthetic recovery
  → 得到 theta_1

第 3 周期
  实现有限差分 FIM
  → 优化 vx/vy/yaw 曲线
  → ensemble 安全筛选
  → 分轮采集 D1

第 4 周期
  得到 theta_active
  → 重训 S2
  → B0/B1/S1/S2 对照
```

其中 SPI-02、SPI-03、SPI-04 是正式参数结果可信性的硬前置；policy 能走只能解除“没有 `pi_collect`”这一项，不能替代全状态测量和 replay 验证。
