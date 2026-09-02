# X1 PACE 参数辨识详细运行手册

> 文档状态：Draft / 首轮执行前评审  
> 适用方案：PACE（Precise Adaptation through Continuous Evolution）  
> 适用对象：X1 下肢 12 个关节，重点包含左右并联踝  
> 上位方案：[X1 参数辨识与 Sim2Real 三方案设计](../../docs/archive/x1_sim2real_system_design.md)  
> 重要说明：当前仓库已经具备踝关节 step/sine 实机采集入口，但尚未实现 chirp、参数化 replay、CMA-ES 和参数回灌。本手册用“当前可运行”和“待实现”明确区分两类步骤。
## 1. 目标与最终产物

PACE 在固定基座、腿部悬空且无外部接触的条件下，通过位置目标激励辨识执行器/关节层参数。当前计划辨识：

```text
p = [I_a, d, tau_f, q_bias, T_d]
```

- `I_a`：每关节等效惯量或 armature。
- `d`：每关节粘性阻尼。
- `tau_f`：每关节库仑摩擦。
- `q_bias`：每关节位置零偏。
- `T_d`：全局或经验证后的分组命令延迟。

不把 PD gains、motor torque constant、位置/速度/力矩饱和、torque-speed curve 当作自由参数。这些量必须先测量或从实机配置获得，并在实机与 replay 中保持一致。

一次完整辨识应产生：

```text
deploy/sim2real/data/pace/round_NN/
├── plan.md
├── config/
│   ├── robot_manifest.yaml
│   ├── collector.yaml
│   ├── parameter_bounds.yaml
│   └── hashes.txt
├── raw/
│   ├── train/
│   └── held_out/
├── processed/
├── optimization/
│   ├── cma_trace.csv
│   ├── best_parameters.yaml
│   ├── covariance.csv
│   └── parameter_correlation.csv
├── validation/
│   ├── metrics.json
│   └── plots/
└── result.md
```

大体积 rosbag 和原始 CSV 默认不提交 Git，只提交 plan、配置、hash、辨识结果和报告。

## 2. 当前代码能力边界

### 2.1 当前可运行

- `AnkleIdentifierModule` 可直接发布 `/joint_cmd`。
- 支持左/右踝、pitch/roll 单轴选择。
- 支持 step 和单频 sine。
- 支持启动姿态 `current/zero/stand`。
- 可设置测试及保持 PD gains。
- 以 1 kHz 记录目标、虚拟关节状态、effort 和 IMU 到 CSV。
- DCU 驱动可额外发布 `/actuator_cmd` 和 `/actuator_states`。
- 运行入口为 `run_identifier.sh`，只加载 `DcuDriverModule` 与 `AnkleIdentifierModule`，不会同时加载 `ControlModule`。

### 2.2 当前不能宣称已经完成

- 没有 chirp 激励。
- 没有覆盖髋、膝和全部 12 个腿部关节的通用采集器。
- CSV 使用采集线程 elapsed time，并把异步到达的最新 joint/IMU 数据写在同一行；它不是严格的消息时间同步数据。
- 当前 CSV 没有 motor-side command/state、电流、电压、温度和饱和标志。
- 没有固定基座批量 replay 环境。
- 没有 CMA-ES 优化程序。
- 没有 `identified_parameters.yaml` 自动回灌入口。
- `auto_stop_after_test: true` 当前只结束测试序列并持续发布 hold，不会自动退出 `aimrt_main`；测试完成后仍需操作者停止进程。

因此，现有 step/sine 结果只用于验证硬件、控制方向、耦合响应和日志链路，不得直接作为完整 PACE 最终结果。

## 3. 角色和现场条件

每次实机运行至少需要：

- 操作者：执行启动、停止和配置切换，全程手持急停。
- 安全观察员：观察机器人、吊架、线缆、执行器声音和温度，不同时操作电脑。
- 数据负责人：确认配置 hash、topic 频率、日志路径、结果完整性。

现场必须具备：

- 正式 PACE 辨识时，机器人 base 必须相对试验架保持静止；普通肩部吊装只提供防跌落保护，不自动等价于 fixed-base。
- 双腿及足端在整个激励范围内不接触地面、吊带、支架或线缆。
- 固定架、吊架和约束件不会在激励中摆动、松弛或产生明显周期形变。
- 关节运动范围内无人员和物体。
- 硬件急停可用，软件停止不能替代硬件急停。
- 网络、供电和散热稳定。

任一条件不满足时不得开始主动激励。

### 3.1 fixed-base 与肩吊的边界

必须区分两个目的：

- **吊装安全条件**：机器人即使失控也不会跌落。
- **fixed-base 辨识条件**：激励过程中 base 的 3 个平移和 3 个转动自由度均不参与被辨识动力学。

仅用一个吊点或两根柔性肩带将机器人吊起时，机器人仍可能发生钟摆、升沉、roll、pitch 和 yaw；吊带的刚度、阻尼及吊架模态也会进入测量响应。该状态不视为正式 PACE 的 fixed-base，不能直接用于最终参数拟合。

固定方式按优先级选择：

1. **推荐：刚性六自由度约束。** 将骨盆/base-link 或躯干承力骨架通过刚性支架连接到地面试验架，在至少 3 个不共线的结构点形成约束；连接点必须由机械设计或整机负责人确认，不得夹持外壳、装饰件或未经确认的肩部覆盖件。
2. **次选：肩部承重加刚性防摆约束。** 肩部吊点只承担重量，另用前后和左右布置的刚性拉杆约束平移、roll、pitch 和 yaw。独立防坠绳保持微松，仅作失效保护，不参与正常受力。
3. **仅限链路验证：普通肩吊。** 单吊点、两根松弛肩带或会明显摆动的吊架，只可用于低幅 step/sine 的方向、急停、日志和温升检查；所得数据标记为 `suspension_only`，不得进入正式 PACE CMA-ES 数据集。

无论采用哪种结构，双腿、足端、线缆和防坠绳在全部激励范围内均不得接触或牵拉运动链。若仿真 replay 使用刚性 fixed-base，实机也必须通过下述静止性检查；“肉眼看起来没动”不能作为验收依据。

### 3.2 base 静止性 Gate

在进入正式 chirp 前，应在躯干/base 上布置 IMU，并使用光学标记、位移传感器或固定相机标记测量平移；仅靠 IMU 不足以可靠判断低频平移。先运行低幅预扫，再检查：

- 全激励频带内没有吊装或固定架共振峰。
- base 运动与关节激励没有明显相干响应，且多次重复结果一致。
- base 转角幅值应显著小于被激励关节幅值；建议以被激励关节幅值的 5% 作为初始工程筛选线，并结合传感器分辨率和目标拟合误差收紧。
- 更换激励幅值或重复装夹后，辨识参数不得出现超出置信区间的系统性漂移。

上述 5% 是项目初始筛选建议，不是 PACE 论文规定的通用阈值。若无法通过静止性 Gate，应加强刚性约束；不能通过调参把吊装摆动吸收到电机惯量、阻尼或时延参数中。

## 4. 总体 Gate

PACE 按以下 Gate 顺序执行。不得跳过前一个 Gate 直接进入下一阶段。

| Gate | 内容 | 通过条件 |
|---|---|---|
| G0 | 软件和模型契约 | observation/action/torque、joint order、offset、limits、时间语义对齐 |
| G1 | 被动检查 | 固定可靠，关节方向、零位、状态频率和急停正常 |
| G2 | 当前 step/sine MVP | 小幅响应稳定，日志完整，无异常耦合和饱和 |
| G3 | chirp 仿真干跑 | 完整命令链和停止条件在仿真中通过 |
| G4 | 正式实机 chirp | train/held-out 数据质量通过 |
| G5 | CMA-ES 拟合 | 收敛、参数未长期贴边界，多次 seed 一致 |
| G6 | held-out 验证 | RMSE、delay 和相对改善达到验收阈值 |
| G7 | 参数回灌与重训 | identified 模型进入训练，B0/B1/P1 对照完成 |
| G8 | sim2sim/实机验证 | 悬空、落地、站立、低速运动逐级通过 |

## 5. G0：软件契约准备

### 5.1 冻结版本

记录以下信息到本轮 `hashes.txt`：

```bash
git -C X1_train rev-parse HEAD
git -C X1_infer rev-parse HEAD 2>/dev/null || echo "X1_infer revision unavailable; use file hashes"
shasum X1_train/resources/robots/x1/urdf/x1.urdf
shasum X1_train/resources/robots/x1/mjcf/xyber_x1_flat.xml
shasum X1_infer/module/sim_module/model/mjcf/xyber_x1_flat.xml
shasum X1_infer/module/dcu_driver_module/cfg/dcu_x1.yaml
shasum X1_infer/module/dcu_driver_module/cfg/ankle_trans_x1.yaml
shasum X1_infer/module/control_module/cfg/rl_x1.yaml
```

当前工作区中的 `X1_infer` 可能是缺少父仓库元数据的 submodule 快照；这种情况下以发布包版本、固件版本和上述逐文件 hash 作为可复现依据，不能把 revision unavailable 当成已冻结版本。

同时人工记录：

- 机器人序列号。
- 左右踝/执行器序列号。
- DCU 固件版本。
- 上电时间和开始测试时温度。
- 供电电压。
- 吊装方式和现场照片编号。

### 5.2 生成并核对 manifest

`robot_manifest.yaml` 至少应包含：

- policy 12 关节固定顺序。
- URDF、Isaac Gym、训练 MuJoCo、部署 MuJoCo、ROS joint name 映射。
- 每个关节方向、零位、装配 offset。
- 虚拟踝 pitch/roll 与左右电机映射。
- 控制周期、策略周期和时间戳定义。
- 位置、速度、虚拟关节力矩、电机力矩、电流和温度限制。
- 实机使用的 Kp/Kd、LPF 和 saturation。

### 5.3 关闭会污染辨识的差异

正式采集前必须完成：

1. 训练、replay、部署使用相同 joint order/sign/offset。
2. 明确训练与部署的 PD gains；当前 knee 和 ankle gains 不完全一致。
3. 明确踝力矩限制；当前训练 URDF、MuJoCo 和旧 sim2sim 数值不一致。
4. replay 复现实机 action delay、PD、LPF、saturation 和并联踝传动。
5. 修复 torque multiplier 每物理步重采样。
6. 修复 DOF friction/damping reset 累计相乘。
7. 固定基座辨识环境关闭 broad domain randomization。
8. 更新总方案中已经过时的“51 维”描述；当前腿部策略是单帧 47 维、66 帧历史。

G0 验收：同一 recorded trace 下，训练端和部署端观测最大误差 `<1e-5`；相同 state/action 下，进入被辨识对象前的最终 command 一致。

## 6. 每轮运行计划

每次上机前复制下面模板到：

```text
deploy/sim2real/data/pace/round_NN/plan.md
```

模板：

```markdown
# PACE Round NN Plan

- 唯一目标：
- 机器人/执行器序列号：
- Git commit：
- 固件版本：
- 固定/吊装方式：
- 测试关节或关节组：
- startup pose：
- Kp/Kd：
- 激励类型：step / sine / chirp / random-step
- 幅值：
- 起止频率：
- 单次时长：
- 重复次数：
- train 或 held-out：
- 日志路径：
- 固定参数：
- 本轮禁止修改项：
- Pass 条件：
- Fail/立即停止条件：
- 操作者：
- 安全观察员：
- 数据负责人：
```

一轮只能有一个唯一目标。改变关节、增益、幅值、频带或固件后必须开始新一轮，不能覆盖旧数据。

## 7. G1：被动检查

### 7.1 软件环境

以下命令在部署产物的 `bin` 目录执行，而不是源码目录。该目录应包含：

```text
aimrt_main
run_identifier.sh
ros2_source.sh
cfg/x1_cfg_identifier.yaml
cfg/dcu_driver_module/dcu_x1.yaml
cfg/ankle_identifier_module/ankle_identifier.yaml
```

加载 ROS 2 环境：

```bash
source ./ros2_source.sh
```

确认网卡名和 EtherCAT 配置与实机一致：

```bash
rg -n "ifname|bind_cpu|cycle_time_ns|enable_actuator" cfg/dcu_driver_module/dcu_x1.yaml
```

### 7.2 上电前检查

- 急停按下或执行器未使能。
- 双腿完全离地。
- 手动缓慢移动各测试关节，确认无机械干涉。
- 线缆在全激励范围内有余量。
- 确认不运行 `run.sh`、`run_with_recording.sh` 或其他 `/joint_cmd` publisher。

### 7.3 只启动状态链检查

首次现场测试建议先用 `enable_actuator: false` 的轮次配置启动驱动，仅验证状态链。不要直接修改并覆盖基线配置，应保存 round-specific 副本并记录 hash。

启动后检查：

```bash
ros2 topic list
ros2 topic hz /joint_states
ros2 topic hz /imu/data
ros2 topic echo --once /joint_states
ros2 topic echo --once /imu/data
```

通过条件：

- `/joint_states` 和 `/imu/data` 接近 1 kHz，且无长时间中断。
- joint name 完整且顺序可映射到 manifest。
- 静止时关节速度接近 0。
- IMU 四元数有效，静止角速度低于计划阈值。
- 时间戳单调递增。

## 8. G2：当前代码可执行的 step/sine MVP

这一步只验证数据链和局部动力学响应，不进行正式 PACE 参数拟合。

### 8.1 配置安全基线

编辑部署产物中的 round-specific `cfg/ankle_identifier_module/ankle_identifier.yaml`。第一轮建议：

```yaml
mode: step
startup_pose_mode: current
test_side: left
test_axis: roll

publish_rate_hz: 1000.0
pre_hold_sec: 3.0
active_sec: 0.5
post_hold_sec: 3.0
repeat_count: 1

step_amplitude_rad: 0.003
test_kp: 30.0
test_kd: 0.5
startup_stable_sec: 2.0
startup_joint_vel_threshold: 0.03
startup_gyro_threshold: 0.1

use_imu: true
auto_stop_after_test: true
csv_path: ./log/pace/round_00/left_roll_step_000.csv
```

注意：

- `startup_pose_mode: zero` 会主动把所有已配置关节移动到零位，不作为首轮默认选择。
- `current` 模式仍会保持启动时位置，必须确认保持 gains 对吊装状态安全。
- 幅值必须从经审批的小幅值开始，禁止直接复制较大幅值。
- `auto_stop_after_test` 不会退出主程序，测试结束后仍持续 hold。

### 8.2 启动 rosbag

建议在独立终端记录所有 topic：

```bash
source ./ros2_source.sh
mkdir -p ./log/pace/round_00
ros2 bag record -a -o ./log/pace/round_00/rosbag_step_000
```

重点确认 rosbag 中包含：

```text
/joint_cmd
/joint_states
/actuator_cmd
/actuator_states
/imu/data
```

### 8.3 启动辨识程序

在部署 `bin` 目录执行：

```bash
source ./ros2_source.sh
./run_identifier.sh
```

程序启动后立即在另一终端检查 publisher：

```bash
ros2 topic info /joint_cmd -v
```

必须只有预期的 identifier publisher。发现第二个控制 publisher 时立即停止。

### 8.4 现场观察顺序

1. 启动后先观察 startup settle。
2. 确认机器人没有向意外姿态运动。
3. 确认 primary/coupled 关节方向与计划一致。
4. 进入 pre-hold 后确认无振荡。
5. 激励期间观察 primary 响应、coupled 响应、两个踝电机和吊架。
6. 进入 post-hold 后确认关节回到 baseline 附近。
7. 日志出现 `Test completed` 后，人工 `Ctrl-C` 结束 `aimrt_main`。
8. 结束 rosbag，记录是否触发过急停或异常。

### 8.5 立即停止条件

发生任一情况立即硬件急停：

- 方向与计划相反。
- 位置、速度、电流、力矩或温度达到批准限制。
- primary 或 coupled 轴持续振荡。
- 并联踝两个电机出现明显不对称冲击或异常声响。
- 吊架摆动、base 运动或足端接触外物。
- ROS/驱动失联、时间戳倒退或状态停更。
- CSV/rosbag 写盘失败。

急停后不要修改多个参数重试。先保存本轮配置、日志和事件时间，再创建下一轮计划。

### 8.6 MVP 数据检查

```bash
wc -l ./log/pace/round_00/left_roll_step_000.csv
head -2 ./log/pace/round_00/left_roll_step_000.csv
tail -2 ./log/pace/round_00/left_roll_step_000.csv
ros2 bag info ./log/pace/round_00/rosbag_step_000
```

检查项：

- CSV 行数与计划的 `publish_rate × 总时长` 同量级。
- 时间列单调递增。
- pre/active/post 三个 phase 均存在。
- target 在 active 段按计划变化。
- actual 没有 NaN、跳变或长时间不更新。
- coupled 轴响应被记录。
- rosbag 包含 virtual joint 与 motor-side command/state。
- 没有命令或状态饱和。

然后以相同幅值依次完成：

```text
left roll step
left pitch step
right roll step
right pitch step
left roll sine
left pitch sine
right roll sine
right pitch sine
```

每改变 side、axis 或 mode 都必须使用新的输出文件，禁止覆盖。

## 9. G3：正式 PACE 前必须实现的采集能力

以下能力实现并通过代码审查、单元测试和仿真干跑后，才能执行正式 chirp。

### 9.1 通用激励器

采集模块应支持：

- `mode: chirp`。
- linear/log chirp。
- 每关节独立 amplitude、start/end frequency、duration。
- 单关节、关节组和全腿 schedule。
- 训练轨迹与 held-out 轨迹使用不同 schedule。
- 命令开始前和结束后的平滑 envelope，避免瞬时跳变。
- 每轮唯一 run ID 和配置 hash。

推荐配置接口示例，实际字段以实现后的 schema 为准：

```yaml
mode: chirp
coordinate: joint       # joint 或 motor
fixed_base_required: true

schedule:
  - joints: [left_knee_pitch_joint]
    amplitude_rad: 0.005
    start_frequency_hz: 0.1
    end_frequency_hz: 2.0
    duration_sec: 30.0
    sweep: logarithmic
    repeat_count: 3
```

示例数值只是首轮评审起点，不是自动授权的实机参数。

### 9.2 同步日志

每一帧至少记录：

```text
run_id
command_sequence
steady_clock_timestamp_ns
source_message_timestamp_ns
joint_name / motor_name
q_target_raw / q_target_after_delay / q_target_after_filter
dq_target
Kp / Kd
tau_ff / tau_pd_raw / tau_after_filter / tau_after_saturation
q / dq / reported_effort
motor_q / motor_dq / motor_effort / motor_current
voltage / temperature
position_saturated / velocity_saturated / torque_saturated
imu quaternion / gyro / acceleration
safety_state / stop_reason
```

日志必须保留 command、joint state、actuator state 各自的原始时间戳，离线对齐时不能假设回调到达时间就是采样时间。

### 9.3 仿真干跑

正式上机前在固定基座 MuJoCo/Isaac 环境完成：

- 全 schedule 播放。
- 任意阶段的软件停止。
- 状态 topic 中断后的停止。
- 超位置/速度/力矩后的停止。
- 输出路径不存在或磁盘写失败后的停止。
- 并联踝 pitch/roll 到 motor-side 命令检查。

## 10. G4：正式实机 chirp 采集

### 10.1 建议采集顺序

按风险由低到高：

1. 单个髋/膝关节小幅 chirp，其他关节保持。
2. 左右对称髋/膝关节分别采集。
3. 单侧踝 motor-coordinate 小幅 chirp。
4. 另一侧踝 motor-coordinate 小幅 chirp。
5. 踝 pitch/roll 联合激励，用于验证交叉耦合。
6. 在前述结果安全后，再考虑多关节同时激励。

### 10.2 训练集

每个关节/关节组至少覆盖：

- 低、中两个安全幅值。
- 经机械和控制带宽评审的频率范围。
- 至少 3 次重复。
- 左右侧。
- 至少两个温度区间或两个独立 session。

初始频率可从约 0.1 Hz 开始；最高频率必须根据结构带宽和现场安全评审确定，不能默认直接使用论文上限。

### 10.3 Held-out 集

Held-out 数据不能从训练轨迹相邻帧随机切分，应使用完整、独立的运行：

- 不同起止频率的 chirp。
- 未参与拟合的 amplitude。
- random-step。
- 一组不同但已知的 Kp/Kd。
- 独立上电或不同时间 session。
- 踝 pitch/roll coupled motion。

### 10.4 单轮数据质量 Gate

每轮结束后立即检查，满足后才能开始下一轮：

- 实际状态采样率不低于 400 Hz，目标为 1 kHz。
- 时间戳单调，无无法解释的长间隔。
- 丢帧率和最大连续丢帧满足本轮计划。
- 激励频带内有足够响应，不全被高增益压制。
- 状态和 command 没有 NaN/Inf。
- 饱和样本占比接近 0；存在持续饱和则本轮作废。
- base 保持固定、足端无接触。
- 电机温度和供电电压在批准范围内。
- 左右电机和虚拟踝数据可互相映射。

## 11. 数据预处理

正式拟合前执行：

1. 按 source timestamp 对齐 command、joint state、actuator state 和 IMU。
2. 保留原始数据，处理结果写入 `processed/`，不得覆盖 raw。
3. 标记但不静默删除丢帧、饱和、急停和通信异常区间。
4. 将关节和 motor 数据转换到 manifest 统一方向和单位。
5. 只使用 base 固定、无接触、无饱和的有效区间。
6. 计算实际采样周期分布和 command-to-state 粗延迟。
7. 生成每个 session 的幅值、频谱、温度和有效时长摘要。
8. 按完整 run 划分 train 和 held-out。

预处理产物应包含可被 replay 直接读取的统一序列格式，以及一份数据质量报告。

## 12. G5：固定基座 Replay 与 CMA-ES

### 12.1 Replay 环境要求

Replay 环境必须：

- 固定 base。
- 移除足端和吊带接触。
- 使用与实机相同初始姿态。
- 回放最终实际施加的 command，不重新生成 chirp。
- 复现实机 Kp/Kd、delay、LPF、saturation。
- 对并联踝复现 virtual-to-motor transmission。
- 支持数千个并行环境，每个环境一组候选参数。
- 能分别计算每关节、每关节组和全轨迹 loss。

### 12.2 参数边界

`parameter_bounds.yaml` 的每个边界必须有来源：

- CAD/URDF/MJCF 标称值。
- 执行器或减速器规格。
- 被动测试估计。
- 当前 step/sine MVP 响应。
- 制造和装配公差。

禁止为得到更低 loss 无限制扩大搜索区间。PD gain、torque constant 和全局 torque scale 不与动力学参数同时优化。

### 12.3 Loss

基础 loss：

```text
L = weighted_mean((q_sim - q_real)^2) + regularization
```

建议额外报告但不要未经评审随意混入主 loss：

- 每关节 q RMSE。
- dq RMSE。
- phase/frequency band error。
- 踝 coupled-axis error。
- 参数先验正则项。
- 达到参数边界的惩罚。

### 12.4 待实现 CLI 契约

以下命令是需要实现的目标接口，当前仓库中尚不存在，不能直接执行：

```bash
# [待实现] 数据预处理
python -m humanoid.scripts.pace_prepare \
  --manifest deploy/sim2real/data/pace/round_NN/config/robot_manifest.yaml \
  --input deploy/sim2real/data/pace/round_NN/raw \
  --output deploy/sim2real/data/pace/round_NN/processed

# [待实现] nominal replay
python -m humanoid.scripts.pace_replay \
  --dataset deploy/sim2real/data/pace/round_NN/processed/train \
  --parameters nominal \
  --output deploy/sim2real/data/pace/round_NN/validation/nominal

# [待实现] CMA-ES
python -m humanoid.scripts.pace_identify \
  --dataset deploy/sim2real/data/pace/round_NN/processed/train \
  --bounds deploy/sim2real/data/pace/round_NN/config/parameter_bounds.yaml \
  --num-envs 4096 \
  --seed 1 \
  --output deploy/sim2real/data/pace/round_NN/optimization
```

### 12.5 优化过程检查

- 先运行 nominal replay，确认初始状态、command 和 loss 正确。
- 用已知仿真参数生成 synthetic data，确认优化器可以找回参数。
- 至少使用 3 个 CMA seed。
- 观察 best/mean loss，而不是只看单次最优个体。
- 检查参数是否长期贴上下界。
- 检查协方差和参数相关矩阵，识别不可辨识组合。
- 训练 loss 降低但 held-out 变差时停止，不接受该结果。

## 13. G6：Held-out 验证

分别运行：

```text
nominal simulator
identified simulator
```

在完全相同的 held-out 输入上比较：

- 每关节和总体 q RMSE。
- 踝 pitch/roll coupled-motion RMSE。
- 不同幅值、频率、Kp/Kd 和 session 的泛化。
- 20–60 秒 rollout 是否漂移或发散。
- delay 估计是否稳定。
- 参数是否落在合理物理范围。

PACE 通过标准：

- held-out 髋膝 `q RMSE < 0.02 rad`。
- held-out 踝 `q RMSE < 0.015 rad`。
- delay 误差 `< 2 ms` 或 `< 实测值的 20%`。
- 相对 nominal held-out MSE 改善至少 40%。
- 跨幅值、频率、已知 PD gains 和 session 保持改善。
- 参数置信区间没有覆盖大部分搜索范围。

未通过时，按以下顺序排查：

1. 时间同步和 delay 定义。
2. joint order/sign/offset。
3. 实际 PD、LPF 和 saturation。
4. 并联踝 motor-side 映射。
5. 未建模接触或 base 运动。
6. 参数边界和参数相关性。
7. 白盒模型表达能力不足。

不得通过扩大所有参数边界来掩盖上述问题。

## 14. G7：参数回灌和策略重训

### 14.1 回灌映射

将通过 held-out 的参数写入统一配置生成链：

| 辨识量 | 训练侧 | 部署/验证侧 |
|---|---|---|
| `I_a` | Isaac DOF armature / MJCF armature | MuJoCo joint armature |
| `d` | Isaac DOF damping | MJCF damping |
| `tau_f` | frictionloss 或一致的摩擦模型 | 同一摩擦模型 |
| `q_bias` | motor offset/observation offset | 经验证后的 `joint_offset` 语义 |
| `T_d` | action delay buffer | replay/deployment测量值，不重复增加延迟 |

并联踝若在 motor coordinates 辨识，必须通过同一个 transmission 映射进入训练模型；不能直接把两个电机参数无条件当成 pitch/roll 两个独立参数。

### 14.2 Domain Randomization

建立三组独立配置：

- B0：nominal，无 broad DR。
- B1：修复实现错误后的原 broad DR。
- P1：identified mean + posterior narrow DR。

P1 的随机化范围来自 CMA 协方差、不同 session 变化和装配差异，不沿用原来的任意宽范围。

### 14.3 训练与导出

训练命令：

```bash
cd X1_train
python humanoid/scripts/train.py \
  --task=x1_dh_stand \
  --run_name=pace_p1_seed1 \
  --headless
```

每组至少 3 个 seed，并保持 policy 架构、reward、训练预算一致。

按现有工程流程导出 JIT 和 ONNX：

```bash
python humanoid/scripts/export_policy_dh.py \
  --task=x1_dh_stand \
  --load_run=<run_name>

python humanoid/scripts/export_onnx_dh.py \
  --task=x1_dh_stand \
  --load_run=<exported_policy_run>
```

导出后验证 ONNX：

- 输入为 `47 × 66 = 3102`。
- 输出为 12。
- 同一输入下 JIT 与 ONNX 输出误差满足约定阈值。
- action scale、clip 和 joint order 与部署 manifest 一致。

## 15. G8：Sim2sim 和实机回归

### 15.1 Sim2sim 前置修复

使用现有 `sim2sim.py` 前必须先完成：

- 修复 body name 条件恒真的表达式。
- 使用与训练/部署相同 phase generator。
- 删除硬编码 `500 Nm` 力矩限制。
- 使用 canonical MJCF。
- 复现 delay、LPF、PD 和并联踝控制路径。
- 加入输入/输出 shape、NaN、limit 和 model hash 检查。

### 15.2 分级验证

严格按顺序执行：

1. 固定基座、无接触 replay。
2. 悬空闭环策略运行。
3. 双脚落地，仅保持站立。
4. 原地站立扰动恢复。
5. 极低速前进。
6. 低速后退、横移、转向。
7. 组合命令。

每一级单独形成 plan/result，上一等级未通过不得进入下一等级。

### 15.3 最终报告

`result.md` 至少包含：

- 使用的数据 run ID。
- 所有配置和代码 hash。
- nominal/identified 参数表。
- CMA 收敛曲线和参数协方差。
- train/held-out 指标。
- B0/B1/P1 训练结果和 seed 分布。
- Isaac-MuJoCo gap。
- 实机站立/速度/姿态/接触/饱和/功率/温度指标。
- 所有安全事件和丢帧情况。
- 结论：Pass、Fail 或需要下一轮补充数据。

## 16. 常见问题和处理

### 16.1 程序提示完成但机器人仍保持

这是当前实现行为。`auto_stop_after_test` 只停止测试序列，主循环继续发布 baseline hold。确认已经进入 post-hold 并稳定后，由操作者 `Ctrl-C` 停止；异常时直接急停。

### 16.2 CSV 有 1 kHz 行，但 joint state 不是 1 kHz 新数据

当前 CSV 按控制循环记录“最新一次回调值”，可能重复写同一状态。正式 PACE 应以 source timestamp 去重并对齐；现有 CSV 只能做 MVP 检查。

### 16.3 踝 pitch 激励导致 roll 明显运动

先检查 transmission direction、零位、motor command 和 Jacobian 映射。若映射正确，说明独立 pitch/roll 白盒不足，应在 motor coordinates 或 2×2 耦合模型中辨识。

### 16.4 训练 loss 很低但 held-out 很差

优先检查 session 泄漏、时间同步、参数相关性和搜索边界。不能随机拆相邻帧作为验证集。

### 16.5 delay 总是贴搜索边界

检查 command、driver、joint state 使用的时间戳定义，以及 LPF 相位延迟是否被重复算入 `T_d`。

### 16.6 参数合理但策略仍无法实机工作

PACE 只辨识固定基座执行器层参数，不覆盖 base mass/COM、地面摩擦、接触和整机耦合。先确认策略训练控制链一致，再决定是否启动 SPI；存在稳定执行器非线性残差时再评估 UAN。

## 17. 首轮推荐工作拆分

```text
PACE-00：完成 manifest、时间戳和控制链 G0
PACE-01：现有左踝 roll 小幅 step，验证方向和日志
PACE-02：四个踝轴小幅 step/sine，建立 MVP 基线
PACE-03：实现通用 chirp、同步日志和安全状态机
PACE-04：建立固定基座 replay，synthetic parameter recovery
PACE-05：一个髋/膝关节 chirp + CMA-ES 闭环
PACE-06：左右髋膝扩展
PACE-07：并联踝 motor-coordinate/2×2 模型
PACE-08：全 12 关节 held-out 验证
PACE-09：参数回灌、B0/B1/P1 重训和 sim2sim
PACE-10：分阶段实机验证
```

第一阶段完成标志不是“采到一份 CSV”，而是 PACE-05 能在独立 held-out 轨迹上稳定优于 nominal，并且整个过程可以由 plan、配置和 hash 完整复现。
