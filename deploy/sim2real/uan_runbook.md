# X1 UAN 详细运行步骤

> 状态：Runbook / 分阶段待实施  
> 日期：2026-08-11  
> 依据：`x1_sim2real_system_design.md`、当前 `X1_train` 和 `X1_infer` 代码  
> 参考：[Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation](https://arxiv.org/html/2502.10894v1)

## 0. 结论与当前起点

X1 当前具备 UAN MVP 的实机采集起点：

- 已有独立 `AnkleIdentifierModule`，不会与 locomotion policy 同时运行。
- 已支持单踝 pitch/roll 的 step 和单频 sine。
- 控制命令和 joint state 目标频率为 1000 Hz。
- `DcuDriverModule` 可发布 `/joint_cmd`、`/joint_states`、`/actuator_cmd`、`/actuator_states` 和 `/imu/data`。
- 当前 `actuator_debug: true`，可获得并联踝两个 motor-side actuator 的 position/velocity/effort 和 command。
- 左右踝已有 virtual joint 与两个 actuator 间的非线性 transmission 映射。

当前配置快照：

```text
X1_infer/module/ankle_identifier_module/cfg/ankle_identifier.yaml
SHA1: 578f7d91109b32f9ec0d895a0e4a13a7890fae41

X1_infer/install/linux/bin/cfg/x1_cfg_identifier.yaml
SHA1: 605e8028f4e0d4b95d86182a40e0a3472aaf39c3

X1_infer/module/dcu_driver_module/cfg/dcu_x1.yaml
SHA1: 9ad1a42844f5544303eeba571afeb01a6347ff08

X1_infer/module/dcu_driver_module/cfg/ankle_trans_x1.yaml
SHA1: 584a2e6dfaa4a6a861fc98e7cbaaab3a2a29b69b
```

但当前仓库还没有完整 UAN。正式训练前必须补齐：

1. 连续 square wave、多幅值/频率 sine 和 seeded Gaussian 激励器。
2. joint-side 与 motor-side command/state 的统一 source timestamp；当前 actuator topic 的 header 未赋时间戳。
3. 电流、电压、温度、fault 和 saturation 的采集与在线停止条件。
4. 从连续 run 生成 transition/sequence 数据集的预处理工具。
5. 能按真实状态初始化、连续回放实机 command 的 UAN 训练环境。
6. 100 ms error history、二维踝残差网络和 UAN 专用 PPO。
7. 在 simulator torque path 中注入 `delta_tau` 的接口。
8. 20 秒与 5 分钟长 rollout 验证工具。
9. 冻结 UAN 后的 locomotion 训练入口和 UAN/DR 去重配置。

本文使用：

- **[当前可运行]**：仓库已有对应入口。
- **[需扩展]**：在现有模块上补字段或模式。
- **[待实现]**：当前没有对应程序。

UAN 的完整闭环是：

```text
固定基座采集 actuator excitation
  → 形成连续真实 sequence 和 transition
  → 仿真按真实初始状态和 command rollout
  → UAN 根据 tracking-error history 输出 delta_tau
  → PPO 训练 UAN 使 q_sim 接近 q_real
  → 未见波形与长 rollout 验证
  → 冻结 UAN，插入 locomotion 训练仿真
  → 重训 locomotion policy
  → 实机只部署新 locomotion policy，不运行 UAN
```

---

## 1. UAN 的辨识边界

### 1.1 UAN 学到什么

UAN 不输出 mass、friction、armature 或 delay 等白盒参数。它学习 simulator-only corrective torque：

```text
delta_tau_t = pi_UAN(error_history_t)
tau_sim_t = tau_baseline_t + delta_tau_t
```

训练目标不是监督拟合真实 torque，而是让加入残差后的模拟 transition 与真实 joint encoder transition 一致。因此真实 `JointState.effort` 或 motor current 可以用于诊断，但不是必须的 torque label。

UAN 适合补偿：

- 非线性摩擦和粘滞效应。
- 死区、滞回和负载相关响应。
- 减速器和传动的未建模动态。
- 简单 armature/friction/delay 白盒模型无法稳定拟合的残差。

UAN 不应补偿：

- joint order、sign、offset 错误。
- 错误的 Kp/Kd、LPF、delay 或 saturation 语义。
- 错误的刚体质量、COM、inertia。
- 地面接触、吊带摆动或 base 未固定。
- 时间戳错位、数据丢帧和测量跳变。

### 1.2 UAN 输入的 X1 契约

论文将 actor 输入限制为 20 个 5 ms 时刻的 position/velocity error，即 100 ms 历史，以减少对全身状态和数据集轨迹的过拟合。

X1 必须把 error 定义成在任意 locomotion policy 训练中都可计算的 actuator tracking error，而不是 `real - sim` 真值：

```text
e_q(t)  = q_des(t)  - q_sim(t)
e_dq(t) = dq_des(t) - dq_sim(t)
```

真实数据中对应计算：

```text
e_q_real(t)  = q_des_real(t)  - q_real(t)
e_dq_real(t) = dq_des_real(t) - dq_real(t)
```

训练 rollout 时 actor 只能看到 simulator 当下的 tracking-error history；`q_real(t+1)` 只用于 reward，不能泄漏到 actor observation。

如果 X1 的最终低层接口为 torque command，没有 `q_des/dq_des`，则必须在 UAN schema 中明确使用另一套可部署输入，例如 torque command、q、dq 的历史；这属于独立网络版本，不能与 tracking-error 版本混用。

### 1.3 UAN 只运行在仿真

UAN 在以下位置运行：

- UAN 自身训练环境。
- 加入 UAN 的 locomotion policy 训练环境。
- UAN replay/validation。

UAN 不放入 `X1_infer` 的实机 torque loop。最终上机只部署在 UAN-corrected simulator 中重训得到的 locomotion ONNX policy。

### 1.4 第一目标只做并联踝

当前 X1 最有价值的 UAN MVP 是左右并联踝，不是全部 12 个腿关节：

```text
[delta_tau_pitch, delta_tau_roll]
  = pi_ankle([e_q_pitch, e_dq_pitch, e_q_roll, e_dq_roll] past 20 steps)
```

原因：

- pitch/roll 由两个 actuator 非线性耦合。
- 当前 simulator 把踝建成 serial virtual joints，没有显式复现实机平行机构。
- 髋膝若可由 PACE/白盒模型充分解释，不应额外增加黑盒网络。

MVP 采用 joint-side 二维 UAN。只有 simulator 也实现 motor-side 平行机构、Jacobian 和 actuator limit 后，才升级为 motor-side UAN。

左右踝可先共享网络，但必须把左右 coordinate/sign 映射到 canonical ankle frame。若 held-out 显示稳定左右差异，再拆成 left/right checkpoint。

---

## 2. 输出和目录

计划目录：

```text
deploy/sim2real/
├── uan_runbook.md
├── configs/
│   ├── x1_robot_manifest.yaml
│   ├── uan_collection.yaml
│   ├── uan_dataset.yaml
│   ├── uan_train.yaml
│   ├── uan_limits.yaml
│   └── uan_locomotion.yaml
├── plans/
│   └── uan_round_NN_<name>.md
├── results/
│   └── uan_round_NN_<name>.md
└── artifacts/uan/<run_id>/
    ├── manifest.yaml
    ├── hashes.txt
    ├── raw/
    ├── processed/
    ├── sequences/
    ├── checkpoints/
    ├── validation/
    └── locomotion/
```

`run_id`：

```text
uan_YYYYMMDD_robotSerial_scope_roundNN
```

最终输出至少包含：

```text
dataset_manifest.yaml
normalization.yaml
uan_checkpoint.pt
uan_limits.yaml
train_curve.csv
one_step_report.md
rollout_20s_report.md
rollout_5min_report.md
heldout_waveform_report.md
locomotion_ablation_report.md
```

每个 checkpoint 必须绑定：dataset hash、network schema、normalization、residual limits、simulator hash、training seed 和验证报告。

---

## 3. 现场条件与安全

### 3.1 固定与吊装

正式 ankle UAN actuator 数据应使用与 PACE 一致的 fixed-base 条件：

- base 的 3 个平移和 3 个转动自由度相对试验架保持静止。
- 双腿和足端在整个激励范围内无接触。
- 普通肩吊只防跌落、不等价于 fixed-base。
- 吊带、线缆和防坠绳不牵引腿部。

如果只做普通肩吊的低幅 step/sine，数据只能标记为 `suspension_only`，用于链路验证，不能作为正式 UAN 动态数据。

若后续需要负载覆盖，应使用可测量、可复现的静态负载夹具建立独立 session，不能依靠随机吊带受力形成“负载变化”。

### 3.2 角色

- 操作者：启动、停止和配置切换，全程手持硬件急停。
- 安全观察员：观察固定架、踝电机、腿部、线缆、温升和异常声响。
- 数据负责人：核对 hash、topic、采样率、数据文件和 trial 结果。

### 3.3 立即停止条件

- primary/coupled 方向错误。
- base 或固定架出现周期运动。
- 足端、腿部、吊带或线缆发生接触。
- joint/motor position、velocity、effort/current/temperature 超限。
- torque-speed 或总功率限制触发。
- pitch/roll 或两个 actuator 出现持续振荡、不对称冲击或异常声响。
- command、joint state、actuator state 或 IMU heartbeat 超时。
- 时间戳倒退、连续丢帧、CSV/rosbag 写盘失败。
- operator 急停。

所有阈值必须来自整机、执行器和固定架安全规范；本文不提供未经硬件审批的通用幅值。

---

## 4. 总体 Gate

| Gate | 内容 | 通过条件 |
|---|---|---|
| G0 | 软件/模型契约 | joint、motor、transmission、PD、delay、limit 和时间语义一致 |
| G1 | 当前 step/sine MVP | 小幅响应、耦合、日志和急停链通过 |
| G2 | 正式采集器 | square/sine/Gaussian、motor-side 数据和安全字段完整 |
| G3 | 数据集 | 连续 sequence、transition、split、normalization 和质量报告正确 |
| G4 | UAN 训练环境 | reset、baseline torque、5 ms 更新和 synthetic recovery 通过 |
| G5 | UAN 训练 | 多 seed 收敛，输出受限且无 reward exploit |
| G6 | 离线验证 | 未见波形、20 s 和 5 min rollout 均优于基线 |
| G7 | locomotion 集成 | UAN 只进 simulator，torque/power limit 顺序正确 |
| G8 | policy 重训 | nominal/DR/white-box/UAN 至少 3 seeds 对照 |
| G9 | sim2sim | UAN-trained policy 在 Isaac/MuJoCo 行为安全一致 |
| G10 | 实机验证 | 新 policy 逐级上机并证明 sim2real 改善 |

---

## 5. G0：冻结契约与判断是否需要 UAN

### 5.1 冻结版本

从工作区根目录执行：

```bash
git -C X1_train rev-parse HEAD
shasum X1_train/resources/robots/x1/urdf/x1.urdf
shasum X1_train/resources/robots/x1/mjcf/xyber_x1_flat.xml
shasum X1_infer/module/ankle_identifier_module/cfg/ankle_identifier.yaml
shasum X1_infer/install/linux/bin/cfg/x1_cfg_identifier.yaml
shasum X1_infer/module/dcu_driver_module/cfg/dcu_x1.yaml
shasum X1_infer/module/dcu_driver_module/cfg/ankle_trans_x1.yaml
shasum X1_infer/module/control_module/cfg/rl_x1.yaml
```

记录：

- robot、左右踝 actuator 和 DCU 序列号/版本。
- virtual joint 与 actuator 的 name、sign、offset 和单位。
- 左右踝 transmission 类型、参数表 hash 和坐标系。
- command/state 频率、source clock 和延迟。
- Kp/Kd、LPF、position/velocity/effort limit。
- torque-speed curve、总功率和温度限制。
- 固定方式、地面/无接触状态和现场照片编号。

### 5.2 先排除软件契约 Gap

正式 UAN 前必须完成总体公共 G0，并特别修复：

- torque multiplier 当前在 `_compute_torques()` 每次调用重新采样的问题。
- friction/damping reset 累乘问题。
- Isaac 与 MuJoCo torque limit 不一致。
- deployment position clamp 被注释的问题。
- action delay、LPF 和并联踝 command coordinate 不一致。

否则 UAN 会学习软件 bug，而不是硬件残差。

### 5.3 建立启用判据

UAN 不是默认给所有关节增加的模块。启动 UAN 前应至少有一份 nominal 或白盒 replay 报告，证明：

- 时间同步、PD、delay、limit 和刚体模型已对齐。
- held-out 中仍存在可重复的非线性残差。
- 残差随方向、负载、历史或耦合轴变化，简单 friction/armature/delay 难以同时拟合。
- 残差不是单次通信异常、温漂或固定架运动。

如果 PACE 白盒模型已经达到指标，UAN 只保留为独立对照，不应强行进入主线。

---

## 6. G1：当前 step/sine MVP

这一步只验证采集链，不训练正式 UAN。

### 6.1 当前能力

**[当前可运行]**：

- `mode: step` 或 `mode: sine`。
- `test_side: left/right`。
- `test_axis: pitch/roll`。
- 1 kHz command loop。
- pre-hold、active、post-hold 和 repeat。
- startup stable joint velocity/gyro Gate。
- primary 和 coupled joint position/velocity/effort。
- IMU quaternion 和 gyro。
- 测试完成后持续 hold。

当前 step 是一个有限持续时间的 position-target pulse，不是论文中的 direct-torque square wave；当前 sine 只有单一 amplitude/frequency。

### 6.2 首轮安全配置

使用 round-specific 配置，不直接覆盖发布包。首轮沿用已审批的 PACE 小幅基线：

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
csv_path: ./log/uan/round_00/left_roll_step_000.csv
```

`0.003 rad` 只作为已在 PACE 文档中提出的首轮工程起点，仍需现场批准。当前仓库默认 `0.015 rad` 不能直接视为首轮安全值。

`startup_pose_mode: zero` 会主动移动到零位，首轮不使用。`auto_stop_after_test` 不退出进程，完成后仍需人工停止。

### 6.3 启动和记录

在部署 `bin` 目录：

```bash
source ./ros2_source.sh
mkdir -p ./log/uan/round_00
ros2 bag record -a -o ./log/uan/round_00/rosbag_step_000
```

另一个终端：

```bash
source ./ros2_source.sh
./run_identifier.sh
```

立即确认：

```bash
ros2 topic info /joint_cmd -v
ros2 topic hz /joint_states
ros2 topic hz /actuator_states
```

`/joint_cmd` 必须只有 identifier publisher。禁止同时启动 `ControlModule`。

### 6.4 MVP 顺序

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

每个 run 使用新配置 hash 和输出文件。MVP 通过条件：

- primary/coupled 方向正确。
- joint 与 motor topic 均有数据。
- pre/active/post 完整。
- 无饱和、振荡、异常声响和固定架运动。
- command/state 时间单调，采样率符合配置。
- pitch 激励时 roll coupled response、roll 激励时 pitch coupled response 均被记录。

---

## 7. G2：实现正式 UAN 采集器

### 7.1 激励模式

**[需扩展]** `AnkleIdentifierModule`：

```yaml
mode: square | sine | gaussian
coordinate: joint | motor
seed: 20260811

schedule:
  - side: left
    axes: [pitch]
    amplitude: <approved>
    frequency_hz: <approved>
    duration_sec: <approved>
```

必须支持：

- 连续 square wave，而不是只做单次 step pulse。
- 多 amplitude/frequency sine schedule。
- seeded piecewise-constant Gaussian。
- 单轴、pitch/roll 双轴和左右踝组合 schedule。
- 平滑 start/end envelope。
- 独立 train/held-out schedule。
- 配置 hash、run id、waveform id 和 sample index。

### 7.2 位置激励与 torque 激励

论文使用 torque waveforms；X1 当前安全入口是 position target + 已知 Kp/Kd。第一阶段使用位置激励，但必须：

- 记录最终 q_des/dq_des/tau_ff/Kp/Kd。
- 在 simulator 中复现相同 PD、LPF、delay 和 saturation。
- 数据标签写成 `position_excitation`，不能称为 direct-torque dataset。

只有 DCU 和机械团队批准独立 torque-mode 安全接口、在线限幅和急停后，才建立 `torque_excitation` 数据。禁止仅把 Kp/Kd 置零后直接发送高幅 Gaussian effort。

### 7.3 波形覆盖

论文对每个 actuator 的 square/sine 使用 12 个幅值/频率组合、约 50 秒数据，并采集约 5 分钟 simultaneous Gaussian；Gaussian 每 5–400 ms 重采样。X1 将其作为覆盖结构参考，不复制幅值。

X1 ankle 建议分组：

| 数据组 | 内容 | 用途 |
|---|---|---|
| W0 | 当前低幅 step/sine | 链路和方向 |
| W1 | 单轴 square 多组合 | 死区、上升/下降和方向不对称 |
| W2 | 单轴 sine 多组合 | 频率、幅值和相位响应 |
| W3 | 单轴 Gaussian | 单轴随机 transition |
| W4 | pitch/roll 同时 Gaussian | 并联耦合 |
| W5 | 不同静态姿态/可测负载 | 负载泛化 |
| W6 | 独立日期/温度 session | 最终 held-out |

Gaussian 必须先短时、低幅验证，再分轮延长；不能从 5 分钟满幅开始。

### 7.4 必录字段

| 类别 | 字段 |
|---|---|
| 身份 | run/session/waveform id、config/robot/transmission hash |
| 时间 | source/arrival/monotonic timestamp、sequence、实际周期 |
| joint command | q_des、dq_des、tau_ff、Kp、Kd、pre/post filter/limit |
| joint state | q、dq、effort estimate、name/sign/offset |
| actuator command | motor q/dq/tau/Kp/Kd，transmission 后最终值 |
| actuator state | motor q/dq/effort estimate |
| 安全 | current、voltage、temperature、fault、saturation、estop |
| 结构 | primary/coupled、Jacobian/condition indicator、coordinate mode |
| IMU/base | quaternion、gyro、base stationary flag |

当前 `DcuDriverModule` 已给 `/joint_states` 加 timestamp，但 `/actuator_cmd` 和 `/actuator_states` 当前没有赋 header timestamp。正式数据前必须修复并验证三个 topic 使用同一采样周期的同一 source stamp。

当前驱动只公开 position/velocity/effort；电流、电压、温度和 fault 需要从 SDK/固件能力扩展。UAN 不依赖 torque label，但这些字段是实机安全硬要求。

### 7.5 数据采集器

**[待实现]**：

```text
X1_infer/assistant/uan_data_recorder/
```

要求：

- rosbag 保存原始 topic。
- 同时写统一 HDF5/Parquet/NPZ sequence 文件。
- 所有 command/state 按 name 对齐，不依赖 map 遍历顺序。
- 记录每个 sample 的 source timestamp 和 waveform phase。
- heartbeat、限位和温度异常实时触发停止。
- 急停仍保存已完成数据和事件时间。
- 每个 run 自动生成 `data_quality.json`。

---

## 8. 正式实机采集步骤

### 8.1 每轮准备

1. 冻结 plan、配置、waveform seed 和输出路径。
2. 核对 fixed-base、足端无接触和防坠状态。
3. 核对 `/joint_cmd` 只有一个 publisher。
4. 核对 `/joint_states`、`/actuator_states`、`/actuator_cmd`、`/imu/data` 频率和 timestamp。
5. 核对 current/temperature/fault safety channel。
6. 以 `startup_pose_mode: current` 进入稳定 hold。
7. 开始 rosbag 和 UAN recorder。
8. arm 唯一 waveform plan。

### 8.2 单轴 square/sine

1. 先左/右 roll、再左/右 pitch。
2. 一次只改变 amplitude 或 frequency。
3. 每个组合至少重复 3 次。
4. 正负方向必须覆盖。
5. 每组间保留足够 cooldown/hold。
6. 每轮结束检查 primary/coupled 和两个 actuator。

幅值/频率网格从硬件批准范围构造，不因论文写了 12 组合就默认执行 12 个危险工况。12 只是达到覆盖后的目标数量级。

### 8.3 Gaussian

推荐渐进顺序：

```text
单轴，最小 sigma，短 duration
  → 单轴，多 dwell time
  → pitch/roll 同时，小 sigma
  → pitch/roll 同时，多 dwell time
  → 左右踝共享 schedule
  → 长 duration
```

每个 Gaussian run 固定 seed，并保存实际经过安全 limiter 的 waveform。训练使用 executed command，不使用 limiter 前计划值。

### 8.4 单轮结束

1. waveform 平滑回到 baseline。
2. 保持 post-hold，确认稳定。
3. 停止 identifier 进程和 rosbag。
4. 生成数据质量报告。
5. 标记 Pass/fail 和任何 safety event。
6. 未审查结果前不执行下一 amplitude/sigma/duration。

---

## 9. G3：数据预处理与 split

### 9.1 同步和清洗

**[待实现]**：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.uan_prepare_data \
  --manifest deploy/sim2real/artifacts/uan/<run_id>/manifest.yaml \
  --raw deploy/sim2real/artifacts/uan/<run_id>/raw \
  --output deploy/sim2real/artifacts/uan/<run_id>/processed \
  --target-rate 200
```

步骤：

1. 按 source timestamp 对齐 joint command/state、actuator command/state 和 IMU。
2. 转换到 manifest 的 canonical sign/unit。
3. 应用 joint offset 的位置必须与部署一致，不重复减 offset。
4. 保留 1 kHz raw；使用 anti-alias filter 后生成 200 Hz UAN sequence。
5. command 用零阶保持；q 可插值；dq 优先使用传感器值并报告滤波方法。
6. 标记丢帧、急停、饱和、固定架运动和通信异常区间。
7. 无效区间不进入训练，但不能从 raw 中删除。
8. 保存 transmission 输入/输出及映射残差。

### 9.2 数据 schema

每个 5 ms sample 至少包含：

```text
t
q_des, dq_des, tau_ff, kp, kd
q_real, dq_real
tau_baseline_real_definition
motor_cmd, motor_state
waveform_type, amplitude, frequency/dwell, seed
temperature, voltage, saturation, valid_mask
```

transition：

```text
(s_t, command_t, s_t+1)
s_t = [q_t, dq_t]
```

同时保留连续 sequence；UAN 不能只用随机独立 transition 训练，因为 100 ms history 和 20 秒稳定性都依赖时间连续性。

### 9.3 split

禁止相邻帧随机拆分。按完整 run/session 分：

- train：部分 square/sine/Gaussian 组合。
- validation：未参与训练的完整组合和 seed。
- test-waveform：未见 amplitude/frequency/dwell。
- test-session：不同日期、温度或装夹。
- test-coupled：pitch/roll simultaneous waveform。

normalization 只用 train 统计量，并绑定 dataset hash。

### 9.4 数据质量 Gate

- 200 Hz 序列无时间倒退和无法解释的 gap。
- command/state 相对延迟已估计。
- joint 与 motor coordinate 可往返检查。
- valid sequence 足够覆盖 20 秒 rollout。
- 正负方向、幅值、频率、速度和负载分布可视化完成。
- 饱和样本占比接近 0；持续饱和 run 作废。
- train/validation/test 无同一 run 泄漏。

---

## 10. G4：UAN 训练环境

### 10.1 新增独立环境

**[待实现]**：

```text
X1_train/humanoid/envs/x1/x1_uan_env.py
X1_train/humanoid/envs/x1/x1_uan_config.py
X1_train/humanoid/algo/uan/
X1_train/humanoid/scripts/train_uan.py
X1_train/humanoid/scripts/eval_uan.py
```

当前 `DHPPO` 是 locomotion actor-critic，使用 Adam、共享单一 optimizer 和 locomotion storage，不等价于论文 UAN 训练配置。UAN 应使用独立 network、runner、storage 和 optimizer，不能把 2 维 residual action 塞进现有 12 维 locomotion PPO。

### 10.2 每个 episode 的执行语义

1. 从一个完整有效 sequence 采样起点。
2. 将 simulator q/dq 设置为该时刻真实 q/dq。
3. 初始化 error history；首帧 q_sim=q_real 时用零误差或使用明确的 100 ms warm-up replay。
4. 从 sequence 回放连续 q_des/dq_des/tau_ff/Kp/Kd。
5. simulator 用自己的 q_sim/dq_sim 计算 baseline tracking torque。
6. UAN 每 5 ms 根据 simulator error history 输出 `delta_tau`。
7. `tau_baseline + delta_tau` 经过残差、torque-speed 和 power limit 后施加。
8. physics 以 1 ms substep 前进，5 ms 时刻比较 q_sim 与 q_real。
9. episode 持续约 20 秒；PPO rollout horizon 可短于 episode，但不得每 96 步重置物理状态。

禁止每步把 q_sim 强制重置为 q_real，否则网络只学到单步修正，无法证明长时稳定。

### 10.3 X1 baseline torque

位置接口下：

```text
tau_baseline = Kp * (q_des - q_sim)
             + Kd * (dq_des - dq_sim)
             + tau_ff
```

必须加入与实机相同的：

- q_des LPF。
- command/action delay。
- joint offset 语义。
- parallel ankle coordinate mapping。
- torque-speed 和 total-power limit。

若 MVP simulator 仍是 serial ankle，`tau_baseline` 和 `delta_tau` 都在 virtual pitch/roll joint coordinate 中定义；报告中必须明确这是等效 joint-side 模型。

### 10.4 Synthetic recovery

先构造 hidden actuator residual，例如已知 dead-zone、Coulomb+viscous friction、delay 或二维 coupling：

1. 用 hidden residual simulator 生成 synthetic sequence。
2. 使用 nominal simulator 训练 UAN。
3. 验证 UAN 能降低未见 waveform 的 q error。
4. 验证 20 秒和 5 分钟 rollout 不发散。
5. 验证限制器会拒绝异常大 `delta_tau`。

synthetic recovery 不通过时不得训练实机数据。

---

## 11. G5：网络、限制和 PPO 训练

### 11.1 MVP 网络

```text
Target: left/right ankle, canonical joint coordinate
Input per step: [e_q_pitch, e_dq_pitch, e_q_roll, e_dq_roll]
History: 20 × 5 ms = 100 ms
Input dim: 80
MLP: [128, 128], ELU
Output: [delta_tau_pitch, delta_tau_roll]
Update: 200 Hz
```

actor 输出建议经过：

```text
delta_tau = delta_tau_max * tanh(raw_output)
```

再经过 rate limit。`delta_tau_max` 和 rate limit 来自硬件/仿真安全与训练数据覆盖，不能让网络任意输出。

### 11.2 可选 ablation

仅在 MVP held-out 明确不足时比较：

- scalar shared UAN：pitch/roll 独立处理。
- 2D coupled UAN：推荐 MVP。
- left/right separate UAN。
- 增加 baseline torque history。
- 增加 motor position/velocity history。

禁止把 base pose、完整全身 observation、waveform id 或未来真实状态直接输入 actor，以免记忆数据集。

### 11.3 Reward

论文 reward 由 joint-position match 和 action smoothness 组成。可用其系数作为初始复现实验：

```text
-1.5 * |q_real - q_sim|_1
+4.0 * exp(-100  * ||q_real - q_sim||^2)
+4.0 * exp(-300  * ||q_real - q_sim||^2)
+5.0 * exp(-1000 * ||q_real - q_sim||^2)
+0.5 * exp(-0.5 * ||delta_tau_t - delta_tau_t-1||)
```

X1 在使用前应先按角度单位、轴数和 baseline error 归一化。额外加入硬约束而不是可被 reward trade-off 的软项：

- `delta_tau` amplitude/rate limit。
- total torque-speed envelope。
- total power limit。
- NaN/Inf 立即终止。
- simulator instability 立即终止并施加大 penalty。

### 11.4 PPO 初始配置

论文参考配置：

```text
gamma: 0.995
GAE lambda: 0.95
entropy: 0
critic learning rate: 5e-4 fixed
actor learning rate: adaptive, target KL 0.01
rollout horizon: 96
environments: 4096
mini epochs: 5
optimizer: AdamW
weight decay: 0.01
```

4096 env 是资源目标，不是硬性要求。减少 env 时必须重新检查 gradient variance 和验证误差，不能声称严格复现论文。

### 11.5 训练命令

**[待实现]**：

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.train_uan \
  --task x1_uan_ankle \
  --dataset deploy/sim2real/artifacts/uan/<run_id>/sequences \
  --config deploy/sim2real/configs/uan_train.yaml \
  --limits deploy/sim2real/configs/uan_limits.yaml \
  --run_name <run_id> \
  --seed 1 \
  --headless
```

至少运行 3 seeds。保存：

- actor/critic checkpoint。
- actor deterministic mean checkpoint。
- normalization。
- optimizer 和 random state。
- train/validation reward 和 q error。
- residual amplitude/rate/saturation 分布。
- episode termination 原因。

### 11.6 防止 reward exploit

训练中检查：

- 网络是否长期贴 `delta_tau_max`。
- correction 是否出现高频正负抖动。
- q error 降低是否仅因过大 damping。
- 短 rollout 改善但长 rollout 发散。
- 一个 waveform 改善但其他 waveform 恶化。
- correction 是否补偿错误 delay/offset。

任何一项存在时先修数据/契约或收紧 actor input/output，不直接扩大网络。

---

## 12. G6：离线验证

### 12.1 基线

至少比较：

| 组别 | 模型 |
|---|---|
| A0 | nominal actuator simulator |
| A1 | 修复实现后的 actuator DR |
| A2 | 最佳白盒 friction/armature/delay 模型 |
| U1 | joint-side 2D UAN |
| U2 | UAN ablation/可选 motor-side 模型 |

PACE 可作为 A2 的来源之一，但第一轮 UAN 与 PACE 仍分别报告，不把组合模型伪装成独立方案结果。

### 12.2 指标

- one-step q/dq error。
- 0.1/1/5/20 秒 rollout q RMSE/MAE。
- phase lag、overshoot、settling time。
- pitch→roll、roll→pitch coupled error。
- 不同 amplitude/frequency/dwell 的误差。
- left/right、cold/hot、load session 泛化。
- `delta_tau` RMS、peak、rate 和 saturation ratio。
- torque-speed/power limit 触发比例。

### 12.3 未见波形

必须包含：

- 未见 square amplitude/frequency。
- 未见 sine amplitude/frequency。
- 未见 Gaussian seed 和 dwell time。
- simultaneous pitch/roll coupling。
- 至少一个不同 session。

不允许只报告从同一长 Gaussian run 随机切出的 test frames。

### 12.4 长 rollout

```bash
PYTHONPATH=X1_train python -m humanoid.scripts.eval_uan \
  --checkpoint deploy/sim2real/artifacts/uan/<run_id>/checkpoints/best.pt \
  --dataset deploy/sim2real/artifacts/uan/<run_id>/sequences/test \
  --durations 20,300 \
  --output deploy/sim2real/artifacts/uan/<run_id>/validation
```

通过条件：

- 未见 waveforms 均优于 A0/A1/A2 或至少不劣于最佳白盒基线。
- 20 秒和 300 秒不发散、不出现 NaN/Inf。
- q error 随时间不持续单调增长。
- residual 不长期贴 amplitude/rate limit。
- coupled-axis error 明显下降。
- 多 seed 结论一致。

单步误差改善而 20 秒发散视为失败。

---

## 13. G7：把冻结 UAN 接入 locomotion simulator

### 13.1 Torque path

在 `X1_train/humanoid/envs/base/legged_robot.py::_compute_torques()` 建立显式顺序：

```text
policy action
  → action scale / delay
  → baseline PD + fixed white-box terms
  → UAN error-history update every 5 ms
  → add delta_tau on selected ankle axes
  → torque-speed clipping
  → total power clipping
  → final hard torque limit
  → simulator
```

UAN 不得在每个 1 ms physics step 重复更新；每 5 ms 推理一次，其余 substep 保持输出。

### 13.2 冻结要求

- `eval()` 模式。
- 禁用 gradient。
- normalization 和 checkpoint 固定。
- locomotion 训练不更新 UAN 权重。
- 每个 env reset 时清零或按定义初始化 100 ms history。
- 记录 correction 和 limiter 统计用于训练监控。

### 13.3 与 DR 去重

对 UAN 覆盖的 ankle actuator properties：

- 不再保留当前宽范围 friction/damping/armature/torque-scale DR。
- 可根据 UAN 多 seed、session residual 建立小范围 residual uncertainty。
- base mass/COM、terrain、sensor noise、external pushes 等未由 UAN 覆盖的 DR 继续保留。
- policy/action lag 若已经被 UAN history稳定覆盖，仍需通过 ablation 决定是否收窄，不能自动删除。

当前 torque multiplier 每 `_compute_torques()` 重采样的 bug 必须先修复，否则 UAN 和 DR 会发生不可解释叠加。

### 13.4 训练配置入口

**[待实现]** 增加显式配置：

```yaml
actuator_residual:
  enabled: true
  simulator_only: true
  checkpoint: <uan_checkpoint.pt>
  normalization: <normalization.yaml>
  joints:
    - left_ankle_pitch
    - left_ankle_roll
    - right_ankle_pitch
    - right_ankle_roll
  update_period_sec: 0.005
  history_steps: 20
```

禁止通过手工改代码常量切换 UAN，使不同 seed 无法复现。

---

## 14. G8：重训 locomotion policy

### 14.1 对照组

| 组别 | 训练 simulator |
|---|---|
| B0 | 修复契约后的 nominal |
| B1 | 修复 bug 后的 broad DR |
| W1 | 最佳独立白盒 actuator model |
| U1 | nominal + frozen UAN + residual DR |

每组共享 X1 manifest、policy 架构、reward、命令 curriculum、训练步数和至少 3 seeds。

### 14.2 训练命令

当前入口：

```bash
cd X1_train
python humanoid/scripts/train.py \
  --task x1_dh_stand \
  --experiment_name x1_dh_stand \
  --run_name uan_<run_id> \
  --seed 1 \
  --headless
```

在新增 `--actuator_residual_config` 或注册 UAN task 前，上述命令仍不会加载 UAN。必须先实现显式入口并在日志首行输出 checkpoint/normalization hash。

### 14.3 训练监控

- locomotion reward 和 episode length。
- ankle correction RMS/peak/rate。
- correction saturation ratio。
- final ankle torque-speed/power limit。
- policy 是否刻意驱动 UAN 到数据覆盖外。
- action、contact、base 和 foot metrics。

如果 policy 学会利用 UAN 外推或限制器边界，应扩大真实安全数据覆盖或收紧 policy action domain，不能让 UAN 在实机侧补偿，因为实机不会运行 UAN。

### 14.4 导出

最终只导出 locomotion policy：

```bash
cd X1_train
python humanoid/scripts/export_onnx_dh.py \
  --task x1_dh_stand \
  --load_run <run_dir>
```

部署包中不得加入 UAN checkpoint 到 real-time ControlModule。实验 manifest 仍需记录 policy 是在哪个 UAN checkpoint 上训练得到的。

---

## 15. G9/G10：sim2sim 与实机验证

### 15.1 Sim2sim

按以下顺序比较 B0/B1/W1/U1 policy：

1. 原地站立。
2. 低速前进/后退。
3. 左右横移。
4. 左右转向。
5. 组合速度命令。
6. action/torque/power 极限附近但仍在批准包络内的仿真测试。

Isaac 和 MuJoCo 都不运行实机 UAN；对 U1 policy 的 MuJoCo 验证可选择：

- 在 MuJoCo simulator 中同样加入 frozen UAN，验证训练模型行为；以及
- 关闭 UAN，观察 policy 对 simulator 差异的敏感度。

两种结果必须分别标注，不能混为一组。

### 15.2 实机

U1 policy 上机顺序：

1. 防坠条件下 zero/stand。
2. 原地站立，不行走。
3. 25% 已批准前进命令。
4. 50% 前进并回零。
5. 低速后退、横移和转向。
6. 小幅组合命令。
7. 扩展到原 `pi_collect` 已批准包络。

每级比较：

- ankle pitch/roll trajectory 和 coupled motion。
- foot contact timing 和打滑。
- base tracking 和姿态。
- action/torque/current/power/temperature。
- 与 UAN-corrected simulation 的差异。
- 安全事件、急停和人工救援。

UAN 离线拟合改善但 locomotion 实机不改善，不能视为方案成功。

---

## 16. 建议首批工作单

### UAN-00：建立启用判据

```text
目标：证明并联踝存在白盒模型难以解释的稳定 held-out 残差
实机：复用 PACE/step-sine 数据
输出：uan_need_assessment.md
```

### UAN-01：冻结采集契约

```text
目标：冻结 joint/motor name、sign、offset、transmission、PD、limit 和时间语义
实机：不需要
输出：manifest.yaml
```

### UAN-02：补齐 actuator telemetry

```text
目标：给 actuator cmd/state 加 source timestamp，接入 current/voltage/temperature/fault
实机：被动状态 + 极低幅 step
输出：telemetry_quality.md
```

### UAN-03：扩展 waveform generator

```text
目标：实现 square/sine schedule、seeded Gaussian、envelope 和 run id
实机：先仿真，后低幅短时
输出：uan_collection.yaml、waveform test
```

### UAN-04：采集 W1/W2

```text
目标：四个踝轴单轴 square/sine 多组合
实机：需要，多轮
输出：single-axis dataset
```

### UAN-05：采集 W3/W4

```text
目标：单轴及 pitch/roll simultaneous Gaussian
实机：需要，严格渐进放量
输出：Gaussian/coupled dataset
```

### UAN-06：transition 数据集

```text
目标：完成 1 kHz raw、200 Hz sequence、split、normalization 和质量报告
实机：不需要
输出：dataset_manifest.yaml
```

### UAN-07：训练环境与 synthetic recovery

```text
目标：验证 reset、20 s rollout、residual limits 和已知 hidden residual recovery
实机：不需要
输出：synthetic_recovery.md
```

### UAN-08：训练 joint-side 2D UAN

```text
目标：至少 3 seeds 得到受限 ankle residual network
实机：不需要
输出：uan_checkpoint.pt、normalization、limits
```

### UAN-09：held-out 与长 rollout

```text
目标：未见 waveform、20 s 和 5 min 均优于 A0/A1/A2
实机：不需要
输出：uan_validation_report.md
```

### UAN-10：接入 locomotion simulator

```text
目标：frozen UAN 以 200 Hz 注入 ankle torque，完成 DR 去重
实机：不需要
输出：integration tests、torque-path report
```

### UAN-11：重训对照

```text
目标：B0/B1/W1/U1 至少 3 seeds
实机：不需要
输出：locomotion_ablation.md
```

### UAN-12：逐级实机验证

```text
目标：证明 U1 policy 的踝耦合和 sim2real 指标实际改善
实机：需要
输出：UAN 最终报告和是否进入主线的结论
```

---

## 17. 最小可执行路径

```text
第 1 周期
  UAN-00 判断是否值得做
  → UAN-01 冻结契约
  → UAN-02 补 telemetry/timestamp
  → UAN-03 实现 waveforms

第 2 周期
  UAN-04 单轴 square/sine
  → UAN-05 小幅 Gaussian/coupled
  → UAN-06 sequence dataset

第 3 周期
  UAN-07 synthetic recovery
  → UAN-08 训练 2D ankle UAN
  → UAN-09 held-out + 20 s + 5 min

第 4 周期
  UAN-10 simulator integration
  → UAN-11 locomotion retraining
  → UAN-12 staged hardware validation
```

对当前 X1，最重要的第一实现不是立即写 MLP，而是补齐 actuator timestamp、安全 telemetry 和正式 waveform。没有可信连续 sequence，UAN 即使训练 loss 很低也没有辨识意义。

