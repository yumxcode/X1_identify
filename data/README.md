# data/ — 真机数据目录（数据契约与目录约定）

> 本仓库是 **F1 数据观测体系的数据消费者**。数据由 F1 侧（`weilai-robot/F1`，`doc/测试体系/` D0–D7）按其采集流程产生并上传到这里；本仓库只做分析、系统辨识与 sim2real，不定义采集端字段。

## 1. 目录结构

```
data/
├── raw/        原始真机数据（原样保存，永不修改；后续真机数据上传落点）
│   ├── walk_diag_*.csv          行走诊断日志（100 Hz，F1 DATA-01）
│   └── *_step_*.csv             单关节阶跃激励日志（1 kHz）
├── derived/     由 raw 派生的中间产物与证据文件（脚本可再生，来源可追溯）
│   ├── x1_clips.npz             prepare_dataset.py 切片产物（远端生成后回传）
│   └── step_m1_regression_all.json   阶跃数据 M1 回归（κs 证据带）
└── README.md   本文件
```

## 2. 上游数据契约（F1 测试体系）

| 数据层 | 载体 | 频率 | F1 观测点 | 本仓库用途 |
|---|---|---|---|---|
| L1 walk_diag CSV | `data/raw/walk_diag_*.csv` | 100 Hz | DATA-01 | SPI 全身辨识主输入（关节 pos/vel/effort、pos_des/tau_des raw+lpf、IMU quat/gyro/accel、cmd、phase） |
| 阶跃激励 CSV | `data/raw/*_step_*.csv` | 1 kHz | —（专项激励） | 串联关节 M1 回归 → κs 证据带（辨识交叉校验） |
| L2 tm_obs bin | （暂未接入） | 100 Hz | DATA-02 | 策略观测向量校验（G0 契约对齐，待用） |
| L4 actuator 话题对 | rosbag（暂未接入） | 1 kHz | DATA-04 | 通信/执行误差分离、盲区补齐（待用） |

字段语义、时钟源与跨体系对齐注意事项见 F1 仓库 `doc/测试体系/D0_公共基础.md` 与 `D2_控制维度.md`（唯一权威）。

## 3. 上传约定（后续真机数据）

1. 新数据一律放 `data/raw/`，保持采集端原始文件名，不做任何清洗改名。
2. 同轮多文件（如一轮 S1–S3 场景采集）建议放 `data/raw/<轮次日期>_.../` 子目录，并在提交信息中注明 F1 侧对应的场景/采集流程。
3. 每个新数据集若要进入辨识流水线：在 `spi_identify/configs/x1_spi.yaml` 的 `data.sources` 增加条目（files + 该轮 kp/kd），其余阶段自动生效。
4. 上传后先跑 `remote_sysid.py`（gradmotion 远端）走一遍 dataset 阶段，确认 parse 通过、clip 数量合理，再决定是否全量重辨识。

## 4. 现有数据档案

| 文件 | 内容 | 采集日期 | 用途 |
|---|---|---|---|
| `raw/walk_diag_20260824_103222.csv` | 15.0 s @100 Hz 行走诊断（cmd 0.25 m/s，rl_walk_leg） | 2026-08-24 | SPI v13–v15 辨识主数据（与 F1 侧 md5 一致：`4e7af10d…`） |
| `raw/*_step_*.csv`（12 个） | 单关节阶跃（幅值 -0.24…0.24 rad，kp40/kd2 或 kp40/kd3） | 2026-05-08/09 | M1 回归 → κs 锚定 |
| `derived/step_m1_regression_all.json` | 12 关节 τ≈α·kp·e+β·kd·q̇+c 回归（α=0.34–0.71） | — | SPI 完成标准 4（ACTUATOR）证据带 |
| `derived/left_knee_pitch_step_20260509_091930.m1_regression.json` | knee 单关节回归（α=0.55，R²≈0.85） | — | κs 早期证据（与上者一致） |

## 5. 数据质量红线（消费者侧检查）

上传的数据进入辨识前确认：

- `timestamp_ns` 单调且周期稳定（walk_diag ≈10 ms，step ≈1 ms）；
- `imu_accel_*` 静置段 ≈ Rᵀ·g（z 轴 9.5–9.8 m/s²），否则 IMU 约定/单位有变；
- 串联关节 `effort_*` 非 NaN；`is_parallel_*` 与该轮控制模式一致；
- 覆盖工况足够（多速度/转向/起步停止），否则提示上游补采。
