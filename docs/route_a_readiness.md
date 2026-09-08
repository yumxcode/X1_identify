# 路线 A 数据采集协议与管线就绪度（2026-09-08，R10/R10V 后落定）

> **背景**：P1V9→R10→R10V 干预实验链（均 2026-09-08）已证明当前传感配置（仅 IMU + 关节编码器 + 吊架）下：
> - ACCEL 门禁（零模型 3× bar）结构性不可达——比力残差为接触冲击瞬态（best 8.5 = 零模型 10.6×）；
> - m–κs 结构性简并不随回放保真度修复破除（κs 跨语义 0.370→0.377）；
> - 再辨识零参数增益（同语义下 R9 参数追平反超 R10）。
>
> 因此"真值级"X1 URDF 需要**动捕/VIO 全局位姿**或**固定基座工装**数据（`sysid_path.md` §4）。本文档把该外部前置落成**可执行协议**：采集什么、怎么验收、到位后管线怎么接线、一次任务如何闭环。

## 1. 采集协议（动捕/VIO 路线，首选）

### 1.1 需要记录的通道

| 通道 | 最低要求 | 说明 |
|---|---|---|
| 基座全局位姿 (x,y,z) | ≥100 Hz，与 walk_diag 同源时钟 | **当前 CSV 无此通道**（dataset.py 只解析 imu_quat/gyro/accel）——需新增列 `base_pos_{x,y,z}`（世界系，m） |
| 基座全局速度（可选） | ≥100 Hz | 可由位姿差分；直接给 `base_linvel_{x,y,z}` 更好 |
| 既有全部通道 | 与现行 walk_diag 契约一致 | q/qd/tau/cmd/kp/kd/imu_*（见 `data/README.md`） |

### 1.2 工况清单（最小集）

1. **直行行走** ≥3 条、每条 ≥60 s、速度分档 0.15/0.25/0.35 m/s（覆盖现行 train 桶工况）；
2. **侧移/转向** ≥2 条（现行 cross 桶工况）；
3. **激励增强工况** ≥2 条：变速/停走交替、原地踏步交替频率（提升惯量可观测性，非必需但强烈建议）；
4. 记录吊架是否介入（分段标注）。

### 1.3 验收红线（数据入库前过 `prepare_dataset.py --config`）

- 位姿通道无跳变（相邻帧位移 < 0.5 m）；静态段漂移 < 2 cm/min；
- 与 IMU 姿态一致性：动捕四元数与 imu_quat 夹角均值 < 5°（时标对齐证据）；
- clip 切分通过现行 G 质检（parse 通过、clip 数合理）。

## 2. 管线接线清单（数据到位后的代码改动，预估 1 个工作日）

| # | 改动 | 位置 | 现状 |
|---|---|---|---|
| W1 | `parse_csv` 新增 `base_pos`（及可选 `base_linvel`）列解析 | `spi/dataset.py::parse_csv`（现只读 imu_*） | **需新增** |
| W2 | clip 化时保留基座位姿参考轨迹 | `spi/dataset.py`（Clip.ref_* 模式） | **需新增** |
| W3 | 回放代价加入基座位置/速度项：`cost.base_pos`/`base_linvel` 权重从 0 打开（如 2.0/0.25，对齐 quat/angvel 量级） | `configs/x1_spi.yaml::cost`（键已存在，值为 0） | **配置即可** |
| W4 | 初态基座速度改由位姿差分（替代 `base_linvel_mode` 里程计/zero） | `spi/rollout.py`（`solve_stance_base_linvel` 已有，新增差分模式） | **小改** |
| W5 | 验证门禁重定标：移除零模型 accel 地板分支、收紧 `accel_rms_max` 至 1.5 量级（`sysid_path.md` §4 预置值） | `configs/x1_spi.yaml::validation` | **配置即可** |
| W6 | 单测：合成数据上 base_pos 通道往返 + 代价项梯度有限性 | `tests/test_dataset.py` 模式 | **需新增** |

W1–W4 完成后即可重跑全量辨识（fidelity 三键保持现行开启值），W5 在首轮结果后按零模型实测再定。

## 3. 闭环命令（数据 + 接线就绪后）

```
# 登记 kp/kd → data/README.md 契约，然后：
gm-run X1_identify/spi_identify/scripts/remote_sysid.py --seed 1 --n-trials 250
# PASS（新门禁）→ apply 回写官方 export/ → python3 scripts/make_output.py → 新轮次档
```

## 4. 固定基座工装路线（备选）

工装就位后走 `deploy/sim2real/pace_runbook.md` 悬空 chirp 激励（已有 runbook），执行器参数 (Ia, d, τf, qbias, Td) 满秩回归；与 SPI 路线互补（关节级真值 + 基座参数来自动捕），不替代。

## 5. 责任与状态

- 采集执行：硬件/采集侧（本仓库外部）；
- 接线 W1–W6：本仓库，**等数据到位后同批落地**（避免无数据的半成品滞留——见经验教训"批次半成品"）；
- 就绪度本文档维护：`docs/methods_log.md` 轮次记录引用本页。
