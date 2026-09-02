# SPI / SPI-Active Sim2Real Pipeline for X1（X1_identify 工作区版）

> 目录名 `spi_identify/`（2026-09 由 `sim2real/` 改名，与 `prime_identify/` 对称命名：一条辨识路线一个目录）；历史文档/任务日志中的 `sim2real/` 路径即指本目录。

基于 **SPI-Active**（*Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning*, CoRL 2025 Oral, arXiv:2505.14266）的 X1 系统辨识与 sim2real 流水线。

> vendored from F1 `dev/sim2real-spi`（v15 PASS），路径已适配本仓库布局。
> 完整设计（论文精读 + X1 适配决策 + v1–v15 迭代史）见 [`docs/spi_design.md`](../docs/spi_design.md)。

## 流水线总览

```
真实日志 (data/raw/walk_diag_*.csv, 100Hz; 1kHz 阶跃日志作 κs 证据)
   │ prepare_dataset.py          变长 clips (H~U(1s,2s)), 初态对齐（含 IMU 三轴比力）
   ▼
data/derived/x1_clips.npz
   │ run_spi.py                  Optuna CMA-ES × MuJoCo 开环回放
   │                             log-Cholesky (m,r,I) + tanh 电机 κ + κs
   │                             train/val 划分，val 为 holdout
   ▼
logs/spi_sysid/gm_play/identified_params.json|.pt
   │ validate_spi.py             完成标准验证：val 提升/物理合理域/IMU 比力/κs 交叉校验
   ▼
logs/spi_sysid/validation.json  (verdict: PASS/FAIL, 退出码 0/1)
   │ apply_params.py             回写 URDF/MJCF + 生成 DR 配置
   ▼
spi_identify/export/{x1_identified.urdf, xyber_x1_identified.xml, dr_x1_spi.json}
   └─→ gradmotion 远端重训（X1_train = agibot_x1_train 框架）→ 新 ONNX → F1 部署
```

Stage-2（可选迭代）：`run_active.py` 用 FIM(tr F⁻¹)+Bézier 命令优化生成最优激励命令序列，真机执行后再回到 `run_spi.py`（前置：多行为策略，见 `docs/sysid_path.md` 路线 C）。

## 执行方式：一切走 gradmotion 远程（本仓库不本地装依赖）

### 一键远程

```bash
# startScript（已含 pip 依赖安装 + 单测 + 全流水线）:
gm-run X1_identify/spi_identify/scripts/remote_sysid.py
```

`remote_sysid.sh` 在远端镜像内完成：`pip install mujoco optuna pyyaml matplotlib` → numpy 级单测（57 个，门禁；含 4 个 vendored MJCF 守卫）→ 依次跑 prepare/run_spi/mass_landscape/apply_params。模型 MJCF 取自 `spi_identify/resources/mjcf/`（vendored 自 X1_infer，剥离 mujoco-3.x-only 传感器以兼容镜像 mujoco 2.3.6；守卫单测钉住与源一致），**无外部兄弟仓库依赖**。产物落在 `logs/`（gradmotion SDK 扫描上传）。

validate-only 模式（跳过 ~15 min CMA-ES，用 `spi_identify/results/identified_params.json` 冻结参数复验）：

```bash
gm-run X1_identify/spi_identify/scripts/remote_sysid.py --validate-only
```

### 本地（仅 numpy 级，无需安装——但按项目约束验证仍走远端）

```bash
python3 -m unittest discover -s spi_identify/tests   # 57 个单元测试
python3 spi_identify/scripts/prepare_dataset.py ...  # 数据准备仅需 numpy+pyyaml
```

## 目录结构

```
spi_identify/
├── configs/x1_spi.yaml        # 参数空间/代价权重/clip/active 全部可配
│                              #   model.mjcf -> X1_infer/.../xyber_x1_flat.xml
│                              #   data.sources -> data/raw/walk_diag_*.csv
├── spi/
│   ├── param_space.py         # log-Cholesky ↔ (m,r,I)；tanh 电机模型；物理合理域罚
│   ├── dataset.py             # 真实 CSV → clips（并联踝=力矩指令，串联=位置指令；含 IMU 比力）
│   ├── rollout.py             # MuJoCo 开环回放（初态对齐/惯量注入/IMU 比力预测）
│   ├── cost.py                # 论文 Table 3 代价（无动捕项可关；+IMU 比力项）
│   ├── optimizer.py           # Optuna CMA-ES 驱动（+物理合理域硬约束）
│   └── validate.py            # 完成标准判定（numpy 级，可单测）
├── active/
│   ├── fim.py                 # 有限差分 FIM（delta_param/ksync_steps）
│   ├── bezier.py              # Bézier 命令重参数化
│   └── command_opt.py         # tr(F⁻¹)+终止惩罚 命令优化
├── scripts/                   # prepare/run_spi/validate_spi/mass_landscape/apply/active/remote_sysid
├── tests/                     # 单测 57（53 numpy 级 + 4 vendored MJCF 守卫）
├── results/                   # 辨识参数与远端任务日志：原生 R4+R7 PASS（r4_native_identified_params.json）+ F1 v15 基准 + R1-R7 迭代日志
├── export/                    # 导出工件（URDF/MJCF/DR/报告）
├── resources/mjcf/            # vendored 2.3.6 兼容 MJCF（serial+flat+env；meshdir 指回 X1_infer）
└── resources/x1_nominal.urdf  # 名义 URDF（与 X1_train resources 逐位一致）
```

## 关键适配（X1 vs 论文）

| 项 | 决策 |
|---|---|
| 无动捕 | base_pos/base_linvel 代价默认 0；改用 **IMU 三轴比力**（imu_accel_*，日志已有）对比仿真 Rᵀ(a−g)，约束基座平移动力学（质量/质心） |
| 物理合理域 | 参数范围收紧（质量 3.0–5.5 kg、质心 ±0.06 m、惯量 0.005–0.38 kg·m²、惯量积 ≤0.03），优化目标加入超域硬罚（penalty_scale=1e8），杜绝超物理结果 |
| 完成标准 | `validate_spi.py` 四项全过才 PASS：① holdout 代价 ≤ nominal 70%；② 物理域（质量/质心/惯量特征值+惯量积）；③ IMU 比力 RMS ≤ min(15, max(地板 13.5, 0.35×nominal))——地板为无动捕开环回放的方法学实测界（3 次运行 12.55–13.01）；④ **κs 落在 1kHz 阶跃数据 M1 回归证据带 [0.34, 0.71]**（独立于行走数据的交叉校验）。输出 `validation.json` 与 PASS/FAIL 退出码 |
| 仿真器 | MuJoCo（`X1_infer` MJCF，与 F1 Humanoid_motion 同源） |
| 执行器 | 串联髋/膝：驱动器 PD 回放；并联踝：τ_des_lpf 直接回放；统一过 κ·tanh 饱和 |
| 辨识对象 | 骨盆 x1-body (4.304 kg) + 4 组电机 κ（hip_pitch/hip_rolleyaw/knee/ankle）+ κs |
| 命令空间 | 当前 3 维 cmd_vel；接入 WTW 式多行为策略后扩展 Stage-2 |

## 远端重训衔接

`apply_params.py` 产出的 `x1_identified.urdf` 替换 `X1_train/resources/robots/x1/urdf/x1.urdf`，`dr_x1_spi.json` 的范围替换 `x1_dh_stand_config.py` 的 `domain_rand` 段（以辨识值为中心的窄带 DR，论文 nominal range 策略），随后按常规流程创建 gradmotion 训练任务。
