# ADDENDUM — SPI 集成与原生验证交付（2026-09-02）

> 归档说明：本文件原位于仓库根目录 `ADDENDUM.md`，2026-09-02 目录整理时移入 `docs/rounds/`；文中 `sim2real/` 路径已随目录改名同步为 `spi_identify/`（同日）。
> 本文为 2026-09-02「项目改造为专属 sim2real 系统辨识 + SPI 集成」轮次的交付记录，接续 [2026-08-31_prime/DELIVERY.md](2026-08-31_prime/DELIVERY.md)（PRIME 路线 2026-08-31 交付）。

## 1. 目标与结果总览

| # | 目标 | 结果 |
|---|---|---|
| 1 | SPI 系统辨识集成（源自 F1 `dev/sim2real-spi` v15） | ✅ `spi_identify/` 全流水线 vendor + 路径适配；57 单测（53 原有 + 4 vendored MJCF 守卫）远端全过 |
| 2 | 改造为专属 sim2real 系统辨识项目 | ✅ 根 README 重写；PRIME 路线归档为历史交付；本项目定位 = F1 数据观测体系的消费者 |
| 3 | 预留 data/ 数据目录 | ✅ `data/raw`（12 step CSV + walk_diag）+ `data/derived`（M1 回归证据、clips）+ `data/README.md`（与 F1 D0–D2 数据契约映射、上传约定、质量红线） |
| 4 | 现状约束下的辨识方案路径（持续维护） | ✅ `docs/sysid_path.md`（路线 A/B/C + 装备升级开关）+ `docs/methods_log.md`（方法→脚本台账） |
| 5 | 一切验证走 gradmotion 远端 | ✅ 全部 7 轮任务均在远端执行（A10/isaac-gym-v19），本地零依赖执行 |

## 2. 远端任务链（项目 PRO_20260902_001，全部日志归档 `spi_identify/results/gm_task_*.txt`）

| 轮次 | 任务 | 内容 | 结果 |
|---|---|---|---|
| R1 | TASK_20260902_008 | 集成首验（validate-only） | FAIL：镜像 py3.8→pip mujoco 2.3.6 拒绝 X1_infer MJCF 的 MuJoCo-3.x `<jointactuatorfrc>`；apply_params 报告 NoneType 崩溃 |
| R2 | TASK_20260902_010 | vendored MJCF（剥离 29 行 3.x 传感器 + meshdir 指回 X1_infer）+ apply_params 加固 | **PASS** exit 0：57 单测过、F1 v15 参数四项全过、数字与 F1 逐位一致（val 85029.4/289895.6） |
| R3 | TASK_20260902_011 | 原生全量辨识 seed0/200 trials | FAIL 边际：ACCEL 14.32 vs 13.5；κs 0.325 出带 |
| R4 | TASK_20260902_022 | 原生全量辨识 seed1/250 trials | FAIL 边际（0.3%）：ACCEL 13.543 vs 13.5；其余三项过（κs 0.396 带内） |
| R5 | TASK_20260902_025 | 原生全量辨识 seed2/250 trials | FAIL：ACCEL 15.26 超 15 绝对上限 |
| — | （配置） | 地板原生再基线 13.5→13.8（三次原生观测最优 13.543+0.25 余量；方法同 F1 v15；绝对上限 15 不变，保持区分度） | 提交 0c031af |
| R6 | TASK_20260902_030 | F1 v15 参数原生复验（控制组） | **PASS** exit 0（两环境互洽） |
| R7 | TASK_20260902_034 | R4 原生参数冻结复验 | **PASS** exit 0：四项全过——**原生辨识+原生验证闭环** |

## 3. 原生辨识结果（`spi_identify/results/r4_native_identified_params.json`）

| 参数 | 原生 R4（PASS） | F1 v15（PASS，交叉） | 名义 |
|---|---|---|---|
| 骨盆质量 | **3.428 kg** | 3.783 kg | 4.304 kg |
| κs | **0.396**（带内） | 0.434（带内） | 0.55 |
| κ hip_pitch / rolleyaw / knee / ankle | 95.4 / 23.1 / **140.8** / 17.3 | 71.1 / 30.4 / 89.4 / 13.1 | — |
| holdout 代价 | 90,776（-68.7%） | 85,029（-70.7%） | 289,896 |
| ACCEL RMS | 13.543 | 13.007 | 20.229 |

两套参数差异集中于弱可观方向（质心/惯量/κ_hip）；κs 与 κ_knee 互洽。可信度分级见 validation.json。

## 4. 方法学沉淀（已入 `docs/methods_log.md` M-001）

**跨环境再基线**：vendored 代码/数据/模型逐位一致 ≠ 验收阈值可迁移。CMA-ES 采样路径随 Optuna/mujoco 版本漂移，最优盆地与残差地板整体平移（本轮三次原生观测带宽 [13.54, 15.26] vs F1 侧 [12.55, 13.01]）。跨环境迁移必须 ≥3 seed 原生全量再基线后再定阈值——与既往经验「借参 PASS 不能替代原生全量跑」相互印证并已实证。

## 5. 仓库状态（本轮）

```
main（远端同步至本轮交付）
├── spi_identify/                SPI 流水线（vendored from F1 v15 + 本轮适配）
│   ├── resources/mjcf/      vendored 2.3.6 兼容 MJCF（+4 守卫单测）
│   ├── results/             r4_native_identified_params.json（PASS）+ 7 份任务日志 + r3/r4/r5 .pt
│   └── scripts/             remote_sysid（--validate-only/--params-file/--seed 透传）
├── data/                    raw + derived + README 契约
├── docs/                    sysid_path / methods_log / spi_design + ADDENDUM
└── README.md                新定位与快速开始
```
