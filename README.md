# X1_identify — X1 专属 Sim2Real 系统辨识项目

> 定位：消费 F1 数据观测体系的真机数据，专注 **系统辨识 + sim2real**。
> 上游：[weilai-robot/F1](https://github.com/weilai-robot/F1)（正式代码发布项目，含数据观测体系 `doc/测试体系/` D0–D7）。
> 执行约束：**一切验证走 gradmotion 远端**，本地不装依赖、不跑训练/仿真。

## 1. 这个仓库做什么

```
F1 真机数据（DATA-01 walk_diag 100Hz / 1kHz 阶跃 / 后续 DATA-02/04）
        │  上传至 data/raw/（见 data/README.md 数据契约）
        ▼
┌── 本仓库（消费者）────────────────────────────────────────────┐
│ ① 数据分析：质量红线检查、IMU 约定校验、工况覆盖评估          │
│ ② 系统辨识：SPI 主路线（sim2real/）+ PRIME 回归路线（prime_identify/）│
│ ③ 产出回写：identified URDF/MJCF + 以辨识值为中心的 DR 配置    │
│ ④ 远端重训衔接：gradmotion 任务（X1_train 框架）→ 新策略 ONNX  │
└──────────────────────────────────────────────────────────────┘
```

**现状约束与方案路径**：无动捕、无固定基座工装，仅有吊架 + 一个稳定行走的算法模型。在此约束下如何做数据分析与辨识，见 [docs/sysid_path.md](docs/sysid_path.md)（持续维护的主文档）。

## 2. 目录结构

```
├── sim2real/            SPI 系统辨识流水线（主路线，vendored from F1 dev/sim2real-spi）
│   ├── spi/             log-Cholesky 参数化 / clip 数据集 / MuJoCo 回放 / 代价 / 优化器
│   ├── active/          SPI-Active Stage-2（FIM + Bézier 命令优化，待多行为策略后启用）
│   ├── configs/x1_spi.yaml    参数空间/代价权重/数据源（全部可配）
│   ├── scripts/         prepare / run_spi / validate / mass_landscape / apply / remote_sysid
│   ├── tests/           43 个 numpy 级单测（远端流水线首阶段执行）
│   ├── results/         F1 v15 PASS 辨识参数（同一数据/模型，参考基准）
│   └── export/          上一轮导出工件（URDF/MJCF/DR/报告）
├── prime_identify/      PRIME 回归路线（既有）：PFIE-IRLS + 接触 QP + 可辨识性证据链
├── data/                真机数据（raw/ 原样 + derived/ 派生证据），见 data/README.md
├── docs/
│   ├── spi_design.md    SPI 方案设计全文（论文精读 + X1 适配决策 + v1–v15 迭代史）
│   └── sysid_path.md    现状约束下的数据分析与辨识路径（持续维护）
├── X1_infer/            小脑仿真模块（MJCF 模型来源，与 F1 Humanoid_motion 同源）
├── X1_train/            远端重训框架（agibot_x1_train，辨识产物替换 URDF/DR）
├── deploy/sim2real/     三方案（SPI/UAN/PACE）实机轮次 runbook
├── x1_sim2real_system_design.md   三方案早期设计（历史文档）
├── DELIVERY.md / PASS_CRITERIA.md PRIME 路线上一轮交付与门禁（历史记录）
```

## 3. 快速开始（gradmotion 远端）

SPI 全流水线（单测 → 数据切片 → CMA-ES 辨识 → 完成标准验证 → 参数回写），startScript：

```
gm-run X1_identify/sim2real/scripts/remote_sysid.py
```

validate-only 模式（跳过 ~15 min 辨识，复用已提交参数做验证）：

```
gm-run X1_identify/sim2real/scripts/remote_sysid.py --validate-only
```

任务退出码 = 验证判定（0=PASS / 1=FAIL）；产物经 gradmotion SDK 从 `logs/` 回传（`logs/spi_sysid/validation.json` 为判定报告）。

接入新真机数据：放入 `data/raw/` → 在 `sim2real/configs/x1_spi.yaml` 的 `data.sources` 登记该轮 kp/kd → 远端重跑。详见 [sim2real/README.md](sim2real/README.md) 与 [data/README.md](data/README.md)。

## 4. 辨识方法记录（方法 → 脚本沉淀）

每轮辨识/分析沉淀到 [docs/methods_log.md](docs/methods_log.md)：方法名、输入数据、核心假设、脚本入口、远端任务号、结论与可信度。被验证有用的方法直接沉淀为 `sim2real/scripts/` 或 `prime_identify/scripts/` 下的可复用脚本，避免一次性 ad-hoc 分析。

已有方法（详见 methods_log）：

| 方法 | 脚本 | 状态 |
|---|---|---|
| SPI 采样式白盒辨识（CMA-ES + MuJoCo 开环回放） | `sim2real/scripts/run_spi.py` | ✅ v15 PASS |
| SPI 完成标准四项验证 | `sim2real/scripts/validate_spi.py` | ✅ |
| 阶跃数据 M1 回归（κs 锚定） | F1 侧脚本 → `data/derived/step_m1_regression_all.json` | ✅ |
| 质量-代价地形诊断 | `sim2real/scripts/mass_landscape.py` | ✅ |
| GRF 整机质量估计（无动捕） | `prime_identify/scripts/gm_validate.py` | ✅（37.355 kg） |
| 惯性参数可辨识性证伪链（7 实验） | `prime_identify/scripts/diag_*.py` | ✅（结论：本传感配置下原理性不可辨识） |
| SPI-Active 命令优化（FIM + Bézier） | `sim2real/scripts/run_active.py` | ⏸ 待多行为策略 |

## 5. 相关文档

- [docs/sysid_path.md](docs/sysid_path.md) — 现状约束下的辨识路径（主文档）
- [sim2real/README.md](sim2real/README.md) — SPI 流水线使用手册
- [docs/spi_design.md](docs/spi_design.md) — SPI 论文精读与 X1 适配全记录
- [data/README.md](data/README.md) — 数据契约与上传约定
- [prime_identify/IDENTIFIABILITY.md](prime_identify/IDENTIFIABILITY.md) — PRIME 路线不可辨识证据链
- [deploy/sim2real/](deploy/sim2real/) — 实机轮次 runbook（一轮一计划、一轮一结果）
