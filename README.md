# X1_identify — X1 系统辨识与 Sim2Real 项目

> **定位**：消费 F1 数据观测体系的真机数据，沉淀一套**成熟可靠的 X1 参数辨识 + sim2real 分析方案与脚本**——当新数据到来时，能自主完成质量检查 → 参数辨识 → 完成标准验证 → 参数回写（URDF/MJCF/DR）→ 远端重训衔接。
> **上游**：[weilai-robot/F1](https://github.com/weilai-robot/F1)（正式代码发布项目，含数据观测体系 `doc/测试体系/` D0–D7）。
> **执行约束**：一切验证走 gradmotion 远端，本地不装依赖、不跑训练/仿真。

## 1. 项目现状（2026-09-02，结论先行）

**SPI 主流水线已闭环验证 PASS**（原生辨识 + 原生验证，远端任务 TASK_20260902_034），新数据到来即可按 §4 流程复用；项目处于"边分析边沉淀"阶段——方法经实践验证后沉淀为脚本，最终收敛为稳定辨识流水线。

| 结论 | 状态 | 依据 |
|---|---|---|
| **SPI 全身辨识（主路线）**：原生 R4 参数四项全过——骨盆 3.428 kg（名义 4.304）、κs 0.396（证据带内）、holdout 代价 -68.7% | ✅ PASS | R7 冻结复验，`spi_identify/results/r4_native_identified_params.json` |
| **F1 v15 交叉基准**：F1 侧参数（3.783 kg / κs 0.434）在本仓库原生复验 | ✅ PASS | TASK_20260902_030，两环境互洽 |
| **全身惯性参数回归（PRIME 路线）**：无基座线运动传感时原理性不可辨识 | ❌ 已证伪 | 7 实验证据链，`prime_identify/IDENTIFIABILITY.md` |
| **GRF 整机质量估计**：37.355 kg（+5.8% vs URDF 35.323，电池/外接件） | ✅ 有效 | 与 SPI 结果交叉参照，`prime_identify/results/gm_validation.json` |
| **SPI-Active 主动激励（Stage-2）** | ⏸ 待多行为策略 | `spi_identify/active/` 代码就绪 |

现状约束（无动捕、无固定基座工装、仅吊架 + 1 个稳定行走策略）下的完整路径与装备升级开关见 **[docs/sysid_path.md](docs/sysid_path.md)（方案主文档，生效中）**。

## 2. 目录结构

```
X1_辨识/
├── README.md                ← 本文件：项目总览
│
├── docs/                    ── 文档（方案 + 实践记录）──────────────────
│   ├── sysid_path.md          【方案·现行】现状约束下的辨识路径（主文档，持续维护）
│   ├── spi_design.md          【方案·SPI】论文精读 + X1 适配决策 + v1–v15 迭代史
│   ├── methods_log.md         【实践·台账】方法→脚本（方法视角，跨轮次沉淀）
│   ├── rounds/                【实践·轮次】每轮工作的交付快照（时间视角，一轮一档 + 索引）
│   └── archive/               【归档】三方案早期设计（2026-08-10 v0.2）+ 方案对比图
│
├── data/                    ── 真机数据 ─────────────────────────────
│   ├── raw/                   原始数据（原样保存：walk_diag 100Hz + 阶跃 1kHz）
│   ├── derived/               派生证据（M1 回归等，脚本可再生）
│   └── README.md              数据契约、上传约定、质量红线
│
├── spi_identify/                ── ★ SPI 辨识流水线（主路线代码）──────────
│   ├── spi/                   log-Cholesky 参数化 / 数据集 / 回放 / 代价 / 优化器 / 验证
│   ├── active/                SPI-Active Stage-2（FIM + Bézier，待多行为策略）
│   ├── configs/x1_spi.yaml    参数空间 / 代价权重 / 数据源（全部可配）
│   ├── scripts/               prepare / run_spi / validate / apply / remote_sysid …
│   ├── tests/                 57 单测（53 numpy 级 + 4 vendored MJCF 守卫）
│   ├── results/               辨识参数 + R1–R7 远端任务日志（原生基准在此）
│   └── export/                导出工件（辨识 URDF / MJCF / DR 配置 / 报告）
│
├── prime_identify/          ── PRIME 回归路线（已证伪，证据链留档）─────
│   ├── prime/ + scripts/      PFIE-IRLS 实现、诊断实验、GRF 质量估计
│   ├── IDENTIFIABILITY.md     不可辨识证据链（7 实验）——新数据先过此检查
│   └── results/               x1_gmass_anchored.urdf、gm_validation.json …
│
├── deploy/sim2real/         ── 实机轮次 runbook（SPI-Active / UAN / PACE）
├── X1_infer/                  部署框架（MJCF 模型来源，与 F1 同源）
└── X1_train/                  远端重训框架（agibot_x1_train，辨识产物替换 URDF/DR）
```

## 3. 快速开始（gradmotion 远端）

SPI 全流水线（单测 → 数据切片 → CMA-ES 辨识 → 完成标准验证 → 参数回写）：

```
gm-run X1_identify/spi_identify/scripts/remote_sysid.py
```

validate-only 模式（跳过 ~15 min 辨识，复用已提交参数做验证）：

```
gm-run X1_identify/spi_identify/scripts/remote_sysid.py --validate-only
gm-run X1_identify/spi_identify/scripts/remote_sysid.py --validate-only --params-file=spi_identify/results/r4_native_identified_params.json
```

任务退出码 = 验证判定（0=PASS / 1=FAIL）；产物经 gradmotion SDK 从 `logs/` 回传（`logs/spi_sysid/validation.json` 为判定报告）。详见 [spi_identify/README.md](spi_identify/README.md)。

## 4. 新数据到来时的标准动作

```
data/raw/ 放入新数据（保持原始文件名）
   → spi_identify/configs/x1_spi.yaml 的 data.sources 登记该轮 kp/kd
   → gm-run remote_sysid.py 走 dataset 阶段质检（parse 通过、clip 数合理，见 sysid_path.md §3 五项检查）
   → 全量辨识（或 validate-only 复验）
   → 记录：docs/methods_log.md 追加方法结论 + docs/rounds/ 新建轮次档案
```

上传约定与质量红线详见 [data/README.md](data/README.md)。

## 5. 方法台账（方法 → 脚本，详见 docs/methods_log.md）

| 方法 | 脚本 | 状态 |
|---|---|---|
| SPI 采样式白盒辨识（CMA-ES + MuJoCo 开环回放） | `spi_identify/scripts/run_spi.py` | ✅ 原生 R4/R7 PASS + F1 v15 交叉 PASS |
| SPI 完成标准四项验证 | `spi_identify/scripts/validate_spi.py` | ✅（地板 13.8 原生再基线） |
| 阶跃数据 M1 回归（κs 锚定） | F1 侧脚本 → `data/derived/step_m1_regression_all.json` | ✅ |
| 质量-代价地形诊断 | `spi_identify/scripts/mass_landscape.py` | ✅ |
| GRF 整机质量估计（无动捕） | `prime_identify/scripts/gm_validate.py` | ✅（37.355 kg） |
| 惯性参数可辨识性证伪链（7 实验） | `prime_identify/scripts/diag_*.py` | ✅（结论：本传感配置下原理性不可辨识） |
| SPI-Active 命令优化（FIM + Bézier） | `spi_identify/scripts/run_active.py` | ⏸ 待多行为策略 |

## 6. 文档导航

| 类别 | 文档 | 说明 |
|---|---|---|
| 方案·现行 | [docs/sysid_path.md](docs/sysid_path.md) | 现状约束下的辨识路径（路线 A/B/C + 装备升级开关） |
| 方案·SPI | [docs/spi_design.md](docs/spi_design.md) | SPI 论文精读与 X1 适配全记录 |
| 方案·实机 | [deploy/sim2real/](deploy/sim2real/) | 三方案实机 runbook（一轮一计划、一轮一结果） |
| 实践·台账 | [docs/methods_log.md](docs/methods_log.md) | 方法→脚本→结论，跨轮次沉淀 |
| 实践·轮次 | [docs/rounds/](docs/rounds/) | 每轮交付快照 + 时间线索引 |
| 历史·归档 | [docs/archive/](docs/archive/) | 三方案早期设计（2026-08-10）+ PRIME 轮交付（2026-08-31）+ SPI 集成轮（2026-09-02）等 |
| 数据契约 | [data/README.md](data/README.md) | 与 F1 D0–D2 映射、上传约定、质量红线 |
| 证据链 | [prime_identify/IDENTIFIABILITY.md](prime_identify/IDENTIFIABILITY.md) | 惯性参数不可辨识证据（新数据先过此检查） |
| 使用手册 | [spi_identify/README.md](spi_identify/README.md) | SPI 流水线目录结构与关键适配 |
