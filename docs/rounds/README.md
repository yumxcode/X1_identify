# 轮次交付记录（rounds）— 项目实践时间线

> 本目录按时间归档**每轮有结论的工作**（辨识轮次、数据采集、方案设计、证伪实验等），一轮一档、只追加不修改。
> 与 [../methods_log.md](../methods_log.md) 的分工：**methods_log 是方法视角**（什么方法 → 什么脚本 → 什么结论，跨轮次沉淀可复用资产）；**本目录是时间视角**（某轮做了什么、交付了什么、任务号与日志在哪，完整快照）。

## 使用规则

1. 每完成一轮有结论的工作，在本目录新建 `YYYY-MM-DD_<topic>.md`（topic 用短英文 slug），内容含：目标、做法、结果（PASS/FAIL 如实）、关键数字、任务号/日志位置、遗留问题。
2. 同轮多份关联文档放同名子目录（如 `2026-08-31_prime/`）。
3. 新档创建后**必须**更新下方时间线索引表。
4. 门禁级 PASS/FAIL 结论只能引用带单测的检查代码（`validate_spi.py`），ad-hoc 结论注明"参考"。

## 时间线索引

| 日期 | 轮次 | 类型 | 主题与结论 | 详细记录 |
|---|---|---|---|---|
| 2026-05-08/09 | 阶跃数据采集 | 数据 | 吊架悬空单关节阶跃 ×12（1 kHz，kp40/kd2-3）→ 后续 M1 回归得 κs 证据带 [0.34, 0.71] | [data/README.md §4](../../data/README.md)、methods_log M-002 |
| 2026-08-10 | 三方案设计 | 设计 | SPI / UAN / PACE 三候选方案定义 + G0 软件契约 Gap 清单（v0.2，后由 SPI 路线主导，UAN/PACE 留待装备升级） | [archive/x1_sim2real_system_design.md](../archive/x1_sim2real_system_design.md) |
| 2026-08-24 | walk_diag 采集 | 数据 | 15 s @100 Hz 行走诊断（rl_walk_leg，cmd 0.25 m/s）→ SPI 主数据（与 F1 侧 md5 一致） | [data/README.md §4](../../data/README.md) |
| 2026-08-25 | F1 侧 SPI 辨识 | 辨识 | F1 环境 v14 辨识（TASK_20260825_018）+ v15 验证 PASS（TASK_20260825_030）：质量 3.783 kg、κs 0.434、holdout -70.7% | methods_log M-001、[../spi_design.md](../spi_design.md) v1–v15 迭代史 |
| 2026-08-31 | PRIME 路线交付 | 辨识/证伪 | 无 mocap 适配 PRIME 复现：G1/G2 FAIL——**惯性参数在本传感配置下原理性不可辨识**（7 实验证据链）；G3/G4/G5 PASS（GRF 整机 37.355 kg，TASK_20260831_122） | [2026-08-31_prime/DELIVERY.md](2026-08-31_prime/DELIVERY.md)、[prime_identify/IDENTIFIABILITY.md](../../prime_identify/IDENTIFIABILITY.md) |
| 2026-09-02 | SPI 集成 + 原生验证 | 辨识 | 仓库改造为专属辨识项目；SPI 流水线 vendor + R1–R7 原生任务：**原生辨识+原生验证闭环 PASS**（R4 参数：质量 3.428 kg、κs 0.396、holdout -68.7%；地板再基线 13.5→13.8） | [2026-09-02_spi_integration.md](2026-09-02_spi_integration.md) |

## 当前有效基准（速查）

- **原生辨识基准**：R4（TASK_20260902_022）+ R7 冻结复验 PASS（TASK_20260902_034）→ `spi_identify/results/r4_native_identified_params.json`
- **交叉基准**：F1 v15 参数原生复验 PASS（TASK_20260902_030）
- 两套参数差异集中于弱可观方向（质心/惯量/κ_hip）；κs 与 κ_knee 互洽 → 详见 [../sysid_path.md §2 路线 A](../sysid_path.md)
