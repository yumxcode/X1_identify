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
| 2026-09-02/03 | 多数据集辨识轮（x1_data 新数据） | 辨识 | 10 条新 walk_diag（270 s，三策略组）入 train/cross 双桶；新增完成标准 5 CROSS-DATASET 与关节级 GATE-J1..J5；**SPI 五项全 PASS**（R9 域内参数：骨盆 3.152 kg、κs 0.370、cross 0.307/13.940≤14.24，T9 TASK_20260903_073 exit 0）；关节 J4/J5 PASS、J1/J2/J3 边际 FAIL（根因量化；post-hoc 修订 T10 改善 knee R² +0.04~0.05 未翻转判定）；GRF 质量 G6 v2 PASS（直行 35.94 kg） | [2026-09-03_multidataset_sysid_report.md](2026-09-03_multidataset_sysid_report.md) |
| 2026-09-04 | 多 seed 地板标定 | 辨识/标定 | seed2/3 补跑（TASK_20260904_013/014，seed 传递验证生效）：3-seed 地板再基线 holdout 15.0（带宽 1.66 占满至绝对上限，单 seed 地板假 FAIL 实证）/ cross 14.405（带宽 0.21 稳定）；参数 seed 方差量化（骨盆 ±14%、κs 全带内）；R9 判定不变，官方参数维持 seed1（§2.5） | [2026-09-03_multidataset_sysid_report.md §2.5](2026-09-03_multidataset_sysid_report.md) |
| 2026-09-04 | **辨识方法与结果评审**（事后复算） | 评审 | 对 R9 与门禁体系的事后复算（**参考性，不推翻 T9 判定**）：R-1 开环回放在 clip 内已发散（holdout 平均姿态误差 62.8°→65.3°，比力残差 13.541 vs 零模型 1.42，差 9.5×，代价改善全部来自幅值类通道而 quat/q 反而变差）；R-2 m 与 κs 结构性简并（三 seed 各散 27–29%，m/κs 只散 3.7%）；R-3 三 seed 一致呈质量↓+惯量↑3–9× 的非物理组合；R-4 ACTUATOR 带下沿由 R²=0.58 的 hip_yaw 独撑，按 R²≥0.80 重取带 [0.546,0.706] 后三 seed 的 κs 全部出带；R-5 PHYSICAL 重言 + ACCEL 地板已退化为绝对上限；R-6 关节级悬空前提已被 IDENTIFIABILITY §2.6 证伪；R-7 伪惯量参考点定义。**处置**：暂缓回写惯量/质心、DR 窗口放宽、P0 补基座初速 + 发散诊断 | [2026-09-04_sysid_review.md](2026-09-04_sysid_review.md)（复现：`spi_identify/scripts/review_diag.py`） |

| 2026-09-04 | 辨识方法事后评审 + P0 发散诊断 | 评审/诊断 | R-1~R-7 参考性结论（回放窗口内发散 62.8-65.3°、m-κs 简并、ACTUATOR 带敏感、Steiner 语义）；P0 扫描（TASK_20260904_098）：**有效窗口 ~0.2-0.3 s**（§11.3 触发→装备升级上调为必要条件）、初速修复不改善（根因转回放保真度）、accel 残差=冲击瞬态 | [2026-09-04_sysid_review.md](2026-09-04_sysid_review.md)（§12 P0 结果） |

## 当前有效基准（速查）

- **多数据集基准（现行，R9）**：T8 域内再辨识（TASK_20260903_015）+ **T9 终判五项全 PASS**（TASK_20260903_073，exit 0；判定时地板 14.04/14.24）+ 3-seed 复核全 PASS（MS TASK_20260904_013/014）→ `spi_identify/results/r9_indomain_params.json`；现行地板 holdout 15.0 / cross 14.405（3-seed 再基线 2026-09-04，报告 §2.5）
- **原生单数据基准（R4）**：R4 辨识 + R7 冻结复验 PASS（TASK_20260902_034）→ `spi_identify/results/r4_native_identified_params.json`；R4 参数跨策略复验 ratio 0.318（T1）
- **交叉基准**：F1 v15 参数原生复验 PASS（TASK_20260902_030）
- κs 跨轮稳定 0.36–0.48（含 3-seed 方差）；骨盆质量 R4 3.43 / F1 3.78 / R9 3.15，seed 方差 ±14%（3.15/3.62/4.21，无动捕质量方向固有不确定度，报告 §2.5）→ 详见 [../sysid_path.md §2 路线 A](../sysid_path.md)
- ⚠️ **使用 R9 参数前必读** [2026-09-04_sysid_review.md](2026-09-04_sysid_review.md)：事后复算显示 m 与 κs 结构性简并（可辨识量是 **κs/m ≈ 0.113–0.117**，非 m 或 κs 各自），骨盆惯量与质心证据不足（建议暂缓回写，§9）；T9 门禁判定本身不变
