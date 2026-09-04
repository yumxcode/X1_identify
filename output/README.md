# X1 辨识输出汇总（顶层交付快照）

> 本目录是全仓库辨识工作的**单一对外输出口**：辨识了哪些指标、结果是多少、辨识后的模型文件是什么。
> 由 `scripts/make_output.py` 于 **2026-09-04 03:21 UTC** 生成（快照，可再生：`python3 scripts/make_output.py`）。
> 参数版本 **R9**（seed1，T8 TASK_20260903_015 辨识，T9 TASK_20260903_073 正式终判 exit 0）；
> 完整方法与证据链见 `docs/rounds/2026-09-03_multidataset_sysid_report.md`。
> 模型工件与 R9 参数一致性校验：**✅ 全部一致**

## 1. 辨识指标清单与结果总览

| 层面 | 辨识指标 | 结果 | 判定 |
|---|---|---|---|
| 整机·SPI | 骨盆质量 / 质心 / 惯量张量 | **3.1520 kg**（nominal 4.3042）/ com [0.02255454, 0.02409425, -0.01261008] / I 见 §2 | 五项全 PASS（T9 exit 0） |
| 整机·SPI | 电机刚度 κ ×4（hip_pitch/hip_rolleyaw/knee/ankle） | hip_pitch 76.1 / hip_rolleyaw 39.8 / knee 73.1 / ankle 19.8 | 域内（域见 `x1_spi.yaml`） |
| 整机·SPI | 力矩缩放 κs | **0.3701**（独立证据带 [0.34, 0.71]） | ✅ |
| 关节·模组 | 串联关节 J_eff / τc / τv / 延迟 / α / k_t（×8 串联 + ×4 并联踝参考） | 延迟 6–9 ms、k_t R²=1.000（12/12）、参数表见 §3 | J4/J5 PASS；J1/J2/J3 边际 FAIL（参考值交付） |
| 整机·质量 | GRF 反算整机总质量（直行组） | **35.943 kg**（n=7，URDF 35.323 的 +1.8%） | ✅ G6 v2 PASS |
| 负结果 | 全身惯性参数（PRIME 回归） | 本传感配置下原理性不可辨识 | 已证伪留档（防重复投入） |

## 2. SPI 参数（R9，官方参数集）

| quantity | nominal | identified (R9) |
|---|---|---|
| 骨盆质量 [kg] | 4.3042 | **3.1520** |
| 骨盆质心 [m] | [0.00252285, -0.00063439, 0.03023409] | [0.02255454, 0.02409425, -0.01261008] |
| 骨盆惯量（对角）[kg·m²] | [0.0268, 0.0108, 0.0218] | [0.019006, 0.110257, 0.111269] |
| 骨盆惯量积 [kg·m²] | 0 | ixy -0.019177, ixz 0.010349, iyz 0.006736 |
| κs | 1.0 | **0.3701** |
| 电机 κ | — | hip_pitch 76.1 / hip_rolleyaw 39.8 / knee 73.1 / ankle 19.8 |

**五项完成标准终判**（`validate_spi.py`，实测值 vs 门限）：

| 标准 | R9 实测 | 门限 | 判定 |
|---|---|---|---|
| EFFECTIVENESS | 0.358 | <= 0.70 | ✅ |
| PHYSICAL | in-domain (by construction) | x1_spi.yaml box | ✅ |
| ACCEL | 13.541 | <= 14.04 (T9) / 15.0 (3-seed re-baseline) | ✅ |
| ACTUATOR | 0.3701 | [0.34, 0.71] | ✅ |
| CROSS-DATASET | ratio 0.307 / rms 13.94 | ratio <= 0.70, rms <= 14.24 (3-seed: 14.405) | ✅ |
nominal→R9：holdout 代价 1,184,627 → 423,748（-64.2%）；跨策略组比力 22.26 → 13.94（-37.4%）。

**seed 稳定性**（3-seed 地板标定，TASK_20260904_013/014；官方参数固定为 seed1=R9，不挑最优 seed）：

| seed | 骨盆 [kg] | κs | holdout 比力 | cross 比力 |
|---|---|---|---|---|
| seed1 (R9, official) | 3.152 | 0.37 | 13.541 | 13.94 |
| seed2 | 4.205 | 0.482 | 13.099 | 13.894 |
| seed3 | 3.619 | 0.409 | 14.759 | 14.105 |

可信度：质量/κs/κ(knee,ankle) 高；质心/κ(hip) 中；惯量低（无动捕弱可观——绝对 seed 带宽骨盆 ±14% 应如实认知）。

## 3. 关节级参数（T7，参考值 — 未回写模型文件）

模型：`τ_meas − g(q) = J_eff·q̈ + τc·tanh(q̇/ε) + τv·q̇ + c0（+陀螺耦合，串联）`；g(q) 为悬空重力矩 LUT。

| 关节 | 类型 | J_eff [kg·m²] | τc [Nm] | τv [Nm·s/rad] | 延迟 [ms] | α(M1) | k_t | 动力学 R² |
|---|---|---|---|---|---|---|---|---|
| left_ankle_pitch_joint | parallel-ankle | 0.0643 | 1.24 | 0.551 | 8 | 5.94 | 68.2 | 0.16 |
| left_ankle_roll_joint | parallel-ankle | 0.0212 | 0.83 | 0.145 | 8 | 2.29 | 68.2 | 0.31 |
| left_hip_pitch_joint | serial | 0.3848 | 1.91 | 0.533 | 9 | 0.724 | 190.6 | 0.80 |
| left_hip_roll_joint | serial | 0.2938 | 1.88 | 1.14 | 9 | 0.831 | 190.6 | 0.57 |
| left_hip_yaw_joint | serial | 0.0362 | 0.1 | 0.173 | 7 | 0.528 | 21.9 | 0.86 |
| left_knee_pitch_joint | serial | 0.2549 | 1.71 | 0.392 | 9 | 0.644 | 190.6 | 0.53 |
| right_ankle_pitch_joint | parallel-ankle | 0.0819 | 1.22 | -0.386 | 8 | 6.2 | 68.2 | 0.37 |
| right_ankle_roll_joint | parallel-ankle | 0.0237 | 0.49 | -0.118 | 8 | 1.92 | 68.2 | 0.43 |
| right_hip_pitch_joint | serial | 0.3098 | 1.77 | 0.8 | 9 | 0.702 | 190.6 | 0.70 |
| right_hip_roll_joint | serial | 0.3898 | 1.79 | 0.625 | 8 | 0.755 | 190.6 | 0.66 |
| right_hip_yaw_joint | serial | 0.0362 | 0.13 | 0.124 | 6 | 0.483 | 21.9 | 0.88 |
| right_knee_pitch_joint | serial | 0.2213 | 1.16 | 0.328 | 9 | 0.629 | 190.6 | 0.50 |

**GATE-J 门禁终判**（J2 已修正为 serial-only；T10 post-hoc 方法修订后判定未翻转）：

| 门 | 判定 | 依据 |
|---|---|---|
| J1_DYNAMICS_R2 | ❌ FAIL | left_knee_pitch_joint R2=0.534<0.6; right_knee_pitch_joint R2=0.498<0.6 |
| J2_LR_SYMMETRY | ❌ FAIL | worst left_hip_roll_joint|right_hip_roll_joint.tau_v rel_diff=0.82 (<= 0.35); 4 violations / 16 pairs |
| J3_GAIN_BAND | ❌ FAIL | left_hip_roll_joint alpha=0.831 outside (0.34, 0.71)+-0.10 |
| J4_DELAY | ✅ PASS | serial delays within (-5.0, 30.0) ms: left_hip_pitch_joint:9.0, left_hip_roll_joint:9.0, left_hip_yaw_joint:7.0, left_knee_pitch_joint:9.0, right_hip_pitch_joint:9.0, right_hip_roll_joint:8.0, right_hip_yaw_joint:6.0, right_knee_pitch_joint:9.0 |
| J5_CURRENT_MAP | ✅ PASS | kt R2 >= 0.9 on all 12 joints |

> 并联踝（ankle_pitch/roll）的 α 与动力学 R² 为并联驱动模式伪影，仅作参考；关节级摩擦/惯量参数**尚未回写** URDF/MJCF——回写属下一阶段工作（见报告 §7）。

## 4. 整机质量（GRF，跨数据集一致性）

| 组 | n | 均值 [kg] | 门限 | 判定 |
|---|---|---|---|---|
| 直行（7500/7500dr/legacy） | 7 | **35.943** | [32.0, 40.0] | ✅ PASS |
| 侧移/转向（5999，参考不设门） | 4 | 30.421 | — | 方法工况偏差 -14% 实证 |

## 5. 辨识后模型工件（`model/`）

由 `spi_identify/scripts/apply_params.py` 以 R9 参数回写生成（远端 apply 阶段），本目录为快照副本；sha256 前 12 位供校验：

| 文件 | 回写 body | sha256-12 | vs R9 参数 |
|---|---|---|---|
| `model/x1_identified.urdf` | base_link | 00578fa399f4 | ✅ 一致 |
| `model/xyber_x1_identified.xml` | x1-body | 79bc9abc2cb9 | ✅ 一致 |
| `model/dr_x1_spi.json` | DR config | 642bb25fbc0b | ✅ 一致 |

- **`x1_identified.urdf`**：58 link，总质量 34.17 kg；`base_link` 惯性参数替换为 R9。
- **`xyber_x1_identified.xml`**：MJCF（与 F1 部署同源），30 body，总质量 34.91 kg；`x1-body` 替换为 R9。
  （两模型总质量不同源于各自名义模型差异，本轮仅回写骨盆；κs/κ 为回放模型参数，不写入静态模型文件。）
- **`dr_x1_spi.json`**：域随机化配置——质量 ±5%、com ±0.03 m、增益 ±10% 以 R9 辨识值为中心。

**未纳入**：`prime_identify/results/x1_gmass_anchored.urdf`（37.1 kg 锚定版，G4 单数据集时代产物，与现行 35.94 kg 证据不一致，留档不交付）。

## 6. 溯源与再生成

| 内容 | 来源 |
|---|---|
| SPI R9 参数 + 验证指标 | `spi_identify/results/r9_indomain_params.json`（+ seed2/3 同目录） |
| 关节级参数 + GATE-J | `spi_identify/results/remote_logs/T9_TASK_20260903_073_final_verdict.log`（T7 辨识 TASK_20260902_222） |
| GRF 整机质量 | `prime_identify/results/gm_validation_multidataset.json`（分组规则同 `gm_validate.py` G6 v2） |
| 模型工件原件 | `spi_identify/export/`（`apply_params.py --params r9_indomain_params.json`） |
| 方法与证据链 | `docs/rounds/2026-09-03_multidataset_sysid_report.md`、`docs/methods_survey.md` |

新数据到来 / 参数轮次更新后：先跑 `remote_sysid.py` 终判 PASS，再执行 `python3 scripts/make_output.py` 重建本目录。
