# X1 参数辨识交付报告（PRIME 复现，无 mocap 适配）

> 日期：2026-08-31 · 仓库：github.com/yumxcode/X1_identify（main + validate-lite）
> 通过标准：`PASS_CRITERIA.md` v1.1（G1–G5）· 可辨识性根因：`prime_identify/IDENTIFIABILITY.md`

## 1. 目标与结果总览

| # | 目标 | 结果 |
|---|---|---|
| 1 | 关联远端仓库并推送 | ✅ origin/main 全量同步（455 MB）；validate-lite 轻量分支供远端验证 |
| 2 | 复现 PRIME 参数辨识 | ⚠️ 核心实现完成且自洽（多项单测）；**惯性参数在本数据传感配置下不可辨识**（G1/G2 FAIL，物理边界而非实现缺陷，§4） |
| 3 | 真机数据 + gradmotion 远端验证 | ✅ G3/G4/G5 全 PASS（终版任务 TASK_20260831_122）；远端与本地逐位一致（37.355 kg，偏差 0.000 kg） |
| 4 | 严格通过标准 | ✅ PASS_CRITERIA.md 先行定义、逐条判定；FAIL 项如实记录并给出根因证据链 |

## 2. 已交付（按 PASS_CRITERIA 条目授权）

| 交付物 | 门禁 | 说明 |
|---|---|---|
| `PASS_CRITERIA.md` | — | G1–G5 标准与阈值（v1.1 含 G3 约定裁决修订） |
| `prime_identify/IDENTIFIABILITY.md` | — | 不可辨识结论的完整证据链（5 个可复现实验） |
| `prime_identify/results/x1_gmass_anchored.urdf` | G3+G4 | 总质量锚定 37.10 kg（唯一改动：base_link 质量 +1.777 kg、惯量同比缩放；PD 保持）。**非参数辨识输出** |
| `prime_identify/results/gm_validation.json` | G5 | 远端=本地的验证记录 |
| `prime_identify/results/real_walk_ident.json` | G2（FAIL 记录） | 真机辨识尝试的完整 before/after（恶化如实记录） |
| `prime_identify/`（prime 包 + scripts） | — | PRIME 实现：log-Cholesky、接触 QP、PFIE-IRLS、自测套件 |

**真机核心数字**（walk_diag_20260824，15 s @ 100 Hz）：
- 整机质量（GRF 平衡，定版估计器）：**37.355 kg**，CI95 [32.04, 39.96]（88/120 帧收敛；本地/远端逐位一致）vs URDF 35.323 kg（**+5.8%**，与论文 G1 实机 +2.9 kg 增量同级，来源电池/外接件）
- 名义 URDF 物理一致性：**0/13 违规**（正确 origin 惯量约定 + 双向单测）。v1.0 曾误报 13/13（检查器约定 bug），已裁决修正
- 估计器基建精度：log-Cholesky 往返 4e-16；参数管线 round-trip 1.1e-14；接触 QP 静力平衡 0.1%；GRF/体重 = 0.999

## 3. PRIME 实现与论文的对应与偏离

| 论文要素 | 本实现 | 备注 |
|---|---|---|
| log-Cholesky 参数化（Eq.2–5） | ✅ `log_cholesky.py` | 往返自测 200 随机样本 1e-8 |
| 平滑接触（Eq.17–20，log-barrier） | ⚠️ 改为 Anitescu SOC-QP（κ→∞ 极限） | 论文 barrier 需 φ 精度 ~0.01 mm（mocap）；无 mocap 下 φ 误差 ~5 mm 使 barrier 双支（穿透分支赢）。QP 极限下力由平衡决定，对 φ 噪声鲁棒 |
| FIE/PFIE 目标（Eq.21–22） | ✅ 残差形式等价实现 | base 线速度不可观 → 行 mask |
| FDDP 求解器（Eq.24–34） | ❌ 改为 IRLS/回归 + 线搜索 | 单步残差形式无需轨迹优化 |
| 解析梯度（Eq.32–34） | ❌ 部分（回归子线性 + FD） | ∂λ/∂θ 解析链未实现；FD 被 PGS 数值噪声淹没 |

## 4. 为什么参数辨识 FAIL（一句话 + 证据）

**刚性接触把惯性参数的动力学影响吸收进接触力重解，参数不可观测；论文靠动捕基座测量（独立于接触力）打破该抵消，本数据无此传感。**

证据链（`IDENTIFIABILITY.md`，全部可复现）：
1. 信号通道开放：GT 成本 9.7 vs nominal 67.7（86% 可分）
2. 标量扫描平坦：base 质量 ±3 kg 内成本 82.9–85.9 无谷（GT +2.0 不在极小点）
3. null(J) 消 λ 子空间：扰动方向近零
4. swing 行（无足端接触）仍经基座加速度耦合 λ（M 非对角）→ 膝部扰动恢复 0%
5. 论文 G1 表恢复精度全部建立在 OptiTrack 基座测量上

## 5. 后续建议（按性价比）

1. **重采激励数据**：悬挂/固定基座单关节 chirp（记录实测 q），回归子满秩（设计文档 PACE 方案已有 runbook）
2. **加装传感**：基座 VIO/动捕（论文路线）或足端六维力（λ 直接可测）
3. 已有 13 个单关节 step 数据集补采关节位置列后可直接用于关节级辨识

## 6. 质量过程记录（教训留档）

- v1.0 交付曾含两处被中途审查纠正的问题：(a) 以"GRF 质量+名义修复"包装成辨识输出（已改名收敛为 anchored 单一工件）；(b) "13/13 物理不一致"诊断实为检查器 origin/COM 惯量约定 bug（已以带双向单测的提交代码裁决为 0/13）。教训：诊断结论必须以带单测的代码为准，不得以脚本 docstring 口头改口。

## 7. 仓库状态

```
main（全量）+ validate-lite（远端验证用孤儿分支）
├── PASS_CRITERIA.md            # 通过标准 v1.1
├── DELIVERY.md                 # 本报告
├── prime_identify/
│   ├── prime/                  # log_cholesky / dynamics(含 PC 检查器+单测) / data / pfie
│   ├── scripts/                # 自测、诊断、gm_validate（本地/远端同构）、make_gmass_urdf
│   ├── results/                # x1_gmass_anchored.urdf, gm_validation.json, real_walk_ident.json
│   └── IDENTIFIABILITY.md      # 不可辨识证据链
├── x1_data/                    # 真机数据（原样）
└── X1_train/ X1_infer/ deploy/ # 原工作区（未改动）
```
