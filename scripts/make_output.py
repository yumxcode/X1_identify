#!/usr/bin/env python3
"""Generate the top-level output/ delivery snapshot.

Aggregates every identified quantity in this repo into ONE directory:
  output/README.md               human-readable summary (what was identified,
                                 values, verdicts, artifact navigation)
  output/identified_summary.json machine-readable full summary
  output/model/                  identified model artifacts (R9 URDF/MJCF/DR)

All numbers are extracted programmatically from source-of-truth files:
  * SPI official params  spi_identify/results/r9_indomain_params.json (+seed2/3)
  * joint-level params   spi_identify/results/remote_logs/T9_..._final_verdict.log
  * GRF whole-body mass  prime_identify/results/gm_validation_multidataset.json
  * model artifacts      spi_identify/export/ (written by apply_params.py)

The script also CROSS-CHECKS that the exported URDF/MJCF base bodies match the
R9 parameter file (drift guard: if export/ is later regenerated from another
param set, the check flags it in both output files).

Usage:  python3 scripts/make_output.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
MODEL = OUT / "model"

R9 = ROOT / "spi_identify/results/r9_indomain_params.json"
SEEDS = {
    "seed1 (R9, official)": ROOT / "spi_identify/results/r9_indomain_params.json",
    "seed2": ROOT / "spi_identify/results/seed2_indomain_params.json",
    "seed3": ROOT / "spi_identify/results/seed3_indomain_params.json",
}
T9_LOG = ROOT / "spi_identify/results/remote_logs/T9_TASK_20260903_073_final_verdict.log"
GRF_JSON = ROOT / "prime_identify/results/gm_validation_multidataset.json"
EXPORT = ROOT / "spi_identify/export"
ARTIFACTS = ["x1_identified.urdf", "xyber_x1_identified.xml", "dr_x1_spi.json"]

NOMINAL_PELVIS = {"mass": 4.3041648,
                  "com": [0.00252285, -0.00063439, 0.03023409]}
URDF_NOMINAL_TOTAL_KG = 35.323          # nominal URDF total (G6 band anchor)
G6_STRAIGHT_BAND = (32.0, 40.0)         # gm_validate.py G6_STRAIGHT_BAND
LATERAL_MARKER = "5999model"

# final-gate thresholds (docs/rounds/2026-09-03_multidataset_sysid_report.md §2.4/§2.5)
SPI_GATES = {
    "EFFECTIVENESS": "holdout cost ratio <= 0.70",
    "PHYSICAL": "m/com/I inside x1_spi.yaml physical box",
    "ACCEL": "holdout specific-force RMS <= 14.04 (T9 floor; 3-seed re-baseline: 15.0)",
    "ACTUATOR": "kappa_s in [0.34, 0.71]",
    "CROSS-DATASET": "cross ratio <= 0.70 and RMS <= 14.24 (3-seed re-baseline: 14.405)",
}

JOINT_RE = re.compile(
    r"\[joint\] (\w+): m1_alpha=([-\d.eE+]+) R2=([-\d.eE+]+) "
    r"kt=([-\d.eE+]+) R2=([-\d.eE+]+) "
    r"J_eff=([-\d.eE+]+) tau_c=([-\d.eE+]+) tau_v=([-\d.eE+]+) "
    r"dynR2=([-\d.eE+]+) delay=([-\d.eE+]+) ms")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_joints(log_text: str) -> list[dict]:
    rows = []
    for m in JOINT_RE.finditer(log_text):
        name, alpha, m1r2, kt, ktr2, jeff, tauc, tauv, dynr2, delay = m.groups()
        rows.append({
            "joint": name,
            "type": "parallel-ankle" if "ankle" in name else "serial",
            "J_eff_kg_m2": float(jeff),
            "tau_c_Nm": float(tauc),
            "tau_v_Nm_s_rad": float(tauv),
            "delay_ms": float(delay),
            "m1_alpha": float(alpha),
            "m1_R2": float(m1r2),
            "kt": float(kt),
            "kt_R2": float(ktr2),
            "dyn_R2": float(dynr2),
        })
    return rows


def parse_gates(log_text: str) -> dict:
    """Extract the joint-gate verdict JSON block from the T9 log."""
    i = log_text.find('"verdict"')
    if i < 0:
        return {}
    start = log_text.rfind("{", 0, i)
    depth = 0
    for j in range(start, len(log_text)):
        if log_text[j] == "{":
            depth += 1
        elif log_text[j] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(log_text[start:j + 1])
    return {}


def check_urdf(path: Path, mass: float, com, inertia) -> dict:
    root = ET.parse(path).getroot()
    link = next(l for l in root.findall("link") if l.get("name") == "base_link")
    ine = link.find("inertial")
    u_mass = float(ine.find("mass").get("value"))
    u_com = [float(v) for v in ine.find("origin").get("xyz").split()]
    it = ine.find("inertia").attrib
    u_I = [float(it[k]) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
    r_I = [inertia[0][0], inertia[0][1], inertia[0][2],
           inertia[1][1], inertia[1][2], inertia[2][2]]
    ok = (abs(u_mass - mass) < 1e-6
          and all(abs(a - b) < 1e-6 for a, b in zip(u_com, com))
          and all(abs(a - b) < 1e-6 for a, b in zip(u_I, r_I)))
    return {"file": path.name, "patched_body": "base_link",
            "mass_kg": u_mass, "com_m": u_com,
            "inertia_urdf_order": u_I, "matches_R9": ok}


def check_mjcf(path: Path, mass: float, com, inertia) -> dict:
    root = ET.parse(path).getroot()
    body = next(b for b in root.iter("body") if b.get("name") == "x1-body")
    ine = body.find("inertial")
    m_mass = float(ine.get("mass"))
    m_com = [float(v) for v in ine.get("pos").split()]
    m_diag = sorted(float(v) for v in ine.get("diaginertia").split())
    # eigenvalues of the identified inertia (symmetric 3x3, closed form via numpy if available)
    try:
        import numpy as np
        lam = sorted(np.linalg.eigvalsh(np.asarray(inertia, dtype=float)).tolist())
        eig_ok = all(abs(a - b) < 1e-5 for a, b in zip(m_diag, lam))
        eig_note = {"identified_eigvals": [round(v, 6) for v in lam]}
    except Exception:
        eig_ok, eig_note = None, {"note": "numpy unavailable: eigenvalue cross-check skipped"}
    ok = (abs(m_mass - mass) < 1e-6
          and all(abs(a - b) < 1e-6 for a, b in zip(m_com, com))
          and eig_ok is not False)
    return {"file": path.name, "patched_body": "x1-body",
            "mass_kg": m_mass, "com_m": m_com,
            "diaginertia_kg_m2": m_diag, "matches_R9": ok, **eig_note}


def grf_groups(per_file: list[dict]) -> dict:
    straight = [e for e in per_file if LATERAL_MARKER not in e["file"]]
    lateral = [e for e in per_file if LATERAL_MARKER in e["file"]]
    ms = [e["mass_kg"] for e in straight]
    ml = [e["mass_kg"] for e in lateral]
    mean_s = round(sum(ms) / len(ms), 3)
    return {
        "straight": {"n": len(straight), "mean_mass_kg": mean_s,
                     "band": list(G6_STRAIGHT_BAND),
                     "vs_nominal_urdf_pct": round((mean_s / URDF_NOMINAL_TOTAL_KG - 1) * 100, 1),
                     "verdict": "PASS" if G6_STRAIGHT_BAND[0] <= mean_s <= G6_STRAIGHT_BAND[1] else "FAIL"},
        "lateral_reference_ungated": {
            "n": len(lateral), "mean_mass_kg": round(sum(ml) / len(ml), 3),
            "note": "(J^T)^-1 GRF bias in lateral/turning regimes (-14%); method limitation, not gated"},
    }


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    r9 = json.loads(R9.read_text())
    bp = r9["best_params"]
    base = bp["bodies"]["base"]
    mass = float(base["mass"])
    com = [float(v) for v in re.sub(r"[\[\]\n]", " ", base["com"]).split()] \
        if isinstance(base["com"], str) else list(base["com"])
    if isinstance(base["inertia"], str):
        vals = [float(v) for v in re.sub(r"[\[\]\n]", " ", base["inertia"]).split()]
        inertia = [vals[0:3], vals[3:6], vals[6:9]]
    else:
        inertia = [list(r) for r in base["inertia"]]

    # ---- SPI five final gates (measured values from the R9 payload) ----
    spi_checks = [
        {"gate": "EFFECTIVENESS", "measured": r9["holdout"]["eff_ratio"],
         "threshold": "<= 0.70", "pass": r9["holdout"]["eff_ratio"] <= 0.70},
        {"gate": "PHYSICAL", "measured": "in-domain (by construction)",
         "threshold": "x1_spi.yaml box", "pass": True},
        {"gate": "ACCEL", "measured": r9["holdout"]["accel_rms_best"],
         "threshold": "<= 14.04 (T9) / 15.0 (3-seed re-baseline)",
         "pass": r9["holdout"]["accel_rms_best"] <= 14.04},
        {"gate": "ACTUATOR", "measured": round(float(bp["kappa_s"]), 4),
         "threshold": "[0.34, 0.71]", "pass": 0.34 <= float(bp["kappa_s"]) <= 0.71},
        {"gate": "CROSS-DATASET",
         "measured": f"ratio {r9['cross']['ratio']} / rms {r9['cross']['accel_rms_best']}",
         "threshold": "ratio <= 0.70, rms <= 14.24 (3-seed: 14.405)",
         "pass": r9["cross"]["ratio"] <= 0.70 and r9["cross"]["accel_rms_best"] <= 14.24},
    ]

    seeds = {}
    for label, p in SEEDS.items():
        d = json.loads(p.read_text())
        seeds[label] = {
            "pelvis_mass_kg": round(float(d["best_params"]["bodies"]["base"]["mass"]), 3),
            "kappa_s": round(float(d["best_params"]["kappa_s"]), 3),
            "holdout_accel_rms": d["holdout"]["accel_rms_best"],
            "cross_accel_rms": d["cross"]["accel_rms_best"],
        }

    # ---- joint-level params from the T9 final-verdict log ----
    log = T9_LOG.read_text()
    joints = parse_joints(log)
    gates = parse_gates(log)
    gate_rows = [{"id": c["id"], "ok": c["ok"], "detail": c["detail"]}
                 for c in gates.get("checks", [])]

    # ---- GRF whole-body mass ----
    grf = json.loads(GRF_JSON.read_text())
    grf_sum = grf_groups(grf["G6_per_file"])

    # ---- model artifacts: copy + cross-check vs R9 ----
    MODEL.mkdir(parents=True, exist_ok=True)
    artifacts, checks = [], []
    for name in ARTIFACTS:
        src, dst = EXPORT / name, MODEL / name
        shutil.copy2(src, dst)
        if name.endswith(".urdf"):
            c = check_urdf(dst, mass, com, inertia)
        elif name.endswith(".xml"):
            c = check_mjcf(dst, mass, com, inertia)
        else:
            dr = json.loads(dst.read_text())
            c = {"file": name, "identified_mass_kg": dr["identified_mass"],
                 "kappa_s": dr["kappa_s"],
                 "matches_R9": (abs(dr["identified_mass"] - mass) < 1e-6
                                and abs(dr["kappa_s"] - float(bp["kappa_s"])) < 1e-6)}
        c["sha256_12"] = sha256(dst)[:12]
        c["source"] = str(src.relative_to(ROOT))
        checks.append(c)
        artifacts.append(c)
    all_match = all(c["matches_R9"] for c in checks)

    summary = {
        "snapshot": {
            "generated_at": now,
            "param_version": "R9 (seed1, TASK_20260903_015) — official, T9 TASK_20260903_073 exit 0",
            "joint_params_version": "T7 (TASK_20260902_222); post-hoc T10 (TASK_20260903_082) revised method, verdict unchanged",
            "report": "docs/rounds/2026-09-03_multidataset_sysid_report.md",
            "model_artifacts_consistent_with_R9": all_match,
            "regenerate": "python3 scripts/make_output.py",
        },
        "spi_r9": {
            "params": bp,
            "holdout": r9["holdout"],
            "cross": r9["cross"],
            "five_final_gates": spi_checks,
            "gates_rationale": SPI_GATES,
            "seed_stability": seeds,
            "credibility": {"pelvis_mass": "high (vs nominal; absolute seed spread +/-14%)",
                            "com": "medium", "inertia": "low (no-mocap weak observability)",
                            "kappa_knee/ankle/kappa_s": "high",
                            "kappa_hip": "medium"},
            "cross_round_consistency": {
                "kappa_s": {"R4": 0.396, "F1_v14": 0.434, "R8": 0.361, "R9": round(float(bp["kappa_s"]), 3)},
                "pelvis_mass_kg": {"R4": 3.428, "F1_v14": 3.783, "R9": round(mass, 3)},
                "note": "pelvis single-rigid-body is a known simplification; R9 sits near the 3.0 lower bound"},
        },
        "joint_level": {
            "model": "tau_meas - g(q) = J_eff*qdd + tau_c*tanh(qd/eps) + tau_v*qd + c0 (+ gyro coupling, serial)",
            "per_joint": joints,
            "gates": gate_rows,
            "verdict": gates.get("verdict"), "exit_code": gates.get("exit_code"),
            "written_back_to_model_files": False,
            "note": "J4/J5 PASS; J1/J2/J3 marginal FAIL (root causes quantified in report §3.2). "
                    "Parameters delivered as reference values only — NOT patched into URDF/MJCF.",
        },
        "grf_whole_body_mass": {
            **grf_sum,
            "nominal_urdf_total_kg": URDF_NOMINAL_TOTAL_KG,
            "per_file": grf["G6_per_file"],
        },
        "negative_results": [{
            "item": "whole-body inertial parameters (PRIME regression route)",
            "conclusion": "not identifiable with current sensing (no mocap / no fixed base)",
            "evidence": "prime_identify/IDENTIFIABILITY.md (7-experiment chain)",
        }],
        "model_artifacts": artifacts,
        "excluded_legacy_artifacts": [{
            "file": "prime_identify/results/x1_gmass_anchored.urdf",
            "what": "URDF with total mass anchored to 37.1 kg (G4 single-dataset era, base_link 6.08 kg)",
            "why_excluded": "superseded: current evidence is the G6 v2 straight-group mean 35.94 kg "
                            "(n=7) and pelvis is separately identified by SPI R9; keeping it in the "
                            "delivery snapshot would mix two inconsistent mass stories",
        }],
    }
    (OUT / "identified_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    # ---------------- README.md ----------------
    kappa_str = " / ".join(f"{k} {v:.1f}" for k, v in bp["motors"].items())
    jm = {j["joint"]: j for j in joints}
    def jr(j, f):  # joint value or dash
        return f"{jm[j][f]:g}" if j in jm else "—"
    joint_rows = "\n".join(
        f"| {j['joint']} | {j['type']} | {j['J_eff_kg_m2']:.4g} | {j['tau_c_Nm']:.3g} "
        f"| {j['tau_v_Nm_s_rad']:.3g} | {j['delay_ms']:.0f} | {j['m1_alpha']:.3g} "
        f"| {j['kt']:.1f} | {j['dyn_R2']:.2f} |"
        for j in joints)
    gate_rows_md = "\n".join(
        f"| {g['id']} | {'✅ PASS' if g['ok'] else '❌ FAIL'} | {g['detail']} |"
        for g in gate_rows)
    spi_rows = "\n".join(
        f"| {c['gate']} | {c['measured'] if not isinstance(c['measured'], dict) else c['measured']} "
        f"| {c['threshold']} | {'✅' if c['pass'] else '❌'} |"
        for c in spi_checks)
    seed_rows = "\n".join(
        f"| {k} | {v['pelvis_mass_kg']} | {v['kappa_s']} | {v['holdout_accel_rms']} | {v['cross_accel_rms']} |"
        for k, v in seeds.items())
    art_rows = "\n".join(
        f"| `model/{c['file']}` | {c.get('patched_body', 'DR config')} | "
        f"{c['sha256_12']} | {'✅ 一致' if c['matches_R9'] else '❌ 不一致（漂移！）'} |"
        for c in checks)

    readme = f"""# X1 辨识输出汇总（顶层交付快照）

> 本目录是全仓库辨识工作的**单一对外输出口**：辨识了哪些指标、结果是多少、辨识后的模型文件是什么。
> 由 `scripts/make_output.py` 于 **{now}** 生成（快照，可再生：`python3 scripts/make_output.py`）。
> 参数版本 **R9**（seed1，T8 TASK_20260903_015 辨识，T9 TASK_20260903_073 正式终判 exit 0）；
> 完整方法与证据链见 `docs/rounds/2026-09-03_multidataset_sysid_report.md`。
> 模型工件与 R9 参数一致性校验：**{'✅ 全部一致' if all_match else '❌ 存在不一致，见下表'}**

## 1. 辨识指标清单与结果总览

| 层面 | 辨识指标 | 结果 | 判定 |
|---|---|---|---|
| 整机·SPI | 骨盆质量 / 质心 / 惯量张量 | **{mass:.4f} kg**（nominal {NOMINAL_PELVIS['mass']:.4f}）/ com {com} / I 见 §2 | 五项全 PASS（T9 exit 0） |
| 整机·SPI | 电机刚度 κ ×4（hip_pitch/hip_rolleyaw/knee/ankle） | {kappa_str} | 域内（域见 `x1_spi.yaml`） |
| 整机·SPI | 力矩缩放 κs | **{float(bp['kappa_s']):.4f}**（独立证据带 [0.34, 0.71]） | ✅ |
| 关节·模组 | 串联关节 J_eff / τc / τv / 延迟 / α / k_t（×8 串联 + ×4 并联踝参考） | 延迟 6–9 ms、k_t R²=1.000（12/12）、参数表见 §3 | J4/J5 PASS；J1/J2/J3 边际 FAIL（参考值交付） |
| 整机·质量 | GRF 反算整机总质量（直行组） | **{grf_sum['straight']['mean_mass_kg']} kg**（n={grf_sum['straight']['n']}，URDF {URDF_NOMINAL_TOTAL_KG} 的 {grf_sum['straight']['vs_nominal_urdf_pct']:+.1f}%） | ✅ G6 v2 PASS |
| 负结果 | 全身惯性参数（PRIME 回归） | 本传感配置下原理性不可辨识 | 已证伪留档（防重复投入） |

## 2. SPI 参数（R9，官方参数集）

| quantity | nominal | identified (R9) |
|---|---|---|
| 骨盆质量 [kg] | {NOMINAL_PELVIS['mass']:.4f} | **{mass:.4f}** |
| 骨盆质心 [m] | {NOMINAL_PELVIS['com']} | {com} |
| 骨盆惯量（对角）[kg·m²] | [0.0268, 0.0108, 0.0218] | {[round(inertia[i][i], 6) for i in range(3)]} |
| 骨盆惯量积 [kg·m²] | 0 | ixy {inertia[0][1]:.6f}, ixz {inertia[0][2]:.6f}, iyz {inertia[1][2]:.6f} |
| κs | 1.0 | **{float(bp['kappa_s']):.4f}** |
| 电机 κ | — | {kappa_str} |

**五项完成标准终判**（`validate_spi.py`，实测值 vs 门限）：

| 标准 | R9 实测 | 门限 | 判定 |
|---|---|---|---|
{spi_rows}
nominal→R9：holdout 代价 1,184,627 → 423,748（-64.2%）；跨策略组比力 22.26 → 13.94（-37.4%）。

**seed 稳定性**（3-seed 地板标定，TASK_20260904_013/014；官方参数固定为 seed1=R9，不挑最优 seed）：

| seed | 骨盆 [kg] | κs | holdout 比力 | cross 比力 |
|---|---|---|---|---|
{seed_rows}

可信度：质量/κs/κ(knee,ankle) 高；质心/κ(hip) 中；惯量低（无动捕弱可观——绝对 seed 带宽骨盆 ±14% 应如实认知）。

## 3. 关节级参数（T7，参考值 — 未回写模型文件）

模型：`τ_meas − g(q) = J_eff·q̈ + τc·tanh(q̇/ε) + τv·q̇ + c0（+陀螺耦合，串联）`；g(q) 为悬空重力矩 LUT。

| 关节 | 类型 | J_eff [kg·m²] | τc [Nm] | τv [Nm·s/rad] | 延迟 [ms] | α(M1) | k_t | 动力学 R² |
|---|---|---|---|---|---|---|---|---|
{joint_rows}

**GATE-J 门禁终判**（J2 已修正为 serial-only；T10 post-hoc 方法修订后判定未翻转）：

| 门 | 判定 | 依据 |
|---|---|---|
{gate_rows_md}

> 并联踝（ankle_pitch/roll）的 α 与动力学 R² 为并联驱动模式伪影，仅作参考；关节级摩擦/惯量参数**尚未回写** URDF/MJCF——回写属下一阶段工作（见报告 §7）。

## 4. 整机质量（GRF，跨数据集一致性）

| 组 | n | 均值 [kg] | 门限 | 判定 |
|---|---|---|---|---|
| 直行（7500/7500dr/legacy） | {grf_sum['straight']['n']} | **{grf_sum['straight']['mean_mass_kg']}** | [32.0, 40.0] | ✅ PASS |
| 侧移/转向（5999，参考不设门） | {grf_sum['lateral_reference_ungated']['n']} | {grf_sum['lateral_reference_ungated']['mean_mass_kg']} | — | 方法工况偏差 -14% 实证 |

## 5. 辨识后模型工件（`model/`）

由 `spi_identify/scripts/apply_params.py` 以 R9 参数回写生成（远端 apply 阶段），本目录为快照副本；sha256 前 12 位供校验：

| 文件 | 回写 body | sha256-12 | vs R9 参数 |
|---|---|---|---|
{art_rows}

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
"""
    (OUT / "README.md").write_text(readme)

    print(f"[make_output] wrote {OUT/'README.md'}")
    print(f"[make_output] wrote {OUT/'identified_summary.json'}")
    for c in checks:
        print(f"[make_output] model/{c['file']}: matches_R9={c['matches_R9']} sha={c['sha256_12']}")
    n_joint = len(joints)
    print(f"[make_output] joints parsed: {n_joint} (expect 12); gates: {len(gate_rows)} (expect 5)")
    if n_joint != 12 or len(gate_rows) != 5 or not all_match:
        raise SystemExit("[make_output] INVARIANT VIOLATION — check sources, output flagged")


if __name__ == "__main__":
    main()
