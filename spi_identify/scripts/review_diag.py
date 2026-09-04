#!/usr/bin/env python3
"""SPI 辨识结果事后复算诊断（review diagnostics）— 本地、纯 numpy、无仿真依赖。

用途：为 docs/rounds/2026-09-04_sysid_review.md 的每一个数字提供可复现来源。
本脚本**不产生门禁判定**（门禁判定只能来自 validate_spi.py / validate_joint.py，
见 docs/rounds/README.md 使用规则 4）；这里的一切输出都是 *参考性* 事后复算，
用于回答"已通过的门禁到底测量了什么"。

复算项：
  R-1a  比力通道零模型基线：把 a_pred ≡ [0,0,g] 当作"模型"，在 train/cross
        两桶上算与 validate_spi.py 同定义的 3 轴 RMS 残差，作为 ACCEL 门槛
        (15.0 m/s^2) 的参照尺度。
  R-1b  从 T9 终判日志的 per-signal 代价反推物理量（姿态角误差、关节角误差、
        关节速度误差），判断开环回放在 clip 内是否已经发散。
  R-2   三 seed 参数简并分析：m / kappa_s / (m/kappa_s) 的离散度对比。
  R-3   物理自洽性：辨识质量与惯量的变化方向 + 惯量特征值三角不等式余量。
  R-4   kappa_s 证据带（ACTUATOR 门禁）的敏感性：全关节带 vs 按回归质量
        (R^2) 筛选后的带。

用法（仓库根目录）：
    python3 spi_identify/scripts/review_diag.py
    python3 spi_identify/scripts/review_diag.py --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "spi_identify/configs/x1_spi.yaml"
T9_LOG = REPO / "spi_identify/results/remote_logs/T9_TASK_20260903_073_final_verdict.log"
M1_JSON = REPO / "data/derived/step_m1_regression_all.json"
SEED_PARAMS = [
    ("seed1 (R9, official)", REPO / "spi_identify/results/r9_indomain_params.json"),
    ("seed2", REPO / "spi_identify/results/seed2_indomain_params.json"),
    ("seed3", REPO / "spi_identify/results/seed3_indomain_params.json"),
]
G = 9.81
# 名义骨盆（MJCF x1-body，见 x1_spi.yaml bodies[0].nominal）
NOMINAL_MASS = 4.3041648
NOMINAL_INERTIA = np.array([[0.02680559, -5.49e-06, 5.389e-05],
                            [-5.49e-06, 0.01083128, -0.00011229],
                            [5.389e-05, -0.00011229, 0.02180955]])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def box_filter(a: np.ndarray, win: int) -> np.ndarray:
    """与 spi/cost.py::box_filter 同定义（edge padding 的滑动平均）。"""
    win = int(win)
    if win <= 1:
        return a
    a = np.asarray(a, dtype=float)
    n, pad = a.shape[0], win // 2
    k = np.ones(win) / win
    out = np.empty_like(a)
    for i in range(a.shape[1]):
        p = np.pad(a[:, i], (pad, pad), mode="edge")
        y = np.convolve(p, k, mode="same")
        out[:, i] = y[pad:pad + n]
    return out


def rms3(d: np.ndarray) -> float:
    """与 validate_spi.py::accel_rms 同定义：sqrt(sum_t ||d_t||^2 / n)，3 轴合并。"""
    return float(np.sqrt(np.sum(d ** 2) / len(d)))


def parse_matrix(s) -> np.ndarray:
    """结果 json 里 com/inertia 被存成 numpy repr 字符串，解析回数组。"""
    if isinstance(s, (list, tuple)):
        return np.asarray(s, dtype=float)
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(s))]
    return np.asarray(v, dtype=float)


def load_cfg() -> dict:
    import yaml  # 只在需要读桶划分时才用
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def extract_json_blocks(text: str) -> List[dict]:
    """抽出日志里所有行首为 '{' 的顶层平衡 JSON 块。"""
    out = []
    for m in re.finditer(r"^\{", text, flags=re.M):
        depth, i, in_str, esc = 0, m.start(), False, False
        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        out.append(json.loads(text[m.start():i + 1]))
                    except json.JSONDecodeError:
                        pass
                    break
            i += 1
    return out


# ---------------------------------------------------------------------------
# R-1a  比力通道零模型基线
# ---------------------------------------------------------------------------

def r1a_null_baseline(win: int) -> dict:
    cfg = load_cfg()
    buckets: Dict[str, List[Path]] = {"train": [], "cross": []}
    for src in cfg["data"]["sources"]:
        buckets[src["role"]].extend(REPO / f for f in src["files"])

    res = {"filter_win": win, "buckets": {}, "per_file": []}
    for role, files in buckets.items():
        acc = []
        for f in files:
            with open(f, newline="") as fh:
                rows = list(csv.DictReader(fh))
            a = np.array([[float(r[f"imu_accel_{c}"]) for c in "xyz"] for r in rows])
            af = box_filter(a, win)
            res["per_file"].append({
                "file": str(f.relative_to(REPO)), "role": role, "n": len(af),
                "signal_rms": round(rms3(af), 3),
                "null_gravity_rms": round(rms3(af - np.array([0.0, 0.0, G])), 3),
                "null_mean_rms": round(rms3(af - af.mean(0)), 3),
            })
            acc.append(af)
        A = np.vstack(acc)
        res["buckets"][role] = {
            "n_files": len(files), "n_steps": int(len(A)),
            "signal_rms": round(rms3(A), 3),
            "null_gravity_rms": round(rms3(A - np.array([0.0, 0.0, G])), 3),
            "null_mean_rms": round(rms3(A - A.mean(0)), 3),
        }
    return res


# ---------------------------------------------------------------------------
# R-1b  从 T9 代价分解反推物理量
# ---------------------------------------------------------------------------

def r1b_cost_inversion() -> Optional[dict]:
    if not T9_LOG.exists():
        return None
    blocks = [b for b in extract_json_blocks(T9_LOG.read_text()) if "accel_rms_val" in b]
    if not blocks:
        return None
    v = blocks[-1]
    w = {"quat": 2.0, "angvel": 0.25, "accel": 1.0, "q": 3.0, "qd": 0.1, "tau": 0.002}
    val = v["costs"]["val"]
    # n_val_steps 由 accel 项与已报 RMS 反推：rms = sqrt(term/(w*n))
    n = int(round(val["best"]["accel"] / (w["accel"] * v["accel_rms_val"] ** 2)))

    def inv(which: str) -> dict:
        c = val[which]
        # 平均姿态误差：1-<q,qr>^2 = sin^2(theta/2)
        s2 = c["quat"] / w["quat"] / n
        theta = 2.0 * math.degrees(math.asin(min(1.0, math.sqrt(max(s2, 0.0)))))
        return {
            "total_cost": round(sum(c.values()), 1),
            "quat_mean_err_deg": round(theta, 1),
            "q_rms_rad_alljoints": round(math.sqrt(c["q"] / w["q"] / n), 3),
            "q_rms_rad_perjoint": round(math.sqrt(c["q"] / w["q"] / n / 12.0), 3),
            "qd_rms_alljoints": round(math.sqrt(c["qd"] / w["qd"] / n), 2),
            "accel_rms": round(math.sqrt(c["accel"] / w["accel"] / n), 3),
        }

    nom, best = inv("nominal"), inv("best")
    share = {k: round(val["best"][k] / sum(val["best"].values()) * 100, 1) for k in val["best"]}
    delta = {k: round((val["best"][k] / val["nominal"][k] - 1) * 100, 1) for k in val["best"]}
    return {"n_val_steps": n, "weights": w, "nominal": nom, "best": best,
            "raw_costs": val, "best_cost_share_pct": share, "best_vs_nominal_pct": delta,
            "cross": v.get("cross_dataset", {})}


# ---------------------------------------------------------------------------
# R-2 / R-3  seed 简并 + 物理自洽性
# ---------------------------------------------------------------------------

def r2_r3_seeds() -> dict:
    rows = []
    for name, p in SEED_PARAMS:
        d = json.loads(p.read_text())
        bp = d["best_params"]
        b = bp["bodies"]["base"]
        I = parse_matrix(b["inertia"]).reshape(3, 3)
        lam = np.linalg.eigvalsh(0.5 * (I + I.T))
        rows.append({
            "run": name, "file": str(p.relative_to(REPO)),
            "mass": b["mass"], "kappa_s": bp["kappa_s"],
            "m_over_ks": b["mass"] / bp["kappa_s"],
            "com": parse_matrix(b["com"]).tolist(),
            "inertia_eig": lam.tolist(),
            "motors": bp["motors"],
            "holdout_accel": d.get("holdout", {}).get("accel_rms_best"),
            "cross_accel": d.get("cross", {}).get("accel_rms_best"),
            "holdout_eff": d.get("holdout", {}).get("eff_ratio"),
            # 物理可实现性的真实条件（log-Cholesky 应保证，此处显式核对）
            "triangle_margin_pct": (lam[0] + lam[1] - lam[2]) / lam[2] * 100,
        })

    def spread(vals):
        vals = np.asarray(vals, dtype=float)
        return {"mean": round(float(vals.mean()), 4),
                "range_pct": round(float((vals.max() - vals.min()) / vals.mean() * 100), 1),
                "cv_pct": round(float(vals.std() / vals.mean() * 100), 1)}

    keys = {"mass": [r["mass"] for r in rows], "kappa_s": [r["kappa_s"] for r in rows],
            "m_over_ks": [r["m_over_ks"] for r in rows]}
    for g in ("hip_pitch", "hip_rolleyaw", "knee", "ankle"):
        keys[f"kappa.{g}"] = [r["motors"][g] for r in rows]
    degeneracy = {k: spread(v) for k, v in keys.items()}

    lam_nom = np.linalg.eigvalsh(NOMINAL_INERTIA)
    physical = {
        "nominal_mass": NOMINAL_MASS,
        "nominal_inertia_eig": lam_nom.tolist(),
        "per_seed": [{
            "run": r["run"],
            "mass_vs_nominal_pct": round((r["mass"] / NOMINAL_MASS - 1) * 100, 1),
            "inertia_eig_ratio_vs_nominal": [round(float(a) / float(b), 2)
                                             for a, b in zip(r["inertia_eig"], lam_nom)],
            "triangle_margin_pct": round(r["triangle_margin_pct"], 1),
        } for r in rows],
    }
    return {"rows": rows, "degeneracy": degeneracy, "physical_consistency": physical}


# ---------------------------------------------------------------------------
# R-4  ACTUATOR 证据带敏感性
# ---------------------------------------------------------------------------

def r4_alpha_band(r2_thresholds=(0.0, 0.80, 0.84)) -> dict:
    d = json.loads(M1_JSON.read_text())
    serial = []
    for k, v in sorted(d.items()):
        if not isinstance(v, dict) or "alpha" not in v:
            continue
        if "ankle" in k:      # 并联踝：alpha 非同一语义，不入带（与 J2/J3 设计一致）
            continue
        serial.append({"joint": k, "alpha": v["alpha"], "R2": v["R2"],
                       "tau_p98": v.get("tau_p98"), "gyro_rms": v.get("gyro_rms")})
    bands = {}
    for th in r2_thresholds:
        sub = [s for s in serial if s["R2"] >= th]
        if not sub:
            continue
        a = [s["alpha"] for s in sub]
        bands[f"R2>={th:.2f}"] = {
            "n_joints": len(sub), "band": [round(min(a), 3), round(max(a), 3)],
            "joints": [s["joint"].replace("_joint", "") for s in sub],
        }
    ks = {name: json.loads(p.read_text())["best_params"]["kappa_s"] for name, p in SEED_PARAMS}
    verdicts = {}
    for label, b in bands.items():
        lo, hi = b["band"]
        verdicts[label] = {n: ("in" if lo <= v <= hi else "OUT") for n, v in ks.items()}
    return {"serial_joints": serial, "bands": bands,
            "gate_band_in_use": [0.34, 0.71], "kappa_s": ks,
            "kappa_s_vs_band": verdicts}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter-win", type=int, default=20,
                    help="比力盒滤波窗口，须与 x1_spi.yaml cost.accel_filter_win 一致（默认 20）")
    ap.add_argument("--json", type=str, default=None, help="把完整结果写入 JSON")
    args = ap.parse_args()

    out: Dict[str, object] = {}
    print("=" * 78)
    print("SPI 辨识事后复算诊断 — 参考性结论，非门禁判定")
    print("=" * 78)

    # ---- R-1a ----
    r1a = r1a_null_baseline(args.filter_win)
    out["R1a_null_baseline"] = r1a
    print(f"\n[R-1a] 比力通道零模型基线（box win={args.filter_win}，与 cost.py 同定义）")
    print(f"  {'桶':<8}{'文件':>4}{'步数':>8}{'信号RMS':>10}{'零模型[0,0,g]':>14}{'零模型=均值':>12}")
    for role, b in r1a["buckets"].items():
        print(f"  {role:<8}{b['n_files']:>4}{b['n_steps']:>8}{b['signal_rms']:>10.2f}"
              f"{b['null_gravity_rms']:>14.2f}{b['null_mean_rms']:>12.2f}")
    print("  参照：ACCEL 门限 15.0 / R9 holdout 13.541 / R9 cross 13.940 m/s^2")

    # ---- R-1b ----
    r1b = r1b_cost_inversion()
    out["R1b_cost_inversion"] = r1b
    if r1b:
        print(f"\n[R-1b] T9 终判 holdout 代价反推（n_val_steps={r1b['n_val_steps']}）")
        print(f"  {'量':<26}{'nominal':>12}{'R9(best)':>12}")
        for key, label in [("quat_mean_err_deg", "基座姿态平均误差 [deg]"),
                           ("q_rms_rad_perjoint", "关节角误差 RMS [rad/关节]"),
                           ("qd_rms_alljoints", "关节速度误差 RMS [rad/s]"),
                           ("accel_rms", "比力误差 RMS [m/s^2]")]:
            print(f"  {label:<26}{r1b['nominal'][key]:>12}{r1b['best'][key]:>12}")
        print("  各通道 best vs nominal 变化 %: " +
              ", ".join(f"{k} {v:+.1f}" for k, v in r1b["best_vs_nominal_pct"].items()))
        print("  best 代价构成 %:            " +
              ", ".join(f"{k} {v}" for k, v in r1b["best_cost_share_pct"].items()))
    else:
        print("\n[R-1b] 跳过：未找到 T9 日志或其中的 validation JSON 块")

    # ---- R-2 / R-3 ----
    r23 = r2_r3_seeds()
    out["R2_R3_seed_degeneracy"] = r23
    print("\n[R-2] 三 seed 参数简并")
    print(f"  {'run':<22}{'m[kg]':>8}{'kappa_s':>9}{'m/ks':>8}{'holdout':>9}{'cross':>8}")
    for r in r23["rows"]:
        print(f"  {r['run']:<22}{r['mass']:>8.3f}{r['kappa_s']:>9.3f}{r['m_over_ks']:>8.2f}"
              f"{(r['holdout_accel'] or 0):>9.3f}{(r['cross_accel'] or 0):>8.3f}")
    print(f"  {'量':<22}{'均值':>8}{'极差%':>9}{'CV%':>8}")
    for k, v in r23["degeneracy"].items():
        print(f"  {k:<22}{v['mean']:>8.3f}{v['range_pct']:>9.1f}{v['cv_pct']:>8.1f}")
    print("  读法：m 与 kappa_s 各自离散而 m/kappa_s 稳定 => 数据约束的是比值而非各自")

    print("\n[R-3] 物理自洽性（质量与惯量的变化方向 + 三角不等式余量）")
    print(f"  {'run':<22}{'m vs nom %':>12}{'I 特征值/名义':>24}{'三角余量%':>12}")
    for p in r23["physical_consistency"]["per_seed"]:
        ratio = " / ".join(f"{x:.1f}x" for x in p["inertia_eig_ratio_vs_nominal"])
        print(f"  {p['run']:<22}{p['mass_vs_nominal_pct']:>+12.1f}"
              f"{ratio:>24}{p['triangle_margin_pct']:>12.1f}")
    print("  读法：质量下降同时惯量放大数倍，无法对应任何真实刚体变化 => 参数在吸收模型误差")

    # ---- R-4 ----
    r4 = r4_alpha_band()
    out["R4_actuator_band"] = r4
    print("\n[R-4] ACTUATOR 证据带敏感性（串联关节 M1 回归）")
    print(f"  {'关节':<26}{'alpha':>8}{'R2':>7}{'tau_p98':>9}{'gyro_rms':>10}")
    for s in r4["serial_joints"]:
        print(f"  {s['joint']:<26}{s['alpha']:>8.3f}{s['R2']:>7.3f}"
              f"{(s['tau_p98'] or 0):>9.2f}{(s['gyro_rms'] or 0):>10.3f}")
    print(f"  门禁在用带: {r4['gate_band_in_use']}")
    for label, b in r4["bands"].items():
        v = r4["kappa_s_vs_band"][label]
        print(f"  {label:<10} n={b['n_joints']} band={b['band']}  -> " +
              ", ".join(f"{n.split(' ')[0]}={r4['kappa_s'][n]:.3f}:{s}" for n, s in v.items()))
    print("  读法：带下沿由 R2 最低、力矩最小的 hip_yaw 单独撑起；"
          "按回归质量筛选后三个 seed 的 kappa_s 全部落在带外")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"\n[写出] {args.json}")


if __name__ == "__main__":
    main()
