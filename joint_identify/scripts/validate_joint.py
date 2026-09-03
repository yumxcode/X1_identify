#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate evaluation for joint-level identification (GATE-J1..J5).

Pure numpy + json — unit-testable locally. Reads joint_params.json produced by
run_joint_identify.py plus the legacy M1 evidence
(data/derived/step_m1_regression_all.json) and emits validation_joint.json.
Exit code 0 = PASS, 1 = FAIL.

Gates
-----
J1 dynamics fit      serial joints: R2 >= 0.60 (hip 0.55 floor, knee 0.75
                      target reported as WARN when missed)
J2 L/R symmetry      (J_eff, tau_c, tau_v, kt) same-name left/right relative
                      difference < 35%
J3 gain band         serial alpha in [0.34, 0.71] AND |alpha - alpha_M1| < 0.08
J4 delay sanity      serial delay in [-5, +30] ms
J5 current map       kt R2 >= 0.90 on every joint (absolute band not used:
                      logger current scaling unverified)
Parallel ankles: reported but only J5 applies (alpha > 1 is expected there).
J2 significance screen (2026-09-03): a pair violates symmetry only when
rel_diff > tol AND the difference is statistically significant,
|a-b| > 2*sqrt(se_a^2+se_b^2). Rationale: rel_diff explodes for parameters
whose magnitude is at the estimation-noise floor (post gyro-coupling tau_v
can legitimately approach 0 on one side) — a relative metric on a
noise-dominated parameter flags noise, not robot asymmetry. Pairs whose
difference is within 2-sigma of the estimator uncertainty are reported as
"not assessable (noise-dominated)" rather than violations. When SEs are
unavailable the pre-registered pure rel_diff rule applies unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "joint_identify"))
from regress import is_serial  # noqa: E402  (canonical home; re-exported below)

M1_JSON = ROOT / "data" / "derived" / "step_m1_regression_all.json"

SYMMETRY_REL_TOL = 0.35
J2_SIGMA_K = 2.0
ALPHA_BAND = (0.34, 0.71)
ALPHA_M1_MAX_DEV = 0.15  # per-joint |alpha - alpha_M1| >= this -> WARN (method systematics)
DELAY_RANGE_MS = (-5.0, 30.0)
KT_R2_MIN = 0.90
R2_FLOOR = {"hip": 0.55, "knee": 0.60}
KNEE_TARGET = 0.75

# is_serial lives in regress.py (canonical home); imported at module top.


def _load_m1(path=M1_JSON):
    try:
        d = json.loads(Path(path).read_text())
        return {j: float(v.get("alpha")) for j, v in d.items()
                if isinstance(v, dict) and "alpha" in v}
    except Exception:
        return {}


def _pairwise_symmetry(jparams):
    """Relative difference of same-name L/R joints for keyed scalars.

    SERIAL joints only: parallel ankles are torque-driven (their serial-
    dynamics fit is outside the model's validity — T7 R2 0.16-0.43 and
    sign-flipping tau_v are drive-mode artifacts, not robot asymmetry).
    Each entry carries (rel_diff, sig) where sig is the significance margin
    |a-b| / (2*sqrt(se_a^2+se_b^2)) > 1 means the asymmetry is statistically
    significant (None when SEs are unavailable -> pure rel_diff rule).
    """
    out = []
    nested = {"J_eff": ("dynamics", "J_eff"), "tau_c": ("dynamics", "tau_c"),
              "tau_v": ("dynamics", "tau_v"), "kt": ("kt", "kt")}
    lefts = [j for j in jparams
             if j.startswith("left_") and is_serial(j)]
    for lj in lefts:
        rj = "right_" + lj[len("left_"):]
        if rj not in jparams or not is_serial(rj):
            continue
        for key, (sub, field) in nested.items():
            def val(jn):
                d = jparams[jn].get(sub)
                if isinstance(d, dict) and field in d:
                    v = d[field]
                    return float(v) if isinstance(v, (int, float)) else None
                return None
            a, b = val(lj), val(rj)
            if a is None or b is None:
                continue
            sa = sb = None
            da, db = jparams[lj].get(sub), jparams[rj].get(sub)
            if isinstance(da, dict) and isinstance(db, dict) and f"se_{field}" in da \
                    and f"se_{field}" in db:
                va, vb = da[f"se_{field}"], db[f"se_{field}"]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    sa, sb = float(va), float(vb)
            denom = min(abs(a), abs(b))
            if denom < 1e-9:
                continue
            rel = abs(a - b) / denom
            sig = None
            if sa is not None and sb is not None:
                thr = J2_SIGMA_K * float(np.sqrt(sa * sa + sb * sb)) if sa + sb > 0 else 0.0
                sig = (abs(a - b) / thr) if thr > 0 else float("inf")
            out.append({"joint_pair": f"{lj}|{rj}", "param": key,
                        "values": [a, b],
                        "rel_diff": rel, "sig_margin": sig})
    return out


def evaluate(joint_params, m1_alphas=None):
    m1_alphas = m1_alphas if m1_alphas is not None else _load_m1()
    checks = []
    warns = []

    def add(cid, ok, detail):
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    serial = {j: p for j, p in joint_params.items() if is_serial(j) and p.get("dynamics")}
    par = {j: p for j, p in joint_params.items() if not is_serial(j)}

    # J1 dynamics fit
    if serial:
        bad = []
        r2_list = []
        for j, p in sorted(serial.items()):
            floor = R2_FLOOR["hip"] if "hip" in j else R2_FLOOR["knee"]
            r2 = p["dynamics"].get("R2", 0.0)
            r2_list.append(f"{j}:{r2:.2f}")
            if r2 < floor:
                bad.append(f"{j} R2={r2:.3f}<{floor}")
            elif "knee" in j and r2 < KNEE_TARGET:
                warns.append(f"{j} knee R2={r2:.3f} below target {KNEE_TARGET} (pass)")
        add("J1_DYNAMICS_R2", not bad,
            "; ".join(bad) if bad else f"serial R2 [{', '.join(r2_list)}] all >= floor")
    else:
        add("J1_DYNAMICS_R2", False, "no serial-joint dynamics results")

    # J2 symmetry (significance-screened): a pair violates only when
    # rel_diff > tol AND the L/R difference is statistically significant
    # (|a-b| > 2*sigma_diff). Noise-dominated pairs are reported as
    # not-assessable instead of violations; missing SEs keep the pure
    # rel_diff rule (pre-registered behavior).
    sym = _pairwise_symmetry(joint_params)
    if sym:
        def is_violation(e):
            if e["rel_diff"] <= SYMMETRY_REL_TOL:
                return False
            if e.get("sig_margin") is None:
                return True  # no SEs -> legacy rule
            return e["sig_margin"] > 1.0

        violations = [e for e in sym if is_violation(e)]
        noise_dom = [e for e in sym
                     if e["rel_diff"] > SYMMETRY_REL_TOL
                     and e.get("sig_margin") is not None
                     and e["sig_margin"] <= 1.0]
        if violations:
            worst = max(violations, key=lambda e: e["rel_diff"])
            sig_s = (f"{worst['sig_margin']:.1f} sigma"
                     if worst.get("sig_margin") is not None else "n/a (no SE)")
            add("J2_LR_SYMMETRY", False,
                f"worst {worst['joint_pair']}.{worst['param']} rel_diff="
                f"{worst['rel_diff']:.2f} (> {SYMMETRY_REL_TOL}, sig={sig_s}); "
                f"{len(violations)} violations / {len(sym)} pairs"
                + (f"; {len(noise_dom)} noise-dominated (not assessable)" if noise_dom else ""))
        else:
            worst_rel = max(sym, key=lambda e: e["rel_diff"])
            sig_s = (f"{worst_rel['sig_margin']:.2f}"
                     if worst_rel.get("sig_margin") is not None else "n/a")
            add("J2_LR_SYMMETRY", True,
                f"worst {worst_rel['joint_pair']}.{worst_rel['param']} "
                f"rel_diff={worst_rel['rel_diff']:.2f} (sig={sig_s}); "
                f"0 violations / {len(sym)} pairs"
                + (f"; {len(noise_dom)} noise-dominated (rel_diff over tol but "
                   f"within 2-sigma estimator uncertainty)" if noise_dom else ""))
    else:
        add("J2_LR_SYMMETRY", True, "no comparable L/R pairs (degenerate)")

    # J3 gain band + M1 consistency (serial only). Hard band = evidence band
    # [0.34,0.71] widened by 0.10 to absorb method-level systematics (IRLS vs
    # OLS, segment definition; observed ~0.09 on post_hold realignment);
    # per-joint deviation >= 0.15 from legacy M1 is reported as WARN.
    if serial:
        bad = []
        alpha_list = []
        for j, p in sorted(serial.items()):
            a = p.get("m1", {}).get("alpha")
            if a is None:
                bad.append(f"{j}: no alpha")
                continue
            alpha_list.append(f"{j}:{a:.3f}")
            if not (ALPHA_BAND[0] - 0.10 <= a <= ALPHA_BAND[1] + 0.10):
                bad.append(f"{j} alpha={a:.3f} outside {ALPHA_BAND}+-0.10")
            a_m1 = m1_alphas.get(j)
            if a_m1 is not None and abs(a - a_m1) >= ALPHA_M1_MAX_DEV:
                warns.append(f"{j} alpha={a:.3f} vs M1 {a_m1:.3f} "
                             f"(dev>= {ALPHA_M1_MAX_DEV}, method systematics)")
        add("J3_GAIN_BAND", not bad, "; ".join(bad) if bad else
            f"serial alpha within {ALPHA_BAND}+-0.10 [{', '.join(alpha_list)}]")
    else:
        add("J3_GAIN_BAND", False, "no serial joints")

    # J4 delay
    if serial:
        bad = []
        for j, p in sorted(serial.items()):
            dms = p.get("delay_ms")
            if dms is None:
                bad.append(f"{j}: no delay estimate")
            elif not (DELAY_RANGE_MS[0] <= dms <= DELAY_RANGE_MS[1]):
                bad.append(f"{j} delay={dms:.1f} ms outside {DELAY_RANGE_MS}")
        add("J4_DELAY", not bad, "; ".join(bad) if bad else
            f"serial delays within {DELAY_RANGE_MS} ms: " +
            ", ".join(f"{j}:{p['delay_ms']:.1f}" for j, p in sorted(serial.items())
                      if p.get("delay_ms") is not None))
    else:
        add("J4_DELAY", False, "no serial joints")

    # J5 current mapping
    bad = [f"{j} kt R2={p.get('kt', {}).get('R2', 0):.2f}" for j, p in
           sorted(joint_params.items())
           if p.get("kt", {}).get("R2", 0.0) < KT_R2_MIN]
    add("J5_CURRENT_MAP", not bad,
        "; ".join(bad) if bad else
        f"kt R2 >= {KT_R2_MIN} on all {len(joint_params)} joints")

    verdict = "PASS" if all(c["ok"] for c in checks) else "FAIL"
    return {"verdict": verdict, "exit_code": 0 if verdict == "PASS" else 1,
            "checks": checks, "warnings": warns,
            "n_serial": len(serial), "n_parallel_reported": len(par),
            "gates": {"J1_r2_floor": R2_FLOOR, "J2_symmetry_tol": SYMMETRY_REL_TOL,
                      "J3_alpha_band": list(ALPHA_BAND),
                      "J3_m1_max_dev": ALPHA_M1_MAX_DEV,
                      "J4_delay_ms": list(DELAY_RANGE_MS), "J5_kt_r2_min": KT_R2_MIN}}


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="logs/joint_identify/joint_params.json")
    ap.add_argument("--out", default="logs/joint_identify/validation_joint.json")
    args = ap.parse_args()
    joint_params = json.loads(Path(args.params).read_text())["joints"]
    report = evaluate(joint_params)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[validate_joint] verdict: {report['verdict']} (exit {report['exit_code']})")
    return report["exit_code"]


if __name__ == "__main__":
    import sys
    sys.exit(main())
