#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint-level system identification on the 12 suspended single-joint step
logs (1 kHz): dynamics (J_eff, Coulomb/viscous friction), actuation delay,
M1 gain reproduction, torque-current mapping.

Run (gradmotion image with mujoco, or any machine with numpy+mujoco):
  gm-run X1_identify/joint_identify/scripts/run_joint_identify.py
Artifacts (gradmotion SDK pickup):
  logs/joint_identify/gm_play/joint_params.{json,pt}
  logs/joint_identify/validation_joint.json
Exit code = gate verdict (0 PASS / 1 FAIL).

Column semantics are VERIFIED at runtime (detect_feedback_semantics) instead
of assumed: target_pos/target_vel/target_effort are measured feedback,
pos_target_rad is the step command. Evidence is printed per file.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# repo root on sys.path so `import joint_identify.*` resolves under gm-run
# (cwd may differ from the repo root on the platform)
sys.path.insert(0, str(ROOT))

from joint_identify.gravity_torque import (build_gravity_lut, gravity_lookup,  # noqa: E402
                                            mujoco_available, zero_lut)
from joint_identify.regress import (detect_feedback_semantics, fit_dynamics,  # noqa: E402
                                    fit_kt, fit_m1, is_serial, savgol_ddq,
                                    xcorr_delay)

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = Path("logs/joint_identify")
GM_DIR = OUT_DIR / "gm_play"
STEP_CSV_GLOB = "*_step_*.csv"
SG_WIN = 21          # 1 kHz -> ~21 ms smoothing window
EPS_V = 0.05         # tanh transition speed [rad/s]
XCORR_MAX_LAG = 0.030

M1_JSON = ROOT / "data" / "derived" / "step_m1_regression_all.json"


def parse_step_csv(path):
    """Parse one step CSV -> dict of 1-D arrays + scalars."""
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        rows = [r for r in rdr if r]
    col = {h: i for i, h in enumerate(hdr)}

    def arr(name, cast=float):
        j = col.get(name)
        if j is None:
            return None
        return np.array([cast(r[j]) for r in rows], dtype=float)

    t = arr("time_sec")
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 1e-3
    return {
        "t": t, "dt": dt,
        "phase": np.array([r[col["phase"]] for r in rows]),
        "cmd": arr("pos_target_rad"),
        "q": arr("target_pos"), "qd": arr("target_vel"), "tau": arr("target_effort"),
        "i_act": arr("actual_current_A"),
        "gyro": (np.stack([arr("imu_gyro_x"), arr("imu_gyro_y"), arr("imu_gyro_z")], axis=1)
                 if col.get("imu_gyro_x") is not None else None),
        "kp": float(rows[0][col["sent_kp"]]), "kd": float(rows[0][col["sent_kd"]]),
        "joint": rows[0][col["target_joint"]],
        "n": len(rows),
    }


def identify_one(path, use_mujoco):
    d = parse_step_csv(path)
    joint = d["joint"]
    entry = {"file": path.name, "kp": d["kp"], "kd": d["kd"], "n": d["n"],
             "dt_ms": round(d["dt"] * 1e3, 3)}

    # --- runtime column-semantics guard -----------------------------------
    pre_mask = d["phase"] == "pre_hold"
    is_fb, ev = detect_feedback_semantics(d["cmd"], d["q"], d["tau"], phases=pre_mask)
    entry["feedback_semantics_ok"] = bool(is_fb)
    entry["semantics_evidence"] = {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in ev.items()}
    print(f"[joint] {joint}: semantics is_feedback={is_fb} ev={entry['semantics_evidence']}")

    qd = d["qd"]
    tau = d["tau"]

    # --- M1 reproduction (post_hold segment, same as legacy M1 n=108000; all
    #     joints; gates apply to serial only) -------------------------------
    post = d["phase"] == "post_hold"
    seg = post if post.sum() > 1000 else np.ones_like(post, bool)
    entry["m1_segment"] = "post_hold" if post.sum() > 1000 else "full"
    entry["m1"] = fit_m1(tau[seg], d["cmd"][seg], d["q"][seg], qd[seg],
                         d["kp"], d["kd"])

    # --- torque-current mapping --------------------------------------------
    entry["kt"] = fit_kt(tau, d["i_act"])

    # --- serial-joint dynamics ----------------------------------------------
    if is_fb:
        seg_dyn = seg
        lut = (build_gravity_lut(joint) if use_mujoco else zero_lut(joint))
        g = gravity_lookup(d["q"][seg_dyn], lut)
        qdd = savgol_ddq(d["q"], d["dt"], window=SG_WIN)[seg_dyn]
        # gyro coupling for ALL serial joints: knee steps excite the largest
        # base motion (gyro_rms 0.29 rad/s, highest serial — M1 evidence);
        # T9 knee R2 0.50-0.53 without coupling traced to exactly this.
        gyro = d["gyro"][seg_dyn] if (d["gyro"] is not None and is_serial(joint)) else None
        entry["dynamics"] = fit_dynamics(tau[seg_dyn] - g, qdd, qd[seg_dyn],
                                         gyro=gyro, eps=EPS_V)
        entry["dynamics"]["gravity_source"] = lut["source"]
        # actuation delay: reconstructed PD command vs measured torque
        tau_cmd = d["kp"] * (d["cmd"] - d["q"]) - d["kd"] * qd
        lag_s, corr = xcorr_delay(tau_cmd[seg_dyn], tau[seg_dyn], d["dt"],
                                  max_lag_s=XCORR_MAX_LAG)
        entry["delay_ms"] = round(lag_s * 1e3, 2)
        entry["delay_peak_corr"] = round(float(corr), 3)
    else:
        entry["dynamics"] = None
        entry["delay_ms"] = None
        entry["semantics_note"] = "columns did not verify as feedback; dynamics skipped"

    slim = {k: v for k, v in entry.items() if k != "semantics_evidence"}
    print(f"[joint] {joint}: m1_alpha={entry['m1']['alpha']:.3f} "
          f"R2={entry['m1']['R2']:.2f} kt={entry['kt']['kt']:.1f} "
          f"R2={entry['kt']['R2']:.3f} " +
          (f"J_eff={entry['dynamics']['J_eff']:.4f} tau_c={entry['dynamics']['tau_c']:.2f} "
           f"tau_v={entry['dynamics']['tau_v']:.3f} dynR2={entry['dynamics']['R2']:.2f} "
           f"delay={entry['delay_ms']} ms" if entry.get("dynamics") else "(no dynamics)"))
    return joint, entry


def main():
    files = sorted(RAW_DIR.glob(STEP_CSV_GLOB))
    if not files:
        print(f"FATAL: no step CSVs under {RAW_DIR}", file=sys.stderr)
        return 2
    use_mujoco = mujoco_available()
    print(f"[joint] mujoco available: {use_mujoco} ({len(files)} files)")

    joints = {}
    t0 = time.time()
    for p in files:
        try:
            j, entry = identify_one(p, use_mujoco)
            joints[j] = entry
        except Exception as exc:  # keep going; gate will fail on missing data
            print(f"[joint] ERROR {p.name}: {exc}", file=sys.stderr)
            joints[p.name] = {"file": p.name, "error": str(exc)}

    payload = {"joints": joints,
               "meta": {"n_joints": len(joints), "mujoco": bool(use_mujoco),
                        "sg_window": SG_WIN, "eps_v": EPS_V,
                        "elapsed_s": round(time.time() - t0, 1),
                        "m1_reference": str(M1_JSON.relative_to(ROOT))}}
    GM_DIR.mkdir(parents=True, exist_ok=True)
    (GM_DIR / "joint_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    try:
        import torch
        torch.save(payload, GM_DIR / "joint_params.pt")
        print(f"[joint] saved {GM_DIR / 'joint_params.pt'}")
    except ImportError:
        print("[joint] torch unavailable; json only")

    # gates (repo root already on sys.path; scripts dir for validate_joint)
    sys.path.insert(0, str(ROOT / "joint_identify" / "scripts"))
    from validate_joint import evaluate
    report = evaluate(joints)
    (OUT_DIR / "validation_joint.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[joint] verdict: {report['verdict']} (exit {report['exit_code']})")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
