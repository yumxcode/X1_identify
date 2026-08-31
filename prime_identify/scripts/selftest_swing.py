"""G1' test: swing-leg-only regression recovery of leg-group perturbations.

Swing frames: one foot clearly airborne -> its leg rows have NO contact
force coupling; the regressor there is a pure, exact linear channel.
"""
import sys
import time

import numpy as np
import pinocchio as pin

sys.path.insert(0, "/workspace/X1_identify_lite/prime_identify")
sys.path.insert(0, "/workspace/X1_identify_lite/prime_identify/scripts")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from selftest_sim import perturb_pi_from, KNEE_L, KNEE_R, BODY_NAME

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/urdf/x1.urdf"
CSV = "/workspace/X1_identify_lite/x1_data/walk_diag_20260824_103222.csv"


def main():
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = len(wd.t) - 1

    # GT: knees +0.4 kg + COM + I (base also perturbed: must NOT leak)
    pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
    pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
    theta_gt = dyn.pi_to_theta(pi_gt)

    # synthesize v_out for ALL frames with GT
    t0 = time.time()
    dyn.set_theta(theta_gt)
    rng = np.random.default_rng(7)
    v_out_syn = np.zeros((n, dyn.nv))
    for k in range(n):
        vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
        v_out_syn[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
    v_out_syn[:, 6:] += rng.normal(0, 0.004, (n, 12))
    wd.v_out = v_out_syn
    print(f"synth rollout {time.time()-t0:.0f}s")

    # swing masks from kinematics (both legs, using raw q)
    dyn.set_theta(dyn.theta_hat)
    swingL = np.zeros(n, bool)
    swingR = np.zeros(n, bool)
    for k in range(n):
        pin.forwardKinematics(dyn.model, dyn.data, wd.q[k])
        pin.updateFramePlacements(dyn.model, dyn.data)
        zl = min(dyn.data.oMf[f].translation[2] for f in dyn.contact_frame_ids[:2])
        zr = min(dyn.data.oMf[f].translation[2] for f in dyn.contact_frame_ids[2:])
        swingL[k] = zl > 0.03
        swingR[k] = zr > 0.03
    print(f"swing frames L={swingL.sum()} R={swingR.sum()} of {n}")

    # regression: rows = swing leg's 6 joints; params = all 130 (or groups)
    # use per-body columns; solve LS with small ridge to nominal
    rowsL = list(range(6, 12))   # left leg joints
    rowsR = list(range(12, 18))
    H = np.zeros((130, 130))
    g = np.zeros(130)
    dyn.set_theta(dyn.theta_hat)
    p_nom = np.concatenate(
        [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
    )
    cnt = 0
    for k in range(n):
        a = (wd.v_out[k] - wd.v[k]) / wd.dt
        Y = dyn.regressor(wd.q[k], wd.v[k], a)
        Bu = np.zeros(dyn.nv)
        Bu[6:] = wd.u[k]
        r = Y @ p_nom - Bu
        if swingL[k]:
            Yr, rr = Y[rowsL], r[rowsL]
            H += Yr.T @ Yr
            g += Yr.T @ (-rr)
            cnt += 1
        if swingR[k]:
            Yr, rr = Y[rowsR], r[rowsR]
            H += Yr.T @ Yr
            g += Yr.T @ (-rr)
    print(f"regression rows: {cnt}")
    H += 1e-5 * np.trace(H) / 130 * np.eye(130)
    dp = np.linalg.solve(H, g)
    p_est = p_nom + dp

    names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]
    print("\n=== G1' RECOVERY (swing-channel) ===")
    ok = True
    for name, true_dm in ((KNEE_L, 0.4), (KNEE_R, 0.4), BODY_NAME and ("left_hip_yaw_joint", 0.0)):
        b = names130.index(name)
        rec = p_est[10 * b] - p_nom[10 * b]
        frac = 100 * rec / true_dm if true_dm else None
        st = "PASS" if abs(rec - true_dm) <= 0.15 else "FAIL"
        ok &= st == "PASS"
        print(f"{name:24s} rec {rec:+.3f} / true {true_dm:+.3f}"
              + (f" ({frac:.0f}%)" if frac is not None else "")
              + f"  [{st}]")
    # also report where the base delta went (should be ~0 on this channel)
    b = names130.index(BODY_NAME)
    print(f"{BODY_NAME:24s} rec {p_est[10*b]-p_nom[10*b]:+.3f} / true +2.000 "
          "(expected ~0: canceled on this channel)")
    print("G1'", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
