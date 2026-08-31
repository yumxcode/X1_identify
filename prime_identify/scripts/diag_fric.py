"""Real-data swing-row residual analysis: is it friction-dominated?

Swing rows are exactly contact-free (tau = Y(a)p - Bu); the residual at
nominal params on REAL data includes joint friction (unmodeled). Correlate
the residual with qd (viscous) and sign(qd) (coulomb) per joint.
"""
import os
import sys
import numpy as np
import pinocchio as pin

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(REPO, "X1_train", "resources", "robots", "x1", "urdf", "x1.urdf")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(REPO, "x1_data", "walk_diag_20260824_103222.csv")


def main():
    dyn = X1Dynamics(URDF)
    wd = load_walk_diag(CSV, dyn)
    n = len(wd.t) - 1

    dyn.set_theta(dyn.theta_hat)
    swingL = np.zeros(n, bool)
    swingR = np.zeros(n, bool)
    for k in range(n):
        pin.forwardKinematics(dyn.model, dyn.data, wd.q[k])
        pin.updateFramePlacements(dyn.model, dyn.data)
        zl = min(dyn.data.oMf[f].translation[2] for f in dyn.contact_frame_ids[:2])
        zr = min(dyn.data.oMf[f].translation[2] for f in dyn.contact_frame_ids[2:])
        swingL[k] = zl > 0.025
        swingR[k] = zr > 0.025

    p_nom = np.concatenate(
        [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
    )
    rowsL = list(range(6, 12))
    rowsR = list(range(12, 18))

    res = {j: [] for j in range(6, 18)}
    qd = {j: [] for j in range(6, 18)}
    for k in range(n):
        a = (wd.v_out[k] - wd.v[k]) / wd.dt
        Y = dyn.regressor(wd.q[k], wd.v[k], a)
        Bu = np.zeros(dyn.nv)
        Bu[6:] = wd.u[k]
        r = Y @ p_nom - Bu
        if swingL[k]:
            for i, j in enumerate(rowsL):
                res[j].append(r[j]); qd[j].append(wd.v[k, j])
        if swingR[k]:
            for i, j in enumerate(rowsR):
                res[j].append(r[j]); qd[j].append(wd.v[k, j])

    print(f"{'joint':26s} {'n':>5s} {'RMS':>7s} {'visc f_v':>9s} {'coul f_c':>9s} {'R2 fric':>7s}")
    from prime.dynamics import JOINT_ORDER
    tot_ss, tot_res = 0.0, 0.0
    for j in range(6, 18):
        r = np.array(res[j]); q = np.array(qd[j])
        A = np.stack([q, np.tanh(8 * q)], axis=1)  # viscous + smoothed coulomb
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        pred = A @ coef
        ss = float(((r - r.mean()) ** 2).sum())
        sse = float(((r - pred) ** 2).sum())
        r2 = 1 - sse / max(ss, 1e-12)
        print(f"{JOINT_ORDER[j-6]:26s} {len(r):5d} {np.sqrt((r**2).mean()):7.3f} "
              f"{coef[0]:+9.3f} {coef[1]:+9.3f} {r2:7.2f}")
        tot_ss += ss; tot_res += sse
    print(f"\noverall friction-explained variance: {1 - tot_res/tot_ss:.1%}")


if __name__ == "__main__":
    main()
