"""Single-joint step data: hypothesis test (suspended leg, fixed base).

Channel: tau_meas = KT_eff * actual_current_A, tau_model = Y(q,v,a) p_nom
(fixed-base pinocchio model, distal subtree of the stepped joint).
High R2 of tau_model ~ current supports the suspended/fixed-base hypothesis
and yields a calibrated torque scale for identification.
"""
import csv
import os
import sys
import numpy as np
import pinocchio as pin
from scipy.signal import butter, filtfilt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
from prime.dynamics import JOINT_ORDER

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(REPO, "X1_train", "resources", "robots", "x1", "urdf", "x1.urdf")


def load_step(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = list(r)
    col = {h: i for i, h in enumerate(hdr)}
    t = np.array([float(x[col["time_sec"]]) for x in rows])
    q = np.array([float(x[col["target_pos"]]) for x in rows])
    qd = np.array([float(x[col["target_vel"]]) for x in rows])
    cur = np.array([float(x[col["actual_current_A"]]) for x in rows])
    ph = np.array([x[col["phase"]] for x in rows])
    jname = rows[len(rows) // 2][col["target_joint"]]
    kp = float(rows[len(rows) // 2][col["sent_kp"]])
    kd = float(rows[len(rows) // 2][col["sent_kd"]])
    return t, q, qd, cur, ph, jname, kp, kd


def lp(x, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x)


def analyze(path, fc=30.0, verbose=True):
    t, q, qd, cur, ph, jname, kp, kd = load_step(path)
    fs = 1000.0
    v_f = lp(qd, fc, fs)
    a_f = lp(np.gradient(v_f, t), fc, fs)

    # fixed-base model
    model = pin.buildModelFromUrdf(URDF)  # fixed base
    data = model.createData()
    jid = model.getJointId(jname)
    # fixed-base joint order = JOINT_ORDER (0..11)
    dof = model.joints[jid].idx_v

    n = len(t)
    tau_nom = np.zeros(n)
    p_nom = np.concatenate(
        [model.inertias[j].toDynamicParameters() for j in range(1, model.njoints)]
    )
    # only the stepped dof moves; others held at their initial values
    q0 = pin.neutral(model)
    mask = ph != "pre_hold"  # use stepping + post-hold
    for k in np.where(mask)[0]:
        qq = q0.copy()
        qq[dof] = q[k]
        vv = np.zeros(model.nv)
        vv[dof] = v_f[k]
        aa = np.zeros(model.nv)
        aa[dof] = a_f[k]
        tau_nom[k] = pin.rnea(model, data, qq, vv, aa)[dof]

    ok = mask & (np.abs(t - 0) > 0)
    A = np.stack([cur[ok], np.ones(ok.sum())], axis=1)
    coef, res_, *_ = np.linalg.lstsq(A, tau_nom[ok], rcond=None)
    pred = A @ coef
    ss = ((tau_nom[ok] - tau_nom[ok].mean()) ** 2).sum()
    sse = ((tau_nom[ok] - pred) ** 2).sum()
    r2 = 1 - sse / ss
    if verbose:
        print(f"{jname:26s} dof={dof} kp={kp} kd={kd}")
        print(f"  KT_eff = {coef[0]:8.2f} Nm/A, offset = {coef[1]:+.3f} Nm, R2 = {r2:.4f}")
        print(f"  tau_nom range [{tau_nom[ok].min():.2f}, {tau_nom[ok].max():.2f}] Nm; "
              f"cur range [{cur[ok].min():.3f}, {cur[ok].max():.3f}] A")
        print(f"  a_f range [{a_f[mask].min():.1f}, {a_f[mask].max():.1f}] rad/s2 "
              f"(stepping only: [{a_f[ph=='stepping'].min():.1f}, {a_f[ph=='stepping'].max():.1f}])")
    return r2, coef[0], jname


if __name__ == "__main__":
    import glob
    base = os.path.join(REPO, "x1_data") + "/"
    for p in sorted(glob.glob(base + "*_step_*.csv"))[:4]:
        analyze(p)
        print()
