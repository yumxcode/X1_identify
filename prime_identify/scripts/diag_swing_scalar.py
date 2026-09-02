"""Scalar diagnostic: single-parameter LS for knee-mass on synthetic swing rows.

Left-leg rows during LEFT swing are EXACTLY contact-free:
  - left-foot contacts: lam = 0 (airborne)
  - right-foot force coupling into left-leg rows: J_R[:, left-leg dofs] = 0
    (right-toe velocity is kinematically independent of left-leg joints)
So Y_L(a) p = Bu_L holds exactly for GT params on synthetic v_out.
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
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(REPO, "X1_train", "resources", "robots", "x1", "urdf", "x1.urdf")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(REPO, "data/raw", "walk_diag_20260824_103222.csv")

dyn = X1Dynamics(URDF)
wd = load_walk_diag(CSV, dyn)
n = 400

pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
theta_gt = dyn.pi_to_theta(pi_gt)

# synth v_out with GT
dyn.set_theta(theta_gt)
rng = np.random.default_rng(7)
v_out_syn = np.zeros((n, dyn.nv))
for k in range(n):
    vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
    v_out_syn[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
v_out_syn[:, 6:] += rng.normal(0, 0.004, (n, 12))

# swing masks (use raw q; feet heights via kinematics)
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
print(f"swing frames: L={swingL.sum()} R={swingR.sum()} / {n}")

rowsL = list(range(6, 12))
rowsR = list(range(12, 18))
names130 = [dyn.model.names[j] for j in range(1, dyn.model.njoints)]

dyn.set_theta(dyn.theta_hat)
p_nom = np.concatenate(
    [dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]
)

# collect swing-row equations
Ys, rs = [], []
for k in range(n):
    a = (v_out_syn[k] - wd.v[k]) / wd.dt
    Y = dyn.regressor(wd.q[k], wd.v[k], a)
    Bu = np.zeros(dyn.nv)
    Bu[6:] = wd.u[k]
    r = Y @ p_nom - Bu
    if swingL[k]:
        Ys.append(Y[rowsL]); rs.append(r[rowsL])
    if swingR[k]:
        Ys.append(Y[rowsR]); rs.append(r[rowsR])
Ys = np.vstack(Ys); rs = np.concatenate(rs)
print(f"swing equations: {Ys.shape[0]} rows x 130 params")

# scalar single-parameter LS: dm for kneeL / kneeR / base
for name, true_dm in ((KNEE_L, 0.4), (KNEE_R, 0.4), (BODY_NAME, 2.0)):
    b = names130.index(name)
    col = Ys[:, 10 * b]
    dm = -(col @ rs) / (col @ col)
    print(f"{name:26s} scalar dm = {dm:+.4f} kg (true {true_dm:+.1f})")

# full LS with PER-COLUMN normalized ridge
cn = np.linalg.norm(Ys, axis=0)
keep = cn > 1e-9
H = Ys[:, keep].T @ Ys[:, keep]
g = Ys[:, keep].T @ (-rs)
rid = 1e-4 * np.trace(H) / keep.sum()
dp = np.linalg.solve(H + rid * np.eye(keep.sum()), g)
dp_full = np.zeros(130)
dp_full[keep] = dp
for name, true_dm in ((KNEE_L, 0.4), (KNEE_R, 0.4), (BODY_NAME, 2.0)):
    b = names130.index(name)
    print(f"{name:26s} fullLS dm = {dp_full[10*b]:+.4f} kg (true {true_dm:+.1f})")
print("cond(H) = %.2e" % np.linalg.cond(H))
