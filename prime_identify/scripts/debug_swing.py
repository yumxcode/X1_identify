"""Pure swing-leg identification sanity check.

Uses ONLY swing-leg joint rows (no contact force coupling) from synthetic
data generated with theta_gt. If the LS recovers the GT direction here,
the bug is in the contact handling; if not, the regressor/p mapping is wrong.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = "/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv"

dyn = X1Dynamics(URDF)
wd = load_walk_diag(CSV, dyn)

# GT: base +2.0 kg only
pi_gt = dyn.pi_nominal.copy()
k = dyn.bodies.index(dyn.model.getJointId("root_joint"))
pi_gt[k, 0] += 2.0
theta_gt = dyn.pi_to_theta(pi_gt)

# find swing frames: left foot clearly above ground
import pinocchio as pin
n = 300
swing_left = []
for kk in range(n):
    pin.forwardKinematics(dyn.model, dyn.data, wd.q[kk])
    pin.updateFramePlacements(dyn.model, dyn.data)
    zl = min(dyn.data.oMf[fid].translation[2] for fid in dyn.contact_frame_ids[:2])
    zr = min(dyn.data.oMf[fid].translation[2] for fid in dyn.contact_frame_ids[2:])
    swing_left.append(zl > 0.02 and zr < 0.005)
swing_left = np.array(swing_left)
print("left-swing frames:", swing_left.sum(), "/", n)

# synthesize v+ on left-swing frames ONLY, using theta_gt
rng = np.random.default_rng(3)
v_meas = wd.v[: n + 1].copy()
dyn.set_theta(theta_gt)
for kk in np.where(swing_left)[0]:
    vp, imp, conv = dyn.solve_contact_step(wd.q[kk], wd.v[kk], wd.u[kk], wd.dt)
    v_meas[kk + 1] = vp
v_meas[1:, 6:] += rng.normal(0, 0.004, (n, 12))

# regress ONLY on left-leg joint rows (swing leg: hip3+knee, 4 rows), which
# contain NO contact force, for the FULL p (130) using nominal as start
dyn.set_theta(dyn.theta_hat)
rowsL = [6, 7, 8, 9]  # left hip_pitch/roll/yaw + knee
H = np.zeros((130, 130))
g = np.zeros(130)
for kk in np.where(swing_left)[0]:
    a = (v_meas[kk + 1] - wd.v[kk]) / wd.dt
    Y = dyn.regressor(wd.q[kk], wd.v[kk], a)
    Bu = np.zeros(dyn.nv)
    Bu[6:] = wd.u[kk]
    r = Y @ np.concatenate([dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)]) - Bu
    Yr = Y[rowsL]
    rr = r[rowsL]
    H += Yr.T @ Yr
    g += Yr.T @ (-rr)
# tiny prior
H += 1e-6 * np.trace(H) / 130 * np.eye(130)
dp = np.linalg.solve(H, g)
p_hat = np.concatenate([dyn.model.inertias[j].toDynamicParameters() for j in range(1, dyn.model.njoints)])
p_new = p_hat + dp
print("base mass: nominal %.3f -> est %.3f (GT %.3f)" % (p_hat[0], p_new[0], p_hat[0] + 2.0))
print("left knee mass delta: %+.4f (GT 0)" % dp[10 * 4])
print("cond(H) = %.1e" % np.linalg.cond(H))
