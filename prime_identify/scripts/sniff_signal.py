import os
import sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prime_identify"))
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(REPO, "X1_train", "resources", "robots", "x1", "urdf", "x1.urdf")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(REPO, "data/raw", "walk_diag_20260824_103222.csv")

dyn = X1Dynamics(URDF)
wd = load_walk_diag(CSV, dyn)

# GT params
pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
theta_gt = dyn.pi_to_theta(pi_gt)

# synthesize v_out on ALL frames (full data for max SNR)
n = len(wd.t) - 1
rng = np.random.default_rng(7)
dyn.set_theta(theta_gt)
v_out_syn = np.zeros((n, dyn.nv))
for k in range(n):
    vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
    v_out_syn[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
v_out_syn[:, 6:] += rng.normal(0, 0.004, (n, 12))
v_out_syn[:, 3:6] += rng.normal(0, 0.005, (n, 3))
wd.v_out = v_out_syn

pf = PFIE(dyn, wd, PFIEConfig(n_iters=1, holdout=0.0))
pf.idx_tr = np.arange(0, n)
pf.idx_ho = np.array([], dtype=int)
rng2 = np.random.default_rng(1)
ls_idx = pf.idx_tr[rng2.choice(n, 200, replace=False)]

c_nom = pf._accept_cost(dyn.theta_hat, ls_idx)
c_gt = pf._accept_cost(theta_gt, ls_idx)
print(f"accept-cost nominal : {c_nom:.2f}")
print(f"accept-cost GT      : {c_gt:.2f}")
print(f"signal (nominal-GT) : {c_nom - c_gt:.2f}  ({100*(c_nom-c_gt)/c_nom:.1f}% of nominal)")

# half-way point
c_half = pf._accept_cost(dyn.theta_hat + 0.5 * (theta_gt - dyn.theta_hat), ls_idx)
print(f"accept-cost halfway : {c_half:.2f}")
