"""Scalar scan: accept-cost vs base mass offset (synthetic GT data)."""
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
CSV = os.path.join(REPO, "x1_data", "walk_diag_20260824_103222.csv")

dyn = X1Dynamics(URDF)
wd = load_walk_diag(CSV, dyn)

pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
theta_gt = dyn.pi_to_theta(pi_gt)

n = 500
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
ls_idx = pf.idx_tr[np.random.default_rng(1).choice(n, 120, replace=False)]

# scan base mass offset: theta variant = theta_hat with base group pi[0] shifted
from prime.log_cholesky import pi_to_theta as p2t, theta_to_pi
from prime.dynamics import _PC_TO_LC

pi_hat = dyn.theta_to_pi_groups(dyn.theta_hat)
print("dm    accept-cost")
for dm in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    pi_v = pi_hat.copy()
    pi_v[0, 0] += dm
    th_v = dyn.theta_hat.copy()
    th_v[0:10] = p2t(pi_v[0][_PC_TO_LC])
    c = pf._accept_cost(th_v, ls_idx)
    print(f"{dm:+5.1f}  {c:9.2f}")
