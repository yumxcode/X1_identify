"""Sensitivity diagnosis: FD gradient of accept-cost over all 90 theta dims
on synthetic GT data; report top-k sensitive directions."""
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig
from selftest_sim import perturb_pi_from, BODY_NAME, KNEE_L, KNEE_R

URDF = "/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf"
CSV = "/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv"

dyn = X1Dynamics(URDF)
wd = load_walk_diag(CSV, dyn)
n = 300
pi_gt = perturb_pi_from(dyn.pi_nominal, dyn, BODY_NAME, +2.0, [0.02, 0.0, -0.03], 1.2)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_L, +0.4, [0.01, 0.0, 0.0], 1.1)
pi_gt = perturb_pi_from(pi_gt, dyn, KNEE_R, +0.4, [0.01, 0.0, 0.0], 1.1)
theta_gt = dyn.pi_to_theta(pi_gt)
dyn.set_theta(theta_gt)
rng = np.random.default_rng(7)
v_out_syn = np.zeros((n, dyn.nv))
for k in range(n):
    vp, imp, conv = dyn.solve_contact_step(wd.q[k], wd.v[k], wd.u[k], wd.dt)
    v_out_syn[k] = vp if np.all(np.isfinite(vp)) else wd.v[k]
v_out_syn[:, 6:] += rng.normal(0, 0.004, (n, 12))
wd.v_out = v_out_syn

pf = PFIE(dyn, wd, PFIEConfig(n_iters=1, holdout=0.0))
ev_idx = np.arange(0, n, 3)  # 100 frames
theta_hat = dyn.theta_hat.copy()

t0 = time.time()
c0 = pf._accept_cost(theta_hat, ev_idx)
eps = 2e-2
grad = np.zeros(dyn.n_theta)
for i in range(dyn.n_theta):
    tp = theta_hat.copy()
    tp[i] += eps
    grad[i] = (pf._accept_cost(tp, ev_idx) - c0) / eps
print(f"c0={c0:.2f}, gradient in {time.time()-t0:.0f}s")

LC_NAMES = ["alpha", "d1", "d2", "d3", "s12", "s23", "s13", "t1", "t2", "t3"]
body_names = [dyn.model.names[dyn.bodies[g]] for g in range(dyn.n_groups)]
order = np.argsort(-np.abs(grad))
print("top-25 |grad| directions:")
for i in order[:25]:
    print(f"  {body_names[i//10]:26s} {LC_NAMES[i%10]:6s} grad={grad[i]:+10.1f}")
print(f"grad norm: {np.linalg.norm(grad):.1f}")
np.save("/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/results/sens_grad.npy", grad)
