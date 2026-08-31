import sys
import numpy as np

sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify/scripts")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig
from run_real_walk import fie_cost

dyn = X1Dynamics("/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf")
wd = load_walk_diag("/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv", dyn)
pf = PFIE(dyn, wd, PFIEConfig(n_iters=1, holdout=0.3))
lam, vplus, conv, Jc_all, active = pf._sweep(dyn.theta_hat, pf.idx_tr)
theta1, _, rms, n_used = pf._regression(dyn.theta_hat, pf.idx_tr, lam, conv, Jc_all, active)
print("single-step regression: frames=%d" % n_used)

for tag, idxs in (("train", pf.idx_tr), ("holdout", pf.idx_ho)):
    c0, _, _ = fie_cost(dyn, wd, idxs, dyn.theta_hat)
    c1, _, _ = fie_cost(dyn, wd, idxs, theta1)
    print(f"{tag}: nominal {c0:.1f} -> 1-step {c1:.1f} ({100*(c1/c0-1):+.1f}%)")

# mass-scaling candidate: add the GRF-implied mass to the base
pi = dyn.theta_to_pi_groups(dyn.theta_hat)
pi[0, 0] += 37.10 - 35.323
from prime.log_cholesky import pi_to_theta as p2t
from prime.dynamics import _PC_TO_LC
th_scale = dyn.theta_hat.copy()
th_scale[0:10] = p2t(pi[0][_PC_TO_LC])
for tag, idxs in (("train", pf.idx_tr), ("holdout", pf.idx_ho)):
    c2, _, _ = fie_cost(dyn, wd, idxs, th_scale)
    print(f"{tag}: nominal -> mass-scaled {c2:.1f}")
