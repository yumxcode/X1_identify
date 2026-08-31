import numpy as np
import sys
sys.path.insert(0, "/Users/yumx/code/robot_x/X1/X1_辨识/prime_identify")
from prime.dynamics import X1Dynamics
from prime.data import load_walk_diag
from prime.pfie import PFIE, PFIEConfig

dyn = X1Dynamics("/Users/yumx/code/robot_x/X1/X1_辨识/X1_train/resources/robots/x1/urdf/x1.urdf")
wd = load_walk_diag("/Users/yumx/code/robot_x/X1/X1_辨识/x1_data/walk_diag_20260824_103222.csv", dyn)
cfg = PFIEConfig(n_iters=1, holdout=0.2)
pf = PFIE(dyn, wd, cfg)
pf.idx_tr = np.arange(0, 40)
pf.idx_ho = np.array([], dtype=int)

theta = dyn.theta_hat.copy()
lam, vplus, conv, Jc_all, active = pf._sweep(theta, pf.idx_tr)
print("converged frames:", conv.sum(), "/", len(conv))
theta_new, _, rmse, n_used = pf._regression(theta, pf.idx_tr, lam, conv, Jc_all, active)
print("regression done, rmse=%.3f, n_used=%d" % (rmse, n_used))
pi_g = dyn.theta_to_pi_groups(theta)
pi_new = dyn.theta_to_pi_groups(theta_new)
d = pi_new - pi_g
print("max |dpi| per body:")
for gi in range(min(dyn.n_groups, 13)):
    print(f"  {dyn.model.names[dyn.bodies[gi]]:24s} dm={d[gi,0]:+.4f} dmc={np.round(d[gi,1:4],4)} dI={np.round(d[gi,4:10],4)}")
# check legality of pi_new -> write directly
try:
    dyn.set_theta(theta_new)
    print("set_theta(theta_new) OK")
except Exception as e:
    print("set_theta FAILED:", e)
    # find offending body
    pi_b = dyn.pi_groups_to_pi_bodies(pi_new)
    for gi, b in enumerate(dyn.bodies):
        m = pi_b[gi, 0]
        Io = np.array([[pi_b[gi,4],pi_b[gi,5],pi_b[gi,7]],[pi_b[gi,5],pi_b[gi,6],pi_b[gi,8]],[pi_b[gi,7],pi_b[gi,8],pi_b[gi,9]]])
        import pinocchio as pin
        C = pin.skew(pi_b[gi,1:4]/m)
        Ic = Io + m*(C@C)
        ev = np.linalg.eigvalsh(0.5*(Ic+Ic.T))
        if ev.min() < 0:
            print(f"  offending: {dyn.model.names[b]} eigs={np.round(ev,6)}")
