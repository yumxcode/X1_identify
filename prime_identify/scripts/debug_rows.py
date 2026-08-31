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
frames = np.arange(0, 60)
pf.idx_tr = frames
lam, vplus, conv, Jc_all, active = pf._sweep(dyn.theta_hat, frames)

dyn.set_theta(dyn.theta_hat)
p = pf._current_p(dyn)
names = ["bx", "by", "bz", "wx", "wy", "wz"] + [j.replace("_joint", "") for j in dyn.model.names[2:]]
r_rows = np.zeros((conv.sum(), dyn.nv))
i = 0
for fi, k in enumerate(frames):
    if not conv[fi]:
        continue
    a = (wd.v[k + 1] - wd.v[k]) / wd.dt
    Y = dyn.regressor(wd.q[k], wd.v[k], a)
    Bu = np.zeros(dyn.nv)
    Bu[dyn.v_joint] = wd.u[k]
    tc = Jc_all[fi].T @ lam[fi].reshape(-1) / wd.dt
    r_rows[i] = Y @ p - Bu - tc
    i += 1
print("RMS per row (Nm):")
for j in range(dyn.nv):
    print(f"  {names[j]:22s} rms={np.sqrt((r_rows[:,j]**2).mean()):10.2f}")
print("overall rms: %.2f" % np.sqrt((r_rows**2).mean()))
